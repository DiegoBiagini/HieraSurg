import argparse
import os
from typing import List
from tqdm import tqdm
from functools import partial
import torch
import json
import cv2
import numpy as np
import imageio
import numpy as np
from PIL import Image

from inference.evaluate_metrics import pipeline_builder, dataloader_builder, pipeline_inference_builder  
from torchmetrics import Metric

from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
from natsort import natsorted
os.environ['YOLO_VERBOSE'] = 'False'

from skimage.metrics import peak_signal_noise_ratio, structural_similarity



class DetectorAgreement(Metric):
    def __init__(self, detector_path : str, match_labels : bool = False, iou_threshold : float = 0.5, conf_threshold : float = 0.5, max_frames : int = 16):
        super().__init__(dist_sync_on_step=False)
        self.add_state("iou", default=[])
        self.add_state("hit_bb_gt", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("hit_bb_gen", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_bb_gt", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_bb_gen", default=torch.tensor(0), dist_reduce_fx="sum")
        self.detector = YOLO(detector_path, verbose=False)

        self.max_frames = max_frames

        # Settings for detector evaluation
        self.match_labels = match_labels
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
    
    def preprocess_single(self, video, real=False):
        if real:
            video = (video+1)/2
            video = video.permute(1,0,2,3)
        # Change channel order
        return video[:self.max_frames, ...]

    def filter_bbs(self, results):
        boxes, labels, scores = [], [], []
        for result in results:
            filtered_boxes = []
            filtered_labels = []
            filtered_scores = []

            for box in result.boxes:
                score = box.conf.cpu().item()
                label = box.cls.cpu().item()
                if score > self.conf_threshold:
                    filtered_boxes.append(box)
                    filtered_labels.append(label)
                    filtered_scores.append(score)
            
            boxes.append(filtered_boxes)
            labels.append(filtered_labels)
            scores.append(filtered_scores)

            #result.show()
        return boxes, labels

    def compute_iou(self, box1, box2):
        # Compute the intersection over union of two boxes
        x1, y1, x2, y2 = box1.xyxy.cpu().numpy()[0]
        x1_p, y1_p, x2_p, y2_p = box2.xyxy.cpu().numpy()[0]

        xi1 = max(x1, x1_p)
        yi1 = max(y1, y1_p)
        xi2 = min(x2, x2_p)
        yi2 = min(y2, y2_p)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_p - x1_p) * (y2_p - y1_p)
        union_area = box1_area + box2_area - inter_area

        iou = inter_area / union_area
        return iou

    def update(self, gen_samples: torch.Tensor, gt_samples: torch.Tensor):
        assert gen_samples.shape[0] == gt_samples.shape[0]
        # Preprocess videos

        for i in range(gen_samples.shape[0]):

            gen_video = self.preprocess_single(gen_samples[i], real=False)
            gt_video = self.preprocess_single(gt_samples[i], real=False)

            assert gen_video.shape[0] == gt_video.shape[0]

            # This considers a batch of images
            # Run detector on both gen and real
            gen_detect_results = self.detector.predict(gen_video, show_labels=False, show_conf=False, augment=False, verbose=False, save=False)
            gt_detect_results = self.detector.predict(gt_video, show_labels=False, show_conf=False, augment=False, verbose=False, save=False)

            gt_boxes, gt_labels = self.filter_bbs(gt_detect_results)
            gen_boxes, gen_labels = self.filter_bbs(gen_detect_results)

            for frame_id in range(len(gt_boxes)):
                frame_gt_boxes, frame_gt_labels = gt_boxes[frame_id], gt_labels[frame_id]
                frame_gen_boxes, frame_gen_labels = gen_boxes[frame_id], gen_labels[frame_id]

                if len(frame_gt_boxes) == 0 or len(frame_gen_boxes) == 0:
                    self.total_bb_gt += torch.as_tensor(len(frame_gt_boxes))
                    self.total_bb_gen += torch.as_tensor(len(frame_gen_boxes))
                    continue

                iou_matrix = np.zeros((len(frame_gt_boxes), len(frame_gen_boxes)))

                for j, gt_box in enumerate(frame_gt_boxes):
                    for k, gen_box in enumerate(frame_gen_boxes):
                        if self.match_labels and frame_gt_labels[j] != frame_gen_labels[k]:
                            iou_matrix[j, k] = 0
                        else:
                            iou_matrix[j, k] = self.compute_iou(gt_box, gen_box)

                row_ind, col_ind = linear_sum_assignment(-iou_matrix)

                for j, k in zip(row_ind, col_ind):
                    if iou_matrix[j, k] >= self.iou_threshold:
                        self.iou.append(iou_matrix[j, k])
                        self.hit_bb_gt += torch.tensor(1)
                        self.hit_bb_gen += torch.tensor(1)

                self.total_bb_gt += torch.as_tensor(len(frame_gt_boxes))
                self.total_bb_gen += torch.as_tensor(len(frame_gen_boxes))


    def compute(self):        
        return {'mIoU': np.mean(self.iou), 'gen_agreement_ratio': (self.hit_bb_gen/self.total_bb_gen).item(), 'gt_agreement_ratio': (self.hit_bb_gt/ self.total_bb_gt).item()}

# Constants
#N_FRAMES = 16
#COGVIDEO_FRAMES = 16 # Native cogvideo number of frames
H, W = 256, 384

N_SAMPLES_UCOND = 1000 # How many videos to generate in the ucond setting
NUM_STEPS_INFERENCE = 100
SEED = 42

device = "cuda"

def load_video(video_path, max_num_frames=16, fps=1):
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame indices to sample at the specified fps
    sample_interval = int(video_fps / fps)
    sample_indices = [i * sample_interval for i in range(int(total_frames / sample_interval))][:max_num_frames]
    
    frames = []
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx in sample_indices:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (256, 384), interpolation=cv2.INTER_AREA)          
            #print(frame.shape) # (384,256,3)
            frames.append(frame)
            
        frame_idx += 1
        
    cap.release()
    
    # Stack frames and convert to tensor
    frames = np.stack(frames)
    video_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2) # [T, C, H, W]
    video_tensor = video_tensor.float() / 255.0 # Normalize to [0,1]
    
    return video_tensor


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate metrics on video generation results')
    
    parser.add_argument('--main_model_path', type=str, required=True,
                        help='Path to the model checkpoint to evaluate')
    parser.add_argument('--cnet_model_path', type=str, required=False,
                        help='Path to the CNet model checkpoint to evaluate')

    parser.add_argument('--model_type', type=str, required=True, choices=['cogvideo2b', 'cogvideo2b_cnet', 'cogvideo2b_segmap', 'cogvideo5b', 'cogvideo5b_cnet', 'cogvideo5b_segmap'],
                        help='Type of the model to evaluate')

    parser.add_argument('--cogvideo_frames', type=int, default=16, 
                        help='Native frame length of CogVideo, how many are generated')
    parser.add_argument('--out_frames', type=int, default=16, 
                        help='How many frames will be outputted')
    
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='Base directory containing the evaluation dataset')
    
    parser.add_argument('--detector_path', type=str, required=True,
                        help='Path to the weights of the yolo detector')
    
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for evaluation')
    
    parser.add_argument('--num_batches', type=int, default=-1,
                        help='Number of batches to run eval on')
    parser.add_argument('--fps', type=int, default=1,
                        help='FPS at which to evaluate detection')    
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')

    parser.add_argument("--use_precomputed", action="store_true",
                        help="If true, use will not use the model to compute metrics, but will use precomputed samples instead")
    parser.add_argument("--precomputed_samples_dir", type=str, default=None,
                        help="Directory containing precomputed samples")
    parser.add_argument("--precomputed_real_samples_dir", type=str, default=None,
                        help="Directory containing the real precomputed samples")    
    
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--max_out_samples', type=int, default=-1, help='How many batches of samples to save for the model in the out dir, all if -1')
    
    return parser.parse_args()

def validate_args(args):
    # Check if model path exists
    if not args.use_precomputed and not os.path.exists(args.main_model_path):
        raise ValueError(f"Model path does not exist: {args.main_model_path}")
    # Check that if using cnet then cnet_model_path exists
    if args.model_type.endswith("_cnet") and not os.path.exists(args.cnet_model_path):
        raise ValueError(f"CNet model path does not exist: {args.cnet_model_path}")
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        raise ValueError(f"Data directory does not exist: {args.data_dir}")
    
    # Check if weights of detector exist
    if not os.path.exists(args.detector_path):
        raise ValueError(f"Detector weights do not exist: {args.detector_path}")    
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

def main():
    args = parse_args()
    validate_args(args)

    if args.use_precomputed:
        assert args.precomputed_samples_dir is not None, "Precomputed samples directory is required when using precomputed metrics"
        print(f"Using precomputed samples from {args.precomputed_samples_dir}")
    else:
        print(f"Evaluating model: {args.main_model_path}")
        load_args = {'model_path': args.main_model_path}
        if args.model_type.endswith("_cnet"):
            print(f"CNet model: {args.cnet_model_path}")
            load_args['cnet_model_path'] = args.cnet_model_path
        print(f"Dataset directory: {args.data_dir}")
        print(f"Using detector: {args.detector_path}")
    
        # Model choice loading
        pipeline = pipeline_builder[args.model_type](**load_args)

    # Instantiate detector
    detector_metric = DetectorAgreement(detector_path=args.detector_path, max_frames=args.out_frames)
    frames_psnr = []
    frames_ssim = []
    if args.use_precomputed:
        # Load the gt samples
        precomputed_samples = natsorted([os.path.join(args.precomputed_samples_dir, f) for f in os.listdir(args.precomputed_samples_dir) if f.endswith('.mp4')])
        precomputed_real_samples = natsorted([os.path.join(args.precomputed_real_samples_dir, f) for f in os.listdir(args.precomputed_real_samples_dir) if f.endswith('.mp4')])

        for i, sample in enumerate(precomputed_real_samples):
            gt_sample = load_video(sample, fps=args.fps)
            gen_sample = load_video(precomputed_samples[i], fps=args.fps)

            for f in range(gen_sample.shape[0]):
                frame_gt = (np.moveaxis(gt_sample[f].cpu().numpy(),0,-1)*255).astype(np.uint8)
                frame_gen = (np.moveaxis(gen_sample[f].cpu().numpy(),0,-1)*255).astype(np.uint8)

                frames_psnr.append(peak_signal_noise_ratio(frame_gt,frame_gen))    
                frames_ssim.append(structural_similarity(frame_gt,frame_gen, channel_axis=2, multichannel=True))           

            detector_metric.update(torch.unsqueeze(gen_sample,0), torch.unsqueeze(gt_sample,0))
            
            if args.num_batches != -1 and i > args.num_batches:
                break

    else:
        # Data loading depending on the model choice
        dl = dataloader_builder[args.model_type](batch_size=args.batch_size, n_workers=args.num_workers, data_path = args.data_dir, 
                                                out_frames=args.out_frames, cogvideo_frames=args.cogvideo_frames)

        saved_videos = 0
        n_batch = 0
        for batch in tqdm(dl):
            gen_samples, conds = pipeline_inference_builder[args.model_type](pipeline, batch, batch_size=args.batch_size, device=device,
                                                                            out_frames=args.out_frames, cogvideo_frames=args.cogvideo_frames)
            
            gt_samples = batch['videos'].to(device)
            detector_metric.update(gen_samples, gt_samples)
            # Save generated videos if output folder is specified
            if args.output_dir is not None:
                os.makedirs(args.output_dir, exist_ok=True)
                # Iterate through batch dimension
                for b in range(gen_samples.shape[0]):
                    if args.max_out_samples != -1 and saved_videos >= args.max_out_samples:
                        break
                    # Convert to uint8 format expected by video writer
                    video = (gen_samples[b].permute(0,2,3,1).cpu().numpy() * 255).astype(np.uint8)
                    out_path = os.path.join(args.output_dir, f"gen_video_{saved_videos:04d}.mp4")
                    with imageio.get_writer(out_path, fps=1, codec='libx264', quality=8) as writer:
                        for frame in video:
                            writer.append_data(frame)

                    if 'init_img' in conds:
                        img = ((conds['init_img'][b]+1)/2*255).byte().numpy()
                        img = np.transpose(img, (1, 2, 0))

                        img = Image.fromarray(img)

                        out_path = os.path.join(args.output_dir, f"init_img_{saved_videos:04d}.png")
                        img.save(out_path)                        
                    if 'segmap' in conds:
                        segmap = torch.unsqueeze(conds['segmap'][b], axis=1).repeat(1,3,1,1)/torch.amax(conds['segmap'][b])
                        segmap = (segmap.permute(0,2,3,1).cpu().numpy() * 255).astype(np.uint8)
                        out_path = os.path.join(args.output_dir, f"segmap_{saved_videos:04d}.mp4")
                        with imageio.get_writer(out_path, fps=1, codec='libx264', quality=8) as writer:
                            for frame in segmap:
                                writer.append_data(frame)
                    if 'cnet' in conds:
                        cnet = (conds['cnet'][b]+1)/2
                        cnet = (cnet.permute(0,2,3,1).cpu().numpy() * 255).astype(np.uint8)
                        out_path = os.path.join(args.output_dir, f"cnet_{saved_videos:04d}.mp4")
                        with imageio.get_writer(out_path, fps=1, codec='libx264', quality=8) as writer:
                            for frame in cnet:
                                writer.append_data(frame)
                    saved_videos += 1

            n_batch += 1
            if args.num_batches != -1 and n_batch > args.num_batches:
                break
    
    results = detector_metric.compute()
    results['SSIM'] = np.mean(np.asarray(frames_ssim), axis=0)
    results['PSNR'] = np.mean(np.asarray(frames_psnr), axis=0)
    for k in results:
        print(f"{k}: {results[k]}")

    # Write results to a JSON file
    output_path = os.path.join(args.output_dir, "detector_metrics.json")
    print(output_path)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
		