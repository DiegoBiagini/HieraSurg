import argparse
import logging
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = "[0]"
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from diffusers import AutoencoderKLCogVideoX, CogVideoXDPMScheduler, CogVideoXTransformer3DModel
from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid
from tqdm.auto import tqdm

from finetune.dataset.utils_sd import get_dataloader_videosegmap
from argparse import Namespace
from segcond.cogvideo_segpred import CogVideoXTransformer3DModel_SegPred_I2V, CogVideoXTransformer3DModel_SegPred_I2V_Old
from segcond.segpred_pipeline import CogVideoXPipeline_VideoSegPred_I2V, CogVideoXPipeline_VideoSegPred_I2V_Img
from finetune.train_cogvideox_cholec_segpred_i2v import apply_colormap_to_videos, export_to_video, to_segpred_batch
import numpy as np
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import blosc
from scipy.ndimage import zoom
from segcond.cogvideo_segcond import CogVideoXTransformer3DModel_SegCond_I2V
from segcond.segcond_pipeline import CogVideoXPipeline_VideoSegmap_I2V
import imageio
from copy import copy
import torch.nn.functional as F


logger = logging.getLogger(__name__)

def set_ablation():
    # Substitute with unknown phase and triplets
    pass         

def move_to_device(obj, device, dtype=None):
    """
    Move a tensor, a dictionary of tensors, a list of tensors, or a list of dictionaries of tensors to the specified device and dtype.
    
    Args:
        obj: A torch.Tensor, a dictionary of torch.Tensors, a list of torch.Tensors, or a list of dictionaries of torch.Tensors.
        device: The target device (e.g., 'cuda' or 'cpu').
        dtype: The target data type (e.g., torch.float32). Default is None.
        
    Returns:
        The object with all tensors moved to the specified device and dtype.
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device, dtype=dtype) if dtype else obj.to(device)
    elif isinstance(obj, dict):
        return {k: v.to(device, dtype=dtype) if isinstance(v, torch.Tensor) else v for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(item, device, dtype) for item in obj]
    else:
        raise TypeError("The input object must be a torch.Tensor, a dictionary of torch.Tensors, or a list of these.")

def naive_convert_to_bw(color_image):
    """
    Convert a color image to black and white (grayscale).

    Parameters:
    color_image (numpy.ndarray): Input color image of shape (H, W, 3).

    Returns:
    numpy.ndarray: Grayscale image of shape (H, W).
    """
    # Define the weights for the RGB channels
    weights = np.array([0.2989, 0.5870, 0.1140])

    # Convert the color image to grayscale using the dot product
    bw_image = np.dot(color_image[...,:3], weights)

    return bw_image
def upscale_and_pad_video(array, new_height, new_width, pad_width):
    # Initialize the new array with the correct shape
    pad_value = np.max(array)-1 # Since the background is the second to last, ideally, wont matter anyway
    upscaled_padded_array = np.zeros((array.shape[0], new_height, new_width + 2 * pad_width))
    
    for i in range(array.shape[0]):
        # Calculate the zoom factors for height and width
        zoom_height = new_height / array.shape[1]
        zoom_width = new_width / array.shape[2]
        
        # Perform the zoom (interpolation) for each slice
        upscaled_slice = zoom(array[i], (zoom_height, zoom_width), order=3)  # order=3 for bicubic interpolation
        
        # Pad the array symmetrically on the left and right sides
        padded_slice = np.pad(upscaled_slice, ((0, 0), (pad_width, pad_width)), mode='constant', constant_values=pad_value)
        
        # Assign the padded slice back to the new array
        upscaled_padded_array[i] = padded_slice
    
    return upscaled_padded_array

"""
def invert_colormap_to_videos(colored_videos, colormap_name='tab20'):
    
    Invert the colormap applied to a batch of colored videos to get back the grayscale videos.

    Parameters:
    - colored_videos: np.ndarray, the input batch of colored videos with shape (B, F, H, W, 3) and values between -1 and 1.
    - colormap_name: str, the name of the matplotlib colormap used.

    Returns:
    - grayscale_videos: np.ndarray, the output batch of grayscale videos with shape (B, F, H, W) and values between 0 and 255.
    
    B, F, H, W, _ = colored_videos.shape
    grayscale_videos = np.zeros((B, F, H, W), dtype=np.float32)
    cmap = plt.get_cmap(colormap_name)
    
    for b in range(B):
        # Identify unique colors in the current video
        unique_colors = np.unique(colored_videos[b].reshape(-1, 3), axis=0)
        
        # Create a color palette for the current video
        color_palette = {tuple((np.array(cmap(i / len(unique_colors))[:3])).astype(np.float32)): i
                         for i in range(len(unique_colors))}
        
        for f in range(F):
            for color, grayscale_value in color_palette.items():
                mask = np.all(colored_videos[b, f] == color, axis=-1)
                grayscale_videos[b, f][mask] = grayscale_value
        grayscale_videos[b,...] = grayscale_videos[b,...]/np.max(grayscale_videos[b,...])
    
    return grayscale_videos
"""

def invert_colormap_to_videos(colored_videos, colormap_name='tab20'):
    """
    Invert the colormap applied to a batch of colored videos to get back the grayscale videos.

    Parameters:
    - colored_videos: np.ndarray, the input batch of colored videos with shape (B, F, H, W, 3) and values between -1 and 1.
    - colormap_name: str, the name of the matplotlib colormap used.

    Returns:
    - grayscale_videos: np.ndarray, the output batch of grayscale videos with shape (B, F, H, W) and values between 0 and 255.
    """
    B, F, H, W, _ = colored_videos.shape
    grayscale_videos = np.zeros((B, F, H, W), dtype=np.float32)
    pkl_to_save = np.zeros((B, F, H, W), dtype=np.uint8)    
    cmap = plt.get_cmap(colormap_name)
    
    # Get the 20 different colors of the original color palette
    color_palette = np.array([cmap(i)[:3] for i in range(20)], dtype=np.float32)

    for b in range(B):
        # Identify unique colors in the current video
        unique_colors = np.unique(colored_videos[b].reshape(-1, 3), axis=0)
        
        # Compute the cost matrix based on L2 distance
        cost_matrix = np.linalg.norm(unique_colors[:, np.newaxis] - color_palette, axis=2)
        
        # Solve the linear sum assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Create a mapping between unique_colors and the closest colors in the palette
        color_mapping = {tuple(unique_colors[row]): color_palette[col] for row, col in zip(row_ind, col_ind)}
        
        # Map the colors in the video to the closest colors in the palette
        mapped_video = np.zeros_like(colored_videos[b])
        for color, mapped_color in color_mapping.items():
            mask = np.all(colored_videos[b] == color, axis=-1)
            mapped_video[mask] = mapped_color
        
        assign_reordered = sorted(list(col_ind))
        new_palette = [color_palette[i] for i in range(color_palette.shape[0]) if i in assign_reordered]
        
        # Convert the mapped colors to grayscale values
        for i, color in enumerate(new_palette): #color_palette
            mask = np.all(mapped_video == color, axis=-1)
            grayscale_videos[b][mask] = i
        
        pkl_to_save[b,...] = grayscale_videos[b,...].astype(np.uint8)
        grayscale_videos[b,...] = (grayscale_videos[b,...] / np.max(grayscale_videos[b,...]))
    
    return grayscale_videos, np.squeeze(pkl_to_save)

def find_optimal_k(colors, max_k=20):
    Ks = np.arange(5, max_k+1)
    inertias = []
    for k in Ks:
        km = KMeans(n_clusters=k, random_state=42).fit(colors)
        inertias.append(km.inertia_)

    # first and second derivatives
    d1 = np.diff(inertias)        # length = len(Ks)-1
    d2 = np.diff(d1)              # length = len(Ks)-2

    # elbow is where the second derivative is most negative
    elbow_idx = np.argmin(d2)     # because inertias are decreasing, d1<0, so d2<0
    optimal_k = Ks[elbow_idx+1]   # +1 to map back from d2 to Ks
    return optimal_k

def quantize_video(video, k = None):
    # Reshape the image to a 2D array of pixels
    video = (video*255).astype(np.uint8)
    video = np.stack([cv2.cvtColor(video[i,...],cv2.COLOR_RGB2LAB) for i in range(video.shape[0])])
    F,H,W,C= video.shape
    pixels = video.reshape(-1, 3)

    # Extract unique colors
    unique_colors, unique_indices = np.unique(pixels, axis=0, return_inverse=True)

    if k is None:
        k = find_optimal_k(unique_colors)
    print(f"Detected {k} colors in the video")
    
    # Train k-means on sampled colors
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(unique_colors)

    labels = kmeans.predict(pixels)
    quantized = kmeans.cluster_centers_[labels]
    
    # Reshape back
    quantized_frame = quantized.reshape(F,H,W,3)
    
    out_frames = [cv2.cvtColor(quantized_frame[i,...].astype(np.uint8), cv2.COLOR_LAB2RGB)/255 for i in range(quantized_frame.shape[0])]
    return np.stack(out_frames)

def get_args():
    parser = argparse.ArgumentParser(description="Simple example of an inference script for CogVideoX.")

    # Model information
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models for segpred.",
    )
    parser.add_argument(
        "--second_step_model_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models for segmap(i2v).",
    )    

    parser.add_argument(
        "--val_data_path",
        type=str,
        default=None,
        help=("A folder containing the validation data."),
    )

    parser.add_argument(
        "--annotations_path",
        type=str,
        default=None,
        help=("A folder containing the annotations."),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )

    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")

    parser.add_argument(
        "--output_dir",
        type=str,
        default="cogvideox-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="All input videos are resized to this height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=720,
        help="All input videos are resized to this width.",
    )
    parser.add_argument(
        "--max_num_frames", type=int, default=17, help="All input videos will be truncated to these many frames, should be 17/33/41/49."
    )
    parser.add_argument(
        "--val_batch_size", type=int, default=4, help="Batch size (per device) for the validation dataloader."
    )    
    parser.add_argument("--segmap_dropout", type=float, default=0, help="Dropout rate for objects in the video_segmap.")
    parser.add_argument("--num_inference_steps", type=int, default=100, help="Number of denoising steps")

    parser.add_argument(
        "--text_cond",
        type=str,
        default="label_emb",
        choices=["label_emb", "SurgVLP"],
        help=(
            "Type of conditioning for the triplet/phase"
        ),
    )    
    parser.add_argument("--save_pkl", action="store_true", help="If true save the gen segmaps as pkl files(compressed with blosc).")


    return parser.parse_args()
  

def main(args):
    device="cuda:0"
    if args.seed is not None:
        torch.manual_seed(args.seed)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    load_dtype = torch.bfloat16
    weight_dtype = torch.bfloat16
    transformer = CogVideoXTransformer3DModel_SegPred_I2V.from_pretrained( # or I2V_Old
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=load_dtype,
    )
    

    vae = AutoencoderKLCogVideoX.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", torch_dtype=torch.float32
    ).to(weight_dtype)

    scheduler = CogVideoXDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    vae.requires_grad_(False)

    text_cond_model = {}

    if args.text_cond == "SurgVLP":
        import surgvlp
        from mmengine.config import Config
        configs = Config.fromfile('./segcond/surgvlp_cfgs/config_peskavlp.py')['config']
        # Change the config file to load different models: config_surgvlp.py / config_hecvl.py / config_peskavlp.py

        model, preprocess = surgvlp.load(configs.model_config, device=device)
                
        text_cond_model['name'], text_cond_model['model'], text_cond_model['preprocessor'] = args.text_cond, model, preprocess
    

    transformer.to(device if torch.cuda.is_available() else "cpu", dtype=weight_dtype)
    vae.to(device if torch.cuda.is_available() else "cpu")

    val_dataset_args = Namespace(data_path=args.val_data_path, image_size=(args.height,args.width), dataset_name="CholecT80VideoSegmapHiera", 
                             global_seed=args.seed, gpu_batch_size=args.val_batch_size, num_workers=args.dataloader_num_workers, video_length=16, 
                             vae=None, segmap_dropout=args.segmap_dropout, max_num_frames=16, clip_step=1)
    val_dataset_args.annotations_path = args.annotations_path
    val_dataset_args.text_cond = args.text_cond
    _, val_dataloader = get_dataloader_videosegmap(0, val_dataset_args, logger, precompute="compute_on_the_fly")    

    pipe = CogVideoXPipeline_VideoSegPred_I2V_Img.from_pretrained(
    #pipe = CogVideoXPipeline_VideoSegPred_I2V.from_pretrained(    
        args.pretrained_model_name_or_path,
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
        torch_dtype=weight_dtype,
    ).to(device)

    pipeline_args = {
        "height": 256, #128,
        "width": 384, #192,
        "num_frames": 16, #args.max_num_frames,
        "text_cond_model": text_cond_model,
        "num_inference_steps": args.num_inference_steps,
        "latent_encoding": "slice"#"full" #"slice"

    }

    scheduler_args = {}

    if "variance_type" in pipe.scheduler.config:
        variance_type = pipe.scheduler.config.variance_type

        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"

        scheduler_args["variance_type"] = variance_type

    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, **scheduler_args)
    pipe = pipe.to(device if torch.cuda.is_available() else "cpu")
    
    # Then load the second step model
    transformer2 = CogVideoXTransformer3DModel_SegCond_I2V.from_pretrained(
        args.second_step_model_path,
        subfolder="transformer",
        torch_dtype=load_dtype,
    )
    vae2 = AutoencoderKLCogVideoX.from_pretrained(args.second_step_model_path, subfolder="vae", torch_dtype=load_dtype)

    scheduler2 = CogVideoXDPMScheduler.from_pretrained(args.second_step_model_path, subfolder="scheduler")

    pipe2 = CogVideoXPipeline_VideoSegmap_I2V.from_pretrained(
        args.second_step_model_path,
        transformer=transformer2,
        vae=vae2,
        scheduler=scheduler2,
    ).to(device)


    generator = torch.Generator(device=device if torch.cuda.is_available() else "cpu").manual_seed(args.seed) if args.seed else None


    n_saved_maps = 0
    n_saved_videos = 0
    n_saved_videos_2 = 0

    for idx, val_sample in enumerate(val_dataloader):
        videos = []
        real_videos = []
        q_videos = []
        bw_videos = []
        pkls = []
        init_images = []    
        naive_bw_batch = []

        pipeline_args['real_init_img'] = val_sample['init_img'].to(dtype=weight_dtype,device=device)
        #pipeline_args['real_init_img'] = F.interpolate(val_sample['init_img'].to(dtype=weight_dtype,device=device), size=(128, 192), mode='bilinear', align_corners=False, antialias=False)
        val_sample_segpred = to_segpred_batch(copy(val_sample))
        #pipeline_args['init_img'] = F.interpolate(val_sample_segpred['init_img'].to(dtype=weight_dtype,device=device), size=(128, 192), mode='bilinear', align_corners=False, antialias=False)
        pipeline_args['init_img'] = val_sample_segpred['init_img'].to(dtype=weight_dtype,device=device)
        pipeline_args['phase'] = move_to_device(val_sample_segpred['sem_info']['phase'],device)
        pipeline_args['triplet'] = move_to_device(val_sample_segpred['sem_info']['triplet'],device)
        pipeline_args['text_cond'] = args.text_cond

        with torch.no_grad():
            video = pipe(**pipeline_args, generator=generator, output_type="np").frames#[0]
            #video = np.zeros(shape=(args.val_batch_size, 16,256,384,3))
        q_video_batch = np.zeros_like(video)

        for i in range(video.shape[0]):
            print("real number of colors ", len(torch.unique(val_sample['video_segmap'][i])))
            q_video_batch[i] = quantize_video(video[i], k = None) #len(torch.unique(val_sample['video_segmap'][i])))
            #q_video_batch[i] = video[i]

            real_videos.append(val_sample['videos'][i])
            videos.append(video[i])
            q_videos.append(q_video_batch[i])      
            init_images.append(val_sample_segpred['init_img'][i])
        bw_video_batch, pkl_for_inference = invert_colormap_to_videos(q_video_batch)
        for i in range(video.shape[0]):
            bw_videos.append(bw_video_batch[i])
            #pkls.append(pkl_for_inference[i])              
            pkls.append(upscale_and_pad_video(pkl_for_inference[i], 256,384,0))              
                
        for i, (video_s, q_video_s, video_r, init_img, bw_video, pkl) in enumerate(zip(videos, q_videos, real_videos, init_images, bw_videos, pkls)):
            video_filename = os.path.join(args.output_dir, f"segpred_{n_saved_videos}.mp4")
            real_video_filename = os.path.join(args.output_dir, f"real_video_{n_saved_videos}.mp4")

            video_q_filename = os.path.join(args.output_dir, f"segpred_q_{n_saved_videos}.mp4")
            init_image_filename = os.path.join(args.output_dir, f"segpred_{n_saved_videos}.png")

            export_to_video(video_s, video_filename, fps=1)
            export_to_video(q_video_s, video_q_filename, fps=1)
            export_to_video((np.moveaxis(video_r.cpu().numpy(),0,-1)+1)/2, real_video_filename, fps=1 if args.max_num_frames == 17 else 8)

            if args.save_pkl:
                # Go BW
                video_bw_filename = os.path.join(args.output_dir, f"segpred_bw_{n_saved_videos}.mp4")
                video_nbw_filename = os.path.join(args.output_dir, f"segpred_nbw_{n_saved_videos}.mp4")
                export_to_video(np.repeat(np.expand_dims(np.squeeze(bw_video),axis=-1),3,-1), video_bw_filename, fps=1)
                # Compress and save

                # Resize to 480,720
                pkl_to_save = upscale_and_pad_video(pkl, 480,720,67)
                # Pad to 480,854
                compressed_pickle = blosc.compress(pkl)
                Path(os.path.join(args.output_dir,"q_pkl")).mkdir(parents=True, exist_ok=True)

                pkl_file_path = os.path.join(args.output_dir, f"q_pkl/{n_saved_maps:06d}.pkl")
                
                with open(pkl_file_path, 'wb') as f:
                    f.write(compressed_pickle)   

                # Also save the non quantized version using a simple color2bw script       
                Path(os.path.join(args.output_dir,"pkl")).mkdir(parents=True, exist_ok=True)
                naive_bw_video = np.zeros(shape=video_s.shape[:-1])  
                for i in range(video_s.shape[0]):
                    naive_bw_video[i] = naive_convert_to_bw(video_s[i])

                naive_bw_batch.append(torch.as_tensor(naive_bw_video))
                export_to_video(np.repeat(np.expand_dims(np.squeeze(naive_bw_video),axis=-1),3,-1), video_nbw_filename, fps=1)

                video_s = video_s /np.max(video)      
                pkl_to_save = upscale_and_pad_video(naive_bw_video, 480,720,67)
                compressed_pickle = blosc.compress(pkl_to_save)
                pkl_file_path = os.path.join(args.output_dir, f"pkl/{n_saved_maps:06d}.pkl")
                
                with open(pkl_file_path, 'wb') as f:
                    f.write(compressed_pickle)                   

                n_saved_maps += 1        

            init_image_cv2 = (((init_img.permute(1, 2, 0).float().cpu().numpy()+1)/2)*255).astype(np.uint8)
            Image.fromarray(init_image_cv2).save(init_image_filename) 
            n_saved_videos += 1 
        
        # Second step
        naive_bw_batch = torch.stack(naive_bw_batch)
        pipeline_args_2 = {
            "height": args.height,
            "width": args.width,
            "num_frames": args.max_num_frames,
            "out_frames": args.max_num_frames -1,
            "text_cond_model": text_cond_model,
            "num_inference_steps": args.num_inference_steps
        }
        pipeline_args_2['init_img'] = val_sample['init_img'].to(device).to(pipe2.dtype)
        #pipeline_args_2['segmap'] = torch.as_tensor(bw_video_batch).to(device).to(pipe2.dtype)
        #pipeline_args_2['segmap'] = naive_bw_batch.to(device).to(pipe2.dtype)                   
        norm = torch.as_tensor(pkl_for_inference).view(torch.as_tensor(pkl_for_inference).size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1)        
        pipeline_args_2['segmap'] = (torch.as_tensor(pkl_for_inference)/norm).to(device).to(pipe2.dtype)
        if args.max_num_frames == 49:
            pipeline_args_2['segmap'] = pipeline_args_2['segmap'][:,:6,...].contiguous()              
        gen_samples = pipe2(**pipeline_args_2, generator=generator, output_type="pt").frames      
        if args.output_dir is not None:
            # Iterate through batch dimension             
            for b in range(gen_samples.shape[0]):
                # Convert to uint8 format expected by video writer
                video = (gen_samples[b].permute(0,2,3,1).float().cpu().numpy() * 255).astype(np.uint8)
                out_path = os.path.join(args.output_dir, f"gen_video_{n_saved_videos_2:04d}.mp4")
                with imageio.get_writer(out_path, fps=1 if args.max_num_frames == 17 else 8, codec='libx264', quality=10) as writer:
                    for frame in video:
                        writer.append_data(frame)   
                n_saved_videos_2 += 1


if __name__ == "__main__":
    args = get_args()
    main(args)