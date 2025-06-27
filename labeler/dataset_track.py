
import os
import torch

import argparse
from sam2.build_sam import build_sam2_video_predictor
from tracker_utils import filter_keypoints, show_mask, refine_masks, get_sam_keypoints, add_bg_black_masks
import cv2
import numpy as np
import random
import pickle
import hashlib

import argparse
import matplotlib.pyplot as plt
import matplotlib
from radio_utils import load_model
from einops import rearrange
from collections import OrderedDict
from PIL import Image

matplotlib.use('Agg')

def sam_fast_init_state(
    images,
    sam_model,
    offload_video_to_cpu=False,
    offload_state_to_cpu=False,
):
    """Initialize an inference state."""
    compute_device = sam_model.device  # device of the model

    inference_state = {}

    img_mean=(0.485, 0.456, 0.406)
    img_std=(0.229, 0.224, 0.225)
    img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
    img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]
    image_size = sam_model.image_size
    processed_images = np.array([cv2.resize(frame, (image_size, image_size)) for frame in images])/255.0
    processed_images_tensor = torch.tensor(processed_images).permute(0, 3, 1, 2)
    processed_images_normalized = (processed_images_tensor - img_mean) / img_std
    processed_images = processed_images_normalized
    inference_state["images"] = processed_images.to(compute_device)
    inference_state["num_frames"] = len(images)
    # whether to offload the video frames to CPU memory
    # turning on this option saves the GPU memory with only a very small overhead
    inference_state["offload_video_to_cpu"] = offload_video_to_cpu
    # whether to offload the inference state to CPU memory
    # turning on this option saves the GPU memory at the cost of a lower tracking fps
    # (e.g. in a test case of 768x768 model, fps dropped from 27 to 24 when tracking one object
    # and from 24 to 21 when tracking two objects)
    inference_state["offload_state_to_cpu"] = offload_state_to_cpu
    # the original video height and width, used for resizing final output scores
    inference_state["video_height"] = images[0].shape[0]
    inference_state["video_width"] = images[0].shape[1]
    inference_state["device"] = compute_device
    if offload_state_to_cpu:
        inference_state["storage_device"] = torch.device("cpu")
    else:
        inference_state["storage_device"] = compute_device
    # inputs on each frame
    inference_state["point_inputs_per_obj"] = {}
    inference_state["mask_inputs_per_obj"] = {}
    inference_state["frames_tracked_per_obj"] = {}
    # visual features on a small number of recently visited frames for quick interactions
    inference_state["cached_features"] = {}
    # values that don't change across frames (so we only need to hold one copy of them)
    inference_state["constants"] = {}
    # mapping between client-side object id and model-side object index
    inference_state["obj_id_to_idx"] = OrderedDict()
    inference_state["obj_idx_to_id"] = OrderedDict()
    inference_state["obj_ids"] = []
    # A storage to hold the model's tracking results and states on each frame
    inference_state["output_dict"] = {
        "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
        "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
    }
    # Slice (view) of each object tracking results, sharing the same memory with "output_dict"
    inference_state["output_dict_per_obj"] = {}
    # A temporary storage to hold new outputs when user interact with a frame
    # to add clicks or mask (it's merged into "output_dict" before propagation starts)
    inference_state["temp_output_dict_per_obj"] = {}
    # Frames that already holds consolidated outputs from click or mask inputs
    # (we directly use their consolidated outputs during tracking)
    inference_state["consolidated_frame_inds"] = {
        "cond_frame_outputs": set(),  # set containing frame indices
        "non_cond_frame_outputs": set(),  # set containing frame indices
    }
    # metadata for each tracking frame (e.g. which direction it's tracked)
    inference_state["tracking_has_started"] = False
    inference_state["frames_already_tracked"] = {}
    # Warm up the visual backbone and cache the image feature on frame 0
    # sam_model._get_image_feature(inference_state, frame_idx=0, batch_size=1)
    for i in range(len(images)):    
        sam_model._get_image_feature(inference_state, frame_idx=i, batch_size=8)    
    return inference_state

def load_radio_model():
    from transformers import AutoModel, CLIPImageProcessor
    hf_repo = "nvidia/RADIO-L"  # For RADIO-L.
    image_processor = CLIPImageProcessor.from_pretrained(hf_repo)

    model_version="radio_v2.5-l" # for RADIOv2.5-L model (ViT-L/16)

    model, preprocessor, info = load_model(model_version, vitdet_window_size=None, adaptor_names=['clip', 'sam'],
                                           torchhub_repo="NVlabs/RADIO", use_local_lib=True)
    if "e-radio" in model_version:
        model.model.set_optimal_window_size((256, 448)) #where it expects a tuple of (height, width) of the input image.

    model = model.cuda()
    return model, image_processor

def predict_features_for_frame(model, image_processor, frame):
    enc_type = ['backbone', 'clip', 'sam']
    out_height, out_width = frame.shape[0], (frame.shape[1] + 15) // 16 * 16

    frame = cv2.resize(frame, (out_width, out_height))  # Resize the frame to the target dimensions
    pad_height = out_height - frame.shape[0]
    pad_width = out_width - frame.shape[1]
    frame = np.pad(frame, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant', constant_values=0)

    pixel_values = image_processor(images=frame, return_tensors='pt', do_resize=False, do_center_crop=True, 
                                   crop_size={'height': out_height, 'width': out_width}).pixel_values.cuda()

    out = model(pixel_values)
    feature_dict = {}
    for k in enc_type:
        features = out[k].features
        patch_size = 16
        n_rows, n_cols = out_height // patch_size, out_width // patch_size
        features = rearrange(features, 'b (h w) c -> b h w c', h=n_rows, w=n_cols).float()
        feature_dict[k] = features

    return feature_dict, torch.squeeze(pixel_values, dim=0)

# Load SAM2
def load_sam_model(sam_weights_path, weights_filename='sam2.1_hiera_large.pt', model_cfg='configs/sam2.1/sam2.1_hiera_l.yaml'):
    """
    Load the SAM model for segmentation.

    Args:
        sam_weights_path (str): Path to the folder containing the SAM model weights.
        weights_filename (str): Name of the weights file for the SAM model. Default is 'sam2.1_hiera_large.pt'.
        model_cfg (str): Path to the configuration file for the SAM model. Default is 'configs/sam2.1/sam2.1_hiera_l.yaml'.

    Returns:
        predictor: The initialized SAM model predictor.
    """
    checkpoint_path = os.path.join(sam_weights_path, weights_filename)
    predictor = build_sam2_video_predictor(model_cfg, checkpoint_path)
    return predictor


def quick_vis(masks):
    return np.argmax(np.stack([masks[i] for i in sorted(masks.keys())], axis=0), axis=0)[np.newaxis, :, :]

import copy
def pipeline(sam_model, clip_frames, radio_model, image_processor, automask_cache=None, gladio_feature_cache=None, visualize=False, points_per_side=16):
    """
    Function to run the detection pipeline
    Args:
        sam_model: The initialized SAM model for segmentation.
        clip_frames: A list of frames to process through the pipeline.
        radio_model: The loaded radio model for feature extraction.
        image_processor: The image processor for the radio model.
        automask_cache: A dictionary to cache automasks for frames.
        visualize: A boolean flag to determine whether to compute visualization results.
        points_per_side: The number of points to sample for SAM automask.
    Returns:
        vis_output: The visualization output for the input frames (if visualize is True).
        out_masks: The segmentation masks for the input frames.
    """

    results_list = [] 
    feature_groups = {}

    markers_stride = 1

    if automask_cache is None:
        automask_cache = {}
    if gladio_feature_cache is None:
        gladio_feature_cache = {}

    for frame_idx, frame in enumerate(clip_frames):
        if frame_idx % markers_stride != 0:
            continue

        frame_hash = hashlib.sha256(pickle.dumps(frame, protocol=1)).hexdigest()
        if frame_hash in automask_cache:
            automasks = automask_cache[frame_hash]
            feature_dict = gladio_feature_cache[frame_hash]

        else:
            try:
                automasks = get_sam_keypoints(frame, sam_model, return_masks=True, points_per_side=points_per_side)
            except:
                automasks = []

            # Get radio features of stitched frames(choose the correct stitching
            frame_for_features = frame # stitched_frames[frame_idx//total_stitched_frames]
            feature_dict, pixel_values = predict_features_for_frame(radio_model, image_processor, frame_for_features)
            feature_dict = feature_dict['clip'].detach().cpu()

            automask_cache[frame_hash] = automasks
            gladio_feature_cache[frame_hash] = feature_dict

        # Sample 4 points from each mask in automasks
        n_samples = 4
        sampled_points = []
        updated_automasks = []     
        for i, mask in enumerate(automasks):
            # Get the indices of True values in the mask
            true_indices = np.argwhere(mask.squeeze())
            
            # If there are less than 4 True values, use all of them
            if len(true_indices) <= n_samples:
                sampled_points.append((i, true_indices))
                updated_automasks.append(mask)
            else:
                # Randomly sample 4 points
                from sklearn.cluster import KMeans

                # Use KMeans clustering to spread points evenly
                kmeans = KMeans(n_clusters=n_samples, random_state=0).fit(true_indices)
                sampled_indices = kmeans.cluster_centers_.astype(int)
                kp = sampled_indices / frame.shape[:2]
                kp = filter_keypoints([kp] , frame)   
                if len(kp) > 0: 
                    sampled_points.append((i, kp))
                    updated_automasks.append(mask)
        automasks = updated_automasks

        # Convert sampled points to the same format as keypoints
        keypoints = [points[1] for points in sampled_points]
        keypoints_idx = [points[0] for points in sampled_points]

        # And then of points  
        for i, kp_group in enumerate(keypoints):
            patch_size = 16
            def map_pixel_to_feat(pixel_coords):
                x = pixel_coords[0]
                y = pixel_coords[1]
                return int(y//patch_size), int(x//patch_size)
            def map_pixel_to_feat_stitched(pixel_coords, single_image_size, stitch_idx):
                h, w = single_image_size
                ph, pw = int(h//patch_size), int(w//patch_size)
                x = pixel_coords[0]
                y = pixel_coords[1]
                return int(y//patch_size) + ph*(stitch_idx//2), int(x//patch_size) + pw*(stitch_idx%2)
            
            feat_indices = []
            
            for kp in kp_group:
                patch_pos = map_pixel_to_feat((kp[1]*frame.shape[1], kp[0]*frame.shape[0]))
                # patch_pos = map_pixel_to_feat_stitched((kp[1]*frame.shape[1], kp[0]*frame.shape[0]), frame.shape[:2], frame_idx%total_stitched_frames)

                if patch_pos not in feat_indices:
                    feat_indices.append(patch_pos)

            # Collect embeddings and possibly average
            points_features = torch.stack([feature_dict[0, patch_pos[0], patch_pos[1], :] for patch_pos in feat_indices])
            points_features = points_features.mean(dim=0)

            # Add to feature groups
            if len(feature_groups) == 0:
                feature_groups[0] =points_features
                keypoints_idx[i] = 0
            else:
                min_dist = 1
                closest_feature_group = -1
                for idx in feature_groups:
                    dist = torch.nn.functional.mse_loss(feature_groups[idx], points_features)
                    if dist < min_dist and dist < 1e-4:
                        min_dist = dist
                        closest_feature_group = idx
                if closest_feature_group != -1:
                    keypoints_idx[i] = closest_feature_group
                    feature_groups[closest_feature_group] = (feature_groups[closest_feature_group] + points_features) / 2
                else:
                    # If no match, add new feature
                    new_idx = len(feature_groups)
                    feature_groups[new_idx] = points_features
                    keypoints_idx[i] = new_idx

        new_kps = []
        new_kps_idx = []
        for i, kp_group in enumerate(keypoints):
            new_kps.extend(kp_group)
            new_kps_idx.extend([keypoints_idx[i]] * len(kp_group))

        # Store the results in a dictionary and append to the results list
        results_list.append({
            'keypoints': new_kps,
            'keypoints_idx': new_kps_idx,
            'masks': automasks,
            'masks_idx': keypoints_idx,
        })

    processed_frame_markers = []  # List to store results for each frame

    for idx, res in enumerate(results_list):
        processed_markers = {}  # Initialize an empty dictionary for the current frame


        # Group keypoints that are close together
        processed_markers['grouped_keypoints'] = res['keypoints']
        processed_markers['grouped_keypoints_idx'] = res['keypoints_idx']

        processed_markers['masks'] = res['masks']
        processed_markers['masks_idx'] = res['masks_idx']

        # Add the frame result to the results list
        processed_frame_markers.append(processed_markers)
    
    # Perform sam
    
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = sam_fast_init_state(clip_frames, sam_model)
        video_segments = {i: None for i in range(len(clip_frames))}
        
        for frame_idx, frame in enumerate(clip_frames):
            for rep in range(2):
                sam_model.reset_state(state)

                if frame_idx % markers_stride != 0:
                    continue    
                else:
                    s_frame_idx = frame_idx // markers_stride

                processed_markers = processed_frame_markers[s_frame_idx]

                # Collect bounding boxes and keypoints
                grouped_keypoints = processed_markers['grouped_keypoints']  # CAREFUL, possibly wrong point format for SAM  
                grouped_keypoints_idx = processed_markers['grouped_keypoints_idx']  

                masks = processed_markers['masks']
                masks_idx = processed_markers['masks_idx']

                if len(masks) == 0:
                    continue

                for idx, mask in enumerate(masks):
                    frame_idx, object_ids, masks = sam_model.add_new_mask(
                        state, 
                        frame_idx=frame_idx, 
                        obj_id=masks_idx[idx],
                        mask=np.squeeze(mask)
                    )  

                # Propagate the prompts to get masklets throughout the video
                for out_frame_idx, out_obj_ids, out_mask_logits in sam_model.propagate_in_video(state, 
                                                                                                reverse=True if rep == 1 else False):
                    if video_segments[out_frame_idx] is None:
                        video_segments[out_frame_idx] = {}
                    
                    for i, out_obj_id in enumerate(out_obj_ids):
                        video_segments[out_frame_idx][out_obj_id] = (out_mask_logits[i] > 0.0).cpu().numpy()    

        refined_masks = refine_masks(copy.deepcopy(video_segments.copy()))
        final_masks = add_bg_black_masks(copy.deepcopy(refined_masks.copy()), clip_frames)
        if visualize:
            grid_images = []
            original_images = []

            for out_frame_idx, masks in final_masks.items():
                plt.figure(figsize=(6, 4))
                plt.title(f"Frame {out_frame_idx}")
                plt.imshow(clip_frames[out_frame_idx])  # Assuming video_frames contains the actual frames
                for out_obj_id, out_mask in masks.items():
                    show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
                
                # Instead of saving and reloading, directly append the figure to grid_images
                plt.gcf().canvas.draw()
                grid_images.append(np.array(plt.gcf().canvas.buffer_rgba()))  # Convert the current figure to a numpy array and append to the grid_images list
                plt.close()

                # Append the original image to original_images
                original_images.append(clip_frames[out_frame_idx])

            # Create a grid of all images
            if grid_images:
                grid_width = 4
                grid_height = 4
                
                image_height, image_width = grid_images[0].shape[:2]
                total_width = grid_width * image_width
                total_height = grid_height * image_height

                grid_image = Image.new('RGB', (total_width, total_height))

                for i, im in enumerate(grid_images):
                    im = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_RGBA2RGB))                 
                    x_offset = (i % grid_width) * image_width
                    y_offset = (i // grid_width) * image_height
                    grid_image.paste(im, (x_offset, y_offset))

            # Create a grid of original images
            if original_images:
                image_height, image_width = original_images[0].shape[:2]
                total_width = grid_width * image_width
                total_height = grid_height * image_height

                original_grid_image = Image.new('RGB', (total_width, total_height))
                for i, im in enumerate(original_images):
                    im = Image.fromarray(im)
                    x_offset = (i % grid_width) * image_width
                    y_offset = (i // grid_width) * image_height
                    original_grid_image.paste(im, (x_offset, y_offset))

            plt.close("all")

            return grid_image, original_grid_image, final_masks
        else:
            return None, None, final_masks

import os
import io
from tqdm import tqdm
from itertools import islice
from contextlib import redirect_stdout, redirect_stderr
def label_dataset(sam_model, dataset_folder, visualize=False):
    torch.manual_seed(0)  # Set the seed for PyTorch
    np.random.seed(0)     # Set the seed for NumPy
    random.seed(0)        # Set the seed for the random module

    radio_model, image_processor = load_radio_model()

    def extract_16_frame_clips(image_files):
        for i in range(0, len(image_files)):
            yield image_files[i:i+16]

    for subfolder in os.listdir(dataset_folder):
        print(f"Processing subfolder: {subfolder}")
        subfolder_path = os.path.join(dataset_folder, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        # Create output folders for this subfolder
        if visualize:
            vis_output_folder = os.path.join(dataset_folder, f"{subfolder}_visualization10")
            os.makedirs(vis_output_folder, exist_ok=True)
        data_output_folder = os.path.join(dataset_folder, f"{subfolder}_masks")
        os.makedirs(data_output_folder, exist_ok=True)

        # Get all image files in the subfolder
        image_files = sorted([f for f in os.listdir(subfolder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        automask_cache = {}
        gladio_feature_cache = {}
        # Extract 16-frame clips
        total_clips = len(list(extract_16_frame_clips(image_files)))
        for clip_idx, clip in enumerate(tqdm(extract_16_frame_clips(image_files), total=total_clips, desc=f"Processing {subfolder}")):
            start_frame_name = os.path.splitext(os.path.basename(clip[0]))[0]
            clip_frames = [cv2.cvtColor(cv2.imread(os.path.join(subfolder_path, img)), cv2.COLOR_BGR2RGB) for img in clip]
            
            # Call the pipeline function for each clip
            # Redirect both stdout and stderr to devnull to suppress all output
            with open(os.devnull, 'w') as devnull:
                with redirect_stdout(devnull), redirect_stderr(devnull):
                    output_masks_image, output_original_image, output_masks = pipeline(sam_model, clip_frames, radio_model, image_processor, 
                                                                                       automask_cache, gladio_feature_cache, visualize, points_per_side=8)

            if visualize:
                # Save the output image
                output_masks_image_path = os.path.join(vis_output_folder, f"{start_frame_name}.png")
                output_masks_image.save(output_masks_image_path)

                output_original_image_path = os.path.join(vis_output_folder, f"{start_frame_name}_og.png")
                output_original_image.save(output_original_image_path)

            # Save the output masks as a single pickle file
            masks_output_path = os.path.join(data_output_folder, f"{start_frame_name}.pkl")
            with open(masks_output_path, 'wb') as f:
                pickle.dump(output_masks, f)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triplet Predictor Configuration")
    parser.add_argument('--sam_weights_folder', type=str, required=True, help='Path to the folder containing the saved embeddings and indices')
    parser.add_argument('--dataset_folder', type=str, required=True, help='Path to the folder containing a dataset')
    parser.add_argument('--visualize', action='store_true', help='Flag to enable visualization')
    parser.add_argument('--cuda_device', type=int, default=0, help='CUDA device number to use')

    args = parser.parse_args()

    # Set CUDA device before any GPU operations
    #os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    device = torch.device(f"cuda:{args.cuda_device}")
    torch.cuda.set_device(device)

    sam_model = load_sam_model(sam_weights_path=args.sam_weights_folder, weights_filename='sam2.1_hiera_large.pt', model_cfg='configs/sam2.1/sam2.1_hiera_l.yaml')
    sam_model = sam_model.to(device)  # Move model to specified device
    print("SAM model initialized successfully.")

    label_dataset(sam_model, dataset_folder=args.dataset_folder, visualize=args.visualize)


