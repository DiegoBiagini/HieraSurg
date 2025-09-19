import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from diffusers import AutoencoderKLCogVideoX, CogVideoXDPMScheduler, CogVideoXTransformer3DModel
from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid
from tqdm.auto import tqdm

from finetune.dataset.utils_sd import get_dataloader_videosegmap
from argparse import Namespace
from segcond.cogvideo_segpred import CogVideoXTransformer3DModel_SegPred_I2V
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


logger = logging.getLogger(__name__)

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
        
        pkl_to_save = grayscale_videos.astype(np.uint8)
        grayscale_videos[b,...] = (grayscale_videos[b,...] / np.max(grayscale_videos[b,...]))
    
    return grayscale_videos, np.squeeze(pkl_to_save)

def find_optimal_k(colors, max_k=20):
    Ks = np.arange(8, max_k+1)
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

def quantize_video(video):
    # Reshape the image to a 2D array of pixels
    video = (video*255).astype(np.uint8)
    video = np.stack([cv2.cvtColor(video[i,...],cv2.COLOR_RGB2LAB) for i in range(video.shape[0])])
    F,H,W,C= video.shape
    pixels = video.reshape(-1, 3)

    # Extract unique colors
    unique_colors, unique_indices = np.unique(pixels, axis=0, return_inverse=True)

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
        help="Path to pretrained model or model identifier from huggingface.co/models.",
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
    device="cuda"
    if args.seed is not None:
        torch.manual_seed(args.seed)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    load_dtype = torch.bfloat16
    weight_dtype = torch.bfloat16
    transformer = CogVideoXTransformer3DModel_SegPred_I2V.from_pretrained(
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
    

    transformer.to("cuda" if torch.cuda.is_available() else "cpu", dtype=weight_dtype)
    vae.to("cuda" if torch.cuda.is_available() else "cpu")

    val_dataset_args = Namespace(data_path=args.val_data_path, image_size=(args.height,args.width), dataset_name="CholecT80VideoSegmapHiera", 
                             global_seed=args.seed, gpu_batch_size=args.val_batch_size, num_workers=args.dataloader_num_workers, video_length=16, 
                             vae=None, segmap_dropout=args.segmap_dropout, max_num_frames=args.max_num_frames)
    val_dataset_args.annotations_path = args.annotations_path
    val_dataset_args.text_cond = args.text_cond
    _, val_dataloader = get_dataloader_videosegmap(0, val_dataset_args, logger, precompute="compute_on_the_fly")    

    pipe = CogVideoXPipeline_VideoSegPred_I2V.from_pretrained(
        args.pretrained_model_name_or_path,
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
        torch_dtype=weight_dtype,
    ).to(device)

    pipeline_args = {
        "height": args.height,
        "width": args.width,
        "num_frames": args.max_num_frames,
        "text_cond_model": text_cond_model,
        "num_inference_steps": args.num_inference_steps,
        "latent_encoding": "full"
    }

    scheduler_args = {}

    if "variance_type" in pipe.scheduler.config:
        variance_type = pipe.scheduler.config.variance_type

        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"

        scheduler_args["variance_type"] = variance_type

    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, **scheduler_args)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(args.seed) if args.seed else None

    videos = []
    q_videos = []
    init_images = []
    n_saved_maps = 0

    for idx, val_sample in enumerate(val_dataloader):
        val_sample = to_segpred_batch(val_sample)
        pipeline_args['init_img'] = val_sample['init_img'].to(dtype=weight_dtype,device=device)
        pipeline_args['phase'] = val_sample['sem_info']['phase'].to(device)
        pipeline_args['triplet'] = val_sample['sem_info']['triplet'].to(device)
        pipeline_args['text_cond'] = args.text_cond

        with torch.no_grad():
            video = pipe(**pipeline_args, generator=generator, output_type="np").frames[0]

        quantized_video = quantize_video(video)
        videos.append(video)
        q_videos.append(quantized_video)
        init_images.append(val_sample['init_img'][0])
        
        for i, (video, q_video, init_img) in enumerate(zip(videos, q_videos, init_images)):
            video_filename = os.path.join(args.output_dir, f"segpred_{i}.mp4")
            video_q_filename = os.path.join(args.output_dir, f"segpred_q_{i}.mp4")
            init_image_filename = os.path.join(args.output_dir, f"segpred_{i}.png")

            export_to_video(video, video_filename, fps=1)
            export_to_video(q_video, video_q_filename, fps=1)

            if args.save_pkl:
                # Go BW
                bw_videos, pkl_to_save = invert_colormap_to_videos(np.expand_dims(q_video,axis=0))
                video_bw_filename = os.path.join(args.output_dir, f"segpred_bw_{i}.mp4")
                export_to_video(np.repeat(np.expand_dims(np.squeeze(bw_videos),axis=-1),3,-1), video_bw_filename, fps=1)
                # Compress and save

                # Resize to 480,720
                pkl_to_save = upscale_and_pad_video(pkl_to_save, 480,720,67)
                # Pad to 480,854
                compressed_pickle = blosc.compress(pkl_to_save)
                Path(os.path.join(args.output_dir,"q_pkl")).mkdir(parents=True, exist_ok=True)

                pkl_file_path = os.path.join(args.output_dir, f"q_pkl/{n_saved_maps:06d}.pkl")
                
                with open(pkl_file_path, 'wb') as f:
                    f.write(compressed_pickle)   

                # Also save the non quantized version using a simple color2bw script       
                Path(os.path.join(args.output_dir,"pkl")).mkdir(parents=True, exist_ok=True)
                naive_bw_video = np.zeros(shape=video.shape[:-1])  
                for i in range(video.shape[0]):
                    naive_bw_video[i] = naive_convert_to_bw(video[i])
                video = video /np.max(video)      
                pkl_to_save = upscale_and_pad_video(naive_bw_video, 480,720,67)
                compressed_pickle = blosc.compress(pkl_to_save)
                pkl_file_path = os.path.join(args.output_dir, f"pkl/{n_saved_maps:06d}.pkl")
                
                with open(pkl_file_path, 'wb') as f:
                    f.write(compressed_pickle)                   

                n_saved_maps += 1        

            init_image_cv2 = (((init_img.permute(1, 2, 0).float().cpu().numpy()+1)/2)*255).astype(np.uint8)
            Image.fromarray(init_image_cv2).save(init_image_filename)  

if __name__ == "__main__":
    args = get_args()
    main(args)