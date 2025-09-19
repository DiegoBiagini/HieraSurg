# Copyright 2024 The CogView team, Tsinghua University & ZhipuAI and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import logging
import math
import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
from diffusers import AutoencoderKLCogVideoX, CogVideoXDPMScheduler, CogVideoXTransformer3DModel
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.optimization import get_scheduler
from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid
from diffusers.training_utils import (
    cast_training_params,
    free_memory,
)
from diffusers.utils import check_min_version, export_to_video, is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module

from finetune.dataset.utils_sd import get_dataloader_video, get_dataloader_videosegmap
from finetune.dataset.data_utils import move_tensors_to_device
import sys
import os
import cv2
import numpy as np
from PIL import Image
from copy import copy
import json 
# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from segcond.segpred_pipeline import CogVideoXPipeline_VideoSegPred
from argparse import Namespace
from segcond.cogvideo_segpred import CogVideoXTransformer3DModel_SegPred_I2V
from segcond.segpred_pipeline import CogVideoXPipeline_VideoSegPred_I2V_Img
from transformers import AutoTokenizer
import torch.nn.functional as F

def print_tensor_statistics(tensor):
    """
    Print various statistics of a PyTorch tensor.

    Args:
        tensor (torch.Tensor): The input tensor.
    """
    mean = torch.mean(tensor)
    std_dev = torch.std(tensor)
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    sum_val = torch.sum(tensor)
    median = torch.median(tensor)

    #print(f"Tensor: \n{tensor}")
    print(f"Mean: {mean.item()}")
    print(f"Standard Deviation: {std_dev.item()}")
    print(f"Minimum: {min_val.item()}")
    print(f"Maximum: {max_val.item()}")
    print(f"Sum: {sum_val.item()}")
    print(f"Median: {median.item()}")
"""
def to_segpred_batch(batch):
    # Take an original batch that contains videos, segmap, init_img and change it to the segpred objective
    # I.e. vidoes is now segmap, init_img is first of the segmap

    batch['videos'] = (torch.unsqueeze(batch['video_segmap'], dim=1).repeat(1,3,1,1,1)*2)-1
    batch['init_img'] = (torch.unsqueeze(batch['video_segmap'][:,0,...], dim=1).repeat(1,3,1,1)*2)-1
    del batch['video_segmap']
    return batch
"""
import matplotlib.pyplot as plt
def apply_colormap_to_videos(grayscale_videos, colormap_name='tab20'):
    """
    Apply a colormap to a batch of grayscale videos to generate colored videos.

    Parameters:
    - grayscale_videos: np.ndarray, the input batch of grayscale videos with shape (B, F, H, W).
    - colormap_name: str, the name of the matplotlib colormap to use.

    Returns:
    - colored_videos: np.ndarray, the output batch of colored videos with shape (B, F, H, W, 3) and values between -1 and 1.
    """
    B, F, H, W = grayscale_videos.shape
    colored_videos = np.zeros((B, F, H, W, 3), dtype=np.float32)
    cmap = plt.get_cmap(colormap_name)
    
    for b in range(B):
        unique_grayscale_levels = np.unique(grayscale_videos[b])
        color_palette = {level: (np.array(cmap(i / len(unique_grayscale_levels))[:3]) * 2 - 1).astype(np.float32)
                         for i, level in enumerate(unique_grayscale_levels)}
        
        for f in range(F):
            for grayscale_value, color in color_palette.items():
                colored_videos[b, f][grayscale_videos[b, f] == grayscale_value] = color
    
    return colored_videos

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
    
    for b in range(B):
        # Identify unique colors in the current video
        unique_colors = np.unique(colored_videos[b].reshape(-1, 3), axis=0)
        
        # Create a color palette for the current video
        color_palette = {tuple((np.array(cmap(i / len(unique_colors))[:3]) * 2 - 1).astype(np.float32)): i
                         for i in range(len(unique_colors))}
        
        for f in range(F):
            for color, grayscale_value in color_palette.items():
                mask = np.all(colored_videos[b, f] == color, axis=-1)
                grayscale_videos[b, f][mask] = grayscale_value
        grayscale_videos[b,...] = grayscale_videos[b,...]/np.max(grayscale_videos[b,...])
    
    return (grayscale_videos*2)-1

def to_segpred_batch(batch, color=True):
    # Take an original batch that contains videos, segmap, init_img and change it to the segpred objective
    # I.e. vidoes is now segmap, init_img is first of the segmap
    if color:
        og_dtype, og_device = batch['video_segmap'].dtype, batch['video_segmap'].device
        batch['real_init_img'] = batch['init_img']
        new_videos = torch.as_tensor(apply_colormap_to_videos(batch['video_segmap'].cpu().numpy()), dtype=og_dtype, device=og_device)
        batch['videos'] = torch.moveaxis(new_videos,-1,1)
        batch['init_img'] = batch['videos'][:,:,0,...]
    else:
        batch['real_init_img'] = batch['init_img']
        batch['videos'] = (torch.unsqueeze(batch['video_segmap'], dim=1).repeat(1,3,1,1,1)*2)-1
        batch['init_img'] = (torch.unsqueeze(batch['video_segmap'][:,0,...], dim=1).repeat(1,3,1,1)*2)-1        
    del batch['video_segmap']
    return batch


def print_trainable_parameters(model, print_trainable=True):
    """
    Prints the number of trainable or frozen parameters in the model and lists their names.
    
    Args:
        model: The model to analyze
        print_trainable (bool): If True, print trainable parameters. If False, print frozen parameters.
    """
    trainable_params = 0
    frozen_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
            if print_trainable:
                print(f"🔥 Trainable: {name} - {param.shape}")
        else:
            frozen_params += num_params
            if not print_trainable:
                print(f"❄️ Frozen: {name} - {param.shape}")
    
    if print_trainable:
        print(
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.2f}%"
        )
    else:
        print(
            f"frozen params: {frozen_params:,d} || all params: {all_param:,d} || frozen%: {100 * frozen_params / all_param:.2f}%"
        )

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0.dev0")

logger = get_logger(__name__)

def vae_encode(vae, frames_tensor):
    frames_tensor = frames_tensor.to(vae.dtype)
    with torch.no_grad():
        batching_latents = []
        batching_vae = []
        to_encode = frames_tensor.to(dtype=vae.dtype, device=vae.device)
        B, C, og_frames, H, W = to_encode.shape
        for b in range(B):
            # 16,3,1,...
            #to_encode = torch.moveaxis(to_encode,2,1).view(B*og_frames, C, H, W)
            sample = to_encode[b:b+1].permute(2,1,0,3,4)
            vae_out = vae.encode(sample)[0]
            out_latents = vae_out.sample() * vae.config.scaling_factor 
            batching_latents.append(torch.squeeze(out_latents))
            batching_vae.append(vae_out)
        return torch.stack(batching_latents,dim=0), batching_vae

def vae_decode(vae, latents):
    B,F,C,H,W = latents.shape
    videos = []
    for b in range(B):
        vid = torch.unsqueeze(latents[b,...],dim=1)
        vid = vid.permute(0, 2, 1, 3, 4)  # [batch_size, num_channels, num_frames, height, width]
        vid = 1 / vae.config.scaling_factor * vid

        decoded_vid = vae.decode(vid).sample
        videos.append(torch.squeeze(decoded_vid))
    video = torch.stack(videos).permute(0,2,3,4,1)
    return video
    

def vae_sample(dist_list, scaling_factor=1):
    batching_latents = []
    for dist in dist_list:
        out_latents = dist.sample() * scaling_factor
        batching_latents.append(torch.squeeze(out_latents))
    return torch.stack(batching_latents,dim=0)


def get_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script for CogVideoX.")
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
    # Training information
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")

    parser.add_argument(
        "--output_dir",
        type=str,
        default="vis",
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
        "--val_batch_size", type=int, default=1, help="Batch size (per device) for the validation dataloader."
    )    
    parser.add_argument("--segmap_dropout", type=float, default=0, help="Dropout rate for objects in the video_segmap.")

    parser.add_argument(
        "--text_cond",
        type=str,
        default="label_emb",
        choices=["label_emb", "SurgVLP"],
        help=(
            "Type of conditioning for the triplet/phase"
        ),
    )    
    parser.add_argument("--use_spawn", action="store_true", default=False, help="Whether or not to use spawn for data loading.")
    return parser.parse_args()

def save_video_frames_as_image(video_tensor, filename):
    # Convert the tensor to a NumPy array and transpose to (frames, height, width, channels)
    video_tensor = (video_tensor + 1) / 2
    
    # Convert the tensor to a NumPy array and transpose to (frames, height, width, channels)
    video_np = video_tensor.permute(1, 2, 3, 0).numpy()
    
    # Convert the range from [0, 1] to [0, 255]
    video_np = (video_np * 255).astype(np.uint8)
    
    # Create a blank image to hold the 4x4 grid of frames
    num_frames, frame_height, frame_width, num_channels = video_np.shape
    grid_height = 4 * frame_height
    grid_width = 4 * frame_width
    grid_image = np.zeros((grid_height, grid_width, num_channels), dtype=np.uint8)
    
    # Fill the grid image with the frames
    for i in range(num_frames):
        row = i // 4
        col = i % 4
        grid_image[row*frame_height:(row+1)*frame_height, col*frame_width:(col+1)*frame_width, :] = video_np[i]
    
    # Convert the NumPy array to a PIL Image and save it
    grid_image_pil = Image.fromarray(grid_image)
    grid_image_pil.save(filename)

def main(args):

    val_dataset_args = Namespace(data_path=args.val_data_path, image_size=(args.height,args.width), dataset_name="CholecT80VideoSegmapHiera", 
                             global_seed=args.seed, gpu_batch_size=args.val_batch_size, num_workers=args.dataloader_num_workers, video_length=16, 
                             vae=None, segmap_dropout=args.segmap_dropout, max_num_frames=args.max_num_frames)
    val_dataset_args.annotations_path = args.annotations_path
    val_dataset_args.text_cond = args.text_cond
    _, val_dataloader = get_dataloader_videosegmap(0, val_dataset_args, logger, precompute="compute_on_the_fly")    
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    j = 0
    textual_info = ""
    for step, batch in enumerate(val_dataloader):
        col_batch = to_segpred_batch(copy(batch))
        for i in range(batch['videos'].shape[0]):
            save_video_frames_as_image(batch['videos'][i], Path(args.output_dir)/f"{j:06}.png")
            save_video_frames_as_image(col_batch['videos'][i], Path(args.output_dir)/f"{j:06}_col.png")

            text_phase = str(tokenizer.batch_decode(batch['sem_info']['phase'][i]['input_ids'], skip_special_tokens=True))            
            text_triplets = str(tokenizer.batch_decode(batch['sem_info']['triplet'][i]['input_ids'], skip_special_tokens=True))          
            with open(Path(args.output_dir)/f"{j:06}.json", "wt") as f:
                json.dump({'phases':text_phase, 'triplets':text_triplets}, f)
            j+=1




if __name__ == "__main__":
    args = get_args()
    torch.multiprocessing.set_sharing_strategy("file_system")    
    if args.use_spawn:
        torch.multiprocessing.set_start_method('spawn', force=True)
    main(args)
