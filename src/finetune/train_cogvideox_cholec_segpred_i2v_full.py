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

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from segcond.segpred_pipeline import CogVideoXPipeline_VideoSegPred
from argparse import Namespace
from segcond.cogvideo_segpred import CogVideoXTransformer3DModel_SegPred_I2V
from segcond.segpred_pipeline import CogVideoXPipeline_VideoSegPred_I2V_Img
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

    # Model information
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    ) 
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )

    parser.add_argument(
        "--train_data_path",
        type=str,
        default=None,
        help=("A folder containing the training data."),
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

    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=50,
        help=(
            "Run validation every X epochs. Validation consists of running the prompt `args.validation_prompt` multiple times: `args.num_validation_videos`."
        ),
    )
    parser.add_argument(
        "--num_validation_videos",
        type=int,
        default=1,
        help="Number of videos to generate during validation.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=6,
        help="The guidance scale to use while sampling validation videos.",
    )
    parser.add_argument(
        "--use_dynamic_cfg",
        action="store_true",
        default=False,
        help="Whether or not to use the default cosine dynamic guidance schedule when sampling validation videos.",
    )

    # Training information
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")

    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
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
        "--validation_prompt",
        type=str,
        default=None,
        help="Prompt to use for validation videos.",
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip videos horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--val_batch_size", type=int, default=1, help="Batch size (per device) for the validation dataloader."
    )    
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides `--num_train_epochs`.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=1,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--enable_slicing",
        action="store_true",
        default=False,
        help="Whether or not to use VAE slicing for saving memory.",
    )
    parser.add_argument(
        "--enable_tiling",
        action="store_true",
        default=False,
        help="Whether or not to use VAE tiling for saving memory.",
    )
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm.")
    parser.add_argument("--segmap_dropout", type=float, default=0, help="Dropout rate for objects in the video_segmap.")
    parser.add_argument("--full_finetune", action="store_true", default=False, help="Whether or not to finetune the entire model.")

    parser.add_argument(
        "--text_cond",
        type=str,
        default="label_emb",
        choices=["label_emb", "SurgVLP"],
        help=(
            "Type of conditioning for the triplet/phase"
        ),
    )    
    parser.add_argument(
        "--add_noise_i2v",
        action="store_true", default=False, help="Whether to add noise before encoding through the VAE or not."        
    )

    # Optimizer
    parser.add_argument(
        "--optimizer",
        type=lambda s: s.lower(),
        default="adam",
        choices=["adam", "adamw", "prodigy"],
        help=("The optimizer type to use."),
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.95, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="The epsilon parameter for the Adam and Prodigy optimizers.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-4, help="The weight decay parameter for the Adam and Prodigy optimizers.")
    # Other information
    parser.add_argument("--tracker_name", type=str, default=None, help="Project tracker name")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="Directory where logs are stored.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default=None,
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--use_spawn", action="store_true", default=False, help="Whether or not to use spawn for data loading.")
    return parser.parse_args()



def log_validation(
    val_dataloader,
    pipe,
    args,
    accelerator,
    pipeline_args,
    epoch,
    is_final_validation: bool = False,
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_videos} videos."
    )
    # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
    scheduler_args = {}

    if "variance_type" in pipe.scheduler.config:
        variance_type = pipe.scheduler.config.variance_type

        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"

        scheduler_args["variance_type"] = variance_type

    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, **scheduler_args)
    pipe = pipe.to(accelerator.device)
    # pipe.set_progress_bar_config(disable=True)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None

    for tracker in accelerator.trackers:
        videos = []
        init_images = []  # List to store the init images
        for idx, val_sample in enumerate(val_dataloader):
            # Add the other pipeline args
            pipeline_args['real_init_img'] = val_sample['init_img']
            val_sample = to_segpred_batch(val_sample)
            pipeline_args['init_img'] = val_sample['init_img']
            pipeline_args['phase'] = val_sample['sem_info']['phase']
            pipeline_args['triplet'] = val_sample['sem_info']['triplet']
            pipeline_args['text_cond'] = args.text_cond

            with torch.no_grad():
                video = pipe(**pipeline_args, generator=generator, output_type="np").frames[0]

            videos.append(video)
            init_images.append(val_sample['init_img'][0])  # Store the init image
            
            phase_name = "test" if is_final_validation else "validation"
            if tracker.name == "wandb":
                video_filenames = []
                init_image_filenames = []  # List to store the filenames of init images
                
                for i, (video, init_img) in enumerate(zip(videos, init_images)):
                    video_filename = os.path.join(args.output_dir, f"{phase_name}_video_{i}.mp4")
                    init_image_filename = os.path.join(args.output_dir, f"{phase_name}_init_img_{i}.png")  # Save init image as PNG

                    export_to_video(video, video_filename, fps=1)
                    # to (16,256,384,3)

                    # Save the init image
                    init_image_cv2 = (((init_img.permute(1, 2, 0).float().cpu().numpy()+1)/2)*255).astype(np.uint8)                  
                    #init_image_cv2 = cv2.cvtColor(init_image_cv2, cv2.COLOR_RGB2BGR)                    
                    #init_img = init_img.cpu().numpy()  # Convert to numpy if needed
                    #init_img = (init_img * 255).astype(np.uint8)  # Scale to [0, 255] if necessary
                    Image.fromarray(init_image_cv2).save(init_image_filename)  # Save as PNG

                    video_filenames.append(video_filename)
                    init_image_filenames.append(init_image_filename)  # Store the filename of the init image

                tracker.log(
                    {
                        phase_name: [
                            wandb.Video(filename, caption=f"{i}")
                            for i, filename in enumerate(video_filenames)
                        ],
                        f"{phase_name}_init_img": [
                            wandb.Image(init_image_filename, caption=f"Init Image {i}")  # Log the init image
                            for i, init_image_filename in enumerate(init_image_filenames)
                        ],
                    }
                )

            if idx == args.num_validation_videos - 1:
                break            


    free_memory()
def prepare_rotary_positional_embeddings(
    height: int,
    width: int,
    num_frames: int,
    vae_scale_factor_spatial: int = 8,
    patch_size: int = 2,
    attention_head_dim: int = 64,
    device: Optional[torch.device] = None,
    base_height: int = 480,
    base_width: int = 720,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grid_height = height // (vae_scale_factor_spatial * patch_size)
    grid_width = width // (vae_scale_factor_spatial * patch_size)
    base_size_width = base_width // (vae_scale_factor_spatial * patch_size)
    base_size_height = base_height // (vae_scale_factor_spatial * patch_size)

    grid_crops_coords = get_resize_crop_region_for_grid((grid_height, grid_width), base_size_width, base_size_height)
    freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
        embed_dim=attention_head_dim,
        crops_coords=grid_crops_coords,
        grid_size=(grid_height, grid_width),
        temporal_size=num_frames,
    )

    freqs_cos = freqs_cos.to(device=device)
    freqs_sin = freqs_sin.to(device=device)
    return freqs_cos, freqs_sin


def get_optimizer(args, params_to_optimize, use_deepspeed: bool = False):
    # Use DeepSpeed optimzer
    if use_deepspeed:
        from accelerate.utils import DummyOptim

        return DummyOptim(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )

    # Optimizer creation
    supported_optimizers = ["adam", "adamw", "prodigy"]
    if args.optimizer not in supported_optimizers:
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}. Supported optimizers include {supported_optimizers}. Defaulting to AdamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and not (args.optimizer.lower() not in ["adam", "adamw"]):
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'Adam' or 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

    if args.optimizer.lower() == "adamw":
        optimizer_class = bnb.optim.AdamW8bit if args.use_8bit_adam else torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )
    elif args.optimizer.lower() == "adam":
        optimizer_class = bnb.optim.Adam8bit if args.use_8bit_adam else torch.optim.Adam

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )
    elif args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    return optimizer


def main(args):

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)


    # Prepare models and scheduler

    # CogVideoX-2b weights are stored in float16
    # CogVideoX-5b and CogVideoX-5b-I2V weights are stored in bfloat16
    load_dtype = torch.bfloat16 if "5b" in args.pretrained_model_name_or_path.lower() else torch.float16
    transformer_og = CogVideoXTransformer3DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=load_dtype,
        revision=args.revision,
        variant=args.variant,
    )
    config = transformer_og.config

    config['text_cond'] = args.text_cond 
    config['in_channels'] = 48       
    transformer = CogVideoXTransformer3DModel_SegPred_I2V.from_config(config) #_SegPred.from_config(config)
    
    # Load weights from transformer_og to transformer for matching parameter names
    transformer_state_dict = transformer.state_dict()
    for name, param in transformer_og.state_dict().items():
        if name in transformer_state_dict:
            try:
                transformer_state_dict[name].copy_(param)
            except:
                print(f"Couldnt copy param {name}, is this intended?")
    transformer.load_state_dict(transformer_state_dict)

    vae = AutoencoderKLCogVideoX.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant, torch_dtype=torch.float32
    )

    scheduler = CogVideoXDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    if args.enable_slicing:
        vae.enable_slicing()
    if args.enable_tiling:
        vae.enable_tiling()

    vae.requires_grad_(False)

    transformer.requires_grad_(True)
 
        #print(name, param.requires_grad)

    # Add the parameter printing here
    
    if accelerator.is_main_process:
        logger.info("=== Trainable parameters ===")
        print_trainable_parameters(transformer, print_trainable=False)
    

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.state.deepspeed_plugin:
        # DeepSpeed is handling precision, use what's in the DeepSpeed config
        if (
            "fp16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["fp16"]["enabled"]
        ):
            weight_dtype = torch.float16
        if (
            "bf16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["bf16"]["enabled"]
        ):
            weight_dtype = torch.float16
    else:
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )


    transformer.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()


    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                if isinstance(model, type(unwrap_model(transformer))):
                    model.save_pretrained(os.path.join(output_dir, "transformer"))
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")
                
                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

    def load_model_hook(models, input_dir):
        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(unwrap_model(transformer))):
                # Load the transformer model
                model_path = os.path.join(input_dir, "transformer")
                if os.path.exists(model_path):
                    model.from_pretrained(model_path)
                else:
                    logger.warning(f"No transformer model found in {model_path}. Skipping loading.")
            else:
                raise ValueError(f"Unexpected save model: {model.__class__}")

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            # only upcast trainable parameters into fp32
            cast_training_params([model])

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        # only upcast trainable parameters (LoRA) into fp32L
        cast_training_params([transformer], dtype=torch.float32)

    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    # Optimization parameters
    transformer_parameters_with_lr = {"params": transformer_lora_parameters, "lr": args.learning_rate}
    params_to_optimize = [transformer_parameters_with_lr]

    use_deepspeed_optimizer = (
        accelerator.state.deepspeed_plugin is not None
        and "optimizer" in accelerator.state.deepspeed_plugin.deepspeed_config
    )
    use_deepspeed_scheduler = (
        accelerator.state.deepspeed_plugin is not None
        and "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    )

    optimizer = get_optimizer(args, params_to_optimize, use_deepspeed=use_deepspeed_optimizer)

    # Dataset and DataLoader@

    train_dataset_args = Namespace(data_path=args.train_data_path, image_size=(args.height,args.width), dataset_name="CholecT80VideoSegmapHiera", 
                             global_seed=args.seed, gpu_batch_size=args.train_batch_size, num_workers=args.dataloader_num_workers, video_length=16, vae=None, segmap_dropout=args.segmap_dropout, max_num_frames=args.max_num_frames)
    train_dataset_args.annotations_path = args.annotations_path
    train_dataset_args.text_cond = args.text_cond

    _, train_dataloader = get_dataloader_videosegmap(accelerator.local_process_index, train_dataset_args, logger)
    val_dataset_args = Namespace(data_path=args.val_data_path, image_size=(args.height,args.width), dataset_name="CholecT80VideoSegmapHiera", 
                             global_seed=args.seed, gpu_batch_size=args.val_batch_size, num_workers=args.dataloader_num_workers, video_length=16, 
                             vae=None, segmap_dropout=args.segmap_dropout, max_num_frames=args.max_num_frames)
    val_dataset_args.annotations_path = args.annotations_path
    val_dataset_args.text_cond = args.text_cond
    _, val_dataloader = get_dataloader_videosegmap(accelerator.local_process_index, val_dataset_args, logger, precompute="compute_on_the_fly")    


    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if use_deepspeed_scheduler:
        from accelerate.utils import DummyScheduler

        lr_scheduler = DummyScheduler(
            name=args.lr_scheduler,
            optimizer=optimizer,
            total_num_steps=args.max_train_steps * accelerator.num_processes,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        )
    else:
        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            num_cycles=args.lr_num_cycles,
            power=args.lr_power,
        )

    # Prepare everything with our `accelerator`.
    transformer, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = args.tracker_name or "cogvideox-segmap"
        accelerator.init_trackers(tracker_name, config=vars(args))

    text_cond_model = {}

    if args.text_cond == "SurgVLP":
        import surgvlp
        from mmengine.config import Config
        configs = Config.fromfile('./segcond/surgvlp_cfgs/config_peskavlp.py')['config']
        # Change the config file to load different models: config_surgvlp.py / config_hecvl.py / config_peskavlp.py

        model, preprocess = surgvlp.load(configs.model_config, device=accelerator.device)
                
        text_cond_model['name'], text_cond_model['model'], text_cond_model['preprocessor'] = args.text_cond, model, preprocess
    
    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_trainable_parameters = sum(param.numel() for model in params_to_optimize for param in model["params"])

    logger.info("***** Running training *****")
    logger.info(f"  Num trainable parameters = {num_trainable_parameters}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if not args.resume_from_checkpoint:
        initial_global_step = 0
    else:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    vae_scale_factor_spatial = 2 ** (len(vae.config.block_out_channels) - 1)

    # For DeepSpeed training
    model_config = transformer.module.config if hasattr(transformer, "module") else transformer.config

    latents_cache = []
    img_latents_cache = []
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()

        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer]
            batch = to_segpred_batch(batch)
            with accelerator.accumulate(models_to_accumulate):
                # Input: B C F H W
                if 'latents' in batch:
                    sample = batch["latents"]          
                else:
                    if len(latents_cache) <= step:
                        # Compute latents and add to cache
                        video = batch['videos']
                        sample, to_cache = vae_encode(vae, video)

                        for i,dist in enumerate(to_cache):
                            to_cache[i] = move_tensors_to_device(dist, 'cpu')

                        latents_cache.append(to_cache)

                        images = batch["init_img"]
                        images_real = batch["real_init_img"]

                        # Add frame dimension to images [B,C,H,W] -> [B,C,F,H,W]
                        images = images.unsqueeze(2).to(dtype=weight_dtype)
                        images_real = images_real.unsqueeze(2).to(dtype=weight_dtype)
                        if args.add_noise_i2v:
                            # Add noise to images OR DO NOT
                            image_noise_sigma = torch.normal(mean=-3.0, std=0.5, size=(1,), device=accelerator.device)
                            image_noise_sigma = torch.exp(image_noise_sigma).to(dtype=images.dtype)

                            images = images + torch.randn_like(images) * image_noise_sigma[:, None, None, None, None]
                        #image_latent_dist = vae.encode(images.to(dtype=vae.dtype)).latent_dist
                        #image_latents = image_latent_dist.sample() * vae.config.scaling_factor
                        image_latents, image_latent_dist = vae_encode(vae, images.to(dtype=vae.dtype))
                        image_real_latents, image_real_latent_dist = vae_encode(vae, images_real.to(dtype=vae.dtype))

                        img_latents_cache.append((move_tensors_to_device(image_latent_dist, 'cpu'), move_tensors_to_device(image_real_latent_dist, 'cpu')))
                    else:
                        cached_dist = latents_cache[step]
                        for i,dist in enumerate(cached_dist):
                            cached_dist[i] = move_tensors_to_device(dist, accelerator.device)                        
                        sample = vae_sample(cached_dist, scaling_factor=vae.config.scaling_factor) 

                        for i,dist in enumerate(cached_dist):
                            cached_dist[i] = move_tensors_to_device(dist, 'cpu')
                        
                        image_latent_dist, image_real_latent_dist = img_latents_cache[step]
                        image_latents = vae_sample(move_tensors_to_device(image_latent_dist, accelerator.device), scaling_factor=vae.config.scaling_factor) 
                        image_real_latents= vae_sample(move_tensors_to_device(image_real_latent_dist, accelerator.device) , scaling_factor=vae.config.scaling_factor) 

                        img_latents_cache[i] = (move_tensors_to_device(image_latent_dist, 'cpu'), move_tensors_to_device(image_real_latent_dist, 'cpu'))

                latent = sample.to(dtype=weight_dtype)#.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
                patch_size_t = getattr(model_config, 'patch_size_t', None)
                if patch_size_t is not None:
                    ncopy = latent.shape[2] % patch_size_t
                    # Copy the first frame ncopy times to match patch_size_t
                    first_frame = latent[:, :, :1, :, :]  # Get first frame [B, C, 1, H, W]
                    latent = torch.cat([first_frame.repeat(1, 1, ncopy, 1, 1), latent], dim=2)
                    assert latent.shape[2] % patch_size_t == 0

                batch_size, num_channels, num_frames, height, width = latent.shape

                # Sample a random timestep for each sample
                timesteps = torch.randint(
                    0, scheduler.config.num_train_timesteps, (batch_size,), device=accelerator.device
                )
                timesteps = timesteps.long()

                # from [B, C, F, H, W] to [B, F, C, H, W]
                #latent = latent.permute(0, 2, 1, 3, 4)
                image_latents = torch.unsqueeze(image_latents, dim=2)
                image_latents = image_latents.permute(0, 2, 1, 3, 4)
                assert (latent.shape[0], *latent.shape[2:]) == (image_latents.shape[0], *image_latents.shape[2:])

                # Padding image_latents to the same frame number as latent
                padding_shape = (latent.shape[0], latent.shape[1] - 1, *latent.shape[2:])
                latent_padding = image_latents.new_zeros(padding_shape)
                image_latents = torch.cat([image_latents, latent_padding], dim=1)

                # Same for the images_real
                image_real_latents = torch.unsqueeze(image_real_latents, dim=2)
                image_real_latents = image_real_latents.permute(0, 2, 1, 3, 4)
                assert (latent.shape[0], *latent.shape[2:]) == (image_real_latents.shape[0], *image_real_latents.shape[2:])

                # Padding image_latents to the same frame number as latent
                latent_padding = image_real_latents.new_zeros(padding_shape)
                image_real_latents = torch.cat([image_real_latents, latent_padding], dim=1)

                # Add noise to latent
                noise = torch.randn_like(latent)
                latent_noisy = scheduler.add_noise(latent, noise, timesteps)

                # Concatenate latent and image_latents in the channel dimension
                latent_img_noisy = torch.cat([latent_noisy, image_latents, image_real_latents], dim=2)

                # Prepare rotary embeds
                vae_scale_factor_spatial = 2 ** (len(vae.config.block_out_channels) - 1)

                rotary_emb = (
                    prepare_rotary_positional_embeddings(
                        height=height * vae_scale_factor_spatial,
                        width=width * vae_scale_factor_spatial,
                        num_frames=num_frames,
                        transformer_config=model_config,
                        vae_scale_factor_spatial=vae_scale_factor_spatial,
                        device=accelerator.device,
                    )
                    if model_config.use_rotary_positional_embeddings
                    else None
                )

                # Predict the noise residual
                if args.text_cond == "label_emb":
                    model_output = transformer(
                        hidden_states=latent_img_noisy,
                        phase_emb=batch['sem_info']['phase'],
                        triplet_emb=batch['sem_info']['triplet'],
                        timestep=timesteps,
                        image_rotary_emb=rotary_emb,
                        return_dict=False,
                    )[0]
                elif args.text_cond == "SurgVLP":
                    # Get an embedding for each batch and restack them
                    with torch.no_grad():
                        surgvlp_embs_phase = torch.stack([text_cond_model['model'](inputs_text=text_sample, mode='text')['text_emb'] 
                                            for text_sample in batch['sem_info']['phase']])
                        surgvlp_embs_triplet = torch.stack([text_cond_model['model'](inputs_text=text_sample, mode='text')['text_emb'] 
                                                for text_sample in batch['sem_info']['triplet']])
                    model_output = transformer(
                        hidden_states=latent_img_noisy,
                        phase_emb=surgvlp_embs_phase,
                        triplet_emb=surgvlp_embs_triplet,
                        timestep=timesteps,
                        image_rotary_emb=rotary_emb,
                        return_dict=False,
                    )[0]                        

                model_pred = scheduler.get_velocity(model_output, latent_noisy, timesteps)

                alphas_cumprod = scheduler.alphas_cumprod[timesteps]
                weights = 1 / (1 - alphas_cumprod)
                while len(weights.shape) < len(model_pred.shape):
                    weights = weights.unsqueeze(-1)

                target = latent

                loss = torch.mean((weights * (model_pred - target) ** 2).reshape(batch_size, -1), dim=1)
                loss = loss.mean()

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    params_to_clip = transformer.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                if accelerator.state.deepspeed_plugin is None:
                    optimizer.step()
                    optimizer.zero_grad()

                lr_scheduler.step()


            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)

                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"Removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            if (epoch + 1) % args.validation_epochs == 0:
                # Create pipeline
                pipe = CogVideoXPipeline_VideoSegPred_I2V_Img.from_pretrained(
                    args.pretrained_model_name_or_path,
                    transformer=unwrap_model(transformer),
                    vae=unwrap_model(vae),
                    scheduler=scheduler,
                    torch_dtype=weight_dtype,
                )


                pipeline_args = {
                    "height": args.height,
                    "width": args.width,
                    "num_frames": args.max_num_frames,
                    "text_cond_model": text_cond_model,
                    "latent_encoding": "slice"
                }
                # Get a few validation samples from the val_dataloader
                try:
                    validation_outputs = log_validation(
                        val_dataloader=val_dataloader,
                        pipe=pipe,
                        args=args,
                        accelerator=accelerator,
                        pipeline_args=pipeline_args,
                        epoch=epoch,
                    )
                except:
                    print("Could not visualize results")

    accelerator.wait_for_everyone()

    accelerator.end_training()


if __name__ == "__main__":
    args = get_args()
    torch.multiprocessing.set_sharing_strategy("file_system")    
    if args.use_spawn:
        torch.multiprocessing.set_start_method('spawn', force=True)
    main(args)
