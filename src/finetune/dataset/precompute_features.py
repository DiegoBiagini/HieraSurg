import os
import shutil
from pathlib import Path
from tqdm import tqdm
import torch
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor
from diffusers import AutoencoderKLCogVideoX

from frame_dataset import default_loader, make_dataset_cholec, load_and_transform_frames_cholec, IMG_EXTENSIONS

def compute_and_save_features(data_root, resolution, video_length, subset_split, clip_step, vae, spatial_transform="", gpu_id=0):
    # Setup paths
    split_file = os.path.join(data_root, f"{subset_split}_split.txt")
    with open(split_file, "r") as f:
        video_names = f.readlines()
    video_paths = [os.path.join(data_root, "videos", v.strip()) for v in video_names]
    annotation_dir = os.path.join(data_root, "labels")

    # Create dataset
    clips, videos, mappings = make_dataset_cholec(
        video_paths_list=video_paths, annotations_path=annotation_dir, nframes=video_length, clip_step=clip_step, skip_empty=False)
    
    if len(clips) == 0:
        raise RuntimeError("Found 0 clips. Supported image extensions are: " + ",".join(IMG_EXTENSIONS))

    # Split dataset based on GPU ID
    total_clips = len(clips)
    mid_point = total_clips // 2
    if gpu_id == 0:
        clips = clips[:mid_point]
    elif gpu_id == 1:
        clips = clips[mid_point:]
    else:
        raise ValueError("Invalid GPU ID. Must be 0 or 1.")

    # Data transform
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Video transform
    if spatial_transform == "crop_resize":
        from torchvision.transforms.functional import crop
        def cropper(img):
            top, left, bottom, right = 0, 67, 480, 787
            if img is None:
                return top, left, bottom, right
            return crop(img, top, left, bottom, right)
        video_transform = transforms.Compose([
            transforms.Lambda(cropper),
            transforms.Resize(resolution),
        ])
    elif spatial_transform == "pad_resize":
        video_transform = transforms.Compose([
            transforms.Pad(padding=(0, 40), fill=-1),
            transforms.CenterCrop(size=(560, 840)),
            transforms.Resize(size=resolution, antialias=True)
        ])
    elif spatial_transform == "resize":
        video_transform = transforms.Resize((resolution, resolution))
    elif spatial_transform == "random_crop":
        video_transform = transforms.Compose([
            transforms_video.RandomCropVideo(resolution),
        ])
    else:
        video_transform = None

    # Precompute samples
    save_dir = Path(data_root) / f"precomputed_samples_{subset_split}"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    batch_size = 16
    frame_cache = {}

    device = f'cuda:{gpu_id}'
    vae.to(device)

    for i in tqdm(range(0, len(clips), batch_size), desc=f"Precomputing and saving samples (GPU {gpu_id})"):
        batch_clips = clips[i:i+batch_size]
        batch_frames = []
        batch_labels = []

        def process_clip(clip):
            frames, labels = load_and_transform_frames_cholec(clip, mappings, default_loader, img_transform, frame_cache)
            frames = torch.cat(frames, 1)
            if video_transform is not None:
                frames = video_transform(frames)
            return frames, labels

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(process_clip, batch_clips))

        for frames, labels in results:
            batch_frames.append(frames)
            batch_labels.append(labels)

        batch_frames_tensor = torch.stack(batch_frames).to(dtype=vae.dtype, device=device)
        
        with torch.no_grad():
            batch_latents = vae.encode(batch_frames_tensor).latent_dist.sample() * vae.config.scaling_factor

        for j, (frames, labels, latents) in enumerate(zip(batch_frames, batch_labels, batch_latents)):
            example = {
                "latents": latents.to(torch.float32).cpu(),
                "frame_info": labels,
                "clip_info": batch_clips[j]
            }

            img_path_parts = batch_clips[j][0]["img_path"].split('/')
            clip_name = f"{img_path_parts[-2]}_{img_path_parts[-1].split('.')[0]}"
            save_path = os.path.join(save_dir, f"{clip_name}.pt")
            torch.save(example, save_path)

        # Clear CUDA cache to prevent OOM
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Example usage
    data_root = "/mnt/m2/Projects/SG2VID/datasets/CholecT50"
    resolution = [480, 720]
    video_length = 16
    subset_split = "train"
    clip_step = 1

    vae = AutoencoderKLCogVideoX.from_pretrained(
        "/mnt/m2/Projects/SG2VID/CogVideo/models/CogvideoX2b/", subfolder="vae"
    ).to(torch.bfloat16)
    vae.enable_slicing()
    vae.enable_tiling()
    vae.requires_grad_(False)

    compute_and_save_features(data_root, resolution, video_length, subset_split, clip_step, 
                              vae, spatial_transform="crop_resize", gpu_id=1)
    #compute_and_save_features(data_root, resolution, video_length, subset_split, clip_step, 
    #                          vae, spatial_transform="crop_resize", gpu_id=1)
