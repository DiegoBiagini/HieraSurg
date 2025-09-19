import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
import copy
from pathlib import Path
from typing import List
import numpy as np
import multiprocessing as mp
import ctypes
from multiprocessing import shared_memory
import blosc


class VideoFrameDatasetCholec80FPS(Dataset):
    def __init__(self,
                 data_root,
                 resolution: List,
                 dataset_name: str,
                 video_length,  # clip length
                 annotation_dir=None,
                 spatial_transform="",
                 subset_split="",
                 frame_stride=1,
                 clip_step=None,
                 precompute="load_precomputed",  # One of load_precomputed, compute_on_the_fly, compute_and_save, compute_and_cache
                 vae=None,
                 pad_to=49,
                 sample_cache=None,
                 horizontal_flip=True,
                 vertical_flip=True,
                 fps=8,
                 sequence_length=48
                 ):
        self.fps=fps
        self.sequence_length=sequence_length
        self.loader = self.default_loader
        self.video_length = video_length
        self.spatial_transform = spatial_transform
        self.frame_stride = frame_stride
        self.dataset_name = dataset_name
        self.precompute = precompute

        self.resize_shape = resolution
        self.video_files = [os.path.join(data_root, f) for f in os.listdir(data_root) if f.endswith('.mp4')]
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])        
        self.sequences = self._preload_frames()

        self.vae = vae

        print('[VideoFrameDatasetCholec80] number of videos:', len(self.video_files))
        print('[VideoFrameDatasetCholec80] number of clips', len(self.sequences))

        # data transform
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip

        mirror_transforms = []
        if self.horizontal_flip:
            mirror_transforms.append(transforms.RandomHorizontalFlip(p=0.3))
        if self.vertical_flip:
            mirror_transforms.append(transforms.RandomVerticalFlip(p=0.3))


        if self.spatial_transform == "crop_resize":
            print('Spatial transform: crop and then resize')
            self.video_transform = transforms.Compose([
                *mirror_transforms,
                transforms.Lambda(cropper),
                transforms.Resize(resolution),
            ])
        elif self.spatial_transform == "pad_resize":
            self.video_transform = transforms.Compose([
                *mirror_transforms,
                transforms.Pad(padding=(0, 40), fill=-1),  # Padding the image from (480, 854) to (560, 854)
                transforms.CenterCrop(size=(560, 840)),  # Cropping the padded image to (560, 840)
                transforms.Resize(size=resolution, antialias=True)  # Finally resizing to required resolution
            ])
        elif self.spatial_transform == "resize":
            print('Spatial transform: resize with no crop')
            self.video_transform = transforms.Compose([
                *mirror_transforms,
                transforms.Resize((resolution[0], resolution[1]))
            ])
        elif self.spatial_transform == "random_crop":
            self.video_transform = transforms.Compose([
                *mirror_transforms,
                transforms.RandomCrop(resolution),
            ])
        elif self.spatial_transform == "":
            self.video_transform = transforms.Compose(mirror_transforms) if mirror_transforms else None
        else:
            raise NotImplementedError

        if sample_cache is not None:
            self.sample_cache = sample_cache
        else:
            self.sample_cache = {}
        self.pad_to = pad_to

        # Precompute samples if required
        if self.precompute == "compute_and_save":
            self.precomputed_samples = []
            self._compute_and_save_samples(save_dir=Path(data_root) / f"precomputed_samples_{subset_split}")
        elif self.precompute == "load_precomputed":
            self.precomputed_samples = []
            self._load_precomputed_samples(load_dir=Path(data_root) / f"precomputed_samples_{subset_split}")

    def _resize_frame(self, frame):
        return cv2.resize(frame, self.resize_shape)

    def _preload_frames(self):
        sequences = []
        for video_file in self.video_files:
            cap = cv2.VideoCapture(video_file)
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = int(video_fps // self.fps)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print("Video file: frames ", total_frames)
            num_sequences = total_frames // (frame_interval)

            frames = []
            for frame_num in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_num % frame_interval == 0:
                    frame = self.img_transform(frame)
                    frames.append(frame)

            for seq_num in range(len(frames)):
                start_idx = seq_num * self.fps
                end_idx = start_idx + self.sequence_length
                sequence = frames[start_idx:end_idx]
                if len(sequence) == self.sequence_length:
                    sequence_array = np.array(sequence, dtype=np.uint8)
                    compressed_sequence = blosc.compress(sequence_array)
                    sequences.append(compressed_sequence)

            cap.release()
            print("Loaded ", video_file)

        return sequences
    
    def _print_memory_usage(self):
        total_memory = self.shared_memory.nbytes
        total_memory_mb = total_memory / (1024 * 1024)
        print(f"Total memory used by sequences: {total_memory_mb:.2f} MB")

    def __len__(self):
        return len(self.shared_memory)
    def decompress_sequence(self, compressed_sequence):
        decompressed_sequence = blosc.decompress(compressed_sequence)
        sequence_array = np.frombuffer(decompressed_sequence, dtype=np.uint8)
        sequence_array = sequence_array.reshape((self.sequence_length, *sequence_array.shape[1:]))
        return sequence_array
    def _compute_sample(self, index):
                
        frames = self.decompress_sequence(self.sequences[index])

        frames_tensor = torch.cat(frames, 1)  # Concatenate frames (c,t,h,w)
        del frames  # Release memory for frames
        
        if self.video_transform:
            frames_tensor = self.video_transform(frames_tensor)

        if self.pad_to != self.video_length:
            c,t,h,w = frames_tensor.shape
 
            frames_tensor = torch.cat((frames_tensor, torch.full(size=(c,self.pad_to-t,h,w), fill_value=-1)), dim=1)

        example = {}
        if self.vae is None:
            example['videos'] = frames_tensor  
        else:           
            raise NotImplementedError

        example["frame_stride"] = self.frame_stride

        return example    

    def _compute_and_cache_sample(self, index):
        # Check if sample exists in cache
        if hasattr(self, 'sample_cache') and index in self.sample_cache:
            return self.sample_cache[index]

        # If not in cache, compute the sample
        example = self._compute_sample(index)

        # Initialize cache if it doesn't exist
        if not hasattr(self, 'sample_cache'):
            self.sample_cache = {}
            
        to_save = copy.deepcopy(example)
        if 'videos' in to_save:
            del to_save['videos']

        self.sample_cache[index] = to_save
        return example    


    def default_loader(self, path):
        return cv2.imread(path)


    def _compute_and_save_samples(self, save_dir):
        raise NotImplementedError()

    def _load_precomputed_samples(self, load_dir):
        raise NotImplementedError()
from torchvision.transforms.functional import crop
def cropper(img):
    top, left, bottom, right = 0, 67, 480, 787
    if img is None:
        return top, left, bottom, right
    return crop(img, top, left, bottom, right)
