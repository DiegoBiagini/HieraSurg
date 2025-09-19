import torch
import torch.utils.data as data
from torchvision import transforms
from torchvision.transforms.functional import crop
import os
import shutil
import random
from typing import List
from finetune.dataset.frame_dataset import masks_to_tensor, load_and_transform_frames_cholec80, inpaint_object_in_masks, make_dataset_cholec80, default_loader
from multiprocessing import Manager
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp']
from pathlib import Path
import copy
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video
from finetune.dataset.frame_dataset import make_dataset_cholec, cropper, VideoFrameDatasetCholecSegmap
import pickle
import blosc
from finetune.dataset.frame_dataset import PairedRandomFlip
import re
from transformers import AutoTokenizer

def tokenize_surgvlp(
    text,
    tokenizer_clinical,
    padding: str = 'max_length',
    max_length: int = 77,
    truncation: bool = True,
    device: str = 'cpu'
):
    ixtoword = {v: k for k, v in tokenizer_clinical.get_vocab().items()}

    if isinstance(text, str):
        text = [text]

    processed_text_tensors = []
    for t in text:
        text_tensors = tokenizer_clinical(
            t,
            return_tensors="pt",
            truncation=truncation,
            padding=padding,
            max_length=max_length,
        )
        text_tensors["sent"] = [ixtoword[ix] for ix in text_tensors["input_ids"][0].tolist()]
        processed_text_tensors.append(text_tensors)

    caption_ids = torch.stack([x["input_ids"] for x in processed_text_tensors])
    attention_mask = torch.stack([x["attention_mask"] for x in processed_text_tensors])
    token_type_ids = torch.stack([x["token_type_ids"] for x in processed_text_tensors])

    # Squeeze if only one text
    if len(text) == 1:
        caption_ids = caption_ids.squeeze(0).to(device)
        attention_mask = attention_mask.squeeze(0).to(device)
        token_type_ids = token_type_ids.squeeze(0).to(device)
    else:
        caption_ids = caption_ids.squeeze().to(device)
        attention_mask = attention_mask.squeeze().to(device)
        token_type_ids = token_type_ids.squeeze().to(device)

    cap_lens = [len([w for w in txt if not w.startswith("[")]) for txt in text]

    return {
        "input_ids": caption_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "cap_lens": cap_lens,
    }

def videosegmap_collate_fn_hiera(batch):    
    # Batch the frame_info
    batched_sem_info = {}
    if 'sem_info' in batch[0]:
        for k in batch[0]['sem_info']:
            batching_values = []
            for item in batch:
                batching_values.append(item['sem_info'][k])
            try:
                batched_sem_info[k] = torch.stack(batching_values)
            except:
                batched_sem_info[k] = batching_values # In case of textual emb, keep the batch as a list

    batched_frame_stride = []
    if 'frame_stride' in batch[0]:
        for item in batch:
            batched_frame_stride.append(item['frame_stride'])

    batched_clip_info = []
    for item in batch:
        batched_clip_info.append(item['clip_info'])
        
    batched_segmaps = []
    for item in batch:
        batched_segmaps.append(item['video_segmap'])
    batched_segmaps = torch.stack(batched_segmaps)
    # Include init_img if present
    batched_init_img = None
    if 'init_img' in batch[0]:
        try:
            batched_init_img = torch.stack([item['init_img'] for item in batch])
        except KeyError:
            # Get first item's init_img as default
            default_init_img = batch[0]['init_img']
            # Replace missing init_imgs with default
            init_imgs = [item.get('init_img', default_init_img) for item in batch]
            batched_init_img = torch.stack(init_imgs)

    if 'latents' in batch[0]:
        raise NotImplementedError
    elif 'videos' in batch[0]:
        try:
            batched_videos = torch.stack([item['videos'] for item in batch])
        except KeyError:
            # Get first item's init_img as default
            default_videos = batch[0]['videos']
            # Replace missing init_imgs with default
            videos = [item.get('videos', default_videos) for item in batch]
            batched_videos = torch.stack(videos)        

        return {
            'videos': batched_videos,
            'frame_stride': batched_frame_stride,
            'clip_info': batched_clip_info,
            'video_segmap': batched_segmaps,
            'init_img': batched_init_img,
            'sem_info': batched_sem_info,
        }
    return {
        'frame_stride': batched_frame_stride,
        'clip_info': batched_clip_info,
        'video_segmap': batched_segmaps,
        'init_img': batched_init_img,
        'sem_info': batched_sem_info,
    }

def load_and_transform_frames_cholecsegmap(frame_list, mappings, loader, img_transform=None, frame_cache = {}, is_segmap_compressed = False, fps=None):
    assert (isinstance(frame_list, list))
    clip = []
    cache_frames = frame_cache != {}
    labels = []

    # Define a transformation that takes relative coordinates of a bbox
    # And considers the image transformation to warp it to the correct position
    def transform_bbox(bbox, img_w, img_h):
        return bbox

    for frame in frame_list:        
        fpath, annotations = frame["img_path"], frame['annotations']
        
        _, f_id = os.path.split(fpath)
        f_id = int(os.path.splitext(f_id)[0])
        _, v_id = os.path.split(_)
        if cache_frames:
            if fpath in frame_cache:
                img = frame_cache[fpath]
            else:
                img = loader(fpath)
                frame_cache[fpath] = img
        else:
            img = loader(fpath)            
        if img_transform is not None:
            img = img_transform(img)
        img = img.view(img.size(0), 1, img.size(1), img.size(2))

        # Transform the annotation to be readable, separating the various parts
        labels_frame = []
        for ann in annotations:
            label = {
                "triplet_id": ann[0],
                "triplet": mappings['triplet'][str(int(ann[0]))] if ann[0] != -1 else "",
                "instrument_id": ann[1],
                "instrument": mappings['instrument'][str(int(ann[1]))] if ann[1] != -1 else "",
                "instrument_conf": ann[2],
                "instrument_rel_bbox": ann[3:7],
                "instrument_bbox": transform_bbox(ann[3:7], img.size(2), img.size(3)), # TODO make sure they are w h
                "target_id": ann[8],
                "target": mappings['target'][str(int(ann[8]))] if ann[8] != -1 else "",
                "target_conf": ann[9],
                "target_rel_bbox": ann[10:14],
                "target_bbox": transform_bbox(ann[10:14], img.size(2), img.size(3)), # TODO make sure they are w h
                "verb_id": ann[7],
                "verb": mappings['verb'][str(int(ann[7]))] if ann[7] != -1 else "",
                "phase_id": ann[14],
                "phase": mappings['phase'][str(int(ann[14]))] if ann[14] != -1 else "",    

                "vid_id": v_id,
                "frame_id": f_id
            }
            labels_frame.append(label)

        labels.append(labels_frame)
        clip.append(img)
    # Load the segmap file
    if not fps is None:
        #segmap_path = Path("/mnt/data1/Cholec80/frames1fps/Done/") / f"{Path(frame_list[0]['img_path']).parent.stem}_masks"
        segmap_path = Path(frame_list[0]['img_path']).parent.parent / f"{Path(frame_list[0]['img_path']).parent.stem}_masks"

        corr_frame = int(int(frame_list[0]['img_path'].split('/')[-1].split('.')[0])/fps)
        corr_frame_path = f"{corr_frame:06}.pkl"
        segmap_path = segmap_path/corr_frame_path
    else:
        segmap_path = Path(frame_list[0]['img_path']).parent.parent / f"{Path(frame_list[0]['img_path']).parent.stem}_masks"
        segmap_path = segmap_path/frame_list[0]['img_path'].split('/')[-1].replace('.jpg', '.pkl')

    if is_segmap_compressed:
        try:            
            with open(segmap_path, 'rb') as f:
                compressed_pickle = f.read()
            segmap_masks = pickle.loads(blosc.decompress(compressed_pickle))          
        except:
            print(f"Error loading/decompressing segmap for {segmap_path}")
            segmap_masks = None
            return clip, None, None
    else:
        segmap_masks = pickle.load(open(segmap_path, 'rb'))
    if not fps is None:
        segmap_new_length = len(frame_list)//fps
        segmap_masks = {k:v for k,v in segmap_masks.items() if k < segmap_new_length}

    return clip, segmap_masks, labels

import surgvlp

class VideoFrameDatasetCholecSegmapHiera(VideoFrameDatasetCholecSegmap):
    def __init__(self,
                 data_root,
                 resolution : List,
                 video_length,  # clip length
                 dataset_name="",
                 spatial_transform="",
                 frame_stride=1,
                 clip_step=None,
                 precompute="load_precomputed",  # One of load_precomputed, compute_on_the_fly, compute_and_save
                 vae=None,
                 subset_split="",
                 segmap_dropout=0,
                 load_compressed_segmap=False,
                 sample_cache=None,
                 lock=None,
                 pad_to = 49,
                 annotations_path=None,
                 vertical_flip=False,
                 horizontal_flip=False,
                 text_cond = 'label_emb',
                 ):
        self.loader = default_loader
        self.video_length = video_length
        self.spatial_transform = spatial_transform
        self.frame_stride = frame_stride
        self.dataset_name = dataset_name
        self.precompute = precompute 
        self.segmap_dropout = segmap_dropout
        self.clip_step = clip_step
        self.text_cond = text_cond

        manager = Manager()
        assert subset_split in ["train", "val", "test", ""]
        #  discard videos that are not in the annotation path
        annotated_videos = [str(Path(f).stem).replace('VID','video') for f in os.listdir(annotations_path)]
        video_paths = [os.path.join(data_root, folder) for folder in os.listdir(data_root) if folder.startswith("video") and folder[5:].isdigit()]
        video_paths = [p for p in video_paths if Path(p).stem in annotated_videos]

        segmap_path = data_root        
        regex_pattern = "^video\d+_masks$"
        segmap_videos = [d for d in os.listdir(segmap_path) if os.path.isdir(os.path.join(segmap_path, d)) and re.match(regex_pattern, d)]
        segmap_videos = [f.replace('_masks','') for f in segmap_videos]

        video_paths = [p for p in video_paths if Path(p).stem in segmap_videos]           
        self.clips, self.videos , self.mappings = make_dataset_cholec(
            video_paths_list=video_paths, annotations_path=annotations_path, nframes=video_length, clip_step=clip_step, skip_empty=False)
        assert (len(self.clips[0]) == video_length), f"Invalid clip length = {len(self.clips[0])}"

        self.clips = manager.list(self.clips)
        self.videos = manager.list(self.videos)
        self.vae = vae

        print('[VideoFrameDatasetSegMapHiera] number of videos:', len(self.videos))
        print('[VideoFrameDatasetSegMapHiera] number of clips', len(self.clips))

        # check data
        if len(self.clips) == 0:
            raise (RuntimeError(f"Found 0 clips. \n"
                                "Supported image extensions are: " +
                                ",".join(IMG_EXTENSIONS)))

        self.vertical_flip_prob = 0.3 if vertical_flip else 0
        self.horizontal_flip_prob = 0.3 if horizontal_flip else 0

        # data transform
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        if self.spatial_transform == "crop_resize":
            print('Spatial transform:crop and then resize')
            self.video_transform = transforms.Compose([
                transforms.Lambda(cropper),
                transforms.Resize(size=resolution, antialias=True)  # Finally resizing to required resolution
            ])
            self.segmap_transform = transforms.Compose([
                transforms.Lambda(cropper),
                transforms.Resize(size=resolution, antialias=False, interpolation=transforms.InterpolationMode.NEAREST)  # Finally resizing to required resolution
            ])            
        elif self.spatial_transform == "pad_resize":
            raise NotImplementedError
        elif self.spatial_transform == "resize":
            raise NotImplementedError
        elif self.spatial_transform == "random_crop":
            raise NotImplementedError
        else:
            raise NotImplementedError

        self.load_compressed_segmap = load_compressed_segmap
        if sample_cache is not None:    
            self.sample_cache = sample_cache
        else:
            self.sample_cache = {}
        self.lock = lock
        self.pad_to = pad_to
        if self.text_cond == "SurgVLP":
            self.tokenizer_clinical = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

        # Precompute samples if required
        if self.precompute == "compute_and_save":
            self.precomputed_samples = []
            self._compute_and_save_samples(save_dir=Path(data_root)/f"precomputed_samples_{subset_split}")
        elif self.precompute == "load_precomputed":
            self.precomputed_samples = []
            self._load_precomputed_samples(load_dir=Path(data_root)/f"precomputed_samples_{subset_split}")

    def paired_transform(self, frames_tensor, segmap, video_transform, segmap_transform):
        paired_flip_transform = PairedRandomFlip(horizontal_flip_prob=self.horizontal_flip_prob, vertical_flip_prob=self.vertical_flip_prob)
        frames_tensor, segmap = paired_flip_transform(frames_tensor, segmap)
        frames_tensor = video_transform(frames_tensor)
        segmap = segmap_transform(segmap)
        return frames_tensor, segmap
    
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
        to_save['video_segmap'] = to_save['video_segmap'].to(torch.int8)
        del to_save['init_img'] # Already cached features

        if self.lock is not None:
            with self.lock:
                self.sample_cache[index] = to_save
        else:
            self.sample_cache[index] = to_save
        return example    

    def _compute_sample(self, index):
        clip = self.clips[index]
        
        frames, segmap, labels = load_and_transform_frames_cholecsegmap(clip, self.mappings, self.loader, self.img_transform, is_segmap_compressed=self.load_compressed_segmap, fps=self.clip_step)
        #frames, segmap = load_and_transform_frames_cholec80(clip, self.loader, self.img_transform, is_segmap_compressed=self.load_compressed_segmap, fps=self.clip_step if self.clip_step != 1 else None)        
        assert (len(frames) == self.video_length), f'clip_length={len(frames)}, target={self.video_length}, {clip}'
        if segmap is None:
            # Choose another random clip
            return self.__getitem__((index+1)%len(self))
        
        if random.random() < self.segmap_dropout:
            objects = list(segmap[0].keys())
            # Choose a random object to drop(except the last two)
            if len(objects) > 2:
                object_to_drop = random.choice(objects[:-2])
                
                # Remove it from the images
                frames = inpaint_object_in_masks(frames, segmap, object_to_drop)

                # Remove it from the segmap
                for i in range(len(segmap)):
                    del segmap[i][object_to_drop]

                    # Shift the keys of the dictionary after removal
                    new_segmap = {}
                    new_key = 0
                    for key in sorted(segmap[i].keys()):
                        new_segmap[new_key] = segmap[i][key]
                        new_key += 1
                    segmap[i] = new_segmap
        segmap = masks_to_tensor(segmap)
        frames_tensor = torch.cat(frames, 1)  # Concatenate frames (c,t,h,w)
        del frames  # Release memory for frames
        
        if self.video_transform:
            frames_tensor, segmap = self.paired_transform(frames_tensor, segmap, self.video_transform, self.segmap_transform)

        if self.pad_to != self.video_length:
            c,t,h,w = frames_tensor.shape
            frames_tensor = torch.cat((frames_tensor, torch.full(size=(c,self.pad_to-t,h,w), fill_value=-1)), dim=1)
        
        if self.pad_to != segmap.shape[0] and self.clip_step == 1: 
            t,h,w = segmap.shape
            segmap = torch.cat((segmap, torch.zeros(size=(self.pad_to-t,h,w))), dim=0)
        segmap = segmap/torch.max(segmap)
        
        example = {}
        if self.vae is None:
            example['videos'] = frames_tensor  
        else:           
            raise NotImplementedError

        example["init_img"] = frames_tensor[:,0,...]
        example["frame_stride"] = self.frame_stride
        example["clip_info"] = clip
        example["video_segmap"] = segmap
        example["sem_info"] = {}

        # Extract phase and triplet
        phase_ids = []
        triplet_ids = []
        
        for f_info in labels:
            phase_ids.append(f_info[0]['phase_id'])
            triplet_ids.append(f_info[0]['triplet_id']) # TODO multiple HOW
        if self.text_cond == "label_emb":
            example['sem_info']['phase'] = torch.as_tensor(phase_ids)
            example['sem_info']['triplet'] = torch.as_tensor(triplet_ids) + 1 # TO fix values = -1
        elif self.text_cond == "SurgVLP":
            # Extract the textual description
            phase_texts = []
            for phase_id in phase_ids:
                if phase_id == -1:
                    phase_text = "unk"
                else:
                    phase_text = self.mappings['phase'][str(phase_id)].replace('-', ' ')
                phase_texts.append(phase_text)
            # Tokenize
            tokenized_phases = tokenize_surgvlp(phase_texts, tokenizer_clinical=self.tokenizer_clinical, max_length=32)
            #tokenized_phases = surgvlp.tokenize(phase_texts, max_length=32)            

            triplet_texts = []
            for triplet_id in triplet_ids:
                if triplet_id == -1:
                    triplet_text = "unk"
                else:
                    triplet_text = self.mappings['triplet'][str(triplet_id)].replace(',',' ')
                triplet_texts.append(triplet_text)
            tokenized_triplets = tokenize_surgvlp(triplet_texts, tokenizer_clinical=self.tokenizer_clinical, max_length=32)                
            #tokenized_triplets = surgvlp.tokenize(triplet_texts, max_length=32)

            example['sem_info']['phase'] = tokenized_phases
            example['sem_info']['triplet'] = tokenized_triplets   

        del labels                     
        return example    
    
    def _load_precomputed_samples(self, load_dir):
        raise NotImplementedError

    def _compute_and_save_samples(self, save_dir):
        raise NotImplementedError
