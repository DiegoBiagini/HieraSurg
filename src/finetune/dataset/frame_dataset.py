import os
import random
import re
from PIL import ImageFile
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video
from transformers import CLIPTokenizer
import numpy as np
from decord import VideoReader, cpu
from typing import List
from pathlib import Path
import json
from .data_utils import masks_to_tensor, inpaint_object_in_masks
import shutil
import copy

""" VideoFrameDataset """

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def videosegmap_collate_fn(batch):    
    # Batch the frame_info
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
        try:
            batched_latents = torch.stack([item['latents'] for item in batch])
        except KeyError:
            # Get first item's init_img as default
            default_latents = batch[0]['latents']
            # Replace missing init_imgs with default
            latents = [item.get('latents', default_latents) for item in batch]
            batched_latents = torch.stack(latents)        

        return {
            'latents': batched_latents,
            'frame_stride': batched_frame_stride,
            'clip_info': batched_clip_info,
            'video_segmap': batched_segmaps,
            'init_img': batched_init_img
        }
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
        }
    return {
        'frame_stride': batched_frame_stride,
        'clip_info': batched_clip_info,
        'video_segmap': batched_segmaps,
        'init_img': batched_init_img,
    }

def cholec_collate_fn(batch):
    #images = torch.stack([item['image'] for item in batch])
    

    # Batch the frame_info
    batched_frame_info = []

    if 'frame_info' in batch[0]:
        for item in batch:
            batched_frame_info.append(item['frame_info'])

    batched_clip_info = []
    for item in batch:
        batched_clip_info.append(item['clip_info'])

    if 'latents' in batch[0]:    
        latents = torch.stack([item['latents'] for item in batch])

        return {
            # 'image': images,
            'latents': latents,
            'frame_info': batched_frame_info,
            'clip_info': batched_clip_info,
        }
    else:
        if 'videos' in batch[0]:
            videos = torch.stack([item['videos'] for item in batch])

            return {
                # 'image': images,
                'videos': videos,
                'frame_info': batched_frame_info,
                'clip_info': batched_clip_info,
            }
    raise Exception("Wrong data stuff in the collate fn")
    batched_bboxes = []
    for item in batch:
        batched_bboxes.append(item['bboxes'])
    kwargs = {}

    # Batch the dsg mhhh
    paths = None
    if batch[0]['dsgraph'].__class__ == dgl.DGLGraph:
        batched_dsg = dgl.batch([item['dsgraph'] for item in batch])
    elif batch[0]['dsgraph'].__class__ == Data:
        batched_dsg = Batch.from_data_list([item['dsgraph'] for item in batch])
        paths = batched_shortest_path_distance(batched_dsg)
        kwargs['paths'] = paths

    # Batch the frame_stride
    if 'frame_stride' in batch[0]:
        frame_stride = torch.tensor([item['frame_stride'] for item in batch])
        return {
            'image': images,
            'frame_info': batched_frame_info,
            'frame_stride': frame_stride,
            'dsgraph': batched_dsg,
            'kwargs': kwargs
        }
    # Autoreg case
    if "prev_image" in batch[0]:
        prev_images = torch.stack([item['prev_image'] for item in batch])
        
        paths = None
        if batch[0]['prev_dsgraph'].__class__ == dgl.DGLGraph:
            batched_prev_dsg = dgl.batch([item['prev_dsgraph'] for item in batch])
        elif batch[0]['prev_dsgraph'].__class__ == Data:
            batched_dsg = Batch.from_data_list([item['prev_dsgraph'] for item in batch])
            paths = batched_shortest_path_distance(batched_dsg)
            kwargs['prev_paths'] = paths
        return {
            'image': images,
            'prev_image': prev_images,
            'frame_info': batched_frame_info,
            'prev_frame_info': [item['prev_frame_info'] for item in batch],
            'dsgraph': batched_dsg,
            'prev_dsgraph': batched_prev_dsg,

            'kwargs': kwargs
            
        }       
    else:
        return {
            'image': images,
            'frame_info': batched_frame_info,
            'bboxes': batched_bboxes,
            'dsgraph': batched_dsg,
            'kwargs': kwargs
            
        }        


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    '''
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
    '''
    Im = Image.open(path)
    return Im.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    """
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
    """
    return pil_loader(path)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(dir):
    assert (os.path.exists(dir)), f'{dir} does not exist'
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def class_name_to_idx(annotation_dir):
    """
    return class indices from 0 ~ num_classes-1
    """
    fpath = os.path.join(annotation_dir, "classInd.txt")
    with open(fpath, "r") as f:
        data = f.readlines()
        class_to_idx = {x.strip().split(" ")[1].lower(): int(x.strip().split(" ")[0]) - 1 for x in data}
    return class_to_idx


def split_by_captical(s):
    s_list = re.sub(r"([A-Z])", r" \1", s).split()
    string = ""
    for s in s_list:
        string += s + " "
    return string.rstrip(" ").lower()


def _read_video(video_path, video_id, sample_frame_num, is_train=True):
    """
    read frames from long video
    args:
        video_id: str,
        sample_frame_num: frames used
    return:
        img_arrays: [num_frm, 3, H, W]
        chunk_mask: [num_frm, n_clip], , mask for indicating frames belong to each clip

    """

    video_path = os.path.join(video_path, video_id + '.mp4')
    vr = VideoReader(video_path, ctx=cpu(0))
    num_frame = len(vr)
    if is_train:
        interval = int(num_frame / (sample_frame_num - 1))
        start = np.random.randint(0, interval + 1)
        end = np.random.randint(num_frame - 1 - interval, num_frame)
        frame_idx = np.linspace(start, end, sample_frame_num).astype(int)
    else:
        frame_idx = np.linspace(0, num_frame - 1, sample_frame_num).astype(int)

    img_arrays = vr.get_batch(frame_idx)

    img_arrays = img_arrays.float() / 255

    img_arrays = img_arrays.permute(0, 3, 1, 2)  # N,C,H,W
    vr.save_frames(video_path, video_id, img_arrays)

    return img_arrays

def tokenize(tokenizer, texts, total_chunk=8, max_length=50, use_split=True, **kwargs):
    '''
    tokenizing text for pretraining
    args:
        tokenizer: for tokenize texts
        texts: list of text segment
        total_chunk: num of text segments
        use_split: whether split the texts
    return:
        text_ids: sequence of token id
        attention_mask: segment_ids to distinguish the sentences
        chunk: index of [CLS]

    '''
    if use_split:

        def merge(texts, tolen=8):
            if len(texts) <= tolen:
                return texts
            else:
                while len(texts) > tolen:
                    texts_2g = [len(texts[i]) + len(texts[i + 1]) for i in range(len(texts) - 1)]
                    min_index = texts_2g.index(min(texts_2g))
                    texts_group = []
                    for i in range(len(texts)):
                        if i != min_index and i != min_index + 1:
                            texts_group.append(texts[i])
                        elif i == min_index:
                            texts_group.append(' '.join(texts[i:i + 2]))
                        else:
                            continue
                    texts = texts_group
                return texts

        if len(texts) > total_chunk:
            texts = merge(texts, tolen=total_chunk)

        encoded = [tokenizer(x, padding='max_length', truncation=True, max_length=max_length) for x in texts]

        text_ids = [x.input_ids for x in encoded]
        attention_mask = [x.attention_mask for x in encoded]

        if len(texts) < total_chunk:
            for i in range(total_chunk - len(texts)):
                text_ids.append([0 for x in range(max_length)])
                attention_mask.append([0 for x in range(max_length)])

    else:
        texts = ' '.join(texts)
        encoded = [tokenizer(x, padding='max_length', truncation=True, max_length=max_length) for x in [texts]]

        text_ids = [x.input_ids for x in encoded]
        attention_mask = [x.attention_mask for x in encoded]

    return text_ids, attention_mask


def make_dataset(dir, nframes, texts, frame_stride=1, clip_step=None,
                        tokenizer=None, total_chunk=8, max_length=50, use_split=True, **kwargs):
    """
        Load videos from MSR-VTT or activityNet
        assert videos are saved in first-level directory:
            dir:
                videoxxx1
                    frame1.jpg
                    frame2.jpg
                videoxxx2
        """
    if clip_step is None:
        # consecutive clips with no frame overlap
        clip_step = nframes
    # make videos
    clips = []  # 2d list
    videos = []  # 2d list
    for video_name in sorted(os.listdir(dir)):
        if video_name != '_broken_clips':
            video_path = os.path.join(dir, video_name)
            assert (os.path.isdir(video_path))

            frames = []
            for i, fname in enumerate(sorted(os.listdir(video_path))):
                assert (is_image_file(fname)), f'fname={fname},video_path={video_path},dir={dir}'

                # get frame info
                img_path = os.path.join(video_path, fname)
                class_name = tokenize(tokenizer=tokenizer, texts=texts, total_chunk=total_chunk, max_length=max_length,
                                      use_split=use_split)
                frame_info = {
                    "img_path": img_path,
                    "class_index": class_name,
                    "class_name": texts,
                    "class_caption": texts  # boxing speed bag
                }
                frames.append(frame_info)

            # make videos
            if len(frames) >= nframes:
                videos.append(frames)

            # make clips
            frames = frames[::frame_stride]
            start_indices = list(range(len(frames)))[::clip_step]
            for i in start_indices:
                clip = frames[i:i + nframes]
                if len(clip) == nframes:
                    clips.append(clip)
    return clips, videos

def make_dataset_cholec(video_paths_list, annotations_path, nframes, frame_stride=1, clip_step=None, skip_empty = False, **kwargs):
    """
        Load videos from Cholec
        assert videos are saved in first-level directory:
            dir:
                videos
                    VIDx1
                        xxxx1.jpg
                        xxxx2.jpg
                    VIDx2
        """
    if clip_step is None:
        # consecutive clips with no frame overlap
        clip_step = nframes
    # make videos
    clips = []  # 2d list
    videos = []  # 2d list
    mappings = None
    for video_path in video_paths_list:
        # Get the path of the annotation
        video_name = str(Path(video_path).stem)
        if video_name.startswith("VID"):
            vid_annotation_path = os.path.join(annotations_path, video_name + ".json")
        else:
            video_number = str(video_name)[5:]
            json_name = f"VID{video_number}.json"
            vid_annotation_path = os.path.join(annotations_path, json_name)
        with open(vid_annotation_path, "r") as f:
            ann_json = json.load(f)
        assert (os.path.isdir(video_path))

        # Save the mappings
        if mappings is None:
            mappings = {k:ann_json['categories'][k] for k in ['triplet', 'instrument', 'verb', 'target', 'phase']}

        frames = []
        for i, fname in enumerate(sorted(os.listdir(video_path))):
            assert (is_image_file(fname)), f'fname={fname},video_path={video_path},dir={dir}'

            # get frame info
            img_path = os.path.join(video_path, fname)

            #Load the annotation for the frame
            if clip_step == 1 or clip_step is None:
                img_path_corr = str(int(Path(img_path).stem))
            else:
                img_path_corr = str(int(int(Path(img_path).stem)/clip_step))                
                
            frame_info = {
                "img_path": img_path,
                "annotations": ann_json['annotations'][img_path_corr]
            }
            frames.append(frame_info)

        # make videos
        #if len(frames) >= nframes:
        videos.append(frames)

        # make clips
        frames = frames[::frame_stride]
        start_indices = list(range(len(frames)))[::clip_step]
        for i in start_indices:
            clip = frames[i:i + nframes]

            # If skip_empty is set skip any clip that contains at least a blacked out frame
            save_clip = True
            if skip_empty:
                for f in clip:
                    img = default_loader(f['img_path'])
                    if is_mostly_black(img):
                        save_clip = False
                        break

            if len(clip) == nframes and save_clip:
                clips.append(clip)
    return clips, videos, mappings

def make_dataset_cholec80(video_paths_list, nframes, frame_stride=1, clip_step=None, skip_empty = False, **kwargs):
    """
        Load videos from Cholec
        assert videos are saved in first-level directory:
            dir:
                videos
                    VIDx1
                        xxxx1.jpg
                        xxxx2.jpg
                    VIDx2
        """
    if clip_step is None:
        # consecutive clips with no frame overlap
        clip_step = nframes
    # make videos
    clips = []  # 2d list
    videos = []  # 2d list
    for video_path in video_paths_list:
        vid_clip_counter = 0
        assert (os.path.isdir(video_path))

        frames = []
        for i, fname in enumerate(sorted(os.listdir(video_path))):
            assert (is_image_file(fname)), f'fname={fname},video_path={video_path},dir={dir}'

            # get frame info
            img_path = os.path.join(video_path, fname)

            #Load the annotation for the frame

            frame_info = {
                "img_path": img_path,
            }
            frames.append(frame_info)

        # make videos
        #if len(frames) >= nframes:
        videos.append(frames)

        # make clips
        frames = frames[::frame_stride]
        start_indices = list(range(len(frames)))[::clip_step]
        for i in start_indices:
            clip = frames[i:i + nframes]

            # If skip_empty is set skip any clip that contains at least a blacked out frame
            save_clip = True
            if skip_empty:
                for f in clip:
                    img = default_loader(f['img_path'])
                    if is_mostly_black(img):
                        save_clip = False
                        break

            if len(clip) == nframes and save_clip:
                clips.append(clip)
                vid_clip_counter += 1
    return clips, videos

def is_mostly_black(image, threshold=0.9):
    black_pixels = 0
    total_pixels = image.size[0] * image.size[1]
    
    for pixel in image.getdata():
        if sum(pixel) < 10:
            black_pixels += 1
    
    return (black_pixels / total_pixels) > threshold

def make_frame_dataset_cholec(video_paths_list, annotations_path, remove_black=False):
    """
        Load frames from Cholec
        assert videos are saved in first-level directory:
            dir:
                videos
                    VIDx1
                        xxxx1.jpg
                        xxxx2.jpg
                    VIDx2
        """
    # load frames
    frames = []
    mappings = None
    for video_path in video_paths_list:
        # Get the path of the annotation
        vid_annotation_path = os.path.join(annotations_path, str(Path(video_path).stem) + ".json")
        with open(vid_annotation_path, "r") as f:
            ann_json = json.load(f)
        assert (os.path.isdir(video_path))

        # Save the mappings
        if mappings is None:
            mappings = {k:ann_json['categories'][k] for k in ['triplet', 'instrument', 'verb', 'target', 'phase']}

        for i, fname in enumerate(sorted(os.listdir(video_path))):
            assert (is_image_file(fname)), f'fname={fname},video_path={video_path},dir={dir}'

            # get frame info
            img_path = os.path.join(video_path, fname)
            
            if remove_black:
                with Image.open(img_path) as img:
                    # Check if more than 90% are all black pixels
                    if is_mostly_black(img):
                        continue              

            #Load the annotation for the frame
            frame_info = {
                "img_path": img_path,
                "annotations": ann_json['annotations'][str(int(Path(img_path).stem))]
            }
            frames.append(frame_info)

    return frames, mappings


def make_dataset_ucf(dir, nframes, class_to_idx, frame_stride=1, clip_step=None):
    """
    Load consecutive clips and consecutive frames from `dir`.

    args:
        nframes: num of frames of every video clips
        class_to_idx: for mapping video name to video id
        frame_stride: select frames with a stride.
        clip_step: select clips with a step. if clip_step< nframes, 
            there will be overlapped frames among two consecutive clips.

    assert videos are saved in first-level directory:
        dir:
            videoxxx1
                frame1.jpg
                frame2.jpg
            videoxxx2
    """
    if clip_step is None:
        # consecutive clips with no frame overlap
        clip_step = nframes
    # make videos
    clips = []  # 2d list
    videos = []  # 2d list
    for video_name in sorted(os.listdir(dir)):
        if video_name != '_broken_clips':
            video_path = os.path.join(dir, video_name)
            assert (os.path.isdir(video_path))

            frames = []
            for i, fname in enumerate(sorted(os.listdir(video_path))):
                assert (is_image_file(fname)), f'fname={fname},video_path={video_path},dir={dir}'

                # get frame info
                img_path = os.path.join(video_path, fname)
                class_name = video_name.split("_")[1].lower()  # v_BoxingSpeedBag_g12_c05 -> boxingspeedbag
                class_caption = split_by_captical(
                    video_name.split("_")[1])  # v_BoxingSpeedBag_g12_c05 -> BoxingSpeedBag -> boxing speed bag
                frame_info = {
                    "img_path": img_path,
                    "class_index": class_to_idx[class_name],
                    "class_name": class_name,  # boxingspeedbag
                    "class_caption": class_caption  # boxing speed bag
                }
                frames.append(frame_info)

            # make videos
            if len(frames) >= nframes:
                videos.append(frames)

            # make clips
            frames = frames[::frame_stride]
            start_indices = list(range(len(frames)))[::clip_step]
            for i in start_indices:
                clip = frames[i:i + nframes]
                if len(clip) == nframes:
                    clips.append(clip)
    return clips, videos


def load_and_transform_frames(frame_list, loader, img_transform=None):
    assert (isinstance(frame_list, list))
    clip = []
    labels = []
    for frame in frame_list:

        if isinstance(frame, tuple):
            fpath, label = frame
        elif isinstance(frame, dict):
            fpath = frame["img_path"]
            label = {
                "class_index": frame["class_index"],
                "class_name": frame["class_name"],
                "class_caption": frame["class_caption"],
            }

        labels.append(label)
        img = loader(fpath)
        if img_transform is not None:
            img = img_transform(img)
        img = img.view(img.size(0), 1, img.size(1), img.size(2))
        clip.append(img)
    return clip, labels[0]  # all frames have same label.

def load_and_transform_frames_cholec(frame_list, mappings, loader, img_transform=None, frame_cache = {}):
    assert (isinstance(frame_list, list))
    clip = []
    labels = []

    # Define a transformation that takes relative coordinates of a bbox
    # And considers the image transformation to warp it to the correct position
    def transform_bbox(bbox, img_w, img_h):
        # TODO: Not sure if correct
        return bbox
        bx, by, bw, bh = bbox
        # First get the actual coordinates
        bx, by, bw, bh = bx*img_w, by*img_h, bw*img_w, bh*img_h

        # Get displacement and adjust those
        top, left, bottom, right = img_transform[0](None)
        bx, by = bx -left, by-top

        # Get resizing and adjust those
        new_w, new_h = img_transform[1].size
        bw, bh = bw*(img_w/new_w), bh*(img_h/new_h)
        return bx, by, bw, bh

    for frame in frame_list:

        fpath, annotations = frame["img_path"], frame['annotations']
        
        _, f_id = os.path.split(fpath)
        f_id = int(os.path.splitext(f_id)[0])
        _, v_id = os.path.split(_)
        if fpath in frame_cache:
            img = frame_cache[fpath]
        else:
            img = loader(fpath)
            frame_cache[fpath] = img
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
    return clip, labels

def load_and_transform_frames_cholec2(frame_list, mappings, loader, img_transform=None):
    assert (isinstance(frame_list, list))
    clip = []

    for frame in frame_list:

        fpath = frame["img_path"]
        
        _, f_id = os.path.split(fpath)
        f_id = int(os.path.splitext(f_id)[0])
        _, v_id = os.path.split(_)

        img = loader(fpath)
        if img_transform is not None:
            img = img_transform(img)
        img = img.view(img.size(0), 1, img.size(1), img.size(2))

        clip.append(img)
    return clip
import pickle
import blosc
def load_and_transform_frames_cholec80(frame_list, loader, img_transform=None, frame_cache = {}, is_segmap_compressed = False, fps=None):
    assert (isinstance(frame_list, list))
    cache_frames = frame_cache != {}
    clip = []

    for frame in frame_list:

        fpath = frame["img_path"]
        
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
        
        clip.append(img)
    
    # Load the segmap file
    if not fps is None:
        # TODO
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
            return clip, None
    else:
        segmap_masks = pickle.load(open(segmap_path, 'rb'))
    if not fps is None:
        segmap_new_length = len(frame_list)//fps
        segmap_masks = {k:v for k,v in segmap_masks.items() if k < segmap_new_length}

    return clip, segmap_masks

def load_and_transform_frame_cholec(frame_info, mappings, loader, img_transform=None, filter_black = False):

    fpath, annotations = frame_info["img_path"], frame_info['annotations']
    def transform_bbox(bbox, img_w, img_h):
        # The BBOX is relative to a 240-430 
        pad_left, pad_top, rect_width, rect_height = 0, 40, 856, 560
        og_size = 240, 430 # It's divided by 2
        bbox_fullres = [c*2 for c in bbox] # Constant but I could also find the ratio directly
        bbox_rect = [bbox_fullres[0]+pad_left, bbox_fullres[1]+pad_top, bbox_fullres[2]*rect_width/img_w, bbox_fullres[3]*rect_height/img_h] 
        normalized_bbox = [bbox_rect[0]/rect_width, bbox_fullres[1]/rect_height, bbox_fullres[2]/rect_width, bbox_fullres[3]/rect_height] 
        return [int(c) for c in normalized_bbox]
        
    _, f_id = os.path.split(fpath)
    f_id = int(os.path.splitext(f_id)[0])
    _, v_id = os.path.split(_)
    img = loader(fpath)
    if filter_black:
        if is_mostly_black(img):
            return None
    if img_transform is not None:
        img = img_transform(img)
    frame = img.view(img.size(0), img.size(1), img.size(2))

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
            "instrument_bbox": transform_bbox(ann[3:7], frame.size(1), frame.size(2)), # TODO make sure they are w h
            "target_id": ann[8],
            "target": mappings['target'][str(int(ann[8]))] if ann[8] != -1 else "",
            "target_conf": ann[9],
            "target_rel_bbox": ann[10:14],
            "target_bbox": transform_bbox(ann[10:14], frame.size(1), frame.size(2)), # TODO make sure they are w h
            "verb_id": ann[7],
            "verb": mappings['verb'][str(int(ann[7]))] if ann[7] != -1 else "",
            "phase_id": ann[14],
            "phase": mappings['phase'][str(int(ann[14]))] if ann[14] != -1 else "",    

            "vid_id": v_id,
            "frame_id": f_id
        }
        labels_frame.append(label)

    return frame, labels_frame

from multiprocessing import Manager


class VideoFrameDatasetCholec(data.Dataset):
    def __init__(self,
                 data_root,
                 resolution : List,
                 video_length,  # clip length
                 dataset_name="",
                 subset_split="",
                 annotation_dir=None,
                 spatial_transform="",
                 frame_stride=1,
                 clip_step=None,
                 precompute="load_precomputed",  # One of load_precomputed, compute_on_the_fly, compute_and_save, compute_and_cache
                 vae=None,
                 pad_to=49,
                 horizontal_flip=False,
                 vertical_flip=False              
                 #tokenizer=None,
                 ):

        self.loader = default_loader
        self.video_length = video_length
        self.subset_split = subset_split
        self.spatial_transform = spatial_transform
        self.frame_stride = frame_stride
        self.dataset_name = dataset_name
        self.precompute = precompute 

        assert (subset_split in ["train", "test", "all", ""])  # "" means no subset_split directory.

        split_file = os.path.join(data_root, f"{subset_split}_split.txt")
        with open(split_file, "r") as f:
            video_names = f.readlines()
        video_paths = [os.path.join(data_root, "videos", v.strip()) for v in video_names]
        annotation_dir = os.path.join(data_root, "labels")

        self.clips, self.videos, self.mappings = make_dataset_cholec(
            video_paths_list=video_paths, annotations_path=annotation_dir, nframes=video_length, clip_step=clip_step, skip_empty=False)
        assert (len(self.clips[0]) == video_length), f"Invalid clip length = {len(self.clips[0])}"

        self.clips = self.clips
        self.vae = vae
        #self.mappings = manager.list(self.mappings)
        print('[VideoFrameDataset] number of videos:', len(self.videos))
        print('[VideoFrameDataset] number of clips', len(self.clips))

        # check data
        if len(self.clips) == 0:
            raise (RuntimeError(f"Found 0 clips. \n"
                                "Supported image extensions are: " +
                                ",".join(IMG_EXTENSIONS)))

        # data transform
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip

        mirror_transforms = []
        if self.horizontal_flip:
            mirror_transforms.append(transforms.RandomHorizontalFlip(p=0.3))
        if self.vertical_flip:
            mirror_transforms.append(transforms.RandomVerticalFlip(p=0.3))        
        # Precompute samples if required

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        if self.spatial_transform == "crop_resize":

            print('Spatial transform:crop and then resize')
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
                transforms.Resize((resolution, resolution))
            ])
        elif self.spatial_transform == "random_crop":
            self.video_transform = transforms.Compose([
                *mirror_transforms,
                transforms_video.RandomCropVideo(resolution),
            ])
        elif self.spatial_transform == "":
            self.video_transform = transforms.Compose(mirror_transforms) if mirror_transforms else None
        else:
            raise NotImplementedError
        
        # DSG loading
        """
        ann_path = Path(data_root) / "labels"
        videos_path = Path(data_root)/ "videos"

        vid_labels_paths = [] 
        for root, dirs, files in os.walk(ann_path):
            for file in files:
                if file.endswith('.json'):
                    # Construct the full file path and add it to the list
                    vid_labels_paths.append((ann_path/file, videos_path/Path(file).stem))        
        """


        self.pad_to = pad_to
        if self.precompute == "compute_and_save":
            self.precomputed_samples = []
            self._compute_and_save_samples(save_dir=Path(data_root)/f"precomputed_samples_{subset_split}")
        elif self.precompute == "load_precomputed":
            self.precomputed_samples = []
            self._load_precomputed_samples(load_dir=Path(data_root)/f"precomputed_samples_{subset_split}")
        
    def _compute_sample(self, index):
        
        clip = self.clips[index]
        
        frames = load_and_transform_frames_cholec2(clip, self.mappings, self.loader, self.img_transform)
        assert (len(frames) == self.video_length), f'clip_length={len(frames)}, target={self.video_length}, {clip}'
        
        frames_tensor = torch.cat(frames, 1)  # Concatenate frames (c,t,h,w)
        del frames  # Release memory for frames
        
        if self.video_transform:
            frames_tensor = self.video_transform(frames_tensor)
        if self.pad_to != self.video_length:
            c,t,h,w = frames_tensor.shape

            # -1 is 0
            frames_tensor = torch.cat((frames_tensor, torch.full(size=(c,self.pad_to-t,h,w), fill_value=-1)), dim=1)
        example = {}
        if self.vae is None:
            example['videos'] = frames_tensor
        else:
            with torch.no_grad():
                to_encode = frames_tensor.unsqueeze(0).to(dtype=self.vae.dtype, device=self.vae.device)
                og_frames = to_encode.shape[2]
                # Ensure the encoder receives exactly 49 frames
                if to_encode.shape[2] < 49:
                    pad_size = 49 - to_encode.shape[2]
                    to_encode = torch.nn.functional.pad(to_encode, (0, 0, 0, 0, 0, pad_size))
                elif to_encode.shape[2] > 49:
                    to_encode = to_encode[:, :, :49, :, :]
                del frames_tensor  # Free memory from frames_tensor
                
                out_latents = torch.squeeze(self.vae.encode(to_encode).latent_dist.sample() * self.vae.config.scaling_factor)
                out_latents = out_latents[:,:og_frames//4,:,:]

                example["latents"] = out_latents
                del out_latents
                del to_encode

        example["frame_stride"] = self.frame_stride
        example["clip_info"] = clip

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
            # Remove the videos if they are there
        
        to_save = copy.deepcopy(example)
        if 'videos' in to_save:
            del to_save['videos']
        self.sample_cache[index] = to_save
        return example

    def _compute_and_save_samples(self, save_dir):
        raise NotImplementedError()
        try:
            if os.path.exists(save_dir):
                shutil.rmtree(save_dir)
        except:
            pass
        os.makedirs(save_dir, exist_ok=True)
        from tqdm import tqdm
        import numpy as np
        batch_size = 4 
        for i in tqdm(range(0, len(self.clips), batch_size), desc="Precomputing and saving samples"):
            batch_clips = self.clips[i:i+batch_size]
            batch_frames = []

            for clip in batch_clips:
                frames = load_and_transform_frames_cholec2(clip, self.mappings, self.loader, self.img_transform)
                frames = torch.cat(frames, 1)  # Concatenate frames (c,t,h,w)
                if self.video_transform:
                    frames = self.video_transform(frames)
                batch_frames.append(frames)

            batch_frames_tensor = torch.stack(batch_frames).to(dtype=self.vae.dtype, device=self.vae.device)
            og_frames = batch_frames_tensor.shape[2]
            
            # Ensure the encoder receives exactly 49 frames
            if batch_frames_tensor.shape[2] < 49:
                pad_size = 49 - batch_frames_tensor.shape[2]
                batch_frames_tensor = torch.nn.functional.pad(batch_frames_tensor, (0, 0, 0, 0, 0, pad_size))
            elif batch_frames_tensor.shape[2] > 49:
                batch_frames_tensor = batch_frames_tensor[:, :, :49, :, :]

            batch_latents = self.vae.encode(batch_frames_tensor).latent_dist.sample() * self.vae.config.scaling_factor
            batch_latents = batch_latents[:,:,:og_frames//4,:,:]

            for j, (frames, latents) in enumerate(zip(batch_frames, batch_latents)):
                example = dict()
                example["latents"] = latents
                example["frame_stride"] = self.frame_stride
                example["clip_info"] = batch_clips[j]
                self.precomputed_samples.append(example)

    def _load_precomputed_samples(self, load_dir):
        from glob import glob
        sample_files = glob(os.path.join(load_dir, "*.pt"))
        self.precomputed_samples = []
        for sample_file in sample_files:
            data = torch.load(sample_file)
            example = dict()

            example["latents"] = data["latents"]
            example["frame_info"] = data["frame_info"]
            example["clip_info"] = data["clip_info"]
            self.precomputed_samples.append(example)

    def __getitem__(self, index):
        if self.precompute == "load_precomputed" or self.precompute == "compute_and_save":
            if index < len(self.precomputed_samples):
                return self.precomputed_samples[index]
            else:
                random_index = random.randint(0, len(self.precomputed_samples) - 1)
                print(f"Warning: Index {index} out of range. Returning random sample at index {random_index}.")
                return self.precomputed_samples[random_index]
        elif self.precompute == "compute_on_the_fly":
            return self._compute_sample(index)
        elif self.precompute == "compute_and_cache":
            return self._compute_and_cache_sample(index)

    def __len__(self):
        return len(self.clips)

class VideoFrameDatasetCholec80(data.Dataset):
    def __init__(self,
                 data_root,
                 resolution : List,
                 dataset_name : str,
                 video_length,  # clip length
                 annotation_dir=None,
                 spatial_transform="",
                 subset_split="",
                 frame_stride=1,
                 clip_step=None,
                 precompute="load_precomputed",  # One of load_precomputed, compute_on_the_fly, compute_and_save, compute_and_cache
                 vae=None,
                 pad_to=17,
                 sample_cache = None,
                 horizontal_flip = False,
                 vertical_flip = False
                 ):

        self.loader = default_loader
        self.video_length = video_length
        self.spatial_transform = spatial_transform
        self.frame_stride = frame_stride
        self.dataset_name = dataset_name
        self.precompute = precompute 

        video_paths = [os.path.join(data_root, folder) for folder in os.listdir(data_root) if folder.startswith("video") and folder[5:].isdigit()]
        print(clip_step)
        self.clips, self.videos = make_dataset_cholec80(
            video_paths_list=video_paths, nframes=video_length, clip_step=clip_step, skip_empty=False)
        assert (len(self.clips[0]) == video_length), f"Invalid clip length = {len(self.clips[0])}"

        self.vae = vae

        print('[VideoFrameDatasetCholec80] number of videos:', len(self.videos))
        print('[VideoFrameDatasetCholec80] number of clips', len(self.clips))
        self.clips = self.clips
        # check data
        if len(self.clips) == 0:
            raise (RuntimeError(f"Found 0 clips. \n"
                                "Supported image extensions are: " +
                                ",".join(IMG_EXTENSIONS)))

        # data transform
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip

        mirror_transforms = []
        if self.horizontal_flip:
            mirror_transforms.append(transforms.RandomHorizontalFlip(p=0.3))
        if self.vertical_flip:
            mirror_transforms.append(transforms.RandomVerticalFlip(p=0.3))        
        # Precompute samples if required

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        if self.spatial_transform == "crop_resize":

            print('Spatial transform:crop and then resize')
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
                transforms.Resize((resolution, resolution))
            ])
        elif self.spatial_transform == "random_crop":
            self.video_transform = transforms.Compose([
                *mirror_transforms,
                transforms_video.RandomCropVideo(resolution),
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
            self._compute_and_save_samples(save_dir=Path(data_root)/f"precomputed_samples_{subset_split}")
        elif self.precompute == "load_precomputed":
            self.precomputed_samples = []
            self._load_precomputed_samples(load_dir=Path(data_root)/f"precomputed_samples_{subset_split}")
    
    def _compute_sample(self, index):
        
        clip = self.clips[index]
        
        frames = load_and_transform_frames_cholec2(clip, None, self.loader, self.img_transform)
        assert (len(frames) == self.video_length), f'clip_length={len(frames)}, target={self.video_length}, {clip}'

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
        example["clip_info"] = clip

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


    def _compute_and_save_samples(self, save_dir):
        raise NotImplementedError()

    def _load_precomputed_samples(self, load_dir):
        raise NotImplementedError
    
    def __getitem__(self, index):
        if self.precompute == "load_precomputed" or self.precompute == "compute_and_save":
            if index < len(self.precomputed_samples):
                return self.precomputed_samples[index]
            else:
                random_index = random.randint(0, len(self.precomputed_samples) - 1)
                print(f"Warning: Index {index} out of range. Returning random sample at index {random_index}.")
                return self.precomputed_samples[random_index]
        elif self.precompute == "compute_on_the_fly":
            return self._compute_sample(index)
        elif self.precompute == "compute_and_cache":
            return self._compute_and_cache_sample(index)

    def __len__(self):
        return len(self.clips)

from torchvision.transforms.functional import crop            

def cropper(img):
    top, left, bottom, right = 0, 67, 480, 787
    if img is None:
        return top, left, bottom, right
    return crop(img, top, left, bottom, right)

import torch
import random
from torchvision import transforms

class PairedRandomFlip:
    def __init__(self, horizontal_flip_prob=0.5, vertical_flip_prob=0.5):
        self.horizontal_flip_prob = horizontal_flip_prob
        self.vertical_flip_prob = vertical_flip_prob

    def __call__(self, img, segmap):
        if random.random() < self.horizontal_flip_prob:
            img = transforms.functional.hflip(img)
            segmap = transforms.functional.hflip(segmap)
        if random.random() < self.vertical_flip_prob:
            img = transforms.functional.vflip(img)
            segmap = transforms.functional.vflip(segmap)
        return img, segmap


class VideoFrameDatasetCholecSegmap(data.Dataset):
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
                 horizontal_flip = False,
                 vertical_flip = False
                 #tokenizer=None,
                 ):

        self.loader = default_loader
        self.video_length = video_length
        self.spatial_transform = spatial_transform
        self.frame_stride = frame_stride
        self.dataset_name = dataset_name
        self.precompute = precompute 
        self.segmap_dropout = segmap_dropout
        self.clip_step = clip_step

        manager = Manager()
        assert subset_split in ["train", "val", "test", ""]
        video_paths = [os.path.join(data_root, folder) for folder in os.listdir(data_root) if folder.startswith("video") and folder[5:].isdigit()]
        segmap_path = data_root #Path("/mnt/data1/Cholec80/frames1fps/Done/") 
        
        regex_pattern = "^video\d+_masks$"
        segmap_videos = [d for d in os.listdir(segmap_path) if os.path.isdir(os.path.join(segmap_path, d)) and re.match(regex_pattern, d)]
        segmap_videos = [f.replace('_masks','') for f in segmap_videos]

        video_paths = [p for p in video_paths if Path(p).stem in segmap_videos]        

        self.clips, self.videos = make_dataset_cholec80(
            video_paths_list=video_paths, nframes=video_length, clip_step=clip_step, skip_empty=False)
        assert (len(self.clips[0]) == video_length), f"Invalid clip length = {len(self.clips[0])}"

        self.clips = manager.list(self.clips)
        self.videos = manager.list(self.videos)
        self.vae = vae

        print('[VideoFrameDatasetSegMap] number of videos:', len(self.videos))
        print('[VideoFrameDatasetSegMap] number of clips', len(self.clips))

        # check data
        if len(self.clips) == 0:
            raise (RuntimeError(f"Found 0 clips. \n"
                                "Supported image extensions are: " +
                                ",".join(IMG_EXTENSIONS)))

        # data transform
        self.vertical_flip_prob = 0.3 if vertical_flip else 0
        self.horizontal_flip_prob = 0.3 if horizontal_flip else 0

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
            self.video_transform = transforms.Compose([
                transforms.Pad(padding=(0, 40), fill=-1),  # Padding the image from (480, 854) to (560, 854)
                transforms.CenterCrop(size=(560, 840)),  # Cropping the padded image to (560, 840)
                transforms.Resize(size=resolution, antialias=True)  # Finally resizing to required resolution
            ])
            self.segmap_transform = transforms.Compose([
                transforms.Pad(padding=(0, 40), fill=-1),  # Padding the image from (480, 854) to (560, 854)
                transforms.CenterCrop(size=(560, 840)),  # Cropping the padded image to (560, 840)
                transforms.Resize(size=resolution, antialias=False, interpolation=transforms.InterpolationMode.NEAREST_EXACT)  # Finally resizing to required resolution
            ])

        elif self.spatial_transform == "resize":
            print('Spatial transform: resize with no crop')
            self.video_transform = transforms.Resize((resolution, resolution))
        elif self.spatial_transform == "random_crop":
            self.video_transform = transforms.Compose([
                transforms_video.RandomCropVideo(resolution),
            ])
        elif self.spatial_transform == "":
            self.video_transform = None
        else:
            raise NotImplementedError

        self.load_compressed_segmap = load_compressed_segmap
        if sample_cache is not None:    
            self.sample_cache = sample_cache
        else:
            self.sample_cache = {}
        self.lock = lock
        self.pad_to = pad_to
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
    
    def _compute_sample(self, index):
        
        clip = self.clips[index]
        
        frames, segmap = load_and_transform_frames_cholec80(clip, self.loader, self.img_transform, is_segmap_compressed=self.load_compressed_segmap, 
                                                            fps=self.clip_step if self.clip_step != 1 else None)
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
            
            #frames_tensor = self.video_transform(frames_tensor)
            #segmap = self.segmap_transform(segmap)

        if self.pad_to != self.video_length:
            c,t,h,w = frames_tensor.shape
 
            frames_tensor = torch.cat((frames_tensor, torch.full(size=(c,self.pad_to-t,h,w), fill_value=-1)), dim=1)
        if self.pad_to != segmap.shape[0] and self.clip_step == 1:
            t,h,w = segmap.shape
            segmap = torch.cat((segmap, torch.zeros(size=(self.pad_to-t,h,w))), dim=0)
        segmap = segmap/torch.max(segmap)
            #print('Frames', frames_tensor.shape)
            #print("segmap", segmap.shape)
        
        example = {}
        if self.vae is None:
            example['videos'] = frames_tensor  
        else:           
            with torch.no_grad():
                to_encode = frames_tensor.unsqueeze(0).to(dtype=self.vae.dtype, device=self.vae.device)
                og_frames = to_encode.shape[2]
                # Ensure the encoder receives exactly 49 frames
                if to_encode.shape[2] < 49:
                    pad_size = 49 - to_encode.shape[2]
                    to_encode = torch.nn.functional.pad(to_encode, (0, 0, 0, 0, 0, pad_size))
                elif to_encode.shape[2] > 49:
                    to_encode = to_encode[:, :, :49, :, :]
                
                out_latents = torch.squeeze(self.vae.encode(to_encode).latent_dist.sample() * self.vae.config.scaling_factor)
                out_latents = out_latents[:,:og_frames//4,:,:]

                example["latents"] = out_latents
                del out_latents
                del to_encode

        example["init_img"] = frames_tensor[:,0,...]
        example["frame_stride"] = self.frame_stride
        example["clip_info"] = clip
        example["video_segmap"] = segmap

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
        to_save['video_segmap'] = to_save['video_segmap'].to(torch.int8)
        del to_save['init_img'] # Already cached features

        if self.lock is not None:
            with self.lock:
                self.sample_cache[index] = to_save
        else:
            self.sample_cache[index] = to_save
        return example    

    def _compute_and_save_samples(self, save_dir):
        try:
            if os.path.exists(save_dir):
                shutil.rmtree(save_dir)
        except:
            pass
        os.makedirs(save_dir, exist_ok=True)
        from tqdm import tqdm
        import numpy as np
        batch_size = 16 
        frame_cache = {}
        for i in tqdm(range(0, len(self.clips), batch_size), desc="Precomputing and saving samples"):
            batch_clips = self.clips[i:i+batch_size]
            batch_frames = []
            batch_segmap = []
            from concurrent.futures import ThreadPoolExecutor

            def process_clip(clip):
                frames, segmap = load_and_transform_frames_cholec80(clip, self.loader, self.img_transform, frame_cache, self.load_compressed_segmap)
                frames = torch.cat(frames, 1)
                if self.video_transform is not None:
                    frames = self.video_transform(frames)
                    segmap = self.video_transform(segmap)
                return frames, segmap

            with ThreadPoolExecutor() as executor:
                results = list(executor.map(process_clip, batch_clips))

            for frames in results:
                batch_frames.append(frames)
                batch_segmap.append(segmap)
            
            batch_frames_tensor = torch.stack(batch_frames).to(dtype=self.vae.dtype, device=self.vae.device)
            batch_latents = self.vae.encode(batch_frames_tensor).latent_dist.sample() * self.vae.config.scaling_factor
            
            for j, (frames, segmap, latents) in enumerate(zip(batch_frames, batch_segmap, batch_latents)):
                example = dict()
                #example["image"] = frames
                example["latents"] = latents.to(torch.float32)
                example['init_img'] = frames[:,0,...]
                example["video_segmap"] = segmap
                example["frame_stride"] = self.frame_stride
                example["clip_info"] = batch_clips[j]
                
                img_path_parts = batch_clips[j][0]["img_path"].split('/')
                clip_name = f"{img_path_parts[-2]}_{img_path_parts[-1].split('.')[0]}"
                
                save_path = os.path.join(save_dir, f"{clip_name}.pt")
                torch.save(example, save_path)

                self.precomputed_samples.append(example)

    def _load_precomputed_samples(self, load_dir):
        from glob import glob
        sample_files = glob(os.path.join(load_dir, "*.pt"))
        self.precomputed_samples = []
        for sample_file in sample_files:
            data = torch.load(sample_file)
            example = dict()
            #example["image"] = data["image"]
            example['init_img'] = data['init_img']
            example["video_segmap"] = data["video_segmap"]
            example["latents"] = data["latents"]
            example["frame_stride"] = data["frame_stride"]
            example["clip_info"] = data["clip_info"]
            self.precomputed_samples.append(example)

    def __getitem__(self, index):
        if self.precompute == "load_precomputed" or self.precompute == "compute_and_save":
            if index < len(self.precomputed_samples):
                return self.precomputed_samples[index]
            else:
                random_index = random.randint(0, len(self.precomputed_samples) - 1)
                print(f"Warning: Index {index} out of range. Returning random sample at index {random_index}.")
                return self.precomputed_samples[random_index]
        elif self.precompute == "compute_on_the_fly":
            return self._compute_sample(index)
        elif self.precompute == "compute_and_cache":
            return self._compute_and_cache_sample(index)        

    def __len__(self):
        return len(self.clips)

class FrameDatasetCholec(data.Dataset):
    def __init__(self,
                 data_root,
                 resolution : List,
                 dataset_name="",
                 subset_split="",
                 annotation_dir=None,
                spatial_transform="",
                graph_type="dgl",
                init_dsg_ds=None
                 ):

        self.loader = default_loader
        self.subset_split = subset_split
        self.spatial_transform = spatial_transform
        self.dataset_name = dataset_name

        assert (subset_split in ["train", "test", "all", ""])  # "" means no subset_split directory.

        split_file = os.path.join(data_root, f"{subset_split}_split.txt")
        with open(split_file, "r") as f:
            video_names = f.readlines()
        video_paths = [os.path.join(data_root, "videos", v.strip()) for v in video_names]
        annotation_dir = os.path.join(data_root, "labels")

        self.frames, self.mappings = make_frame_dataset_cholec(video_paths_list=video_paths, annotations_path=annotation_dir, remove_black=False)

        print('[VideoFrameDataset] number of frames:', len(self.frames))

        # data transform
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        if self.spatial_transform == "crop_resize":
            from torchvision.transforms.functional import crop            
            def cropper(img):
                top, left, bottom, right = 0, 36, 480, 768
                if img is None:
                    return top, left, bottom, right
                return crop(img, top, left, bottom, right)
            print('Spatial transform:crop and then resize')
            self.edit_transform = transforms.Compose([
                transforms.Lambda(cropper),
                transforms.Resize(resolution),
            ])
        elif self.spatial_transform == "pad_resize":
            from torchvision.transforms.functional import crop            

            def cropper(img):
                top, left, bottom, right = 0, 36, 480, 768
                if img is None:
                    return top, left, bottom, right
                return crop(img, top, left, bottom, right)               
            self.edit_transform = transforms.Compose([
                transforms.Lambda(cropper),             
                transforms.Pad(padding=(0, 37), fill=-1),  # Padding the image from (480, 854) to (554, 854)
                transforms.CenterCrop(size=(554, 732)),  # Cropping the padded image to (554, 732) -122 to left and right, -61
                transforms.Resize(size=resolution, antialias=True)  # Finally resizing to required resolution
            ])

        elif self.spatial_transform == "resize":
            print('Spatial transform: resize with no crop')
            self.edit_transform = transforms.Resize((resolution, resolution))
        elif self.spatial_transform == "random_crop":
            self.edit_transform = transforms.Compose([
                transforms_video.RandomCropVideo(resolution),
            ])
        elif self.spatial_transform == "":
            self.edit_transform = None
        else:
            raise NotImplementedError
        
        # DSG loading
        ann_path = Path(data_root) / "labels"
        videos_path = Path(data_root)/ "videos"
        sg_ann_path = Path(data_root)/"scene_graph"

        vid_labels_paths = [] 
        for root, dirs, files in os.walk(ann_path):
            for file in files:
                if file.endswith('.json'):
                    # Construct the full file path and add it to the list
                    vid_labels_paths.append((ann_path/file, videos_path/Path(file).stem))        
        if graph_type is not None:
            self.dsg_ds = VideoSceneGraphDataset(vid_labels_paths, sg_ann_path, init_dsg_ds=init_dsg_ds)
            self.graph_type = graph_type
        else:
            self.dsg_ds = None

    
    def __getitem__(self, index):
        # get clip info
        frame = self.frames[index]

        # make clip tensor
        out = load_and_transform_frame_cholec(frame, self.mappings, self.loader, self.img_transform, filter_black=True)
        if out is None:
            return self.__getitem__((index+random.randint(1,256))%len(self))
        C, og_H, og_W = out[0].shape
        
        frame, labels = out
        temp = frame
        C,H,W = frame.shape

        if self.edit_transform is not None:
            frame = self.edit_transform(frame)

        example = dict()
        example["image"] = frame
        example["frame_info"] = labels

        if self.dsg_ds is not None:
            dsg = self.dsg_ds.get_dict(labels[0]['vid_id'], labels[0]['frame_id'])
            bboxes = self.dsg_ds.get_bboxes(labels[0]['vid_id'], labels[0]['frame_id'])
            def transform_bbox(bbox, img_w, img_h):
                # The BBOX is relative to a 240-430 
                pad_left, pad_top, rect_width, rect_height = -61, 37, og_W, og_H
                img_w += pad_left*2
                img_h += pad_top*2
                og_size = 240, 430 # It's divided by 2
                bbox_fullres = [c*2 for c in bbox] # Constant but I could also find the ratio directly
                bbox_rect = [bbox_fullres[0]+pad_left, bbox_fullres[1]+pad_top, bbox_fullres[2]+pad_left, bbox_fullres[3]+pad_top] 
                normalized_bbox = [bbox_rect[0]/img_w, bbox_rect[1]/img_h, bbox_rect[2]/img_w, bbox_rect[3]/img_h] 
                return normalized_bbox   
            example["bboxes"] = [(transform_bbox(bbox[0], W, H), bbox[1]) for bbox in bboxes]

            if self.graph_type == "dgl":
                example["dsgraph"] = to_dgl(dsg)
            elif self.graph_type == "pyg":
                example["dsgraph"] = dsg             
        # Put together into list
        return example

    def __len__(self):
        return len(self.frames)


def extra():
    import torch
    from torchvision import transforms
    from PIL import Image
    import os

    video_dir = "./sam2/notebooks/videos/bedroom"

    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    # Define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load frames into a single tensor
    frames = []
    for frame_name in frame_names:
        frame_path = os.path.join(video_dir, frame_name)
        frame = Image.open(frame_path)
        frame_tensor = transform(frame)
        frames.append(frame_tensor)

    # Stack frames into a single tensor
    frames_tensor = torch.stack(frames)