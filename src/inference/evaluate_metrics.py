import argparse
import os
from typing import List
from tqdm import tqdm
from functools import partial
import torch
from diffusers import CogVideoXTransformer3DModel, AutoencoderKLCogVideoX, CogVideoXDPMScheduler
from finetune.pipeline import CogVideoXPipelineNoPrompt
from inference.controlnet_inference_tests import CogVideoXControlnet, ControlnetCogVideoXPipelineNoPrompt
from segcond.segcond_pipeline import CogVideoXPipeline_VideoSegmap
from segcond.cogvideo_segcond import CogVideoXTransformer3DModel_SegCond, CogVideoXTransformer3DModel_SegCond_Hierarchical
from cogvideox_controlnet.cogvideo_transformer import CustomCogVideoXTransformer3DModel
from segcond.feature_utils import load_radio_model, predict_features_for_frame
from torch.utils.data import Dataset, DataLoader
import json
import cv2
import numpy as np
import imageio
import numpy as np
from PIL import Image
from torchmetrics.image.inception import InceptionScore

# Constants
#N_FRAMES = 16
#COGVIDEO_FRAMES = 16 # Native cogvideo number of frames
H, W = 256, 384

N_SAMPLES_UCOND = 4000 # How many videos to generate in the ucond setting
NUM_STEPS_INFERENCE = 100
SEED = 42

device = "cuda:1"

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

def create_pipeline_ucond(model_path, param = "2b"):
    load_dtype = torch.bfloat16 #if "5b" in param else torch.float16

    transformer = CogVideoXTransformer3DModel.from_pretrained(
        model_path,
        subfolder="transformer",
        torch_dtype=load_dtype,
    )

    vae = AutoencoderKLCogVideoX.from_pretrained(
        model_path, subfolder="vae", torch_dtype=load_dtype
    )

    scheduler = CogVideoXDPMScheduler.from_pretrained(model_path, subfolder="scheduler")

    pipe = CogVideoXPipelineNoPrompt.from_pretrained(
        model_path,
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
    )
    return pipe


def create_pipeline_cnet(model_path, cnet_model_path, param = "2b"):
    load_dtype = torch.bfloat16 #if "5b" in param else torch.float16
    
    transformer = CustomCogVideoXTransformer3DModel.from_pretrained(
        model_path,
        subfolder="transformer", 
        torch_dtype=load_dtype,
    )

    vae = AutoencoderKLCogVideoX.from_pretrained(
        model_path, 
        subfolder="vae",
        torch_dtype=load_dtype
    )

    scheduler = CogVideoXDPMScheduler.from_pretrained(model_path, subfolder="scheduler")

    # Load controlnet from checkpoint
    controlnet = CogVideoXControlnet.from_pretrained(cnet_model_path, torch_dtype=load_dtype)

    pipe = ControlnetCogVideoXPipelineNoPrompt.from_pretrained(
        model_path,
        transformer=transformer,
        vae=vae,
        controlnet=controlnet,
        scheduler=scheduler,
    )#.to(torch.bfloat16)
    return pipe

def create_pipeline_segmap(model_path, param = "2b"):
    load_dtype = torch.bfloat16 #if "5b" in param else torch.float16

    
    transformer = CogVideoXTransformer3DModel_SegCond.from_pretrained(
        model_path,
        subfolder="transformer",
        torch_dtype=load_dtype,
    )

    vae = AutoencoderKLCogVideoX.from_pretrained(
        model_path, subfolder="vae", torch_dtype=load_dtype
    )

    scheduler = CogVideoXDPMScheduler.from_pretrained(model_path, subfolder="scheduler")

    # Load radio model for image features
    radio_model, radio_preprocessor = load_radio_model() 

    pipe = CogVideoXPipeline_VideoSegmap.from_pretrained(
        model_path,
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
    )
    pipe.img_cond_model = {'name':'RADIO', 'model':radio_model.to(device), 'preprocessor':radio_preprocessor}
    return pipe

def create_pipeline_segmap_i2v(model_path, param = "2b"):
    from segcond.cogvideo_segcond import CogVideoXTransformer3DModel_SegCond_I2V
    from segcond.segcond_pipeline import CogVideoXPipeline_VideoSegmap_I2V
    load_dtype = torch.bfloat16 #if "5b" in param else torch.float16

    
    transformer = CogVideoXTransformer3DModel_SegCond_I2V.from_pretrained(
        model_path,
        subfolder="transformer",
        torch_dtype=load_dtype,
    )

    vae = AutoencoderKLCogVideoX.from_pretrained(
        model_path, subfolder="vae", torch_dtype=load_dtype
    )

    scheduler = CogVideoXDPMScheduler.from_pretrained(model_path, subfolder="scheduler")

    pipe = CogVideoXPipeline_VideoSegmap_I2V.from_pretrained(
        model_path,
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
    )
    return pipe

def create_pipeline_segmap_hiera_i2v(model_path, param = "2b", text_cond=None):
    from segcond.cogvideo_segcond import CogVideoXTransformer3DModel_SegCond_Hierarchical_I2V
    from segcond.segcond_pipeline import CogVideoXPipeline_VideoSegmap_I2V
    load_dtype = torch.bfloat16 #if "5b" in param else torch.float16

    
    transformer = CogVideoXTransformer3DModel_SegCond_Hierarchical_I2V.from_pretrained(
        model_path,
        subfolder="transformer",
        torch_dtype=load_dtype,
    )

    vae = AutoencoderKLCogVideoX.from_pretrained(
        model_path, subfolder="vae", torch_dtype=load_dtype
    )

    scheduler = CogVideoXDPMScheduler.from_pretrained(model_path, subfolder="scheduler")

    pipe = CogVideoXPipeline_VideoSegmap_I2V.from_pretrained(
        model_path,
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
    )
    if text_cond == "label_embd":
        pipe.text_cond_model = None
    elif text_cond == "SurgVLP":
        import surgvlp
        from mmengine.config import Config
        configs = Config.fromfile('./segcond/surgvlp_cfgs/config_peskavlp.py')['config']
        # Change the config file to load different models: config_surgvlp.py / config_hecvl.py / config_peskavlp.py

        model, preprocess = surgvlp.load(configs.model_config, device=device)            
        pipe.text_cond_model = {'name':'SurgVLP', 'model':model.to(pipe.dtype).to(device), 'preprocessor':preprocess}

    return pipe


def create_pipeline_segmap_surgvlp(model_path, param = "2b", global_feat=False):
    load_dtype = torch.bfloat16 #if "5b" in param else torch.float16
    
    transformer = CogVideoXTransformer3DModel_SegCond.from_pretrained(
        model_path,
        subfolder="transformer",
        torch_dtype=load_dtype,
    )

    vae = AutoencoderKLCogVideoX.from_pretrained(
        model_path, subfolder="vae", torch_dtype=load_dtype
    )

    scheduler = CogVideoXDPMScheduler.from_pretrained(model_path, subfolder="scheduler")

    # Load radio model for image features
    import surgvlp
    from mmengine.config import Config
    configs = Config.fromfile('./segcond/surgvlp_cfgs/config_peskavlp.py')['config']
    # Change the config file to load different models: config_surgvlp.py / config_hecvl.py / config_peskavlp.py

    model, preprocess = surgvlp.load(configs.model_config)

    pipe = CogVideoXPipeline_VideoSegmap.from_pretrained(
        model_path,
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
    )
    if not global_feat:
        pipe.img_cond_model = {'name':'SurgVLP', 'model':model.to(pipe.dtype).to(device), 
                            'preprocessor':preprocess}
    else:
        pipe.img_cond_model = {'name':'SurgVLP_global', 'model':model.to(pipe.dtype).to(device), 
                            'preprocessor':preprocess}        
    return pipe

def create_pipeline_segmap_hiera(model_path, param = "2b", cond_model = 'RADIO'):
    load_dtype = torch.bfloat16 #if "5b" in param else torch.float16
    
    transformer = CogVideoXTransformer3DModel_SegCond_Hierarchical.from_pretrained(
        model_path,
        subfolder="transformer",
        torch_dtype=load_dtype,
    )

    vae = AutoencoderKLCogVideoX.from_pretrained(
        model_path, subfolder="vae", torch_dtype=load_dtype
    )

    scheduler = CogVideoXDPMScheduler.from_pretrained(model_path, subfolder="scheduler")

    # Load radio model for image features

    pipe = CogVideoXPipeline_VideoSegmap.from_pretrained(
        model_path,
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
    )
    if cond_model == "RADIO":
        radio_model, radio_preprocessor = load_radio_model() 

        pipe.img_cond_model = {'name':'RADIO', 'model':radio_model, 'preprocessor':radio_preprocessor}
    elif cond_model == "SurgVLP":
        import surgvlp
        from mmengine.config import Config
        configs = Config.fromfile('./segcond/surgvlp_cfgs/config_peskavlp.py')['config']
        # Change the config file to load different models: config_surgvlp.py / config_hecvl.py / config_peskavlp.py

        model, preprocess = surgvlp.load(configs.model_config, device=device)            
        pipe.text_cond_model = {'name':'SurgVLP', 'model':model.to(pipe.dtype).to(device), 'preprocessor':preprocess}
        pipe.img_cond_model = {'name':'SurgVLP', 'model':model.to(pipe.dtype).to(device), 'preprocessor':preprocess}        

    return pipe

def create_dataset(data_path, batch_size, n_workers, model_type, out_frames, cogvideo_frames, ext_args=None):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if model_type == 'ucond':
        # Create fake dataset that returns random stuff
        """
        class DummyDataset(Dataset):
            def __init__(self, size=N_SAMPLES_UCOND):
                self.size = size
            
            def __len__(self):
                return self.size
                
            def __getitem__(self, idx):
                return torch.randn(3, N_FRAMES, H, W)
        
        dataset = DummyDataset()
        dl = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=n_workers,
            shuffle=False
        )
        """
        from finetune.dataset.frame_dataset import VideoFrameDatasetCholec80, cholec_collate_fn

        kwargs = {
            'data_root': data_path,
            'resolution': [H, W],
            "dataset_name": "CholecT80Video",
            "video_length": out_frames,
            "subset_split": "test",
            "spatial_transform": "crop_resize",
            "clip_step": 1,
            "precompute": "compute_on_the_fly", # One of load_precomputed, compute_on_the_fly, compute_and_save, compute_and_cache
            "vae": None,
            'pad_to': cogvideo_frames

        }
        dataset = VideoFrameDatasetCholec80(**kwargs)
        
        dl = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=n_workers,
            pin_memory=False,
            drop_last=True,
            collate_fn=cholec_collate_fn,
            prefetch_factor=2
        )

    elif model_type == 'cnet':
        from cogvideox_controlnet.training.controlnet_dataset_cholec import VideoFrameDatasetCholecControlNet, cholec_collate_fn
        kwargs = {
            'data_root': data_path,
            'resolution': [H, W],
            "dataset_name": "CholecT80VideoControlNet",
            "video_length": out_frames,
            "spatial_transform": "crop_resize",
            "clip_step": 1,
            "precompute": "compute_on_the_fly",
            "vae": None,
            "segmap_dropout": 0,
            "load_compressed_segmap": True,
            'pad_to': cogvideo_frames,
            'horizontal_flip': False,
            'vertical_flip': False            
        }
        
        dataset = VideoFrameDatasetCholecControlNet(**kwargs)
        
        dl = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=n_workers,
            pin_memory=False,
            drop_last=True,
            collate_fn=cholec_collate_fn,
            prefetch_factor=2
        )
    elif model_type == 'segmap':
        from finetune.dataset.frame_dataset import VideoFrameDatasetCholecSegmap, videosegmap_collate_fn
        kwargs = {
            'data_root': data_path,
            'resolution': [H, W],
            "dataset_name": "CholecT80VideoSegmap",
            "video_length": out_frames,
            "spatial_transform": "crop_resize",
            "clip_step": 1,
            "precompute": "compute_on_the_fly",
            "vae": None,
            "segmap_dropout": 0,
            "load_compressed_segmap": True,
            'pad_to': cogvideo_frames,
            'horizontal_flip': False,
            'vertical_flip': False
        }
        
        dataset = VideoFrameDatasetCholecSegmap(**kwargs)
        
        dl = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=n_workers,
            pin_memory=False,
            drop_last=True,
            collate_fn=videosegmap_collate_fn,
            prefetch_factor=2
        )
    elif model_type == 'segmap_hiera' or model_type == "segmap_hiera_surgvlp":
        from finetune.dataset.frame_dataset_hiera import VideoFrameDatasetCholecSegmapHiera, videosegmap_collate_fn_hiera
        kwargs = {
            'data_root': data_path,
            'resolution': [H, W],
            "dataset_name": "CholecT80VideoSegmapHiera",
            "video_length": out_frames,
            "spatial_transform": "crop_resize",
            "clip_step": 1,
            "precompute": "compute_on_the_fly",
            "vae": None,
            "segmap_dropout": 0,
            "load_compressed_segmap": True,
            'pad_to': cogvideo_frames,
            'annotations_path': ext_args['annotations_path'],
            'horizontal_flip': False,
            'vertical_flip': False,
            'text_cond': "label_emb" if model_type == "segmap_hiera" else "SurgVLP"         
        }
        
        dataset = VideoFrameDatasetCholecSegmapHiera(**kwargs)
        
        dl = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=n_workers,
            pin_memory=False,
            drop_last=True,
            collate_fn=videosegmap_collate_fn_hiera,
            prefetch_factor=2
        )        
    return dl

def pipeline_inference(pipeline, batch, device, model_type, batch_size, out_frames, cogvideo_frames, ext_args=None):
    global SEED
    with torch.no_grad():
        scheduler_args = {}
        generator = torch.Generator().manual_seed(SEED)
        pipeline = pipeline.to(device)

        if "variance_type" in pipeline.scheduler.config:
            variance_type = pipeline.scheduler.config.variance_type

            if variance_type in ["learned", "learned_range"]:
                variance_type = "fixed_small"

            scheduler_args["variance_type"] = variance_type

        if model_type == 'ucond':
            pipeline_args = {
                "num_inference_steps": NUM_STEPS_INFERENCE,
                "height": H,
                "width": W,
                "num_frames": cogvideo_frames, 
                "batch_size": batch_size           
            }
            out = pipeline(**pipeline_args, generator=generator, output_type="pt").frames
            conds = {}
        elif model_type == 'cnet':
            pipeline_args = {
                "num_inference_steps": NUM_STEPS_INFERENCE,
                "height": H,
                "width": W,
                "num_frames": cogvideo_frames,
                "device": torch.device(device),
                #"batch_size": batch_size           

            }
            controlnet_latents = batch['controlnet_videos'].to(device).to(pipeline.dtype)
            if pipeline.controlnet.in_channels == 3 and controlnet_latents.shape[2] == 1:
                controlnet_latents = controlnet_latents.repeat(1,1,3,1,1)
            #n_frames_latent = controlnet_latents.shape[1]
            #rep = (COGVIDEO_FRAMES+n_frames_latent-1)//n_frames_latent
            #controlnet_latents = controlnet_latents.repeat(1,rep,1,1,1)[:,:COGVIDEO_FRAMES,...]
            pipeline_args['controlnet_latents'] = controlnet_latents       

            out = pipeline(**pipeline_args, generator=generator, output_type="pt").frames.to(torch.float32)
            conds = {'cnet':controlnet_latents.to(torch.float32)}                
        elif model_type == 'segmap':
            pipeline_args = {
                "num_inference_steps": NUM_STEPS_INFERENCE,
                "height": H,
                "width": W,
                "num_frames": cogvideo_frames,
                "out_frames": out_frames
                #"device": torch.device(device)       
            }
            pipeline_args['init_img'] = batch['init_img'].to(device).to(pipeline.dtype)
            #pipeline_args['init_img'] = (torch.unsqueeze(batch['video_segmap'][:,0,...], dim=1).repeat(1,3,1,1).to(device).to(pipeline.dtype)*2)-1

            #n_frames_segmap = video_segmap.shape[1] 
            #rep = (COGVIDEO_FRAMES+n_frames_segmap-1)//n_frames_segmap
            #video_segmap = video_segmap.repeat(1,rep,1,1,1)[:,:COGVIDEO_FRAMES,...]
            pipeline_args['segmap'] = batch['video_segmap'].to(device).to(pipeline.dtype)
            out = pipeline(**pipeline_args, generator=generator, output_type="pt").frames
            conds = {'init_img':pipeline_args['init_img'].cpu(), 'segmap': batch['video_segmap']}                
        elif model_type == 'segmap_hiera' or model_type == "segmap_hiera_surgvlp":
            pipeline_args = {
                "num_inference_steps": NUM_STEPS_INFERENCE,
                "height": H,
                "width": W,
                "num_frames": cogvideo_frames,
                "out_frames": out_frames,
                #"device": torch.device(device) 
                'hiera_args':{
                    'phase_start_step': ext_args['phase_start_step'], 
                    'phase_end_step': ext_args['phase_end_step'],
                    'triplet_start_step': ext_args['triplet_start_step'],
                    'triplet_end_step': ext_args['triplet_end_step'],
                    'phase': move_to_device(batch['sem_info']['phase'], device,),
                    'triplet': move_to_device(batch['sem_info']['triplet'], device),
                    "text_cond": "label_emb" if model_type == "segmap_hiera" else "SurgVLP"
                }               
            }
            pipeline_args['init_img'] = batch['init_img'].to(device).to(pipeline.dtype)
            #n_frames_segmap = video_segmap.shape[1] 
            #rep = (COGVIDEO_FRAMES+n_frames_segmap-1)//n_frames_segmap
            #video_segmap = video_segmap.repeat(1,rep,1,1,1)[:,:COGVIDEO_FRAMES,...]
            pipeline_args['segmap'] = batch['video_segmap'].to(device).to(pipeline.dtype)
            out = pipeline(**pipeline_args, generator=generator, output_type="pt").frames
            conds = {'init_img':batch['init_img'], 'segmap': batch['video_segmap']}                            

        SEED += batch_size
        out = out[:,:out_frames,...]
        return out, conds

def setup_metric(metric):
    if metric == 'fid':
        # FID is available in torchmetrics
        from torchmetrics.image.fid import FrechetInceptionDistance
        return FrechetInceptionDistance(feature=2048, normalize= False)
    elif metric == 'fvd':
        # FVD requires custom implementation since not in torchmetrics
        from tools.fvd import FrechetVideoDistance
        return FrechetVideoDistance()
    elif metric == 'kid':
        from torchmetrics.image.kid import KernelInceptionDistance
        return KernelInceptionDistance(feature=2048)
    elif metric == 'kvd':
        # KVD requires custom implementation since not in torchmetrics 
        from tools.fvd import KernelVideoDistance
        return KernelVideoDistance()
    elif metric == 'is':
        return InceptionScore(normalize=False)
    else:
        raise ValueError(f"Metric {metric} not supported")

def load_video(video_path):
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame indices to sample at 1 fps
    sample_indices = [int(i * fps) for i in range(int(total_frames / fps))]
    
    frames = []
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx in sample_indices:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            
        frame_idx += 1
        
    cap.release()
    
    # Stack frames and convert to tensor
    frames = np.stack(frames)
    video_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2) # [T, C, H, W]
    video_tensor = video_tensor.float() / 255.0 # Normalize to [0,1]
    
    return video_tensor

def metric_update(metric_name, metric, samples, real=True):
    # Input should be a tensor of shape [B, C, T, H, W], range [0,1]
    # 
    if metric_name == 'fid':
        if not real:
            samples = torch.permute(samples, dims=(0,2,1,3,4))
        # Sample is a video, so we need to pass it to FID as a much bigger batch of frames
        # New shape [B*T, C, H, W]
        b,c,t,h,w = samples.shape
        new_samples = (samples.reshape(b*t, c,h,w) * 255.0).to(torch.uint8)
        metric.update(new_samples, real=real)
    elif metric_name == 'fvd':
        if not real:
            samples = torch.permute(samples, dims=(0,2,1,3,4))
        else:
            samples = (samples+1)/2
        metric.update(samples, real=real)
    elif metric_name == 'kvd':
        if not real:
            samples = torch.permute(samples, dims=(0,2,1,3,4))
        else:
            samples = (samples+1)/2
        metric.update(samples, real=real)
    elif metric_name == 'is':
        if not real:
            samples = torch.permute(samples, dims=(0,2,1,3,4))
        # Sample is a video, so we need to pass it to IS as a much bigger batch of frames
        # New shape [B*T, C, H, W]
        b,c,t,h,w = samples.shape
        new_samples = (samples.reshape(b*t, c,h,w) * 255.0).to(torch.uint8)
        metric.update(new_samples)        

pipeline_builder = {
    "cogvideo2b": partial(create_pipeline_ucond, param="2b"),
    "cogvideo2b_cnet": partial(create_pipeline_cnet, param="2b"),
    "cogvideo2b_segmap": partial(create_pipeline_segmap, param="2b"),
    "cogvideo2b_segmap_i2v": partial(create_pipeline_segmap_i2v, param="2b"),
    "cogvideo2b_segmap_hiera_i2v_surgvlp": partial(create_pipeline_segmap_hiera_i2v, param="2b", text_cond="SurgVLP"),
    "cogvideo2b_segmap_surgvlp": partial(create_pipeline_segmap_surgvlp, param="2b"),
    "cogvideo2b_segmap_surgvlp_global": partial(create_pipeline_segmap_surgvlp, param="2b", global_feat=True),    
    "cogvideo2b_segmap_hiera": partial(create_pipeline_segmap_hiera, param="2b"),
    "cogvideo2b_segmap_hiera_surgvlp": partial(create_pipeline_segmap_hiera, param="2b", cond_model="SurgVLP"),
    "cogvideo5b": partial(create_pipeline_ucond, param="5b"),
    "cogvideo5b_cnet": partial(create_pipeline_cnet, param="5b"),
    "cogvideo5b_segmap": partial(create_pipeline_segmap, param="5b"),
}

dataloader_builder = {
    "cogvideo2b": partial(create_dataset, model_type = 'segmap'), #Segmap for compatibility
    "cogvideo2b_cnet": partial(create_dataset, model_type = 'cnet'),
    "cogvideo2b_segmap": partial(create_dataset, model_type = 'segmap'),
    "cogvideo2b_segmap_i2v": partial(create_dataset, model_type = 'segmap'),
    "cogvideo2b_segmap_hiera_i2v_surgvlp": partial(create_dataset, model_type = 'segmap_hiera_surgvlp'),
    "cogvideo2b_segmap_surgvlp": partial(create_dataset, model_type = 'segmap'),
    "cogvideo2b_segmap_surgvlp_global": partial(create_dataset, model_type = 'segmap'),
    "cogvideo2b_segmap_hiera": partial(create_dataset, model_type = 'segmap_hiera'),
    "cogvideo2b_segmap_hiera_surgvlp": partial(create_dataset, model_type = 'segmap_hiera_surgvlp'),
    "cogvideo5b": partial(create_dataset, model_type = 'ucond'),
    "cogvideo5b_cnet": partial(create_dataset, model_type = 'cnet'),
    "cogvideo5b_segmap": partial(create_dataset, model_type = 'segmap'),
}

pipeline_inference_builder = {
    "cogvideo2b": partial(pipeline_inference, model_type = 'ucond'),
    "cogvideo2b_cnet": partial(pipeline_inference, model_type = 'cnet'),
    "cogvideo2b_segmap": partial(pipeline_inference, model_type = 'segmap'),
    "cogvideo2b_segmap_i2v": partial(pipeline_inference, model_type = 'segmap'),
    "cogvideo2b_segmap_hiera_i2v_surgvlp": partial(pipeline_inference, model_type = 'segmap_hiera_surgvlp'),
    "cogvideo2b_segmap_surgvlp": partial(pipeline_inference, model_type = 'segmap'),
    "cogvideo2b_segmap_surgvlp_global": partial(pipeline_inference, model_type = 'segmap'),
    "cogvideo2b_segmap_hiera": partial(pipeline_inference, model_type = 'segmap_hiera'),
    "cogvideo2b_segmap_hiera_surgvlp": partial(pipeline_inference, model_type = 'segmap_hiera_surgvlp'),
    "cogvideo5b": partial(pipeline_inference, model_type = 'ucond'),
    "cogvideo5b_cnet": partial(pipeline_inference, model_type = 'cnet'),
    "cogvideo5b_segmap": partial(pipeline_inference, model_type = 'segmap'),    
}

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate metrics on video generation results')
    
    parser.add_argument('--main_model_path', type=str, required=True,
                        help='Path to the model checkpoint to evaluate')
    parser.add_argument('--cnet_model_path', type=str, required=False,
                        help='Path to the CNet model checkpoint to evaluate')

    parser.add_argument('--model_type', type=str, required=True, choices=['cogvideo2b', 'cogvideo2b_cnet', 'cogvideo2b_segmap', 'cogvideo2b_segmap_surgvlp', 'cogvideo2b_segmap_surgvlp_global',
                                                                          'cogvideo5b', 'cogvideo5b_cnet', 'cogvideo5b_segmap', 'cogvideo2b_segmap_hiera', "cogvideo2b_segmap_hiera_surgvlp",
                                                                          "cogvideo2b_segmap_i2v", "cogvideo2b_segmap_hiera_i2v_surgvlp"],
                                    help='Type of the model to evaluate')

    parser.add_argument('--cogvideo_frames', type=int, default=49, 
                        help='Native frame length of CogVideo, how many are generated')
    parser.add_argument('--out_frames', type=int, default=16, 
                        help='How many frames will be outputted')
    ### Args for hiera
    parser.add_argument(
        "--phase_start_step",
        type=int,
        default=1000,
        help=("The step at which to start using phase embeddings."),
    )
    parser.add_argument(
        "--phase_end_step",
        type=int,
        default=750,
        help=("The step at which to stop using phase embeddings."),
    )    
    parser.add_argument(
        "--triplet_start_step",
        type=int,
        default=1000,
        help=("The step at which to start using triplet embeddings."),
    )
    parser.add_argument(
        "--triplet_end_step",
        type=int,
        default=500,
        help=("The step at which to stop using triplet embeddings."),
    )
    parser.add_argument(
        "--annotations_path",
        type=str,
        default=None,
        help=("A folder containing the annotations."),
    )
    
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='Base directory containing the evaluation dataset')
    parser.add_argument('--metrics', type=str, nargs='+', default=['fvd', 'fid', 'kvd', 'is'],
                        help='List of metrics to evaluate. Options: fvd, fid, kvd, is')
    
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for evaluation')
    
    parser.add_argument('--num_batches', type=int, default=-1,
                        help='Number of batches to run eval on')
    
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')

    parser.add_argument("--use_precomputed", action="store_true",
                        help="If true, use will not use the model to compute metrics, but will use precomputed samples instead")
    parser.add_argument("--precomputed_samples_dir", type=str, default=None,
                        help="Directory containing precomputed samples")
    
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--max_out_samples', type=int, default=-1, help='How many batches of samples to save for the model in the out dir, all if -1')
    
    return parser.parse_args()

def validate_args(args):
    # Check if model path exists
    if not os.path.exists(args.main_model_path):
        raise ValueError(f"Model path does not exist: {args.main_model_path}")
    # Check that if using cnet then cnet_model_path exists
    if args.model_type.endswith("_cnet") and not os.path.exists(args.cnet_model_path):
        raise ValueError(f"CNet model path does not exist: {args.cnet_model_path}")
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        raise ValueError(f"Data directory does not exist: {args.data_dir}")
    
    # Validate metrics
    valid_metrics = {'fvd', 'fid', 'kvd', 'is'}

    invalid_metrics = set(args.metrics) - valid_metrics
    if invalid_metrics:
        raise ValueError(f"Invalid metrics: {invalid_metrics}. Valid options are: {valid_metrics}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

def main():
    args = parse_args()
    validate_args(args)
    ext_args = {key: getattr(args, key) for key in ['phase_start_step', 'phase_end_step', 'triplet_start_step', 'triplet_end_step', 'annotations_path'] if hasattr(args, key)}


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
        print(f"Metrics to evaluate: {args.metrics}")
    
        # Model choice loading
        pipeline = pipeline_builder[args.model_type](**load_args)

    # Data loading depending on the model choice
    dl = dataloader_builder[args.model_type](batch_size=args.batch_size, n_workers=args.num_workers, data_path = args.data_dir, 
                                            out_frames=args.out_frames, cogvideo_frames=args.cogvideo_frames, ext_args=ext_args)

    # Instantiate metrics
    metrics = [setup_metric(metric).to(device) for metric in args.metrics]

    if args.use_precomputed:
        # Load the gt samples
        n_batch = 0
        for batch in tqdm(dl):
            gt_samples = batch['videos'].to(device)
            for i, metric_name in enumerate(args.metrics):
                metric_update(metric_name, metrics[i], gt_samples, real=True)
            n_batch += 1
            if args.num_batches != -1 and n_batch > args.num_batches:
                break
        # Load the precomputed samples, which are videos in a folder
        # List the videos in the folder
        
        precomputed_samples = [os.path.join(args.precomputed_samples_dir, f) for f in os.listdir(args.precomputed_samples_dir) if f.endswith('.mp4')]

        # Create batches of videos
        cur_batch = []
        for video_path in precomputed_samples:
            video = load_video(video_path)
            cur_batch.append(video)
            if len(cur_batch) == args.batch_size:
                for i, metric_name in enumerate(args.metrics):
                    metric_update(metric_name, metrics[i], torch.stack(cur_batch).to(device), real=False)
                cur_batch = []
    else:
        # Inference and Metrics computation
        saved_videos = 0
        saved_videos_real = 0
        n_batch = 0
        for batch in tqdm(dl):
            print(batch['clip_info'])
            gen_samples, conds = pipeline_inference_builder[args.model_type](pipeline, batch, batch_size = args.batch_size, device=device,
                                                                             out_frames=args.out_frames, cogvideo_frames=args.cogvideo_frames, ext_args=ext_args)
            
            gt_samples = batch['videos'].to(device)
            for i, metric_name in enumerate(args.metrics):
                metric_update(metric_name, metrics[i], gen_samples, real=False)
                metric_update(metric_name, metrics[i], gt_samples, real=True)             


            # Save generated videos if output folder is specified
            if args.output_dir is not None:
                os.makedirs(args.output_dir, exist_ok=True)
                # Iterate through batch dimension
                for b in range(gt_samples.shape[0]):
                    video = ((gt_samples[b].permute(1,2,3,0).cpu().numpy()+1)/2 * 255).astype(np.uint8)                      
                    out_path = os.path.join(args.output_dir, f"real_video_{saved_videos_real:04d}.mp4")
                    with imageio.get_writer(out_path, fps=1, codec='libx264', quality=10) as writer:
                        for frame in video:
                            writer.append_data(frame)     
                    saved_videos_real += 1                     
                for b in range(gen_samples.shape[0]):
                    if args.max_out_samples != -1 and saved_videos >= args.max_out_samples:
                        break
                    # Convert to uint8 format expected by video writer
                    video = (gen_samples[b].permute(0,2,3,1).float().cpu().numpy() * 255).astype(np.uint8)
                    out_path = os.path.join(args.output_dir, f"gen_video_{saved_videos:04d}.mp4")
                    with imageio.get_writer(out_path, fps=1, codec='libx264', quality=10) as writer:
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
                        with imageio.get_writer(out_path, fps=1, codec='libx264', quality=10) as writer:
                            for frame in segmap:
                                writer.append_data(frame)
                    if 'cnet' in conds:
                        cnet = (conds['cnet'][b]+1)/2
                        cnet = (cnet.permute(0,2,3,1).cpu().numpy() * 255).astype(np.uint8)
                        out_path = os.path.join(args.output_dir, f"cnet_{saved_videos:04d}.mp4")
                        with imageio.get_writer(out_path, fps=1, codec='libx264', quality=10) as writer:
                            for frame in cnet:
                                writer.append_data(frame)
                    saved_videos += 1

            n_batch += 1
            if args.num_batches != -1 and n_batch >= args.num_batches:
                break
    
    results = {}
    for i, metric in enumerate(metrics):
        metric_name = args.metrics[i]
        metric_value = metric.compute()
        if isinstance(metric_value, torch.Tensor):
            metric_value = metric_value.item()
        if metric_name == 'is':
            metric_value = metric_value[0].item()
        results[metric_name] = metric_value
        print(f"{args.metrics[i]}: {metric_value}")

    # Write results to a JSON file
    output_path = os.path.join(args.output_dir, "metrics.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
if __name__ == "__main__":
    main()
