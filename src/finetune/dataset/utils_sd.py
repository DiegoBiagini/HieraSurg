
from glob import glob
import os
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from finetune.dataset.dataset import get_dataset


from .frame_dataset import cholec_collate_fn, videosegmap_collate_fn
from .frame_dataset_hiera import videosegmap_collate_fn_hiera



def get_dataloader(rank, args, logger, split = 'train'):
    # Setup data:
    kwargs = {
        'data_root': args.data_path,
        'resolution': args.image_size,
        "dataset_name": "CholecT50",
        "subset_split": split,
        "spatial_transform": "pad_resize",
        "sd_training": True,
        "graph_type": args.graph_type,
        "init_dsg_ds": args.dsg_ds if hasattr(args, 'dsg_ds') else None
        }

    dataset = get_dataset(args.dataset_name, **kwargs)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=args.gpu_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=cholec_collate_fn if args.dataset_name == "CholecT50" else None # TODO
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    return sampler, loader

def get_dataloader_video(rank, args, logger, split = 'train', precompute="compute_and_cache"):
    # Setup data:
    kwargs = {
        'data_root': args.data_path,
        'resolution': args.image_size,
        "dataset_name": args.dataset_name,
        "video_length": args.video_length,
        "subset_split": split,
        "spatial_transform": "crop_resize",
        "clip_step": args.clip_step if hasattr(args, "clip_step") else 1,
        "precompute": precompute, # One of load_precomputed, compute_on_the_fly, compute_and_save, compute_and_cache
        "vae": args.vae,
        "pad_to": args.max_num_frames
        }
    kw_args = vars(args).copy()
    # im sorry
    del kw_args['data_path']
    del kw_args['image_size']
    del kw_args['max_num_frames']
    del kw_args['global_seed']
    del kw_args['gpu_batch_size']
    del kw_args['num_workers']


    for key, value in kw_args.items():
        if key not in kwargs:
            kwargs[key] = value
    dataset = get_dataset(args.dataset_name, **kwargs)
    if dist.is_initialized():
        
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=rank,
            shuffle=True,
            seed=args.global_seed
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    loader = DataLoader(
        dataset,
        batch_size=args.gpu_batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
        collate_fn=cholec_collate_fn if args.dataset_name == "CholecT50" or args.dataset_name == "CholecT50Video" else None # TODO
    )
    try:
        logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")
    except:
        logger.print(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    return sampler, loader

def get_dataloader_videosegmap(rank, args, logger, precompute="compute_and_cache"):
    # Setup data:
    from multiprocessing import Manager
    manager = Manager()
    kwargs = {
        'data_root': args.data_path,
        'resolution': args.image_size,
        "dataset_name": "CholecT80VideoSegmap",
        "video_length": args.video_length,
        "spatial_transform": "crop_resize",
        "clip_step": args.clip_step if hasattr(args, "clip_step") else 1,
        "precompute": precompute,
        "vae": args.vae,
        "load_compressed_segmap": True,
        "pad_to": args.max_num_frames,
        #"sample_cache": None if args.num_workers == 0 else manager.dict(),
        #"lock": None if args.num_workers == 0 else manager.Lock()
        }
    kw_args = vars(args).copy()
    # im sorry
    del kw_args['data_path']
    del kw_args['image_size']
    del kw_args['max_num_frames']
    del kw_args['global_seed']
    del kw_args['gpu_batch_size']
    del kw_args['num_workers']


    for key, value in kw_args.items():
        if key not in kwargs:
            kwargs[key] = value

    dataset = get_dataset(args.dataset_name, **kwargs)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size() if dist.is_initialized() else 1,
        rank=rank,
        shuffle=True,
        seed=args.global_seed if not args.global_seed is None else 0
    )
    if args.dataset_name == "CholecT50" or args.dataset_name == "CholecT50Video":
        collate_fn = cholec_collate_fn
    elif args.dataset_name == "CholecT80VideoSegmap":
        collate_fn = videosegmap_collate_fn
    elif args.dataset_name == "CholecT80VideoSegmapHiera":
        collate_fn = videosegmap_collate_fn_hiera
    else:
        collate_fn = None

    loader = DataLoader(
        dataset,
        batch_size=args.gpu_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
        collate_fn=collate_fn,
        persistent_workers=args.num_workers > 0
    )
    print(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    return sampler, loader