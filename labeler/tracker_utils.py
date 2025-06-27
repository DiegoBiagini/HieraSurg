import os
import random
import cv2
import torch
#from stable_keypoints_square import run_image_with_context_augmented, find_max_pixel
import numpy as np
import matplotlib.pyplot as plt
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

def create_mask_grid(masks, n_rows=None, n_cols=None):
    L = len(masks)
    
    # If rows/cols not specified, try to create a roughly square grid
    if n_rows is None and n_cols is None:
        n_rows = int(np.ceil(np.sqrt(L)))
        n_cols = int(np.ceil(L / n_rows))
    elif n_rows is None:
        n_rows = int(np.ceil(L / n_cols))
    elif n_cols is None:
        n_cols = int(np.ceil(L / n_rows))
    
    # Get mask dimensions
    h, w = masks[0].shape
    
    # Create empty grid
    grid = np.zeros((h * n_rows, w * n_cols), dtype=masks[0].dtype)
    
    # Fill the grid with masks
    for idx, mask in enumerate(masks):
        if idx >= L:
            break
        i, j = idx // n_cols, idx % n_cols
        grid[i*h:(i+1)*h, j*w:(j+1)*w] = mask
        
    return grid


def get_sam_keypoints(frame, sam2, return_masks=False, points_per_side=16):
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=points_per_side,
        pred_iou_thresh=0.9,
        stability_score_thresh=0.9,
        stability_score_offset=0.9,
        crop_n_layers=1,
        box_nms_thresh=0.9,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=500.0,
        use_m2m=True,
    )
    masks = mask_generator.generate(frame)

    masks = {i: np.expand_dims(mask["segmentation"], axis=0) for i, mask in enumerate(masks)}

    areas = [np.sum(m) for idx, m in masks.items()]
    min_size = 1000
    max_size = 100000
    # Size filtering   
    filtering = [
        (area > min_size) and (area < max_size)
        for area in areas
    ]
    # Use list comprehension to filter masks based on the boolean list
    masks = [mask for mask, keep in zip(masks.values(), filtering) if keep]

    # Find connected components for each mask and discard small ones
    cc_masks_list = []
    for mask in masks:
        new_mask = np.zeros_like(mask)
        # Label connected components
        num_labels, labels = cv2.connectedComponents(np.squeeze(mask).astype(np.uint8), connectivity=4)
        
        biggest_cc = None
        biggest_cc_size = 0
        # Create a new mask for each connected component
        for label in range(1, num_labels):  # Skip background (label 0)
            component_mask = (labels == label)

            if np.sum(component_mask) > biggest_cc_size:
                biggest_cc_size = np.sum(component_mask)
                biggest_cc = component_mask
                
        if biggest_cc is not None and biggest_cc_size > min_size:
            new_mask[np.expand_dims(biggest_cc, axis=0)] = 1                    
            cc_masks_list.append(new_mask)
    masks = cc_masks_list

    # Compare masks and remove highly overlapping ones
    to_keep = [True for _ in range(len(masks))]
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            intersection = np.logical_and(masks[i], masks[j])
            intersection_area = np.sum(intersection)
            area_i = np.sum(masks[i])
            area_j = np.sum(masks[j])
            smaller_area = min(area_i, area_j)

            if intersection_area >= 0.8 * smaller_area:
                if area_i < area_j:
                    to_keep[i] = False
                else:
                    to_keep[j] = False
    masks = [mask for mask, keep in zip(masks, to_keep) if keep]

    if return_masks:
        return masks

    def find_centroid(mask):
        """Find the centroid of a binary mask."""
        mask = np.squeeze(mask)
        if np.sum(mask) == 0:
            return None  # No points in the mask

        # Get the coordinates of the points in the mask
        y_indices, x_indices = np.where(mask)
        
        # Calculate the centroid
        centroid_x = np.mean(x_indices)
        centroid_y = np.mean(y_indices)
        
        return (centroid_x, centroid_y)

    # Calculate centroids for each mask
    return np.array([[find_centroid(mask) for mask in masks]] )

def filter_keypoints(keypoints, frame, radius=5):
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    filtered_keypoints = []

    for keypoint in keypoints[0]:
        x, y = int(keypoint[1].item() * frame.shape[1]), int(keypoint[0].item() * frame.shape[0])
        # Define the region around the keypoint
        x_start = max(0, x - radius)
        x_end = min(gray_frame.shape[1], x + radius)
        y_start = max(0, y - radius)
        y_end = min(gray_frame.shape[0], y + radius)

        # Check if the region is all black
        region = gray_frame[y_start:y_end, x_start:x_end]
        if not np.mean(region) <= 32:
            filtered_keypoints.append(keypoint)

    return filtered_keypoints


def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab20")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.99])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def refine_masks(video_segments):
    refined_masks = {}
    if list(video_segments.items())[0][1] is None:
        return video_segments

    """
    # Get the maximum number of masks in any frame
    max_masks = max(len(masks) for masks in video_segments.values() if masks is not None)

    # Iterate through all frames
    for frame_idx, masks in video_segments.items():
        if masks is None:
            continue
        
        # Get the shape of the masks (assuming all masks have the same shape)
        mask_shape = next(iter(masks.values())).shape if masks else None
        
        # Add missing mask indices with zero masks
        for mask_idx in range(max_masks):
            if mask_idx not in masks:
                # Create a zero mask with the same shape as other masks
                zero_mask = np.zeros(mask_shape, dtype=bool)
                masks[mask_idx] = zero_mask
        
        # Update the video_segments with the modified masks
        video_segments[frame_idx] = masks
    """
    overridden = {i:0 for i in range(len(list(video_segments.items())[0][1]))}
    overridden_by = {i:{} for i in range(len(list(video_segments.items())[0][1]))}
    existing = {i:0 for i in range(len(list(video_segments.items())[0][1]))}

    for frame_idx, masks in video_segments.items():
        # Remove masks that are too small, too big, too similar to others, parts that have too many holes
        # INFO: a mask is a (H, W) array of booleans, masks is a dict of such arrays
        
        areas = [np.sum(m) for idx, m in masks.items()]
        # Size filtering   
        min_size = 1000
        max_size = 100000
        filtering = [
            (area > min_size) and (area < max_size)
            for area in areas
        ]
        for idx, mask in masks.items():
            masks[idx] = mask if filtering[idx] else np.full_like(mask, False, dtype=bool)
        
        # Find connected components for each mask and discard small ones
        for idx, mask in masks.items():
            new_mask = np.zeros_like(mask)
            # Label connected components
            num_labels, labels = cv2.connectedComponents(np.squeeze(mask).astype(np.uint8), connectivity=4)
            
            # Create a new mask for each connected component
            for label in range(1, num_labels):  # Skip background (label 0)
                component_mask = (labels == label)

                if np.sum(component_mask) > min_size:
                    new_mask[np.expand_dims(component_mask, axis=0)] = 1                    
    
            if np.sum(new_mask) > min_size:
                masks[idx] = np.asarray(new_mask, dtype=bool)
            else:
                masks[idx] = np.full_like(mask, False, dtype=bool)
        

        for idx, mask in masks.items():
            binary = np.squeeze(mask).astype(np.uint8)
            binary_inv = cv2.bitwise_not(binary)

            # Copy the binary image for flood filling
            flood_filled = binary_inv.copy()

            # Create a mask for flood filling
            h, w = binary_inv.shape[:2]
            fill_mask = np.zeros((h+2, w+2), np.uint8)

            # Flood fill from the edges
            cv2.floodFill(flood_filled, fill_mask, (0, 0), 255)

            # Invert the result to get the final filled holes
            filled_image = cv2.bitwise_not(fill_mask[1:-1, 1:-1])
            
            # Only use filled version if area isn't too much larger
            filled_area = np.sum(filled_image == 255)
            original_area = np.sum(binary)
            
            if filled_area <= 1.2 * original_area:
                masks[idx] = np.asarray(np.expand_dims(filled_image, axis=0) == 255, dtype=bool)
            else:
                masks[idx] = mask

        # Check which are still existing after the first filtering round
        areas = [np.sum(m) for idx, m in masks.items()]
        for i, area in enumerate(areas):
            if area > 0:
                existing[i] += 1
        # Compare masks and remove highly overlapping ones
        to_keep = [True for _ in range(len(masks.values()))]
        for i in range(len(masks.values())):
            for j in range(i + 1, len(masks.values())):
                intersection = np.logical_and(masks[i], masks[j])
                intersection_area = np.sum(intersection)
                area_i = np.sum(masks[i])
                area_j = np.sum(masks[j])
                smaller_area = min(area_i, area_j)

                iou = intersection_area / (area_i + area_j - intersection_area)
                #if intersection_area > 0.8 * smaller_area:
                if iou >= 0.8:
                    if area_i < area_j:
                        to_keep[i] = False
                        if smaller_area > 0:
                            overridden_by[i][j] = 1 if j not in overridden_by[i] else overridden_by[i][j] + 1
                    else:
                        to_keep[j] = False
                        if smaller_area > 0:
                            overridden_by[j][i] = 1 if i not in overridden_by[j] else overridden_by[j][i] + 1
        for i, keep in enumerate(to_keep):
            if not keep and areas[i] > 0:
                overridden[i] += 1
        for idx, mask in masks.items(): 
            masks[idx] = mask if to_keep[idx] else np.full_like(mask, False, dtype=bool)
        
        refined_masks[frame_idx] = masks
    print(existing)
    print(overridden)
    keep_object = {}
    for i in range(2): # Do it 2 times such that the overlapping objects are merged even when there is no direct overlap of discarded ones
        for obj_id in existing.keys():
            if existing[obj_id] == 0:
                keep_object[obj_id] = False
                continue
            if overridden[obj_id] / existing[obj_id] > 0.5:
                keep_object[obj_id] = False
                # Merge with most overlapping one
                most_overlapping = max(overridden_by[obj_id], key=overridden_by[obj_id].get)
                for frame_idx, masks in refined_masks.items():
                    combined_mask = np.logical_or(refined_masks[frame_idx][most_overlapping], refined_masks[frame_idx][obj_id])
                    if np.sum(combined_mask) < max_size:
                        refined_masks[frame_idx][most_overlapping] = combined_mask
                    else:
                        print("muroi")
            else:
                keep_object[obj_id] = True
    final_masks = {}
    # Create a mapping of old indices to new ordinal indices
    kept_indices = sorted([idx for idx in keep_object.keys() if keep_object[idx]])
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(kept_indices)}
    
    for frame_idx, masks in refined_masks.items():
        final_masks[frame_idx] = {index_map[idx]: m if len(m.shape) == 3 else np.squeeze(m, axis=0) for idx, m in masks.items() if keep_object[idx]}
    return final_masks

def add_bg_black_masks(video_segments, clip_frames):
    final_masks = {}
    if list(video_segments.items())[0][1] is None:
        max_obj_idx = 0
    else:
        max_obj_idx = max(video_segments[0].keys())
    bg_index = max_obj_idx + 1
    black_index = bg_index + 1
       
    for frame_idx, masks in video_segments.items():
        if masks is None:
            masks = {}
        frame = clip_frames[frame_idx]
        
        total_mask = np.expand_dims(np.zeros_like(frame[:,:,0], dtype=bool), 0)
        for idx, mask in masks.items():
            total_mask = np.logical_or(total_mask, mask)
            
        bg_mask = np.logical_not(total_mask)

        # Get the black part of the bg_mask, by looking at the frame
        black_mask = np.all(frame < [30, 30, 30], axis=-1)  # Threshold to identify almost black pixels
        # Remove connected components/submasks in black_mask that have a small area
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(black_mask.astype(np.uint8), connectivity=8)
        min_area = 100  # Define a threshold for the minimum area of connected components

        for label in range(1, num_labels):  # Start from 1 to skip the background
            if stats[label, cv2.CC_STAT_AREA] < min_area:
                black_mask[labels == label] = False
        black_bg_mask = np.logical_and(bg_mask, black_mask)

        bg_mask = np.logical_and(bg_mask, np.logical_not(black_mask))

        masks[bg_index] = np.expand_dims(bg_mask, 0) if len(bg_mask.shape) == 2 else bg_mask
        masks[black_index] = np.expand_dims(black_bg_mask, 0) if len(black_bg_mask.shape) == 2 else black_bg_mask

        final_masks[frame_idx] = masks
    return final_masks
          

from PIL import Image, ImageDraw

def show_anns(anns, pil_image):
    if len(anns) == 0:
        return pil_image
    if "area" in anns[0]:
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    else:
        sorted_anns = anns

    draw = ImageDraw.Draw(pil_image, 'RGBA')

    for ann in sorted_anns:
        if 'segmentation' in ann:
            m = ann['segmentation']
        else:
            m = np.squeeze(ann)
        color = tuple(np.random.randint(0, 256, 3).tolist() + [90])  # Random color with alpha=90
        mask_image = Image.fromarray((m * 255).astype(np.uint8), mode='L')
        draw.bitmap((0, 0), mask_image, fill=color)

    return pil_image

def stitch_frames(clip_frames, stitching_stride  =  2, total_stitched_frames = 4):
    stitched_frames = []
    for i in range(0, len(clip_frames), stitching_stride):
        if i + 3 < len(clip_frames):  # Ensure there are enough frames to stitch
            top_row = np.hstack(clip_frames[i:i+stitching_stride])  # Stitch the first two frames horizontally
            bottom_row = np.hstack(clip_frames[i+stitching_stride:i+total_stitched_frames])  # Stitch the next two frames horizontally
            stitched_frame = np.vstack((top_row, bottom_row))  # Stitch the two rows vertically
            stitched_frames.append(stitched_frame)        

    return stitched_frames
