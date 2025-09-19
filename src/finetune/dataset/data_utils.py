import torch
import numpy as np
import cv2

def masks_to_tensor(masks):

    if not masks:
        return torch.empty(0)

    num_frames = len(masks)
    num_objects = len(next(iter(masks.values())))
    
    # Get the shape of a single mask
    sample_mask = next(iter(next(iter(masks.values())).values()))
    _, H, W = sample_mask.shape

    # Initialize the output tensor
    output = torch.zeros((num_frames, H, W), dtype=torch.long)

    for frame, frame_masks in masks.items():
        for obj_id, mask in frame_masks.items():
            mask = mask.squeeze()
            output[frame][mask] = obj_id

    return output


def inpaint_object_in_masks(frames, frames_masks, to_remove):
    edited_frames = []
    for idx, og_frame in enumerate(frames):
        for obj_id, mask in frames_masks[idx].items():
            if obj_id == to_remove:
                frame = og_frame.numpy()
                # Reshape frame to (H, W, 3) for cv2 compatibility
                frame = np.transpose(frame, (2, 3, 0, 1)).squeeze(axis=-1)
                # Convert frame from range [-1, 1] to [0, 255]
                frame = ((frame + 1) * 127.5).astype(np.uint8)
                # Create a copy of the region
                mask_bool = np.squeeze(mask).astype(bool)
                
                inpainted = cv2.inpaint(frame, mask_bool.astype(np.uint8), 
                                    inpaintRadius=20, flags=cv2.INPAINT_TELEA)
                
                frame[mask_bool] = inpainted[mask_bool]
                # Apply Gaussian blur to the masked region and its edges
                # Create a dilated mask to include edges
                kernel = np.ones((7,7), np.uint8)
                dilated_mask = cv2.dilate(mask_bool.astype(np.uint8), kernel, iterations=1)
                # Apply blur to the region including edges
                blurred = cv2.GaussianBlur(frame, (11,11), 0)
                frame[dilated_mask.astype(bool)] = blurred[dilated_mask.astype(bool)]
                
                # Convert frame back to range [-1, 1]
                frame = (frame.astype(np.float32) - 127.5) / 127.5
                # Reshape frame back to (3, 1, H, W)
                frame = np.expand_dims(np.transpose(frame, (2, 0, 1)), axis=1)
                
                edited_frames.append(torch.as_tensor(frame))
    return edited_frames

def segmentation_to_edges(segmap):
    # Segmap is a 16xHxW tensor with flat surfaces of the same value(they are masks)
    # We want to get the edges of each mask and output a single image for each frame

    edges = []
    for frame_idx in range(segmap.shape[0]):
        frame = segmap[frame_idx].cpu().numpy()
        edge_frame = np.zeros_like(frame)
        for obj_id in np.unique(frame):
            mask = (frame == obj_id).astype(np.uint8)*255
            edges_mask = cv2.Canny(mask, 100, 200)
            edge_frame = np.maximum(edge_frame, edges_mask)
        # add to list as rgb images
        edge_frame = cv2.cvtColor(edge_frame.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        edges.append(edge_frame)
    return edges

def move_tensors_to_device(obj, device):
    for attr_name in dir(obj):
        attr_value = getattr(obj, attr_name)
        if isinstance(attr_value, torch.Tensor):
            setattr(obj, attr_name, attr_value.to(device))
    return obj
