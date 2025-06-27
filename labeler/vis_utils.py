import torch
import torch.nn.functional as F
import numpy as np
import cv2

def get_pca_map(
    feature_map: torch.Tensor,
    img_size,
    interpolation="bicubic",
    return_pca_stats=False,
    pca_stats=None,
):
    """
    feature_map: (1, h, w, C) is the feature map of a single image.
    """
    if feature_map.shape[0] != 1:
        # make it (1, h, w, C)
        feature_map = feature_map[None]
    if pca_stats is None:
        reduct_mat, color_min, color_max = get_robust_pca(
            feature_map.reshape(-1, feature_map.shape[-1])
        )
    else:
        reduct_mat, color_min, color_max = pca_stats
    pca_color = feature_map @ reduct_mat
    pca_color = (pca_color - color_min) / (color_max - color_min)
    pca_color = pca_color.clamp(0, 1)
    pca_color = F.interpolate(
        pca_color.permute(0, 3, 1, 2),
        size=img_size,
        mode=interpolation,
    ).permute(0, 2, 3, 1)
    pca_color = pca_color.cpu().numpy().squeeze(0)
    if return_pca_stats:
        return pca_color, (reduct_mat, color_min, color_max)
    return pca_color

def get_robust_pca(features: torch.Tensor, m: float = 2, remove_first_component=False):
    # features: (N, C)
    # m: a hyperparam controlling how many std dev outside for outliers
    assert len(features.shape) == 2, "features should be (N, C)"
    with torch.amp.autocast(device_type=features.device.type, enabled=False):    
        reduction_mat = torch.pca_lowrank(features, q=3, niter=20)[2]
    colors = features @ reduction_mat
    if remove_first_component:
        colors_min = colors.min(dim=0).values
        colors_max = colors.max(dim=0).values
        tmp_colors = (colors - colors_min) / (colors_max - colors_min)
        fg_mask = tmp_colors[..., 0] < 0.2
        reduction_mat = torch.pca_lowrank(features[fg_mask], q=3, niter=20)[2]
        colors = features @ reduction_mat
    else:
        fg_mask = torch.ones_like(colors[:, 0]).bool()
    d = torch.abs(colors[fg_mask] - torch.median(colors[fg_mask], dim=0).values)
    mdev = torch.median(d, dim=0).values
    s = d / mdev
    try:
        rins = colors[fg_mask][s[:, 0] < m, 0]
        gins = colors[fg_mask][s[:, 1] < m, 1]
        bins = colors[fg_mask][s[:, 2] < m, 2]
        rgb_min = torch.tensor([rins.min(), gins.min(), bins.min()])
        rgb_max = torch.tensor([rins.max(), gins.max(), bins.max()])
    except:
        rins = colors
        gins = colors
        bins = colors
        rgb_min = torch.tensor([rins.min(), gins.min(), bins.min()])
        rgb_max = torch.tensor([rins.max(), gins.max(), bins.max()])

    return reduction_mat, rgb_min.to(reduction_mat), rgb_max.to(reduction_mat)


def create_image_grid_with_annotations(images, annotations, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, font_color=(255, 255, 255), thickness=2):
    """
    Arrange a nested list of MxN numpy arrays into a single image grid with column annotations.

    Args:
    images (list of list of np.array): Nested list containing the image arrays.
    annotations (list of str): List of annotations for each column.
    font (int): Font type for the annotations.
    font_scale (float): Font scale for the annotations.
    font_color (tuple): Font color for the annotations.
    thickness (int): Thickness of the font.

    Returns:
    np.array: The resulting image grid with annotations.
    """
    # Check if the input is valid
    if not images or not isinstance(images, list) or not all(isinstance(row, list) for row in images):
        raise ValueError("Input must be a nested list of numpy arrays.")

    # Determine the number of rows and columns
    num_rows = len(images)
    num_cols = len(images[0])

    # Check if all rows have the same number of columns
    if not all(len(row) == num_cols for row in images):
        raise ValueError("All rows must have the same number of columns.")

    # Get the dimensions of each image
    first_image_shape = images[0][0].shape
    img_height, img_width = first_image_shape[:2]

    # Check if all images have the same dimensions
    for row in images:
        for img in row:
            if img.shape[:2] != (img_height, img_width):
                raise ValueError("All images must have the same dimensions.")

    # Determine the number of channels
    num_channels = first_image_shape[2] if len(first_image_shape) == 3 else 1

    # Create an empty array for the resulting grid
    annotation_height = 50  # Height reserved for annotations
    grid_height = num_rows * img_height + annotation_height
    grid_width = num_cols * img_width

    if num_channels == 1:
        grid = np.zeros((grid_height, grid_width), dtype=images[0][0].dtype)
    else:
        grid = np.zeros((grid_height, grid_width, num_channels), dtype=images[0][0].dtype)

    # Add annotations at the top of each column
    for col_idx, annotation in enumerate(annotations):
        text_size = cv2.getTextSize(annotation, font, font_scale, thickness)[0]
        text_x = col_idx * img_width + (img_width - text_size[0]) // 2
        text_y = (annotation_height + text_size[1]) // 2
        cv2.putText(grid, annotation, (text_x, text_y), font, font_scale, font_color, thickness)

    # Populate the grid with the images
    for row_idx, row in enumerate(images):
        for col_idx, img in enumerate(row):
            start_y = annotation_height + row_idx * img_height
            start_x = col_idx * img_width
            grid[start_y:start_y+img_height, start_x:start_x+img_width] = img

    return grid