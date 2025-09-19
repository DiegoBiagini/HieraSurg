import torch
import os
import math
import torch.nn.functional as F

# https://github.com/universome/fvd-comparison


def load_i3d_pretrained(device=torch.device('cpu')):
    i3D_WEIGHTS_URL = "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt"
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'i3d_torchscript.pt')
    print(filepath)
    if not os.path.exists(filepath):
        print(f"preparing for download {i3D_WEIGHTS_URL}, you can download it by yourself.")
        os.system(f"wget {i3D_WEIGHTS_URL} -O {filepath}")
    i3d = torch.jit.load(filepath).eval().to(device)
    #i3d = torch.nn.DataParallel(i3d)
    return i3d
    

def get_feats(videos, detector, device, bs=10):
    # videos : torch.tensor BCTHW [0, 1]
    detector_kwargs = dict(rescale=False, resize=False, return_features=True) # Return raw features before the softmax layer.
    feats = np.empty((0, 400))
    with torch.no_grad():
        for i in range((len(videos)-1)//bs + 1):
            feats = np.vstack([feats, detector(torch.stack([preprocess_single(video) for video in videos[i*bs:(i+1)*bs]]).to(device), **detector_kwargs).detach().cpu().numpy()])
    return feats


def get_fvd_feats(videos, i3d, device, bs=10):
    # videos in [0, 1] as torch tensor BCTHW
    # videos = [preprocess_single(video) for video in videos]
    embeddings = get_feats(videos, i3d, device, bs)
    return embeddings


def preprocess_single(video, resolution=224, sequence_length=None):
    # video: CTHW, [0, 1]
    c, t, h, w = video.shape

    # temporal crop
    if sequence_length is not None:
        video = video[:, :sequence_length]

    # scale shorter side to resolution
    scale = resolution / min(h, w)
    if h < w:
        target_size = (resolution, math.ceil(w * scale))
    else:
        target_size = (math.ceil(h * scale), resolution)
    video = F.interpolate(video, size=target_size, mode='bilinear', align_corners=False)

    # center crop
    c, t, h, w = video.shape
    w_start = (w - resolution) // 2
    h_start = (h - resolution) // 2
    video = video[:, :, h_start:h_start + resolution, w_start:w_start + resolution]

    # [0, 1] -> [-1, 1]
    video = (video - 0.5) * 2

    return video.contiguous()


"""
Copy-pasted from https://github.com/cvpr2022-stylegan-v/stylegan-v/blob/main/src/metrics/frechet_video_distance.py
"""
from typing import Tuple
from scipy.linalg import sqrtm
import numpy as np


def compute_stats(feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = feats.mean(axis=0) # [d]
    sigma = np.cov(feats, rowvar=False) # [d, d]
    return mu, sigma


def frechet_distance(feats_fake: np.ndarray, feats_real: np.ndarray) -> float:
    mu_gen, sigma_gen = compute_stats(feats_fake)
    mu_real, sigma_real = compute_stats(feats_real)
    m = np.square(mu_gen - mu_real).sum()
    if feats_fake.shape[0]>1:
        s, _ = sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
        fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
    else:
        fid = np.real(m)
    return float(fid)

from tools.fvd import load_i3d_pretrained, get_fvd_feats, frechet_distance
from torchmetrics import Metric
import torch

class FrechetVideoDistance(Metric):
    def __init__(self):
        super().__init__(dist_sync_on_step=False)
        self.add_state("real_feats", default=[], dist_reduce_fx=None)
        self.add_state("fake_feats", default=[], dist_reduce_fx=None)
        self.i3d = load_i3d_pretrained()

    def update(self, samples: torch.Tensor, real: bool = True):
        # Preprocess videos
        new_samples = [] 
        for i in range(samples.shape[0]):
            new_samples.append(preprocess_single(samples[i], sequence_length=16))

        samples = torch.stack(new_samples).float()
        # Check for ordering
        # Extract features using I3D model
        feats = get_fvd_feats(samples, self.i3d, samples.device)
        
        # Store features based on whether they are real or fake
        if real:
            self.real_feats.append(feats)
        else:
            self.fake_feats.append(feats)

    def compute(self):
        fake_feats = np.concatenate(self.fake_feats, axis=0)
        real_feats = np.concatenate(self.real_feats, axis=0)
        return frechet_distance(fake_feats, real_feats)

class KernelVideoDistance(Metric):
    def __init__(self, kernel_function="rbf", num_subsets=100, subset_size=1000, degree=3, gamma=None):
        super().__init__(dist_sync_on_step=False)
        self.add_state("real_feats", default=[], dist_reduce_fx=None)
        self.add_state("fake_feats", default=[], dist_reduce_fx=None)
        self.i3d = load_i3d_pretrained()
        self.kernel_function = kernel_function
        self.num_subsets = num_subsets
        self.subset_size = subset_size
        self.degree = degree
        self.gamma = gamma

    def update(self, samples: torch.Tensor, real: bool = True):
        # Preprocess videos
        new_samples = [] 
        for i in range(samples.shape[0]):
            new_samples.append(preprocess_single(samples[i], sequence_length=16))


        samples = torch.stack(new_samples).float()
        # Check for ordering

        # Extract features using I3D model
        feats = get_fvd_feats(samples, self.i3d, samples.device)
        
        # Store features based on whether they are real or fake
        if real:
            self.real_feats.append(feats)
        else:
            self.fake_feats.append(feats)

    def compute(self):
        fake_feats = np.concatenate(self.fake_feats, axis=0)
        real_feats = np.concatenate(self.real_feats, axis=0)
        
        n_samples = min(len(fake_feats), len(real_feats))
        subset_size = min(self.subset_size, n_samples)
        
        results = []
        for _ in range(self.num_subsets):
            # Randomly sample subset_size samples from each set
            fake_idx = np.random.choice(len(fake_feats), subset_size, replace=False)
            real_idx = np.random.choice(len(real_feats), subset_size, replace=False)
            
            fake_subset = torch.from_numpy(fake_feats[fake_idx])
            real_subset = torch.from_numpy(real_feats[real_idx])
            
            mmd = poly_mmd(real_subset, fake_subset, degree=self.degree, gamma=self.gamma)
                
            results.append(mmd.item())
            
        return np.mean(results)

from torch import Tensor
from typing import Optional
def maximum_mean_discrepancy(k_xx: Tensor, k_xy: Tensor, k_yy: Tensor) -> Tensor:
    """Adapted from `KID Score`_."""
    m = k_xx.shape[0]

    diag_x = torch.diag(k_xx)
    diag_y = torch.diag(k_yy)

    kt_xx_sums = k_xx.sum(dim=-1) - diag_x
    kt_yy_sums = k_yy.sum(dim=-1) - diag_y
    k_xy_sums = k_xy.sum(dim=0)

    kt_xx_sum = kt_xx_sums.sum()
    kt_yy_sum = kt_yy_sums.sum()
    k_xy_sum = k_xy_sums.sum()

    value = (kt_xx_sum + kt_yy_sum) / (m * (m - 1))
    value -= 2 * k_xy_sum / (m**2)
    return value


def poly_kernel(f1: Tensor, f2: Tensor, degree: int = 3, gamma: Optional[float] = None, coef: float = 1.0) -> Tensor:
    """Adapted from `KID Score`_."""
    if gamma is None:
        gamma = 1.0 / f1.shape[1]
    return (f1 @ f2.T * gamma + coef) ** degree


def poly_mmd(
    f_real: Tensor, f_fake: Tensor, degree: int = 3, gamma: Optional[float] = None, coef: float = 1.0
) -> Tensor:
    """Adapted from `KID Score`_."""
    k_11 = poly_kernel(f_real, f_real, degree, gamma, coef)
    k_22 = poly_kernel(f_fake, f_fake, degree, gamma, coef)
    k_12 = poly_kernel(f_real, f_fake, degree, gamma, coef)
    return maximum_mean_discrepancy(k_11, k_12, k_22)