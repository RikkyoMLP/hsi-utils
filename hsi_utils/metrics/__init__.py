import torch
from hsi_utils.core.metrics_utils import torch_ssim, torch_psnr

def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, size_average: bool = True) -> torch.Tensor:
    return torch_ssim(img1, img2, window_size, size_average)

def psnr(img: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return torch_psnr(img, ref)