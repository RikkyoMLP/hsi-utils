import torch
import torch.nn as nn
from torchmetrics.functional.image import image_gradients as ig_


def tv(x: torch.Tensor):
    dx, dy = ig_(x)
    return dx.abs().mean() + dy.abs().mean()

def nuc_loss_v2(x: torch.Tensor, patch_size: int, eps: float = 1e-8):
    B, C, H, W = x.shape
    stride = int(patch_size / 2 - 1)
    unfold = nn.Unfold(kernel_size=patch_size, stride=stride)
    x = unfold(x)
    x = x.permute(0, 2, 1)
    x = x.reshape(B, -1, C, patch_size*patch_size)
    S = torch.linalg.svdvals(x)
    nuc_norm_log = torch.mean(torch.log(S + eps))
    
    return nuc_norm_log