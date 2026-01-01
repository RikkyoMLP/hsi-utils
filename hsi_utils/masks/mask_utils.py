import os
import scipy.io as sio
import numpy as np
import torch
from hsi_utils.physics import shift

def generate_masks(mask_path: str, batch_size: int) -> torch.Tensor:
    """
    Generate 3D masks for CASSI system.
    
    Args:
        mask_path: Path to the directory containing mask files.
        batch_size: Number of masks to generate (batch size).
        
    Returns:
        torch.Tensor: Batch of 3D masks with shape [batch_size, nC, H, W].
    """
    # Handle path with or without trailing slash
    # file_path = os.path.join(mask_path, "fixmask_3d.mat")
    file_path = os.path.join(mask_path, "mask.mat")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Mask file not found at {file_path}")
    mask = sio.loadmat(file_path)
    mask = mask["mask"]
    # mask_3d = mask["mask_3d"]
    mask3d = np.tile(mask[:, :, np.newaxis], (1, 1, 28))
    mask3d = np.transpose(mask3d, [2, 0, 1])
    mask3d = torch.from_numpy(mask3d)
    [nC, H, W] = mask3d.shape
    mask3d_batch = mask3d.expand([batch_size, nC, H, W]).cuda().float()
    return mask3d_batch

# def generate_masks(mask_path, batch_size):
#     mask = sio.loadmat("/root/gpufree-data/CASSI-SSL/dataset/mask.mat")
#     mask = mask["mask"]
#     mask3d = np.tile(mask[:, :, np.newaxis], (1, 1, 28))
#     mask3d = np.transpose(mask3d, [2, 0, 1])
#     mask3d = torch.from_numpy(mask3d)
#     [nC, H, W] = mask3d.shape
#     mask3d_batch = mask3d.expand([batch_size, nC, H, W]).cuda().float()
#     return mask3d_batch

def generate_shift_masks(mask_path: str, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate shifted 3D masks and their squared sum.
    
    Args:
        mask_path: Path to the directory containing mask files.
        batch_size: Number of masks to generate.
        
    Returns:
        tuple[torch.Tensor, torch.Tensor]: 
            - Phi_batch: Expanded shifted masks [batch_size, nC, H, W].
            - Phi_s_batch: Sum of squared shifted masks [batch_size, H, W] (or similar, depending on reduction).
    """
    file_path = os.path.join(mask_path, "mask_3d_shift.mat")
    mask = sio.loadmat(file_path)
    mask_3d_shift = mask["mask_3d_shift"]
    mask_3d_shift = np.transpose(mask_3d_shift, [2, 0, 1])
    mask_3d_shift = torch.from_numpy(mask_3d_shift)
    nC, H, W = mask_3d_shift.shape
    
    Phi_batch = mask_3d_shift.expand([batch_size, nC, H, W]).cuda().float()
    Phi_s_batch = torch.sum(Phi_batch**2, 1)
    Phi_s_batch[Phi_s_batch == 0] = 1
    
    return Phi_batch, Phi_s_batch

def init_mask(mask_path: str, mask_type: str, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Initialize masks.

    Args:
        mask_path: Path to mask directory.
        mask_type: Type of mask.
        batch_size: Batch size.
        
    Returns:
        tuple[torch.Tensor, torch.Tensor]: The mask batch and input mask.
    """
    mask3d_batch = generate_masks(mask_path, batch_size)
    
    if mask_type == "Phi":
        shift_mask3d_batch = shift(mask3d_batch)
        input_mask = shift_mask3d_batch
    elif mask_type == "Phi_PhiPhiT":
        Phi_batch, Phi_s_batch = generate_shift_masks(mask_path, batch_size)
        input_mask = (Phi_batch, Phi_s_batch)
    elif mask_type == "Mask":
        input_mask = mask3d_batch
    elif mask_type is None:
        input_mask = None
    else:
        # Default fallback
        input_mask = mask3d_batch

    print(f"Mask shape: {mask3d_batch.shape}")
    return mask3d_batch, input_mask

