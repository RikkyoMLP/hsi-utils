from pathlib import Path

from hsi_utils.datasets.io import loadmat
import numpy as np
import torch
from hsi_utils.physics import shift
from hsi_utils.logger import logger


def _log_mask_path_to_console(file_path: str | Path) -> None:
    try:
        logger.info(f"Mask {file_path} loaded")
    except Exception:
        print(f"Mask {file_path} loaded")


def _mask_file(mask_path: str | Path, filename: str) -> Path:
    path = Path(mask_path)
    return path if path.is_file() else path / filename


def _target_device(device: torch.device | str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_masks(
    mask_path: str | Path,
    batch_size: int,
    *,
    channels: int = 28,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Generate 3D masks for CASSI system.

    Args:
        mask_path: Path to the directory containing mask files.
        batch_size: Number of masks to generate (batch size).
        channels: Number of spectral channels.
        device: Output device. Defaults to CUDA when available, otherwise CPU.
        dtype: Output floating dtype.

    Returns:
        torch.Tensor: Batch of 3D masks with shape [batch_size, nC, H, W].
    """
    if batch_size <= 0 or channels <= 0:
        raise ValueError("batch_size and channels must be positive")
    file_path = _mask_file(mask_path, "mask.mat")
    _log_mask_path_to_console(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Mask file not found at {file_path}")
    document = loadmat(file_path, variable_names=["mask"])
    if "mask" not in document:
        raise KeyError(f"MAT key 'mask' missing from {file_path}")
    mask = np.asarray(document["mask"])
    if mask.ndim != 2:
        raise ValueError(f"base mask must be 2D, got shape {mask.shape}")
    mask3d = np.repeat(mask[None], channels, axis=0)
    tensor = torch.from_numpy(np.ascontiguousarray(mask3d)).to(
        device=_target_device(device),
        dtype=dtype,
    )
    return tensor.unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()


# def generate_masks(mask_path, batch_size):
#     mask = sio.loadmat("/root/gpufree-data/CASSI-SSL/dataset/mask.mat")
#     mask = mask["mask"]
#     mask3d = np.tile(mask[:, :, np.newaxis], (1, 1, 28))
#     mask3d = np.transpose(mask3d, [2, 0, 1])
#     mask3d = torch.from_numpy(mask3d)
#     [nC, H, W] = mask3d.shape
#     mask3d_batch = mask3d.expand([batch_size, nC, H, W]).cuda().float()
#     return mask3d_batch


def generate_shift_masks(
    mask_path: str | Path,
    batch_size: int,
    nC: int = 28,
    step: int = 2,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate shifted 3D masks and their squared sum from base 2D mask.

    Spectral shift: zero-pad width from W to W+(nC-1)*step, then roll
    each channel t by step*t pixels.

    Args:
        mask_path: Path to the directory containing mask.mat.
        batch_size: Number of masks to generate.
        nC: Number of spectral channels.
        step: Shift step size.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - Phi_batch: Shifted masks [batch_size, nC, H, W_shifted].
            - Phi_s_batch: Sum of squared shifted masks [batch_size, H, W_shifted].
    """
    if batch_size <= 0 or nC <= 0 or step < 0:
        raise ValueError("batch_size/nC must be positive and step non-negative")
    file_path = _mask_file(mask_path, "mask.mat")
    _log_mask_path_to_console(file_path)
    document = loadmat(file_path, variable_names=["mask"])
    if "mask" not in document:
        raise KeyError(f"MAT key 'mask' missing from {file_path}")
    mask_2d = np.asarray(document["mask"])
    if mask_2d.ndim != 2:
        raise ValueError(f"base mask must be 2D, got shape {mask_2d.shape}")
    H, W = mask_2d.shape

    # Tile to 3D and zero-pad width for spectral shift
    W_shifted = W + (nC - 1) * step
    mask_3d_shift = np.zeros((H, W_shifted, nC), dtype=np.float32)
    for t in range(nC):
        mask_3d_shift[:, 0:W, t] = mask_2d
        mask_3d_shift[:, :, t] = np.roll(mask_3d_shift[:, :, t], step * t, axis=1)

    # [H, W_shifted, nC] -> [nC, H, W_shifted]
    mask_3d_shift = torch.from_numpy(
        np.ascontiguousarray(np.transpose(mask_3d_shift, [2, 0, 1]))
    ).to(device=_target_device(device), dtype=dtype)

    Phi_batch = mask_3d_shift.unsqueeze(0).expand(
        batch_size, -1, -1, -1
    ).contiguous()
    Phi_s_batch = torch.sum(Phi_batch**2, 1)
    Phi_s_batch = torch.where(
        Phi_s_batch == 0,
        torch.ones_like(Phi_s_batch),
        Phi_s_batch,
    )

    return Phi_batch, Phi_s_batch


def load_shifted_masks(
    mask_path: str | Path,
    batch_size: int,
    *,
    channels: int = 28,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    key: str = "mask_3d_shift",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a calibrated shifted-mask asset without regenerating it.

    This is intentionally separate from :func:`generate_shift_masks`.  Some
    SCI datasets publish a shifted mask whose boundary values differ from a
    mask synthesized from ``mask.mat``; callers that need dataset identity
    should use this loader.
    """

    if batch_size <= 0 or channels <= 0:
        raise ValueError("batch_size and channels must be positive")
    file_path = _mask_file(mask_path, "mask_3d_shift.mat")
    _log_mask_path_to_console(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Shifted mask file not found at {file_path}")
    document = loadmat(file_path, variable_names=[key])
    if key not in document:
        raise KeyError(f"MAT key {key!r} missing from {file_path}")
    array = np.asarray(document[key])
    if array.ndim != 3:
        raise ValueError(f"shifted mask must be 3D, got shape {array.shape}")
    if array.shape[-1] == channels:
        chw = np.transpose(array, (2, 0, 1))
    elif array.shape[0] == channels:
        chw = array
    else:
        raise ValueError(
            f"shifted mask shape {array.shape} has no {channels}-channel axis"
        )
    tensor = torch.from_numpy(np.ascontiguousarray(chw)).to(
        device=_target_device(device),
        dtype=dtype,
    )
    phi = tensor.unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()
    phi_sum = (phi**2).sum(dim=1)
    phi_sum = torch.where(phi_sum == 0, torch.ones_like(phi_sum), phi_sum)
    return phi, phi_sum


def init_mask(
    mask_path: str | Path,
    mask_type: str | None,
    batch_size: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None]:
    """
    Initialize masks.

    Args:
        mask_path: Path to mask directory.
        mask_type: Type of mask.
        batch_size: Batch size.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The mask batch and input mask.
    """
    mask3d_batch = generate_masks(
        mask_path,
        batch_size,
        device=device,
        dtype=dtype,
    )

    if mask_type == "Phi":
        shift_mask3d_batch = shift(mask3d_batch)
        input_mask = shift_mask3d_batch
    elif mask_type == "Phi_PhiPhiT":
        Phi_batch, Phi_s_batch = generate_shift_masks(
            mask_path,
            batch_size,
            device=device,
            dtype=dtype,
        )
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
