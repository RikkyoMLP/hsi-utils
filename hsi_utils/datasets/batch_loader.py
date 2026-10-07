"""Cached full-image loading and deterministic augmented batch sampling."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from .dataset_utils import shuffle_crop
from .io import loadmat


class BatchLoader:
    """
    Load ordered .mat files once and sample a batch at an absolute update.
    """

    def __init__(
        self,
        files: Sequence[str | Path],
        *,
        batch_size: int = 4,
        crop_size: int = 256,
        key: str = "img_expand",
        scale: float = 1 / 65536,
        seed: int = 42,
    ):
        self.files = [Path(path) for path in files]
        if not self.files:
            raise ValueError("files must contain at least one .mat file")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if crop_size != 256:
            raise ValueError("BatchLoader currently supports only crop_size=256")
        self.batch_size = batch_size
        self.crop_size = crop_size
        self.seed = seed
        self.data = []
        for index, path in enumerate(self.files, start=1):
            cube = np.asarray(loadmat(path, variable_names=[key])[key], dtype=np.float32)
            if cube.ndim != 3:
                raise ValueError(f"Expected a 3D HSI cube in {path}, got {cube.shape}")
            if cube.shape[-1] == 28:
                pass
            elif cube.shape[0] == 28:
                cube = cube.transpose(1, 2, 0)
            else:
                raise ValueError(f"Expected 28 bands in {path}, got {cube.shape}")
            if min(cube.shape[:2]) <= crop_size:
                raise ValueError(f"Image height and width must exceed {crop_size}: {path}")
            self.data.append(cube * scale)
            if index % 25 == 0 or index == len(self.files):
                print(f"batch_loader_loaded={index}/{len(self.files)}", flush=True)

    def batch_at(self, global_update: int) -> torch.Tensor:
        """Return a CUDA batch using ``seed + global_update`` without RNG side effects."""
        if global_update < 0:
            raise ValueError("global_update must be non-negative")
        return shuffle_crop(
            self.data,
            self.batch_size,
            crop_size=self.crop_size,
            seed=self.seed + global_update,
        )
