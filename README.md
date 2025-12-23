# hsi-utils

Shared utility library for hyperspectral image related tasks.

## Usage

This library can be used as a monorepo dependency. To do this, you will need to install [uv](https://docs.astral.sh/uv/) as your python package manager.

```bash
# Install uv
pip install uv
# Or via curl
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then, you can install this library as a monorepo dependency.

```bash
uv sync
```

### Example usage


An example library is provided in \[double-blind redacted\].

## Modules & API Reference

### 1. Masks (`hsi_utils.masks`)

Utilities for generating and managing optical masks, specifically for CASSI systems.

- `generate_masks(mask_path: str, batch_size: int) -> torch.Tensor`
  Generates a batch of 3D fixed masks.
  
- `generate_shift_masks(mask_path: str, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]`
  Generates shifted 3D masks and their squared sum, used for dispersion modeling.

- `init_mask(mask_path: str, mask_type: str, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]`
  High-level entry point to initialize masks.

### 2. Datasets (`hsi_utils.datasets`)

Functions for loading, processing, and augmenting hyperspectral datasets.

- `LoadTraining(path: str) -> List[np.ndarray]`
  Loads training scenes from a specified directory.

- `LoadTest(path_test: str) -> torch.Tensor`
  Loads test data and formats it into a tensor of shape `[N, 28, 256, 256]`.

- `LoadMeasurement(path_test_meas: str) -> torch.Tensor`
  Loads pre-simulated measurement data for testing.

- `shuffle_crop(train_data: List[np.ndarray], batch_size: int, crop_size: int = 256, argument: bool = True) -> torch.Tensor`
  Performs random cropping and data augmentation (rotation, flipping, and mosaic/stitching) on the training data.

### 3. Physics (`hsi_utils.physics`)

Implements the physical forward models for CASSI (Coded Aperture Snapshot Spectral Imaging).

- `shift(inputs: torch.Tensor, step: int = 2) -> torch.Tensor`
  Simulates the dispersion effect by shifting spectral channels.

- `shift_back(inputs: torch.Tensor, step: int = 2) -> torch.Tensor`
  Reverses the dispersion shift effect.

- `gen_meas_torch(data_batch: torch.Tensor, mask3d_batch: torch.Tensor, Y2H: bool = True, mul_mask: bool = False) -> torch.Tensor`
  The forward model: generates 2D compressed measurements from 3D hyperspectral cubes and masks. Can also return pseudo-HSI if `Y2H=True`.

- `init_meas(gt: torch.Tensor, mask: torch.Tensor, input_setting: str) -> torch.Tensor`
  Wrapper to generate measurements from ground truth.
