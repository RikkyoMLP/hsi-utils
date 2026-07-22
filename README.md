# hsi-utils

Shared utility library for hyperspectral image related tasks.

## Usage

### Install as monorepo dependency (recommended)

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

### Install as an editable package

If you prefer to install this library as an editable package, you can do so by running the following command:

```bash
git clone https://github.com/RikkyoMLP/hsi-utils.git
cd hsi-utils
pip install -e .
```

### Example usage

You can find an example monorepo [here](https://github.com/RikkyoMLP/python-monorepo-example).

## Modules & API Reference

### Config (`hsi_utils.config`)

Utilities for loading and merging configurations.

- `load_merge_config(args: List[str], base_config_path: Union[Path, str, None] = None, inject_template: bool = False) -> OmegaConf`
  Loads base config and merges with CLI arguments.
- `from_args(base_config_path: Union[Path, str, None] = None, inject_template: bool = False) -> OmegaConf`
  Loads base config and merges with CLI arguments.

### Datasets (`hsi_utils.datasets`)

Classes and functions for loading, processing, and augmenting hyperspectral datasets.

- `HSIDataset(path, keys=("img",), scale=1.0, max_scene=None)` -- `torch.utils.data.Dataset`
  Lazy-loading dataset for HSI `.mat` files. Configurable key lookup order, normalization scale, and optional scene number filtering. Each item is a `[C, H, W]` float32 tensor.

- `HSITrainDataset(path, max_scene=205)` -- Convenience constructor
  Returns an `HSIDataset` pre-configured for training: keys=`("img_expand", "img", "data_slice")`, scale=`1/65536`.

- `LoadTraining(path)` / `LoadTest(path)` -- Legacy, prefer `HSIDataset` + `DataLoader`

- `LoadMeasurement(path_test_meas: str) -> torch.Tensor`
  Loads pre-simulated measurement data for testing.

- `shuffle_crop(train_data: List[np.ndarray], batch_size: int, crop_size: int = 256, argument: bool = True) -> torch.Tensor`
  Performs random cropping and data augmentation (rotation, flipping, and mosaic/stitching) on the training data.

#### Example: Loading data with DataLoader

```python
from torch.utils.data import DataLoader
from hsi_utils.datasets import HSIDataset, HSITrainDataset

# Test (default: key="img", no normalization)
test_loader = DataLoader(HSIDataset("/path/to/test"), batch_size=1)

# Training (fallback keys, /65536 normalization, scene filtering)
train_loader = DataLoader(HSITrainDataset("/path/to/train"), batch_size=4, shuffle=True)

for batch in train_loader:  # batch: [B, C, H, W], float32
    batch = batch.cuda()
    ...
```

### Logger (`hsi_utils.logger`)

Utilities for logging and logging exceptions.

- `logger`
  Global logger instance.

- `setup_logger(log_path: str, level: int = logging.INFO) -> None`
  Setup global Root Logger.

- `log_exception(func: Callable[..., Any]) -> Callable[..., Any]`
  Decorator to capture exceptions raised in the decorated function, log the full traceback to the configured logger, and re-raise the exception.

#### Weights & Biases

- `WandbLogger.from_config(settings, *, run_name=None, run_config=None, run_id=None, resume="must", default_project=None, api_key=None, strict=False) -> WandbLogger`
  Initializes an optional W&B run from a YAML file or mapping. Missing/disabled settings and initialization failures produce a safe no-op logger; use `strict=True` when failures should be raised.
- `WandbLogger.log(data, *, step=None, **kwargs) -> None`
  Uploads metrics or media to the active run.
- `WandbLogger.image(data, **kwargs) -> Any`
  Wraps image-like data as `wandb.Image` when tracking is active.
- `WandbLogger.autolog(*, step=None)`
  Activates a logger and step for decorated calls within a `with` block.
- `WandbLogger.finish(**kwargs) -> None`
  Finishes the active run. Repeated calls are safe.
- `wandb_capture(namespace)`
  Decorates a function returning a mapping and automatically uploads its scalar metrics, indexed metrics, and RGB image sequences.

Install the optional backend and create a logger:

```bash
uv add "hsi-utils[wandb]"
```

```yaml
# configs/wandb.yaml
enabled: true
project: my-project
entity: my-team
group: baseline
```

```python
from hsi_utils.logger import WandbLogger, wandb_capture

@wandb_capture("train")
def train_epoch(...):
    ...
    return {"loss": loss, "lr": lr, "epoch_time": elapsed}

@wandb_capture("eval")
def evaluate(...):
    ...
    return {
        "psnr_mean": psnr_mean,
        "psnr_per_scene": psnr_per_scene,
        "recon_rgb": recon_images,
        "gt_rgb": gt_images,
    }

tracker = WandbLogger.from_config(
    "configs/wandb.yaml",
    run_name="experiment-01",
    run_config={"learning_rate": 1e-4},
    run_id=None,  # Set an existing ID to resume it with resume="must".
)

with tracker.autolog(step=1):
    train_epoch(...)
    evaluate(...)

tracker.finish()
```

The decorator follows these conventions:

- Scalar keys are logged below the decorator namespace, such as `loss` becoming `train/loss`.
- Numeric `*_per_scene` sequences expand to `*_S01`, `*_S02`, and so on.
- Numeric `stage_*_mean` sequences expand to `stage_1_*`, `stage_2_*`, and so on.
- Image sequences named `*_rgb` become media panels such as `recon/S01`; `gt_rgb` is uploaded only once per logger instance.

Authentication uses `WANDB_API_KEY` when it is set and otherwise falls back to the normal `wandb.login()` credential lookup. YAML keys other than `enabled`, `login`, and `api_key_env` are forwarded to `wandb.init`, so standard options such as `tags`, `group`, `mode`, and `notes` remain available.

### Loss Functions (`hsi_utils.loss_functions`)

Regularization losses for HSI reconstruction.

- `tv(x: torch.Tensor) -> torch.Tensor`
  Total variation loss. Computes the mean absolute image gradient (horizontal + vertical) via `torchmetrics.functional.image.image_gradients`.

- `nuc_loss_v2(x: torch.Tensor, patch_size: int, eps: float = 1e-8) -> torch.Tensor`
  Log-nuclear-norm loss. Unfolds the input into overlapping patches, computes SVD singular values per patch, and returns `mean(log(S + eps))`. Promotes spectral low-rank structure.

### Masks (`hsi_utils.masks`)

Utilities for generating and managing optical masks, specifically for CASSI systems.

- `generate_masks(mask_path: str, batch_size: int) -> torch.Tensor`
  Generates a batch of 3D fixed masks.
- `generate_shift_masks(mask_path: str, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]`
  Generates shifted 3D masks and their squared sum, used for dispersion modeling.

- `init_mask(mask_path: str, mask_type: str, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]`
  High-level entry point to initialize masks.

### Metrics (`hsi_utils.metrics`)

Utilities for calculating metrics such as PSNR and SSIM.

- `psnr(img: torch.Tensor, ref: torch.Tensor) -> float`
  Calculates the PSNR between two images.
- `ssim(img: torch.Tensor, ref: torch.Tensor) -> float`
  Calculates the SSIM between two images.

### Misc (`hsi_utils.misc`)

Small standalone utilities.

- `is_none(val: Any) -> bool`
  Returns `True` if `val` is `None` or a string that reads `"none"` (case-insensitive, stripped). Useful for CLI/config parsing where `None` may arrive as a string.

### Models (`hsi_utils.models`)

- `get_nb_trainable_parameters(model: nn.Module) -> tuple[int, int]`
  Framework-agnostic function to get the number of trainable and all parameters in the model.

### Physics (`hsi_utils.physics`)

Implements the physical forward models for CASSI (Coded Aperture Snapshot Spectral Imaging).

- `shift(inputs: torch.Tensor, step: int = 2) -> torch.Tensor`
  Simulates the dispersion effect by shifting spectral channels.

- `shift_back(inputs: torch.Tensor, step: int = 2) -> torch.Tensor`
  Reverses the dispersion shift effect.

- `gen_meas_torch(data_batch: torch.Tensor, mask3d_batch: torch.Tensor, Y2H: bool = True, mul_mask: bool = False) -> torch.Tensor`
  The forward model: generates 2D compressed measurements from 3D hyperspectral cubes and masks. Can also return pseudo-HSI if `Y2H=True`.

- `init_meas(gt: torch.Tensor, mask: torch.Tensor, input_setting: str) -> torch.Tensor`
  Wrapper to generate measurements from ground truth.

### Plotting (`hsi_utils.plotting`)

Visualization tools for optimization curves and spectral analysis.

#### General Plotting

- `PlotInput` -- Dataclass
  Describes one data series on a plot. Fields: `data` (array), `identifier` (legend label), `show_max`/`show_min` (mark extrema), `line_color`, `line_style`, `line_width`, `fill_color`.

- `BaselineInput` -- Dataclass
  Describes a horizontal baseline reference line. Fields: `value`, `label`, `line_color`, `line_style`, `line_width`, `fill_color`, `fill_alpha`.

- `draw_plot(left_axis_plots, left_axis_label, x_axis_label, title, ...) -> PIL.Image.Image`
  Generic dual Y-axis plotting function. Accepts two sets of `PlotInput` (left/right axis), optional `BaselineInput` for each axis. Returns a PIL Image and optionally saves to `output_path`.

#### Spectral Density

- `SpectralInput` -- Dataclass
  One HSI cube with metadata for spectral density comparison. Fields: `cube` (`(H,W,C)` array), `label`, `color`, `is_ground_truth`.

- `compute_spectral_density(cube: np.ndarray, roi: tuple[int,int,int,int], clip_max: float | None = None) -> np.ndarray`
  Extracts mean spectral density from an ROI, normalized to max=1. Returns a `(C,)` float64 array.

- `draw_spectral_density(inputs: list[SpectralInput], roi, wavelengths=None, clip_max=None, output_path=None, figsize=(8,6)) -> PIL.Image.Image`
  Draws spectral density curves for multiple reconstructions. Computes Pearson correlation of each prediction against the ground truth and displays it in the legend.

### Rendering (`hsi_utils.rendering`)

HSI-to-image rendering utilities for visualization and figure generation.

#### Pseudo-coloring

- `colorize_channel(gray_image: np.ndarray, wavelength_nm: float, brightness: float = 5.0) -> np.ndarray`
  Applies wavelength-dependent pseudo-coloring to a single spectral band using CIE 1964 color matching functions. Matches the MATLAB `dispCubeAshwin.m` algorithm. Returns `(H, W, 3)` uint8 RGB.

- `colorize_cube(cube: np.ndarray, wavelengths=None, brightness=5.0, channels=None) -> list[tuple[float, np.ndarray]]`
  Colorizes multiple channels from an HSI cube. Returns a list of `(wavelength_nm, colored_image)` tuples.

#### RGB Reconstruction

- `hsi_to_rgb(cube: np.ndarray, wavelengths=None, gamma=2.2) -> np.ndarray`
  Reconstructs an sRGB image from a multi-band HSI cube via CIE 1964 spectral integration + XYZ-to-sRGB conversion. Returns `(H, W, 3)` uint8.

- `load_or_reconstruct_rgb(mat_data: dict, cube_key="truth", rgb_key="rgb", wavelengths=None) -> np.ndarray`
  Loads a pre-stored RGB from `.mat` data if available, otherwise falls back to `hsi_to_rgb`.

#### Magnified Inset

- `InsetPosition` -- Enum (`TOP_LEFT`, `TOP_RIGHT`, `BOTTOM_LEFT`, `BOTTOM_RIGHT`)
  Corner placement for zoom insets.

- `draw_magnified_inset(image: np.ndarray, roi, inset_position=InsetPosition.TOP_LEFT, inset_scale=3.0, border_color=(255,255,0), border_width=2, margin=4) -> np.ndarray`
  Draws a zoomed inset overlay on an RGB image. Highlights the source ROI with a rectangle, scales and pastes it at the specified corner. Returns `(H, W, 3)` uint8.

#### Measurement Rendering

- `render_measurement(measurement: np.ndarray) -> np.ndarray`
  Min-max normalizes a raw CASSI measurement and returns a `(H, W)` uint8 grayscale image.

### Templates (`hsi_utils.templates`)

Utilities for generating templates for the model.

- `get_template(args: Union[DictConfig, Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]`
  Determines input_setting and input_mask based on args.template.
- `get_template_list() -> List[Dict[str, Optional[str]]]`
  Gets list of templates.

### Training (`hsi_utils.training`)

Utilities for training the model.

- `set_gpu_id(gpu_id: int | str | list[int]) -> None`
  Sets GPU ID for multiple / single GPU training.
- `set_seed(seed: int) -> None`
  Sets seed for reproducibility.
- `setup_cudnn(benchmark: bool = False, deterministic: bool = False, enabled: bool = True) -> None`
  Explicitly setup CuDNN environment.
