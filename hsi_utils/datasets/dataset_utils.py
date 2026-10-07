import os
import numpy as np
import torch
import tqdm
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import Dataset
from typing import List, Tuple, Union, Dict, Any, TypedDict
from pathlib import Path


class HSIDataset(Dataset):
    """
    Lazy-loading Dataset for HSI .mat files.

    Only stores file paths at init; each __getitem__ reads a single .mat file.
    Configurable key lookup order, normalization scale, and filename filtering.
    """

    def __init__(
        self,
        path: str,
        keys: Tuple[str, ...] = ("img",),
        scale: float = 1.0,
        max_scene: int | None = None,
    ):
        self.keys = keys
        self.scale = scale
        self.paths = []
        for name in sorted(os.listdir(path)):
            if not name.endswith(".mat"):
                continue
            if max_scene is not None:
                try:
                    scene_num = int(name.split(".")[0][5:])
                except (ValueError, IndexError):
                    continue
                if scene_num > max_scene:
                    continue
            self.paths.append(os.path.join(path, name))

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:  # type: ignore  # fix later
        from hsi_utils.datasets.io import loadmat

        mat = loadmat(self.paths[idx])
        for key in self.keys:
            if key in mat:
                img = mat[key].astype(np.float32) * self.scale
                # [H, W, C] -> [C, H, W]
                return torch.from_numpy(img.transpose(2, 0, 1))
        raise KeyError(f"No valid key in {self.paths[idx]}, tried {self.keys}")


def HSITrainDataset(path: str, max_scene: int = 205) -> HSIDataset:
    """Convenience constructor for training datasets."""
    return HSIDataset(
        path,
        keys=("img_expand", "img", "data_slice"),
        scale=1.0 / 65536.0,
        max_scene=max_scene,
    )


def _describe_value(val):
    """Recursively describe the structure of a value from a .mat file."""
    if isinstance(val, np.ndarray):
        # Structured or object arrays: recurse into dtype fields
        if val.dtype.names:
            return {
                "type": "ndarray",
                "dtype": str(val.dtype),
                "shape": val.shape,
                "fields": {name: _describe_value(val[name]) for name in val.dtype.names},
            }
        # Squeeze scalar object arrays (MATLAB cells / nested structs)
        if val.dtype == object:
            flat = val.flat
            if val.size == 1:
                return _describe_value(flat[0])
            return {
                "type": "ndarray[object]",
                "shape": val.shape,
                "elements": [_describe_value(flat[i]) for i in range(min(val.size, 8))],
            }
        return {"type": "ndarray", "dtype": str(val.dtype), "shape": val.shape}
    if isinstance(val, dict):
        return {k: _describe_value(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_describe_value(v) for v in val[:8]]
    if isinstance(val, (bytes, str)):
        return type(val).__name__
    # Scalar numbers, etc.
    return type(val).__name__


class MatMetadata(TypedDict):
    keys: List[str]
    structure: Dict[str, Any]


class MatFileInfo(TypedDict):
    data: Dict[str, Any]
    metadata: MatMetadata


def load_raw_data(path: Union[str, Path]) -> MatFileInfo:
    """
    Read one .mat file and return structured data.

    Args:
        path: Path to the .mat file.

    Returns:
        {
            "data": Any,
            "metadata": {
                "keys": List[str],
                "structure": nested description of each key's value,
            },
        }
    """
    from hsi_utils.datasets.io import loadmat

    mat = loadmat(str(path))
    return {
        "data": mat,
        "metadata": {
            "keys": list(mat.keys()),
            "structure": {
                key: _describe_value(mat[key]) for key in mat.keys()
            }
        },
    }


def _load_single_training_scene(scene_path):
    """Load a single .mat training scene. Returns np.float32 array or None."""
    from hsi_utils.datasets.io import loadmat

    img_dict = loadmat(scene_path)
    if "img_expand" in img_dict:
        img = img_dict["img_expand"] / 65536.0
    elif "img" in img_dict:
        img = img_dict["img"] / 65536.0
    elif "data_slice" in img_dict:
        img = img_dict["data_slice"] / 65536.0
    else:
        return None
    return img.astype(np.float32)


def _load_single_test_scene(scene_path):
    """Load a single .mat test scene. Returns np.ndarray."""
    from hsi_utils.datasets.io import loadmat

    return loadmat(scene_path)["img"]


def LoadTrainingLegacy(path: str, num_workers: int = 8) -> List[np.ndarray]:
    """
    Load training data from directory with multithreading.

    Args:
        path: Directory path.
        num_workers: Number of threads for parallel loading.

    Returns:
        List[np.ndarray]: List of loaded images (numpy arrays).
    """
    scene_list = sorted(os.listdir(path))
    print("training sences:", len(scene_list))

    valid_paths = []
    for name in scene_list:
        if "mat" not in name:
            continue
        try:
            scene_num = int(name.split(".")[0][5:])
        except (ValueError, IndexError):
            continue
        if scene_num <= 205:
            valid_paths.append(os.path.join(path, name))

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        imgs = list(
            tqdm.tqdm(
                executor.map(_load_single_training_scene, valid_paths),
                total=len(valid_paths),
            )
        )

    imgs = [img for img in imgs if img is not None]
    print("{} training scenes loaded.".format(len(imgs)))
    return imgs


def LoadTestLegacy(path_test: str, num_workers: int = 8) -> torch.Tensor:
    """
    Load test data with multithreading.

    Args:
        path_test: Path to test data directory.
        num_workers: Number of threads for parallel loading.

    Returns:
        torch.Tensor: Test data tensor of shape [N, 28, 256, 256].
    """
    scene_list = sorted(os.listdir(path_test))
    scene_paths = [os.path.join(path_test, name) for name in scene_list]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(
            tqdm.tqdm(
                executor.map(_load_single_test_scene, scene_paths),
                total=len(scene_paths),
            )
        )

    test_data = np.zeros((len(results), 256, 256, 28))
    for i, img in enumerate(results):
        test_data[i, :, :, :] = img
    test_data = torch.from_numpy(np.transpose(test_data, (0, 3, 1, 2)))
    return test_data


def LoadMeasurement(path_test_meas: str) -> torch.Tensor:
    """
    Load simulation test measurement.

    Args:
        path_test_meas: Path to measurement file.

    Returns:
        torch.Tensor: Measurement tensor.
    """
    from hsi_utils.datasets.io import loadmat

    img = loadmat(path_test_meas)["simulation_test"]
    test_data = img
    test_data = torch.from_numpy(test_data)
    return test_data


def _augment_single(x: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """
    Apply random rotation and flipping to a single sample.

    Args:
        x: Input tensor [C, H, W]

    Returns:
        torch.Tensor: Augmented tensor.
    """
    rotTimes = rng.integers(0, 4)
    vFlip = rng.integers(0, 2)
    hFlip = rng.integers(0, 2)
    # Random rotation
    for _ in range(rotTimes):
        x = torch.rot90(x, dims=(1, 2))
    # Random vertical Flip
    for _ in range(vFlip):
        x = torch.flip(x, dims=(2,))
    # Random horizontal Flip
    for _ in range(hFlip):
        x = torch.flip(x, dims=(1,))
    return x


def _augment_mosaic(generate_gt: torch.Tensor) -> torch.Tensor:
    """
    Stitch 4 cropped patches (mosaic) into one large patch.

    Args:
        generate_gt: Tensor of 4 patches [4, C, H/2, W/2]

    Returns:
        torch.Tensor: Mosaicked tensor [C, H, W]
    """
    c = generate_gt.shape[1]
    # Assuming standard size 256x256 based on usage
    h, w = 256, 256
    divid_point_h = 128
    divid_point_w = 128
    output_img = torch.zeros(c, h, w).cuda()
    output_img[:, :divid_point_h, :divid_point_w] = generate_gt[0]
    output_img[:, :divid_point_h, divid_point_w:] = generate_gt[1]
    output_img[:, divid_point_h:, :divid_point_w] = generate_gt[2]
    output_img[:, divid_point_h:, divid_point_w:] = generate_gt[3]
    return output_img


def shuffle_crop(
    train_data: List[np.ndarray],
    batch_size: int,
    crop_size: int = 256,
    argument: bool = True,
    *,
    seed: int = 42,
) -> torch.Tensor:
    """
    Randomly crop and augment training data.

    Args:
        train_data: List of training images.
        batch_size: Batch size.
        crop_size: Size of the crop.
        argument: Whether to apply data augmentation.
        seed: NumPy random seed. Use a different seed per training step;
            the same seed and inputs replay the same batch.

    Returns:
        torch.Tensor: Batch of processed training samples.
    """
    rng = np.random.default_rng(seed)
    if argument:
        gt_batch = []
        # The first half data use the original data.
        half_batch = batch_size // 2
        index = rng.choice(range(len(train_data)), half_batch)
        processed_data = np.zeros(
            (half_batch, crop_size, crop_size, 28), dtype=np.float32
        )
        for i in range(half_batch):
            img = train_data[index[i]]
            h, w, _ = img.shape
            x_index = rng.integers(0, h - crop_size)
            y_index = rng.integers(0, w - crop_size)
            processed_data[i, :, :, :] = img[
                x_index : x_index + crop_size, y_index : y_index + crop_size, :
            ]
        processed_data_torch = (
            torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2))).cuda().float()
        )
        for i in range(processed_data_torch.shape[0]):
            gt_batch.append(_augment_single(processed_data_torch[i], rng))

        # The other half data use splicing.
        remaining_batch = batch_size - half_batch
        processed_data_2 = np.zeros((4, 128, 128, 28), dtype=np.float32)

        # Note: Code assumes crop_size is 256 for the mosaic logic (128*2)
        # If crop_size changes, this logic needs adjustment, but keeping original logic for now.

        for _ in range(remaining_batch):
            sample_list = rng.integers(0, len(train_data), 4)
            for j in range(4):
                # Retrieve random sample to get dimensions
                img_sample = train_data[sample_list[j]]
                h, w, _ = img_sample.shape

                x_index = rng.integers(0, h - crop_size // 2)
                y_index = rng.integers(0, w - crop_size // 2)
                processed_data_2[j] = img_sample[
                    x_index : x_index + crop_size // 2,
                    y_index : y_index + crop_size // 2,
                    :,
                ]
            gt_batch_2 = torch.from_numpy(
                np.transpose(processed_data_2, (0, 3, 1, 2))
            ).cuda()  # [4,28,128,128]
            gt_batch.append(_augment_mosaic(gt_batch_2))
        gt_batch = torch.stack(gt_batch, dim=0)
        return gt_batch
    else:
        index = rng.choice(range(len(train_data)), batch_size)
        processed_data = np.zeros(
            (batch_size, crop_size, crop_size, 28), dtype=np.float32
        )
        for i in range(batch_size):
            img = train_data[index[i]]
            h, w, _ = img.shape
            x_index = rng.integers(0, h - crop_size)
            y_index = rng.integers(0, w - crop_size)
            processed_data[i, :, :, :] = img[
                x_index : x_index + crop_size, y_index : y_index + crop_size, :
            ]
        gt_batch = (
            torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2))).cuda().float()
        )
        return gt_batch
