import os
import random
import scipy.io as sio
import numpy as np
import torch
from typing import List, Tuple, Union

def LoadTraining(path: str) -> List[np.ndarray]:
    """
    Load training data from directory.
    
    Args:
        path: Directory path.
        
    Returns:
        List[np.ndarray]: List of loaded images (numpy arrays).
    """
    imgs = []
    scene_list = os.listdir(path)
    scene_list.sort()
    print("training sences:", len(scene_list))
    for i in range(len(scene_list)):
        scene_path = os.path.join(path, scene_list[i])
        
        # Parse scene number from filename (assuming format like 'scene001.mat')
        try:
            scene_num = int(scene_list[i].split(".")[0][5:])
        except ValueError:
            continue
            
        if scene_num <= 205:
            if "mat" not in scene_path:
                continue
            img_dict = sio.loadmat(scene_path)
            if "img_expand" in img_dict:
                img = img_dict["img_expand"] / 65536.0
            elif "img" in img_dict:
                img = img_dict["img"] / 65536.0
            elif "data_slice" in img_dict:
                img = img_dict["data_slice"] / 65536.0
            else:
                continue
                
            img = img.astype(np.float32)
            imgs.append(img)
            print("Sence {} is loaded. {}".format(i, scene_list[i]))
    return imgs

def LoadTest(path_test: str) -> torch.Tensor:
    """
    Load test data.
    
    Args:
        path_test: Path to test data directory.
        
    Returns:
        torch.Tensor: Test data tensor of shape [N, 28, 256, 256].
    """
    scene_list = os.listdir(path_test)
    scene_list.sort()
    test_data = np.zeros((len(scene_list), 256, 256, 28))
    for i in range(len(scene_list)):
        scene_path = os.path.join(path_test, scene_list[i])
        img = sio.loadmat(scene_path)["img"]
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
    img = sio.loadmat(path_test_meas)["simulation_test"]
    test_data = img
    test_data = torch.from_numpy(test_data)
    return test_data

def _augment_single(x: torch.Tensor) -> torch.Tensor:
    """
    Apply random rotation and flipping to a single sample.
    
    Args:
        x: Input tensor [C, H, W]
        
    Returns:
        torch.Tensor: Augmented tensor.
    """
    rotTimes = random.randint(0, 3)
    vFlip = random.randint(0, 1)
    hFlip = random.randint(0, 1)
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

def shuffle_crop(train_data: List[np.ndarray], batch_size: int, crop_size: int = 256, argument: bool = True) -> torch.Tensor:
    """
    Randomly crop and augment training data.
    
    Args:
        train_data: List of training images.
        batch_size: Batch size.
        crop_size: Size of the crop.
        argument: Whether to apply data augmentation.
        
    Returns:
        torch.Tensor: Batch of processed training samples.
    """
    if argument:
        gt_batch = []
        # The first half data use the original data.
        half_batch = batch_size // 2
        index = np.random.choice(range(len(train_data)), half_batch)
        processed_data = np.zeros(
            (half_batch, crop_size, crop_size, 28), dtype=np.float32
        )
        for i in range(half_batch):
            img = train_data[index[i]]
            h, w, _ = img.shape
            x_index = np.random.randint(0, h - crop_size)
            y_index = np.random.randint(0, w - crop_size)
            processed_data[i, :, :, :] = img[
                x_index : x_index + crop_size, y_index : y_index + crop_size, :
            ]
        processed_data_torch = (
            torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2))).cuda().float()
        )
        for i in range(processed_data_torch.shape[0]):
            gt_batch.append(_augment_single(processed_data_torch[i]))

        # The other half data use splicing.
        remaining_batch = batch_size - half_batch
        processed_data_2 = np.zeros((4, 128, 128, 28), dtype=np.float32)
        
        # Note: Code assumes crop_size is 256 for the mosaic logic (128*2)
        # If crop_size changes, this logic needs adjustment, but keeping original logic for now.
        
        for i in range(remaining_batch):
            sample_list = np.random.randint(0, len(train_data), 4)
            for j in range(4):
                # Retrieve random sample to get dimensions
                img_sample = train_data[sample_list[j]]
                h, w, _ = img_sample.shape
                
                x_index = np.random.randint(0, h - crop_size // 2)
                y_index = np.random.randint(0, w - crop_size // 2)
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
        index = np.random.choice(range(len(train_data)), batch_size)
        processed_data = np.zeros(
            (batch_size, crop_size, crop_size, 28), dtype=np.float32
        )
        for i in range(batch_size):
            img = train_data[index[i]]
            h, w, _ = img.shape
            x_index = np.random.randint(0, h - crop_size)
            y_index = np.random.randint(0, w - crop_size)
            processed_data[i, :, :, :] = img[
                x_index : x_index + crop_size, y_index : y_index + crop_size, :
            ]
        gt_batch = torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2))).cuda().float()
        return gt_batch
