import os
import torch


def set_gpu_id(gpu_id: int | str | list[int]) -> None:
    """
    Set GPU ID for multiple / single GPU training.

    Args:
        gpu_id: GPU ID or list of GPU IDs. Must follow the format of CUDA_VISIBLE_DEVICES.

    Examples:
        >>> set_gpu_id("0,1,2,3")
        >>> set_gpu_id([0, 1, 2, 3])
        >>> set_gpu_id("0,1,2,3")
    """
    if gpu_id is None:
        return
    if isinstance(gpu_id, list):
        gpu_id = ",".join(gpu_id)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if not torch.cuda.is_available():
        raise Exception("No available GPU, please check your GPU configuration.")
