import torch.backends.cudnn as cudnn


def setup_cudnn(
    benchmark: bool = False,
    deterministic: bool = False,
    enabled: bool = True,
) -> None:
    """
    Explicitly setup CuDNN environment.

    Args:
        `benchmark`: Whether to enable CuDNN benchmark mode.
        `deterministic`: Whether to enable CuDNN deterministic mode.
        `enabled`: Whether to enable CuDNN.
    """
    cudnn.benchmark = benchmark
    cudnn.deterministic = deterministic
    cudnn.enabled = enabled
