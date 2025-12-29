import random
import numpy as np
import torch
from hsi_utils.logger import logger, setup_logger


def set_seed(seed: int) -> None:
    """
    Set seed for reproducibility.

    Args:
        `seed`: Seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    try:
        if not logger.handlers:
            setup_logger(logger.handlers[0].baseFilename)
        logger.info(f"Seed has been set to {seed}")
    except Exception as e:
        # likely logger is not initialized yet
        print(f"Seed has been set to {seed}, not able to log to file")
