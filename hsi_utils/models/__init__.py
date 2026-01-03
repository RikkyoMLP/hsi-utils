from torch import nn

def get_nb_trainable_parameters(model: nn.Module) -> tuple[int, int]:
    """
    Framework-agnostic function to get the number of trainable and all parameters in the model.

    Args:
        model (nn.Module): The model to get the number of trainable and all parameters from

    Returns:
        tuple[int, int]: The number of trainable and all parameters in the model
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    return trainable_params, all_params