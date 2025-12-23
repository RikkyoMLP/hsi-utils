import torch

def shift(inputs: torch.Tensor, step: int = 2) -> torch.Tensor:
    """
    Simulate the dispersion (shift) effect in CASSI.
    
    Args:
        inputs: Input tensor of shape [bs, nC, row, col].
        step: Shift step size.
        
    Returns:
        torch.Tensor: Shifted tensor.
    """
    bs, nC, row, col = inputs.shape
    output = torch.zeros(bs, nC, row, col + (nC - 1) * step).cuda().float()
    for i in range(nC):
        output[:, i, :, step * i : step * i + col] = inputs[:, i, :, :]
    return output

def shift_back(inputs: torch.Tensor, step: int = 2) -> torch.Tensor:
    """
    Reverse the dispersion (shift) effect.
    
    Args:
        inputs: Input tensor of shape [bs, row, col_shifted].
        step: Shift step size.
        
    Returns:
        torch.Tensor: Back-shifted tensor of shape [bs, nC, row, col].
    """
    bs, row, col = inputs.shape
    nC = 28 # Fixed nC as per original code
    output = torch.zeros(bs, nC, row, col - (nC - 1) * step).cuda().float()
    for i in range(nC):
        output[:, i, :, :] = inputs[:, :, step * i : step * i + col - (nC - 1) * step]
    return output

def gen_meas_torch(data_batch: torch.Tensor, mask3d_batch: torch.Tensor, Y2H: bool = True, mul_mask: bool = False) -> torch.Tensor:
    """
    Generate measurements from data and mask (Forward model).
    
    Args:
        data_batch: Ground truth data batch.
        mask3d_batch: Mask batch.
        Y2H: Whether to convert Y (measurement) back to H (pseudo-HSI).
        mul_mask: Whether to multiply H by mask.
        
    Returns:
        torch.Tensor: The generated measurement or pseudo-HSI.
    """
    # nC = data_batch.shape[1]
    temp = shift(mask3d_batch * data_batch, 2)
    Phi_batch = torch.sum(shift(mask3d_batch, 2), 1)
    meas = torch.sum(temp, 1)
    
    if Y2H:
        meas = meas / (Phi_batch + 0.000001)
        H = shift_back(meas, 2)
        if mul_mask:
            HM = torch.mul(H, mask3d_batch)
            return HM
        return H
    return meas

def init_meas(gt: torch.Tensor, mask: torch.Tensor, input_setting: str) -> torch.Tensor:
    """
    Initialize measurement from ground truth and mask.
    
    Args:
        gt: Ground truth tensor.
        mask: Mask tensor.
        input_setting: Setting string (unused in current logic).
        
    Returns:
        torch.Tensor: Generated measurement.
    """
    input_meas = gen_meas_torch(gt, mask, Y2H=True, mul_mask=False)
    return input_meas

