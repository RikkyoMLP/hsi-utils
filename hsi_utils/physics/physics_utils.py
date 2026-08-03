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
    if inputs.ndim != 4:
        raise ValueError(f"shift expects [B,C,H,W], got shape {tuple(inputs.shape)}")
    if step < 0:
        raise ValueError("step must be non-negative")
    bs, nC, row, col = inputs.shape
    output = inputs.new_zeros(bs, nC, row, col + (nC - 1) * step)
    for i in range(nC):
        output[:, i, :, step * i : step * i + col] = inputs[:, i]
    return output


def shift_back(
    inputs: torch.Tensor,
    step: int = 2,
    channels: int = 28,
) -> torch.Tensor:
    """
    Reverse the dispersion (shift) effect.

    Args:
        inputs: Input tensor of shape [bs, row, col_shifted].
        step: Shift step size.
        channels: Number of spectral channels to recover.

    Returns:
        torch.Tensor: Back-shifted tensor of shape [bs, nC, row, col].
    """
    if inputs.ndim != 3:
        raise ValueError(f"shift_back expects [B,H,Wm], got shape {tuple(inputs.shape)}")
    if step < 0:
        raise ValueError("step must be non-negative")
    if channels <= 0:
        raise ValueError("channels must be positive")
    bs, row, col = inputs.shape
    width = col - (channels - 1) * step
    if width <= 0:
        raise ValueError(
            f"shifted width {col} is too small for channels={channels}, step={step}"
        )
    output = inputs.new_zeros(bs, channels, row, width)
    for i in range(channels):
        output[:, i] = inputs[:, :, step * i : step * i + width]
    return output


def cassi_forward(
    cube: torch.Tensor,
    mask: torch.Tensor,
    step: int = 2,
) -> torch.Tensor:
    """Apply the raw (unnormalized) CASSI forward operator.

    Both tensors must be ``[B,C,H,W]`` and may reside on any device or use
    any floating dtype.  Each sample keeps its own mask; no batch element is
    implicitly copied over another one.
    """

    if cube.ndim != 4 or mask.ndim != 4:
        raise ValueError("cassi_forward expects cube and mask shaped [B,C,H,W]")
    if cube.shape != mask.shape:
        raise ValueError(
            f"cube/mask shape mismatch: {tuple(cube.shape)} != {tuple(mask.shape)}"
        )
    if cube.device != mask.device:
        raise ValueError(f"cube/mask device mismatch: {cube.device} != {mask.device}")
    if cube.dtype != mask.dtype:
        raise ValueError(f"cube/mask dtype mismatch: {cube.dtype} != {mask.dtype}")
    return shift(cube * mask, step=step).sum(dim=1)


def cassi_adjoint(
    measurement: torch.Tensor,
    mask: torch.Tensor,
    step: int = 2,
) -> torch.Tensor:
    """Apply the adjoint of :func:`cassi_forward`."""

    if measurement.ndim != 3 or mask.ndim != 4:
        raise ValueError(
            "cassi_adjoint expects measurement [B,H,Wm] and mask [B,C,H,W]"
        )
    expected_width = mask.shape[-1] + (mask.shape[1] - 1) * step
    expected_shape = (mask.shape[0], mask.shape[2], expected_width)
    if tuple(measurement.shape) != expected_shape:
        raise ValueError(
            "measurement/mask shape mismatch: "
            f"expected {expected_shape}, got {tuple(measurement.shape)}"
        )
    if measurement.device != mask.device:
        raise ValueError(
            f"measurement/mask device mismatch: {measurement.device} != {mask.device}"
        )
    if measurement.dtype != mask.dtype:
        raise ValueError(
            f"measurement/mask dtype mismatch: {measurement.dtype} != {mask.dtype}"
        )
    return shift_back(
        measurement,
        step=step,
        channels=mask.shape[1],
    ) * mask


def gen_meas_torch(
    data_batch: torch.Tensor,
    mask3d_batch: torch.Tensor,
    Y2H: bool = True,
    mul_mask: bool = False,
) -> torch.Tensor:
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
    if data_batch.ndim != 4 or mask3d_batch.ndim != 4:
        raise ValueError("gen_meas_torch expects data and mask shaped [B,C,H,W]")
    batch_size, nC, H, W = data_batch.shape
    if tuple(mask3d_batch.shape[1:]) != (nC, H, W):
        raise ValueError(
            "data/mask shape mismatch: "
            f"{tuple(data_batch.shape)} != {tuple(mask3d_batch.shape)}"
        )
    if mask3d_batch.shape[0] == 1 and batch_size != 1:
        mask3d_batch = mask3d_batch.expand(batch_size, -1, -1, -1)
    elif mask3d_batch.shape[0] != batch_size:
        raise ValueError(
            f"mask batch must be 1 or {batch_size}, got {mask3d_batch.shape[0]}"
        )
    mask3d_batch = mask3d_batch.to(device=data_batch.device, dtype=data_batch.dtype)
    meas = cassi_forward(data_batch, mask3d_batch, step=2)

    if Y2H:
        meas = meas / nC * 2
        H = shift_back(meas, channels=nC)
        if mul_mask:
            HM = torch.mul(H, mask3d_batch)
            return HM
        return H
    return meas


def init_meas(
    gt: torch.Tensor,
    mask: torch.Tensor,
    input_setting: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Initialize measurement from ground truth and mask.

    Args:
        gt: Ground truth tensor.
        mask: Mask tensor.
        input_setting: Setting string (unused in current logic).

    Returns:
        torch.Tensor: Generated measurement.
    """
    if input_setting == "H":
        Y_meas = gen_meas_torch(gt, mask, Y2H=False, mul_mask=False)
        Y_meas_normalized = Y_meas / 28 * 2  # Normalize like in forward_model
        H_meas = gen_meas_torch(gt, mask, Y2H=True, mul_mask=False)
        return H_meas, Y_meas_normalized
    elif input_setting == "HM":
        Y_meas = gen_meas_torch(gt, mask, Y2H=False, mul_mask=False)
        Y_meas_normalized = Y_meas / 28 * 2
        HM_meas = gen_meas_torch(gt, mask, Y2H=True, mul_mask=True)
        return HM_meas, Y_meas_normalized
    elif input_setting == "Y":
        input_meas = gen_meas_torch(gt, mask, Y2H=False, mul_mask=True)
        Y_meas_normalized = input_meas / 28 * 2
        return input_meas, Y_meas_normalized
    else:
        raise NotImplementedError("Unknown input setting")


def forward_model(x: torch.Tensor, Phi: torch.Tensor, step: int = 2) -> torch.Tensor:
    """
    Forward physical model: converts HSI cube to compressed measurement Y.
    This simulates the CASSI imaging process.

    Args:
        x: HSI cube [B, C, H, W]
        Phi: Mask (either shifted or unshifted)
        step: Shift step

    Returns:
        Y: Compressed measurement [B, H, W_shifted]
    """
    if x.ndim != 4 or Phi.ndim != 4:
        raise ValueError("forward_model expects x and Phi shaped [B,C,H,W]")
    if x.shape[:3] != Phi.shape[:3]:
        raise ValueError(
            f"x/Phi leading shape mismatch: {tuple(x.shape)} != {tuple(Phi.shape)}"
        )
    if x.device != Phi.device or x.dtype != Phi.dtype:
        raise ValueError("x and Phi must share device and dtype")
    nC = Phi.shape[1]
    x_shifted = shift(x, step)

    # Phi already shifted (matches shifted width) vs unshifted (needs shifting)
    if Phi.shape[-1] == x_shifted.shape[-1]:
        Phi_shifted = Phi
    else:
        Phi_shifted = shift(Phi, step)

    mea = torch.sum(x_shifted * Phi_shifted, dim=1) / nC * 2
    return mea
