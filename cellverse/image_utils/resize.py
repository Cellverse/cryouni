"""
This module provides image resizing utilities.
Resizing performed in the image are cropping and padding.
"""

from __future__ import annotations

from numbers import Number

import torch

from cellverse.math_utils.number import get_next_even

__all__ = [
    "crop_pad_center",
    "frequency_bin_center",
    "frequency_crop_center",
]


def crop_pad_center(x: torch.Tensor, target_shape: torch.Size, pad_value: Number | torch.Tensor = 0.0) -> torch.Tensor:
    """
    Cropping or padding the center of the data to the target size (supports batched input)
    by copying the center of the data and padding the rest.

    Example:
        >>> x = torch.randn(2, 3, 4, 5)
        >>> x = crop_pad_center(x, (3, 4))
        >>> x.shape
        torch.Size([2, 3, 3, 4])

    Args:
        `x` (torch.Tensor): Input data.
        `target_shape` (torch.Size): Target shape.
        `pad_value` (float | torch.Tensor): Value to pad with when padding, defaults to 0.0.
            If a tensor is provided, it be of shape [] or [...].

    Returns:
        (torch.Tensor): data of shape [..., *target_shape].
    """
    dims = len(target_shape)
    batch_shape, input_shape = x.shape[:-dims], x.shape[-dims :]
    output_shape = batch_shape + target_shape

    # Create the final output canvas, pre-filled with the padding value.
    if isinstance(pad_value, Number):
        y = x.new_full(output_shape, pad_value)
    elif isinstance(pad_value, torch.Tensor):
        y = x.new_ones(output_shape) * pad_value.reshape(batch_shape + (1,) * dims).to(x.device)
    else:
        raise TypeError(f"pad_value must be a Number or torch.Tensor, not {type(pad_value)}")

    src_slices = [...]
    dst_slices = [...]

    # Calculate source and destination slices for each dimension
    for input_dim, target_dim in zip(input_shape, target_shape):
        if input_dim > target_dim: # Cropping
            src_start = (input_dim - target_dim) // 2
            src_end = src_start + target_dim
            dst_start, dst_end = 0, target_dim
        else:                      # Padding
            dst_start = (target_dim - input_dim) // 2
            dst_end = dst_start + input_dim
            src_start, src_end = 0, input_dim

        src_slices.append(slice(src_start, src_end))
        dst_slices.append(slice(dst_start, dst_end))

    y[tuple(dst_slices)] = x[tuple(src_slices)]

    return y


def frequency_crop_center(x: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    """
    Crop the input tensor in the frequency domain.

    Args:
        `x` (torch.Tensor): Input data of shape [..., H, W].
        `target_shape` (torch.Size): Target shape.

    Returns:
        (torch.Tensor): Cropped data of shape [..., *target_shape].
    """
    assert len(target_shape) == 2, f"Target shape must be 2D, got {target_shape = }."
    assert x.shape[-2 :] >= target_shape, f"Input shape must be larger than the target shape, got {x.shape[-2 :] = } and {target_shape = }."

    L = get_next_even(max(x.shape[-2 :]))
    target_L = get_next_even(max(target_shape))

    # --- 1. Pad the input data spatially with the mean value to a square of even size L. ---
    x = crop_pad_center(x, (L, L), pad_value=x.mean(dim=(-2, -1)))

    # --- 2. Convert the input data to the frequency domain. ---
    x_freq = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))

    # --- 3. Crop the center of the frequency domain to a square of even size target_L. ---
    x_freq = crop_pad_center(x_freq, (target_L, target_L))

    # --- 4. Convert the frequency domain back to the spatial domain. ---
    x = torch.fft.ifft2(torch.fft.ifftshift(x_freq, dim=(-2, -1))).real

    # --- 5. Crop the spatial domain to the target shape. ---
    x = crop_pad_center(x, (target_shape))

    # Because the IFFT is performed on a smaller grid (target_L) than the input FFT (L),
    # the normalization factor changes. Scaling factor ~ area ratio.
    intensity_scale = (target_L * target_L) / (L * L)
    x = x * intensity_scale

    return x


def frequency_bin_center(x: torch.Tensor, bin_factor: Number) -> torch.Tensor:
    """
    Binning the input spatial domain data by a factor of `bin_factor` in the frequency domain.

    Args:
        `x` (torch.Tensor): Input data of shape [..., H, W].
        `bin_factor` (float): Binning factor (must be >= 1.0).

    Returns:
        (torch.Tensor): Binned data of shape [..., H / bin_factor, W / bin_factor].
    """
    assert bin_factor >= 1.0, f"Binning factor is the ratio of the input shape to the target shape, must be >= 1.0, got {bin_factor = }."

    return frequency_crop_center(x, (int(x.shape[-2] / bin_factor), int(x.shape[-1] / bin_factor)))
