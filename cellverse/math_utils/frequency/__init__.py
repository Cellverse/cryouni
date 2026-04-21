"""
This module provides functions and objects that are related to frequency domain operations.

Objects:
    - `ft` (FourierTransform): Fourier transform object.
    - `ht` (HartleyTransform): Hartley transform object.
"""

import torch
import torch.nn.functional as F

from .fourier import (
    FourierTransform,
    NormType,
)
from .hartley import HartleyTransform

ft_object = FourierTransform()
"""Fourier transform object."""
ht_object = HartleyTransform()
"""Hartley transform object."""


def circularize(frequency_signal: torch.Tensor, last_n_dims: int | None = None) -> torch.Tensor:
    """
    Apply circular padding to a frequency domain signal along specified dimensions.

    This function expands the frequency signal one more by applying circular padding
    to the specified dimensions, effectively creating a periodic extension of the signal.

    Usually, the spatial domain signal should be odd-sized before Fourier/Hartley transform.
    Therefore, if the input signal is even-sized, it should be circularized to make it odd-sized.

    Args:
        `frequency_signal` (torch.Tensor): Input frequency domain signal.
        `last_n_dims` (int | None): Number of last dimensions to circularize.

    Returns:
        (torch.Tensor): Circularized frequency signal,
            shape [..., N + 1], where N is the original shape.
    """
    pad = [0, 1] * (last_n_dims if last_n_dims is not None else frequency_signal.ndim)
    if frequency_signal.ndim == len(pad) // 2:
        frequency_signal = frequency_signal.unsqueeze(0)
        frequency_signal = F.pad(frequency_signal, pad, mode="circular")
        frequency_signal = frequency_signal.squeeze(0)
    else:
        frequency_signal = F.pad(frequency_signal, pad, mode="circular")
    return frequency_signal


def uncircularize(frequency_signal: torch.Tensor, last_n_dims: int | None = None) -> torch.Tensor:
    """
    Extract the original signal from a circularized frequency domain signal.

    This function reverses the circularization process by extracting the central
    region of the signal one less that corresponds to the original dimensions.

    Args:
        `frequency_signal` (torch.Tensor): Input frequency domain signal.
        `last_n_dims` (int | None): Number of last dimensions to uncircularize.

    Returns:
        (torch.Tensor): Extracted frequency signal,
            shape [..., N - 1], where N is the original shape.
    """
    slices = [slice(0, -1)] * (last_n_dims if last_n_dims is not None else frequency_signal.ndim)
    return frequency_signal[..., *slices]


__all__ = [k for k in globals().keys() if not k.startswith("_")]
