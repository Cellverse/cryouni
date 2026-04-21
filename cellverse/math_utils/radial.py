"""
This module provides functions to create radial grids in both frequency and spatial domains.
Also, operations that is based on radial grids are provided.

Due to the constraints of centered transforms and the implementation uniform grids,
only odd-sized 2D and 3D inputs are supported, which is enough for common use cases.

Note that `shell` and `radial` are the same meaning in this context.
"""

from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn.functional as F

from .grid import (
    get_frequency_grid,
    get_spatial_grid,
)
from .number import get_nyquist_index

__all__ = [
    "expand_from_radial",
    "get_frequency_radial_grid",
    "get_spatial_radial_grid",
    "avg_over_radial",
    "sum_over_radial",
]


@lru_cache
def get_frequency_radial_grid(
    shape: torch.Size,
    scale: bool = True,
    round: bool = True,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Generate a radial distance grid in the frequency domain,
    range [0, sqrt(0.5 ** len(shape))) for `scale` is False or [0, sqrt(0.5 * max(shape) ** len(shape))) for `scale` is True.

    Note:
        The returned grid is automatically cached.

    Args:
        `shape` (torch.Size): The shape of the grid in (Z, Y, X) order.
        `scale` (bool): Whether to scale the grid, default to True.
        `round` (bool): Whether to round the scaled grid to integer.
            Only works when `scale` is True, default to True.
        `device` (torch.device | None): The device of the grid, default to cpu.

    Returns:
        (torch.Tensor): The radial distance of each frequency point, shape [Z, Y, X] for 3D or [Y, X] for 2D.
    """
    grid = get_frequency_grid(shape, device=device)
    if scale:
        grid = grid * torch.as_tensor(tuple(shape[::-1]), dtype=grid.dtype, device=device)
    radial_distance = grid.norm(dim=-1)
    if scale and round:
        radial_distance = radial_distance.round().long()
    return radial_distance


@lru_cache
def get_spatial_radial_grid(
    shape: torch.Size,
    scale: bool = True,
    round: bool = True,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Generate a radial distance grid in the spatial domain,
    range [0, 1] for `scale` is False or [0, sqrt(max(shape) ** len(shape))] for `scale` is True.

    Note:
        The returned grid is automatically cached.

    Args:
        `shape` (torch.Size): The shape of the grid in (Z, Y, X) order.
        `scale` (bool): Whether to scale the grid, default to True.
        `round` (bool): Whether to round the scaled grid to integer.
            Only works when `scale` is True, default to True.
        `device` (torch.device | None): The device of the grid, default to cpu.

    Returns:
        The radial distance of each spatial point, shape [Z, Y, X] for 3D or [Y, X] for 2D.
    """
    grid = get_spatial_grid(shape, device=device)
    if scale:
        grid = grid * torch.as_tensor(tuple(shape[::-1]), dtype=grid.dtype, device=device)
    radial_distance = grid.norm(dim=-1)
    if scale and round:
        radial_distance = radial_distance.round().long()
    return radial_distance


def avg_over_radial(data: torch.Tensor, last_n_dims: int | None = None) -> torch.Tensor:
    """
    Average over the radial direction in the frequency domain.

    Args:
        `data` (torch.Tensor): The data to average over, shape [..., Z, Y, X] for 3D or [..., Y, X] for 2D.
        `last_n_dims` (int | None): Number of last dimensions to transform.
            If None, it will be treated as all dimensions.

    Returns:
        (torch.Tensor): The average over the radial direction, shape [..., nyquist_index(data.shape[-last_n_dims:])].
    """
    last_n_dims = last_n_dims if last_n_dims is not None else data.ndim
    assert last_n_dims in (2, 3), f"Only 2D and 3D are supported, got {last_n_dims = }."

    batch, shape = data.shape[:-last_n_dims], data.shape[-last_n_dims :]

    data_flat = data.reshape(batch + (-1,))
    radial_flat = get_frequency_radial_grid(shape, device=data.device).flatten()
    radial_max = radial_flat[0].item() + 1 # The first element distance is always the maximum radial.

    index = radial_flat.expand_as(data_flat)
    count = radial_flat.bincount(minlength=radial_max).clamp_min_(1)
    value = data.new_zeros(batch + (radial_max,)).scatter_add_(len(batch), index, data_flat)
    result = value / count

    return result[..., : get_nyquist_index(shape)]


def sum_over_radial(data: torch.Tensor, last_n_dims: int | None = None) -> torch.Tensor:
    """
    Summing over the frequency domain data along the radial direction.

    Args:
        `data` (torch.Tensor): The data to sum over, shape [..., Z, Y, X] for 3D or [..., Y, X] for 2D.
        `last_n_dims` (int | None): Number of last dimensions to transform.
            If None, it will be treated as all dimensions.

    Returns:
        (torch.Tensor): The sum over the radial direction, shape [..., nyquist_index(data.shape[-last_n_dims:])].
    """
    last_n_dims = last_n_dims if last_n_dims is not None else data.ndim
    assert last_n_dims in (2, 3), f"Only 2D and 3D are supported, got {last_n_dims = }."

    batch, shape = data.shape[:-last_n_dims], data.shape[-last_n_dims :]

    data_flat = data.reshape(batch + (-1,))
    radial_flat = get_frequency_radial_grid(shape, device=data.device).flatten()
    radial_max = radial_flat[0].item() + 1 # The first element distance is always the maximum radial.

    index = radial_flat.expand_as(data_flat)
    result = data.new_zeros(batch + (radial_max,)).scatter_add_(len(batch), index, data_flat)

    return result[..., : get_nyquist_index(shape)]


def expand_from_radial(radial: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    Expand a radial signal to a 2D or 3D volume. It can be considered as the inverse operation of `avg_over_radial`.

    Args:
        `radial` (torch.Tensor): The radial signal to expand, shape [..., nyquist_index(shape)].
        `shape` (torch.Size): The shape of the volume in (Z, Y, X) order.

    Returns:
        (torch.Tensor): The expanded volume, shape [..., Z, Y, X] for 3D or [..., Y, X] for 2D.
    """
    radial_flat = get_frequency_radial_grid(shape, device=radial.device).flatten()

    batch, length = radial.shape[:-1], radial.shape[-1]
    radial_max = radial_flat[0].item() + 1 # The first element distance is always the maximum radial.

    # Use the last value of radial to pad radial.
    if length < radial_max:
        pad_len = radial_max - length
        padding = radial[..., -1 :].expand(*batch, pad_len)
        radial = torch.cat([radial, padding], dim=-1)

    return radial[..., radial_flat].reshape(*batch, *shape)
