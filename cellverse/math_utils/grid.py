"""
This module provides functions to create uniform grids in both frequency and spatial domains.

- For frequency grids, the coordinates are uniformly spaced in the range (-0.5, 0.5) for each dimension.
- For spatial grids, the coordinates are uniformly spaced in the range [-1, 1] for each dimension.

Due to `torch.nn.functional.affine_grid`'s implementation, only 1D, 2D and 3D grids are supported,
which is enough for common use cases.
"""

from __future__ import annotations

from functools import lru_cache
import warnings

import torch
import torch.nn.functional as F

from .number import is_odd

__all__ = [
    "get_frequency_grid",
    "get_spatial_grid",
]


def _get_grid(
    shape: torch.Size,
    R: torch.Tensor | None = None,
    t: torch.Tensor | None = None,
    device: torch.device | None = None,
    align_corners: bool = True,
) -> torch.Tensor:
    ndim = len(shape)

    if device is None:
        if R is not None:
            device = R.device
        elif t is not None:
            device = t.device
        else:
            device = torch.device("cpu")

    # 1D case
    if ndim == 1:
        if align_corners:
            return torch.linspace(-1.0, 1.0, shape[0], device=device)
        else:
            return torch.fft.fftshift(torch.fft.fftfreq(shape[0], device=device))

    # Record whether R and t have batch dimension
    R_has_batch = R is not None and R.ndim > 2
    t_has_batch = t is not None and t.ndim > 1

    if R is None:
        R = torch.eye(ndim, device=device)
    else:
        R = R.to(device)

    if t is None:
        t = torch.zeros(ndim, device=device)
    else:
        t = t.to(device)

    if not R_has_batch and not t_has_batch:
        R = R.unsqueeze(0)
        t = t.unsqueeze(0)
    elif not R_has_batch:
        R = R.unsqueeze(0).expand(t.size(0), -1, -1)
    elif not t_has_batch:
        t = t.unsqueeze(0).expand(R.size(0), -1)
    else:
        assert R.size(0) == t.size(0), f"R and t must have the same batch dimension, but got {R.size() = } and {t.size() = }"

    theta = torch.cat([R, t.unsqueeze(-1)], dim=-1) # [B, N, N + 1]

    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="ignore",
            message="Since version 1.3.0, affine_grid behavior has changed for unit-size grids when align_corners=True.",
            category=UserWarning,
        )
        grid = F.affine_grid(
            theta=theta,
            size=(theta.size(0), 1, *shape),
            align_corners=align_corners,
        )                                    # [B, Z, Y, X, (x, y, z)] | [B, Y, X, (x, y)]

    if not align_corners:
        grid *= 0.5

    # Remove batch dimension if both R and t have no batch dimension
    if not (R_has_batch or t_has_batch):
        grid = grid.squeeze(0)

    return grid


@lru_cache
def _get_cached_grid(shape: torch.Size, device: torch.device | None = None, align_corners: bool = True) -> torch.Tensor:
    return _get_grid(shape, device=device, align_corners=align_corners)


def get_frequency_grid(
    shape: torch.Size,
    R: torch.Tensor | None = None,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Generate a uniform spaced 3D frequency grid, range (-0.5, 0.5).

    Args:
        `shape` (torch.Size): The shape of the grid in (Z, Y, X) order,
            only odd dimensions are supported.
        `R` (torch.Tensor | None): The rotation matrix of the unit grid,
            - For 3D, shape [3, 3] or [B, 3, 3].
            - For 2D, shape [2, 2] or [B, 2, 2].
            If None, the identity matrix is used.
        `device` (torch.device | None): The device of the grid, default to cpu.

    Returns:
        (torch.Tensor): The frequency grid coordinates,
            shape [Z, Y, X (x, y, z)] for 3D and [Y, X (x, y)] for 2D if R has no batch dimension,
            shape [B, Z, Y, X (x, y, z)] for 3D and [B, Y, X (x, y)] for 2D if R has batch dimension.
    """
    assert all(is_odd(n) for n in shape), f"Expect all dimensions to be odd, got {shape = }."
    if R is None:
        return _get_cached_grid(shape, device, align_corners=False)
    else:
        return _get_grid(shape, R, None, device, align_corners=False)


def get_spatial_grid(
    shape: torch.Size,
    R: torch.Tensor | None = None,
    t: torch.Tensor | None = None,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Generate a uniform spaced 3D spatial grid, range [-1, 1].

    Args:
        `shape` (torch.Size): The shape of the grid in (Z, Y, X) order,
            both odd and even dimensions are supported.
        `R` (torch.Tensor | None): The rotation matrix of the unit grid,
            - For 3D, shape [3, 3] or [B, 3, 3].
            - For 2D, shape [2, 2] or [B, 2, 2].
            If None, the identity matrix is used.
        `t` (torch.Tensor | None): The translation vector of the unit grid,
            - For 3D, shape [3] or [B, 3].
            - For 2D, shape [2] or [B, 2].
            If None, the zero vector is used.
        `device` (torch.device | None): The device of the grid, default to cpu.

    Returns:
        (torch.Tensor): The spatial grid coordinates,
            shape [Z, Y, X (x, y, z)] for 3D and [Y, X (x, y)] for 2D if R, t has no batch dimension,
            shape [B, Z, Y, X (x, y, z)] for 3D and [B, Y, X (x, y)] for 2D if R, t has batch dimension.
    """
    if R is None and t is None:
        return _get_cached_grid(shape, device, align_corners=True)
    else:
        return _get_grid(shape, R, t, device, align_corners=True)
