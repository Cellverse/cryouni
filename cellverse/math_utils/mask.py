"""
A module for masks in the spatial domain.
"""

from __future__ import annotations

from functools import lru_cache

import torch

from .radial import get_spatial_radial_grid

__all__ = [
    "get_radial_mask",
]


def _normalize_radius(radius: int | float, shape: torch.Size, max_radius: int) -> float:
    """
    Check radius and normalize radius to the range [0, 1].

    Args:
        `radius` (int | float): The radius to normalize.
        `shape` (torch.Size): The shape of the grid.
        `max_radius_val` (int): The maximum possible integer radius.

    Returns:
        (float): The normalized radius.
    """
    if isinstance(radius, int):
        if not (0 <= radius <= max_radius):
            raise ValueError(f"Integer radius {radius} is out of the valid range [0, {max_radius}] for shape {shape}")
        return radius / max_radius

    if not (0.0 <= radius <= 1.0):
        raise ValueError(f"Float radius {radius} must be in the range [0.0, 1.0]")
    return radius


@lru_cache
def get_radial_mask(
    shape: torch.Size,
    inner_radius: int | float,
    outer_radius: int | float | None = None,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Generate a radial mask in the spatial domain, with an optional soft edge.

    - If `outer_radius` is not provided, it creates a hard mask where values
      inside `inner_radius` are True and outside are False.
    - If `outer_radius` is provided, it creates a soft mask with a value of 1.0
      inside `inner_radius`, and a linear transition to 0.0 at `outer_radius`.

    The radius can be specified as:
    - An `int`, representing the radius **in pixels**, in the range [0, min(shape) // 2].
    - A `float`, representing the normalized radius in the range [0, 1].

    A 1D cross-section of the soft mask's profile:

      1.0 |*****
          |      **
          |         ** (linear falloff)
          |            **
      0.0 |_______________*****__________
              ^            ^
         inner_radius outer_radius

    Note:
        - The `shape` should be equal along each dimension.
        - The returned grid is automatically cached for performance.

    Args:
        `shape` (torch.Size): The shape of the grid in (Z, Y, X) order.
        `inner_radius` (int | float): The inner radius of the mask.
        `outer_radius` (int | float | None): The outer radius for the soft transition.
            If None, a hard mask is returned. Defaults to None.
        `device` (torch.device | None): The device of the mask. Defaults to 'cpu' if None.

    Returns:
        (torch.Tensor): A boolean tensor for a hard mask or a float tensor for a soft mask.
    """
    assert all(s == shape[0] for s in shape), f"Expect all dimensions to be equal, got {shape = }."

    grid = get_spatial_radial_grid(shape, scale=False, round=False, device=device)

    max_radius = min(shape) // 2
    norm_inner_radius = _normalize_radius(inner_radius, shape, max_radius)

    # Hard mask case
    if outer_radius is None:
        return grid <= norm_inner_radius

    # Soft mask case
    norm_outer_radius = _normalize_radius(outer_radius, shape, max_radius)

    if norm_inner_radius > norm_outer_radius:
        raise ValueError(f"Expect inner_radius <= outer_radius, got {inner_radius = }, {outer_radius = }")

    # Handle the edge case where the falloff region has zero width.
    # This should behave identically to a hard mask, but return a float tensor.
    if norm_inner_radius == norm_outer_radius:
        return (grid <= norm_inner_radius).float()

    mask = (grid < norm_inner_radius).float()

    transition_region = (grid >= norm_inner_radius) & (grid <= norm_outer_radius)
    denominator = norm_outer_radius - norm_inner_radius

    linear_falloff = (norm_outer_radius - grid[transition_region]) / denominator
    mask[transition_region] = linear_falloff

    return mask
