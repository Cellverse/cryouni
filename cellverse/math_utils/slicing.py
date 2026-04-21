"""
This module provides a encapsulation of the slicing operation for 3D volumes,
which supports both Fourier and Hilbert transforms.
"""

import torch
import torch.nn.functional as F

__all__ = [
    "slice_volume",
]


def slice_volume(volume: torch.Tensor, spatial_grid: torch.Tensor) -> torch.Tensor:
    """
    Slice a 3D volume at specific spatial positions.

    Args:
        `volume` (torch.Tensor): The 3D volume to be sliced, shape [Z, Y, X].
        `spatial_grid` (torch.Tensor): The spatial coordinates of the slicing positions,
            shape [B, 1, Y, X, (x, y, z)].

    Returns:
        (torch.Tensor): The sliced 2D images, shape [B, Y, X].
    """
    assert volume.ndim == 3, "Only 3D volumes are supported."

    # [1, C, Z, Y, X]
    if torch.is_complex(volume):
        input = torch.view_as_real(volume).permute(3, 0, 1, 2).unsqueeze(0)
    else:
        input = volume.unsqueeze(0).unsqueeze(0)

    B = spatial_grid.size(0)
    sliced = F.grid_sample(
        input.expand(B, -1, -1, -1, -1), # [B, C, Z, Y, X]
        spatial_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )                                    # [B, C, 1, Y, X]

    # [B, Y, X]
    if torch.is_complex(volume):
        sliced = torch.view_as_complex(sliced.permute(0, 2, 3, 4, 1).contiguous()).squeeze(1)
    else:
        sliced = sliced.squeeze(1).squeeze(1)

    return sliced
