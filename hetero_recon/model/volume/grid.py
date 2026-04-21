from typing import Any

from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

from cellverse.math_utils.frequency import ht_object, uncircularize
from cellverse.math_utils.grid import get_spatial_grid
from cellverse.math_utils.mask import get_radial_mask
from cellverse.math_utils.number import get_next_odd
from coach_pl.configuration import configurable

from .build import VOLUME_REGISTRY
from .volume import Volume

__all__ = ["GridVolume"]


@VOLUME_REGISTRY.register()
class GridVolume(Volume):

    @configurable
    def __init__(self, hartley_image_size: int, conformation_channels: int) -> None:
        super().__init__()

        self.hartley_image_size = get_next_odd(hartley_image_size)
        self.conformation_channels = conformation_channels

        D = self.hartley_image_size
        C = self.conformation_channels

        self.grid_ht = nn.Parameter(torch.zeros(1, C, D, D, D))

    @classmethod
    def from_config(cls, cfg: DictConfig) -> dict[str, Any]:
        return {
            "hartley_image_size": cfg.DATAMODULE.DATASET.IMAGE_SIZE.HARTLEY,
            "conformation_channels": cfg.MODEL.HEAD.Z_CHANNELS,
        }

    def forward(self, rotations: torch.Tensor, conformations: torch.Tensor, radius: int = None) -> torch.Tensor:
        B = conformations.shape[0]
        D = self.hartley_image_size
        device = rotations.device
        radius = radius if radius is not None else D // 2

        # Rotate the canonical coordinate slice
        coords_slice = get_spatial_grid((1, D, D), device=device).reshape(-1, 3) # [D * D, 3]
        coords = torch.einsum("bji, nj -> bni", rotations, coords_slice)         # [B, D * D, 3]

        # Apply mask
        mask = get_radial_mask((D, D), inner_radius=radius, device=device).reshape(-1) # [D * D]
        masked_coords = coords[:, mask]                                                # [B, N_kept, 3]

        # Reshape for grid_sample
        grid_query = masked_coords.view(B, 1, 1, -1, 3)     # [B, 1, 1, N_kept, 3]
        batch_grid = self.grid_ht.expand(B, -1, -1, -1, -1) # [B, C, D, D, D]

        # Sample
        sampled_basis = F.grid_sample(batch_grid, grid_query, align_corners=True) # [B, C, 1, 1, N_kept]
        sampled_basis = sampled_basis.view(B, self.conformation_channels, -1)     # [B, C, N_kept]

        # Linear Combination
        image = (sampled_basis * conformations.unsqueeze(-1)).sum(dim=1) # [B, N_kept]

        return image

    @torch.inference_mode()
    def eval_volume(self, conformations: torch.Tensor, noise_std, radius: int = None) -> torch.Tensor:
        B = conformations.shape[0]
        D = self.hartley_image_size
        device = conformations.device
        radius = radius if radius is not None else D // 2

        # Reshape z for broadcasting: [B, C] -> [B, C, 1, 1, 1]
        z_reshaped = conformations.view(B, -1, 1, 1, 1) # [B, C, 1, 1, 1]

        # Weighting basis volumes
        volume_ht = (self.grid_ht * z_reshaped).sum(dim=1) # [B, D, D, D]

        # Apply mask
        mask = get_radial_mask((D, D, D), inner_radius=radius, device=device)
        volume_ht[:, ~mask] = 0

        # Scale and transform back to spatial domain
        volume_ht = volume_ht * noise_std
        volume_sp = uncircularize(ht_object.iht3_center(volume_ht), last_n_dims=3) # [B, D, D, D]
        return volume_sp

    def __repr__(self) -> str:
        return f"GridVolume(hartley_image_size={self.hartley_image_size}, conformation_channels={self.conformation_channels})"
