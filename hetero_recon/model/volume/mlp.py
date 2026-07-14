from typing import Any

from omegaconf import DictConfig
import torch
import torch.nn as nn
from torch.nn import functional as F

from cellverse.math_utils.frequency import ht_object, uncircularize
from cellverse.math_utils.grid import get_frequency_grid
from cellverse.math_utils.mask import get_radial_mask
from cellverse.math_utils.number import get_next_odd
from coach_pl.configuration import configurable

from .build import VOLUME_REGISTRY
from .volume import Volume
from hetero_recon.model.layer import SwiGLUResidualLinearMLP

__all__ = [
    "FullyConnectedMlpVolume",
]


@VOLUME_REGISTRY.register()
class FullyConnectedMlpVolume(Volume):

    @configurable
    def __init__(
        self,
        conformation_channels: int,
        pe_channels: int,
        embed_channels: int,
        depth: int,
        pe_feat_sigma: float,
        hartley_image_size: int,
    ) -> None:
        super().__init__()

        self.hartley_image_size = get_next_odd(hartley_image_size)
        self.conformation_channels = conformation_channels

        # Random Fourier Feature encoding parameter
        rand_freqs = torch.randn((1, 1, pe_channels // 2, 3)) * pe_feat_sigma
        self.rand_freqs = nn.Parameter(rand_freqs, requires_grad=False)

        self.mlp = SwiGLUResidualLinearMLP(
            in_dim=pe_channels + conformation_channels,
            depth=depth,
            embed_dim=embed_channels,
            out_dim=1,
            mlp_ratio=2.0,
        )

    @classmethod
    def from_config(cls, cfg: DictConfig) -> dict[str, Any]:
        return {
            "conformation_channels": cfg.MODEL.HEAD.Z_CHANNELS,
            "pe_channels": cfg.MODEL.VOLUME.PE_CHANNELS,
            "embed_channels": cfg.MODEL.VOLUME.EMBED_CHANNELS,
            "depth": cfg.MODEL.VOLUME.DEPTH,
            "pe_feat_sigma": cfg.MODEL.VOLUME.PE_FEAT_SIGMA,
            "hartley_image_size": cfg.DATAMODULE.DATASET.IMAGE_SIZE.HARTLEY,
        }

    def _embed_coords(self, coords: torch.Tensor) -> torch.Tensor:
        freqs = self.rand_freqs * (self.hartley_image_size // 2)
        kx_ky_kz = coords.unsqueeze(-2) * freqs
        k = kx_ky_kz.sum(-1)
        return torch.cat([torch.sin(k), torch.cos(k)], dim=-1)

    def _forward_with_coords(self, z: torch.Tensor, embedded_coords: torch.Tensor) -> torch.Tensor:
        z_expanded = z.unsqueeze(1).expand(-1, embedded_coords.shape[1], -1)
        mlp_input = torch.cat([embedded_coords, z_expanded], dim=-1)
        return self.mlp(mlp_input).squeeze(-1)

    def forward(self, rotations: torch.Tensor, conformations: torch.Tensor, radius: int = None) -> torch.Tensor:
        """
        Generates a predicted 2D Hartley image slice.

        This method orchestrates the decoding process:
        1. Rotates the canonical 2D coordinate slice.
        2. Masks coordinates outside the given radius.
        3. Embeds the valid coordinates using the subclass-specific method.
        4. Decodes the final values using the subclass-specific MLP.
        """
        device = rotations.device
        D = self.hartley_image_size
        radius = radius if radius is not None else D // 2

        # Rotate the canonical coordinate slice
        coords_slice = get_frequency_grid((1, D, D), device=device).reshape(-1, 3)
        coords = torch.einsum("bji, nj -> bni", rotations, coords_slice)

        # Apply mask, embed, and decode
        mask = get_radial_mask((D, D), inner_radius=radius, device=device).reshape(-1)
        masked_coords = coords[:, mask]
        embedded_coords = self._embed_coords(masked_coords)
        y_hat = self._forward_with_coords(conformations, embedded_coords)

        return y_hat

    @torch.inference_mode()
    def eval_volume(self, conformations: torch.Tensor, noise_std: float, radius: int = None, volume_size: int = None) -> torch.Tensor:
        """
        Reconstructs a full 3D volume in real space by decoding slice by slice.
        This is a shared, high-level process that relies on the subclass-specific
        implementation of `_embed_coords` and `_forward_with_coords`.

        Args:
            conformations: Latent conformations of shape [B, C].
            noise_std: Noise standard deviation for scaling.
            radius: Radius of the spherical mask in Hartley domain.
            volume_size: Output spatial dimension D for volume.
                If None, uses the model's native hartley_image_size.
        """
        D = self.hartley_image_size
        device = conformations.device
        radius = radius if radius is not None else D // 2

        volume_ht = torch.zeros((D, D, D), device=device)
        conformations = conformations.reshape(1, -1)

        # Prepare 3D coordinate grid and spherical mask
        coords_3d = get_frequency_grid((D, D, D), device=device)
        sphere_mask = get_radial_mask((D, D, D), inner_radius=radius, device=device)

        # Decode slice by slice to avoid memory issues
        for i in range(D):
            if not sphere_mask[i].any():
                continue

            coords_slice_i = coords_3d[i, ...].reshape(1, -1, 3)

            mask_i = sphere_mask[i, :].reshape(-1)
            masked_coords = coords_slice_i[:, mask_i]

            # Use subclass-specific methods for embedding and decoding
            embedded_coords = self._embed_coords(masked_coords)
            y_hat_slice = self._forward_with_coords(conformations, embedded_coords).flatten()

            # Place the decoded values back into the Hartley volume
            volume_ht[i].view(-1)[mask_i] = y_hat_slice

        # Scale and transform back to spatial domain
        volume_ht = volume_ht * noise_std
        volume_sp = uncircularize(ht_object.iht3_center(volume_ht), last_n_dims=3)

        # Resample to target volume size in spatial domain
        if volume_size is not None and volume_size != D:
            volume_sp = F.interpolate(
                volume_sp.unsqueeze(0).unsqueeze(0),
                size=(volume_size, volume_size, volume_size),
                mode='trilinear', align_corners=False,
            ).squeeze(0).squeeze(0)

        return volume_sp
