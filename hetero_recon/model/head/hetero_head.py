from typing import Any

from omegaconf import DictConfig
import torch
import torch.nn as nn

from coach_pl.configuration import configurable

from hetero_recon.model.head.build import HEAD_REGISTRY
from hetero_recon.model.layer.normalization import LayerNorm2d
from hetero_recon.model.utils.shape_specification import ShapeSpecification

__all__ = ["ClassPatchTokenHeteroHead"]


@HEAD_REGISTRY.register()
class ClassPatchTokenHeteroHead(nn.Module):

    @configurable
    def __init__(
        self,
        in_channels: int,
        embed_channels: int,
        z_channels: int,
        image_size: int,
        patch_size: int,
        variational: bool,
        cls_norm: bool,
        std_z_init: float,
    ):
        super().__init__()

        self.neck_patch = nn.Sequential(
            LayerNorm2d(in_channels),
            nn.Conv2d(
                in_channels,
                embed_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )
        self.neck_cls = nn.LayerNorm(in_channels) if cls_norm else nn.Identity()

        self.fc_z_mu = nn.Linear(in_channels + (image_size // patch_size) * (image_size // patch_size) * embed_channels, z_channels)
        if variational:
            self.fc_z_logvar = nn.Linear(in_channels + (image_size // patch_size) * (image_size // patch_size) * embed_channels, z_channels)

        self.variational = variational
        self.grid_size = image_size // patch_size
        self.std_z_init = std_z_init
        self.z_channels = z_channels

    @classmethod
    def from_config(cls, cfg: DictConfig, input_shape: ShapeSpecification) -> dict[str, Any]:
        return {
            "in_channels": input_shape.out_channels,
            "embed_channels": cfg.MODEL.HEAD.EMBED_CHANNELS,
            "z_channels": cfg.MODEL.HEAD.Z_CHANNELS,
            "image_size": cfg.DATAMODULE.DATASET.IMAGE_SIZE.SPATIAL,
            "patch_size": cfg.MODEL.BACKBONE.PATCH_SIZE,
            "variational": cfg.MODEL.HEAD.VARIATIONAL,
            "cls_norm": not cfg.MODEL.BACKBONE.FINAL_NORM,
            'std_z_init': cfg.MODEL.HEAD.Z_STD_INIT,
        }

    def random_init(
        self,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.randn(batch_size, self.z_channels, device=device) * self.std_z_init

    def forward(
        self,
        cls_tokens: torch.Tensor,
        patch_tokens: torch.Tensor,
    ) -> torch.Tensor:
        B, L, E = patch_tokens.shape

        patch_tokens = patch_tokens.view(B, self.grid_size, self.grid_size, E).permute(0, 3, 1, 2)
        patch_tokens = self.neck_patch(patch_tokens)
        patch_tokens = patch_tokens.flatten(-3)
        cls_tokens = self.neck_cls(cls_tokens)

        tokens = torch.cat([cls_tokens, patch_tokens], dim=1)
        z_mu = self.fc_z_mu(tokens)
        if self.variational:
            z_logvar = self.fc_z_logvar(tokens)
        else:
            z_logvar = None

        if z_logvar is None:
            z = z_mu + self.std_z_init * torch.randn_like(z_mu)
        else:
            z = z_mu + torch.randn_like(z_mu) * torch.exp(0.5 * z_logvar)

        return z, z_mu, z_logvar
