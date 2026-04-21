from typing import Any

from omegaconf import DictConfig
import torch
import torch.nn as nn

from coach_pl.configuration import configurable
from coach_pl.model import MODEL_REGISTRY
from coach_pl.utils.checkpoint import load_pretrained
from coach_pl.utils.logging import setup_logger

from hetero_recon.model.backbone import Backbone, build_backbone
from hetero_recon.model.head import build_head
from hetero_recon.model.volume import build_volume, Volume

__all__ = ["HeterogeneousOnlyReconstructor"]

logger = setup_logger(__name__, rank_zero_only=True)


@MODEL_REGISTRY.register()
class HeterogeneousOnlyReconstructor(nn.Module):

    @configurable
    def __init__(
        self,
        backbone: Backbone,
        head: nn.Module,
        volume: Volume,
    ) -> None:
        super().__init__()

        self.backbone = backbone
        self.head = head
        self.volume = volume

    @classmethod
    def from_config(cls, cfg: DictConfig) -> dict[str, Any]:
        backbone = build_backbone(cfg)
        if cfg.MODEL.BACKBONE.PRETRAINED_PATH:
            load_pretrained(backbone, cfg.MODEL.BACKBONE.PRETRAINED_PATH, cfg.MODEL.BACKBONE.PRETRAINED_PREFIX)

        return {
            "backbone": backbone,
            "head": build_head(cfg, backbone.output_shape),
            "volume": build_volume(cfg),
        }

    def forward(
        self,
        y_real: torch.Tensor,
        radius: int,
        rotations: torch.Tensor,
        translations: torch.Tensor,
        encode_only: bool = False,
        return_z_only: bool = False,
    ) -> dict[str, Any]:
        # Forward backbone
        tokens_dict = self.backbone(y_real.unsqueeze(1))
        cls_tokens = tokens_dict["x_norm_clstoken"]
        patch_tokens = tokens_dict["x_norm_patchtokens"]

        # Forward head
        conformations, conformations_mu, conformations_logvar = self.head(cls_tokens, patch_tokens)

        if encode_only or return_z_only:
            return {
                "z": conformations,
                "z_mu": conformations_mu,
                "z_logvar": conformations_logvar,
                "rots": rotations,
                "trans": translations,
            }

        with torch.autocast(device_type="cuda", dtype=torch.float32):
            y_hat = self.volume(rotations, conformations, radius=radius)

        return {
            "y_hat": y_hat,
            "z": conformations,
            "z_mu": conformations_mu,
            "z_logvar": conformations_logvar,
            "rots": rotations,
            "trans": translations,
        }

    @torch.jit.ignore
    def load_pretrained(self, checkpoint_path: str, prefix: str = "") -> None:
        load_pretrained(self, checkpoint_path, prefix)
