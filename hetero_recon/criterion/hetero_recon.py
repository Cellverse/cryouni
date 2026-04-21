from typing import Any

from omegaconf import DictConfig
import torch
import torch.nn as nn

from cellverse.math_utils import ht_object
from cellverse.math_utils.grid import get_frequency_grid
from cellverse.math_utils.helper import to_2tuple
from cellverse.math_utils.mask import get_radial_mask
from cellverse.math_utils.number import get_next_odd
from coach_pl.configuration import configurable
from coach_pl.criterion import CRITERION_REGISTRY

__all__ = ["HeterogeneousReconstructionCriterion"]


@CRITERION_REGISTRY.register()
class HeterogeneousReconstructionCriterion(nn.Module):

    @configurable
    def __init__(self, kl_divergence_loss_beta: float, hartley_image_size: int) -> None:
        super().__init__()

        self.kl_divergence_loss_beta = kl_divergence_loss_beta
        self.hartley_image_size = get_next_odd(hartley_image_size)

    @classmethod
    def from_config(cls, cfg: DictConfig) -> dict[str, Any]:
        return {
            "kl_divergence_loss_beta": cfg.CRITERION.KL_DIVERGENCE_LOSS_BETA,
            "hartley_image_size": cfg.DATAMODULE.DATASET.IMAGE_SIZE.HARTLEY,
        }

    def calc_l2_recon_loss(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        ctf: torch.Tensor,
        trans: torch.Tensor,
        mask: torch.Tensor,
    ):
        # y_hat: shape (B, N)
        y = ht_object.translate(
            ht=y,
            t=trans * self.hartley_image_size,
            frequency_grid=get_frequency_grid(to_2tuple(self.hartley_image_size), device=y.device),
        )
        y = y[:, mask]

        # ctf: shape (B, N)
        ctf = ctf[:, mask]

        y_hat = y_hat * ctf

        return nn.functional.mse_loss(y_hat, y)

    @staticmethod
    def calc_kl_divergence_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)

    def forward(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        ctf: torch.Tensor,
        trans: torch.Tensor,
        radius: int,
        z_mu: torch.Tensor | None = None,
        z_logvar: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        D = self.hartley_image_size

        mask = get_radial_mask((D, D), inner_radius=radius, device=y_hat.device) # [D, D]

        # l2_recon_loss
        l2_recon_loss = self.calc_l2_recon_loss(y_hat, y, ctf, trans, mask)

        # kl_divergence_loss
        if z_mu is not None and z_logvar is not None:
            kl_divergence_loss = self.calc_kl_divergence_loss(z_mu, z_logvar)
        else:
            kl_divergence_loss = 0

        # loss
        loss = l2_recon_loss + self.kl_divergence_loss_beta * kl_divergence_loss / mask.sum()

        return {
            "loss": loss,
            "l2_recon_loss": l2_recon_loss,
            "kl_divergence_loss": kl_divergence_loss,
        }
