from typing import Any

import torch

from coach_pl.module import MODULE_REGISTRY
from coach_pl.utils.logging import setup_logger

from hetero_recon.module.base import BaseModule

logger = setup_logger(__name__, rank_zero_only=True)

__all__ = ["HeterogeneousOnlyModule"]


@MODULE_REGISTRY.register()
class HeterogeneousOnlyModule(BaseModule):

    def model_forward(self, batch: Any, encode_only: bool, return_z_only: bool) -> Any:
        return self.model(
            y_real=batch['y_real'],
            radius=self.radius,
            rotations=batch['rots'],
            translations=batch['trans'],
            encode_only=encode_only,
            return_z_only=return_z_only,
        )

    @torch.autocast(device_type="cuda", dtype=torch.float32)
    def criterion_forward(self, predictions: Any, batch: Any) -> Any:
        return self.criterion(
            y_hat=predictions['y_hat'],
            y=batch['y'],
            ctf=batch['ctf'],
            trans=predictions['trans'],       # using refined translation
            radius=self.radius,
            z_mu=predictions['z_mu'],
            z_logvar=predictions['z_logvar'],
        )
