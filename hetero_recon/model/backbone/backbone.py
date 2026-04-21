from abc import ABCMeta, abstractmethod
from typing import Any, Sequence

from omegaconf import DictConfig
import torch
import torch.nn as nn

from hetero_recon.model.utils.shape_specification import ShapeSpecification

__all__ = ["Backbone"]


class Backbone(nn.Module, metaclass=ABCMeta):
    """
    Abstract base class for all the Vision Transformers (ViTs).
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()

        self._output_shape = ShapeSpecification(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
        )

    @classmethod
    def from_config(cls, cfg: DictConfig) -> dict[str, Any]:
        args = cls.get_scale(cfg.MODEL.BACKBONE.VIT_SCALE)
        args.update({
            "img_size": cfg.DATAMODULE.DATASET.IMAGE_SIZE.SPATIAL,
            "patch_size": cfg.MODEL.BACKBONE.PATCH_SIZE,
            "in_chans": cfg.MODEL.BACKBONE.IN_CHANS,
            "final_norm": cfg.MODEL.BACKBONE.FINAL_NORM,
            "dynamic_img_size": cfg.MODEL.BACKBONE.DYNAMIC_IMG_SIZE,
            "dynamic_img_pad": cfg.MODEL.BACKBONE.DYNAMIC_IMG_PAD,
            "num_register_tokens": cfg.MODEL.BACKBONE.NUM_REGISTER_TOKENS,
            "drop_path_rate": cfg.MODEL.BACKBONE.DROP_PATH_RATE,
        })
        return args

    @property
    def output_shape(self) -> ShapeSpecification:
        """
        Get the output shape of the backbone, which is used for downstream modules.
        """
        return self._output_shape

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def _image_to_tokens(self, x: torch.Tensor) -> Any:
        """
        Convert image to tokens to be processed by Transformer blocks.
        """
        raise NotImplementedError

    def _tokens_to_features(self, x: torch.Tensor, apply_norm: bool = True) -> torch.Tensor:
        """
        Process image tokens with Transformer blocks.
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, x: torch.Tensor, apply_norm: bool = True) -> dict[str, torch.Tensor | int]:
        """
        Convert image to feature.

        Args:
            `x` (torch.Tensor): The input tensor, shape [B, C, H, W].
            `apply_norm` (bool): Whether to apply final norm to the output tensor.

        Returns:
            (dict[str, torch.Tensor | int]): The output tensors or the height and width of the patch tokens.
        """
        raise NotImplementedError

    def get_intermediate_features(
        self,
        x: torch.Tensor,
        block_taken_indices: Sequence[int],
        reshape: bool = True,
        return_class_token: bool = False,
        apply_norm: bool = True,
    ) -> Sequence[tuple[torch.Tensor, torch.Tensor | None]]:
        """
        Get intermediate layer features from the input tensor.

        Args:
            `x` (torch.Tensor): The input tensor, shape [B, C, H, W].
            `block_taken_indices` (Sequence[int]): The indices of the blocks to take intermediate features from.
            `reshape` (bool): Whether to reshape the output tensor to [B, H // P, W // P, E].
            `return_class_token` (bool): Whether to return the class token.
            `apply_norm` (bool): Whether to apply final norm to the output tensor.

        Returns:
            (Sequence[tuple[torch.Tensor, torch.Tensor | None]]):
                - A tuple of intermediate layer features, shape [B, H // P, W // P, E] if `reshape` is True, otherwise [B, H // P * W // P, E].
                - A tuple of class tokens, shape [B, E] if `return_class_token` is True, otherwise None.
        """
        raise NotImplementedError

    def get_intermediate_attn_map(self, x: torch.Tensor, block_taken_indices: Sequence[int]) -> Sequence[torch.Tensor]:
        """
        Get intermediate attention from the input tensor.

        Args:
            `x` (torch.Tensor): The input tensor, shape [B, C, H, W].
            `block_taken_indices` (Sequence[int]): The indices of the blocks to take intermediate features from.

        Returns:
            (Sequence[torch.Tensor]): A sequence of intermediate attention maps, shape [B, N, L, L].
        """
        raise NotImplementedError

    @classmethod
    def get_scale(cls, scale: str) -> dict[str, int]:
        """
        See the definition of the scale of ViT in the following papers:
            1. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (https://arxiv.org/abs/2010.11929)
            2. Scaling Vision Transformer (https://arxiv.org/abs/2106.04560)
            3. DINOv2: Learning Robust Visual Features without Supervision (https://arxiv.org/abs/https://arxiv.org/abs/2304.07193)
        """
        SCALE = {
            "tiny": {
                "embed_dim": 192,
                "depth": 12,
                "num_heads": 3
            },
            "small": {
                "embed_dim": 384,
                "depth": 12,
                "num_heads": 6
            },
            "base": {
                "embed_dim": 768,
                "depth": 12,
                "num_heads": 12
            },
            "large": {
                "embed_dim": 1152,
                "depth": 24,
                "num_heads": 18
            },
            "huge": {
                "embed_dim": 1280,
                "depth": 32,
                "num_heads": 16
            },
            "giant": {
                "embed_dim": 1536,
                "depth": 40,
                "num_heads": 24
            },
        }
        return SCALE[scale]

    @classmethod
    def random_masking(cls, x: torch.Tensor, mask_ratio: float) -> torch.BoolTensor:
        """
        Generate a random mask for the input tensor.

        Args:
            `x` (torch.Tensor): Tensor of shape [B, H, W, E].
            `mask_ratio` (float): Ratio of masked tokens.

        Returns:
            (torch.BoolTensor): Mask of shape [B, H * W], where 0 for unmasked, and 1 for masked.
        """
        B, H, W, E = x.shape
        num_masked = int(H * W * mask_ratio)

        noise = torch.rand(B, H * W, device=x.device) # [B, H * W]
        rank = noise.argsort(dim=1)                   # [B, H * W]
        mask = rank < num_masked                      # [B, H * W]

        return mask
