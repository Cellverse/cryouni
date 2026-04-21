"""
This module provides functions to preprocess cryo-EM images.

Currently, the following normalization methods are supported:
    - Contrast normalization
    - Z-score standardization
"""

from __future__ import annotations

from abc import (
    ABCMeta,
    abstractmethod,
)

from einops import rearrange
import torch

__all__ = [
    "ContrastNormalizer",
    "ZScoreStandardizer",
]


class _Normalizer(metaclass=ABCMeta):
    """
    Abstract base class for batch-wise cryo-EM image normalization.
    """

    @abstractmethod
    def compute_stats(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Compute normalization statistics.

        Args:
            `images` (torch.Tensor): Input images of shape [..., H, W].

        Returns:
            (dict[str, torch.Tensor]): Computed statistics.
        """
        raise NotImplementedError

    @abstractmethod
    def apply_stats(self, images: torch.Tensor, stats: dict[str, torch.Tensor], inplace: bool = False) -> torch.Tensor:
        """
        Apply normalization statistics to images.

        Args:
            `images` (torch.Tensor): Input images of shape [..., H, W].
            `stats` (dict[str, torch.Tensor]): Pre-computed statistics.
            `inplace` (bool): Whether to apply normalization in-place (default: False).

        Returns:
            (torch.Tensor): Normalized images of the same shape as input.
        """
        raise NotImplementedError

    def __call__(self, images: torch.Tensor, inplace: bool = False) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Apply normalization statistics to images and return both normalized images and statistics.

        Args:
            `images` (torch.Tensor): Input images of shape [..., H, W].
            `inplace` (bool): Whether to apply normalization in-place (default: False).

        Returns:
            (tuple[torch.Tensor, dict[str, torch.Tensor]]):
                - Normalized images of the same shape as input.
                - Computed statistics.
        """
        stats = self.compute_stats(images)
        normalized = self.apply_stats(images, stats, inplace)
        return normalized, stats


class ContrastNormalizer(_Normalizer):
    """
    Contrast normalizer for cryo-EM images, which restricts the contrast of an image to a certain range.

    Args:
        `lower_quantile` (float): Lower quantile of a cryo-EM image for contrast estimation (default: 0.02).
        `upper_quantile` (float): Upper quantile of a cryo-EM image for contrast estimation (default: 0.98).
        `patch_size` (int): Patch size for contrast estimation (default: 128).
        `expansion` (float): Expansion factor for contrast normalization (default: 1.5).

    Example:
        >>> contrast_normalizer = ContrastNormalizer()
        >>> stats = contrast_normalizer.compute_stats(images)
        >>> normalized = contrast_normalizer.apply_stats(images, stats)
        >>>
        >>> # Equivalently:
        >>>
        >>> normalized, stats = contrast_normalizer(images)
    """

    def __init__(
        self,
        lower_quantile: float = 0.02,
        upper_quantile: float = 0.98,
        patch_size: int = 128,
        expansion: float = 1.5,
    ) -> None:
        super().__init__()

        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.patch_size = patch_size
        self.expansion = expansion

    def compute_stats(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Compute contrast normalization statistics.

        Args:
            `images` (torch.Tensor): Input images of shape [..., H, W].

        Returns:
            (dict[str, torch.Tensor]): Computed statistics:
                - `v_min`: Minimum value for contrast normalization.
                - `v_max`: Maximum value for contrast normalization.
        """
        *batch_dims, h, w = images.shape
        p = self.patch_size

        h_patch, w_patch = h // p, w // p
        h_cut, w_cut = h_patch * p, w_patch * p
        h_add, w_add = 1 if h_cut < h else 0, 1 if w_cut < w else 0
        quantiles = images.new_full((*batch_dims, 2, h_patch + h_add, w_patch + w_add), fill_value=torch.nan)

        # Letf top
        if h_cut > 0 and w_cut > 0:
            left_top = images[..., : h_cut, : w_cut]
            left_top_flat = rearrange(left_top, "... (nh p1) (nw p2) -> ... nh nw (p1 p2)", p1=p, p2=p)
            quantiles[..., 0, : h_patch, : w_patch] = left_top_flat.quantile(self.upper_quantile, dim=-1)
            quantiles[..., 1, : h_patch, : w_patch] = left_top_flat.quantile(self.lower_quantile, dim=-1)

        # Right top
        if h_cut > 0 and w_cut < w:
            right_top = images[..., : h_cut, w_cut :]
            right_top_flat = rearrange(right_top, "... (nh p) w -> ... nh (p w)", p=p)
            quantiles[..., 0, : h_patch, -1] = right_top_flat.quantile(self.upper_quantile, dim=-1)
            quantiles[..., 1, : h_patch, -1] = right_top_flat.quantile(self.lower_quantile, dim=-1)

        # Left bottom
        if h_cut < h and w_cut > 0:
            left_bottom = images[..., h_cut :, : w_cut]
            left_bottom_flat = rearrange(left_bottom, "... h (nw p) -> ... nw (h p)", p=p)
            quantiles[..., 0, -1, : w_patch] = left_bottom_flat.quantile(self.upper_quantile, dim=-1)
            quantiles[..., 1, -1, : w_patch] = left_bottom_flat.quantile(self.lower_quantile, dim=-1)

        # Right bottom
        if h_cut < h and w_cut < w:
            right_bottom = images[..., h_cut :, w_cut :]
            right_bottom_flat = rearrange(right_bottom, "... h w -> ... (h w)")
            quantiles[..., 0, -1, -1] = right_bottom_flat.quantile(self.upper_quantile, dim=-1)
            quantiles[..., 1, -1, -1] = right_bottom_flat.quantile(self.lower_quantile, dim=-1)

        p_upper, p_lower = quantiles.flatten(-2).nanquantile(0.5, dim=-1).unbind(-1)

        v_middle = (p_lower + p_upper) * 0.5
        v_range = (p_upper - p_lower) * 0.5

        return {
            "v_min": (v_middle - self.expansion * v_range).unsqueeze(-1).unsqueeze(-1),
            "v_max": (v_middle + self.expansion * v_range).unsqueeze(-1).unsqueeze(-1),
        }

    def apply_stats(self, images: torch.Tensor, stats: dict[str, torch.Tensor], inplace: bool = False) -> torch.Tensor:
        """
        Apply contrast normalization to images.

        Args:
            `images` (torch.Tensor): Input images of shape [..., H, W].
            `stats` (dict[str, torch.Tensor]):
                - `v_min`: Minimum value for contrast normalization.
                - `v_max`: Maximum value for contrast normalization.
            `inplace` (bool): Whether to apply normalization in-place (default: False).

        Returns:
            (torch.Tensor): Contrast normalized images of the same shape as input.
        """
        if inplace:
            images.clamp_(min=stats["v_min"], max=stats["v_max"])
        else:
            images = images.clamp(min=stats["v_min"], max=stats["v_max"])
        return images


class ZScoreStandardizer(_Normalizer):
    """
    Z-score normalizer for images, which standardizes the image to have a mean of 0 and a standard deviation of 1.

    Example:
        >>> z_score_normalizer = ZScoreStandardizer()
        >>> stats = z_score_normalizer.compute_stats(images)
        >>> standardized = z_score_normalizer.apply_stats(images, stats)
        >>>
        >>> # Equivalently:
        >>>
        >>> standardized, stats = z_score_normalizer(images)
    """

    def compute_stats(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Compute Z-score standardized statistics.

        Args:
            `images` (torch.Tensor): Input images of shape [..., H, W].

        Returns:
            (dict[str, torch.Tensor]): Computed statistics.
        """
        # Z-score standardization is to make the image have a mean of 0 and a standard deviation of 1.
        # Each pixel is not considered as a sample, so the unbiased flag is set to False.
        return {
            "mean": torch.mean(images, dim=(-2, -1), keepdim=True),
            "std": torch.std(images, dim=(-2, -1), unbiased=False, keepdim=True),
        }

    def apply_stats(self, images: torch.Tensor, stats: dict[str, torch.Tensor], inplace: bool = False) -> torch.Tensor:
        """
        Apply Z-score standardized to images.

        Args:
            `images` (torch.Tensor): Input images of shape [..., H, W].
            `stats` (dict[str, torch.Tensor]):
                - `mean`: Mean of images.
                - `std`: Standard deviation of images.
            `inplace` (bool): Whether to apply normalization in-place (default: False).

        Returns:
            (torch.Tensor): Z-score standardized images of the same shape as input.
        """
        if inplace:
            images.sub_(stats["mean"]).div_(stats["std"])
        else:
            images = (images - stats["mean"]) / stats["std"]
        return images
