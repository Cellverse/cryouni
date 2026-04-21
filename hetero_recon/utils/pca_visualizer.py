"""PCA feature map visualizer for backbone features."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
import torch
import torch.nn as nn


def get_rotated_flipped_image(
    image: torch.Tensor,
    rotation_index: int,
    flipping_index: int,
    inverse: bool = False,
    dims: tuple[int, int] = (-2, -1),
) -> torch.Tensor:
    """90-degree rotate and flip a tensor."""
    assert rotation_index in range(4)
    assert flipping_index in range(2)
    if inverse:
        image = torch.rot90(image, k=-rotation_index, dims=dims)
        if flipping_index == 1:
            image = image.flip(dims=dims[-1 :])
    else:
        if flipping_index == 1:
            image = image.flip(dims=dims[-1 :])
        image = torch.rot90(image, k=rotation_index, dims=dims)
    return image


class FeatureExtractor:
    """Extract patch features from a backbone, with optional 8-fold augmentation."""

    def __init__(self, backbone: nn.Module, augment: bool) -> None:
        self.backbone = backbone
        self.augment = augment

    @torch.inference_mode()
    def _get_features(self, images: torch.Tensor) -> torch.Tensor:
        features_dict = self.backbone(images)
        H = features_dict["patch_tokens_h"]
        W = features_dict["patch_tokens_w"]
        patch_features = features_dict["x_norm_patchtokens"]
        return patch_features.reshape(patch_features.size(0), H, W, -1) # [B, H, W, E]

    @torch.inference_mode()
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        if self.augment:
            features = torch.stack(
                [
                    get_rotated_flipped_image(
                        self._get_features(get_rotated_flipped_image(images, r, f)),
                        r,
                        f,
                        inverse=True,
                        dims=(-3, -2),
                    ) for f in range(2) for r in range(4)
                ],
                dim=0,
            ).mean(dim=0)
        else:
            features = self._get_features(images)
        return features


class PCAVisualizer:
    """Visualize backbone patch features via PCA into RGB maps."""

    N_COMPONENTS = 3

    def __init__(self, backbone: nn.Module, augment: bool = False) -> None:
        self.patch_size: tuple[int, int] = backbone.patch_embed.patch_size
        self.pca = PCA(n_components=self.N_COMPONENTS)
        self.feature_extractor = FeatureExtractor(backbone, augment)

    @torch.inference_mode()
    def __call__(self, x: torch.Tensor) -> list[dict[str, np.ndarray]]:
        """
        Args:
            `x`: Input images [B, C, H, W].

        Returns:
            List of dicts per image with keys:
                "Origin Image", "PCA Features",
        """
        patch_tokens = self.feature_extractor(x) # [B, H, W, E]
        B, H, W, _ = patch_tokens.shape

        flat = patch_tokens.reshape(B * H * W, -1).detach().cpu().numpy()
        pca_features = self.pca.fit_transform(flat) # [B*H*W, 3]

        pca_features = pca_features.reshape(B, H, W, -1)
        vmin = pca_features.min(axis=(1, 2), keepdims=True)
        vmax = pca_features.max(axis=(1, 2), keepdims=True)
        pca_features = (pca_features - vmin) / (vmax - vmin + 1e-8)

        results = []
        for i in range(B):
            image = x[i].detach().cpu().numpy().transpose(1, 2, 0) # [h, w, C]
            result = {
                "Origin Image": image,
                "PCA Features": pca_features[i],
            }
            results.append(result)

        return results
