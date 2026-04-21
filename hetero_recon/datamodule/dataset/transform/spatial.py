import numpy as np
import torch

from cellverse.cryoem.ctf import (
    CTFCorrectionMode,
    CTFCorrector,
    CTFParameters,
    ImageParameters,
)
from cellverse.cryoem.preprocess import ContrastNormalizer, ZScoreStandardizer
from cellverse.image_utils.resize import crop_pad_center
from cellverse.math_utils import ht_object
from cellverse.math_utils.frequency import circularize, uncircularize
from cellverse.math_utils.grid import get_frequency_grid
from cellverse.math_utils.helper import to_2tuple
from cellverse.math_utils.number import get_next_odd

__all__ = ["DracoSpatialTransform"]


class DracoSpatialTransform(object):

    def __init__(self, input_is_light_particles: bool) -> None:
        """Initializes the spatial domain transform pipeline.

        Args:
            input_is_light_particles (bool): Controls the contrast of the final output.
                If True, the CTF corrector ensures that the output images have dark particles on a light
                background, which is a standard convention.
        """
        super().__init__()

        self.ctf_corrector = CTFCorrector(
            mode=CTFCorrectionMode.PHASE_FLIPPING,
            uninvert_data=not input_is_light_particles,
        )
        self.contrast_normalizer = ContrastNormalizer()
        self.z_score_standardizer = ZScoreStandardizer()

    def __call__(
        self,
        images: np.ndarray | torch.Tensor,
        image_params: ImageParameters,
        ctf_params: CTFParameters,
        trans: torch.Tensor = None,
    ) -> torch.Tensor:
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images).float()
        elif isinstance(images, list):
            images = torch.stack([torch.from_numpy(img.get()).float() for img in images])

        original_shape = images.shape
        if images.ndim == 2:
            images = images.unsqueeze(0) # Add batch dimension if missing
        assert images.ndim == 3, "Input images must be 2D or 3D (batch of 2D images)"

        ctf_filter = self.ctf_corrector.get_ctf_filter(
            image_params=image_params,
            ctf_params=ctf_params,
        )

        target_size = get_next_odd(image_params.image_size)
        target_shape = to_2tuple(target_size)

        images_ht = ht_object.ht2_center(circularize(images, last_n_dims=2))
        images_ht = crop_pad_center(images_ht, target_shape)

        if trans is not None:
            images_ht = ht_object.translate(images_ht, trans * target_size, get_frequency_grid(target_shape, device=images_ht.device))

        images_ht = images_ht * ctf_filter

        processed_images = uncircularize(ht_object.iht2_center(images_ht), last_n_dims=2)

        processed_images, _ = self.contrast_normalizer(processed_images)

        processed_images, _ = self.z_score_standardizer(processed_images)

        return processed_images.reshape(original_shape[:-2] + processed_images.shape[-2 :])
