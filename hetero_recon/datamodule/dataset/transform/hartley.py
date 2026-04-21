import numpy as np
import torch

from cellverse.cryoem.ctf import (
    CTFCalculator,
    CTFParameters,
    ImageParameters,
)
from cellverse.image_utils.resize import crop_pad_center
from cellverse.math_utils import ht_object
from cellverse.math_utils.frequency import circularize
from cellverse.math_utils.mask import get_radial_mask
from cellverse.math_utils.number import get_next_odd

__all__ = ["ReconstructionHartleyTransform"]


class ReconstructionHartleyTransform(object):

    def __init__(self, input_is_light_particles: bool) -> None:
        """Initializes the spatial domain transform pipeline.

        Args:
            input_is_light_particles (bool): Controls the contrast of the final output.
                If True, the CTF corrector ensures that the
                output images have dark particles on a light
                background, which is a standard convention.
        """
        super().__init__()
        self.invert_contrast = input_is_light_particles

    def __call__(
        self,
        images: np.ndarray | torch.Tensor,
        image_params: ImageParameters,
        ctf_params: CTFParameters,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images).float()
        elif isinstance(images, list):
            images = torch.stack([torch.from_numpy(img.get()).float() for img in images])

        original_shape = images.shape
        if images.ndim == 2:
            images = images.unsqueeze(0) # Add batch dimension if missing
        assert images.ndim == 3, "Input images must be 2D or 3D (batch of 2D images)"

        mask = get_radial_mask(images.shape[-2 :], inner_radius=0.85, outer_radius=0.99, device=images.device)
        images = images * mask[None]

        images_ht = ht_object.ht2_center(circularize(images, last_n_dims=2))

        if self.invert_contrast:
            images_ht = -images_ht

        ctfs = CTFCalculator.get_ctf_2d(
            image_params=image_params,
            ctf_params=ctf_params,
        )

        target_size = get_next_odd(image_params.image_size)
        target_shape = (target_size, target_size)
        images_ht = crop_pad_center(images_ht, target_shape)
        ctfs = crop_pad_center(ctfs, target_shape)

        images_ht = images_ht.reshape(original_shape[:-2] + target_shape)
        ctfs = ctfs.reshape(original_shape[:-2] + target_shape)

        return images_ht, ctfs
