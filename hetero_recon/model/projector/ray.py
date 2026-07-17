from collections.abc import Callable

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from cellverse.math_utils.grid import get_spatial_grid

__all__ = ["RealSpaceRayProjector"]


class RealSpaceRayProjector(nn.Module):
    """Project a continuous real-space density field along rotated viewing rays.

    The projector samples the density on the spatial grid used before CryoUNI's
    circular padding. Samples outside the unit reconstruction sphere are skipped,
    and rays are processed in chunks so a future fused implementation can retain
    the same public interface.
    """

    def __init__(self, spatial_image_size: int, ray_chunk_size: int, checkpoint_density: bool = False) -> None:
        super().__init__()

        if spatial_image_size < 3:
            raise ValueError(f"spatial_image_size must be at least 3, got {spatial_image_size}")
        if ray_chunk_size < 1:
            raise ValueError(f"ray_chunk_size must be positive, got {ray_chunk_size}")

        self.spatial_image_size = spatial_image_size
        self.ray_chunk_size = ray_chunk_size
        self.checkpoint_density = checkpoint_density

        xy_grid = get_spatial_grid((spatial_image_size, spatial_image_size)).reshape(-1, 2)
        ray_mask = xy_grid.square().sum(dim=-1) <= 1.0

        self.register_buffer("ray_xy", xy_grid[ray_mask], persistent=False)
        self.register_buffer("ray_pixel_indices", ray_mask.nonzero(as_tuple=False).flatten(), persistent=False)
        self.register_buffer(
            "depth_samples",
            torch.linspace(-1.0, 1.0, spatial_image_size),
            persistent=False,
        )

    def forward(
        self,
        density_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        rotations: torch.Tensor,
        conformations: torch.Tensor,
    ) -> torch.Tensor:
        """Return real-space projections with shape ``[B, D, D]``.

        ``rotations`` follows the same convention as the Fourier decoder: camera
        coordinates are mapped into the canonical molecular frame with ``R.T``.
        """

        if rotations.ndim != 3 or rotations.shape[-2 :] != (3, 3):
            raise ValueError(f"rotations must have shape [B, 3, 3], got {tuple(rotations.shape)}")
        if conformations.ndim != 2 or conformations.shape[0] != rotations.shape[0]:
            raise ValueError(
                "conformations must have shape [B, C] with the same batch size as rotations, "
                f"got {tuple(conformations.shape)} and {tuple(rotations.shape)}"
            )

        batch_size = rotations.shape[0]
        spatial_size = self.spatial_image_size
        ray_values = []

        ray_xy = self.ray_xy.to(device=rotations.device, dtype=rotations.dtype)
        depth_samples = self.depth_samples.to(device=rotations.device, dtype=rotations.dtype)

        for start in range(0, ray_xy.shape[0], self.ray_chunk_size):
            xy = ray_xy[start : start + self.ray_chunk_size]
            ray_count = xy.shape[0]

            xy_expanded = xy[:, None, :].expand(-1, spatial_size, -1)
            z_expanded = depth_samples[None, :, None].expand(ray_count, -1, -1)
            camera_points = torch.cat([xy_expanded, z_expanded], dim=-1).reshape(-1, 3)

            valid_samples = camera_points.square().sum(dim=-1) <= 1.0
            camera_points = camera_points[valid_samples]
            canonical_points = torch.einsum("bji,nj->bni", rotations, camera_points)

            if self.checkpoint_density and self.training:
                densities = checkpoint(density_fn, canonical_points, conformations, use_reentrant=False)
            else:
                densities = density_fn(canonical_points, conformations)
            if densities.shape != canonical_points.shape[: 2]:
                raise ValueError(
                    "density_fn must return one scalar per query point, "
                    f"got {tuple(densities.shape)} for {tuple(canonical_points.shape)}"
                )

            sample_ray_indices = (valid_samples.nonzero(as_tuple=False).flatten() // spatial_size).expand(batch_size, -1)
            projected_rays = densities.new_zeros((batch_size, ray_count))
            # A discrete sum preserves the Hartley/Fourier projection-slice scaling.
            projected_rays = projected_rays.scatter_add(1, sample_ray_indices, densities)
            ray_values.append(projected_rays)

        projected_values = torch.cat(ray_values, dim=1)
        pixel_indices = self.ray_pixel_indices.to(rotations.device).expand(batch_size, -1)
        projection_flat = projected_values.new_zeros((batch_size, spatial_size * spatial_size))
        projection_flat = projection_flat.scatter(1, pixel_indices, projected_values)
        return projection_flat.reshape(batch_size, spatial_size, spatial_size)
