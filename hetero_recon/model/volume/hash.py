import math
from typing import Any

from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

from cellverse.math_utils import ht_object
from cellverse.math_utils.frequency import circularize
from cellverse.math_utils.grid import get_spatial_grid
from cellverse.math_utils.mask import get_radial_mask
from cellverse.math_utils.number import get_next_odd
from coach_pl.configuration import configurable

from .build import VOLUME_REGISTRY
from .volume import Volume
from hetero_recon.model.projector import RealSpaceRayProjector

__all__ = [
    "ConditionalHashGridVolume",
    "MultiResolutionHashEncoder",
]


class MultiResolutionHashEncoder(nn.Module):
    """Pure-PyTorch multi-resolution 3D hash-grid encoding.

    This implementation prioritizes correctness and a stable interface. The hash
    tables and conditional field can later be replaced by tiny-cuda-nn without
    changing the volume or projector APIs.
    """

    _HASH_PRIMES = (1, 2_654_435_761, 805_459_861)

    def __init__(
        self,
        num_levels: int,
        features_per_level: int,
        log2_hashmap_size: int,
        base_resolution: int,
        max_resolution: int,
    ) -> None:
        super().__init__()

        if num_levels < 1:
            raise ValueError(f"num_levels must be positive, got {num_levels}")
        if features_per_level < 1:
            raise ValueError(f"features_per_level must be positive, got {features_per_level}")
        if log2_hashmap_size < 1:
            raise ValueError(f"log2_hashmap_size must be positive, got {log2_hashmap_size}")
        if base_resolution < 1 or max_resolution < base_resolution:
            raise ValueError("expected 1 <= base_resolution <= max_resolution, "
                             f"got {base_resolution} and {max_resolution}")

        self.num_levels = num_levels
        self.features_per_level = features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.max_resolution = max_resolution

        if num_levels == 1:
            resolutions = [max_resolution]
        else:
            scale = math.exp(math.log(max_resolution / base_resolution) / (num_levels - 1))
            resolutions = [int(math.floor(base_resolution * scale ** level)) for level in range(num_levels)]
            resolutions[-1] = max_resolution

        max_table_size = 1 << log2_hashmap_size
        self.resolutions = tuple(resolutions)
        dense_levels = []
        self.tables = nn.ParameterList()
        for resolution in resolutions:
            dense_vertex_count = (resolution + 1) ** 3
            table_size = min(max_table_size, dense_vertex_count)
            dense_levels.append(table_size == dense_vertex_count)
            table = nn.Parameter(torch.empty(table_size, features_per_level))
            nn.init.uniform_(table, -1.0e-4, 1.0e-4)
            self.tables.append(table)
        self.dense_levels = tuple(dense_levels)

        corners = torch.tensor(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
            ],
            dtype=torch.long,
        )
        self.register_buffer("corner_offsets", corners, persistent=False)
        self.register_buffer("hash_primes", torch.tensor(self._HASH_PRIMES, dtype=torch.long), persistent=False)

    @property
    def output_dim(self) -> int:
        return self.num_levels * self.features_per_level

    def _hash(self, vertices: torch.Tensor, table_size: int) -> torch.Tensor:
        hashed = vertices[..., 0] * self.hash_primes[0]
        hashed = torch.bitwise_xor(hashed, vertices[..., 1] * self.hash_primes[1])
        hashed = torch.bitwise_xor(hashed, vertices[..., 2] * self.hash_primes[2])
        return torch.remainder(hashed, table_size)

    @staticmethod
    def _linearize(vertices: torch.Tensor, resolution: int) -> torch.Tensor:
        vertex_resolution = resolution + 1
        return (vertices[..., 0] + vertex_resolution * vertices[..., 1] + vertex_resolution * vertex_resolution * vertices[..., 2])

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.shape[-1] != 3:
            raise ValueError(f"coords must end in dimension 3, got {tuple(coords.shape)}")
        if not coords.is_floating_point():
            raise TypeError(f"coords must be floating point, got {coords.dtype}")

        coords_unit = ((coords + 1.0) * 0.5).clamp(0.0, 1.0)
        encoded_levels = []

        for resolution, dense_level, table in zip(self.resolutions, self.dense_levels, self.tables):
            scaled = coords_unit * resolution
            lower = torch.floor(scaled).long().clamp(0, resolution - 1)
            fraction = scaled - lower.to(scaled.dtype)

            vertices = lower[..., None, :] + self.corner_offsets
            if dense_level:
                indices = self._linearize(vertices, resolution)
            else:
                indices = self._hash(vertices, table.shape[0])
            corner_features = F.embedding(indices, table)

            corner_offsets = self.corner_offsets.to(fraction.dtype)
            weights = torch.where(
                corner_offsets.bool(),
                fraction[..., None, :],
                1.0 - fraction[..., None, :],
            ).prod(dim=-1)
            encoded = (corner_features * weights[..., None]).sum(dim=-2)
            encoded_levels.append(encoded)

        return torch.cat(encoded_levels, dim=-1)


class ConditionalHashField(nn.Module):
    """Latent-conditioned real-space scalar field with per-layer FiLM."""

    def __init__(
        self,
        encoder: MultiResolutionHashEncoder,
        conformation_channels: int,
        hidden_channels: int,
        depth: int,
    ) -> None:
        super().__init__()

        if depth < 1:
            raise ValueError(f"depth must be positive, got {depth}")

        self.encoder = encoder
        self.conformation_channels = conformation_channels
        self.hidden_channels = hidden_channels
        self.depth = depth

        layers = [nn.Linear(encoder.output_dim, hidden_channels)]
        layers.extend(nn.Linear(hidden_channels, hidden_channels) for _ in range(depth - 1))
        self.layers = nn.ModuleList(layers)
        self.film = nn.Linear(conformation_channels, depth * 2 * hidden_channels)
        self.output = nn.Linear(hidden_channels, 1)

        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.normal_(self.film.weight, std=1.0e-4)
        nn.init.zeros_(self.film.bias)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, coords: torch.Tensor, conformations: torch.Tensor) -> torch.Tensor:
        if coords.ndim != 3 or coords.shape[-1] != 3:
            raise ValueError(f"coords must have shape [B, N, 3], got {tuple(coords.shape)}")
        if conformations.ndim != 2 or conformations.shape[0] != coords.shape[0]:
            raise ValueError(
                "conformations must have shape [B, C] with the same batch size as coords, "
                f"got {tuple(conformations.shape)} and {tuple(coords.shape)}"
            )

        features = self.encoder(coords)
        modulation = self.film(conformations).reshape(
            conformations.shape[0],
            self.depth,
            2,
            self.hidden_channels,
        )

        hidden = features
        for layer_index, layer in enumerate(self.layers):
            hidden = layer(hidden)
            scale = 1.0 + modulation[:, layer_index, 0].unsqueeze(1)
            shift = modulation[:, layer_index, 1].unsqueeze(1)
            hidden = F.silu(hidden * scale + shift)

        return self.output(hidden).squeeze(-1)


@VOLUME_REGISTRY.register()
class ConditionalHashGridVolume(Volume):
    """Real-space conditional hash field rendered into Hartley-domain slices."""

    @configurable
    def __init__(
        self,
        conformation_channels: int,
        hartley_image_size: int,
        num_levels: int,
        features_per_level: int,
        log2_hashmap_size: int,
        base_resolution: int,
        hidden_channels: int,
        depth: int,
        ray_chunk_size: int,
        checkpoint_rays: bool,
        eval_chunk_size: int,
    ) -> None:
        super().__init__()

        self.hartley_image_size = get_next_odd(hartley_image_size)
        self.spatial_image_size = self.hartley_image_size - 1
        self.conformation_channels = conformation_channels
        self.eval_chunk_size = eval_chunk_size
        if eval_chunk_size < 1:
            raise ValueError(f"eval_chunk_size must be positive, got {eval_chunk_size}")

        encoder = MultiResolutionHashEncoder(
            num_levels=num_levels,
            features_per_level=features_per_level,
            log2_hashmap_size=log2_hashmap_size,
            base_resolution=base_resolution,
            max_resolution=self.spatial_image_size,
        )
        self.field = ConditionalHashField(
            encoder=encoder,
            conformation_channels=conformation_channels,
            hidden_channels=hidden_channels,
            depth=depth,
        )
        self.projector = RealSpaceRayProjector(
            spatial_image_size=self.spatial_image_size,
            ray_chunk_size=ray_chunk_size,
            checkpoint_density=checkpoint_rays,
        )

    @classmethod
    def from_config(cls, cfg: DictConfig) -> dict[str, Any]:
        return {
            "conformation_channels": cfg.MODEL.HEAD.Z_CHANNELS,
            "hartley_image_size": cfg.DATAMODULE.DATASET.IMAGE_SIZE.HARTLEY,
            "num_levels": cfg.MODEL.VOLUME.NUM_LEVELS,
            "features_per_level": cfg.MODEL.VOLUME.FEATURES_PER_LEVEL,
            "log2_hashmap_size": cfg.MODEL.VOLUME.LOG2_HASHMAP_SIZE,
            "base_resolution": cfg.MODEL.VOLUME.BASE_RESOLUTION,
            "hidden_channels": cfg.MODEL.VOLUME.HIDDEN_CHANNELS,
            "depth": cfg.MODEL.VOLUME.DEPTH,
            "ray_chunk_size": cfg.MODEL.VOLUME.RAY_CHUNK_SIZE,
            "checkpoint_rays": cfg.MODEL.VOLUME.CHECKPOINT_RAYS,
            "eval_chunk_size": cfg.MODEL.VOLUME.EVAL_CHUNK_SIZE,
        }

    def forward(
        self,
        rotations: torch.Tensor,
        conformations: torch.Tensor,
        radius: int = None,
    ) -> torch.Tensor:
        radius = radius if radius is not None else self.hartley_image_size // 2
        projection_sp = self.projector(self.field, rotations, conformations)
        projection_ht = ht_object.ht2_center(circularize(projection_sp, last_n_dims=2))

        mask = get_radial_mask(
            (self.hartley_image_size, self.hartley_image_size),
            inner_radius=radius,
            device=projection_ht.device,
        ).reshape(-1)
        return projection_ht.flatten(1)[:, mask]

    @torch.inference_mode()
    def eval_volume(
        self,
        conformations: torch.Tensor,
        noise_std: float,
        radius: int = None,
    ) -> torch.Tensor:
        del radius # The field is already restricted to the real-space reconstruction box.

        if conformations.ndim == 1:
            conformations = conformations.unsqueeze(0)
        if conformations.ndim != 2:
            raise ValueError(f"conformations must have shape [C] or [B, C], got {tuple(conformations.shape)}")

        spatial_size = self.spatial_image_size
        coords_3d = get_spatial_grid(
            (spatial_size, spatial_size, spatial_size),
            device=conformations.device,
        )
        sphere_mask = get_radial_mask(
            (spatial_size, spatial_size, spatial_size),
            inner_radius=spatial_size // 2,
            device=conformations.device,
        )
        coords = coords_3d[sphere_mask]

        density_chunks = []
        for start in range(0, coords.shape[0], self.eval_chunk_size):
            chunk = coords[start : start + self.eval_chunk_size]
            chunk = chunk.unsqueeze(0).expand(conformations.shape[0], -1, -1)
            density_chunks.append(self.field(chunk, conformations))

        densities = torch.cat(density_chunks, dim=1)
        volumes = densities.new_zeros((conformations.shape[0], spatial_size, spatial_size, spatial_size))
        volumes[:, sphere_mask] = densities
        volumes = volumes * noise_std
        return volumes[0] if volumes.shape[0] == 1 else volumes
