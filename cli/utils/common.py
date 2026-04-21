"""
Common utilities shared across CLI tools.

This module provides shared functionality for model loading, dataset creation,
and latent vector encoding used by all CLI analysis tools.
"""

from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from cellverse.cryoem.ctf import ImageParameters
from cellverse.math_utils.number import get_next_odd
from coach_pl.configuration import CfgNode
from coach_pl.model import build_model

from hetero_recon.datamodule.dataset.transform import DracoSpatialTransform
from hetero_recon.utils.io import read_h5_particle


class HDF5Dataset(Dataset):
    """HDF5 particle dataset for inference."""

    def __init__(self, particle_path: str, image_size: int, is_light: bool, hartley_image_size: int):
        super().__init__()
        self.raw_particles, self.rots, self.trans, self.raw_image_params, self.ctf_params = read_h5_particle(particle_path, lazy=True)

        target_size_odd = get_next_odd(image_size)
        hartley_size_odd = get_next_odd(hartley_image_size)

        pixel_size = self.raw_image_params.pixel_size_angstrom
        raw_size = self.raw_image_params.image_size

        self.spatial_image_params = ImageParameters(target_size_odd, pixel_size * (raw_size - 1) / (target_size_odd - 1))
        self.spatial_transform = DracoSpatialTransform(is_light)
        self.psize_A = pixel_size * (raw_size - 1) / (hartley_size_odd - 1)

    def __len__(self):
        return len(self.raw_particles)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        slc = slice(idx, idx + 1)
        img_real = self.spatial_transform(self.raw_particles[slc], self.spatial_image_params, self.ctf_params[slc], self.trans[slc])[0]
        return {
            'y_real': img_real,
            'indices': idx,
        }


def check_paths_exist(paths: list[Path]):
    """Ensure all paths in the list exist."""
    for p in paths:
        if p is None:
            raise ValueError("A required path argument is None.")
        if not p.exists():
            raise FileNotFoundError(f"File/Directory does not exist: {p}")


def setup_system(
    config_path: Path,
    checkpoint_path: Path,
    images_path: Path,
    is_dark: bool = False,
) -> tuple[Any, HDF5Dataset, Any]:
    """
    Setup model and dataset with standard configuration.

    Args:
        config_path: Path to model config YAML file.
        checkpoint_path: Path to model checkpoint file.
        images_path: Path to HDF5 images dataset.
        is_dark: Whether images are dark-on-light.

    Returns:
        Tuple of (model, dataset, cfg).
    """
    # Load Config
    cfg = CfgNode.load_yaml_with_base(config_path)
    cfg.MODEL.BACKBONE.PRETRAINED_PATH = None
    cfg.DATAMODULE.DATASET.PARTICLE_PATH = images_path
    CfgNode.set_readonly(cfg, True)
    print(f"Config loaded from {config_path}")

    # Load Model
    model = build_model(cfg)
    model.load_pretrained(checkpoint_path)
    model.eval()
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Model loaded from {checkpoint_path}")

    # Load Dataset
    dataset = HDF5Dataset(
        particle_path=images_path,
        image_size=cfg.DATAMODULE.DATASET.IMAGE_SIZE.SPATIAL,
        is_light=not is_dark,
        hartley_image_size=cfg.DATAMODULE.DATASET.IMAGE_SIZE.HARTLEY,
    )
    print(f"Dataset loaded from {images_path}")

    return model, dataset, cfg


def encode_latent_vectors(
    model,
    dataset,
    cache_path: Path,
    batch_size: int = 64,
    num_workers: int = 10,
) -> np.ndarray:
    """
    Encode images to latent vectors with caching.

    Runs model inference to encode all images in the dataset to latent space.
    Results are cached to disk to avoid recomputation on subsequent runs.

    Args:
        `model`: Trained model with encoder.
        `dataset`: Image dataset.
        `cache_path` (Path): Path to cache file for storing/loading latent vectors.
        `batch_size` (int): Batch size for encoding.
        `num_workers` (int): Number of data loading workers.

    Returns:
        `all_z` (np.ndarray): Encoded latent vectors of shape [N, latent_dim].
    """
    if cache_path.exists():
        print(f"Loading cached latent vectors from {cache_path.name}...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print("Running inference to encode images...")
    all_z = []
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    for batch_data in tqdm(dataloader, desc="Encoding"):
        with torch.no_grad():
            pred = model(
                y_real=batch_data["y_real"].to(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
                radius=None,
                rotations=None,
                translations=None,
                encode_only=True,
            )
            all_z.append(pred["z_mu"].detach().cpu().numpy())

    all_z = np.concatenate(all_z, axis=0)

    print(f"Caching latent vectors to {cache_path.name}...")
    with open(cache_path, "wb") as f:
        pickle.dump(all_z, f)

    return all_z
