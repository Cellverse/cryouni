from pathlib import Path
from typing import Any

from omegaconf import DictConfig
from pytorch_lightning.trainer.states import RunningStage
import torch
from torch.utils.data import Dataset

from cellverse.cryoem.ctf import ImageParameters
from cellverse.math_utils.number import get_next_odd, get_prev_even
from coach_pl.configuration import configurable
from coach_pl.datamodule import DATASET_REGISTRY
from coach_pl.utils.logging import setup_logger

from .transform import (
    DracoSpatialTransform,
    ReconstructionHartleyTransform,
)
from hetero_recon.utils.io import (
    read_cryodrgn_particle,
    read_h5_particle,
    read_pkl_ctf,
    read_pkl_indices,
    read_pkl_poses,
)

logger = setup_logger(__name__, rank_zero_only=True)

__all__ = ["HeterogeneousReconstructionDataset"]


@DATASET_REGISTRY.register()
class HeterogeneousReconstructionDataset(Dataset):

    @configurable
    def __init__(
        self,
        hartley_image_size: int,
        spatial_image_size: int,
        input_is_light_particles: bool,
        particle_path: Path,
        particle_dir: Path | None,
        poses_path: Path | None,
        ctf_path: Path | None,
        indices_path: Path | None,
        dataset_len: int,
        is_training: bool,
        lazy_mode: bool,
        centerlize_spatial_image: bool,
        ignore_gt_poses: bool,
    ) -> None:
        super().__init__()

        self.lazy_mode = lazy_mode
        self.centerlize_spatial_image = centerlize_spatial_image
        self.ignore_gt_poses = ignore_gt_poses

        # Step 1: Load all raw data from disk
        self._load_raw_data(particle_path, particle_dir, poses_path, ctf_path)

        # Step 2: Set up image parameters and transformations
        self._setup_parameters_and_transforms(hartley_image_size, spatial_image_size, input_is_light_particles)

        # Step 3: Select particles based on indices and desired number
        self._filter_indices(indices_path, dataset_len, is_training)

        # Step 4: If not in lazy mode, preprocess all data and store in memory
        if not self.lazy_mode:
            self._preprocess_eager_data()

        # Step 5: Estimate noise from data
        self._estimate_statistics()

    @classmethod
    def from_config(cls, cfg: DictConfig, stage: RunningStage) -> dict[str, Any]:
        return {
            "hartley_image_size": cfg.DATAMODULE.DATASET.IMAGE_SIZE.HARTLEY,
            "spatial_image_size": cfg.DATAMODULE.DATASET.IMAGE_SIZE.SPATIAL,
            "input_is_light_particles": cfg.DATAMODULE.DATASET.INPUT_IS_LIGHT_PARTICLES,
            "particle_path": cfg.DATAMODULE.DATASET.PARTICLE_PATH,
            "particle_dir": cfg.DATAMODULE.DATASET.PARTICLE_DIR,
            "poses_path": cfg.DATAMODULE.DATASET.POSES_PATH,
            "ctf_path": cfg.DATAMODULE.DATASET.CTF_PATH,
            "indices_path": cfg.DATAMODULE.DATASET.INDICES_PATH,
            "dataset_len":
            cfg.DATAMODULE.DATASET.TRAIN_PARTICLES_NUM if stage == RunningStage.TRAINING else cfg.DATAMODULE.DATASET.VAL_PARTICLES_NUM,
            "is_training": stage == RunningStage.TRAINING,
            "lazy_mode": cfg.DATAMODULE.DATASET.LAZY_MODE,
            'centerlize_spatial_image': cfg.DATAMODULE.DATASET.CENTERLIZE_SPATIAL_IMAGE,
            "ignore_gt_poses": cfg.DATAMODULE.DATASET.IGNORE_GT_POSES,
        }

    def _load_raw_data(self, particle_path: Path, particle_dir: Path, poses_path: Path | None, ctf_path: Path | None):
        """Loads raw particle, CTF, and pose data from specified files."""
        particle_path = Path(particle_path)

        if particle_path.suffix == ".h5":
            self.raw_particles, self.rots, self.trans, self.raw_image_params, self.ctf_params = read_h5_particle(particle_path, lazy=self.lazy_mode)
        elif particle_path.suffix in ['.mrc', '.mrcs', ".txt", ".cs", '.star']:
            self.raw_particles = read_cryodrgn_particle(particle_path, particle_dir, lazy=self.lazy_mode)
            self.raw_image_params, self.ctf_params = read_pkl_ctf(ctf_path)
            self.rots, self.trans = None, None
        else:
            raise ValueError(f"Unsupported particle file format: {particle_path.suffix}")

        if self.ignore_gt_poses:
            self.rots = None
            self.trans = None

        if poses_path:
            if self.rots is not None:
                logger.warning("Overwriting poses from HDF5 file with those from the provided poses file.")
            self.rots, self.trans = read_pkl_poses(poses_path)

        if self.trans is None and self.centerlize_spatial_image:
            raise ValueError("Cannot centerlize spatial image without pose data.")

    def _setup_parameters_and_transforms(self, h_size: int, s_size: int, is_light: bool):
        """Initializes image parameters and data transformation pipelines."""
        h_size, s_size = get_next_odd(h_size), get_next_odd(s_size)
        raw_size = self.raw_image_params.image_size
        raw_psize = self.raw_image_params.pixel_size_angstrom

        self.hartley_image_params = ImageParameters(h_size, raw_psize * (raw_size - 1) / (h_size - 1))
        self.spatial_image_params = ImageParameters(s_size, raw_psize * (raw_size - 1) / (s_size - 1))

        self.hartley_transform = ReconstructionHartleyTransform(is_light)
        self.spatial_transform = DracoSpatialTransform(is_light)

    def _filter_indices(self, indices_path: Path | None, dataset_len: int, is_training: bool):
        """Determines the final set of particle indices to be used."""
        if indices_path:
            indices = torch.sort(read_pkl_indices(indices_path))[0]
        else:
            indices = torch.arange(len(self.raw_particles))
        self.indices = indices

        if dataset_len < 0:
            dataset_len = len(indices)
        elif dataset_len > len(indices) and not self.random_sample:
            logger.error(f"dataset_len is set to {dataset_len} as it exceeds the number of available particles.")
            dataset_len = len(indices)
        self.dataset_len = dataset_len

    def __len__(self):
        return self.dataset_len

    def _preprocess_eager_data(self):
        """Pre-processes all data and stores it in memory (for non-lazy mode)."""
        logger.info(f"Eager mode: Pre-processing all {len(self)} particles...")
        h_size = self.hartley_image_params.image_size
        s_size = self.spatial_image_params.image_size

        self.images_ht = torch.zeros((len(self), h_size, h_size), dtype=torch.half)
        self.images_real = torch.zeros((len(self), get_prev_even(s_size), get_prev_even(s_size)), dtype=torch.half)
        self.ctfs = torch.zeros((len(self), h_size, h_size), dtype=torch.half)

        for i, idx in enumerate(self.indices):
            img_ht, ctf = self.hartley_transform(self.raw_particles[idx:idx+1], self.hartley_image_params, self.ctf_params[idx:idx+1])
            img_real = self.spatial_transform(
                self.raw_particles[idx : idx + 1],
                self.spatial_image_params,
                self.ctf_params[idx : idx + 1],
                self.trans[idx : idx + 1] if self.centerlize_spatial_image else None
            )

            self.images_ht[i] = img_ht.half()
            self.images_real[i] = img_real.half()
            self.ctfs[i] = ctf.half()

            if (i + 1) % 5000 == 0:
                logger.info(f"Processed {i + 1} / {len(self)} particles.")
        logger.info("Eager mode pre-processing complete.")

    def _estimate_statistics(self, num_samples_for_lazy_est: int = 1000):
        """Estimates noise and amplitude spectrum from the dataset."""
        logger.info("Estimating noise variance and amplitude spectrum...")

        if self.lazy_mode:
            # For lazy mode, estimate from a random subset of data
            sample_indices = self.indices[torch.randperm(len(self))[: num_samples_for_lazy_est]]
            sampled_images_ht, sampled_ctf = [], []
            for idx in sample_indices:
                img_ht, ctf = self.hartley_transform(self.raw_particles[idx:idx+1], self.hartley_image_params, self.ctf_params[idx:idx+1])
                sampled_images_ht.append(img_ht[0])
                sampled_ctf.append(ctf[0])
            sampled_images_ht = torch.stack(sampled_images_ht)
            sampled_ctf = torch.stack(sampled_ctf)
        else:
            # For eager mode, use all pre-processed data
            sampled_images_ht = self.images_ht
            sampled_ctf = self.ctfs

        # Estimate noise standard deviation
        ht_std = torch.std(sampled_images_ht, unbiased=False)
        self.estimated_noise_std = ht_std
        self.estimated_noise_var = self.estimated_noise_std ** 2
        logger.info(f"Estimated noise std: {self.estimated_noise_std:.4f} (variance: {self.estimated_noise_var:.4f})")

    def __getitem__(self, idx: int) -> dict[str, Any]:
        original_index = self.indices[idx]

        if self.lazy_mode:
            img_ht, ctf = self.hartley_transform(self.raw_particles[original_index:original_index+1], self.hartley_image_params, self.ctf_params[original_index:original_index+1])
            img_real = self.spatial_transform(
                self.raw_particles[original_index : original_index + 1],
                self.spatial_image_params,
                self.ctf_params[original_index : original_index + 1],
                self.trans[original_index : original_index + 1] if self.centerlize_spatial_image else None
            )

            batch = {
                "y_real": img_real[0],
                "y": img_ht[0] / self.estimated_noise_std,
                "ctf": ctf[0],
                "indices": original_index.item(),
            }
        else:
            # For eager mode, use pre-processed data, which has already been permuted
            batch = {
                "y_real": self.images_real[idx].float(),
                "y": self.images_ht[idx].float() / self.estimated_noise_std,
                "ctf": self.ctfs[idx].float(),
                "indices": original_index.item(),
            }

        if self.rots is not None:
            batch["rots"] = self.rots[original_index]
        if self.trans is not None:
            batch["trans"] = self.trans[original_index]

        return batch

    @property
    def collate_fn(self) -> None:
        return None
