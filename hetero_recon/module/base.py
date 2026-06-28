from abc import ABCMeta, abstractmethod
from collections import defaultdict
from pathlib import Path
import pickle
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import LRSchedulerTypeUnion
from sklearn.decomposition import PCA
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
import torch
import torch.nn as nn

from cellverse.math_utils.frequency import ht_object, uncircularize
from cellverse.math_utils.mask import get_radial_mask
from cellverse.math_utils.number import get_next_odd
from coach_pl.configuration import configurable
from coach_pl.criterion import build_criterion
from coach_pl.model import build_model
from coach_pl.utils.logging import setup_logger

from hetero_recon.utils.io import save_volume
from hetero_recon.utils.plot_utils import (
    cluster_gmm,
    get_nearest_point,
    plot_pca_2d,
    plot_pca_explained_variance_ratio,
    plot_spatial_domain,
    plot_umap_2d,
    run_pca,
    run_umap,
)

logger = setup_logger(__name__, rank_zero_only=True)

__all__ = ["BaseModule"]


def _unique_latents_by_index(z: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = np.asarray(z)
    indices = np.asarray(indices).reshape(-1)
    if z.ndim != 2:
        raise ValueError("z must be a 2D latent array")
    if len(z) != len(indices):
        raise ValueError("z and indices must have the same first dimension")

    unique_indices, first_positions = np.unique(indices, return_index=True)
    order = np.argsort(unique_indices)
    unique_indices = unique_indices[order].astype(np.int64, copy=False)
    unique_z = z[first_positions[order]]
    return unique_indices, unique_z


def _save_latents_with_indices(output_dir: Path, z: np.ndarray, indices: np.ndarray) -> dict[str, np.ndarray]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    unique_indices, unique_z = _unique_latents_by_index(z, indices)

    with (output_dir / "all_z.pkl").open("wb") as f:
        pickle.dump(unique_z, f)
    with (output_dir / "all_z_indices.pkl").open("wb") as f:
        pickle.dump(unique_indices, f)
    with (output_dir / "all_z_with_indices.pkl").open("wb") as f:
        pickle.dump(
            {
                "latent_source": "z_mu",
                "z_mu": unique_z,
                "indices": unique_indices,
            },
            f,
        )
    return {"z": unique_z, "indices": unique_indices}


class BaseModule(LightningModule, metaclass=ABCMeta):
    """
    A project-specific base module that extends PyTorch Lightning's LightningModule.
    """

    @configurable
    def __init__(self, model: nn.Module, criterion: nn.Module, cfg: DictConfig) -> None:
        super().__init__()

        self.model = model
        self.criterion = criterion
        logger.info(model)

        self.save_hyperparameters(cfg)

        # Dataset & dataloader configuration
        self.batch_size = cfg.DATAMODULE.DATALOADER.TRAIN.BATCH_SIZE

        # Optimizer configuration
        self.base_lr = cfg.MODULE.OPTIMIZER.BASE_LR
        self.head_lr = cfg.MODULE.OPTIMIZER.BASE_LR * cfg.MODULE.OPTIMIZER.HEAD_LR_SCALE
        self.optimizer_name = cfg.MODULE.OPTIMIZER.NAME
        self.optimizer_params = {
            k.lower(): v
            for k, v in cfg.MODULE.OPTIMIZER.PARAMS.items()
        }

        # Scheduler configuration
        self.step_on_epochs = cfg.MODULE.SCHEDULER.STEP_ON_EPOCHS
        self.scheduler_name = cfg.MODULE.SCHEDULER.NAME
        self.scheduler_params = {
            k.lower(): v
            for k, v in cfg.MODULE.SCHEDULER.PARAMS.items()
        }
        self.scheduler_params["num_epochs"] = cfg.TRAINER.MAX_EPOCHS

        # PCA
        self.pca = PCA(n_components=3)

        # Dimension
        self.hartley_image_size = get_next_odd(cfg.DATAMODULE.DATASET.IMAGE_SIZE.HARTLEY)
        self.z_dim = cfg.MODEL.HEAD.Z_CHANNELS
        self.random_init_image_num = 0

    @classmethod
    def from_config(cls, cfg: DictConfig) -> dict[str, Any]:
        return {
            "model": build_model(cfg),
            "criterion": build_criterion(cfg),
            "cfg": cfg
        }

    def configure_optimizers(self) -> Any:

        hyperparameters = self.rescale_hyperparameters()

        # Define parameter groups with their prefixes and learning rates
        param_group_configs = {
            "head.": hyperparameters["head_lr"],
        }

        # Group parameters by prefix, default to base_lr
        param_groups = defaultdict(list)
        for name, param in self.model.named_parameters():
            lr = hyperparameters["lr"] # Default learning rate
            for prefix, group_lr in param_group_configs.items():
                if name.startswith(prefix):
                    lr = group_lr
                    break
            param_groups[lr].append(param)

        # Create optimizer groups
        optimizer_groups = [
            {
                "params": params,
                "lr": lr,
            } for lr, params in param_groups.items() if params # Only include non-empty groups
        ]

        optimizer = create_optimizer_v2(
            model_or_params=optimizer_groups,
            opt=self.optimizer_name,
            **self.optimizer_params,
        )

        if self.step_on_epochs:
            updates_per_epoch = 1
        else:
            updates_per_epoch = self.trainer.estimated_stepping_batches // self.scheduler_params["num_epochs"]

        scheduler, _ = create_scheduler_v2(
            optimizer=optimizer,
            sched=self.scheduler_name,
            step_on_epochs=self.step_on_epochs,
            updates_per_epoch=updates_per_epoch,
            **self.scheduler_params,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch" if self.step_on_epochs else "step",
                "frequency": 1,
            }
        }

    def lr_scheduler_step(self, scheduler: LRSchedulerTypeUnion, metric: Any | None) -> None:
        if self.step_on_epochs:
            scheduler.step(self.current_epoch, metric)
        else:
            scheduler.step_update(self.global_step, metric)

    def add(self, predictions: Any, batch: Any, sync: bool = False) -> None:
        if not self.trainer.is_global_zero and not sync:
            return

        def process_tensor(t):
            if sync:
                gathered = self.all_gather(t)
                return gathered.detach().cpu().numpy()
            else:
                return t.detach().cpu().numpy()

        self.result['z'].append(process_tensor(predictions['z_mu']).reshape(-1, self.z_dim))
        self.result['indices'].append(process_tensor(batch['indices']).reshape(-1))

    def on_train_start(self):
        self.tb_writer = self.trainer.logger.experiment
        self.exp_version = self.trainer.log_dir.split("/")[-1]
        self.root_dir = Path(self.trainer.default_root_dir)
        self.first_single_batch = None
        self.first_multiple_y_real = None

    def on_train_epoch_start(self):
        self.result = defaultdict(list)

    @abstractmethod
    def model_forward(self, batch: Any, encode_only: bool, return_z_only: bool) -> Any:
        raise NotImplementedError

    @abstractmethod
    def criterion_forward(self, predictions: Any, batch: Any) -> Any:
        raise NotImplementedError

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        if self.first_single_batch is None:
            # Only take the first data element of the first batch
            first_single_batch = {}
            for key, value in batch.items():
                first_single_batch[key] = value[0 : 1]
            self.first_single_batch = first_single_batch

            self.first_multiple_y_real = batch["y_real"][: 8]

        predictions = self.model_forward(batch, encode_only=False, return_z_only=False)
        loss_dict = self.criterion_forward(predictions, batch)

        # Logging
        for key, value in loss_dict.items():
            self.log(f"train/{key}", value, prog_bar=True, logger=True, sync_dist=True)
        self.log("train/radius", self.radius, prog_bar=True, logger=True, sync_dist=True)

        # Collect results
        self.add(predictions, batch, sync=False)

        return loss_dict["loss"]

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        if self.trainer.is_global_zero and self.time_to_eval:
            with torch.no_grad():
                first_single_predictions = self.model_forward(self.first_single_batch, encode_only=False, return_z_only=False)

            self.visualize_reconstruction(first_single_predictions, self.first_single_batch)
            self.visualize_pca_feature(self.first_multiple_y_real)

            z = np.concatenate(self.result["z"])
            indices = np.concatenate(self.result["indices"])
            self.result = defaultdict(list)
            self.result["z"] = [z]
            self.result["indices"] = [indices]

        if self.trainer.is_global_zero and self.trainer.is_last_batch:
            self.save_volume()
            self.result = defaultdict(list)

    def on_validation_epoch_start(self):
        self.result = defaultdict(list)

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        predictions = self.model_forward(batch, encode_only=True, return_z_only=self.current_epoch < self.trainer.max_epochs - 1)

        # Collect results
        self.add(predictions, batch, sync=True)

    def on_validation_epoch_end(self):
        if self.trainer.is_global_zero:
            self.visualize_conformation()

    def on_test_epoch_start(self):
        self.result = defaultdict(list)

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        predictions = self.model_forward(batch, encode_only=True, return_z_only=False)

        self.add(predictions, batch, sync=True)

    def on_test_epoch_end(self):
        if self.trainer.is_global_zero:
            self.visualize_conformation()
            self.save_conformation()

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        model = self.model
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        checkpoint["model"] = model.state_dict()

    def rescale_hyperparameters(self) -> dict[str, float]:
        """
        Rescale hyperparameters, e.g., learning rate, exponential moving average decay, etc.,
        based on the total batch size.
        """
        hyperparameters = {}

        total_batch_size = self.batch_size * self.trainer.accumulate_grad_batches * self.trainer.world_size
        self.total_batch_size = total_batch_size
        logger.info(
            f"Total training batch size ({total_batch_size}) = single batch size ({self.batch_size}) * accumulate ({self.trainer.accumulate_grad_batches}) * world size ({self.trainer.world_size})"
        )

        scale_factor = 32
        lr = self.base_lr * total_batch_size / scale_factor
        head_lr = self.head_lr * total_batch_size / scale_factor

        logger.info(f"Learning rate ({lr:0.6g}) = base_lr ({self.base_lr:0.6g}) * total_batch_size ({total_batch_size}) / {scale_factor}")

        hyperparameters.update({
            "lr": lr,
            "head_lr": head_lr,
        })
        return hyperparameters

    @property
    def radius(self):
        return self.hartley_image_size // 2

    @property
    def noise_std(self) -> float:
        return self.trainer.datamodule.train_dataset.estimated_noise_std

    @property
    def psize_A(self) -> float:
        return self.trainer.datamodule.train_dataset.hartley_image_params.pixel_size_angstrom

    @torch.inference_mode()
    def visualize_reconstruction(self, predictions: Any, batch: Any) -> None:
        D = self.hartley_image_size
        device = batch["y"].device

        # Ground Truth
        y = batch["y"][0]
        y_real = uncircularize(ht_object.iht2_center(y), last_n_dims=2).detach().cpu().numpy()

        # Prediction
        y_hat = torch.zeros((D * D), device=device)
        mask = get_radial_mask((D, D), inner_radius=self.radius, device=device).reshape(-1)
        y_hat[mask] = predictions["y_hat"][0].detach().flatten()
        y_hat = y_hat.reshape(D, D)
        y_hat_real = uncircularize(ht_object.iht2_center(y_hat), last_n_dims=2).detach().cpu().numpy()

        # Index
        index = batch["indices"].item()

        # Plot
        fig = plot_spatial_domain(y_real, y_hat_real, index)
        self.tb_writer.add_figure('train/Spatial Image', fig, global_step=self.global_step)
        plt.close(fig)

    @torch.inference_mode()
    def visualize_pca_feature(self, images: torch.Tensor) -> None:
        if hasattr(self.model, "backbone"):
            patch_tokens = self.model.backbone(images.unsqueeze(1))
            H = patch_tokens["patch_tokens_h"]
            W = patch_tokens["patch_tokens_w"]
            patch_tokens = patch_tokens["x_norm_patchtokens"].detach().flatten(0, -2)  # [B * L, E]
            pca_features = self.pca.fit_transform(patch_tokens.detach().cpu().numpy()) # [B * L, 3]
            pca_features = pca_features.reshape(-1, H, W, 3)

            # Images
            images = images.unsqueeze(-1).detach().cpu().numpy() # [B, H, W, 1]
            vmin = np.min(pca_features, axis=(1, 2), keepdims=True)
            vmax = np.max(pca_features, axis=(1, 2), keepdims=True)
            pca_features = (pca_features - vmin) / (vmax - vmin)

            # Plot
            fig, axs = plt.subplots(nrows=2, ncols=len(images), figsize=(len(images) * 5, 10), tight_layout=True)
            for image, ax in zip(images, axs[0]):
                ax.imshow(image, cmap="gray")
                ax.axis("off")
            for pca, ax in zip(pca_features, axs[1]):
                ax.imshow(pca)
                ax.axis("off")
            self.tb_writer.add_figure("Patch Token PCA", fig, global_step=self.global_step)
            plt.close(fig)

    @torch.inference_mode()
    def visualize_conformation(self) -> None:
        if "z" in self.result:
            z = np.concatenate(self.result["z"])

            # PCA & UMAP
            pc, pca = run_pca(z)
            umap = run_umap(z)
            cluster_num = 10 #! Hardcode
            gmm_labels, gmm_centers = cluster_gmm(z, cluster_num)
            _, gmm_centers_ind = get_nearest_point(z, gmm_centers)

            fig = plot_pca_2d(pc, cluster_num, gmm_labels, gmm_centers_ind)
            self.tb_writer.add_figure("val/conf_pca", fig, self.global_step)
            plt.close(fig)
            fig = plot_umap_2d(umap, cluster_num, gmm_labels, gmm_centers_ind)
            self.tb_writer.add_figure("val/conf_umap", fig, self.global_step)
            plt.close(fig)
            fig = plot_pca_explained_variance_ratio(pca.explained_variance_ratio_)
            self.tb_writer.add_figure("val/conf_pca_variance_ratio", fig, self.global_step)
            plt.close(fig)

            # GMM volume
            device = next(self.model.parameters()).device
            output_dir = self.root_dir / "gmm_volumes" / self.exp_version / f"epoch_{self.current_epoch}"
            output_dir.mkdir(parents=True, exist_ok=True)
            for i, center_z in enumerate(gmm_centers):
                conformation = torch.from_numpy(center_z).to(device)
                volume = self.model.volume.eval_volume(conformation, noise_std=self.noise_std, radius=self.radius)
                volume_path = output_dir / f"volume.{i:03d}.mrc"
                save_volume(volume_path, volume.detach().cpu().numpy(), self.psize_A)

    @torch.inference_mode()
    def save_conformation(self) -> None:
        if "z" in self.result:
            z = np.concatenate(self.result["z"])
            indices = np.concatenate(self.result["indices"])
            _save_latents_with_indices(self.root_dir, z, indices)

    @torch.inference_mode()
    def save_volume(self) -> None:
        if "z" in self.result:
            z = np.concatenate(self.result["z"])
            indices = np.concatenate(self.result["indices"])
            _, first_idx = np.unique(indices, return_index=True)
            z = z[first_idx]
            _, gmm_centers = cluster_gmm(z, 1)

            # GMM volume
            device = next(self.model.parameters()).device
            output_dir = self.root_dir / "center_vols" / self.exp_version
            output_dir.mkdir(parents=True, exist_ok=True)
            conformation = torch.from_numpy(gmm_centers[0]).to(device)
            volume = self.model.volume.eval_volume(conformation, noise_std=self.noise_std, radius=self.radius)
            volume_path = output_dir / f"volume.ep{self.current_epoch:03d}.mrc"
            save_volume(volume_path, volume.detach().cpu().numpy(), self.psize_A)

    @property
    def time_to_eval(self) -> bool:
        return self.global_step % 1000 == 0 or self.global_step == 1 or self.trainer.is_last_batch or (
            self.total_batch_size * self.global_step <= self.random_init_image_num and self.global_step % 100 == 0
        )
