"""
Visualize PCA feature maps from the CryoUNI backbone for a set of particles.
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import typer
from typing_extensions import Annotated

from cli.utils.common import setup_system

from hetero_recon.utils.pca_visualizer import PCAVisualizer

app = typer.Typer(help="Visualize PCA feature maps from the backbone.", add_completion=False)


@app.command()
def main(
    config: Annotated[Path, typer.Option("--config", "-c", help="Config YAML file", exists=True)],
    ckpt: Annotated[Path, typer.Option("--ckpt", "-p", help="Model checkpoint", exists=True)],
    images: Annotated[Path, typer.Option("--images", "-i", help="HDF5 particle images", exists=True)],
    out: Annotated[Path, typer.Option("--out", "-o", help="Output directory", resolve_path=True)],
    num_images: Annotated[int, typer.Option("--num-images", "-n", help="Number of images to visualize")] = 9,
    start_idx: Annotated[int, typer.Option("--start-idx", help="Starting particle index")] = 0,
    augment: Annotated[bool, typer.Option("--augment/--no-augment", help="8-fold augmentation")] = True,
    num_cols: Annotated[int, typer.Option("--num-cols", help="Columns in output grid")] = 9,
    noise_std: Annotated[float, typer.Option("--noise-std")] = 300.0,
) -> None:
    out.mkdir(parents=True, exist_ok=True)

    model, dataset, cfg = setup_system(config, ckpt, images)
    backbone = model.model.backbone

    visualizer = PCAVisualizer(backbone, augment=augment)

    indices = list(range(start_idx, min(start_idx + num_images, len(dataset))))
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=num_images, num_workers=0)
    batch = next(iter(loader))

    inputs = batch["y_real"].unsqueeze(1).to(torch.device("cuda" if torch.cuda.is_available() else "cpu")) # [B, 1, H, W]
    results = visualizer(inputs)

    # Determine rows: Origin Image, PCA Features
    row_keys = ["Origin Image", "PCA Features"]

    n_rows = len(row_keys)
    n_cols = min(num_cols, len(results))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2), dpi=150, tight_layout=True)

    if n_rows == 1:
        axs = axs[np.newaxis, :]
    if n_cols == 1:
        axs = axs[:, np.newaxis]

    for col, result in enumerate(results[: n_cols]):
        for row, key in enumerate(row_keys):
            ax = axs[row, col]
            img = result[key]
            if img.ndim == 3 and img.shape[-1] == 1:
                img = img[..., 0]
            if img.ndim == 2 or (img.ndim == 3 and img.shape[-1] in (1, 3)):
                ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
            else:
                ax.imshow(img)
            ax.axis("off")
            if col == 0:
                ax.set_title(key, fontsize=6)

    save_path = out / "pca_feature_map.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    app()
