"""
PC Trajectory Sampling CLI: generate volumes along PCA-based trajectories in latent space.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import typer
from typing_extensions import Annotated

# Setup Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from cli.utils.common import encode_latent_vectors, setup_system
from hetero_recon.utils.io import  save_volume
from hetero_recon.utils.latent_analysis import LatentAnalysisPipeline
from hetero_recon.utils.plot_utils import get_nearest_point

app = typer.Typer(
    help="PC Trajectory Sampling: sample latent space along PCA dimensions and generate volumes.",
    add_completion=False,
)


@app.command()
def main(
    output_dir: Annotated[Path, typer.Option("--out", "-o", help="Output directory", resolve_path=True)],
    config_path: Annotated[Path, typer.Option("--config", "-c", help="Config YAML file", exists=True)],
    checkpoint_path: Annotated[Path, typer.Option("--ckpt", "-p", help="Model checkpoint file", exists=True)],
    images_path: Annotated[Path, typer.Option("--images", "-i", help="HDF5 images dataset", exists=True)],
    pc_dim: Annotated[str, typer.Option("--pc-dim", help="PC dimensions to sample along (e.g., 0 1)")] = "0",
    num_samples: Annotated[int, typer.Option("--num-samples", help="Number of points to sample along each dimension")] = 10,
    not_on_data: Annotated[bool, typer.Option("--not-on-data", help="Sample off the data manifold (otherwise snap to nearest data point)")] = False,
    traj_type: Annotated[str, typer.Option("--traj-type", help="Type of trajectory")] = "linear",
    spiral_loops: Annotated[float, typer.Option("--spiral-loops", help="Number of loops for spiral trajectory")] = 3.0,
    spiral_scale: Annotated[float, typer.Option("--spiral-scale", help="Scale factor for spiral/circle radius (1.0 = auto fit to 95% data range)")] = 1.0,
    noise_std: Annotated[float, typer.Option("--noise-std", help="Noise std for volume generation")] = 300.0,
) -> None:
    """Sample latent space along PCA trajectories and generate corresponding volumes.

    Performs:
    1. Encode particles to latent vectors (or load cached)
    2. Run PCA on latent vectors using LatentAnalysisPipeline
    3. Generate a trajectory in PCA space (linear, spiral, or circle)
    4. Optionally snap trajectory points to the nearest data point
    5. Generate volumes for each trajectory point
    6. Visualize the trajectory in PCA space
    """
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    pc_dim = [int(d) for d in pc_dim.split()]
    print(f"Trajectory type: {traj_type}, PC dims: {pc_dim}, Points: {num_samples}")
    print("=" * 80)

    # Setup output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model and dataset
    print("\n[1/6] Loading model and dataset...")
    model, dataset, _ = setup_system(
        config_path,
        checkpoint_path,
        images_path,
    )

    # Inference to get latent vectors (or load cache)
    print("\n[2/6] Encoding latent vectors...")
    all_z_path = output_dir.parent / "all_z.pkl"
    all_z = encode_latent_vectors(model, dataset, all_z_path)
    print(f"Latent vectors shape: {all_z.shape}")

    # Initialize LatentAnalysisPipeline and run PCA (full components)
    print("\n[3/6] Running PCA via LatentAnalysisPipeline...")
    pipeline = LatentAnalysisPipeline(all_z)
    # Run PCA retaining all components
    pc = pipeline.run_reduction(method="pca", n_components=all_z.shape[1])
    pca_model = pipeline.reduction_models["pca"]
    # If model is from cuML, ensure we can use inverse_transform with numpy arrays
    # (cuML PCA accepts numpy arrays as well)
    print(f"PCA explained variance ratio (first {len(pc_dim)}): {pca_model.explained_variance_ratio_[:len(pc_dim)]}")

    # Prepare save directory for trajectory
    if traj_type == "spiral":
        traj_name = f"spiral_dims_{pc_dim[0]}_{pc_dim[1]}_loops_{spiral_loops}_num_{num_samples}"
    elif traj_type == "circle":
        traj_name = f"circle_dims_{pc_dim[0]}_{pc_dim[1]}_scale_{spiral_scale}_num_{num_samples}"
    else:  # linear
        traj_name = f"linear_dims_{'_'.join(map(str, pc_dim))}_num_{num_samples}"
    save_dir = output_dir / traj_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Generate trajectory in PCA space
    print(f"\n[4/6] Generating {traj_type} trajectory with {num_samples} points...")
    z_dim = all_z.shape[-1]
    traj_pca = np.zeros((num_samples, z_dim))  # [N, Z_dim] initialized to 0

    if traj_type == "spiral":
        if len(pc_dim) != 2:
            raise ValueError("Spiral trajectory requires exactly 2 PC dimensions (e.g., --pc-dim 0 1)")
        dim_x, dim_y = pc_dim[0], pc_dim[1]

        # Auto-compute radius from data distribution (95% range)
        range_x = (np.percentile(pc[:, dim_x], 95) - np.percentile(pc[:, dim_x], 5)) / 2
        range_y = (np.percentile(pc[:, dim_y], 95) - np.percentile(pc[:, dim_y], 5)) / 2
        max_radius = min(range_x, range_y) * spiral_scale

        t = np.linspace(0, 1, num_samples, endpoint=False)
        theta = t * spiral_loops * 2 * np.pi
        radius = t * max_radius
        traj_pca[:, dim_x] = radius * np.cos(theta)
        traj_pca[:, dim_y] = radius * np.sin(theta)
        plot_x_dim, plot_y_dim = dim_x, dim_y

    elif traj_type == "circle":
        if len(pc_dim) != 2:
            raise ValueError("Circular trajectory requires exactly 2 PC dimensions (e.g., --pc-dim 0 1)")
        dim_x, dim_y = pc_dim[0], pc_dim[1]
        range_x = (np.percentile(pc[:, dim_x], 95) - np.percentile(pc[:, dim_x], 5)) / 2
        range_y = (np.percentile(pc[:, dim_y], 95) - np.percentile(pc[:, dim_y], 5)) / 2
        radius = min(range_x, range_y) * spiral_scale

        t = np.linspace(0, 1, num_samples, endpoint=False)
        theta = t * 2 * np.pi
        traj_pca[:, dim_x] = radius * np.cos(theta)
        traj_pca[:, dim_y] = radius * np.sin(theta)
        plot_x_dim, plot_y_dim = dim_x, dim_y

    else:  # linear
        for d in pc_dim:
            start, end = np.percentile(pc[:, d], [5, 95])
            traj_pca[:, d] = np.linspace(start, end, num_samples)
        # choose two dimensions for plotting
        plot_x_dim = pc_dim[0]
        plot_y_dim = pc_dim[1] if len(pc_dim) > 1 else (pc_dim[0] + 1 if pc_dim[0] + 1 < pc.shape[1] else 0)

    # Convert PCA coordinates back to latent space
    # (pca_model.inverse_transform works with numpy arrays)
    z_traj = pca_model.inverse_transform(traj_pca.astype(np.float32))

    # Optionally snap to nearest data point
    if not not_on_data:
        z_traj, _ = get_nearest_point(all_z, z_traj)
        print("Snapped trajectory points to nearest data points.")
    else:
        z_traj = z_traj.astype(np.float32)
        print("Using off-data manifold points.")

    # Plot trajectory in PCA space
    print("\n[5/6] Plotting trajectory...")
    fig, ax = plt.subplots(figsize=(10, 8))
    # Background particles
    ax.scatter(pc[:, plot_x_dim], pc[:, plot_y_dim], c='lightgray', s=5, alpha=0.3, rasterized=True)
    # Trajectory line
    ax.plot(traj_pca[:, plot_x_dim], traj_pca[:, plot_y_dim], c='red', linewidth=2, alpha=0.8, linestyle='--')
    # Trajectory points colored by order
    sc = ax.scatter(
        traj_pca[:, plot_x_dim], traj_pca[:, plot_y_dim],
        c=range(num_samples), cmap='jet', s=60, marker='*', edgecolors='k', zorder=10
    )
    ax.text(traj_pca[0, plot_x_dim], traj_pca[0, plot_y_dim], "START", fontsize=12, fontweight='bold', color='blue')
    ax.text(traj_pca[-1, plot_x_dim], traj_pca[-1, plot_y_dim], "END", fontsize=12, fontweight='bold', color='blue')
    ax.set_xlabel(f"PC {plot_x_dim + 1}")
    ax.set_ylabel(f"PC {plot_y_dim + 1}")
    ax.set_title(f"Trajectory: {traj_type.title()}")
    plt.colorbar(sc, label='Frame Index')
    ax.set_aspect('equal', adjustable='datalim')

    # Set limits based on data percentiles
    x_vmin, x_vmax = np.percentile(pc[:, plot_x_dim], [2, 98])
    y_vmin, y_vmax = np.percentile(pc[:, plot_y_dim], [2, 98])
    x_vmid = (x_vmin + x_vmax) / 2
    y_vmid = (y_vmin + y_vmax) / 2
    x_vrange = x_vmax - x_vmin
    y_vrange = y_vmax - y_vmin
    ax.set_xlim(x_vmid - x_vrange * 0.75, x_vmid + x_vrange * 0.75)
    ax.set_ylim(y_vmid - y_vrange * 0.75, y_vmid + y_vrange * 0.75)

    plot_path = save_dir / "pc_trajectory.png"
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"Saved trajectory plot to {plot_path}")

    # Generate volumes for each trajectory point
    print("\n[6/6] Generating volumes...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i, z in enumerate(tqdm(z_traj, desc="Volumes")):
        zval = torch.from_numpy(z).to(device)
        volume = model.volume.eval_volume(zval, noise_std=noise_std, radius=None)
        save_volume(save_dir / f"volume.{i:03d}.mrc", volume.detach().cpu().numpy(), dataset.psize_A)

    # Summary
    print("\n" + "=" * 80)
    print("PC TRAJECTORY SAMPLING COMPLETE")
    print("=" * 80)
    print(f"Generated {num_samples} volumes in: {save_dir}")
    print(f"Trajectory plot: {plot_path}")
    print("=" * 80)


if __name__ == "__main__":
    app()
