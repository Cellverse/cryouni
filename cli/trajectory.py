"""
Trajectory Planning CLI.

Plans smooth trajectories through conformational space connecting density peaks.
Supports N-dimensional input with visualization for 2D/3D.
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from cli.utils.common import encode_latent_vectors, setup_system

from hetero_recon.utils.latent_analysis import LatentAnalysisPipeline
from hetero_recon.utils.peak_detector import PeakDetector
from hetero_recon.utils.plot_utils import (
    plot_energy_profile,
    plot_trajectory_2d,
    plot_trajectory_3d,
)
from hetero_recon.utils.trajectory_planner import TrajectoryPlanner

app = typer.Typer(
    help="Trajectory planning: generate smooth paths through conformational space.",
    add_completion=False,
)


def find_nearest_points(
    trajectory: np.ndarray,
    pc: np.ndarray,
    pc_dims: list[int],
) -> np.ndarray:
    """
    Find nearest actual points in PC space for trajectory coordinates.
    """
    n_components = pc.shape[1]
    final_pca_points = np.zeros((len(trajectory), n_components), dtype=np.float32)
    pc_proj = pc[:, pc_dims]

    for idx, traj_coord in enumerate(trajectory):
        distances = np.linalg.norm(pc_proj - traj_coord, axis=1)
        nearest_idx = np.argmin(distances)
        final_pca_points[idx] = pc[nearest_idx]

    return final_pca_points


def create_zero_padded_pc_coords(
    trajectory: np.ndarray,
    n_components: int,
    pc_dims: list[int],
) -> np.ndarray:
    """
    Create full PC coordinates by zero-padding other dimensions.
    """
    final_pca_points = np.zeros((len(trajectory), n_components), dtype=np.float32)
    for i, dim_idx in enumerate(pc_dims):
        final_pca_points[:, dim_idx] = trajectory[:, i]
    return final_pca_points


def generate_volumes_from_trajectory(
    trajectory: np.ndarray,
    pc_dims: list[int],
    pca,
    model,
    dataset,
    output_dir: Path,
    noise_std: float,
    pc: np.ndarray | None = None,
    use_nearest_point: bool = True,
) -> np.ndarray:
    """Generate MRC volumes along trajectory in reduced space.

    For each trajectory point, computes latent vector and generates 3D volume.

    Args:
        `trajectory` (np.ndarray): Trajectory points of shape [N, D].
        `pc_dims` (list[int]): Indices of PC dimensions used.
        `pca`: Fitted PCA model for inverse transform.
        `model`: Neural volume decoder.
        `dataset`: Dataset with pixel size info.
        `output_dir` (Path): Directory to save volumes.
        `noise_std` (float): Noise std for volume generation.
        `pc` (np.ndarray | None): Full PC array, required if use_nearest_point=True.
        `use_nearest_point` (bool): If True, find nearest actual points; if False, zero-pad.

    Returns:
        `z_traj` (np.ndarray): Full latent trajectory of shape [N, latent_dim].
    """
    from hetero_recon.utils.io import save_volume

    output_dir.mkdir(parents=True, exist_ok=True)

    # Choose coordinate generation method
    if use_nearest_point:
        if pc is None:
            raise ValueError("pc must be provided when use_nearest_point=True")
        final_traj_pca = find_nearest_points(trajectory, pc, pc_dims)
    else:
        final_traj_pca = create_zero_padded_pc_coords(trajectory, pca.n_components_, pc_dims)

    # Pad if needed for inverse transform
    if final_traj_pca.shape[1] < pca.n_components_:
        final_traj_pca = np.pad(final_traj_pca, ((0, 0), (0, pca.n_components_ - final_traj_pca.shape[1])))

    # Inverse transform to latent Z
    z_traj = pca.inverse_transform(final_traj_pca)
    np.save(output_dir / "trajectory_z.npy", z_traj)

    print(f"Generating {len(z_traj)} MRC volumes...")
    with torch.no_grad():
        for idx, z in enumerate(tqdm(z_traj, desc="Generating volumes")):
            z_gpu = torch.from_numpy(z).to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).unsqueeze(0)
            vol = model.volume.eval_volume(z_gpu, noise_std=noise_std)
            save_volume(
                output_dir / f"volume.{idx:03d}.mrc",
                vol.detach().cpu().numpy().squeeze(),
                dataset.psize_A,
            )
    print(f"Saved {len(z_traj)} volumes to {output_dir}")

    return z_traj


@app.command()
def main(
    output_dir: Annotated[Path, typer.Option("--out", "-o", help="Output directory", resolve_path=True)],
    config_path: Annotated[Path, typer.Option("--config", "-c", help="Config YAML file", exists=True)],
    checkpoint_path: Annotated[Path, typer.Option("--ckpt", "-p", help="Model checkpoint file", exists=True)],
    images_path: Annotated[Path, typer.Option("--images", "-i", help="HDF5 images dataset", exists=True)],
    pc_dims: Annotated[str, typer.Option("--pc-dims", "-d", help="PC dimensions for N-dimensional analysis")] = "0 1",
    num_steps: Annotated[int, typer.Option("--num-steps", "-n", help="Number of frames in trajectory")] = 60,
    method: Annotated[str, typer.Option("--method", "-m", help="Trajectory planning method")] = "gradient",
    bw_multiplier: Annotated[float | None, typer.Option("--bw-multiplier", help="Bandwidth multiplier for KDE")] = None,
    peak_threshold_rel: Annotated[float, typer.Option("--peak-threshold-rel", help="Relative threshold for peak detection as a fraction of max density")] = 0.01,
    close: Annotated[bool, typer.Option("--close/--open", "-e", help="Generate closed loop trajectory")] = True,
    noise_std: Annotated[float, typer.Option("--noise-std")] = 300.0,
) -> None:
    """
    Plan trajectory through conformational space.

    Detects density peaks and plans smooth path connecting them.
    Supports N-dimensional input (visualization for 2D/3D only).

    Methods:
    - "gradient": Gradient-driven path refinement (N-D supported)
    - "fmm": Fast Marching Method / Eikonal solver (2D/3D only)
    - "linear": Simple linear interpolation (N-D supported)
    """
    # Parse PC dimensions
    pc_dims_list = [int(x) for x in pc_dims.split()]
    ndim = len(pc_dims_list)

    # FMM only supports 2D/3D
    if method == "fmm" and ndim > 3:
        print(f"Error: FMM method only supports 2D/3D, got {ndim}D")
        return

    print("=" * 70)
    print(f"TRAJECTORY PLANNING: {ndim}D (PC{', PC'.join(map(str, pc_dims_list))})")
    print("=" * 70)
    print(f"Method: {method.upper()}")
    print(f"Trajectory type: {'Closed loop' if close else 'Open path'}")
    print(f"Number of frames: {num_steps}")

    output_dir.mkdir(parents=True, exist_ok=True)
    dims_str = "_".join(f"PC{d}" for d in pc_dims_list)
    traj_dir = output_dir / f"{dims_str}_{method}"
    traj_dir.mkdir(parents=True, exist_ok=True)

    # Load model and dataset
    print("\n[1/5] Loading model and dataset...")
    model, dataset, _ = setup_system(
        config_path,
        checkpoint_path,
        images_path,
    )

    # Encode latent vectors
    print("\n[2/5] Encoding latent vectors...")
    all_z_path = output_dir.parent / "all_z.pkl"
    all_z = encode_latent_vectors(model, dataset, all_z_path)
    print(f"Latent vectors shape: {all_z.shape}")

    # PCA reduction
    print("\n[3/5] Performing PCA...")
    pipeline = LatentAnalysisPipeline(all_z)
    pc = pipeline.run_reduction(method="pca")
    pca = pipeline.reduction_models["pca"]
    print(f"PCA shape: {pc.shape}")

    # Extract selected dimensions
    pc_selected = pc[:, pc_dims_list]

    # Detect peaks for waypoints
    print(f"\n[4/5] Planning {ndim}D trajectory...")
    print("  - Detecting density peaks...")
    detector = PeakDetector(pc_selected, bw_multiplier=bw_multiplier, peak_threshold_rel=peak_threshold_rel)
    peak_coords = detector.get_peaks()
    print(f"  - Found {len(peak_coords)} peaks")

    # Plan trajectory
    print("  - Planning path...")
    planner = TrajectoryPlanner(pc_selected, waypoints=peak_coords)
    trajectory = planner.plan(
        method=method,
        num_steps=num_steps,
        close=close,
    )
    waypoints = planner.get_waypoints()
    ordered_waypoints = planner.get_ordered_waypoints()
    print(f"  - Planned: {len(ordered_waypoints)} waypoints -> {len(trajectory)} steps")

    # Save trajectory
    np.save(traj_dir / "trajectory.npy", trajectory)
    np.save(traj_dir / "waypoints.npy", ordered_waypoints)

    # Visualization (2D/3D only)
    if ndim <= 3:
        print("  - Creating visualization...")
        if ndim == 2:
            density_np = detector.kde.pdf(detector.kde.dataset).cpu().numpy()
            fig = plot_trajectory_2d(pc_selected, trajectory, waypoints, ordered_waypoints, density_np, pc_dims=pc_dims_list)
        else:
            fig = plot_trajectory_3d(pc_selected, trajectory, waypoints, ordered_waypoints, pc_dims=pc_dims_list)
        fig.savefig(traj_dir / "trajectory_summary.png")
        plt.close(fig)

    # Energy profile (always, regardless of ndim)
    fig_energy = plot_energy_profile(pc_selected, trajectory, waypoints)
    fig_energy.savefig(traj_dir / "energy_profile.png")
    plt.close(fig_energy)

    # Generate volumes
    print("\n[5/5] Generating volumes...")

    # Generate zero-padded volumes
    print("  - Generating zero_padded volumes...")
    zero_padded_dir = traj_dir / "zero_padded"
    generate_volumes_from_trajectory(
        trajectory,
        pc_dims_list,
        pca,
        model,
        dataset,
        zero_padded_dir,
        noise_std,
        pc=None,
        use_nearest_point=False,
    )

    # Generate nearest-point volumes
    print("  - Generating nearest_point volumes...")
    nearest_point_dir = traj_dir / "nearest_point"
    generate_volumes_from_trajectory(
        trajectory,
        pc_dims_list,
        pca,
        model,
        dataset,
        nearest_point_dir,
        noise_std,
        pc=pc,
        use_nearest_point=True,
    )

    # Summary
    print("\n" + "=" * 70)
    print("TRAJECTORY PLANNING COMPLETE")
    print("=" * 70)
    print(f"Output directory: {traj_dir}")
    print(f"\nGenerated files:")
    print(f"  - trajectory.npy, waypoints.npy")
    if ndim <= 3:
        print(f"  - trajectory_summary.png")
    print(f"  - energy_profile.png")
    print(f"  - zero_padded/")
    print(f"    - trajectory_z.npy (latent coordinates)")
    print(f"    - volume.*.mrc ({num_steps} volumes)")
    print(f"  - nearest_point/")
    print(f"    - trajectory_z.npy (latent coordinates)")
    print(f"    - volume.*.mrc ({num_steps} volumes)")
    print("=" * 70)


if __name__ == "__main__":
    app()
