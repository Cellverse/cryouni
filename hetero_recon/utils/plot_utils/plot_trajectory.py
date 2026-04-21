from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
import seaborn as sns
import torch

from .axis_utils import (
    _draw_density_scatter_2d,
    _draw_density_scatter_3d,
)
from .color_utils import get_faded_cmap
from hetero_recon.utils.kde_gpu import GaussianKDE


def plot_trajectory_2d(
    data: np.ndarray,
    trajectory: np.ndarray,
    waypoints: np.ndarray,
    ordered_waypoints: np.ndarray,
    density: np.ndarray,
    pc_dims: list[int],
) -> plt.Figure:
    """
    Plot 2D trajectory with density background and colored path matching InteractiveExplorer.
    """
    sns.set_theme(style="ticks")
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300, tight_layout=True)

    _draw_density_scatter_2d(
        ax=ax,
        x=data[:, pc_dims[0]],
        y=data[:, pc_dims[1]],
        density=density,
        zorder=1,
    )

    # 2. Plot trajectory with bwr gradient (Match InteractiveExplorer: optimized_line)
    points = trajectory.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1 :]], axis=1)

    lc = LineCollection(segments, cmap="bwr", linewidth=3, zorder=10)
    lc.set_array(np.linspace(0, 1, len(segments))) # Color based on progress
    ax.add_collection(lc)

    # 3. Plot waypoints (Match InteractiveExplorer: path_scatter)
    # Color corresponds to the position in ordered_waypoints, label corresponds to index in waypoints
    wp_colors = []
    for wp in waypoints:
        idx = np.where((ordered_waypoints == wp).all(axis=1))[0][0]
        wp_colors.append(idx / max(1, len(ordered_waypoints) - 1))

    ax.scatter(waypoints[:, 0], waypoints[:, 1], s=80, c=wp_colors, cmap="bwr", edgecolors="black", marker="o", zorder=20)
    for i, wp in enumerate(waypoints):
        ax.annotate(str(i), xy=wp, xytext=(5, 5), textcoords="offset points", fontweight='bold')

    # Axis setup
    ax.set_xlabel(f"PC {pc_dims[0]}")
    ax.set_ylabel(f"PC {pc_dims[1]}")

    ax.set_title("2D Trajectory")
    ax.grid(True, alpha=0.25)

    return fig


def plot_energy_profile(
    data: np.ndarray,
    trajectory: np.ndarray,
    waypoints: np.ndarray | None = None,
    kde: "GaussianKDE | None" = None,
) -> plt.Figure:
    """
    Plot 1D energy profile along the trajectory, with optional peak markers.

    Energy is defined as -log(density), evaluated at each trajectory point.

    Args:
        `data` (np.ndarray): All data points used to fit KDE, shape [N, D].
        `trajectory` (np.ndarray): Trajectory points, shape [M, D].
        `waypoints` (np.ndarray | None): Peak waypoints, shape [K, D].
            If provided, each peak is marked on the profile.
        `kde`: Pre-fitted GaussianKDE instance. If None, a new one is fitted from data.

    Returns:
        `fig` (plt.Figure): The energy profile figure.
    """
    if kde is None:
        kde = GaussianKDE(torch.from_numpy(data).float())

    traj_tensor = torch.from_numpy(trajectory.astype(np.float32)).to(kde.dataset.device)
    log_density = kde.logpdf(traj_tensor, batch_size=512).cpu().numpy()
    energy = -log_density

    steps = np.arange(len(energy))

    sns.set_theme(style="ticks")
    fig, ax = plt.subplots(figsize=(10, 3), dpi=300, tight_layout=True)

    # Color line by progress (bwr, matching trajectory plots)
    points = np.array([steps, energy]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1 :]], axis=1)
    from matplotlib.collections import LineCollection
    lc = LineCollection(segments, cmap="bwr", linewidth=2, zorder=5)
    lc.set_array(np.linspace(0, 1, len(segments)))
    ax.add_collection(lc)
    ax.set_xlim(steps[0], steps[-1])
    ax.set_ylim(energy.min() - 0.5, energy.max() + 0.5)

    # Mark peak positions
    if waypoints is not None and len(waypoints) > 0:
        # Find the nearest trajectory frame for each waypoint
        dists = np.linalg.norm(trajectory[:, np.newaxis, :] - waypoints[np.newaxis, :, :], axis=-1) # [M, K]
        peak_frames = dists.argmin(axis=0)                                                          # [K]
        cmap_bwr = plt.get_cmap("bwr")
        for k, frame in enumerate(peak_frames):
            color = cmap_bwr(k / max(1, len(peak_frames) - 1))
            ax.scatter(frame, energy[frame], color=color, edgecolors="black", s=60, zorder=10)
            ax.text(frame, energy[frame] - 0.35, str(k), ha="center", va="top", fontsize=8, fontweight="bold", color=color, zorder=11)

    ax.set_xlabel("Frame")
    ax.set_ylabel("Energy  (-log density)")
    ax.set_title("Energy Profile Along Trajectory")
    ax.grid(True, alpha=0.25)

    return fig


def plot_trajectory_3d(
    data: np.ndarray,
    trajectory: np.ndarray,
    waypoints: np.ndarray,
    ordered_waypoints: np.ndarray,
    pc_dims: list[int] | None = None,
) -> plt.Figure:
    """
    Plot 3D trajectory with colored path matching InteractiveExplorer.
    """
    sns.set_theme(style="white")
    fig = plt.figure(figsize=(10, 10), dpi=300, tight_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    # 1. Background Density Scatter
    kde = GaussianKDE(torch.from_numpy(data).float().to("cuda" if torch.cuda.is_available() else "cpu"))
    dens = kde.pdf(kde.dataset, batch_size=1024).cpu().numpy()
    density = (dens - dens.min()) / (dens.max() - dens.min())

    cmap = get_faded_cmap(name="turbo", gamma=4)
    color = cmap(density)
    color[:, 3] = density ** 3

    _draw_density_scatter_3d(
        ax=ax,
        data_3d=data,
        c=color,
        zorder=10,
    )

    # 2. 3D Trajectory with bwr gradient
    points = trajectory.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1 :]], axis=1)

    lc = Line3DCollection(segments, cmap="bwr", linewidth=4, zorder=15)
    lc.set_array(np.linspace(0, 1, len(segments)))
    ax.add_collection3d(lc)

    # 3. 3D Waypoints
    # Color corresponds to the position in ordered_waypoints, label corresponds to index in waypoints
    wp_colors = []
    for wp in waypoints:
        idx = np.where((ordered_waypoints == wp).all(axis=1))[0][0]
        wp_colors.append(idx / max(1, len(ordered_waypoints) - 1))

    ax.scatter(
        waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], s=100, c=wp_colors, cmap="bwr", edgecolors="black", depthshade=False, zorder=20
    )
    for i, wp in enumerate(waypoints):
        ax.text(wp[0], wp[1], wp[2], str(i), fontweight='bold', zorder=30)

    # Axis setup
    if pc_dims is not None:
        ax.set_xlabel(f"PC {pc_dims[0]}")
        ax.set_ylabel(f"PC {pc_dims[1]}")
        ax.set_zlabel(f"PC {pc_dims[2]}")
    else:
        ax.set_xlabel("Dim 0")
        ax.set_ylabel("Dim 1")
        ax.set_zlabel("Dim 2")
    ax.set_title("3D Trajectory")
    ax.grid(True, alpha=0.25)

    return fig
