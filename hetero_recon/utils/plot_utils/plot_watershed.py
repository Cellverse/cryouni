import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .axis_utils import (
    _draw_cluster_center_scatter,
    _draw_cluster_scatter,
    _draw_density_projection_3d,
    _draw_density_scatter_2d,
)
from .color_utils import (
    _get_colors,
    get_faded_cmap,
)
from hetero_recon.utils.kde_gpu import GaussianKDE


def plot_watershed_2d(
    delta_val: float | None,
    data: np.ndarray,
    peaks: np.ndarray,
    segmentation: np.ndarray,
    grid: np.ndarray,
    density: np.ndarray,
    pc_dims: list[int] | None = None,
    batch_size: int = 2048,
) -> plt.Figure:
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300, tight_layout=True)

    data_2d = data[:, :2]
    kde = GaussianKDE(data_2d.astype(np.float32))

    # Density-colored scatter background
    density_pts = kde.pdf(kde.dataset, batch_size=batch_size).cpu().numpy()
    _draw_density_scatter_2d(ax, data_2d[:, 0], data_2d[:, 1], density_pts, alpha=0.5, zorder=1)

    # Grid coordinates and basin labels from segmenter
    XX = grid[:, :, 0]
    YY = grid[:, :, 1]
    label_grid = segmentation.astype(float) - 1  # 0-based, -1 for background

    # Basin boundaries (pairwise)
    unique_labels = np.unique(label_grid[label_grid >= 0])
    fig_dummy, ax_dummy = plt.subplots()
    for i in range(len(unique_labels)):
        for j in range(i + 1, len(unique_labels)):
            c, d = unique_labels[i], unique_labels[j]
            pair_map = np.full(label_grid.shape, np.nan)
            pair_map[label_grid == c] = 1
            pair_map[label_grid == d] = 0
            cs = ax_dummy.contour(XX, YY, pair_map, levels=[0.5])
            for verts in cs.allsegs[0]:
                ax.plot(verts[:, 0], verts[:, 1], color="black", linewidth=2, linestyle="--", zorder=3)
    plt.close(fig_dummy)

    # kT contours per basin
    if delta_val is not None:
        ZZ = np.log1p(density)
        for k in unique_labels:
            k = int(k)
            basin_mask = label_grid == k
            rho_max = density[basin_mask].max()
            z_val = float(np.log1p(rho_max * np.exp(-delta_val)))
            ZZ_masked = np.where(basin_mask, ZZ, np.nan)
            cs = plt.contour(XX, YY, ZZ_masked, levels=[z_val])
            for path in cs.get_paths():
                verts = path.vertices
                if len(verts) == 0:
                    continue
                ax.plot(verts[:, 0], verts[:, 1], color="#FF8080", linestyle="--", linewidth=2, zorder=2)
            plt.close(cs.figure)

        _draw_cluster_center_scatter(ax=ax, centers=peaks, zorder=10)

    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    n_clusters = len(unique_labels)
    colors = _get_colors(n_clusters)
    for k in unique_labels:
        k = int(k)
        masked = np.ma.masked_where(label_grid != k, np.ones_like(label_grid))
        ax.pcolormesh(XX, YY, masked, cmap=ListedColormap([colors[k]]), alpha=0.3, zorder=0, vmin=0, vmax=1)
    legend_handles = [Patch(facecolor=colors[int(k)], label=f"Cluster {int(k)}") for k in unique_labels]
    ax.legend(handles=legend_handles, loc="upper right", frameon=True, framealpha=0.5)

    if pc_dims is not None:
        ax.set_xlabel(f"PC {pc_dims[0]}")
        ax.set_ylabel(f"PC {pc_dims[1]}")
    else:
        ax.set_xlabel("PC 0")
        ax.set_ylabel("PC 1")
    ax.set_title("2D Watershed Segmentation")
    ax.set_aspect("equal", adjustable="box")

    return fig


def plot_watershed_3d(
    data: np.ndarray,
    labels: np.ndarray,
    peaks: np.ndarray | None = None,
    pc_dims: list[int] | None = None,
) -> plt.Figure:
    """
    Plot 3D watershed segmentation.

    Args:
        `data` (np.ndarray): Data points of shape [N, 3].
        `labels` (np.ndarray): Cluster labels of shape [N].
        `peaks` (np.ndarray, optional): Peak locations of shape [K, 3].
        `pc_dims` (list[int], optional): PC dimension indices for axis labels.

    Returns:
        `fig` (plt.Figure): Matplotlib figure.
    """
    sns.set_theme(style="ticks")
    fig = plt.figure(figsize=(10, 10), dpi=300, tight_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.computed_zorder = False

    cluster_num = len(np.unique(labels[labels >= 0]))
    colors = _get_colors(cluster_num)

    # Add density projections on walls (low alpha background)
    x_min, x_max = data[:, 0].min(), data[:, 0].max()
    y_min, y_max = data[:, 1].min(), data[:, 1].max()
    z_min, z_max = data[:, 2].min(), data[:, 2].max()
    x_pad = (x_max - x_min) * 0.01
    y_pad = (y_max - y_min) * 0.01
    z_pad = (z_max - z_min) * 0.01

    _draw_density_projection_3d(ax, data[:, [1, 2]], zdir="x", offset=x_min - x_pad, cmap=get_faded_cmap())
    _draw_density_projection_3d(ax, data[:, [0, 2]], zdir="y", offset=y_max + y_pad, cmap=get_faded_cmap())
    _draw_density_projection_3d(ax, data[:, [0, 1]], zdir="z", offset=z_min - z_pad, cmap=get_faded_cmap())

    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_zlim(z_min - z_pad, z_max + z_pad)

    # Scatter plot by cluster (overlayed on top)
    for i in range(cluster_num):
        mask = labels == i
        data_sub = data[mask]
        _draw_cluster_scatter(
            ax=ax,
            data=data_sub,
            color=colors[i],
            alpha=0.1,
            zorder=1,
        )

    # Plot boundary points
    boundary_mask = labels == -1
    if np.any(boundary_mask):
        _draw_cluster_scatter(
            ax=ax,
            data=data[boundary_mask],
            color="gray",
            alpha=0.1,
            zorder=1,
        )

    # Plot peaks on top
    _draw_cluster_center_scatter(
        ax=ax,
        centers=peaks,
        zorder=10,
    )

    # Add legend
    legend_handles = []
    for i in range(cluster_num):
        legend_handles.append(
            plt.Line2D(
                xdata=[0],
                ydata=[0],
                marker="o",
                color="w",
                label=str(i),
                markerfacecolor=colors[i],
                markersize=5,
            )
        )
    legend_handles.append(plt.Line2D(
        xdata=[0],
        ydata=[0],
        marker="o",
        color="w",
        label=str(-1),
        markerfacecolor="gray",
        markersize=5,
    ))
    ax.legend(handles=legend_handles, loc="upper right", frameon=True, fancybox=True, framealpha=0.5)

    # Set axis labels with proper PC dimensions
    if pc_dims is not None:
        ax.set_xlabel(f"PC {pc_dims[0]}")
        ax.set_ylabel(f"PC {pc_dims[1]}")
        ax.set_zlabel(f"PC {pc_dims[2]}")
    else:
        ax.set_xlabel("PC 0")
        ax.set_ylabel("PC 1")
        ax.set_zlabel("PC 2")
    ax.set_title("3D Watershed Segmentation")
    ax.grid(False)
    ax.set_aspect("auto")

    return fig
