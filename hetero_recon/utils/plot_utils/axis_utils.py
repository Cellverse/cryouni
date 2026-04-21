import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
import seaborn as sns
import torch

from .color_utils import (
    _get_colors,
    get_faded_cmap,
)
from hetero_recon.utils.kde_gpu import GaussianKDE


def _draw_cluster_scatter(
    ax,
    data,
    color,
    alpha: float = 1.0,
    zorder: int = 0,
):
    """
    Args:
        `ax` (plt.Axes): Matplotlib axes object, 2D and 3D axes are supported.
        `data` (np.ndarray): Data points of shape [N, 2] or [N, 3].
        `c` (str | np.ndarray): Color values.
    """
    dim = data.shape[-1]
    kwargs = {
        "s": 5,
        "edgecolors": "none",
        "alpha": alpha,
        "rasterized": True,
        "zorder": zorder,
    }

    if dim == 2:
        scatter = ax.scatter(
            x=data[:, 0],
            y=data[:, 1],
            color=color,
            **kwargs,
        )
    else:
        scatter = ax.scatter(
            xs=data[:, 0],
            ys=data[:, 1],
            zs=data[:, 2],
            color=color,
            **kwargs,
        )
    return scatter


def _draw_cluster_center_scatter(
    ax,
    centers,
    zorder: int = 0,
):
    """
    Args:
        `ax` (plt.Axes): Matplotlib axes object, 2D and 3D axes are supported.
        `centers` (np.ndarray): Data points of shape [N, 2] or [N, 3].
    """
    dim = centers.shape[-1]
    kwargs = {
        "s": 25,
        "c": "k",
        "edgecolors": "none",
        "marker": "o",
        "rasterized": True,
        "zorder": zorder,
    }
    if dim == 2:
        scatter = ax.scatter(
            x=centers[:, 0],
            y=centers[:, 1],
            **kwargs,
        )

        for i, center in enumerate(centers):
            ax.annotate(
                text=str(i),
                xy=center,
                xytext=(5, 5),
                textcoords="offset points",
                color="black",
                zorder=zorder,
            )
    else:
        scatter = ax.scatter(
            xs=centers[:, 0],
            ys=centers[:, 1],
            zs=centers[:, 2],
            depthshade=True,
            **kwargs,
        )

        for i, center in enumerate(centers):
            ax.text(
                x=center[0],
                y=center[1],
                z=center[2],
                s=str(i),
                color="black",
                zorder=zorder,
            )
    return scatter


def _draw_density_scatter_2d(
    ax,
    x,
    y,
    density,
    alpha: float = 0.5,
    zorder: int = 0,
):
    """
    Args:
        `ax` (plt.Axes): Matplotlib axes object, only 2D axes are supported.
        `x` (np.ndarray): 2D data points of shape [H, W].
        `y` (np.ndarray): 2D data points of shape [H, W].
        `density` (np.ndarray): 2D density grid of shape [H, W].
    """
    norm_density = (density - density.min()) / (density.max() - density.min() + 1e-12)
    cmap = sns.cubehelix_palette(start=.5, rot=-.75, as_cmap=True, reverse=True)
    scatter = ax.scatter(
        x,
        y,
        c=norm_density,
        cmap=cmap,
        s=3,
        alpha=alpha,
        edgecolors="none",
        rasterized=True,
        zorder=zorder,
    )
    return scatter


def _draw_density_projection_3d(
    ax,
    data_2d,
    zdir,
    offset,
    cmap,
):
    # Compute density on grid
    kde = GaussianKDE(torch.from_numpy(data_2d).float().to("cuda" if torch.cuda.is_available() else "cpu"))
    grid = kde.get_grid()

    # Get grid & density
    X = grid[..., 0].cpu().numpy()
    Y = grid[..., 1].cpu().numpy()
    density = kde.pdf(grid, batch_size=1024).cpu().numpy()

    kwargs = {
        "zdir": zdir,
        "offset": offset,
        "levels": 40,
        "cmap": cmap,
        "alpha": 0.2,
        "zorder": 0,
    }

    if zdir == "x":
        ax.contourf(density, X, Y, **kwargs)
    elif zdir == "y":
        ax.contourf(X, density, Y, **kwargs)
    elif zdir == "z":
        ax.contourf(X, Y, density, **kwargs)


def _draw_density_scatter_3d(
    ax,
    data_3d,
    c,
    cmap: str = "turbo",
    zorder: int = 0,
):
    """
    Args:
        `ax` (plt.Axes): Matplotlib axes object, only 3D axes are supported.
        `data_3d` (np.ndarray): 3D data points of shape [N, 3].
        `c` (np.ndarray): Color values of shape [N].
    """
    scatter = ax.scatter(
        xs=data_3d[:, 0],
        ys=data_3d[:, 1],
        zs=data_3d[:, 2],
        c=c,
        s=2.5,
        edgecolors="none",
        antialiased=True,
        rasterized=True,
        zorder=zorder,
    )

    x_min, x_max = data_3d[:, 0].min(), data_3d[:, 0].max()
    y_min, y_max = data_3d[:, 1].min(), data_3d[:, 1].max()
    z_min, z_max = data_3d[:, 2].min(), data_3d[:, 2].max()
    x_pad = (x_max - x_min) * 0.01
    y_pad = (y_max - y_min) * 0.01
    z_pad = (z_max - z_min) * 0.01

    _draw_density_projection_3d(ax, data_3d[:, [1, 2]], zdir="x", offset=x_min - x_pad, cmap=get_faded_cmap())
    _draw_density_projection_3d(ax, data_3d[:, [0, 2]], zdir="y", offset=y_max + y_pad, cmap=get_faded_cmap())
    _draw_density_projection_3d(ax, data_3d[:, [0, 1]], zdir="z", offset=z_min - z_pad, cmap=get_faded_cmap())
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_zlim(z_min - z_pad, z_max + z_pad)

    return scatter


def _draw_peak_contour_2d(
    ax,
    x,
    y,
    density,
    peaks,
    alpha: float = 0.5,
    zorder: int = 0,
):
    """
    Args:
        `ax` (plt.Axes): Matplotlib axes object, only 2D axes are supported.
        `x` (np.ndarray): 2D data points of shape [H, W].
        `y` (np.ndarray): 2D data points of shape [H, W].
        `density` (np.ndarray): 2D density grid of shape [H, W].
        `peaks` (np.ndarray): 2D data points of shape [K, 2].
    """
    scatter_density = _draw_density_scatter_2d(
        ax,
        x,
        y,
        density,
        alpha=alpha,
        zorder=zorder,
    )

    scatter_centers = _draw_cluster_center_scatter(
        ax,
        centers=peaks,
        zorder=zorder + 10,
    )

    return scatter_density, scatter_centers


def _draw_density_hull_3d(
    ax,
    data_3d,
    peaks,
    zorder: int = 0,
):
    dist = cdist(data_3d, peaks)
    labels = dist.argmin(axis=1)
    colors = _get_colors(len(peaks))

    for i in range(len(peaks)):
        cluster_points = data_3d[labels == i]

        if len(cluster_points) < 4:
            continue

        dists_to_peak = np.linalg.norm(cluster_points - peaks[i], axis=-1)
        threshold = np.percentile(dists_to_peak, 50)
        core_points = cluster_points[dists_to_peak <= threshold]

        if len(core_points) > 4:
            hull = ConvexHull(core_points)
            ax.plot_trisurf(
                core_points[:, 0],
                core_points[:, 1],
                core_points[:, 2],
                triangles=hull.simplices,
                color=colors[i],
                alpha=0.25,
                edgecolor="none",
                zorder=zorder,
            )


def _draw_peak_line_3d(
    ax,
    data_3d,
    peaks,
    zorder: int = 0,
):

    x_min, x_max = data_3d[:, 0].min(), data_3d[:, 0].max()
    y_min, y_max = data_3d[:, 1].min(), data_3d[:, 1].max()
    z_min, z_max = data_3d[:, 2].min(), data_3d[:, 2].max()
    x_pad = (x_max - x_min) * 0.01
    y_pad = (y_max - y_min) * 0.01
    z_pad = (z_max - z_min) * 0.01

    for px, py, pz in peaks:
        line_style = {
            "color": "black",
            "linestyle": "--",
            "linewidth": 0.8,
            "alpha": 0.3,
            "zorder": zorder,
        }
        ax.plot([px, x_min - x_pad], [py, py], [pz, pz], **line_style) # to X wall
        ax.plot([px, px], [py, y_max + y_pad], [pz, pz], **line_style) # to Y wall
        ax.plot([px, px], [py, py], [pz, z_min - z_pad], **line_style) # to Z wall


def _draw_peak_scatter_3d(
    ax,
    data_3d,
    peaks,
    c,
    zorder: int = 0,
):
    """
    Args:
        `ax` (plt.Axes): Matplotlib axes object, only 3D axes are supported.
        `x` (np.ndarray): 3D data points of shape [H, W, D].
        `y` (np.ndarray): 3D data points of shape [H, W, D].
        `z` (np.ndarray): 3D data points of shape [H, W, D].
        `density` (np.ndarray): 3D density grid of shape [H, W, D].
        `peaks` (np.ndarray): 3D data points of shape [K, 3].
    """
    scatter = _draw_density_scatter_3d(ax, data_3d, c, zorder=zorder)

    _draw_cluster_center_scatter(ax, peaks, zorder=zorder + 10)

    _draw_peak_line_3d(ax, data_3d, peaks, zorder=zorder + 5)

    return scatter
