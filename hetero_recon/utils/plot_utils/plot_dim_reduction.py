from __future__ import annotations

from matplotlib import cm
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from .axis_utils import (
    _draw_cluster_center_scatter,
    _draw_cluster_scatter,
    _draw_density_scatter_2d,
    _draw_density_scatter_3d,
)
from .color_utils import (
    _get_colors,
    get_faded_cmap,
)
from hetero_recon.utils.kde_gpu import GaussianKDE


def _plot_cluster_2d(
    data_2d: np.ndarray,
    cluster_num: int,
    cluster_labels: int,
    cluster_centers: np.ndarray | None = None,
    cluster_center_indices: np.ndarray | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    sns.set_theme(style="ticks")
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300, tight_layout=True)

    colors = _get_colors(cluster_num)

    # Scatter plot of each cluster
    for i in range(cluster_num):
        mask = cluster_labels == i
        data_2d_sub = data_2d[mask]
        _draw_cluster_scatter(
            ax,
            data_2d_sub,
            color=colors[i],
            alpha=0.5,
            zorder=1,
        )

    if cluster_center_indices is not None:
        cluster_centers = data_2d[cluster_center_indices]

    # Scatter plot of cluster centers
    if cluster_centers is not None:
        _draw_cluster_center_scatter(
            ax,
            cluster_centers,
            zorder=10,
        )

    # Add legend
    legend_elements = []
    for i in range(cluster_num):
        legend_elements.append(
            Line2D(
                xdata=[0],
                ydata=[0],
                marker="o",
                color="w",
                label=str(i),
                markerfacecolor=colors[i],
                markersize=10,
            )
        )
    ax.legend(handles=legend_elements, loc="upper right", frameon=True, fancybox=True, framealpha=0.5)

    data_min, data_max = data_2d.min(axis=0), data_2d.max(axis=0)
    data_pad = (data_max - data_min) * 0.01
    ax.set_xlim(data_min[0] - data_pad[0], data_max[0] + data_pad[0])
    ax.set_ylim(data_min[1] - data_pad[1], data_max[1] + data_pad[1])

    ax.grid(True, alpha=0.25)
    ax.set_aspect("auto")

    return fig, ax


def plot_pca_2d(
    data: np.ndarray,
    cluster_num: int,
    cluster_labels: np.ndarray,
    cluster_center_indices: np.ndarray,
) -> plt.Figure:
    fig, ax = _plot_cluster_2d(
        data_2d=data[:, : 2],
        cluster_num=cluster_num,
        cluster_labels=cluster_labels,
        cluster_center_indices=cluster_center_indices,
    )

    ax.set_xlabel("PC0")
    ax.set_ylabel("PC1")
    ax.set_title("2D PCA with GMM Clustering")

    return fig


def plot_umap_2d(
    data: np.ndarray,
    cluster_num: int,
    cluster_labels: np.ndarray,
    cluster_center_indices: np.ndarray,
) -> plt.Figure:
    fig, ax = _plot_cluster_2d(
        data_2d=data[:, : 2],
        cluster_num=cluster_num,
        cluster_labels=cluster_labels,
        cluster_center_indices=cluster_center_indices,
    )

    ax.set_xlabel("UMAP0")
    ax.set_ylabel("UMAP1")
    ax.set_title("2D UMAP with GMM Clustering")

    return fig


def _plot_density_2d(data_2d: np.ndarray, cbar: bool = True) -> plt.Figure:
    sns.set_theme(style="ticks")
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300, tight_layout=True)

    kde = GaussianKDE(torch.from_numpy(data_2d).float().to("cuda" if torch.cuda.is_available() else "cpu"))
    density = kde.pdf(kde.dataset, batch_size=1024).cpu().numpy()

    scatter = _draw_density_scatter_2d(
        ax,
        data_2d[:, 0],
        data_2d[:, 1],
        density,
        alpha=0.5,
        zorder=1,
    )

    # Add colorbar
    if cbar:
        colobar = fig.colorbar(scatter, ax=ax, shrink=0.9, label="Density")
        colobar.set_alpha(1.0)

    ax.set_aspect("equal", adjustable="box")
    return fig, ax


def plot_pca_density_2d(data: np.ndarray, cbar: bool = True) -> plt.Figure:
    fig, ax = _plot_density_2d(data[:, : 2], cbar)

    ax.set_xlabel("PC0")
    ax.set_ylabel("PC1")
    ax.set_title("2D PCA Feature Density")

    return fig


def plot_umap_density_2d(data: np.ndarray, cbar: bool = True) -> plt.Figure:
    fig, ax = _plot_density_2d(data[:, : 2], cbar)

    ax.set_xlabel("UMAP0")
    ax.set_ylabel("UMAP1")
    ax.set_title("2D UMAP Feature Density")

    return fig


def _plot_cluster_3d(
    data_3d: np.ndarray,
    cluster_num: int,
    cluster_labels: int,
    cluster_centers: np.ndarray | None = None,
    cluster_center_indices: np.ndarray | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    sns.set_theme(style="ticks")
    fig = plt.figure(figsize=(10, 10), dpi=300, tight_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    colors = _get_colors(cluster_num)

    # Scatter plot of each cluster
    for i in range(cluster_num):
        mask = cluster_labels == i
        data_3d_sub = data_3d[mask]
        _draw_cluster_scatter(
            ax,
            data=data_3d_sub,
            color=colors[i],
            alpha=0.1,
            zorder=1,
        )

    if cluster_center_indices is not None:
        cluster_centers = data_3d[cluster_center_indices]

    # Scatter plot of cluster centers
    if cluster_centers is not None:
        _draw_cluster_center_scatter(
            ax,
            cluster_centers,
            zorder=10,
        )

    # Add legend
    legend_elements = []
    for i in range(cluster_num):
        legend_elements.append(
            Line2D(
                xdata=[0],
                ydata=[0],
                marker="o",
                color="w",
                label=str(i),
                markerfacecolor=colors[i],
                markersize=10,
            )
        )
    ax.legend(handles=legend_elements, loc="upper right", frameon=True, fancybox=True, framealpha=0.5)

    data_min, data_max = np.min(data_3d, axis=0), np.max(data_3d, axis=0)
    data_pad = (data_max - data_min) * 0.01
    ax.set_xlim(data_min[0] - data_pad[0], data_max[0] + data_pad[0])
    ax.set_ylim(data_min[1] - data_pad[1], data_max[1] + data_pad[1])
    ax.set_zlim(data_min[2] - data_pad[2], data_max[2] + data_pad[2])

    ax.grid(True, alpha=0.25)
    ax.set_aspect("auto")

    return fig, ax


def plot_pca_3d(
    data: np.ndarray,
    cluster_num: int,
    cluster_labels: np.ndarray,
    cluster_center_indices: np.ndarray,
) -> plt.Figure:
    fig, ax = _plot_cluster_3d(
        data_3d=data[:, : 3],
        cluster_num=cluster_num,
        cluster_labels=cluster_labels,
        cluster_center_indices=cluster_center_indices,
    )

    ax.set_xlabel("PC0")
    ax.set_ylabel("PC1")
    ax.set_zlabel("PC2")
    ax.set_title("3D PCA with GMM Clustering")

    return fig


def plot_umap_3d(
    data: np.ndarray,
    cluster_num: int,
    cluster_labels: np.ndarray,
    cluster_center_indices: np.ndarray,
) -> plt.Figure:
    fig, ax = _plot_cluster_3d(
        data_3d=data[:, : 3],
        cluster_num=cluster_num,
        cluster_labels=cluster_labels,
        cluster_center_indices=cluster_center_indices,
    )

    ax.set_xlabel("UMAP0")
    ax.set_ylabel("UMAP1")
    ax.set_zlabel("UMAP2")
    ax.set_title("3D UMAP with GMM Clustering")

    return fig


def _plot_density_3d(data_3d: np.ndarray, cbar: bool = True) -> tuple[plt.Figure, plt.Axes]:
    sns.set_theme(style="ticks")
    fig = plt.figure(figsize=(10, 10), dpi=300, tight_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.computed_zorder = False

    # Compute density at each point
    kde = GaussianKDE(torch.from_numpy(data_3d).float().to("cuda" if torch.cuda.is_available() else "cpu"))
    density = kde.pdf(kde.dataset, batch_size=1024).cpu().numpy()

    # Normalize density for better visualization
    density = (density - density.min()) / (density.max() - density.min())

    cmap = get_faded_cmap(name="turbo", gamma=4)
    color = cmap(density)
    color[:, 3] = density ** 3

    # Scatter plot with density-based coloring
    scatter = _draw_density_scatter_3d(
        ax,
        data_3d,
        color,
        cmap=get_faded_cmap(),
        zorder=10,
    )

    # Add colorbar
    if cbar:
        sm = cm.ScalarMappable(cmap=cmap)
        sm.set_array([])
        cbar_obj = fig.colorbar(sm, ax=ax, shrink=0.8, label="Density")
        cbar_obj.solids.set_alpha(1.0)

    ax.grid(False)
    ax.set_aspect("auto")

    return fig, ax


def plot_pca_density_3d(pc: np.ndarray, cbar: bool = True) -> plt.Figure:
    fig, ax = _plot_density_3d(pc[:, : 3], cbar=cbar)

    ax.set_xlabel("PC0")
    ax.set_ylabel("PC1")
    ax.set_zlabel("PC2")
    ax.set_title("3D PCA Feature Density")

    return fig


def plot_umap_density_3d(umap: np.ndarray, cbar: bool = True) -> plt.Figure:
    fig, ax = _plot_density_3d(umap[:, : 3], cbar=cbar)

    ax.set_xlabel("UMAP0")
    ax.set_ylabel("UMAP1")
    ax.set_zlabel("UMAP2")
    ax.set_title("3D UMAP Feature Density")

    return fig


def plot_pca_explained_variance_ratio(pca_explained_variance: np.ndarray) -> plt.Figure:
    n_features = len(pca_explained_variance)

    sns.set_theme(style="ticks")
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300, tight_layout=True)

    ax.bar(np.arange(n_features), pca_explained_variance, edgecolor="white", linewidth=1)
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance")
    ax.set_title("PCA Explained Variance")
    ax.set_xticks(np.arange(n_features))
    ax.grid(True)

    return fig
