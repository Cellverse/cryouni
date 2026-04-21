import matplotlib.pyplot as plt
import seaborn as sns

from .axis_utils import (
    _draw_peak_contour_2d,
    _draw_peak_scatter_3d,
)
from .color_utils import get_faded_cmap
from hetero_recon.utils.peak_detector import PeakDetector


def plot_peak_2d(
    detector: PeakDetector,
    cbar: bool = True,
    xlabel: str = "X",
    ylabel: str = "Y",
    title: str | None = None,
) -> plt.Figure:
    sns.set_theme(style="ticks")
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300, tight_layout=True)

    # Get density grid
    density = detector.kde.pdf(detector.kde.dataset, batch_size=1024).cpu().numpy()

    # Plot with scatter
    scatter_density, _ = _draw_peak_contour_2d(
        ax,
        detector.points[:, 0],
        detector.points[:, 1],
        density,
        detector.get_peaks(),
    )

    # Add colorbar
    if cbar:
        colorbar = fig.colorbar(scatter_density, ax=ax, shrink=0.9, label="Density")
        colorbar.set_alpha(1.0)

    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.set_aspect("auto")

    return fig


def plot_peak_3d(
    detector: PeakDetector,
    xlabel: str = "X",
    ylabel: str = "Y",
    zlabel: str = "Z",
    title: str | None = None,
) -> plt.Figure:
    """
    Create a 3D density map with labeled peaks.
    """
    sns.set_theme(style="ticks")
    fig = plt.figure(figsize=(10, 10), dpi=300, tight_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.computed_zorder = False

    _, density = detector.get_density_grid()

    # Compute density at each point for coloring
    density = detector.kde.pdf(detector.kde.dataset, batch_size=1024).cpu().numpy()
    density = (density - density.min()) / (density.max() - density.min())

    cmap = get_faded_cmap(name="turbo", gamma=4)
    color = cmap(density)
    color[:, 3] = density ** 3

    scatter = _draw_peak_scatter_3d(
        ax=ax,
        data_3d=detector.points,
        peaks=detector.get_peaks(),
        c=color,
        zorder=10,
    )

    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    if title:
        ax.set_title(title)

    ax.grid(False)

    return fig
