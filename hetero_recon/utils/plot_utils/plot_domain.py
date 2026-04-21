from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_hartley_domain(y: np.ndarray, y_pred: np.ndarray, index: int | None = None) -> plt.Figure:
    D = y.shape[-1]

    sns.set_theme(style="ticks")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150, tight_layout=True)

    im1 = axes[0].imshow(np.abs(y) + 1, norm=colors.LogNorm(), cmap="RdPu")
    axes[0].set_title(f"Ground Truth ({D}x{D})" + (f" (Index: {index})" if index is not None else ""))
    fig.colorbar(im1, ax=axes[0], shrink=0.9)

    im2 = axes[1].imshow(np.abs(y_pred) + 1, norm=colors.LogNorm(), cmap="RdPu")
    axes[1].set_title(f"Prediction ({D}x{D})" + (f" (Index: {index})" if index is not None else ""))
    fig.colorbar(im2, ax=axes[1], shrink=0.9)

    return fig


def plot_spatial_domain(y: np.ndarray, y_pred: np.ndarray, index: int | None = None) -> plt.Figure:
    D = y.shape[-1]

    sns.set_theme(style="ticks")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), dpi=150, tight_layout=True)

    im1 = axes[0].imshow(y, cmap="viridis")
    axes[0].set_title(f"GT ({D}x{D})" + (f"\nIdx: {index}" if index is not None else ""))
    fig.colorbar(im1, ax=axes[0], shrink=0.9)

    im2 = axes[1].imshow(y_pred, cmap="viridis")
    axes[1].set_title(f"Pred ({D}x{D})" + (f"\nIdx: {index}" if index is not None else ""))
    fig.colorbar(im2, ax=axes[1], shrink=0.9)

    return fig
