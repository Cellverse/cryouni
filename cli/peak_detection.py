"""
Peak Detection Analysis for PCA Space.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import typer
from typing_extensions import Annotated

# Setup Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from cli.utils.common import encode_latent_vectors, setup_system
from cli.utils.constant import CONTOUR_LEVELS

from hetero_recon.utils.io import save_volume
from hetero_recon.utils.latent_analysis import LatentAnalysisPipeline
from hetero_recon.utils.peak_detector import PeakDetector
from hetero_recon.utils.plot_utils import (
    get_faded_cmap,
    plot_peak_2d,
    plot_peak_3d,
)
from hetero_recon.utils.plot_utils.axis_utils import _draw_peak_contour_2d

app = typer.Typer(
    help="Peak detection analysis: systematic exploration of all PC pair combinations.",
    add_completion=False,
)

# =============================================================================
# Volume Generation Functions
# =============================================================================


def find_nearest_points(
    peak_coords: np.ndarray,
    pc: np.ndarray,
    dims: list[int],
) -> np.ndarray:
    """
    Find nearest actual points in PC space for given peak coordinates.
    """
    n_components = pc.shape[1]
    final_pca_points = np.zeros((len(peak_coords), n_components), dtype=np.float32)
    pc_proj = pc[:, dims]

    for idx, peak_coord in enumerate(peak_coords):
        distances = np.linalg.norm(pc_proj - peak_coord, axis=1)
        nearest_idx = np.argmin(distances)
        final_pca_points[idx] = pc[nearest_idx]

    return final_pca_points


def create_zero_padded_pc_coords(
    peak_coords: np.ndarray,
    n_components: int,
    dims: list[int],
) -> np.ndarray:
    """
    Create full PC coordinates by zero-padding other dimensions.
    """
    final_pca_points = np.zeros((len(peak_coords), n_components), dtype=np.float32)
    for i, dim_idx in enumerate(dims):
        final_pca_points[:, dim_idx] = peak_coords[:, i]
    return final_pca_points


def export_peak_volumes(
    peak_coords: np.ndarray,
    dims: list[int],
    pca,
    model,
    dataset,
    output_dir: Path,
    noise_std: float,
    pc: np.ndarray | None = None,
    not_on_data: bool = False,
    max_components: int | None = None,
    volume_size: int | None = None,
) -> None:
    """Export MRC volumes for detected peaks.

    Generates 3D volumes from latent vectors corresponding to each peak.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if not_on_data:
        n_components = max_components if max_components else pca.n_components_
        final_pca_points = create_zero_padded_pc_coords(peak_coords, n_components, dims)
    else:
        if pc is None:
            raise ValueError("pc must be provided when not_on_data=False")
        final_pca_points = find_nearest_points(peak_coords, pc, dims)

    # Pad if needed for inverse transform
    if final_pca_points.shape[1] < pca.n_components_:
        final_pca_points = np.pad(final_pca_points, ((0, 0), (0, pca.n_components_ - final_pca_points.shape[1])))

    # Inverse transform to latent space
    z_peaks = pca.inverse_transform(final_pca_points)

    # Generate and save volumes
    with torch.no_grad():
        for idx, z_vec in enumerate(z_peaks):
            z_gpu = torch.from_numpy(z_vec).to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).unsqueeze(0)
            vol = model.volume.eval_volume(z_gpu, noise_std=noise_std, volume_size=volume_size)
            save_volume(
                output_dir / f"volume.{idx:02d}.mrc",
                vol.detach().cpu().numpy().squeeze(),
                dataset.psize_A,
            )


# =============================================================================
# Visualization Functions
# =============================================================================


def plot_pc_pair_peaks(ax: plt.Axes, detector: PeakDetector) -> None:
    """
    Plot peaks on a corner plot axis.
    """
    grid, density = detector.get_density_grid()
    X = grid[..., 0]
    Y = grid[..., 1]

    # Plot density contours
    _draw_peak_contour_2d(
        ax,
        X,
        Y,
        density,
        detector.get_peaks(),
    )


# =============================================================================
# Analysis Pipelines
# =============================================================================


def analyze_peaks_2d(
    pc: np.ndarray,
    pca,
    model,
    dataset,
    output_dir: Path,
    noise_std: float,
    max_components: int,
    bw_multiplier: float,
    peak_threshold_rel: float = 0.01,
    not_on_data: bool = False,
    volume_size: int | None = None,
) -> None:
    """Analyze 2D PC pair combinations with corner plot and peak volumes."""
    n_pcs = min(max_components, pc.shape[1])
    plots_dir = output_dir / "plots_2d"
    vols_dir = output_dir / "volumes_2d"
    mode_dir = "off_data" if not_on_data else "on_data"
    plots_dir.mkdir(exist_ok=True)
    vols_dir.mkdir(exist_ok=True)

    sns.set_theme(style="white")
    fig, axes = plt.subplots(
        n_pcs,
        n_pcs,
        figsize=(n_pcs * 5, n_pcs * 5),
        squeeze=False,
        tight_layout=True,
    )

    # Process each subplot
    for row in range(n_pcs):
        for col in range(n_pcs):
            ax = axes[row, col]

            # Diagonal: histogram
            if row == col:
                sns.histplot(
                    pc[:, row],
                    kde=True,
                    ax=ax,
                    color="blue",
                    alpha=0.3,
                )
                ax.set_title(f"PC{row}")

            # Upper triangle: off
            elif col > row:
                ax.axis("off")

            # Lower triangle: density + peaks
            else:
                print(f"Processing PC{col} vs PC{row}...")

                detector = PeakDetector(pc[:, [col, row]], bw_multiplier=bw_multiplier, peak_threshold_rel=peak_threshold_rel)
                peak_coords = detector.get_peaks()
                print(f"  Found {len(peak_coords)} peaks")

                plot_pc_pair_peaks(ax, detector)

                # Individual plot
                fig_sub = plot_peak_2d(
                    detector,
                    xlabel=f"PC{col}",
                    ylabel=f"PC{row}",
                    title=f"Density of PC{col} & PC{row}",
                )
                fig_sub.savefig(plots_dir / f"PC{col}_PC{row}.png")
                plt.close(fig_sub)

                # Export peak volumes
                export_peak_volumes(
                    peak_coords,
                    [col, row],
                    pca,
                    model,
                    dataset,
                    vols_dir / f"PC{col}_PC{row}" / mode_dir,
                    noise_std,
                    pc=pc,
                    not_on_data=not_on_data,
                    volume_size=volume_size,
                )

            if row == n_pcs - 1:
                ax.set_xlabel(f"PC{col}")
            if col == 0:
                ax.set_ylabel(f"PC{row}")

    matrix_path = output_dir / "pca_corner_plot.png"
    fig.savefig(matrix_path, dpi=200)
    plt.close(fig)

    print(f"\nCorner plot saved: {matrix_path.name}")
    print(f"Individual plots: {plots_dir.name}/ ({n_pcs * (n_pcs - 1) // 2} PC pairs)")
    print(f"Peak volumes: {vols_dir.name}/ (organized by PC pair)")


def analyze_peaks_3d(
    pc: np.ndarray,
    pca,
    model,
    dataset,
    output_dir: Path,
    noise_std: float,
    max_components: int,
    bw_multiplier: float,
    peak_threshold_rel: float = 0.01,
    not_on_data: bool = False,
    volume_size: int | None = None,
) -> None:
    """Analyze 3D PC triplet combinations with peak detection and volumes."""
    n_pcs = min(max_components, pc.shape[1])
    plots_dir = output_dir / "plots_3d"
    vols_root = output_dir / "volumes_3d"
    mode_dir = "off_data" if not_on_data else "on_data"
    plots_dir.mkdir(exist_ok=True)
    vols_root.mkdir(exist_ok=True)

    triplets = list(combinations(range(n_pcs), 3))

    print(f"\nAnalyzing {len(triplets)} PC triplet combinations from first {n_pcs} PCs...")
    print(f"Triplets: {triplets}\n")

    for dim_x, dim_y, dim_z in triplets:
        print(f"Processing PC{dim_x}, PC{dim_y}, PC{dim_z}...")

        # Detect peaks in 3D
        detector = PeakDetector(pc[:, [dim_x, dim_y, dim_z]], bw_multiplier=bw_multiplier, peak_threshold_rel=peak_threshold_rel)
        peak_coords = detector.get_peaks()
        print(f"  Found {len(peak_coords)} peaks")

        fig = plot_peak_3d(
            detector,
            xlabel=f"PC{dim_x}",
            ylabel=f"PC{dim_y}",
            zlabel=f"PC{dim_z}",
            title=f"PC{dim_x} & PC{dim_y} & PC{dim_z}",
        )
        fig.savefig(plots_dir / f"PC{dim_x}_PC{dim_y}_PC{dim_z}.png")
        plt.close(fig)

        # Export peak volumes
        export_peak_volumes(
            peak_coords,
            [dim_x, dim_y, dim_z],
            pca,
            model,
            dataset,
            vols_root / f"PC{dim_x}_PC{dim_y}_PC{dim_z}" / mode_dir,
            noise_std,
            pc=pc,
            not_on_data=not_on_data,
            max_components=max_components,
            volume_size=volume_size,
        )


def analyze_peaks_nd(
    pc: np.ndarray,
    pca,
    dims: list[int],
    model,
    dataset,
    output_dir: Path,
    noise_std: float,
    bw_multiplier: float,
    peak_threshold_rel: float = 0.01,
    not_on_data: bool = False,
    volume_size: int | None = None,
) -> tuple[np.ndarray, int]:
    """
    Analyze N-dimensional PC space and export peak volumes without plotting.

    This function performs peak detection in arbitrary N-dimensional PC space
    and exports the corresponding 3D volumes using either on-data snapping
    (default) or zero-padded off-data coordinates.

    Args:
        pc: Principal component data of shape [N_samples, N_components]
        pca: Fitted PCA object for inverse transformation
        dims: List of PC indices to analyze (e.g., [0, 1, 2] for PC0, PC1, PC2)
        model: Neural network model for volume generation
        dataset: Dataset containing pixel size information
        output_dir: Directory to save volumes
        noise_std: Noise standard deviation for volume generation
        bw_multiplier: Bandwidth multiplier for peak detection
        peak_threshold_rel: Relative threshold for peak detection as a fraction of max density
        not_on_data: If True, use zero-padded off-data coordinates; otherwise snap to nearest data point

    Returns:
        tuple: (peak_coords, num_peaks) - coordinates of detected peaks and count

    Example:
        >>> # Analyze 4D PC space (PC0, PC1, PC2, PC3)
        >>> peaks, n_peaks = analyze_nd_peaks(
        ...     pc, pca, [0, 1, 2, 3], model, dataset,
        ...     output_dir / "4d_analysis", noise_std=300.0
        ... )
        >>> print(f"Found {n_peaks} peaks in 4D space")
    """
    n = len(dims)
    plots_dir = output_dir / f"plots_{n}d"
    vols_root = output_dir / f"volumes_{n}d"
    mode_dir = "off_data" if not_on_data else "on_data"
    plots_dir.mkdir(exist_ok=True)
    vols_root.mkdir(exist_ok=True)

    # Detect peaks in N-D space
    print(f"Processing {n}D analysis: PC{dims}...")
    detector = PeakDetector(pc[:, dims], bw_multiplier=bw_multiplier, peak_threshold_rel=peak_threshold_rel)
    peak_coords = detector.get_peaks()
    print(f"  Found {len(peak_coords)} peaks")

    # Plot peaks
    print(f"  Plotting peaks, only PC{dims[:3]} are shown...")
    detector_plot = PeakDetector(pc[:, dims[: 3]], bw_multiplier=bw_multiplier, peak_threshold_rel=peak_threshold_rel)
    detector_plot.peak_coords = detector.peak_coords[:, : 3]
    fig = plot_peak_3d(
        detector_plot,
        xlabel=f"PC{dims[0]}",
        ylabel=f"PC{dims[1]}",
        zlabel=f"PC{dims[2]}",
        title=f"PC{dims[0]} & PC{dims[1]} & PC{dims[2]}",
    )
    fig.savefig(plots_dir / f"PC{dims[0]}_PC{dims[1]}_PC{dims[2]}.png")
    plt.close(fig)

    # Export peak volumes
    export_peak_volumes(
        peak_coords,
        dims,
        pca,
        model,
        dataset,
        vols_root / mode_dir,
        noise_std,
        pc=pc,
        not_on_data=not_on_data,
        volume_size=volume_size,
    )


# =============================================================================
# Main Entry Point
# =============================================================================


@app.command()
def main(
    output_dir: Annotated[Path, typer.Option("--out", "-o", help="Output directory", resolve_path=True)],
    config_path: Annotated[Path, typer.Option("--config", "-c", help="Config YAML file", exists=True)],
    checkpoint_path: Annotated[Path, typer.Option("--ckpt", "-p", help="Model checkpoint file", exists=True)],
    images_path: Annotated[Path, typer.Option("--images", "-i", help="HDF5 images dataset", exists=True)],
    max_pcs_2d: Annotated[int, typer.Option("--max-pcs-2d", help="Maximum number of PCs for 2D analysis")] = 4,
    max_pcs_3d: Annotated[int, typer.Option("--max-pcs-3d", help="Maximum number of PCs for 3D analysis")] = 4,
    pcs_nd: Annotated[str, typer.Option("--pcs-nd", help="PCs for N-dimensional analysis")] = "0 1 2 3",
    bw_multiplier: Annotated[float, typer.Option("--bw-multiplier", help="Bandwidth multiplier")] = None,
    peak_threshold_rel: Annotated[float, typer.Option("--peak-threshold-rel", help="Relative threshold for peak detection as a fraction of max density. Lower (e.g., 0.005) = more peaks (sensitive); higher (e.g., 0.05) = fewer peaks (stringent)")] = 0.01,
    not_on_data: Annotated[bool, typer.Option("--not-on-data", help="Use zero-padded off-data coordinates instead of snapping to nearest data point")] = False,
    volume_size: Annotated[int | None, typer.Option("--volume-size", help="Output volume spatial dimension (D×D×D). Default: use model's native size")] = None,
    noise_std: Annotated[float, typer.Option("--noise-std", help="Noise std for volume generation")] = 300.0,
) -> None:
    """Peak detection across PC combinations: 2D pairs, 3D triplets, N-D analysis.

    Systematically analyzes all PC pair/triplet combinations from first N components.
    For each combination, detects density peaks and exports volumes in two methods:
    1. Nearest point: volume from closest actual data point
    2. Zero-padded: volume with only specified dimensions, zeros elsewhere
    """
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print("=" * 70)

    # Setup output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load model and dataset
    print("\n1. Loading model and dataset...")
    model, dataset, _ = setup_system(
        config_path,
        checkpoint_path,
        images_path,
    )

    all_z_path = output_dir.parent / "all_z.pkl"
    all_z = encode_latent_vectors(model, dataset, all_z_path)
    print(f"Latent vectors shape: {all_z.shape}")

    # Step 2: Run dimensionality reduction
    print("\n2. Running dimensionality reduction...")
    pipeline = LatentAnalysisPipeline(all_z)
    pc = pipeline.run_reduction(method="pca")
    pca = pipeline.reduction_models["pca"]

    # Step 3a: Run 2D analysis
    print("\n3a. Creating 2D corner plot and detecting peaks...")
    analyze_peaks_2d(pc, pca, model, dataset, output_dir, noise_std, max_components=max_pcs_2d, bw_multiplier=bw_multiplier, peak_threshold_rel=peak_threshold_rel, not_on_data=not_on_data, volume_size=volume_size)

    # Step 3b: Run 3D analysis
    print("\n3b. Analyzing 3D PC triplets...")
    analyze_peaks_3d(pc, pca, model, dataset, output_dir, noise_std, max_components=max_pcs_3d, bw_multiplier=bw_multiplier, peak_threshold_rel=peak_threshold_rel, not_on_data=not_on_data, volume_size=volume_size)

    # Step 3c: Run nD analysis (example)
    pcs_nd_list = [int(x) for x in pcs_nd.split()]
    print(f"\n3c. Analyzing {len(pcs_nd_list)}D PC space...")
    analyze_peaks_nd(pc, pca, pcs_nd_list, model, dataset, output_dir, noise_std, bw_multiplier=bw_multiplier, peak_threshold_rel=peak_threshold_rel, not_on_data=not_on_data, volume_size=volume_size)

    # Summary
    mode_dir_name = "off_data" if not_on_data else "on_data"
    n = len(pcs_nd_list)
    print("\n" + "=" * 70)
    print("PEAK DETECTION COMPLETE")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    mode_label = "off_data (zero-padded)" if not_on_data else "on_data (snap to nearest)"
    print(f"\nGenerated files:")
    print(f"  Mode: {mode_label}")
    print(f"  2D Analysis:")
    print(f"    - pca_corner_plot.png (all PC pairs visualization)")
    print(f"    - plots_2d/ (individual high-resolution PC pair plots)")
    print(f"    - volumes_2d/ (peak volumes organized by PC pair/{mode_dir_name})")
    print(f"  3D Analysis:")
    print(f"    - plots_3d/ (3D visualizations for PC triplets)")
    print(f"    - volumes_3d/ (peak volumes organized by PC triplet/{mode_dir_name})")
    print(f"  {n}D Analysis:")
    print(f"    - volumes_{n}d/ (peak volumes for {n}D PC space/{mode_dir_name})")
    print("=" * 70)


if __name__ == "__main__":
    app()
