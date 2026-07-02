"""
Watershed Segmentation CLI.

Performs density-based watershed clustering on reduced latent space.
Supports N-dimensional input with visualization for 2D/3D.
"""

from __future__ import annotations

from pathlib import Path
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from typing_extensions import Annotated

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from cli.utils.common import encode_latent_vectors, setup_system

from hetero_recon.utils.io import save_pickle
from hetero_recon.utils.latent_analysis import LatentAnalysisPipeline
from hetero_recon.utils.plot_utils import plot_watershed_2d, plot_watershed_3d
from hetero_recon.utils.watershed_segmenter import WatershedSegmenter

app = typer.Typer(
    help="Watershed segmentation: density-based clustering of conformational states.",
    add_completion=False,
)


def _save_delta_results(
    delta_val: float,
    labels: np.ndarray,
    peaks: np.ndarray,
    pc_selected: np.ndarray,
    pc_dims_list: list,
    ndim: int,
    delta_dir: Path,
    segmentation: np.ndarray = None,
    grid: np.ndarray = None,
    density: np.ndarray = None,
) -> None:
    """Save results for a specific delta_delta_g value."""
    # Save clustering results
    save_pickle(labels, delta_dir / "watershed_labels.pkl")
    save_pickle(peaks, delta_dir / "watershed_peaks.pkl")

    n_clusters = len(np.unique(labels[labels >= 0]))

    # Visualization (2D/3D only)
    if ndim <= 3:
        if ndim == 2:
            fig = plot_watershed_2d(delta_val, pc_selected, peaks, segmentation, grid, density, pc_dims=pc_dims_list)
        else:
            fig = plot_watershed_3d(pc_selected, labels, peaks, pc_dims=pc_dims_list)
        fig.savefig(delta_dir / "watershed.png")
        plt.close(fig)

    # Save cluster indices
    for i in range(-1, n_clusters):
        save_path = delta_dir / f"cluster_{i:03d}.pkl" if i >= 0 else delta_dir / "cluster_-1.pkl"
        indices = np.where(labels == i)[0].tolist()
        save_pickle(indices, save_path)


@app.command()
def main(
    output_dir: Annotated[Path, typer.Option("--out", "-o", help="Output directory", resolve_path=True)],
    config_path: Annotated[Path, typer.Option("--config", "-c", help="Config YAML file", exists=True)],
    checkpoint_path: Annotated[Path, typer.Option("--ckpt", "-p", help="Model checkpoint file", exists=True)],
    images_path: Annotated[Path, typer.Option("--images", "-i", help="HDF5 images dataset", exists=True)],
    pc_dims: Annotated[str, typer.Option("--pc-dims", "-d", help="PC dimensions for N-dimensional analysis")] = "0 1",
    delta_delta_g: Annotated[str, typer.Option("--delta-delta-g", "-g", help="Energy difference in kT.")] = "1 2 3 4 None",
    gt_labels_path: Annotated[Path | None, typer.Option("--gt-labels", help="Path to ground truth labels pkl file")] = None,
    resolution: Annotated[int, typer.Option("--resolution", "-r", help="KDE resolution")] = None,
    bw_multiplier: Annotated[float | None, typer.Option("--bw-multiplier", help="Bandwidth multiplier for KDE")] = None,
    peak_threshold_rel: Annotated[float, typer.Option("--peak-threshold-rel", help="Relative threshold for peak detection as a fraction of max density")] = 0.01,
) -> None:
    """
    Watershed-based clustering in reduced space.

    Performs density-based clustering using watershed algorithm on selected
    principal components. Supports N-dimensional input (visualization for 2D/3D only).
    """
    # Determine delta_delta_g values to process
    delta_delta_g_values = [float(x) if x.lower() != "none" else None for x in delta_delta_g.split()]

    # Parse PC dimensions
    pc_dims_list = [int(x) for x in pc_dims.split()]
    ndim = len(pc_dims_list)
    print("=" * 70)
    print(f"WATERSHED SEGMENTATION: {ndim}D (PC{', PC'.join(map(str, pc_dims_list))})")
    print(f"Delta_delta_g values: {delta_delta_g_values}")
    print("=" * 70)

    # Setup output directory
    output_dir.mkdir(parents=True, exist_ok=True)

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
    print(f"PCA shape: {pc.shape}")

    # Create PC-named subfolder
    dims_str = "_".join(f"PC{d}" for d in pc_dims_list)
    watershed_base_dir = output_dir / dims_str
    watershed_base_dir.mkdir(parents=True, exist_ok=True)

    # Extract selected dimensions
    pc_selected = pc[:, pc_dims_list]

    # Initialize watershed segmenter once (reused for all delta_delta_g values)
    print(f"\n[4/5] Initializing {ndim}D watershed segmenter...")
    segmenter = WatershedSegmenter(
        points=pc_selected,
        latent_vectors=all_z,
        resolution=resolution,
        bw_multiplier=bw_multiplier,
        peak_threshold_rel=peak_threshold_rel,
    )

    # Process each delta_delta_g value
    print("\n[5/5] Running segmentation for each delta_delta_g value...")
    for delta_val in delta_delta_g_values:
        delta_str = f"delta_{delta_val}" if delta_val is not None else "delta_none"
        print(f"\n  Processing {delta_str}...")

        # Create delta-specific subdirectory
        delta_dir = watershed_base_dir / delta_str
        delta_dir.mkdir(parents=True, exist_ok=True)

        # Run segmentation with current delta_delta_g
        labels = segmenter.segment(delta_delta_g=delta_val)
        peaks = segmenter.get_peaks()
        grid, density, segmentation = segmenter.get_visualization_data()
        n_clusters = len(np.unique(labels[labels >= 0]))

        print(f"    Found {n_clusters} clusters, {len(peaks)} peaks")
        for cluster_id in range(n_clusters):
            count = int(np.sum(labels == cluster_id))
            print(f"      Cluster {cluster_id}: {count} particles ({count / len(labels) * 100:.2f}%)")

        # GT label analysis if provided
        reassigned_labels = labels.copy()
        if gt_labels_path is not None and gt_labels_path.exists():
            print(f"    Analyzing GT labels and reassigning...")

            # Load GT labels
            with open(gt_labels_path, 'rb') as f:
                gt_labels = pickle.load(f)
            gt_labels = np.array(gt_labels)
            reassigned_peaks = np.zeros((np.unique(gt_labels).shape[0], peaks.shape[1]))

            # Analyze each cluster's GT label distribution
            cluster_stats = []
            cluster_reassignment = {}

            for cluster_id in range(n_clusters):
                mask = labels == cluster_id
                cluster_gt_labels = gt_labels[mask]

                # Compute histogram
                unique_gt, counts = np.unique(cluster_gt_labels, return_counts=True)

                # Find most frequent GT label
                max_idx = np.argmax(counts)
                dominant_gt_label = unique_gt[max_idx]
                dominant_count = counts[max_idx]
                total_count = len(cluster_gt_labels)

                # Store statistics
                for gt_label, count in zip(unique_gt, counts):
                    cluster_stats.append({
                        'delta': delta_str,
                        'cluster_id': dominant_gt_label,
                        'gt_label': int(gt_label),
                        'count': int(count),
                        'percentage': float(count / total_count * 100)
                    })

                # Reassign cluster label to dominant GT label
                cluster_reassignment[cluster_id] = dominant_gt_label
                reassigned_labels[mask] = dominant_gt_label
                reassigned_peaks[dominant_gt_label] = peaks[cluster_id]

                print(
                    f"      Cluster {cluster_id}: {total_count} particles, dominant GT label {dominant_gt_label} ({dominant_count}/{total_count}, {dominant_count/total_count*100:.1f}%)"
                )

            # Save GT analysis results
            stats_df = pd.DataFrame(cluster_stats)
            stats_csv_path = delta_dir / "cluster_gt_analysis.csv"
            stats_df.to_csv(stats_csv_path, index=False)

            # Save reassignment mapping
            reassignment_path = delta_dir / "cluster_reassignment.pkl"
            save_pickle(cluster_reassignment, reassignment_path)

            # Save reassigned labels
            reassigned_labels_path = delta_dir / "reassigned_labels.pkl"
            save_pickle(reassigned_labels[labels >= 0], reassigned_labels_path)

            # Save masked GT labels
            masked_gt_labels_path = delta_dir / "masked_gt_labels.pkl"
            save_pickle(gt_labels[labels >= 0], masked_gt_labels_path)

            # Save reassigned peaks
            reassigned_peak_path = delta_dir / "reassigned_peaks.pkl"
            save_pickle(reassigned_peaks, reassigned_peak_path)

            # Save reassigned cluster indices
            reassigned_unique = np.unique(reassigned_labels[reassigned_labels >= 0])
            for gt_label in reassigned_unique:
                mask = reassigned_labels == gt_label
                cluster_indices = np.where(mask)[0]

                reassigned_index_path = delta_dir / f"reassigned_index_{int(gt_label):03d}.pkl"
                save_pickle(cluster_indices, reassigned_index_path)

            # Plot with reassigned labels if GT provided
            if ndim <= 3:
                if ndim == 2:
                    fig = plot_watershed_2d(delta_val, pc_selected, reassigned_peaks, segmentation, grid, density, pc_dims=pc_dims_list)
                else:
                    fig = plot_watershed_3d(pc_selected, reassigned_labels, reassigned_peaks, pc_dims=pc_dims_list)
                fig.savefig(delta_dir / "watershed_reassigned.png", dpi=150, bbox_inches="tight")
                plt.close(fig)

        # Save results for this delta value
        _save_delta_results(
            delta_val,
            labels,
            peaks,
            pc_selected,
            pc_dims_list,
            ndim,
            delta_dir,
            segmentation=segmentation,
            grid=grid,
            density=density,
        )
        print(f"    Saved results to {delta_str}/")

    # Summary
    print("\n" + "=" * 70)
    print("WATERSHED SEGMENTATION COMPLETE")
    print("=" * 70)
    print(f"Output directory: {watershed_base_dir}")
    print(f"\nGenerated subdirectories:")
    for delta_val in delta_delta_g_values:
        delta_str = f"delta_{delta_val}" if delta_val is not None else "delta_none"
        print(f"  - {delta_str}/")
    print("=" * 70)


if __name__ == "__main__":
    app()
