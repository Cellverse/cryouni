"""
GPU-Accelerated Latent Space Analysis Pipeline.
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

from hetero_recon.utils.io import save_pickle, save_volume
from hetero_recon.utils.latent_analysis import LatentAnalysisPipeline
from hetero_recon.utils.plot_utils import (
    plot_pca_2d,
    plot_pca_3d,
    plot_pca_density_2d,
    plot_pca_density_3d,
    plot_pca_explained_variance_ratio,
    plot_umap_2d,
    plot_umap_3d,
    plot_umap_density_2d,
    plot_umap_density_3d,
)

app = typer.Typer(
    help="Latent space analysis: GPU-accelerated dimensionality reduction, clustering, and visualization.",
    add_completion=False,
)


@app.command()
def main(
    output_dir: Annotated[Path, typer.Option("--out", "-o", help="Output directory", resolve_path=True)],
    config_path: Annotated[Path, typer.Option("--config", "-c", help="Config YAML file", exists=True)],
    checkpoint_path: Annotated[Path, typer.Option("--ckpt", "-p", help="Model checkpoint file", exists=True)],
    images_path: Annotated[Path, typer.Option("--images", "-i", help="HDF5 images dataset", exists=True)],
    cluster_num: Annotated[int, typer.Option("--cluster-num", "-k", help="Number of clusters")] = 10,
    noise_std: Annotated[float, typer.Option("--noise-std", help="Noise std for volume generation")] = 300.0,
) -> None:
    """GPU-accelerated latent space analysis: reduction, clustering, visualization.

    Performs complete LatentAnalysisPipeline workflow:
    1. Encode particles to latent vectors (or load cached)
    2. GPU-accelerated dimensionality reduction (UMAP & PCA)
    3. GMM clustering in high-dimensional space
    4. 2D/3D visualization with KDE density and cluster scatter
    5. Generate volumes for cluster centers
    6. Generate cluster membership indices
    """
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Cluster number: {cluster_num}")
    print("=" * 80)

    # Setup output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model and dataset
    print("\n[1/7] Loading model and dataset...")
    model, dataset, _ = setup_system(
        config_path,
        checkpoint_path,
        images_path,
    )

    # Inference to get latent vectors
    print("\n[2/7] Encoding latent vectors...")
    all_z_path = output_dir.parent / "all_z.pkl"
    all_z = encode_latent_vectors(model, dataset, all_z_path)
    print(f"Latent vectors shape: {all_z.shape}")

    # Initialize Pipeline
    print("\n[3/7] Initializing LatentAnalysisPipeline...")
    pipeline = LatentAnalysisPipeline(all_z)

    # Run Dimensionality Reduction
    print("\n[4/7] Running dimensionality reduction...")

    print("  - UMAP 2D...")
    z_umap_2d = pipeline.run_reduction(method="umap", n_components=2, n_neighbors=30, min_dist=0.01)

    print("  - PCA 2D...")
    z_pca_2d = pipeline.run_reduction(method="pca")

    print("  - UMAP 3D...")
    z_umap_3d = pipeline.run_reduction(method="umap", n_components=3, n_neighbors=30, min_dist=0.01)

    print("  - PCA 3D...")
    z_pca_3d = pipeline.run_reduction(method="pca")

    # Run GMM Clustering
    print(f"\n[5/7] Running GMM clustering with {cluster_num} clusters...")
    gmm_labels, gmm_centers = pipeline.run_clustering(
        cluster_num=cluster_num,
        covariance_type="full",
        random_state=0,
    )

    # Save clustering results
    save_pickle(gmm_labels, output_dir / "gmm_labels.pkl")
    save_pickle(gmm_centers, output_dir / "gmm_centers.pkl")

    # Print cluster summary
    summary = pipeline.get_cluster_summary()
    print("\nCluster Summary:")
    print(f"  Total samples: {len(all_z)}")
    print(f"  Number of clusters: {summary['cluster_num']}")
    for i, (size, prop) in enumerate(zip(summary['cluster_sizes'], summary['cluster_proportions'])):
        print(f"    Cluster {i}: {size:6d} samples ({prop * 100:5.2f}%)")

    # Generate Visualizations
    print("\n[6/7] Generating visualizations...")

    # 1. 2D PCA scatter
    print("  - 2D PCA scatter plot...")
    fig_pca_2d = plot_pca_2d(z_pca_2d, cluster_num, gmm_labels, pipeline.cluster_centers_indices)
    fig_pca_2d.savefig(output_dir / "pca_2d.png")
    plt.close(fig_pca_2d)

    # 2. 3D PCA scatter
    print("  - 3D PCA scatter plot...")
    fig_pca_3d = plot_pca_3d(z_pca_3d, cluster_num, gmm_labels, pipeline.cluster_centers_indices)
    fig_pca_3d.savefig(output_dir / "pca_3d.png")
    plt.close(fig_pca_3d)

    # 3. 2D UMAP scatter
    print("  - 2D UMAP scatter plot...")
    fig_umap_2d = plot_umap_2d(z_umap_2d, cluster_num, gmm_labels, pipeline.cluster_centers_indices)
    fig_umap_2d.savefig(output_dir / "umap_2d.png")
    plt.close(fig_umap_2d)

    # 4. 3D UMAP scatter
    print("  - 3D UMAP scatter plot...")
    fig_umap_3d = plot_umap_3d(z_umap_3d, cluster_num, gmm_labels, pipeline.cluster_centers_indices)
    fig_umap_3d.savefig(output_dir / "umap_3d.png")
    plt.close(fig_umap_3d)

    # 5. 2D PCA density
    print("  - 2D PCA density plot...")
    fig_pca_2d_density = plot_pca_density_2d(z_pca_2d)
    fig_pca_2d_density.savefig(output_dir / "pca_2d_density.png")
    plt.close(fig_pca_2d_density)

    # 6. 3D PCA density (using all 3 components)
    print("  - 3D PCA density plot...")
    fig_pca_3d_density = plot_pca_density_3d(z_pca_3d)
    fig_pca_3d_density.savefig(output_dir / "pca_3d_density.png")
    plt.close(fig_pca_3d_density)

    # 7. 2D UMAP density
    print("  - 2D UMAP density plot...")
    fig_umap_2d_density = plot_umap_density_2d(z_umap_2d)
    fig_umap_2d_density.savefig(output_dir / "umap_2d_density.png")
    plt.close(fig_umap_2d_density)

    # 8. 3D UMAP density (using all 3 components)
    print("  - 3D UMAP density plot...")
    fig_umap_3d_density = plot_umap_density_3d(z_umap_3d)
    fig_umap_3d_density.savefig(output_dir / "umap_3d_density.png")
    plt.close(fig_umap_3d_density)

    # 9. PCA explained variance
    print("  - PCA explained variance plot...")
    pca_model = pipeline.reduction_models["pca"]
    explained_var = pca_model.explained_variance_ratio_
    # Convert from cupy to numpy if needed
    if hasattr(explained_var, "get"):
        explained_var = explained_var.get()

    fig_pca_var = plot_pca_explained_variance_ratio(explained_var)
    fig_pca_var.savefig(output_dir / "pca_explained_variance_ratio.png")
    plt.close(fig_pca_var)

    # Generate volumes for cluster centers
    print("\n[7/7] Generating volumes and splitting star file...")

    gmm_vol_dir = output_dir / "gmm_vols"
    gmm_vol_dir.mkdir(parents=True, exist_ok=True)

    print("  - Generating volumes for cluster centers...")
    for i, center_z in enumerate(tqdm(gmm_centers, desc="Volumes")):
        zval = torch.from_numpy(center_z).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        volume = model.volume.eval_volume(zval, noise_std=noise_std, radius=None)
        save_volume(gmm_vol_dir / f"volume.{i:03d}.mrc", volume.detach().cpu().numpy(), dataset.psize_A)

    # Save cluster indices as pickle files (instead of writing .star files)
    print("  - Saving cluster indices as pickle files by clusters...")
    for i in range(cluster_num):
        save_path = output_dir / f"cluster_{i:03d}.pkl"
        indices = np.where(gmm_labels == i)[0].tolist()
        save_pickle(indices, save_path)
        print(f"    Cluster {i}: {len(indices)} particles -> {save_path.name}")

    # Summary
    print("\n" + "=" * 80)
    print("LATENT SPACE ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Detected {cluster_num} conformational clusters")
    print(f"Output directory: {output_dir}")
    print(f"\nGenerated files:")
    print(f"  Data:")
    print(f"    - all_z.pkl (latent vectors)")
    print(f"    - gmm_labels.pkl, gmm_centers.pkl (clustering results)")
    print(f"  Visualizations (9 images):")
    print(f"    - pca_2d.png (2D PCA scatter with GMM clusters)")
    print(f"    - pca_3d.png (3D PCA scatter with GMM clusters)")
    print(f"    - umap_2d.png (2D UMAP scatter with GMM clusters)")
    print(f"    - umap_3d.png (3D UMAP scatter with GMM clusters)")
    print(f"    - pca_2d_density.png (2D PCA density)")
    print(f"    - pca_3d_density.png (3D PCA density)")
    print(f"    - umap_2d_density.png (2D UMAP density)")
    print(f"    - umap_3d_density.png (3D UMAP density)")
    print(f"    - pca_explained_var.png (PCA explained variance ratio)")
    print(f"  Volumes:")
    print(f"    - gmm_vols/volume.*.mrc ({cluster_num} cluster center volumes)")
    print(f"  Cluster indices:")
    print(f"    - cluster_*.pkl ({cluster_num} split index pickle files)")
    print("=" * 80)


if __name__ == "__main__":
    app()
