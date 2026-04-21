from typing import Literal

import numpy as np
import torch

from cellverse.math_utils.gmm import GaussianMixture

try:
    from cuml.decomposition import PCA
    from cuml.manifold import UMAP
except ImportError:
    import warnings
    warnings.warn("cuML not available. Falling back to CPU-based sklearn. Performance will be significantly slower.")

    from sklearn.decomposition import PCA
    from umap import UMAP

from scipy.spatial.distance import cdist


class LatentAnalysisPipeline:
    """
    GPU-accelerated pipeline for latent space analysis.

    This class manages dimensionality reduction and clustering of high-dimensional
    latent vectors using GPU-accelerated algorithms.

    Attributes:
        `data` (np.ndarray): Original high-dimensional latent vectors of shape (N, D).
        `reduced_data` (dict): Dictionary storing reduced representations.
        `cluster_num` (int): Number of clusters.
        `cluster_labels` (np.ndarray): GMM cluster assignments.
        `cluster_centers` (np.ndarray): GMM cluster centers in original space.

    Args:
        `data` (np.ndarray): Latent vectors of shape (N, D) where N is the number
            of samples and D is the dimensionality.

    Example:
        >>> # Initialize with latent vectors
        >>> pipeline = LatentAnalysisPipeline(latent_vectors)
        >>>
        >>> # Run dimensionality reduction
        >>> umap_2d = pipeline.run_reduction(method='umap', n_components=2)
        >>>
        >>> # Perform clustering
        >>> labels, centers = pipeline.run_clustering(cluster_num=10)
    """

    def __init__(self, data: np.ndarray) -> None:
        self.data = np.asanyarray(data)

        # Storage for reduced representations
        self.reduced_data = {}
        self.reduction_models = {}

        # Storage for clustering results
        self.cluster_num: int | None = None
        self.cluster_labels: np.ndarray | None = None
        self.cluster_centers: np.ndarray | None = None
        self.cluster_centers_indices: np.ndarray | None = None

    def run_reduction(self, method: Literal["umap", "pca"] = "umap", n_components: int | None = None, **kwargs) -> np.ndarray:
        """
        Execute GPU-accelerated dimensionality reduction.

        Args:
            `method` (str): Reduction method, either 'umap' or 'pca'. Default: 'umap'
            `n_components` (int, optional): Number of dimensions in reduced space.
                For UMAP: required (default: 2).
                For PCA: Fixed to full dimensions for internal optimization.
            `**kwargs`: Additional parameters passed to the reduction algorithm.

        Returns:
            `reduced` (np.ndarray): Reduced representation of shape (N, n_components).

        Example:
            >>> # UMAP with custom parameters
            >>> umap_2d = pipeline.run_reduction(
            ...     method='umap',
            ...     n_components=2,
            ...     n_neighbors=30,
            ...     min_dist=0.01
            ... )
            >>>
            >>> # PCA reduction (uses all components)
            >>> pca_all = pipeline.run_reduction(method='pca', n_components=None)
        """
        # For PCA, use string key without n_components if None
        if method.lower() == "pca":
            cache_key = f"{method}"
        else:
            cache_key = f"{method}_{n_components}"

        # Return cached result if available
        if cache_key in self.reduced_data:
            return self.reduced_data[cache_key]

        # Perform reduction
        if method.lower() == "umap":
            umap_params = {
                "n_components": n_components
            }
            umap_params.update(kwargs)

            reducer = UMAP(**umap_params)
            reduced = reducer.fit_transform(self.data)

        elif method.lower() == "pca":
            reducer = PCA(**kwargs)
            reduced = reducer.fit_transform(self.data)

        else:
            raise ValueError(f"Unknown reduction method: {method}. Use 'umap' or 'pca'.")

        # Cache results
        self.reduced_data[cache_key] = reduced
        self.reduction_models[cache_key] = reducer

        return reduced

    def run_clustering(
        self,
        cluster_num: int = 10,
        covariance_type: str = "full",
        random_state: int | None = None,
        on_data: bool = True,
        **kwargs
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform GMM clustering on the high-dimensional latent space.

        Clustering is performed on the original high-dimensional data, not the
        reduced representation, to preserve all information for cluster assignment.

        Args:
            `cluster_num` (int): Number of Gaussian components. Default: 10
            `covariance_type` (str): Type of covariance parameters.
                Options: 'full', 'tied', 'diag', 'spherical'. Default: 'full'.
            `random_state` (int, optional): Random seed for reproducibility.
            `on_data` (bool): If True, compute cluster centers as nearest points on
                the data manifold. Default: True.
            `**kwargs`: Additional parameters passed to GaussianMixture.

        Returns:
            (tuple[np.ndarray, np.ndarray]):
                - labels: Cluster assignments of shape (N,)
                - centers: Cluster centers of shape (K, D) in original space

        Example:
            >>> labels, centers = pipeline.run_clustering(
            ...     cluster_num=10,
            ...     covariance_type='full',
            ...     random_state=42
            ... )
            >>> print(f"Found {len(np.unique(labels))} clusters")
        """
        gmm_params = {
            "n_components": cluster_num,
            "covariance_type": covariance_type,
            "random_state": random_state,
        }
        gmm_params.update(kwargs)

        # Initialize GMM model
        gmm = GaussianMixture(**gmm_params)

        # Handle device transfer
        data = torch.from_numpy(self.data).to("cuda" if torch.cuda.is_available() else "cpu")
        labels = gmm.fit_predict(data).cpu().numpy()
        centers = gmm.means_.cpu().numpy()

        # Find nearest data points to cluster centers if requested
        if on_data:
            centers, centers_indices = self._get_nearest_point(self.data, centers)
        else:
            centers_indices = None

        # Store results
        self.cluster_num = cluster_num
        self.cluster_labels = labels
        self.cluster_centers = centers
        self.cluster_centers_indices = centers_indices

        return labels, centers

    @staticmethod
    def _get_nearest_point(data: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Find closest points in data to query points.

        Args:
            `data` (np.ndarray): Data points of shape (N, D).
            `query` (np.ndarray): Query points of shape (K, D).

        Returns:
            `nearest` (np.ndarray): Nearest data points of shape (K, D).
            `indices` (np.ndarray): Indices of nearest points of shape (K,).
        """
        indices = cdist(query, data).argmin(axis=1)
        return data[indices], indices

    def get_cluster_summary(self) -> dict:
        """
        Get summary statistics about the clustering results.

        Returns:
            (dict): Dictionary containing:
                - cluster_num: Number of clusters
                - cluster_sizes: Array of cluster sizes
                - cluster_proportions: Array of cluster proportions

        Example:
            >>> summary = pipeline.get_cluster_summary()
            >>> print(f"Cluster sizes: {summary['cluster_sizes']}")
        """
        if self.cluster_labels is None:
            raise RuntimeError("Must run run_clustering() before getting summary")

        _, counts = np.unique(self.cluster_labels, return_counts=True)
        proportions = counts / len(self.cluster_labels)

        return {
            "cluster_num": self.cluster_num,
            "cluster_sizes": counts,
            "cluster_proportions": proportions
        }
