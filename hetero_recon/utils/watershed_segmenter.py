"""Watershed segmentation on reduced latent space."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from skimage.segmentation import watershed

from hetero_recon.utils.peak_detector import PeakDetector


class WatershedSegmenter:
    """Watershed segmentation on reduced space."""

    def __init__(
        self,
        points: np.ndarray,
        latent_vectors: np.ndarray = None,
        resolution: int = None,
        bw_multiplier: float | None = None,
        peak_threshold_rel: float = 0.01,
    ) -> None:
        """
        Args:
            `points` (np.ndarray): Reduced data of shape [N, D].
            `latent_vectors` (np.ndarray): Original latent vectors of shape [N, latent_dim].
            `resolution` (int): Grid resolution.
            `bw_multiplier` (float): Bandwidth multiplier for KDE.
            `peak_threshold_rel` (float): Relative threshold for peak detection
                as a fraction of the maximum density. Default: 0.01.
        """
        self.points = np.asarray(points)
        self.latent_vectors = latent_vectors
        self.ndim = self.points.shape[1]

        # Use PeakDetector for consistent peak detection
        self.detector = PeakDetector(self.points, resolution=resolution, bw_multiplier=bw_multiplier, peak_threshold_rel=peak_threshold_rel)
        self.grid, self.density = self.detector.get_density_grid()

        self._peak_indices = None
        self._peak_coords = None
        self._segmentation = None
        self._labels = None
        self._n_clusters = 0

    def segment(self, delta_delta_g: float | None = None) -> np.ndarray:
        """
        Args:
            `delta_delta_g` (float | None): Maximum allowed energy difference in kT.

        Returns:
            (np.ndarray): Cluster labels of shape [N], -1 = excluded.
        """
        # Use PeakDetector to find peaks
        self._peak_coords = self.detector.get_peaks()
        self._peak_indices = self.detector.peak_indices

        self._n_clusters = len(self._peak_indices)

        markers = np.zeros(self.density.shape, dtype=np.int32)
        for i, idx in enumerate(self._peak_indices):
            markers[tuple(idx)] = i + 1

        # Use threshold (0.001 by default)
        mask = np.ones_like(self.density, dtype=bool)
        self._segmentation = watershed(-self.density, markers, mask=mask)

        self._labels = self._map_labels_to_particles()

        # Shift labels for background
        self._labels = self._labels - 1

        if delta_delta_g is not None and delta_delta_g > 0:
            particle_densities = self._get_particle_densities()

            # Iterate through each detected peak (i.e., each cluster)
            for cluster_label, peak_idx in enumerate(self._peak_indices):
                rho_max = self.density[tuple(peak_idx)]
                min_density_threshold = rho_max * np.exp(-delta_delta_g)

                # Create a mask for all particles belonging to the current cluster
                cluster_mask = self._labels == cluster_label
                exclude_mask = cluster_mask & (particle_densities < min_density_threshold)

                # Mark unqualified particles as -1 (background/excluded)
                self._labels[exclude_mask] = -1
        else:
            # Hard classification for outliers
            outlier_mask = self._labels == -1

            if np.any(outlier_mask):
                outlier_points = self.points[outlier_mask]

                # Calculate squared Euclidean distances from outliers to each peak
                # outlier_points shape: [N_outliers, D]
                # self._peak_coords shape: [N_clusters, D]
                dist_sq = cdist(outlier_points, self._peak_coords) ** 2

                # Find the index of the closest peak for each outlier.
                # Conveniently, peak indices (0, 1, 2...) exactly match the cluster labels!
                nearest_labels = np.argmin(dist_sq, axis=1)

                # Reassign the outliers to their nearest valid cluster
                self._labels[outlier_mask] = nearest_labels

        return self._labels

    def get_labels(self) -> np.ndarray:
        return self._labels

    def get_peaks(self) -> np.ndarray:
        return self._peak_coords

    def get_visualization_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.grid, self.density, self._segmentation

    @property
    def n_clusters(self) -> int:
        return self._n_clusters

    def _coords_to_indices(self, coords: np.ndarray) -> np.ndarray:
        """
        Map coordinates to nearest grid indices.

        The grid is structured as meshgrid with indexing='ij':
        - For 2D: grid[i, j, 0] = x_range[i], grid[i, j, 1] = y_range[j]
        - For 3D: grid[i, j, k, 0] = x_range[i], grid[i, j, k, 1] = y_range[j], grid[i, j, k, 2] = z_range[k]
        """
        indices = []
        grid_shape = self.density.shape # Shape without the last coordinate dimension

        for d in range(self.ndim):
            # Extract 1D array of coordinate values for dimension d
            # For dimension d, we need to vary index along axis d while keeping others at 0
            idx_tuple = tuple([slice(None) if i == d else 0 for i in range(self.ndim)] + [d])
            grid_1d = self.grid[idx_tuple]

            # Get min and max
            grid_min = grid_1d[0]
            grid_max = grid_1d[-1]

            # Normalize and map to indices (same as reference)
            normalized = (coords[:, d] - grid_min) / (grid_max - grid_min)
            idx = (normalized * (grid_shape[d] - 1)).round().astype(int)
            idx = np.clip(idx, 0, grid_shape[d] - 1)
            indices.append(idx)

        return np.stack(indices, axis=1)

    def _map_labels_to_particles(self) -> np.ndarray:
        indices = self._coords_to_indices(self.points)
        labels = self._segmentation[tuple(indices.T)]
        return labels.astype(np.int32)

    def _get_particle_densities(self) -> np.ndarray:
        indices = self._coords_to_indices(self.points)
        return self.density[tuple(indices.T)]
