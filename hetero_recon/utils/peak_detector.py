from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter
import torch

from hetero_recon.utils.kde_gpu import GaussianKDE


class PeakDetector:
    """
    Dimension-agnostic peak detector using GPU-accelerated KDE.

    This class computes density maps from point clouds and identifies
    local maxima (peaks) representing distinct conformational states or clusters.

    Attributes:
        `points` (np.ndarray): Input point coordinates of shape [N, D].
        `kde` (GaussianKDE): KDE instance for density computation.
        `grid` (np.ndarray): Coordinate grid of shape [res, ..., res, D].
        `density` (np.ndarray): Computed density values.
        `peak_indices` (np.ndarray): Grid indices of detected peaks.
        `peak_coords` (np.ndarray): Actual coordinates of detected peaks.

    Example:
        >>> # Initialize and detect peaks
        >>> detector = PeakDetector(points_2d)
        >>> print(f"Found {len(detector.get_peaks())} peaks")
        >>>
        >>> # Access density grid for visualization
        >>> grid, density = detector.get_density_grid()
    """

    def __init__(self, points: np.ndarray, resolution: int | None = None, bw_multiplier: float | None = None) -> None:
        self.points = points
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize KDE and compute density on GPU
        self.kde = GaussianKDE(torch.from_numpy(points).float().to(device), bw_multiplier=bw_multiplier)

        # Generate coordinate grid and evaluate PDF
        grid_torch = self.kde.get_grid(resolution=resolution)
        density_torch = self.kde.pdf(grid_torch)

        # Transfer results to CPU for post-processing
        self.grid = grid_torch.cpu().numpy()
        self.density = density_torch.cpu().numpy()

        # Identify peaks using automated local maxima detection
        self.peak_indices = self._find_maxima(self.density)
        self.peak_coords = self._indices_to_coords(self.peak_indices)

    def get_peaks(self) -> np.ndarray:
        return self.peak_coords

    def get_density_grid(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the computed density grid with coordinate grids.

        Returns:
            tuple:
                - grid (np.ndarray): Coordinate grid of shape [res, ..., res, D].
                - density (np.ndarray): Density values of shape [res, ..., res].
        """
        return self.grid, self.density

    def _indices_to_coords(self, indices: np.ndarray) -> np.ndarray:
        """
        Convert grid indices to actual coordinates.

        Args:
            `indices` (np.ndarray): Grid indices of shape [N_peaks, D].

        Returns:
            np.ndarray: Coordinates of shape [N_peaks, D].
        """
        return self.grid[tuple(indices.T)]

    @staticmethod
    def _find_maxima(density: np.ndarray) -> np.ndarray:
        """
        Find local maxima in the density grid using statistical thresholding.

        Args:
            `density` (np.ndarray): Density grid of arbitrary shape.

        Returns:
            (np.ndarray): Indices of local maxima of shape [N_peaks, D].
        """
        win_size = max(4, max(density.shape) // 64)

        # Apply maximum filter
        density_max = maximum_filter(density, size=win_size, mode="constant")

        # Identify peaks: points that equal the filtered max and exceed threshold
        threshold = density.max() * 0.01

        # A point is a peak if it equals the local maximum and exceeds the threshold
        maxima_mask = (density == density_max) & (density > threshold)

        # Return indices of peaks
        maxima_points = np.argwhere(maxima_mask)
        return maxima_points
