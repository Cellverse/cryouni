from __future__ import annotations

import torch

from cellverse.math_utils.kde import GaussianKDE as G

KDE_RESOLUTION_2D = 256
KDE_RESOLUTION_3D = 128
KDE_RESOLUTION_4D = 64
KDE_RESOLUTION_ND = 16


class GaussianKDE(G):
    """
    A wrapper for the GaussianKDE class.
    """

    def __init__(self, dataset, bw_method: str = None, bw_multiplier: float = None, weights: torch.Tensor = None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset = torch.as_tensor(dataset, device=device)

        if bw_multiplier is not None:
            if bw_method is None or bw_method == "scott":
                bw_method = lambda kde: kde.scotts_factor() * bw_multiplier
            elif bw_method == "silverman":
                bw_method = lambda kde: kde.silverman_factor() * bw_multiplier
            elif callable(bw_method):
                bw_method = lambda kde: bw_method(kde) * bw_multiplier
            else:
                bw_method = bw_method * bw_multiplier

        super().__init__(dataset, bw_method, weights)

    def get_grid(self, resolution: int = None, pad_ratio: float = 0.01) -> torch.Tensor:
        """
        Generate grid covering the data range with padding (dimension-agnostic).

        Args:
            `resolution` (int): Grid resolution per dimension.
                If not specified, it will be set based on data dimensionality.
            `pad_ratio` (float): Padding ratio relative to data range (default: 0.01).

        Returns:
            Stacked grid tensor of shape:
                - For 2D: [resolution, resolution, 2]
                - For 3D: [resolution, resolution, resolution, 3]
                - For ND: [resolution, ..., resolution, D] (D dimensions stacked in last axis)

        Example:
            >>> kde = KernelDensityEstimatorGPU(data_2d) # data_2d is (N, 2)
            >>> grid = kde.get_grid(resolution=128)
            >>> grid.shape                               # [128, 128, 2]

            >>> kde = KernelDensityEstimatorGPU(data_3d) # data_3d is (N, 3)
            >>> grid = kde.get_grid(resolution=64)
            >>> grid.shape                               # [64, 64, 64, 3]
        """
        if resolution is None:
            if self.d == 2:
                resolution = KDE_RESOLUTION_2D
            elif self.d == 3:
                resolution = KDE_RESOLUTION_3D
            elif self.d == 4:
                resolution = KDE_RESOLUTION_4D
            else:
                resolution = KDE_RESOLUTION_ND

        # Calculate min/max and padding for each dimension
        ranges = []
        for dim in range(self.d):
            dim_min = self.dataset[:, dim].min().item()
            dim_max = self.dataset[:, dim].max().item()
            padding = (dim_max - dim_min) * pad_ratio
            ranges.append(torch.linspace(dim_min - padding, dim_max + padding, resolution, device=self.dataset.device))

        # Generate meshgrid for all dimensions: [..., D]
        grid = torch.stack(torch.meshgrid(*ranges, indexing="ij"), dim=-1)

        return grid

    def evaluate(self, points: torch.Tensor, batch_size: int = 1024) -> torch.Tensor:
        return torch.exp(self.logpdf(points, batch_size=batch_size))

    __call__ = evaluate

    def pdf(self, x: torch.Tensor, batch_size: int = 1024) -> torch.Tensor:
        return self.evaluate(x, batch_size=batch_size)

    def logpdf(self, x: torch.Tensor, batch_size: int = 1024) -> torch.Tensor:
        original_shape = x.shape[:-1]
        x_reshaped = x.reshape(-1, self.d).to(self.dataset.device)
        flattned_shape = len(x_reshaped)

        logpdf = x.new_empty(flattned_shape)
        for i in range(0, flattned_shape, batch_size):
            logpdf[i : i + batch_size] = super().logpdf(x_reshaped[i : i + batch_size])
        return logpdf.reshape(original_shape)

    def pdf_gradient(self, x: torch.Tensor, batch_size: int = 1024) -> torch.Tensor:
        original_shape = x.shape
        x_reshaped = x.reshape(-1, self.d).to(self.dataset.device)
        flattned_shape = len(x)

        gradient = torch.empty_like(x)
        for i in range(0, flattned_shape, batch_size):
            gradient[i : i + batch_size] = super().pdf_gradient(x_reshaped[i : i + batch_size])
        return gradient.reshape(original_shape)
