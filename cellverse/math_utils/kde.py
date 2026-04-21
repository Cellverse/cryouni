"""
This module provides functions to compute kernel density estimates using Gaussian kernels in PyTorch.

The code is adapted from scipy:
https://github.com/scipy/scipy/blob/main/scipy/stats/_kde.py
"""

from __future__ import annotations

import math
from typing import Callable

import torch

from ._constants import LOG_TAU


class GaussianKDE:
    """
    Representation of a kernel-density estimate using Gaussian kernels in PyTorch.

    This implementation follows scipy.stats.gaussian_kde.
    It includes automatic bandwidth determination and supports both uni-variate and multi-variate data.

    Args:
        `dataset` (torch.Tensor): Datapoints to estimate from.
            Shape (n, d) where n is the number of points and d is the number of dimensions.
        `bw_method` (str | None): The method used to calculate the bandwidth factor.
            This can be 'scott', 'silverman' or None.
            If a scalar, this will be used directly as `factor`.
            If None (default), 'scott' is used.
        `weights` (torch.Tensor | None): Weights of the datapoints. Shape (n,).
            If None, all samples are assumed to be equally weighted.

    Attributes:
        `dataset` (torch.Tensor): The dataset with which `GaussianKDE` was initialized.
        `n` (int): Number of data points.
        `d` (int): Number of dimensions.
        `neff` (float): Effective number of dataset.
        `factor` (float): The bandwidth factor obtained from `covariance_factor`.
    """

    def __init__(self, dataset: torch.Tensor, bw_method: str | None = None, weights: torch.Tensor | None = None) -> None:
        # Handle input shape: ensure (n, d)
        self.dataset = torch.atleast_2d(torch.as_tensor(dataset))
        if not self.dataset.numel() > 1:
            raise ValueError("`dataset` input should have multiple elements.")

        # Updated shape extraction: (n, d)
        self.n, self.d = self.dataset.shape

        if weights is not None:
            self._weights = torch.atleast_1d(torch.as_tensor(weights, dtype=self.dataset.dtype, device=self.dataset.device))
            self._weights /= self._weights.sum()

            if self.weights.ndim != 1:
                raise ValueError("`weights` input should be one-dimensional.")
            if len(self._weights) != self.n:
                raise ValueError(f"`weights` input should be of length {self.n}")

            self._neff = (1.0 / self._weights.square().sum()).item()

        if self.d > self.n:
            msg = (
                "Number of dimensions is greater than number of samples. "
                "This results in a singular data covariance matrix, "
                "which cannot be treated using the algorithms implemented in `GaussianKDE`."
            )
            raise ValueError(msg)

        # Triggers covariance and kernel distribution calculation
        try:
            self.set_bandwidth(bw_method=bw_method)
        except RuntimeError:
            msg = (
                "The data appears to lie in a lower-dimensional subspace of the space in which it is expressed. "
                "This results in a singular data covariance matrix, "
                "which cannot be treated using the algorithms implemented in `GaussianKDE`. "
                "Consider performing principal component analysis / dimensionality reduction and using `GaussianKDE` with the transformed data."
            )
            raise ValueError(msg)

    def evaluate(self, points: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the estimated pdf on a set of points.

        Args:
            `points` (torch.Tensor): Points to evaluate at.
                Shape (m, d) where m is the number of points and d is the number of dimensions.

        Returns:
            (torch.Tensor): Values at each point. Shape (m,)
        """
        return torch.exp(self.logpdf(points))

    __call__ = evaluate

    def scotts_factor(self) -> float:
        return math.pow(self.neff, -1.0 / (self.d + 4))

    def silverman_factor(self) -> float:
        return math.pow(self.neff * (self.d + 2) / 4.0, -1.0 / (self.d + 4))

    # Default method to calculate bandwidth, can be overwritten by subclass
    covariance_factor = scotts_factor
    covariance_factor.__doc__ = "Computes the bandwidth factor `factor`. The default method is `scotts_factor`."

    def set_bandwidth(self, bw_method: str | float | Callable | None = None) -> None:
        """Compute the bandwidth factor with a given method and re-calculate the covariance."""
        if bw_method is None:
            pass
        elif bw_method == "scott":
            self.covariance_factor = self.scotts_factor
        elif bw_method == "silverman":
            self.covariance_factor = self.silverman_factor
        elif isinstance(bw_method, (int, float)):
            self._bw_method = "use constant"
            self.covariance_factor = lambda: bw_method
        elif callable(bw_method):
            self._bw_method = bw_method
            self.covariance_factor = lambda: self._bw_method(self)
        else:
            msg = "`bw_method` should be 'scott', 'silverman', a scalar or a callable."
            raise ValueError(msg)

        self._compute_covariance()

    def _compute_covariance(self) -> None:
        """Computes the covariance matrix for each Gaussian kernel using covariance_factor()."""
        self.factor = self.covariance_factor()

        # Cache covariance and Cholesky decomp of covariance
        if not hasattr(self, "_data_cho_cov"):
            self._data_covariance = torch.atleast_2d(torch.cov(self.dataset.T, aweights=self.weights))
            self._data_cho_cov = torch.linalg.cholesky(self._data_covariance, upper=False)

            eye = torch.eye(self.d, dtype=self.dataset.dtype, device=self.dataset.device)
            self._data_inv_cho_cov = torch.linalg.solve_triangular(self._data_cho_cov, eye, upper=False)

        self.covariance = self._data_covariance * (self.factor ** 2)
        self.cho_cov = self._data_cho_cov * self.factor

        self.inv_cho_cov_t = (self._data_inv_cho_cov / self.factor).T
        self.inv_cov = self.inv_cho_cov_t @ self.inv_cho_cov_t.T

        self.log_det = 2 * self.cho_cov.diagonal().log().sum().item() + self.d * LOG_TAU
        self._dataset_whitened = self.dataset @ self.inv_cho_cov_t

    def pdf(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the estimated pdf on a provided set of points.
        """
        return self.evaluate(x)

    def logpdf(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the log of the estimated pdf on a provided set of points.

        Args:
            `x` (torch.Tensor): Points to evaluate. Shape (m, d).

        Returns:
            (torch.Tensor): Values at each point. Shape (m,)
        """
        points = torch.atleast_2d(torch.as_tensor(x, dtype=self.dataset.dtype, device=self.dataset.device))

        m, d = points.shape
        if d != self.d:
            raise ValueError(f"points have dimension {d}, dataset has dimension {self.d}")

        # Compute log probabilities:
        points_whitened = points @ self.inv_cho_cov_t
        dist_sq = torch.cdist(points_whitened, self._dataset_whitened).square()
        log_probs = -0.5 * (dist_sq + self.log_det)

        # Numerically stable weighted sum in log space:
        # log(sum(weight * exp(log_prob))) = logsumexp(log(weight) + log_prob)
        log_weights = torch.log(self.weights)
        log_pdf = torch.logsumexp(log_probs + log_weights, dim=1)

        return log_pdf

    def pdf_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the gradient of the estimated pdf at a provided set of points.

        Args:
            `x` (torch.Tensor): Points to evaluate. Shape (m, d).

        Returns:
            (torch.Tensor): Gradient at each point. Shape (m, d).
        """
        points = torch.atleast_2d(torch.as_tensor(x, dtype=self.dataset.dtype, device=self.dataset.device))

        m, d = points.shape
        if d != self.d:
            raise ValueError(f"points have dimension {d}, dataset has dimension {self.d}")

        # 1. Compute Probabilities (weights * N(x|mu, cov))
        # log_probs: (m, n)
        points_whitened = points @ self.inv_cho_cov_t
        dist_sq = torch.cdist(points_whitened, self._dataset_whitened).square()
        log_probs = -0.5 * (dist_sq + self.log_det)

        # weighted_probs: (m, n) corresponding to w_i * N(x; mu_i)
        log_weights = torch.log(self.weights)
        weighted_probs = torch.exp(log_probs + log_weights)

        # 2. Compute Difference Vectors (x - mu_i)
        # points: (m, d) -> (m, 1, d)
        # dataset (mu): (n, d) -> (1, n, d)
        # diffs: (m, n, d)
        diffs = points.unsqueeze(1) - self.dataset.unsqueeze(0)

        # 3. Compute Weighted Sum of Differences
        # Expand probs for broadcasting: (m, n, 1)
        # weighted_diffs: (m, n, d)
        weighted_diffs = weighted_probs.unsqueeze(-1) * diffs

        # Sum over kernels (dim 1): -> (m, d)
        sum_weighted_diffs = weighted_diffs.sum(dim=1)

        # 4. Multiply by Inverse Covariance (Precision Matrix)
        # Formula: - (Sum [ w_i * N_i * (x-mu_i) ]) @ Sigma^-1
        # (m, d) @ (d, d) -> (m, d)
        grad = -sum_weighted_diffs @ self.inv_cov

        return grad

    @property
    def weights(self) -> torch.Tensor:
        try:
            return self._weights
        except AttributeError:
            self._weights = self.dataset.new_ones(self.n) / self.n
            return self._weights

    @property
    def neff(self) -> float:
        try:
            return self._neff
        except AttributeError:
            self._neff = (1.0 / self.weights.square().sum()).item()
            return self._neff
