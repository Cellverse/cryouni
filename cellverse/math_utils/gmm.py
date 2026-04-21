"""
This module provides functions to compute Gaussian Mixture Model (GMM) using PyTorch.

The code is adapted from scikit-learn:
https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/mixture/_gaussian_mixture.py
"""

from __future__ import annotations

import math
from typing import Literal
import warnings

import torch

from ._constants import LOG_TAU


def _check_data(X: torch.Tensor, device: torch.device) -> torch.Tensor:
    if not torch.is_tensor(X):
        X = torch.as_tensor(X)

    # Ensure at least float32
    if X.dtype not in [torch.float32, torch.float64]:
        X = X.float()

    # Ensure 2D array
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got {X.ndim}D array instead")

    X = X.to(device)
    return X


def _check_weights(weights: torch.Tensor, n_components: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if not torch.is_tensor(weights):
        weights = torch.as_tensor(weights)

    if weights.shape != (n_components,):
        raise ValueError(f"The parameter 'weights' should have the shape of ({n_components},), but got {weights.shape}")

    # Check range [0, 1]
    if (weights < 0.0).any() or (weights > 1.0).any():
        raise ValueError(
            f"The parameter 'weights' should be in the range [0, 1], but got max value {weights.max():.5f}, min value {weights.min():.5f}"
        )

    # Check normalization
    atol = 1e-6 if weights.dtype == torch.float32 else 1e-8
    if not torch.allclose(torch.abs(1.0 - weights.sum()), torch.tensor(0.0, dtype=weights.dtype, device=weights.device), atol=atol):
        raise ValueError(f"The parameter 'weights' should be normalized, but got sum(weights) = {weights.sum():.5f}")

    weights = weights.to(device=device, dtype=dtype)
    return weights


def _check_means(means: torch.Tensor, n_components: int, n_features: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if not torch.is_tensor(means):
        means = torch.as_tensor(means)

    if means.shape != (n_components, n_features):
        raise ValueError(f"The parameter 'means' should have the shape of ({n_components}, {n_features}), but got {means.shape}")

    means = means.to(device=device, dtype=dtype)
    return means


def _check_precisions(
    precisions: torch.Tensor,
    covariance_type: Literal["full", "diag", "spherical", "tied"],
    n_components: int,
    n_features: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not torch.is_tensor(precisions):
        precisions = torch.as_tensor(precisions)

    expected_shapes = {
        "full": (n_components, n_features, n_features),
        "diag": (n_components, n_features),
        "spherical": (n_components,),
        "tied": (n_features, n_features),
    }

    expected_shape = expected_shapes[covariance_type]
    if precisions.shape != expected_shape:
        raise ValueError(f"The parameter 'precisions' should have the shape of {expected_shape}, but got {precisions.shape}")

    precisions = precisions.to(device=device, dtype=dtype)
    return precisions


def _estimate_gaussian_covariance_full(resp, X, nk, means, reg_covar: float) -> torch.Tensor:
    term_1 = torch.einsum("sc, sf, sg -> cfg", resp, X, X) / nk.view(-1, 1, 1)
    term_2 = torch.einsum("cf, cg -> cfg", means, means)
    covariances = term_1 - term_2
    covariances.diagonal(dim1=1, dim2=2).add_(reg_covar)
    return covariances


def _estimate_gaussian_covariance_diag(resp, X, nk, means, reg_covar: float) -> torch.Tensor:
    term_1 = torch.einsum("sc, sf -> cf", resp, X.square()) / nk.unsqueeze(1)
    term_2 = means.square()
    covariances = term_1 - term_2 + reg_covar
    return covariances


def _estimate_gaussian_covariance_spherical(resp, X, nk, means, reg_covar: float) -> torch.Tensor:
    return _estimate_gaussian_covariance_diag(resp, X, nk, means, reg_covar).mean(1)


def _estimate_gaussian_covariance_tied(resp, X, nk, means, reg_covar: float) -> torch.Tensor:
    term_1 = torch.einsum("sf, sg -> fg", X, X)
    term_2 = torch.einsum("c, cf, cg -> fg", nk, means, means)
    covariances = (term_1 - term_2) / nk.sum()
    covariances.diagonal().add_(reg_covar)
    return covariances


def _compute_precision_cholesky_tied(covariances: torch.Tensor) -> torch.Tensor:
    n_features, _ = covariances.shape

    try:
        covariances_cholesky = torch.linalg.cholesky(covariances, upper=False)
    except RuntimeError:
        raise ValueError(
            "Fitting the mixture model failed because some components have "
            "ill-defined empirical covariance (for instance caused by singleton "
            "or collapsed samples). Try to decrease the number of components, "
            "increase reg_covar, or scale the input data."
        )

    # Get the transposed inverse matrix of the cholesky decomposition.
    precisions_cholesky = torch.linalg.solve_triangular(
        covariances_cholesky,
        torch.eye(n_features, dtype=covariances.dtype, device=covariances.device),
        upper=False,
    ).permute(1, 0)
    return precisions_cholesky


def _compute_precision_cholesky_full(covariances: torch.Tensor) -> torch.Tensor:
    n_components, n_features, _ = covariances.shape
    precisions_cholesky = covariances.new_empty((n_components, n_features, n_features))
    for k, covariance in enumerate(covariances):
        try:
            precisions_cholesky[k] = _compute_precision_cholesky_tied(covariance)
        except ValueError:
            # Re-raise with component information
            raise ValueError(
                f"Fitting the mixture model failed because component {k} has "
                "ill-defined empirical covariance (for instance caused by singleton "
                "or collapsed samples). Try to decrease the number of components, "
                "increase reg_covar, or scale the input data."
            )
    return precisions_cholesky


def _compute_precision_cholesky_diag_spherical(covariances: torch.Tensor) -> torch.Tensor:
    if (covariances <= 0.0).any():
        raise ValueError(
            "Fitting the mixture model failed because some components have "
            "ill-defined empirical covariance (for instance caused by singleton "
            "or collapsed samples). Try to decrease the number of components, "
            "increase reg_covar, or scale the input data."
        )
    return torch.rsqrt(covariances)


def _compute_precision_cholesky_from_precisions_full(precisions: torch.Tensor) -> torch.Tensor:
    return torch.linalg.cholesky(precisions.flip((-2, -1)), upper=False).flip((-2, -1))


def _compute_precision_cholesky_from_precisions_diag_spherical(precisions: torch.Tensor) -> torch.Tensor:
    return precisions.sqrt()


def _compute_precision_cholesky_from_precisions_tied(precisions: torch.Tensor) -> torch.Tensor:
    return torch.linalg.cholesky(precisions.flip((-2, -1)), upper=False).flip((-2, -1))


def _compute_log_det_cholesky_full(cholesky: torch.Tensor, n_features: int) -> torch.Tensor:
    return cholesky.diagonal(dim1=1, dim2=2).log().sum(1)


def _compute_log_det_cholesky_diag(cholesky: torch.Tensor, n_features: int) -> torch.Tensor:
    return cholesky.log().sum(1)


def _compute_log_det_cholesky_spherical(cholesky: torch.Tensor, n_features: int) -> torch.Tensor:
    return n_features * cholesky.log()


def _compute_log_det_cholesky_tied(cholesky: torch.Tensor, n_features: int) -> torch.Tensor:
    return cholesky.diagonal().log().sum()


def _compute_log_quadratic_full(X: torch.Tensor, means: torch.Tensor, cholesky: torch.Tensor) -> torch.Tensor:
    term_1 = torch.einsum("sf, cfg -> scg", X, cholesky)
    term_2 = torch.einsum("cf, cfg -> cg", means, cholesky)
    quadratic = term_1 - term_2.unsqueeze(0)
    return quadratic.square_().sum(2)


def _compute_log_quadratic_diag(X: torch.Tensor, means: torch.Tensor, cholesky: torch.Tensor) -> torch.Tensor:
    term_1 = torch.einsum("sf, cf -> scf", X, cholesky)
    term_2 = torch.einsum("cf, cf -> cf", means, cholesky)
    quadratic = term_1 - term_2.unsqueeze(0)
    return quadratic.square_().sum(2)


def _compute_log_quadratic_spherical(X: torch.Tensor, means: torch.Tensor, cholesky: torch.Tensor) -> torch.Tensor:
    term_1 = torch.einsum("sf, c -> scf", X, cholesky)
    term_2 = torch.einsum("cf, c -> cf", means, cholesky)
    quadratic = term_1 - term_2.unsqueeze(0)
    return quadratic.square_().sum(2)


def _compute_log_quadratic_tied(X: torch.Tensor, means: torch.Tensor, cholesky: torch.Tensor) -> torch.Tensor:
    term_1 = torch.einsum("sf, fg -> sg", X, cholesky)
    term_2 = torch.einsum("cf, fg -> cg", means, cholesky)
    quadratic = term_1.unsqueeze(1) - term_2.unsqueeze(0)
    return quadratic.square_().sum(2)


class GaussianMixture:
    """
    Gaussian Mixture Model.

    Representation of a Gaussian mixture model probability distribution.
    This class allows to estimate the parameters of a Gaussian mixture
    distribution.

    Attributes:
        `weights_` (torch.Tensor): The weights of each mixture component. Shape [n_components].
        `means_` (torch.Tensor): The mean of each mixture component. Shape [n_components, n_features].
        `covariances_` (torch.Tensor): The covariance of each mixture component. Shape depends on `covariance_type`.
        `precisions_cholesky_` (torch.Tensor): The cholesky decomposition of the precision matrices of each mixture component.
        `n_iter_` (int): Number of iterations run by the last call to fit().
        `converged_` (bool): True when convergence was reached in fit(), False otherwise.
        `lower_bound_` (float): Lower bound value on the log probability (of the training data) of the best fit of EM.

    Args:
        `n_components` (int): The number of mixture components.
        `covariance_type` (str): String describing the type of covariance parameters to use.
            Must be one of 'full', 'tied', 'diag', 'spherical'.
        `tol` (float): The convergence threshold. EM iterations will stop when the
            lower bound average gain is below this threshold.
        `reg_covar` (float): Non-negative regularization added to the diagonal of covariance.
            Ensures the covariance matrices are positive definite.
        `max_iter` (int): The number of EM iterations to perform.
        `n_init` (int): The number of initializations to perform. The best results are kept.
        `init_params` (str): The method used to initialize the weights, the means and the precisions.
            Must be one of 'kmeans', 'random', 'random_from_data', 'kmeans++'.
        `random_state` (int, optional): A random number generator seed.
        `weights_init` (torch.Tensor, optional): The initial weights to use. Shape [n_components].
        `means_init` (torch.Tensor, optional): The initial means to use. Shape [n_components, n_features].
        `precisions_init` (torch.Tensor, optional): The initial precisions to use. Shape depends on `covariance_type`.
    """

    def __init__(
        self,
        n_components: int = 1,
        *,
        covariance_type: Literal["full", "diag", "spherical", "tied"] = "full",
        tol: float = 1e-3,
        reg_covar: float = torch.finfo(torch.float32).eps,
        max_iter: int = 100,
        n_init: int = 1,
        init_params: Literal["kmeans", "random", "random_from_data", "kmeans++"] = "kmeans",
        random_state: int | None = None,
        weights_init: torch.Tensor | None = None,
        means_init: torch.Tensor | None = None,
        precisions_init: torch.Tensor | None = None,
    ) -> None:
        super().__init__()

        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.random_state = random_state

        self.weights_init = weights_init
        self.means_init = means_init
        self.precisions_init = precisions_init

        self.weights_ = None
        self.means_ = None
        self.covariances_ = None
        self.precisions_cholesky_ = None

        self._estimate_gaussian_covariances = {
            "full": _estimate_gaussian_covariance_full,
            "diag": _estimate_gaussian_covariance_diag,
            "spherical": _estimate_gaussian_covariance_spherical,
            "tied": _estimate_gaussian_covariance_tied,
        }[self.covariance_type]
        self._compute_precision_cholesky = {
            "full": _compute_precision_cholesky_full,
            "diag": _compute_precision_cholesky_diag_spherical,
            "spherical": _compute_precision_cholesky_diag_spherical,
            "tied": _compute_precision_cholesky_tied,
        }[self.covariance_type]
        self._compute_precision_cholesky_from_precisions = {
            "full": _compute_precision_cholesky_from_precisions_full,
            "diag": _compute_precision_cholesky_from_precisions_diag_spherical,
            "spherical": _compute_precision_cholesky_from_precisions_diag_spherical,
            "tied": _compute_precision_cholesky_from_precisions_tied,
        }[self.covariance_type]
        self._compute_log_det_cholesky = {
            "full": _compute_log_det_cholesky_full,
            "diag": _compute_log_det_cholesky_diag,
            "spherical": _compute_log_det_cholesky_spherical,
            "tied": _compute_log_det_cholesky_tied,
        }[self.covariance_type]
        self._compute_log_quadratic = {
            "full": _compute_log_quadratic_full,
            "diag": _compute_log_quadratic_diag,
            "spherical": _compute_log_quadratic_spherical,
            "tied": _compute_log_quadratic_tied,
        }[self.covariance_type]

    def _check_parameters(self, X: torch.Tensor) -> None:
        """Check the parameters of the Gaussian mixture model."""

        _, n_features = X.shape

        if self.weights_init is not None:
            self.weights_init = _check_weights(self.weights_init, self.n_components, X.device, X.dtype)

        if self.means_init is not None:
            self.means_init = _check_means(self.means_init, self.n_components, n_features, X.device, X.dtype)

        if self.precisions_init is not None:
            self.precisions_init = _check_precisions(
                self.precisions_init, self.covariance_type, self.n_components, n_features, X.device, X.dtype
            )

    def fit(self, X: torch.Tensor) -> "GaussianMixture":
        """Estimate model parameters with the EM algorithm.

        The method fits the model `n_init` times and sets the parameters with
        which the model has the largest probability or lower bound.

        Args:
            `X` (torch.Tensor): Input data of shape [n_samples, n_features].

        Returns:
            (GaussianMixtureModel): The fitted model.
        """
        self.fit_predict(X)
        return self

    @torch.inference_mode()
    def fit_predict(self, X: torch.Tensor) -> torch.Tensor:
        """Estimate model parameters using EM and predict the labels for X.

        The method fits the model `n_init` times and sets the parameters with
        which the model has the largest probability or lower bound. Within each
        initialization, the loop iterates at least `max_iter` times.

        Args:
            `X` (torch.Tensor): Input data of shape [n_samples, n_features].

        Returns:
            (torch.Tensor): Component labels for each sample of shape [n_samples].
        """
        # Check X
        X = _check_data(X, X.device)

        n_samples, _ = X.shape
        if n_samples < self.n_components:
            raise ValueError(f"Expected n_samples >= n_components, but got {n_samples = } < {self.n_components = }")

        self._check_parameters(X)

        # Initialize other attributes
        max_lower_bound = -torch.inf
        self.converged_ = False

        rng = torch.Generator(device=X.device)
        if self.random_state is not None:
            rng.manual_seed(self.random_state)

        for init in range(self.n_init):

            self._initialize_parameters(X, rng)
            lower_bound = -torch.inf

            if self.max_iter == 0:
                best_params = self._get_parameters()
                best_n_iter = 0
            else:
                converged = False
                for n_iter in range(self.max_iter):
                    prev_lower_bound = lower_bound

                    # E-M iteration
                    log_prob, log_resp = self._e_step(X)
                    self._m_step(X, log_resp)
                    lower_bound = log_prob

                    # Check for convergence
                    change = lower_bound - prev_lower_bound
                    if abs(change) < self.tol:
                        converged = True
                        break

                if lower_bound > max_lower_bound or max_lower_bound == -torch.inf:
                    max_lower_bound = lower_bound
                    best_params = self._get_parameters()
                    best_n_iter = n_iter
                    self.converged_ = converged

        # Should only warn about convergence if max_iter > 0, otherwise
        # the user is assumed to have used 0-iters initialization
        # to get the initial means.
        if not self.converged_ and self.max_iter > 0:
            warnings.warn("Best performance not reached, increase n_init or max_iter to improve the results.")

        self._set_parameters(best_params)
        self.n_iter_ = best_n_iter
        self.lower_bound_ = max_lower_bound

        # Always do a final e-step to guarantee that the labels returned by
        # fit_predict(X) are always consistent with fit(X).predict(X)
        # for any value of max_iter and tol (and any random_state).
        _, log_resp = self._e_step(X)

        return log_resp.argmax(1)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict the labels for the data samples in X using trained model.

        Args:
            `X` (torch.Tensor): Input data of shape [n_samples, n_features].

        Returns:
            (torch.Tensor): Component labels for each sample of shape [n_samples].
        """
        X = _check_data(X, X.device)
        return self._estimate_weighted_log_prob(X).argmax(1)

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """Posterior probabilities for each sample in X.

        Args:
            `X` (torch.Tensor): Input data of shape [n_samples, n_features].

        Returns:
            (torch.Tensor): Posterior probabilities of shape [n_samples, n_components].
        """
        X = _check_data(X, X.device)
        _, log_resp = self._estimate_log_prob_resp(X)
        return log_resp.exp()

    def score_samples(self, X: torch.Tensor) -> torch.Tensor:
        """Log probabilities of the given data.

        Args:
            `X` (torch.Tensor): Input data of shape [n_samples, n_features].

        Returns:
            (torch.Tensor): Log probabilities of each sample of shape [n_samples].
        """
        X = _check_data(X, X.device)
        return self._estimate_weighted_log_prob(X).logsumexp(1)

    def score(self, X: torch.Tensor) -> float:
        """Average log probabilities of the given data under the Gaussian mixture model.

        Args:
            `X` (torch.Tensor): Input data of shape [n_samples, n_features].

        Returns:
            (float): Average log probabilities of the given data under the Gaussian mixture model.
        """
        return self.score_samples(X).mean().item()

    def _initialize_parameters(self, X: torch.Tensor, rng: torch.Generator) -> None:
        """Initialize the model parameters."""
        if not (
            self.weights_init is None or \
            self.means_init is None or \
            self.precisions_init is None
        ):
            self._initialize(X, None)
            return

        n_samples, _ = X.shape

        if self.init_params == "kmeans":
            # Simple PyTorch K-Means implementation for initialization
            resp = X.new_zeros((n_samples, self.n_components))

            indices = torch.randperm(n_samples, generator=rng, device=X.device)[: self.n_components]
            centroids = X[indices]

            # Simple K-Means Loop (fixed 10 iters for init speed)
            for _ in range(10):
                # E-step: distance to centroids
                dists = torch.cdist(X, centroids)
                labels = dists.argmin(1)

                # M-step: update centroids
                new_centroids = centroids.clone()
                for k in range(self.n_components):
                    mask = labels == k
                    if mask.any():
                        new_centroids[k] = X[mask].mean(0)
                    else:
                        # Empty cluster
                        rand_idx = torch.randint(0, n_samples, (1,), generator=rng, device=X.device)
                        new_centroids[k] = X[rand_idx].squeeze(0)

                # Exit condition
                if torch.allclose(centroids, new_centroids, atol=1e-4):
                    centroids = new_centroids
                    break

                centroids = new_centroids

            # Create hard responsibilities
            dists = torch.cdist(X, centroids)
            labels = dists.argmin(1)
            resp[torch.arange(n_samples, device=X.device), labels] = 1.0

        elif self.init_params == "random":
            resp = torch.rand((n_samples, self.n_components), generator=rng, device=X.device)
            resp /= resp.sum(1, keepdim=True)

        elif self.init_params == "random_from_data":
            resp = torch.zeros((n_samples, self.n_components), device=X.device)
            indices = torch.randperm(n_samples, generator=rng, device=X.device)[: self.n_components]
            resp[indices, torch.arange(self.n_components, device=X.device)] = 1.0

        elif self.init_params == "kmeans++":
            raise NotImplementedError("KMeans++ is not implemented yet.")

        self._initialize(X, resp)

    def _initialize(self, X: torch.Tensor, resp: torch.Tensor | None) -> None:
        """Initialize model parameters from responsibilities."""
        n_samples, _ = X.shape
        weights, means, covariances = None, None, None

        if resp is not None:
            weights, means, covariances = self._estimate_gaussian_parameters(X, resp)
            if self.weights_init is None:
                weights /= n_samples

        self.weights_ = weights if self.weights_init is None else self.weights_init
        self.means_ = means if self.means_init is None else self.means_init

        if self.precisions_init is None:
            self.covariances_ = covariances
            self.precisions_cholesky_ = self._compute_precision_cholesky(covariances)
        else:
            self.precisions_cholesky_ = self._compute_precision_cholesky_from_precisions(self.precisions_init)

    def _e_step(self, X: torch.Tensor) -> tuple[float, torch.Tensor]:
        """E step.

        Args:
            `X` (torch.Tensor): Input data of shape [n_samples, n_features].

        Returns:
            (float): Log probability of the model.
            (torch.Tensor): Log-responsibility of shape [n_samples, n_components].
        """
        log_prob, log_resp = self._estimate_log_prob_resp(X)
        return log_prob.mean().item(), log_resp

    def _m_step(self, X: torch.Tensor, log_resp: torch.Tensor) -> None:
        """M step.

        Args:
            `X` (torch.Tensor): Input data of shape [n_samples, n_features].
            `log_resp` (torch.Tensor): Log-responsibility of shape [n_samples, n_components].
        """
        self.weights_, self.means_, self.covariances_ = self._estimate_gaussian_parameters(X, log_resp.exp())
        self.weights_ /= self.weights_.sum()
        self.precisions_cholesky_ = self._compute_precision_cholesky(self.covariances_)

    def _estimate_log_prob_resp(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Estimate log probabilities and responsibilities for each sample.

        Args:
            `X` (torch.Tensor): Input data of shape [n_samples, n_features].

        Returns:
            (torch.Tensor): Log probabilities of each sample of shape [n_samples].
            (torch.Tensor): Log responsibility of shape [n_samples, n_components].
        """
        weighted_log_prob = self._estimate_weighted_log_prob(X)
        log_prob = weighted_log_prob.logsumexp(1)
        log_resp = weighted_log_prob - log_prob.unsqueeze(1)
        return log_prob, log_resp

    def _estimate_weighted_log_prob(self, X: torch.Tensor) -> torch.Tensor:
        """Estimate the weighted log-probabilities."""
        return self._estimate_log_weights() + self._estimate_log_gaussian_prob(X)

    def _estimate_log_weights(self) -> torch.Tensor:
        """Estimate the log of the weights of the mixture components of shape [n_components]."""
        return self.weights_.log()

    def _estimate_log_gaussian_prob(self, X: torch.Tensor) -> torch.Tensor:
        """Estimate the log Gaussian probability.

        Args:
            `X` (torch.Tensor): Input data of shape [n_samples, n_features].

        Returns:
            (torch.Tensor): Log-probabilities of shape [n_samples, n_components].
        """
        _, n_features = X.shape

        log_det_cholesky = self._compute_log_det_cholesky(self.precisions_cholesky_, n_features)
        log_quadratic = self._compute_log_quadratic(X, self.means_, self.precisions_cholesky_)

        return log_det_cholesky - 0.5 * (n_features * LOG_TAU + log_quadratic)

    def _estimate_gaussian_parameters(self, X: torch.Tensor, resp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Estimate the Gaussian distribution parameters.

        Args:
            `X` (torch.Tensor): Input data of shape [n_samples, n_features].
            `resp` (torch.Tensor): Responsibilities of shape [n_samples, n_components].

        Returns:
            (tuple):
                `nk` (torch.Tensor): Effective number of samples of shape [n_components].
                `means` (torch.Tensor): Component means of shape [n_components, n_features].
                `covariances` (torch.Tensor): Covariances of shape [n_components, n_features, n_features].
        """
        nk = resp.sum(0) + 10 * torch.finfo(resp.dtype).eps
        means = torch.einsum("sc, sf -> cf", resp, X) / nk.unsqueeze(1)
        covariances = self._estimate_gaussian_covariances(resp, X, nk, means, self.reg_covar)
        return nk, means, covariances

    def _get_parameters(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.weights_, self.means_, self.covariances_, self.precisions_cholesky_

    def _set_parameters(self, params):
        self.weights_, self.means_, self.covariances_, self.precisions_cholesky_ = params

    def _n_parameters(self) -> int:
        """Return the number of free parameters in the model."""
        _, n_features = self.means_.shape

        mean_params = self.n_components * n_features
        if self.covariance_type == "full":
            cov_params = self.n_components * n_features * (n_features + 1) // 2
        elif self.covariance_type == "diag":
            cov_params = self.n_components * n_features
        elif self.covariance_type == "spherical":
            cov_params = self.n_components
        elif self.covariance_type == "tied":
            cov_params = n_features * (n_features + 1) // 2
        weight_params = self.n_components - 1 # Sum of weights is 1, which reduces one degree of freedom.

        return mean_params + cov_params + weight_params

    def aic(self, X: torch.Tensor) -> float:
        """Akaike information criterion.

        Args:
            `X` (torch.Tensor): Input data of shape [n_samples, n_features].

        Returns:
            (float): Akaike information criterion.
        """
        n_samples, _ = X.shape
        return -2 * self.score(X) * n_samples + 2 * self._n_parameters()

    def bic(self, X: torch.Tensor) -> float:
        """Bayesian information criterion.

        Args:
            `X` (torch.Tensor): Input data of shape [n_samples, n_features].

        Returns:
            (float): Bayesian information criterion.
        """
        n_samples, _ = X.shape
        return -2 * self.score(X) * n_samples + self._n_parameters() * math.log(n_samples)
