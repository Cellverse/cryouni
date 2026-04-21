"""
This module provides functions to compute centered Hartley transforms and
other related Hartley domain operations.

The definition and implementation of the centered Hartley transform is based on
the centered Fourier transform. Here, the Hartley transform is defined as:

>>> HT(x) = real(FT(x)) - imag(FT(x))

where FT(x) is the Fourier transform.

Both forward and inverse transforms are implemented using the same formula,
as the Hartley transform is its own inverse when properly normalized.
"""

from __future__ import annotations

import math

import torch

from ..number import is_even
from ..radial import sum_over_radial
from ._dimension import _get_last_n_dims
from .fourier import (
    FourierTransform,
    NormType,
)

__all__ = [
    "HartleyTransform",
]


class HartleyTransform(object):
    """
    Args:
        `norm` (str): Normalization mode of choices ["backward", "ortho", "forward"].
    """

    def __init__(self, norm: NormType = NormType.BACKWARD) -> None:
        self.ft = FourierTransform(norm)

    @property
    def norm(self) -> NormType:
        return self.ft.norm

    @norm.setter
    def norm(self, val: NormType) -> None:
        self.ft.norm = val

    def ft_to_ht(self, ft: torch.Tensor) -> torch.Tensor:
        """
        Convert Fourier domain signal to Hartley domain signal.

        Args:
            `ft` (torch.Tensor): Fourier domain signal.

        Returns:
            (torch.Tensor): Hartley domain signal.
        """
        return ft.real - ft.imag

    def ht_to_ft(self, ht: torch.Tensor, last_n_dims: int | None = None) -> torch.Tensor:
        """
        Convert Hartley domain signal to Fourier domain signal.

        Args:
            `ht` (torch.Tensor): Hartley domain signal.
            `last_n_dims` (int | None): Number of last dimensions to compute.

        Returns:
            (torch.Tensor): Fourier domain signal.
        """
        ht_flip = self._flip(ht, last_n_dims)
        ht_e = (ht_flip + ht) * 0.5
        ht_o = (ht_flip - ht) * 0.5
        return torch.complex(ht_e, ht_o)

    def ht1_center(self, x: torch.Tensor) -> torch.Tensor:
        """
        1D centered Hartley Transform.

        Args:
            `x` (torch.Tensor): Spatial domain signal of shape [..., X].

        Returns:
            (torch.Tensor): Centered 1D Hartley transform of same shape as input.
        """
        ft = self.ft.ft1_center(x)
        return self.ft_to_ht(ft)

    def iht1_center(self, x: torch.Tensor) -> torch.Tensor:
        """
        1D centered inverse Hartley Transform.

        Args:
            `x` (torch.Tensor): Hartley domain signal of shape [..., X].

        Returns:
            (torch.Tensor): Centered inverse 1D Hartley transform of same shape as input.
        """
        if self.norm == NormType.BACKWARD:
            return self.ht1_center(x) / x.shape[-1]
        elif self.norm == NormType.ORTHO:
            return self.ht1_center(x)
        elif self.norm == NormType.FORWARD:
            return self.ht1_center(x) * x.shape[-1]

    def ht2_center(self, x: torch.Tensor) -> torch.Tensor:
        """
        2D centered Hartley Transform.

        Args:
            `x` (torch.Tensor): Spatial domain signal of shape [..., Y, X].

        Returns:
            (torch.Tensor): Centered 2D Hartley transform of same shape as input.
        """
        ft = self.ft.ft2_center(x)
        return self.ft_to_ht(ft)

    def iht2_center(self, x: torch.Tensor) -> torch.Tensor:
        """
        2D centered inverse Hartley Transform.

        Args:
            `x` (torch.Tensor): Hartley domain signal of shape [..., Y, X].

        Returns:
            (torch.Tensor): Centered inverse 2D Hartley transform of same shape as input.
        """
        if self.norm == NormType.BACKWARD:
            return self.ht2_center(x) / math.prod(x.shape[-2 :])
        elif self.norm == NormType.ORTHO:
            return self.ht2_center(x)
        elif self.norm == NormType.FORWARD:
            return self.ht2_center(x) * math.prod(x.shape[-2 :])

    def ht3_center(self, x: torch.Tensor) -> torch.Tensor:
        """
        3D centered Hartley Transform.

        Args:
            `x` (torch.Tensor): Spatial domain signal of shape [..., Z, Y, X].

        Returns:
            (torch.Tensor): Centered 3D Hartley transform of same shape as input.
        """
        ft = self.ft.ft3_center(x)
        return self.ft_to_ht(ft)

    def iht3_center(self, x: torch.Tensor) -> torch.Tensor:
        """
        3D centered inverse Hartley Transform.

        Args:
            `x` (torch.Tensor): Hartley domain signal of shape [..., Z, Y, X].

        Returns:
            (torch.Tensor): Centered inverse 3D Hartley transform of same shape as input.
        """
        if self.norm == NormType.BACKWARD:
            return self.ht3_center(x) / math.prod(x.shape[-3 :])
        elif self.norm == NormType.ORTHO:
            return self.ht3_center(x)
        elif self.norm == NormType.FORWARD:
            return self.ht3_center(x) * math.prod(x.shape[-3 :])

    def htn_center(self, x: torch.Tensor, last_n_dims: int | None = None) -> torch.Tensor:
        """
        N-D centered Hartley Transform.

        Args:
            `x` (torch.Tensor): Spatial domain signal.
            `last_n_dims` (int | None): Number of last dimensions to transform.

        Returns:
            (torch.Tensor): Centered N-D Hartley transform of same shape as input.
        """
        ft = self.ft.ftn_center(x, last_n_dims)
        return self.ft_to_ht(ft)

    def ihtn_center(self, x: torch.Tensor, last_n_dims: int | None = None) -> torch.Tensor:
        """
        N-D centered inverse Hartley Transform.

        Args:
            `x` (torch.Tensor): Hartley domain signal.
            `last_n_dims` (int | None): Number of last dimensions to transform.

        Returns:
            (torch.Tensor): Centered inverse N-D Hartley transform of same shape as input.
        """
        if self.norm == NormType.BACKWARD:
            return self.htn_center(x, last_n_dims) / math.prod(x.shape[-last_n_dims :])
        elif self.norm == NormType.ORTHO:
            return self.htn_center(x, last_n_dims)
        elif self.norm == NormType.FORWARD:
            return self.htn_center(x, last_n_dims) * math.prod(x.shape[-last_n_dims :])

    def _phase_shift(self, t: torch.Tensor, frequency_grid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute phase shift for translation in Hartley domain.

        Args:
            `t` (torch.Tensor): Translation vector of shape [B, D], units are pixels/voxels.
            `frequency_grid` (torch.Tensor): Frequency coordinates of shape [..., D].

        Returns:
            (tuple[torch.Tensor, torch.Tensor]): Tuple of (cosine, sine) phase shift tensors of shape [B, ...].
        """
        phase = 2.0 * torch.pi * torch.tensordot(t, frequency_grid, dims=([-1], [-1]))
        phase_shift_cos = phase.cos()
        phase_shift_sin = phase.sin()
        return phase_shift_cos, phase_shift_sin

    def translate(self, ht: torch.Tensor, t: torch.Tensor, frequency_grid: torch.Tensor) -> torch.Tensor:
        """
        Translate Hartley domain signal, which is

        >>> HT(f(x - t)) = HT(f(x)) * cos(2 * pi * dot(t, k)) + HT(f(x)) * sin(2 * pi * dot(t, k))

        Args:
            `ht` (torch.Tensor): Input Hartley domain signal of shape [..., Y, X] for 2D or [..., Z, Y, X] for 3D.
            `t` (torch.Tensor): Translation vector of shape [..., 2] for 2D or [..., 3] for 3D, units are pixels/voxels.
            `frequency_grid` (torch.Tensor): Frequency coordinates of shape [Y, X, (x, y)] for 2D or [Z, Y, X, (x, y, z)] for 3D.

        Returns:
            (torch.Tensor): Translated Hartley domain signal of shape [..., Y, X] for 2D or [..., Z, Y, X] for 3D.
        """
        phase_shift_cos, phase_shift_sin = self._phase_shift(t, frequency_grid)
        return ht * phase_shift_cos + self._flip(ht, t.size(-1)) * phase_shift_sin

    def inv_translate(self, ht: torch.Tensor, t: torch.Tensor, frequency_grid: torch.Tensor) -> torch.Tensor:
        """
        Inverse translation in Hartley domain, which is

        >>> HT(f(x + t)) = HT(f(x)) * cos(2 * pi * dot(t, k)) - HT(f(x)) * sin(2 * pi * dot(t, k))

        Args:
            `ht` (torch.Tensor): Input Hartley domain signal of shape [..., Y, X] for 2D or [..., Z, Y, X] for 3D.
            `t` (torch.Tensor): Translation vector of shape [..., 2] for 2D or [..., 3] for 3D, units are pixels/voxels.
            `frequency_grid` (torch.Tensor): Frequency coordinates of shape [Y, X, (x, y)] for 2D or [Z, Y, X, (x, y, z)] for 3D.

        Returns:
            (torch.Tensor): Inverse translated Hartley domain signal of shape [..., Y, X] for 2D or [..., Z, Y, X] for 3D.
        """
        phase_shift_cos, phase_shift_sin = self._phase_shift(t, frequency_grid)
        return ht * phase_shift_cos - self._flip(ht, t.size(-1)) * phase_shift_sin

    def covariance(self, ht1: torch.Tensor, ht2: torch.Tensor, last_n_dims: int | None = None) -> torch.Tensor:
        """
        Compute Hartley domain covariance, which is:

        >>> cov(HT(x), HT(y)) = [HT(x) * HT(y) + HT(x) * HT(-y) - HT(-x) * HT(y) + HT(-x) * HT(-y)] / 2

        Args:
            `ht1` (torch.Tensor): Input Hartley domain signal of shape [...].
            `ht2` (torch.Tensor): Input Hartley domain signal of shape [...].
            `last_n_dims` (int | None): Number of last dimensions to compute.

        Returns:
            (torch.Tensor): Covariance tensor of the same shape as input.
        """
        ht1_flip = self._flip(ht1, last_n_dims)
        ht2_flip = self._flip(ht2, last_n_dims)
        covariance = (ht1 * (ht2 - ht2_flip) + ht1_flip * (ht2 + ht2_flip)) * 0.5
        return covariance

    def variance(self, ht: torch.Tensor, last_n_dims: int | None = None) -> torch.Tensor:
        """
        Compute Hartley domain variance.

        >>> var(HT(x)) = (HT(x)^2 + HT(-x)^2) / 2

        Args:
            `ht` (torch.Tensor): Input Hartley domain signal of shape [...].
            `last_n_dims` (int | None): Number of last dimensions to compute.

        Returns:
            (torch.Tensor): Variance tensor of the same shape as input.
        """
        ht_flip = self._flip(ht, last_n_dims)
        return (ht.square() + ht_flip.square()) * 0.5

    def htn_psd(self, ht: torch.Tensor, last_n_dims: int | None = None) -> torch.Tensor:
        """
        Compute Power Spectral Density (PSD) in Hartley domain, which is an alias of variance.

        Args:
            `ht` (torch.Tensor): Input Hartley domain signal of shape [...].
            `last_n_dims` (int | None): Number of last dimensions to transform.

        Returns:
            (torch.Tensor): PSD tensor of same shape as input.
        """
        return self.variance(ht, last_n_dims)

    def shell_correlation(self, ht1: torch.Tensor, ht2: torch.Tensor, last_n_dims: int | None = None) -> torch.Tensor:
        """
        Compute the shell correlation between two volumes, which is:

        >>> SC(HT(x), HT(y)) = cov_real(HT(x), HT(y)) / sqrt(var(HT(x)) * var(HT(y)))

        Args:
            `ht1` (torch.Tensor): Input Hartley domain signal of shape [..., Z, Y, X] for 3D or [..., Y, X] for 2D.
            `ht2` (torch.Tensor): Input Hartley domain signal of shape [..., Z, Y, X] for 3D or [..., Y, X] for 2D.
            `last_n_dims` (int | None): Number of last dimensions to compute the shell correlation.

        Returns:
            (torch.Tensor): The shell correlation between the two volumes, shape [..., nyquist_index(ht.shape[-last_n_dims:])].
        """
        covariance = sum_over_radial(self._real_covariance(ht1, ht2, last_n_dims), last_n_dims)
        variance_1 = sum_over_radial(self.variance(ht1, last_n_dims), last_n_dims)
        variance_2 = sum_over_radial(self.variance(ht2, last_n_dims), last_n_dims)
        shell_correlation = covariance / (variance_1 * variance_2).sqrt()
        shell_correlation.nan_to_num_(0.0, 0.0, 0.0)
        shell_correlation[..., 0] = shell_correlation[..., 1]
        return shell_correlation

    def _real_covariance(self, ht1: torch.Tensor, ht2: torch.Tensor, last_n_dims: int | None = None) -> torch.Tensor:
        """
        Compute the real part of Fourier domain covariance in Hartley domain.

        >>> cov_real(HT(x), HT(y)) = [HT(x) * HT(y) + HT(-x) * HT(-y)] / 2

        Args:
            `ht1` (torch.Tensor): Input Hartley domain signal of shape [...].
            `ht2` (torch.Tensor): Input Hartley domain signal of shape [...].
            `last_n_dims` (int | None): Number of last dimensions to compute.

        Returns:
            (torch.Tensor): Covariance tensor of the same shape as input.
        """
        ht1_flip = self._flip(ht1, last_n_dims)
        ht2_flip = self._flip(ht2, last_n_dims)
        covariance = (ht1 * ht2 + ht1_flip * ht2_flip) * 0.5
        return covariance

    def _flip(self, ht: torch.Tensor, last_n_dims: int | None = None) -> torch.Tensor:
        """
        Get the flipped Hartley domain signal, i.e. `ht(-f) = ht(N - f)`.
        Here the 0th frequency is at the geometric center of `ht`.

        Args:
            `ht` (torch.Tensor): Input Hartley domain signal of shape [...].
            `last_n_dims` (int | None): Number of last dimensions to flip.

        Returns:
            (torch.Tensor): Flipped Hartley domain signal of the same shape as input.
        """

        # Select all the even dimensions
        flip_dims = _get_last_n_dims(last_n_dims if last_n_dims is not None else ht.ndim)
        roll_dims = tuple(d for d in flip_dims if is_even(ht.size(d)))

        if len(roll_dims) == 0:
            return ht.flip(dims=flip_dims)
        else:
            return ht.flip(dims=flip_dims).roll([1] * len(roll_dims), dims=roll_dims)
