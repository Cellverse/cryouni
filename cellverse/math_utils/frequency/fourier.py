"""
This module provides functions to compute centered Fourier transforms and
other related Fourier domain operations.

The input signals are treated with the origin `(0, 0)` at their geometric center.
Therefore, for centered transforms, only odd-sized inputs are supported.
However, we will not check for this for the sake of speed.
Therefore, the result for even-sized inputs is undefined.

To compute the centered Fourier transform:
    1. The input signal is shifted to relocate its center to the array corner (via ifftshift).
    2. The standard Fourier transform is applied (via fft).
    3. The result is shifted to position the zero-frequency component at the center (via fftshift).
"""

from __future__ import annotations

from enum import StrEnum

import torch
import torch.fft as fft

from ..radial import sum_over_radial
from ._dimension import _get_last_n_dims

__all__ = [
    "FourierTransform",
    "NormType",
]


class NormType(StrEnum):
    BACKWARD = "backward"
    ORTHO = "ortho"
    FORWARD = "forward"


class FourierTransform(object):
    """
    Args:
        `norm` (str): Normalization mode of choices ["backward", "ortho", "forward"].
    """

    def __init__(self, norm: NormType = NormType.BACKWARD) -> None:
        self._norm = norm

    @property
    def norm(self) -> NormType:
        return self._norm

    @norm.setter
    def norm(self, val: NormType) -> None:
        self._norm = val

    def ft1_center(self, x: torch.Tensor) -> torch.Tensor:
        """
        1D centered Fourier Transform.

        Args:
            `x` (torch.Tensor): Spatial domain signal of shape [..., X].

        Returns:
            (torch.Tensor): Centered 1D Fourier transform of same shape as input.
        """
        return fft.fftshift(fft.fft(fft.ifftshift(x, dim=-1), norm=self.norm.value), dim=-1)

    def ift1_center(self, x: torch.Tensor) -> torch.Tensor:
        """
        1D centered inverse Fourier Transform.

        Args:
            `x` (torch.Tensor): Frequency domain signal of shape [..., X].

        Returns:
            (torch.Tensor): Centered inverse 1D Fourier transform of same shape as input.
        """
        return fft.fftshift(fft.ifft(fft.ifftshift(x, dim=-1), norm=self.norm.value), dim=-1)

    def ft2_center(self, x: torch.Tensor) -> torch.Tensor:
        """
        2D centered Fourier Transform.

        Args:
            `x` (torch.Tensor): Spatial domain signal of shape [..., Y, X].

        Returns:
            (torch.Tensor): Centered 2D Fourier transform of same shape as input.
        """
        dims = _get_last_n_dims(2)
        return fft.fftshift(fft.fft2(fft.ifftshift(x, dim=dims), norm=self.norm.value), dim=dims)

    def ift2_center(self, x: torch.Tensor) -> torch.Tensor:
        """
        2D centered inverse Fourier Transform.

        Args:
            `x`(torch.Tensor): Frequency domain signal of shape [..., Y, X].

        Returns:
            (torch.Tensor): Centered inverse 2D Fourier transform of same shape as input.
        """
        dims = _get_last_n_dims(2)
        return fft.fftshift(fft.ifft2(fft.ifftshift(x, dim=dims), norm=self.norm.value), dim=dims)

    def ft3_center(self, x: torch.Tensor) -> torch.Tensor:
        """
        3D centered Fourier Transform.

        Args:
            `x` (torch.Tensor): Spatial domain signal of shape [..., Z, Y, X].

        Returns:
            (torch.Tensor): Centered 3D Fourier transform of same shape as input.
        """
        dims = _get_last_n_dims(3)
        return fft.fftshift(fft.fftn(fft.ifftshift(x, dim=dims), dim=dims, norm=self.norm.value), dim=dims)

    def ift3_center(self, x: torch.Tensor) -> torch.Tensor:
        """
        3D centered inverse Fourier Transform.

        Args:
            `x` (torch.Tensor): Frequency domain signal of shape [..., Z, Y, X].

        Returns:
            (torch.Tensor): Centered inverse 3D Fourier transform of same shape as input.
        """
        dims = _get_last_n_dims(3)
        return fft.fftshift(fft.ifftn(fft.ifftshift(x, dim=dims), dim=dims, norm=self.norm.value), dim=dims)

    def ftn_center(self, x: torch.Tensor, last_n_dims: int | None = None) -> torch.Tensor:
        """
        N-D centered Fourier Transform.

        Args:
            `x` (torch.Tensor): Spatial domain signal.
            `last_n_dims` (int | None): Number of last dimensions to transform.

        Returns:
            (torch.Tensor): Centered N-D Fourier transform of same shape as input.
        """
        dims = _get_last_n_dims(last_n_dims if last_n_dims is not None else x.ndim)
        return fft.fftshift(fft.fftn(fft.ifftshift(x, dim=dims), norm=self.norm.value, dim=dims), dim=dims)

    def iftn_center(self, x: torch.Tensor, last_n_dims: int | None = None) -> torch.Tensor:
        """
        N-D centered inverse Fourier Transform.

        Args:
            `x` (torch.Tensor): Frequency domain signal.
            `last_n_dims` (int | None): Number of last dimensions to transform.

        Returns:
            (torch.Tensor): Centered inverse N-D Fourier transform of same shape as input.
        """
        dims = _get_last_n_dims(last_n_dims if last_n_dims is not None else x.ndim)
        return fft.fftshift(fft.ifftn(fft.ifftshift(x, dim=dims), norm=self.norm.value, dim=dims), dim=dims)

    def _phase_shift(self, t: torch.Tensor, frequency_grid: torch.Tensor) -> torch.Tensor:
        """
        Compute phase shift for translation in Fourier domain.

        Args:
            `t` (torch.Tensor): Translation vector of shape [B, D], units are pixels/voxels.
            `frequency_grid` (torch.Tensor): Frequency coordinates of shape [..., D].

        Returns:
            (torch.Tensor): Phase shift tensor of shape [B, ...].
        """
        phase = -2.0 * torch.pi * torch.tensordot(t, frequency_grid, dims=([-1], [-1]))
        return torch.polar(torch.ones_like(phase), phase)

    def translate(self, ft: torch.Tensor, t: torch.Tensor, frequency_grid: torch.Tensor) -> torch.Tensor:
        """
        Translate Fourier domain signal, which is

        >>> FT(f(x - t)) = FT(f(x)) * exp(-i * 2 * pi * dot(t, k))

        Args:
            `ft` (torch.Tensor): Input Fourier domain signal of shape [..., Y, X] for 2D or [..., Z, Y, X] for 3D.
            `t` (torch.Tensor): Translation vector of shape [..., 2] for 2D or [..., 3] for 3D, units are pixels/voxels.
            `frequency_grid` (torch.Tensor): Frequency coordinates of shape [Y, X, (x, y)] for 2D or [Z, Y, X, (x, y, z)] for 3D.

        Returns:
            (torch.Tensor): Translated Fourier domain signal of shape [..., Y, X] for 2D or [..., Z, Y, X] for 3D.
        """
        phase_shift = self._phase_shift(t, frequency_grid)
        return ft * phase_shift

    def inv_translate(self, ft: torch.Tensor, t: torch.Tensor, frequency_grid: torch.Tensor) -> torch.Tensor:
        """
        Inverse translation in Fourier domain, which is

        >>> FT(f(x + t)) = FT(f(x)) * exp(i * 2 * pi * dot(t, k))

        Args:
            `ft` (torch.Tensor): Input Fourier domain signal of shape [..., Y, X] for 2D or [..., Z, Y, X] for 3D.
            `t` (torch.Tensor): Translation vector of shape [..., 2] for 2D or [..., 3] for 3D, units are pixels/voxels.
            `frequency_grid` (torch.Tensor): Frequency coordinates of shape [Y, X, (x, y)] for 2D or [Z, Y, X, (x, y, z)] for 3D.

        Returns:
            (torch.Tensor): Inverse translated Fourier domain signal of shape [..., Y, X] for 2D or [..., Z, Y, X] for 3D.
        """
        phase_shift = self._phase_shift(t, frequency_grid)
        return ft * phase_shift.conj()

    def covariance(self, ft1: torch.Tensor, ft2: torch.Tensor) -> torch.Tensor:
        """
        Compute Fourier domain covariance, which is:

        >>> cov(FT(x), FT(y)) = conj(FT(x)) * FT(y)

        Args:
            `ft1` (torch.Tensor): Input Fourier domain signal of shape [...].
            `ft2` (torch.Tensor): Input Fourier domain signal of shape [...].

        Returns:
            (torch.Tensor): Covariance tensor of same shape as input.
        """
        return ft1.conj() * ft2

    def variance(self, ft: torch.Tensor) -> torch.Tensor:
        """
        Compute Fourier domain variance, which is:

        >>> var(FT(x)) = |FT(x)|^2

        Args:
            `ft` (torch.Tensor): Input Fourier domain signal of shape [...].

        Returns:
            (torch.Tensor): Variance tensor of of same shape as input.
        """
        return ft.abs().square()

    def psd(self, ft: torch.Tensor) -> torch.Tensor:
        """
        Compute Power Spectral Density (PSD) of a N-D Fourier domain signal, which is an alias of variance.

        Args:
            `ft` (torch.Tensor): Input Fourier domain signal of shape [...].

        Returns:
            (torch.Tensor): PSD tensor of same shape as input.
        """
        return self.variance(ft)

    def shell_correlation(self, ft1: torch.Tensor, ft2: torch.Tensor, last_n_dims: int | None = None) -> torch.Tensor:
        """
        Compute the shell correlation between two volumes, which is:

        >>> SC(FT(x), FT(y)) = real(cov(FT(x), FT(y))) / sqrt(var(FT(x)) * var(FT(y)))

        Args:
            `ft1` (torch.Tensor): Input Fourier domain signal of shape [..., Z, Y, X] for 3D or [..., Y, X] for 2D.
            `ft2` (torch.Tensor): Input Fourier domain signal of shape [..., Z, Y, X] for 3D or [..., Y, X] for 2D.
            `last_n_dims` (int | None): Number of last dimensions to compute the shell correlation.

        Returns:
            (torch.Tensor): The shell correlation between the two volumes, shape [..., nyquist_index(ft.shape[-last_n_dims:])].
        """
        covariance = sum_over_radial(self.covariance(ft1, ft2).real, last_n_dims)
        variance_1 = sum_over_radial(self.variance(ft1), last_n_dims)
        variance_2 = sum_over_radial(self.variance(ft2), last_n_dims)
        shell_correlation = covariance / (variance_1 * variance_2).sqrt()
        shell_correlation.nan_to_num_(0.0, 0.0, 0.0)
        shell_correlation[..., 0] = shell_correlation[..., 1]
        return shell_correlation
