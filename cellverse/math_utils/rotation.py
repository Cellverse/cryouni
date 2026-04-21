"""
This module provides functions that is related to rotations,
including axis-angle, quaternion, rotation matrix, so3 and s2s2.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = [
    "axis_angle_to_rotation_matrix",
    "axis_angle_to_quaternion",
    "quaternion_multiply",
    "quaternion_to_axis_angle",
    "quaternion_to_rotation_matrix",
    "rotation_matrix_to_axis_angle",
    "rotation_matrix_to_quaternion",
    "standardize_quaternion",
]


def standardize_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion to standard form
        - Quaternions are in the form (s, i, j, k), where s is the scalar part and i, j, k are the vector part.
        - The scalar part is positive.

    Args:
        `quaternion` (torch.Tensor): Input quaternion of shape [..., 4].

    Returns:
        (torch.Tensor): Standard quaternion of shape [..., 4].
    """
    return torch.where(quaternion[..., 0 : 1] < 0, -quaternion, quaternion)


def quaternion_multiply(quaternion1: torch.Tensor, quaternion2: torch.Tensor) -> torch.Tensor:
    s1, i1, j1, k1 = quaternion1.unbind(-1)
    s2, i2, j2, k2 = quaternion2.unbind(-1)
    quaternion = torch.empty_like(quaternion1)
    quaternion[..., 0] = s1 * s2 - i1 * i2 - j1 * j2 - k1 * k2
    quaternion[..., 1] = s1 * i2 + i1 * s2 + j1 * k2 - k1 * j2
    quaternion[..., 2] = s1 * j2 - i1 * k2 + j1 * s2 + k1 * i2
    quaternion[..., 3] = s1 * k2 + i1 * j2 - j1 * i2 + k1 * s2
    return quaternion


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Equivalent to `x.relu().sqrt()`, but with a zero gradient at `x = 0`.
    """
    if torch.is_grad_enabled():
        mask = x > 0
        out = torch.zeros_like(x)
        out[mask] = x[mask].sqrt()
    else:
        out = x.relu().sqrt()
    return out


def axis_angle_to_rotation_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert axis-angle vector to rotation matrix.

    Args:
        `axis_angle` (torch.Tensor): Input axis-angle vector of shape [..., 3],
            where its direction aligning with the axis direction and its norm represents rotation angles.

    Returns:
        (torch.Tensor): Rotation matrix of shape [..., 3, 3].
    """

    def _axis_angle_to_rotation_matrix(axis_angle: torch.Tensor, theta_sq: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Works for large `theta` value."""

        theta = theta_sq.clamp_min(1e-12).sqrt()        # [...]
        axis = axis_angle / (theta.unsqueeze(-1) + eps) # [..., 3]
        x, y, z = axis.unbind(-1)                       # [...]

        xx, yy, zz = x * x, y * y, z * z
        xy, yz, zx = x * y, y * z, z * x

        cos_theta = theta.cos()
        sin_theta = theta.sin()
        one_m_cos = 1 - cos_theta

        matrix = axis_angle.new_empty(axis_angle.shape[:-1] + (3, 3)) # [..., 3, 3]
        matrix[..., 0, 0] = one_m_cos * xx + cos_theta
        matrix[..., 0, 1] = one_m_cos * xy - sin_theta * z
        matrix[..., 0, 2] = one_m_cos * zx + sin_theta * y
        matrix[..., 1, 0] = one_m_cos * xy + sin_theta * z
        matrix[..., 1, 1] = one_m_cos * yy + cos_theta
        matrix[..., 1, 2] = one_m_cos * yz - sin_theta * x
        matrix[..., 2, 0] = one_m_cos * zx - sin_theta * y
        matrix[..., 2, 1] = one_m_cos * yz + sin_theta * x
        matrix[..., 2, 2] = one_m_cos * zz + cos_theta
        return matrix

    def _axis_angle_to_rotation_matrix_taylor(axis_angle: torch.Tensor) -> torch.Tensor:
        """Using Taylor's expansion, works for small `theta` value."""

        x, y, z = axis_angle.unbind(-1) # [...]
        o = torch.ones_like(x)

        matrix = x.new_empty(axis_angle.shape[:-1] + (3, 3)) # [..., 3, 3]
        matrix[..., 0, 0] = o
        matrix[..., 0, 1] = -z
        matrix[..., 0, 2] = y
        matrix[..., 1, 0] = z
        matrix[..., 1, 1] = o
        matrix[..., 1, 2] = -x
        matrix[..., 2, 0] = -y
        matrix[..., 2, 1] = x
        matrix[..., 2, 2] = o
        return matrix

    theta_sq = axis_angle.square().sum(-1)

    matrix_large = _axis_angle_to_rotation_matrix(axis_angle, theta_sq)
    matrix_small = _axis_angle_to_rotation_matrix_taylor(axis_angle)

    mask = (theta_sq > 1e-6).unsqueeze(-1).unsqueeze(-1) # [..., 1, 1]

    matrix = torch.where(mask, matrix_large, matrix_small)

    return matrix


def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert axis-angle vector to quaternion.

    Args:
        `axis_angle` (torch.Tensor): Input axis-angle vector of shape [..., 3],
            where its direction aligning with the axis direction and its norm represents rotation angles.

    Returns:
        (torch.Tensor): Quaternion of shape [..., 4].
    """
    x, y, z = axis_angle.unbind(-1)
    theta = axis_angle.norm(dim=-1)
    theta_half = theta * 0.5

    w = theta_half.cos()
    k = torch.where(
        theta > 0.0,
        theta_half.sin() / theta,
        0.5 * torch.ones_like(theta),
    )

    quaternion = torch.stack([w, x * k, y * k, z * k], dim=-1)
    return quaternion


def quaternion_to_axis_angle(quaternion: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion to axis-angle vector.
        - Quaternions are in the form (s, i, j, k), where s is the scalar part and i, j, k are the vector part.
        - Quaternions can be **unnormalized**.

    Args:
        `quaternion` (torch.Tensor): Input quaternion of shape [..., 4].

    Returns:
        (torch.Tensor): Axis-angle vector of shape [..., 3].
    """
    cos_theta_half, x, y, z = quaternion.unbind(-1)
    sin_theta_half = (x * x + y * y + z * z).sqrt()

    sign = torch.sign(cos_theta_half)
    theta = 2.0 * torch.atan2(sign * sin_theta_half, sign * cos_theta_half)

    k = torch.where(
        sin_theta_half > 0.0,
        theta / sin_theta_half,
        2.0 * torch.ones_like(theta),
    )

    axis_angle = torch.stack([x * k, y * k, z * k], dim=-1)
    return axis_angle


def quaternion_to_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion to rotation matrix.
        - Quaternions are in the form (s, i, j, k), where s is the scalar part and i, j, k are the vector part.
        - Quaternions can be **unnormalized**.

    Args:
        `quaternion` (torch.Tensor): Input quaternion of shape [..., 4].

    Returns:
        (torch.Tensor): Rotation matrix of shape [..., 3, 3].
    """
    s, i, j, k = quaternion.unbind(-1)

    n = 2.0 / quaternion.square().sum(dim=-1)

    matrix = quaternion.new_empty(quaternion.shape[:-1] + (3, 3)) # [..., 3, 3]
    matrix[..., 0, 0] = 1 - n * (j ** 2 + k ** 2)
    matrix[..., 0, 1] = n * (i * j - k * s)
    matrix[..., 0, 2] = n * (i * k + j * s)
    matrix[..., 1, 0] = n * (i * j + k * s)
    matrix[..., 1, 1] = 1 - n * (i ** 2 + k ** 2)
    matrix[..., 1, 2] = n * (j * k - i * s)
    matrix[..., 2, 0] = n * (i * k - j * s)
    matrix[..., 2, 1] = n * (j * k + i * s)
    matrix[..., 2, 2] = 1 - n * (i ** 2 + j ** 2)
    return matrix


def rotation_matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    return quaternion_to_axis_angle(rotation_matrix_to_quaternion(matrix))


def rotation_matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrix to quaternion.
        - Quaternions are in the form (s, i, j, k), where s is the scalar part and i, j, k are the vector part.
        - Quaternions are **normalized** and standardized.

    Args:
        `matrix` (torch.Tensor): Input rotation matrix of shape [..., 3, 3].

    Returns:
        (torch.Tensor): Quaternion of shape [..., 4].
    """
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = matrix.flatten(-2, -1).unbind(-1)

    q_abs_double = matrix.new_empty(matrix.shape[:-2] + (4,)) # [..., 4]
    q_abs_double[..., 0] = 1.0 + m00 + m11 + m22
    q_abs_double[..., 1] = 1.0 + m00 - m11 - m22
    q_abs_double[..., 2] = 1.0 - m00 + m11 - m22
    q_abs_double[..., 3] = 1.0 - m00 - m11 + m22
    q_abs_double = _sqrt_positive_part(q_abs_double)          # 2 * |q| [..., 4]

    q_by_sijk = matrix.new_empty(matrix.shape[:-2] + (4, 4)) # [..., 4, 4]
    q_by_sijk[..., 0, 0] = q_abs_double[..., 0].square()
    q_by_sijk[..., 0, 1] = m21 - m12
    q_by_sijk[..., 0, 2] = m02 - m20
    q_by_sijk[..., 0, 3] = m10 - m01                         # 4s * q
    q_by_sijk[..., 1, 0] = m21 - m12
    q_by_sijk[..., 1, 1] = q_abs_double[..., 1].square()
    q_by_sijk[..., 1, 2] = m10 + m01
    q_by_sijk[..., 1, 3] = m02 + m20                         # 4i * q
    q_by_sijk[..., 2, 0] = m02 - m20
    q_by_sijk[..., 2, 1] = m10 + m01
    q_by_sijk[..., 2, 2] = q_abs_double[..., 2].square()
    q_by_sijk[..., 2, 3] = m12 + m21                         # 4j * q
    q_by_sijk[..., 3, 0] = m10 - m01
    q_by_sijk[..., 3, 1] = m02 + m20
    q_by_sijk[..., 3, 2] = m12 + m21
    q_by_sijk[..., 3, 3] = q_abs_double[..., 3].square()     # 4k * q

    # We floor here at 0.1 but the exact level is not important;
    # if `q_abs_double` is small, the candidate won't be picked.
    q_candidates = q_by_sijk / (2.0 * q_abs_double.unsqueeze(-1).clamp_min(0.1)) # [..., 4, 4]

    # If not for numerical problems, `q_candidates[i]` should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator).
    out = q_candidates[F.one_hot(q_abs_double.argmax(dim=-1), 4) > 0.5, :].reshape(matrix.shape[:-2] + (4,))

    return standardize_quaternion(out)


def random_quaternion(shape: torch.Size, dtype: torch.dtype | None = None, device: torch.device | None = None) -> torch.Tensor:
    """
    Generate random quaternions representing uniformly distributed rotations.
        - The distribution is uniform over SO(3) (Haar measure).
        - Quaternions are in the form (s, i, j, k), where s is the scalar part and i, j, k are the vector part.
        - Quaternions are **normalized** and standardized.

    Args:
        `shape` (torch.Size): The batch shape of the output quaternions.
        `dtype` (torch.dtype | None): Data type of the output tensor.
        `device` (torch.device | None): Device of the output tensor.

    Returns:
        (torch.Tensor): Random quaternions of shape [..., 4].
    """
    # Uniformly sampling over S3 using Shoemake's algorithm (Subgroup Algorithm)
    # Reference: "Uniform Random Rotations", Ken Shoemake, Graphics Gems III.
    u1, u2, u3 = torch.rand(3, *shape, dtype=dtype, device=device)

    quaternion = u1.new_empty(*shape, 4)
    quaternion[..., 0] = torch.sqrt(u1) * torch.cos(2 * torch.pi * u3)     # s
    quaternion[..., 1] = torch.sqrt(1 - u1) * torch.sin(2 * torch.pi * u2) # i
    quaternion[..., 2] = torch.sqrt(1 - u1) * torch.cos(2 * torch.pi * u2) # j
    quaternion[..., 3] = torch.sqrt(u1) * torch.sin(2 * torch.pi * u3)     # k
    return standardize_quaternion(quaternion)


def random_rotation_matrix(shape: torch.Size, dtype: torch.dtype | None = None, device: torch.device | None = None) -> torch.Tensor:
    """
    Generate random rotation matrices representing uniformly distributed rotations.
        - The distribution is uniform over SO(3) (Haar measure).

    Args:
        `shape` (torch.Size): The batch shape of the output matrices.
        `dtype` (torch.dtype | None): Data type of the output tensor.
        `device` (torch.device | None): Device of the output tensor.

    Returns:
        (torch.Tensor): Random rotation matrices of shape [..., 3, 3].
    """
    return quaternion_to_rotation_matrix(random_quaternion(shape, dtype=dtype, device=device))
