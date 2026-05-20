# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Torch math utilities for the Newton repro toolkit.

The functions in this module are **copied verbatim** from
:mod:`isaaclab.utils.math` so per-bundle MDP scripts can use the same
quaternion / frame-transform / scaling / sampling helpers without importing
Isaac Lab. The Warp-native counterparts live in :mod:`envs.math.warp` and a
pytest-driven parity check keeps the two surfaces in sync
(:mod:`test.test_math`).

Source: ``source/isaaclab/isaaclab/utils/math.py``. When re-syncing this
module, walk the upstream file and copy each function listed in the
``__all__`` below.

Quaternion convention is ``(x, y, z, w)`` (Hamilton, xyzw layout).
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional

__all__ = [
    # scaling
    "scale_transform",
    "unscale_transform",
    "saturate",
    "normalize",
    "wrap_to_pi",
    "copysign",
    # quaternion
    "quat_unique",
    "matrix_from_quat",
    "quat_conjugate",
    "quat_inv",
    "quat_from_euler_xyz",
    "quat_from_matrix",
    "euler_xyz_from_quat",
    "axis_angle_from_quat",
    "quat_from_angle_axis",
    "quat_mul",
    "yaw_quat",
    "quat_apply",
    "quat_apply_inverse",
    # frame transforms
    "skew_symmetric_matrix",
    "combine_frame_transforms",
    "subtract_frame_transforms",
    "compute_pose_error",
    "transform_points",
    # sampling
    "sample_uniform",
    "sample_log_uniform",
    "sample_gaussian",
]


# ----------------------------------------------------------------------------
# Scaling / general
# ----------------------------------------------------------------------------


@torch.jit.script
def scale_transform(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Normalizes a given input tensor to a range of [-1, 1].

    .. note::
        It uses pytorch broadcasting functionality to deal with batched input.

    Args:
        x: Input tensor of shape (N, dims).
        lower: The minimum value of the tensor. Shape is (N, dims) or (dims,).
        upper: The maximum value of the tensor. Shape is (N, dims) or (dims,).

    Returns:
        Normalized transform of the tensor. Shape is (N, dims).
    """
    offset = (lower + upper) * 0.5
    return 2 * (x - offset) / (upper - lower)


@torch.jit.script
def unscale_transform(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """De-normalizes a given input tensor from range of [-1, 1] to (lower, upper).

    .. note::
        It uses pytorch broadcasting functionality to deal with batched input.

    Args:
        x: Input tensor of shape (N, dims).
        lower: The minimum value of the tensor. Shape is (N, dims) or (dims,).
        upper: The maximum value of the tensor. Shape is (N, dims) or (dims,).

    Returns:
        De-normalized transform of the tensor. Shape is (N, dims).
    """
    offset = (lower + upper) * 0.5
    return x * (upper - lower) * 0.5 + offset


@torch.jit.script
def saturate(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Clamps a given input tensor to (lower, upper).

    It uses pytorch broadcasting functionality to deal with batched input.

    Args:
        x: Input tensor of shape (N, dims).
        lower: The minimum value of the tensor. Shape is (N, dims) or (dims,).
        upper: The maximum value of the tensor. Shape is (N, dims) or (dims,).

    Returns:
        Clamped transform of the tensor. Shape is (N, dims).
    """
    return torch.max(torch.min(x, upper), lower)


@torch.jit.script
def normalize(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Normalizes a given input tensor to unit length.

    Args:
        x: Input tensor of shape (N, dims).
        eps: A small value to avoid division by zero. Defaults to 1e-9.

    Returns:
        Normalized tensor of shape (N, dims).
    """
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)


@torch.jit.script
def wrap_to_pi(angles: torch.Tensor) -> torch.Tensor:
    r"""Wraps input angles (in radians) to the range :math:`[-\pi, \pi]`.

    Args:
        angles: Input angles of any shape.

    Returns:
        Angles in the range :math:`[-\pi, \pi]`.
    """
    wrapped_angle = (angles + torch.pi) % (2 * torch.pi)
    return torch.where((wrapped_angle == 0) & (angles > 0), torch.pi, wrapped_angle - torch.pi)


@torch.jit.script
def copysign(mag: float, other: torch.Tensor) -> torch.Tensor:
    """Create a new floating-point tensor with the magnitude of input and the sign of other, element-wise.

    Args:
        mag: The magnitude scalar.
        other: The tensor containing values whose signbits are applied to magnitude.

    Returns:
        The output tensor.
    """
    mag_torch = abs(mag) * torch.ones_like(other)
    return torch.copysign(mag_torch, other)


# ----------------------------------------------------------------------------
# Quaternion / rotation
# ----------------------------------------------------------------------------


@torch.jit.script
def quat_unique(q: torch.Tensor) -> torch.Tensor:
    """Convert a unit quaternion to a standard form where the real part is non-negative.

    Args:
        q: The quaternion orientation in (x, y, z, w). Shape is (..., 4).

    Returns:
        Standardized quaternions. Shape is (..., 4).
    """
    return torch.where(q[..., 3:4] < 0, -q, q)


@torch.jit.script
def matrix_from_quat(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: The quaternion orientation in (x, y, z, w). Shape is (..., 4).

    Returns:
        Rotation matrices. The shape is (..., 3, 3).
    """
    i, j, k, r = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


@torch.jit.script
def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Computes the conjugate of a quaternion.

    Args:
        q: The quaternion orientation in (x, y, z, w). Shape is (..., 4).

    Returns:
        The conjugate quaternion in (x, y, z, w). Shape is (..., 4).
    """
    shape = q.shape
    q = q.reshape(-1, 4)
    return torch.cat((-q[..., :3], q[..., 3:4]), dim=-1).view(shape)


@torch.jit.script
def quat_inv(q: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Computes the inverse of a quaternion.

    Args:
        q: The quaternion orientation in (x, y, z, w). Shape is (N, 4).
        eps: A small value to avoid division by zero. Defaults to 1e-9.

    Returns:
        The inverse quaternion in (x, y, z, w). Shape is (N, 4).
    """
    return quat_conjugate(q) / q.pow(2).sum(dim=-1, keepdim=True).clamp(min=eps)


@torch.jit.script
def quat_from_euler_xyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as Euler angles in radians to Quaternions.

    Args:
        roll: Rotation around x-axis (in radians). Shape is (N,).
        pitch: Rotation around y-axis (in radians). Shape is (N,).
        yaw: Rotation around z-axis (in radians). Shape is (N,).

    Returns:
        The quaternion in (x, y, z, w). Shape is (N, 4).
    """
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp
    return torch.stack([qx, qy, qz, qw], dim=-1)


@torch.jit.script
def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


@torch.jit.script
def quat_from_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: The rotation matrices. Shape is (..., 3, 3).

    Returns:
        The quaternion in (x, y, z, w). Shape is (..., 4). Rows whose input is not a
        valid rotation are filled with NaN.
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(batch_dim + (9,)), dim=-1)

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    quat_by_rijk = torch.stack(
        [
            torch.stack([m21 - m12, m02 - m20, m10 - m01, q_abs[..., 0] ** 2], dim=-1),
            torch.stack([q_abs[..., 1] ** 2, m10 + m01, m02 + m20, m21 - m12], dim=-1),
            torch.stack([m10 + m01, q_abs[..., 2] ** 2, m12 + m21, m02 - m20], dim=-1),
            torch.stack([m20 + m02, m21 + m12, q_abs[..., 3] ** 2, m10 - m01], dim=-1),
        ],
        dim=-2,
    )

    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    quat = quat_candidates[torch.nn.functional.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(
        batch_dim + (4,)
    )
    invalid = (quat.norm(p=2, dim=-1, keepdim=True) - 1.0).abs() > 2e-5
    return torch.where(invalid, torch.full_like(quat, float("nan")), quat)


@torch.jit.script
def euler_xyz_from_quat(
    quat: torch.Tensor, wrap_to_2pi: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert rotations given as quaternions to Euler angles in radians.

    Args:
        quat: The quaternion orientation in (x, y, z, w). Shape is (N, 4).
        wrap_to_2pi: Whether to wrap output Euler angles into [0, 2π). If False,
            angles are returned in the default range (−π, π]. Defaults to False.

    Returns:
        A tuple containing roll-pitch-yaw. Each element is a tensor of shape (N,).
    """
    q_x, q_y, q_z, q_w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    sin_roll = 2.0 * (q_w * q_x + q_y * q_z)
    cos_roll = 1 - 2 * (q_x * q_x + q_y * q_y)
    roll = torch.atan2(sin_roll, cos_roll)

    sin_pitch = 2.0 * (q_w * q_y - q_z * q_x)
    pitch = torch.where(torch.abs(sin_pitch) >= 1, copysign(torch.pi / 2.0, sin_pitch), torch.asin(sin_pitch))

    sin_yaw = 2.0 * (q_w * q_z + q_x * q_y)
    cos_yaw = 1 - 2 * (q_y * q_y + q_z * q_z)
    yaw = torch.atan2(sin_yaw, cos_yaw)

    if wrap_to_2pi:
        return roll % (2 * torch.pi), pitch % (2 * torch.pi), yaw % (2 * torch.pi)
    return roll, pitch, yaw


@torch.jit.script
def axis_angle_from_quat(quat: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """Convert rotations given as quaternions to axis/angle.

    Args:
        quat: The quaternion orientation in (x, y, z, w). Shape is (..., 4).
        eps: The tolerance for Taylor approximation. Defaults to 1.0e-6.

    Returns:
        Rotations given as a vector in axis angle form. Shape is (..., 3).
    """
    quat = quat * (1.0 - 2.0 * (quat[..., 3:4] < 0.0))
    mag = torch.linalg.norm(quat[..., :3], dim=-1)
    half_angle = torch.atan2(mag, quat[..., 3])
    angle = 2.0 * half_angle
    sin_half_angles_over_angles = torch.where(
        angle.abs() > eps, torch.sin(half_angle) / angle, 0.5 - angle * angle / 48
    )
    return quat[..., :3] / sin_half_angles_over_angles.unsqueeze(-1)


@torch.jit.script
def quat_from_angle_axis(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as angle-axis to quaternions.

    Args:
        angle: The angle turned anti-clockwise in radians around the vector's direction. Shape is (N,).
        axis: The axis of rotation. Shape is (N, 3).

    Returns:
        The quaternion in (x, y, z, w). Shape is (N, 4).
    """
    theta = (angle / 2).unsqueeze(-1)
    xyz = normalize(axis) * theta.sin()
    w = theta.cos()
    return normalize(torch.cat([xyz, w], dim=-1))


@torch.jit.script
def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply two quaternions together.

    Args:
        q1: The first quaternion in (x, y, z, w). Shape is (..., 4).
        q2: The second quaternion in (x, y, z, w). Shape is (..., 4).

    Returns:
        The product of the two quaternions in (x, y, z, w). Shape is (..., 4).
    """
    if q1.shape != q2.shape:
        msg = f"Expected input quaternion shape mismatch: {q1.shape} != {q2.shape}."
        raise ValueError(msg)
    shape = q1.shape
    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)
    x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    return torch.stack([x, y, z, w], dim=-1).view(shape)


@torch.jit.script
def yaw_quat(quat: torch.Tensor) -> torch.Tensor:
    """Extract the yaw component of a quaternion.

    Args:
        quat: The orientation in (x, y, z, w). Shape is (..., 4)

    Returns:
        A quaternion with only yaw component in (x, y, z, w).
    """
    shape = quat.shape
    quat_yaw = quat.view(-1, 4)
    qx = quat_yaw[:, 0]
    qy = quat_yaw[:, 1]
    qz = quat_yaw[:, 2]
    qw = quat_yaw[:, 3]
    yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    quat_yaw = torch.zeros_like(quat_yaw)
    quat_yaw[:, 2] = torch.sin(yaw / 2)
    quat_yaw[:, 3] = torch.cos(yaw / 2)
    quat_yaw = normalize(quat_yaw)
    return quat_yaw.view(shape)


@torch.jit.script
def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Apply a quaternion rotation to a vector.

    Args:
        quat: The quaternion in (x, y, z, w). Shape is (..., 4).
        vec: The vector in (x, y, z). Shape is (..., 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (..., 3).
    """
    shape = vec.shape
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)
    xyz = quat[:, :3]
    t = xyz.cross(vec, dim=-1) * 2
    return (vec + quat[:, 3:4] * t + xyz.cross(t, dim=-1)).view(shape)


@torch.jit.script
def quat_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Apply an inverse quaternion rotation to a vector.

    Args:
        quat: The quaternion in (x, y, z, w). Shape is (..., 4).
        vec: The vector in (x, y, z). Shape is (..., 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (..., 3).
    """
    shape = vec.shape
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)
    xyz = quat[:, :3]
    t = xyz.cross(vec, dim=-1) * 2
    return (vec - quat[:, 3:4] * t + xyz.cross(t, dim=-1)).view(shape)


# ----------------------------------------------------------------------------
# Frame transforms
# ----------------------------------------------------------------------------


@torch.jit.script
def skew_symmetric_matrix(vec: torch.Tensor) -> torch.Tensor:
    """Computes the skew-symmetric matrix of a vector.

    Args:
        vec: The input vector. Shape is (3,) or (N, 3).

    Returns:
        The skew-symmetric matrix. Shape is (1, 3, 3) or (N, 3, 3).
    """
    if vec.shape[-1] != 3:
        raise ValueError(f"Expected input vector shape mismatch: {vec.shape} != (..., 3).")
    if vec.ndim == 1:
        vec = vec.unsqueeze(0)
    skew_sym_mat = torch.zeros(vec.shape[0], 3, 3, device=vec.device, dtype=vec.dtype)
    skew_sym_mat[:, 0, 1] = -vec[:, 2]
    skew_sym_mat[:, 0, 2] = vec[:, 1]
    skew_sym_mat[:, 1, 2] = -vec[:, 0]
    skew_sym_mat[:, 1, 0] = vec[:, 2]
    skew_sym_mat[:, 2, 0] = -vec[:, 1]
    skew_sym_mat[:, 2, 1] = vec[:, 0]
    return skew_sym_mat


@torch.jit.script
def combine_frame_transforms(
    t01: torch.Tensor, q01: torch.Tensor, t12: torch.Tensor | None = None, q12: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Combine transformations between two reference frames into a stationary frame.

    :math:`T_{02} = T_{01} \times T_{12}`.

    Args:
        t01: Position of frame 1 w.r.t. frame 0. Shape is (N, 3).
        q01: Quaternion orientation of frame 1 w.r.t. frame 0 in (x, y, z, w). Shape is (N, 4).
        t12: Position of frame 2 w.r.t. frame 1. Shape is (N, 3). Defaults to None.
        q12: Quaternion orientation of frame 2 w.r.t. frame 1 in (x, y, z, w). Shape is (N, 4). Defaults to None.

    Returns:
        A tuple containing the position and orientation of frame 2 w.r.t. frame 0.
    """
    if q12 is not None:
        q02 = quat_mul(q01, q12)
    else:
        q02 = q01
    if t12 is not None:
        t02 = t01 + quat_apply(q01, t12)
    else:
        t02 = t01
    return t02, q02


def subtract_frame_transforms(
    t01: torch.Tensor, q01: torch.Tensor, t02: torch.Tensor | None = None, q02: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Subtract transformations between two reference frames.

    :math:`T_{12} = T_{01}^{-1} \times T_{02}`.

    Args:
        t01: Position of frame 1 w.r.t. frame 0. Shape is (N, 3).
        q01: Quaternion orientation of frame 1 w.r.t. frame 0 in (x, y, z, w). Shape is (N, 4).
        t02: Position of frame 2 w.r.t. frame 0. Shape is (N, 3). Defaults to None.
        q02: Quaternion orientation of frame 2 w.r.t. frame 0 in (x, y, z, w). Shape is (N, 4). Defaults to None.

    Returns:
        A tuple containing the position and orientation of frame 2 w.r.t. frame 1.
    """
    q10 = quat_inv(q01)
    if q02 is not None:
        q12 = quat_mul(q10, q02)
    else:
        q12 = q10
    if t02 is not None:
        t12 = quat_apply(q10, t02 - t01)
    else:
        t12 = quat_apply(q10, -t01)
    return t12, q12


def compute_pose_error(
    t01: torch.Tensor,
    q01: torch.Tensor,
    t02: torch.Tensor,
    q02: torch.Tensor,
    rot_error_type: Literal["quat", "axis_angle"] = "axis_angle",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the position and orientation error between source and target frames.

    Args:
        t01: Position of source frame. Shape is (N, 3).
        q01: Quaternion orientation of source frame in (x, y, z, w). Shape is (N, 4).
        t02: Position of target frame. Shape is (N, 3).
        q02: Quaternion orientation of target frame in (x, y, z, w). Shape is (N, 4).
        rot_error_type: The rotation error type to return: "quat", "axis_angle". Defaults to "axis_angle".

    Returns:
        A tuple containing position and orientation error.
    """
    source_quat_norm = quat_mul(q01, quat_conjugate(q01))[:, 3]
    source_quat_inv = quat_conjugate(q01) / source_quat_norm.unsqueeze(-1)
    quat_error = quat_mul(q02, source_quat_inv)

    pos_error = t02 - t01

    if rot_error_type == "quat":
        return pos_error, quat_error
    elif rot_error_type == "axis_angle":
        axis_angle_error = axis_angle_from_quat(quat_error)
        return pos_error, axis_angle_error
    else:
        raise ValueError(f"Unsupported orientation error type: {rot_error_type}. Valid: 'quat', 'axis_angle'.")


def transform_points(
    points: torch.Tensor, pos: torch.Tensor | None = None, quat: torch.Tensor | None = None
) -> torch.Tensor:
    r"""Transform input points in a given frame to a target frame.

    :math:`p_{target} = R_{target} \times p_{source} + t_{target}`.

    Args:
        points: Points to transform. Shape is (N, P, 3) or (P, 3).
        pos: Position of the target frame. Shape is (N, 3) or (3,). Defaults to None.
        quat: Quaternion orientation of the target frame in (x, y, z, w). Shape is (N, 4) or (4,). Defaults to None.

    Returns:
        Transformed points in the target frame.
    """
    points_batch = points.clone()
    is_batched = points_batch.dim() == 3
    if points_batch.dim() == 2:
        points_batch = points_batch[None]
    if points_batch.dim() != 3:
        raise ValueError(f"Expected points to have dim = 2 or dim = 3: got shape {points.shape}")
    if not (pos is None or pos.dim() == 1 or pos.dim() == 2):
        raise ValueError(f"Expected pos to have dim = 1 or dim = 2: got shape {pos.shape}")
    if not (quat is None or quat.dim() == 1 or quat.dim() == 2):
        raise ValueError(f"Expected quat to have dim = 1 or dim = 2: got shape {quat.shape}")
    if quat is not None:
        rot_mat = matrix_from_quat(quat)
        if rot_mat.dim() == 2:
            rot_mat = rot_mat[None]
        points_batch = torch.matmul(rot_mat, points_batch.transpose_(1, 2))
        points_batch = points_batch.transpose_(1, 2)
    if pos is not None:
        if pos.dim() == 1:
            pos = pos[None, None, :]
        else:
            pos = pos[:, None, :]
        points_batch += pos
    if not is_batched:
        points_batch = points_batch.squeeze(0)
    return points_batch


# ----------------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------------


def sample_uniform(
    lower: torch.Tensor | float, upper: torch.Tensor | float, size: int | tuple[int, ...], device: str
) -> torch.Tensor:
    """Sample uniformly within a range.

    Args:
        lower: Lower bound of uniform range.
        upper: Upper bound of uniform range.
        size: The shape of the tensor.
        device: Device to create tensor on.

    Returns:
        Sampled tensor. Shape is based on ``size``.
    """
    if isinstance(size, int):
        size = (size,)
    return torch.rand(*size, device=device) * (upper - lower) + lower


def sample_log_uniform(
    lower: torch.Tensor | float, upper: torch.Tensor | float, size: int | tuple[int, ...], device: str
) -> torch.Tensor:
    r"""Sample using log-uniform distribution within a range.

    :math:`x = \exp(\text{uniform}(\log(\text{lower}), \log(\text{upper})))`.

    Args:
        lower: Lower bound of uniform range.
        upper: Upper bound of uniform range.
        size: The shape of the tensor.
        device: Device to create tensor on.

    Returns:
        Sampled tensor.
    """
    if not isinstance(lower, torch.Tensor):
        lower = torch.tensor(lower, dtype=torch.float, device=device)
    if not isinstance(upper, torch.Tensor):
        upper = torch.tensor(upper, dtype=torch.float, device=device)
    return torch.exp(sample_uniform(torch.log(lower), torch.log(upper), size, device))


def sample_gaussian(
    mean: torch.Tensor | float, std: torch.Tensor | float, size: int | tuple[int, ...], device: str
) -> torch.Tensor:
    """Sample using gaussian distribution.

    Args:
        mean: Mean of the gaussian.
        std: Std of the gaussian.
        size: The shape of the tensor.
        device: Device to create tensor on.

    Returns:
        Sampled tensor.
    """
    if isinstance(mean, float):
        if isinstance(size, int):
            size = (size,)
        return torch.normal(mean=mean, std=std, size=size).to(device=device)
    else:
        return torch.normal(mean=mean, std=std).to(device=device)
