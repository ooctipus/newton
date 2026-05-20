# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp-native math utilities for the Newton repro toolkit.

This module mirrors :mod:`envs.math.torch` for operations where Isaac Lab's
implementation does something Warp's standard library does not provide
directly. **Operations that are one-liner wrappers around a Warp builtin are
not re-exposed here** -- callers should use the Warp builtin directly so
there is exactly one canonical name for each operation. The omitted
wrappers, and the Warp builtin to use in their place:

* ``quat_apply(q, v)`` -> :func:`wp.quat_rotate`
* ``quat_apply_inverse(q, v)`` -> :func:`wp.quat_rotate_inv`
* ``matrix_from_quat(q)`` -> :func:`wp.quat_to_matrix`
* ``quat_from_matrix(m)`` -> :func:`wp.quat_from_matrix`

The two surfaces are kept in sync by :mod:`test.test_math` (random-input
fuzz with ``np.allclose``); the parity tests compare the torch helpers
against the Warp builtins directly for the operations above.

Quaternion convention is ``(x, y, z, w)`` (Hamilton, xyzw layout) -- this
matches both Isaac Lab and Warp's native ``quatf`` / ``quatd`` types.

Multi-precision strategy
========================

The patterns follow Newton's own utility-math modules (see
``newton/_src/solvers/kamino/_src/core/math.py`` and
``newton/_src/solvers/style3d/linear_solver.py``):

* **Scalar -> scalar** helpers (e.g. ``wrap_to_pi``, ``saturate``,
  ``scale_transform``) take ``wp.Float`` type-variables so Warp specializes
  the function per call site for any supported float width (f16/f32/f64).
* **Typed vec / quat / matrix** helpers (e.g. ``quat_mul``, ``quat_from_euler_xyz``)
  use explicit ``wp.quatf`` / ``wp.vec3f`` / ``wp.mat33f`` typings. v1 ships
  f32 only; an f64 overload is a pure addition later (see footnote on
  precision at the bottom of this module).
* **Array-launched kernels** (e.g. ``sample_uniform_kernel``) declare
  ``wp.array[Any]`` and pre-instantiate dtype variants via ``wp.overload``
  so first-call JIT does not happen inside a CUDA graph capture context
  (matches Newton's style3d linear_solver.py).
* ``wp.set_module_options({"enable_backward": False})`` is set at the top so
  the module compiles fast and stays small; repro math never needs gradients.

Precision footnote (Warp issue #485): if you add f64 paths, declare
precision constants **at module level** (e.g.
``_PI_F64 = wp.float64(wp.PI)``). The expression ``wp.float64(wp.PI)`` inside
a kernel truncates ``wp.PI`` to f32 first before casting up, losing
double-precision bits.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp
from warp._src.types import Float

__all__ = [
    # scaling
    "scale_transform",
    "unscale_transform",
    "saturate",
    "normalize_vec3",
    "wrap_to_pi",
    # quaternion
    "quat_unique",
    "quat_conjugate",
    "quat_inv",
    "quat_from_euler_xyz",
    "euler_xyz_from_quat",
    "axis_angle_from_quat",
    "quat_from_angle_axis",
    "quat_mul",
    "yaw_quat",
    # frame transforms
    "skew_symmetric_matrix",
    "combine_frame_transforms",
    "subtract_frame_transforms",
    "transform_point",
    # sampling (per-thread, returning scalars)
    "sample_uniform_scalar",
    "sample_log_uniform_scalar",
    "sample_gaussian_scalar",
    # array-launched kernels (dtype-overloaded)
    "sample_uniform_kernel",
    "sample_log_uniform_kernel",
    "sample_gaussian_kernel",
]

wp.set_module_options({"enable_backward": False})


# ----------------------------------------------------------------------------
# Typed constants
# ----------------------------------------------------------------------------

PI = wp.constant(wp.float32(np.pi))
"""Single-precision PI."""

TWO_PI = wp.constant(wp.float32(2.0 * np.pi))
"""Single-precision 2*PI."""

HALF_PI = wp.constant(wp.float32(0.5 * np.pi))
"""Single-precision PI/2."""

UNIT_X = wp.constant(wp.vec3f(1.0, 0.0, 0.0))
"""Unit X-axis vector."""

UNIT_Y = wp.constant(wp.vec3f(0.0, 1.0, 0.0))
"""Unit Y-axis vector."""

UNIT_Z = wp.constant(wp.vec3f(0.0, 0.0, 1.0))
"""Unit Z-axis vector."""

QUAT_IDENTITY = wp.constant(wp.quatf(0.0, 0.0, 0.0, 1.0))
"""Identity quaternion (x=0, y=0, z=0, w=1)."""


# ----------------------------------------------------------------------------
# Scalar generics (specialize on any float width)
# ----------------------------------------------------------------------------


@wp.func
def scale_transform(x: Float, lower: Float, upper: Float) -> Float:
    """Normalize *x* in ``[lower, upper]`` to ``[-1, 1]``."""
    offset = (lower + upper) * 0.5
    return 2.0 * (x - offset) / (upper - lower)


@wp.func
def unscale_transform(x: Float, lower: Float, upper: Float) -> Float:
    """De-normalize *x* in ``[-1, 1]`` to ``[lower, upper]``."""
    offset = (lower + upper) * 0.5
    return x * (upper - lower) * 0.5 + offset


@wp.func
def saturate(x: Float, lower: Float, upper: Float) -> Float:
    """Clamp scalar *x* to ``[lower, upper]``."""
    return wp.max(wp.min(x, upper), lower)


@wp.func
def wrap_to_pi(angle: Float) -> Float:
    """Wrap a scalar angle (radians) to ``[-pi, pi]``.

    Uses Python ``math.pi`` literals; Warp narrows them to the caller's
    precision at compile time (safe for f32; if f64 is added in the future,
    declare ``_PI_F64 = wp.float64(wp.PI)`` at module level and substitute).
    """
    two_pi = 2.0 * np.pi
    pi = np.pi
    wrapped = (angle + pi) - wp.floor((angle + pi) / two_pi) * two_pi
    return wrapped - pi


# ----------------------------------------------------------------------------
# Vec3 (f32 only for v1)
# ----------------------------------------------------------------------------


@wp.func
def normalize_vec3(x: wp.vec3f, eps: wp.float32) -> wp.vec3f:
    """Normalize a vec3 to unit length, clamping the norm at *eps*."""
    n = wp.length(x)
    return x / wp.max(n, eps)


# ----------------------------------------------------------------------------
# Quaternion (f32 only for v1)
# ----------------------------------------------------------------------------


@wp.func
def quat_unique(q: wp.quatf) -> wp.quatf:
    """Standardize *q* so the real part is non-negative."""
    if q[3] < 0.0:
        return wp.quatf(-q[0], -q[1], -q[2], -q[3])
    return q


@wp.func
def quat_conjugate(q: wp.quatf) -> wp.quatf:
    """Quaternion conjugate (negate the vector part)."""
    return wp.quatf(-q[0], -q[1], -q[2], q[3])


@wp.func
def quat_inv(q: wp.quatf, eps: wp.float32) -> wp.quatf:
    """Quaternion inverse (conjugate / norm^2)."""
    n2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]
    inv_n2 = 1.0 / wp.max(n2, eps)
    return wp.quatf(-q[0] * inv_n2, -q[1] * inv_n2, -q[2] * inv_n2, q[3] * inv_n2)


@wp.func
def quat_mul(q1: wp.quatf, q2: wp.quatf) -> wp.quatf:
    """Hamilton product of two quaternions (xyzw layout)."""
    x1 = q1[0]
    y1 = q1[1]
    z1 = q1[2]
    w1 = q1[3]
    x2 = q2[0]
    y2 = q2[1]
    z2 = q2[2]
    w2 = q2[3]
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return wp.quatf(x, y, z, w)


@wp.func
def quat_from_euler_xyz(roll: wp.float32, pitch: wp.float32, yaw: wp.float32) -> wp.quatf:
    """Build a quaternion from XYZ-convention Euler angles (matches Isaac Lab)."""
    cy = wp.cos(yaw * 0.5)
    sy = wp.sin(yaw * 0.5)
    cr = wp.cos(roll * 0.5)
    sr = wp.sin(roll * 0.5)
    cp = wp.cos(pitch * 0.5)
    sp = wp.sin(pitch * 0.5)
    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp
    return wp.quatf(qx, qy, qz, qw)


@wp.func
def euler_xyz_from_quat(q: wp.quatf) -> wp.vec3f:
    """Recover XYZ-convention Euler angles from a quaternion (matches Isaac Lab).

    Returns ``(roll, pitch, yaw)`` in radians; pitch is clamped to ``[-pi/2, pi/2]``
    via ``asin`` of a saturated input.
    """
    q_x = q[0]
    q_y = q[1]
    q_z = q[2]
    q_w = q[3]
    sin_roll = 2.0 * (q_w * q_x + q_y * q_z)
    cos_roll = 1.0 - 2.0 * (q_x * q_x + q_y * q_y)
    roll = wp.atan2(sin_roll, cos_roll)

    sin_pitch = 2.0 * (q_w * q_y - q_z * q_x)
    sin_pitch_clamped = wp.clamp(sin_pitch, -1.0, 1.0)
    pitch = wp.asin(sin_pitch_clamped)

    sin_yaw = 2.0 * (q_w * q_z + q_x * q_y)
    cos_yaw = 1.0 - 2.0 * (q_y * q_y + q_z * q_z)
    yaw = wp.atan2(sin_yaw, cos_yaw)
    return wp.vec3f(roll, pitch, yaw)


@wp.func
def axis_angle_from_quat(q: wp.quatf, eps: wp.float32) -> wp.vec3f:
    """Convert a quaternion to an axis-angle vector (magnitude = angle)."""
    sign = wp.float32(1.0)
    if q[3] < 0.0:
        sign = wp.float32(-1.0)
    qx = q[0] * sign
    qy = q[1] * sign
    qz = q[2] * sign
    qw = q[3] * sign
    mag = wp.sqrt(qx * qx + qy * qy + qz * qz)
    half_angle = wp.atan2(mag, qw)
    angle = 2.0 * half_angle
    if wp.abs(angle) > eps:
        scale = wp.sin(half_angle) / angle
    else:
        scale = 0.5 - angle * angle / 48.0
    return wp.vec3f(qx / scale, qy / scale, qz / scale)


@wp.func
def quat_from_angle_axis(angle: wp.float32, axis: wp.vec3f) -> wp.quatf:
    """Build a quaternion from an axis (normalized inside) and an angle."""
    axis_n = normalize_vec3(axis, 1.0e-9)
    half = angle * 0.5
    s = wp.sin(half)
    c = wp.cos(half)
    return wp.normalize(wp.quatf(axis_n[0] * s, axis_n[1] * s, axis_n[2] * s, c))


@wp.func
def yaw_quat(q: wp.quatf) -> wp.quatf:
    """Extract the yaw-only quaternion from a full quaternion."""
    qx = q[0]
    qy = q[1]
    qz = q[2]
    qw = q[3]
    yaw = wp.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )
    half = yaw * 0.5
    return wp.normalize(wp.quatf(0.0, 0.0, wp.sin(half), wp.cos(half)))


@wp.func
def skew_symmetric_matrix(v: wp.vec3f) -> wp.mat33f:
    """Return the 3x3 skew-symmetric matrix of *v*."""
    return wp.mat33f(
        0.0,
        -v[2],
        v[1],
        v[2],
        0.0,
        -v[0],
        -v[1],
        v[0],
        0.0,
    )


@wp.func
def combine_frame_transforms(t01: wp.vec3f, q01: wp.quatf, t12: wp.vec3f, q12: wp.quatf) -> wp.transformf:
    """Compose two SE(3) transforms ``T_{02} = T_{01} * T_{12}``."""
    t02 = t01 + wp.quat_rotate(q01, t12)
    q02 = quat_mul(q01, q12)
    return wp.transformf(t02, q02)


@wp.func
def subtract_frame_transforms(t01: wp.vec3f, q01: wp.quatf, t02: wp.vec3f, q02: wp.quatf) -> wp.transformf:
    """Compute the relative transform ``T_{12} = T_{01}^{-1} * T_{02}``."""
    q10 = quat_conjugate(q01)
    t12 = wp.quat_rotate(q10, t02 - t01)
    q12 = quat_mul(q10, q02)
    return wp.transformf(t12, q12)


@wp.func
def transform_point(p: wp.vec3f, pos: wp.vec3f, q: wp.quatf) -> wp.vec3f:
    """Transform a point ``p`` by ``(pos, q)``: ``q * p + pos``."""
    return wp.quat_rotate(q, p) + pos


@wp.func
def sample_uniform_scalar(state: wp.uint32, lower: wp.float32, upper: wp.float32) -> wp.float32:
    """Single uniform sample in ``[lower, upper)`` using a per-thread RNG state."""
    return lower + (upper - lower) * wp.randf(state)


@wp.func
def sample_log_uniform_scalar(state: wp.uint32, lower: wp.float32, upper: wp.float32) -> wp.float32:
    """Single log-uniform sample in ``[lower, upper)``."""
    return wp.exp(wp.log(lower) + (wp.log(upper) - wp.log(lower)) * wp.randf(state))


@wp.func
def sample_gaussian_scalar(state: wp.uint32, mean: wp.float32, std: wp.float32) -> wp.float32:
    """Single Gaussian sample with the given mean and std."""
    return mean + std * wp.randn(state)


@wp.kernel
def sample_uniform_kernel(seed: wp.int32, lower: wp.float32, upper: wp.float32, out: wp.array[Any]):
    tid = wp.tid()
    state = wp.rand_init(seed + tid)
    out[tid] = lower + (upper - lower) * wp.randf(state)


# Forward-declare instances of the generic kernel to support graph capture on CUDA <12.3 drivers
wp.overload(sample_uniform_kernel, {"out": wp.array[wp.float32]})


@wp.kernel
def sample_log_uniform_kernel(seed: wp.int32, lower: wp.float32, upper: wp.float32, out: wp.array[Any]):
    tid = wp.tid()
    state = wp.rand_init(seed + tid)
    out[tid] = wp.exp(wp.log(lower) + (wp.log(upper) - wp.log(lower)) * wp.randf(state))


# Forward-declare instances of the generic kernel to support graph capture on CUDA <12.3 drivers
wp.overload(sample_log_uniform_kernel, {"out": wp.array[wp.float32]})


@wp.kernel
def sample_gaussian_kernel(seed: wp.int32, mean: wp.float32, std: wp.float32, out: wp.array[Any]):
    tid = wp.tid()
    state = wp.rand_init(seed + tid)
    out[tid] = mean + std * wp.randn(state)


# Forward-declare instances of the generic kernel to support graph capture on CUDA <12.3 drivers
wp.overload(sample_gaussian_kernel, {"out": wp.array[wp.float32]})
