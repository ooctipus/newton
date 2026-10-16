# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""FP64 references for Warp's transform and quaternion laws, for cache tests."""

import numpy as np


def quat_multiply(a, b):
    """Multiply xyzw quaternions independently in FP64."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return np.concatenate(
        (
            a[..., 3:] * b[..., :3] + b[..., 3:] * a[..., :3] + np.cross(a[..., :3], b[..., :3]),
            a[..., 3:] * b[..., 3:] - np.sum(a[..., :3] * b[..., :3], axis=-1, keepdims=True),
        ),
        axis=-1,
    )


def rotate_inverse(q, vector):
    """Express a world vector in its current normalized orientation in FP64."""
    q = np.asarray(q, np.float64).copy()
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    q[..., :3] *= -1
    vector = np.asarray(vector, np.float64)
    twice = 2 * np.cross(q[..., :3], vector)
    return vector + q[..., 3:] * twice + np.cross(q[..., :3], twice)


def rotate(q, vector):
    """Evaluate Warp's quaternion-rotation expression in FP64 without renormalizing inputs."""
    return (
        vector * (2 * q[..., 3:] ** 2 - 1)
        + q[..., :3] * (2 * np.sum(q[..., :3] * vector, axis=-1, keepdims=True))
        + 2 * q[..., 3:] * np.cross(q[..., :3], vector)
    )


def compose(a, b):
    """Compose two transforms independently in FP64."""
    return np.concatenate((a[..., :3] + rotate(a[..., 3:], b[..., :3]), quat_multiply(a[..., 3:], b[..., 3:])), axis=-1)


def inverse(transform):
    """Invert a transform with the conjugate quaternion in FP64."""
    q = transform[..., 3:].copy()
    q[..., :3] *= -1
    return np.concatenate((rotate(q, -transform[..., :3]), q), axis=-1)
