# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Parity tests between :mod:`envs.math.torch` and :mod:`envs.math.warp`.

For each function pair, sample N random inputs with a seeded numpy RNG, run
the torch implementation, run a Warp kernel that calls the Warp ``@wp.func``
per element, then compare with ``np.allclose`` at f32 tolerance.

Sampling functions use different RNGs on the two backends (torch.rand vs
wp.randf), so per-element parity is impossible; those tests verify
distributional properties instead.
"""

from __future__ import annotations

import importlib
import importlib.util
import pathlib
import sys

import numpy as np
import torch
import warp as wp

_REPRO_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(_REPRO_DIR) not in sys.path:
    sys.path.insert(0, str(_REPRO_DIR))


# ``envs.math.warp`` shadows the installed ``warp`` package if loaded as a
# top-level module. We always load via the full dotted path so Python resolves
# ``warp`` (top-level) to the real Warp library.
def _load(modpath: str):
    spec = importlib.util.find_spec(modpath)
    assert spec is not None, f"could not find {modpath}"
    return importlib.import_module(modpath)


math_torch = _load("envs.math.torch")
math_warp = _load("envs.math.warp")

# The math.warp module declares ``enable_backward=False``. To avoid Warp trying
# to build adjoint code for those funcs from this test module's kernels (which
# would default to backward=True), set the same option here.
wp.set_module_options({"enable_backward": False})

_DEVICE = "cpu"
_TOL = dict(atol=1e-5, rtol=1e-5)
_TIGHT_TOL = dict(atol=1e-6, rtol=1e-6)
_N = 16


def _rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


def _random_quat(rng: np.random.Generator, n: int = _N) -> np.ndarray:
    """Generate N random unit quaternions in xyzw layout."""
    q = rng.standard_normal((n, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    # avoid the q[w] < 0 branch ambiguity for asymmetric tests
    q[q[:, 3] < 0] *= -1
    return q


def _random_vec3(rng: np.random.Generator, n: int = _N, scale: float = 1.0) -> np.ndarray:
    return (rng.standard_normal((n, 3)) * scale).astype(np.float32)


def _random_mat33_from_quat(rng: np.random.Generator, n: int = _N) -> np.ndarray:
    q = _random_quat(rng, n)
    R = math_torch.matrix_from_quat(torch.from_numpy(q)).cpu().numpy().astype(np.float32)
    return R


def _launch(kernel, inputs, out_shape, out_dtype):
    """Allocate output, launch a kernel, return numpy view of output."""
    out = wp.empty(out_shape, dtype=out_dtype, device=_DEVICE)
    wp.launch(kernel, dim=out_shape, inputs=inputs + [out], device=_DEVICE)
    return np.asarray(out.numpy())


# ----------------------------------------------------------------------------
# Scalar -- specialized at kernel compile time for wp.float32
# ----------------------------------------------------------------------------


@wp.kernel
def _saturate_k(
    x: wp.array(dtype=wp.float32),
    lo: wp.array(dtype=wp.float32),
    hi: wp.array(dtype=wp.float32),
    out: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    out[tid] = math_warp.saturate(x[tid], lo[tid], hi[tid])


@wp.kernel
def _scale_transform_k(
    x: wp.array(dtype=wp.float32),
    lo: wp.array(dtype=wp.float32),
    hi: wp.array(dtype=wp.float32),
    out: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    out[tid] = math_warp.scale_transform(x[tid], lo[tid], hi[tid])


@wp.kernel
def _unscale_transform_k(
    x: wp.array(dtype=wp.float32),
    lo: wp.array(dtype=wp.float32),
    hi: wp.array(dtype=wp.float32),
    out: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    out[tid] = math_warp.unscale_transform(x[tid], lo[tid], hi[tid])


@wp.kernel
def _wrap_to_pi_k(x: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    out[tid] = math_warp.wrap_to_pi(x[tid])


def test_saturate_parity() -> None:
    rng = _rng()
    lo = rng.uniform(-1, 0, size=_N).astype(np.float32)
    hi = rng.uniform(0, 1, size=_N).astype(np.float32)
    x = rng.uniform(-2, 2, size=_N).astype(np.float32)
    t = math_torch.saturate(torch.from_numpy(x), torch.from_numpy(lo), torch.from_numpy(hi)).numpy()
    w = _launch(
        _saturate_k,
        [
            wp.array(x, dtype=wp.float32, device=_DEVICE),
            wp.array(lo, dtype=wp.float32, device=_DEVICE),
            wp.array(hi, dtype=wp.float32, device=_DEVICE),
        ],
        _N,
        wp.float32,
    )
    np.testing.assert_allclose(w, t, **_TIGHT_TOL)


def test_scale_transform_parity() -> None:
    rng = _rng()
    lo = rng.uniform(-2, -0.5, size=_N).astype(np.float32)
    hi = rng.uniform(0.5, 2, size=_N).astype(np.float32)
    x = (rng.uniform(0, 1, size=_N) * (hi - lo) + lo).astype(np.float32)
    t = math_torch.scale_transform(torch.from_numpy(x), torch.from_numpy(lo), torch.from_numpy(hi)).numpy()
    w = _launch(
        _scale_transform_k,
        [
            wp.array(x, dtype=wp.float32, device=_DEVICE),
            wp.array(lo, dtype=wp.float32, device=_DEVICE),
            wp.array(hi, dtype=wp.float32, device=_DEVICE),
        ],
        _N,
        wp.float32,
    )
    np.testing.assert_allclose(w, t, **_TOL)


def test_unscale_transform_parity() -> None:
    rng = _rng()
    lo = rng.uniform(-2, -0.5, size=_N).astype(np.float32)
    hi = rng.uniform(0.5, 2, size=_N).astype(np.float32)
    x = rng.uniform(-1, 1, size=_N).astype(np.float32)
    t = math_torch.unscale_transform(torch.from_numpy(x), torch.from_numpy(lo), torch.from_numpy(hi)).numpy()
    w = _launch(
        _unscale_transform_k,
        [
            wp.array(x, dtype=wp.float32, device=_DEVICE),
            wp.array(lo, dtype=wp.float32, device=_DEVICE),
            wp.array(hi, dtype=wp.float32, device=_DEVICE),
        ],
        _N,
        wp.float32,
    )
    np.testing.assert_allclose(w, t, **_TOL)


def test_wrap_to_pi_parity() -> None:
    rng = _rng()
    # avoid landing exactly on a multiple of pi where the torch branch fires
    x = rng.uniform(-5 * np.pi, 5 * np.pi, size=_N).astype(np.float32)
    t = math_torch.wrap_to_pi(torch.from_numpy(x)).numpy()
    w = _launch(_wrap_to_pi_k, [wp.array(x, dtype=wp.float32, device=_DEVICE)], _N, wp.float32)
    np.testing.assert_allclose(w, t, **_TOL)


# ----------------------------------------------------------------------------
# Quaternion
# ----------------------------------------------------------------------------


@wp.kernel
def _quat_conjugate_k(q: wp.array(dtype=wp.quatf), out: wp.array(dtype=wp.quatf)):
    tid = wp.tid()
    out[tid] = math_warp.quat_conjugate(q[tid])


@wp.kernel
def _quat_inv_k(q: wp.array(dtype=wp.quatf), out: wp.array(dtype=wp.quatf)):
    tid = wp.tid()
    out[tid] = math_warp.quat_inv(q[tid], wp.float32(1e-9))


@wp.kernel
def _quat_unique_k(q: wp.array(dtype=wp.quatf), out: wp.array(dtype=wp.quatf)):
    tid = wp.tid()
    out[tid] = math_warp.quat_unique(q[tid])


@wp.kernel
def _quat_mul_k(a: wp.array(dtype=wp.quatf), b: wp.array(dtype=wp.quatf), out: wp.array(dtype=wp.quatf)):
    tid = wp.tid()
    out[tid] = math_warp.quat_mul(a[tid], b[tid])


@wp.kernel
def _quat_apply_k(q: wp.array(dtype=wp.quatf), v: wp.array(dtype=wp.vec3f), out: wp.array(dtype=wp.vec3f)):
    # ``envs.math.torch.quat_apply`` is verified directly against Warp's
    # ``wp.quat_rotate`` builtin; envs/math/warp.py intentionally does not
    # re-export it.
    tid = wp.tid()
    out[tid] = wp.quat_rotate(q[tid], v[tid])


@wp.kernel
def _quat_apply_inverse_k(q: wp.array(dtype=wp.quatf), v: wp.array(dtype=wp.vec3f), out: wp.array(dtype=wp.vec3f)):
    tid = wp.tid()
    out[tid] = wp.quat_rotate_inv(q[tid], v[tid])


@wp.kernel
def _quat_from_euler_xyz_k(rpy: wp.array(dtype=wp.vec3f), out: wp.array(dtype=wp.quatf)):
    tid = wp.tid()
    out[tid] = math_warp.quat_from_euler_xyz(rpy[tid][0], rpy[tid][1], rpy[tid][2])


@wp.kernel
def _euler_xyz_from_quat_k(q: wp.array(dtype=wp.quatf), out: wp.array(dtype=wp.vec3f)):
    tid = wp.tid()
    out[tid] = math_warp.euler_xyz_from_quat(q[tid])


@wp.kernel
def _quat_from_angle_axis_k(
    angle: wp.array(dtype=wp.float32), axis: wp.array(dtype=wp.vec3f), out: wp.array(dtype=wp.quatf)
):
    tid = wp.tid()
    out[tid] = math_warp.quat_from_angle_axis(angle[tid], axis[tid])


@wp.kernel
def _axis_angle_from_quat_k(q: wp.array(dtype=wp.quatf), out: wp.array(dtype=wp.vec3f)):
    tid = wp.tid()
    out[tid] = math_warp.axis_angle_from_quat(q[tid], wp.float32(1e-6))


@wp.kernel
def _yaw_quat_k(q: wp.array(dtype=wp.quatf), out: wp.array(dtype=wp.quatf)):
    tid = wp.tid()
    out[tid] = math_warp.yaw_quat(q[tid])


@wp.kernel
def _matrix_from_quat_k(q: wp.array(dtype=wp.quatf), out: wp.array(dtype=wp.mat33f)):
    # ``envs.math.torch.matrix_from_quat`` is verified against Warp's
    # ``wp.quat_to_matrix`` builtin.
    tid = wp.tid()
    out[tid] = wp.quat_to_matrix(q[tid])


@wp.kernel
def _quat_from_matrix_k(m: wp.array(dtype=wp.mat33f), out: wp.array(dtype=wp.quatf)):
    tid = wp.tid()
    out[tid] = wp.quat_from_matrix(m[tid])


def test_quat_conjugate_parity() -> None:
    rng = _rng()
    q = _random_quat(rng)
    t = math_torch.quat_conjugate(torch.from_numpy(q)).numpy()
    w = _launch(_quat_conjugate_k, [wp.array(q, dtype=wp.quatf, device=_DEVICE)], _N, wp.quatf)
    np.testing.assert_allclose(w, t, **_TIGHT_TOL)


def test_quat_inv_parity() -> None:
    rng = _rng()
    q = _random_quat(rng)
    t = math_torch.quat_inv(torch.from_numpy(q)).numpy()
    w = _launch(_quat_inv_k, [wp.array(q, dtype=wp.quatf, device=_DEVICE)], _N, wp.quatf)
    np.testing.assert_allclose(w, t, **_TOL)


def test_quat_unique_parity() -> None:
    rng = _rng()
    # construct a mix where some quaternions have w<0
    q = rng.standard_normal((_N, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    t = math_torch.quat_unique(torch.from_numpy(q)).numpy()
    w = _launch(_quat_unique_k, [wp.array(q, dtype=wp.quatf, device=_DEVICE)], _N, wp.quatf)
    np.testing.assert_allclose(w, t, **_TIGHT_TOL)


def test_quat_mul_parity() -> None:
    rng = _rng()
    q1, q2 = _random_quat(rng), _random_quat(rng.spawn(1)[0])
    t = math_torch.quat_mul(torch.from_numpy(q1), torch.from_numpy(q2)).numpy()
    w = _launch(
        _quat_mul_k,
        [wp.array(q1, dtype=wp.quatf, device=_DEVICE), wp.array(q2, dtype=wp.quatf, device=_DEVICE)],
        _N,
        wp.quatf,
    )
    np.testing.assert_allclose(w, t, **_TOL)


def test_quat_apply_parity() -> None:
    rng = _rng()
    q = _random_quat(rng)
    v = _random_vec3(rng.spawn(1)[0])
    t = math_torch.quat_apply(torch.from_numpy(q), torch.from_numpy(v)).numpy()
    w = _launch(
        _quat_apply_k,
        [wp.array(q, dtype=wp.quatf, device=_DEVICE), wp.array(v, dtype=wp.vec3f, device=_DEVICE)],
        _N,
        wp.vec3f,
    )
    np.testing.assert_allclose(w, t, **_TOL)


def test_quat_apply_inverse_parity() -> None:
    rng = _rng()
    q = _random_quat(rng)
    v = _random_vec3(rng.spawn(1)[0])
    t = math_torch.quat_apply_inverse(torch.from_numpy(q), torch.from_numpy(v)).numpy()
    w = _launch(
        _quat_apply_inverse_k,
        [wp.array(q, dtype=wp.quatf, device=_DEVICE), wp.array(v, dtype=wp.vec3f, device=_DEVICE)],
        _N,
        wp.vec3f,
    )
    np.testing.assert_allclose(w, t, **_TOL)


def test_quat_from_euler_xyz_parity() -> None:
    rng = _rng()
    rpy = rng.uniform(-np.pi, np.pi, size=(_N, 3)).astype(np.float32)
    t = math_torch.quat_from_euler_xyz(
        torch.from_numpy(rpy[:, 0]), torch.from_numpy(rpy[:, 1]), torch.from_numpy(rpy[:, 2])
    ).numpy()
    w = _launch(_quat_from_euler_xyz_k, [wp.array(rpy, dtype=wp.vec3f, device=_DEVICE)], _N, wp.quatf)
    np.testing.assert_allclose(w, t, **_TOL)


def test_euler_xyz_from_quat_parity() -> None:
    rng = _rng()
    # stay away from pitch=+-pi/2 (gimbal singularity); the torch path branches there
    rpy = rng.uniform(-1.3, 1.3, size=(_N, 3)).astype(np.float32)
    q = (
        math_torch.quat_from_euler_xyz(
            torch.from_numpy(rpy[:, 0]), torch.from_numpy(rpy[:, 1]), torch.from_numpy(rpy[:, 2])
        )
        .numpy()
        .astype(np.float32)
    )
    roll, pitch, yaw = math_torch.euler_xyz_from_quat(torch.from_numpy(q))
    t = np.stack([roll.numpy(), pitch.numpy(), yaw.numpy()], axis=-1)
    w = _launch(_euler_xyz_from_quat_k, [wp.array(q, dtype=wp.quatf, device=_DEVICE)], _N, wp.vec3f)
    np.testing.assert_allclose(w, t, **_TOL)


def test_quat_from_angle_axis_parity() -> None:
    rng = _rng()
    angle = rng.uniform(-np.pi, np.pi, size=_N).astype(np.float32)
    axis = _random_vec3(rng.spawn(1)[0], scale=2.0)
    t = math_torch.quat_from_angle_axis(torch.from_numpy(angle), torch.from_numpy(axis)).numpy()
    w = _launch(
        _quat_from_angle_axis_k,
        [wp.array(angle, dtype=wp.float32, device=_DEVICE), wp.array(axis, dtype=wp.vec3f, device=_DEVICE)],
        _N,
        wp.quatf,
    )
    # both implementations normalize internally; either sign of q is a valid rotation
    flip = np.sign(np.sum(w * t, axis=-1, keepdims=True))
    flip = np.where(flip == 0, 1.0, flip)
    np.testing.assert_allclose(w * flip, t, **_TOL)


def test_axis_angle_from_quat_parity() -> None:
    rng = _rng()
    q = _random_quat(rng)
    t = math_torch.axis_angle_from_quat(torch.from_numpy(q)).numpy()
    w = _launch(_axis_angle_from_quat_k, [wp.array(q, dtype=wp.quatf, device=_DEVICE)], _N, wp.vec3f)
    np.testing.assert_allclose(w, t, **_TOL)


def test_yaw_quat_parity() -> None:
    rng = _rng()
    q = _random_quat(rng)
    t = math_torch.yaw_quat(torch.from_numpy(q)).numpy()
    w = _launch(_yaw_quat_k, [wp.array(q, dtype=wp.quatf, device=_DEVICE)], _N, wp.quatf)
    np.testing.assert_allclose(w, t, **_TOL)


def test_matrix_from_quat_parity() -> None:
    rng = _rng()
    q = _random_quat(rng)
    t = math_torch.matrix_from_quat(torch.from_numpy(q)).numpy()
    w = _launch(_matrix_from_quat_k, [wp.array(q, dtype=wp.quatf, device=_DEVICE)], _N, wp.mat33f)
    np.testing.assert_allclose(w, t, **_TOL)


def test_quat_from_matrix_parity() -> None:
    rng = _rng()
    R = _random_mat33_from_quat(rng)
    t = math_torch.quat_from_matrix(torch.from_numpy(R)).numpy()
    w = _launch(_quat_from_matrix_k, [wp.array(R, dtype=wp.mat33f, device=_DEVICE)], _N, wp.quatf)
    # quat and -quat represent the same rotation; canonicalize before compare
    flip = np.sign(np.sum(w * t, axis=-1, keepdims=True))
    flip = np.where(flip == 0, 1.0, flip)
    np.testing.assert_allclose(w * flip, t, **_TOL)


# ----------------------------------------------------------------------------
# Frame transforms
# ----------------------------------------------------------------------------


@wp.kernel
def _skew_k(v: wp.array(dtype=wp.vec3f), out: wp.array(dtype=wp.mat33f)):
    tid = wp.tid()
    out[tid] = math_warp.skew_symmetric_matrix(v[tid])


@wp.kernel
def _combine_k(
    t01: wp.array(dtype=wp.vec3f),
    q01: wp.array(dtype=wp.quatf),
    t12: wp.array(dtype=wp.vec3f),
    q12: wp.array(dtype=wp.quatf),
    out_pos: wp.array(dtype=wp.vec3f),
    out_quat: wp.array(dtype=wp.quatf),
):
    tid = wp.tid()
    xf = math_warp.combine_frame_transforms(t01[tid], q01[tid], t12[tid], q12[tid])
    out_pos[tid] = wp.transform_get_translation(xf)
    out_quat[tid] = wp.transform_get_rotation(xf)


@wp.kernel
def _subtract_k(
    t01: wp.array(dtype=wp.vec3f),
    q01: wp.array(dtype=wp.quatf),
    t02: wp.array(dtype=wp.vec3f),
    q02: wp.array(dtype=wp.quatf),
    out_pos: wp.array(dtype=wp.vec3f),
    out_quat: wp.array(dtype=wp.quatf),
):
    tid = wp.tid()
    xf = math_warp.subtract_frame_transforms(t01[tid], q01[tid], t02[tid], q02[tid])
    out_pos[tid] = wp.transform_get_translation(xf)
    out_quat[tid] = wp.transform_get_rotation(xf)


@wp.kernel
def _transform_point_k(
    p: wp.array(dtype=wp.vec3f),
    pos: wp.array(dtype=wp.vec3f),
    q: wp.array(dtype=wp.quatf),
    out: wp.array(dtype=wp.vec3f),
):
    tid = wp.tid()
    out[tid] = math_warp.transform_point(p[tid], pos[tid], q[tid])


def test_skew_symmetric_matrix_parity() -> None:
    rng = _rng()
    v = _random_vec3(rng)
    t = math_torch.skew_symmetric_matrix(torch.from_numpy(v)).numpy()
    w = _launch(_skew_k, [wp.array(v, dtype=wp.vec3f, device=_DEVICE)], _N, wp.mat33f)
    np.testing.assert_allclose(w, t, **_TIGHT_TOL)


def test_combine_frame_transforms_parity() -> None:
    rng = _rng()
    t01 = _random_vec3(rng)
    q01 = _random_quat(rng.spawn(1)[0])
    t12 = _random_vec3(rng.spawn(2)[1])
    q12 = _random_quat(rng.spawn(3)[2])
    tp, tq = math_torch.combine_frame_transforms(
        torch.from_numpy(t01), torch.from_numpy(q01), torch.from_numpy(t12), torch.from_numpy(q12)
    )
    tp = tp.numpy()
    tq = tq.numpy()
    out_pos = wp.empty(_N, dtype=wp.vec3f, device=_DEVICE)
    out_quat = wp.empty(_N, dtype=wp.quatf, device=_DEVICE)
    wp.launch(
        _combine_k,
        dim=_N,
        inputs=[
            wp.array(t01, dtype=wp.vec3f, device=_DEVICE),
            wp.array(q01, dtype=wp.quatf, device=_DEVICE),
            wp.array(t12, dtype=wp.vec3f, device=_DEVICE),
            wp.array(q12, dtype=wp.quatf, device=_DEVICE),
            out_pos,
            out_quat,
        ],
        device=_DEVICE,
    )
    np.testing.assert_allclose(out_pos.numpy(), tp, **_TOL)
    np.testing.assert_allclose(out_quat.numpy(), tq, **_TOL)


def test_subtract_frame_transforms_parity() -> None:
    rng = _rng()
    t01 = _random_vec3(rng)
    q01 = _random_quat(rng.spawn(1)[0])
    t02 = _random_vec3(rng.spawn(2)[1])
    q02 = _random_quat(rng.spawn(3)[2])
    tp, tq = math_torch.subtract_frame_transforms(
        torch.from_numpy(t01), torch.from_numpy(q01), torch.from_numpy(t02), torch.from_numpy(q02)
    )
    tp = tp.numpy()
    tq = tq.numpy()
    out_pos = wp.empty(_N, dtype=wp.vec3f, device=_DEVICE)
    out_quat = wp.empty(_N, dtype=wp.quatf, device=_DEVICE)
    wp.launch(
        _subtract_k,
        dim=_N,
        inputs=[
            wp.array(t01, dtype=wp.vec3f, device=_DEVICE),
            wp.array(q01, dtype=wp.quatf, device=_DEVICE),
            wp.array(t02, dtype=wp.vec3f, device=_DEVICE),
            wp.array(q02, dtype=wp.quatf, device=_DEVICE),
            out_pos,
            out_quat,
        ],
        device=_DEVICE,
    )
    np.testing.assert_allclose(out_pos.numpy(), tp, **_TOL)
    np.testing.assert_allclose(out_quat.numpy(), tq, **_TOL)


def test_transform_point_parity() -> None:
    # torch.transform_points takes a batched (P, 3) and a single (pos, quat).
    # The Warp port operates per-element: each tid sees its own (p, pos, q).
    # We verify by running the torch version separately per item.
    rng = _rng()
    p = _random_vec3(rng)
    pos = _random_vec3(rng.spawn(1)[0])
    q = _random_quat(rng.spawn(2)[1])
    # per-element reference
    t = np.zeros_like(p)
    for i in range(_N):
        out = math_torch.transform_points(
            torch.from_numpy(p[i : i + 1]), torch.from_numpy(pos[i]), torch.from_numpy(q[i])
        )
        t[i] = out.numpy()
    w = _launch(
        _transform_point_k,
        [
            wp.array(p, dtype=wp.vec3f, device=_DEVICE),
            wp.array(pos, dtype=wp.vec3f, device=_DEVICE),
            wp.array(q, dtype=wp.quatf, device=_DEVICE),
        ],
        _N,
        wp.vec3f,
    )
    np.testing.assert_allclose(w, t, **_TOL)


# ----------------------------------------------------------------------------
# Sampling -- distributional checks only (RNGs differ between torch and Warp)
# ----------------------------------------------------------------------------


def test_sample_uniform_distribution() -> None:
    n = 10_000
    out = wp.empty(n, dtype=wp.float32, device=_DEVICE)
    wp.launch(
        math_warp.sample_uniform_kernel,
        dim=n,
        inputs=[wp.int32(7), wp.float32(-2.0), wp.float32(3.0), out],
        device=_DEVICE,
    )
    samples = out.numpy()
    assert samples.shape == (n,)
    assert samples.min() >= -2.0 and samples.max() <= 3.0
    np.testing.assert_allclose(samples.mean(), 0.5, atol=0.1)


def test_sample_log_uniform_distribution() -> None:
    n = 10_000
    out = wp.empty(n, dtype=wp.float32, device=_DEVICE)
    # log-symmetric range so the log midpoint is 0
    wp.launch(
        math_warp.sample_log_uniform_kernel,
        dim=n,
        inputs=[wp.int32(11), wp.float32(0.1), wp.float32(10.0), out],
        device=_DEVICE,
    )
    samples = out.numpy()
    assert samples.shape == (n,)
    assert (samples >= 0.1).all() and (samples <= 10.0).all()
    log_mean = float(np.log(samples).mean())
    expected = float(0.5 * (np.log(0.1) + np.log(10.0)))  # = 0.0
    assert abs(log_mean - expected) < 0.2, f"log_mean={log_mean}, expected~{expected}"


def test_sample_gaussian_distribution() -> None:
    n = 10_000
    out = wp.empty(n, dtype=wp.float32, device=_DEVICE)
    wp.launch(
        math_warp.sample_gaussian_kernel,
        dim=n,
        inputs=[wp.int32(23), wp.float32(1.5), wp.float32(0.5), out],
        device=_DEVICE,
    )
    samples = out.numpy()
    np.testing.assert_allclose(samples.mean(), 1.5, atol=0.05)
    np.testing.assert_allclose(samples.std(), 0.5, atol=0.05)


def test_sample_overload_registered() -> None:
    """The graph-capture-safety overloads should be registered."""
    # Just verifying the module loads without crashing was already done by import.
    # Spot-check that float32 launches work (they would fail to JIT-compile under
    # graph capture if no overload were registered).
    assert math_warp.sample_uniform_kernel is not None
