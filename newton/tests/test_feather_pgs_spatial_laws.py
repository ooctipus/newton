# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The spatial kit: one source for Warp and CUDA, and each law against its float64 definition.

Each law's two renderings agree bit for bit (up to the sign of a zero). Each
law matches its definition within a stated bound, in float32 units of the
magnitude of its terms (``term_ulps``); the joint transform matches the
Featherstone solver's, the same law, bit for bit.
"""

import unittest

import numpy as np
import warp as wp

from newton._src.sim import JointType
from newton._src.solvers.feather_pgs import law_source, spatial_laws
from newton._src.solvers.feather_pgs.device_preludes import dual_native
from newton._src.solvers.featherstone import kernels as featherstone
from newton.tests.feather_pgs_law_harness import exec_kernel, mismatches, term_ulps
from newton.tests.unittest_utils import add_function_test, get_test_devices

N = 4096
EPS = float(np.finfo(np.float32).eps)


# ---------------------------------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------------------------------


def _quaternions(rng, n):
    q = rng.standard_normal((n, 4))
    return (q / np.linalg.norm(q, axis=1, keepdims=True)).astype(np.float32)


def _inertias(rng, n):
    a = rng.standard_normal((n, 3, 3))
    return (np.einsum("nij,nkj->nik", a, a) * 0.05 + np.eye(3) * 0.01).astype(np.float32)


def _value(annotation, rng, n):
    if annotation in (float, wp.float32):
        return rng.uniform(0.2, 3.0, n).astype(np.float32)
    if annotation is wp.vec3:
        return rng.standard_normal((n, 3)).astype(np.float32)
    if annotation is wp.quat:
        return _quaternions(rng, n)
    if annotation is wp.transform:
        return np.concatenate([rng.standard_normal((n, 3)).astype(np.float32), _quaternions(rng, n)], axis=1)
    if annotation is wp.spatial_vector:
        return rng.standard_normal((n, 6)).astype(np.float32)
    if annotation is wp.mat33:
        return _inertias(rng, n)
    if annotation is bool:
        return rng.integers(0, 2, n).astype(np.int32)
    raise TypeError(annotation)


def _array(values, annotation, device):
    dtype = wp.int32 if annotation is bool else annotation
    return wp.array(values, dtype=dtype, device=device)


def _results(law):
    sig = law_source.law_signature(law)
    return [wp.int32 if r is bool else r for r in sig.results]


# ---------------------------------------------------------------------------------------------------------------------
# Twins: each law's Warp function and CUDA rendering
# ---------------------------------------------------------------------------------------------------------------------


def _twin_kernels(law):
    sig = law_source.law_signature(law)
    namespace = {"wp": wp, "law": law}
    inputs, warp_args, native_args = [], [], []
    for k, (name, annotation) in enumerate(sig.parameters.items()):
        namespace[f"T_{k}"] = wp.int32 if annotation is bool else annotation
        inputs.append(f"{name}: wp.array[T_{k}]")
        warp_args.append(f"{name}[i] != 0" if annotation is bool else f"{name}[i]")
        native_args.append(f"wp::index({name}, i) != 0" if annotation is bool else f"wp::index({name}, i)")
    outputs = []
    for j, result in enumerate(_results(law)):
        namespace[f"R_{j}"] = result
        outputs.append(f"out_{j}: wp.array[R_{j}]")
    signature = ", ".join(inputs + outputs)
    names = [f"r_{j}" for j in range(len(outputs))]
    stores = "\n".join(f"    out_{j}[i] = r_{j}" for j in range(len(outputs)))
    warp_source = f"def warp_twin({signature}):\n    i = wp.tid()\n    {', '.join(names)} = law({', '.join(warp_args)})\n{stores}\n"
    if len(outputs) == 1:
        native_stores = "wp::index(out_0, i) = result;"
    else:
        native_stores = "".join(f"wp::index(out_{j}, i) = result.v{j};" for j in range(len(outputs)))
    body = f"    const auto result = Spatial::{sig.name}({', '.join(native_args)});\n    {native_stores}\n"
    namespace["snippet"] = dual_native(body, preludes=spatial_laws.SPATIAL.preludes())
    arguments = ", ".join(part.split(":")[0] for part in inputs + outputs)
    native_source = (
        f"@wp.func_native(snippet)\ndef native(i: int, {signature}): ...\n\n"
        f"def native_twin({signature}):\n    native(wp.tid(), {arguments})\n"
    )
    return exec_kernel("warp_twin", warp_source, dict(namespace)), exec_kernel(
        "native_twin", native_source, dict(namespace)
    )


CUDA_LAWS = [
    getattr(spatial_laws, name)
    for name in spatial_laws.SPATIAL.laws
    if law_source.law_signature(getattr(spatial_laws, name)).cuda
]


def test_twins_agree(test, device):
    """Each law's Warp function and CUDA rendering return the same bits (a zero's sign aside)."""
    rng = np.random.default_rng(11)
    for law in CUDA_LAWS:
        sig = law_source.law_signature(law)
        with test.subTest(law=sig.name):
            warp_kernel, native_kernel = _twin_kernels(law)
            values = [_array(_value(a, rng, N), a, device) for a in sig.parameters.values()]
            outputs = []
            for kernel in (warp_kernel, native_kernel):
                outs = [wp.zeros(N, dtype=r, device=device) for r in _results(law)]
                wp.launch(kernel, N, inputs=values + outs, device=device)
                outputs.append([o.numpy() for o in outs])
            for a, b in zip(*outputs, strict=True):
                test.assertEqual(int(mismatches(a, b, signed_zeros=False).sum()), 0)


class TestSpatialLawSource(unittest.TestCase):
    def test_every_law_but_the_matrix_pack_and_joint_laws_renders(self):
        """Native code reaches every law but the three whose consumers are Warp only."""
        names = {law_source.law_signature(law).name for law in CUDA_LAWS}
        self.assertEqual(set(spatial_laws.SPATIAL.laws) - names, {"inertia_matrix", "joint_transform", "joint_motion"})
        source = spatial_laws.SPATIAL.prelude().source
        self.assertIn("return wp::transform_multiply(parent, child);", source)


# ---------------------------------------------------------------------------------------------------------------------
# References: each law against its float64 definition
# ---------------------------------------------------------------------------------------------------------------------


def _quat_matrix(q):
    """Rotation matrices of unit quaternions ``(x, y, z, w)`` (float64)."""
    x, y, z, w = (q[:, k].astype(np.float64) for k in range(4))
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], axis=-1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], axis=-1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=-1),
        ],
        axis=1,
    )


def _quat_multiply(a, b):
    x1, y1, z1, w1 = (a[:, k].astype(np.float64) for k in range(4))
    x2, y2, z2, w2 = (b[:, k].astype(np.float64) for k in range(4))
    return np.stack(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        axis=1,
    )


def _compose(a, b):
    """Poses ``a * b`` (float64), each ``(p, q)``."""
    p = a[:, :3] + np.einsum("nij,nj->ni", _quat_matrix(a[:, 3:]), b[:, :3])
    return np.concatenate([p, _quat_multiply(a[:, 3:], b[:, 3:])], axis=1)


def _inverse(a):
    q = a[:, 3:].astype(np.float64) * np.array([-1.0, -1.0, -1.0, 1.0])
    return np.concatenate([-np.einsum("nij,nj->ni", _quat_matrix(q), a[:, :3]), q], axis=1)


def _skew(v):
    zero = np.zeros(v.shape[0])
    return np.stack(
        [
            np.stack([zero, -v[:, 2], v[:, 1]], axis=-1),
            np.stack([v[:, 2], zero, -v[:, 0]], axis=-1),
            np.stack([-v[:, 1], v[:, 0], zero], axis=-1),
        ],
        axis=1,
    )


def _kernel(name, law, arguments, results, device):
    """Launch one law over arrays; returns its results as numpy."""
    namespace = {"wp": wp, "law": law}
    inputs, args = [], []
    for k, values in enumerate(arguments):
        namespace[f"T_{k}"] = values.dtype
        inputs.append(f"a_{k}: wp.array[T_{k}]")
        args.append(f"a_{k}[i]")
    outputs = []
    for j, dtype in enumerate(results):
        namespace[f"R_{j}"] = dtype
        outputs.append(f"r_{j}: wp.array[R_{j}]")
    names = [f"v_{j}" for j in range(len(results))]
    stores = "\n".join(f"    r_{j}[i] = v_{j}" for j in range(len(results)))
    source = (
        f"def {name}({', '.join(inputs + outputs)}):\n    i = wp.tid()\n"
        f"    {', '.join(names)} = law({', '.join(args)})\n{stores}\n"
    )
    kernel = exec_kernel(name, source, namespace)
    outs = [wp.zeros(arguments[0].shape[0], dtype=r, device=device) for r in results]
    wp.launch(kernel, arguments[0].shape[0], inputs=list(arguments) + outs, device=device)
    return [o.numpy() for o in outs]


def _close(test, got, expected, magnitude, bound, tag):
    gap = float(term_ulps(got, expected, magnitude).max())
    test.assertLess(gap, bound, tag)


def test_compose_laws(test, device):
    """The pose and twist combines and the joint frame pose against their float64 definitions."""
    rng = np.random.default_rng(1)
    poses = [_value(wp.transform, rng, N) for _ in range(3)]
    a, b, c = (wp.array(x, dtype=wp.transform, device=device) for x in poses)
    (got,) = _kernel("compose", spatial_laws.transform_compose, [a, b], [wp.transform], device)
    _close(test, got, _compose(poses[0], poses[1]), 4.0, 8.0, "transform_compose")
    (got,) = _kernel("frame_pose", spatial_laws.joint_frame_pose, [a, b, c], [wp.transform], device)
    expected = _compose(_compose(poses[0], poses[1]), _inverse(poses[2]))
    _close(test, got, expected, 8.0, 16.0, "joint_frame_pose")
    twists = [rng.standard_normal((N, 6)).astype(np.float32) for _ in range(4)]
    arrays = [wp.array(x, dtype=wp.spatial_vector, device=device) for x in twists]
    velocity, bias = _kernel(
        "twist_compose", spatial_laws.twist_compose, arrays, [wp.spatial_vector, wp.spatial_vector], device
    )
    pv, pa, v, acc = (x.astype(np.float64) for x in twists)
    linear = np.cross(pv[:, 3:], v[:, :3]) + np.cross(pv[:, :3], v[:, 3:])
    angular = np.cross(pv[:, 3:], v[:, 3:])
    _close(test, velocity, pv + v, np.abs(pv) + np.abs(v), 2.0, "twist_compose velocity")
    expected = pa + acc + np.concatenate([linear, angular], axis=1)
    _close(test, bias, expected, 8.0, 16.0, "twist_compose bias")


def test_twist_and_wrench_laws(test, device):
    """Frame changes, point velocities, wrench shifts and the transport."""
    rng = np.random.default_rng(2)
    frame = _value(wp.transform, rng, N)
    twist = rng.standard_normal((N, 6)).astype(np.float32)
    wrench = rng.standard_normal((N, 6)).astype(np.float32)
    offset = rng.standard_normal((N, 3)).astype(np.float32)
    f = wp.array(frame, dtype=wp.transform, device=device)
    t = wp.array(twist, dtype=wp.spatial_vector, device=device)
    w = wp.array(wrench, dtype=wp.spatial_vector, device=device)
    o = wp.array(offset, dtype=wp.vec3, device=device)
    rotation = _quat_matrix(frame[:, 3:])
    angular = np.einsum("nij,nj->ni", rotation, twist[:, 3:])
    linear = np.einsum("nij,nj->ni", rotation, twist[:, :3]) + np.cross(frame[:, :3], angular)
    (got,) = _kernel("twist_transform", spatial_laws.twist_transform, [f, t], [wp.spatial_vector], device)
    _close(test, got, np.concatenate([linear, angular], axis=1), 8.0, 16.0, "twist_transform")
    (got,) = _kernel("point_velocity", spatial_laws.point_velocity, [t, o], [wp.vec3], device)
    _close(test, got, twist[:, :3] + np.cross(twist[:, 3:], offset), 4.0, 4.0, "point_velocity")
    (got,) = _kernel("wrench_shift", spatial_laws.wrench_shift, [w, o], [wp.spatial_vector], device)
    expected = np.concatenate([wrench[:, :3], wrench[:, 3:] + np.cross(offset, wrench[:, :3])], axis=1)
    _close(test, got, expected, 4.0, 4.0, "wrench_shift")
    v3 = wp.array(twist[:, :3], dtype=wp.vec3, device=device)
    w3 = wp.array(twist[:, 3:], dtype=wp.vec3, device=device)
    (got,) = _kernel("transport", spatial_laws.free_root_transport, [v3, w3], [wp.vec3], device)
    _close(test, got, np.cross(twist[:, 3:], twist[:, :3]), 4.0, 4.0, "free_root_transport")


def test_inertia_laws(test, device):
    """Placing, applying, packing and acting with one inertia representation against the dense 6x6 algebra."""
    rng = np.random.default_rng(4)
    mass = _value(float, rng, N)
    inertia = _inertias(rng, N)
    rotation = _quaternions(rng, N)
    offset = rng.standard_normal((N, 3)).astype(np.float32)
    twist = rng.standard_normal((N, 6)).astype(np.float32)
    m = wp.array(mass, dtype=float, device=device)
    i3 = wp.array(inertia, dtype=wp.mat33, device=device)
    q = wp.array(rotation, dtype=wp.quat, device=device)
    o = wp.array(offset, dtype=wp.vec3, device=device)
    tw = wp.array(twist, dtype=wp.spatial_vector, device=device)
    placed_mass, moment, rotational = _kernel(
        "inertia_at", spatial_laws.inertia_at, [m, i3, q, o], [float, wp.vec3, wp.mat33], device
    )
    R = _quat_matrix(rotation)
    c = offset.astype(np.float64)
    expected = np.einsum("nij,njk,nlk->nil", R, inertia, R) + mass[:, None, None] * (
        np.einsum("ni,ni->n", c, c)[:, None, None] * np.eye(3) - np.einsum("ni,nj->nij", c, c)
    )
    np.testing.assert_array_equal(placed_mass, mass)
    _close(test, moment, mass[:, None] * c, np.abs(mass[:, None] * c), 1.0, "inertia_at moment")
    _close(test, rotational, expected, np.abs(expected).max(axis=(1, 2))[:, None, None] + 1.0, 16.0, "inertia_at")
    dense = np.zeros((N, 6, 6))
    dense[:, :3, :3] = mass[:, None, None] * np.eye(3)
    dense[:, :3, 3:] = -_skew(mass[:, None] * c)
    dense[:, 3:, :3] = _skew(mass[:, None] * c)
    dense[:, 3:, 3:] = rotational.astype(np.float64)
    pm = wp.array(placed_mass, dtype=float, device=device)
    pv = wp.array(moment, dtype=wp.vec3, device=device)
    pr = wp.array(rotational, dtype=wp.mat33, device=device)
    (got,) = _kernel("inertia_apply", spatial_laws.inertia_apply, [pm, pv, pr, tw], [wp.spatial_vector], device)
    _close(test, got, np.einsum("nij,nj->ni", dense, twist), 8.0 * np.abs(dense).max(), 8.0, "inertia_apply")
    (got,) = _kernel("inertia_matrix", spatial_laws.inertia_matrix, [pm, pv, pr], [wp.spatial_matrix], device)
    np.testing.assert_allclose(got, dense, rtol=1e-6, atol=1e-6)
    (got,) = _kernel("inertia_action", spatial_laws.inertia_action, [q, i3, o], [wp.vec3], device)
    expected = np.einsum("nij,njk,nlk,nl->ni", R, inertia, R, c)
    _close(test, got, expected, 8.0 * np.abs(expected).max(), 8.0, "inertia_action")
    # The inverse-dynamics wrench, Newton-Euler at the center of mass, against the spatial algebra at the
    # origin: I a + v x* (I v) - (m g, c x m g), with I the float64 placed inertia.
    acceleration = rng.standard_normal((N, 6)).astype(np.float32)
    gravity = rng.standard_normal((N, 3)).astype(np.float32)
    placed = np.einsum("nij,njk,nlk->nil", R, inertia, R) + mass[:, None, None] * (
        np.einsum("ni,ni->n", c, c)[:, None, None] * np.eye(3) - np.einsum("ni,nj->nij", c, c)
    )
    spatial = np.zeros((N, 6, 6))
    spatial[:, :3, :3] = mass[:, None, None] * np.eye(3)
    spatial[:, :3, 3:] = -_skew(mass[:, None] * c)
    spatial[:, 3:, :3] = _skew(mass[:, None] * c)
    spatial[:, 3:, 3:] = placed
    momentum = np.einsum("nij,nj->ni", spatial, twist)
    gyroscopic = np.concatenate(
        [
            np.cross(twist[:, 3:], momentum[:, :3]),
            np.cross(twist[:, 3:], momentum[:, 3:]) + np.cross(twist[:, :3], momentum[:, :3]),
        ],
        axis=1,
    )
    weight = mass[:, None] * gravity
    expected = (
        np.einsum("nij,nj->ni", spatial, acceleration)
        + gyroscopic
        - np.concatenate([weight, np.cross(c, weight)], axis=1)
    )
    a = wp.array(acceleration, dtype=wp.spatial_vector, device=device)
    g = wp.array(gravity, dtype=wp.vec3, device=device)
    (got,) = _kernel("body_wrench", spatial_laws.body_wrench, [m, q, i3, o, tw, a, g], [wp.spatial_vector], device)
    # The terms' size: the inertia times the acceleration, the velocity squared and gravity.
    terms = np.abs(acceleration).max(axis=1) + (1.0 + np.abs(twist).max(axis=1)) ** 2 + np.abs(gravity).max(axis=1)
    _close(test, got, expected, np.abs(spatial).max(axis=(1, 2))[:, None] * terms[:, None], 16.0, "body_wrench")


@wp.kernel(enable_backward=False)
def _records(
    mass: wp.array[float],
    moment: wp.array[wp.vec3],
    rotational: wp.array[wp.mat33],
    records: wp.array[spatial_laws.inertia_record],
    unpacked_mass: wp.array[float],
    unpacked_moment: wp.array[wp.vec3],
    unpacked_rotational: wp.array[wp.mat33],
):
    i = wp.tid()
    records[i] = spatial_laws.pack_inertia(mass[i], moment[i], rotational[i])
    m, h, r = spatial_laws.unpack_inertia(records[i])
    unpacked_mass[i] = m
    unpacked_moment[i] = h
    unpacked_rotational[i] = r


def test_inertia_records(test, device):
    """An inertia record holds mass, first moment and rotational inertia exactly, row-major."""
    rng = np.random.default_rng(5)
    mass = _value(float, rng, N)
    moment = rng.standard_normal((N, 3)).astype(np.float32)
    rotational = _inertias(rng, N)
    inputs = [
        wp.array(mass, dtype=float, device=device),
        wp.array(moment, dtype=wp.vec3, device=device),
        wp.array(rotational, dtype=wp.mat33, device=device),
    ]
    records = wp.zeros(N, dtype=spatial_laws.inertia_record, device=device)
    outputs = [
        wp.zeros(N, dtype=float, device=device),
        wp.zeros(N, dtype=wp.vec3, device=device),
        wp.zeros(N, dtype=wp.mat33, device=device),
    ]
    wp.launch(_records, N, inputs=inputs, outputs=[records, *outputs], device=device)
    np.testing.assert_array_equal(records.numpy(), np.c_[mass, moment, rotational.reshape(N, 9)])
    for got, value in zip(outputs, (mass, moment, rotational), strict=True):
        np.testing.assert_array_equal(got.numpy(), value)


@wp.kernel(enable_backward=False)
def _joints(
    joint_type: wp.array[int],
    lin_count: wp.array[int],
    ang_count: wp.array[int],
    joint_axis: wp.array[wp.vec3],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    anchor: wp.array[wp.transform],
    law_transform: wp.array[wp.transform],
    reference_transform: wp.array[wp.transform],
    twist: wp.array[wp.spatial_vector],
    bias: wp.array[wp.spatial_vector],
    columns: wp.array[wp.spatial_vector],
):
    i = wp.tid()
    t = joint_type[i]
    law_transform[i] = spatial_laws.joint_transform(t, joint_axis, 6 * i, lin_count[i], ang_count[i], joint_q, 7 * i)
    reference_transform[i] = featherstone.jcalc_transform(
        t, joint_axis, 6 * i, lin_count[i], ang_count[i], joint_q, 7 * i
    )
    velocity, own = spatial_laws.joint_motion(
        t, joint_axis, lin_count[i], ang_count[i], anchor[i], joint_q, 7 * i, joint_qd, 6 * i, columns
    )
    twist[i] = velocity
    bias[i] = own


def _rotate(axis, angle, vector):
    """``vector`` turned by ``angle`` about the unit ``axis`` (Rodrigues, float64)."""
    return (
        vector * np.cos(angle)
        + np.cross(axis, vector) * np.sin(angle)
        + axis * np.dot(axis, vector) * (1.0 - np.cos(angle))
    )


def _motion_cross(first, second):
    """The motion cross product of two twists in Newton's layout ``(v, w)``."""
    return np.r_[np.cross(first[3:], second[:3]) + np.cross(first[:3], second[3:]), np.cross(first[3:], second[3:])]


def _d6_chain(R, p, axes, q, qd, lin, ang):
    """A D6 joint's columns, twist and own bias as the chain of its axes (float64): the linear axes slide along the
    anchor's; each angular axis, carried through the rotations before it, turns about the joint origin the linear
    coordinates moved; the bias sums the motion cross products of each link of the chain with every later one."""
    origin = p + R @ sum((axes[k] * q[k] for k in range(lin)), np.zeros(3))
    columns = [np.r_[R @ axes[k], 0.0, 0.0, 0.0] for k in range(lin)]
    turned = []
    for k in range(ang):
        axis = axes[lin + k].astype(np.float64)
        for previous, angle in turned:
            axis = _rotate(previous, angle, axis)
        turned.append((axis, q[lin + k]))
        w = R @ axis
        columns.append(np.r_[np.cross(origin, w), w])
    chain = [sum((columns[k] * qd[k] for k in range(lin)), np.zeros(6))]
    chain += [columns[lin + k] * qd[lin + k] for k in range(ang)]
    bias = sum(
        (_motion_cross(chain[j], chain[k]) for j in range(len(chain)) for k in range(j + 1, len(chain))), np.zeros(6)
    )
    return np.asarray(columns), sum(chain, np.zeros(6)), bias


def test_joint_laws(test, device):
    """Joint transforms against the Featherstone solver's (the same law), and motion columns and twists against
    their float64 definitions, for every joint type the passes see."""
    rng = np.random.default_rng(6)
    kinds = [
        (int(JointType.PRISMATIC), 1, 0),
        (int(JointType.REVOLUTE), 0, 1),
        (int(JointType.BALL), 0, 3),
        (int(JointType.FIXED), 0, 0),
        (int(JointType.FREE), 3, 3),
        (int(JointType.D6), 2, 1),
        (int(JointType.D6), 0, 2),
        (int(JointType.D6), 1, 3),
    ]
    count = 512
    pick = rng.integers(0, len(kinds), count)
    types = np.array([kinds[k][0] for k in pick], dtype=np.int32)
    lin = np.array([kinds[k][1] for k in pick], dtype=np.int32)
    ang = np.array([kinds[k][2] for k in pick], dtype=np.int32)
    axes = rng.standard_normal((count * 6, 3))
    axes = (axes / np.linalg.norm(axes, axis=1, keepdims=True)).astype(np.float32)
    # A D6's angular axes are orthonormal, as importers build them (Newton's axis transport orthonormalizes a pair).
    for i in np.flatnonzero(types == int(JointType.D6)):
        frame = np.linalg.qr(rng.standard_normal((3, 3)))[0].T.astype(np.float32)
        axes[6 * i + lin[i] : 6 * i + lin[i] + ang[i]] = frame[: ang[i]]
    q = rng.uniform(-1.0, 1.0, count * 7).astype(np.float32)
    for i in np.flatnonzero((types == int(JointType.BALL)) | (types == int(JointType.FREE))):
        first = 7 * i + (3 if types[i] == int(JointType.FREE) else 0)
        q[first : first + 4] /= np.linalg.norm(q[first : first + 4])
    qd = rng.uniform(-1.0, 1.0, count * 6).astype(np.float32)
    anchor = _value(wp.transform, rng, count)
    arrays = [
        wp.array(types, dtype=int, device=device),
        wp.array(lin, dtype=int, device=device),
        wp.array(ang, dtype=int, device=device),
        wp.array(axes, dtype=wp.vec3, device=device),
        wp.array(q, dtype=float, device=device),
        wp.array(qd, dtype=float, device=device),
        wp.array(anchor, dtype=wp.transform, device=device),
    ]
    outputs = [
        wp.zeros(count, dtype=wp.transform, device=device),
        wp.zeros(count, dtype=wp.transform, device=device),
        wp.zeros(count, dtype=wp.spatial_vector, device=device),
        wp.zeros(count, dtype=wp.spatial_vector, device=device),
        wp.zeros(count * 6, dtype=wp.spatial_vector, device=device),
    ]
    wp.launch(_joints, count, inputs=arrays, outputs=outputs, device=device)
    law, reference, twist, bias, columns = (o.numpy() for o in outputs)
    np.testing.assert_array_equal(law, reference)
    R = _quat_matrix(anchor[:, 3:])
    p = anchor[:, :3].astype(np.float64)
    for i in range(count):
        if types[i] == int(JointType.FIXED):
            np.testing.assert_array_equal(twist[i], 0.0)
            continue
        own = np.zeros(6)
        if types[i] == int(JointType.FREE):
            expected = [np.r_[R[i] @ e, 0, 0, 0] for e in np.eye(3)] + [np.r_[0, 0, 0, R[i] @ e] for e in np.eye(3)]
        elif types[i] == int(JointType.BALL):
            expected = [np.r_[np.cross(p[i], R[i] @ e), R[i] @ e] for e in np.eye(3)]
        elif types[i] == int(JointType.D6):
            expected, _, own = _d6_chain(
                R[i], p[i], axes[6 * i : 6 * i + 6], q[7 * i : 7 * i + 7], qd[6 * i : 6 * i + 6], lin[i], ang[i]
            )
        else:
            kinds_of = [False] * lin[i] + [True] * ang[i]
            expected = []
            for k, angular in enumerate(kinds_of):
                w = R[i] @ axes[6 * i + k]
                expected.append(np.r_[np.cross(p[i], w), w] if angular else np.r_[w, 0, 0, 0])
        expected = np.asarray(expected)
        got = columns[6 * i : 6 * i + len(expected)]
        scale = 1.0 + np.abs(p[i]).max() + 2.0 * float(types[i] == int(JointType.D6))
        _close(test, got, expected, 4.0 * scale, 8.0, f"joint_motion columns {types[i]}")
        _close(test, twist[i], expected.T @ qd[6 * i : 6 * i + len(expected)], 16.0 * scale, 16.0, "joint_motion twist")
        _close(test, bias[i], own, 16.0 * scale, 32.0, f"joint_motion own bias {types[i]}")


@wp.kernel(enable_backward=False)
def _forces(
    motion: wp.array[wp.spatial_vector],
    wrench: wp.array[wp.spatial_vector],
    values: wp.array2d[float],
    out: wp.array[float],
):
    i = wp.tid()
    out[i] = spatial_laws.coordinate_force(
        motion[i], wrench[i], values[i, 0], values[i, 1], values[i, 2], values[i, 3], values[i, 4], values[i, 5]
    )


def test_coordinate_force(test, device):
    """A coordinate's generalized force: control force less the projected wrench, then spring and damper."""
    rng = np.random.default_rng(7)
    motion = rng.standard_normal((N, 6)).astype(np.float32)
    wrench = rng.standard_normal((N, 6)).astype(np.float32)
    values = rng.standard_normal((N, 6)).astype(np.float32)
    out = wp.zeros(N, dtype=float, device=device)
    wp.launch(
        _forces,
        N,
        inputs=[
            wp.array(motion, dtype=wp.spatial_vector, device=device),
            wp.array(wrench, dtype=wp.spatial_vector, device=device),
            wp.array(values, dtype=float, device=device),
        ],
        outputs=[out],
        device=device,
    )
    v = values.astype(np.float64)
    expected = -np.einsum("ni,ni->n", motion, wrench) + v[:, 0] + v[:, 1] * (v[:, 2] - v[:, 4]) - v[:, 3] * v[:, 5]
    _close(test, out.numpy(), expected, 16.0, 16.0, "coordinate_force")


class TestSpatialLaws(unittest.TestCase):
    pass


for _function in (
    test_twins_agree,
    test_compose_laws,
    test_twist_and_wrench_laws,
    test_inertia_laws,
    test_inertia_records,
    test_joint_laws,
    test_coordinate_force,
):
    add_function_test(TestSpatialLaws, _function.__name__, _function, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main(verbosity=2)
