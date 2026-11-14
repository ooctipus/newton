# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The laws primitive's contract: each law written once, in Warp's language, against its definition.

Every law matches its float64 definition within a stated bound; its CUDA rendering agrees with its Warp function (bit
for bit but for the sign of a zero, or within an iterative law's tolerance); the rendering keeps the Python
statements and refuses what it cannot keep equal. Each section names the interface it reads.
"""

import os
import unittest

import numpy as np
import warp as wp

from newton._src.sim import JointType
from newton._src.solvers.feather_pgs import (
    contact_laws,
    law_source,
    projection_laws,
    row_laws,
    spatial_laws,
)
from newton._src.solvers.feather_pgs import projection_laws as laws
from newton._src.solvers.feather_pgs.device_preludes import cuda_native
from newton._src.solvers.feather_pgs.row_laws import ROW_CONTACT
from newton._src.solvers.featherstone import kernels as featherstone
from newton.tests.feather_pgs_law_harness import (
    FAMILIES,
    LAWS,
    TWIN_AGREEING,
    TWIN_SAMPLES,
    TWIN_TOLERANCES,
    call,
    exec_kernel,
    launch,
    mismatches,
    register_device_tests,
    scalars,
    term_ulps,
    twin_inputs,
    twin_kernels,
)

# ---------------------------------------------------------------------------------------------------------------------
# spatial_laws
#
# The spatial kit: one source for Warp and CUDA, and each law against its float64 definition.
#
# Each law's two renderings agree bit for bit (up to the sign of a zero). Each law matches its definition within a
# stated bound, in float32 units of the magnitude of its terms (``term_ulps``); the joint transform matches the
# Featherstone solver's, the same law, bit for bit.
# ---------------------------------------------------------------------------------------------------------------------


N = 4096


EPS = float(np.finfo(np.float32).eps)


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


CUDA_LAWS = [
    getattr(spatial_laws, name)
    for name in spatial_laws.SPATIAL.laws
    if law_source.law_signature(getattr(spatial_laws, name)).cuda
]


def test_spatial_twins_agree(test, device):
    """Each law's Warp function and CUDA rendering return the same bits (a zero's sign aside)."""
    rng = np.random.default_rng(11)
    for law in CUDA_LAWS:
        sig = law_source.law_signature(law)
        with test.subTest(law=sig.name):
            inputs = {name: _value(a, rng, N) for name, a in sig.parameters.items()}
            warp_results, native_results = (launch(kernel, N, inputs, sig, device) for kernel in twin_kernels(law))
            for a, b in zip(warp_results, native_results, strict=True):
                test.assertEqual(int(mismatches(a, b, signed_zeros=False).sum()), 0)


class TestSpatialLawSource(unittest.TestCase):
    def test_every_law_but_the_matrix_pack_and_joint_laws_renders(self):
        """Native code reaches every law but the three whose consumers are Warp only."""
        names = {law_source.law_signature(law).name for law in CUDA_LAWS}
        self.assertEqual(set(spatial_laws.SPATIAL.laws) - names, {"inertia_matrix", "joint_transform", "joint_motion"})
        source = spatial_laws.SPATIAL.prelude().source
        self.assertIn("return wp::transform_multiply(parent, child);", source)


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


# ---------------------------------------------------------------------------------------------------------------------
# projection_laws
#
# The projection laws solve the problems they name, on every test device.
#
# Each law meets its defining conditions, checked in float64: the tangent pair's disk problem, the friction triple's
# elliptic cone problem against its exact solve, the contact's Coulomb problem (cone, normal complementarity and, for
# Daviet's Newton, maximal dissipation), the regularized step's fixed point and its block-solve diagonal, and the
# velocity-limit, drive and set projections. The laws' CUDA renderings agree with these Warp functions
# (the twins below).
# ---------------------------------------------------------------------------------------------------------------------


@wp.kernel
def _pair(
    a: wp.array[float],
    c: wp.array[float],
    d: wp.array[float],
    residual: wp.array[wp.vec2],
    old: wp.array[wp.vec2],
    radius: wp.array[float],
    omega: wp.array[float],
    out: wp.array[wp.vec2],
    relaxed: wp.array[wp.vec2],
):
    i = wp.tid()
    out[i] = laws.disk(laws.friction_pair(a[i], c[i], d[i], residual[i], old[i], radius[i], 1.0), radius[i])
    relaxed[i] = laws.friction_pair(a[i], c[i], d[i], residual[i], old[i], radius[i], omega[i])


def test_friction_pair_solves_the_disk_problem(test, device):
    """Minimize the pair's quadratic on its disk: the residual at the result opposes every feasible move.

    At ``omega = 1`` the pair ``x`` and its residual ``r(x) = residual + A (x - old)`` meet the variational
    inequality ``r . x + radius |r| = 0``: zero inside the disk (sticking), anti-parallel on it (sliding).
    A relaxed pair moves the old one ``omega`` of the way.
    """
    rng = np.random.default_rng(61)
    n = 8192
    values = twin_inputs("friction_pair", rng, n, device)
    outputs = call(_pair, n, device, **values, out=np.zeros((n, 2), np.float32), relaxed=np.zeros((n, 2), np.float32))
    x = outputs["out"].numpy().astype(np.float64)
    relaxed = outputs["relaxed"].numpy().astype(np.float64)
    old = values["old"].astype(np.float64)
    a, c, d = (values[k].astype(np.float64) for k in ("a", "c", "d"))
    block = np.stack([np.stack([a, c], 1), np.stack([c, d], 1)], 1)
    r = values["residual"].astype(np.float64) + np.einsum("nij,nj->ni", block, x - old)
    radius = values["radius"].astype(np.float64)
    norm = np.linalg.norm(x, axis=1)
    test.assertTrue(np.all(norm <= radius * (1.0 + 1.0e-6) + 1.0e-30))
    test.assertTrue(np.all(x[radius == 0.0] == 0.0))
    loaded = radius > 0.0
    scale = np.abs(block).max(axis=(1, 2)) * np.maximum(radius, np.linalg.norm(old, axis=1))
    scale += np.linalg.norm(values["residual"], axis=1)
    gap = np.einsum("ni,ni->n", r, x) + radius * np.linalg.norm(r, axis=1)
    gap = gap[loaded] / (scale[loaded] * radius[loaded])
    test.assertGreater(gap.min(), -1.0e-5)
    test.assertLess(np.quantile(gap, 0.995), 1.0e-5)
    test.assertLess(gap.max(), 1.0e-3)
    # A sticking pair relaxes to old + omega (pair - old), before the caller's disk.
    omega = values["omega"].astype(np.float64)[:, None]
    sticking = loaded & (norm < radius * (1.0 - 1.0e-3))
    expected = old + omega * (x - old)
    test.assertLess(np.max(np.abs(relaxed[sticking] - expected[sticking]) / (1.0 + np.abs(expected[sticking]))), 1.0e-4)


@wp.kernel
def _pair_from_metric(
    largest: wp.array[float],
    smallest: wp.array[float],
    axis: wp.array[wp.vec2],
    residual: wp.array[wp.vec2],
    old: wp.array[wp.vec2],
    radius: wp.array[float],
    omega: wp.array[float],
    out: wp.array[wp.vec2],
):
    i = wp.tid()
    out[i] = laws.friction_pair_from_metric(largest[i], smallest[i], axis[i], residual[i], old[i], radius[i], omega[i])


def test_friction_pair_keeps_a_sliding_pair_at_its_solution(test, device):
    """Keep a sliding pair already at its disk problem's solution to the floor, bit for bit, at any radius.

    The hold is the root solve's own stop (:data:`~.projection_laws.PAIR_FLOOR`): the impulse on its circle and the
    residual anti-parallel to it, both within the floor. Impulses ``(3, 4) 2^e`` have exact norms ``5 2^e`` and
    residuals ``-2^k`` times them exact cross products; radii 0 or 2 ulps above that norm (1.9e-7 of the radius, inside
    the floor) keep the impulse, including radii under 1e-15 that the squared-norm screen leaves to the test; 8 ulps
    above (outside the floor, inside the screen) and 4e-6 above (outside the screen) re-solve it.
    """
    rng = np.random.default_rng(71)
    n = 4096
    values = twin_inputs("friction_pair_from_metric", rng, n, device)
    exponent = rng.integers(-60, 57, n)
    sign = rng.choice([-1.0, 1.0], (n, 2))
    legs = np.where(rng.integers(0, 2, n)[:, None] == 0, [3.0, 4.0], [4.0, 3.0])
    old = (sign * legs * np.ldexp(1.0, exponent)[:, None]).astype(np.float32)
    norm = np.ldexp(5.0, exponent).astype(np.float32)
    residual = (-old * np.ldexp(1.0, rng.integers(-2, 7, n))[:, None]).astype(np.float32)
    ulp = np.nextafter(norm, np.float32(np.inf)) - norm
    radii = {
        "on the circle": (norm, True),
        "2 ulps inside": (norm + 2 * ulp, True),
        "8 ulps inside": (norm + 8 * ulp, False),
        "4e-6 inside": (norm * np.float32(1.0 + 4.0e-6), False),
    }
    for name, (radius, kept) in radii.items():
        with test.subTest(radius=name):
            out = call(
                _pair_from_metric,
                n,
                device,
                **{k: values[k] for k in ("largest", "smallest", "axis", "omega")},
                residual=residual,
                old=old,
                radius=radius.astype(np.float32),
                out=np.zeros((n, 2), np.float32),
            )["out"].numpy()
            held = np.all(out == old, axis=1)
            if kept:
                test.assertTrue(held.all(), f"{int((~held).sum())} of {n} re-solved, radius {radius[~held][:3]}")
            else:
                test.assertFalse(held.any(), f"{int(held.sum())} of {n} kept")


def _ball_problems(rng, n):
    """Random friction triples: SPD blocks of condition up to 1e6 before their rows' scales (two decades), friction
    coefficients and loads over two decades, torsional coefficients 0.003 to 0.3 times the friction ones, old
    impulses near the cone and residuals from sticking to sliding (float64)."""
    rotation, _ = np.linalg.qr(rng.normal(size=(n, 3, 3)))
    condition = 10.0 ** rng.uniform(0.0, 6.0, n)
    spectrum = np.stack([np.ones(n), condition ** -rng.uniform(0.0, 1.0, n), 1.0 / condition], 1)
    rows = 10.0 ** rng.uniform(-1.0, 1.0, (n, 3))
    block = np.einsum("nij,nj,nkj->nik", rotation, spectrum, rotation) * rows[:, :, None] * rows[:, None, :]
    mu = 10.0 ** rng.uniform(-1.0, 1.0, n)
    mu_t = mu * 10.0 ** rng.uniform(-2.5, -0.5, n)
    load = 10.0 ** rng.uniform(-1.0, 1.0, n)
    coefficients = np.stack([mu, mu, mu_t], 1)
    old = rng.normal(size=(n, 3)) * coefficients * load[:, None] * 0.3
    size = np.einsum("nii->ni", block) * coefficients * load[:, None]
    residual = rng.normal(size=(n, 3)) * size * 10.0 ** rng.uniform(-3.0, 0.5, (n, 1))
    return block, residual, old, mu, mu_t, load


def _ball_reference(block, residual, old, coefficients, load):
    """Solve ``min 1/2 dx^T G dx + r^T dx`` on ``|(old + dx) / c| <= load`` exactly in float64: in ``y = x / c`` the
    trust-region problem, its multiplier by bisection to the float64 floor. Returns the solution and whether it
    sticks (lies inside the cone)."""
    scaled = block * coefficients[:, :, None] * coefficients[:, None, :]
    gradient = coefficients * residual - np.einsum("nij,nj->ni", scaled, old / coefficients)
    values, vectors = np.linalg.eigh(scaled)
    rotated = np.einsum("nji,nj->ni", vectors, gradient)
    inside = np.linalg.norm(rotated / values, axis=1) <= load
    lo, hi = np.zeros(len(load)), np.linalg.norm(rotated, axis=1) / load + values.max(axis=1)
    for _ in range(200):
        alpha = 0.5 * (lo + hi)
        outside = np.linalg.norm(rotated / (values + alpha[:, None]), axis=1) > load
        lo, hi = np.where(outside, alpha, lo), np.where(outside, hi, alpha)
    alpha = np.where(inside, 0.0, hi)
    y = -np.einsum("nij,nj->ni", vectors, rotated / (values + alpha[:, None]))
    return coefficients * y, inside


@wp.kernel
def _ball(
    block: wp.array[wp.mat33],
    residual: wp.array[wp.vec3],
    old: wp.array[wp.vec3],
    mu: wp.array[float],
    mu_t: wp.array[float],
    load: wp.array[float],
    omega: wp.array[float],
    out: wp.array[wp.vec3],
    relaxed: wp.array[wp.vec3],
):
    i = wp.tid()
    out[i] = laws.friction_ball(block[i], residual[i], old[i], mu[i], mu_t[i], load[i], 1.0)
    relaxed[i] = laws.friction_ball(block[i], residual[i], old[i], mu[i], mu_t[i], load[i], omega[i])


def _run_ball(device, block, residual, old, mu, mu_t, load, omega=None):
    n = len(load)
    omega = np.ones(n) if omega is None else omega
    values = {"block": block, "residual": residual, "old": old, "mu": mu, "mu_t": mu_t, "load": load, "omega": omega}
    values = {name: np.asarray(value, np.float32) for name, value in values.items()}
    zeros = np.zeros((n, 3), np.float32)
    outputs = call(_ball, n, device, **values, out=zeros, relaxed=zeros)
    return values, outputs["out"].numpy().astype(np.float64), outputs["relaxed"].numpy().astype(np.float64)


def test_friction_ball_solves_the_cone_problem(test, device):
    """Minimize the triple's quadratic on its elliptic cone: the float64 solution's objective to the rounding floor.

    On 10^4 random triples (:func:`_ball_problems`) the objective gap to the exact float64 solve, relative to the
    size of its terms over the cone ``|c r| load + |G| (c load)^2``, is at most 1e-6 (eight float32 ulps) on 99.9%
    and 1e-5 on every one; the result leaves the cone by at most 2e-6 of the load (the caller projects it). A
    sticking triple relaxes to ``old + omega (x - old)``.
    """
    rng = np.random.default_rng(83)
    n = 10000
    problem = _ball_problems(rng, n)
    values, x, relaxed = _run_ball(device, *problem, omega=rng.choice([1.0, 0.8, 1.2], n))
    block, residual, old, mu, mu_t, load = (
        values[k].astype(np.float64) for k in ("block", "residual", "old", "mu", "mu_t", "load")
    )
    coefficients = np.stack([mu, mu, mu_t], 1)
    reference, inside = _ball_reference(block, residual, old, coefficients, load)
    test.assertTrue(0.1 < inside.mean() < 0.9, f"sticking fraction {inside.mean()}")

    def objective(impulse):
        change = impulse - old
        return 0.5 * np.einsum("ni,nij,nj->n", change, block, change) + np.einsum("ni,ni->n", residual, change)

    terms = np.linalg.norm(coefficients * residual, axis=1) * load
    terms += np.abs(block).max(axis=(1, 2)) * (coefficients.max(axis=1) * load) ** 2
    gap = (objective(x) - objective(reference)) / terms
    test.assertLess(np.quantile(np.abs(gap), 0.999), 1.0e-6)
    test.assertLess(np.abs(gap).max(), 1.0e-5)
    test.assertLess((np.linalg.norm(x / coefficients, axis=1) / load - 1.0).max(), 2.0e-6)
    omega = values["omega"].astype(np.float64)[:, None]
    expected = old + omega * (x - old)
    error = np.abs(relaxed[inside] - expected[inside]) / (coefficients[inside] * load[inside, None])
    test.assertLess(error.max(), 1.0e-4)


def test_friction_ball_keeps_a_sliding_triple_at_its_solution(test, device):
    """Keep a sliding triple already at its cone problem's solution to the floor, bit for bit, at any load.

    The pair law's hold (:data:`~.projection_laws.PAIR_FLOOR`) in the scaled impulse ``y = (t1 / mu, t2 / mu,
    s / mu_t)``: on its sphere and its gradient ``c r`` anti-parallel to it, both within the floor. With power-of-two
    coefficients, ``y = (2, 3, 6) 2^e`` has the exact norm ``7 2^e`` and residuals ``-2^k y / c`` exact cross
    products: loads 0 or 2 ulps above that norm keep the impulse; 8 ulps above and 4e-6 above re-solve it.
    """
    rng = np.random.default_rng(73)
    n = 4096
    exponent = rng.integers(-40, 40, n)
    mu = np.ldexp(1.0, rng.integers(-3, 4, n))
    mu_t = mu * np.ldexp(1.0, rng.integers(-8, -1, n))
    coefficients = np.stack([mu, mu, mu_t], 1)
    legs = np.array([[2.0, 3.0, 6.0], [6.0, 2.0, 3.0], [3.0, 6.0, 2.0]])[rng.integers(0, 3, n)]
    y = rng.choice([-1.0, 1.0], (n, 3)) * legs * np.ldexp(1.0, exponent)[:, None]
    old = coefficients * y
    residual = -y / coefficients * np.ldexp(1.0, rng.integers(-2, 7, n))[:, None]
    block = _ball_problems(rng, n)[0]
    norm = np.ldexp(7.0, exponent).astype(np.float32)
    ulp = np.nextafter(norm, np.float32(np.inf)) - norm
    loads = {
        "on the sphere": (norm, True),
        "2 ulps inside": (norm + 2 * ulp, True),
        "8 ulps inside": (norm + 8 * ulp, False),
        "4e-6 inside": (norm * np.float32(1.0 + 4.0e-6), False),
    }
    for name, (load, kept) in loads.items():
        with test.subTest(load=name):
            values, x, _ = _run_ball(device, block, residual, old, mu, mu_t, load)
            held = np.all(x == values["old"], axis=1)
            if kept:
                test.assertTrue(held.all(), f"{int((~held).sum())} of {n} re-solved, load {load[~held][:3]}")
            else:
                test.assertFalse(held.any(), f"{int(held.sum())} of {n} kept")


def test_friction_ball_reduces_to_the_pair(test, device):
    """Solve the pair alone where the spin takes no part, and the spin alone where the tangents take none.

    ``mu_t = 0`` is :func:`~.projection_laws.friction_pair` bit for bit, the spin pinned at zero. A spin row
    decoupled from the pair (no coupling in the block, no residual, no impulse) stays at zero at any ``mu_t``, and
    the joint solve meets the pair's disk problem as the pair law does: the objectives agree to the rounding floor
    (the impulses themselves only to the blocks' conditioning, up to 1e6 times it). ``mu = 0`` pins the tangents and
    leaves the spin's box solve, ``clamp(s - r_s / G_ss, -mu_t load, mu_t load)``.
    """
    rng = np.random.default_rng(79)
    n = 4096
    block, residual, old, mu, mu_t, load = _ball_problems(rng, n)
    old[:, 2] = 0.0
    values, pinned, _ = _run_ball(device, block, residual, old, mu, np.zeros(n), load)
    pair = call(
        _pair,
        n,
        device,
        a=np.ascontiguousarray(values["block"][:, 0, 0]),
        c=np.ascontiguousarray(values["block"][:, 0, 1]),
        d=np.ascontiguousarray(values["block"][:, 1, 1]),
        residual=np.ascontiguousarray(values["residual"][:, :2]),
        old=np.ascontiguousarray(values["old"][:, :2]),
        radius=values["mu"] * values["load"],
        omega=np.ones(n, np.float32),
        out=np.zeros((n, 2), np.float32),
        relaxed=np.zeros((n, 2), np.float32),
    )["relaxed"].numpy()
    np.testing.assert_array_equal(pinned[:, :2], pair)
    np.testing.assert_array_equal(pinned[:, 2], 0.0)
    decoupled = block.copy()
    decoupled[:, :2, 2] = decoupled[:, 2, :2] = 0.0
    spinless = residual.copy()
    spinless[:, 2] = 0.0
    values, joint, _ = _run_ball(device, decoupled, spinless, old, mu, mu_t, load)
    test.assertLess((np.abs(joint[:, 2]) / (values["mu_t"] * values["load"])).max(), 1.0e-6)
    g = values["block"][:, :2, :2].astype(np.float64)
    r = values["residual"][:, :2].astype(np.float64)
    start = values["old"][:, :2].astype(np.float64)

    def objective(impulse):
        change = impulse - start
        return 0.5 * np.einsum("ni,nij,nj->n", change, g, change) + np.einsum("ni,ni->n", r, change)

    radius = values["mu"].astype(np.float64) * values["load"]
    terms = np.linalg.norm(r, axis=1) * radius + np.abs(g).max(axis=(1, 2)) * radius**2
    gap = np.abs(objective(joint[:, :2]) - objective(pair.astype(np.float64))) / terms
    test.assertLess(np.quantile(gap, 0.999), 1.0e-6)
    test.assertLess(gap.max(), 1.0e-5)
    spin_old = np.zeros_like(old)
    spin_old[:, 2] = old[:, 2] + rng.normal(size=n) * mu_t * load
    values, spin, _ = _run_ball(device, block, residual, spin_old, np.zeros(n), mu_t, load)
    np.testing.assert_array_equal(spin[:, :2], 0.0)
    bound = values["mu_t"].astype(np.float64) * values["load"]
    step = values["residual"][:, 2].astype(np.float64) / values["block"][:, 2, 2]
    expected = np.clip(values["old"][:, 2] - step, -bound, bound)
    test.assertLess((np.abs(spin[:, 2] - expected) / bound).max(), 1.0e-5)


@wp.kernel
def _scalar_laws(
    old: wp.array[float],
    residual: wp.array[float],
    diagonal: wp.array[float],
    w: wp.array[float],
    bound: wp.array[float],
    out: wp.array2d[float],
):
    i = wp.tid()
    step = -residual[i] / diagonal[i]
    out[i, 0] = laws.relaxed_step(old[i], laws.regularized_delta(step, w[i], old[i]), 0.7)
    out[i, 1] = laws.regularized_diagonal(diagonal[i], w[i])
    out[i, 2] = laws.unilateral(old[i])
    out[i, 3] = laws.box(old[i], -bound[i], bound[i])
    new, delta = laws.velocity_limit_update(old[i], residual[i] / diagonal[i])
    push_new, push_delta = laws.velocity_limit_update(0.0, residual[i] / diagonal[i])
    out[i, 4] = new
    out[i, 5] = delta
    out[i, 6] = push_new
    out[i, 7] = push_delta
    row = laws.speed_row(wp.abs(residual[i]) * 3.0, bound[i], diagonal[i], 0.0)
    out[i, 9] = row[2]


@wp.kernel
def _disk(pair: wp.array[wp.vec2], radius: wp.array[float], out: wp.array[wp.vec2]):
    i = wp.tid()
    out[i] = laws.disk(pair[i], radius[i])


def test_disk_keeps_a_pair_inside_it(test, device):
    """Keep a pair inside its disk bit for bit and scale one outside onto it; the CUDA rendering screens squared
    norms and takes its root and division only outside, so a sticking pair's visit takes neither. (A pair on the
    boundary goes either way with the device's rounding of its squared magnitude, as before.)"""
    rng = np.random.default_rng(29)
    n = 4096
    pair = rng.normal(size=(n, 2)).astype(np.float32)
    magnitude = np.sqrt((pair.astype(np.float64) ** 2).sum(axis=1)).astype(np.float32)
    scale = rng.choice(np.array([0.5, 2.0], np.float32), n)
    radius = (magnitude * scale).astype(np.float32)
    radius[:8] = 0.0
    pair[:4] = 0.0
    out = call(_disk, n, device, pair=pair, radius=radius, out=np.zeros((n, 2), np.float32))["out"].numpy()
    inside = ~pair.any(axis=1) | ((scale == 2.0) & (radius > 0.0))
    test.assertTrue(inside.any() and (~inside).any())
    np.testing.assert_array_equal(out[inside].view(np.uint32), pair[inside].view(np.uint32))
    scaled = np.sqrt((out[~inside].astype(np.float64) ** 2).sum(axis=1))
    np.testing.assert_allclose(scaled, radius[~inside], rtol=1e-6, atol=1e-30)
    source = laws.PROJECTION_LAW.prelude().source
    body = source[source.index(" disk(wp::vec2 pair, float radius) {") :]
    body = body[: body.index("\n        }\n")]
    test.assertIn(
        "if (magnitude_sq > radius * radius) {\n                return wp::mul(pair, radius / wp::sqrt(magnitude_sq));",
        body,
    )


def test_scalar_laws(test, device):
    """Hold each scalar law's definition: steps, sets, velocity limits and the friction gate.

    The regularized step's fixed point, residual ``-(1 / w - 1) d lambda``, is the exact solve against
    :func:`~.projection_laws.regularized_diagonal`.
    """
    rng = np.random.default_rng(63)
    n = 4096
    old = scalars(rng, n)
    residual = scalars(rng, n)
    diagonal = rng.uniform(0.1, 3.0, n).astype(np.float32)
    w = rng.choice(np.array([1.0, 0.5, 0.98, 0.25], np.float32), n)
    bound = rng.uniform(0.05, 2.0, n).astype(np.float32)
    out = call(
        _scalar_laws,
        n,
        device,
        old=old,
        residual=residual,
        diagonal=diagonal,
        w=w,
        bound=bound,
        out=np.zeros((n, 10), np.float32),
    )["out"].numpy()
    o, r, d, w64, b = (v.astype(np.float64) for v in (old, residual, diagonal, w, bound))
    np.testing.assert_allclose(out[:, 0], o + 0.7 * (w64 * (-r / d) - (1.0 - w64) * o), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(out[:, 1], d / w64, rtol=1e-6)
    np.testing.assert_array_equal(out[:, 2], np.where(old < 0.0, 0.0, old))
    np.testing.assert_array_equal(out[:, 3], np.clip(old, -bound, bound))
    np.testing.assert_allclose(out[:, 4], np.maximum(o - r / d, 0.0), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(out[:, 5], out[:, 4] - old, rtol=1e-6, atol=1e-6)
    # From a zero total the update pushes a violated row onto its limit (the tiers' last word).
    push = np.where(r < 0.0, -r / d, 0.0)
    np.testing.assert_allclose(out[:, 6], push, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(out[:, 7], push, rtol=1e-6, atol=1e-7)
    # A free body over its speed limit keeps its direction and ends on the limit.
    speed = np.abs(r) * 3.0
    over = speed > b
    np.testing.assert_allclose(speed[over] * (1.0 + out[over, 9]), b[over], rtol=1e-5)
    np.testing.assert_array_equal(out[~over, 9], 0.0)
    # The regularized step's fixed point solves the row against the regularized diagonal.
    b_free = r
    fixed = -b_free * w64 / d
    test.assertTrue(np.allclose(fixed, -b_free / (d / w64)))
    stepped = fixed + 0.7 * (w64 * (-(d * fixed + b_free) / d) - (1.0 - w64) * fixed)
    np.testing.assert_allclose(stepped, fixed, rtol=1e-12, atol=1e-12)
    test.assertEqual(ROW_CONTACT, 0)


# ---------------------------------------------------------------------------------------------------------------------
# law_source
#
# One source per FeatherPGS law: the families render, compile and agree with their Warp functions.
#
# The laws (``contact_laws``, ``row_laws``, ``projection_laws``) are written once in Warp's language; ``law_source``
# renders each as a CUDA function for native sweeps. The tests check the rendering mechanism, the constants the laws
# share with the code they replace, and the agreement of each law's two renderings.
# ---------------------------------------------------------------------------------------------------------------------


class TestLawSource(unittest.TestCase):
    """The families render once per law, compile for CUDA and keep the constants of the code they replace."""

    def test_families_render_every_cuda_law(self):
        """Render one struct per family with a static function per law that has a CUDA form."""
        for family in FAMILIES:
            source = family.prelude().source
            self.assertEqual(source.count(f"struct {family.name} {{"), 1)
            self.assertIs(family.prelude(), family.prelude())
            for name in family.laws:
                sig = law_source.law_signature(LAWS.get(name) or getattr(contact_laws, name))
                rendered = f" {name}(" in source
                self.assertEqual(rendered, sig.cuda, name)
        self.assertFalse(law_source.law_signature(spatial_laws.inertia_matrix).cuda)
        self.assertEqual(
            [prelude.name for prelude in law_source.law_preludes(row_laws.ROW_LAW, row_laws.ROW_LAW)], ["RowLaw"]
        )

    def test_renderings_keep_the_python_statements(self):
        """Render one C++ statement per Python statement, in order, with Warp's rounding of float constants."""
        source = projection_laws.PROJECTION_LAW.prelude().source
        for line in (
            "auto scale = wp::max(wp::max(a, d), 1e-20f);",
            "auto largest = 0.5f * (a + d + wp::sqrt((a - d) * (a - d) + 4.0f * c * c));",
            "return wp::add(old, wp::mul(omega, wp::sub(result, old)));",
        ):
            self.assertIn(line, source)
        for value in (1.0e-6, 2.0e-7, 1.0e-20, 0.1, 1.0 / 3.0, 1.0e20, 6.0e-8):
            text = law_source._float_literal(value)
            self.assertEqual(np.float32(float(text[:-1])), np.float32(value), text)

    def test_subset_is_enforced(self):
        """Refuse what the CUDA rendering cannot keep equal to the Warp function, naming the law."""
        family = law_source.LawFamily("ScratchLaw")

        @family.law
        def chained(a: float, b: float) -> bool:
            return 0.0 < a < b

        @family.law
        def late_read(a: float) -> float:
            if a > 0.0:
                b = a
            return b

        for law in (chained, late_read):
            with self.assertRaises(law_source.LawSourceError) as raised:
                law_source.render_cuda(law)
            self.assertIn(law_source.law_signature(law).name, str(raised.exception))

    def test_cuda_renderings_compile(self):
        """Compile the families' CUDA renderings through NVRTC, without a device (sm_80)."""
        import tempfile  # noqa: PLC0415

        body = "    return 0;\n#endif\n"
        namespace = {
            "wp": wp,
            "snippet": cuda_native(body, preludes=law_source.law_preludes(*FAMILIES)),
        }
        source = (
            "@wp.func_native(snippet)\ndef families() -> int: ...\n\n"
            "def compile_families(out: wp.array[int]):\n    out[0] = families()\n"
        )
        kernel = exec_kernel("compile_families", source, namespace)
        with tempfile.TemporaryDirectory() as directory:
            try:
                wp.compile_aot_module(kernel.module, arch=80, module_dir=directory)
            except Exception as error:
                if "NVRTC" in str(error) and "not" in str(error):
                    self.skipTest(f"NVRTC unavailable: {error}")
                raise
            self.assertTrue(any(name.endswith((".cubin", ".ptx")) for name in os.listdir(directory)))


def test_family_twins_agree(test, device):
    """Return equal results from each law's Warp function and CUDA rendering, to an iterative law's tolerance.

    Every other law's results are bitwise equal up to the sign of a zero: Warp negates by subtracting from zero.
    An iterative law's samples agree within its tolerance but for rare ones whose solve takes the other branch of
    a convergence or sticking test, and the ball metric's to the rounding floor (``TWIN_TOLERANCES``).
    """
    rng = np.random.default_rng(1234)
    for name, law in LAWS.items():
        sig = law_source.law_signature(law)
        if not sig.cuda:
            continue
        with test.subTest(law=name):
            warp_kernel, native_kernel = twin_kernels(law)
            inputs = twin_inputs(name, rng, TWIN_SAMPLES, device)
            warp_results = launch(warp_kernel, TWIN_SAMPLES, inputs, sig, device)
            native_results = launch(native_kernel, TWIN_SAMPLES, inputs, sig, device)
            for k, (a, b) in enumerate(zip(warp_results, native_results, strict=True)):
                if name in TWIN_TOLERANCES and a.dtype.kind == "f":
                    key, bound = TWIN_TOLERANCES[name]
                    a64 = a.reshape(TWIN_SAMPLES, -1).astype(np.float64)
                    b64 = b.reshape(TWIN_SAMPLES, -1).astype(np.float64)
                    np.testing.assert_array_equal(np.isnan(a64), np.isnan(b64))
                    if key is None:
                        scale = np.maximum(np.nan_to_num(np.abs(a64)).max(axis=1, keepdims=True), 1.0e-3)
                    else:
                        scale = np.maximum(np.asarray(inputs[key], dtype=np.float64), 1.0e-6)[:, None]
                    gap = (np.nan_to_num(np.abs(a64 - b64)) / scale).max(axis=1)
                    test.assertLessEqual(float(np.quantile(gap, TWIN_AGREEING)), bound, f"result {k}")
                    test.assertLess(float(gap.max()), 1.0e3 * bound, f"result {k}")
                    continue
                bad = np.flatnonzero(mismatches(a, b, signed_zeros=False))
                test.assertEqual(bad.size, 0, f"result {k}: {bad.size} of {TWIN_SAMPLES} differ, first at {bad[:3]}")


class TestFeatherPGSLaws(unittest.TestCase):
    """Each law against its definition, and its two renderings, on every test device."""


register_device_tests(TestFeatherPGSLaws, globals())


if __name__ == "__main__":
    unittest.main(verbosity=2)
