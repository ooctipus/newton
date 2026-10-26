# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The spatial kit: one source for Warp and CUDA, and the sites it replaces.

Each law's two renderings agree bit for bit (up to the sign of a zero). Each
site that spells a kit law today is compared with the law on random inputs:
bitwise where the site computes the same operations in the same order, and
otherwise within a stated bound, in float32 units of the magnitude of the
terms (``term_ulps``), with the reason the order differs.
"""

import unittest

import numpy as np
import warp as wp

from newton._src.math import velocity_at_point
from newton._src.math.spatial import transform_twist
from newton._src.sim import JointType
from newton._src.solvers.feather_pgs import (
    cached_force,
    kernels,
    law_source,
    scalar_branches,
    spatial_laws,
    tree_kinetic_state,
)
from newton._src.solvers.feather_pgs.device_preludes import cuda_native, dual_native
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
    def test_every_law_but_the_matrix_pack_and_joint_transform_renders(self):
        """Native code reaches every law but the two whose consumers are Warp only."""
        names = {law_source.law_signature(law).name for law in CUDA_LAWS}
        self.assertEqual(set(spatial_laws.SPATIAL.laws) - names, {"inertia_matrix", "joint_transform"})
        source = spatial_laws.SPATIAL.prelude().source
        self.assertIn("return wp::transform_multiply(parent, child);", source)


# ---------------------------------------------------------------------------------------------------------------------
# Sites: each spelling of a kit law today against the law
# ---------------------------------------------------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _compose_sites(
    parent_v: wp.array[wp.spatial_vector],
    parent_a: wp.array[wp.spatial_vector],
    joint_v: wp.array[wp.spatial_vector],
    law_v: wp.array[wp.spatial_vector],
    law_a: wp.array[wp.spatial_vector],
    serial_v: wp.array[wp.spatial_vector],
    serial_a: wp.array[wp.spatial_vector],
):
    i = wp.tid()
    v, a = spatial_laws.twist_compose(parent_v[i], parent_a[i], joint_v[i], wp.spatial_vector())
    law_v[i] = v
    law_a[i] = a
    # kernels.compute_link_kinematics: the velocity product of the child's total twist with the joint's.
    v_s = parent_v[i] + joint_v[i]
    serial_v[i] = v_s
    serial_a[i] = parent_a[i] + kernels.spatial_cross(v_s, joint_v[i])


@wp.kernel(enable_backward=False)
def _twist_sites(
    frame: wp.array[wp.transform],
    twist: wp.array[wp.spatial_vector],
    offset: wp.array[wp.vec3],
    law: wp.array[wp.spatial_vector],
    newton_twist: wp.array[wp.spatial_vector],
    law_point: wp.array[wp.vec3],
    finalize_point: wp.array[wp.vec3],
    body_terms_point: wp.array[wp.vec3],
    newton_point: wp.array[wp.vec3],
):
    i = wp.tid()
    law[i] = spatial_laws.twist_transform(frame[i], twist[i])
    newton_twist[i] = transform_twist(frame[i], twist[i])
    law_point[i] = spatial_laws.point_velocity(twist[i], offset[i])
    v_s = twist[i]
    # kernels.finalize_body_dynamics_values and update_body_qd_from_featherstone.
    finalize_point[i] = wp.spatial_top(v_s) + wp.cross(wp.spatial_bottom(v_s), offset[i])
    # tree_kinetic_state.get_helpers._body_terms (the public COM twist) and scalar_branches' leaf velocity.
    linear, omega = wp.spatial_top(v_s), wp.spatial_bottom(v_s)
    body_terms_point[i] = linear + wp.cross(omega, offset[i])
    # friction_patches._point_velocity.
    newton_point[i] = velocity_at_point(v_s, offset[i])


@wp.kernel(enable_backward=False)
def _wrench_sites(
    wrench: wp.array[wp.spatial_vector],
    offset: wp.array[wp.vec3],
    flags: wp.array[int],
    zeros: wp.array[wp.spatial_vector],
    poses: wp.array[wp.transform],
    law: wp.array[wp.spatial_vector],
    scalar: wp.array[wp.spatial_vector],
    net: wp.array[wp.spatial_vector],
    transport_law: wp.array[wp.vec3],
    transport_site: wp.array[wp.vec3],
):
    i = wp.tid()
    law[i] = spatial_laws.wrench_shift(wrench[i], offset[i])
    scalar[i] = scalar_branches._external_at_origin(i, offset[i], wrench, flags)
    # kernels._compute_body_net_wrench subtracts the shifted wrench; with no other wrench it returns its negative.
    origin = wp.vec3()
    net[i] = -kernels._compute_body_net_wrench(
        i,
        wp.spatial_vector(),
        origin,
        zeros,
        wrench,
        flags,
        poses,
        offset,
    )
    linear, angular = wp.spatial_top(wrench[i]), wp.spatial_bottom(wrench[i])
    transport_law[i] = spatial_laws.free_root_transport(linear, angular)
    # kernels.apply_free_root_transport_to_predictor and _light_remove_free_root_transport_from_qdd.
    transport_site[i] = wp.cross(angular, linear)


@wp.kernel(enable_backward=False)
def _inertia_sites(
    mass: wp.array[float],
    inertia: wp.array[wp.mat33],
    pose: wp.array[wp.transform],
    twist: wp.array[wp.spatial_vector],
    law_moment: wp.array[wp.vec3],
    law_rotational: wp.array[wp.mat33],
    held_rotational: wp.array[wp.mat33],
    compact_com: wp.array[wp.vec3],
    compact_rotational: wp.array[wp.mat33],
    law_apply: wp.array[wp.spatial_vector],
    compact_apply: wp.array[wp.spatial_vector],
    dense_apply: wp.array[wp.spatial_vector],
    law_matrix: wp.array[wp.spatial_matrix],
    compact_matrix: wp.array[wp.spatial_matrix],
    adjoint_matrix: wp.array[wp.spatial_matrix],
    law_action: wp.array[wp.vec3],
    held_action: wp.array[wp.vec3],
):
    i = wp.tid()
    rotation = wp.transform_get_rotation(pose[i])
    offset = wp.transform_get_translation(pose[i])
    m, h, rotational = spatial_laws.inertia_at(mass[i], inertia[i], rotation, offset)
    law_moment[i] = h
    law_rotational[i] = rotational
    # tree_kinetic_state: _body_terms rotates the inertia, _STORE_TERMS_SOURCE adds the parallel-axis term entry by entry.
    matrix = wp.quat_to_matrix(rotation)
    rotated = matrix * inertia[i] * wp.transpose(matrix)
    square = wp.dot(offset, offset)
    held = wp.mat33()
    for r in range(3):
        for c in range(3):
            diagonal = float(0.0)
            if r == c:
                diagonal = square
            held[r, c] = rotated[r, c] + m * (diagonal - offset[r] * offset[c])
    held_rotational[i] = held
    # kernels.transform_com_inertia_terms: the parallel-axis term as -m [c]x [c]x.
    com, inertia_origin = kernels.transform_com_inertia_terms(pose[i], m, inertia[i])
    compact_com[i] = com
    compact_rotational[i] = inertia_origin
    law_apply[i] = spatial_laws.inertia_apply(m, h, rotational, twist[i])
    compact_apply[i] = kernels.mul_com_spatial_inertia(m, com, inertia_origin, twist[i])
    law_matrix[i] = spatial_laws.inertia_matrix(m, h, rotational)
    compact_matrix[i] = kernels.assemble_com_spatial_inertia(m, com, inertia_origin)
    dense_apply[i] = compact_matrix[i] * twist[i]
    # kernels.refresh_masked_body_inertia: the dense adjoint transform of the body-frame matrix.
    body = wp.spatial_matrix()
    for r in range(3):
        body[r, r] = m
        for c in range(3):
            body[3 + r, 3 + c] = inertia[i][r, c]
    adjoint_matrix[i] = kernels.transform_spatial_inertia(pose[i], body)
    law_action[i] = spatial_laws.inertia_action(rotation, inertia[i], offset)
    held_action[i] = tree_kinetic_state._inertia_action(rotation, inertia[i], offset)


@wp.kernel(enable_backward=False)
def _joint_sites(
    joint_type: wp.array[int],
    joint_axis: wp.array[wp.vec3],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    lin_count: wp.array[int],
    ang_count: wp.array[int],
    anchor: wp.array[wp.transform],
    child: wp.array[wp.transform],
    law_transform: wp.array[wp.transform],
    site_transform: wp.array[wp.transform],
    law_axes: wp.array2d[wp.spatial_vector],
    site_axes: wp.array[wp.spatial_vector],
    law_leaf: wp.array[wp.transform],
    scalar_leaf: wp.array[wp.transform],
    law_leaf_axis: wp.array[wp.spatial_vector],
    scalar_leaf_axis: wp.array[wp.spatial_vector],
):
    i = wp.tid()
    t = joint_type[i]
    law_transform[i] = spatial_laws.joint_transform(t, joint_axis, 6 * i, lin_count[i], ang_count[i], joint_q, 7 * i)
    site_transform[i] = kernels.jcalc_transform(t, joint_axis, 6 * i, lin_count[i], ang_count[i], joint_q, 7 * i)
    # jcalc_motion stores one column per coordinate; the kit forms each column with motion_axis, its kind a
    # constant in each branch as at the call sites (a kind chosen at run time compiles to another FMA grouping).
    kernels.jcalc_motion(t, joint_axis, lin_count[i], ang_count[i], anchor[i], joint_qd, 6 * i, site_axes)
    for k in range(lin_count[i] + ang_count[i]):
        if k < lin_count[i]:
            law_axes[i, k] = spatial_laws.motion_axis(anchor[i], joint_axis[6 * i + k], False)
        else:
            law_axes[i, k] = spatial_laws.motion_axis(anchor[i], joint_axis[6 * i + k], True)
    # scalar_branches: one-coordinate leaves, each kind in its own branch.
    axis = joint_axis[6 * i]
    q = joint_q[7 * i]
    inverse_child = wp.transform_inverse(child[i])
    if t == JointType.REVOLUTE:
        law_leaf[i] = anchor[i] * spatial_laws.axis_transform(True, axis, q) * inverse_child
        law_leaf_axis[i] = spatial_laws.motion_axis(anchor[i], axis, True)
        scalar_leaf[i] = scalar_branches._revolute_pose(anchor[i], axis, q, child[i])
        scalar_leaf_axis[i] = scalar_branches._revolute_motion(anchor[i], wp.vec3(), axis)
    else:
        law_leaf[i] = anchor[i] * spatial_laws.axis_transform(False, axis, q) * inverse_child
        law_leaf_axis[i] = spatial_laws.motion_axis(anchor[i], axis, False)
        scalar_leaf[i] = scalar_branches._prismatic_pose(anchor[i], axis, q, child[i])
        scalar_leaf_axis[i] = scalar_branches._prismatic_motion(anchor[i], axis)


def _launch(kernel, inputs, outputs, device, n=N):
    wp.launch(kernel, n, inputs=inputs, outputs=outputs, device=device)
    return [o.numpy() for o in outputs]


def test_compose_sites(test, device):
    """The scans' twist compose against the serial route's (ulps: ``(v_p + v) x v`` there, ``v_p x v`` here)."""
    rng = np.random.default_rng(1)
    arrays = [
        wp.array(rng.standard_normal((N, 6)).astype(np.float32), dtype=wp.spatial_vector, device=device)
        for _ in range(3)
    ]
    outs = [wp.zeros(N, dtype=wp.spatial_vector, device=device) for _ in range(4)]
    law_v, law_a, serial_v, serial_a = _launch(_compose_sites, arrays, outs, device)
    test.assertEqual(int(mismatches(law_v, serial_v).sum()), 0)
    # Terms of the acceleration: the parent's and the velocity products, of order |v|^2.
    v = np.linalg.norm(arrays[2].numpy(), axis=1, keepdims=True)
    scale = np.abs(arrays[1].numpy()) + (np.linalg.norm(arrays[0].numpy(), axis=1, keepdims=True) + v) * v
    test.assertLess(float(term_ulps(law_a, serial_a, scale).max()), 2.0)


def test_twist_sites(test, device):
    """Twist transform and point velocity: every site's spelling is the law's arithmetic."""
    rng = np.random.default_rng(2)
    frame = wp.array(_value(wp.transform, rng, N), dtype=wp.transform, device=device)
    twist = wp.array(_value(wp.spatial_vector, rng, N), dtype=wp.spatial_vector, device=device)
    offset = wp.array(_value(wp.vec3, rng, N), dtype=wp.vec3, device=device)
    outs = [wp.zeros(N, dtype=wp.spatial_vector, device=device) for _ in range(2)]
    outs += [wp.zeros(N, dtype=wp.vec3, device=device) for _ in range(4)]
    law, site, point, *points = _launch(_twist_sites, [frame, twist, offset], outs, device)
    test.assertEqual(int(mismatches(law, site).sum()), 0)
    for other in points:
        test.assertEqual(int(mismatches(point, other, signed_zeros=False).sum()), 0)


def test_wrench_sites(test, device):
    """Wrench shift and free-root transport: the scalar branches', the net wrench's and the transport sites'."""
    rng = np.random.default_rng(3)
    wrench = wp.array(_value(wp.spatial_vector, rng, N), dtype=wp.spatial_vector, device=device)
    offset = wp.array(_value(wp.vec3, rng, N), dtype=wp.vec3, device=device)
    flags = wp.zeros(N, dtype=int, device=device)
    zeros = wp.zeros(N, dtype=wp.spatial_vector, device=device)
    poses = wp.array(np.tile([0, 0, 0, 0, 0, 0, 1], (N, 1)).astype(np.float32), dtype=wp.transform, device=device)
    outs = [wp.zeros(N, dtype=wp.spatial_vector, device=device) for _ in range(3)]
    outs += [wp.zeros(N, dtype=wp.vec3, device=device) for _ in range(2)]
    law, scalar, net, transport_law, transport_site = _launch(
        _wrench_sites, [wrench, offset, flags, zeros, poses], outs, device
    )
    test.assertEqual(int(mismatches(law, scalar).sum()), 0)
    test.assertEqual(int(mismatches(law, net, signed_zeros=False).sum()), 0)
    test.assertEqual(int(mismatches(transport_law, transport_site).sum()), 0)


def test_inertia_sites(test, device):
    """One inertia representation against the held tree's (bitwise) and the compact and dense forms (ulps)."""
    rng = np.random.default_rng(4)
    mass = wp.array(_value(float, rng, N), dtype=float, device=device)
    inertia = wp.array(_inertias(rng, N), dtype=wp.mat33, device=device)
    pose = wp.array(_value(wp.transform, rng, N), dtype=wp.transform, device=device)
    twist = wp.array(_value(wp.spatial_vector, rng, N), dtype=wp.spatial_vector, device=device)
    types = [wp.vec3, wp.mat33, wp.mat33, wp.vec3, wp.mat33, wp.spatial_vector, wp.spatial_vector, wp.spatial_vector]
    types += [wp.spatial_matrix] * 3 + [wp.vec3] * 2
    outs = [wp.zeros(N, dtype=t, device=device) for t in types]
    (
        moment,
        rotational,
        held,
        com,
        compact,
        law_apply,
        compact_apply,
        dense_apply,
        law_matrix,
        compact_matrix,
        adjoint_matrix,
        law_action,
        held_action,
    ) = _launch(_inertia_sites, [mass, inertia, pose, twist], outs, device)
    m = mass.numpy()
    test.assertEqual(int(mismatches(rotational, held).sum()), 0)
    test.assertEqual(int(mismatches(moment, m[:, None] * com).sum()), 0)
    # The parallel-axis term: entry by entry here, -m [c]x [c]x there; terms of order |I| + m |c|^2.
    c = pose.numpy()[:, :3].astype(np.float64)
    scale = np.abs(inertia.numpy()).max(axis=(1, 2)) + m * np.sum(c * c, axis=1)
    test.assertLess(float(term_ulps(rotational, compact, scale[:, None, None]).max()), 4.0)
    test.assertEqual(int(mismatches(law_action, held_action).sum()), 0)
    # Applying the inertia: (m v + w x h, h x v + I w) here, m (v - c x w) and m c x v + I w in the compact form.
    v = np.abs(twist.numpy()).max(axis=1)
    apply_scale = (m * (1.0 + np.linalg.norm(c, axis=1)) + scale) * v
    test.assertLess(float(term_ulps(law_apply, compact_apply, apply_scale[:, None]).max()), 4.0)
    test.assertLess(float(term_ulps(law_apply, dense_apply, apply_scale[:, None]).max()), 4.0)
    # The dense pack: bitwise but for the parallel-axis block, which follows the rotational inertia.
    law_m, compact_m = law_matrix.reshape(N, 6, 6), compact_matrix.reshape(N, 6, 6)
    test.assertEqual(int(mismatches(law_m[:, :, :3], compact_m[:, :, :3]).sum()), 0)
    test.assertEqual(int(mismatches(law_m[:, :3, 3:], compact_m[:, :3, 3:]).sum()), 0)
    # The adjoint form sums 6x6 triple products: terms of the size of the matrix's largest entry.
    matrix_scale = np.abs(law_m).max(axis=(1, 2))
    test.assertLess(float(term_ulps(law_matrix, adjoint_matrix, matrix_scale[:, None, None]).max()), 8.0)


def test_joint_sites(test, device):
    """Joint transform and motion subspace: jcalc's and the scalar branches' one-axis copies."""
    rng = np.random.default_rng(5)
    kinds = [
        (int(JointType.PRISMATIC), 1, 0),
        (int(JointType.REVOLUTE), 0, 1),
        (int(JointType.BALL), 0, 3),
        (int(JointType.FIXED), 0, 0),
        (int(JointType.FREE), 3, 3),
        (int(JointType.D6), 2, 1),
        (int(JointType.D6), 1, 2),
        (int(JointType.D6), 3, 3),
    ]
    choice = rng.integers(0, len(kinds), N)
    types = np.array([kinds[k][0] for k in choice], np.int32)
    lin = np.array([kinds[k][1] for k in choice], np.int32)
    ang = np.array([kinds[k][2] for k in choice], np.int32)
    axes = rng.standard_normal((N * 6, 3))
    axes = (axes / np.linalg.norm(axes, axis=1, keepdims=True)).astype(np.float32)
    q = rng.uniform(-1.5, 1.5, (N, 7)).astype(np.float32)
    quaternion_joints = np.isin(types, [int(JointType.BALL), int(JointType.FREE)])
    ball = types == int(JointType.BALL)
    q[ball, :4] = _quaternions(rng, int(ball.sum()))
    free = types == int(JointType.FREE)
    q[free, 3:7] = _quaternions(rng, int(free.sum()))
    del quaternion_joints
    arrays = [
        wp.array(types, dtype=int, device=device),
        wp.array(axes, dtype=wp.vec3, device=device),
        wp.array(q.reshape(-1), dtype=float, device=device),
        wp.array(rng.uniform(-1, 1, N * 6).astype(np.float32), dtype=float, device=device),
        wp.array(lin, dtype=int, device=device),
        wp.array(ang, dtype=int, device=device),
        wp.array(_value(wp.transform, rng, N), dtype=wp.transform, device=device),
        wp.array(_value(wp.transform, rng, N), dtype=wp.transform, device=device),
    ]
    outs = [
        wp.zeros(N, dtype=wp.transform, device=device),
        wp.zeros(N, dtype=wp.transform, device=device),
        wp.zeros((N, 6), dtype=wp.spatial_vector, device=device),
        wp.zeros(N * 6, dtype=wp.spatial_vector, device=device),
        wp.zeros(N, dtype=wp.transform, device=device),
        wp.zeros(N, dtype=wp.transform, device=device),
        wp.zeros(N, dtype=wp.spatial_vector, device=device),
        wp.zeros(N, dtype=wp.spatial_vector, device=device),
    ]
    law_t, site_t, law_axes, site_axes, law_leaf, scalar_leaf, law_leaf_axis, scalar_leaf_axis = _launch(
        _joint_sites, arrays, outs, device
    )
    test.assertEqual(int(mismatches(law_t, site_t).sum()), 0)
    site_axes = site_axes.reshape(N, 6, 6)
    axis_joints = ~np.isin(types, [int(JointType.FREE), int(JointType.FIXED)])
    for i in np.flatnonzero(axis_joints)[:2048]:
        count = lin[i] + ang[i]
        if types[i] == int(JointType.BALL):
            # A ball joint's columns are the anchor's world axes, which jcalc_motion forms from unit twists.
            continue
        test.assertEqual(int(mismatches(law_axes[i, :count], site_axes[i, :count]).sum()), 0)
    scalar = np.isin(types, [int(JointType.PRISMATIC), int(JointType.REVOLUTE)])
    if wp.get_device(device).is_cuda:
        # The same expression, inlined through the law instead of the scalar branch's function: NVCC groups the
        # composed pose's multiply-adds differently in about a fifth of the samples (CPU code is identical).
        scale = np.abs(scalar_leaf[scalar]).max(axis=1, keepdims=True)
        test.assertLess(float(term_ulps(law_leaf[scalar], scalar_leaf[scalar], scale).max()), 4.0)
    else:
        test.assertEqual(int(mismatches(law_leaf[scalar], scalar_leaf[scalar]).sum()), 0)
    test.assertEqual(int(mismatches(law_leaf_axis[scalar], scalar_leaf_axis[scalar], signed_zeros=False).sum()), 0)


def test_cached_force_shift(test, device):
    """The held predictor's native wrench shift (``cached_force.shift_wrench``) is the law's arithmetic."""
    if not wp.get_device(device).is_cuda:
        return
    rng = np.random.default_rng(6)
    snippet = cuda_native(
        "    wp::spatial_vector out;\n    float* o=reinterpret_cast<float*>(&out);\n"
        + cached_force.shift_wrench("wrench", "offset", "o[k]", "o[3+k]")
        + "    return out;\n#else\n    return wp::spatial_vector();\n#endif\n"
    )

    @wp.func_native(snippet)
    def shift(wrench: wp.spatial_vector, offset: wp.vec3) -> wp.spatial_vector: ...

    @wp.kernel(enable_backward=False)
    def run(
        wrench: wp.array[wp.spatial_vector],
        offset: wp.array[wp.vec3],
        law: wp.array[wp.spatial_vector],
        site: wp.array[wp.spatial_vector],
    ):
        i = wp.tid()
        law[i] = spatial_laws.wrench_shift(wrench[i], offset[i])
        site[i] = shift(wrench[i], offset[i])

    wrench = wp.array(_value(wp.spatial_vector, rng, N), dtype=wp.spatial_vector, device=device)
    offset = wp.array(_value(wp.vec3, rng, N), dtype=wp.vec3, device=device)
    outs = [wp.zeros(N, dtype=wp.spatial_vector, device=device) for _ in range(2)]
    law, site = _launch(run, [wrench, offset], outs, device)
    test.assertEqual(int(mismatches(law, site).sum()), 0)


class TestSpatialLaws(unittest.TestCase):
    pass


for _function in (
    test_twins_agree,
    test_compose_sites,
    test_twist_sites,
    test_wrench_sites,
    test_inertia_sites,
    test_joint_sites,
    test_cached_force_shift,
):
    add_function_test(TestSpatialLaws, _function.__name__, _function, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main(verbosity=2)
