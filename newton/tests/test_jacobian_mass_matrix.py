# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for eval_jacobian() and eval_mass_matrix() functions."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices

_SENTINEL = np.float32(-7.25)
"""Stale buffer value that no Jacobian or mass-matrix entry of the fixtures takes."""


def _build_translated_prismatic_chain(device):
    """Build a two-link revolute/prismatic chain with translated joint frames.

    Args:
        device: Device on which to finalize the model.

    Returns:
        Tuple containing the finalized model, base body index, and slider body index.
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Y)

    base = builder.add_link(mass=1.5)
    slider = builder.add_link(mass=0.9)

    builder.add_shape_box(base, hx=0.2, hy=0.1, hz=0.1)
    builder.add_shape_box(slider, hx=0.15, hy=0.1, hz=0.08)

    builder.body_com[base] = wp.vec3(0.2, 0.0, 0.0)
    builder.body_com[slider] = wp.vec3(0.35, 0.0, -0.1)

    j0 = builder.add_joint_revolute(
        parent=-1,
        child=base,
        axis=newton.Axis.Z,
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
    )
    j1 = builder.add_joint_prismatic(
        parent=base,
        child=slider,
        axis=newton.Axis.X,
        parent_xform=wp.transform(wp.vec3(1.0, 0.0, 0.4), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.2, 0.0, -0.15), wp.quat_identity()),
    )
    builder.add_articulation([j0, j1], label="translated_slider")

    return builder.finalize(device=device), base, slider


def _build_free_body_with_com(device):
    """Build a single free body with a nonzero center-of-mass offset.

    Args:
        device: Device on which to finalize the model.

    Returns:
        Tuple containing the finalized model and body index.
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Y)

    body = builder.add_body(
        xform=wp.transform(
            wp.vec3(1.0, -0.5, 2.0),
            wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.35),
        ),
        mass=1.7,
    )
    builder.add_shape_box(body, hx=0.2, hy=0.1, hz=0.15)
    builder.body_com[body] = wp.vec3(0.4, -0.15, 0.2)

    return builder.finalize(device=device), body


def _build_descendant_free_with_rotated_parent(device):
    """Build a free child body under a rotated fixed parent.

    Args:
        device: Device on which to finalize the model.

    Returns:
        Tuple containing the finalized model, base body index, and child body index.
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Y)

    base = builder.add_link(is_kinematic=True, mass=1.0)
    child = builder.add_link(mass=1.8)

    builder.add_shape_box(base, hx=0.1, hy=0.1, hz=0.1)
    builder.add_shape_box(child, hx=0.18, hy=0.12, hz=0.09)
    builder.body_com[child] = wp.vec3(0.25, -0.1, 0.2)

    j0 = builder.add_joint_fixed(
        parent=-1,
        child=base,
        parent_xform=wp.transform(
            wp.vec3(),
            wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi * 0.5),
        ),
        child_xform=wp.transform_identity(),
    )
    j1 = builder.add_joint_free(
        parent=base,
        child=child,
        child_xform=wp.transform(
            wp.vec3(0.15, 0.0, -0.1),
            wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.35),
        ),
    )
    builder.add_articulation([j0, j1], label="descendant_free")

    return builder.finalize(device=device), base, child


def _build_d6_three_angular(device):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Y)

    cfg = newton.ModelBuilder.JointDofConfig.create_unlimited
    child = builder.add_link(mass=1.0)
    builder.add_shape_box(child, hx=0.2, hy=0.1, hz=0.15)
    builder.body_com[child] = wp.vec3(0.3, -0.1, 0.2)

    j = builder.add_joint_d6(
        parent=-1,
        child=child,
        angular_axes=[
            cfg(axis=newton.Axis.X),
            cfg(axis=newton.Axis.Y),
            cfg(axis=newton.Axis.Z),
        ],
    )
    builder.add_articulation([j], label="d6_three_angular")

    return builder.finalize(device=device), child


def _offset(x, y, z):
    """Return a translation-only transform."""
    return wp.transform(wp.vec3(x, y, z), wp.quat_identity())


def _build_mixed_articulations(device, world_count=3):
    """Build replicated worlds of three articulations whose Jacobian blocks all carry padding.

    Each world holds a fixed-base chain (4 links, 6 DOFs: revolute, prismatic, ball,
    revolute), a fan of six one-DOF leaves on a fixed root (7 links, 6 DOFs) and a
    floating chain (2 links, 7 DOFs). The fan sets the link count and the floating chain
    the DOF count, so every articulation's block has padded rows or columns. Joint
    coordinates are moved away from the reference pose before forward kinematics.

    Args:
        device: Device on which to finalize the model.
        world_count: Number of replicated worlds.

    Returns:
        Tuple containing the finalized model and the posed state.
    """
    inertia = wp.mat33(np.eye(3, dtype=np.float32) * 0.01)
    world = newton.ModelBuilder()

    links = [world.add_link(mass=1.0 + 0.2 * i, com=wp.vec3(0.05 * i, -0.03, 0.02), inertia=inertia) for i in range(4)]
    chain = [
        world.add_joint_revolute(-1, links[0], axis=newton.Axis.Z, parent_xform=_offset(0.1, 0.0, 0.2)),
        world.add_joint_prismatic(links[0], links[1], axis=newton.Axis.X, parent_xform=_offset(0.3, 0.0, 0.0)),
        world.add_joint_ball(links[1], links[2], parent_xform=_offset(0.0, 0.25, 0.0)),
        world.add_joint_revolute(links[2], links[3], axis=newton.Axis.Y, parent_xform=_offset(0.2, 0.0, 0.1)),
    ]
    world.add_articulation(chain, label="chain")

    root = world.add_link(mass=2.0, inertia=inertia)
    fan = [world.add_joint_fixed(-1, root, parent_xform=_offset(1.0, 0.0, 0.0))]
    for i in range(6):
        leaf = world.add_link(mass=0.3, com=wp.vec3(0.0, 0.01 * i, 0.02), inertia=inertia)
        add_joint, axis = (
            (world.add_joint_prismatic, newton.Axis.Z) if i % 2 == 0 else (world.add_joint_revolute, newton.Axis.X)
        )
        fan.append(add_joint(root, leaf, axis=axis, parent_xform=_offset(0.1 * i, 0.05, 0.0)))
    world.add_articulation(fan, label="fan")

    base = world.add_link(mass=1.5, com=wp.vec3(0.02, 0.0, -0.01), inertia=inertia)
    tip = world.add_link(mass=0.7, com=wp.vec3(0.1, 0.0, 0.0), inertia=inertia)
    floater = [
        world.add_joint_free(base, parent_xform=_offset(0.0, 1.0, 0.5)),
        world.add_joint_revolute(base, tip, axis=newton.Axis.Y, parent_xform=_offset(0.2, 0.0, 0.0)),
    ]
    world.add_articulation(floater, label="floater")

    scene = newton.ModelBuilder()
    scene.replicate(world, world_count=world_count)
    model = scene.finalize(device=device)

    state = model.state()
    q = state.joint_q.numpy()
    q_start = model.joint_q_start.numpy()
    rng = np.random.default_rng(7)
    for joint, joint_type in enumerate(model.joint_type.numpy()):
        start = q_start[joint]
        if joint_type in (newton.JointType.REVOLUTE, newton.JointType.PRISMATIC):
            q[start] = rng.uniform(-0.8, 0.8)
        elif joint_type in (newton.JointType.BALL, newton.JointType.FREE):
            if joint_type == newton.JointType.FREE:
                q[start : start + 3] = rng.uniform(-0.5, 0.5, 3)
                start += 3
            quat = rng.normal(size=4)
            q[start : start + 4] = quat / np.linalg.norm(quat)
    state.joint_q.assign(q)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return model, state


def _build_fans(device, leaf_count, world_count=4):
    """Build replicated worlds of one fan of one-DOF leaves on a fixed root.

    The leaves alternate between prismatic and revolute joints. Each Jacobian block
    holds ``6 * (leaf_count + 1) * leaf_count`` entries.

    Args:
        device: Device on which to finalize the model.
        leaf_count: Number of leaves.
        world_count: Number of replicated worlds.

    Returns:
        Tuple containing the finalized model and the state after forward kinematics.
    """
    inertia = wp.mat33(np.eye(3, dtype=np.float32) * 0.01)
    world = newton.ModelBuilder()
    root = world.add_link(mass=2.0, inertia=inertia)
    fan = [world.add_joint_fixed(-1, root)]
    for i in range(leaf_count):
        leaf = world.add_link(mass=0.3, com=wp.vec3(0.0, 0.01 * i, 0.02), inertia=inertia)
        add_joint, axis = (
            (world.add_joint_prismatic, newton.Axis.Z) if i % 2 == 0 else (world.add_joint_revolute, newton.Axis.X)
        )
        fan.append(add_joint(root, leaf, axis=axis, parent_xform=_offset(0.1 * i, 0.05, 0.0)))
    world.add_articulation(fan, label="fan")

    scene = newton.ModelBuilder()
    scene.replicate(world, world_count=world_count)
    model = scene.finalize(device=device)
    state = model.state()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return model, state


def _mixed_selections(model):
    """Return named articulation selections over :func:`_build_mixed_articulations` as boolean arrays."""
    labels = np.asarray(model.articulation_label)
    world = model.articulation_world.numpy()
    return {
        "fan": labels == "fan",
        "mixed": ((labels == "chain") & (world == 0)) | ((labels != "chain") & (world == 1)),
        "none": np.zeros(model.articulation_count, dtype=bool),
        "all": np.ones(model.articulation_count, dtype=bool),
    }


def _jacobian_shape(model):
    """Return the model-wide padded Jacobian shape."""
    return (model.articulation_count, model.max_joints_per_articulation * 6, model.max_dofs_per_articulation)


def _filled(shape, value, device):
    """Return a float array of ``shape`` holding ``value`` everywhere."""
    return wp.array(np.full(shape, value, dtype=np.float32), dtype=float, device=device)


def _jacobian_extents(model):
    """Return each articulation's Jacobian extent within its block: (6 times its joint count, its DOF count)."""
    start = model.articulation_start.numpy()
    end = model.articulation_end.numpy()
    qd_start = model.joint_qd_start.numpy()
    return [(6 * (end[i] - start[i]), qd_start[end[i]] - qd_start[start[i]]) for i in range(model.articulation_count)]


def _jacobian_entries(model, selected, shape):
    """Return a Boolean array of ``shape`` that marks the Jacobian entries of the selected articulations."""
    entries = np.zeros(shape, dtype=bool)
    for i, (rows, cols) in enumerate(_jacobian_extents(model)):
        if selected[i]:
            entries[i, :rows, :cols] = True
    return entries


def _bits(array):
    """Return the float32 bit patterns of ``array``, so that comparisons tell ``-0.0`` from ``0.0``."""
    return np.ascontiguousarray(array, dtype=np.float32).view(np.uint32)


def _assert_bits_equal(actual, expected):
    """Assert that two float32 arrays hold the same bits, so that ``-0.0`` does not pass for ``0.0``."""
    np.testing.assert_array_equal(_bits(actual), _bits(expected))


def _assert_masked_jacobian(actual, expected, before, entries):
    """Assert what an eval_jacobian guarantees for a buffer that held ``before``.

    The selected articulations' Jacobians, the ``entries``, equal the unmasked result
    ``expected`` bitwise, and every other entry still holds ``before``, bit for bit.

    Args:
        actual: Jacobian after the call, shape [articulation_count, rows, columns].
        expected: Unmasked Jacobian of the same state.
        before: Buffer contents before the call, broadcastable to ``actual``.
        entries: Boolean array of ``actual``'s shape marking the selected Jacobians.
    """
    _assert_bits_equal(actual[entries], expected[entries])
    before = np.broadcast_to(np.asarray(before, dtype=np.float32), actual.shape)
    _assert_bits_equal(actual[~entries], before[~entries])


def _record_clears():
    """Patch whole-array clears and return the patches and the list of cleared array pointers.

    ``wp.zeros`` clears through :meth:`warp.array.zero_`, so zero-initialized
    allocations are recorded as well.
    """
    cleared = []
    zero_, fill_ = wp.array.zero_, wp.array.fill_

    def record_zero(array):
        cleared.append((array.ptr, array.shape, array.dtype))
        zero_(array)

    def record_fill(array, value):
        cleared.append((array.ptr, array.shape, array.dtype))
        fill_(array, value)

    patches = (mock.patch.object(wp.array, "zero_", record_zero), mock.patch.object(wp.array, "fill_", record_fill))
    return patches, cleared


def _nan_filled_float_allocations():
    """Return a patch under which every float array from ``wp.empty`` starts as NaN.

    Uninitialized memory may hold anything; NaN makes any read of an entry that was
    never written show up in the results.
    """
    empty = wp.empty

    def nan_empty(*args, **kwargs):
        array = empty(*args, **kwargs)
        if array.dtype == wp.float32:
            array.fill_(np.nan)
        return array

    return mock.patch.object(wp, "empty", nan_empty)


def _diagonal_inertia(ix, iy, iz):
    """Return a diagonal inertia tensor in the body COM frame."""
    return wp.matrix_from_rows(
        wp.vec3(ix, 0.0, 0.0),
        wp.vec3(0.0, iy, 0.0),
        wp.vec3(0.0, 0.0, iz),
    )


def _floating_base_pendulum_mass_matrix(base_mass, child_mass, length, base_inertia, child_inertia):
    """Analytical mass matrix for a free base with one revolute pendulum child.

    The configuration is evaluated at identity base pose and zero revolute angle.
    The generalized velocity order is the public Newton order:
    free translation, free angular velocity, then the revolute velocity.
    """
    child_twist_from_qd = np.zeros((6, 7), dtype=np.float64)

    # Child COM linear velocity from free-base translation.
    child_twist_from_qd[0, 0] = 1.0
    child_twist_from_qd[1, 1] = 1.0
    child_twist_from_qd[2, 2] = 1.0

    # Child COM linear velocity from angular velocity crossed with r=(length, 0, 0).
    child_twist_from_qd[2, 4] = -length
    child_twist_from_qd[1, 5] = length
    child_twist_from_qd[1, 6] = length

    # Child angular velocity is base angular velocity plus the revolute z velocity.
    child_twist_from_qd[3, 3] = 1.0
    child_twist_from_qd[4, 4] = 1.0
    child_twist_from_qd[5, 5] = 1.0
    child_twist_from_qd[5, 6] = 1.0

    child_spatial_inertia = np.diag([child_mass, child_mass, child_mass, *child_inertia])
    expected = child_twist_from_qd.T @ child_spatial_inertia @ child_twist_from_qd

    expected[0, 0] += base_mass
    expected[1, 1] += base_mass
    expected[2, 2] += base_mass
    expected[3, 3] += base_inertia[0]
    expected[4, 4] += base_inertia[1]
    expected[5, 5] += base_inertia[2]

    return expected


def _kinetic_energy_from_body_twists(model, state, bodies):
    """Compute total body kinetic energy from world-space twists.

    Args:
        model: Model providing body masses and inertia tensors.
        state: State providing body poses and velocities.
        bodies: Body indices to include in the energy sum.

    Returns:
        Total kinetic energy for the selected bodies.
    """
    body_q = state.body_q.numpy()
    body_qd = state.body_qd.numpy()
    body_inertia = model.body_inertia.numpy()
    body_mass = model.body_mass.numpy()

    kinetic = 0.0
    for body in bodies:
        quat = wp.quat(
            float(body_q[body, 3]),
            float(body_q[body, 4]),
            float(body_q[body, 5]),
            float(body_q[body, 6]),
        )
        R = np.array(wp.quat_to_matrix(quat), dtype=np.float64).reshape(3, 3)
        I_world = R @ body_inertia[body].astype(np.float64) @ R.T
        v_com = body_qd[body, :3].astype(np.float64)
        omega = body_qd[body, 3:6].astype(np.float64)
        kinetic += 0.5 * float(body_mass[body]) * float(v_com @ v_com)
        kinetic += 0.5 * float(omega @ (I_world @ omega))

    return kinetic


def test_jacobian_simple_pendulum(test, device):
    """Test Jacobian computation for a simple 2-link pendulum."""
    builder = newton.ModelBuilder()

    # Create a 2-link pendulum
    b1 = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
    )
    b2 = builder.add_link(
        xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
    )

    j1 = builder.add_joint_revolute(
        parent=-1,
        child=b1,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
    )
    j2 = builder.add_joint_revolute(
        parent=b1,
        child=b2,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
    )
    builder.add_articulation([j1, j2], label="pendulum")

    model = builder.finalize(device=device)
    state = model.state()

    # Compute FK first
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    # Compute Jacobian (convenience pattern - let function allocate)
    J = newton.eval_jacobian(model, state)

    test.assertIsNotNone(J)
    test.assertEqual(J.shape[0], model.articulation_count)
    test.assertEqual(J.shape[1], model.max_joints_per_articulation * 6)
    test.assertEqual(J.shape[2], model.max_dofs_per_articulation)

    J_np = J.numpy()

    # For a revolute joint about Z-axis at identity:
    # Motion subspace should be [0, 0, 0, 0, 0, 1] (linear velocity from angular motion)
    # At identity configuration, first joint affects both links
    # Check that Jacobian has non-zero entries for angular velocity (index 5)
    test.assertNotEqual(J_np[0, 5, 0], 0.0)  # First link, angular z, first dof
    test.assertNotEqual(J_np[0, 11, 0], 0.0)  # Second link, angular z, first dof
    test.assertNotEqual(J_np[0, 11, 1], 0.0)  # Second link, angular z, second dof


def test_jacobian_numerical_verification(test, device):
    """Verify Jacobian shape and basic properties."""
    builder = newton.ModelBuilder()

    # Create a simple pendulum
    b1 = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
    )

    j1 = builder.add_joint_revolute(
        parent=-1,
        child=b1,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
    )
    builder.add_articulation([j1], label="pendulum")

    model = builder.finalize(device=device)
    state = model.state()

    # Set a non-zero joint angle
    joint_q = state.joint_q.numpy()
    joint_q[0] = 0.5
    state.joint_q.assign(joint_q)

    # Compute FK
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    # Compute Jacobian (convenience pattern)
    J = newton.eval_jacobian(model, state)
    J_np = J.numpy()

    # Verify shape
    test.assertEqual(J_np.shape[0], 1)  # One articulation
    test.assertEqual(J_np.shape[1], 6)  # One link * 6
    test.assertEqual(J_np.shape[2], 1)  # One DOF

    # For revolute joint about z-axis, the angular z component (index 5) should be 1.0
    test.assertAlmostEqual(J_np[0, 5, 0], 1.0, places=5)


def test_mass_matrix_symmetry(test, device):
    """Test that mass matrix is symmetric."""
    builder = newton.ModelBuilder()

    # Create a 2-link pendulum with different masses
    b1 = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
    )
    b2 = builder.add_link(
        xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()),
        mass=2.0,
    )

    j1 = builder.add_joint_revolute(
        parent=-1,
        child=b1,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
    )
    j2 = builder.add_joint_revolute(
        parent=b1,
        child=b2,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
    )
    builder.add_articulation([j1, j2], label="pendulum")

    model = builder.finalize(device=device)
    state = model.state()

    # Set some joint angles
    joint_q = state.joint_q.numpy()
    joint_q[0] = 0.3
    joint_q[1] = 0.5
    state.joint_q.assign(joint_q)

    # Compute FK first
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    # Compute mass matrix (convenience pattern)
    H = newton.eval_mass_matrix(model, state)

    test.assertIsNotNone(H)
    test.assertEqual(H.shape[0], model.articulation_count)
    test.assertEqual(H.shape[1], model.max_dofs_per_articulation)
    test.assertEqual(H.shape[2], model.max_dofs_per_articulation)

    H_np = H.numpy()

    # Check symmetry for the valid portion of the matrix
    num_dofs = 2
    H_valid = H_np[0, :num_dofs, :num_dofs]

    np.testing.assert_allclose(H_valid, H_valid.T, rtol=1e-5, atol=1e-6)


def test_mass_matrix_positive_definite(test, device):
    """Test that mass matrix is positive definite."""
    builder = newton.ModelBuilder()

    # Create a pendulum with non-trivial inertia
    b1 = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
    )
    builder.add_shape_box(body=b1, hx=0.1, hy=0.1, hz=0.1)

    j1 = builder.add_joint_revolute(
        parent=-1,
        child=b1,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
    )
    builder.add_articulation([j1], label="pendulum")

    model = builder.finalize(device=device)
    state = model.state()

    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    # Compute mass matrix (convenience pattern)
    H = newton.eval_mass_matrix(model, state)
    H_np = H.numpy()

    # For a single DOF, the mass matrix should be a positive scalar
    test.assertGreater(H_np[0, 0, 0], 0.0)


def test_fixed_base_simple_pendulum_mass_matrix_matches_analytical(test, device):
    """Test a fixed-base simple pendulum mass matrix against the closed-form scalar inertia."""
    mass = 2.0
    length = 0.75
    inertia_zz = 0.2

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)
    body = builder.add_link(mass=mass, inertia=_diagonal_inertia(0.1, 0.15, inertia_zz))
    joint = builder.add_joint_revolute(
        parent=-1,
        child=body,
        axis=newton.Axis.Z,
        child_xform=wp.transform(wp.vec3(-length, 0.0, 0.0), wp.quat_identity()),
    )
    builder.add_articulation([joint], label="fixed_base_pendulum")

    model = builder.finalize(device=device)
    state = model.state()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    H = newton.eval_mass_matrix(model, state).numpy()
    expected = inertia_zz + mass * length**2

    test.assertEqual(H.shape, (1, 1, 1))
    np.testing.assert_allclose(H[0, 0, 0], expected, rtol=1.0e-6, atol=1.0e-6)


def test_floating_base_simple_pendulum_mass_matrix_matches_analytical(test, device):
    """Test a free-base pendulum mass matrix against its closed-form 7-DOF inertia."""
    base_mass = 3.0
    child_mass = 2.0
    length = 0.6
    base_inertia = (0.4, 0.5, 0.6)
    child_inertia = (0.2, 0.25, 0.3)

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)
    base = builder.add_link(mass=base_mass, inertia=_diagonal_inertia(*base_inertia))
    child = builder.add_link(mass=child_mass, inertia=_diagonal_inertia(*child_inertia))

    free_joint = builder.add_joint_free(child=base)
    revolute_joint = builder.add_joint_revolute(
        parent=base,
        child=child,
        axis=newton.Axis.Z,
        child_xform=wp.transform(wp.vec3(-length, 0.0, 0.0), wp.quat_identity()),
    )
    builder.add_articulation([free_joint, revolute_joint], label="floating_base_pendulum")

    model = builder.finalize(device=device)
    state = model.state()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    H = newton.eval_mass_matrix(model, state).numpy()[0, : model.joint_dof_count, : model.joint_dof_count]
    expected = _floating_base_pendulum_mass_matrix(base_mass, child_mass, length, base_inertia, child_inertia)

    test.assertEqual(H.shape, (7, 7))
    np.testing.assert_allclose(H, expected, rtol=1.0e-6, atol=1.0e-6)


def test_jacobian_multiple_articulations(test, device):
    """Test Jacobian computation with multiple articulations."""
    builder = newton.ModelBuilder()

    # Create 3 independent pendulums
    for i in range(3):
        b1 = builder.add_link(
            xform=wp.transform(wp.vec3(i * 2.0, 0.0, 0.0), wp.quat_identity()),
            mass=1.0,
        )

        j1 = builder.add_joint_revolute(
            parent=-1,
            child=b1,
            axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform(wp.vec3(i * 2.0, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        )
        builder.add_articulation([j1], label=f"pendulum_{i}")

    model = builder.finalize(device=device)
    state = model.state()

    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    # Compute Jacobian (convenience pattern)
    J = newton.eval_jacobian(model, state)

    test.assertEqual(J.shape[0], 3)  # 3 articulations
    test.assertEqual(model.articulation_count, 3)


def test_jacobian_with_mask(test, device):
    """Test Jacobian computation with articulation mask."""
    builder = newton.ModelBuilder()

    # Create 2 pendulums
    for i in range(2):
        b1 = builder.add_link(
            xform=wp.transform(wp.vec3(i * 2.0, 0.0, 0.0), wp.quat_identity()),
            mass=1.0,
        )

        j1 = builder.add_joint_revolute(
            parent=-1,
            child=b1,
            axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform(wp.vec3(i * 2.0, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        )
        builder.add_articulation([j1], label=f"pendulum_{i}")

    model = builder.finalize(device=device)
    state = model.state()

    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    # Compute with mask - only first articulation (performance pattern - pre-allocate)
    J = wp.zeros(
        (model.articulation_count, model.max_joints_per_articulation * 6, model.max_dofs_per_articulation),
        dtype=float,
        device=device,
    )
    mask = wp.array([True, False], dtype=bool, device=device)
    J_returned = newton.eval_jacobian(model, state, J, mask=mask)

    # Verify same array is returned
    test.assertIs(J_returned, J)

    J_np = J.numpy()

    # First articulation should have non-zero Jacobian
    test.assertNotEqual(np.abs(J_np[0]).max(), 0.0)

    # The second articulation is masked out: a masked eval_jacobian leaves its block
    # unchanged, and the buffer was zero-initialized.
    test.assertEqual(np.abs(J_np[1]).max(), 0.0)


def test_mass_matrix_with_mask(test, device):
    """Test mass matrix computation with articulation mask."""
    builder = newton.ModelBuilder()

    # Create 2 pendulums
    for i in range(2):
        b1 = builder.add_link(
            xform=wp.transform(wp.vec3(i * 2.0, 0.0, 0.0), wp.quat_identity()),
            mass=1.0 + i,  # Different masses
        )
        builder.add_shape_box(body=b1, hx=0.1, hy=0.1, hz=0.1)

        j1 = builder.add_joint_revolute(
            parent=-1,
            child=b1,
            axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform(wp.vec3(i * 2.0, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        )
        builder.add_articulation([j1], label=f"pendulum_{i}")

    model = builder.finalize(device=device)
    state = model.state()

    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    # Compute with mask - only second articulation (performance pattern - pre-allocate)
    H = wp.zeros(
        (model.articulation_count, model.max_dofs_per_articulation, model.max_dofs_per_articulation),
        dtype=float,
        device=device,
    )
    mask = wp.array([False, True], dtype=bool, device=device)
    H_returned = newton.eval_mass_matrix(model, state, H, mask=mask)

    # Verify same array is returned
    test.assertIs(H_returned, H)

    H_np = H.numpy()

    # First articulation should be zero (masked out)
    test.assertEqual(H_np[0, 0, 0], 0.0)

    # Second articulation should have non-zero mass matrix
    test.assertNotEqual(H_np[1, 0, 0], 0.0)


def test_jacobian_writes_only_selected_jacobians(test, device):
    """Overwrite each selected articulation's Jacobian, links by DOFs, and nothing else in ``J``.

    Every selected Jacobian, zeros of DOFs that do not move a link included, must equal
    the unmasked result bitwise whatever the buffer held before. The rows and columns
    past each articulation's links and DOFs, and the blocks of unselected articulations,
    keep the caller's values. Without a mask every articulation is selected.
    """
    model, state = _build_mixed_articulations(device)
    J_ref = newton.eval_jacobian(model, state).numpy()
    masks = {name: wp.array(selected, dtype=bool, device=device) for name, selected in _mixed_selections(model).items()}
    masks["no mask"] = None

    for name, mask in masks.items():
        with test.subTest(selection=name):
            J = _filled(J_ref.shape, _SENTINEL, device)
            test.assertIs(newton.eval_jacobian(model, state, J, mask=mask), J)

            selected = np.ones(model.articulation_count, dtype=bool) if mask is None else mask.numpy()
            _assert_masked_jacobian(J.numpy(), J_ref, _SENTINEL, _jacobian_entries(model, selected, J_ref.shape))


def test_jacobian_mask_without_buffer_zeroes_unselected(test, device):
    """Return zero blocks for unselected articulations when eval_jacobian allocates J."""
    model, state = _build_mixed_articulations(device)
    J_ref = newton.eval_jacobian(model, state).numpy()
    selected = _mixed_selections(model)["mixed"]

    J = newton.eval_jacobian(model, state, mask=wp.array(selected, dtype=bool, device=device)).numpy()

    _assert_bits_equal(J[selected], J_ref[selected])
    _assert_bits_equal(J[~selected], np.zeros_like(J_ref[~selected]))


def _view_slots(view):
    """Return the Model articulation index of each slot of ``view``'s view-shaped outputs (world-major)."""
    return view.articulation_ids.numpy().reshape(-1)


def _view_selections():
    """Return named per-world masks of the 3-world mixed model: every world, and world 1 only."""
    return {"view": None, "world 1": np.array([False, True, False])}


def test_articulation_view_jacobian_writes_view_slots(test, device):
    """Write each selected view articulation's Jacobian into its slot of a view-shaped J, and nothing else.

    Slot ``i`` holds the Jacobian of the view's ``i``-th articulation, bitwise the entries of the
    model-level result for it. Slots outside a per-world mask, and entries past an articulation's links
    and DOFs (a view of every articulation pads to its largest), keep the caller's values.
    """
    model, state = _build_mixed_articulations(device)
    J_ref = newton.eval_jacobian(model, state).numpy()
    extents = _jacobian_extents(model)

    for pattern in ("fan", "*"):
        view = newton.selection.ArticulationView(model, pattern, verbose=False)
        shape = (view.count, 6 * view.max_joints_per_articulation, view.max_dofs_per_articulation)
        for name, world_mask in _view_selections().items():
            with test.subTest(pattern=pattern, selection=name):
                J = _filled(shape, _SENTINEL, device)
                view_mask = None if world_mask is None else wp.array(world_mask, dtype=bool, device=device)
                test.assertIs(view.eval_jacobian(state, J, mask=view_mask), J)

                selected = view.get_model_articulation_mask(mask=view_mask).numpy()
                expected = np.full(shape, _SENTINEL, dtype=np.float32)
                for slot, articulation in enumerate(_view_slots(view)):
                    if selected[articulation]:
                        rows, cols = extents[articulation]
                        expected[slot, :rows, :cols] = J_ref[articulation, :rows, :cols]
                _assert_bits_equal(J.numpy(), expected)


def test_articulation_view_mass_matrix_writes_view_slots(test, device):
    """Write each selected view articulation's mass matrix into its slot of a view-shaped H, each entry once.

    Slot ``i`` holds the mass matrix of the view's ``i``-th articulation, bitwise the model-level result
    for it, with the Jacobian computed internally or supplied from :meth:`eval_jacobian`. Slots outside
    a per-world mask and entries past an articulation's DOFs keep the caller's values: nothing is cleared.
    """
    model, state = _build_mixed_articulations(device)
    H_ref = newton.eval_mass_matrix(model, state).numpy()
    extents = _jacobian_extents(model)

    for pattern in ("fan", "*"):
        view = newton.selection.ArticulationView(model, pattern, verbose=False)
        shape = (view.count, view.max_dofs_per_articulation, view.max_dofs_per_articulation)
        for name, world_mask in _view_selections().items():
            view_mask = None if world_mask is None else wp.array(world_mask, dtype=bool, device=device)
            J = view.eval_jacobian(state, mask=view_mask)
            for source, supplied in (("internal", None), ("supplied", J)):
                with test.subTest(pattern=pattern, selection=name, jacobian=source):
                    H = _filled(shape, _SENTINEL, device)
                    test.assertIs(view.eval_mass_matrix(state, H, J=supplied, mask=view_mask), H)

                    selected = view.get_model_articulation_mask(mask=view_mask).numpy()
                    expected = np.full(shape, _SENTINEL, dtype=np.float32)
                    for slot, articulation in enumerate(_view_slots(view)):
                        if selected[articulation]:
                            dofs = extents[articulation][1]
                            expected[slot, :dofs, :dofs] = H_ref[articulation, :dofs, :dofs]
                    _assert_bits_equal(H.numpy(), expected)


def test_articulation_view_rejects_other_shapes(test, device):
    """Refuse outputs and scratch that do not have the view's shapes, such as a model-wide Jacobian."""
    model, state = _build_mixed_articulations(device)
    view = newton.selection.ArticulationView(model, "fan", verbose=False)
    joints, dofs = view.max_joints_per_articulation, view.max_dofs_per_articulation
    with test.assertRaises(ValueError):
        view.eval_jacobian(state, wp.zeros(_jacobian_shape(model), dtype=float, device=device))
    with test.assertRaises(ValueError):
        view.eval_mass_matrix(state, wp.zeros((view.count, dofs + 1, dofs + 1), dtype=float, device=device))
    with test.assertRaises(ValueError):
        view.eval_mass_matrix(state, body_I_s=wp.zeros(view.count * joints, dtype=wp.spatial_matrix, device=device))


def test_articulation_view_of_every_articulation_matches_unmasked(test, device):
    """Evaluate a view of every articulation like the unmasked functions, without clearing a Jacobian.

    Without a mask such a view selects everything, so the unmasked paths give the same
    values. The caller's Jacobian and the internal one of eval_mass_matrix are written,
    never cleared. A mask still restricts the selection.
    """
    model, state = _build_mixed_articulations(device)
    view = newton.selection.ArticulationView(model, "*", verbose=False)
    test.assertEqual(view.count, model.articulation_count)
    J_ref = newton.eval_jacobian(model, state).numpy()
    H_ref = newton.eval_mass_matrix(model, state).numpy()
    J = _filled(J_ref.shape, _SENTINEL, device)

    patches, cleared = _record_clears()
    with patches[0], patches[1]:
        test.assertIs(view.eval_jacobian(state, J), J)
        H = view.eval_mass_matrix(state)

    test.assertFalse([ptr for ptr, shape, _dtype in cleared if shape == J_ref.shape])
    every = np.ones(model.articulation_count, dtype=bool)
    _assert_masked_jacobian(J.numpy(), J_ref, _SENTINEL, _jacobian_entries(model, every, J_ref.shape))
    _assert_bits_equal(H.numpy(), H_ref)

    J = _filled(J_ref.shape, _SENTINEL, device)
    view.eval_jacobian(state, J, mask=wp.array([False, True, False], dtype=bool, device=device))
    selected = model.articulation_world.numpy() == 1
    _assert_masked_jacobian(J.numpy(), J_ref, _SENTINEL, _jacobian_entries(model, selected, J_ref.shape))


def test_mass_matrix_mask_zeroes_unselected_articulations(test, device):
    """Return exact selected blocks and zeros elsewhere from a masked eval_mass_matrix.

    This is the mask convention that eval_inverse_dynamics_passive documents as
    mirroring eval_mass_matrix. It must hold when H holds stale values and the
    supplied Jacobian, evaluated into a NaN-filled buffer, carries NaN everywhere
    outside the selected articulations' Jacobians.
    """
    model, state = _build_mixed_articulations(device)
    H_ref = newton.eval_mass_matrix(model, state).numpy()

    for name, selected in _mixed_selections(model).items():
        mask = wp.array(selected, dtype=bool, device=device)
        J = _filled(_jacobian_shape(model), np.nan, device)
        newton.eval_jacobian(model, state, J, mask=mask)
        for source, supplied in (("supplied", J), ("internal", None)):
            with test.subTest(selection=name, jacobian=source):
                H = _filled(H_ref.shape, _SENTINEL, device)
                test.assertIs(newton.eval_mass_matrix(model, state, H, J=supplied, mask=mask), H)

                H_np = H.numpy()
                _assert_bits_equal(H_np[selected], H_ref[selected])
                _assert_bits_equal(H_np[~selected], np.zeros_like(H_ref[~selected]))


def test_mass_matrix_scratch_skips_redundant_clears(test, device):
    """Clear eval_mass_matrix's internal scratch only where its kernels read it.

    The internal Jacobian is never cleared: eval_jacobian writes every entry that the
    mass-matrix kernel reads. The spatial-inertia scratch is fully overwritten and
    never zero-filled. H keeps its single clear, which the documented zeros of
    unselected articulations need.
    """
    model, state = _build_mixed_articulations(device)
    mask = wp.array(_mixed_selections(model)["fan"], dtype=bool, device=device)
    J_shape = _jacobian_shape(model)
    H_shape = (model.articulation_count, model.max_dofs_per_articulation, model.max_dofs_per_articulation)

    for name, call_mask in (("masked", mask), ("unmasked", None)):
        patches, cleared = _record_clears()
        with patches[0], patches[1]:
            newton.eval_mass_matrix(model, state, mask=call_mask)

        with test.subTest(call=name):
            test.assertEqual(sum(shape == J_shape for _ptr, shape, _dtype in cleared), 0)
            test.assertEqual(sum(shape == H_shape for _ptr, shape, _dtype in cleared), 1)
            test.assertFalse([dtype for _ptr, _shape, dtype in cleared if dtype == wp.spatial_matrix])


def test_jacobian_mask_accepts_masks_the_launch_converts(test, device):
    """Select the same articulations from masks that only the kernel launch accepts as from an exact mask.

    The launch reads the first ``articulation_count`` entries of a longer Warp mask
    and, on the CPU, takes a NumPy Boolean mask through its array interface. The
    masked mass matrix, whose internal Jacobian goes through the same launch, must
    agree as well.
    """
    model, state = _build_mixed_articulations(device)
    selected = _mixed_selections(model)["mixed"]
    exact = wp.array(selected, dtype=bool, device=device)
    masks = {"longer Warp mask": wp.array(np.append(selected, [True, False]), dtype=bool, device=device)}
    if device.is_cpu:
        masks["NumPy mask"] = selected

    J_ref = _filled(_jacobian_shape(model), _SENTINEL, device)
    newton.eval_jacobian(model, state, J_ref, mask=exact)
    H_ref = newton.eval_mass_matrix(model, state, mask=exact).numpy()
    for name, mask in masks.items():
        with test.subTest(mask=name):
            J = _filled(_jacobian_shape(model), _SENTINEL, device)
            newton.eval_jacobian(model, state, J, mask=mask)
            _assert_bits_equal(J.numpy(), J_ref.numpy())
            _assert_bits_equal(newton.eval_mass_matrix(model, state, mask=mask).numpy(), H_ref)


def test_jacobian_mask_graph_replay_matches_eager(test, device):
    """Write the Jacobians that the mask selects when a captured masked evaluation replays.

    Each replay starts from a reset buffer and a new mask: the selected Jacobians must
    equal the eager unmasked result, and every other entry keeps the reset value.
    """
    model, state = _build_mixed_articulations(device)
    J_ref = newton.eval_jacobian(model, state).numpy()
    selections = _mixed_selections(model)
    mask = wp.array(selections["mixed"], dtype=bool, device=device)
    joint_S_s = wp.zeros(model.joint_dof_count, dtype=wp.spatial_vector, device=device)
    J = _filled(J_ref.shape, _SENTINEL, device)

    # Load the modules eagerly; a capture cannot compile them.
    newton.eval_jacobian(model, state, J, joint_S_s=joint_S_s, mask=mask)
    with wp.ScopedCapture(device) as capture:
        newton.eval_jacobian(model, state, J, joint_S_s=joint_S_s, mask=mask)

    for name in ("mixed", "fan", "all"):
        with test.subTest(selection=name):
            selected = selections[name]
            mask.assign(selected)
            J.assign(np.full(J_ref.shape, _SENTINEL, dtype=np.float32))
            wp.capture_launch(capture.graph)

            _assert_masked_jacobian(J.numpy(), J_ref, _SENTINEL, _jacobian_entries(model, selected, J_ref.shape))


def test_jacobian_mask_graph_replay_fills_strided_view(test, device):
    """Capture a masked eval_jacobian into a non-contiguous view and replay it.

    Warp's APIC capture, which every CPU capture uses and a CUDA capture with
    ``apic=True`` adds, cannot record a fill of a non-contiguous array, and the
    evaluation records none. Each replay starts from a reset buffer and a new mask: the
    selected blocks must equal the eager unmasked result bitwise, the unselected ones
    keep the reset values, and nothing outside the view is written.
    """
    captures = {"default": {}}
    if device.is_cuda:
        captures["apic"] = {"apic": True}
    selections = {"some": np.array([True, False, False, True]), "others": np.array([False, True, True, False])}
    for leaf_count in (6, 18):  # blocks of 252 and 2052 entries, every entry an articulation's Jacobian
        model, state = _build_fans(device, leaf_count)
        J_ref = newton.eval_jacobian(model, state).numpy()
        count, rows, cols = J_ref.shape
        whole = _filled((count, rows + 2, cols + 3), _SENTINEL, device)
        window = (slice(None), slice(1, 1 + rows), slice(2, 2 + cols))
        J = whole[window]
        test.assertFalse(J.is_contiguous)
        mask = wp.array(selections["some"], dtype=bool, device=device)
        joint_S_s = wp.zeros(model.joint_dof_count, dtype=wp.spatial_vector, device=device)
        # Load the modules eagerly; a capture cannot compile them.
        newton.eval_jacobian(model, state, J, joint_S_s=joint_S_s, mask=mask)

        for capture_name, options in captures.items():
            with test.subTest(leaves=leaf_count, capture=capture_name):
                with wp.ScopedCapture(device, **options) as capture:
                    newton.eval_jacobian(model, state, J, joint_S_s=joint_S_s, mask=mask)

                for name, selected in selections.items():
                    with test.subTest(selection=name):
                        mask.assign(selected)
                        whole.assign(np.full(whole.shape, _SENTINEL, dtype=np.float32))
                        wp.capture_launch(capture.graph)

                        expected = np.full(whole.shape, _SENTINEL, dtype=np.float32)
                        expected[window][selected] = J_ref[selected]
                        _assert_bits_equal(whole.numpy(), expected)


def test_mass_matrix_mask_graph_replay_matches_eager(test, device):
    """Replay a captured masked eval_mass_matrix whose internal Jacobian is allocated during the capture.

    New float allocations start as NaN, as reused memory may hold anything, so a
    replay must write every Jacobian entry that the mass-matrix kernel reads.
    """
    if device.is_cuda and not wp.is_mempool_enabled(device):
        test.skipTest("Allocating during CUDA graph capture requires the memory pool")
    model, state = _build_mixed_articulations(device)
    H_ref = newton.eval_mass_matrix(model, state).numpy()
    selections = _mixed_selections(model)
    mask = wp.array(selections["mixed"], dtype=bool, device=device)
    H = _filled(H_ref.shape, _SENTINEL, device)
    body_I_s = wp.empty(model.body_count, dtype=wp.spatial_matrix, device=device)
    joint_S_s = wp.zeros(model.joint_dof_count, dtype=wp.spatial_vector, device=device)

    # Load the modules eagerly; a capture cannot compile them.
    newton.eval_mass_matrix(model, state, H, body_I_s=body_I_s, joint_S_s=joint_S_s, mask=mask)
    with _nan_filled_float_allocations(), wp.ScopedCapture(device) as capture:
        newton.eval_mass_matrix(model, state, H, body_I_s=body_I_s, joint_S_s=joint_S_s, mask=mask)

    for name in ("mixed", "fan"):
        with test.subTest(selection=name):
            selected = selections[name]
            mask.assign(selected)
            H.assign(np.full(H_ref.shape, _SENTINEL, dtype=np.float32))
            wp.capture_launch(capture.graph)

            H_np = H.numpy()
            _assert_bits_equal(H_np[selected], H_ref[selected])
            _assert_bits_equal(H_np[~selected], np.zeros_like(H_ref[~selected]))


def test_prismatic_joint_jacobian(test, device):
    """Test Jacobian for prismatic joint."""
    builder = newton.ModelBuilder()

    b1 = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
    )

    j1 = builder.add_joint_prismatic(
        parent=-1,
        child=b1,
        axis=wp.vec3(1.0, 0.0, 0.0),  # Slide along X
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
    )
    builder.add_articulation([j1], label="slider")

    model = builder.finalize(device=device)
    state = model.state()

    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    # Compute Jacobian (convenience pattern)
    J = newton.eval_jacobian(model, state)
    J_np = J.numpy()

    # For prismatic joint along X, the Jacobian should have:
    # Linear velocity in X direction (index 0)
    test.assertNotEqual(J_np[0, 0, 0], 0.0)
    # Angular velocity should be zero
    test.assertEqual(J_np[0, 3, 0], 0.0)
    test.assertEqual(J_np[0, 4, 0], 0.0)
    test.assertEqual(J_np[0, 5, 0], 0.0)


def test_empty_model(test, device):
    """Test that functions handle empty model gracefully."""
    builder = newton.ModelBuilder()
    model = builder.finalize(device=device)
    state = model.state()

    J = newton.eval_jacobian(model, state)
    H = newton.eval_mass_matrix(model, state)

    test.assertIsNone(J)
    test.assertIsNone(H)


def test_articulation_view_api(test, device):
    """Test Jacobian and mass matrix via ArticulationView API."""
    builder = newton.ModelBuilder()

    # Create 2 pendulums with different keys
    for i, key in enumerate(["robot_a", "robot_b"]):
        b1 = builder.add_link(
            xform=wp.transform(wp.vec3(i * 2.0, 0.0, 0.0), wp.quat_identity()),
            mass=1.0,
        )
        builder.add_shape_box(body=b1, hx=0.1, hy=0.1, hz=0.1)

        j1 = builder.add_joint_revolute(
            parent=-1,
            child=b1,
            axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform(wp.vec3(i * 2.0, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        )
        builder.add_articulation([j1], label=key)

    model = builder.finalize(device=device)
    state = model.state()

    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    J_model = newton.eval_jacobian(model, state).numpy()
    H_model = newton.eval_mass_matrix(model, state).numpy()

    # Create ArticulationView for just robot_a: one pendulum, one joint and one DOF
    view_a = newton.selection.ArticulationView(model, pattern="robot_a")
    test.assertEqual((view_a.max_joints_per_articulation, view_a.max_dofs_per_articulation), (1, 1))

    # Test eval_jacobian via ArticulationView (convenience pattern): view-shaped, robot_a only
    J = view_a.eval_jacobian(state)
    test.assertEqual(J.shape, (1, 6, 1))
    _assert_bits_equal(J.numpy()[0], J_model[0, :6, :1])
    test.assertNotEqual(np.abs(J.numpy()).max(), 0.0)

    # Test eval_mass_matrix via ArticulationView (convenience pattern)
    H = view_a.eval_mass_matrix(state)
    test.assertEqual(H.shape, (1, 1, 1))
    _assert_bits_equal(H.numpy()[0], H_model[0, :1, :1])
    test.assertNotEqual(H.numpy()[0, 0, 0], 0.0)

    # Test with pre-allocated view-shaped buffers (performance pattern), for robot_b
    view_b = newton.selection.ArticulationView(model, pattern="robot_b")
    J2 = wp.zeros(
        (view_b.count, 6 * view_b.max_joints_per_articulation, view_b.max_dofs_per_articulation),
        dtype=float,
        device=device,
    )
    H2 = wp.zeros(
        (view_b.count, view_b.max_dofs_per_articulation, view_b.max_dofs_per_articulation),
        dtype=float,
        device=device,
    )

    test.assertIs(view_b.eval_jacobian(state, J2), J2)
    test.assertIs(view_b.eval_mass_matrix(state, H2), H2)
    _assert_bits_equal(J2.numpy()[0], J_model[1, :6, :1])
    _assert_bits_equal(H2.numpy()[0], H_model[1, :1, :1])


def test_floating_base_jacobian(test, device):
    """Test Jacobian for a floating base articulation (FREE joint at root)."""
    builder = newton.ModelBuilder()

    # Base link with FREE joint (6 DOFs)
    b_base = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
        mass=2.0,
    )
    builder.add_shape_box(body=b_base, hx=0.2, hy=0.2, hz=0.2)

    j_free = builder.add_joint_free(
        child=b_base,
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
    )

    # Child link with revolute joint (1 DOF)
    b_child = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
        mass=1.0,
    )
    builder.add_shape_box(body=b_child, hx=0.1, hy=0.1, hz=0.1)

    j_rev = builder.add_joint_revolute(
        parent=b_base,
        child=b_child,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.5, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
    )
    builder.add_articulation([j_free, j_rev], label="floating_robot")

    model = builder.finalize(device=device)
    state = model.state()

    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    # FREE joint has 6 DOFs, revolute has 1 -> total 7 DOFs, 2 links
    test.assertEqual(model.max_dofs_per_articulation, 7)
    test.assertEqual(model.max_joints_per_articulation, 2)

    J = newton.eval_jacobian(model, state)
    test.assertEqual(J.shape, (1, 12, 7))  # 1 articulation, 2*6 rows, 7 DOFs

    J_np = J.numpy()

    # Base link (rows 0-5): should be affected by the 6 FREE DOFs (columns 0-5)
    base_block = J_np[0, 0:6, 0:6]
    test.assertNotEqual(np.abs(base_block).max(), 0.0)

    # Base link should NOT be affected by the revolute DOF (column 6)
    test.assertEqual(J_np[0, 0, 6], 0.0)
    test.assertEqual(J_np[0, 1, 6], 0.0)
    test.assertEqual(J_np[0, 2, 6], 0.0)
    test.assertEqual(J_np[0, 3, 6], 0.0)
    test.assertEqual(J_np[0, 4, 6], 0.0)
    test.assertEqual(J_np[0, 5, 6], 0.0)

    # Child link (rows 6-11): should be affected by all 7 DOFs
    child_free_block = J_np[0, 6:12, 0:6]
    test.assertNotEqual(np.abs(child_free_block).max(), 0.0)
    # Revolute DOF should give angular z velocity on the child
    test.assertNotEqual(J_np[0, 11, 6], 0.0)

    # Mass matrix should be 7x7, symmetric, and positive definite
    H = newton.eval_mass_matrix(model, state)
    test.assertEqual(H.shape, (1, 7, 7))

    H_np = H.numpy()
    H_valid = H_np[0, :7, :7]
    np.testing.assert_allclose(H_valid, H_valid.T, rtol=1e-5, atol=1e-6)

    # Check positive definiteness via Cholesky
    np.linalg.cholesky(H_valid)


def test_jacobian_matches_body_qd_com(test, device):
    """`J @ joint_qd` should match COM-referenced `state.body_qd` per link."""
    model, base, slider = _build_translated_prismatic_chain(device)
    state = model.state()

    q = state.joint_q.numpy()
    qd = state.joint_qd.numpy()
    q[0] = 0.55
    q[1] = 0.8
    qd[0] = 1.1
    qd[1] = -0.35

    state.joint_q.assign(q)
    state.joint_qd.assign(qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    J = newton.eval_jacobian(model, state).numpy()
    body_qd = state.body_qd.numpy()

    base_block = J[0, 0:6, : model.joint_dof_count]
    slider_block = J[0, 6:12, : model.joint_dof_count]
    base_twist = base_block @ qd
    slider_twist = slider_block @ qd

    np.testing.assert_allclose(base_twist, body_qd[base], atol=1.0e-5, rtol=1.0e-6)
    np.testing.assert_allclose(slider_twist, body_qd[slider], atol=1.0e-5, rtol=1.0e-6)


def test_d6_three_angular_jacobian_matches_body_qd(test, device):
    """`J @ joint_qd` should match `state.body_qd` for a D6 joint with three
    angular DOFs at a non-identity configuration.

    The FK path transports each angular axis through the rotations applied by
    the preceding angular DOFs, so the Jacobian's angular columns must use the
    same transported axes rather than the raw joint axes.
    """
    model, child = _build_d6_three_angular(device)
    state = model.state()

    q = state.joint_q.numpy()
    qd = state.joint_qd.numpy()
    q[:3] = [0.5, -0.4, 0.7]
    qd[:3] = [0.9, -0.6, 0.3]
    state.joint_q.assign(q)
    state.joint_qd.assign(qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    J = newton.eval_jacobian(model, state).numpy()[0, :6, : model.joint_dof_count]
    body_qd = state.body_qd.numpy()[child]

    np.testing.assert_allclose(J @ qd, body_qd, atol=1.0e-5, rtol=1.0e-6)


def test_mass_matrix_matches_com_kinetic_energy(test, device):
    """`0.5 * qd^T H qd` should equal kinetic energy from COM twists."""
    model, base, slider = _build_translated_prismatic_chain(device)
    state = model.state()

    q = state.joint_q.numpy()
    qd = state.joint_qd.numpy()
    q[0] = 0.4
    q[1] = 0.6
    qd[0] = 0.9
    qd[1] = -0.25

    state.joint_q.assign(q)
    state.joint_qd.assign(qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    H = newton.eval_mass_matrix(model, state).numpy()[0, : model.joint_dof_count, : model.joint_dof_count]
    kinetic_from_h = 0.5 * float(qd @ H @ qd)
    kinetic_from_bodies = _kinetic_energy_from_body_twists(model, state, (base, slider))
    np.testing.assert_allclose(kinetic_from_h, kinetic_from_bodies, atol=1.0e-5, rtol=1.0e-6)


def test_floating_free_jacobian_matches_body_qd_com(test, device):
    """Floating-base Jacobian should match COM-referenced body twists."""
    model, body = _build_free_body_with_com(device)
    state = model.state()

    qd = state.joint_qd.numpy()
    qd[:] = np.array([0.35, -0.2, 0.15, 0.25, -0.4, 0.7], dtype=np.float32)
    state.joint_qd.assign(qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    J = newton.eval_jacobian(model, state).numpy()[0, :6, : model.joint_dof_count]
    body_qd = state.body_qd.numpy()[body]

    np.testing.assert_allclose(J @ qd, body_qd, atol=1.0e-5, rtol=1.0e-6)


def test_floating_free_mass_matrix_matches_com_kinetic_energy(test, device):
    """Floating-base mass matrix should match kinetic energy from COM twists."""
    model, body = _build_free_body_with_com(device)
    state = model.state()

    qd = state.joint_qd.numpy()
    qd[:] = np.array([0.3, -0.1, 0.2, 0.45, -0.25, 0.6], dtype=np.float32)
    state.joint_qd.assign(qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    H = newton.eval_mass_matrix(model, state).numpy()[0, : model.joint_dof_count, : model.joint_dof_count]
    kinetic_from_h = 0.5 * float(qd @ H @ qd)
    kinetic_from_body = _kinetic_energy_from_body_twists(model, state, (body,))
    np.testing.assert_allclose(kinetic_from_h, kinetic_from_body, atol=1.0e-5, rtol=1.0e-6)


def test_descendant_free_jacobian_matches_body_qd_com(test, device):
    """A rotated-parent descendant FREE chain should satisfy `J @ joint_qd == body_qd[child]` at the child COM."""
    model, _base, child = _build_descendant_free_with_rotated_parent(device)
    state = model.state()

    q = state.joint_q.numpy()
    qd = state.joint_qd.numpy()
    q[:] = np.array([0.35, -0.25, 0.45, *wp.quat_rpy(0.3, -0.2, 0.4)], dtype=np.float32)
    qd[:] = np.array([0.7, -0.15, 0.25, 0.35, -0.4, 0.5], dtype=np.float32)
    state.joint_q.assign(q)
    state.joint_qd.assign(qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    J = newton.eval_jacobian(model, state).numpy()[0, 6:12, : model.joint_dof_count]
    body_qd = state.body_qd.numpy()[child]

    np.testing.assert_allclose(J @ qd, body_qd, atol=1.0e-5, rtol=1.0e-6)


def test_descendant_free_mass_matrix_matches_com_kinetic_energy(test, device):
    """A rotated-parent descendant FREE chain should match COM-based kinetic energy under the public body_qd contract."""
    model, _base, child = _build_descendant_free_with_rotated_parent(device)
    state = model.state()

    q = state.joint_q.numpy()
    qd = state.joint_qd.numpy()
    q[:] = np.array([-0.15, 0.4, 0.3, *wp.quat_rpy(-0.25, 0.15, 0.5)], dtype=np.float32)
    qd[:] = np.array([0.25, -0.45, 0.3, 0.55, -0.2, 0.35], dtype=np.float32)
    state.joint_q.assign(q)
    state.joint_qd.assign(qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    H = newton.eval_mass_matrix(model, state).numpy()[0, : model.joint_dof_count, : model.joint_dof_count]
    kinetic_from_h = 0.5 * float(qd @ H @ qd)

    kinetic_from_body = _kinetic_energy_from_body_twists(model, state, (child,))
    np.testing.assert_allclose(kinetic_from_h, kinetic_from_body, atol=1.0e-5, rtol=1.0e-6)


class TestJacobianMassMatrix(unittest.TestCase):
    pass


devices = get_test_devices()

add_function_test(
    TestJacobianMassMatrix, "test_jacobian_simple_pendulum", test_jacobian_simple_pendulum, devices=devices
)
add_function_test(
    TestJacobianMassMatrix,
    "test_jacobian_numerical_verification",
    test_jacobian_numerical_verification,
    devices=devices,
)
add_function_test(TestJacobianMassMatrix, "test_mass_matrix_symmetry", test_mass_matrix_symmetry, devices=devices)
add_function_test(
    TestJacobianMassMatrix, "test_mass_matrix_positive_definite", test_mass_matrix_positive_definite, devices=devices
)
add_function_test(
    TestJacobianMassMatrix,
    "test_fixed_base_simple_pendulum_mass_matrix_matches_analytical",
    test_fixed_base_simple_pendulum_mass_matrix_matches_analytical,
    devices=devices,
)
add_function_test(
    TestJacobianMassMatrix,
    "test_floating_base_simple_pendulum_mass_matrix_matches_analytical",
    test_floating_base_simple_pendulum_mass_matrix_matches_analytical,
    devices=devices,
)
add_function_test(
    TestJacobianMassMatrix,
    "test_jacobian_multiple_articulations",
    test_jacobian_multiple_articulations,
    devices=devices,
)
add_function_test(TestJacobianMassMatrix, "test_jacobian_with_mask", test_jacobian_with_mask, devices=devices)
add_function_test(TestJacobianMassMatrix, "test_mass_matrix_with_mask", test_mass_matrix_with_mask, devices=devices)
add_function_test(
    TestJacobianMassMatrix,
    "test_jacobian_writes_only_selected_jacobians",
    test_jacobian_writes_only_selected_jacobians,
    devices=devices,
)
add_function_test(
    TestJacobianMassMatrix,
    "test_jacobian_mask_without_buffer_zeroes_unselected",
    test_jacobian_mask_without_buffer_zeroes_unselected,
    devices=devices,
)
add_function_test(
    TestJacobianMassMatrix,
    "test_articulation_view_jacobian_writes_view_slots",
    test_articulation_view_jacobian_writes_view_slots,
    devices=devices,
)
add_function_test(
    TestJacobianMassMatrix,
    "test_articulation_view_mass_matrix_writes_view_slots",
    test_articulation_view_mass_matrix_writes_view_slots,
    devices=devices,
)
add_function_test(
    TestJacobianMassMatrix,
    "test_articulation_view_rejects_other_shapes",
    test_articulation_view_rejects_other_shapes,
    devices=devices,
)
add_function_test(
    TestJacobianMassMatrix,
    "test_articulation_view_of_every_articulation_matches_unmasked",
    test_articulation_view_of_every_articulation_matches_unmasked,
    devices=devices,
)
add_function_test(
    TestJacobianMassMatrix,
    "test_mass_matrix_mask_zeroes_unselected_articulations",
    test_mass_matrix_mask_zeroes_unselected_articulations,
    devices=devices,
)
add_function_test(
    TestJacobianMassMatrix,
    "test_mass_matrix_scratch_skips_redundant_clears",
    test_mass_matrix_scratch_skips_redundant_clears,
    devices=devices,
)
add_function_test(
    TestJacobianMassMatrix,
    "test_jacobian_mask_accepts_masks_the_launch_converts",
    test_jacobian_mask_accepts_masks_the_launch_converts,
    devices=devices,
)
add_function_test(
    TestJacobianMassMatrix,
    "test_jacobian_mask_graph_replay_matches_eager",
    test_jacobian_mask_graph_replay_matches_eager,
    devices=devices,
)
add_function_test(
    TestJacobianMassMatrix,
    "test_jacobian_mask_graph_replay_fills_strided_view",
    test_jacobian_mask_graph_replay_fills_strided_view,
    devices=devices,
)
add_function_test(
    TestJacobianMassMatrix,
    "test_mass_matrix_mask_graph_replay_matches_eager",
    test_mass_matrix_mask_graph_replay_matches_eager,
    devices=devices,
)
add_function_test(
    TestJacobianMassMatrix, "test_prismatic_joint_jacobian", test_prismatic_joint_jacobian, devices=devices
)
add_function_test(TestJacobianMassMatrix, "test_empty_model", test_empty_model, devices=devices)
add_function_test(TestJacobianMassMatrix, "test_articulation_view_api", test_articulation_view_api, devices=devices)
add_function_test(TestJacobianMassMatrix, "test_floating_base_jacobian", test_floating_base_jacobian, devices=devices)
add_function_test(
    TestJacobianMassMatrix, "test_jacobian_matches_body_qd_com", test_jacobian_matches_body_qd_com, devices=devices
)
add_function_test(
    TestJacobianMassMatrix,
    "test_d6_three_angular_jacobian_matches_body_qd",
    test_d6_three_angular_jacobian_matches_body_qd,
    devices=devices,
)
add_function_test(
    TestJacobianMassMatrix,
    "test_mass_matrix_matches_com_kinetic_energy",
    test_mass_matrix_matches_com_kinetic_energy,
    devices=devices,
)
add_function_test(
    TestJacobianMassMatrix,
    "test_floating_free_jacobian_matches_body_qd_com",
    test_floating_free_jacobian_matches_body_qd_com,
    devices=devices,
)
add_function_test(
    TestJacobianMassMatrix,
    "test_descendant_free_jacobian_matches_body_qd_com",
    test_descendant_free_jacobian_matches_body_qd_com,
    devices=devices,
)
add_function_test(
    TestJacobianMassMatrix,
    "test_floating_free_mass_matrix_matches_com_kinetic_energy",
    test_floating_free_mass_matrix_matches_com_kinetic_energy,
    devices=devices,
)
add_function_test(
    TestJacobianMassMatrix,
    "test_descendant_free_mass_matrix_matches_com_kinetic_energy",
    test_descendant_free_mass_matrix_matches_com_kinetic_energy,
    devices=devices,
)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
