# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for eval_jacobian() and eval_mass_matrix() functions."""

from __future__ import annotations

import gc
import threading
import unittest
import weakref
from unittest import mock

import numpy as np
import warp as wp

import newton
import newton._src.sim.articulation as articulation_module
from newton._src.sim.articulation import (
    _CLEAR_REPLAY_CAPACITY,
    _clear_launch_arguments,
    _clear_replays,
    _launch_zero_selected_blocks,
    _zero_selected_articulation_blocks,
    _zero_selected_block_words,
    _zero_selected_blocks,
)
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


def _assert_bits_equal(actual, expected):
    """Assert that two float32 arrays hold the same bits, so that ``-0.0`` does not pass for ``0.0``."""
    np.testing.assert_array_equal(
        np.ascontiguousarray(actual, dtype=np.float32).view(np.uint32),
        np.ascontiguousarray(expected, dtype=np.float32).view(np.uint32),
    )


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

    # The second articulation is masked out, so its block keeps the zeros of the
    # zero-initialized buffer: a masked eval_jacobian leaves unselected blocks untouched.
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


def test_jacobian_mask_overwrites_only_selected_blocks(test, device):
    """Overwrite whole selected blocks and leave unselected articulations' entries untouched.

    Every selected block, padding included, must equal the unmasked result bitwise
    whatever the buffer held before; every unselected block keeps the caller's values.
    """
    model, state = _build_mixed_articulations(device)
    J_ref = newton.eval_jacobian(model, state).numpy()

    for name, selected in _mixed_selections(model).items():
        with test.subTest(selection=name):
            J = _filled(J_ref.shape, _SENTINEL, device)
            mask = wp.array(selected, dtype=bool, device=device)
            test.assertIs(newton.eval_jacobian(model, state, J, mask=mask), J)

            J_np = J.numpy()
            _assert_bits_equal(J_np[selected], J_ref[selected])
            _assert_bits_equal(J_np[~selected], np.full_like(J_ref[~selected], _SENTINEL))


def test_jacobian_mask_skips_whole_buffer_clear(test, device):
    """Clear the whole Jacobian buffer only when no mask is given."""
    model, state = _build_mixed_articulations(device)
    J = wp.zeros(_jacobian_shape(model), dtype=float, device=device)
    mask = wp.array(_mixed_selections(model)["fan"], dtype=bool, device=device)

    patches, cleared = _record_clears()
    with patches[0], patches[1]:
        newton.eval_jacobian(model, state, J, mask=mask)
        masked_clears = [ptr for ptr, _shape, _dtype in cleared]
        cleared.clear()
        newton.eval_jacobian(model, state, J)
        unmasked_clears = [ptr for ptr, _shape, _dtype in cleared]

    test.assertNotIn(J.ptr, masked_clears)
    test.assertIn(J.ptr, unmasked_clears)


def test_jacobian_mask_without_buffer_zeroes_unselected(test, device):
    """Return zero blocks for unselected articulations when eval_jacobian allocates J."""
    model, state = _build_mixed_articulations(device)
    J_ref = newton.eval_jacobian(model, state).numpy()
    selected = _mixed_selections(model)["mixed"]

    J = newton.eval_jacobian(model, state, mask=wp.array(selected, dtype=bool, device=device)).numpy()

    _assert_bits_equal(J[selected], J_ref[selected])
    _assert_bits_equal(J[~selected], np.zeros_like(J_ref[~selected]))


def test_articulation_view_jacobian_writes_only_view_blocks(test, device):
    """Write only the view's articulation blocks, narrowed by a per-world mask, through the view."""
    model, state = _build_mixed_articulations(device)
    J_ref = newton.eval_jacobian(model, state).numpy()
    view = newton.selection.ArticulationView(model, "fan", verbose=False)
    world_mask = np.array([False, True, False])

    for name, mask in (("view", None), ("world 1", world_mask)):
        with test.subTest(selection=name):
            J = _filled(J_ref.shape, _SENTINEL, device)
            view_mask = None if mask is None else wp.array(mask, dtype=bool, device=device)
            test.assertIs(view.eval_jacobian(state, J, mask=view_mask), J)

            selected = view.get_model_articulation_mask(mask=view_mask).numpy()
            test.assertEqual(int(selected.sum()), 3 if mask is None else 1)
            J_np = J.numpy()
            _assert_bits_equal(J_np[selected], J_ref[selected])
            _assert_bits_equal(J_np[~selected], np.full_like(J_ref[~selected], _SENTINEL))


def test_articulation_view_of_every_articulation_clears_whole_buffers(test, device):
    """Evaluate a view of every articulation like the unmasked functions, whole-buffer clears included.

    Without a mask such a view selects everything, so the unmasked paths give the same
    values and clear the caller's Jacobian and the internal one with a memset each. A
    mask still restricts the selection.
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

    jacobian_clears = [ptr for ptr, shape, _dtype in cleared if shape == J_ref.shape]
    test.assertEqual(len(jacobian_clears), 2)
    test.assertIn(J.ptr, jacobian_clears)
    _assert_bits_equal(J.numpy(), J_ref)
    _assert_bits_equal(H.numpy(), H_ref)

    J = _filled(J_ref.shape, _SENTINEL, device)
    view.eval_jacobian(state, J, mask=wp.array([False, True, False], dtype=bool, device=device))
    selected = model.articulation_world.numpy() == 1
    J_np = J.numpy()
    _assert_bits_equal(J_np[selected], J_ref[selected])
    _assert_bits_equal(J_np[~selected], np.full_like(J_ref[~selected], _SENTINEL))


def test_mass_matrix_mask_zeroes_unselected_articulations(test, device):
    """Return exact selected blocks and zeros elsewhere from a masked eval_mass_matrix.

    This is the mask convention that eval_inverse_dynamics_passive documents as
    mirroring eval_mass_matrix. It must hold when H holds stale values and the
    supplied Jacobian carries NaN in every block eval_jacobian left unselected.
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

    The internal Jacobian is cleared once without a mask and never as a whole with one;
    the spatial-inertia scratch is fully overwritten and never zero-filled. H keeps its
    single clear, which the documented zeros of unselected articulations need.
    """
    model, state = _build_mixed_articulations(device)
    mask = wp.array(_mixed_selections(model)["fan"], dtype=bool, device=device)
    J_shape = _jacobian_shape(model)
    H_shape = (model.articulation_count, model.max_dofs_per_articulation, model.max_dofs_per_articulation)

    patches, cleared = _record_clears()
    with patches[0], patches[1]:
        newton.eval_mass_matrix(model, state, mask=mask)
        masked = list(cleared)
        cleared.clear()
        newton.eval_mass_matrix(model, state)
        unmasked = list(cleared)

    for name, events, jacobian_clears in (("masked", masked, 0), ("unmasked", unmasked, 1)):
        with test.subTest(call=name):
            test.assertEqual(sum(shape == J_shape for _ptr, shape, _dtype in events), jacobian_clears)
            test.assertEqual(sum(shape == H_shape for _ptr, shape, _dtype in events), 1)
            test.assertFalse([dtype for _ptr, _shape, dtype in events if dtype == wp.spatial_matrix])


def test_zero_selected_blocks_matches_numpy(test, device):
    """Zero exactly the selected per-articulation blocks, through the device path and through the kernels.

    Block shapes sit on both sides of the blockwise kernel's lane count and of the
    block size at which it replaces word stores, with odd sizes that word stores
    cannot cover. A buffer may hold more blocks than the mask covers, a misaligned
    buffer cannot take word stores, and a strided view must change only inside its
    own window, whether or not it starts on an 8-byte boundary.
    """
    rng = np.random.default_rng(3)
    selected = np.array([True, False, True, True, False])
    mask = wp.array(selected, dtype=bool, device=device)
    shapes = ((5, 1, 1), (5, 6, 1), (5, 12, 2), (5, 1, 300), (5, 66, 31), (5, 37, 33), (5, 64, 32), (5, 654, 108))
    for clear in (_zero_selected_blocks, _launch_zero_selected_blocks):
        for shape in shapes:
            with test.subTest(clear=clear.__name__, shape=shape):
                values = rng.normal(size=shape).astype(np.float32)
                blocks = wp.array(values, dtype=float, device=device)
                clear(blocks, mask, shape[0], device)
                expected = values.copy()
                expected[selected] = 0.0
                _assert_bits_equal(blocks.numpy(), expected)

        with test.subTest(clear=clear.__name__, shape="more blocks than articulations"):
            values = rng.normal(size=(7, 6, 2)).astype(np.float32)
            blocks = wp.array(values, dtype=float, device=device)
            clear(blocks, wp.array(np.append(selected, True), dtype=bool, device=device), 5, device)
            expected = values.copy()
            expected[:5][selected] = 0.0
            _assert_bits_equal(blocks.numpy(), expected)

        with test.subTest(clear=clear.__name__, shape="misaligned buffer"):
            values = rng.normal(size=1 + 5 * 12).astype(np.float32)
            whole = wp.array(values, dtype=float, device=device)
            clear(whole[1:].reshape((5, 6, 2)), mask, 5, device)
            expected = values.copy()
            expected[1:].reshape(5, 12)[selected] = 0.0
            _assert_bits_equal(whole.numpy(), expected)

        # The first view starts 256 bytes into the buffer, the second 380 bytes.
        for name, rows, cols in (("aligned", slice(2, 10), slice(4, 16)), ("misaligned", slice(3, 11), slice(5, 17))):
            with test.subTest(clear=clear.__name__, shape=f"{name} strided view"):
                values = rng.normal(size=(5, 20, 30)).astype(np.float32)
                whole = wp.array(values, dtype=float, device=device)
                clear(whole[:, rows, cols], mask, 5, device)
                expected = values.copy()
                expected[selected, rows, cols] = 0.0
                _assert_bits_equal(whole.numpy(), expected)


def test_zero_selected_blocks_kernel_by_block_size(test, device):
    """Clear small blocks word by word and large ones block by block.

    The word kernel runs one thread per 8-byte word of the blocks that the mask
    covers and stores only the selected words. A block walk spends its setup on every
    block, which outweighs the stores of a small block; word stores keep a warp's
    256 bytes contiguous at any block size. Odd blocks, misaligned buffers and
    strided views cannot take word stores.
    """
    mask = wp.array([True, False, True, True, False], dtype=bool, device=device)
    misaligned = wp.zeros(1 + 5 * 12, dtype=float, device=device)[1:].reshape((5, 6, 2))
    # 8 x 12 windows of 20 x 30 blocks, starting 256 bytes into the buffer.
    aligned_strided = wp.zeros((5, 20, 30), dtype=float, device=device)[:, 2:10, 4:16]
    cases = {
        "6 x 1": (wp.zeros((5, 6, 1), dtype=float, device=device), _zero_selected_block_words, 15),
        "66 x 31": (wp.zeros((5, 66, 31), dtype=float, device=device), _zero_selected_block_words, 5 * 33 * 31),
        "37 x 33": (wp.zeros((5, 37, 33), dtype=float, device=device), _zero_selected_articulation_blocks, (5, 256)),
        "64 x 32": (wp.zeros((5, 64, 32), dtype=float, device=device), _zero_selected_articulation_blocks, (5, 256)),
        "misaligned": (misaligned, _zero_selected_articulation_blocks, (5, 32)),
        "aligned strided": (aligned_strided, _zero_selected_articulation_blocks, (5, 96)),
    }
    for name, (blocks, kernel, dim) in cases.items():
        with (
            test.subTest(blocks=name),
            mock.patch.dict(_clear_replays, clear=True),
            mock.patch.object(wp, "launch", wraps=wp.launch) as launch,
        ):
            _launch_zero_selected_blocks(blocks, mask, 5, device)
            test.assertEqual(launch.call_count, 1)
            test.assertIs(launch.call_args.args[0], kernel)
            test.assertEqual(launch.call_args.kwargs["dim"], dim)


def test_zero_selected_blocks_word_view_spans_covered_blocks(test, device):
    """Pack only the blocks that the mask covers into the word kernel's view, owned or not.

    A buffer may hold more blocks than ``articulation_count``. Warp rejects a view of
    2**31 or more words, so a view of the whole buffer could fail where the launch
    reaches far fewer words.
    """
    values = np.ones((7, 6, 2), dtype=np.float32)
    expected = values.copy()
    expected[[0, 2, 3]] = 0.0
    blocks = wp.array(values, dtype=float, device=device)
    mask = wp.array([True, False, True, True, False, True, True], dtype=bool, device=device)
    for owned in (True, False):
        with test.subTest(owned=owned):
            blocks.assign(values)
            kernel, dim, inputs, outputs = _clear_launch_arguments(blocks, mask, 5, owned=owned)
            test.assertIs(kernel, _zero_selected_block_words)
            test.assertEqual(dim, 5 * 6)
            test.assertEqual((outputs[0].ptr, outputs[0].shape), (blocks.ptr, (5 * 6,)))
            wp.launch(kernel, dim=dim, inputs=inputs, outputs=outputs, device=device)
            _assert_bits_equal(blocks.numpy(), expected)


def test_zero_selected_blocks_replays_recorded_launch(test, device):
    """Replay the clear recorded for the same buffer and mask instead of packing a new launch.

    ``warp.launch`` packs every argument on each call, which costs more host time
    than the clear of a small model. A replay reads the mask's current values, a
    new buffer records its own launch, and a graph capture launches directly so the
    graph holds the launch.
    """
    values = np.arange(4 * 6 * 2, dtype=np.float32).reshape(4, 6, 2) + 1.0
    blocks = wp.array(values, dtype=float, device=device)
    mask = wp.array([True, False, False, True], dtype=bool, device=device)

    def expect(selected):
        expected = values.copy()
        expected[np.asarray(selected)] = 0.0
        _assert_bits_equal(blocks.numpy(), expected)

    with mock.patch.dict(_clear_replays, clear=True), mock.patch.object(wp, "launch", wraps=wp.launch) as launch:
        _launch_zero_selected_blocks(blocks, mask, 4, device)
        test.assertEqual(launch.call_count, 1)
        test.assertTrue(launch.call_args.kwargs["record_cmd"])
        expect([True, False, False, True])

        for selected in ([True, False, False, True], [False, True, True, False]):
            blocks.assign(values)
            mask.assign(selected)
            _launch_zero_selected_blocks(blocks, mask, 4, device)
            expect(selected)
        test.assertEqual(launch.call_count, 1)
        test.assertEqual(len(_clear_replays), 1)

        other = wp.array(values, dtype=float, device=device)
        _launch_zero_selected_blocks(other, mask, 4, device)
        test.assertEqual(launch.call_count, 2)
        _assert_bits_equal(other.numpy()[[1, 2]], np.zeros((2, 6, 2), dtype=np.float32))
        _assert_bits_equal(other.numpy()[[0, 3]], values[[0, 3]])

        blocks.assign(values)
        with wp.ScopedCapture(device) as capture:
            _launch_zero_selected_blocks(blocks, mask, 4, device)
        test.assertEqual(launch.call_count, 3)
        test.assertFalse(launch.call_args.kwargs.get("record_cmd", False))
        wp.capture_launch(capture.graph)
        expect([False, True, True, False])


def test_zero_selected_blocks_replay_holds_no_buffers(test, device):
    """Keep no buffer or mask alive through a recorded clear, and keep a bounded number of them.

    A recorded launch reaches its arrays through views that do not own memory, so a
    Jacobian freed by its caller, or allocated per call as in eval_mass_matrix, is
    released at once.
    """
    with mock.patch.dict(_clear_replays, clear=True):
        blocks = wp.zeros((4, 6, 2), dtype=float, device=device)
        mask = wp.array([True, False, True, False], dtype=bool, device=device)
        _launch_zero_selected_blocks(blocks, mask, 4, device)
        test.assertEqual(len(_clear_replays), 1)
        references = (weakref.ref(blocks), weakref.ref(mask))
        del blocks, mask
        gc.collect()
        test.assertEqual([reference() for reference in references], [None, None])

        kept = [wp.zeros((4, 6, 2), dtype=float, device=device) for _ in range(_CLEAR_REPLAY_CAPACITY + 3)]
        mask = wp.array([True, False, True, False], dtype=bool, device=device)
        for blocks in kept:
            _launch_zero_selected_blocks(blocks, mask, 4, device)
        test.assertEqual(len(_clear_replays), _CLEAR_REPLAY_CAPACITY)

        # The oldest records went first: the last buffers replay and the first records anew.
        with mock.patch.object(wp, "launch", wraps=wp.launch) as launch:
            for blocks in kept[3:]:
                _launch_zero_selected_blocks(blocks, mask, 4, device)
            test.assertEqual(launch.call_count, 0)
            _launch_zero_selected_blocks(kept[0], mask, 4, device)
            test.assertEqual(launch.call_count, 1)


def test_zero_selected_blocks_launches_directly_for_warp_debugging(test, device):
    """Launch the clear through warp.launch on every call while a Warp debugging aid is on.

    Printed launches and CUDA error checks happen in warp.launch, and a recorded launch
    keeps the module build it was recorded with, so replays would skip them, and the
    checks of a debug or floating-point-checking build. ``warp.launch`` is mocked, so
    no debug build is compiled.
    """
    blocks = wp.zeros((4, 6, 2), dtype=float, device=device)
    mask = wp.array([True, False, True, False], dtype=bool, device=device)
    aids = {"none": ("print_launches", False), "debug mode": ("mode", "debug")}
    aids.update({name: (name, True) for name in ("verify_fp", "print_launches", "verify_cuda")})
    for aid, (setting, value) in aids.items():
        with (
            test.subTest(aid=aid),
            mock.patch.dict(_clear_replays, clear=True),
            mock.patch.object(wp.config, setting, value),
            mock.patch.object(wp, "launch") as launch,
        ):
            for _ in range(2):
                _launch_zero_selected_blocks(blocks, mask, 4, device)
            recorded = [call.kwargs.get("record_cmd", False) for call in launch.call_args_list]
            test.assertEqual(recorded, [True] if aid == "none" else [False, False])


def test_zero_selected_blocks_replay_cache_takes_concurrent_records(test, device):
    """Keep the replay cache bounded and consistent when two threads record clears at once.

    The cache below makes each thread read its size, or its oldest key, and then
    wait until the other thread has read it too, as an unlucky thread switch can.
    Without a lock around the eviction and the insertion, both threads then see room
    for one more record and the cache overflows, or both drop the same oldest record
    and the second drop raises KeyError. With the lock, the second thread reads only
    after the first has finished, and the wait times out.
    """
    meet_at_size, meet_at_oldest = threading.Barrier(2, timeout=0.1), threading.Barrier(2, timeout=0.1)

    def meet(barrier):
        try:
            barrier.wait()
        except threading.BrokenBarrierError:
            pass

    class InterleavingDict(dict):
        def __len__(self):
            size = super().__len__()
            meet(meet_at_size)
            return size

        def __iter__(self):
            keys = list(super().__iter__())
            meet(meet_at_oldest)
            return iter(keys)

    mask = wp.array([True, False, True, False], dtype=bool, device=device)
    values = np.ones((4, 6, 2), dtype=np.float32)
    expected = values.copy()
    expected[[0, 2]] = 0.0
    # Load the module before the threads record their launches.
    _launch_zero_selected_blocks(wp.array(values, dtype=float, device=device), mask, 4, device)
    for free_records in (1, 0):
        with test.subTest(free_records=free_records):
            for barrier in (meet_at_size, meet_at_oldest):
                barrier.reset()
            cache = InterleavingDict({("other", i): None for i in range(_CLEAR_REPLAY_CAPACITY - free_records)})
            buffers = [wp.array(values, dtype=float, device=device) for _ in range(2)]
            errors = []

            def clear(blocks, errors=errors):
                try:
                    _launch_zero_selected_blocks(blocks, mask, 4, device)
                except Exception as error:
                    errors.append(error)

            with mock.patch.object(articulation_module, "_clear_replays", cache):
                threads = [threading.Thread(target=clear, args=(blocks,)) for blocks in buffers]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()
            for barrier in (meet_at_size, meet_at_oldest):
                barrier.abort()

            test.assertEqual(errors, [])
            test.assertEqual(len(cache), _CLEAR_REPLAY_CAPACITY)
            test.assertEqual(sum(key[0] == "other" for key in cache), _CLEAR_REPLAY_CAPACITY - 2)
            for blocks in buffers:
                _assert_bits_equal(blocks.numpy(), expected)


def test_zero_selected_blocks_fills_cpu_memory_on_host(test, device):
    """Clear selected CPU blocks with a host fill, not the kernel, whose per-entry loop runs far slower on CPU."""
    blocks = wp.array(np.ones((4, 6, 3), dtype=np.float32), dtype=float, device=device)
    mask = wp.array([True, False, True, False], dtype=bool, device=device)

    with mock.patch.object(wp, "launch", wraps=wp.launch) as launch:
        _zero_selected_blocks(blocks, mask, 4, device)

    test.assertEqual(launch.call_count, 0)
    expected = np.ones((4, 6, 3), dtype=np.float32)
    expected[[0, 2]] = 0.0
    _assert_bits_equal(blocks.numpy(), expected)


def test_jacobian_mask_accepts_masks_the_launch_converts(test, device):
    """Select the same articulations from masks that only the kernel launch accepts as from an exact mask.

    The launch reads the first ``articulation_count`` entries of a longer Warp mask
    and, on the CPU, takes a NumPy Boolean mask through its array interface. The
    masked mass matrix, whose internal Jacobian goes through the same clear, must
    agree as well.
    """
    model, state = _build_mixed_articulations(device)
    selected = _mixed_selections(model)["mixed"]
    exact = wp.array(selected, dtype=bool, device=device)
    J_ref = _filled(_jacobian_shape(model), _SENTINEL, device)
    newton.eval_jacobian(model, state, J_ref, mask=exact)
    H_ref = newton.eval_mass_matrix(model, state, mask=exact).numpy()

    masks = {"longer Warp mask": wp.array(np.append(selected, [True, False]), dtype=bool, device=device)}
    if device.is_cpu:
        masks["NumPy mask"] = selected
    for name, mask in masks.items():
        with test.subTest(mask=name):
            J = _filled(_jacobian_shape(model), _SENTINEL, device)
            newton.eval_jacobian(model, state, J, mask=mask)
            _assert_bits_equal(J.numpy(), J_ref.numpy())
            _assert_bits_equal(newton.eval_mass_matrix(model, state, mask=mask).numpy(), H_ref)


def test_jacobian_mask_graph_replay_matches_eager(test, device):
    """Clear and fill the blocks that the mask selects when a captured masked evaluation replays.

    Each replay starts from a reset buffer and a new mask: the selected blocks must
    equal the eager unmasked result and the others keep the reset values.
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

            J_np = J.numpy()
            _assert_bits_equal(J_np[selected], J_ref[selected])
            _assert_bits_equal(J_np[~selected], np.full_like(J_ref[~selected], _SENTINEL))


def test_mass_matrix_mask_graph_replay_matches_eager(test, device):
    """Replay a captured masked eval_mass_matrix whose internal Jacobian is allocated during the capture.

    New float allocations start as NaN, as reused memory may hold anything, so a
    replay must clear the blocks that the mask selects when it runs.
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


def test_jacobian_mask_clear_is_not_taped(test, device):
    """Keep the masked clear off a wp.Tape, as the whole-buffer clear it replaces.

    A taped clear would run its kernel's adjoint in ``tape.backward()``, which
    overwrites ``J.grad``: the word kernel's adjoint zeroes whole selected blocks,
    and the per-block kernel's writes into unselected blocks and past ``J``. Graph
    captures and, on the CPU, NumPy masks launch the clear through
    :func:`warp.launch`. ``J`` views the front of a larger buffer, so a gradient
    write past ``J`` shows in the buffer's tail. Every case must leave the gradient
    of the eager call, whose clear the tape never sees.
    """
    selected = np.array([False, True, False, True])
    for leaf_count in (6, 18):  # 252- and 2052-entry blocks: the word and the per-block kernel
        model, state = _build_fans(device, leaf_count)
        shape = _jacobian_shape(model)
        size = int(np.prod(shape))
        joint_S_s = wp.zeros(model.joint_dof_count, dtype=wp.spatial_vector, device=device)
        mask = wp.array(selected, dtype=bool, device=device)

        def eager(J, model=model, state=state, joint_S_s=joint_S_s, mask=mask):
            newton.eval_jacobian(model, state, J, joint_S_s=joint_S_s, mask=mask)

        def captured(J, model=model, state=state, joint_S_s=joint_S_s, mask=mask):
            with wp.ScopedCapture(device) as capture:
                newton.eval_jacobian(model, state, J, joint_S_s=joint_S_s, mask=mask)
            wp.capture_launch(capture.graph)

        def numpy_mask(J, model=model, state=state, joint_S_s=joint_S_s):
            newton.eval_jacobian(model, state, J, joint_S_s=joint_S_s, mask=selected)

        # Load the modules eagerly; a capture cannot compile them.
        eager(wp.zeros(shape, dtype=float, device=device))
        cases = {"eager": eager, "capture": captured}
        if device.is_cpu:
            cases["NumPy mask"] = numpy_mask
        gradients = {}
        for name, evaluate in cases.items():
            with test.subTest(leaves=leaf_count, case=name):
                buffer = wp.zeros(size + 4096, dtype=float, device=device, requires_grad=True)
                tape = wp.Tape()
                with tape:
                    evaluate(buffer[:size].reshape(shape))
                taped = [launch[0].key for launch in tape.launches]
                test.assertNotIn(_zero_selected_articulation_blocks.key, taped)
                test.assertNotIn(_zero_selected_block_words.key, taped)

                buffer.grad.fill_(1.0)
                tape.backward()
                gradients[name] = buffer.grad.numpy()
                np.testing.assert_array_equal(gradients[name][size:], 1.0)
                np.testing.assert_array_equal(gradients[name][:size].reshape(shape[0], -1)[~selected], 1.0)
                _assert_bits_equal(gradients[name], gradients["eager"])


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

    # Create ArticulationView for just robot_a
    view_a = newton.selection.ArticulationView(model, pattern="robot_a")

    # Test eval_jacobian via ArticulationView (convenience pattern)
    J = view_a.eval_jacobian(state)
    test.assertIsNotNone(J)
    test.assertEqual(J.shape[0], model.articulation_count)

    J_np = J.numpy()
    # robot_a (index 0) should have non-zero Jacobian
    test.assertNotEqual(np.abs(J_np[0]).max(), 0.0)
    # robot_b (index 1) should be zero (not in view)
    test.assertEqual(np.abs(J_np[1]).max(), 0.0)

    # Test eval_mass_matrix via ArticulationView (convenience pattern)
    H = view_a.eval_mass_matrix(state)
    test.assertIsNotNone(H)
    test.assertEqual(H.shape[0], model.articulation_count)

    H_np = H.numpy()
    # robot_a should have non-zero mass matrix
    test.assertNotEqual(H_np[0, 0, 0], 0.0)
    # robot_b should be zero
    test.assertEqual(H_np[1, 0, 0], 0.0)

    # Test with pre-allocated buffers (performance pattern)
    J2 = wp.zeros(
        (model.articulation_count, model.max_joints_per_articulation * 6, model.max_dofs_per_articulation),
        dtype=float,
        device=device,
    )
    H2 = wp.zeros(
        (model.articulation_count, model.max_dofs_per_articulation, model.max_dofs_per_articulation),
        dtype=float,
        device=device,
    )

    # Create view for robot_b
    view_b = newton.selection.ArticulationView(model, pattern="robot_b")

    J2_returned = view_b.eval_jacobian(state, J2)
    H2_returned = view_b.eval_mass_matrix(state, H2)

    test.assertIs(J2_returned, J2)
    test.assertIs(H2_returned, H2)

    J2_np = J2.numpy()
    H2_np = H2.numpy()

    # robot_a is not in view_b. Its Jacobian block keeps the zeros of the zero-initialized
    # J2, since the view leaves blocks outside it untouched; eval_mass_matrix zeroes its H2
    # entries.
    test.assertEqual(np.abs(J2_np[0]).max(), 0.0)
    test.assertEqual(H2_np[0, 0, 0], 0.0)
    # robot_b should have values
    test.assertNotEqual(np.abs(J2_np[1]).max(), 0.0)
    test.assertNotEqual(H2_np[1, 0, 0], 0.0)


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
    "test_jacobian_mask_overwrites_only_selected_blocks",
    test_jacobian_mask_overwrites_only_selected_blocks,
    devices=devices,
)
add_function_test(
    TestJacobianMassMatrix,
    "test_jacobian_mask_skips_whole_buffer_clear",
    test_jacobian_mask_skips_whole_buffer_clear,
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
    "test_articulation_view_jacobian_writes_only_view_blocks",
    test_articulation_view_jacobian_writes_only_view_blocks,
    devices=devices,
)
add_function_test(
    TestJacobianMassMatrix,
    "test_articulation_view_of_every_articulation_clears_whole_buffers",
    test_articulation_view_of_every_articulation_clears_whole_buffers,
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
    "test_zero_selected_blocks_matches_numpy",
    test_zero_selected_blocks_matches_numpy,
    devices=devices,
)
add_function_test(
    TestJacobianMassMatrix,
    "test_zero_selected_blocks_kernel_by_block_size",
    test_zero_selected_blocks_kernel_by_block_size,
    devices=devices,
)
add_function_test(
    TestJacobianMassMatrix,
    "test_zero_selected_blocks_word_view_spans_covered_blocks",
    test_zero_selected_blocks_word_view_spans_covered_blocks,
    devices=devices,
)
add_function_test(
    TestJacobianMassMatrix,
    "test_zero_selected_blocks_replays_recorded_launch",
    test_zero_selected_blocks_replays_recorded_launch,
    devices=devices,
)
add_function_test(
    TestJacobianMassMatrix,
    "test_zero_selected_blocks_replay_holds_no_buffers",
    test_zero_selected_blocks_replay_holds_no_buffers,
    devices=devices,
)
add_function_test(
    TestJacobianMassMatrix,
    "test_zero_selected_blocks_launches_directly_for_warp_debugging",
    test_zero_selected_blocks_launches_directly_for_warp_debugging,
    devices=devices,
)
add_function_test(
    TestJacobianMassMatrix,
    "test_zero_selected_blocks_replay_cache_takes_concurrent_records",
    test_zero_selected_blocks_replay_cache_takes_concurrent_records,
    devices=[device for device in devices if device.is_cpu],
)
add_function_test(
    TestJacobianMassMatrix,
    "test_zero_selected_blocks_fills_cpu_memory_on_host",
    test_zero_selected_blocks_fills_cpu_memory_on_host,
    devices=[device for device in devices if device.is_cpu],
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
    "test_mass_matrix_mask_graph_replay_matches_eager",
    test_mass_matrix_mask_graph_replay_matches_eager,
    devices=devices,
)
add_function_test(
    TestJacobianMassMatrix,
    "test_jacobian_mask_clear_is_not_taped",
    test_jacobian_mask_clear_is_not_taped,
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
