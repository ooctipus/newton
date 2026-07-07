# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Analytical tests for batched inverse dynamics."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _diagonal_inertia(x: float, y: float, z: float) -> wp.mat33:
    return wp.matrix_from_rows(wp.vec3(x, 0.0, 0.0), wp.vec3(0.0, y, 0.0), wp.vec3(0.0, 0.0, z))


def _build_pendulum(device):
    mass = 2.3
    length = 0.65
    inertia_z = 0.17
    builder = newton.ModelBuilder(gravity=-9.81, up_axis=newton.Axis.Y)
    body = builder.add_link(mass=mass)
    builder.body_com[body] = wp.vec3(length, 0.0, 0.0)
    builder.body_inertia[body] = _diagonal_inertia(0.11, 0.13, inertia_z)
    joint = builder.add_joint_revolute(parent=-1, child=body, axis=newton.Axis.Z)
    builder.add_articulation([joint], label="pendulum")
    return builder.finalize(device=device), body, mass, length, inertia_z


def _build_free_body(device):
    mass = 3.1
    inertia = np.diag([0.22, 0.31, 0.47]).astype(np.float32)
    builder = newton.ModelBuilder(gravity=-9.81, up_axis=newton.Axis.Z)
    body = builder.add_body(mass=mass)
    builder.body_com[body] = wp.vec3(0.37, -0.21, 0.16)
    builder.body_inertia[body] = wp.mat33(*inertia.reshape(-1))
    return builder.finalize(device=device), body, mass, inertia


def _build_d6_body(device):
    inertia = np.diag([0.23, 0.37, 0.51]).astype(np.float32)
    builder = newton.ModelBuilder(gravity=0.0)
    body = builder.add_link(mass=1.8)
    builder.body_inertia[body] = wp.mat33(*inertia.reshape(-1))
    axis = newton.ModelBuilder.JointDofConfig
    joint = builder.add_joint_d6(
        parent=-1,
        child=body,
        angular_axes=[axis(axis=newton.Axis.X), axis(axis=newton.Axis.Y), axis(axis=newton.Axis.Z)],
    )
    builder.add_articulation([joint], label="d6_body")
    return builder.finalize(device=device), body, inertia


def _rotate_vector(axis: np.ndarray, angle: float, value: np.ndarray) -> np.ndarray:
    return (
        value * np.cos(angle)
        + np.cross(axis, value) * np.sin(angle)
        + axis * np.dot(axis, value) * (1.0 - np.cos(angle))
    )


def _arrays(model, batch_size, joint_q=None, joint_qd=None, joint_qdd=None):
    if joint_q is None:
        joint_q = np.repeat(model.joint_q.numpy()[None], batch_size, axis=0)
    if joint_qd is None:
        joint_qd = np.zeros((batch_size, model.joint_dof_count), dtype=np.float32)
    if joint_qdd is None:
        joint_qdd = np.zeros((batch_size, model.joint_dof_count), dtype=np.float32)
    return (
        wp.array(joint_q, dtype=wp.float32, device=model.device),
        wp.array(joint_qd, dtype=wp.float32, device=model.device),
        wp.array(joint_qdd, dtype=wp.float32, device=model.device),
        wp.empty((batch_size, model.joint_dof_count), dtype=wp.float32, device=model.device),
    )


def test_pendulum_gravity_and_acceleration(test, device):
    model, _body, mass, length, inertia_z = _build_pendulum(device)
    angle = 0.43
    acceleration = 1.7
    joint_q = np.array([[angle]], dtype=np.float32)
    joint_qdd = np.array([[acceleration]], dtype=np.float32)
    q, qd, qdd, joint_f = _arrays(model, 1, joint_q=joint_q, joint_qdd=joint_qdd)

    newton.dynamics.DynamicsInverse(model, 1).compute(q, qd, qdd, joint_f)

    expected = (inertia_z + mass * length**2) * acceleration + mass * 9.81 * length * np.cos(angle)
    np.testing.assert_allclose(joint_f.numpy()[0, 0], expected, atol=2.0e-5, rtol=2.0e-5)


def test_mass_matrix_identity(test, device):
    model, _body, _mass, _length, _inertia_z = _build_pendulum(device)
    joint_q = np.array([[0.37]], dtype=np.float32)
    joint_qd = np.array([[-0.81]], dtype=np.float32)
    joint_qdd = np.array([[1.23]], dtype=np.float32)
    q, qd, qdd, joint_f = _arrays(model, 1, joint_q, joint_qd, joint_qdd)
    zero_qdd = wp.zeros_like(qdd)
    joint_bias = wp.empty_like(joint_f)
    evaluator = newton.dynamics.DynamicsInverse(model, 1)

    evaluator.compute(q, qd, qdd, joint_f)
    evaluator.compute(q, qd, zero_qdd, joint_bias)

    state = model.state()
    state.joint_q.assign(joint_q[0])
    state.joint_qd.assign(joint_qd[0])
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    mass_matrix = newton.eval_mass_matrix(model, state).numpy()[0, :1, :1]
    expected = mass_matrix @ joint_qdd[0] + model.joint_armature.numpy() * joint_qdd[0]
    np.testing.assert_allclose(joint_f.numpy()[0] - joint_bias.numpy()[0], expected, atol=2.0e-5, rtol=2.0e-5)


def test_free_root_public_com_world_convention(test, device):
    model, body, mass, inertia_body = _build_free_body(device)
    rotation = np.asarray(wp.quat_rpy(0.41, -0.27, 0.19), dtype=np.float32)
    joint_q = np.array([[0.8, -0.5, 1.4, *rotation]], dtype=np.float32)
    joint_qd = np.array([[0.31, -0.17, 0.22, 0.44, -0.36, 0.29]], dtype=np.float32)
    joint_qdd = np.array([[-0.28, 0.52, 0.33, -0.21, 0.37, 0.18]], dtype=np.float32)
    q, qd, qdd, joint_f = _arrays(model, 1, joint_q, joint_qd, joint_qdd)
    evaluator = newton.dynamics.DynamicsInverse(model, 1)

    evaluator.compute(q, qd, qdd, joint_f)

    body_q = wp.empty((1, model.body_count), dtype=wp.transform, device=device)
    body_qd = wp.empty((1, model.body_count), dtype=wp.spatial_vector, device=device)
    newton.eval_fk_batched(model, q, qd, body_q, body_qd)
    body_rotation = body_q.numpy()[0, body, 3:7]
    rotation_matrix = np.asarray(wp.quat_to_matrix(wp.quat(*body_rotation)), dtype=np.float64).reshape(3, 3)
    inertia_world = rotation_matrix @ inertia_body.astype(np.float64) @ rotation_matrix.T
    velocity_angular = joint_qd[0, 3:].astype(np.float64)
    acceleration_angular = joint_qdd[0, 3:].astype(np.float64)
    expected_force = mass * (joint_qdd[0, :3].astype(np.float64) - np.array([0.0, 0.0, -9.81]))
    expected_torque = inertia_world @ acceleration_angular + np.cross(
        velocity_angular, inertia_world @ velocity_angular
    )
    np.testing.assert_allclose(joint_f.numpy()[0, :3], expected_force, atol=3.0e-5, rtol=3.0e-5)
    np.testing.assert_allclose(joint_f.numpy()[0, 3:], expected_torque, atol=3.0e-5, rtol=3.0e-5)

    zero_qdd = wp.zeros_like(qdd)
    joint_bias = wp.empty_like(joint_f)
    evaluator.compute(q, qd, zero_qdd, joint_bias)
    state = model.state()
    state.joint_q.assign(joint_q[0])
    state.joint_qd.assign(joint_qd[0])
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    mass_matrix = newton.eval_mass_matrix(model, state).numpy()[0, :6, :6]
    expected_delta = mass_matrix @ joint_qdd[0] + model.joint_armature.numpy() * joint_qdd[0]
    np.testing.assert_allclose(joint_f.numpy()[0] - joint_bias.numpy()[0], expected_delta, atol=5.0e-5, rtol=5.0e-5)


def test_d6_compound_axis_bias(test, device):
    model, body, inertia_body = _build_d6_body(device)
    joint_q = np.array([[0.31, -0.42, 0.27]], dtype=np.float32)
    joint_qd = np.array([[0.73, -0.48, 0.36]], dtype=np.float32)
    joint_qdd = np.array([[-0.29, 0.61, -0.17]], dtype=np.float32)
    q, qd, qdd, joint_f = _arrays(model, 1, joint_q, joint_qd, joint_qdd)
    newton.dynamics.DynamicsInverse(model, 1).compute(q, qd, qdd, joint_f)

    axis_0 = np.array([1.0, 0.0, 0.0])
    axis_1 = _rotate_vector(axis_0, joint_q[0, 0], np.array([0.0, 1.0, 0.0]))
    axis_2 = _rotate_vector(
        axis_1,
        joint_q[0, 1],
        _rotate_vector(axis_0, joint_q[0, 0], np.array([0.0, 0.0, 1.0])),
    )
    axes = np.stack((axis_0, axis_1, axis_2))
    velocity_angular = joint_qd[0].astype(np.float64) @ axes
    acceleration_angular = joint_qdd[0].astype(np.float64) @ axes
    acceleration_angular += np.cross(axis_0 * joint_qd[0, 0], axis_1) * joint_qd[0, 1]
    acceleration_angular += np.cross(axis_0 * joint_qd[0, 0] + axis_1 * joint_qd[0, 1], axis_2) * joint_qd[0, 2]

    body_q = wp.empty((1, model.body_count), dtype=wp.transform, device=device)
    body_qd = wp.empty((1, model.body_count), dtype=wp.spatial_vector, device=device)
    newton.eval_fk_batched(model, q, qd, body_q, body_qd)
    body_rotation = body_q.numpy()[0, body, 3:7]
    rotation_matrix = np.asarray(wp.quat_to_matrix(wp.quat(*body_rotation)), dtype=np.float64).reshape(3, 3)
    inertia_world = rotation_matrix @ inertia_body.astype(np.float64) @ rotation_matrix.T
    body_torque = inertia_world @ acceleration_angular + np.cross(velocity_angular, inertia_world @ velocity_angular)
    expected = axes @ body_torque + model.joint_armature.numpy() * joint_qdd[0]
    np.testing.assert_allclose(joint_f.numpy()[0], expected, atol=5.0e-5, rtol=5.0e-5)


def test_external_body_force_sign_and_com_torque(test, device):
    model, body, _mass, length, _inertia_z = _build_pendulum(device)
    angle = 0.31
    joint_q = np.array([[angle]], dtype=np.float32)
    q, qd, qdd, joint_f = _arrays(model, 1, joint_q=joint_q)
    body_f = np.zeros((1, model.body_count, 6), dtype=np.float32)
    body_f[0, body, 1] = 7.0
    body_f[0, body, 5] = 0.8

    newton.dynamics.DynamicsInverse(model, 1).compute(
        q, qd, qdd, joint_f, wp.array(body_f, dtype=wp.spatial_vector, device=device)
    )

    gravity = 2.3 * 9.81 * length * np.cos(angle)
    external_generalized = 7.0 * length * np.cos(angle) + 0.8
    np.testing.assert_allclose(joint_f.numpy()[0, 0], gravity - external_generalized, atol=3.0e-5, rtol=3.0e-5)


def test_batch_matches_individual_evaluations(test, device):
    model, _body, _mass, _length, _inertia_z = _build_pendulum(device)
    joint_q = np.array([[-0.5], [-0.1], [0.3], [0.8]], dtype=np.float32)
    joint_qd = np.array([[0.2], [-0.7], [0.4], [0.1]], dtype=np.float32)
    joint_qdd = np.array([[0.9], [-0.3], [1.4], [-0.8]], dtype=np.float32)
    q, qd, qdd, joint_f = _arrays(model, 4, joint_q, joint_qd, joint_qdd)
    newton.dynamics.DynamicsInverse(model, 4).compute(q, qd, qdd, joint_f)

    expected = np.empty((4, 1), dtype=np.float32)
    evaluator = newton.dynamics.DynamicsInverse(model, 1)
    for index in range(4):
        q_one, qd_one, qdd_one, force_one = _arrays(
            model, 1, joint_q[index : index + 1], joint_qd[index : index + 1], joint_qdd[index : index + 1]
        )
        evaluator.compute(q_one, qd_one, qdd_one, force_one)
        expected[index] = force_one.numpy()[0]
    np.testing.assert_allclose(joint_f.numpy(), expected, atol=1.0e-6, rtol=1.0e-6)


def test_cuda_graph_capture(test, device):
    if not device.is_cuda:
        test.skipTest("CUDA graph capture requires a CUDA device")
    model, _body, _mass, _length, _inertia_z = _build_pendulum(device)
    q, qd, qdd, joint_f = _arrays(model, 8)
    evaluator = newton.dynamics.DynamicsInverse(model, 8)
    evaluator.compute(q, qd, qdd, joint_f)
    with wp.ScopedCapture(device) as capture:
        evaluator.compute(q, qd, qdd, joint_f)
    wp.capture_launch(capture.graph)
    test.assertTrue(np.all(np.isfinite(joint_f.numpy())))


class TestDynamicsInverse(unittest.TestCase):
    pass


devices = get_test_devices()
add_function_test(
    TestDynamicsInverse,
    "test_pendulum_gravity_and_acceleration",
    test_pendulum_gravity_and_acceleration,
    devices=devices,
)
add_function_test(TestDynamicsInverse, "test_mass_matrix_identity", test_mass_matrix_identity, devices=devices)
add_function_test(
    TestDynamicsInverse,
    "test_free_root_public_com_world_convention",
    test_free_root_public_com_world_convention,
    devices=devices,
)
add_function_test(
    TestDynamicsInverse,
    "test_external_body_force_sign_and_com_torque",
    test_external_body_force_sign_and_com_torque,
    devices=devices,
)
add_function_test(TestDynamicsInverse, "test_d6_compound_axis_bias", test_d6_compound_axis_bias, devices=devices)
add_function_test(
    TestDynamicsInverse,
    "test_batch_matches_individual_evaluations",
    test_batch_matches_individual_evaluations,
    devices=devices,
)
add_function_test(TestDynamicsInverse, "test_cuda_graph_capture", test_cuda_graph_capture, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
