# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
import newton.ik as ik
from newton.ik import IKOptimizerLM
from newton._src.sim.ik.ik_lm_optimizer import (
    _lm_global_workspace_bytes,
    _lm_tiled_solve_fits,
    _lm_tiled_solve_shared_memory_bytes,
)
from newton.tests.unittest_utils import add_function_test, assert_np_equal, get_selected_cuda_test_devices


def _build_revolute_chain(n_dofs: int, device: wp.Device) -> newton.Model:
    builder = newton.ModelBuilder()
    parent = -1
    joints = []
    for _ in range(n_dofs):
        link = builder.add_link(xform=wp.transform_identity(), inertia=wp.mat33(np.eye(3)), mass=1.0)
        joint = builder.add_joint_revolute(
            parent=parent,
            child=link,
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform_identity(),
            axis=wp.vec3(0.0, 0.0, 1.0),
        )
        joints.append(joint)
        parent = link
    builder.add_articulation(joints)
    return builder.finalize(device=device, requires_grad=True)


def test_lm_tiled_solve_75_by_126(test, device):
    """The SMPL-sized LM system must launch and match a dense reference solve."""
    with wp.ScopedDevice(device):
        batch = 2
        n_dofs = 75
        n_residuals = 126
        rng = np.random.default_rng(42)
        jacobian_np = rng.normal(size=(batch, n_residuals, n_dofs)).astype(np.float32)
        residuals_np = rng.normal(size=(batch, n_residuals, 1)).astype(np.float32)
        lambda_np = np.array([0.1, 0.3], dtype=np.float32)

        optimizer_type = IKOptimizerLM._build_specialized((n_dofs, n_residuals, device.arch, True))
        optimizer = object.__new__(optimizer_type)
        optimizer.device = device
        jacobian = wp.array(jacobian_np, dtype=wp.float32, device=device)
        residuals = wp.array(residuals_np, dtype=wp.float32, device=device)
        lambda_values = wp.array(lambda_np, dtype=wp.float32, device=device)
        delta = wp.empty((batch, n_dofs), dtype=wp.float32, device=device)
        predicted_reduction = wp.empty(batch, dtype=wp.float32, device=device)

        optimizer._solve_normal_equations(jacobian, residuals, lambda_values, delta, predicted_reduction)

        expected_delta = np.empty((batch, n_dofs), dtype=np.float32)
        expected_reduction = np.empty(batch, dtype=np.float32)
        for row in range(batch):
            gradient = jacobian_np[row].T @ residuals_np[row, :, 0]
            normal = jacobian_np[row].T @ jacobian_np[row] + lambda_np[row] * np.eye(n_dofs)
            expected_delta[row] = np.linalg.solve(normal, -gradient)
            expected_reduction[row] = 0.5 * expected_delta[row] @ (lambda_np[row] * expected_delta[row] - gradient)

        assert_np_equal(delta.numpy(), expected_delta, tol=2.0e-4)
        assert_np_equal(predicted_reduction.numpy(), expected_reduction, tol=2.0e-4)


def test_lm_global_solve_matches_tiled(test, device):
    """The oversized-system fallback must preserve the fused solve equations."""
    with wp.ScopedDevice(device):
        batch = 3
        n_dofs = 7
        n_residuals = 13
        rng = np.random.default_rng(7)
        jacobian_np = rng.normal(size=(batch, n_residuals, n_dofs)).astype(np.float32)
        residuals_np = rng.normal(size=(batch, n_residuals, 1)).astype(np.float32)
        lambda_np = np.array([0.05, 0.2, 0.7], dtype=np.float32)
        jacobian = wp.array(jacobian_np, dtype=wp.float32, device=device)
        residuals = wp.array(residuals_np, dtype=wp.float32, device=device)
        lambda_values = wp.array(lambda_np, dtype=wp.float32, device=device)

        outputs = []
        for use_tiled_solve in (True, False):
            optimizer_type = IKOptimizerLM._build_specialized((n_dofs, n_residuals, device.arch, use_tiled_solve))
            optimizer = object.__new__(optimizer_type)
            optimizer.device = device
            if use_tiled_solve:
                optimizer._lm_normal_matrix = None
                optimizer._lm_gradient = None
                optimizer._lm_zero_diagonal = None
            else:
                optimizer._lm_normal_matrix = wp.empty(batch * n_dofs * n_dofs, dtype=wp.float32, device=device)
                optimizer._lm_gradient = wp.empty(batch * n_dofs, dtype=wp.float32, device=device)
                optimizer._lm_zero_diagonal = wp.zeros(n_dofs, dtype=wp.float32, device=device)
            delta = wp.empty((batch, n_dofs), dtype=wp.float32, device=device)
            predicted_reduction = wp.empty(batch, dtype=wp.float32, device=device)
            optimizer._solve_normal_equations(jacobian, residuals, lambda_values, delta, predicted_reduction)
            outputs.append((delta.numpy(), predicted_reduction.numpy()))

        assert_np_equal(outputs[1][0], outputs[0][0], tol=2.0e-4)
        assert_np_equal(outputs[1][1], outputs[0][1], tol=2.0e-4)


def test_lm_oversized_global_solve_matches_dense_reference(test, device):
    """An automatically oversized LM system must use global memory and remain accurate."""
    with wp.ScopedDevice(device):
        batch = 1
        n_dofs = 160
        n_residuals = 3
        use_tiled_solve = _lm_tiled_solve_fits(n_dofs, n_residuals, device.max_shared_memory_per_block)
        test.assertFalse(use_tiled_solve)
        rng = np.random.default_rng(19)
        jacobian_np = rng.normal(size=(batch, n_residuals, n_dofs)).astype(np.float32)
        residuals_np = rng.normal(size=(batch, n_residuals, 1)).astype(np.float32)
        lambda_np = np.array([0.5], dtype=np.float32)
        optimizer_type = IKOptimizerLM._build_specialized((n_dofs, n_residuals, device.arch, use_tiled_solve))
        optimizer = object.__new__(optimizer_type)
        optimizer.device = device
        optimizer._lm_normal_matrix = wp.empty(batch * n_dofs * n_dofs, dtype=wp.float32, device=device)
        optimizer._lm_gradient = wp.empty(batch * n_dofs, dtype=wp.float32, device=device)
        optimizer._lm_zero_diagonal = wp.zeros(n_dofs, dtype=wp.float32, device=device)
        jacobian = wp.array(jacobian_np, dtype=wp.float32, device=device)
        residuals = wp.array(residuals_np, dtype=wp.float32, device=device)
        lambda_values = wp.array(lambda_np, dtype=wp.float32, device=device)
        delta = wp.empty((batch, n_dofs), dtype=wp.float32, device=device)
        predicted_reduction = wp.empty(batch, dtype=wp.float32, device=device)

        optimizer._solve_normal_equations(jacobian, residuals, lambda_values, delta, predicted_reduction)

        gradient = jacobian_np[0].T @ residuals_np[0, :, 0]
        normal = jacobian_np[0].T @ jacobian_np[0] + lambda_np[0] * np.eye(n_dofs)
        expected_delta = np.linalg.solve(normal, -gradient).astype(np.float32)
        expected_reduction = 0.5 * expected_delta @ (lambda_np[0] * expected_delta - gradient)
        assert_np_equal(delta.numpy()[0], expected_delta, tol=3.0e-4)
        test.assertAlmostEqual(predicted_reduction.numpy()[0], expected_reduction, places=4)


def test_lm_global_workspace_memory_estimate_matches_allocations(test, device):
    """The public estimate must include every fallback workspace allocation."""
    with wp.ScopedDevice(device):
        n_problems = 2
        model = _build_revolute_chain(160, device)
        targets = wp.zeros(n_problems, dtype=wp.vec3, device=device)
        objective = ik.IKObjectivePosition(model.body_count - 1, wp.vec3(), targets)
        estimate = ik.IKSolver.estimate_memory(model, n_problems, [objective], jacobian_mode=ik.IKJacobianType.ANALYTIC)
        solver = ik.IKSolver(model, n_problems, [objective], jacobian_mode=ik.IKJacobianType.ANALYTIC)
        test.assertFalse(solver._impl.USE_TILED_SOLVE)

        arrays = {}

        def add_arrays(value):
            if isinstance(value, wp.array):
                if value.device == model.device:
                    arrays[(str(value.device), value.ptr)] = value.capacity
                    if value.grad is not None:
                        add_arrays(value.grad)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    add_arrays(item)

        for owner in (solver, solver._impl, objective):
            for value in vars(owner).values():
                add_arrays(value)
        arrays.pop((str(targets.device), targets.ptr))
        test.assertEqual(estimate.total_bytes, sum(arrays.values()))


class TestIKOptimizerLM(unittest.TestCase):
    def test_lm_shared_memory_selector_and_workspace(self):
        self.assertEqual(_lm_tiled_solve_shared_memory_bytes(75, 126), 84_880)
        self.assertTrue(_lm_tiled_solve_fits(75, 126, 101_376))
        self.assertTrue(_lm_tiled_solve_fits(75, 126, 84_880))
        self.assertFalse(_lm_tiled_solve_fits(75, 126, 84_879))
        self.assertTrue(_lm_tiled_solve_fits(160, 3, None))
        self.assertFalse(_lm_tiled_solve_fits(160, 3, 101_376))
        self.assertEqual(_lm_global_workspace_bytes(2, 160, 3, None), 0)
        self.assertEqual(
            _lm_global_workspace_bytes(2, 160, 3, 101_376),
            4 * (2 * (160 * 160 + 160) + 160),
        )


cuda_devices = get_selected_cuda_test_devices()

add_function_test(
    TestIKOptimizerLM,
    "test_lm_tiled_solve_75_by_126",
    test_lm_tiled_solve_75_by_126,
    cuda_devices,
)
add_function_test(
    TestIKOptimizerLM,
    "test_lm_global_solve_matches_tiled",
    test_lm_global_solve_matches_tiled,
    cuda_devices[:1],
)
add_function_test(
    TestIKOptimizerLM,
    "test_lm_global_workspace_memory_estimate_matches_allocations",
    test_lm_global_workspace_memory_estimate_matches_allocations,
    cuda_devices[:1],
)
add_function_test(
    TestIKOptimizerLM,
    "test_lm_oversized_global_solve_matches_dense_reference",
    test_lm_oversized_global_solve_matches_dense_reference,
    cuda_devices[:1],
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
