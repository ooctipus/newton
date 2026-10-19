# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Project velocity-limit rows with the stateless or the accumulated law.

A velocity-limit row is unilateral, ``r = J v + b >= 0``. The stateless law is
PhysX's clamp: each visit applies only the impulse that removes the current
overshoot and stores that correction, so an impulse that later rows made
unnecessary is never released. The accumulated law keeps a non-negative total,
``max(old - r / d, 0)``, and converges to the complementarity solution, which
is independent of the row order and is the fixed point of the parallel tiers.

The CPU tests check the generated sources: the stateless law keeps them (the
point-stream pins hold every family's stateless text), and the accumulated law
reaches every visit of a velocity-limit row in every serial sweep generator.
The CUDA tests run the kernels against a float64 reference.
"""

import inspect
import itertools
import unittest
from unittest import mock

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import mf_gs_source
from newton._src.solvers.feather_pgs import solver_feather_pgs as solver_source
from newton._src.solvers.feather_pgs.kernels import PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT as VELOCITY_LIMIT
from newton._src.solvers.feather_pgs.response_block_sweep import (
    get_response_block_sweep_lane_kernel,
    get_response_block_sweep_warp_kernel,
)
from newton._src.solvers.feather_pgs.sparse_pgs import _get_pgs_solve_sparse_kernel

# Every serial sweep generator with velocity-limit rows: (factory, arguments, options, accumulated sites).
# Each site is the accumulated expression one visit of a velocity-limit row compiles.
_GENERATORS = {
    "mf_gs": (
        solver_source._get_pgs_solve_mf_gs_kernel.__wrapped__,
        (64, 32, 12, "120"),
        {"has_drive_rows": True},
        (
            "fmaxf(old_impulse - residual / denom, 0.0f)",  # dense rows of a velocity-limit row phase
            "fmaxf(old_vlim - residual_vlim / denom, 0.0f)",  # end-of-iteration dense pass
            "fmaxf(old_mf_vlim - residual_mf_vlim * mf_diag, 0.0f)",  # end-of-iteration free-body pass
            "fmaxf(old_impulse - residual * mf_diag, 0.0f)",  # the contact loop's unreachable free-body branch
        ),
    ),
    "mf_gs_without_dense_limits": (
        solver_source._get_pgs_solve_mf_gs_kernel.__wrapped__,
        (64, 32, 12, "120"),
        {"has_drive_rows": False, "has_dense_velocity_limit_rows": False},
        (
            "fmaxf(old_impulse - residual / denom, 0.0f)",
            "fmaxf(old_mf_vlim - residual_mf_vlim * mf_diag, 0.0f)",
            "fmaxf(old_impulse - residual * mf_diag, 0.0f)",
        ),
    ),
    "mf_gs_streamed": (
        solver_source._get_pgs_solve_mf_gs_kernel.__wrapped__,
        (64, 32, 12, "120"),
        {"has_drive_rows": True, "stream_mf_impulses": True},
        (
            "fmaxf(old_impulse - residual / denom, 0.0f)",
            "fmaxf(old_vlim - residual_vlim / denom, 0.0f)",
            "fmaxf(old_mf_vlim - residual_mf_vlim * mf_diag, 0.0f)",
            "fmaxf(old_impulse - residual * mf_diag, 0.0f)",
        ),
    ),
    "mf_gs_global_metadata": (
        solver_source._get_pgs_solve_mf_gs_kernel.__wrapped__,
        (64, 32, 12, "120"),
        {"has_drive_rows": True, "shared_metadata": False, "fuse_vel_limits": True},
        (
            "fmaxf(old_impulse - residual / denom, 0.0f)",
            "fmaxf(old_vlim - residual_vlim / denom, 0.0f)",
            "fmaxf(old_mf_vlim - residual_mf_vlim * mf_diag, 0.0f)",
            "fmaxf(old_impulse - residual * mf_diag, 0.0f)",
        ),
    ),
    "incremental": (
        solver_source._get_pgs_solve_mf_gs_incremental_kernel.__wrapped__,
        (64, 32, 12, "120"),
        {"incremental_rows": 32},
        (
            "fmaxf(old_impulse - residual * s_inv_diag[i], 0.0f)",
            "fmaxf(old_vlim - residual_vlim * s_inv_diag[i], 0.0f)",
        ),
    ),
    "propagation_full_iteration": (
        solver_source._get_pgs_solve_propagation_full_iteration_kernel.__wrapped__,
        (64, 32, 18, 48, 8, 18, "120"),
        {"fuse_vel_limits": True},
        (
            "fmaxf(old_impulse - residual / denom, 0.0f)",
            "fmaxf(old_impulse - residual * mf_diag, 0.0f)",
        ),
    ),
    "response_block_lane": (
        get_response_block_sweep_lane_kernel,
        (64, 32, 18),
        {"lane_rows": 2, "has_drive_rows": False},
        (
            "fmaxf(old_impulse - residual * ca[0], 0.0f)",
            "fmaxf(old_impulse - residual * ca[1], 0.0f)",
            "fmaxf(lam[0] - residual * ca[0], 0.0f)",
            "fmaxf(lam[1] - residual * ca[1], 0.0f)",
        ),
    ),
    "response_block_warp": (
        get_response_block_sweep_warp_kernel,
        (64, 32, 18),
        {"rows": 64, "lane_rows": 16, "has_drive_rows": False},
        (
            "fmaxf(old_impulse - residual * ca_, 0.0f)",
            "fmaxf(old_vlim - residual * ca_, 0.0f)",
        ),
    ),
    "sparse_limits": (
        _get_pgs_solve_sparse_kernel.__wrapped__,
        (48, 18, 12),
        {"velocity_limits": True},
        ("fmaxf(old_vlim - residual / denominator, 0.0f)",),
    ),
    "sparse_free_rows": (
        _get_pgs_solve_sparse_kernel.__wrapped__,
        (48, 18, 12),
        {"mf_max_constraints": 32, "free_row_dofs": 6},
        ("fmaxf(old_impulse - residual * inverse_diagonal, 0.0f)",),
    ),
}

# Text of the stateless law at its visits; none may survive in an accumulated kernel.
_STATELESS_TEXT = (
    "PhysX applies the clamp delta directly",
    "if (residual_vlim < 0.0f)",
    "if (residual_mf_vlim < 0.0f)",
    "(residual < 0.0f) ? delta : 0.0f",
    "(residual < 0.0f) ? -residual",
    "residual < 0.0f ? -residual",
    "delta_impulse = delta;",
    "kind == 4 ? new_impulse",
)


def _native_sources(factory, *args, **kwargs):
    """Build a kernel and return it with the native snippets its factory compiled."""
    snippets = []
    original = wp.func_native

    def capture(snippet, *rest, **options):
        snippets.append(snippet)
        return original(snippet, *rest, **options)

    with mock.patch.object(wp, "func_native", capture):
        kernel = factory(*args, **kwargs)
    return kernel, snippets


class TestVelocityLimitLawSources(unittest.TestCase):
    def test_stateless_law_keeps_the_generated_sources(self):
        """Build every generator's stateless kernel from exactly the source and name it had without the law.

        The point-stream pins hold these sources against the code before the law existed.
        """
        for name, (factory, args, options, _) in _GENERATORS.items():
            with self.subTest(generator=name):
                kernel, sources = _native_sources(factory, *args, **options)
                stateless, stateless_sources = _native_sources(
                    factory, *args, velocity_limit_law="stateless", **options
                )
                self.assertEqual(stateless.key, kernel.key)
                self.assertEqual(stateless_sources, sources)
                self.assertNotIn("_vacc", kernel.key)
        self.assertEqual(
            mf_gs_source.get_source(64, 32, 12, "120", velocity_limit_law="stateless"),
            mf_gs_source.get_source(64, 32, 12, "120"),
        )

    def test_accumulated_law_reaches_every_velocity_limit_visit(self):
        """Project every visit of a velocity-limit row with ``max(old - r / d, 0)`` and name the kernel for it."""
        for name, (factory, args, options, sites) in _GENERATORS.items():
            with self.subTest(generator=name):
                kernel, sources = _native_sources(factory, *args, velocity_limit_law="accumulated", **options)
                text = "\n".join(sources)
                self.assertIn("_vacc", kernel.key)
                for site in sites:
                    self.assertEqual(text.count(site), 1, site)
                for stateless in _STATELESS_TEXT:
                    self.assertNotIn(stateless, text)

    def test_streamed_impulses_read_the_total_once(self):
        """Read a streamed free-body total through one lane-0 broadcast, never a broadcast inside a broadcast."""
        _, sources = _native_sources(
            solver_source._get_pgs_solve_mf_gs_kernel.__wrapped__,
            64,
            32,
            12,
            "120",
            stream_mf_impulses=True,
            velocity_limit_law="accumulated",
        )
        text = "\n".join(sources)
        self.assertIn("const float old_mf_vlim = __shfl_sync(MASK, lane == 0 ? g_lam_mf[i] : 0.0f, 0);", text)
        self.assertNotIn("lane == 0 ? __shfl_sync", text)

    def test_sparse_sweeps_without_limit_rows_ignore_the_law(self):
        """Fold the law away where a sparse sweep holds no velocity-limit row."""
        factory = _get_pgs_solve_sparse_kernel.__wrapped__
        kernel, sources = _native_sources(factory, 48, 18, 12)
        accumulated, accumulated_sources = _native_sources(factory, 48, 18, 12, velocity_limit_law="accumulated")
        self.assertEqual(accumulated.key, kernel.key)
        self.assertEqual(accumulated_sources, sources)

    def test_unknown_laws_are_rejected(self):
        """Reject a law that is neither stateless nor accumulated in every generator."""
        for name, (factory, args, options, _) in _GENERATORS.items():
            with self.subTest(generator=name), self.assertRaisesRegex(ValueError, "velocity_limit_law"):
                factory(*args, velocity_limit_law="clamped", **options)
        with self.assertRaisesRegex(ValueError, "velocity_limit_law"):
            mf_gs_source.get_source(64, 32, 12, "120", velocity_limit_law="clamped")


def serial_velocity_limit_reference(mass, velocity, jacobian, bias, *, law, iterations):
    """Sweep velocity-limit rows in row order with one law in float64; return the velocity and the impulses."""
    response = np.linalg.solve(mass, jacobian.T).T
    diagonal = np.einsum("ij,ij->i", jacobian, response)
    velocity, impulses = np.array(velocity, np.float64), np.zeros(len(jacobian))
    for _ in range(iterations):
        for row in range(len(jacobian)):
            residual = jacobian[row] @ velocity + bias[row]
            if law == "accumulated":
                new = max(impulses[row] - residual / diagonal[row], 0.0)
                change = new - impulses[row]
            else:
                new = change = max(-residual / diagonal[row], 0.0)
            impulses[row] = new
            velocity += response[row] * change
    return velocity, impulses


# A two-link chain whose elbow limit slows the shoulder: braking the elbow (row 1) brings the shoulder inside
# its own limit, so the shoulder row (row 0) carries no impulse at the complementarity solution. The stateless
# law applies the shoulder correction first when row 0 leads and can never release it.
_CHAIN_MASS = np.array([[3.0, -1.4], [-1.4, 1.0]])
_CHAIN_VELOCITY = np.array([1.2, 3.0])
_CHAIN_JACOBIAN = np.array([[-1.0, 0.0], [0.0, -1.0]])  # upper limits: limit - qd >= 0
_CHAIN_BIAS = np.array([1.0, 1.0])
_CHAIN_SOLUTION = np.array([1.2 - 1.4 * 2.0 / 3.0, 1.0])  # elbow at its limit, shoulder 0.2667 inside its own
_CHAIN_OVERBRAKED = np.array([0.1973333333, 1.0])  # stateless, shoulder row first


class TestVelocityLimitLawReference(unittest.TestCase):
    def test_accumulated_rows_reach_the_order_free_solution(self):
        """Reach the complementarity solution in either row order, where the stateless law over-brakes."""
        results = {}
        for law, order in itertools.product(("stateless", "accumulated"), ((0, 1), (1, 0))):
            rows = list(order)
            velocity, impulses = serial_velocity_limit_reference(
                _CHAIN_MASS, _CHAIN_VELOCITY, _CHAIN_JACOBIAN[rows], _CHAIN_BIAS[rows], law=law, iterations=64
            )
            results[law, order] = velocity
            if law == "accumulated":
                np.testing.assert_allclose(velocity, _CHAIN_SOLUTION, atol=1.0e-12)
                np.testing.assert_allclose(impulses[rows.index(0)], 0.0, atol=1.0e-12)
        np.testing.assert_allclose(results["stateless", (0, 1)], _CHAIN_OVERBRAKED, atol=1.0e-9)
        np.testing.assert_allclose(results["stateless", (1, 0)], _CHAIN_SOLUTION, atol=1.0e-12)


def _dense_row_values(device, *, order, row_phase, iterations):
    """Inputs of a matrix-free sweep kernel holding the chain's two velocity-limit rows in ``order``."""
    rows = list(order)
    capacity, dofs = 4, 2
    jacobian = np.zeros((1, capacity, dofs), np.float32)
    jacobian[0, :2] = _CHAIN_JACOBIAN[rows]
    response = np.zeros_like(jacobian)
    response[0, :2] = np.linalg.solve(_CHAIN_MASS, _CHAIN_JACOBIAN[rows].T).T
    diagonal = np.zeros((1, capacity), np.float32)
    diagonal[0, :2] = np.einsum("ij,ij->i", jacobian[0, :2], response[0, :2])
    bias = np.zeros((1, capacity), np.float32)
    bias[0, :2] = _CHAIN_BIAS[rows]
    row_type = np.full((1, capacity), VELOCITY_LIMIT, np.int32)

    def array(value, dtype=wp.float32):
        return wp.array(value, dtype=dtype, device=device)

    zero_int, zero_float = array([0], wp.int32), array([[0.0]])
    zero_spatial = wp.zeros((1, 1, 6), dtype=wp.float32, device=device)
    values = {
        "general_world_count": array([1], wp.int32),
        "general_worlds": zero_int,
        "general_world_grid_stride": 1,
        "use_general_world_queue": 0,
        "rb_class": zero_int,
        "world_constraint_count": array([2], wp.int32),
        # Both rows are the velocity-limit segment of a phase-5 launch.
        "dense_phase_bounds": array([[0, 2]], wp.int32),
        "local_solve_owner": zero_int,
        "world_dof_indices": array([[0, 1]], wp.int32),
        "world_deferred_dof_mask": array([[0, 0]], wp.int32),
        "rhs_bias": array(bias),
        "world_diag": array(diagonal),
        "world_row_w": array(np.ones((1, capacity), np.float32)),
        "world_impulses": array(np.zeros((1, capacity), np.float32)),
        "J_world": array(jacobian),
        "Y_world": array(response),
        "world_row_type": array(row_type, wp.int32),
        "world_row_parent": array(np.full((1, capacity), -1, np.int32), wp.int32),
        "world_row_mu": array(np.zeros((1, capacity), np.float32)),
        "world_torsion_group": array([[-1]], wp.int32),
        "contact_torsion_radius": 0.0,
        "mf_constraint_count": zero_int,
        "mf_contact_rows_end": zero_int,
        "mf_meta": array([[0]], wp.int32),
        "mf_impulses": zero_float,
        "mf_J_a": zero_spatial,
        "mf_J_b": zero_spatial,
        "mf_MiJt_a": zero_spatial,
        "mf_MiJt_b": zero_spatial,
        "mf_row_mu": zero_float,
        "mf_row_w": zero_float,
        "iterations": iterations,
        "omega": 1.0,
        "regularize": 0,
        "row_phase": row_phase,
        "friction_start_iteration": 0,
        "iteration_offset": 0,
        "freeze_drive_rows": 0,
        "defer_dense_response": 0,
        "v_out": array(_CHAIN_VELOCITY.astype(np.float32)),
    }
    for name in ("target_vel_bias", "vel_multiplier", "impulse_multiplier", "max_impulse", "vel_limit"):
        values[f"world_drive_{name}"] = zero_float
    return values


@unittest.skipUnless(wp.is_cuda_available(), "Matrix-free sweep kernels require CUDA")
class TestVelocityLimitLawKernels(unittest.TestCase):
    def test_serial_sweeps_follow_the_reference_in_either_row_order(self):
        """Run the chain's rows through the serial sweeps in both orders and both row phases.

        Phase 0 visits velocity-limit rows in the end-of-iteration pass, phase 5 in
        the main row loop (the velocity-limit phase of the propagation and
        multi-block schedules). Reversing the row order moves the accumulated
        result by at most 1e-6; the stateless law's move documents the
        order-dependent over-braking the accumulated law removes.
        """
        device = wp.get_device("cuda:0")
        arch = str(device.arch)
        factories = {
            "mf_gs": lambda law: solver_source._get_pgs_solve_mf_gs_kernel(
                4, 1, 2, arch, has_drive_rows=False, velocity_limit_law=law
            ),
            "incremental": lambda law: solver_source._get_pgs_solve_mf_gs_incremental_kernel(
                4, 1, 2, arch, incremental_rows=4, has_drive_rows=False, velocity_limit_law=law
            ),
        }
        for (name, factory), law, row_phase in itertools.product(
            factories.items(), ("stateless", "accumulated"), (0, 5)
        ):
            kernel = factory(law)
            velocities = {}
            for order in ((0, 1), (1, 0)):
                values = _dense_row_values(device, order=order, row_phase=row_phase, iterations=64)
                wp.launch_tiled(
                    kernel,
                    dim=[1],
                    inputs=[values[parameter] for parameter in inspect.signature(kernel.func).parameters],
                    block_dim=32,
                    device=device,
                )
                velocities[order] = values["v_out"].numpy().astype(np.float64)
                expected, impulses = serial_velocity_limit_reference(
                    _CHAIN_MASS,
                    _CHAIN_VELOCITY,
                    _CHAIN_JACOBIAN[list(order)],
                    _CHAIN_BIAS[list(order)],
                    law=law,
                    iterations=64,
                )
                with self.subTest(kernel=name, law=law, row_phase=row_phase, order=order):
                    np.testing.assert_allclose(velocities[order], expected, atol=2.0e-6)
                    np.testing.assert_allclose(values["world_impulses"].numpy()[0, :2], impulses, atol=2.0e-6)
            move = float(np.max(np.abs(velocities[(0, 1)] - velocities[(1, 0)])))
            with self.subTest(kernel=name, law=law, row_phase=row_phase):
                if law == "accumulated":
                    self.assertLessEqual(move, 1.0e-6)
                    np.testing.assert_allclose(velocities[(0, 1)], _CHAIN_SOLUTION, atol=2.0e-6)
                else:
                    # The stateless law brakes the shoulder to 0.197 instead of 0.267 when its row leads.
                    self.assertGreater(move, 0.05)


if __name__ == "__main__":
    unittest.main()
