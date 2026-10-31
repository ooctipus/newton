# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Project velocity-limit rows with their law: a non-negative total.

A velocity-limit row is unilateral, ``r = J v + b >= 0``. It keeps a
non-negative total, ``max(old - r / d, 0)``, like a contact normal or a
position limit, and converges to the complementarity solution, which is
independent of the row order and is the fixed point of the parallel tiers.

The CPU tests check that the law reaches every visit of a velocity-limit row in
every serial sweep generator. The CUDA tests run the kernels against a float64
reference and roll out solvers.
"""

import inspect
import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import sweep_routes
from newton._src.solvers.feather_pgs.kernels import PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT as VELOCITY_LIMIT
from newton._src.solvers.feather_pgs.sweep_routes_sparse import sparse_kernel
from newton.solvers import SolverFeatherPGS
from newton.tests.feather_pgs_law_harness import articulation_factor, whitened_rows

# Every serial sweep generator with velocity-limit rows: (factory, arguments, options, law sites).
# Each site is the law call one visit of a velocity-limit row compiles (projection_laws.velocity_limit_update and
# its free-body form speed_row), from the row's total.
_DENSE_PASS = "ProjectionLaw::velocity_limit_update(old_vlim, residual_vlim / denom)"
_SPEED_PASS = "ProjectionLaw::speed_row(speed_mf_vlim, __int_as_float(meta.z), mf_diag, old_mf_vlim)"
_GENERATORS = {
    "mf_gs": (
        sweep_routes.mf_gs_kernel.__wrapped__,
        (64, 32, 12, "120"),
        {},
        # The end-of-iteration dense pass and free-body speed pass.
        (_DENSE_PASS, _SPEED_PASS),
    ),
    "mf_gs_without_dense_limits": (
        sweep_routes.mf_gs_kernel.__wrapped__,
        (64, 32, 12, "120"),
        {"has_dense_velocity_limit_rows": False},
        (_SPEED_PASS,),
    ),
    "mf_gs_streamed": (
        sweep_routes.mf_gs_kernel.__wrapped__,
        (64, 32, 12, "120"),
        {"stream_mf_impulses": True},
        (_DENSE_PASS, _SPEED_PASS),
    ),
    "mf_gs_global_metadata": (
        sweep_routes.mf_gs_kernel.__wrapped__,
        (64, 32, 12, "120"),
        {"shared_metadata": False},
        (_DENSE_PASS, _SPEED_PASS),
    ),
    "sparse_limits": (
        sparse_kernel.__wrapped__,
        (48, 18, 12),
        {"velocity_limits": True},
        (_DENSE_PASS,),
    ),
}


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
    def test_law_reaches_every_velocity_limit_visit(self):
        """Project every visit of a velocity-limit row with the law, from the row's total."""
        for name, (factory, args, options, sites) in _GENERATORS.items():
            with self.subTest(generator=name):
                _, sources = _native_sources(factory, *args, **options)
                text = "\n".join(sources)
                for site in sites:
                    self.assertEqual(text.count(site), 1, site)

    def test_streamed_impulses_read_the_total_once(self):
        """Read a streamed free-body total through one lane-0 broadcast, never a broadcast inside a broadcast."""
        _, sources = _native_sources(
            sweep_routes.mf_gs_kernel.__wrapped__,
            64,
            32,
            12,
            "120",
            stream_mf_impulses=True,
        )
        text = "\n".join(sources)
        self.assertIn("const float old_mf_vlim = READ(lane == 0 ? g_lam_mf[i] : 0.0f, 0);", text)
        self.assertNotIn("lane == 0 ? READ", text)


def serial_velocity_limit_reference(mass, velocity, jacobian, bias, *, iterations):
    """Sweep velocity-limit rows in row order with their law in float64; return the velocity and the impulses."""
    response = np.linalg.solve(mass, jacobian.T).T
    diagonal = np.einsum("ij,ij->i", jacobian, response)
    velocity, impulses = np.array(velocity, np.float64), np.zeros(len(jacobian))
    for _ in range(iterations):
        for row in range(len(jacobian)):
            residual = jacobian[row] @ velocity + bias[row]
            new = max(impulses[row] - residual / diagonal[row], 0.0)
            change = new - impulses[row]
            impulses[row] = new
            velocity += response[row] * change
    return velocity, impulses


# A two-link chain whose elbow limit slows the shoulder: braking the elbow (row 1) brings the shoulder inside
# its own limit, so the shoulder row (row 0) carries no impulse at the complementarity solution. When row 0 leads,
# its first visit brakes the shoulder, and the row releases that impulse once the elbow row brakes.
_CHAIN_MASS = np.array([[3.0, -1.4], [-1.4, 1.0]])
_CHAIN_VELOCITY = np.array([1.2, 3.0])
_CHAIN_JACOBIAN = np.array([[-1.0, 0.0], [0.0, -1.0]])  # upper limits: limit - qd >= 0
_CHAIN_BIAS = np.array([1.0, 1.0])
_CHAIN_SOLUTION = np.array([1.2 - 1.4 * 2.0 / 3.0, 1.0])  # elbow at its limit, shoulder 0.2667 inside its own


class TestVelocityLimitLawReference(unittest.TestCase):
    def test_rows_reach_the_order_free_solution(self):
        """Reach the complementarity solution in either row order, releasing the shoulder's first impulse."""
        for order in ((0, 1), (1, 0)):
            with self.subTest(order=order):
                rows = list(order)
                velocity, impulses = serial_velocity_limit_reference(
                    _CHAIN_MASS, _CHAIN_VELOCITY, _CHAIN_JACOBIAN[rows], _CHAIN_BIAS[rows], iterations=64
                )
                np.testing.assert_allclose(velocity, _CHAIN_SOLUTION, atol=1.0e-12)
                np.testing.assert_allclose(impulses[rows.index(0)], 0.0, atol=1.0e-12)


def _dense_row_values(device, *, order, iterations, first_row=0, mass=_CHAIN_MASS):
    """Inputs of a matrix-free sweep kernel holding the chain's two velocity-limit rows in ``order``.

    The rows are packed (:mod:`.packed_rows`): whitened by the chain's leaf-first factor of ``mass``, at their
    coordinates. ``first_row`` puts that many inactive velocity-limit rows (no Jacobian, zero diagonal) before the
    chain's, which then occupy rows ``first_row`` and ``first_row + 1``.
    """
    rows = list(order)
    count = first_row + 2
    capacity, dofs = max(4, count), 2
    chain = slice(first_row, count)
    factors, factor = articulation_factor(mass, device)
    whitened = np.zeros((1, capacity, dofs))
    whitened[0, chain] = whitened_rows(_CHAIN_JACOBIAN[rows], factor)
    diagonal = np.zeros((1, capacity), np.float32)
    diagonal[0, chain] = np.einsum("ij,ij->i", whitened[0, chain], whitened[0, chain])
    bias = np.zeros((1, capacity), np.float32)
    bias[0, chain] = _CHAIN_BIAS[rows]
    incident = np.zeros((1, capacity), np.float32)
    incident[0, chain] = _CHAIN_JACOBIAN[rows] @ _CHAIN_VELOCITY
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
        "world_constraint_count": array([count], wp.int32),
        # Every row is in the velocity-limit segment.
        "dense_phase_bounds": array([[0, count]], wp.int32),
        "local_solve_owner": zero_int,
        "world_dof_indices": array([[0, 1]], wp.int32),
        "rhs_bias": array(bias),
        "row_incident": array(incident),
        "world_diag": array(diagonal),
        "world_row_w": array(np.ones((1, capacity), np.float32)),
        "world_impulses": array(np.zeros((1, capacity), np.float32)),
        "response_dofs": array([dofs], wp.int32),
        "articulation_world_dof_offset": array([0], wp.int32),
        "packed_dense": 1,
        "packed_Z": array(whitened.astype(np.float32)),
        "row_sides": wp.array(np.tile(np.array([0, -1], np.int32), (1, capacity, 1)), dtype=wp.vec2i, device=device),
        "world_row_type": array(row_type, wp.int32),
        "world_row_parent": array(np.full((1, capacity), -1, np.int32), wp.int32),
        "world_row_mu": array(np.zeros((1, capacity), np.float32)),
        "mf_constraint_count": zero_int,
        "mf_contact_rows_end": zero_int,
        "mf_meta": array([[0]], wp.int32),
        "mf_impulses": zero_float,
        "mf_Z_a": zero_spatial,
        "mf_Z_b": zero_spatial,
        "factors": factors,
        "mf_row_mu": zero_float,
        "mf_row_w": zero_float,
        "iterations": iterations,
        "omega": 1.0,
        "regularize": 0,
        "v_out": array(_CHAIN_VELOCITY.astype(np.float32)),
    }
    return values


def _launch_inputs(kernel, values):
    """Return ``kernel``'s launch inputs from ``values``, by parameter name; every parameter must be in ``values``."""
    return [values[name] for name in inspect.signature(kernel.func).parameters]


@unittest.skipUnless(wp.is_cuda_available(), "Matrix-free sweep kernels require CUDA")
class TestVelocityLimitLawKernels(unittest.TestCase):
    def test_serial_sweep_follows_the_reference_in_either_row_order(self):
        """Run the chain's rows through the serial sweep in both orders.

        The sweep visits velocity-limit rows in the end-of-iteration pass. Each order follows the float64
        reference, and reversing the row order moves the result by at most 1e-6.
        """
        device = wp.get_device("cuda:0")
        kernel = sweep_routes.mf_gs_kernel(4, 1, 2, str(device.arch))
        velocities = {}
        for order in ((0, 1), (1, 0)):
            values = _dense_row_values(device, order=order, iterations=64)
            wp.launch_tiled(kernel, dim=[1], inputs=_launch_inputs(kernel, values), block_dim=32, device=device)
            velocities[order] = values["v_out"].numpy().astype(np.float64)
            expected, impulses = serial_velocity_limit_reference(
                _CHAIN_MASS, _CHAIN_VELOCITY, _CHAIN_JACOBIAN[list(order)], _CHAIN_BIAS[list(order)], iterations=64
            )
            with self.subTest(order=order):
                np.testing.assert_allclose(velocities[order], expected, atol=2.0e-6)
                np.testing.assert_allclose(values["world_impulses"].numpy()[0, :2], impulses, atol=2.0e-6)
        self.assertLessEqual(float(np.max(np.abs(velocities[(0, 1)] - velocities[(1, 0)]))), 1.0e-6)
        np.testing.assert_allclose(velocities[(0, 1)], _CHAIN_SOLUTION, atol=2.0e-6)


def _pushed_arm(device, limit):
    """One revolute link about z without gravity, its velocity limited to ``limit``."""
    builder = newton.ModelBuilder(gravity=0.0)
    body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    builder.add_articulation([builder.add_joint_revolute(-1, body, axis=wp.vec3(0.0, 0.0, 1.0))])
    model = builder.finalize(device=device)
    model.joint_velocity_limit.assign(np.full(model.joint_dof_count, limit, np.float32))
    return model


def _matrix_free_solver(model, *, pins=None, **options):
    settings = {
        "enable_joint_velocity_limits": True,
        "friction_anchor_beta": 0.0,
        "pgs_iterations": 8,
        "dense_max_constraints": 32,
        "mf_max_constraints": 32,
    }
    settings.update(options)
    with mock.patch.dict(SolverFeatherPGS._route_pins, {"storage": "dense", **(pins or {})}):
        return SolverFeatherPGS(model, **settings)


@unittest.skipUnless(wp.is_cuda_available(), "Matrix-free FPGS requires CUDA")
class TestVelocityLimitLawRollouts(unittest.TestCase):
    def test_constant_torque_never_pushes_past_the_limit(self):
        """Hold a revolute DOF that a constant torque pushes past its limit at the limit (the matrix-free GS route)."""
        limit, dt = 1.0, 1.0 / 240.0
        model = _pushed_arm("cuda:0", limit)
        solver = _matrix_free_solver(model)
        self.assertIsNone(solver.get_path_selection()["routes"]["sweep"]["tiers"])
        state, out, control = model.state(), model.state(), model.control()
        control.joint_f.assign(np.array([120.0], np.float32))  # 0.5 rad/s per step
        speeds = []
        for _ in range(8):
            solver.step(state, out, control, None, dt)
            state, out = out, state
            speeds.append(float(abs(state.joint_qd.numpy()[0])))
        self.assertGreater(max(speeds), 0.99 * limit, speeds)
        self.assertLessEqual(max(speeds), limit * (1.0 + 1.0e-5), speeds)

    def test_serial_sweep_converges_to_the_parallel_tier(self):
        """Converge the serial sweep to the parallel tier's fixed point.

        A palm joint carries three two-link fingers, every joint velocity-limited.
        One step from rest with torques that push some joints past their limits and
        leave others inside couples the limit rows through the hand's mass matrix.
        The parallel tier (a test pin: 2000 sweeps) solves the complementarity
        problem; at 2000 iterations the matrix-free GS kernel matches it to 1e-5.
        PhysX's clamp, which keeps each visit's correction instead of a total,
        converged 1.1e-3 away in a float64 reference.
        """
        model = _limited_hand("cuda:0")
        # Unconstrained step velocities (0.05, 0.33, -1.42, -0.45, 2.07, -0.46, 0.62) rad/s: three limits bind at
        # the solution, and the coupling keeps four joints inside theirs.
        torques = np.array([0.03, 0.006, -0.005, 0.0002, 0.008, -0.007, 0.001], np.float32)
        velocities = {}
        for name, pins in (("serial", {}), ("tier", {"tiers": 32, "tier_sweeps": 2000})):
            solver = _matrix_free_solver(model, pins=pins, pgs_iterations=2000)
            self.assertEqual(solver.placement.ink_rows, 32 if name == "tier" else 0)
            state, out, control = model.state(), model.state(), model.control()
            control.joint_f.assign(torques)
            solver.step(state, out, control, None, 1.0 / 240.0)
            counts = solver.placement.constraint_count.numpy()
            self.assertLessEqual(int(counts.max()), 32)
            velocities[name] = out.joint_qd.numpy().astype(np.float64)
        limited = np.abs(velocities["serial"])
        self.assertTrue(np.any(limited >= _HAND_LIMIT * (1.0 - 1.0e-4)), "no limit binds")
        self.assertTrue(np.any(limited <= 0.9 * _HAND_LIMIT), "every limit binds")
        np.testing.assert_allclose(velocities["serial"], velocities["tier"], atol=1.0e-5)


_HAND_LIMIT = 0.5


def _limited_hand(device):
    """A palm joint carrying three two-link fingers in one plane, every joint limited to ``_HAND_LIMIT``."""
    builder = newton.ModelBuilder(gravity=0.0)
    axis = wp.vec3(0.0, 0.0, 1.0)
    palm = builder.add_link(mass=0.4, inertia=wp.mat33(np.eye(3) * 2.0e-3))
    joints = [builder.add_joint_revolute(-1, palm, axis=axis)]
    for finger in range(3):
        parent, offset = palm, wp.vec3(0.03 * (finger - 1), 0.08, 0.0)
        for _ in range(2):
            link = builder.add_link(mass=0.05, inertia=wp.mat33(np.eye(3) * 2.0e-5))
            joints.append(
                builder.add_joint_revolute(
                    parent, link, axis=axis, parent_xform=wp.transform(offset, wp.quat_identity())
                )
            )
            parent, offset = link, wp.vec3(0.0, 0.04, 0.0)
    builder.add_articulation(joints)
    model = builder.finalize(device=device)
    model.joint_velocity_limit.assign(np.full(model.joint_dof_count, _HAND_LIMIT, np.float32))
    return model


if __name__ == "__main__":
    unittest.main()
