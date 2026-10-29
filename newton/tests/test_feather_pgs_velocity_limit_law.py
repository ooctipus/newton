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
The CUDA tests run the kernels against a float64 reference and roll out solvers.
"""

import ast
import inspect
import itertools
import re
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import solver_feather_pgs as solver_source
from newton._src.solvers.feather_pgs import sweep_routes
from newton._src.solvers.feather_pgs.kernels import PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT as VELOCITY_LIMIT
from newton._src.solvers.feather_pgs.sweep_routes_sparse import sparse_kernel
from newton.solvers import SolverFeatherPGS
from newton.tests.feather_pgs_law_harness import articulation_factor, whitened_rows

# Every serial sweep generator with velocity-limit rows: (factory, arguments, options, accumulated sites).
# Each site is the law call one visit of a velocity-limit row compiles (projection_laws.velocity_limit_update and
# its free-body form speed_row, whose last argument selects the accumulated total).
_DENSE_PHASE = "ProjectionLaw::velocity_limit_update(old_impulse, residual, residual / denom, 1)"
_DENSE_PASS = "ProjectionLaw::velocity_limit_update(old_vlim, residual_vlim, residual_vlim / denom, 1)"
_SPEED_PASS = "ProjectionLaw::speed_row(speed_mf_vlim, __int_as_float(meta.z), mf_diag, old_mf_vlim, 1)"
_GENERATORS = {
    "mf_gs": (
        sweep_routes.mf_gs_kernel.__wrapped__,
        (64, 32, 12, "120"),
        {"has_drive_rows": True},
        # Dense rows of a velocity-limit row phase, the end-of-iteration dense pass and free-body speed pass.
        (_DENSE_PHASE, _DENSE_PASS, _SPEED_PASS),
    ),
    "mf_gs_without_dense_limits": (
        sweep_routes.mf_gs_kernel.__wrapped__,
        (64, 32, 12, "120"),
        {"has_drive_rows": False, "has_dense_velocity_limit_rows": False},
        (_DENSE_PHASE, _SPEED_PASS),
    ),
    "mf_gs_streamed": (
        sweep_routes.mf_gs_kernel.__wrapped__,
        (64, 32, 12, "120"),
        {"has_drive_rows": True, "stream_mf_impulses": True},
        (_DENSE_PHASE, _DENSE_PASS, _SPEED_PASS),
    ),
    "mf_gs_global_metadata": (
        sweep_routes.mf_gs_kernel.__wrapped__,
        (64, 32, 12, "120"),
        {"has_drive_rows": True, "shared_metadata": False, "fuse_vel_limits": True},
        (_DENSE_PHASE, _DENSE_PASS, _SPEED_PASS),
    ),
    "incremental": (
        sweep_routes.incremental_kernel.__wrapped__,
        (64, 32, 12, "120"),
        {"incremental_rows": 32},
        (
            "ProjectionLaw::velocity_limit_update(old_impulse, residual, residual * s_inv_diag[i], 1)",
            "ProjectionLaw::velocity_limit_update(s_lam[i], residual_vlim, residual_vlim * s_inv_diag[i], 1)",
        ),
    ),
    "sparse_limits": (
        sparse_kernel.__wrapped__,
        (48, 18, 12),
        {"velocity_limits": True},
        ("ProjectionLaw::velocity_limit_update(old_vlim, residual_vlim, residual_vlim / denom, 1)",),
    ),
}

# The stateless law at a visit: a law call that selects it; none may survive in an accumulated kernel.
_STATELESS_CALL = re.compile(r"(velocity_limit_update|speed_row)\([^;]*, 0\);")


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
            sweep_routes.mf_gs_source(64, 32, 12, velocity_limit_law="stateless"),
            sweep_routes.mf_gs_source(64, 32, 12),
        )

    def test_accumulated_law_reaches_every_velocity_limit_visit(self):
        """Project every visit of a velocity-limit row with the accumulated law and name the kernel for it."""
        for name, (factory, args, options, sites) in _GENERATORS.items():
            with self.subTest(generator=name):
                kernel, sources = _native_sources(factory, *args, velocity_limit_law="accumulated", **options)
                text = "\n".join(sources)
                self.assertIn("_vacc", kernel.key)
                for site in sites:
                    self.assertEqual(text.count(site), 1, site)
                self.assertIsNone(_STATELESS_CALL.search(text))

    def test_streamed_impulses_read_the_total_once(self):
        """Read a streamed free-body total through one lane-0 broadcast, never a broadcast inside a broadcast."""
        _, sources = _native_sources(
            sweep_routes.mf_gs_kernel.__wrapped__,
            64,
            32,
            12,
            "120",
            stream_mf_impulses=True,
            velocity_limit_law="accumulated",
        )
        text = "\n".join(sources)
        self.assertIn("const float old_mf_vlim = READ(lane == 0 ? g_lam_mf[i] : 0.0f, 0);", text)
        self.assertNotIn("lane == 0 ? READ", text)

    def test_sparse_sweeps_without_limit_rows_ignore_the_law(self):
        """Fold the law away where a sparse sweep holds no velocity-limit row."""
        factory = sparse_kernel.__wrapped__
        kernel, sources = _native_sources(factory, 48, 18, 12)
        accumulated, accumulated_sources = _native_sources(factory, 48, 18, 12, velocity_limit_law="accumulated")
        self.assertEqual(accumulated.key, kernel.key)
        self.assertEqual(accumulated_sources, sources)

    def test_solver_passes_its_law_to_every_sweep_it_builds(self):
        """Pass ``velocity_limit_law`` at every call of a sweep factory that takes one in the solver module.

        The solver builds each matrix-free sweep with ``_velocity_limit_kernel_law()``, so its kernels apply the
        law that ``routes.velocity_limits.law`` reports. A call that omits the argument builds the stateless law
        under either option. A sweep built through ``_sweep_kernel(route, factory, ...)``, which applies the solve
        policy, is a call of its factory.
        """
        factories = set()
        for module_name, module in vars(solver_source).items():
            if not inspect.ismodule(module) or not module_name.startswith("sweep_routes"):
                continue
            for name, value in vars(module).items():
                if inspect.isfunction(getattr(value, "__wrapped__", value)) and "velocity_limit_law" in (
                    inspect.signature(value).parameters
                ):
                    factories.add(f"{module_name}.{name}")
        self.assertLessEqual(
            {"sweep_routes.mf_gs_kernel", "sweep_routes.incremental_kernel", "sweep_routes_sparse.sparse_kernel"},
            factories,
        )

        def dotted(node):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                return f"{node.value.id}.{node.attr}"
            return node.id if isinstance(node, ast.Name) else None

        calls, missing = 0, []
        for node in ast.walk(ast.parse(inspect.getsource(solver_source))):
            if not isinstance(node, ast.Call):
                continue
            name = dotted(node.func)
            if name == "self._sweep_kernel" and len(node.args) > 1:
                # _sweep_kernel(route, factory, *args, **kwargs) calls the factory under the solve policy.
                name = dotted(node.args[1])
            if name not in factories:
                continue
            calls += 1
            if "velocity_limit_law" not in {keyword.arg for keyword in node.keywords}:
                missing.append(f"{name} at line {node.lineno}")
        self.assertGreaterEqual(calls, 3)
        self.assertEqual(missing, [])

    def test_unknown_laws_are_rejected(self):
        """Reject a law that is neither stateless nor accumulated in every generator and the solver."""
        for name, (factory, args, options, _) in _GENERATORS.items():
            with self.subTest(generator=name), self.assertRaisesRegex(ValueError, "velocity_limit_law"):
                factory(*args, velocity_limit_law="clamped", **options)
        with self.assertRaisesRegex(ValueError, "velocity_limit_law"):
            sweep_routes.mf_gs_source(64, 32, 12, velocity_limit_law="clamped")


def _cpu_arm():
    builder = newton.ModelBuilder()
    body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    builder.add_articulation([builder.add_joint_revolute(-1, body)])
    return builder.finalize(device="cpu")


def _law_probe(**changes):
    """A stand-in with the attributes the sweep-law decision reads: a matrix-free solver with joint limits."""
    probe = {
        "_matrix_free": True,
        "pgs_schedule": "interleaved",
        "enable_joint_velocity_limits": True,
        "_has_rigid_body_velocity_limits": False,
        "_free_rigid_body_count": 0,
        "velocity_limit_activation_fraction": 0.0,
        "velocity_limit_law": "accumulated",
    }
    probe.update(changes)
    return SimpleNamespace(**probe)


class TestVelocityLimitLawOption(unittest.TestCase):
    def test_sweep_law_follows_the_schedule_and_the_rows(self):
        """Build the requested law where velocity-limit rows exist, except under PhysX's grasp schedule."""
        decide = SolverFeatherPGS._velocity_limit_sweep_law
        for changes, law, words in (
            ({}, "accumulated", "velocity_limit_law='accumulated'"),
            ({"velocity_limit_law": "stateless"}, "stateless", "velocity_limit_law='stateless'"),
            ({"pgs_schedule": "contact_then_internal"}, "accumulated", "velocity_limit_law="),
            ({"pgs_schedule": "physx_grasp"}, "stateless", "physx_grasp"),
            ({"enable_joint_velocity_limits": False}, "stateless", "No velocity-limit rows"),
            # Free-body velocity limits are rows of their own.
            (
                {
                    "enable_joint_velocity_limits": False,
                    "_has_rigid_body_velocity_limits": True,
                    "_free_rigid_body_count": 1,
                },
                "accumulated",
                "velocity_limit_law=",
            ),
            ({"velocity_limit_activation_fraction": float("inf")}, "stateless", "No velocity-limit rows"),
            ({"_matrix_free": False}, None, "CPU's split sweeps"),
        ):
            with self.subTest(**{key: str(value) for key, value in changes.items()}):
                probe = _law_probe(**changes)
                selected, reason = decide(probe)
                self.assertEqual(selected, law)
                self.assertIn(words, reason)

    def test_option_records_the_requested_and_the_applied_law(self):
        """Default to the stateless law, record the request, and apply none on the CPU.

        The applied option and ``routes.velocity_limits.law`` agree: the CPU's split mode builds
        no matrix-free sweep, so it applies neither law.
        """
        model = _cpu_arm()
        default = SolverFeatherPGS(model)
        accumulated = SolverFeatherPGS(model, velocity_limit_law="accumulated")
        self.assertEqual(default.velocity_limit_law, "stateless")
        self.assertEqual(accumulated.velocity_limit_law, "accumulated")
        for solver, requested in ((default, "stateless"), (accumulated, "accumulated")):
            selected = solver.get_path_selection()
            self.assertEqual(
                selected["options"]["velocity_limit_law"],
                {
                    "applied": None,
                    "requested": requested,
                    "default": "stateless",
                    "source": "default" if requested == "stateless" else "argument",
                },
            )
            limits = selected["routes"]["velocity_limits"]
            self.assertIsNone(limits["law"])
            self.assertIn("CPU's split sweeps", limits["law_reason"])
        with self.assertRaisesRegex(ValueError, "velocity_limit_law"):
            SolverFeatherPGS(model, velocity_limit_law="clamped")


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


def _dense_row_values(device, *, order, row_phase, iterations, first_row=0, mass=_CHAIN_MASS):
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
        # Every row is in the velocity-limit segment of a phase-5 launch.
        "dense_phase_bounds": array([[0, count]], wp.int32),
        "local_solve_owner": zero_int,
        "world_dof_indices": array([[0, 1]], wp.int32),
        "rhs_bias": array(bias),
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
        "world_torsion_group": array([[-1]], wp.int32),
        "contact_torsion_radius": 0.0,
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
        "row_phase": row_phase,
        "friction_start_iteration": 0,
        "iteration_offset": 0,
        "freeze_drive_rows": 0,
        "v_out": array(_CHAIN_VELOCITY.astype(np.float32)),
    }
    for name in ("target_vel_bias", "vel_multiplier", "impulse_multiplier", "max_impulse", "vel_limit"):
        values[f"world_drive_{name}"] = zero_float
    return values


def _launch_inputs(kernel, values):
    """Return ``kernel``'s launch inputs from ``values``, by parameter name; every parameter must be in ``values``."""
    return [values[name] for name in inspect.signature(kernel.func).parameters]


@unittest.skipUnless(wp.is_cuda_available(), "Matrix-free sweep kernels require CUDA")
class TestVelocityLimitLawKernels(unittest.TestCase):
    def _check_chain(self, values, velocities, *, order, law, first_row=0, **labels):
        """Compare one launch on the chain with the float64 reference sweep and keep its velocity."""
        velocities[order] = values["v_out"].numpy().astype(np.float64)
        expected, impulses = serial_velocity_limit_reference(
            _CHAIN_MASS, _CHAIN_VELOCITY, _CHAIN_JACOBIAN[list(order)], _CHAIN_BIAS[list(order)], law=law, iterations=64
        )
        chain = values["world_impulses"].numpy()[0, first_row : first_row + 2]
        with self.subTest(law=law, order=order, **labels):
            np.testing.assert_allclose(velocities[order], expected, atol=2.0e-6)
            np.testing.assert_allclose(chain, impulses, atol=2.0e-6)

    def _check_order_move(self, velocities, *, law, **labels):
        """Hold the accumulated result in either row order; record the stateless law's order dependence."""
        move = float(np.max(np.abs(velocities[(0, 1)] - velocities[(1, 0)])))
        with self.subTest(law=law, **labels):
            if law == "accumulated":
                self.assertLessEqual(move, 1.0e-6)
                np.testing.assert_allclose(velocities[(0, 1)], _CHAIN_SOLUTION, atol=2.0e-6)
            else:
                # The stateless law brakes the shoulder to 0.197 instead of 0.267 when its row leads.
                self.assertGreater(move, 0.05)

    def test_serial_sweeps_follow_the_reference_in_either_row_order(self):
        """Run the chain's rows through the serial sweeps in both orders and both row phases.

        Phase 0 visits velocity-limit rows in the end-of-iteration pass, phase 5 in
        the main row loop (the velocity-limit phase of the PhysX-grasp
        schedule). Reversing the row order moves the accumulated
        result by at most 1e-6; the stateless law's move documents the
        order-dependent over-braking the accumulated law removes.
        """
        device = wp.get_device("cuda:0")
        arch = str(device.arch)
        factories = {
            "mf_gs": lambda law: sweep_routes.mf_gs_kernel(4, 1, 2, arch, has_drive_rows=False, velocity_limit_law=law),
            "incremental": lambda law: sweep_routes.incremental_kernel(
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
                wp.launch_tiled(kernel, dim=[1], inputs=_launch_inputs(kernel, values), block_dim=32, device=device)
                self._check_chain(values, velocities, order=order, law=law, kernel=name, row_phase=row_phase)
            self._check_order_move(velocities, law=law, kernel=name, row_phase=row_phase)


def _pushed_arm(device, limit):
    """One revolute link about z without gravity, its velocity limited to ``limit``."""
    builder = newton.ModelBuilder(gravity=0.0)
    body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    builder.add_articulation([builder.add_joint_revolute(-1, body, axis=wp.vec3(0.0, 0.0, 1.0))])
    model = builder.finalize(device=device)
    model.joint_velocity_limit.assign(np.full(model.joint_dof_count, limit, np.float32))
    return model


# Matrix-free routes that sweep dense velocity-limit rows. Sparse factors roll out in
# test_feather_pgs_sparse_velocity_limits.
_DENSE_ROUTES = {
    "mf_gs": {"mf_gs_incremental_rows": 0},
    "incremental": {"mf_gs_incremental_rows": 32},
}


def _matrix_free_solver(model, law, **options):
    settings = {
        "enable_joint_velocity_limits": True,
        "friction_anchor_beta": 0.0,
        "pgs_iterations": 8,
        "dense_max_constraints": 32,
        "mf_max_constraints": 32,
        "velocity_limit_law": law,
    }
    settings.update(options)
    with mock.patch.dict(SolverFeatherPGS._route_pins, {"storage": "dense", "evaluator": "ordinary"}):
        return SolverFeatherPGS(model, **settings)


@unittest.skipUnless(wp.is_cuda_available(), "Matrix-free FPGS requires CUDA")
class TestVelocityLimitLawRollouts(unittest.TestCase):
    def test_constant_torque_never_pushes_past_the_limit(self):
        """Hold a revolute DOF that a constant torque pushes past its limit at the limit under either law.

        Every sweep kernel of the route is built with the requested law: the
        matrix-free GS kernel and the incremental tiers.
        """
        limit, dt = 1.0, 1.0 / 240.0
        model = _pushed_arm("cuda:0", limit)
        for (route, options), law in itertools.product(_DENSE_ROUTES.items(), ("stateless", "accumulated")):
            with self.subTest(route=route, law=law):
                solver = _matrix_free_solver(model, law, **options)
                routes = solver.get_path_selection()["routes"]
                self.assertEqual(routes["velocity_limits"]["law"], law)
                tiers = routes["sweep"]["tiers"]
                self.assertEqual(tiers is not None and tiers["family"] == "incremental", route == "incremental")
                kernels = [routes["sweep"]["mf_gs"]]
                if route == "incremental":
                    kernels += [tier["kernel"] for tier in tiers["tiers"]]
                for key in kernels:
                    self.assertEqual("_vacc" in key, law == "accumulated", key)
                state, out, control = model.state(), model.state(), model.control()
                control.joint_f.assign(np.array([120.0], np.float32))  # 0.5 rad/s per step
                speeds = []
                for _ in range(8):
                    solver.step(state, out, control, None, dt)
                    state, out = out, state
                    speeds.append(float(abs(state.joint_qd.numpy()[0])))
                self.assertGreater(max(speeds), 0.99 * limit, speeds)
                self.assertLessEqual(max(speeds), limit * (1.0 + 1.0e-5), speeds)

    def test_accumulated_serial_sweep_converges_to_the_parallel_tier(self):
        """Converge every serial sweep route under the accumulated law to the parallel tier's fixed point.

        A palm joint carries three two-link fingers, every joint velocity-limited.
        One step from rest with torques that push some joints past their limits and
        leave others inside couples the limit rows through the hand's mass matrix.
        The parallel tier (tolerance 0, 2000 sweeps) solves the complementarity
        problem; at 2000 iterations the matrix-free GS kernel and the incremental
        tier match it to 1e-5 under the accumulated law. The stateless serial
        sweep converges more than 1e-4 away (1.1e-3 in a float64 reference), so
        a route built with the stateless law cannot pass.
        """
        model = _limited_hand("cuda:0")
        # Unconstrained step velocities (0.05, 0.33, -1.42, -0.45, 2.07, -0.46, 0.62) rad/s: three limits bind at
        # the solution, and the coupling keeps four joints inside theirs.
        torques = np.array([0.03, 0.006, -0.005, 0.0002, 0.008, -0.007, 0.001], np.float32)
        converged = {"pgs_iterations": 2000, "mf_gs_parallel_sweeps": 2000}
        velocities = {}
        for name, law, options in (
            ("serial_stateless", "stateless", {"mf_gs_incremental_rows": 0}),
            ("serial", "accumulated", {"mf_gs_incremental_rows": 0}),
            ("incremental", "accumulated", {"mf_gs_incremental_rows": 32}),
            ("tier", "accumulated", {"mf_gs_incremental_rows": 0, "mf_gs_parallel_rows": 32}),
            (
                "tier_matrix_free",
                "accumulated",
                {"mf_gs_incremental_rows": 0, "mf_gs_parallel_rows": 32, "mf_gs_parallel_matrix_free": True},
            ),
        ):
            solver = _matrix_free_solver(model, law, **converged, **options)
            self.assertEqual(solver._ink_rows, 32 if name.startswith("tier") else 0)
            # Each serial route sweeps the hand's rows with kernels of the requested law.
            if name == "incremental":
                route_kernels = [tier.kernel for tier in solver._mf_gs_tiers]
                self.assertTrue(route_kernels)
            else:
                route_kernels = [solver._pgs_solve_mf_gs_kernel]
            for kernel in route_kernels:
                self.assertEqual("_vacc" in kernel.key, law == "accumulated", kernel.key)
            state, out, control = model.state(), model.state(), model.control()
            control.joint_f.assign(torques)
            solver.step(state, out, control, None, 1.0 / 240.0)
            counts = solver.constraint_count.numpy()
            self.assertLessEqual(int(counts.max()), 32)
            velocities[name] = out.joint_qd.numpy().astype(np.float64)
        limited = np.abs(velocities["serial"])
        self.assertTrue(np.any(limited >= _HAND_LIMIT * (1.0 - 1.0e-4)), "no limit binds")
        self.assertTrue(np.any(limited <= 0.9 * _HAND_LIMIT), "every limit binds")
        stateless_gap = float(np.max(np.abs(velocities["serial_stateless"] - velocities["tier"])))
        # The comparison below discriminates the laws only while the stateless law lands elsewhere.
        self.assertGreater(stateless_gap, 1.0e-4)
        for route in ("serial", "incremental", "tier_matrix_free"):
            np.testing.assert_allclose(
                velocities[route],
                velocities["tier"],
                atol=1.0e-5,
                err_msg=f"{route}; the stateless serial sweep lies {stateless_gap:.3e} from the tier",
            )


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
