# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Count the Gauss-Seidel sweeps every solve executes without changing a computed bit.

Under ``pgs_solve_readout="sweeps"`` each sweep kernel records, per world, the
sweeps a solve executed and whether it stopped before its budget. A sweep
stops a world after a sweep that changes no impulse, so a world that stops
after ``s`` sweeps has the same impulses and velocities after ``s - 1`` and
after ``s`` sweeps. The kernel tests replay a solve at every sweep budget and
check the counts against that rule; the solver tests check that every route's
readout leaves the trajectory bitwise unchanged and counts each world-solve
once.

``pgs_velocity_change_tolerance`` stops a solve earlier, after the first sweep
whose velocity-scaled changes fit the tolerance. Before that sweep it runs
exactly the exact-law sweeps, so its outputs equal the exact law's at that
budget, and the trace of an exact solve predicts that sweep for every grid
tolerance. On the solver routes the trace changes no result and records every
solve once per grid tolerance, and a tolerance reaches every counted sweep.
"""

import unittest
import warnings
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
    PGS_CONSTRAINT_TYPE_MIMIC,
    PGS_LOCAL_SOLVE_OWNER_SINGLE,
)
from newton._src.solvers.feather_pgs.solve_policy import (
    SOLVE_PASSES,
    TRACE_GRID,
    TRACE_KINDS,
    TRACE_REASONS,
    SolvePolicy,
    SolveReadout,
    SolveTrace,
)
from newton._src.solvers.feather_pgs.solver_feather_pgs import _get_pgs_solve_local_owned_kernel
from newton._src.solvers.feather_pgs.sparse_mass_matrix import _SparseRowSupport
from newton._src.solvers.feather_pgs.sparse_pgs import _get_pgs_solve_sparse_kernel
from newton.solvers import SolverFeatherPGS

SWEEPS = SolvePolicy(readout="sweeps")
STATE_FIELDS = ("joint_q", "joint_qd", "body_q", "body_qd")
BUDGET = 8


def _readout(device, worlds, budget=BUDGET):
    """A fresh counter operand for ``worlds`` worlds and a budget of ``budget`` sweeps."""
    operand = SolveReadout()
    operand.sweeps = wp.zeros((budget + 1, worlds), dtype=wp.int32, device=device)
    operand.exhausted = wp.zeros(worlds, dtype=wp.int32, device=device)
    return operand


def _trace(device, worlds, budget=BUDGET):
    """A fresh trace operand for ``worlds`` worlds and a budget of ``budget`` sweeps."""
    operand = SolveTrace()
    operand.sweeps = wp.zeros((budget + 1, worlds), dtype=wp.int32, device=device)
    operand.exhausted = wp.zeros(worlds, dtype=wp.int32, device=device)
    operand.stops = wp.zeros((len(TRACE_GRID), budget + 2), dtype=wp.int64, device=device)
    operand.reasons = wp.zeros(len(TRACE_REASONS), dtype=wp.int64, device=device)
    operand.limits = wp.zeros((len(TRACE_KINDS), 26), dtype=wp.int64, device=device)
    return operand


def _violation_problem():
    """Two contact rows on one coordinate: the first one's update violates the second, unselected at entry.

    In ``_problem``'s layout: identity factor, rows ``[1, 0, 0]`` and ``[-3, 1, 0]``,
    incident velocity zero and bias ``[-1, 0.5]``. The first sweep visits only the
    first row (the second is satisfied at entry); its velocity-scaled change is 1
    and leaves the second row violated by 2.5.
    """
    rows = np.array([[1.0, 0.0, 0.0], [-3.0, 1.0, 0.0]])
    types = np.full(2, PGS_CONSTRAINT_TYPE_CONTACT, dtype=np.int32)
    diagonal = np.sum(rows * rows, axis=1) + 1.0e-6
    zeros = np.zeros(2)
    return (
        np.eye(3),
        np.arange(3),
        rows,
        np.zeros(3),
        np.array([-1.0, 0.5]),
        types,
        np.full(2, -1, dtype=np.int32),
        zeros,
        zeros,
        rows,
        rows,
        zeros,
        diagonal,
    )


def _expected_counts(outputs, *, offset=0, friction_start=0):
    """Return ``(executed, stopped)`` per world from per-budget outputs ``outputs[s][world]``.

    A world stops after the first sweep ``s`` that leaves its outputs bitwise
    unchanged, once friction rows are active (global sweep index at least
    ``friction_start``); otherwise it executes the whole budget.
    """
    budget = len(outputs) - 1
    executed, stopped = [], []
    for world in range(len(outputs[0])):
        stop = next(
            (
                s
                for s in range(1, budget + 1)
                if offset + s - 1 >= friction_start
                and all(np.array_equal(a, b) for a, b in zip(outputs[s][world], outputs[s - 1][world], strict=True))
            ),
            None,
        )
        executed.append(budget if stop is None else stop)
        stopped.append(stop is not None)
    return executed, stopped


def _assert_counts(test, operand, executed, stopped, solved, *, repeats=1):
    """Check one record of ``repeats`` identical solves per world in ``solved``, none elsewhere."""
    sweeps, exhausted = operand.sweeps.numpy(), operand.exhausted.numpy()
    for world, (count, stop) in enumerate(zip(executed, stopped, strict=True)):
        with test.subTest(world=world):
            expected = np.zeros(sweeps.shape[0], dtype=np.int32)
            if world in solved:
                expected[count] = repeats
            np.testing.assert_array_equal(sweeps[:, world], expected)
            test.assertEqual(exhausted[world], repeats if world in solved and not stop else 0)


class _SparseCase:
    """One sparse sweep problem in three worlds (the middle one without rows) and its launch buffers."""

    def __init__(self, device, problem, options, friction_start, offset, omega):
        _, indices, rows, _, bias, types, parents, mu, impulses, _, _, incident, diagonal = problem
        if options.get("active_rows"):
            parents[types == 0] = -1  # A linked normal keeps the full sweep.
        count, worlds, dofs, support = len(types), 3, rows.shape[1], len(indices)
        capacity = count + 2
        self.shape, self.options, self.device = (capacity, dofs, support), options, device

        def padded(values, dtype=np.float32):
            result = np.zeros((worlds, capacity), dtype=dtype)
            result[:, : len(values)] = values
            return wp.array(result, device=device)

        row_dof = np.full((worlds, capacity, support), -1, dtype=np.int32)
        row_dof[:, :count] = indices
        row_factor = np.zeros((worlds, capacity, support), dtype=np.float32)
        row_factor[:, :count] = rows[:, indices]
        self.initial = padded(impulses)
        self.impulses = wp.clone(self.initial)
        self.delta = wp.full((worlds, dofs), float("nan"), device=device)
        self.head = [worlds, wp.array([count, 0, count], dtype=int, device=device), padded(bias), padded(diagonal)]
        self.tail = [
            wp.array(row_dof, device=device),
            wp.array(row_factor, device=device),
            wp.zeros((worlds, capacity, 12), device=device),
            padded(incident),
            padded(types, np.int32),
            padded(parents, np.int32),
            padded(mu),
            wp.zeros((worlds, dofs), dtype=int, device=device),
            wp.array(np.arange(worlds * dofs).reshape(worlds, dofs), dtype=int, device=device),
            wp.zeros(worlds * dofs, device=device),
            wp.zeros(worlds, dtype=int, device=device),
            wp.zeros((worlds, 4), dtype=int, device=device),
            wp.zeros((worlds, 1), device=device),
            *[wp.zeros((worlds, 1, 6), device=device) for _ in range(4)],
            wp.zeros((worlds, 1), device=device),
        ]
        self.solve = (omega, 0.0, 0.0, friction_start, offset)
        worlds_per_block = 64 // options.get("lanes_per_world", 32)
        self.blocks = (worlds + worlds_per_block - 1) // worlds_per_block
        self.worlds = worlds

    def kernel(self, policy=None):
        options = self.options if policy is None else {**self.options, "solve_policy": policy}
        return _get_pgs_solve_sparse_kernel(*self.shape, **options)

    def run(self, kernel, iterations, *extra):
        """Restore the initial impulses and sweep; capturable."""
        wp.copy(self.impulses, self.initial)
        wp.launch_tiled(
            kernel,
            dim=[self.blocks],
            inputs=[*self.head, self.impulses, *self.tail, iterations, *self.solve],
            outputs=[self.delta, _SparseRowSupport(), *extra],
            block_dim=64,
            device=self.device,
        )

    def read(self):
        """Per-world outputs: the factor velocity change and the impulses."""
        return list(zip(self.delta.numpy(), self.impulses.numpy(), strict=True))


class _LocalCase:
    """A local owner's three-coordinate problem: world 0 owns a mimic and a joint-limit row, world 1 is not owned."""

    def __init__(self, device):
        dofs, rows, local_rows = 3, 4, 2
        self.shape, self.device, self.worlds = (rows, local_rows, dofs), device, 2
        lower = np.array([[2.0, 0.0, 0.0], [0.25, 1.5, 0.0], [-0.1, 0.2, 1.25]], dtype=np.float32)
        jacobian = np.zeros((2, rows, dofs), dtype=np.float32)
        jacobian[0, 0] = [1.0, 0.5, -0.25]
        jacobian[0, 1] = [-0.4, 0.3, 0.8]
        jacobian[1, 0] = [0.2, -0.1, 0.7]
        rhs = np.zeros((2, rows), dtype=np.float32)
        rhs[0, :2] = [-0.2, 0.15]
        cfm = np.zeros((2, rows), dtype=np.float32)
        cfm[0, :2] = [0.01, 0.02]
        row_type = np.zeros((2, rows), dtype=np.int32)
        row_type[0, :2] = [PGS_CONSTRAINT_TYPE_MIMIC, PGS_CONSTRAINT_TYPE_JOINT_LIMIT]
        row_type[1, 0] = PGS_CONSTRAINT_TYPE_MIMIC
        candidates = wp.array([0, 1], dtype=wp.int32, device=device)
        lower_group = wp.array(np.stack([lower, np.eye(dofs, dtype=np.float32)]), dtype=wp.float32, device=device)
        jacobian_group = wp.array(jacobian, dtype=wp.float32, device=device)
        # Outputs: the diagonal (initially the compliance), the impulses and the articulations' velocities.
        self.initial = (
            wp.array(cfm, dtype=wp.float32, device=device),
            wp.zeros((2, rows), dtype=wp.float32, device=device),
            wp.array([0.3, -0.5, 0.1, 9.0, 8.0, 7.0], dtype=wp.float32, device=device),
        )
        self.outputs = tuple(wp.clone(array) for array in self.initial)
        self.inputs = [
            candidates,
            candidates,
            PGS_LOCAL_SOLVE_OWNER_SINGLE,
            wp.array([0, 1], dtype=wp.int32, device=device),
            wp.array([0, 1], dtype=wp.int32, device=device),
            wp.array([0, 3], dtype=wp.int32, device=device),
            # World 1 belongs to another owner: its solve returns before any sweep.
            wp.array([PGS_LOCAL_SOLVE_OWNER_SINGLE, 0], dtype=wp.int32, device=device),
            wp.array([2, 1], dtype=wp.int32, device=device),
            lower_group,
            jacobian_group,
            lower_group,
            jacobian_group,
            wp.array(rhs, dtype=wp.float32, device=device),
            wp.array(row_type, dtype=wp.int32, device=device),
            wp.full((2, rows), -1, dtype=wp.int32, device=device),
            wp.zeros((2, rows), dtype=wp.float32, device=device),
            wp.zeros(2, dtype=wp.int32, device=device),
            wp.zeros((2, 4), dtype=wp.int32, device=device),
            wp.zeros((2, 1), dtype=wp.float32, device=device),
            *[wp.zeros((2, 1, 6), dtype=wp.float32, device=device) for _ in range(4)],
            wp.zeros((2, 1), dtype=wp.float32, device=device),
            0,
            wp.zeros((2, rows), dtype=wp.float32, device=device),
            wp.zeros((2, 1), dtype=wp.float32, device=device),
        ]

    def kernel(self, policy=None):
        options = {} if policy is None else {"solve_policy": policy}
        return _get_pgs_solve_local_owned_kernel(*self.shape, self.device.arch, **options)

    def run(self, kernel, iterations, *extra):
        """Restore the initial diagonal, impulses and velocities and solve."""
        for output, initial in zip(self.outputs, self.initial, strict=True):
            wp.copy(output, initial)
        wp.launch_tiled(
            kernel,
            dim=[self.worlds],
            inputs=[*self.inputs, iterations, 1.0, 0, 0],
            outputs=[*self.outputs, *extra],
            block_dim=32,
            device=self.device,
        )

    def read(self):
        """Per-world outputs: the impulses, the articulation's velocities and the diagonal."""
        diagonal, impulses, velocities = (array.numpy() for array in self.outputs)
        return [(impulses[world], velocities[3 * world : 3 * world + 3], diagonal[world]) for world in range(2)]


def _assert_bitwise(actual, expected):
    """Check per-world outputs bitwise equal."""
    for actual_world, expected_world in zip(actual, expected, strict=True):
        for actual_array, expected_array in zip(actual_world, expected_world, strict=True):
            np.testing.assert_array_equal(actual_array, expected_array)


@unittest.skipUnless(wp.is_cuda_available(), "Native Gauss-Seidel sweeps run on CUDA")
class TestSolveReadoutKernels(unittest.TestCase):
    """Counts of single kernels against the stationary-sweep rule, replayed at every budget."""

    def test_sparse_counts_sweeps_to_the_stationary_exit(self):
        """Count each world's sweeps to its first stationary sweep, bitwise outputs, eager and replayed."""
        from newton.tests.test_feather_pgs_sparse_pgs import _problem  # noqa: PLC0415

        device = wp.get_device("cuda:0")
        cases = (
            ((43, 18, False, None), {}, (0, 0, 1.0)),
            ((43, 18, True, None), {}, (3, 1, 0.7)),
            ((43, 18, False, None), {"active_rows": True}, (0, 0, 1.0)),
            ((3, 3, False, "late_activation"), {"lanes_per_world": 16}, (0, 0, 1.0)),
        )
        for (dofs, support, seeded, activity), options, (friction_start, offset, omega) in cases:
            with self.subTest(dofs=dofs, seeded=seeded, activity=activity, options=options):
                problem = _problem(dofs, support, seeded, contact_activity=activity)
                case = _SparseCase(device, problem, options, friction_start, offset, omega)
                exact, counted = case.kernel(), case.kernel(SWEEPS)
                outputs = []
                for budget in range(BUDGET + 1):
                    case.run(exact, budget)
                    outputs.append(case.read())
                executed, stopped = _expected_counts(outputs, offset=offset, friction_start=friction_start)
                operand = _readout(device, case.worlds)
                case.run(counted, BUDGET, operand)
                for actual, expected in zip(case.read(), outputs[BUDGET], strict=True):
                    for actual_array, expected_array in zip(actual, expected, strict=True):
                        np.testing.assert_array_equal(actual_array, expected_array)
                _assert_counts(self, operand, executed, stopped, solved=range(case.worlds))
                # Graph replays add one record per replay to the captured operand.
                operand = _readout(device, case.worlds)
                with wp.ScopedCapture(device=device) as capture:
                    case.run(counted, BUDGET, operand)
                for _ in range(2):
                    wp.capture_launch(capture.graph)
                for actual, expected in zip(case.read(), outputs[BUDGET], strict=True):
                    for actual_array, expected_array in zip(actual, expected, strict=True):
                        np.testing.assert_array_equal(actual_array, expected_array)
                _assert_counts(self, operand, executed, stopped, solved=range(case.worlds), repeats=2)

    def test_trace_predicts_where_each_tolerance_stops(self):
        """Stop each tolerance run where the exact trace predicted, with the exact law's outputs at that sweep."""
        from newton.tests.test_feather_pgs_sparse_pgs import _problem  # noqa: PLC0415

        device = wp.get_device("cuda:0")
        cases = (
            ((43, 18, False, None), {}, (0, 0, 1.0)),
            ((43, 18, True, None), {}, (3, 1, 0.7)),
            ((43, 18, False, None), {"active_rows": True}, (0, 0, 1.0)),
        )
        for (dofs, support, seeded, activity), options, (friction_start, offset, omega) in cases:
            with self.subTest(dofs=dofs, seeded=seeded, options=options):
                problem = _problem(dofs, support, seeded, contact_activity=activity)
                case = _SparseCase(device, problem, options, friction_start, offset, omega)
                exact = case.kernel()
                outputs = []
                for budget in range(BUDGET + 1):
                    case.run(exact, budget)
                    outputs.append(case.read())
                trace = _trace(device, case.worlds)
                case.run(case.kernel(SolvePolicy(readout="trace")), BUDGET, trace)
                for actual, expected in zip(case.read(), outputs[BUDGET], strict=True):
                    for actual_array, expected_array in zip(actual, expected, strict=True):
                        np.testing.assert_array_equal(actual_array, expected_array)
                stops = trace.stops.numpy()
                self.assertEqual(int(trace.reasons.numpy().sum()), case.worlds)
                for point in (4, 8, 12):
                    with self.subTest(tolerance=TRACE_GRID[point]):
                        policy = SolvePolicy(readout="sweeps", velocity_change_tolerance=TRACE_GRID[point])
                        operand = _readout(device, case.worlds)
                        case.run(case.kernel(policy), BUDGET, operand)
                        sweeps, exhausted = operand.sweeps.numpy(), operand.exhausted.numpy()
                        histogram = sweeps.sum(axis=1)
                        np.testing.assert_array_equal(histogram[1:BUDGET], stops[point, 1:BUDGET])
                        self.assertEqual(histogram[BUDGET], stops[point, BUDGET] + stops[point, BUDGET + 1])
                        self.assertEqual(int(exhausted.sum()), stops[point, BUDGET + 1])
                        # Before its exit a tolerance run sweeps exactly as the exact law does.
                        result = case.read()
                        for world in range(case.worlds):
                            executed = int(np.flatnonzero(sweeps[:, world])[0])
                            for actual_array, expected_array in zip(
                                result[world], outputs[executed][world], strict=True
                            ):
                                np.testing.assert_array_equal(actual_array, expected_array)

    def test_active_rows_wait_for_a_violated_unvisited_row(self):
        """Keep sweeping while the visit of a row the active scan skipped would change it beyond the tolerance.

        The first sweep visits only the first row: its velocity-scaled change is ``omega`` and it leaves
        the second row violated by ``3 * omega - 0.5``, which a visit would change by ``omega`` times
        that. At ``omega = 1`` that is 2.5, beyond a tolerance of 1.5, so the world sweeps on. At
        ``omega = 0.5`` it is 0.5, within a tolerance of 0.75, so the world stops after the first sweep,
        although the second row's violation itself (1) exceeds the tolerance.
        """
        device = wp.get_device("cuda:0")
        rows = _violation_problem()[2]
        for omega, tolerance, violation in ((1.0, 1.5, 2.5), (0.5, 0.75, 1.0)):
            with self.subTest(omega=omega):
                case = _SparseCase(device, _violation_problem(), {"active_rows": True}, 0, 0, omega)
                outputs = []
                for budget in range(BUDGET + 1):
                    case.run(case.kernel(), budget)
                    outputs.append(case.read())
                delta, impulses = outputs[1][0]
                np.testing.assert_allclose(impulses * np.array([1.0, 10.0]), [omega, 0.0], rtol=1.0e-5)
                np.testing.assert_allclose(rows[1] @ delta + 0.5, -violation, rtol=1.0e-5)
                operand = _readout(device, case.worlds)
                policy = SolvePolicy(readout="sweeps", velocity_change_tolerance=tolerance)
                case.run(case.kernel(policy), BUDGET, operand)
                executed = int(np.flatnonzero(operand.sweeps.numpy()[:, 0])[0])
                if omega * violation > tolerance:
                    self.assertGreater(executed, 1)
                else:
                    self.assertEqual(executed, 1)
                for actual_array, expected_array in zip(case.read()[0], outputs[executed][0], strict=True):
                    np.testing.assert_array_equal(actual_array, expected_array)

    def test_local_counts_sweeps_to_the_stationary_exit(self):
        """Count only the owned world's sweeps to its first stationary sweep, with bitwise outputs."""
        case = _LocalCase(wp.get_cuda_device())
        exact, counted = case.kernel(), case.kernel(SWEEPS)
        outputs = []
        for budget in range(BUDGET + 1):
            case.run(exact, budget)
            outputs.append(case.read())
        executed, stopped = _expected_counts(outputs)
        operand = _readout(case.device, case.worlds)
        case.run(counted, BUDGET, operand)
        _assert_bitwise(case.read(), outputs[BUDGET])
        _assert_counts(self, operand, executed, stopped, solved=(0,))

    def test_local_trace_predicts_where_each_tolerance_stops(self):
        """Stop the local owner's tolerance runs where its exact trace predicted, with the exact outputs there."""
        case = _LocalCase(wp.get_cuda_device())
        exact = case.kernel()
        outputs = []
        for budget in range(BUDGET + 1):
            case.run(exact, budget)
            outputs.append(case.read())
        trace = _trace(case.device, case.worlds)
        case.run(case.kernel(SolvePolicy(readout="trace")), BUDGET, trace)
        _assert_bitwise(case.read(), outputs[BUDGET])
        # Only the owned world solves: one record per grid tolerance, exit reason and final sweep.
        stops = trace.stops.numpy()
        np.testing.assert_array_equal(stops.sum(axis=1), np.ones(len(TRACE_GRID), dtype=np.int32))
        self.assertEqual((int(trace.reasons.numpy().sum()), int(trace.limits.numpy().sum())), (1, 1))
        for point in (4, 8, 12):
            with self.subTest(tolerance=TRACE_GRID[point]):
                stop = int(np.flatnonzero(stops[point])[0])
                operand = _readout(case.device, case.worlds)
                policy = SolvePolicy(readout="sweeps", velocity_change_tolerance=TRACE_GRID[point])
                case.run(case.kernel(policy), BUDGET, operand)
                sweeps, exhausted = operand.sweeps.numpy(), operand.exhausted.numpy()
                np.testing.assert_array_equal(sweeps[:, 1], 0)
                executed = int(np.flatnonzero(sweeps[:, 0])[0])
                self.assertEqual(executed, min(stop, BUDGET))
                self.assertEqual(int(exhausted[0]), int(stop > BUDGET))
                # Before its exit a tolerance run sweeps exactly as the exact law does.
                _assert_bitwise(case.read(), [outputs[executed][0], outputs[BUDGET][1]])


def _sparse_route():
    """Fixed-base legs pressing spheres into the ground, on native sparse factors (explicit contacts)."""
    from newton.tests.test_feather_pgs_sparse_tolerance import _build  # noqa: PLC0415

    model, contacts = _build(wp.get_device("cuda:0"))

    def prepare(solver, state):
        state.joint_qd.fill_(0.3)

    return {
        "model": model,
        "options": {"pgs_mode": "split", "sparse_mass_matrix": "on", "friction_anchor_beta": 0.0},
        "contacts": contacts,
        "labels": {"sparse_pgs"},
        "owners": ({"sparse_pgs"},),
        "solving": {"sparse_pgs"},
        "prepare": prepare,
    }


def _diagonal_route():
    """Tilted sliding branches on the ground, owned by the diagonal-mass contact sweep."""
    from newton.tests.test_feather_pgs_launch_config import _build_sparse_contact_friction_model  # noqa: PLC0415

    def prepare(solver, state):
        joint_qd = state.joint_qd.numpy()
        for art in np.flatnonzero(solver._model_plan.response_dof_count == 16):
            start = int(solver._model_plan.articulation_dof_start[art])
            joint_qd[start : start + 16] = -0.3
        state.joint_qd.assign(joint_qd)

    return {
        "model": _build_sparse_contact_friction_model(device="cuda:0"),
        "options": {
            "pgs_mode": "matrix_free",
            "dense_max_constraints": 96,
            "mf_max_constraints": 32,
            "enable_joint_limits": True,
        },
        "labels": {"sparse_diagonal"},
        "owners": ({"sparse_diagonal"},),
        "solving": {"sparse_diagonal"},
        "prepare": prepare,
    }


def _mixed_route(hinv_jt_kernel, dofs, labels, *, owners, solving):
    """An arm struck by a sliding free box: local owners under ``par_row``, the paired factor sweep for 23 DOFs.

    Its one world always holds the arm's mimic row, so one owner solves it at every solve.
    """
    from newton.tests.test_feather_pgs_response_diagonal import _build_mixed_response_model  # noqa: PLC0415

    def prepare(solver, state):
        joint_qd = state.joint_qd.numpy()
        free = int(np.flatnonzero(solver._model_plan.is_free_rigid)[0])
        start = int(solver._model_plan.articulation_dof_start[free])
        joint_qd[start], joint_qd[start + 2] = 2.0, -3.0
        state.joint_qd.assign(joint_qd)

    return {
        "model": _build_mixed_response_model("cuda:0", dof_count=dofs, friction=0.7),
        "options": {
            "pgs_mode": "matrix_free",
            "enable_joint_limits": True,
            "joint_limit_activation_gap": 0.0,
            "dense_max_constraints": 96,
            "mf_max_constraints": 32,
        },
        "overrides": {"hinv_jt_kernel": hinv_jt_kernel},
        "labels": labels,
        "owners": owners,
        "solving": solving,
        "prepare": prepare,
    }


def _matrix_free_route():
    """Contacting tripods beside free spheres: the matrix-free Gauss-Seidel sweep owns their worlds.

    The spheres' ground contacts are matrix-free rows, which the incremental tier leaves to the
    general sweep (``_incremental_route`` covers the tier).
    """
    from newton.tests.test_feather_pgs_sparse_solver import _build_model  # noqa: PLC0415

    return {
        "model": _build_model(free_count=2, robot_count=2),
        "options": {
            "pgs_mode": "matrix_free",
            "sparse_mass_matrix": "off",
            "friction_anchor_beta": 0.0,
            "enable_joint_limits": True,
            "dense_max_constraints": 96,
            "mf_max_constraints": 64,
        },
        "labels": {"mf_gs", "mf_gs_incremental"},
        "owners": ({"mf_gs", "mf_gs_incremental"},),
        "solving": {"mf_gs"},
    }


def _incremental_route():
    """Contacting tripods without free bodies: every world holds only dense rows, which the incremental tier owns.

    Each world's three feet rest in the ground and one leg starts beyond its limit, about a dozen rows,
    within the tier's 32; without free bodies no world has matrix-free rows or a local owner.
    """
    from newton.tests.test_feather_pgs_sparse_solver import _build_model  # noqa: PLC0415

    return {
        "model": _build_model(),
        "options": {
            "pgs_mode": "matrix_free",
            "sparse_mass_matrix": "off",
            "friction_anchor_beta": 0.0,
            "enable_joint_limits": True,
            "dense_max_constraints": 96,
            "mf_max_constraints": 64,
        },
        "labels": {"mf_gs_incremental"},
        "owners": ({"mf_gs_incremental"},),
        "solving": {"mf_gs_incremental"},
    }


_LOCAL = {"local_internal", "local_pair", "local_residual"}


def _local_route():
    """The mixed arm under ``par_row``: local owners claim the world, the paired owner while the box touches it."""
    return _mixed_route(
        "par_row",
        13,
        {*_LOCAL, "mf_gs"},
        owners=({*_LOCAL, "mf_gs", "mf_gs_incremental"},),
        solving={"local_pair"},
    )


def _paired_route():
    """The mixed arm with 23 DOFs: the paired factor sweep owns the world, which has no matrix-free rows."""
    return _mixed_route("auto", 23, {"paired_factor"}, owners=({"paired_factor", "mf_gs"},), solving={"paired_factor"})


def _policy_routes():
    """The routes whose local, paired, matrix-free and incremental sweeps compile the velocity law and trace."""
    return (_local_route(), _paired_route(), _matrix_free_route(), _incremental_route())


@unittest.skipUnless(wp.is_cuda_available(), "Native Gauss-Seidel sweeps run on CUDA")
class TestSolveReadoutRoutes(unittest.TestCase):
    """Every counted route leaves the trajectory unchanged and counts its world-solves."""

    @staticmethod
    def _solvers(route, variants):
        """Build one solver per option set of ``variants`` on the route's model, with the route's overrides."""
        options = {**route["options"], "double_buffer": False, "pgs_iterations": BUDGET}
        with mock.patch.object(SolverFeatherPGS, "_kernel_overrides", route.get("overrides", {})):
            return [SolverFeatherPGS(route["model"], **options, **variant) for variant in variants]

    @staticmethod
    def _steps(route, solvers, steps, dt=1.0 / 240.0):
        """Step every solver from one prepared state on shared contacts; yield each step's output states."""
        model = route["model"]
        states = [(model.state(), model.state()) for _ in solvers]
        for solver, (state, _) in zip(solvers, states, strict=True):
            if "prepare" in route:
                route["prepare"](solver, state)
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        contacts, pipeline = route.get("contacts"), None
        if contacts is None:
            pipeline = newton.CollisionPipeline(model, broad_phase="nxn", reduce_contacts=False, deterministic=True)
            contacts = pipeline.contacts()
        control = model.control()
        for step in range(steps):
            if pipeline is not None:
                pipeline.collide(states[0][0], contacts)
            for solver, (state_in, state_out) in zip(solvers, states, strict=True):
                state_in.clear_forces()
                solver.step(state_in, state_out, control, contacts, dt)
            states = [(state_out, state_in) for state_in, state_out in states]
            yield step, [state for state, _ in states]

    def _lockstep(self, route, *, readout="sweeps", steps=4):
        """Step two solvers without and one with the readout from one state on shared contacts.

        The second solver without readout is a control: where the route itself is
        bitwise reproducible, the readout solver must match it bitwise; where
        atomically ordered bookkeeping makes the route vary between runs, it must
        stay as close as the control does. Returns the readout solver, its
        summary and the model.
        """
        model = route["model"]
        variants = [{"pgs_solve_readout": "off"}, {"pgs_solve_readout": "off"}, {"pgs_solve_readout": readout}]
        solvers = self._solvers(route, variants)
        counted = solvers[-1]
        self.assertTrue(set(counted._solve_readout.kernels.values()) & route["labels"], counted._solve_readout.kernels)
        self.assertEqual(solvers[0]._solve_readout.kernels, {})
        for step, states in self._steps(route, solvers, steps):
            for name in STATE_FIELDS:
                reference, control_run, actual = (getattr(state, name).numpy() for state in states)
                if np.array_equal(control_run, reference):
                    np.testing.assert_array_equal(actual, reference, err_msg=f"{name} at step {step}")
                else:
                    spread = float(np.abs(control_run - reference).max())
                    np.testing.assert_allclose(actual, reference, rtol=0.0, atol=10.0 * spread, err_msg=name)
        record = counted.get_path_selection()
        self.assertEqual(record["options"]["pgs_solve_readout"]["applied"], readout)
        counts = record["routes"]["sweep"]["readout"]
        self.assertEqual(counts["mode"], readout)
        self.assertEqual(counts["counted"], counted._solve_readout.kernels)
        summary = record["measurements"]["solve_readout"]
        self.assertEqual(summary["readout"], readout)
        for slot in summary["slots"]:
            with self.subTest(kernel=slot["kernel"]):
                self.assertEqual(slot["budget"], BUDGET)
                self.assertEqual(slot["solves"], sum(slot["sweeps"]))
                self.assertEqual(slot["sweeps"][0], 0)
                self.assertLessEqual(slot["exhausted"], slot["sweeps"][BUDGET])
                self.assertLessEqual(slot["worlds"], model.world_count)
        self._assert_owned(route, summary, steps)
        return counted, summary, model

    def _assert_owned(self, route, summary, steps):
        """Check that one kernel of each owner group solved every world at every step, and each solving label ran."""
        world_count = route["model"].world_count
        for owners in route["owners"]:
            with self.subTest(owners=sorted(owners)):
                solves = sum(slot["solves"] for slot in summary["slots"] if slot["route"] in owners)
                self.assertEqual(solves, steps * world_count)
        for label in route["solving"]:
            with self.subTest(solving=label):
                self.assertGreater(sum(slot["solves"] for slot in summary["slots"] if slot["route"] == label), 0)

    def test_sparse_route_counts_every_world_once_per_solve(self):
        """Count each world once per solve on the sparse route, eagerly and under graph replay."""
        route = _sparse_route()
        solver, summary, model = self._lockstep(route, steps=4)
        (slot,) = summary["slots"]
        self.assertEqual(slot["route"], "sparse_pgs")
        # The sparse sweep enters its loop for every world, rows or not.
        self.assertEqual(slot["solves"], 4 * model.world_count)
        state_in, state_out = model.state(), model.state()
        newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
        with wp.ScopedCapture(device=model.device) as capture:
            solver.step(state_in, state_out, model.control(), route["contacts"], 1.0 / 240.0)
        for _ in range(3):
            wp.capture_launch(capture.graph)
        (replayed,) = solver.get_path_selection()["measurements"]["solve_readout"]["slots"]
        self.assertEqual(replayed["solves"], 7 * model.world_count)

    def test_diagonal_route_counts_every_world_once_per_solve(self):
        """Count the diagonal-mass contact sweep of every world once per solve."""
        _, summary, model = self._lockstep(_diagonal_route(), steps=4)
        (slot,) = [slot for slot in summary["slots"] if slot["route"] == "sparse_diagonal"]
        self.assertEqual(slot["solves"], 4 * model.world_count)

    def test_local_route_counts_without_changing_results(self):
        """Count the local owners' sweeps without changing the trajectory, every world-solve once."""
        self._lockstep(_local_route())

    def test_paired_route_counts_without_changing_results(self):
        """Count the paired factor sweep without changing the trajectory, every world-solve once."""
        solver, _, _ = self._lockstep(_paired_route())
        self.assertIsNotNone(solver._paired_factor_solve_kernel)

    def test_matrix_free_route_counts_without_changing_results(self):
        """Count the matrix-free Gauss-Seidel sweep without changing the trajectory, every world-solve once."""
        self._lockstep(_matrix_free_route())

    def test_incremental_route_counts_without_changing_results(self):
        """Count the incremental tier's sweeps without changing the trajectory, every world at every solve."""
        solver, _, _ = self._lockstep(_incremental_route())
        self.assertTrue(solver._pgs_solve_mf_gs_incremental_kernels)

    def test_declined_trace_counts_without_it_and_warns(self):
        """Warn about, record and count without the trace a sweep variant that cannot trace.

        The matrix-free sweep traces only the current friction law; under bisection friction it
        keeps the plain count, which a trace request records under the readout, not the exit law.
        """
        route = _matrix_free_route()
        with self.assertWarnsRegex(UserWarning, "pgs_solve_readout='trace'"):
            (solver,) = self._solvers(route, [{"pgs_solve_readout": "trace", "friction_mode": "bisection"}])
        key = solver._pgs_solve_mf_gs_kernel.key
        sweep = solver.get_path_selection()["routes"]["sweep"]
        self.assertIn(key, sweep["readout"]["declined"])
        self.assertEqual((sweep["readout"]["mode"], sweep["exit_law"]["declined"]), ("trace", {}))
        for _ in self._steps(route, [solver], 2):
            pass
        (slot,) = [
            slot
            for slot in solver.get_path_selection()["measurements"]["solve_readout"]["slots"]
            if slot["kernel"] == key
        ]
        self.assertNotIn("trace", slot)
        self.assertGreater(slot["solves"], 0)

    def test_velocity_pass_counts_in_its_own_slots(self):
        """Count the velocity pass's sweeps in slots of their own budget, each world-solve once per pass."""
        route, steps = _incremental_route(), 4
        (solver,) = self._solvers(route, [{"pgs_solve_readout": "sweeps", "pgs_velocity_iterations": 2}])
        for _ in self._steps(route, [solver], steps):
            pass
        summary = solver.get_path_selection()["measurements"]["solve_readout"]
        slots = {(slot["route"], slot["pass"]): slot for slot in summary["slots"]}
        # Every counted kernel holds one slot per pass.
        self.assertEqual(
            sorted((slot["kernel"], slot["pass"]) for slot in summary["slots"]),
            sorted((kernel, solve_pass) for kernel in solver._solve_readout.kernels for solve_pass in SOLVE_PASSES),
        )
        for solve_pass, budget in (("position", BUDGET), ("velocity", 2)):
            with self.subTest(solve_pass=solve_pass):
                slot = slots[("mf_gs_incremental", solve_pass)]
                self.assertEqual(slot["budget"], budget)
                self.assertEqual(slot["solves"], steps * route["model"].world_count)
                self.assertEqual(len(slot["sweeps"]), budget + 1)

    def test_trace_changes_no_result_and_records_every_solve(self):
        """Trace the local, paired and matrix-free sweeps unchanged, once per solve and grid tolerance."""
        for route in _policy_routes():
            with self.subTest(labels=sorted(route["labels"])):
                solver, summary, _ = self._lockstep(route, readout="trace")
                self.assertEqual(solver.get_path_selection()["routes"]["sweep"]["readout"]["declined"], {})
                for slot in summary["slots"]:
                    with self.subTest(kernel=slot["kernel"]):
                        trace = slot["trace"]
                        self.assertEqual([sum(row) for row in trace["stops"]], [slot["solves"]] * len(TRACE_GRID))
                        self.assertEqual(sum(trace["reasons"].values()), slot["solves"])
                        self.assertEqual(sum(kind["count"] for kind in trace["final_change"].values()), slot["solves"])

    def test_tolerance_reaches_every_counted_sweep(self):
        """Compile a velocity tolerance into every counted local, paired and matrix-free sweep, and step on it."""
        for route in _policy_routes():
            with self.subTest(labels=sorted(route["labels"])):
                variant = {"pgs_solve_readout": "sweeps", "pgs_velocity_change_tolerance": 1.0e-3}
                with warnings.catch_warnings():
                    # A route's fixed-iteration sweeps keep their own law, about which the solver may warn.
                    warnings.simplefilter("ignore", UserWarning)
                    (solver,) = self._solvers(route, [variant])
                counted = solver._solve_readout.kernels
                self.assertTrue(set(counted.values()) & route["labels"], counted)
                law = solver.get_path_selection()["routes"]["sweep"]["exit_law"]
                self.assertEqual((law["law"], law["declined"]), ("velocity", {}))
                self.assertEqual(set(law["honored"]), set(counted))
                for step, (state,) in self._steps(route, [solver], steps=4):
                    for name in STATE_FIELDS:
                        self.assertTrue(np.isfinite(getattr(state, name).numpy()).all(), f"{name} at step {step}")
                summary = solver.get_path_selection()["measurements"]["solve_readout"]
                self._assert_owned(route, summary, 4)


if __name__ == "__main__":
    unittest.main()
