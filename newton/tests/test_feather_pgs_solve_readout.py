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
once: in the slot of the sweep that solved it, or, for a world without rows
that the matrix-free sweeps do not enter, as skipped. The parallel tiers count
their own worlds, against their own sweep budget.

``pgs_velocity_change_tolerance`` stops a solve earlier, after the first sweep
whose velocity-scaled changes fit the tolerance. Before that sweep it runs
exactly the exact-law sweeps, so its outputs equal the exact law's at that
budget, and the trace of an exact solve predicts that sweep for every grid
tolerance. On the solver routes the trace changes no result and records every
solve once per grid tolerance, and a tolerance compiled into every counted
sweep keeps the trajectory near the exact law's in no more sweeps. Velocity-limit
rows vote under both velocity-limit laws in the kernel tests, and bind under the
accumulated law on a route of their own.
"""

import itertools
import unittest
import warnings
from functools import partial
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import sweep_routes, sweep_routes_local
from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
    PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT,
    PGS_CONSTRAINT_TYPE_MIMIC,
    PGS_LOCAL_SOLVE_OWNER_SINGLE,
)
from newton._src.solvers.feather_pgs.row_metadata import _DENSE_META_ROW_TYPE_MASK
from newton._src.solvers.feather_pgs.solve_policy import (
    SOLVE_PASSES,
    TRACE_GRID,
    TRACE_KINDS,
    TRACE_REASONS,
    SolvePolicy,
    SolveReadout,
    SolveTrace,
)
from newton._src.solvers.feather_pgs.sparse_mass_matrix import _SparseRowSupport
from newton._src.solvers.feather_pgs.sweep_routes_sparse import sparse_kernel
from newton._src.solvers.feather_pgs.velocity_limit_law import VELOCITY_LIMIT_LAWS
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


# Two velocity-limit DOFs whose positive mass coupling makes braking either one speed the other up.
_PAIR_MASS = np.array([[3.0, 0.6], [0.6, 1.0]])
_PAIR_VELOCITY = np.array([2.0, 2.5])


def _velocity_limit_pair(device, *, row_phase, iterations):
    """One world's launch values of two coupled velocity-limit rows that keep pushing each other's DOF past its limit.

    The values of test_feather_pgs_velocity_limit_law's chain (upper limits of 1 on both DOFs, the
    rows in order) with another mass matrix and start: both DOFs start beyond their limits, and
    braking one speeds the other up, so every correction is a push and the stateless corrections
    add up to the accumulated impulses. The sweeps then converge to both limits geometrically under
    either law, by about 0.12 per sweep: the largest velocity-scaled change falls from about 2 in
    the first sweep to about 1e-6 in the eighth.
    """
    from newton.tests.test_feather_pgs_velocity_limit_law import _dense_row_values  # noqa: PLC0415

    values = _dense_row_values(device, order=(0, 1), row_phase=row_phase, iterations=iterations)
    jacobian = values["J_world"].numpy()[0, :2]
    response, diagonal = values["Y_world"].numpy(), values["world_diag"].numpy()
    response[0, :2] = np.linalg.solve(_PAIR_MASS, jacobian.T).T
    diagonal[0, :2] = np.einsum("ij,ij->i", jacobian, response[0, :2])
    values.update(
        Y_world=wp.array(response, dtype=wp.float32, device=device),
        world_diag=wp.array(diagonal, dtype=wp.float32, device=device),
        v_out=wp.array(_PAIR_VELOCITY, dtype=wp.float32, device=device),
    )
    return values


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
            padded(incident),
            padded(types, np.int32),
            padded(parents, np.int32),
            padded(mu),
            wp.array(np.arange(worlds * dofs).reshape(worlds, dofs), dtype=int, device=device),
            wp.zeros(worlds * dofs, device=device),
            wp.zeros(worlds, dtype=int, device=device),
            wp.zeros((worlds, 4), dtype=int, device=device),
            wp.zeros((worlds, 1), device=device),
            *[wp.zeros((worlds, 1, 6), device=device) for _ in range(2)],
            wp.zeros((worlds, 1), device=device),
        ]
        self.solve = (omega, friction_start, offset)
        worlds_per_block = 64 // options.get("lanes_per_world", 32)
        self.blocks = (worlds + worlds_per_block - 1) // worlds_per_block
        self.worlds = worlds

    def kernel(self, policy=None):
        options = self.options if policy is None else {**self.options, "solve_policy": policy}
        return sparse_kernel(*self.shape, **options)

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
            *[wp.zeros((2, 1, 6), dtype=wp.float32, device=device) for _ in range(2)],
            wp.zeros((2, 1), dtype=wp.float32, device=device),
            0,
            wp.zeros((2, rows), dtype=wp.float32, device=device),
            wp.zeros((2, 1), dtype=wp.float32, device=device),
        ]

    def kernel(self, policy=None):
        options = {} if policy is None else {"solve_policy": policy}
        return sweep_routes_local.local_owned_kernel(*self.shape, self.device.arch, **options)

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


def _rowless(solver):
    """Worlds of the solver's last step without dense or matrix-free rows."""
    dense, free = solver.constraint_count.numpy(), solver.mf_constraint_count.numpy()
    return int(np.count_nonzero((dense == 0) & (free == 0)))


def _skipped(summary, solve_pass="position"):
    """Skipped world-solves of a readout summary in one solve pass."""
    return sum(entry["solves"] for entry in summary["skipped"] if entry["pass"] == solve_pass)


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
                # The impulse buffer holds the case's padded capacity; only the leading rows are live.
                live, padding = impulses[: len(rows)], impulses[len(rows) :]
                np.testing.assert_allclose(live, [omega, 0.0], rtol=1.0e-5, atol=0.0)
                np.testing.assert_array_equal(padding, 0.0)
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

    def test_velocity_limit_trace_predicts_where_each_tolerance_stops(self):
        """Stop the tolerance runs of velocity-limit rows where the exact trace predicted, under either law.

        Two coupled velocity-limit rows (:func:`_velocity_limit_pair`) go through the matrix-free and
        incremental sweeps, which visit them in the end-of-iteration pass of row phase 0 and in the
        main row loop of phase 5. A row votes the correction it applies under the stateless law and
        the change of its stored total under the accumulated law, each scaled by its A_ii. The trace
        of an exact solve changes no output and predicts the sweep after which each tolerance run
        stops; that run has the exact law's outputs there and leaves neither DOF beyond its limit by
        more than the tolerance, and the largest tolerance stops several sweeps before the budget.
        """
        from newton.tests.test_feather_pgs_velocity_limit_law import _launch_inputs  # noqa: PLC0415

        device = wp.get_device("cuda:0")
        arch = str(device.arch)
        factories = {
            "mf_gs": partial(sweep_routes.mf_gs_kernel, 4, 1, 2, arch, has_drive_rows=False),
            "incremental": partial(
                sweep_routes.incremental_kernel, 4, 1, 2, arch, incremental_rows=4, has_drive_rows=False
            ),
        }
        for (name, factory), law, row_phase in itertools.product(factories.items(), VELOCITY_LIMIT_LAWS, (0, 5)):
            with self.subTest(kernel=name, law=law, row_phase=row_phase):

                def run(policy, budget, operand=None, factory=factory, law=law, row_phase=row_phase):
                    kernel = factory(velocity_limit_law=law, solve_policy=policy)
                    values = _velocity_limit_pair(device, row_phase=row_phase, iterations=budget)
                    if operand is not None:
                        values["readout"] = operand
                    inputs = _launch_inputs(kernel, values)
                    wp.launch_tiled(kernel, dim=[1], inputs=inputs, block_dim=32, device=device)
                    return values["v_out"].numpy(), values["world_impulses"].numpy()[0]

                outputs = [run(SolvePolicy(), budget) for budget in range(BUDGET + 1)]
                trace = _trace(device, 1)
                _assert_bitwise([run(SolvePolicy(readout="trace"), BUDGET, trace)], [outputs[BUDGET]])
                self.assertEqual(int(trace.reasons.numpy().sum()), 1)
                stops = trace.stops.numpy()
                earliest = BUDGET + 1
                for point in (4, 8, 12):
                    tolerance = TRACE_GRID[point]
                    stop = int(np.flatnonzero(stops[point])[0])
                    operand = _readout(device, 1)
                    policy = SolvePolicy(readout="sweeps", velocity_change_tolerance=tolerance)
                    velocity, impulses = run(policy, BUDGET, operand)
                    executed = int(np.flatnonzero(operand.sweeps.numpy()[:, 0])[0])
                    self.assertEqual(executed, min(stop, BUDGET), tolerance)
                    self.assertEqual(int(operand.exhausted.numpy()[0]), int(stop > BUDGET), tolerance)
                    # Before its exit a tolerance run sweeps exactly as the exact law does.
                    _assert_bitwise([(velocity, impulses)], [outputs[executed]])
                    if stop <= BUDGET:
                        self.assertLessEqual(float(np.max(velocity - 1.0)), tolerance + 1.0e-6, tolerance)
                    earliest = min(earliest, stop)
                self.assertLess(earliest, BUDGET - 2)

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
        "enters_rowless_worlds": True,
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
        "enters_rowless_worlds": True,
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

    Each world's three feet start in the ground and one leg beyond its limit, within the tier's 32 rows;
    without free bodies no world has matrix-free rows or a local owner. Pushed out of the ground by the
    first step, the tripods rise for a few steps, in which every row of their worlds separates at the
    predictor: the resolved-world classifier leaves those worlds without rows, and no sweep enters them.
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


def _velocity_limit_route():
    """The incremental route's tripods with velocity-limited legs under the accumulated velocity-limit law.

    Each leg joint is limited to 1 rad/s and starts at 2 rad/s, so the velocity-limit rows bind
    from the first step, and the accumulated law keeps each row's total across sweeps, which the
    votes measure as the change of that total. The rows start from half the limit and add at most
    two per leg to a world's rows; the matrix-free sweep stands among the owners in case a world
    outgrows the incremental tier's 32 rows.
    """
    route = _incremental_route()
    model = route["model"]
    joint_type, starts = model.joint_type.numpy(), model.joint_qd_start.numpy()
    legs = starts[np.flatnonzero(joint_type == int(newton.JointType.REVOLUTE))]
    limits = model.joint_velocity_limit.numpy()
    limits[legs] = 1.0
    model.joint_velocity_limit.assign(limits)

    def prepare(solver, state):
        joint_qd = state.joint_qd.numpy()
        joint_qd[legs] = 2.0
        state.joint_qd.assign(joint_qd)

    route["options"] = {
        **route["options"],
        "enable_joint_velocity_limits": True,
        "velocity_limit_activation_fraction": 0.5,
        "velocity_limit_law": "accumulated",
    }
    route.update(
        name="velocity_limits",
        labels={"mf_gs_incremental", "mf_gs"},
        owners=({"mf_gs_incremental", "mf_gs"},),
        solving=set(),
        prepare=prepare,
    )
    return route


def _tier_route():
    """A tripod world beside a free-body world, with the parallel tiers of ``mf_gs_parallel_rows`` built.

    The tripod's world holds only dense rows (24 while its feet touch the ground: six contacts and
    six joint-limit rows), within the tiers' bound, so the tiers solve and count it and the
    matrix-free sweep returns before its first sweep there; the free body's matrix-free rows keep
    its world on the counted matrix-free sweep. The tiled response keeps the local owners out: the
    automatic choice gives the tripod's 9 DOFs the parallel-row response, which makes a world with
    one small non-free articulation, in a model with free bodies, a local-owner candidate, and its
    counted local sweep would claim the world whenever it holds at most 9 rows of one phase.
    """
    from newton.tests.test_feather_pgs_sparse_solver import _build_model  # noqa: PLC0415

    return {
        "model": _build_model(heterogeneous=True),
        "options": {
            "pgs_mode": "matrix_free",
            "sparse_mass_matrix": "off",
            "friction_anchor_beta": 0.0,
            "enable_joint_limits": True,
            "dense_max_constraints": 96,
            "mf_max_constraints": 64,
            "mf_gs_parallel_rows": 32,
        },
        "overrides": {"hinv_jt_kernel": "tiled"},
        "labels": {"mf_gs", "parallel_tier"},
        "owners": ({"mf_gs", "parallel_tier"},),
        "solving": {"mf_gs", "parallel_tier"},
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
    """The routes whose local, paired, matrix-free and incremental sweeps compile the velocity law and trace.

    The last one binds velocity-limit rows under the accumulated velocity-limit law.
    """
    return (_local_route(), _paired_route(), _matrix_free_route(), _incremental_route(), _velocity_limit_route())


def _route_name(route):
    """A route's name in subtests: its own, or its counted labels."""
    return route.get("name", "+".join(sorted(route["labels"])))


@unittest.skipUnless(wp.is_cuda_available(), "Native Gauss-Seidel sweeps run on CUDA")
class TestSolveReadoutRoutes(unittest.TestCase):
    """Every counted route leaves the trajectory unchanged and counts its world-solves."""

    @staticmethod
    def _solvers(route, variants):
        """Build one solver per option set of ``variants`` on the route's model, with the route's overrides; a
        variant's options override the route's."""
        options = {**route["options"], "double_buffer": False, "pgs_iterations": BUDGET}
        with mock.patch.object(SolverFeatherPGS, "_kernel_overrides", route.get("overrides", {})):
            return [SolverFeatherPGS(route["model"], **{**options, **variant}) for variant in variants]

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
        rowless = 0
        for step, states in self._steps(route, solvers, steps):
            rowless += _rowless(counted)
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
        # The route record names the passes whose skipped solves the summary holds.
        self.assertEqual(counts["skipped_passes"], [] if route.get("enters_rowless_worlds") else ["position"])
        self.assertEqual([entry["pass"] for entry in summary["skipped"]], counts["skipped_passes"])
        # A parallel tier counts against its own compiled budget, every other sweep against the solver's.
        budgets = {tier.kernel.key: tier.sweeps for tier in counted._mf_gs_tiers if tier.sweeps}
        for slot in summary["slots"]:
            with self.subTest(kernel=slot["kernel"]):
                budget = budgets.get(slot["kernel"], BUDGET)
                self.assertEqual(slot["budget"], budget)
                self.assertEqual(slot["solves"], sum(slot["sweeps"]))
                self.assertEqual(slot["sweeps"][0], 0)
                self.assertLessEqual(slot["exhausted"], slot["sweeps"][budget])
                self.assertLessEqual(slot["worlds"], model.world_count)
        self._assert_owned(route, summary, steps, rowless)
        return counted, summary, model

    def _assert_owned(self, route, summary, steps, rowless):
        """Check that each owner group's kernels and the skipped record cover every world at every step once.

        The sparse and diagonal-mass sweeps enter every world; the matrix-free sweeps enter no world
        without rows, whose ``rowless`` world-steps the readout must count as skipped. Each solving
        label must also have run.
        """
        world_count = route["model"].world_count
        skipped = _skipped(summary)
        self.assertEqual(skipped, 0 if route.get("enters_rowless_worlds") else rowless)
        for owners in route["owners"]:
            with self.subTest(owners=sorted(owners)):
                solves = sum(
                    slot["solves"]
                    for slot in summary["slots"]
                    if slot["route"] in owners and slot["pass"] == "position"
                )
                self.assertEqual(solves + skipped, steps * world_count)
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
        """Count the incremental tier's sweeps without changing the trajectory, and the solves of resolved worlds.

        While the tripods rise, the resolved-world classifier leaves their worlds without rows and no
        sweep enters them; the readout counts those solves as skipped, so with the tier's slot they
        cover every world at every solve.
        """
        solver, summary, model = self._lockstep(_incremental_route())
        self.assertTrue(solver._mf_gs_tiers)
        self.assertTrue(solver.get_path_selection()["routes"]["resolved_worlds"]["enabled"])
        (skipped,) = summary["skipped"]
        self.assertEqual(skipped["pass"], "position")
        self.assertGreater(skipped["solves"], 0)
        self.assertLessEqual(skipped["worlds"], model.world_count)

    def test_velocity_limit_route_counts_without_changing_results(self):
        """Count the sweeps of binding velocity-limit rows under the accumulated law, every world-solve once.

        Under ``velocity_limit_law="accumulated"`` a velocity-limit row keeps its total across sweeps,
        and the counted sweeps carry that law. The route's first step holds velocity-limit rows with
        positive totals; the readout changes no result and counts every world-solve once.
        """
        route = _velocity_limit_route()
        (probe,) = self._solvers(route, [{"pgs_solve_readout": "sweeps"}])
        self.assertEqual(probe.get_path_selection()["routes"]["velocity_limits"]["law"], "accumulated")
        self.assertTrue(probe._solve_readout.kernels)
        for key in probe._solve_readout.kernels:
            self.assertIn("_vacc", key)
        for _ in self._steps(route, [probe], 1):
            counts = probe.constraint_count.numpy()
            types = probe.row_type.numpy() & _DENSE_META_ROW_TYPE_MASK
            impulses = probe.impulses.numpy()
            totals = [
                impulses[world, :count][types[world, :count] == PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT]
                for world, count in enumerate(counts)
            ]
            self.assertTrue(any(np.any(world_totals > 0.0) for world_totals in totals), totals)
        _, summary, _ = self._lockstep(route)
        self.assertGreater(sum(slot["solves"] for slot in summary["slots"]), 0)

    def test_parallel_tiers_count_their_worlds(self):
        """Count the parallel tiers' world-solves in their own slots, every world-solve once, none twice.

        The tiers solve the tripod's world, the matrix-free sweep the free body's: their slots and the
        skip counter must add up to every world at every step. Under a velocity tolerance every sweep,
        the tiers' included, compiles the law, without a warning.
        """
        route, steps = _tier_route(), 4
        solver, summary, _ = self._lockstep(route, steps=steps)
        # The route's premise: no local owner claims the tripod's world, so the tiers own it.
        self.assertFalse(solver._local_internal_fast_path)
        sweep = solver.get_path_selection()["routes"]["sweep"]
        tier_kernels = {tier["kernel"] for tier in sweep["tiers"]["tiers"]}
        self.assertEqual(sweep["tiers"]["family"], "parallel")
        self.assertLessEqual(tier_kernels, set(sweep["readout"]["counted"]))
        tiers = sum(slot["solves"] for slot in summary["slots"] if slot["route"] == "parallel_tier")
        self.assertGreater(tiers, 0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            (loose,) = self._solvers(route, [{"pgs_velocity_change_tolerance": 1.0e-3}])
        law = loose.get_path_selection()["routes"]["sweep"]["exit_law"]
        self.assertEqual(law["law"], "velocity")
        self.assertLessEqual(
            {loose._pgs_solve_mf_gs_kernel.key, *(tier.kernel.key for tier in loose._mf_gs_tiers)}, set(law["honored"])
        )

    def test_torsion_traces_its_changes(self):
        """Trace the matrix-free sweep with contact torsion, without a warning: every sweep compiles the trace."""
        route = _matrix_free_route()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            (solver,) = self._solvers(
                route, [{"pgs_solve_readout": "trace", "contact_torsion_radius": 0.01, "dense_max_constraints": 256}]
            )
        key = solver._pgs_solve_mf_gs_kernel.key
        self.assertIn("_torsion", key)
        self.assertEqual(solver.get_path_selection()["routes"]["sweep"]["readout"]["mode"], "trace")
        for _ in self._steps(route, [solver], 2):
            pass
        (slot,) = [
            slot
            for slot in solver.get_path_selection()["measurements"]["solve_readout"]["slots"]
            if slot["kernel"] == key
        ]
        self.assertIn("trace", slot)
        self.assertGreater(slot["solves"], 0)
        self.assertEqual(sum(slot["trace"]["reasons"].values()), slot["solves"])

    def test_velocity_pass_counts_in_its_own_slots(self):
        """Count the velocity pass's sweeps in slots of their own budget, each world-solve once per pass."""
        route, steps = _incremental_route(), 4
        (solver,) = self._solvers(route, [{"pgs_solve_readout": "sweeps", "pgs_velocity_iterations": 2}])
        rowless = 0
        for _ in self._steps(route, [solver], steps):
            rowless += _rowless(solver)
        summary = solver.get_path_selection()["measurements"]["solve_readout"]
        slots = {(slot["route"], slot["pass"]): slot for slot in summary["slots"]}
        # Every counted kernel holds one slot per pass, and the route one skip counter per pass.
        self.assertEqual(
            sorted((slot["kernel"], slot["pass"]) for slot in summary["slots"]),
            sorted((kernel, solve_pass) for kernel in solver._solve_readout.kernels for solve_pass in SOLVE_PASSES),
        )
        self.assertEqual([entry["pass"] for entry in summary["skipped"]], list(SOLVE_PASSES))
        for solve_pass, budget in (("position", BUDGET), ("velocity", 2)):
            with self.subTest(solve_pass=solve_pass):
                slot = slots[("mf_gs_incremental", solve_pass)]
                self.assertEqual(slot["budget"], budget)
                # Both passes sweep the same rows: a world without rows skips both.
                self.assertEqual(_skipped(summary, solve_pass), rowless)
                self.assertEqual(slot["solves"] + rowless, steps * route["model"].world_count)
                self.assertEqual(len(slot["sweeps"]), budget + 1)

    def test_trace_changes_no_result_and_records_every_solve(self):
        """Trace the local, paired and matrix-free sweeps unchanged, once per solve and grid tolerance."""
        for route in _policy_routes():
            with self.subTest(route=_route_name(route)):
                _, summary, _ = self._lockstep(route, readout="trace")
                for slot in summary["slots"]:
                    with self.subTest(kernel=slot["kernel"]):
                        trace = slot["trace"]
                        self.assertEqual([sum(row) for row in trace["stops"]], [slot["solves"]] * len(TRACE_GRID))
                        self.assertEqual(sum(trace["reasons"].values()), slot["solves"])
                        self.assertEqual(sum(kind["count"] for kind in trace["final_change"].values()), slot["solves"])

    def test_tolerance_stays_near_the_exact_law_in_fewer_sweeps(self):
        """Step every policy route under a velocity tolerance beside the exact law: near it, in no more sweeps.

        From one state on shared contacts, a 1e-3 tolerance compiled into every counted sweep must keep
        every joint velocity within 1e-2 of the exact law's over 12 steps, count every world-solve once
        (in a slot or as skipped), execute no more sweeps than the exact law on the route (and per route
        label whose solve count matches), and fewer on at least one route.
        """
        steps, tolerance = 12, 1.0e-3
        savings = {}
        for route in _policy_routes():
            name = _route_name(route)
            with self.subTest(route=name):
                variants = [
                    {"pgs_solve_readout": "sweeps"},
                    {"pgs_solve_readout": "sweeps", "pgs_velocity_change_tolerance": tolerance},
                ]
                with warnings.catch_warnings():
                    # A route's fixed-iteration sweeps keep their own law, about which the solver may warn.
                    warnings.simplefilter("ignore", UserWarning)
                    exact, loose = self._solvers(route, variants)
                counted = loose._solve_readout.kernels
                self.assertTrue(set(counted.values()) & route["labels"], counted)
                law = loose.get_path_selection()["routes"]["sweep"]["exit_law"]
                self.assertEqual(law["law"], "velocity")
                self.assertEqual(set(law["honored"]), set(counted))
                deviation = 0.0
                # The two trajectories differ slightly, so each solver's worlds may lack rows at other steps.
                rowless = [0, 0]
                for step, (exact_state, loose_state) in self._steps(route, [exact, loose], steps):
                    rowless = [count + _rowless(solver) for count, solver in zip(rowless, (exact, loose), strict=True)]
                    for field in STATE_FIELDS:
                        self.assertTrue(np.isfinite(getattr(loose_state, field).numpy()).all(), f"{field} at {step}")
                    difference = loose_state.joint_qd.numpy() - exact_state.joint_qd.numpy()
                    deviation = max(deviation, float(np.abs(difference).max()))
                self.assertLess(deviation, 1.0e-2)
                totals = []
                for solver, solver_rowless in zip((exact, loose), rowless, strict=True):
                    summary = solver.get_path_selection()["measurements"]["solve_readout"]
                    self._assert_owned(route, summary, steps, solver_rowless)
                    by_label = {}
                    for slot in summary["slots"]:
                        solves, sweeps = by_label.get(slot["route"], (0, 0))
                        executed = sum(count * sweep for sweep, count in enumerate(slot["sweeps"]))
                        by_label[slot["route"]] = (solves + slot["solves"], sweeps + executed)
                    totals.append(by_label)
                exact_totals, loose_totals = totals
                for label, (solves, sweeps) in loose_totals.items():
                    if solves and exact_totals.get(label, (0, 0))[0] == solves:
                        self.assertLessEqual(sweeps, exact_totals[label][1], label)
                savings[name] = sum(sweeps for _, sweeps in exact_totals.values()) - sum(
                    sweeps for _, sweeps in loose_totals.values()
                )
                self.assertGreaterEqual(savings[name], 0)
        self.assertTrue(any(saving > 0 for saving in savings.values()), savings)


if __name__ == "__main__":
    unittest.main()
