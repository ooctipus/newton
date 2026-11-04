# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Count the Gauss-Seidel sweeps every solve executes without changing a computed bit.

Under ``Diagnostics(solve_readout="sweeps")`` each sweep kernel records, per world, the
sweeps a solve executed and whether it stopped before its budget. A sweep
stops a world after a sweep whose changes all lie within the exit law's
rounding floor, so a world that stops after ``s`` sweeps has the same impulses
and velocities at every budget from ``s`` on, and its outputs change at every
budget before: they settle at ``s``, or at ``s - 1`` when sweep ``s`` changed
nothing. The kernel tests replay a solve at every sweep budget and check the
counts against that rule. That every route's readout leaves the trajectory
unchanged and counts each world-solve is conformance's route matrix
(:mod:`newton.tests.test_feather_pgs_conformance`, the ownership tests).
"""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import sweep_routes_local
from newton._src.solvers.feather_pgs.row_kernels import (
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
    PGS_CONSTRAINT_TYPE_MIMIC,
)
from newton._src.solvers.feather_pgs.solve_policy import (
    TRACE_GRID,
    TRACE_KINDS,
    TRACE_REASONS,
    SolvePolicy,
    SolveReadout,
    SolveTrace,
)
from newton._src.solvers.feather_pgs.sparse_mass_matrix import _SparseRowSupport
from newton._src.solvers.feather_pgs.sweep import PGS_LOCAL_SOLVE_OWNER_SINGLE
from newton._src.solvers.feather_pgs.sweep_routes_sparse import sparse_kernel
from newton.tests.feather_pgs_law_harness import chain_walk, leave_walk, local_packed_rows, row_lines, walked_rows

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


def _expected_counts(outputs):
    """Return the sweeps each world may have executed, from per-budget outputs ``outputs[s][world]``.

    The outputs settle at budget ``c``, the first from which they never change. A world stops after the first
    sweep whose changes were all quiet: sweep ``c`` (its changes within the floor) or ``c + 1`` (no change).
    A world whose outputs settle only at the budget may have run it out. Returns per world the allowed counts.
    """
    budget = len(outputs) - 1
    allowed = []
    for world in range(len(outputs[0])):

        def same(a, b, world=world):
            return all(np.array_equal(x, y) for x, y in zip(outputs[a][world], outputs[b][world], strict=True))

        settle = next(c for c in range(budget + 1) if all(same(c, later) for later in range(c, budget + 1)))
        first = max(settle, 1)
        counts = {first, min(first + 1, budget)} if first == settle else {first}
        allowed.append(counts)
    return allowed


def _assert_counts(test, operand, allowed, solved, *, repeats=1):
    """Check one record of ``repeats`` identical solves per world in ``solved`` at one of its allowed counts (a
    count below the budget stopped the world), none elsewhere."""
    sweeps, exhausted = operand.sweeps.numpy(), operand.exhausted.numpy()
    budget = sweeps.shape[0] - 1
    for world, counts in enumerate(allowed):
        with test.subTest(world=world):
            if world not in solved:
                np.testing.assert_array_equal(sweeps[:, world], 0)
                test.assertEqual(exhausted[world], 0)
                continue
            (recorded,) = np.flatnonzero(sweeps[:, world])
            test.assertIn(int(recorded), counts)
            test.assertEqual(int(sweeps[recorded, world]), repeats)
            if recorded < budget:
                test.assertEqual(exhausted[world], 0)
            else:
                test.assertIn(int(exhausted[world]), (0, repeats))


class _SparseCase:
    """One sparse sweep problem in three worlds (the middle one without rows) and its launch buffers."""

    def __init__(self, device, problem, options, omega):
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
        counts = wp.array([count, 0, count], dtype=int, device=device)
        self.head = [worlds, counts, row_lines(device), padded(bias), padded(diagonal)]
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
            wp.zeros((worlds, 1, 12), device=device),
        ]
        self.omega = omega
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
            inputs=[*self.head, self.impulses, *self.tail, iterations, self.omega],
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
        velocity = np.array([0.3, -0.5, 0.1, 9.0, 8.0, 7.0], dtype=np.float32)
        # The one writer's rows on the walk of each world's factor (z = A^-T J^T), their incident velocities and
        # diagonals |z|^2 + cfm (FL3.6: the local kernel reads them and leaves by the walk).
        stored = [walked_rows(lower, jacobian[0]), jacobian[1]]
        incident = np.stack([jacobian[0] @ velocity[:dofs], jacobian[1] @ velocity[dofs:]])
        diagonal = cfm.copy()
        diagonal[0] += np.sum(stored[0] * stored[0], axis=1)
        self.walk = chain_walk(device, [lower, np.eye(dofs)])
        # Outputs: the diagonal (initially the one writer's), the impulses and the articulations' velocities.
        self.initial = (
            wp.array(diagonal, dtype=wp.float32, device=device),
            wp.zeros((2, rows), dtype=wp.float32, device=device),
            wp.array(velocity, dtype=wp.float32, device=device),
        )
        self.outputs = tuple(wp.clone(array) for array in self.initial)
        self.inputs = [
            candidates,
            candidates,
            PGS_LOCAL_SOLVE_OWNER_SINGLE,
            wp.array([0, 1], dtype=wp.int32, device=device),
            wp.array([0, 3], dtype=wp.int32, device=device),
            # World 1 belongs to another owner: its solve returns before any sweep.
            wp.array([PGS_LOCAL_SOLVE_OWNER_SINGLE, 0], dtype=wp.int32, device=device),
            wp.array([2, 1], dtype=wp.int32, device=device),
            row_lines(device),
            *local_packed_rows(device, [0, 1], stored, incident),
            wp.array(rhs, dtype=wp.float32, device=device),
            wp.array(row_type, dtype=wp.int32, device=device),
            wp.full((2, rows), -1, dtype=wp.int32, device=device),
            wp.zeros((2, rows), dtype=wp.float32, device=device),
            wp.zeros(2, dtype=wp.int32, device=device),
            wp.zeros((2, 4), dtype=wp.int32, device=device),
            wp.zeros((2, 1, 12), dtype=wp.float32, device=device),
            0,
            wp.zeros((2, rows), dtype=wp.float32, device=device),
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
            inputs=[*self.inputs, iterations, 1.0, *self.outputs, self.walk, *extra],
            block_dim=32,
            device=self.device,
        )
        leave_walk(self.walk, self.outputs[2], self.device)

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
            ((43, 18, False, None), {}, 1.0),
            ((43, 18, True, None), {}, 0.7),
            ((43, 18, False, None), {"active_rows": True}, 1.0),
            ((3, 3, False, "late_activation"), {"lanes_per_world": 16}, 1.0),
        )
        for (dofs, support, seeded, activity), options, omega in cases:
            with self.subTest(dofs=dofs, seeded=seeded, activity=activity, options=options):
                problem = _problem(dofs, support, seeded, contact_activity=activity)
                case = _SparseCase(device, problem, options, omega)
                exact, counted = case.kernel(), case.kernel(SWEEPS)
                outputs = []
                for budget in range(BUDGET + 1):
                    case.run(exact, budget)
                    outputs.append(case.read())
                allowed = _expected_counts(outputs)
                operand = _readout(device, case.worlds)
                case.run(counted, BUDGET, operand)
                for actual, expected in zip(case.read(), outputs[BUDGET], strict=True):
                    for actual_array, expected_array in zip(actual, expected, strict=True):
                        np.testing.assert_array_equal(actual_array, expected_array)
                _assert_counts(self, operand, allowed, solved=range(case.worlds))
                # Graph replays add one record per replay to the captured operand.
                operand = _readout(device, case.worlds)
                with wp.ScopedCapture(device=device) as capture:
                    case.run(counted, BUDGET, operand)
                for _ in range(2):
                    wp.capture_launch(capture.graph)
                for actual, expected in zip(case.read(), outputs[BUDGET], strict=True):
                    for actual_array, expected_array in zip(actual, expected, strict=True):
                        np.testing.assert_array_equal(actual_array, expected_array)
                _assert_counts(self, operand, allowed, solved=range(case.worlds), repeats=2)

    def test_local_counts_sweeps_to_the_stationary_exit(self):
        """Count only the owned world's sweeps to its first stationary sweep, with bitwise outputs."""
        case = _LocalCase(wp.get_cuda_device())
        exact, counted = case.kernel(), case.kernel(SWEEPS)
        outputs = []
        for budget in range(BUDGET + 1):
            case.run(exact, budget)
            outputs.append(case.read())
        allowed = _expected_counts(outputs)
        operand = _readout(case.device, case.worlds)
        case.run(counted, BUDGET, operand)
        _assert_bitwise(case.read(), outputs[BUDGET])
        _assert_counts(self, operand, allowed, solved=(0,))


if __name__ == "__main__":
    unittest.main()
