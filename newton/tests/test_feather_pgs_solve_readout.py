# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Count the Gauss-Seidel sweeps every solve executes without changing a computed bit.

Under ``pgs_solve_readout="sweeps"`` each sweep kernel records, per world, the
sweeps a solve executed and whether it stopped before its budget. A sweep
stops a world after a sweep whose changes all lie within the exit law's
rounding floor, so a world that stops after ``s`` sweeps has the same impulses
and velocities at every budget from ``s`` on, and its outputs change at every
budget before: they settle at ``s``, or at ``s - 1`` when sweep ``s`` changed
nothing. The kernel tests replay a solve at every sweep budget and check the
counts against that rule; the solver tests check that every route's readout
leaves the trajectory bitwise unchanged and counts each world-solve once: in
the slot of the sweep that solved it, or, for a world without rows that the
matrix-free sweeps do not enter, as skipped. The parallel tiers count their own
worlds, against their own sweep budget. Velocity-limit rows bind on a route of
their own.
"""

import unittest
import warnings
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import sweep_routes_local
from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
    PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT,
    PGS_CONSTRAINT_TYPE_MIMIC,
    PGS_LOCAL_SOLVE_OWNER_SINGLE,
)
from newton._src.solvers.feather_pgs.row_metadata import _DENSE_META_ROW_TYPE_MASK
from newton._src.solvers.feather_pgs.solve_policy import (
    TRACE_GRID,
    TRACE_KINDS,
    TRACE_REASONS,
    SolvePolicy,
    SolveReadout,
    SolveTrace,
)
from newton._src.solvers.feather_pgs.sparse_mass_matrix import _SparseRowSupport
from newton._src.solvers.feather_pgs.sweep_routes_sparse import sparse_kernel
from newton.solvers import SolverFeatherPGS
from newton.tests.feather_pgs_law_harness import local_packed_rows

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


def _expected_counts(outputs, *, offset=0, friction_start=0):
    """Return the sweeps each world may have executed, from per-budget outputs ``outputs[s][world]``.

    The outputs settle at budget ``c``, the first from which they never change. A world stops after the first
    sweep, once friction rows are active (global sweep index at least ``friction_start``), whose changes were all
    quiet: sweep ``c`` (its changes within the floor) or ``c + 1`` (no change), or the first open sweep after them.
    A world whose outputs settle only at the budget may have run it out. Returns per world the allowed counts.
    """
    budget = len(outputs) - 1
    allowed = []
    for world in range(len(outputs[0])):

        def same(a, b, world=world):
            return all(np.array_equal(x, y) for x, y in zip(outputs[a][world], outputs[b][world], strict=True))

        settle = next(c for c in range(budget + 1) if all(same(c, later) for later in range(c, budget + 1)))
        first = max(settle, 1)
        while first < budget and offset + first - 1 < friction_start:
            first += 1
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
            lower_group,
            *local_packed_rows(device, [0, 1], jacobian),
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


def _skipped(summary):
    """Skipped world-solves of a readout summary."""
    return summary["skipped"]["solves"] if summary["skipped"] else 0


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
                allowed = _expected_counts(outputs, offset=offset, friction_start=friction_start)
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


def _sparse_route():
    """Fixed-base legs pressing spheres into the ground, on native sparse factors (explicit contacts)."""
    from newton.tests.test_feather_pgs_sparse_tolerance import _build  # noqa: PLC0415

    model, contacts = _build(wp.get_device("cuda:0"))

    def prepare(solver, state):
        state.joint_qd.fill_(0.3)

    return {
        "model": model,
        "options": {"friction_anchor_beta": 0.0},
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
        for art in np.flatnonzero(solver.forest.response_dof_count == 16):
            start = int(solver.forest.articulation_dof_start_host[art])
            joint_qd[start : start + 16] = -0.3
        state.joint_qd.assign(joint_qd)

    return {
        "model": _build_sparse_contact_friction_model(device="cuda:0"),
        "options": {
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
        free = int(np.flatnonzero(solver.forest.is_free_rigid_host)[0])
        start = int(solver.forest.articulation_dof_start_host[free])
        joint_qd[start], joint_qd[start + 2] = 2.0, -3.0
        state.joint_qd.assign(joint_qd)

    return {
        "model": _build_mixed_response_model("cuda:0", dof_count=dofs, friction=0.7),
        "options": {
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

    The spheres' ground contacts are matrix-free rows, which the general sweep solves.
    """
    from newton.tests.test_feather_pgs_sparse_solver import _build_model  # noqa: PLC0415

    return {
        "model": _build_model(free_count=2, robot_count=2),
        "options": {
            "friction_anchor_beta": 0.0,
            "enable_joint_limits": True,
            "dense_max_constraints": 96,
            "mf_max_constraints": 64,
        },
        "overrides": {"storage": "dense"},
        "labels": {"mf_gs"},
        "owners": ({"mf_gs"},),
        "solving": {"mf_gs"},
    }


def _dense_route():
    """Contacting tripods without free bodies: every world holds only dense rows, which the general sweep owns.

    Each world's three feet start in the ground and one leg beyond its limit; without free bodies no
    world has matrix-free rows or a local owner. Pushed out of the ground by the first step, the
    tripods rise for a few steps, in which every row of their worlds separates at the predictor: the
    resolved-world classifier leaves those worlds without rows, and no sweep enters them.
    """
    from newton.tests.test_feather_pgs_sparse_solver import _build_model  # noqa: PLC0415

    return {
        "model": _build_model(),
        "options": {
            "friction_anchor_beta": 0.0,
            "enable_joint_limits": True,
            "dense_max_constraints": 96,
            "mf_max_constraints": 64,
        },
        "overrides": {"storage": "dense"},
        "labels": {"mf_gs"},
        "owners": ({"mf_gs"},),
        "solving": {"mf_gs"},
    }


def _velocity_limit_route():
    """The dense route's tripods with velocity-limited legs.

    Each leg joint is limited to 1 rad/s and starts at 2 rad/s, so the velocity-limit rows bind
    from the first step, and each row keeps its total across sweeps, which the votes measure as
    the change of that total. The rows start from half the limit and add at most two per leg to a
    world's rows.
    """
    route = _dense_route()
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
    }
    route.update(
        name="velocity_limits",
        labels={"mf_gs"},
        owners=({"mf_gs"},),
        solving=set(),
        prepare=prepare,
    )
    return route


def _tier_route():
    """The dense route's tripods with the Jacobi tiers a test pin builds (``tiers``): 32 rows.

    Each tripod's world holds only dense rows (24 while its feet touch the ground: six contacts and
    six joint-limit rows), within the tiers' bound, so the tiers solve and count it and the
    matrix-free sweep returns before its first sweep there; a world that outgrows the bound keeps the
    counted matrix-free sweep.
    """
    route = _dense_route()
    route.update(
        overrides={"storage": "dense", "tiers": 32},
        labels={"mf_gs", "parallel_tier"},
        owners=({"mf_gs", "parallel_tier"},),
        solving={"parallel_tier"},
    )
    return route


_LOCAL = {"local_internal", "local_pair", "local_residual"}


def _local_route():
    """The mixed arm under ``par_row``: local owners claim the world, the paired owner while the box touches it."""
    return _mixed_route(
        "par_row",
        13,
        {*_LOCAL, "mf_gs"},
        owners=({*_LOCAL, "mf_gs"},),
        solving={"local_pair"},
    )


def _paired_route():
    """The mixed arm with 23 DOFs: the paired factor sweep owns the world, which has no matrix-free rows."""
    return _mixed_route("auto", 23, {"paired_factor"}, owners=({"paired_factor", "mf_gs"},), solving={"paired_factor"})


def _policy_routes():
    """The routes whose local, paired and matrix-free sweeps compile the velocity law and trace.

    The last one binds velocity-limit rows.
    """
    return (_local_route(), _paired_route(), _matrix_free_route(), _dense_route(), _velocity_limit_route())


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
        options = {**route["options"], "pgs_iterations": BUDGET}
        with mock.patch.object(SolverFeatherPGS, "_route_pins", route.get("overrides", {})):
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
        # The route record says whether the summary holds skipped solves.
        self.assertEqual(counts["counts_skipped"], not route.get("enters_rowless_worlds"))
        self.assertEqual(summary["skipped"] is not None, counts["counts_skipped"])
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
                solves = sum(slot["solves"] for slot in summary["slots"] if slot["route"] in owners)
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

    def test_dense_route_counts_without_changing_results(self):
        """Count the dense rows' sweeps without changing the trajectory, and the solves of resolved worlds.

        While the tripods rise, the resolved-world classifier leaves their worlds without rows and no
        sweep enters them; the readout counts those solves as skipped, so with the sweep's slot they
        cover every world at every solve.
        """
        solver, summary, model = self._lockstep(_dense_route())
        self.assertTrue(solver.get_path_selection()["routes"]["resolved_worlds"]["enabled"])
        skipped = summary["skipped"]
        self.assertGreater(skipped["solves"], 0)
        self.assertLessEqual(skipped["worlds"], model.world_count)

    def test_velocity_limit_route_counts_without_changing_results(self):
        """Count the sweeps of binding velocity-limit rows, every world-solve once.

        A velocity-limit row keeps its total across sweeps. The route's first step holds velocity-limit
        rows with positive totals; the readout changes no result and counts every world-solve once.
        """
        route = _velocity_limit_route()
        (probe,) = self._solvers(route, [{"pgs_solve_readout": "sweeps"}])
        self.assertTrue(probe._solve_readout.kernels)
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

        The tiers solve the tripods' worlds within their bound, the matrix-free sweep any other: their
        slots and the skip counter must add up to every world at every step.
        """
        route, steps = _tier_route(), 4
        solver, summary, _ = self._lockstep(route, steps=steps)
        # The route's premise: no local owner claims the tripod's world, so the tiers own it.
        self.assertFalse(solver._local_internal_fast_path)
        sweep = solver.get_path_selection()["routes"]["sweep"]
        tier_kernels = {tier["kernel"] for tier in sweep["tiers"]["tiers"]}
        self.assertIn(sweep["tiers"]["family"], ("parallel", "keyed_parallel"))
        self.assertLessEqual(tier_kernels, set(sweep["readout"]["counted"]))
        tiers = sum(slot["solves"] for slot in summary["slots"] if slot["route"] == "parallel_tier")
        self.assertGreater(tiers, 0)

    def test_torsion_traces_its_changes(self):
        """Trace the matrix-free sweep with contact torsion, without a warning: every sweep compiles the trace."""
        route = _matrix_free_route()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            (solver,) = self._solvers(
                route, [{"pgs_solve_readout": "trace", "contact_torsion": True, "dense_max_constraints": 256}]
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


if __name__ == "__main__":
    unittest.main()
