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
"""

import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
    PGS_CONSTRAINT_TYPE_MIMIC,
    PGS_LOCAL_SOLVE_OWNER_SINGLE,
)
from newton._src.solvers.feather_pgs.solve_policy import SolvePolicy, SolveReadout
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

    def test_local_counts_sweeps_to_the_stationary_exit(self):
        """Count only the owned world's sweeps to its first stationary sweep, with bitwise outputs."""
        device = wp.get_cuda_device()
        dofs, rows, local_rows = 3, 4, 2
        lower = np.array([[2.0, 0.0, 0.0], [0.25, 1.5, 0.0], [-0.1, 0.2, 1.25]], dtype=np.float32)
        jacobian = np.zeros((2, rows, dofs), dtype=np.float32)
        jacobian[0, 0] = [1.0, 0.5, -0.25]
        jacobian[0, 1] = [-0.4, 0.3, 0.8]
        jacobian[1, 0] = [0.2, -0.1, 0.7]
        velocity = np.array([0.3, -0.5, 0.1, 9.0, 8.0, 7.0], dtype=np.float32)
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
        mf = [
            wp.zeros(2, dtype=wp.int32, device=device),
            wp.zeros((2, 4), dtype=wp.int32, device=device),
            wp.zeros((2, 1), dtype=wp.float32, device=device),
            *[wp.zeros((2, 1, 6), dtype=wp.float32, device=device) for _ in range(4)],
            wp.zeros((2, 1), dtype=wp.float32, device=device),
            0,
            wp.zeros((2, rows), dtype=wp.float32, device=device),
            wp.zeros((2, 1), dtype=wp.float32, device=device),
        ]

        def launch(kernel, iterations, *extra):
            diagonal = wp.array(cfm, dtype=wp.float32, device=device)
            impulses = wp.zeros((2, rows), dtype=wp.float32, device=device)
            velocities = wp.array(velocity, dtype=wp.float32, device=device)
            wp.launch_tiled(
                kernel,
                dim=[2],
                inputs=[
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
                    *mf,
                    iterations,
                    1.0,
                    0,
                    0,
                ],
                outputs=[diagonal, impulses, velocities, *extra],
                block_dim=32,
                device=device,
            )
            velocity_out = velocities.numpy()
            return [
                (impulses.numpy()[world], velocity_out[3 * world : 3 * world + 3], diagonal.numpy()[world])
                for world in range(2)
            ]

        exact = _get_pgs_solve_local_owned_kernel(rows, local_rows, dofs, device.arch)
        counted = _get_pgs_solve_local_owned_kernel(rows, local_rows, dofs, device.arch, solve_policy=SWEEPS)
        outputs = [launch(exact, budget) for budget in range(BUDGET + 1)]
        executed, stopped = _expected_counts(outputs)
        operand = _readout(device, 2)
        result = launch(counted, BUDGET, operand)
        for actual, expected in zip(result, outputs[BUDGET], strict=True):
            for actual_array, expected_array in zip(actual, expected, strict=True):
                np.testing.assert_array_equal(actual_array, expected_array)
        _assert_counts(self, operand, executed, stopped, solved=(0,))


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
        "prepare": prepare,
    }


def _mixed_route(hinv_jt_kernel, dofs, labels):
    """An arm struck by a sliding free box: local owners under ``par_row``, the paired factor sweep for 23 DOFs."""
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
        "prepare": prepare,
    }


def _matrix_free_route():
    """Contacting tripods beside free spheres: matrix-free Gauss-Seidel and its incremental tier."""
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
    }


@unittest.skipUnless(wp.is_cuda_available(), "Native Gauss-Seidel sweeps run on CUDA")
class TestSolveReadoutRoutes(unittest.TestCase):
    """Every counted route leaves the trajectory unchanged and counts its world-solves."""

    def _lockstep(self, route, *, steps=4, dt=1.0 / 240.0):
        """Step two solvers without and one with the readout from one state on shared contacts.

        The second solver without readout is a control: where the route itself is
        bitwise reproducible, the readout solver must match it bitwise; where
        atomically ordered bookkeeping makes the route vary between runs, it must
        stay as close as the control does. Returns the readout solver, its
        summary and the model.
        """
        model, options = route["model"], {**route["options"], "double_buffer": False, "pgs_iterations": BUDGET}
        solvers = []
        with mock.patch.object(SolverFeatherPGS, "_kernel_overrides", route.get("overrides", {})):
            for readout in ("off", "off", "sweeps"):
                solvers.append(SolverFeatherPGS(model, pgs_solve_readout=readout, **options))
        counted = solvers[-1]
        self.assertTrue(set(counted._solve_readout.kernels.values()) & route["labels"], counted._solve_readout.kernels)
        self.assertEqual(solvers[0]._solve_readout.kernels, {})
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
            for name in STATE_FIELDS:
                reference, control_run, actual = (getattr(state, name).numpy() for state, _ in states)
                if np.array_equal(control_run, reference):
                    np.testing.assert_array_equal(actual, reference, err_msg=f"{name} at step {step}")
                else:
                    spread = float(np.abs(control_run - reference).max())
                    np.testing.assert_allclose(actual, reference, rtol=0.0, atol=10.0 * spread, err_msg=name)
        record = counted.get_path_selection()
        self.assertEqual(record["options"]["pgs_solve_readout"]["applied"], "sweeps")
        readout = record["routes"]["sweep"]["readout"]
        self.assertEqual(readout["mode"], "sweeps")
        self.assertEqual(readout["counted"], counted._solve_readout.kernels)
        summary = record["measurements"]["solve_readout"]
        self.assertEqual(summary["readout"], "sweeps")
        for slot in summary["slots"]:
            with self.subTest(kernel=slot["kernel"]):
                self.assertEqual(slot["budget"], BUDGET)
                self.assertEqual(slot["solves"], sum(slot["sweeps"]))
                self.assertEqual(slot["sweeps"][0], 0)
                self.assertLessEqual(slot["exhausted"], slot["sweeps"][BUDGET])
                self.assertLessEqual(slot["worlds"], model.world_count)
        self.assertGreater(sum(slot["solves"] for slot in summary["slots"]), 0)
        return counted, summary, model

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
        """Count the local owners' sweeps without changing the trajectory."""
        self._lockstep(_mixed_route("par_row", 13, {"local_internal", "local_pair", "local_residual", "mf_gs"}))

    def test_paired_route_counts_without_changing_results(self):
        """Count the paired factor sweep without changing the trajectory."""
        solver, summary, _ = self._lockstep(_mixed_route("auto", 23, {"paired_factor"}))
        self.assertIsNotNone(solver._paired_factor_solve_kernel)
        self.assertGreater(sum(slot["solves"] for slot in summary["slots"] if slot["route"] == "paired_factor"), 0)

    def test_matrix_free_route_counts_without_changing_results(self):
        """Count the matrix-free Gauss-Seidel sweep and its incremental tier without changing the trajectory."""
        self._lockstep(_matrix_free_route())


if __name__ == "__main__":
    unittest.main()
