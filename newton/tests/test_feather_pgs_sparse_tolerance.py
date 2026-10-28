# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Native sparse sweeps honor impulse-change and velocity-change tolerances for early exit."""

import functools
import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import sweep_routes_split
from newton._src.solvers.feather_pgs.kernels import PGS_CONSTRAINT_TYPE_FRICTION
from newton._src.solvers.feather_pgs.solve_policy import TRACE_GRID
from newton._src.solvers.feather_pgs.solver_feather_pgs import SolverFeatherPGS

TOLERANCE = TRACE_GRID[8]
"""The velocity tolerance of the tests, 1e-3 [m/s or rad/s], a point of the trace's grid."""

BUDGET = 32
"""Sweep budget of the rollouts: the exact law runs every sweep of their first solve, which a tolerance shortens."""

IMPULSE_TOLERANCES = {"tolerance": 1.0e-4, "relative_tolerance": 1.0e-3}
"""Impulse-change tolerances of the tests: absolute [N s or N m s], and relative to the row's new impulse.

On the CPU twin of the rollouts a relative 1e-2 already stops the first solve after 11 sweeps and moves
the joint velocities by 1.2e-2 from the exact law's, beyond the bound the tests hold a small tolerance to.
"""


def _build(device, worlds=8, *, hip_pitch=False):
    """Fixed-base revolute chains whose tips press spheres into the ground.

    Each leg rolls at the hip and pitches at the knee, so the two feet's six contact rows meet five
    coordinates: the Delassus matrix is singular and the rows' velocity targets are inconsistent, so
    a sweep drifts along its null space instead of settling. ``hip_pitch`` adds a pitch joint at each
    hip; the six rows then meet seven coordinates and the sweep converges.
    """
    builder = newton.ModelBuilder()
    shape_cfg = builder.default_shape_cfg.copy()
    shape_cfg.density = 0.0
    shape_cfg.mu = 0.7
    shapes = []
    legs = ((newton.Axis.X, 0.0), (newton.Axis.Y, 0.0), (newton.Axis.Y, -0.25))
    if not hip_pitch:
        legs = (legs[0], legs[2])
    for _world in range(worlds):
        builder.begin_world()
        # A torso turning on a fixed base, with two legs: the branching keeps the mass factor sparse, and the held
        # tree, which alone evaluates native sparse factors, takes the fixed root.
        base = builder.add_link(mass=2.0, inertia=wp.mat33(np.eye(3) * 0.1), lock_inertia=True)
        torso = builder.add_link(mass=2.0, inertia=wp.mat33(np.eye(3) * 0.1), lock_inertia=True)
        joints = [
            builder.add_joint_fixed(
                parent=-1, child=base, parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.48), wp.quat_identity())
            ),
            builder.add_joint_revolute(parent=base, child=torso, axis=newton.Axis.Z),
        ]
        for side in (-1.0, 1.0):
            parent = torso
            for link, (axis, drop) in enumerate(legs):
                body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * 0.05), lock_inertia=True)
                offset = wp.vec3(0.0, 0.15 * side, 0.0) if link == 0 else wp.vec3(0.0, 0.0, drop)
                joints.append(
                    builder.add_joint_revolute(
                        parent=parent,
                        child=body,
                        axis=axis,
                        parent_xform=wp.transform(offset, wp.quat_identity()),
                    )
                )
                parent = body
            shapes.append(
                builder.add_shape_sphere(
                    parent, radius=0.03, xform=wp.transform(wp.vec3(0.1, 0.0, -0.2), wp.quat_identity()), cfg=shape_cfg
                )
            )
        builder.add_articulation(joints)
        builder.end_world()
    plane = builder.add_ground_plane()
    model = builder.finalize(device=device)
    count = len(shapes)
    model.rigid_contact_max = count
    contacts = newton.Contacts(rigid_contact_max=count, soft_contact_max=0, device=device)
    contacts.rigid_contact_count.assign(np.array([count], dtype=np.int32))
    contacts.rigid_contact_shape0.assign(np.asarray(shapes, dtype=np.int32))
    contacts.rigid_contact_shape1.assign(np.full(count, plane, dtype=np.int32))
    # Feet rest 2 mm into the ground: torso at z=0.48, hip drop 0.25, shin drop 0.2, sphere radius 0.03.
    points = np.zeros((count, 3), dtype=np.float32)
    points[:, 0] = 0.1
    points[:, 2] = -0.2 - 0.03 - 0.002
    contacts.rigid_contact_point0.assign(points)
    ground = np.zeros((count, 3), dtype=np.float32)
    ground[:, 0] = 0.1
    ground[:, 1] = np.tile([-0.15, 0.15], worlds)
    contacts.rigid_contact_point1.assign(ground)
    contacts.rigid_contact_normal.assign(np.tile([0.0, 0.0, -1.0], (count, 1)).astype(np.float32))
    return model, contacts


def _first_solve_sweeps(model, contacts, sweeps, dt=0.005):
    """Replay the rollouts' first solve on the CPU, one sweep at a time, and return each row's change per sweep.

    The CPU split route assembles the rows the native sparse sweep solves (Delassus matrix, right-hand
    side and diagonal, from the rollouts' first state), and its dense loop kernel applies the same
    sweep law, one sweep per launch, from the zero impulses the rollouts start from. ``change`` holds
    each row's impulse change, shape ``[sweeps, world_count, rows]``; times the row's diagonal
    ``A_ii`` (``diagonal``) it is the velocity-scaled change the trace records. ``slack_impulse``
    holds the new impulse that scales the row's relative impulse tolerance in the sparse sweep's
    vote, its own or, for a friction pair's second tangent, the first tangent's. ``live`` marks each
    world's rows, ``[world_count, rows]``.
    """
    solver = SolverFeatherPGS(model, pgs_mode="split", pgs_iterations=1, friction_anchor_beta=0.0)
    state_in, state_out = model.state(), model.state()
    state_in.joint_qd.fill_(0.3)
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
    solver.step(state_in, state_out, model.control(), contacts, dt)
    impulses = wp.zeros_like(solver.impulses)
    history = [impulses.numpy().copy()]
    for sweep in range(sweeps):
        wp.launch(
            sweep_routes_split.loop_kernel(),
            dim=solver.world_count,
            inputs=[
                solver.constraint_count,
                solver.diag,
                solver.C,
                solver.rhs,
                impulses,
                1,
                solver.pgs_omega,
                solver.row_type,
                solver.row_parent,
                solver.row_mu,
                solver.row_w,
                int(solver._regularization_enabled),
                0,
                sweep,
            ],
            device=model.device,
        )
        history.append(impulses.numpy().copy())
    history = np.stack(history)
    counts, parent = solver.constraint_count.numpy(), solver.row_parent.numpy()
    rows = np.arange(parent.shape[1])
    second_tangent = (solver.row_type.numpy() == PGS_CONSTRAINT_TYPE_FRICTION) & (rows == parent + 2)
    slack_row = np.broadcast_to(np.where(second_tangent, parent + 1, rows), history[1:].shape)
    return SimpleNamespace(
        change=np.abs(np.diff(history, axis=0)),
        slack_impulse=np.abs(np.take_along_axis(history[1:], slack_row, axis=2)),
        diagonal=solver.diag.numpy(),
        live=rows < counts[:, None],
    )


def _first_stops(quiet, live):
    """Return, per world, the first sweep after which every live row is quiet, or 0 if no sweep is."""
    quiet = np.where(live, quiet, True).all(axis=-1)
    return np.where(quiet.any(axis=0), quiet.argmax(axis=0) + 1, 0)


class TestSparseToleranceFixture(unittest.TestCase):
    """The tolerance tests' premise, replayed on the CPU twin of the sweep law."""

    def test_first_solve_falls_within_the_tolerance_early(self):
        """Fall within 1e-3 in every world well before the rollouts' budget, all of which the exact law runs.

        The velocity-tolerance test stops these solves at their first sweep within 1e-3 and needs it
        to come before the exact law stops; on legs that pitch at the hip no sweep within the budget
        leaves every impulse unchanged. On legs without hip pitch the replay drifts instead, by about
        0.2 m/s every sweep, so neither law would stop within the budget.
        """
        model, contacts = _build("cpu", hip_pitch=True)
        replay = _first_solve_sweeps(model, contacts, BUDGET)
        self.assertEqual(_first_stops(replay.change == 0.0, replay.live).tolist(), [0] * len(replay.live))
        stops = _first_stops(replay.change * replay.diagonal <= TOLERANCE, replay.live)
        self.assertTrue((stops > 0).all(), stops)
        self.assertLessEqual(int(stops.max()), 3 * BUDGET // 4)

    def test_first_solve_meets_the_impulse_tolerances_early(self):
        """Meet each tested impulse-change tolerance in every world well before the rollouts' budget.

        The sparse sweep's vote holds every row's impulse change to the absolute tolerance plus the
        relative one times the new impulse that scales it; the impulse-tolerance test needs that to
        hold before the exact law stops, which runs every sweep of the budget. On legs without hip
        pitch the impulses drift by about 0.4 N s every sweep, so neither tolerance would stop it.
        """
        model, contacts = _build("cpu", hip_pitch=True)
        replay = _first_solve_sweeps(model, contacts, BUDGET)
        slack = {
            "tolerance": IMPULSE_TOLERANCES["tolerance"],
            "relative_tolerance": IMPULSE_TOLERANCES["relative_tolerance"] * replay.slack_impulse,
        }
        for name, bound in slack.items():
            with self.subTest(name):
                stops = _first_stops(replay.change <= bound, replay.live)
                self.assertTrue((stops > 0).all(), stops)
                self.assertLessEqual(int(stops.max()), 3 * BUDGET // 4)


@unittest.skipUnless(wp.is_cuda_available(), "Native sparse factors are a CUDA path")
class TestFeatherPGSSparseTolerance(unittest.TestCase):
    def _rollout(
        self,
        iterations,
        *,
        tolerance=0.0,
        relative_tolerance=0.0,
        velocity_tolerance=0.0,
        readout="off",
        steps=12,
        dt=0.005,
        hip_pitch=False,
    ):
        device = wp.get_device("cuda:0")
        model, contacts = _build(device, hip_pitch=hip_pitch)
        solver = SolverFeatherPGS(
            model,
            pgs_mode="split",
            sparse_mass_matrix="on",
            pgs_iterations=iterations,
            pgs_change_tolerance=tolerance,
            pgs_relative_change_tolerance=relative_tolerance,
            pgs_velocity_change_tolerance=velocity_tolerance,
            pgs_solve_readout=readout,
            friction_anchor_beta=0.0,
        )
        self.assertIsNotNone(solver._sparse_mass_matrix_size)
        states = [model.state() for _ in range(2)]
        states[0].joint_qd.fill_(0.3)
        newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
        control = model.control()
        for _ in range(steps):
            solver.step(states[0], states[1], control, contacts, dt)
            states.reverse()
        wp.synchronize_device(device)
        qd = states[0].joint_qd.numpy().copy()
        self.assertTrue(np.isfinite(qd).all())
        return qd, solver

    def test_change_tolerance_bounds_the_solution_change(self):
        """Stop earlier under a small impulse-change tolerance and stay close to the exact stationary exit.

        Over a 12-step rollout at a budget of 32 sweeps, all of which the exact law runs in the first
        solve, an absolute 1e-4 and a relative 1e-3 tolerance each stop that solve earlier and keep
        every joint velocity within 1e-2 of the exact law's; a zero tolerance is the exact law. The
        legs pitch at the hip, so the solve converges (``TestSparseToleranceFixture`` replays it on
        the CPU); on legs without it the impulses drift and no tolerance stops within the budget.
        """
        rollout = functools.partial(self._rollout, BUDGET, readout="sweeps", hip_pitch=True)
        exact, exact_solver = rollout()
        again, _ = rollout(tolerance=0.0)
        np.testing.assert_array_equal(exact, again)
        (exact_slot,) = exact_solver.get_path_selection()["measurements"]["solve_readout"]["slots"]
        for name, tolerance in IMPULSE_TOLERANCES.items():
            with self.subTest(name, tolerance=tolerance):
                loose, solver = rollout(**{name: tolerance})
                self.assertLess(float(np.abs(loose - exact).max()), 1.0e-2)
                (slot,) = solver.get_path_selection()["measurements"]["solve_readout"]["slots"]
                self.assertEqual(slot["solves"], exact_slot["solves"])
                self.assertLess(slot["mean_sweeps"], exact_slot["mean_sweeps"])
                self.assertLess(slot["exhausted"], exact_slot["exhausted"])
        with self.assertRaises(ValueError):
            self._rollout(8, tolerance=-1.0)
        with self.assertRaises(ValueError):
            self._rollout(8, relative_tolerance=1.0)

    def test_velocity_tolerance_stops_earlier_near_the_exact_law(self):
        """Stay near the exact law in fewer sweeps under a velocity tolerance, with every exit sweep within it.

        Over a 12-step rollout at a budget of 32 sweeps, all of which the exact law runs in the first
        solve, a 1e-3 tolerance stops that solve earlier, stays within 1e-2 of the exact law, executes
        fewer sweeps, and stops only on final sweeps within the tolerance. On a first solve, identical
        in both runs, at a budget of 64 sweeps the tolerance stops each world exactly where the exact
        law's trace predicts for 1e-3, which is earlier than the exact law. The legs pitch at the hip,
        so the sweep converges (``TestSparseToleranceFixture`` replays that solve on the CPU); on
        legs without it, it drifts and neither law stops within any budget.
        """
        rollout = functools.partial(self._rollout, hip_pitch=True)
        exact, _ = rollout(BUDGET)
        counted, exact_solver = rollout(BUDGET, readout="sweeps")
        np.testing.assert_array_equal(counted, exact)
        again, _ = rollout(BUDGET, velocity_tolerance=0.0, readout="trace")
        np.testing.assert_array_equal(again, exact)
        loose, solver = rollout(BUDGET, velocity_tolerance=TOLERANCE, readout="trace")
        self.assertLess(float(np.abs(loose - exact).max()), 1.0e-2)
        law = solver.get_path_selection()["routes"]["sweep"]["exit_law"]
        self.assertEqual((law["law"], list(law["honored"].values())), ("velocity", ["sparse_pgs"]))
        (slot,) = solver.get_path_selection()["measurements"]["solve_readout"]["slots"]
        (exact_slot,) = exact_solver.get_path_selection()["measurements"]["solve_readout"]["slots"]
        self.assertEqual(slot["solves"], exact_slot["solves"])
        self.assertLess(slot["mean_sweeps"], exact_slot["mean_sweeps"])
        trace = slot["trace"]
        self.assertEqual(sum(trace["reasons"].values()), slot["solves"])
        self.assertGreater(trace["reasons"]["tolerance"], 0)
        # A stopped solve's final sweep changed by at most 1e-3, in magnitude bins below 10^-2.5 (14 and up),
        # so only solves that ran out of budget reach those bins.
        beyond = sum(sum(entry["bins"][14:]) for entry in trace["final_change"].values())
        self.assertLessEqual(beyond, trace["reasons"]["budget"])

        budget = 64
        _, traced = rollout(budget, readout="trace", steps=1)
        _, stopped = rollout(budget, velocity_tolerance=TOLERANCE, readout="trace", steps=1)
        (predicted,) = traced.get_path_selection()["measurements"]["solve_readout"]["slots"]
        (actual,) = stopped.get_path_selection()["measurements"]["solve_readout"]["slots"]
        point = TRACE_GRID.index(TOLERANCE)
        stops = predicted["trace"]["stops"][point]
        self.assertEqual(sum(stops), predicted["solves"])
        np.testing.assert_array_equal(actual["sweeps"][1:budget], stops[1:budget])
        self.assertEqual(actual["sweeps"][budget], stops[budget] + stops[budget + 1])
        self.assertEqual(actual["exhausted"], stops[budget + 1])
        self.assertLess(actual["mean_sweeps"], predicted["mean_sweeps"])
        self.assertGreater(actual["trace"]["reasons"]["tolerance"], 0)
        with self.assertRaises(ValueError):
            rollout(8, velocity_tolerance=-1.0)


if __name__ == "__main__":
    unittest.main()
