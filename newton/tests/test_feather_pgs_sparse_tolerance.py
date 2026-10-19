# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Native sparse sweeps honor impulse-change and velocity-change tolerances for early exit."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.solve_policy import TRACE_GRID
from newton._src.solvers.feather_pgs.solver_feather_pgs import SolverFeatherPGS

TOLERANCE = TRACE_GRID[8]
"""The velocity tolerance of the tests, 1e-3 [m/s or rad/s], a point of the trace's grid."""


def _build(device, worlds=8):
    """Fixed-base revolute chains whose tips press spheres into the ground."""
    builder = newton.ModelBuilder()
    shape_cfg = builder.default_shape_cfg.copy()
    shape_cfg.density = 0.0
    shape_cfg.mu = 0.7
    shapes = []
    for _world in range(worlds):
        builder.begin_world()
        # A fixed-base torso with two legs: the branching keeps the mass factor sparse.
        torso = builder.add_link(mass=2.0, inertia=wp.mat33(np.eye(3) * 0.1), lock_inertia=True)
        joints = [
            builder.add_joint_revolute(
                parent=-1,
                child=torso,
                axis=newton.Axis.Z,
                parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.48), wp.quat_identity()),
            )
        ]
        for side in (-1.0, 1.0):
            parent = torso
            for link in range(2):
                body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * 0.05), lock_inertia=True)
                offset = wp.vec3(0.0, 0.15 * side, 0.0) if link == 0 else wp.vec3(0.0, 0.0, -0.25)
                joints.append(
                    builder.add_joint_revolute(
                        parent=parent,
                        child=body,
                        axis=newton.Axis.Y if link else newton.Axis.X,
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
    ):
        device = wp.get_device("cuda:0")
        model, contacts = _build(device)
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
        """A small sweep tolerance stays close to the exact stationary exit; zero is identical."""
        exact, _ = self._rollout(8)
        again, _ = self._rollout(8, tolerance=0.0)
        np.testing.assert_array_equal(exact, again)
        loose, _ = self._rollout(8, tolerance=1.0e-4)
        self.assertLess(float(np.abs(loose - exact).max()), 1.0e-2)
        relative, _ = self._rollout(8, relative_tolerance=1.0e-2)
        self.assertLess(float(np.abs(relative - exact).max()), 1.0e-2)
        with self.assertRaises(ValueError):
            self._rollout(8, tolerance=-1.0)
        with self.assertRaises(ValueError):
            self._rollout(8, relative_tolerance=1.0)

    def test_velocity_tolerance_stops_earlier_near_the_exact_law(self):
        """Stay near the exact law in fewer sweeps under a velocity tolerance, with every exit sweep within it.

        Over a 12-step rollout a 1e-3 tolerance stays within 1e-2 of the exact law, executes no more
        sweeps, and stops only on final sweeps within the tolerance. On a first solve, identical in
        both runs, at a budget of 64 sweeps the tolerance stops each world exactly where the exact
        law's trace predicts for 1e-3, which is earlier than the exact law.
        """
        exact, _ = self._rollout(8)
        counted, exact_solver = self._rollout(8, readout="sweeps")
        np.testing.assert_array_equal(counted, exact)
        again, _ = self._rollout(8, velocity_tolerance=0.0, readout="trace")
        np.testing.assert_array_equal(again, exact)
        loose, solver = self._rollout(8, velocity_tolerance=TOLERANCE, readout="trace")
        self.assertLess(float(np.abs(loose - exact).max()), 1.0e-2)
        law = solver.get_path_selection()["routes"]["sweep"]["exit_law"]
        self.assertEqual((law["law"], list(law["honored"].values())), ("velocity", ["sparse_pgs"]))
        (slot,) = solver.get_path_selection()["measurements"]["solve_readout"]["slots"]
        (exact_slot,) = exact_solver.get_path_selection()["measurements"]["solve_readout"]["slots"]
        self.assertEqual(slot["solves"], exact_slot["solves"])
        self.assertLessEqual(slot["mean_sweeps"], exact_slot["mean_sweeps"])
        trace = slot["trace"]
        self.assertEqual(sum(trace["reasons"].values()), slot["solves"])
        # A stopped solve's final sweep changed by at most 1e-3, in magnitude bins below 10^-2.5 (14 and up),
        # so only solves that ran out of budget reach those bins.
        beyond = sum(sum(entry["bins"][14:]) for entry in trace["final_change"].values())
        self.assertLessEqual(beyond, trace["reasons"]["budget"])

        budget = 64
        _, traced = self._rollout(budget, readout="trace", steps=1)
        _, stopped = self._rollout(budget, velocity_tolerance=TOLERANCE, readout="trace", steps=1)
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
            self._rollout(8, velocity_tolerance=-1.0)


if __name__ == "__main__":
    unittest.main()
