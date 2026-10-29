# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Native sparse sweeps stop a converging solve at the exit law's rounding floor, and a drifting one never."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.solver_feather_pgs import SolverFeatherPGS

BUDGET = 32
"""Sweep budget of the rollouts: the bitwise exit ran every sweep of their first solve, which the floor shortens."""


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


@unittest.skipUnless(wp.is_cuda_available(), "Native sparse factors are a CUDA path")
class TestFeatherPGSSparseExitLaw(unittest.TestCase):
    def _rollout(self, iterations, *, readout="sweeps", steps=12, dt=0.005, hip_pitch=False):
        device = wp.get_device("cuda:0")
        model, contacts = _build(device, hip_pitch=hip_pitch)
        solver = SolverFeatherPGS(
            model,
            pgs_mode="split",
            sparse_mass_matrix="on",
            pgs_iterations=iterations,
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
        (slot,) = solver.get_path_selection()["measurements"]["solve_readout"]["slots"]
        return qd, slot

    def test_a_converging_solve_stops_at_the_floor_near_the_converged_state(self):
        """Stop the converging rollout's solves before their budget, every stop at the floor, near a long solve.

        The legs pitch at the hip, so the six contact rows meet seven coordinates and the sweep converges: its
        solves stop before the budget, the stopped ones' final sweeps change by at most the floor (magnitude bins
        below 10^-2.5, under the trace's 26 bins), and every joint velocity stays within 1e-3 of a rollout at four
        times the budget, where the floor stops the solves too.
        """
        short, slot = self._rollout(BUDGET, readout="trace", hip_pitch=True)
        long, long_slot = self._rollout(4 * BUDGET, hip_pitch=True)
        self.assertLess(slot["mean_sweeps"], BUDGET)
        self.assertLess(slot["exhausted"], slot["solves"])
        self.assertLess(long_slot["exhausted"], long_slot["solves"])
        self.assertLess(float(np.abs(short - long).max()), 1.0e-3)
        trace = slot["trace"]
        self.assertEqual(sum(trace["reasons"].values()), slot["solves"])
        beyond = sum(sum(entry["bins"][14:]) for entry in trace["final_change"].values())
        self.assertLessEqual(beyond, trace["reasons"]["budget"])

    def test_a_drifting_solve_runs_its_budget(self):
        """Run every sweep of the drifting rollout's first solve: its impulses drift far above any rounding floor.

        On legs without hip pitch the six contact rows meet five coordinates: the Delassus matrix is singular and
        the rows' targets inconsistent, so the sweep drifts along its null space by about 0.2 m/s every sweep.
        """
        _, slot = self._rollout(BUDGET, steps=1)
        self.assertEqual(slot["exhausted"], slot["solves"])
        self.assertEqual(slot["mean_sweeps"], BUDGET)


if __name__ == "__main__":
    unittest.main()
