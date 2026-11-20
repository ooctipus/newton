# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""A joint's velocity limit is its drive's no-load speed, not a constraint (:func:`.spatial_laws.motor_step`).

The drive's force on a step is ``clamp(f_PD(v+), lo(v+), hi(v+))`` with ``hi(v) = clip(E (1 - v / V), -E, E)`` and
``lo(v) = -hi(-v)``: the motor's torque in the direction it drives falls from its stall torque ``E`` (the effort
limit) at rest to zero at ``V`` (the velocity limit) and past it brakes (the back-EMF), down to ``-E`` at ``2 V``;
braking keeps ``E``; an external force drives the joint past ``V`` against that brake. The references are that law's
exact implicit step in float64.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS
from newton.tests import test_feather_pgs_coordinate_trees as branch

H = 1.0 / 240.0


def _motor_force(v, q, ke, kd, q_target, v_target, E, V, dt=H):
    """The motor law's force at the step's end velocity ``v`` (the drive's implicit PD within the motor's reach)."""
    pd = ke * (q_target - q - dt * v) + kd * (v_target - v)
    line = E / V
    return float(np.clip(pd, -np.clip(E + line * v, -E, E), np.clip(E - line * v, -E, E)))


def _leaf_step(inertia, kd, E, V, v_target, v, joint_f=0.0, dt=H):
    """One driven coordinate's exact step from ``v``: ``I (v+ - v) = dt (F(v+) + joint_f)`` solved by bisection (its
    residual is increasing in ``v+``, the law being monotone)."""
    lo, hi = -1.0e4, 1.0e4
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if inertia * (mid - v) - dt * (_motor_force(mid, 0.0, 0.0, kd, 0.0, v_target, E, V) + joint_f) > 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def _leaf_steps(inertia, kd, E, V, v_target, steps, joint_f=0.0, dt=H):
    """One driven coordinate's exact steps from rest (:func:`_leaf_step`)."""
    v, out = 0.0, []
    for _ in range(steps):
        v = _leaf_step(inertia, kd, E, V, v_target, v, joint_f, dt)
        out.append(v)
    return np.asarray(out)


def _kinks(q, ke, kd, q_target, v_target, E, V, dt=H):
    """Every end velocity where the motor law's force bends: the bounds' kinks and the drive's crossings of each
    bound's piece (the force is linear between them)."""
    pd_at_zero, pd_slope, line = ke * (q_target - q) + kd * v_target, ke * dt + kd, E / V
    kinks = [-2.0 * V, 0.0, 2.0 * V]
    # The bounds' pieces as value - slope v: hi's E and E - line v, lo's -E and -E - line v.
    for value, slope in ((E, 0.0), (E, line), (-E, 0.0), (-E, line)):
        if pd_slope != slope:
            kinks.append((pd_at_zero - value) / (pd_slope - slope))
    return sorted(kinks)


def _chain_step(mass, q, v, ke, kd, q_target, v_target, E, V, dt=H):
    """A collinear chain's exact step: ``mass (v+ - v) = dt F(v+)`` by Gauss-Seidel over the coordinates, each
    coordinate's equation solved exactly between the law's kinks, where it is linear (a convex problem: the sweeps
    converge to its one minimum)."""
    n = len(v)
    x = v.copy()
    kinks = [_kinks(q[i], ke, kd, q_target[i], v_target[i], E, V, dt) for i in range(n)]
    for _ in range(500):
        change = 0.0
        for i in range(n):
            rest = mass[i] @ (x - v) - mass[i, i] * (x[i] - v[i])

            def residual(m, i=i, rest=rest):
                force = _motor_force(m, q[i], ke, kd, q_target[i], v_target[i], E, V, dt)
                return mass[i, i] * (m - v[i]) + rest - dt * force

            points = [kinks[i][0] - 1.0e4, *kinks[i], kinks[i][-1] + 1.0e4]
            values = [residual(m) for m in points]
            for a, b, ra, rb in zip(points[:-1], points[1:], values[:-1], values[1:], strict=True):
                if ra <= 0.0 <= rb:
                    solved = a if rb == ra else a - ra * (b - a) / (rb - ra)
                    change = max(change, abs(solved - x[i]))
                    x[i] = solved
                    break
        if change < 1.0e-14:
            break
    return x


def _hinges(inertias, *, ke, kd, E, V, device="cpu"):
    """Collinear hinges in a chain (unit masses at their axes, so ``H`` is the suffix sums of ``inertias``), each
    driven with effort limit ``E`` and velocity limit ``V``, gravity off."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    parent, joints = -1, []
    for inertia in inertias:
        link = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * inertia))
        joints.append(
            builder.add_joint_revolute(
                parent=parent,
                child=link,
                axis=(0.0, 0.0, 1.0),
                target_ke=ke,
                target_kd=kd,
                effort_limit=E,
                velocity_limit=V,
                armature=0.0,
                limit_lower=-1.0e6,
                limit_upper=1.0e6,
            )
        )
        parent = link
    builder.add_articulation(joints)
    return builder.finalize(device=device)


def _roll(model, steps, *, q_target=None, v_target=None, joint_f=None, dt=H):
    """Step ``model`` from rest; each step's joint coordinates and velocities (float64)."""
    solver = SolverFeatherPGS(model, pgs_iterations=4)
    state, out, control = model.state(), model.state(), model.control()
    for name, value in (("joint_target_q", q_target), ("joint_target_qd", v_target), ("joint_f", joint_f)):
        if value is not None:
            getattr(control, name).assign(np.asarray(value, dtype=np.float32))
    qs, vs = [], []
    for _ in range(steps):
        solver.step(state, out, control, None, dt)
        state, out = out, state
        qs.append(state.joint_q.numpy().astype(np.float64))
        vs.append(state.joint_qd.numpy().astype(np.float64))
    return np.asarray(qs), np.asarray(vs)


class TestMotorLaw(unittest.TestCase):
    """The drive's motor law on the held tree's route (CPU and CUDA)."""

    devices = ("cpu", "cuda:0") if wp.is_cuda_available() else ("cpu",)

    def test_a_driven_joint_saturates_at_its_speed(self):
        """A drive asking for 20 times its no-load speed rides its torque-speed line up to ``V`` and never past it."""
        inertia, kd, E, V = 0.01, 10.0, 1.0, 5.0
        expected = _leaf_steps(inertia, kd, E, V, 100.0, 120)
        for device in self.devices:
            with self.subTest(device=device):
                _, v = _roll(_hinges([inertia], ke=0.0, kd=kd, E=E, V=V, device=device), 120, v_target=[100.0])
                np.testing.assert_allclose(v[:, 0], expected, rtol=1.0e-5, atol=1.0e-5)
                self.assertLessEqual(v[:, 0].max(), V * (1.0 + 1.0e-6))
                self.assertGreater(v[-1, 0], 0.999 * V)

    def test_a_back_driven_joint_passes_its_speed(self):
        """An external torque ``3 E`` takes the joint past ``V``, a motor being no cap. With its drive asking for more
        speed still, the motor brakes past ``V`` all the same (its back-EMF), with its full ``E`` past ``2 V``; with its
        drive asking for rest, it brakes with ``E`` from the start. Either way the joint ends gaining ``2 E / I``."""
        inertia, kd, E, V = 0.01, 10.0, 1.0, 5.0
        for v_target in (1000.0, 0.0):
            expected = _leaf_steps(inertia, kd, E, V, v_target, 240, joint_f=3.0 * E)
            for device in self.devices:
                with self.subTest(v_target=v_target, device=device):
                    model = _hinges([inertia], ke=0.0, kd=kd, E=E, V=V, device=device)
                    _, v = _roll(model, 240, v_target=[v_target], joint_f=[3.0 * E])
                    np.testing.assert_allclose(v[:, 0], expected, rtol=1.0e-5, atol=1.0e-4)
                    self.assertGreater(v[-1, 0], 30.0 * V)
                    # Past 2 V the brake takes E from the external torque, whatever the drive asks.
                    gain = (v[-1, 0] - v[-41, 0]) * inertia / (40 * H)
                    self.assertAlmostEqual(gain, 2.0 * E, delta=1.0e-3)

    def test_a_coupled_chain_follows_the_motor_step(self):
        """Two collinear hinges pushed far past their no-load speed, and three on position targets with a stiff PD:
        each step from the solver's own state against the coupled exact step. The two driven the same way stay under
        ``V``; of the three, the middle one is back-driven past it by its neighbours' reactions, as the exact step
        is."""
        cases = (
            ((0.01, 0.01), 0.0, 10.0, 1.0, 5.0, (0.0, 0.0), (100.0, 100.0), 120, True),
            ((0.02, 0.01, 0.005), 400.0, 20.0, 2.0, 3.0, (3.0, -2.0, 4.0), (0.0, 0.0, 0.0), 240, False),
        )
        tolerance = 1.0e-5
        for inertias, ke, kd, E, V, q_target, v_target, steps, under in cases:
            n = len(inertias)
            mass = np.array([[sum(inertias[max(i, j) :]) for j in range(n)] for i in range(n)])
            for device in self.devices:
                with self.subTest(joints=n, device=device):
                    model = _hinges(inertias, ke=ke, kd=kd, E=E, V=V, device=device)
                    qs, vs = _roll(model, steps, q_target=q_target, v_target=v_target)
                    q, v, errors = np.zeros(n), np.zeros(n), []
                    for k in range(steps):
                        x = _chain_step(mass, q, v, ke, kd, np.asarray(q_target), np.asarray(v_target), E, V)
                        errors.append(np.abs(vs[k] - x).max() / max(1.0, np.abs(x).max()))
                        q, v = qs[k], vs[k]
                    # The decision reads its piece from a one-coordinate trial: a step at a kink of a coupled chain
                    # may take the neighbouring piece (continuous there), every other step is the exact one.
                    self.assertLess(np.percentile(errors, 95), tolerance)
                    self.assertLess(errors[-1], tolerance)
                    self.assertLess(max(errors), 0.05)
                    if under:
                        self.assertLessEqual(np.abs(vs).max(), V * (1.0 + 1.0e-5))

    def test_a_floating_articulation_keeps_its_momentum(self):
        """A free base with three driven hinges, no gravity, drives on their torque-speed lines: the drives' forces are
        internal, so linear and angular momentum drift only as the integration's error does, first order in ``h``
        (a speed cap on one body would remove momentum at a rate that does not vanish with ``h``)."""
        drifts = []
        for dt in (1.0 / 240.0, 1.0 / 960.0):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            base = builder.add_link(
                mass=2.0,
                inertia=wp.mat33(np.diag([0.02, 0.03, 0.04])),
                xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
            )
            joints, parent = [builder.add_joint_free(child=base)], base
            for k, axis in enumerate(((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0))):
                link = builder.add_link(
                    mass=0.5,
                    inertia=wp.mat33(np.diag([0.002, 0.003, 0.001])),
                    xform=wp.transform(wp.vec3(0.2 * (k + 1), 0.0, 1.0), wp.quat_identity()),
                )
                joints.append(
                    builder.add_joint_revolute(
                        parent=parent,
                        child=link,
                        axis=axis,
                        parent_xform=wp.transform(wp.vec3(0.2 if k else 0.1, 0.0, 0.0), wp.quat_identity()),
                        child_xform=wp.transform(wp.vec3(-0.1 if k == 0 else 0.0, 0.0, 0.0), wp.quat_identity()),
                        target_kd=5.0,
                        effort_limit=0.5,
                        velocity_limit=4.0,
                        armature=0.0,
                        limit_lower=-1.0e6,
                        limit_upper=1.0e6,
                    )
                )
                parent = link
            builder.add_articulation(joints)
            model = builder.finalize(device="cpu")
            solver = SolverFeatherPGS(model, pgs_iterations=4)
            state, out, control = model.state(), model.state(), model.control()
            newton.eval_fk(model, model.joint_q, model.joint_qd, state)
            control.joint_target_qd.assign(np.asarray([0.0] * 6 + [50.0, -50.0, 50.0], dtype=np.float32))
            start = _momentum(model, state)
            drift = np.zeros(2)
            for _ in range(round(1.0 / dt)):
                solver.step(state, out, control, None, dt)
                state, out = out, state
                drift = np.maximum(
                    drift, [np.abs(m - s).max() for m, s in zip(_momentum(model, state), start, strict=True)]
                )
            self.assertGreater(np.abs(state.joint_qd.numpy()[6:]).max(), 3.0)
            drifts.append(drift)
        # Linear and angular: a quarter of the step, a quarter of the drift (first order), and small at h = 1/240.
        np.testing.assert_array_less(drifts[1], 0.3 * drifts[0])
        np.testing.assert_array_less(drifts[0], 0.02)


def _momentum(model, state):
    """The articulation's linear momentum and its angular momentum about the origin (float64)."""
    pose = state.body_q.numpy().astype(np.float64)
    twist = state.body_qd.numpy().astype(np.float64)
    mass = model.body_mass.numpy().astype(np.float64)
    com = model.body_com.numpy().astype(np.float64)
    inertia = model.body_inertia.numpy().astype(np.float64)
    linear, angular = np.zeros(3), np.zeros(3)
    for body in range(len(mass)):
        rotation = _rotation(pose[body, 3:])
        centre = pose[body, :3] + rotation @ com[body]
        velocity, spin = twist[body, :3], twist[body, 3:]
        linear += mass[body] * velocity
        angular += mass[body] * np.cross(centre, velocity) + rotation @ inertia[body] @ rotation.T @ spin
    return linear, angular


def _rotation(quaternion):
    x, y, z, w = quaternion
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


@unittest.skipUnless(wp.is_cuda_available(), "the one-coordinate class runs on CUDA")
class TestMotorLawOnCoordinateTrees(unittest.TestCase):
    """The same law on the one-coordinate trees' route, whose predict decides each leaf's drive itself."""

    def test_a_leaf_saturates_at_its_speed_and_is_back_driven_past_it(self):
        model = branch.build_branch_model("cuda:0", worlds=2, arm=True)
        model.gravity.fill_(wp.vec3(0.0, 0.0, 0.0))
        size = branch.branch_size(model)
        dofs = np.asarray(model.joint_qd_start.numpy()[model.articulation_start.numpy()[0] + 1]) + np.arange(size)
        child = model.joint_child.numpy()[model.articulation_start.numpy()[0] + 1 + np.arange(size)]
        inertia = (model.body_mass.numpy()[child] + model.joint_armature.numpy()[dofs]).astype(np.float64)
        # Each leaf's no-load speed is what its stall torque gives it in 60 steps; its PD saturates far below its target.
        E = 1.0
        V = E * 60.0 * H / inertia
        kd = 10.0 * E / V
        for name, value in (
            ("target_ke", 0.0),
            ("target_kd", kd),
            ("effort_limit", E),
            ("velocity_limit", V),
            ("damping", 0.0),
        ):
            array = getattr(model, f"joint_{name}")
            values = array.numpy()
            values[dofs] = value
            array.assign(values)
        model.joint_stiffness.zero_()
        model.joint_limit_lower.fill_(-1.0e3)
        model.joint_limit_upper.fill_(1.0e3)
        solver = branch.recipe_solver(model, coordinate=True, update_mass_matrix_interval=1)
        self.assertTrue(solver.get_path_selection()["routes"]["held_tree"]["coordinate_trees"]["enabled"])
        control = model.control()
        targets = np.zeros(model.joint_dof_count, dtype=np.float32)
        targets[dofs[0]] = 20.0 * V[0]
        control.joint_target_qd.assign(targets)
        forces = np.zeros(model.joint_dof_count, dtype=np.float32)
        forces[dofs[1]] = 3.0 * E
        control.joint_f.assign(forces)
        state, out = model.state(), model.state()
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        history = []
        for _ in range(120):
            solver.step(state, out, control, None, H)
            state, out = out, state
            history.append(state.joint_qd.numpy()[dofs[:2]].astype(np.float64))
        history = np.asarray(history)
        # Each step from the solver's own state against the exact step. The leaf route's first step from rest is short
        # of its exact step for a saturated drive whatever the bound (the effort limit alone too, about 8% here), a
        # defect of that route's first step apart from this law; every later step is the law's.
        for k in range(1, 120):
            driven = _leaf_step(inertia[0], kd[0], E, V[0], 20.0 * V[0], history[k - 1, 0])
            pushed = _leaf_step(inertia[1], kd[1], E, V[1], 0.0, history[k - 1, 1], joint_f=3.0 * E)
            self.assertAlmostEqual(history[k, 0], driven, delta=1.0e-4 * V[0])
            self.assertAlmostEqual(history[k, 1], pushed, delta=1.0e-4 * V[1])
        self.assertLessEqual(history[:, 0].max(), V[0] * (1.0 + 1.0e-5))
        self.assertGreater(history[-1, 1], 3.0 * V[1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
