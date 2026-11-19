# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for FeatherPGS sleeping (the rest law, ``feather_pgs/sleep.py``, and the pipeline's sleep filter)."""

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS

_H = 0.02
_DT, _SUBSTEPS = 0.01, 8


def _run(model, steps, bind=True, on_step=None):
    """Step ``model`` at 100 Hz collision and 8 substeps; return the body poses after each step, the asleep flag of
    each body's tree after each step (``None`` unbound) and the solver."""
    pipeline = newton.CollisionPipeline(model, broad_phase="sap", include_static_kinematic_pairs=False)
    solver = SolverFeatherPGS(model, pgs_iterations=8)
    contacts = pipeline.contacts()
    sleep = solver.collision_sleep_filter() if bind else None
    if sleep is not None:
        pipeline.configure_sleep_filter(*sleep)
    s0, s1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    shape_body = model.shape_body.numpy()
    first_shape = [int(np.flatnonzero(shape_body == b)[0]) for b in range(model.body_count)]
    poses, asleep = [], []
    for step in range(steps):
        if on_step is not None:
            on_step(step, solver, s0)
        pipeline.collide(s0, contacts, dt=_DT / _SUBSTEPS)
        for _ in range(_SUBSTEPS):
            solver.step(s0, s1, control, contacts, _DT / _SUBSTEPS)
            s0, s1 = s1, s0
        poses.append(s0.body_q.numpy().copy())
        if sleep is not None:
            index = sleep[0].numpy()[first_shape]
            state = sleep[1].numpy()
            asleep.append(
                np.array([index[b, 1] >= 0 and state[index[b, 0], index[b, 1]] >= 0 for b in range(model.body_count)])
            )
    return np.array(poses), (np.array(asleep) if sleep is not None else None), solver


def _boxes(striker_height=None):
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    for i, x in enumerate((0.0, 0.2, 0.4)):
        body = builder.add_body(xform=wp.transform(wp.vec3(x, 0.0, _H + 0.005 * (i + 1)), wp.quat_identity()))
        builder.add_shape_box(body, hx=_H, hy=_H, hz=_H)
    if striker_height is not None:
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, striker_height), wp.quat_identity()))
        builder.add_shape_box(body, hx=_H, hy=_H, hz=_H)
    return builder.finalize(device="cpu")


class TestFeatherPGSSleep(unittest.TestCase):
    def test_resting_boxes_sleep_and_hold_their_pose(self):
        poses, asleep, _ = _run(_boxes(), 60)
        self.assertTrue(asleep[-1].all(), "three boxes at rest on the ground fall asleep")
        held = [k for k in range(len(poses) - 1) if asleep[k].all() and asleep[k + 1].all()]
        self.assertTrue(held)
        for k in held:
            np.testing.assert_array_equal(poses[k], poses[k + 1])
        self.assertGreater(poses[-1][:, 2].min(), _H - 1.0e-4)

    def test_a_falling_box_wakes_the_box_it_strikes_before_the_impact(self):
        poses, asleep, _ = _run(_boxes(striker_height=0.5), 60)
        # Box 0 rests and sleeps long before the striker arrives (about 0.3 s), wakes before the impact and never
        # sinks into the ground; all four rest asleep at the end.
        self.assertTrue(asleep[20, 0])
        self.assertFalse(asleep[25:35, 0].all())
        self.assertGreater(poses[:, 0, 2].min(), _H - 2.0e-4)
        self.assertGreater(poses[-1, 3, 2], 3.0 * _H - 2.0e-4)
        self.assertTrue(asleep[-1].all())

    def test_a_box_creeping_down_a_ramp_slides_as_if_awake(self):
        # 26.7 degrees against a friction angle of 26.57: the box accelerates at 0.026 m/s^2 and stays under 3 mm/s
        # for over 0.1 s, so only the sticking test tells it from a box at rest.
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.mu = 0.5
        tilt = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), math.radians(26.7))
        builder.add_shape_box(
            -1, xform=wp.transform(wp.quat_rotate(tilt, wp.vec3(0.0, 0.0, -0.05)), tilt), hx=1.0, hy=0.5, hz=0.05
        )
        body = builder.add_body(xform=wp.transform(wp.quat_rotate(tilt, wp.vec3(0.0, 0.0, _H)), tilt))
        builder.add_shape_box(body, hx=_H, hy=_H, hz=_H)
        model = builder.finalize(device="cpu")
        bound, asleep, _ = _run(model, 30)
        unbound, _, _ = _run(model, 30, bind=False)
        self.assertFalse(asleep.any())
        np.testing.assert_array_equal(bound, unbound)

    def test_a_driven_tree_never_sleeps(self):
        builder = newton.ModelBuilder()
        link = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()))
        builder.add_shape_box(link, hx=0.1, hy=0.02, hz=0.02)
        joint = builder.add_joint_revolute(-1, link, axis=newton.Axis.Y, target_ke=100.0, target_kd=10.0)
        builder.add_articulation([joint])
        free = builder.add_body(xform=wp.transform(wp.vec3(2.0, 0.0, _H), wp.quat_identity()))
        builder.add_shape_box(free, hx=_H, hy=_H, hz=_H)
        builder.add_ground_plane()
        model = builder.finalize(device="cpu")
        _, asleep, _ = _run(model, 40)
        self.assertFalse(asleep[:, 0].any(), "the driven link never sleeps")
        self.assertTrue(asleep[-1, 1], "the box beside it does")

    def test_a_box_a_driven_body_holds_never_sleeps(self):
        # A box resting on a platform that a drive holds still is loaded by it, not at rest (as a part a gripper
        # holds): its count restarts while the platform touches it. A box beside it on the ground sleeps.
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        platform = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()))
        builder.add_shape_box(platform, hx=0.1, hy=0.1, hz=0.01)
        joint = builder.add_joint_prismatic(
            -1,
            platform,
            axis=newton.Axis.Z,
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()),
            target_ke=1.0e4,
            target_kd=100.0,
        )
        builder.add_articulation([joint])
        held = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.11 + _H), wp.quat_identity()))
        builder.add_shape_box(held, hx=_H, hy=_H, hz=_H)
        free = builder.add_body(xform=wp.transform(wp.vec3(1.0, 0.0, _H), wp.quat_identity()))
        builder.add_shape_box(free, hx=_H, hy=_H, hz=_H)
        _, asleep, _ = _run(builder.finalize(device="cpu"), 40)
        self.assertFalse(asleep[:, held].any())
        self.assertTrue(asleep[-1, free])

    def test_a_reset_wakes_its_worlds(self):
        def reset_at(step, solver, state):
            if step == 30:
                solver.reset(state, wp.array([True], dtype=wp.bool, device="cpu"))

        _, asleep, _ = _run(_boxes(), 31, on_step=reset_at)
        self.assertTrue(asleep[29].all())
        self.assertFalse(asleep[30].any())

    def test_a_gravity_change_wakes_its_worlds_alone(self):
        # Two worlds of resting boxes: the model's gravity is written for both, changed in world 1 only. World 0's
        # boxes stay asleep (nothing changed for them, and nothing is read again on the host); world 1's wake.
        world = newton.ModelBuilder()
        world.add_ground_plane()
        for x in (0.0, 0.2):
            body = world.add_body(xform=wp.transform(wp.vec3(x, 0.0, _H + 0.005), wp.quat_identity()))
            world.add_shape_box(body, hx=_H, hy=_H, hz=_H)
        builder = newton.ModelBuilder()
        builder.replicate(world, 2)
        model = builder.finalize(device="cpu")

        def lower_gravity(step, solver, state):
            if step == 40:
                gravity = model.gravity.numpy()
                gravity[1, 2] *= 0.5
                model.gravity.assign(gravity)
                solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)

        _, asleep, _ = _run(model, 41, on_step=lower_gravity)
        self.assertTrue(asleep[39].all())
        self.assertTrue(asleep[40, :2].all(), "world 0's gravity is as it was: its boxes sleep on")
        self.assertFalse(asleep[40, 2:].any(), "world 1's gravity changed: its boxes wake")

    def test_a_fast_striker_wakes_the_box_at_the_pass_that_predicts_the_impact(self):
        # Speculative contacts over the 10 ms window: a box dropped 0.5 m reaches the resting box at 3 m/s, 30 mm a
        # window against 10 mm of contact gaps, so only the swept boxes of the pass before the impact overlap. That
        # pass must wake the box it makes contacts with: a held box takes the impact with its velocity zeroed after
        # every sweep, and the striker sinks into it (2 mm before the island joined over the swept boxes).
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.gap = 0.005
        builder.add_ground_plane()
        box = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, _H + 0.001), wp.quat_identity()))
        builder.add_shape_box(box, hx=_H, hy=_H, hz=_H)
        striker = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 3.0 * _H + 0.5), wp.quat_identity()))
        builder.add_shape_box(striker, hx=_H, hy=_H, hz=_H)
        model = builder.finalize(device="cpu")
        pipeline = newton.CollisionPipeline(
            model, broad_phase="sap", include_static_kinematic_pairs=False, speculative_contact_gap_max=0.05
        )
        solver = SolverFeatherPGS(model, pgs_iterations=8)
        contacts = pipeline.contacts()
        shape_sleep_index, tree_asleep = solver.collision_sleep_filter()
        pipeline.configure_sleep_filter(shape_sleep_index, tree_asleep)
        s0, s1, control = model.state(), model.state(), model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
        box_tree = shape_sleep_index.numpy()[int(np.flatnonzero(model.shape_body.numpy() == box)[0])]
        slept, deepest = False, 0.0
        for _ in range(40):
            pipeline.collide(s0, contacts, dt=_DT)
            asleep = tree_asleep.numpy()[box_tree[0], box_tree[1]] >= 0
            slept = slept or asleep
            n = int(contacts.rigid_contact_count.numpy()[0])
            bodies = (
                model.shape_body.numpy()[contacts.rigid_contact_shape0.numpy()[:n]],
                model.shape_body.numpy()[contacts.rigid_contact_shape1.numpy()[:n]],
            )
            struck = bool(
                np.any((bodies[0] == striker) & (bodies[1] == box) | (bodies[0] == box) & (bodies[1] == striker))
            )
            self.assertFalse(struck and asleep, "a pass makes contacts with the box only after waking it")
            for _ in range(_SUBSTEPS):
                solver.step(s0, s1, control, contacts, _DT / _SUBSTEPS)
                s0, s1 = s1, s0
                q = s0.body_q.numpy()
                deepest = max(deepest, (q[box, 2] + _H) - (q[striker, 2] - _H))
        self.assertTrue(slept, "the box rests and sleeps before the striker arrives")
        self.assertLess(deepest, 1.0e-4)

    def test_no_tree_that_can_rest_builds_no_sleep_state(self):
        builder = newton.ModelBuilder()
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()))
        builder.add_shape_box(body, hx=_H, hy=_H, hz=_H)
        builder.body_flags[body] = int(newton.BodyFlags.KINEMATIC)
        solver = SolverFeatherPGS(builder.finalize(device="cpu"))
        self.assertIsNone(solver.collision_sleep_filter())


if __name__ == "__main__":
    unittest.main(verbosity=2)
