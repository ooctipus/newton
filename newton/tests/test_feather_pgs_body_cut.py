# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests of the body cut: the solver's own model is a pure forest, and a step crosses to it and back unchanged."""

import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import body_cut
from newton._src.solvers.feather_pgs.body_cut import BodyCut
from newton.tests import feather_pgs_scenes as scenes


def _revolute_chain(builder, *, parent=-1, x=0.0):
    """A body on a revolute joint about ``y``, returning ``(body, joint)``."""
    body = builder.add_link(xform=wp.transform(wp.vec3(x, 0.0, 0.5)))
    builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.1)
    joint = builder.add_joint_revolute(
        parent=parent,
        child=body,
        axis=newton.Axis.Y,
        parent_xform=wp.transform(wp.vec3(x, 0.0, 0.6)),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.1)),
    )
    return body, joint


def _two_roots(*, foreign: bool = False):
    """Two world roots closed by a loop joint, optionally behind a foreign articulation's free joint."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    left, j_left = _revolute_chain(builder)
    right, j_right = _revolute_chain(builder, x=0.2)
    builder.add_articulation([j_left, j_right])
    if foreign:
        body = builder.add_link(xform=wp.transform(wp.vec3(2.0, 0.0, 0.5)))
        builder.add_shape_sphere(body, radius=0.05)
        builder.add_articulation([builder.add_joint_free(child=body)])
    loop = builder.add_joint_ball(
        parent=left,
        child=right,
        parent_xform=wp.transform(wp.vec3(0.1, 0.0, 0.0)),
        child_xform=wp.transform(wp.vec3(-0.1, 0.0, 0.0)),
    )
    return builder, loop


def _closed(joint_kind: str):
    """A body on a revolute root and a free body, joined by a loop joint of ``joint_kind`` between them."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    first, j_first = _revolute_chain(builder)
    builder.add_articulation([j_first])
    second = builder.add_body(xform=wp.transform(wp.vec3(0.2, 0.0, 0.5)))
    builder.add_shape_box(second, hx=0.05, hy=0.05, hz=0.1)
    frames = {
        "parent_xform": wp.transform(wp.vec3(0.1, 0.0, 0.0)),
        "child_xform": wp.transform(wp.vec3(-0.1, 0.0, 0.0)),
    }
    add = {
        "ball": lambda: builder.add_joint_ball(first, second, **frames),
        "revolute": lambda: builder.add_joint_revolute(first, second, axis=newton.Axis.Y, **frames),
        "prismatic": lambda: builder.add_joint_prismatic(first, second, axis=newton.Axis.X, **frames),
        "fixed": lambda: builder.add_joint_fixed(first, second, **frames),
        "d6": lambda: builder.add_joint_d6(
            first,
            second,
            linear_axes=[newton.ModelBuilder.JointDofConfig(axis=newton.Axis.X)],
            angular_axes=[newton.ModelBuilder.JointDofConfig(axis=newton.Axis.Y)],
            **frames,
        ),
    }
    return builder, add[joint_kind]()


_LOOP_MOTION = {
    "ball": ([0.0, np.sin(0.2), 0.0, np.cos(0.2)], [0.0, 0.9, 0.0]),
    "revolute": ([0.4], [0.9]),
    "prismatic": ([0.05], [0.3]),
    "fixed": ([], []),
    "d6": ([0.03, 0.2], [0.3, 0.9]),
}
"""The loop joint's coordinates and motions each joint type is seated back to."""


class TestPureForest(unittest.TestCase):
    def test_a_pure_forest_is_its_own_model(self):
        """Nothing is gathered or launched for a model without loops: the step's objects pass through."""
        builder = newton.ModelBuilder()
        body, joint = _revolute_chain(builder)
        builder.add_articulation([joint])
        model = builder.finalize(device="cpu")
        cut = BodyCut.of(model)
        self.assertIs(cut.model, model)
        self.assertTrue(cut.pure)
        state_in, state_out, control = model.state(), model.state(), model.control()
        entered = cut.enter(state_in, state_out, control, None)
        self.assertIs(entered[0], state_in)
        self.assertIs(entered[1], state_out)
        self.assertIs(entered[2], control)
        cut.leave(state_in, state_out)
        self.assertEqual(body, 0)

    def test_scenes_without_loops_are_pure(self):
        for scene in (scenes.mimic_pair(), scenes.compliant_stack(), scenes.compliant_pinch()):
            with self.subTest(scene=scene.name):
                model = scene.build(wp.get_device("cpu"))
                self.assertIs(BodyCut.of(model).model, model)


class TestForestLayout(unittest.TestCase):
    def setUp(self):
        self.device = wp.get_device("cpu")

    def test_roots_split_and_the_loop_joint_joins_its_parents_tree(self):
        builder, loop = _two_roots()
        model = builder.finalize(device="cpu")
        cut = BodyCut.of(model)
        forest = cut.model
        self.assertEqual(model.body_count, 2)
        self.assertEqual(forest.body_count, 3)
        self.assertEqual(forest.joint_count, 3)
        self.assertEqual(forest.articulation_count, 2)
        # Tree 0 holds the left root and the loop joint; tree 1 holds the right root.
        self.assertEqual(forest.joint_child.numpy().tolist(), [0, 2, 1])
        self.assertEqual(forest.joint_parent.numpy().tolist(), [-1, 0, -1])
        self.assertEqual(forest.joint_ancestor.numpy().tolist(), [-1, 0, -1])
        self.assertEqual(forest.joint_articulation.numpy().tolist(), [0, 0, 1])
        self.assertEqual(forest.articulation_start.numpy().tolist(), [0, 2, 3])
        self.assertEqual(forest.articulation_end.numpy().tolist(), [2, 3])
        # The ball's four coordinates and three motions follow its tree.
        self.assertEqual(forest.joint_q_start.numpy().tolist(), [0, 1, 5, 6])
        self.assertEqual(forest.joint_qd_start.numpy().tolist(), [0, 1, 4, 5])
        self.assertEqual(cut.link(loop), 0)
        self.assertEqual((cut.primary.tolist(), cut.shadow.tolist()), ([1], [2]))
        self.assertEqual((cut.tree_primary.tolist(), cut.tree_shadow.tolist()), ([1], [0]))
        self.assertEqual(cut.component.tolist(), [0, 0])

    def test_a_foreign_joint_between_a_tree_and_its_loop_joint(self):
        """The add_usd order: the free object's coordinates keep their values, wherever the layout puts them."""
        builder, _ = _two_roots(foreign=True)
        model = builder.finalize(device="cpu")
        forest = BodyCut.of(model).model
        self.assertEqual(forest.articulation_count, 3)
        self.assertEqual(forest.joint_articulation.numpy().tolist(), [0, 0, 1, 2])
        self.assertEqual(forest.joint_type.numpy().tolist()[1], int(newton.JointType.BALL))
        self.assertEqual(forest.joint_type.numpy().tolist()[3], int(newton.JointType.FREE))

    def test_a_world_parent_roots_its_own_tree_beside_its_child(self):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        load = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.8)))
        builder.add_shape_box(load, hx=0.04, hy=0.04, hz=0.04)
        builder.add_joint_ball(parent=-1, child=load, parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.8)))
        model = builder.finalize(device="cpu")
        cut = BodyCut.of(model)
        self.assertEqual(cut.model.articulation_count, 2)
        self.assertEqual(cut.model.joint_parent.numpy().tolist(), [-1, -1])
        self.assertEqual(cut.model.joint_ancestor.numpy().tolist(), [-1, -1])
        self.assertEqual(cut.component.tolist(), [0, 0])

    def test_a_kinematic_parent_takes_the_shadow_in_its_tree(self):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        carrier = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0)), is_kinematic=True)
        builder.add_shape_box(carrier, hx=0.03, hy=0.03, hz=0.03)
        load = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.8)))
        builder.add_shape_box(load, hx=0.04, hy=0.04, hz=0.04)
        builder.add_joint_ball(parent=carrier, child=load)
        model = builder.finalize(device="cpu")
        forest = BodyCut.of(model).model
        self.assertEqual(forest.articulation_count, 2)
        # The carrier's tree: its free joint, then the ball to the shadow, ahead of the load's free joint.
        self.assertEqual(forest.joint_child.numpy().tolist(), [carrier, 2, load])
        self.assertEqual(forest.joint_articulation.numpy().tolist(), [0, 0, 1])
        self.assertEqual(forest.joint_ancestor.numpy().tolist(), [-1, 0, -1])

    def test_a_parent_no_tree_holds_stands_the_joint_on_the_world(self):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        anchor = builder.add_link(xform=wp.transform(wp.vec3(1.0, 0.0, 0.0)))
        body, joint = _revolute_chain(builder)
        builder.add_articulation([joint])
        builder.add_joint_ball(parent=anchor, child=body, parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5)))
        model = builder.finalize(device="cpu")
        forest = BodyCut.of(model).model
        self.assertEqual(forest.joint_parent.numpy().tolist(), [-1, -1])
        np.testing.assert_allclose(forest.joint_X_p.numpy()[1, :3], [1.0, 0.0, 0.5], atol=1e-6)

    def test_a_joint_outside_every_tree_stays_outside(self):
        builder, _ = _two_roots()
        standalone = builder.add_link()
        builder.add_shape_box(standalone, hx=0.05, hy=0.05, hz=0.1)
        builder.add_joint_revolute(parent=-1, child=standalone, axis=newton.Axis.Y)
        model = builder.finalize(device="cpu")
        forest = BodyCut.of(model).model
        self.assertEqual(forest.joint_articulation.numpy().tolist(), [0, 0, 1, -1])
        self.assertEqual(forest.articulation_start.numpy().tolist(), [0, 2, 4])
        self.assertEqual(forest.articulation_end.numpy().tolist(), [2, 3])

    def test_mimic_joints_follow_the_layout(self):
        builder, _ = _two_roots()
        builder.add_constraint_mimic(joint0=1, joint1=0, coef0=0.0, coef1=1.0)
        model = builder.finalize(device="cpu")
        forest = BodyCut.of(model).model
        self.assertEqual(forest.constraint_mimic_joint0.numpy().tolist(), [2])
        self.assertEqual(forest.constraint_mimic_joint1.numpy().tolist(), [0])

    def test_shapes_stay_on_the_primary(self):
        builder, _ = _two_roots()
        model = builder.finalize(device="cpu")
        cut = BodyCut.of(model)
        self.assertLess(int(cut.model.shape_body.numpy().max()), model.body_count)

    def test_the_catalogue_loops_become_pure_forests(self):
        for scene in (scenes.four_bar(), scenes.loaded_four_bar(), scenes.five_bar_leg(), scenes.stewart_platform()):
            with self.subTest(scene=scene.name):
                model = scene.build(self.device)
                forest = BodyCut.of(model).model
                child = forest.joint_child.numpy()
                self.assertEqual(np.unique(child).size, child.size, "a body with two parents")
                self.assertEqual(np.flatnonzero(forest.joint_articulation.numpy() < 0).size, 0)
                ancestor = forest.joint_ancestor.numpy()
                articulation = forest.joint_articulation.numpy()
                roots = np.bincount(articulation[ancestor < 0], minlength=forest.articulation_count)
                self.assertTrue(np.all(roots == 1), "a tree of several roots")
                self.assertTrue(np.all(ancestor < np.arange(ancestor.size)), "not parent-first")
                held = ancestor >= 0
                self.assertTrue(
                    np.all(articulation[ancestor[held]] == articulation[held]), "a tree crossing articulations"
                )

    def test_the_stewart_splits_one_tree_per_leg(self):
        model = scenes.stewart_platform().build(self.device)
        cut = BodyCut.of(model)
        self.assertEqual(cut.model.articulation_count, 6)
        self.assertEqual(cut.shadow.size, 5)
        self.assertEqual(np.unique(cut.component).size, 1)


class TestMassSplit(unittest.TestCase):
    def test_pieces_sum_back_to_the_body(self):
        model = scenes.stewart_platform().build(wp.get_device("cpu"))
        cut = BodyCut.of(model)
        forest = cut.model
        bodies = model.body_count
        mass, inertia = forest.body_mass.numpy(), forest.body_inertia.numpy()
        com = forest.body_com.numpy()
        source_mass, source_inertia = model.body_mass.numpy(), model.body_inertia.numpy()
        platform = int(cut.primary[0])
        self.assertTrue(np.all(cut.primary == platform))
        total = mass[platform] + mass[bodies:].sum()
        self.assertAlmostEqual(float(total), float(source_mass[platform]), delta=1e-5 * float(source_mass[platform]))
        np.testing.assert_allclose(mass[bodies:], source_mass[platform] / 6.0, rtol=1e-6)
        pieces = inertia[platform] + inertia[bodies:].sum(axis=0)
        np.testing.assert_allclose(pieces, source_inertia[platform], rtol=1e-5, atol=1e-9)
        np.testing.assert_array_equal(com[bodies:], np.repeat(com[platform][None], 5, axis=0))
        others = np.delete(np.arange(bodies), platform)
        np.testing.assert_array_equal(mass[others], source_mass[others])
        self.assertIsNone(forest.body_inv_mass)
        self.assertIsNone(forest.body_inv_inertia)

    def test_the_shares_follow_the_body(self):
        """Domain randomization: a notified mass reaches every piece."""
        builder, _ = _two_roots()
        model = builder.finalize(device="cpu")
        cut = BodyCut.of(model)
        mass = model.body_mass.numpy()
        mass[1] = 3.0
        model.body_mass.assign(mass)
        cut.refresh()
        np.testing.assert_allclose(cut.model.body_mass.numpy()[[1, 2]], [1.5, 1.5])

    def test_a_disabled_link_returns_its_share(self):
        builder, loop = _two_roots()
        model = builder.finalize(device="cpu")
        cut = BodyCut.of(model)
        whole = float(model.body_mass.numpy()[1])
        cut.set_enabled(loop, False)
        cut.refresh()
        forest = cut.model
        self.assertAlmostEqual(float(forest.body_mass.numpy()[1]), whole, places=6)
        self.assertEqual(float(forest.body_mass.numpy()[2]), 0.0)
        self.assertEqual(int(forest.body_flags.numpy()[2]), int(newton.BodyFlags.KINEMATIC))
        cut.set_enabled(loop, True)
        cut.refresh()
        self.assertAlmostEqual(float(forest.body_mass.numpy()[2]), 0.5 * whole, places=6)
        self.assertEqual(int(forest.body_flags.numpy()[2]), int(newton.BodyFlags.DYNAMIC))

    def test_properties_follow_the_layout(self):
        builder, _ = _two_roots()
        model = builder.finalize(device="cpu")
        cut = BodyCut.of(model)
        armature = model.joint_armature.numpy()
        armature[:] = np.arange(armature.size)
        model.joint_armature.assign(armature)
        cut.refresh()
        self.assertEqual(cut.model.joint_armature.numpy().tolist(), [0.0, 2.0, 3.0, 4.0, 1.0])


class TestLiveArrays(unittest.TestCase):
    """A step reads the model's gains, limits, frames and inertias live: a write needs no notification."""

    def _entered(self, model, cut):
        valid = wp.ones(cut.model.articulation_count, dtype=wp.int32, device="cpu")
        cut.enter(model.state(), model.state(), None, valid)

    def test_a_write_reaches_the_forest_at_the_next_step(self):
        builder, _ = _two_roots()
        model = builder.finalize(device="cpu")
        cut = BodyCut.of(model)
        cases = (
            ("joint_target_ke", 7.0),
            ("joint_target_kd", 8.0),
            ("joint_limit_lower", -9.0),
            ("joint_limit_upper", 9.5),
            ("joint_effort_limit", 11.0),
            ("joint_velocity_limit", 12.0),
            ("joint_damping", 13.0),
            ("joint_spring_stiffness", 14.0),
            ("joint_spring_ref", 15.0),
        )
        dofs = cut._orders[newton.Model.AttributeFrequency.JOINT_DOF]
        for name, value in cases:
            values = getattr(model, name).numpy()
            values[:] = np.arange(values.size) + value
            getattr(model, name).assign(values)
        self._entered(model, cut)
        for name, _ in cases:
            with self.subTest(name=name):
                np.testing.assert_array_equal(getattr(cut.model, name).numpy(), getattr(model, name).numpy()[dofs])

    def test_a_frame_reaches_the_forest_at_the_next_step(self):
        builder, loop = _two_roots()
        model = builder.finalize(device="cpu")
        cut = BodyCut.of(model)
        frames = model.joint_X_p.numpy()
        frames[loop, :3] = (0.3, 0.2, 0.1)
        model.joint_X_p.assign(frames)
        self._entered(model, cut)
        np.testing.assert_allclose(cut.model.joint_X_p.numpy()[int(cut._position[loop]), :3], (0.3, 0.2, 0.1))

    def test_a_mass_reaches_every_piece_at_the_next_step(self):
        builder, _ = _two_roots()
        model = builder.finalize(device="cpu")
        cut = BodyCut.of(model)
        mass, inertia = model.body_mass.numpy(), model.body_inertia.numpy()
        mass[1] = 3.0
        inertia[1] = 2.0 * np.eye(3)
        model.body_mass.assign(mass)
        model.body_inertia.assign(inertia)
        self._entered(model, cut)
        np.testing.assert_allclose(cut.model.body_mass.numpy()[[1, 2]], [1.5, 1.5])
        np.testing.assert_allclose(cut.model.body_inertia.numpy()[2], np.eye(3))

    def test_arrays_the_layout_leaves_alone_are_the_same_arrays(self):
        builder, _ = _two_roots()
        builder.add_constraint_mimic(joint0=1, joint1=0, coef0=0.0, coef1=1.0)
        model = builder.finalize(device="cpu")
        forest = BodyCut.of(model).model
        for name in ("gravity", "shape_body", "constraint_mimic_coef0", "constraint_mimic_coef1"):
            self.assertIs(getattr(forest, name), getattr(model, name), name)

    def test_a_changed_topology_raises(self):
        builder, _ = _two_roots()
        model = builder.finalize(device="cpu")
        cut = BodyCut.of(model)
        child = model.joint_child.numpy()
        child[0], child[1] = child[1], child[0]
        model.joint_child.assign(child)
        with self.assertRaises(RuntimeError):
            cut.refresh()


class TestBoundary(unittest.TestCase):
    def test_a_step_crosses_unchanged(self):
        """Into the forest and back, every coordinate and body keeps its value."""
        builder, _ = _two_roots(foreign=True)
        model = builder.finalize(device="cpu")
        cut = BodyCut.of(model)
        state_in, state_out = model.state(), model.state()
        q = np.arange(model.joint_coord_count, dtype=np.float32) + 1.0
        qd = np.arange(model.joint_dof_count, dtype=np.float32) + 1.0
        state_in.joint_q.assign(q)
        state_in.joint_qd.assign(qd)
        valid = wp.ones(cut.model.articulation_count, dtype=wp.int32, device="cpu")
        forest_in, forest_out, _ = cut.enter(state_in, state_out, None, valid)
        np.testing.assert_array_equal(
            forest_in.joint_q.numpy(), q[cut._orders[newton.Model.AttributeFrequency.JOINT_COORD]]
        )
        forest_out.joint_q.assign(forest_in.joint_q)
        forest_out.joint_qd.assign(forest_in.joint_qd)
        cut.leave(state_in, state_out)
        np.testing.assert_array_equal(state_out.joint_q.numpy(), q)
        np.testing.assert_array_equal(state_out.joint_qd.numpy(), qd)

    def test_a_state_keeps_its_mirror(self):
        builder, _ = _two_roots()
        model = builder.finalize(device="cpu")
        cut = BodyCut.of(model)
        state_in, state_out = model.state(), model.state()
        valid = wp.ones(cut.model.articulation_count, dtype=wp.int32, device="cpu")
        first = cut.enter(state_in, state_out, None, valid)
        second = cut.enter(state_in, state_out, None, valid)
        self.assertIs(first[0], second[0])
        self.assertIs(first[1], second[1])

    def test_the_control_is_gathered(self):
        builder, _ = _two_roots(foreign=True)
        model = builder.finalize(device="cpu")
        cut = BodyCut.of(model)
        control = model.control()
        force = np.arange(model.joint_dof_count, dtype=np.float32)
        control.joint_f.assign(force)
        valid = wp.ones(cut.model.articulation_count, dtype=wp.int32, device="cpu")
        _, _, mirror = cut.enter(model.state(), model.state(), control, valid)
        dofs = cut._orders[newton.Model.AttributeFrequency.JOINT_DOF]
        np.testing.assert_array_equal(mirror.joint_f.numpy(), force[dofs])

    def test_the_output_leaves_a_joint_outside_every_tree_alone(self):
        builder, _ = _two_roots()
        standalone = builder.add_link()
        builder.add_shape_box(standalone, hx=0.05, hy=0.05, hz=0.1)
        builder.add_joint_revolute(parent=-1, child=standalone, axis=newton.Axis.Y)
        model = builder.finalize(device="cpu")
        cut = BodyCut.of(model)
        state_in, state_out = model.state(), model.state()
        q = state_out.joint_q.numpy()
        q[-1] = 0.25
        state_out.joint_q.assign(q)
        valid = wp.ones(cut.model.articulation_count, dtype=wp.int32, device="cpu")
        cut.enter(state_in, state_out, None, valid)
        cut.leave(state_in, state_out)
        self.assertEqual(float(state_out.joint_q.numpy()[-1]), 0.25)


class TestSeating(unittest.TestCase):
    def _state_at(self, kind: str):
        """The caller's state whose second body stands where the loop joint's ``_LOOP_MOTION`` puts its shadow."""
        builder, _ = _closed(kind)
        model = builder.finalize(device="cpu")
        cut = BodyCut.of(model)
        forest = cut.model
        q_start, qd_start = forest.joint_q_start.numpy(), forest.joint_qd_start.numpy()
        loop = int(cut._position[cut.joints[0]])
        q, qd = _LOOP_MOTION[kind]
        posed = forest.state()
        joint_q, joint_qd = posed.joint_q.numpy(), posed.joint_qd.numpy()
        joint_q[q_start[0]], joint_qd[qd_start[0]] = 0.3, 0.7
        joint_q[q_start[loop] : q_start[loop + 1]] = q
        joint_qd[qd_start[loop] : qd_start[loop + 1]] = qd
        posed.joint_q.assign(joint_q)
        posed.joint_qd.assign(joint_qd)
        newton.eval_fk(forest, posed.joint_q, posed.joint_qd, posed)
        state = model.state()
        bodies = np.append(np.arange(model.body_count - 1), cut.shadow[0])
        state.body_q.assign(posed.body_q.numpy()[bodies])
        state.body_qd.assign(posed.body_qd.numpy()[bodies])
        newton.eval_ik(model, state, state.joint_q, state.joint_qd)
        return model, cut, state, loop

    def test_seating_reads_every_joint_types_coordinates(self):
        """Seating recovers the loop joint's coordinates and motions from the bodies' poses and twists."""
        for kind, (expected_q, expected_qd) in _LOOP_MOTION.items():
            with self.subTest(kind=kind):
                model, cut, state, loop = self._state_at(kind)
                valid = wp.zeros(cut.model.articulation_count, dtype=wp.int32, device="cpu")
                forest_in, _, _ = cut.enter(state, model.state(), None, valid)
                q_start, qd_start = cut.model.joint_q_start.numpy(), cut.model.joint_qd_start.numpy()
                q = forest_in.joint_q.numpy()[q_start[loop] : q_start[loop + 1]]
                qd = forest_in.joint_qd.numpy()[qd_start[loop] : qd_start[loop + 1]]
                if kind == "ball":
                    self.assertGreater(abs(float(np.dot(q, expected_q))), 1.0 - 1.0e-6)
                else:
                    np.testing.assert_allclose(q, expected_q, atol=2.0e-6)
                np.testing.assert_allclose(qd, expected_qd, atol=5.0e-6)

    def test_seating_puts_the_shadow_on_its_primary(self):
        for kind in _LOOP_MOTION:
            with self.subTest(kind=kind):
                model, cut, state, _ = self._state_at(kind)
                valid = wp.zeros(cut.model.articulation_count, dtype=wp.int32, device="cpu")
                forest_in, _, _ = cut.enter(state, model.state(), None, valid)
                forward = cut.model.state()
                newton.eval_fk(cut.model, forest_in.joint_q, forest_in.joint_qd, forward)
                primary, shadow = forward.body_q.numpy()[cut.primary[0]], forward.body_q.numpy()[cut.shadow[0]]
                np.testing.assert_allclose(shadow[:3], primary[:3], atol=2.0e-6)
                self.assertGreater(abs(float(np.dot(shadow[3:], primary[3:]))), 1.0 - 1.0e-6)

    def test_seating_clears_every_tree_of_the_component(self):
        """A cleared bit spreads over the links: seating moves every tree a link joins."""
        builder, _ = _two_roots()
        model = builder.finalize(device="cpu")
        cut = BodyCut.of(model)
        valid = wp.array([1, 0], dtype=wp.int32, device="cpu")
        cut.enter(model.state(), model.state(), None, valid)
        self.assertEqual(valid.numpy().tolist(), [0, 0])

    def test_steady_stepping_seats_nothing(self):
        """A valid cache leaves a loop joint's coordinates as they integrated, past a turn."""
        builder, loop = _closed("revolute")
        model = builder.finalize(device="cpu")
        cut = BodyCut.of(model)
        state_in, state_out = model.state(), model.state()
        q = state_in.joint_q.numpy()
        q[model.joint_q_start.numpy()[loop]] = 7.0
        state_in.joint_q.assign(q)
        valid = wp.ones(cut.model.articulation_count, dtype=wp.int32, device="cpu")
        forest_in, _, _ = cut.enter(state_in, state_out, None, valid)
        self.assertEqual(float(forest_in.joint_q.numpy()[1]), 7.0)


class TestLoopJointApi(unittest.TestCase):
    def test_unknown_joint_is_rejected(self):
        builder, loop = _two_roots()
        cut = BodyCut.of(builder.finalize(device="cpu"))
        with self.assertRaises(ValueError):
            cut.link(loop + 100)
        with self.assertRaises(ValueError):
            cut.link(0)

    def test_anchors_reach_the_forest(self):
        builder, loop = _two_roots()
        model = builder.finalize(device="cpu")
        cut = BodyCut.of(model)
        cut.set_anchors(loop, (0.1, 0.2, 0.3), (0.4, 0.5, 0.6))
        cut.refresh()
        parent, child = cut.anchors(loop)
        np.testing.assert_allclose(parent, [0.1, 0.2, 0.3])
        np.testing.assert_allclose(child, [0.4, 0.5, 0.6])
        np.testing.assert_allclose(cut.model.joint_X_p.numpy()[1, :3], [0.1, 0.2, 0.3])
        np.testing.assert_allclose(cut.model.joint_X_c.numpy()[1, :3], [0.4, 0.5, 0.6])


class TestEveryLiveArrayIsListed(unittest.TestCase):
    def test_a_step_reads_only_gathered_shared_or_topology_arrays(self):
        """Run the live census on cut scenes: an array of the forest's own that a step's kernel takes and the gather
        does not refresh would go stale on a write the caller need not notify."""
        allowed = set(body_cut._LIVE) | set(body_cut._TOPOLOGY) | {"body_world"}
        real = wp.launch
        read: set[str] = set()

        def walk(value, seen, depth=0):
            if isinstance(value, wp.array):
                seen.add(value.ptr)
            elif hasattr(getattr(value, "_cls", None), "vars") and depth < 4:
                for name in value._cls.vars:
                    walk(getattr(value, name, None), seen, depth + 1)

        def census(kernel, dim, inputs=None, outputs=None, **kw):
            seen: set[int] = set()
            for value in (*(inputs or ()), *(outputs or ())):
                walk(value, seen)
            read.update(names[ptr] for ptr in seen if ptr in names)
            return real(kernel, dim, inputs=inputs or [], outputs=outputs or [], **kw)

        for scene, options in (
            (scenes.loaded_four_bar(), {"pgs_warmstart": True}),
            (scenes.five_bar_leg(), {"enable_joint_velocity_limits": True}),
            (scenes.stewart_platform(), {"contact_torsion": True}),
        ):
            with self.subTest(scene=scene.name), wp.ScopedDevice("cpu"):
                model = scene.build(wp.get_device("cpu"))
                solver = newton.solvers.SolverFeatherPGS(model, pgs_iterations=2, dense_max_constraints=64, **options)
                forest = solver.model
                # Only the forest's own arrays: a shared array is the caller's, live by being the same one.
                names = {
                    value.ptr: name
                    for name, value in vars(forest).items()
                    if isinstance(value, wp.array) and value.ptr and getattr(model, name, None) is not value
                }
                pipeline = newton.CollisionPipeline(model, contact_matching=scene.contact_matching or "latest")
                contacts = pipeline.contacts()
                state_0, state_1, control = model.state(), model.state(), model.control()
                newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
                pipeline.collide(state_0, contacts)
                with mock.patch.object(wp, "launch", census):
                    solver.step(state_0, state_1, control, contacts, scene.dt)
        self.assertFalse(read - allowed, "arrays a step reads that no gather refreshes")
        self.assertTrue(read & {"joint_target_ke", "joint_X_p", "body_mass"}, "the census saw nothing")


if __name__ == "__main__":
    unittest.main()
