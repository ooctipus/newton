# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The held tree on the models beside the catalogue: trees of one size with different topologies, articulations
without coordinates, models without any, particles, debug rows and the joint kinds roots and descendants may hold."""

import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS

_INERTIA = wp.mat33(np.eye(3, dtype=np.float32) * 0.01)


def _devices() -> tuple:
    return ("cpu", "cuda:0") if wp.is_cuda_available() else ("cpu",)


def _chain(builder, links, x):
    """A chain of revolute links hanging at ``x`` in a plane."""
    parent, joints = -1, []
    for k in range(links):
        link = builder.add_link(mass=1.0, inertia=_INERTIA, xform=wp.transform(wp.vec3(x, 0.0, 1.0 - 0.2 * k)))
        anchor = wp.vec3(x, 0.0, 1.0) if parent < 0 else wp.vec3(0.0, 0.0, -0.1)
        joints.append(
            builder.add_joint_revolute(
                parent,
                link,
                axis=newton.Axis.Y,
                parent_xform=wp.transform(anchor),
                child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.1)),
            )
        )
        parent = link
    builder.add_articulation(joints)


def _star(builder, leaves, x):
    """A revolute root at ``x`` with revolute leaves: as many coordinates as a chain of ``leaves + 1`` links."""
    root = builder.add_link(mass=1.0, inertia=_INERTIA, xform=wp.transform(wp.vec3(x, 0.0, 1.0)))
    joints = [builder.add_joint_revolute(-1, root, axis=newton.Axis.Y, parent_xform=wp.transform(wp.vec3(x, 0.0, 1.0)))]
    for k in range(leaves):
        leaf = builder.add_link(mass=0.5, inertia=_INERTIA, xform=wp.transform(wp.vec3(x + 0.2 * (k + 1), 0.0, 1.0)))
        joints.append(
            builder.add_joint_revolute(
                root, leaf, axis=newton.Axis.X, parent_xform=wp.transform(wp.vec3(0.2 * (k + 1), 0.0, 0.0))
            )
        )
    builder.add_articulation(joints)


def _fall(builder, device, *, steps=60, **options):
    model = builder.finalize(device=device)
    solver = SolverFeatherPGS(model, pgs_iterations=8, **options)
    state, out, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    for _ in range(steps):
        state.clear_forces()
        solver.step(state, out, control, None, 1.0 / 240.0)
        state, out = out, state
    return solver, model, state


class TestHeldModels(unittest.TestCase):
    def test_trees_of_one_size_group_by_topology(self):
        """Hold a chain and a star of three coordinates each in groups of their own, over one size group's factors."""
        for device in _devices():
            with self.subTest(device=device):
                builder = newton.ModelBuilder()
                _chain(builder, 3, 0.0)
                _star(builder, 2, 2.0)
                _chain(builder, 3, 4.0)
                solver, _, state = _fall(builder, device)
                self.assertEqual(solver.forest.size_groups, [3])
                group_to_art = solver.forest.group_to_art_host[3]
                self.assertEqual(sorted(group_to_art.tolist()), [0, 1, 2])
                self.assertEqual(solver.forest.art_group_idx.numpy()[group_to_art].tolist(), [0, 1, 2])
                held = solver.factor.held
                self.assertIsNotNone(held, solver.factor.held_reason)
                self.assertEqual(sorted(group.group_count for group in held.groups), [1, 2])
                alone = newton.ModelBuilder()
                _star(alone, 2, 2.0)
                _, _, expected = _fall(alone, device)
                np.testing.assert_allclose(state.joint_q.numpy()[3:6], expected.joint_q.numpy(), atol=1e-5)
                np.testing.assert_allclose(state.body_q.numpy()[3:6], expected.body_q.numpy(), atol=1e-5)

    def test_articulations_without_coordinates_follow_their_frames(self):
        """Publish an articulation without coordinates in both banks, whatever moves its joint frame."""
        for device in _devices():
            with self.subTest(device=device):
                builder = newton.ModelBuilder()
                _chain(builder, 2, 0.0)
                body = builder.add_link(mass=1.0, inertia=_INERTIA, xform=wp.transform(wp.vec3(2.0, 0.0, 1.0)))
                builder.add_articulation(
                    [builder.add_joint_fixed(-1, body, parent_xform=wp.transform(wp.vec3(2.0, 0.0, 1.0)))]
                )
                solver, model, state = _fall(builder, device)
                self.assertIsNotNone(solver.factor.held, solver.factor.held_reason)
                np.testing.assert_allclose(state.body_q.numpy()[body, :3], [2.0, 0.0, 1.0], atol=1e-6)
                frames = model.joint_X_p.numpy()
                frames[2, 0] = 3.0
                model.joint_X_p.assign(frames)
                solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)
                out = model.state()
                solver.step(state, out, model.control(), None, 1.0 / 240.0)
                np.testing.assert_allclose(state.body_q.numpy()[body, :3], [3.0, 0.0, 1.0], atol=1e-6)
                np.testing.assert_allclose(out.body_q.numpy()[body, :3], [3.0, 0.0, 1.0], atol=1e-6)

    def test_a_forest_holds_a_tree_without_coordinates(self):
        """Hold a world's tree of no coordinates as a zero family of its forest."""
        for device in _devices():
            with self.subTest(device=device):
                scene = newton.ModelBuilder()
                _chain(scene, 2, 0.0)
                body = scene.add_link(mass=1.0, inertia=_INERTIA, xform=wp.transform(wp.vec3(2.0, 0.0, 1.0)))
                scene.add_articulation(
                    [scene.add_joint_fixed(-1, body, parent_xform=wp.transform(wp.vec3(2.0, 0.0, 1.0)))]
                )
                builder = newton.ModelBuilder()
                builder.replicate(scene, 2)
                with mock.patch.dict(SolverFeatherPGS._route_pins, {"storage": "dense", "grouping": "forest"}):
                    solver, _, state = _fall(builder, device, steps=4)
                self.assertEqual(len(solver.factor.held.groups), 1)
                self.assertEqual(solver.factor.held.groups[0].layout.root_families, (2, 0))
                np.testing.assert_allclose(
                    state.body_q.numpy()[[body, 2 * body + 1], :3], [[2.0, 0.0, 1.0]] * 2, atol=1e-6
                )

    def test_a_model_without_coordinates_steps(self):
        """Step a model of fixed articulations and a joint outside them: one group of no coordinates, every body in
        place."""
        for device in _devices():
            with self.subTest(device=device):
                builder = newton.ModelBuilder()
                fixed = builder.add_link(mass=1.0, inertia=_INERTIA, xform=wp.transform(wp.vec3(0.0, 0.0, 1.0)))
                builder.add_articulation(
                    [builder.add_joint_fixed(-1, fixed, parent_xform=wp.transform(wp.vec3(0.0, 0.0, 1.0)))]
                )
                outside = builder.add_link(mass=1.0, inertia=_INERTIA, xform=wp.transform(wp.vec3(1.0, 0.0, 1.0)))
                builder.add_joint_fixed(-1, outside, parent_xform=wp.transform(wp.vec3(1.0, 0.0, 1.0)))
                solver, _, state = _fall(builder, device, steps=4)
                self.assertIsNotNone(solver.factor.held, solver.factor.held_reason)
                self.assertEqual([group.layout.dofs for group in solver.factor.held.groups], [0])
                np.testing.assert_allclose(state.body_q.numpy()[fixed, :3], [0.0, 0.0, 1.0], atol=1e-6)

    def test_particles_fall_beside_the_tree(self):
        """Integrate particles beside the held tree, which keeps its own trajectory."""
        for device in _devices():
            with self.subTest(device=device):
                builder = newton.ModelBuilder()
                _chain(builder, 2, 0.0)
                builder.add_particle(pos=(0.0, 2.0, 3.0), vel=(0.0, 0.0, 0.0), mass=1.0)
                solver, _, state = _fall(builder, device)
                self.assertIsNotNone(solver.factor.held, solver.factor.held_reason)
                self.assertLess(state.particle_q.numpy()[0, 2], 3.0 - 0.1)
                alone = newton.ModelBuilder()
                _chain(alone, 2, 0.0)
                _, _, expected = _fall(alone, device)
                np.testing.assert_allclose(state.body_q.numpy(), expected.body_q.numpy(), atol=1e-6)

    def test_debug_rows_keep_the_route(self):
        """Keep the held tree and the trajectory under ``pgs_debug``."""
        for device in _devices():
            with self.subTest(device=device):
                arms = [newton.ModelBuilder(), newton.ModelBuilder()]
                for builder in arms:
                    _chain(builder, 3, 0.0)
                solver, _, debug = _fall(arms[0], device, pgs_debug=True)
                self.assertIsNotNone(solver.factor.held, solver.factor.held_reason)
                _, _, plain = _fall(arms[1], device)
                np.testing.assert_array_equal(debug.joint_q.numpy(), plain.joint_q.numpy())

    def test_free_and_distance_joints_in_a_tree(self):
        """Hold a distance root as a free root, and a free joint below a body, on the shared joint laws."""
        for device in _devices():
            for name in ("distance root", "free descendant"):
                with self.subTest(device=device, joint=name):
                    builder = newton.ModelBuilder(gravity=0.0)
                    body = builder.add_link(mass=1.0, inertia=_INERTIA, xform=wp.transform(wp.vec3(0.0, 0.0, 1.0)))
                    tip = builder.add_link(mass=1.0, inertia=_INERTIA, xform=wp.transform(wp.vec3(0.3, 0.0, 1.0)))
                    if name == "distance root":
                        root = builder.add_joint_distance(-1, body, min_distance=-1.0, max_distance=-1.0)
                        tail = builder.add_joint_revolute(
                            body, tip, axis=newton.Axis.Y, parent_xform=wp.transform(wp.vec3(0.3, 0.0, 0.0))
                        )
                    else:
                        root = builder.add_joint_free(body)
                        tail = builder.add_joint_free(tip, parent=body)
                    builder.add_articulation([root, tail])
                    solver, model, state = _fall(builder, device, steps=4)
                    self.assertIsNotNone(solver.factor.held, solver.factor.held_reason)
                    # Nothing acts on the tree: every body stays where its coordinates put it.
                    rest = model.state()
                    newton.eval_fk(model, rest.joint_q, rest.joint_qd, rest)
                    np.testing.assert_allclose(state.body_q.numpy(), rest.body_q.numpy(), atol=1e-5)

    def test_a_second_root_in_one_articulation_is_its_own_tree(self):
        """The forest splits an articulation of two free roots into two trees, so the second root's body coasts
        where the articulation's one origin would carry it away (a free body at 0.5 m/s once ended at x = 5.85 m)."""
        builder = newton.ModelBuilder(gravity=0.0)
        first = builder.add_link(mass=1.0, inertia=_INERTIA, xform=wp.transform(wp.vec3(0.0, 0.0, 1.0)))
        second = builder.add_link(mass=1.0, inertia=_INERTIA, xform=wp.transform(wp.vec3(1.0, 0.0, 1.0)))
        joints = [builder.add_joint_free(first), builder.add_joint_free(second)]
        builder.add_articulation(joints)
        builder.joint_qd[6] = 0.5
        model = builder.finalize(device="cpu")
        solver = SolverFeatherPGS(model)
        self.assertIsNotNone(solver.factor.held)
        self.assertEqual(solver.model.articulation_count, 2)
        state_0, state_1 = model.state(), model.state()
        for _ in range(240):
            solver.step(state_0, state_1, model.control(), None, 1.0 / 240.0)
            state_0, state_1 = state_1, state_0
        body_q = state_0.body_q.numpy()
        np.testing.assert_allclose(body_q[first, :3], (0.0, 0.0, 1.0), atol=1.0e-6)
        np.testing.assert_allclose(body_q[second, :3], (1.5, 0.0, 1.0), atol=1.0e-5)


if __name__ == "__main__":
    unittest.main()
