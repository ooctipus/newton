# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The forest plan's coordinate trees: the vertices of the rows' graph, read from the model's topology alone."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.tree_plan import TreePlan
from newton.solvers import SolverFeatherPGS
from newton.tests import feather_pgs_scenes as scenes

_INERTIA = wp.mat33(np.eye(3, dtype=np.float32) * 0.01)
_AXIS = (0.0, 0.0, 1.0)


def _link(builder) -> int:
    return builder.add_link(mass=1.0, inertia=_INERTIA)


def _plan(model):
    return SolverFeatherPGS(model).forest


def _sizes(plan, world=0) -> list:
    trees = plan.world_trees[plan.world_tree_start[world] : plan.world_tree_start[world + 1]]
    return plan.tree_size[trees].tolist()


class TestCoordinateTrees(unittest.TestCase):
    def test_catalogue(self):
        """Read a star as its keys, a loop on a fixed base as a tree per branch, and a floating hand as one."""
        keyboard = _plan(scenes.keyboard(keys=8, worlds=2).build("cpu"))
        for world in range(2):
            self.assertEqual(sorted(_sizes(keyboard, world), reverse=True), [6] + [1] * 8)
        self.assertEqual(keyboard.diagonal_mass_sizes, frozenset((8,)))
        four_bar = _plan(scenes.four_bar().build("cpu"))
        self.assertEqual(len(_sizes(four_bar)), 2)
        self.assertEqual(min(_sizes(four_bar)), 1)
        hand = _plan(scenes.hand(worlds=2).build("cpu"))
        self.assertEqual(sorted(_sizes(hand, 1), reverse=True), [17, 6])
        for plan in (keyboard, four_bar, hand):
            per_articulation = np.bincount(plan.tree_articulation, plan.tree_size, plan.response_dof_count.size)
            np.testing.assert_array_equal(per_articulation, plan.response_dof_count)

    def test_root_without_coordinates_is_no_node(self):
        """Contract welds at any depth: the base is in no tree, and a welded link is its parent's."""
        builder = newton.ModelBuilder()
        base, arm, weld, tip, side = (_link(builder) for _ in range(5))
        builder.add_articulation(
            [
                builder.add_joint_fixed(-1, base),
                builder.add_joint_revolute(base, arm, axis=_AXIS),
                builder.add_joint_fixed(arm, weld),
                builder.add_joint_revolute(weld, tip, axis=_AXIS),
                builder.add_joint_revolute(base, side, axis=_AXIS),
            ]
        )
        plan = _plan(builder.finalize(device="cpu"))
        self.assertEqual(plan.tree_size.tolist(), [2, 1])
        self.assertEqual(plan.tree_first.tolist(), [0, 2])
        self.assertEqual(plan.tree_articulation.tolist(), [0, 0])
        self.assertEqual(plan.body_tree.tolist(), [-1, 0, 0, 0, 1])

    def test_a_free_root_is_a_coordinate(self):
        """A free body is a one-node tree of six, and a free root holds its whole articulation."""
        builder = newton.ModelBuilder()
        box, trunk, leg = (_link(builder) for _ in range(3))
        builder.add_articulation([builder.add_joint_free(box)])
        builder.add_articulation([builder.add_joint_free(trunk), builder.add_joint_revolute(trunk, leg, axis=_AXIS)])
        plan = _plan(builder.finalize(device="cpu"))
        self.assertEqual(plan.tree_size.tolist(), [6, 7])
        self.assertEqual(plan.body_tree.tolist(), [0, 1, 1])

    def test_only_solved_coordinates_make_a_tree(self):
        """A prescribed free body responds with nothing, so it is no vertex."""
        builder = newton.ModelBuilder()
        carried = builder.add_link(mass=1.0, inertia=_INERTIA, is_kinematic=True)
        free = _link(builder)
        builder.add_articulation([builder.add_joint_free(carried)])
        builder.add_articulation([builder.add_joint_free(free)])
        plan = _plan(builder.finalize(device="cpu"))
        self.assertEqual(plan.prescribed_articulation_host.tolist(), [1, 0])
        self.assertEqual(plan.tree_size.tolist(), [6])
        self.assertEqual(plan.tree_articulation.tolist(), [1])
        self.assertEqual(plan.body_tree.tolist(), [-1, 0])

    def test_a_tree_stops_at_its_articulation(self):
        """Each articulation is its own block of the mass matrix, even hung from another's body: a star on both sides."""
        builder = newton.ModelBuilder()
        upper, lower = _link(builder), _link(builder)
        builder.add_articulation([builder.add_joint_revolute(-1, upper, axis=_AXIS)])
        builder.add_articulation([builder.add_joint_revolute(upper, lower, axis=_AXIS)])
        model = builder.finalize(device="cpu")
        self.assertEqual(model.joint_ancestor.numpy().tolist(), [-1, 0])
        plan = _plan(model)
        self.assertEqual(plan.tree_size.tolist(), [1, 1])
        self.assertEqual(plan.body_tree.tolist(), [0, 1])
        self.assertEqual(plan.diagonal_mass_sizes, frozenset((1,)))

    def test_breadth_first_joints_interleave_trees(self):
        """Keep ``(tree_first, tree_size)`` honest: it counts a tree's coordinates, it does not bound them."""
        builder = newton.ModelBuilder()
        base, left, right, left_tip, right_tip = (_link(builder) for _ in range(5))
        builder.add_articulation(
            [
                builder.add_joint_fixed(-1, base),
                builder.add_joint_revolute(base, left, axis=_AXIS),
                builder.add_joint_revolute(base, right, axis=_AXIS),
                builder.add_joint_revolute(left, left_tip, axis=_AXIS),
                builder.add_joint_revolute(right, right_tip, axis=_AXIS),
            ]
        )
        plan = _plan(builder.finalize(device="cpu"))
        self.assertEqual(plan.tree_first.tolist(), [0, 1])
        self.assertEqual(plan.tree_size.tolist(), [2, 2])
        self.assertEqual(plan.body_tree.tolist(), [-1, 0, 1, 0, 1])

    def test_a_model_without_joints_has_no_trees(self):
        builder = newton.ModelBuilder()
        _link(builder)
        plan = _plan(builder.finalize(device="cpu"))
        self.assertEqual(plan.tree_size.size, 0)
        self.assertEqual(plan.world_trees.size, 0)
        self.assertEqual(plan.body_tree.tolist(), [-1])

    def test_a_star_is_a_diagonal_factor_pattern(self):
        """The star test agrees with the fill-free pattern on random welded trees of one root joint."""
        rng = np.random.default_rng(7)
        found = {True: 0, False: 0}
        for _ in range(60):
            builder = newton.ModelBuilder()
            count = int(rng.integers(1, 9))
            bodies = [_link(builder) for _ in range(count)]
            parents = [-1 if k == 0 else int(rng.integers(0, k)) for k in range(count)]
            welded = rng.random(count) < 0.6
            joints = []
            for k in range(count):
                parent = -1 if parents[k] < 0 else bodies[parents[k]]
                add = builder.add_joint_fixed if welded[k] else builder.add_joint_revolute
                joints.append(add(parent, bodies[k]) if welded[k] else add(parent, bodies[k], axis=_AXIS))
            builder.add_articulation(joints)
            if welded.all():
                continue
            plan = _plan(builder.finalize(device="cpu"))
            size = int((~welded).sum())
            star = TreePlan.build(parents, (~welded).astype(np.int32)).nonzero_count == size
            self.assertEqual(size in plan.diagonal_mass_sizes, star, parents)
            self.assertEqual(bool(np.all(plan.tree_size == 1)), star, parents)
            found[bool(star)] += 1
        self.assertTrue(all(found.values()), found)

    def test_record_reports_the_trees(self):
        """Report the graph's vertices by size."""
        solver = SolverFeatherPGS(scenes.keyboard(keys=8, worlds=2).build("cpu"))
        self.assertEqual(solver.get_path_selection()["routes"]["dynamics"]["tree_sizes"], {"1": 16, "6": 2})


if __name__ == "__main__":
    unittest.main(verbosity=2)
