# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The forest primitive's contract: the model's topology read once into a pure forest of coordinate trees.

The body cut turns every loop into trees and a step crosses into the forest and back unchanged; the plan's trees are
the rows' graph's vertices; tree ownership and response supports follow the joints; the subtree intervals and the
fill-free factor pattern follow the trees; the host index kit the plan is built with matches element-by-element
references. Each section names the interface it reads.
"""

import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import body_cut, host_index
from newton._src.solvers.feather_pgs.body_cut import BodyCut
from newton._src.solvers.feather_pgs.body_mapping import build_body_maps
from newton._src.solvers.feather_pgs.forest_plan import RowLaw
from newton._src.solvers.feather_pgs.tree_plan import TreePlan
from newton.solvers import SolverFeatherPGS
from newton.tests import feather_pgs_scenes as scenes

# ---------------------------------------------------------------------------------------------------------------------
# host_index
#
# Tests of FeatherPGS's host index toolkit against element-by-element references.
# ---------------------------------------------------------------------------------------------------------------------


class TestFeatherPGSHostIndex(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(7)

    def test_offsets_bound_each_segment(self):
        counts = self.rng.integers(0, 5, 40)
        bounds = host_index.offsets(counts)
        self.assertEqual(bounds.dtype, np.int64)
        self.assertEqual(bounds.tolist(), [0, *np.cumsum(counts).tolist()])
        self.assertEqual(host_index.offsets(np.array([3, 4], dtype=np.int32), dtype=np.int32).dtype, np.int32)
        self.assertEqual(host_index.offsets([]).tolist(), [0])

    def test_expand_lists_each_element_with_its_rank(self):
        counts = self.rng.integers(0, 5, 40)
        segment, rank = host_index.expand(counts)
        expected = [(s, k) for s, count in enumerate(counts) for k in range(count)]
        self.assertEqual(list(zip(segment.tolist(), rank.tolist(), strict=True)), expected)

    def test_segmented_rank_restarts_at_each_segment(self):
        segment = np.sort(self.rng.integers(0, 8, 60))
        weight = self.rng.integers(0, 4, 60)
        for weights in (None, weight, weight > 1):
            values = np.ones(60, dtype=np.int64) if weights is None else np.asarray(weights, dtype=np.int64)
            expected = [int(values[:i][segment[:i] == segment[i]].sum()) for i in range(60)]
            self.assertEqual(host_index.segmented_rank(segment, weights).tolist(), expected)
        self.assertEqual(host_index.segmented_rank(np.zeros(0, dtype=np.int32)).size, 0)

    def test_group_by_keeps_each_keys_elements_in_order(self):
        keys = self.rng.integers(0, 6, 50)
        bounds, order = host_index.group_by(keys, 8)
        self.assertEqual(bounds.size, 9)
        for key in range(8):
            self.assertEqual(order[bounds[key] : bounds[key + 1]].tolist(), np.flatnonzero(keys == key).tolist())

    def test_ordered_unique_keeps_first_occurrences(self):
        values = self.rng.integers(0, 9, 50)
        expected = list(dict.fromkeys(values.tolist()))
        self.assertEqual(host_index.ordered_unique(values).tolist(), expected)

    def test_lookup_finds_the_holding_segment(self):
        bounds = np.array([2, 2, 5, 9, 9, 12])
        items = np.arange(-1, 14)
        expected = [next((s for s in range(5) if bounds[s] <= i < bounds[s + 1]), -1) for i in items]
        self.assertEqual(host_index.lookup(bounds, items).tolist(), expected)


# ---------------------------------------------------------------------------------------------------------------------
# forest_plan
#
# The forest plan's coordinate trees: the vertices of the rows' graph, read from the model's topology alone.
# ---------------------------------------------------------------------------------------------------------------------


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
        # The root cut: the base is a root of its own, each joint below it roots a tree, the weld stays in its tree.
        self.assertEqual(plan.coordinate_forest.parent.tolist(), [-1, -1, 1, 2, -1])
        self.assertEqual(plan.tree_root.tolist(), [1, 4])

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

    def test_the_root_cut_makes_a_fixed_star_one_tree_a_key(self):
        """A fixed-base star is as many one-coordinate trees as it has keys, all of one shape."""
        builder = newton.ModelBuilder()
        base = _link(builder)
        keys = [_link(builder) for _ in range(4)]
        builder.add_articulation(
            [builder.add_joint_fixed(-1, base)] + [builder.add_joint_revolute(base, key, axis=_AXIS) for key in keys]
        )
        plan = _plan(builder.finalize(device="cpu"))
        self.assertEqual(plan.tree_size.tolist(), [1] * 4)
        self.assertEqual(plan.tree_root.tolist(), [1, 2, 3, 4])
        self.assertEqual(plan.coordinate_forest.roots.tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(plan.coordinate_forest.dofs.tolist(), [0, 1, 1, 1, 1])
        self.assertEqual(plan.tree_shape.tolist(), [0] * 4)

    def test_tree_shapes_follow_parents_types_and_axes(self):
        """Equal trees share a shape: a chain differs from a fork of as many joints, a slide from a hinge."""
        builder = newton.ModelBuilder()
        for kind in ("chain", "fork", "chain", "slide"):
            root, middle, tip = (_link(builder) for _ in range(3))
            joint = builder.add_joint_prismatic if kind == "slide" else builder.add_joint_revolute
            builder.add_articulation(
                [
                    builder.add_joint_revolute(-1, root, axis=_AXIS),
                    builder.add_joint_revolute(root, middle, axis=_AXIS),
                    joint(middle if kind != "fork" else root, tip, axis=_AXIS),
                ]
            )
        plan = _plan(builder.finalize(device="cpu"))
        shapes = plan.tree_shape.tolist()
        self.assertEqual(len(shapes), 4)
        self.assertEqual(shapes[0], shapes[2])
        self.assertEqual(len(set(shapes)), 3)

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


class TestWorldClasses(unittest.TestCase):
    """Worlds of equal tree shapes and equal row laws present are one class, numbered by their first world."""

    @staticmethod
    def _worlds(arms):
        """One world an arm of ``links`` revolute links; ``limited`` gives its first joint a position limit; ``touch``
        a colliding box on its tip, above the ground."""
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        for links, limited, touch in arms:
            builder.begin_world()
            parent, joints = -1, []
            for k in range(links):
                body = _link(builder)
                limits = {"limit_lower": -1.0, "limit_upper": 1.0} if limited and k == 0 else {}
                joints.append(builder.add_joint_revolute(parent, body, axis=_AXIS, **limits))
                parent = body
            if touch:
                builder.add_shape_box(parent, hx=0.1, hy=0.1, hz=0.1)
            builder.add_articulation(joints)
            builder.end_world()
        model = builder.finalize(device="cpu")
        model.joint_limit_lower.fill_(-np.inf)
        model.joint_limit_upper.fill_(np.inf)
        model.joint_velocity_limit.fill_(0.0)
        lower, upper = model.joint_limit_lower.numpy(), model.joint_limit_upper.numpy()
        dof = 0
        for links, limited, _ in arms:
            if limited:
                lower[dof], upper[dof] = -1.0, 1.0
            dof += links
        model.joint_limit_lower.assign(lower)
        model.joint_limit_upper.assign(upper)
        return _plan(model)

    def test_classes_key_on_shapes_and_laws(self):
        plan = self._worlds(
            [(2, False, False), (3, False, False), (2, False, False), (2, True, False), (2, False, True)]
        )
        self.assertEqual(plan.world_class.tolist(), [0, 1, 0, 2, 3])
        self.assertEqual(plan.world_laws.tolist(), [0, 0, 0, int(RowLaw.LIMIT), int(RowLaw.CONTACT | RowLaw.SPIN)])
        self.assertEqual([c.worlds.tolist() for c in plan.world_classes], [[0, 2], [1], [3], [4]])
        self.assertEqual(plan.world_classes[2].laws, RowLaw.LIMIT)
        self.assertEqual(plan.world_classes[0].shapes, plan.world_classes[2].shapes)
        self.assertNotEqual(plan.world_classes[0].shapes, plan.world_classes[1].shapes)

    def test_a_class_ignores_its_trees_order(self):
        """A world's key holds its trees' shapes ascending, so the catalogue's replicated worlds are one class each."""
        for scene in (scenes.keyboard(keys=8, worlds=3), scenes.hand(worlds=2)):
            plan = _plan(scene.build("cpu"))
            self.assertEqual(len(plan.world_classes), 1)
            (only,) = plan.world_classes
            self.assertEqual(only.worlds.tolist(), list(range(plan.world_count)))
            sizes = sorted(plan.tree_size[plan.tree_shape == shape][0] for shape in only.shapes)
            self.assertEqual(sizes, sorted(_sizes(plan)))

    def test_contacts_are_the_admitted_pairs(self):
        """A world touches through the shape pairs the collision filter admits, and holds a footprint where a body
        pair can touch at two points: two shape pairs, or one pair of shapes not limited to a single point."""

        def laws(*shapes, ground=True):
            builder = newton.ModelBuilder()
            if ground:
                builder.add_ground_plane()
            body = _link(builder)
            builder.add_articulation([builder.add_joint_revolute(-1, body, axis=_AXIS)])
            for add in shapes:
                add(builder, body)
            return RowLaw(int(_plan(builder.finalize(device="cpu")).world_laws[0])) & (RowLaw.CONTACT | RowLaw.SPIN)

        def sphere(x):
            return lambda b, body: b.add_shape_sphere(body, radius=0.1, xform=wp.transform((x, 0.0, 0.0)))

        def box(b, body):
            b.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)

        self.assertEqual(laws(box, ground=False), RowLaw(0))
        self.assertEqual(laws(sphere(0.0)), RowLaw.CONTACT)
        self.assertEqual(laws(sphere(0.0), sphere(0.3)), RowLaw.CONTACT | RowLaw.SPIN)
        self.assertEqual(laws(box), RowLaw.CONTACT | RowLaw.SPIN)

    def test_a_link_is_a_law_of_its_world(self):
        plan = _plan(scenes.four_bar().build("cpu"))
        self.assertTrue(plan.world_laws[0] & RowLaw.LINK)


# ---------------------------------------------------------------------------------------------------------------------
# body_maps
#
# Check tree ownership and response supports built during solver setup.
# ---------------------------------------------------------------------------------------------------------------------


def maps(bodies, child, ancestor, qd, starts, ends, response_start, response_count):
    child, qd, starts, ends, response_start, response_count = (
        np.asarray(values, dtype=np.int32) for values in (child, qd, starts, ends, response_start, response_count)
    )
    forest = TreePlan.build(np.asarray(ancestor, dtype=np.int32))
    return build_body_maps(bodies, child, forest, qd, starts, ends, response_start, response_count)


class TestBodyMaps(unittest.TestCase):
    def test_loop_tail_fixed_root_and_unowned_body(self):
        actual = maps(
            6, [0, 1, 2, 1, 3, 4], [-1, 0, 1, 2, -1, 4], [0, 0, 1, 3, 3, 38, 39], [0, 4, 6], [3, 6], [0, 3], [3, 36]
        )
        expected = (
            [0, 1, 2, 4, 5, -1],
            [0, 0, 0, 1, 1, -1],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 7, 0xFFFFFFFF, 0xFFFFFFFF, 0],
            [-1, 0, -1, -1, -1, -1],
        )
        for observed, reference in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(observed, reference)

    def test_response_window_and_high_singleton(self):
        actual = maps(3, [0, 1, 2], [-1, 0, 0], [0, 6, 7, 46], [0, 3], [3], [6], [40])
        np.testing.assert_array_equal(actual[2], [0, 1, 1])
        np.testing.assert_array_equal(actual[3], [0, 1, 0xFFFFFFFE])
        np.testing.assert_array_equal(actual[4], [-1, 6, -1])
        actual = maps(2, [0, 1], [-1, -1], [0, 39, 40], [0, 2], [2], [0], [40])
        self.assertEqual(actual[2][1], 1)
        self.assertEqual(actual[3][1], 0)
        self.assertEqual(actual[4][1], 39)

    def test_unsorted_acyclic_ancestry(self):
        actual = maps(3, [0, 1, 2], [2, -1, 1], [0, 1, 2, 3], [0, 3], [3], [0], [3])
        np.testing.assert_array_equal(actual[3], [7, 2, 6])
        np.testing.assert_array_equal(actual[4], [-1, 1, -1])

    def test_cyclic_ancestry_is_no_forest(self):
        """A joint ancestry with a cycle anywhere, reachable from an owned body or not, is no joint forest."""
        for ancestor in ([0, -1], [1, 0]):
            with self.subTest(ancestor=ancestor), self.assertRaisesRegex(ValueError, "cycle|other nodes"):
                TreePlan.build(np.asarray(ancestor, dtype=np.int32))

    def test_last_tree_owner_and_empty_articulation(self):
        actual = maps(1, [0, 0], [-1, 0], [0, 1, 2], [0, 0, 2], [0, 2], [0, 0], [0, 2])
        self.assertEqual(actual[0][0], 1)
        self.assertEqual(actual[1][0], 1)
        self.assertEqual(actual[3][0], 3)
        actual = maps(2, [], [], [0], [0, 0], [0], [0], [0])
        np.testing.assert_array_equal(actual[4], [-1, -1])


# ---------------------------------------------------------------------------------------------------------------------
# subtree_reduce
#
# The prefix-scan subtree reduction of the tree finish kernel matches the depth-level loop.
# ---------------------------------------------------------------------------------------------------------------------


class TestDepthFirstIntervals(unittest.TestCase):
    def test_breadth_first_input_is_reordered(self):
        """Legs added breadth-first still get one contiguous interval per subtree."""
        plan = TreePlan.build([-1, 0, 0, 1, 2, 3])
        self.assertEqual(plan.dfs_order.tolist(), [0, 1, 3, 5, 2, 4])
        self.assertEqual(plan.dfs_end.tolist(), [6, 4, 4, 4, 6, 6])

    def test_forest_roots_are_separate_intervals(self):
        plan = TreePlan.build([-1, 0, -1, 2, 2])
        self.assertEqual(plan.dfs_order.tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(plan.dfs_end.tolist(), [2, 2, 5, 4, 5])

    def test_cycle_is_rejected(self):
        with self.assertRaises(ValueError):
            TreePlan.build([-1, 2, 1])


def _interleaved_legs(device, worlds=4):
    """A free trunk whose two three-joint legs are added breadth-first: parent-first but not depth-first."""
    scene = newton.ModelBuilder()
    cfg = scene.default_shape_cfg.copy()
    cfg.mu = 0.6
    stance = wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity())
    trunk = scene.add_link(xform=stance, mass=4.0, inertia=wp.mat33(np.eye(3) * 0.1), lock_inertia=True)
    scene.add_shape_box(trunk, hx=0.2, hy=0.1, hz=0.05, cfg=cfg)
    joints = [scene.add_joint_free(child=trunk, parent_xform=stance)]
    parents = [trunk, trunk]
    for segment in range(3):
        for leg in range(2):
            body = scene.add_link(mass=0.5, inertia=wp.mat33(np.eye(3) * 0.01), lock_inertia=True)
            origin = wp.vec3(0.15 * (1 if leg else -1), 0.0, 0.0) if segment == 0 else wp.vec3(0.0, 0.0, -0.15)
            joints.append(
                scene.add_joint_revolute(
                    parent=parents[leg],
                    child=body,
                    axis=newton.Axis.Y,
                    parent_xform=wp.transform(origin, wp.quat_identity()),
                    target_ke=30.0,
                    target_kd=2.0,
                    limit_lower=-1.5,
                    limit_upper=1.5,
                )
            )
            if segment == 2:
                scene.add_shape_sphere(body, radius=0.03, cfg=cfg)
            parents[leg] = body
    scene.add_articulation(joints)
    # A free box keeps the world on the dense route (the mixed owner's class).
    box = scene.add_body(xform=wp.transform(wp.vec3(0.6, 0.0, 0.06), wp.quat_identity()), mass=0.5)
    scene.add_shape_box(box, hx=0.05, hy=0.05, hz=0.05, cfg=cfg)
    builder = newton.ModelBuilder()
    builder.replicate(scene, worlds, spacing=(3.0, 3.0, 0.0))
    builder.add_ground_plane()
    model = builder.finalize(device=device)
    model.rigid_contact_max = 512
    return model


# ---------------------------------------------------------------------------------------------------------------------
# sparse_mass_matrix
# ---------------------------------------------------------------------------------------------------------------------


def _skew(v):
    return np.array(((0.0, -v[2], v[1]), (v[2], 0.0, -v[0]), (-v[1], v[0], 0.0)))


def _random_inertia(rng):
    """A random rigid body's inertia about the origin: its inertia record and its dense spatial matrix."""
    mass = rng.uniform(0.5, 2.0)
    com = rng.normal(scale=0.3, size=3)
    q = rng.normal(size=(3, 3))
    rotational = q @ q.T + 0.1 * np.eye(3) + mass * (com @ com * np.eye(3) - np.outer(com, com))
    moment = mass * com
    dense = np.block([[mass * np.eye(3), -_skew(moment)], [_skew(moment), rotational]])
    return np.concatenate(([mass], moment, rotational.reshape(-1))), dense


def _endpoint_support(plan, joint_a, joint_b=-1):
    """Sparse coordinates supporting either endpoint joint; -1 denotes the world."""
    mask = 0
    for joint in (joint_a, joint_b):
        if joint != -1:
            mask |= int(plan.joint_ancestor_mask[joint])
    return np.asarray([dof for dof in range(plan.dof_count) if mask & (1 << dof)], dtype=np.int32)


def _branched_humanoid_topology():
    """Return 43 DOFs with a free base, legs, spine, arms, and branching fingers."""
    parents, counts = [-1], [6]

    def chain(parent, length):
        for _ in range(length):
            parents.append(parent)
            counts.append(1)
            parent = len(parents) - 1
        return parent

    for _ in range(2):
        chain(0, 6)
    spine = chain(0, 3)
    for _ in range(2):
        hand = chain(spine, 7)
        for _ in range(4):
            chain(hand, 1)
    return parents, counts


class TestSparseMassMatrix(unittest.TestCase):
    def test_row_support_catalogue(self):
        """Preserve ordered endpoint unions and full multi-DOF limit support."""
        for parents, counts in (([-1, 0, 0, 1], [3, 2, 1, 0]), ([-1], [64])):
            tree = TreePlan.build(parents, counts)
            plan = tree.factor
            body_masks = np.concatenate((np.zeros(1, dtype=np.uint64), plan.joint_ancestor_mask))
            support = tree.support_classes(body_masks, plan.dof_count)
            self.assertIsNotNone(support)
            self.assertEqual(support.body_class[0], 0)
            for first, first_mask in enumerate(body_masks):
                for second, second_mask in enumerate(body_masks):
                    index = support.pair_class[support.body_class[first], support.body_class[second]]
                    mask = int(first_mask | second_mask)
                    expected = [dof for dof in range(plan.dof_count) if mask & (1 << dof)]
                    expected += [-1] * (plan.dof_count - len(expected))
                    np.testing.assert_array_equal(support.class_dof[index], expected)
            for dof, mask in enumerate(plan.ancestor_mask):
                expected = [i for i in range(plan.dof_count) if int(mask) & (1 << i)]
                expected += [-1] * (plan.dof_count - len(expected))
                np.testing.assert_array_equal(support.class_dof[support.limit_class[dof]], expected)
        tree = TreePlan.build([-1, 0, 0, 1], [3, 2, 1, 0])
        plan = tree.factor
        support = tree.support_classes(plan.joint_ancestor_mask, 6)
        np.testing.assert_array_equal(support.class_dof[support.limit_class[0]], [3, 4, 5, -1, -1, -1])
        # Structurally zero coefficients inside a joint block must retain their slots.
        np.testing.assert_array_equal(np.flatnonzero(plan.lookup[:, plan.inverse_permutation[0]] >= 0), [5])

    def test_row_support_rejects_oversized_union(self):
        """Decline a catalogue that would truncate a non-colliding body's union."""
        tree = TreePlan.build([-1, 0, 0], [6, 3, 3])
        self.assertIsNone(tree.support_classes(tree.factor.joint_ancestor_mask, 9))

    def test_factor_schedules_follow_dependencies(self):
        """Cover each packed factor entry once and schedule independent branch pivots together."""
        for parents, counts in [([2, -1, 1, 2, -1, 4], [2, 3, 0, 1, 1, 2]), _branched_humanoid_topology()]:
            plan = TreePlan.build(parents, counts).factor
            levels = np.empty(plan.dof_count, dtype=int)
            for level in range(plan.factor_level_count):
                begin, end = plan.factor_level_offsets[level : level + 2]
                levels[plan.factor_level_columns[begin:end]] = level
            np.testing.assert_array_equal(np.sort(plan.factor_level_columns), np.arange(plan.dof_count))
            np.testing.assert_array_equal(np.sort(plan.column_entries), np.arange(plan.nonzero_count))
            np.testing.assert_array_equal(
                np.sort(plan.factor_level_entries), np.flatnonzero(plan.entry_rows != plan.columns)
            )
            for row in range(plan.dof_count):
                begin, end = plan.row_offsets[row : row + 2]
                first = row - (end - begin) + 1
                np.testing.assert_array_equal(plan.columns[begin:end], np.arange(first, row + 1))
                self.assertTrue(np.all(levels[first:row] < levels[row]))
                begin, end = plan.column_offsets[row : row + 2]
                entries = plan.column_entries[begin:end]
                np.testing.assert_array_equal(plan.columns[entries], row)
                self.assertTrue(np.all(np.diff(plan.entry_rows[entries]) > 0))
            if plan.dof_count == 43:
                self.assertLess(plan.factor_level_count, plan.dof_count // 2)

    def test_forest_fixed_multidof_order(self):
        """Derive fill-free support for unordered forests and fixed joints."""
        plan = TreePlan.build([2, -1, 1, 2, -1, 4], [2, 3, 0, 1, 1, 2]).factor
        np.testing.assert_array_equal(np.sort(plan.permutation), np.arange(9))
        np.testing.assert_array_equal(plan.inverse_permutation[plan.permutation], np.arange(9))
        self.assertEqual(_endpoint_support(plan, 2).size, 3)
        self.assertEqual(_endpoint_support(plan, 0, 5).size, 8)
        rng = np.random.default_rng(54)
        lower = np.zeros((9, 9))
        lower[plan.entry_rows, plan.columns] = rng.normal(size=plan.nonzero_count)
        lower[np.diag_indices(9)] = 2.0
        inverse = np.linalg.inv(lower)
        np.testing.assert_allclose(inverse[plan.lookup < 0], 0.0, atol=1.0e-14)
        matrix = lower @ lower.T
        structural = (plan.lookup >= 0) | (plan.lookup.T >= 0)
        np.testing.assert_allclose(matrix[~structural], 0.0, atol=1.0e-14)

    def test_invalid_topology(self):
        """Reject cycles, invalid parents, counts, and unsupported mask widths."""
        for parents, counts in [
            ([1, 0], [1, 1]),
            ([2], [1]),
            ([-2], [1]),
            ([-1], [-1]),
            ([-1], [1.5]),
            ([-1], []),
            ([], []),
        ]:
            with self.subTest(parents=parents, counts=counts), self.assertRaises(ValueError):
                _ = TreePlan.build(parents, counts).factor

    def test_mask_bit_63_and_fixed_root(self):
        """Preserve the highest mask bit and fixed-root endpoint support."""
        plan = TreePlan.build([-1, 0], [0, 64]).factor
        self.assertEqual(int(plan.joint_ancestor_mask[0]), 0)
        self.assertEqual(int(plan.joint_ancestor_mask[1]), (1 << 64) - 1)
        self.assertEqual(int(plan.ancestor_mask[0]), (1 << 64) - 1)


# ---------------------------------------------------------------------------------------------------------------------
# body_cut
#
# Tests of the body cut: the solver's own model is a pure forest, and a step crosses to it and back unchanged.
# ---------------------------------------------------------------------------------------------------------------------


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
        """Nothing is gathered or launched for a model without loops: the step's objects pass through, since its stage
        list holds no enter and no leave."""
        builder = newton.ModelBuilder()
        body, joint = _revolute_chain(builder)
        builder.add_articulation([joint])
        model = builder.finalize(device="cpu")
        cut = BodyCut.of(model)
        self.assertIs(cut.model, model)
        self.assertTrue(cut.pure)
        self.assertEqual(body, 0)
        state = model.state()
        self.assertIs(cut.mirror(state), state)
        order = [stage.name for stage in newton.solvers.SolverFeatherPGS(model)._plan.stages]
        self.assertNotIn("enter", order)
        self.assertNotIn("leave", order)

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
    unittest.main(verbosity=2)
