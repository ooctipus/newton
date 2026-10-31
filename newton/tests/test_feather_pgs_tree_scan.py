# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The tree plan and the tree scan: the plan's structures, every held-tree scan configuration and the rule
that picks them.

* :class:`TestTreePlan`: ``TreePlan.build``'s structures: depth-first intervals,
  the held tree's jump table and children, the heavy-chain packing, the sparse
  factor pattern and its support classes, articulation depths, body supports
  and local parents.
* :class:`TestHeldTreePasses`: every held-tree scan (pose and motion scans,
  subtree and external reductions, the twist publication) on every schedule it
  runs, close to a float64 reference: on CUDA its schedule, on the CPU its
  serial twin.
* :class:`TestHeldTreeChains`: trees of 32 bodies or fewer whose heavy chains
  need two warps run the chain scans on two warps.
* :class:`TestScheduleRule`: the schedule follows from the plan's data.
* :class:`TestPath`: a PATH visits the root path.
"""

import unittest
import unittest.mock

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import tree_scan as ts
from newton._src.solvers.feather_pgs.device_preludes import dual_native, lane_group
from newton._src.solvers.feather_pgs.tree_plan import UP_FIRST, UP_SUMS, TreePlan, local_parents
from newton.tests.feather_pgs_tree_scan_fixture import (
    EXTERNAL,
    MOTION,
    PASSES,
    POSE,
    SHAPES,
    TWIST_SUM,
    applies,
    chain_lanes,
    lanes_of,
    layout_of,
    plan_of,
    random_scratch,
    run,
)

EPS = float(np.finfo(np.float32).eps)


def _random_forest(rng, n, roots):
    parent = np.full(n, -1)
    for i in range(roots, n):
        parent[i] = rng.integers(0, i)
    return parent


# ---------------------------------------------------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------------------------------------------------


class TestTreePlan(unittest.TestCase):
    def forests(self):
        rng = np.random.default_rng(0)
        for trial in range(300):
            n = int(rng.integers(1, 65))
            roots = 1 if trial % 3 else int(rng.integers(1, max(2, n // 3 + 1)))
            parent = _random_forest(rng, n, min(roots, n))
            dofs = rng.integers(0, 4, n)
            if dofs.sum() == 0:
                dofs[0] = 1
            yield parent, dofs

    def test_depth_first_intervals(self):
        """Children and roots ascending, each subtree one interval of positions."""
        for parent, dofs in self.forests():
            plan = TreePlan.build(parent, dofs)
            n = parent.size
            np.testing.assert_array_equal(np.sort(plan.dfs_order), np.arange(n))
            np.testing.assert_array_equal(plan.dfs_order[plan.dfs_position], np.arange(n))
            for node in range(n):
                subtree = set()
                for other in range(n):
                    up = other
                    while up >= 0 and up != node:
                        up = parent[up]
                    if up == node:
                        subtree.add(other)
                position = plan.dfs_position[node]
                self.assertEqual(set(plan.dfs_order[position : plan.dfs_end[position]].tolist()), subtree)
                children = np.flatnonzero(parent == node)
                walk = plan.dfs_position[children]
                self.assertTrue(np.all(np.diff(walk) > 0), "children ascend")
            roots = np.flatnonzero(parent < 0)
            self.assertTrue(np.all(np.diff(plan.dfs_position[roots]) > 0), "roots ascend")

    def test_held_tree_topology(self):
        """TreeKineticState's jump table, descending children CSR and depths (tree_kinetic_owner, tree_kinetic_plan)."""
        for parent, dofs in self.forests():
            plan = TreePlan.build(parent, dofs)
            bodies = parent.size
            depth = np.zeros(bodies, np.int32)
            for body in range(bodies):
                depth[body] = depth[parent[body]] + 1 if parent[body] >= 0 else 0
            rounds = max(1, int(depth.max()).bit_length())
            jump = np.empty((rounds, bodies), np.int32)
            jump[0] = parent
            for level in range(1, rounds):
                previous = jump[level - 1]
                jump[level] = np.where(previous >= 0, previous[np.maximum(previous, 0)], -1)
            children, offsets = [], [0]
            for body in range(bodies):
                children.extend(np.flatnonzero(parent == body)[::-1].tolist())
                offsets.append(len(children))
            np.testing.assert_array_equal(plan.depth, depth)
            np.testing.assert_array_equal(plan.jump, jump)
            np.testing.assert_array_equal(plan.children, children)
            np.testing.assert_array_equal(plan.children_offsets, offsets)

    def test_heavy_chain_packing(self):
        """Heavy chains partition each tree into paths along the largest subtrees; the packing pads each to a power
        of two, keeps it within a warp and records its nodes, start, length, light level and anchor lane by lane."""
        for parent, dofs in self.forests():
            if np.count_nonzero(parent < 0) != 1:
                continue
            plan = TreePlan.build(parent, dofs)
            n = parent.size
            for node in range(n):
                children = np.flatnonzero(parent == node)
                if children.size:
                    sizes = plan.subtree_size[children]
                    self.assertEqual(plan.heavy[node], children[np.flatnonzero(sizes == sizes.max())[0]])
            packing = plan.chain_packing(2)
            widths = 1 << np.ceil(np.log2(plan.chain_lengths)).astype(np.int64)
            if packing is None:
                # Refused: a chain wider than a warp, or chains that no first-fit packing of two warps holds.
                continue
            meta = packing.meta
            occupied = meta[:, 0] >= 0
            np.testing.assert_array_equal(np.sort(meta[occupied, 0]), np.arange(n))
            for chain, head in enumerate(plan.chain_heads):
                lanes = np.flatnonzero(meta[:, 0] == head)
                self.assertEqual(lanes.size, 1)
                start = int(lanes[0])
                length = int(plan.chain_lengths[chain])
                self.assertEqual(start // 32, (start + widths[chain] - 1) // 32, "within a warp")
                nodes = meta[start : start + length, 0]
                np.testing.assert_array_equal(meta[start : start + length, 1], start)
                np.testing.assert_array_equal(meta[start : start + length, 2], length)
                np.testing.assert_array_equal(meta[start : start + length, 3], plan.chain_levels[chain])
                np.testing.assert_array_equal(meta[start : start + length, 4], parent[head])
                np.testing.assert_array_equal(nodes[1:], plan.heavy[nodes[:-1]])
                self.assertEqual(plan.heavy[nodes[-1]], -1)
            self.assertEqual(packing.width, int(widths.max()))

    def test_factor_pattern_and_supports(self):
        """The factor pattern from its definition, on random forests: leaf-first subtrees, no fill, one entry per
        related coordinate pair in consistent row, column, level and physical indices, and ancestor masks; the
        support classes name each endpoint union and each coordinate's limit support."""
        rng = np.random.default_rng(1)
        for parent, dofs in self.forests():
            if not 1 <= dofs.sum() <= 64:
                continue
            plan = TreePlan.build(parent, dofs)
            factor = plan.factor
            n = factor.dof_count
            node = plan.dof_node[factor.permutation]  # each factor coordinate's node
            related = plan.ancestor_matrix()  # [i, a]: a is i or one of its ancestors
            np.testing.assert_array_equal(factor.permutation[factor.inverse_permutation], np.arange(n))
            # Leaf-first: an ancestor's coordinates follow its descendants'.
            above = related[node[:, None], node[None, :]] & (node[:, None] != node[None, :])  # [c, r]: r's above c's
            rows, cols = np.nonzero(above)
            self.assertTrue(np.all(cols > rows))
            # Entry (row, col), row >= col, exactly when row's node is col's or one of its ancestors.
            pattern = np.tril(related[node[None, :], node[:, None]])
            np.testing.assert_array_equal(factor.lookup >= 0, pattern)
            # Entries row by row (CSR), and column by column, rows ascending (CSC).
            entries = np.arange(factor.nonzero_count)
            np.testing.assert_array_equal(factor.lookup[factor.entry_rows, factor.columns], entries)
            self.assertTrue(np.all(np.diff(factor.entry_rows) >= 0))
            np.testing.assert_array_equal(np.diff(factor.row_offsets), pattern.sum(axis=1))
            np.testing.assert_array_equal(factor.column_entries, np.lexsort((factor.entry_rows, factor.columns)))
            np.testing.assert_array_equal(np.diff(factor.column_offsets), pattern.sum(axis=0))
            physical = factor.physical_lookup[factor.permutation[:, None], factor.permutation[None, :]]
            np.testing.assert_array_equal(physical, np.where(pattern, factor.lookup, factor.lookup.T))
            # No fill: the Cholesky factor of a positive definite matrix on the related pairs keeps to the pattern.
            symmetric = rng.normal(size=(n, n))
            symmetric = (symmetric + symmetric.T) * (pattern | pattern.T)
            lower = np.linalg.cholesky(symmetric + np.diag(np.abs(symmetric).sum(axis=1) + 1.0))
            np.testing.assert_array_equal(lower[~pattern], 0.0)
            # A pivot's level follows every earlier column its row holds; the level lists group by level.
            off = factor.entry_rows != factor.columns
            self.assertTrue(np.all(factor.levels[factor.entry_rows[off]] > factor.levels[factor.columns[off]]))
            for level in range(factor.factor_level_count):
                columns = factor.factor_level_columns[
                    factor.factor_level_offsets[level] : factor.factor_level_offsets[level + 1]
                ]
                np.testing.assert_array_equal(factor.levels[columns], level)
                entries = factor.factor_level_entries[
                    factor.factor_level_entry_offsets[level] : factor.factor_level_entry_offsets[level + 1]
                ]
                np.testing.assert_array_equal(factor.levels[factor.columns[entries]], level)
            np.testing.assert_array_equal(np.sort(factor.factor_level_entries), np.flatnonzero(off))
            # Masks: the factor coordinates of each node's ancestors and itself.
            bits = np.left_shift(np.uint64(1), np.arange(n, dtype=np.uint64))
            expected = np.bitwise_or.reduce(np.where(related[:, node], bits, np.uint64(0)), axis=1)
            np.testing.assert_array_equal(factor.joint_ancestor_mask, expected)
            np.testing.assert_array_equal(factor.ancestor_mask, expected[factor.dof_joint])

            def coordinates(mask, n=n):
                return [f for f in range(n) if int(mask) >> f & 1]

            body_masks = factor.joint_ancestor_mask[rng.integers(0, parent.size, parent.size)]
            for width in (min(4, n), n):
                classes = plan.support_classes(body_masks, width)
                unions = [int(a | b) for a in (0, *body_masks) for b in (0, *body_masks)] + list(factor.ancestor_mask)
                self.assertEqual(classes is None, max(int(m).bit_count() for m in unions) > width)
                if classes is None:
                    continue

                def pattern_of(index, classes=classes):
                    """A class's coordinates, checked ascending and padded with -1."""
                    row = classes.class_dof[index]
                    used = int(np.count_nonzero(row >= 0))
                    self.assertTrue(np.all(np.diff(row[:used]) > 0) and np.all(row[used:] == -1))
                    return row[:used].tolist()

                for a, mask_a in enumerate(body_masks):
                    self.assertEqual(pattern_of(classes.pair_class[classes.body_class[a], 0]), coordinates(mask_a))
                    for b, mask_b in enumerate(body_masks):
                        pair = classes.pair_class[classes.body_class[a], classes.body_class[b]]
                        self.assertEqual(pattern_of(pair), coordinates(mask_a | mask_b))
                for dof, mask in enumerate(factor.ancestor_mask):
                    self.assertEqual(pattern_of(classes.limit_class[dof]), coordinates(mask))

    def test_local_parents(self):
        """The held operators' and the solver's local-parent remaps (one function, both spellings)."""
        rng = np.random.default_rng(2)
        ancestor, starts = [], []
        for _ in range(5):
            starts.append(len(ancestor))
            parent = _random_forest(rng, 9, 1)
            ancestor.extend(np.where(parent >= 0, parent + starts[-1], -1))
        ancestor = np.asarray(ancestor)
        starts = np.asarray(starts)
        ours = local_parents(ancestor, starts, 9)
        for a, start in enumerate(starts):
            slice_ = ancestor[start : start + 9]
            np.testing.assert_array_equal(ours[a], np.where(slice_ < 0, -1, slice_ - start))  # held_tree
            np.testing.assert_array_equal(ours[a], np.where(slice_ >= start, slice_ - start, -1))  # the solver
        # A range that starts inside a tree leaves its first joint's ancestor outside: not a group of whole trees.
        with self.assertRaises(ValueError):
            local_parents(ancestor, starts + 1, 8)

    def test_up_flags_set_every_slot_they_read(self):
        """A descending walk that follows the up flags (the serial up scans' protocol) sums every subtree from slots
        holding NaN: each slot it reads, a first child stored first, so no slot needs clearing."""
        rng = np.random.default_rng(4)
        for parent, dofs in self.forests():
            plan = TreePlan.build(parent, dofs)
            own = rng.standard_normal(parent.size)
            slot = np.full(parent.size, np.nan)
            total = np.empty(parent.size)
            for node in range(parent.size - 1, -1, -1):
                flags = int(plan.up_flags[node])
                children = slot[node] if flags & UP_SUMS else 0.0
                total[node] = own[node] + children
                up = int(parent[node])
                if up >= 0:
                    slot[up] = total[node] if flags & UP_FIRST else slot[up] + total[node]
            expected = own.copy()
            for level in range(plan.max_depth, 0, -1):
                nodes = plan.level_nodes[plan.level_offsets[level] : plan.level_offsets[level + 1]]
                np.add.at(expected, parent[nodes], expected[nodes])
            np.testing.assert_allclose(total, expected, rtol=1e-12, atol=1e-12)
            has_children = np.bincount(parent[parent >= 0], minlength=parent.size) > 0
            np.testing.assert_array_equal((plan.up_flags & UP_SUMS) != 0, has_children)

    def test_cycles_and_bad_parents_raise(self):
        for parent in ([0], [-1, 2, 1], [-1, 5]):
            with self.assertRaises(ValueError):
                TreePlan.build(parent)


# ---------------------------------------------------------------------------------------------------------------------
# A float64 reference of each scan
# ---------------------------------------------------------------------------------------------------------------------


def _quat_multiply(a, b):
    x1, y1, z1, w1 = np.moveaxis(a, -1, 0)
    x2, y2, z2, w2 = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        axis=-1,
    )


def _quat_rotate(q, v):
    u, w = q[..., :3], q[..., 3:]
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def reference(item, plan, layout, scratch, flag, vhat=None, axes=None):
    """The pass's result in float64: the down prefix along each root path, or the subtree sum."""
    s = scratch.astype(np.float64).copy()
    bodies = plan.count
    order = np.arange(bodies)
    if item.combine is POSE:
        pose = s[:, : 7 * bodies].reshape(-1, bodies, 7)
        for node in order:
            p = plan.parent[node]
            if p >= 0:
                t = _quat_rotate(pose[:, p, 3:], pose[:, node, :3]) + pose[:, p, :3]
                q = _quat_multiply(pose[:, p, 3:], pose[:, node, 3:])
                pose[:, node, :3], pose[:, node, 3:] = t, q
        return s, 7 * bodies
    if item.combine is MOTION:
        base = 7 * bodies
        v = s[:, base : base + 6 * bodies].reshape(-1, bodies, 6)
        a = s[:, base + 6 * bodies : base + 12 * bodies].reshape(-1, bodies, 6)
        a[:] = 0.0
        for node in order:
            p = plan.parent[node]
            if p >= 0:
                pl, pw, vl, vw = v[:, p, :3], v[:, p, 3:], v[:, node, :3].copy(), v[:, node, 3:].copy()
                a[:, node, :3] = a[:, p, :3] + np.cross(pw, vl) + np.cross(pl, vw)
                a[:, node, 3:] = a[:, p, 3:] + np.cross(pw, vw)
                v[:, node] = v[:, p] + v[:, node]
        return s, None
    if item.combine is TWIST_SUM:
        own = np.einsum("gnk,gn->gnk", axes.astype(np.float64), vhat.astype(np.float64))
        result = own.copy()
        for node in order:
            p = plan.parent[node]
            if p >= 0:
                result[:, node] += result[:, p]
        return result, None
    groups = [(layout.wrench_offset if item.combine is not EXTERNAL else 0, 6)]
    if item.combine is not EXTERNAL and flag:
        groups.append((layout.moment_offset, 13))
    if item.combine is EXTERNAL and not flag:
        return s, None
    for offset, width in groups:
        block = s[:, offset : offset + width * bodies].reshape(-1, bodies, width)
        for node in order[::-1]:
            p = plan.parent[node]
            if p >= 0:
                block[:, p] += block[:, node]
    return s, None


# ---------------------------------------------------------------------------------------------------------------------
# The held tree's passes
# ---------------------------------------------------------------------------------------------------------------------


def _cases(packed_widths=(0, 8)):
    for shape in SHAPES:
        plan = plan_of(shape)
        for packed in packed_widths:
            for name, item in PASSES.items():
                chain = item.schedule is ts.Schedule.HEAVY_CHAIN
                if chain and packed:
                    continue
                lanes = chain_lanes(plan) if chain else lanes_of(plan, packed)
                if (packed and lanes >= 32) or not applies(item, plan, lanes):
                    continue
                for flag in (0, 1) if item.direction is ts.Direction.UP else (1,):
                    yield shape, plan, lanes, name, item, flag


def _assert_close_to_reference(test, item, plan, layout, start, result, flag, groups, twists=None):
    if item.combine is TWIST_SUM:
        rng = np.random.default_rng(3 + 1)
        axes = rng.standard_normal((groups * plan.count, 6)).astype(np.float32).reshape(groups, plan.count, 6)
        vhat = rng.standard_normal(groups * plan.count).astype(np.float32).reshape(groups, plan.count)
        expected, _ = reference(item, plan, layout, start, flag, vhat, axes)
        got = twists.reshape(groups, plan.count, 6)
    else:
        expected, _ = reference(item, plan, layout, start, flag)
        got = result.reshape(start.shape)
    scale = np.abs(expected).max() + 1.0
    error = np.abs(got.astype(np.float64) - expected).max() / scale
    test.assertLess(error, 64 * plan.count * EPS, f"{item.name} on {plan.count} nodes")


def check_held_passes(test, device):
    """Every pass, shape and lane width (warps of packed trees, a warp or two per tree): the scan close to a
    float64 reference. On the CPU every pass runs its serial twin (the twist publication is CUDA only)."""
    cuda = wp.get_device(device).is_cuda
    groups = 64 if cuda else 4
    for shape, plan, lanes, name, item, flag in _cases((0, 8) if cuda else (0,)):
        if name == "twist_shfl" and not cuda:
            continue
        with test.subTest(shape=shape, lanes=lanes, scan=name, flag=flag):
            layout = layout_of(plan, lanes)
            scratch, twists = run(item, plan, layout, groups, flag, 3, device)
            start = random_scratch(layout, groups, 3)
            _assert_close_to_reference(test, item, plan, layout, start, scratch, flag, groups, twists=twists)


class TestHeldTreePasses(unittest.TestCase):
    pass


def _held_tree_model(shape: str, worlds: int, device):
    """A free-root tree of ``shape``, revolute joints below the root, replicated over ``worlds``."""
    import newton  # noqa: PLC0415

    scene = newton.ModelBuilder()
    bodies, joints = [], []
    for index, parent in enumerate(SHAPES[shape]):
        body = scene.add_link(mass=0.5 + 0.1 * index, inertia=wp.mat33(np.diag([0.04, 0.05, 0.07])))
        bodies.append(body)
        if index == 0:
            joints.append(scene.add_joint_free(body))
            continue
        anchor = wp.transform(wp.vec3(0.07, 0.03 * (index % 5), -0.08), wp.quat_identity())
        axis = (float(index % 3), float((index + 1) % 3), 1.0)
        joints.append(scene.add_joint_revolute(bodies[parent], body, axis=axis, parent_xform=anchor, armature=0.05))
    scene.add_articulation(joints)
    builder = newton.ModelBuilder()
    builder.replicate(scene, worlds)
    return builder.finalize(device=device)


@unittest.skipUnless(wp.is_cuda_available(), "The held tree is a CUDA path")
class TestHeldTreeChains(unittest.TestCase):
    def test_two_warp_chains_of_small_trees(self):
        """Trees of 32 bodies or fewer whose padded heavy chains need two warps (g1_30, binary31) run every scan on
        the heavy chains, a tree's two warps one block: the repair and the finish publish forward kinematics.

        (The owner these schedules replaced launched such a tree's 64 threads with one warp's lane group, so the
        second warp took a second tree's state block, past the block's shared memory: CUDA error 700.)
        """
        import newton  # noqa: PLC0415
        from newton._src.solvers.feather_pgs import tree_kinetic_state  # noqa: PLC0415
        from newton.tests.test_feather_pgs_sparse_solver import _make_solver  # noqa: PLC0415

        device = wp.get_cuda_device()
        on_chains = tree_kinetic_state.HeldSchedules(*(ts.Schedule.HEAVY_CHAIN,) * 4)
        for shape in ("g1_30", "binary31"):
            with self.subTest(shape=shape):
                self.assertEqual(plan_of(shape).chain_packing(2).lanes, 64)
                self.assertLessEqual(plan_of(shape).count, 32)
                model = _held_tree_model(shape, 8, device)
                with (
                    unittest.mock.patch.object(tree_kinetic_state, "held_lanes", lambda tree, groups, *, cuda: 64),
                    unittest.mock.patch.object(
                        tree_kinetic_state, "held_schedules", lambda tree, lanes, *, chains: on_chains
                    ),
                ):
                    solver = _make_solver(model, True)
                self.assertIsNotNone(solver.factor.held)
                # The sparse factor's held group is the owner's only group.
                (owner,) = solver.factor.held.groups
                self.assertEqual((owner.layout.lanes, owner.layout.pack, owner.block_dim), (64, 1, 64))
                self.assertEqual(owner.schedules, on_chains)
                rng = np.random.default_rng(13)
                state = model.state()
                q = model.joint_q.numpy()
                q_start, types = model.joint_q_start.numpy(), model.joint_type.numpy()
                revolute = np.flatnonzero(types == int(newton.JointType.REVOLUTE))
                q[q_start[revolute]] = rng.uniform(-1.0, 1.0, revolute.size)
                state.joint_q.assign(q)
                state.joint_qd.assign(rng.uniform(-0.5, 0.5, model.joint_dof_count).astype(np.float32))
                expected = model.state()
                newton.eval_fk(model, state.joint_q, state.joint_qd, expected)
                owner.begin(state, solver.factor, state.joint_qd, 1.0 / 240.0, True)
                np.testing.assert_allclose(state.body_q.numpy(), expected.body_q.numpy(), atol=2e-6)
                np.testing.assert_allclose(state.body_qd.numpy(), expected.body_qd.numpy(), atol=2e-5)
                np.testing.assert_array_equal(owner.status.numpy(), 0)


# ---------------------------------------------------------------------------------------------------------------------
# The schedule rule
# ---------------------------------------------------------------------------------------------------------------------


class TestScheduleRule(unittest.TestCase):
    def test_rule(self):
        """The rule follows the plan's data: warp-sized trees jump by shuffles; narrow ones sum along heavy chains,
        bushy ones (a star, a binary tree, chains needing two warps) by intervals; wide ones jump through shared
        memory and sum by levels where no chains pack; the CPU and paths run serially."""
        add6 = ts.add(ts.Group("wrench", 6))
        picks = {}
        for shape in SHAPES:
            plan = plan_of(shape)
            lanes = lanes_of(plan)
            down = ts.choose_schedule(plan, ts.Direction.DOWN, ts.TRANSFORM, cuda=True, lanes=lanes)
            up = ts.choose_schedule(plan, ts.Direction.UP, add6, cuda=True, lanes=lanes)
            picks[shape] = (down, up)
            cpu = ts.choose_schedule(plan, ts.Direction.DOWN, ts.TRANSFORM, cuda=False, lanes=lanes)
            self.assertIs(cpu, ts.Schedule.SERIAL)
            self.assertIs(ts.choose_schedule(plan, ts.Direction.PATH, add6, cuda=True, lanes=lanes), ts.Schedule.SERIAL)
            # The same data, the same choice; a launch packing several trees per block leaves no heavy chain.
            again = ts.choose_schedule(plan_of(shape), ts.Direction.UP, add6, cuda=True, lanes=lanes)
            self.assertIs(again, up)
            packed = ts.choose_schedule(plan, ts.Direction.UP, add6, cuda=True, lanes=lanes, chains=False)
            self.assertIsNot(packed, ts.Schedule.HEAVY_CHAIN)
        self.assertEqual(picks["anymal13"], (ts.Schedule.JUMP_SHFL, ts.Schedule.HEAVY_CHAIN))
        self.assertEqual(picks["g1_30"], (ts.Schedule.JUMP_SHFL, ts.Schedule.DFS_INTERVAL))
        self.assertEqual(picks["binary31"], (ts.Schedule.JUMP_SHFL, ts.Schedule.DFS_INTERVAL))
        self.assertEqual(picks["chain64"], (ts.Schedule.JUMP_SMEM, ts.Schedule.LEVEL))
        with self.assertRaises(ValueError):
            ts.choose_schedule(plan_of("ant9"), ts.Direction.UP, ts.TRANSFORM, cuda=True, lanes=32)

    def test_unsupported_configurations_raise(self):
        plan = plan_of("ant9")
        with self.assertRaises(ValueError):
            ts.tree_scan(plan, ts.TRANSFORM, ts.Direction.UP, ts.Schedule.LEVEL)
        with self.assertRaises(ValueError):
            ts.tree_scan(plan, ts.TRANSFORM, ts.Direction.DOWN, ts.Schedule.DFS_INTERVAL)


# ---------------------------------------------------------------------------------------------------------------------
# The root path
# ---------------------------------------------------------------------------------------------------------------------


def check_path(test, device):
    """A PATH visits a node's ancestors and the node, root first (parent-first plans and others), against the
    parent pointers."""
    for shape in ("random24", "binary31", "star32", "chain32", "humanoid44", "random60"):
        parent = np.asarray(SHAPES[shape])
        # The same tree with its nodes renumbered so that children precede parents.
        reverse = np.where(parent[::-1] >= 0, parent.size - 1 - parent[::-1], -1)
        for label, parents in (("parent-first", parent), ("children first", reverse)):
            plan = TreePlan.build(parents)
            n = plan.count
            expected = np.zeros((n, n), np.int32)
            for node in range(n):
                path, a = [], node
                while a >= 0:
                    path.append(a)
                    a = parents[a]
                expected[node, path[::-1]] = np.arange(1, len(path) + 1)
            scan = ts.tree_scan(
                plan,
                ts.add(ts.Group("v", 1)),
                ts.Direction.PATH,
                ts.Schedule.SERIAL,
                visit="        visits.data[node*count+ancestor]=++rank;",
            )

            @wp.func_native("    int rank=0;\n" + dual_native(scan.text, preludes=(lane_group(1), *scan.preludes)))
            def native_path(plan: ts.ScanPlan, node: int, count: int, visits: wp.array[int]): ...

            def run(plan: ts.ScanPlan, count: int, visits: wp.array[int]):
                native_path(plan, wp.tid(), count, visits)

            visits = wp.zeros(n * n, dtype=int, device=device)
            kernel = wp.kernel(enable_backward=False, module="unique")(run)
            wp.launch(kernel, n, inputs=[ts.device_plan(plan, device), n, visits], device=device)
            np.testing.assert_array_equal(visits.numpy().reshape(n, n), expected, err_msg=f"native {shape} {label}")


class TestPath(unittest.TestCase):
    pass


from newton.tests.unittest_utils import add_function_test, get_test_devices  # noqa: E402

add_function_test(TestHeldTreePasses, "test_held_passes", check_held_passes, devices=get_test_devices())
add_function_test(TestPath, "test_path", check_path, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main(verbosity=2)
