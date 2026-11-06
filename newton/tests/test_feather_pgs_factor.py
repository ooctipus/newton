# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The tree factor's contract: each articulation's ``P (H + R + K) P^T = L L^T`` held as the walk, judged through
the factor's interface against independent float64 references.

- **The tree plan and its scans**: :class:`.tree_plan.TreePlan`'s structures (depth-first intervals, the jump table,
  heavy chains, the factor pattern, local parents) and every held-tree scan on every schedule it runs, close to a
  float64 reference; a PATH visits the root path.
- **The tree layout**: the factor pattern of any size, and the state block in shared or global memory stepping alike.
- **The walk is the factor** (Z_UNION §1.2, §5): :class:`.factor.walk.TreeWalk` against the factor of the independent
  float64 mass matrix (:mod:`.feather_pgs_mass_reference`) on every catalogue model (``L^-1``, the leave, a row from its
  seed, the meeting); F1, the step's rows and leave against the walk's on every CPU route; F2, the float32 walk on
  ``long_chain(N)`` without armature; F4 and F5, row entries per seed.
- **The walk on the device** (FL3.1): the native walk up and down on the CPU and CUDA against the float64 walk, F2 on
  the device, a held factor whitening the step's ``J``, one frame per placement, the refresh against the float64 walk
  and the class emitter's walks both ways.
- **The reference route against an independent float64 evaluation**: ``H`` as the finish forms it, ``L L^T``, the
  response and the swept rows' whitening, each within a multiple of the float32 rounding.
- **The drives**: per-coordinate drive rows, mixed target layouts.
- **The generalized laws** under a permuted physical index.
- **The FK/ID cache's publication** across graph replays, and **the device proof** of the frozen arrays it rests on.

Each bound is a multiple ``BOUND`` of ``n u`` (``u`` the float32 rounding) times the factor's condition where a
factorization enters. ``python -m newton.tests.test_feather_pgs_factor report`` prints the walk's measured tables.
"""

import contextlib
import dataclasses
import functools
import hashlib
import os
import sys
import unittest
import warnings
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.core import memory
from newton._src.sim.enums import JointType
from newton._src.solvers.feather_pgs import field_group, tree_kinetic_plan, tree_kinetic_state
from newton._src.solvers.feather_pgs import tree_scan as ts
from newton._src.solvers.feather_pgs.device_preludes import DevicePrelude, dual_native, lane_group, substitute
from newton._src.solvers.feather_pgs.device_proof import _WORDS_PER_THREAD, DeviceProof
from newton._src.solvers.feather_pgs.factor import kernels as factor_kernels
from newton._src.solvers.feather_pgs.factor import walk as factor_walk
from newton._src.solvers.feather_pgs.factor.fields import DriveRows
from newton._src.solvers.feather_pgs.factor.walk import (
    WALK_COORDINATE,
    WALK_FRAME,
    WALK_LINK,
    WALK_NOW,
    TreeWalk,
    local_frames,
    unpack_walk_records,
)
from newton._src.solvers.feather_pgs.kernel_template import KernelTemplate, items, params
from newton._src.solvers.feather_pgs.placement import Placement
from newton._src.solvers.feather_pgs.row_kernels import PGS_CONSTRAINT_TYPE_SPEED
from newton._src.solvers.feather_pgs.tree_factor import prepare_augmented_drives
from newton._src.solvers.feather_pgs.tree_kinetic_layout import TreeLayout
from newton._src.solvers.feather_pgs.tree_plan import UP_FIRST, UP_SUMS, TreePlan, local_parents
from newton.solvers import SolverFeatherPGS
from newton.tests import feather_pgs_scenes as scenes
from newton.tests import test_feather_pgs_conformance as conformance
from newton.tests.feather_pgs_law_harness import dense_factor32, free_body_Z, unpacked_rows, walk_inverse
from newton.tests.feather_pgs_mass_reference import MassReference
from newton.tests.feather_pgs_tree_scan_fixture import (
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
from newton.tests.unittest_utils import add_function_test, get_test_devices

# =====================================================================================================================
# The tree plan and its scans
# =====================================================================================================================

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
    groups = [(layout.wrench_offset, 6)]
    if flag:
        groups.append((layout.moment_offset, 13))
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
        on_chains = tree_kinetic_state.HeldSchedules(*(ts.Schedule.HEAVY_CHAIN,) * 3)
        for shape in ("g1_30", "binary31"):
            with self.subTest(shape=shape):
                self.assertEqual(plan_of(shape).chain_packing(2).lanes, 64)
                self.assertLessEqual(plan_of(shape).count, 32)
                model = _held_tree_model(shape, 8, device)
                with (
                    mock.patch.object(tree_kinetic_state, "held_lanes", lambda tree, groups, device: 64),
                    mock.patch.object(tree_kinetic_state, "held_schedules", lambda tree, lanes, *, chains: on_chains),
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
                owner.begin(state, solver.factor, state.joint_qd, 1.0 / 240.0)
                np.testing.assert_allclose(state.body_q.numpy(), expected.body_q.numpy(), atol=2e-6)
                np.testing.assert_allclose(state.body_qd.numpy(), expected.body_qd.numpy(), atol=2e-5)
                np.testing.assert_array_equal(owner.status.numpy(), 0)


# ---------------------------------------------------------------------------------------------------------------------
# The schedule rule
# ---------------------------------------------------------------------------------------------------------------------


class TestScheduleRule(unittest.TestCase):
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


add_function_test(TestHeldTreePasses, "test_held_passes", check_held_passes, devices=get_test_devices())
add_function_test(TestPath, "test_path", check_path, devices=get_test_devices())


# =====================================================================================================================
# The tree layout
# =====================================================================================================================

# A chain of more bodies and coordinates than a block's lanes.
_LONG_CHAIN = TreePlan.build(np.arange(-1, 105), np.r_[6, np.ones(105, np.int64)])


class TestLargeTrees(unittest.TestCase):
    def test_factor_pattern_holds_any_size(self):
        """Keep the fill-free pattern past the masks' 64 coordinates: a chain's every entry, a star's diagonal."""
        chain = _LONG_CHAIN.factor
        self.assertEqual(chain.nonzero_count, 111 * 112 // 2)
        self.assertEqual(chain.nonzero_count, _LONG_CHAIN.nonzero_count)
        self.assertEqual(np.count_nonzero(chain.lookup >= 0), chain.nonzero_count)
        self.assertFalse(np.any(chain.ancestor_mask) or np.any(chain.joint_ancestor_mask))
        star = TreePlan.build(np.r_[-1, np.zeros(70, np.int64)]).factor
        # The root's coordinate (the last in leaf-first order) couples to every leaf.
        self.assertEqual(star.nonzero_count, 71 + 70)
        self.assertEqual(TreePlan.build(np.arange(-1, 63)).factor.joint_ancestor_mask[-1], (1 << 64) - 1)

    def test_a_tree_past_the_lanes_takes_the_depth_levels(self):
        """Hold a tree of more bodies than a block's lanes: 64 lanes, and the depth-level scans on CUDA."""
        lanes = tree_kinetic_state.held_lanes(_LONG_CHAIN, 3, memory.Device.of(120, 188, 101_376))
        self.assertEqual(lanes, 64)
        schedules = tree_kinetic_state.held_schedules(_LONG_CHAIN, lanes, chains=True)
        self.assertEqual(set(dataclasses.asdict(schedules).values()), {ts.Schedule.LEVEL})
        layout = TreeLayout(106, 111, _LONG_CHAIN.nonzero_count, 111, 105, 6, tuple(range(110, -1, -1)), lanes=lanes)
        self.assertEqual((layout.block, layout.pack), (64, 1))
        with self.assertRaises(ValueError):
            TreeLayout(106, 111, 111, 111, 105, 6, tuple(range(111)), lanes=32)


class TestStateBlockPlacement(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "Placement is a CUDA choice")
    def test_both_placements_step_alike(self):
        """Step two different trees with their state blocks in shared and in global memory: the same state to rounding."""
        scene, steps, finals = scenes.sprawl(worlds=2), 8, []
        for budget in (None, 64):
            with (
                mock.patch.object(tree_kinetic_plan, "static_budget", return_value=budget)
                if budget
                else contextlib.nullcontext()
            ):
                model = scene.build("cuda:0")
                solver = SolverFeatherPGS(model, **scene.solver_options)
            self.assertEqual(solver.factor.held.groups[0].layout.global_block, budget is not None)
            # The row order must not follow thread scheduling: sorted contacts order the rows, and the scene's row limit
            # keeps them all, since a thread race picks the rows that survive a limit.
            pipeline = newton.CollisionPipeline(model, rigid_contact_max=scene.rigid_contact_max, deterministic=True)
            contacts = pipeline.contacts()
            state, out, control = model.state(), model.state(), model.control()
            # Different trees: a block read at another tree's offset then holds other values.
            joint_q = state.joint_q.numpy()
            joint_q[model.joint_coord_count // 2 + 7] += 0.05
            state.joint_q.assign(joint_q)
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
            for _ in range(steps):
                pipeline.collide(state, contacts)
                solver.step(state, out, control, contacts, scene.dt)
                state, out = out, state
            finals.append({name: getattr(state, name).numpy() for name in ("joint_q", "joint_qd", "body_q", "body_qd")})
        # Only the address space differs, so ptxas may fuse multiply-adds apart (BACKBONE rule 4), though both devices
        # now read the same bits. One rounding of a position, u at the scene's size, reads as u / dt in a contact's
        # target; over the steps that sums to steps * u / dt in velocity and steps * (steps + 1) / 2 * u in position.
        # Every joint coordinate moved by one ulp stays inside these, and a lost store of one body fails them 14x over.
        u = np.finfo(np.float32).eps * np.abs(finals[0]["body_q"]).max()
        position, velocity = 0.5 * steps * (steps + 1) * u, steps * u / scene.dt
        for name, tolerance in (
            ("joint_q", position),
            ("body_q", position),
            ("joint_qd", velocity),
            ("body_qd", velocity),
        ):
            np.testing.assert_allclose(finals[1][name], finals[0][name], rtol=0.0, atol=tolerance, err_msg=name)


# =====================================================================================================================
# The walk is the factor
# =====================================================================================================================

ROUNDING = 2.0**-24
BOUND = 16.0
"""The rounding bounds' multiple of ``n u``."""
STEPS = 8
CHAINS = (8, 32, 64, 106, 200, 300, 1000)
SEGMENT = 32
REVERSED = ("gripper_point", "hand_point", "legged_point", "pinch_twist_point_torsion")
"""F1's problems of several worlds, dense and free bodies' rows, run again with every row table reversed."""


# ---------------------------------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------------------------------


def jacobian(walk: TreeWalk, body: int, wrench: np.ndarray) -> np.ndarray:
    """``J^T`` of rows seeded by ``wrench`` at ``body`` (``[6]`` or ``[6, m]``): ``s_c^T w`` at every coordinate of
    the body's root path, the wrench carried rigidly between frames (no joint takes any of it)."""
    w = np.asarray(wrench, dtype=np.float64)
    x = np.zeros((walk.s.shape[0], *w.shape[1:]))
    for b in walk.ancestors(body):
        x[walk.first[b] : walk.first[b + 1]] = walk.s[walk.first[b] : walk.first[b + 1]].astype(np.float64) @ w
        d = walk.offset[b].astype(np.float64)
        w = np.concatenate((w[:3], w[3:] - np.cross(d, w[:3], axis=0)))
    return x


def factor_matrix(walk: TreeWalk) -> np.ndarray:
    """``P^T L P`` by coordinate, float64: the inverse of the walk's footprints of every unit covector."""
    inverse, _ = walk.up(covector=np.eye(walk.s.shape[0]))
    return np.linalg.inv(inverse.astype(np.float64))


def contact_wrenches(frame_offset: np.ndarray, normal=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Unit forces along ``normal`` and two tangents at a contact ``frame_offset`` from a body's frame: the wrenches
    ``(f, r x f)`` at the frame, one column each."""
    n = np.asarray(normal, dtype=np.float64)
    n /= np.linalg.norm(n)
    tangents = np.linalg.svd(n[None, :])[2][1:]
    forces = np.stack((n, *tangents), axis=1)
    return np.concatenate((forces, np.cross(frame_offset[:, None], forces, axis=0)), axis=0)


def relative(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, np.float64) - b) / max(np.linalg.norm(b), 1e-300))


# ---------------------------------------------------------------------------------------------------------------------
# Identities: the walk is the factor (float64, against the independent mass matrix)
# ---------------------------------------------------------------------------------------------------------------------


def identities(model, joint_q: np.ndarray, rng) -> dict:
    """The walk against the dense float64 factor of ``H + D`` (``D`` random armature) of every articulation: its
    ``L^-1`` (footprints of unit covectors), its leave, a body seed's row and the meeting of two rows."""
    reference = MassReference(model)
    worst = dict.fromkeys(("Linv", "leave", "seed", "meet", "twist"), 0.0)
    for art in range(model.articulation_count):
        H = reference.mass_matrix(joint_q, art)
        n = H.shape[0]
        if n == 0:
            continue
        D = rng.uniform(0.0, 1.0e-3, n)
        walk = TreeWalk(*local_frames(model, joint_q, art), D)
        L = np.linalg.cholesky((H + np.diag(D))[::-1, ::-1])
        footprints, _ = walk.up(covector=np.eye(n))
        expected = np.linalg.solve(L, np.eye(n)[::-1])[::-1]
        worst["Linv"] = max(worst["Linv"], relative(footprints, expected))
        u = rng.normal(size=(n, 3))
        qd, twist = walk.down(u)
        worst["leave"] = max(worst["leave"], relative(qd, np.linalg.solve(L.T, u[::-1])[::-1]))
        bodies = walk.parent.size
        k, l = int(rng.integers(bodies)), int(rng.integers(bodies))
        w_k, w_l = rng.normal(size=(6, 2)), rng.normal(size=(6, 3))
        x = jacobian(walk, k, w_k)
        z_k, _ = walk.up(k, w_k)
        worst["seed"] = max(worst["seed"], relative(z_k, np.linalg.solve(L, x[::-1])[::-1]))
        z_l, _ = walk.up(l, w_l)
        worst["meet"] = max(worst["meet"], relative(walk.meet(k, w_k, l, w_l), z_k.T @ z_l))
        # The leave's state at a body is its twist: Gamma_k^T u.
        gamma, _ = walk.up(k, np.eye(6))
        worst["twist"] = max(worst["twist"], relative(twist[k], gamma.T @ u))
    return worst


# ---------------------------------------------------------------------------------------------------------------------
# F1: the catalogue's rows and leaves
# ---------------------------------------------------------------------------------------------------------------------


def catalogue() -> list:
    """Every problem of the CPU catalogue (:func:`.test_feather_pgs_conformance.cpu_groups`) with both CPU routes:
    the reference's (dense storage, the general placement) and the default's (the group's options)."""
    return [(group.physics, route) for group in conformance.cpu_groups() for route in (group.reference, *group.routes)]


class Capture:
    """The rows a step builds: whether the class holds free bodies' rows (their writer stages with free bodies,
    :meth:`.rows.RowLaws.write_free_bodies`), then the step's last values."""

    def __init__(self, solver):
        self.solver = solver
        self.mf = True if solver.forest.has_free_rigid_bodies else None

    def finish(self):
        """The step's end: its final rows and impulses (after any elimination), the predictor and the velocity."""
        placement, solver = self.solver.placement, self.solver
        self.Z_final = placement.packed_Z.numpy().astype(np.float64)
        self.impulses = placement.impulses.numpy().astype(np.float64)
        # The general placement's one writer (FL3.3) runs no pack: its rows' sides and counts are the step's.
        self.sides = placement.row_sides.numpy().copy()
        self.counts = placement.constraint_count.numpy().copy()
        self.mf_final = None
        if self.mf is not None:
            # Each row's sides (its responding bodies' articulations) and their first world coordinates.
            sides = placement.row_sides.numpy()[:, placement.body_base :].copy()
            offsets = solver.forest.articulation_world_dof_offset.numpy()
            dofs = np.where(sides >= 0, offsets[np.maximum(sides, 0)], -1)
            self.mf_final = (
                free_body_Z(solver)[..., :6].astype(np.float64),
                free_body_Z(solver)[..., 6:].astype(np.float64),
                placement.impulses.numpy()[:, placement.body_base :].astype(np.float64),
                placement.mf_constraint_count.numpy().copy(),
                sides[..., 0],
                sides[..., 1],
                dofs[..., 0],
                dofs[..., 1],
                placement.row_type.numpy()[:, placement.body_base :].copy(),
            )
        self.v_hat = solver.v_hat.numpy().astype(np.float64)
        self.v_out = solver.v_out.numpy().astype(np.float64)
        # The line of each family's store where a world's rows start (its row tables), as every step read takes it.
        self.dense_first = placement.dense_first.numpy().copy()
        self.mf_first = placement.dense_first.numpy().copy()


def pipeline_options(scene) -> dict:
    """The collision pipeline's options a scene asks for (as :func:`.test_feather_pgs_conformance.simulate`)."""
    options = {}
    if scene.rigid_contact_max is not None:
        options["rigid_contact_max"] = scene.rigid_contact_max
    if scene.contact_matching is not None:
        options["contact_matching"] = scene.contact_matching
    if scene.deterministic_contacts:
        options["deterministic"] = True
    return options


def run_route(physics, route, device="cpu", steps=STEPS):
    """Step ``physics`` through ``route``; return the solver, the last step's input coordinates (as the forest holds
    them) and its :class:`Capture`."""
    scene = physics.scene
    model = scene.build(device)
    pipeline = newton.CollisionPipeline(model, **pipeline_options(scene))
    contacts = pipeline.contacts()
    options = {**conformance.BASE_OPTIONS, **scene.solver_options, **physics.options(physics.iterations)}
    options.update(route.options)
    # The rows' material law is the sweep's: these contacts hold no per-contact material.
    options.pop("contact_compliance", None)
    with conformance.route_context(route), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver = SolverFeatherPGS(model, pins=route.solver_pins(), **options)
        capture = Capture(solver)
        state, out, control = model.state(), model.state(), model.control()
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        joint_q_in = None
        for step in range(steps):
            state.clear_forces()
            if scene.control is not None:
                scene.control(model, control, state, step)
            pipeline.collide(state, contacts)
            solver.step(state, out, control, contacts, scene.dt)
            joint_q_in = solver.cut.mirror(state).joint_q.numpy().copy()
            state, out = out, state
        capture.finish()
    return solver, joint_q_in, capture


@dataclasses.dataclass
class RowCheck:
    """F1 on one run: the worst of each measure, and the counts behind them.

    A float32 whitening of a row is exact for a factor perturbed by ``n u`` of its norm, so it is within rounding of the
    walk's when ``|Z_step - Z_walk| / |Z_walk| <= BOUND n u kappa``, ``kappa`` the walk's factor's condition number
    (two-norms); the leave likewise, with the publication's rounding of ``v_hat + dv`` added. A measure's ratio to its
    rounding passes at ``BOUND``."""

    rows: int = 0
    sides: int = 0
    seeded: int = 0
    free_rows: int = 0
    kappa: float = 0.0
    """The largest condition number of an articulation's factor."""
    forward_step: float = 0.0
    """Worst ``|Z_step - Z_walk64| / |Z_walk64|`` of a side."""
    forward_walk32: float = 0.0
    """The same of the float32 walk."""
    ratio: float = 0.0
    """Worst ``|Z_step - Z_walk64| / |Z_walk64|`` over ``n u kappa`` (pass: at most ``BOUND``)."""
    writer: float = 0.0
    """Worst ``|J(seed) - J| / |J|`` of a row's recovered seed: how near the writer's float32 row is a body seed."""
    seed: float = 0.0
    """Worst row walked from its seed against the walk of the covector that seed writes (float64)."""
    meet: float = 0.0
    """Worst meeting at the LCA against the seeds' footprints' dot product (float64)."""
    leave: float = 0.0
    """Worst ``|dv_walk - dv_step| / |dv_walk|`` of an articulation's leave."""
    leave_ratio: float = 0.0
    """The same over its rounding, ``n u kappa`` and the publication's (pass: at most ``BOUND``)."""
    ball_worlds: int = 0
    """Worlds whose speed ball acted: their ``du`` is a law's, not ``sum Z lambda``, so their leave is not compared."""
    operators: tuple = ()


def _seed(walk: TreeWalk, x: np.ndarray):
    """The body seed of a row ``x``: the deepest body it reaches, the wrench there nearest to writing it and how near
    (``|J(w) - x| / |x|``), or ``None`` where no wrench at one body writes it (a joint's covector; a row of two bodies of
    one tree)."""
    touched = np.flatnonzero(x != 0.0)
    if not touched.size:
        return None
    body = int(np.searchsorted(walk.first, touched[-1], side="right") - 1)
    G = jacobian(walk, body, np.eye(6))
    path = np.flatnonzero(np.any(G != 0.0, axis=1))
    if np.any(x[np.setdiff1d(np.arange(x.size), path)] != 0.0):
        return None
    w, *_ = np.linalg.lstsq(G[path], x[path], rcond=None)
    near = np.linalg.norm(G @ w - x) / np.linalg.norm(x)
    if near > 1.0e-5:
        return None
    return body, w, near


def check_rows(solver, joint_q: np.ndarray, capture: Capture, rng) -> RowCheck:
    """F1 on one run: every dense row's sides and every free-body row, then the leave of the step's ``du``."""
    out = RowCheck(operators=tuple(getattr(group.sparse, "kind", "scalar") for group in solver.factor.held.groups))
    forest, placement = solver.forest, solver.placement
    sizes = forest.articulation_response_dof_count.numpy()
    offsets = forest.articulation_world_dof_offset.numpy()
    starts = forest.articulation_dof_start.numpy()
    art_world = forest.art_to_world.numpy()
    dense = bool(placement.packed_dense)
    walks = {}

    def walk_of(art):
        if art not in walks:
            walk = solver.factor.walk(joint_q, art)
            kappa = float(np.linalg.cond(factor_matrix(walk)))
            walks[art] = (walk, solver.factor.walk(joint_q, art, dtype=np.float32), kappa)
            out.kappa = max(out.kappa, kappa)
        return walks[art]

    def compare(art, x, z_step, seeds):
        walk, walk32, kappa = walk_of(art)
        z64, _ = walk.up(covector=x)
        norm = np.linalg.norm(z64)
        out.sides += 1
        if norm == 0.0:
            out.forward_step = max(out.forward_step, float(np.abs(z_step).max()))
            return
        z32, _ = walk32.up(covector=x.astype(np.float32))
        forward = np.linalg.norm(z_step - z64) / norm
        out.forward_step = max(out.forward_step, forward)
        out.forward_walk32 = max(out.forward_walk32, np.linalg.norm(z32 - z64) / norm)
        out.ratio = max(out.ratio, forward / (x.size * ROUNDING * kappa))
        seed = _seed(walk, x)
        if seed is not None:
            body, w, near = seed
            out.seeded += 1
            out.writer = max(out.writer, near)
            z_seed, _ = walk.up(body, w)
            written, _ = walk.up(covector=jacobian(walk, body, w))
            out.seed = max(out.seed, np.linalg.norm(z_seed - written) / np.linalg.norm(written))
            seeds.setdefault(art, []).append((body, w, z_seed))

    # The walked free-body rows form no J (walk_rows): test_feather_pgs_walk_rows holds them to their J v_hat.
    # The leave: the step's du is sum Z lambda over its final rows; each articulation leaves it once.
    D = int(forest.max_world_dofs)
    du = np.zeros((int(forest.world_count), max(D, 1)))
    # Every stored row's footprints, the one writer's (FL3.3, whose rows form no J).
    if getattr(solver.placement, "walks", False):
        for world, count in enumerate(capture.counts):
            line = int(capture.dense_first[world])
            for row in range(int(count)):
                lam = capture.impulses[line, row]
                if dense:
                    du[world, :D] += capture.Z_final[line, row, :D] * lam
                    continue
                first = 0
                for art in capture.sides[line, row]:
                    if art < 0:
                        continue
                    n, offset = int(sizes[art]), int(offsets[art])
                    du[world, offset : offset + n] += capture.Z_final[line, row, first : first + n] * lam
                    first += n
    ball = np.zeros(du.shape[0], dtype=bool)
    if capture.mf_final is not None:
        Z_a, Z_b, lam, counts, _, _, dof_a, dof_b, kinds = capture.mf_final
        for world, count in enumerate(counts):
            line = int(capture.mf_first[world])
            for row in range(int(count)):
                if kinds[line, row] == PGS_CONSTRAINT_TYPE_SPEED:
                    # A free body's speed row is its speed ball's: a magnitude against the body's velocity as each
                    # visit finds it, a law and not a row, so its world's du is not sum Z lambda.
                    ball[world] |= lam[line, row] != 0.0
                    continue
                for dof, Z in ((dof_a, Z_a), (dof_b, Z_b)):
                    if dof[line, row] >= 0:
                        du[world, dof[line, row] : dof[line, row] + 6] += Z[line, row] * lam[line, row]
    out.ball_worlds = int(np.count_nonzero(ball))
    for art in range(solver.model.articulation_count):
        n, world = int(sizes[art]), int(art_world[art])
        if n <= 0 or world < 0 or ball[world] or not np.any(du[world, offsets[art] : offsets[art] + n]):
            continue
        walk, _, kappa = walk_of(art)
        dv, _ = walk.down(du[world, offsets[art] : offsets[art] + n])
        dofs = int(starts[art]) + np.arange(n)
        step = capture.v_out[dofs] - capture.v_hat[dofs]
        norm = np.linalg.norm(dv)
        relative_error = np.linalg.norm(dv - step) / norm
        published = 2.0 * ROUNDING * (np.linalg.norm(capture.v_out[dofs]) + np.linalg.norm(capture.v_hat[dofs])) / norm
        out.leave = max(out.leave, relative_error)
        out.leave_ratio = max(out.leave_ratio, relative_error / (n * ROUNDING * kappa + published))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# F2, F2b: long chains in float32
# ---------------------------------------------------------------------------------------------------------------------


def ground_contact(model, poses: np.ndarray, joint: int) -> tuple[int, np.ndarray]:
    """The ground's contact under the last sphere of ``joint``'s child (``long_chain``'s links and head): the walk's
    body and the wrenches of a normal and two tangents at its frame."""
    body = int(model.joint_child.numpy()[joint])
    shape = int(np.flatnonzero(model.shape_body.numpy() == body)[-1])
    centre = transform_point(poses[body], model.shape_transform.numpy()[shape, :3].astype(np.float64))
    point = centre - np.array([0.0, 0.0, float(model.shape_scale.numpy()[shape, 0])])
    parent = int(model.joint_parent.numpy()[joint])
    if int(model.joint_type.numpy()[joint]) == int(newton.JointType.FREE):
        frame = transform_point(poses[body], model.body_com.numpy()[body].astype(np.float64))
    else:
        anchor = model.joint_X_p.numpy()[joint, :3].astype(np.float64)
        frame = transform_point(poses[parent], anchor) if parent >= 0 else anchor
    return joint - int(model.articulation_start.numpy()[0]), contact_wrenches(point - frame)


def newton_state_body_q(model, joint_q: np.ndarray) -> np.ndarray:
    """The body poses at ``joint_q`` (Newton's forward kinematics, float32, as float64)."""
    state = model.state()
    state.joint_q.assign(np.asarray(joint_q, dtype=np.float32))
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return state.body_q.numpy().astype(np.float64)


def transform_point(pose: np.ndarray, point: np.ndarray) -> np.ndarray:
    x, y, z, w = pose[3:]
    q = np.array([x, y, z])
    t = 2.0 * np.cross(q, point)
    return pose[:3] + point + w * t + np.cross(q, t)


@dataclasses.dataclass
class Chain:
    """F2 at one length: errors against the float64 truth (no armature) of a tip row's ``Z``, a leave and the tip
    impulse's response ``dv = H^-1 J^T``."""

    links: int
    coordinates: int
    condition: float
    """``cond(H)``, float64."""
    truth_floor: float
    """The float64 walk against the float64 Cholesky of the independent mass matrix (the truth's own spread)."""
    main_factor: float
    """Main's float32 factor's backward error ``|L L^T - P (H + R + K) P^T| / |H + R + K|``."""
    main: dict
    """Main's float32 factor, applied in float32 (and ``"dv64"``: applied in float64)."""
    walk32: dict
    walk32_poses: dict
    """The float32 walk whose local frames come from Newton's float32 world poses, main's forward kinematics."""
    walk32_pivots_positive: bool
    walk32_pivots: float
    """The float32 walk's worst pivot ``ell^2`` against the truth's, relative."""
    main_finite: bool


def chain_precision(links: int, rng) -> Chain:
    """Main's float32 dense factor of ``long_chain(links, armature=0)``, no armature added, against the
    float32 walk in local frames with none, both against the float64 walk with none (the truth)."""
    scene = scenes.long_chain(links=links, armature=0.0)
    model = scene.build("cpu")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver = SolverFeatherPGS(model, dense_max_constraints=8)
    state, out, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    joint_q = state.joint_q.numpy().astype(np.float64)
    solver.step(state, out, control, None, scene.dt)
    n = links + 6
    H = MassReference(model).mass_matrix(joint_q, 0)
    H32 = MassReference(model, np.float32).mass_matrix(joint_q, 0).astype(np.float64)
    L32 = dense_factor32(H32 + np.diag(solver.factor.diagonal(0)))
    truth = TreeWalk(*local_frames(model, joint_q, 0), np.zeros(n))
    walk32 = TreeWalk(*local_frames(model, joint_q, 0), np.zeros(n), dtype=np.float32)
    body, W = ground_contact(model, newton_state_body_q(model, joint_q), links)
    x = jacobian(truth, body, W)
    z64, _ = truth.up(body, W)
    dv64, _ = truth.down(z64)
    u = rng.normal(size=n)
    v64, _ = truth.down(u)
    floor = relative(dv64, np.linalg.solve(H, x))
    A = (H + np.diag(solver.factor.diagonal(0)))[::-1, ::-1]
    L64 = L32.astype(np.float64)

    from scipy.linalg import solve_triangular

    def solve(lower, rhs, transpose=False):
        return solve_triangular(lower, rhs, lower=True, trans=1 if transpose else 0, check_finite=False)

    with np.errstate(all="ignore"):
        x32 = x.astype(np.float32)[::-1]
        z_main = solve(L32, x32)
        main = {
            "Z": relative(z_main[::-1], z64),
            "leave": relative(solve(L32, u.astype(np.float32)[::-1], True)[::-1], v64),
            "dv": relative(solve(L32, z_main, True)[::-1], dv64),
            "dv64": relative(solve(L64, solve(L64, x[::-1]), True)[::-1], dv64),
        }

        def errors(walk):
            z32, _ = walk.up(body, W.astype(np.float32))
            return {
                "Z": relative(z32, z64),
                "leave": relative(walk.down(u.astype(np.float32))[0], v64),
                "dv": relative(walk.down(z32)[0], dv64),
            }

        mine = errors(walk32)
        poses = newton_state_body_q(model, joint_q)
        from_poses = errors(TreeWalk(*local_frames(model, joint_q, 0, poses), np.zeros(n), dtype=np.float32))
    return Chain(
        links=links,
        coordinates=n,
        condition=float(np.linalg.cond(H)),
        truth_floor=floor,
        main_factor=float(np.abs(L64 @ L64.T - A).max() / np.abs(A).max()) if np.all(np.isfinite(L64)) else np.inf,
        main=main,
        walk32=mine,
        walk32_poses=from_poses,
        walk32_pivots_positive=bool(np.all(np.isfinite(walk32.ell)) and np.all(walk32.ell > 0.0)),
        walk32_pivots=float(np.max(np.abs(walk32.pivots.astype(np.float64) / truth.pivots - 1.0))),
        main_finite=bool(np.all(np.isfinite(L32))),
    )


# ---------------------------------------------------------------------------------------------------------------------
# F4, F5: row entries per seed in the segmented and the merged frames
# ---------------------------------------------------------------------------------------------------------------------


def segments(walk: TreeWalk, size: int = SEGMENT) -> list:
    """A chain's coordinates cut at bodies every ``size`` coordinates from the root: ``(coordinates, last body)``."""
    cuts, coordinates, last = [], [], -1
    for b in range(walk.parent.size):
        own = list(range(walk.first[b], walk.first[b + 1]))
        if coordinates and len(coordinates) + len(own) > size:
            cuts.append((coordinates, last))
            coordinates = []
        coordinates += own
        last = b
    if coordinates:
        cuts.append((coordinates, last))
    return cuts


def segmented_frame(walk: TreeWalk, size: int = SEGMENT) -> np.ndarray:
    """The orthonormal frame that cuts the chain every ``size`` coordinates (Z_UNION §2.3c): each segment rotated by the
    complete QR of the footprints of its last body's six unit wrenches, interface first."""
    n = walk.s.shape[0]
    O = np.zeros((n, n))
    for coordinates, last in segments(walk, size):
        gamma, _ = walk.up(last, np.eye(6))
        Q, _ = np.linalg.qr(gamma[coordinates].astype(np.float64), mode="complete")
        O[np.ix_(coordinates, coordinates)] = Q
    return O


def entries(Z: np.ndarray, threshold: float) -> np.ndarray:
    """Each row's entries above ``threshold`` of its largest (rows are columns of ``Z``)."""
    return np.sum(np.abs(Z) > threshold * np.abs(Z).max(axis=0), axis=0)


@dataclasses.dataclass
class Counts:
    """F4 and F5 on ``long_chain(links)``."""

    links: int
    tip_rows: int
    tip_static: dict
    """Tip rows' entries in the frame cut every 32 coordinates, ``{threshold: (mean, max)}``."""
    structure: int
    """Rows with more entries than the frame allows (their own segment, then six per segment above): none."""
    tip_merged: tuple
    """Tip rows' entries with the seed-free stretch merged: ``(mean, max)``, the window and the stretch's rank."""
    tip_residue: float
    """A tip row's footprints above the merged stretch against ``Gamma_c`` times its carried state (two walks of one
    push): the walk's own rounding, below which entries are residue."""
    tip_gram: float
    ground_rows: int
    ground_main: tuple
    """Main's entries per ground row: ``(mean support, stored width)``."""
    ground_segmented: tuple
    """Segmented entries per ground row (1e-12): ``(mean, max)``."""
    ground_gram: float


def counts(links: int = 300) -> Counts:
    scene = scenes.long_chain(links=links)
    model = scene.build("cpu")
    joint_q = model.joint_q.numpy().astype(np.float64)
    walk = TreeWalk(*local_frames(model, joint_q, 0), model.joint_armature.numpy()[: links + 6])
    poses = newton_state_body_q(model, joint_q)

    def rows(joints):
        seeds = [ground_contact(model, poses, j) for j in joints]
        Z = np.concatenate([walk.up(b, W)[0] for b, W in seeds], axis=1)
        return seeds, Z

    O = segmented_frame(walk)
    cuts = segments(walk)
    segment_of = np.zeros(walk.parent.size, dtype=np.int64)
    for index, (coordinates, _) in enumerate(cuts):
        segment_of[np.searchsorted(walk.first, coordinates, side="right") - 1] = index

    def beyond(seeds, Zs):
        """Rows whose entries pass their own segment's coordinates plus six for each segment above it."""
        allowed = np.repeat([len(cuts[segment_of[b]][0]) + 6 * segment_of[b] for b, _ in seeds], 3)
        return int(np.count_nonzero(entries(Zs, 1e-12) > allowed))

    last = walk.parent.size - 1
    tip_seeds, Z_tip = rows([last - 1, last])
    tip_static = {
        t: (float(entries(O.T @ Z_tip, t).mean()), int(entries(O.T @ Z_tip, t).max())) for t in (1e-12, 1e-10)
    }
    # Merged: the window below the shallowest seeded body's parent c, plus the six numbers that cross c.
    c = min(b for b, _ in tip_seeds) - 1
    stretch = [k for b in walk.ancestors(c) for k in range(walk.first[b], walk.first[b + 1])]
    window = np.setdiff1d(np.arange(walk.s.shape[0]), stretch)
    gamma, _ = walk.up(c, np.eye(6))
    _, R = np.linalg.qr(gamma[stretch])
    merged = np.concatenate(
        [np.concatenate((walk.up(b, W)[0][window], R @ walk.carry(b, W, c)), axis=0) for b, W in tip_seeds], axis=1
    )
    merged_counts = entries(merged, 1e-12)
    rank = int(np.linalg.matrix_rank(gamma[stretch], tol=1e-10 * np.abs(gamma[stretch]).max()))
    b, W = tip_seeds[0]
    residue = relative(walk.up(b, W)[0][stretch], gamma[stretch] @ walk.carry(b, W, c))
    gram = np.abs(merged.T @ merged - Z_tip.T @ Z_tip).max() / np.abs(Z_tip.T @ Z_tip).max()
    # F5: a contact under every link's sphere and the head's.
    ground, Z_ground = rows(range(walk.parent.size))
    support = np.array(
        [sum(walk.first[a + 1] - walk.first[a] for a in walk.ancestors(b)) for b, W in ground for _ in range(3)]
    )
    Zs = O.T @ Z_ground
    seg = entries(Zs, 1e-12)
    return Counts(
        links=links,
        tip_rows=Z_tip.shape[1],
        tip_static=tip_static,
        structure=beyond(tip_seeds, O.T @ Z_tip) + beyond(ground, Zs),
        tip_merged=(float(merged_counts.mean()), int(merged_counts.max()), int(window.size), rank),
        tip_residue=residue,
        tip_gram=float(gram),
        ground_rows=Z_ground.shape[1],
        ground_main=(float(support.mean()), walk.s.shape[0]),
        ground_segmented=(float(seg.mean()), int(seg.max())),
        ground_gram=float(np.abs(Zs.T @ Zs - Z_ground.T @ Z_ground).max() / np.abs(Z_ground.T @ Z_ground).max()),
    )


# ---------------------------------------------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------------------------------------------


class TestWalkIdentities(unittest.TestCase):
    """The walk against the independent float64 factor, on every catalogue model at its first pose."""

    def test_catalogue_models(self):
        rng = np.random.default_rng(1101)
        seen = set()
        for physics, _ in catalogue():
            if physics.scene.name in seen:
                continue
            seen.add(physics.scene.name)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = SolverFeatherPGS(physics.scene.build("cpu")).model  # loops cut at a body
            with self.subTest(scene=physics.scene.name):
                worst = identities(model, model.joint_q.numpy(), rng)
                for name, value in worst.items():
                    self.assertLess(value, 1.0e-9, f"{name}")


_construct = Placement.__init__


def _reversed_lines(self, *args, **kwargs):
    """A placement whose row tables put every world's rows on another world's line
    (:mod:`.test_feather_pgs_row_lines`)."""
    _construct(self, *args, **kwargs)
    lines = self.dense_first.shape[0]
    self.dense_first.assign(np.arange(lines - 1, -1, -1, dtype=np.int32))


class TestWalkF1(unittest.TestCase):
    """F1: the walk's rows and leave against the step's, on every catalogue problem and both CPU routes."""

    def check(self, result):
        self.assertLessEqual(result.ratio, BOUND, f"Z: {result}")
        self.assertLess(result.seed, 1.0e-9, f"seed: {result}")
        self.assertLess(result.meet, 1.0e-9, f"meet: {result}")
        self.assertLessEqual(result.leave_ratio, BOUND, f"leave: {result}")

    def test_rows_read_through_their_tables(self):
        """F1 with every row table reversed: the identity tables would hide a read that takes its line from the world,
        and a world's leave would then sum another world's rows."""
        for physics, route in catalogue():
            if physics.name not in REVERSED or route.name != "cpu":
                continue
            with self.subTest(problem=physics.name):
                with mock.patch.object(Placement, "__init__", _reversed_lines):
                    solver, joint_q, capture = run_route(physics, route)
                worlds = capture.dense_first.size
                self.assertGreater(worlds, 1)
                np.testing.assert_array_equal(capture.dense_first, np.arange(worlds)[::-1])
                self.check(check_rows(solver, joint_q, capture, np.random.default_rng(7)))


def _add_f1_tests():
    for physics, route in catalogue():

        def test(self, physics=physics, route=route):
            solver, joint_q, capture = run_route(physics, route)
            self.check(check_rows(solver, joint_q, capture, np.random.default_rng(7)))

        test.__doc__ = f"F1 on {physics.name} through {route.name}."
        setattr(TestWalkF1, f"test_{physics.name}_{route.name}", test)


_add_f1_tests()


class TestWalkF2(unittest.TestCase):
    """F2 on ``long_chain(N)`` without armature."""

    def test_f2_float32_walk_needs_no_armature(self):
        """The float32 walk builds every chain without armature, and from 32 links it answers a tip impulse at least
        twice as close to the float64 truth as main's float32 dense factor."""
        rng = np.random.default_rng(3)
        for links in CHAINS:
            with self.subTest(links=links):
                chain = chain_precision(links, rng)
                self.assertTrue(chain.walk32_pivots_positive)
                self.assertLess(chain.walk32["dv"], 1.0e-5)
                if links >= 32:
                    # A dense factor that failed (NaN) is not closer either.
                    self.assertFalse(chain.main["dv"] <= 2.0 * chain.walk32["dv"], f"{chain.main}")


class TestWalkCounts(unittest.TestCase):
    """F4 and F5 on ``long_chain(300)``."""

    def test_rows_per_seed(self):
        """Every row keeps the frame's structure; merged, a tip row has at most 10 entries; the chain on the ground
        has at most 0.45 of main's entries per row."""
        result = counts(300)
        self.assertEqual(result.structure, 0)
        self.assertLessEqual(result.tip_merged[1], 10)
        self.assertLessEqual(result.ground_segmented[0], 0.45 * result.ground_main[0])
        self.assertLess(result.tip_gram, 1.0e-12)
        self.assertLess(result.ground_gram, 1.0e-12)


def report_walk() -> None:
    """Print every falsifier's measured table."""
    rng = np.random.default_rng(3)
    print("F2  long_chain(N), armature 0: errors against the float64 walk without armature (the truth); dv = H^-1 J^T")
    print(
        "     N   cond(H) | main factor | Z: main / walk32 | leave: main / "
        "walk32 | dv: main (f64 solves) / walk32  ratio / walk32 on float32 poses | walk32 pivots"
    )
    for links in CHAINS:
        c = chain_precision(links, rng)
        ratio = c.main["dv"] / c.walk32["dv"] if c.walk32["dv"] > 0 else float("nan")
        print(
            f"  {c.links:5d}  {c.condition:8.1e} | {c.main_factor:8.1e} | {c.main['Z']:8.1e} / {c.walk32['Z']:8.1e} | "
            f"{c.main['leave']:8.1e} / {c.walk32['leave']:8.1e} | {c.main['dv']:8.1e} ({c.main['dv64']:8.1e}) / "
            f"{c.walk32['dv']:8.1e} {ratio:8.1f} / {c.walk32_poses['dv']:8.1e} | positive {c.walk32_pivots_positive} "
            f"{c.walk32_pivots:8.1e} (truth floor {c.truth_floor:.0e})",
            flush=True,
        )
    c = counts(300)
    print(
        f"F4  tip of long_chain(300), {c.tip_rows} rows: static s=32 (mean, max) by threshold {c.tip_static}; rows past "
        f"the frame's structure {c.structure}; merged (mean, max, window, rank) {c.tip_merged}; residue "
        f"{c.tip_residue:.1e}; Gram {c.tip_gram:.1e}"
    )
    print(
        f"F5  long_chain(300) on the ground, {c.ground_rows} rows: main support mean {c.ground_main[0]:.1f} "
        f"(stored {c.ground_main[1]}); segmented mean {c.ground_segmented[0]:.1f}, max {c.ground_segmented[1]}; "
        f"ratio {c.ground_segmented[0] / c.ground_main[0]:.3f}; Gram {c.ground_gram:.1e}",
        flush=True,
    )
    print(f"F1  catalogue (Z and leave against their rounding n u kappa: pass at <= {BOUND:g})")
    for physics, route in catalogue():
        solver, joint_q, capture = run_route(physics, route)
        r = check_rows(solver, joint_q, capture, np.random.default_rng(7))
        print(
            f"  {physics.name:44s} {route.name:16s} rows {r.rows:4d} sides {r.sides:4d} seeded {r.seeded:4d} "
            f"free {r.free_rows:3d} kappa {r.kappa:8.1e} | Z: step {r.forward_step:8.1e} walk32 {r.forward_walk32:8.1e} "
            f"ratio {r.ratio:6.2f} | writer {r.writer:7.1e} seed {r.seed:7.1e} meet {r.meet:7.1e} | leave {r.leave:8.1e} "
            f"ratio {r.leave_ratio:6.2f} ball {r.ball_worlds} | {','.join(sorted(set(r.operators)))}",
            flush=True,
        )


# =====================================================================================================================
# The walk on the device
# =====================================================================================================================

UNITS = (1, 2, 3, 4)
"""Rows a lane walks: a limit, a velocity limit's pair, a point contact, a contact with spin."""
UNSTORED = ("sparse", "sparse_triplets", "keyed_parallel")
"""The CUDA catalogue's routes whose rows no store holds as ``J``, packed ``Z`` or the walk's slots: native sparse
factors (rows in factor coordinates, whitened from a ``J`` in shared scratch) and the keyed rows."""


def devices() -> list:
    return ["cpu", *(["cuda:0"] if wp.is_cuda_available() else [])]


# ---------------------------------------------------------------------------------------------------------------------
# The device walk of one topology
# ---------------------------------------------------------------------------------------------------------------------

_UP = """    if (tid >= body.shape[0]) return;
    const int n = z.shape[2], bodies = link.shape[1];
    const float* d = data.data + (size_t)tree.data[tid] * data.shape[1] * $coordinate;
    const float* l = link.data + (size_t)tree.data[tid] * bodies * $link;
    const float* s = now.data + (size_t)tree.data[tid] * now.shape[1] * $now;
    const float* f = now_link.data + (size_t)tree.data[tid] * bodies * $link;
    float* out = z.data + (size_t)tid * $rows * n;
    float* entries = jz.data + (size_t)tid * $rows * n;
    const auto sink = [&](int r, int c, float value, float j) {
        out[(size_t)r * n + c] = value;
        entries[(size_t)r * n + c] = j;
    };
    float p[6 * $rows], w[6 * $rows];
    for (int k = 0; k < 6 * $rows; ++k) p[k] = w[k] = wrench.data[(size_t)tid * 6 * $rows + k];
    int start = top.data[tid];
    if (seeded.data[tid] != 0) {
        float x[$rows];
        for (int r = 0; r < $rows; ++r) x[r] = covector.data[(size_t)tid * $rows + r];
        WALK_SEED(d, start, x, p, sink);
        start -= 1;
        s = nullptr;
    }
    // The fixture's frames are one geometry's: no origin shift between the factor's and the step's.
    const float shift[3] = {0.0f, 0.0f, 0.0f};
    WALK_UP(d, l, s, f, shift, body.data[tid], start, p, w, sink);
"""

_TREE = """    if (tid >= u.shape[0]) return;
    const int n = u.shape[1], bodies = link.shape[1];
    const float* d = data.data + (size_t)tree.data[tid] * data.shape[1] * $coordinate;
    const float* l = link.data + (size_t)tree.data[tid] * bodies * $link;
    const float* in = u.data + (size_t)tid * n;
    float* out = qd.data + (size_t)tid * n;
    float* twists = twist.data + (size_t)tid * bodies * 6;
    WALK_DOWN(
        0, d, l, [&](const int c) -> float { return in[c]; }, [](const int) -> float { return 0.0f; },
        [&](const int c, const float q, const float) { out[c] = q; },
        [&](const int b, const float* V) {
            for (int k = 0; k < 6; ++k) twists[6 * b + k] = V[k];
        });
"""

_UP_PARAMS = params(
    ("data", wp.array3d[float]),
    ("link", wp.array3d[float]),
    ("now", wp.array3d[float]),
    ("now_link", wp.array3d[float]),
    ("tree", wp.array[int]),
    ("body", wp.array[int]),
    ("top", wp.array[int]),
    ("seeded", wp.array[int]),
    ("wrench", wp.array3d[float]),
    ("covector", wp.array2d[float]),
    ("z", wp.array3d[float]),
    ("jz", wp.array3d[float]),
)
_TREE_PARAMS = params(
    ("data", wp.array3d[float]),
    ("link", wp.array3d[float]),
    ("tree", wp.array[int]),
    ("u", wp.array2d[float]),
    ("qd", wp.array2d[float]),
    ("twist", wp.array3d[float]),
)


@functools.cache
def walk_kernels(table: tuple, rows: int) -> tuple:
    """The rows kernel (a thread per seed) and the tree kernel (a thread per tree, the class's emitted walk down) of
    one path table."""
    prelude = factor_walk.walk_prelude((table,), rows)
    tag = hashlib.sha256(repr(table).encode()).hexdigest()[:12]
    sizes = {"coordinate": WALK_COORDINATE, "link": WALK_LINK, "now": WALK_NOW}
    up = KernelTemplate(
        f"walk_up_{rows}_{tag}",
        _UP_PARAMS,
        substitute(_UP, rows=rows, **sizes),
        items("tid", flat=True),
        (prelude,),
        dual=True,
    )
    tree = KernelTemplate(
        f"walk_tree_{tag}",
        _TREE_PARAMS,
        substitute(_TREE, coordinate=WALK_COORDINATE, link=WALK_LINK),
        items("tid", flat=True),
        (factor_walk.walk_down_prelude((ts.WalkClass(*table),)),),
        dual=True,
    )
    return up.kernel(), tree.kernel()


def random_tree(rng, bodies: int, device="cpu"):
    """A random tree of ``bodies`` links (revolute, prismatic and ball joints) with a drive on every scalar coordinate,
    no gravity: its model, built on ``device``."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    links, joints = [], []
    for b in range(bodies):
        parent = -1 if b == 0 else links[int(rng.integers(0, b))]
        inertia = np.diag(rng.uniform(0.02, 0.2, 3))
        link = builder.add_link(mass=float(rng.uniform(0.5, 2.0)), inertia=wp.mat33(inertia))
        offset = wp.transform(wp.vec3(*rng.normal(size=3) * 0.2), wp.quat_identity())
        axis = tuple(np.eye(3)[int(rng.integers(0, 3))])
        kind = int(rng.integers(0, 3)) if b > 0 else 0
        if kind == 0:
            joint = builder.add_joint_revolute(parent=parent, child=link, axis=axis, parent_xform=offset)
        elif kind == 1:
            joint = builder.add_joint_prismatic(parent=parent, child=link, axis=axis, parent_xform=offset)
        else:
            joint = builder.add_joint_ball(parent=parent, child=link, parent_xform=offset)
        links.append(link)
        joints.append(joint)
    builder.add_articulation(joints)
    model = builder.finalize(device=device)
    q = model.joint_q.numpy()
    q[:] = rng.normal(size=q.shape) * 0.3
    # Ball joints hold unit quaternions.
    types, q_start = model.joint_type.numpy(), model.joint_q_start.numpy()
    for j in np.flatnonzero(types == int(newton.JointType.BALL)):
        quat = rng.normal(size=4)
        q[q_start[j] : q_start[j] + 4] = quat / np.linalg.norm(quat)
    return model, q


def frame_points(walk: TreeWalk) -> np.ndarray:
    """Each body's frame point of ``walk``, its root's at the origin: the parent's less the body's offset."""
    frames = np.zeros((walk.parent.size, 3))
    for b in range(walk.parent.size):
        if walk.parent[b] >= 0:
            frames[b] = frames[walk.parent[b]] - np.asarray(walk.offset[b], dtype=np.float64)
    return frames


class DeviceWalk:
    """The device walk of trees of one topology (one path table), their data from their float32 walks."""

    def __init__(self, walks: list, device, rows: int = 3, step: list | None = None, frames: list | None = None):
        self.walks, self.device, self.rows = walks, device, rows
        self.n, self.bodies = walks[0].s.shape[0], walks[0].parent.size
        self.up_kernel, self.tree_kernel = walk_kernels(walks[0].path_table, rows)
        data, link = zip(*(walk.device_data() for walk in walks), strict=True)
        link = np.stack(link)
        # Each body's frame point (the trees' roots at the origin), at the refresh's poses: the link's frame.
        self.frames = np.stack(frames if frames is not None else [frame_points(walk) for walk in walks])
        link[:, :, WALK_FRAME : WALK_FRAME + 3] = self.frames
        self.data = wp.array(np.stack(data), dtype=float, device=device)
        self.link = wp.array(link, dtype=float, device=device)
        # The step's geometry (WalkTrees.now, .now_link): the columns at each body's frame and the links of the walks'
        # own poses, or of ``step``'s walks, the same trees at a step's poses after the refresh's (its frames at the
        # step's).
        step = walks if step is None else step
        self.step_frames = np.stack([frame_points(walk) for walk in step])
        now_link = np.stack([walk.device_data()[1] for walk in step])
        now_link[:, :, WALK_FRAME : WALK_FRAME + 3] = self.step_frames
        self.now = wp.array(
            np.stack([np.asarray(walk.s, dtype=np.float32) for walk in step]), dtype=float, device=device
        )
        self.now_link = wp.array(now_link, dtype=float, device=device)

    def up(self, tree, body, top, seeded, wrench, covector, entries: bool = False):
        """Footprints ``[seeds, rows, n]`` of seeds of ``rows`` rows: wrenches ``[seeds, rows, 6]`` at their bodies'
        frames from coordinate ``top``, plus covectors ``[seeds, rows]`` at ``top`` where ``seeded``; with ``entries``,
        also the step's ``J`` entries the sink received beside each footprint."""
        count = len(body)
        # The wrenches at the bodies' frames at the step's poses, moved to their frames in the walk's links.
        wrench = np.array(np.reshape(wrench, (count, self.rows, 6)), dtype=np.float64)
        for i in range(count):
            moved = self.step_frames[tree[i], body[i]] - self.frames[tree[i], body[i]]
            wrench[i, :, 3:6] += np.cross(moved, wrench[i, :, 0:3])
        z = wp.zeros((count, self.rows, self.n), dtype=float, device=self.device)
        jz = wp.zeros((count, self.rows, self.n), dtype=float, device=self.device)

        def array(values, dtype):
            return wp.array(
                np.asarray(values, dtype=np.float32 if dtype is float else np.int32), dtype=dtype, device=self.device
            )

        wp.launch(
            self.up_kernel,
            dim=count,
            inputs=[
                self.data,
                self.link,
                self.now,
                self.now_link,
                array(tree, int),
                array(body, int),
                array(top, int),
                array(seeded, int),
                array(np.reshape(wrench, (count, self.rows, 6)), float),
                array(np.reshape(covector, (count, self.rows)), float),
                z,
                jz,
            ],
            device=self.device,
        )
        if entries:
            return z.numpy().astype(np.float64), jz.numpy().astype(np.float64)
        return z.numpy().astype(np.float64)

    def down(self, tree, u) -> tuple:
        """The leave of ``u`` ``[cases, n]``: joint velocities and body twists, a thread per case of tree ``tree``."""
        count = len(tree)
        outputs = (
            wp.zeros((count, self.n), dtype=float, device=self.device),
            wp.zeros((count, self.bodies, 6), dtype=float, device=self.device),
        )
        wp.launch(
            self.tree_kernel,
            dim=count,
            inputs=[
                self.data,
                self.link,
                wp.array(np.asarray(tree, dtype=np.int32), dtype=int, device=self.device),
                wp.array(np.asarray(u, dtype=np.float32), dtype=float, device=self.device),
                outputs[0],
                outputs[1],
            ],
            device=self.device,
        )
        return tuple(array.numpy().astype(np.float64) for array in outputs)


def held_step(device) -> dict:
    """A step between refreshes (``update_mass_matrix_interval > 1``) on random trees of revolute, prismatic and ball
    joints: the device walk with the refresh's factor (``q_r``) and the step's geometry (``q_t``, every coordinate moved),
    its contact rows seeded at the step's frames, against ``L_r^-1 P J_t^T`` in float64 (the dense rows' whitening by the
    held factor), as worst ratios to ``n u kappa``; ``stale`` the same rows whitened with the refresh pose's ``J``, what
    the walk gave before it took the step's geometry (a root body's rows are the same in both)."""
    rng = np.random.default_rng(1102)
    out = {"held": 0.0, "stale": 0.0, "seeds": 0}
    for bodies in (3, 6, 9, 12):
        model, q_r = random_tree(rng, bodies)
        q_t = q_r + rng.normal(size=q_r.shape) * 0.2
        types, q_start = model.joint_type.numpy(), model.joint_q_start.numpy()
        for j in np.flatnonzero(types == int(newton.JointType.BALL)):
            quat = q_t[q_start[j] : q_start[j] + 4]
            q_t[q_start[j] : q_start[j] + 4] = quat / np.linalg.norm(quat)
        n = int(model.joint_dof_count)
        diagonal = rng.uniform(0.0, 0.05, n)
        walk_r = TreeWalk(*local_frames(model, q_r, 0), diagonal)
        walk_t = TreeWalk(*local_frames(model, q_t, 0), diagonal)
        walk32 = TreeWalk(*local_frames(model, q_r, 0), diagonal, dtype=np.float32)
        kappa = float(np.linalg.cond(factor_matrix(walk_r)))
        seeds = [(b, contact_wrenches(rng.normal(scale=0.05, size=3), rng.normal(size=3))) for b in range(bodies)]
        z = DeviceWalk([walk32], device, 3, step=[walk_t]).up(
            [0] * bodies,
            [b for b, _ in seeds],
            [int(walk_t.first[b + 1]) - 1 for b, _ in seeds],
            [0] * bodies,
            [w.T for _, w in seeds],
            np.zeros((bodies, 3)),
        )
        for i, (b, w) in enumerate(seeds):
            held, _ = walk_r.up(covector=jacobian(walk_t, b, w))
            stale, _ = walk_r.up(covector=jacobian(walk_r, b, w))
            scale = n * ROUNDING * kappa
            out["held"] = max(out["held"], relative(z[i].T, held) / scale)
            out["stale"] = max(out["stale"], relative(stale, held) / scale)
        out["seeds"] += bodies
    return out


def seed_rows(walk: TreeWalk, rows: int, rng) -> list:
    """Seeds of every unit a lane walks: at every body, ``rows`` wrenches (a contact's normal and tangents at a random
    point, then random); at every coordinate, ``rows`` covectors. Each is ``(body, top, seeded, wrench, covector)``."""
    seeds = []
    for b in range(walk.parent.size):
        top = int(walk.first[b + 1]) - 1
        wrench = contact_wrenches(rng.normal(scale=0.05, size=3), rng.normal(size=3))
        wrench = np.concatenate((wrench, rng.normal(size=(6, max(rows - 3, 0)))), axis=1)[:, :rows]
        seeds.append((b, top, 0, wrench.T, np.zeros(rows)))
        for c in range(int(walk.first[b]), int(walk.first[b + 1])):
            seeds.append((b, c, 1, np.zeros((rows, 6)), rng.choice((-1.0, 1.0), rows) * rng.uniform(0.5, 2.0, rows)))
    return seeds


@functools.cache
def catalogue_walks() -> tuple:
    """Every catalogue model once (loops cut at a body), at its first pose: per articulation, the float64 walk of its
    factor, the float32 one and the factor's condition number."""
    out, seen = [], set()
    for physics, _ in catalogue():
        if physics.scene.name in seen:
            continue
        seen.add(physics.scene.name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            solver = SolverFeatherPGS(physics.scene.build("cpu"))
        joint_q = solver.model.joint_q.numpy()
        for art in range(solver.model.articulation_count):
            walk = solver.factor.walk(joint_q, art)
            if walk.s.shape[0] == 0:
                continue
            kappa = float(np.linalg.cond(factor_matrix(walk)))
            out.append((physics.scene.name, art, walk, solver.factor.walk(joint_q, art, dtype=np.float32), kappa))
    return tuple(out)


def device_f1(device, units=UNITS, rng=None) -> dict:
    """F1 of the device walk on every catalogue model: each unit's rows and their ``J`` entries (``entries``: what a
    row's incident sums against ``v_hat``), the leave and its twists against the float64 walk, as worst ratios to ``n u
    kappa``, and its own rows and leave in one frame, ``|J dv - Z^T du|`` over ``n u |Z| |du|`` (pass: at most
    ``BOUND``), with the counts behind them."""
    rng = np.random.default_rng(1101) if rng is None else rng
    worst = {"rows": 0.0, "entries": 0.0, "leave": 0.0, "twist": 0.0, "frame": 0.0, "seeds": 0, "trees": 0}
    for _scene, _art, walk, walk32, kappa in catalogue_walks():
        n = walk.s.shape[0]
        scale = n * ROUNDING * kappa
        for rows in units:
            device_walk = DeviceWalk([walk32], device, rows)
            seeds = seed_rows(walk, rows, rng)
            body, top, seeded, wrench, covector = zip(*seeds, strict=True)
            z, j = device_walk.up([0] * len(seeds), body, top, seeded, wrench, covector, entries=True)
            for i, (b, c, is_covector, w, x) in enumerate(seeds):
                if is_covector:
                    X = np.zeros((n, rows))
                    X[c] = x
                    z64, _ = walk.up(covector=X)
                    j64 = X
                else:
                    z64, _ = walk.up(b, w.T)
                    j64 = jacobian(walk, b, w.T)
                worst["rows"] = max(worst["rows"], relative(z[i].T, z64) / scale)
                worst["entries"] = max(worst["entries"], relative(j[i].T, j64) / scale)
            worst["seeds"] += len(seeds)
        u = rng.normal(size=(2, n))
        reference = [walk.down(case) for case in u]
        qd, twist = DeviceWalk([walk32], device).down([0, 0], u)
        for case, (qd64, twist64) in enumerate(reference):
            worst["leave"] = max(worst["leave"], relative(qd[case], qd64) / scale)
            worst["twist"] = max(worst["twist"], relative(twist[case], twist64) / scale)
        # One frame: the device walk's rows of the contact seeds and its leave of a du, J dv against Z^T du.
        contacts = [seed for seed in seed_rows(walk, 3, rng) if not seed[2]]
        body, top, seeded, wrench, covector = zip(*contacts, strict=True)
        Z = DeviceWalk([walk32], device, 3).up([0] * len(contacts), body, top, seeded, wrench, covector)
        J = np.stack([jacobian(walk, b, w.T).T for b, _, _, w, _ in contacts])
        du = rng.normal(size=n)
        dv = DeviceWalk([walk32], device).down([0], du[None])[0][0]
        worst["frame"] = max(
            worst["frame"], frame_residual(J.reshape(-1, n), Z.reshape(-1, n), du, dv) / (n * ROUNDING)
        )
        worst["trees"] += 1
    return worst


def frame_residual(J: np.ndarray, Z: np.ndarray, du: np.ndarray, dv: np.ndarray) -> float:
    """``max_i |J_i dv - Z_i^T du| / (|Z_i| |du|)`` of rows ``J`` and their footprints ``Z`` ``[rows, n]``, ``dv`` the
    leave of ``du``. One factorization leaves the rounding of its two triangular passes, at most ``n u kappa`` and
    measured at ``n u``; two leave their factors' difference times ``|dv|``, which grows with ``kappa``."""
    residual = np.abs(J @ dv - Z @ du)
    scale = np.linalg.norm(Z, axis=1) * np.linalg.norm(du)
    keep = scale > 0.0
    return float(np.max(residual[keep] / scale[keep], initial=0.0))


# ---------------------------------------------------------------------------------------------------------------------
# F2 on the device, and one frame against two
# ---------------------------------------------------------------------------------------------------------------------


def chain_on_device(links: int, device) -> dict:
    """``long_chain(links, armature=0)``: a tip contact's three rows up and their responses down through the device
    walk (float32 data in local frames, no armature), against the float64 truth: ``Z``, ``dv = H^-1 J^T`` and the
    float32 walk on the host beside them."""
    model = scenes.long_chain(links=links, armature=0.0).build("cpu")
    joint_q = model.joint_q.numpy().astype(np.float64)
    n = int(model.joint_dof_count)
    parent, first, columns, inertia, offset = local_frames(model, joint_q, 0)
    # The device walk holds each body's float32 frame point and forms its offset as the difference of two (WALK_LINK):
    # the truth and the host's float32 walk take the chain of those offsets, so the bar measures the walk on its data.
    frames = frame_points(TreeWalk(parent, first, columns, inertia, offset, np.zeros(n))).astype(np.float32)
    offset = np.array([frames[up] - frames[b] if up >= 0 else np.zeros(3, np.float32) for b, up in enumerate(parent)])
    offset = offset.astype(np.float64)
    truth = TreeWalk(parent, first, columns, inertia, offset, np.zeros(n))
    walk32 = TreeWalk(parent, first, columns, inertia, offset, np.zeros(n), dtype=np.float32)
    body, W = ground_contact(model, newton_state_body_q(model, joint_q), links)
    z64, _ = truth.up(body, W)
    dv64 = np.stack([truth.down(z64[:, r])[0] for r in range(3)], axis=1)
    device_walk = DeviceWalk([walk32], device, frames=[frames])
    z = device_walk.up([0], [body], [int(truth.first[body + 1]) - 1], [0], [W.T], [np.zeros(3)])[0].T
    dv = device_walk.down([0, 0, 0], z.T)[0].T
    host32, _ = walk32.up(body, W.astype(np.float32))
    return {
        "links": links,
        "Z": relative(z, z64),
        "dv": relative(dv, dv64),
        "host_dv": relative(np.stack([walk32.down(host32[:, r])[0] for r in range(3)], axis=1), dv64),
        "pivots_positive": bool(np.all(walk32.ell > 0.0)),
    }


def frames_on_chain(links: int, device) -> dict:
    """One frame against two on ``long_chain(links)`` without armature (the solver adds none, so the walk and main's
    ``L`` factor one matrix): the device walk's rows of a tip contact with the device walk's leave, and with a leave by
    main's float32 ``L``: ``|J dv - Z^T du|`` over ``n u |Z| |du|``. Main's float32 factor fails on this chain without
    the armature it added (a non-finite leave), which is not within the bound either."""
    model = scenes.long_chain(links=links, armature=0.0).build("cpu")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver = SolverFeatherPGS(model, dense_max_constraints=8)
    state, out, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    joint_q = state.joint_q.numpy().astype(np.float64)
    solver.step(state, out, control, None, 1.0 / 240.0)
    n = int(model.joint_dof_count)
    H32 = MassReference(model, np.float32).mass_matrix(joint_q, 0).astype(np.float64)
    L32 = dense_factor32(H32 + np.diag(solver.factor.diagonal(0))).astype(np.float64)
    walk = solver.factor.walk(joint_q, 0)
    walk32 = solver.factor.walk(joint_q, 0, dtype=np.float32)
    body, W = ground_contact(model, newton_state_body_q(model, joint_q), links)
    J = jacobian(walk, body, W).T
    device_walk = DeviceWalk([walk32], device)
    Z = device_walk.up([0], [body], [int(walk.first[body + 1]) - 1], [0], [W.T], [np.zeros(3)])[0]
    du = Z.sum(axis=0)
    dv_walk = device_walk.down([0], du[None])[0][0]
    from scipy.linalg import solve_triangular

    dv_main = solve_triangular(L32, du[::-1], lower=True, trans=1, check_finite=False)[::-1]
    return {
        "links": links,
        "kappa": float(np.linalg.cond(factor_matrix(walk))),
        "one_frame": frame_residual(J, Z, du, dv_walk) / (n * ROUNDING),
        "two_frames": frame_residual(J, Z, du, dv_main) / (n * ROUNDING),
    }


# ---------------------------------------------------------------------------------------------------------------------
# The step's placements: one frame each, and F1 on CUDA
# ---------------------------------------------------------------------------------------------------------------------


def placement_frames(solver, joint_q: np.ndarray, capture) -> dict:
    """One run's placements against the walk, as ratios to their rounding, ``n u kappa`` and the publication's (pass:
    at most ``BOUND``). Every class with response coordinates walks its rows (the one writer, FL3.3; the sparse
    diagonal's, FL3.5), so no store keeps a row's ``J``:

    - ``rows`` and ``leave``: the one writer's worlds, their rows against their physical ``J v_hat`` and their leave
      against the float64 walk's (:func:`.test_feather_pgs_walk_rows.walked_frames`, ``slot_frames``; Z0's F1).

    The worlds a speed ball acted in are left out (their ``du`` is a law's, not ``sum Z lambda``), and so is every
    world of the support placement, case K1 (``unstored``: its rows are :class:`TestSupportPlacement`'s)."""
    forest, placement = solver.forest, solver.placement
    out = {"rows": 0.0, "leave": 0.0, "general": 0, "unstored": 0}
    if not getattr(placement, "walks", False) or solver.classes.case == "K1":
        out["unstored"] = int(forest.world_count)
        return out
    # The one writer of the general placement (FL3.3) and of the sparse diagonal (FL3.5): its rows are walked from
    # seeds. Its rows against their physical J v_hat and its leave against the float64 walk's
    # (test_feather_pgs_walk_rows).
    from newton.tests.test_feather_pgs_walk_rows import slot_frames, walked_frames  # noqa: PLC0415

    frames = slot_frames if placement.sparse_diagonal_contact_solve else walked_frames
    walked = frames(solver, joint_q, capture)
    out.update(rows=walked["rows"], leave=walked["leave"], general=walked["worlds"])
    return out


def cuda_placements() -> list:
    """The CUDA catalogue's problems on the routes whose rows a store holds, ``(physics, route)`` in catalogue order: a
    route is checked on its first problem with articulated rows (:func:`checked`)."""
    return [
        (group.physics, route)
        for group in conformance.cuda_groups()
        for route in (group.reference, *group.routes)
        if route.name not in UNSTORED
    ]


# ---------------------------------------------------------------------------------------------------------------------
# The class emitter in both directions
# ---------------------------------------------------------------------------------------------------------------------

_BOTH = """    if (tid >= weights.shape[0]) return;
    const int n = values.shape[1], bodies = weights.shape[1];
    WALK_BOTH(weights.data + (size_t)tid * bodies, values.data + (size_t)tid * n, path.data + (size_t)tid * n,
              subtree.data + (size_t)tid * bodies);
"""
_BOTH_PARAMS = params(
    ("weights", wp.array2d[float]),
    ("values", wp.array2d[float]),
    ("path", wp.array2d[float]),
    ("subtree", wp.array2d[float]),
)


@functools.cache
def both_ways_kernel(table: tuple):
    """A thread per case of one class: root first, each coordinate's sum of the weights of the bodies and the values of
    the coordinates on its root path up to it (``path``); leaf first, each body's sum over its subtree (``subtree``)."""
    walk = ts.WalkClass(*table)

    down = walk.walk("$V[0] = $P[0];", "$V[0] += x[$c];\ndown[$c] = $V[0];", begin="$V[0] += w[$b];", width=1)
    up = walk.walk(
        "$P[0] += $V[0];", "$V[0] += x[$c];", begin="$V[0] += w[$b];", end="up[$b] = $V[0];", root_first=False, width=1
    )
    text = f"""    const auto WALK_BOTH = [&](const float* w, const float* x, float* down, float* up) {{
        {{
{ts.indent_text(down, 12)}
        }}
        {{
{ts.indent_text(up, 12)}
        }}
    }};"""

    # The loop form's tables take the one access's WALK_CONSTANT.
    prelude = DevicePrelude(f"walk_both_{walk.tag}", text, cpu=text)
    preludes = (factor_walk.walk_loads_prelude(), prelude)
    return KernelTemplate(
        f"walk_both_{walk.tag}", _BOTH_PARAMS, _BOTH, items("tid", flat=True), preludes, dual=True
    ).kernel()


def both_ways(table: tuple, device, rng) -> tuple:
    """Integer weights and values (sums exact in float32) walked both ways by the emitted class, against numpy."""
    parent, first = table
    bodies, n = len(parent), first[-1]
    cases = 4
    weights = rng.integers(-8, 9, size=(cases, bodies)).astype(np.float32)
    values = rng.integers(-8, 9, size=(cases, n)).astype(np.float32)
    path, subtree = np.zeros((cases, n), np.float32), np.zeros((cases, bodies), np.float32)
    for case in range(cases):
        acc = np.zeros(bodies)
        for b in range(bodies):
            acc[b] = (acc[parent[b]] if parent[b] >= 0 else 0.0) + weights[case, b]
            for c in range(first[b], first[b + 1]):
                acc[b] += values[case, c]
                path[case, c] = acc[b]
        sub = weights[case].astype(np.float64) + [values[case, first[b] : first[b + 1]].sum() for b in range(bodies)]
        for b in reversed(range(bodies)):
            if parent[b] >= 0:
                sub[parent[b]] += sub[b]
        subtree[case] = sub
    outputs = (wp.zeros((cases, n), dtype=float, device=device), wp.zeros((cases, bodies), dtype=float, device=device))
    wp.launch(
        both_ways_kernel(table),
        dim=cases,
        inputs=[wp.array(weights, dtype=float, device=device), wp.array(values, dtype=float, device=device), *outputs],
        device=device,
    )
    return outputs[0].numpy(), path, outputs[1].numpy(), subtree


def emitter_tables(rng) -> list:
    """Classes on both sides of ``WALK_UNROLL``: random trees (free and scalar joints, several roots, joins whose spans
    nest and follow each other), a chain and a star."""
    tables = []
    for bodies, roots in ((5, 1), (12, 2), (30, 1), (48, 3)):
        parent = [-1] + [int(rng.integers(-1, b)) if b < roots else int(rng.integers(0, b)) for b in range(1, bodies)]
        counts = [int(rng.choice((1, 1, 1, 3, 6))) for _ in range(bodies)]
        tables.append((tuple(parent), tuple(np.concatenate(([0], np.cumsum(counts))).tolist())))
    tables.append((tuple(range(-1, 105)), tuple(range(107))))
    tables.append(((-1, *([0] * 69)), (0, *range(6, 76))))
    return tables


def checked(result: dict) -> bool:
    """Whether a run held rows the check could read: worlds the general sweep solves from the store."""
    return result["general"] > 0


def refresh_error(device, bodies: int, rng) -> float:
    """One refresh of a random tree (its joins included) by walk_step_kernel's lane feed against the float64 walk: the
    worst relative error of ``s``, ``ut``, ``1/ell`` and ``P0``."""
    model, q = random_tree(rng, bodies, device)
    model.joint_q.assign(q)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver = SolverFeatherPGS(model, pgs_iterations=4)
    state = model.state()
    factor = solver.factor
    if factor.walk_arts is None:
        factor.plan_walk(np.arange(model.articulation_count, dtype=np.int32))
    # The held tree's repair publishes the state's poses and the walk's step geometry, as a step's would.
    factor.fk_id(state, model.state(), state.joint_qd, 1.0 / 240.0)
    factor.mass_update_mask.fill_(1)
    walk = factor.walk(q, 0)
    n = walk.s.shape[0]
    pivots = walk.own_pivots + factor.armature[:n]
    factor.walk_step(state, 1.0 / 240.0)
    d = unpack_walk_records(factor.walk_data.numpy(), n).astype(np.float64)
    return max(
        np.abs(d[:, 0:6] - walk.s).max(),
        np.abs(d[:, 6:12] - walk.ut).max() / max(np.abs(walk.ut).max(), 1.0e-30),
        np.abs(1.0 / d[:, 12] - walk.ell).max() / max(walk.ell.max(), 1.0e-30),
        np.abs(d[:, 13] - pivots).max() / max(np.abs(walk.own_pivots).max(), 1.0e-30),
    )


# ---------------------------------------------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------------------------------------------


class TestWalkDevice(unittest.TestCase):
    """F1 and F2 of the device walk, and one frame per placement, on the CPU and the CUDA device."""

    def test_f1_device_walk_on_the_catalogue(self):
        """Every unit's rows and their ``J`` entries, the leave and its twists within ``BOUND n u kappa`` of the float64
        walk, and the device walk's own rows and leave in one frame."""
        for device in devices():
            with self.subTest(device=device):
                worst = device_f1(device)
                self.assertGreater(worst["seeds"], 0)
                for name in ("rows", "entries", "leave", "twist", "frame"):
                    self.assertLessEqual(worst[name], BOUND, f"{name}: {worst}")

    def test_f2_on_the_device(self):
        """A tip impulse's response on ``long_chain(N)`` without armature: the device walk's float32 within 1e-5 of the
        float64 truth at every length, as the host's float32 walk."""
        for device in devices():
            for links in CHAINS:
                with self.subTest(device=device, links=links):
                    chain = chain_on_device(links, device)
                    self.assertTrue(chain["pivots_positive"])
                    self.assertLess(chain["dv"], 1.0e-5, f"{chain}")
                    self.assertLess(chain["dv"], 10.0 * chain["host_dv"] + 1.0e-6, f"{chain}")

    def test_a_held_factor_whitens_the_steps_J(self):
        """Between refreshes the factor is the refresh's and ``J`` the step's: the device walk with the step's
        geometry gives ``L_r^-1 P J_t^T`` within ``BOUND n u kappa``, and the refresh pose's ``J`` misses it by far."""
        for device in devices():
            with self.subTest(device=device):
                result = held_step(device)
                self.assertGreater(result["seeds"], 0)
                self.assertLessEqual(result["held"], BOUND, f"{result}")
                self.assertGreater(result["stale"], 1.0e3 * BOUND, f"{result}")

    def test_one_frame_tells_two_apart(self):
        """``long_chain(106)`` without armature: the walk's rows with the walk's leave stay within ``BOUND n u`` of
        ``|Z| |du|``; with a leave by main's float32 factor they do not (a failed factor, NaN, is not within). The
        invariant guards a placement that converts its rows without its leave."""
        for device in devices():
            with self.subTest(device=device):
                frames = frames_on_chain(106, device)
                self.assertLessEqual(frames["one_frame"], BOUND, f"{frames}")
                self.assertFalse(frames["two_frames"] <= BOUND, f"{frames}")

    def test_one_frame_per_placement_cpu(self):
        """Every CPU catalogue route: its rows and its leave within ``BOUND n u kappa`` of the walk's."""
        for physics, route in catalogue():
            with self.subTest(problem=physics.name, route=route.name):
                solver, joint_q, capture = run_route(physics, route)
                result = placement_frames(solver, joint_q, capture)
                for name in ("rows", "leave"):
                    self.assertLessEqual(result[name], BOUND, f"{name}: {result}")

    def test_the_refresh_meets_the_float64_walk(self):
        """walk_step_kernel's refresh (the lane feed) within float32 rounding of the float64 walk, on random trees with
        joins."""
        for device in devices():
            for bodies in (3, 8, 14):
                with self.subTest(device=device, bodies=bodies):
                    error = refresh_error(device, bodies, np.random.default_rng(bodies))
                    self.assertLess(error, 1.0e-4)

    def test_the_class_emitter_walks_both_ways(self):
        """The class emitter's walk root first and leaf first, unrolled and in the loop form, on trees with joins and
        several roots: exact sums along each root path and over each subtree."""
        tables = emitter_tables(np.random.default_rng(1102))
        self.assertTrue(any(first[-1] <= ts.WALK_UNROLL for _, first in tables))
        self.assertTrue(any(first[-1] > ts.WALK_UNROLL for _, first in tables))
        for device in devices():
            for table in tables:
                with self.subTest(device=device, coordinates=table[1][-1]):
                    down, path, up, subtree = both_ways(table, device, np.random.default_rng(len(table[0])))
                    np.testing.assert_array_equal(down, path)
                    np.testing.assert_array_equal(up, subtree)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA placements")
    def test_f1_on_the_cuda_placements(self):
        """Z0's F1 on the CUDA placements: the one writer's worlds' rows and leave."""
        covered, done = 0, set()
        for physics, route in cuda_placements():
            if route.name in done:
                continue
            with self.subTest(problem=physics.name, route=route.name):
                solver, joint_q, capture = run_route(physics, route, device="cuda:0")
                result = placement_frames(solver, joint_q, capture)
                for name in ("rows", "leave"):
                    self.assertLessEqual(result[name], BOUND, f"{name}: {result}")
                if checked(result):
                    done.add(route.name)
                covered += result["general"]
        # The one writer's worlds were checked.
        self.assertGreater(covered, 0)


def report_device() -> None:
    """Print the device walk's measured tables."""
    for device in devices():
        print(f"== {device}: F1 of the device walk (worst ratio to n u kappa)", device_f1(device))
        for links in CHAINS:
            print(f"== {device}: F2", chain_on_device(links, device))
        for links in (32, 106, 300):
            print(f"== {device}: frames", frames_on_chain(links, device))
    for physics, route in catalogue():
        solver, joint_q, capture = run_route(physics, route)
        print(f"== cpu {physics.name} {route.name}", placement_frames(solver, joint_q, capture))
    if wp.is_cuda_available():
        done = set()
        for physics, route in cuda_placements():
            if route.name in done:
                continue
            solver, joint_q, capture = run_route(physics, route, device="cuda:0")
            result = placement_frames(solver, joint_q, capture)
            print(f"== cuda {physics.name} {route.name}", result)
            if checked(result):
                done.add(route.name)


# =====================================================================================================================
# The reference route against an independent float64 evaluation
# =====================================================================================================================

"""The bounds' multiple of ``n u |A|``: every catalogue model measures under 6 on the CPU."""


def _problems() -> list:
    """Every model of the CUDA catalogue once, with its options."""
    seen, problems = set(), []
    for group in conformance.cuda_groups():
        if group.physics.scene.name not in seen:
            seen.add(group.physics.scene.name)
            problems.append(group.physics)
    return problems


@dataclasses.dataclass
class Evaluated:
    """A solver after its last step, with the coordinates around that step."""

    solver: SolverFeatherPGS
    joint_q_in: np.ndarray
    joint_q_out: np.ndarray


def _run(physics, reference, device) -> Evaluated:
    scene = physics.scene
    model = scene.build(device)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    options = {**conformance.BASE_OPTIONS, **scene.solver_options, **physics.options(physics.iterations)}
    # The rows' material law is the sweep's: these contacts hold no per-contact material.
    options.pop("contact_compliance", None)
    with conformance.route_context(reference):
        solver = SolverFeatherPGS(model, pins=reference.solver_pins(), **options)
        state, out, control = model.state(), model.state(), model.control()
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        for step in range(STEPS):
            state.clear_forces()
            if scene.control is not None:
                scene.control(model, control, state, step)
            pipeline.collide(state, contacts)
            solver.step(state, out, control, contacts, scene.dt)
            # The coordinates as the forest holds them: the reference evaluates the solver's own model.
            joint_q_in = solver.cut.mirror(state).joint_q.numpy().copy()
            state, out = out, state
    return Evaluated(solver, joint_q_in, solver.cut.mirror(state).joint_q.numpy().copy())


def _held_groups(solver):
    """Each held tree group of dense or compact coordinates: its operator, its articulations per group, its host."""
    for group in solver.factor.held.groups:
        operator = group.sparse
        if getattr(operator, "prescribed", False) or not hasattr(operator, "host"):
            continue
        arts = getattr(operator, "forest_arts", None)
        arts = operator.plan.group_to_art.numpy()[:, None] if arts is None else arts
        yield group, operator, arts


def _block(reference, joint_q, arts) -> np.ndarray:
    """The mass matrix of the trees ``arts`` side by side."""
    blocks = [reference.mass_matrix(joint_q, int(art)) for art in arts]
    n = sum(block.shape[0] for block in blocks)
    H, at = np.zeros((n, n)), 0
    for block in blocks:
        H[at : at + block.shape[0], at : at + block.shape[0]] = block
        at += block.shape[0]
    return H


def _bound(A, n=None) -> float:
    return BOUND * (A.shape[0] if n is None else n) * ROUNDING * np.abs(A).max()


def _diagonal(solver, art) -> np.ndarray:
    """The armature and drive stiffness of articulation ``art``'s coordinates."""
    forest = solver.forest
    start, size = int(forest.articulation_dof_start_host[art]), int(forest.articulation_dof_count_host[art])
    added = solver.factor.dof_armature.numpy()[start : start + size].astype(np.float64)
    rows = solver.factor.drive_row_by_dof.numpy()[start : start + size]
    # The factor holds every drive's K: a saturated drive's is released by the predictor's border, not the factor.
    K = solver.factor.aug_row_K.numpy()
    added += np.where(rows >= 0, np.maximum(K[np.maximum(rows, 0)], 0.0), 0.0)
    return added


def check_geometry(test, evaluated, reference) -> None:
    """What the finish forms of ``H`` against the float64 mass matrix at the last step's output: each drive's K-free
    mass ``H_ii + R_ii`` (``row_H``) and a compact diagonal's ``H_ii``."""
    solver = evaluated.solver
    forest, factor = solver.forest, solver.factor
    row_H, drive_rows, armature = factor.aug_row_H.numpy(), factor.drive_row_by_dof.numpy(), factor.dof_armature.numpy()
    for group, operator, arts in _held_groups(solver):
        kept = group.mass_diagonal.numpy().astype(np.float64) if group.diagonal_mass else None
        for index in (0, len(arts) - 1):
            H = _block(reference, evaluated.joint_q_out, arts[index])
            dofs = np.concatenate(
                [
                    np.arange(forest.articulation_dof_start_host[a], forest.articulation_dof_start_host[a] + n)
                    for a in arts[index]
                    for n in (int(forest.articulation_dof_count_host[a]),)
                ]
            )
            driven = drive_rows[dofs] >= 0
            mass = np.diag(H)[driven] + armature[dofs[driven]]
            error = np.abs(row_H[drive_rows[dofs[driven]]] - mass).max(initial=0.0)
            test.assertLessEqual(error, _bound(H), f"row_H of group {index}, {operator.kind}")
            if kept is not None:
                error = np.abs(kept[index] - np.diag(H)).max()
                test.assertLessEqual(error, _bound(H), f"H_ii of group {index}, {operator.kind}")


def check_factor(test, evaluated, reference) -> None:
    """The walk (the factor) against the float64 matrix at the last step's input, and its solve."""
    solver = evaluated.solver
    rng = np.random.default_rng(11)
    for _, operator, arts in _held_groups(solver):
        if operator.response == "diagonal" or arts.shape[1] != 1:
            continue
        for index in (0, len(arts) - 1):
            art = int(arts[index, 0])
            A = reference.mass_matrix(evaluated.joint_q_in, art) + np.diag(_diagonal(solver, art))
            inverse = walk_inverse(solver, art)
            L = np.linalg.inv(inverse)
            test.assertLessEqual(np.abs(L @ L.T - A).max(), _bound(A), "L L^T")
            x = rng.normal(size=A.shape[0])
            y = inverse.T @ (inverse @ x)
            test.assertLessEqual(
                np.abs(A @ y - x).max(), _bound(A) * np.abs(y).max() + BOUND * ROUNDING * np.abs(x).max(), "A y - x"
            )


def check_swept_rows(test, evaluated, reference) -> None:
    """The swept rows' whitening against the float64 matrix at the last step's input."""
    solver = evaluated.solver
    if solver.placement.packed_Z.shape[2] <= 1 or not int(solver.placement.constraint_count.numpy().sum()):
        return
    forest = solver.forest
    sizes = forest.articulation_response_dof_count.numpy()
    offsets = forest.articulation_world_dof_offset.numpy()
    worlds = forest.art_to_world.numpy()
    J, Y = unpacked_rows(solver)
    for world, count in enumerate(solver.placement.constraint_count.numpy()):
        if not count:
            continue
        n = int(J.shape[2])
        A = np.zeros((n, n))
        for art in np.flatnonzero((worlds == world) & (sizes > 0)):
            at, size = int(offsets[art]), int(sizes[art])
            H = reference.mass_matrix(evaluated.joint_q_in, int(art))
            A[at : at + size, at : at + size] = H + np.diag(_diagonal(solver, int(art)))
        scale = _bound(A) * np.abs(Y[world, : int(count)]).max(axis=1)
        residual = np.abs(Y[world, : int(count)] @ A - J[world, : int(count)]).max(axis=1)
        test.assertTrue(np.all(residual <= scale + _bound(np.abs(J[world, : int(count)]), n)), "A Y - J")


class _Invariants(unittest.TestCase):
    device = None
    reference_route = None

    def check(self, physics):
        evaluated = _run(physics, self.reference_route, self.device)
        reference = MassReference(evaluated.solver.model)
        check_geometry(self, evaluated, reference)
        check_factor(self, evaluated, reference)
        check_swept_rows(self, evaluated, reference)


def _add_tests(cls) -> None:
    for physics in _problems():

        def test(self, physics=physics):
            self.check(physics)

        test.__doc__ = f"Hold {physics.scene.name}'s held tree to the float64 mass matrix."
        setattr(cls, f"test_{physics.scene.name}", test)


class TestTreeInvariantsCPU(_Invariants):
    device = "cpu"
    reference_route = conformance.REFERENCE_CPU


_add_tests(TestTreeInvariantsCPU)


@unittest.skipUnless(wp.is_cuda_available(), "CUDA")
class TestTreeInvariantsCUDA(_Invariants):
    device = "cuda:0"
    reference_route = conformance.REFERENCE_CUDA


_add_tests(TestTreeInvariantsCUDA)


# =====================================================================================================================
# The drives
# =====================================================================================================================


def _mixed_target_model(coord_layout, device):
    """Place scalar drives before and after free/ball joints in two worlds."""
    with mock.patch.object(newton, "use_coord_layout_targets", coord_layout):
        scene = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        for joint_kind in ("revolute", "free", "ball", "revolute"):
            body = scene.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
            joint = getattr(scene, f"add_joint_{joint_kind}")(parent=-1, child=body)
            scene.add_articulation([joint])
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        builder.replicate(scene, 2)
        model = builder.finalize(device=device)
    joints = np.flatnonzero(model.joint_type.numpy() == int(JointType.REVOLUTE))
    dofs = model.joint_qd_start.numpy()[joints]
    coords = model.joint_q_start.numpy()[joints]
    targets = model.joint_target_q_start.numpy()[joints]
    state, control = model.state(), model.control()
    q, qd = state.joint_q.numpy().copy(), state.joint_qd.numpy().copy()
    q[coords], qd[dofs] = [0.04, 0.05, 0.06, 0.07], [0.06, -0.05, 0.04, -0.03]
    state.joint_q.assign(q)
    state.joint_qd.assign(qd)
    target_q, target_qd = control.joint_target_q.numpy().copy(), control.joint_target_qd.numpy().copy()
    target_q[targets], target_qd[dofs] = [0.2, -0.3, 0.4, -0.5], [0.12, -0.07, 0.03, -0.11]
    control.joint_target_q.assign(target_q)
    control.joint_target_qd.assign(target_qd)
    ke, kd = np.zeros(model.joint_dof_count), np.zeros(model.joint_dof_count)
    ke[dofs], kd[dofs] = 20.0, 3.0
    model.joint_target_ke.assign(ke)
    model.joint_target_kd.assign(kd)
    model.joint_effort_limit.fill_(100.0)
    return model, state, control, joints, dofs, coords, targets


def _drive_rows(row_K, articulations, dofs, device):
    """The arrays a prepare kernel writes around ``row_K``, with no last step and no drive at its limit yet."""
    return field_group.upload(
        DriveRows,
        device,
        row_K=row_K,
        row_dv=wp.zeros_like(row_K),
        drive=wp.zeros(dofs, dtype=wp.vec4, device=device),
        mass_update_requested=wp.zeros(articulations, dtype=int, device=device),
        v_out=wp.zeros(dofs, dtype=float, device=device),
        v_hat=wp.zeros(dofs, dtype=float, device=device),
    )


class TestFeatherPGSAugmentedDrives(unittest.TestCase):
    def test_mixed_joint_target_layout_trajectories(self):
        """Match public solver trajectories across both target layouts."""
        trajectories = []
        for coord_layout in (False, True):
            model, state, control, _joints, dofs, coords, targets = _mixed_target_model(coord_layout, wp.get_device())
            solver = newton.solvers.SolverFeatherPGS(
                model,
                enable_joint_limits=False,
                dense_max_constraints=16,
                mf_max_constraints=16,
                pgs_iterations=8,
            )
            initial_qd = state.joint_qd.numpy()[dofs].copy()
            position_error = control.joint_target_q.numpy()[targets] - state.joint_q.numpy()[coords]
            following = model.state()
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
            history = []
            for step in range(3):
                state.clear_forces()
                solver.step(state, following, control, None, 0.01)
                state, following = following, state
                q, qd = state.joint_q.numpy(), state.joint_qd.numpy()
                if step == 0:
                    self.assertTrue(np.all((qd[dofs] - initial_qd) * position_error > 0.0))
                history.append(np.concatenate((q, qd)))
            trajectories.append(history)
        np.testing.assert_allclose(trajectories[0], trajectories[1], rtol=0.0, atol=2e-6)

    def test_mixed_joint_position_target_layouts(self):
        """Keep distinct world targets aligned after free and ball coordinates."""
        device, dt = wp.get_device(), 0.01
        for coord_layout in (False, True):
            model, state, control, joints, dofs, coords, targets = _mixed_target_model(coord_layout, device)
            max_dofs = 6
            row_count = model.articulation_count * max_dofs
            expected = np.zeros(model.joint_dof_count)
            expected[dofs] = 20.0 * (
                control.joint_target_q.numpy()[targets]
                - state.joint_q.numpy()[coords]
                - dt * state.joint_qd.numpy()[dofs]
            ) + 3.0 * (control.joint_target_qd.numpy()[dofs] - state.joint_qd.numpy()[dofs])
            with self.subTest(coord_layout=coord_layout):
                row_K = wp.zeros(row_count, dtype=float, device=device)
                tau = wp.zeros(model.joint_dof_count, dtype=float, device=device)
                rows = np.full(model.joint_dof_count, -1, dtype=np.int32)
                q_indices = np.zeros(model.joint_dof_count, dtype=np.int32)
                target_indices = np.zeros(model.joint_dof_count, dtype=np.int32)
                rows[dofs], q_indices[dofs] = joints * max_dofs, coords
                target_indices[dofs] = targets
                wp.launch(
                    prepare_augmented_drives,
                    dim=model.joint_dof_count,
                    inputs=[
                        wp.array(np.arange(model.joint_dof_count, dtype=np.int32), dtype=int, device=device),
                        wp.array(rows, dtype=int, device=device),
                        wp.array(q_indices, dtype=int, device=device),
                        wp.array(target_indices, dtype=int, device=device),
                        state.joint_q,
                        state.joint_qd,
                        model.joint_target_ke,
                        model.joint_target_kd,
                        control.joint_target_q,
                        control.joint_target_qd,
                        model.joint_effort_limit,
                        wp.zeros(model.joint_dof_count, dtype=float, device=device),
                        max_dofs,
                        dt,
                    ],
                    outputs=[_drive_rows(row_K, model.articulation_count, model.joint_dof_count, device), tau],
                    device=device,
                )
                np.testing.assert_allclose(tau.numpy(), expected, rtol=0, atol=2e-6)

    def test_per_coordinate_drives(self):
        """Write each coordinate's K and clamped force from its row, clear both without gains, and request a factor
        refresh wherever a drive's K changed, to zero included (one law for every drive)."""
        device = wp.get_device()
        joint_q = wp.array([0.25, -0.35], dtype=wp.float32, device=device)
        joint_qd = wp.array([0.1, -0.2], dtype=wp.float32, device=device)
        target_ke = wp.array([10.0, 20.0], dtype=wp.float32, device=device)
        target_kd = wp.array([1.0, 2.0], dtype=wp.float32, device=device)
        target_q = wp.array([0.1, -0.1], dtype=wp.float32, device=device)
        target_qd = wp.array([0.0, 0.05], dtype=wp.float32, device=device)
        effort_limit = wp.array([1.25, 2.5], dtype=wp.float32, device=device)
        damping = wp.array([0.2, 0.3], dtype=wp.float32, device=device)
        rows = wp.array([0, 1], dtype=wp.int32, device=device)

        def prepare_by_dof(ke, kd, stiffness, tau, passive=damping):
            drive_rows = _drive_rows(stiffness, 1, 2, device)
            wp.launch(
                prepare_augmented_drives,
                dim=2,
                inputs=[
                    rows,
                    rows,
                    rows,
                    rows,
                    joint_q,
                    joint_qd,
                    ke,
                    kd,
                    target_q,
                    target_qd,
                    effort_limit,
                    passive,
                    2,
                    0.1,
                ],
                outputs=[drive_rows, tau],
                device=device,
            )
            return drive_rows.mass_update_requested.numpy()

        stiffness = wp.zeros(2, dtype=wp.float32, device=device)
        tau = wp.zeros(2, dtype=wp.float32, device=device)
        prepare_by_dof(target_ke, target_kd, stiffness, tau)
        # The drive's K, ke dt^2 + kd dt, with the passive damper's, c dt.
        np.testing.assert_allclose(stiffness.numpy(), [0.22, 0.43], rtol=0.0, atol=1.0e-7)
        # Both forces are past their effort limits: -(10 (0.25 - 0.1 + 0.01) + 0.1) and -(20 (-0.25 - 0.02) + 2 (-0.25)).
        np.testing.assert_allclose(tau.numpy(), [-1.25, 2.5], rtol=0.0, atol=1.0e-6)

        # Without gains the preparation keeps no drive force, and the passive damper keeps its K.
        zero_gain = wp.zeros_like(target_ke)
        passive_stiffness = wp.array([7.0, 7.0], dtype=wp.float32, device=device)
        passive_tau = wp.array([3.0, -2.0], dtype=wp.float32, device=device)
        prepare_by_dof(zero_gain, zero_gain, passive_stiffness, passive_tau)
        np.testing.assert_allclose(passive_stiffness.numpy(), [0.02, 0.03], rtol=0.0, atol=1.0e-7)
        np.testing.assert_array_equal(passive_tau.numpy(), [0.0, 0.0])

        # A drive whose K falls to zero (no gains, no damper) holds no K and asks for a factor refresh, as every
        # changed K does; an unchanged zero asks for none.
        held = wp.array([0.22, 0.43], dtype=wp.float32, device=device)
        requested = prepare_by_dof(zero_gain, zero_gain, held, wp.zeros(2, dtype=wp.float32, device=device), zero_gain)
        np.testing.assert_array_equal(held.numpy(), [0.0, 0.0])
        self.assertNotEqual(int(requested[0]), 0)
        requested = prepare_by_dof(zero_gain, zero_gain, held, wp.zeros(2, dtype=wp.float32, device=device), zero_gain)
        self.assertEqual(int(requested[0]), 0)


# =====================================================================================================================
# The generalized laws
# =====================================================================================================================


class TestGeneralizedOperations(unittest.TestCase):
    def test_permuted_physical_indices_and_prescribed_dofs(self):
        """Update each physical DOF once under an unseen nonidentity permutation."""

        @wp.kernel
        def mapped(
            indices: wp.array[int],
            qd: wp.array[float],
            mask: wp.array[int],
            velocity: wp.array[float],
            acceleration: wp.array[float],
        ):
            factor_kernels._light_update_qdd_from_velocity(indices[wp.tid()], qd, mask, 2.0, velocity, acceleration)

        device = os.environ.get("FPGS_TEST_DEVICE", "cpu")
        n = 37
        old = np.linspace(-1.5, 2.5, n, dtype=np.float32)
        velocity = np.sin(np.arange(n)).astype(np.float32)
        mask = (np.arange(n) % 5 == 1).astype(np.int32)
        expected_velocity = np.where(mask != 0, old, velocity)
        expected_acceleration = np.where(mask != 0, 0.0, (velocity - old) * 2.0)
        arrays = [
            wp.array((np.arange(n) * 13 % n).astype(np.int32), dtype=int, device=device),
            wp.array(old, device=device),
            wp.array(mask, dtype=int, device=device),
            wp.array(velocity, device=device),
            wp.full(n, float("nan"), device=device),
        ]
        wp.launch(mapped, dim=n, inputs=arrays, device=device)
        np.testing.assert_array_equal(arrays[3].numpy(), expected_velocity)
        np.testing.assert_array_equal(arrays[4].numpy(), expected_acceleration)


# =====================================================================================================================
# The FK/ID cache's publication across graph replays
# =====================================================================================================================


def _build(device):
    """Three articulated links per world with fabricated ground contacts on the last link."""
    builder = newton.ModelBuilder()
    shape_cfg = builder.default_shape_cfg.copy()
    shape_cfg.density = 0.0
    shape_cfg.mu = 0.6
    shapes = []
    for world in range(6):
        builder.begin_world()
        parent = -1
        joints = []
        for link in range(3):
            body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * 0.05), lock_inertia=True)
            if link == 0:
                joint = builder.add_joint_prismatic(parent=parent, child=body, axis=newton.Axis.Z)
            else:
                joint = builder.add_joint_revolute(
                    parent=parent,
                    child=body,
                    axis=newton.Axis.Y,
                    parent_xform=wp.transform(wp.vec3(0.0, 0.0, -0.3), wp.quat_identity()),
                )
            joints.append(joint)
            parent = body
        builder.add_articulation(joints)
        shapes.append(
            builder.add_shape_sphere(
                parent,
                radius=0.02,
                xform=wp.transform(wp.vec3(0.0, 0.0, -0.05 - 0.002 * world), wp.quat_identity()),
                cfg=shape_cfg,
            )
        )
        builder.end_world()
    plane = builder.add_ground_plane()
    model = builder.finalize(device=device)
    model.rigid_contact_max = len(shapes)
    contacts = newton.Contacts(rigid_contact_max=len(shapes), soft_contact_max=0, device=device)
    count = len(shapes)
    contacts.rigid_contact_count.assign(np.array([count], dtype=np.int32))
    contacts.rigid_contact_shape0.assign(np.asarray(shapes, dtype=np.int32))
    contacts.rigid_contact_shape1.assign(np.full(count, plane, dtype=np.int32))
    points = np.zeros((count, 3), dtype=np.float32)
    points[:, 2] = -0.001
    contacts.rigid_contact_point0.assign(points)
    contacts.rigid_contact_point1.assign(np.zeros((count, 3), dtype=np.float32))
    contacts.rigid_contact_normal.assign(np.tile([0.0, 0.0, -1.0], (count, 1)).astype(np.float32))
    return model, contacts


def _digest(state):
    return hashlib.sha256(
        b"".join(getattr(state, name).numpy().tobytes() for name in ("joint_q", "joint_qd", "body_q", "body_qd"))
    ).hexdigest()


@unittest.skipUnless(wp.is_cuda_available(), "Graph replays are a CUDA path")
class TestFeatherPGSFkCachePublication(unittest.TestCase):
    def _run(self, replays, substeps=3, dt=0.005):
        device = wp.get_device("cuda:0")
        model, contacts = _build(device)
        solver = SolverFeatherPGS(
            model,
            dense_max_constraints=16,
            pgs_iterations=6,
        )
        states = [model.state() for _ in range(2)]
        newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
        control = model.control()

        def substep_block():
            for _ in range(substeps):
                solver.step(states[0], states[1], control, contacts, dt)
                states.reverse()

        # Step one block eagerly, then capture one block and replay it.
        substep_block()
        wp.synchronize_device(device)
        with wp.ScopedCapture(device=device) as capture:
            substep_block()
        digests = []
        for _ in range(replays):
            wp.capture_launch(capture.graph)
            wp.synchronize_device(device)
            digests.append(_digest(states[0]))
        self.assertTrue(np.isfinite(states[0].joint_q.numpy()).all())
        return digests

    def test_in_place_publication_advances_across_replays(self):
        """Advance the trajectory at every replay: the published state is not frozen."""
        self.assertGreater(len(set(self._run(replays=40))), 30)

    def test_in_place_publication_is_deterministic(self):
        """Two independent captures replay identically."""
        self.assertEqual(self._run(replays=20), self._run(replays=20))


# =====================================================================================================================
# The device proof
# =====================================================================================================================


def _proof_arrays(device):
    rng = np.random.default_rng(3)
    # One array shorter than the thread count, one spanning many strides, and a float array.
    return {
        "short": wp.array(rng.integers(0, 1 << 30, 7, dtype=np.int32), dtype=wp.int32, device=device),
        "long": wp.array(rng.integers(0, 1 << 30, 40 * _WORDS_PER_THREAD + 5, dtype=np.int32), device=device),
        "floats": wp.array(rng.normal(size=(300, 3)).astype(np.float32), dtype=wp.vec3, device=device),
    }


@unittest.skipUnless(wp.is_cuda_available(), "the device proof sums on CUDA")
class TestFeatherPGSDeviceProof(unittest.TestCase):
    def _arrays(self, device):
        return _proof_arrays(device)

    def test_unchanged_arrays_validate_and_any_change_is_seen(self):
        device = wp.get_device("cuda:0")
        arrays = self._arrays(device)
        proof = DeviceProof(list(arrays.items()), device)
        proof.validate(arrays.__getitem__, "test")
        for label in arrays:
            with self.subTest(label=label, change="word"):
                original = arrays[label].numpy().copy()
                changed = original.copy()
                flat = changed.reshape(-1)
                flat[len(flat) // 2] += 1
                wp.copy(arrays[label], wp.array(changed, dtype=arrays[label].dtype, device=device))
                with self.assertRaisesRegex(RuntimeError, label):
                    proof.validate(arrays.__getitem__, "test")
                wp.copy(arrays[label], wp.array(original, dtype=arrays[label].dtype, device=device))
            proof.validate(arrays.__getitem__, "test")
        with self.subTest(change="swap"):
            original = arrays["long"].numpy().copy()
            swapped = original.copy()
            swapped[[1, 1000]] = swapped[[1000, 1]]
            wp.copy(arrays["long"], wp.array(swapped, dtype=wp.int32, device=device))
            with self.assertRaisesRegex(RuntimeError, "long"):
                proof.validate(arrays.__getitem__, "test")

    def test_resized_array_is_rejected(self):
        device = wp.get_device("cuda:0")
        arrays = self._arrays(device)
        proof = DeviceProof(list(arrays.items()), device)
        arrays["short"] = wp.zeros(8, dtype=wp.int32, device=device)
        with self.assertRaisesRegex(RuntimeError, "short"):
            proof.validate(arrays.__getitem__, "test")


class TestFeatherPGSDeviceProofWatched(unittest.TestCase):
    """Watched arrays (joint frames, which owners read live) are reported and stored instead of raising."""

    def _check(self, device):
        arrays = _proof_arrays(device)
        proof = DeviceProof(list(arrays.items()), device, watched=("long", "floats"))

        def write(label, offset, change=1):
            values = arrays[label].numpy().copy()
            values.reshape(-1)[offset] += change
            wp.copy(arrays[label], wp.array(values, dtype=arrays[label].dtype, device=device))

        validate = arrays.__getitem__
        self.assertEqual(proof.validate(validate, "test"), [])
        write("floats", 5)
        self.assertEqual(proof.validate(validate, "test"), ["floats"])
        # The changed content is the reference now.
        self.assertEqual(proof.validate(validate, "test"), [])
        write("floats", 7)
        write("long", 3)
        self.assertEqual(proof.validate(validate, "test"), ["long", "floats"])
        self.assertEqual(proof.validate(validate, "test"), [])
        # A frozen change raises before any watched change is stored.
        write("floats", 9)
        write("short", 2)
        with self.assertRaisesRegex(RuntimeError, "short"):
            proof.validate(validate, "test")
        write("short", 2, -1)
        self.assertEqual(proof.validate(validate, "test"), ["floats"])
        # A watched array whose size changed raises as a frozen one does.
        arrays["long"] = wp.zeros(9, dtype=wp.int32, device=device)
        with self.assertRaisesRegex(RuntimeError, "long"):
            proof.validate(validate, "test")

    def test_watched_arrays_are_reported_and_stored_on_the_host(self):
        """Report and store changed watched arrays by host fingerprint, as on the CPU; raise on frozen ones."""
        self._check(wp.get_device("cpu"))

    @unittest.skipUnless(wp.is_cuda_available(), "the device proof sums on CUDA")
    def test_watched_arrays_are_reported_and_stored_on_the_device(self):
        """Report and store changed watched arrays by device checksum; raise on frozen ones."""
        self._check(wp.get_device("cuda:0"))


def report() -> None:
    """Print the walk's and the device walk's measured tables."""
    report_walk()
    report_device()


if __name__ == "__main__":
    if sys.argv[1:2] == ["report"]:
        wp.config.quiet = True
        report()
    else:
        unittest.main()
