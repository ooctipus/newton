# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The tree plan and the tree scan against every tree pass and builder they replace.

* :class:`TestTreePlan`: ``TreePlan.build`` reproduces each host builder's
  structures (depth-first intervals, the held tree's jump table and children,
  the heavy-chain packing, the sparse factor pattern and its support classes,
  articulation depths, body supports, local parents).
* :class:`TestHeldTreePasses`: on CUDA, every held-tree pass (pose and motion
  scans, subtree and external reductions, the twist publication) on every
  schedule it runs, as its existing text and as its ``tree_scan``
  configuration, on the existing plan structs and on the scan plan: bit for
  bit, and close to a float64 reference.
* :class:`TestSerialTwin`: the CPU path of every scan is the serial schedule:
  bitwise with the existing CPU twins that already walk levels, within a few
  units of the terms of those that emulate pointer jumping or chains.
* :class:`TestScheduleRule`: the schedule follows from the plan's data.
* :class:`TestPath`: a PATH visits the root path, natively and in Warp.
* :class:`TestGeneralRoute`: the general route's passes as scan
  configurations against the existing serial kernels, and their gradients.
"""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import tree_chain_scan
from newton._src.solvers.feather_pgs import tree_scan as ts
from newton._src.solvers.feather_pgs.sparse_mass_matrix import _SparseMassMatrixPlan, _SparseRowSupportPlan
from newton._src.solvers.feather_pgs.tree_kinetic_owner import dfs_intervals
from newton._src.solvers.feather_pgs.tree_plan import TreePlan, local_parents
from newton.tests.feather_pgs_law_harness import term_ulps
from newton.tests.feather_pgs_tree_scan_fixture import (
    EXTERNAL,
    MOTION,
    PASSES,
    POSE,
    SERIAL_PASSES,
    SHAPES,
    TWIST_SUM,
    applies,
    compare,
    lanes_of,
    layout_of,
    plan_of,
    random_scratch,
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
        """tree_kinetic_owner.dfs_intervals: children and roots ascending, each subtree one interval."""
        for parent, dofs in self.forests():
            plan = TreePlan.build(parent, dofs)
            order, end = dfs_intervals(parent)
            np.testing.assert_array_equal(plan.dfs_order, order)
            np.testing.assert_array_equal(plan.dfs_end, end)

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
        """tree_chain_scan.make_plan: the same lanes, light children, levels and widths, or the same refusal."""
        for parent, dofs in self.forests():
            if np.count_nonzero(parent < 0) != 1:
                continue
            plan = TreePlan.build(parent, dofs)
            packing = plan.chain_packing(2)
            try:
                reference = tree_chain_scan.make_plan(parent)
            except ValueError:
                self.assertIsNone(packing)
                continue
            np.testing.assert_array_equal(packing.meta, reference["meta"])
            np.testing.assert_array_equal(plan.light_offsets, reference["light_offsets"])
            np.testing.assert_array_equal(plan.light_children, reference["light_children"])
            np.testing.assert_array_equal(packing.stage_width, reference["stage_width"])
            self.assertEqual(packing.levels, reference["levels"])
            self.assertEqual(packing.width, int(reference["widths"].max()))

    def test_factor_pattern_and_supports(self):
        """_SparseMassMatrixPlan.build field by field, and _SparseRowSupportPlan.build's classes."""
        rng = np.random.default_rng(1)
        for parent, dofs in self.forests():
            if dofs.sum() > 64:
                continue
            plan = TreePlan.build(parent, dofs)
            mass = _SparseMassMatrixPlan.build(parent, dofs)
            factor = plan.factor
            for name in (
                "permutation",
                "inverse_permutation",
                "row_offsets",
                "columns",
                "entry_rows",
                "lookup",
                "dof_joint",
                "ancestor_mask",
                "joint_ancestor_mask",
                "column_offsets",
                "column_entries",
                "factor_level_offsets",
                "factor_level_columns",
                "factor_level_entry_offsets",
                "factor_level_entries",
            ):
                np.testing.assert_array_equal(getattr(factor, name), getattr(mass, name), err_msg=name)
            # mixed_tree_kinetic's physical-index map, built entry by entry there.
            physical = np.full((mass.dof_count, mass.dof_count), -1, np.int32)
            for index, (row, col) in enumerate(zip(mass.entry_rows, mass.columns, strict=True)):
                a, b = mass.permutation[row], mass.permutation[col]
                physical[a, b] = physical[b, a] = index
            np.testing.assert_array_equal(factor.physical_lookup, physical)
            body_masks = mass.joint_ancestor_mask[rng.integers(0, parent.size, parent.size)]
            for width in (min(4, mass.dof_count), mass.dof_count):
                ours = plan.support_classes(body_masks, width)
                reference = _SparseRowSupportPlan.build(mass, body_masks, max_row_dofs=width)
                self.assertEqual(ours is None, reference is None)
                if reference is not None:
                    for name in ("body_class", "pair_class", "class_dof", "limit_class"):
                        np.testing.assert_array_equal(getattr(ours, name), getattr(reference, name), err_msg=name)

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
            np.testing.assert_array_equal(ours[a], np.where(slice_ < 0, -1, slice_ - start))  # mixed_tree_kinetic
            np.testing.assert_array_equal(ours[a], np.where(slice_ >= start, slice_ - start, -1))  # the solver
        # A range that starts inside a tree leaves its first joint's ancestor outside: not a group of whole trees.
        with self.assertRaises(ValueError):
            local_parents(ancestor, starts + 1, 8)

    def test_global_forest(self):
        """A whole model's forest: tree depths (conditioning.articulation_depths) and coordinate supports."""
        rng = np.random.default_rng(3)
        parent, dofs = [], []
        for _ in range(40):
            start = len(parent)
            local = _random_forest(rng, int(rng.integers(1, 12)), 1)
            parent.extend(np.where(local >= 0, local + start, -1))
            dofs.extend(rng.integers(0, 3, local.size))
        parent, dofs = np.asarray(parent), np.asarray(dofs)
        plan = TreePlan.build(parent, dofs)
        depth = np.ones(parent.size, np.int64)
        for _ in range(parent.size):
            deeper = np.where(parent >= 0, depth[np.maximum(parent, 0)] + 1, 1)
            if np.array_equal(deeper, depth):
                break
            depth = deeper
        tree = np.zeros(parent.size, np.int64)
        root = plan.root_of
        np.maximum.at(tree, root, depth)
        np.testing.assert_array_equal(plan.tree_depths(), tree[plan.roots])
        # body_mapping.build_body_maps' support: the coordinate bits of a body's joint and its ancestors,
        # counted from its tree's first coordinate.
        start = np.concatenate(([0], np.cumsum(dofs)))
        for node in rng.integers(0, parent.size, 50):
            bits, walk = 0, int(node)
            while walk >= 0:
                for dof in range(start[walk], start[walk + 1]):
                    bits |= 1 << int(dof - start[root[node]])
                walk = int(parent[walk])
            self.assertEqual(int(plan.coordinate_mask[node]), bits)

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
            lanes = lanes_of(plan, packed)
            if packed and lanes >= 32:
                continue
            for name, item in PASSES.items():
                if not applies(item, plan, lanes) or (item.plan_struct == "chain" and packed):
                    continue
                for flag in (0, 1) if item.direction is ts.Direction.UP else (1,):
                    yield shape, plan, lanes, name, item, flag


def _assert_close_to_reference(test, item, plan, layout, start, result, flag, groups, twists=None, data=None):
    if item.combine is TWIST_SUM:
        rng = np.random.default_rng(3 + 1)
        axes = rng.standard_normal((groups * plan.count, 6)).astype(np.float32).reshape(groups, plan.count, 6)
        vhat = rng.standard_normal(groups * plan.count).astype(np.float32).reshape(groups, plan.count)
        expected, _ = reference(item, plan, layout, start, flag, vhat, axes)
        got = twists.reshape(groups, plan.count, 6)
        scale = np.abs(expected).max() + 1.0
    else:
        expected, _ = reference(item, plan, layout, start, flag)
        got = result.reshape(start.shape)
        scale = np.abs(expected).max() + 1.0
    error = np.abs(got.astype(np.float64) - expected).max() / scale
    test.assertLess(error, 64 * plan.count * EPS, f"{item.name} on {plan.count} nodes")


@unittest.skipUnless(wp.is_cuda_available(), "The held tree's schedules are CUDA paths")
class TestHeldTreePasses(unittest.TestCase):
    def test_generated_scans_are_the_existing_passes(self):
        """Every pass, shape and lane width: existing text == generated on the held structs == generated on the
        scan plan, bit for bit; all close to a float64 reference."""
        device = wp.get_cuda_device()
        groups = 64
        for shape, plan, lanes, name, item, flag in _cases():
            with self.subTest(shape=shape, lanes=lanes, scan=name, flag=flag):
                layout = layout_of(plan, lanes)
                results = compare(item, plan, layout, groups, flag, 3, device)
                existing = results["existing"]
                for arm in ("generated-held", "generated-scan"):
                    np.testing.assert_array_equal(existing[0].view(np.uint32), results[arm][0].view(np.uint32))
                    np.testing.assert_array_equal(existing[1].view(np.uint32), results[arm][1].view(np.uint32))
                start = random_scratch(layout, groups, 3)
                _assert_close_to_reference(
                    self, item, plan, layout, start, existing[0], flag, groups, twists=existing[1]
                )


class TestSerialTwin(unittest.TestCase):
    """The CPU path of every scan is the serial schedule."""

    def test_cpu_twins(self):
        level_twins = {"reduce_level", "reduce_interval", "external_level"}
        for shape, plan, lanes, name, item, flag in _cases(packed_widths=(0,)):
            if name == "twist_shfl":
                continue  # CUDA only: the classifier's twists are published from warp-sized plans.
            with self.subTest(shape=shape, scan=name, flag=flag):
                layout = layout_of(plan, lanes)
                results = compare(item, plan, layout, 4, flag, 3, "cpu")
                existing, generated = results["existing"][0], results["generated-scan"][0]
                np.testing.assert_array_equal(results["generated-held"][0].view(np.uint32), generated.view(np.uint32))
                if name in level_twins:
                    # These CPU twins already walk levels, children descending: the serial order.
                    np.testing.assert_array_equal(existing.view(np.uint32), generated.view(np.uint32))
                else:
                    # The old twins emulate pointer jumping or chains: the same combines, regrouped.
                    scale = np.abs(existing).reshape(4, -1).max(axis=1)[:, None]
                    gap = term_ulps(existing.reshape(4, -1), generated.reshape(4, -1), scale)
                    test_bound = 8.0 * max(1, plan.max_depth)
                    self.assertLess(float(gap.max()), test_bound)
                start = random_scratch(layout, 4, 3)
                _assert_close_to_reference(self, item, plan, layout, start, generated, flag, 4)


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
# The general route
# ---------------------------------------------------------------------------------------------------------------------


@wp.struct
class _PathVisits:
    rank: wp.array[int]
    visits: wp.array[int]
    count: int


@wp.func
def _record_visit(ancestor: int, node: int, value: int, d: _PathVisits):
    d.visits[node * d.count + ancestor] = wp.atomic_add(d.rank, node, value) + value


_RECORD_PATH = ts.serial_path(_record_visit)


@wp.kernel
def _warp_path(root: int, order: wp.array[int], position: wp.array[int], end: wp.array[int], d: _PathVisits):
    node = wp.tid()
    _RECORD_PATH(root, node, 1, order, position, end, d)


def check_path(test, device):
    """A PATH visits a node's ancestors and the node, root first: the native walk (parent-first plans and
    others) and the Warp walk, against the parent pointers."""
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

            @wp.func_native("    int rank=0;\n" + ts.native_scan(scan, 1))
            def native_path(plan: ts.ScanPlan, node: int, count: int, visits: wp.array[int]): ...

            def run(plan: ts.ScanPlan, count: int, visits: wp.array[int]):
                native_path(plan, wp.tid(), count, visits)

            visits = wp.zeros(n * n, dtype=int, device=device)
            kernel = wp.kernel(enable_backward=False, module="unique")(run)
            wp.launch(kernel, n, inputs=[ts.device_plan(plan, device), n, visits], device=device)
            np.testing.assert_array_equal(visits.numpy().reshape(n, n), expected, err_msg=f"native {shape} {label}")
            d = _PathVisits()
            d.rank = wp.zeros(n, dtype=int, device=device)
            d.visits = wp.zeros(n * n, dtype=int, device=device)
            d.count = n
            arrays = [
                wp.array(a.astype(np.int32), dtype=int, device=device)
                for a in (plan.dfs_order, plan.dfs_position, plan.dfs_end)
            ]
            wp.launch(_warp_path, n, inputs=[int(plan.roots[0]), *arrays, d], device=device)
            np.testing.assert_array_equal(d.visits.numpy().reshape(n, n), expected, err_msg=f"Warp {shape} {label}")


def check_general_route(test, device):
    """The serial kernels and their scan configurations, one serial scan per pass with fused per-node laws.

    Tau, composite inertia (all three forms) and CRBA keep their arithmetic order: bitwise, but for the
    entries the existing CRBA kernel races on (CUDA). FK and ID compose a joint's pose and twist as the held
    tree does, ``X_p (X_pj X_j X_c^-1)`` and ``a_p + v_p x v_j``, where the serial kernels compute
    ``((X_p X_pj) X_j) X_c^-1`` and ``a_p + (v_p + v_j) x v_j``: equal in exact arithmetic, within a few
    units of the largest value on a 64-joint chain.
    """
    from newton.tests.feather_pgs_tree_scan_fixture import (  # noqa: PLC0415
        articulated_model,
        crba_shared,
        serial_compare,
    )

    model = articulated_model(("anymal13", "humanoid13", "g1_30", "chain64", "random60", "pendulum2"), 2, device)
    results = serial_compare(model)
    cuda = wp.get_device(device).is_cuda
    test.assertEqual(set(results), {p for p in SERIAL_PASSES if cuda or not p.startswith("composite_inertia_")})
    for key, output in (
        ("tau", "tau"),
        ("composite_inertia", "body_I_c"),
        ("composite_inertia_warp", "body_I_c"),
        ("composite_inertia_compact", "body_I_c"),
    ):
        if key in results:
            a, b = results[key][output]
            np.testing.assert_array_equal(a.view(np.uint32), b.view(np.uint32), err_msg=key)
    # CRBA: in the existing kernel two column threads of one multi-coordinate joint both write the joint's
    # off-diagonal entries, S_i.(I S_j) and S_j.(I S_i), and the schedule decides which lands. The scan
    # writes each once, the later column's as a serial run leaves it: bitwise on CPU; on CUDA bitwise but at
    # those entries, there within rounding, and exactly symmetric.
    a, b = results["crba"]["H"]
    shared, mirror = crba_shared(model)
    np.testing.assert_array_equal(b.view(np.uint32)[mirror], b.view(np.uint32), err_msg="crba symmetry")
    if cuda:
        np.testing.assert_array_equal(a.view(np.uint32)[~shared], b.view(np.uint32)[~shared], err_msg="crba")
        gap = np.abs(a[shared].astype(np.float64) - b[shared]).max(initial=0.0) / np.abs(a).max() / EPS
        test.assertLess(gap, 8.0, "crba shared entries")
    else:
        np.testing.assert_array_equal(a.view(np.uint32), b.view(np.uint32), err_msg="crba")
    for key, bound in (("fk_kinematics", 32.0), ("fk_id", 96.0)):
        for output, (existing, generated) in results[key].items():
            scale = max(float(np.abs(existing).max()), 1e-30)
            gap = float(np.abs(existing.astype(np.float64) - generated).max()) / scale / EPS
            test.assertLess(gap, bound, f"{key} {output}")


def _unit_quaternion_tangent(model, gradient):
    """``gradient`` with respect to ``joint_q`` less its components normal to each ball and free joint's unit
    quaternion: the directions a joint can move in."""
    import newton  # noqa: PLC0415

    q, q_start = model.joint_q.numpy(), model.joint_q_start.numpy()
    tangent = gradient.astype(np.float64).copy()
    for joint, kind in enumerate(model.joint_type.numpy()):
        if kind not in (int(newton.JointType.BALL), int(newton.JointType.FREE)):
            continue
        first = q_start[joint] + (3 if kind == int(newton.JointType.FREE) else 0)
        unit = q[first : first + 4] / np.linalg.norm(q[first : first + 4])
        tangent[first : first + 4] -= np.dot(tangent[first : first + 4], unit) * unit
    return tangent


def check_general_route_gradients(test, device):
    """The general route is also the gradient route: Warp's tape through each generated pass.

    Tau and composite inertia give the existing kernels' gradients (bitwise, but tau's wrench and pose inputs
    within rounding). FK and ID match central differences; the arms agree on joint velocities, masses and
    inertias within rounding, and on joint positions along the unit quaternions: off the unit sphere the
    rotation formula is not multiplicative, so the association of the poses' compositions shows there. CRBA
    matches central differences, where the existing kernel's gradient is zero (the tape does not see the joint
    its in-kernel search finds).
    """
    from newton.tests.feather_pgs_tree_scan_fixture import (  # noqa: PLC0415
        GRADIENT_INPUTS,
        articulated_model,
        serial_difference,
        serial_gradients,
    )

    model = articulated_model(("anymal13", "humanoid13", "pendulum2"), 1, device, requires_grad=True)
    rng = np.random.default_rng(7)

    def relative(a, b):
        return float(np.abs(a.astype(np.float64) - b).max() / max(np.abs(b).max(), 1e-30))

    def against_differences(name, key, step, bound):
        gradient = serial_gradients(model, name, "generated")[key].reshape(-1)
        picks = rng.choice(gradient.size, min(gradient.size, 24), replace=False)
        difference = serial_difference(model, name, "generated", key, step, indices=picks).reshape(-1)[picks]
        test.assertGreater(np.abs(difference).max(), 0.0, f"{name} {key}")
        test.assertLess(relative(gradient[picks], difference), bound, f"{name} {key} against differences")

    for name in GRADIENT_INPUTS:
        existing = serial_gradients(model, name, "existing")
        generated = serial_gradients(model, name, "generated")
        for key in GRADIENT_INPUTS[name]:
            a, b = existing[key], generated[key]
            if name == "composite_inertia" or (name == "tau" and key not in ("body_f_s", "body_f_ext", "body_q")):
                np.testing.assert_array_equal(a.view(np.uint32), b.view(np.uint32), err_msg=f"{name} {key}")
            elif name == "crba":
                continue
            elif key == "joint_q" and name != "tau":
                test.assertLess(
                    relative(_unit_quaternion_tangent(model, b), _unit_quaternion_tangent(model, a)), 1e-5, name
                )
            else:
                test.assertLess(relative(b, a), 1e-5, f"{name} {key}")
    for name in ("fk_kinematics", "fk_id"):
        against_differences(name, "joint_qd", 1e-2, 1e-3)
        against_differences(name, "joint_q", 1e-3, 2e-3)
    for key in ("joint_S_s", "body_I_c"):
        against_differences("crba", key, 1e-2, 1e-3)


class TestPath(unittest.TestCase):
    pass


class TestGeneralRoute(unittest.TestCase):
    pass


def _general_route(test, device):
    check_general_route(test, device)


def _general_route_gradients(test, device):
    check_general_route_gradients(test, device)


from newton.tests.unittest_utils import add_function_test, get_test_devices  # noqa: E402

add_function_test(TestPath, "test_path", check_path, devices=get_test_devices())
add_function_test(TestGeneralRoute, "test_general_route", _general_route, devices=get_test_devices())
add_function_test(
    TestGeneralRoute, "test_general_route_gradients", _general_route_gradients, devices=get_test_devices()
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
