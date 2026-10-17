# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Topology and subtree-reduction controls for the packed-chain tree schedule."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import tree_chain_scan, tree_kinetic_state


@wp.func_native("float* s=reinterpret_cast<float*>(address);" + tree_chain_scan.reduction_source(moments=False))
def _reduce_wrenches(address: wp.uint64, plan: tree_chain_scan.ChainPlan): ...


# The owner's per-tree storage mallocs on CPU, which this Warp's CPU JIT cannot
# resolve; the reduction only needs a scratch address, so pass the test's own.
@wp.func_native("return reinterpret_cast<uint64_t>(scratch.data);")
def _scratch_address(scratch: wp.array[float]) -> wp.uint64: ...


@wp.kernel
def _subtrees(
    plan: tree_chain_scan.ChainPlan,
    values: wp.array2d[float],
    scratch: wp.array[float],
    result: wp.array2d[float],
):
    _, logical = wp.tid()
    lanes = tree_kinetic_state._lanes()
    lane, stride = lanes[0], lanes[1]
    if stride == 1 and logical != 0:
        return
    address = _scratch_address(scratch)
    for body in range(lane, 44, stride):
        value = wp.spatial_vector()
        for k in range(6):
            value[k] = values[body, k]
        _store_wrench(address, body, value)
    tree_kinetic_state._sync()
    _reduce_wrenches(address, plan)
    for body in range(lane, 44, stride):
        value = _load_wrench(address, body)
        for k in range(6):
            result[body, k] = value[k]


@wp.func_native("float* s=reinterpret_cast<float*>(address);for(int k=0;k<6;++k)s[6*body+k]=value[k];")
def _store_wrench(address: wp.uint64, body: int, value: wp.spatial_vector): ...


@wp.func_native(
    "float* s=reinterpret_cast<float*>(address);wp::spatial_vector value;"
    "for(int k=0;k<6;++k)value[k]=s[6*body+k];return value;"
)
def _load_wrench(address: wp.uint64, body: int) -> wp.spatial_vector: ...


# One body per joint, parents first: a 44-body humanoid tree with a long and a branched limb.
_TREE_44 = np.asarray(
    [
        -1,
        0,
        1,
        2,
        3,
        4,
        5,
        0,
        0,
        8,
        9,
        10,
        11,
        12,
        0,
        14,
        14,
        14,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        22,
        25,
        22,
        27,
        28,
        14,
        14,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        36,
        39,
        36,
        41,
        42,
    ],
    dtype=np.int32,
)


def _chain_plan(parent, device):
    """Bind a chain plan from its host schedule as the held-tree owner does."""
    host = tree_chain_scan.make_plan(parent)
    plan = tree_chain_scan.ChainPlan()
    plan.chain_meta = wp.array(host["meta"].reshape(-1), dtype=int, device=device)
    for name in ("light_offsets", "light_children", "stage_width"):
        setattr(plan, name, wp.array(host[name], dtype=int, device=device))
    plan.chain_levels = host["levels"]
    plan.chain_width = int(host["widths"].max())
    return plan


class TestChainPlan(unittest.TestCase):
    def test_current_tree_packs_without_cross_warp_segments(self):
        parent = _TREE_44
        plan = tree_chain_scan.make_plan(parent)
        self.assertEqual(plan["levels"], 3)
        meta = plan["meta"]
        np.testing.assert_array_equal(np.sort(meta[meta[:, 0] >= 0, 0]), np.arange(44))
        self.assertEqual(int(np.sum(plan["widths"])), 60)
        self.assertEqual(len(plan["widths"]), 12)
        for lane, (body, start, length, level, anchor) in enumerate(meta):
            if body < 0:
                continue
            self.assertEqual(lane // 32, int(start) // 32)
            self.assertLess(lane - start, length)
            self.assertGreaterEqual(level, 0)
            self.assertEqual(anchor < 0, level == 0)

    def test_generic_tree_and_unsupported_packing(self):
        plan = tree_chain_scan.make_plan(np.asarray([-1, 0, 1, 0, 3, 3], np.int32))
        self.assertEqual(plan["levels"], 2)
        np.testing.assert_array_equal(np.sort(plan["meta"][plan["meta"][:, 0] >= 0, 0]), np.arange(6))
        with self.assertRaises(ValueError):
            tree_chain_scan.make_plan(np.arange(-1, 33, dtype=np.int32))
        with self.assertRaises(ValueError):
            tree_chain_scan.make_plan(np.asarray([-1, 2, 1], np.int32))

    def test_additive_subtrees_do_not_subtract_unrelated_large_branch(self):
        """An unrelated large wrench cannot erase a small descendant subtree."""
        parent = _TREE_44
        values = np.zeros((44, 6), np.float32)
        values[6, 0], values[40, 0] = 1e20, 1.25
        values[38, 1], values[43, 1] = 0.5, -0.25
        expected = values.astype(np.float64)
        for body in range(43, 0, -1):
            expected[parent[body]] += expected[body]
        result = wp.zeros((44, 6), dtype=float, device="cpu")
        wp.launch_tiled(
            _subtrees,
            dim=[1],
            inputs=[
                _chain_plan(parent, "cpu"),
                wp.array(values, dtype=float, device="cpu"),
                wp.zeros(44 * 6, dtype=float, device="cpu"),
                result,
            ],
            block_dim=64,
            device="cpu",
        )
        np.testing.assert_allclose(result.numpy(), expected, rtol=2e-6, atol=1e-7)
        self.assertEqual(float(result.numpy()[39, 0]), 1.25)


if __name__ == "__main__":
    unittest.main()
