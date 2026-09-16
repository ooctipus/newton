# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Topology and dispatch controls for the complete packed-chain tree schedule."""

import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import g1_chain_scan, g1_kinetic_state
from tools.fpgs_bench.test_g1_kinetic_state import DT, public_state, rotated_input, set_anchor
from tools.fpgs_bench.test_sparse_factor import fixture


@wp.func_native("float* s=reinterpret_cast<float*>(address);" + g1_chain_scan.reduction_source(moments=False))
def _reduce_wrenches(address: wp.uint64, plan: g1_chain_scan.ChainPlan): ...


@wp.kernel
def _subtrees(plan: g1_chain_scan.ChainPlan, values: wp.array2d[float], result: wp.array2d[float]):
    _, logical = wp.tid()
    lanes = g1_kinetic_state._lanes()
    lane, stride = lanes[0], lanes[1]
    if stride == 1 and logical != 0:
        return
    address = g1_kinetic_state._storage()
    for body in range(lane, 44, stride):
        value = wp.spatial_vector()
        for k in range(6):
            value[k] = values[body, k]
        _store_wrench(address, body, value)
    g1_kinetic_state._sync()
    _reduce_wrenches(address, plan)
    for body in range(lane, 44, stride):
        value = _load_wrench(address, body)
        for k in range(6):
            result[body, k] = value[k]
    g1_kinetic_state._release(address)


@wp.func_native("float* s=reinterpret_cast<float*>(address);for(int k=0;k<6;++k)s[6*body+k]=value[k];")
def _store_wrench(address: wp.uint64, body: int, value: wp.spatial_vector): ...


@wp.func_native(
    "float* s=reinterpret_cast<float*>(address);wp::spatial_vector value;"
    "for(int k=0;k<6;++k)value[k]=s[6*body+k];return value;"
)
def _load_wrench(address: wp.uint64, body: int) -> wp.spatial_vector: ...


class TestChainPlan(unittest.TestCase):
    def test_current_tree_packs_without_cross_warp_segments(self):
        parent = np.asarray(
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
        plan = g1_chain_scan.make_plan(parent)
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
        plan = g1_chain_scan.make_plan(np.asarray([-1, 0, 1, 0, 3, 3], np.int32))
        self.assertEqual(plan["levels"], 2)
        np.testing.assert_array_equal(np.sort(plan["meta"][plan["meta"][:, 0] >= 0, 0]), np.arange(6))
        with self.assertRaises(ValueError):
            g1_chain_scan.make_plan(np.arange(-1, 33, dtype=np.int32))
        with self.assertRaises(ValueError):
            g1_chain_scan.make_plan(np.asarray([-1, 2, 1], np.int32))

    def test_default_off_and_complete_cpu_owner(self):
        """Exercise both current/held force actions and complete next publication."""
        results = []
        for enabled in (False, True):
            with patch.dict(os.environ, {"FEATHER_PGS_G1_CHAIN_SCAN": str(int(enabled))}):
                case = fixture("cpu")
                owner = g1_kinetic_state.G1KineticState(case["owner"])
            self.assertEqual(owner.chain_scan, enabled)
            self.assertEqual(owner.finish_kernel.key.endswith("_chain"), enabled)
            self.assertEqual(owner.predictor_kernel.key.endswith("_chain"), enabled)
            self.assertIs(owner.plan._cls, g1_chain_scan.ChainPlan if enabled else g1_kinetic_state.KineticPlan)
            model, solver, sparse = case["model"], case["solver"], case["owner"]
            set_anchor(model)
            W = np.linalg.solve(np.linalg.cholesky(case["H"][::-1, ::-1]), np.eye(43))
            packed = W[sparse.host["row"], sparse.host["col"]][None].astype(np.float32)
            sparse.data.W.assign(packed)
            sparse.data.valid.fill_(1)
            control = model.control()
            control.joint_f.assign(np.linspace(-0.2, 0.4, 43, dtype=np.float32))
            epochs = []
            for epoch in range(2):
                state, output = model.state(), model.state()
                rotated_input(model, state, epoch=epoch)
                if epoch:
                    state.body_f.zero_()  # Preserve the original zero-external-force shortcut too.
                augmented = solver._prepare_augmented_state(state, output, control)
                augmented.joint_tau.assign(np.linspace(0.02, -0.01, 43, dtype=np.float32))
                owner.invalidate()
                owner.begin(state, augmented, state.joint_qd, DT, epoch == 0)
                owner.predict(state, augmented, control, state.joint_qd, DT)
                vhat, bias = solver.v_hat.numpy().copy(), owner.bias.numpy().copy()
                solver.v_out.assign(state.joint_qd.numpy() + np.linspace(0.1, -0.1, 43, dtype=np.float32))
                owner.finish(state, augmented, output, DT, epoch == 0)
                public_state(self, model, output)
                np.testing.assert_array_equal(owner.status.numpy(), [0])
                np.testing.assert_array_equal(sparse.data.W.numpy(), packed)
                epochs.append((vhat, bias, output.joint_q.numpy(), output.joint_qd.numpy(), owner.bias.numpy()))
            results.append(epochs)
        for old, new in zip(results[0], results[1], strict=True):
            for expected, actual in zip(old, new, strict=True):
                np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=3e-5)

    def test_additive_subtrees_do_not_subtract_unrelated_large_branch(self):
        """An unrelated large wrench cannot erase a small descendant subtree."""
        with patch.dict(os.environ, {"FEATHER_PGS_G1_CHAIN_SCAN": "1"}):
            case = fixture("cpu")
            owner = g1_kinetic_state.G1KineticState(case["owner"])
        parent = case["model"].joint_parent.numpy()[:44]
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
            inputs=[owner.plan, wp.array(values, dtype=float, device="cpu"), result],
            block_dim=64,
            device="cpu",
        )
        np.testing.assert_allclose(result.numpy(), expected, rtol=2e-6, atol=1e-7)
        self.assertEqual(float(result.numpy()[39, 0]), 1.25)


if __name__ == "__main__":
    unittest.main()
