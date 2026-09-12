# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the prepared-response local-solver source seam without a GPU."""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs.articulated_factor import allocate_factor_data
from newton._src.solvers.feather_pgs.articulated_response import (
    get_whitening_bridge_kernel,
    prepare_local_response_source,
    validate_whitening_layout,
)
from newton._src.solvers.feather_pgs.solver_feather_pgs import (
    _get_paired_hinv_jt_kernel,
    _get_pgs_solve_local_owned_kernel,
    _get_pgs_solve_paired_factor_kernel,
)


class TestPreparedResponseSource(unittest.TestCase):
    def test_actual_local_variants(self):
        """Replace only primary response preparation in all Franka local owners."""
        original = wp.func_native
        for paired, rows, lanes, mf in ((0, 9, 8, 0), (6, 20, 32, 0), (6, 40, 32, 12)):
            snippets = []

            def capture(source, *args, _snippets=snippets, **kwargs):
                _snippets.append(source)
                return original(source, *args, **kwargs)

            _get_pgs_solve_local_owned_kernel.cache_clear()
            with patch.object(wp, "func_native", capture):
                _get_pgs_solve_local_owned_kernel(
                    192,
                    rows,
                    9,
                    "sm_120",
                    paired_dof_count=paired,
                    lanes_per_world=lanes,
                    contact_capable=bool(paired),
                    persistent_queue=bool(paired),
                    warps_per_block=2,
                    mf_max_constraints=64 if mf else 0,
                    local_mf_max_constraints=mf,
                    dense_response_matrix=True,
                )
            self.assertEqual(len(snippets), 1)
            source = snippets[0]
            changed = prepare_local_response_source(source, 9)
            self.assertNotIn("s_L[", changed)
            self.assertIn("response[i] = L_group.data[group_j_base + row * 9 + i]", changed)
            if paired:
                self.assertIn("s_L_secondary[", changed)
            recurrence = "    for (int iteration = 0; iteration < iterations; ++iteration)"
            self.assertEqual(
                source[source.index(recurrence) :].replace("#undef s_L\n", ""),
                changed[changed.index(recurrence) :],
            )

    def test_reject_source_drift(self):
        """Reject a source that does not contain exactly the inherited preparation."""
        with self.assertRaises(ValueError):
            prepare_local_response_source("not the local solver", 9)

    def test_primary_upper_factor_sources(self):
        """Reverse only primary triangle ownership in the inherited paired kernels."""
        original = wp.func_native
        configurations = (
            (_get_paired_hinv_jt_kernel, (23, 6, 192, 29, "sm_120"), {"factor_coordinates": True}),
            (_get_pgs_solve_paired_factor_kernel, (192, 29, 23, 6, "sm_120"), {}),
        )
        for factory, args, kwargs in configurations:
            sources = []

            def capture(source, *args, _sources=sources, **kwargs):
                _sources.append(source)
                return original(source, *args, **kwargs)

            factory.cache_clear()
            with patch.object(wp, "func_native", capture):
                factory(*args, **kwargs)
                factory(*args, **kwargs, primary_upper_factor=True)
            self.assertEqual(len(sources), 2)
            before, after = sources
            if factory is _get_paired_hinv_jt_kernel:
                expected = before.replace("lane < 23 && k <= lane", "lane < 23 && k >= lane")
                expected = expected.replace(
                    "lane < 23 && k >= lane)\n                    y", "lane < 23 && k <= lane)\n                    y"
                )
            else:
                self.assertEqual(before.count("primary_lane <= k"), 2)
                expected = before.replace("primary_lane <= k", "primary_lane >= k")
            self.assertEqual(expected, after)


class TestWhiteningBridge(unittest.TestCase):
    def test_branched_fixed_link_and_held_mask(self):
        """Check actual CPU bridge algebra, fixed ancestors, owner masks and row action."""
        wp.init()
        parents = np.array([-1, 0, 1, 2, 1], dtype=np.int32)
        slots = np.array([-1, 0, -1, 1, 2], dtype=np.int32)
        rng = np.random.default_rng(210)
        motion = rng.normal(size=(5, 6))
        motion[slots < 0] = 0
        raw = rng.normal(size=(5, 6, 6))
        inertia = raw @ raw.transpose(0, 2, 1) + np.eye(6)[None]
        diagonal = np.array([0.21, 0.32, 0.43])
        h = np.diag(diagonal)
        for body in range(5):
            jac = np.zeros((6, 3))
            ancestor = body
            while ancestor >= 0:
                if slots[ancestor] >= 0:
                    jac[:, slots[ancestor]] = motion[ancestor]
                ancestor = parents[ancestor]
            h += jac.T @ inertia[body] @ jac
        ia = inertia.copy()
        u = np.zeros((5, 6))
        inv_d = np.zeros(5)
        for body in reversed(range(5)):
            reduced = ia[body].copy()
            slot = slots[body]
            if slot >= 0:
                u[body] = ia[body] @ motion[body]
                inv_d[body] = 1.0 / (motion[body] @ u[body] + diagonal[slot])
                reduced -= np.outer(u[body], u[body]) * inv_d[body]
            if parents[body] >= 0:
                ia[parents[body]] += reduced
        factor = allocate_factor_data(
            [7, 2],
            np.tile(np.arange(5), (2, 1)),
            np.tile(parents, (2, 1)),
            np.tile(slots, (2, 1)),
            np.tile(slots, (2, 1)),
            [5, 5],
            "cpu",
        )
        factor.S.assign(np.tile(motion.astype(np.float32), (2, 1, 1)))
        factor.U.assign(np.tile(u.astype(np.float32), (2, 1, 1)))
        factor.invD.assign(np.tile(inv_d.astype(np.float32), (2, 1)))
        factor.valid.fill_(1)
        validate_whitening_layout(factor, 3)
        et = wp.full((2, 3, 3), 777.0, dtype=float, device="cpu")
        w = wp.full_like(et, 888.0)
        mask = wp.array(np.array([0, 0, 0, 0, 0, 0, 0, 1], np.int32), dtype=int, device="cpu")
        kernel = get_whitening_bridge_kernel(5, 3)
        wp.launch_tiled(kernel, dim=[2], inputs=[factor, mask, et, w], block_dim=32, device="cpu")
        et_np, w_np = et.numpy(), w.numpy()
        np.testing.assert_array_equal(et_np[1], 777.0)
        np.testing.assert_array_equal(w_np[1], 888.0)
        np.testing.assert_allclose(et_np[0] @ et_np[0].T, h, rtol=2e-6, atol=2e-5)
        np.testing.assert_allclose(w_np[0].T @ w_np[0], np.linalg.inv(h), rtol=3e-6, atol=1e-7)
        np.testing.assert_allclose(et_np[0] @ w_np[0], np.eye(3), rtol=1e-6, atol=1e-6)
        velocity = rng.normal(size=3)
        row = rng.normal(size=3)
        encoded = et_np[0].T @ velocity
        z = w_np[0] @ row
        self.assertAlmostEqual(float(z @ encoded), float(row @ velocity), places=6)
        np.testing.assert_allclose(w_np[0].T @ encoded, velocity, rtol=2e-6, atol=1e-6)
        # A reuse call leaves both transforms intact, even if current S changes.
        before_et, before_w = et_np.copy(), w_np.copy()
        factor.S.fill_(0.0)
        mask.zero_()
        wp.launch_tiled(kernel, dim=[2], inputs=[factor, mask, et, w], block_dim=32, device="cpu")
        np.testing.assert_array_equal(et.numpy(), before_et)
        np.testing.assert_array_equal(w.numpy(), before_w)
        factor.valid.zero_()
        mask.fill_(1)
        wp.launch_tiled(kernel, dim=[2], inputs=[factor, mask, et, w], block_dim=32, device="cpu")
        self.assertTrue(np.isnan(et.numpy()).all())
        self.assertTrue(np.isnan(w.numpy()).all())
        bad_slots = np.tile(slots, (2, 1))
        bad_slots[:, [1, 3]] = bad_slots[:, [3, 1]]
        factor.dof_slots.assign(bad_slots)
        with self.assertRaises(ValueError):
            validate_whitening_layout(factor, 3)


if __name__ == "__main__":
    unittest.main()
