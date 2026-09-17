# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check four independent full-warp worlds without changing row mathematics."""

import inspect
import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import (
    sparse_factor,
    sparse_factor_rows,
    sparse_fullwarp_group,
    sparse_spectral_tangents,
)
from tools.fpgs_bench import test_sparse_metric_tangents as metric
from tools.fpgs_bench import test_sparse_spectral_tangents as spectral
from tools.fpgs_bench.test_sparse_factor import fixture
from tools.fpgs_bench.test_sparse_spectral_tangents import ENV as SPECTRAL_ENV

ENV = {
    **SPECTRAL_ENV,
    "FEATHER_PGS_SPARSE_PARALLEL_LIMITS": "1",
    "FEATHER_PGS_SPARSE_FULLWARP_GROUP": "1",
}


class TestSparseFullwarpGroup(unittest.TestCase):
    def test_factories(self):
        """Keep both complete owner ABIs and expose distinct factory identities."""
        for original, candidate, key in (
            (
                sparse_spectral_tangents.get_solve_kernel(),
                sparse_fullwarp_group.get_solve_kernel(),
                "sparse_spectral_tangent43_s18_c100_w4",
            ),
            (
                sparse_factor_rows.get_parallel_limit_kernel(),
                sparse_fullwarp_group.get_prefix_kernel(),
                "sparse_factor_parallel_limit_prefix43_w4",
            ),
        ):
            self.assertEqual(candidate.key, key)
            self.assertEqual(inspect.signature(candidate.func), inspect.signature(original.func))
            self.assertIsNot(candidate, original)

    def test_source_ownership_only(self):
        """Change only indexing/storage while preserving all warp collectives."""
        for solve, original, candidate in (
            (True, sparse_spectral_tangents.get_solve_kernel(), sparse_fullwarp_group.get_solve_kernel()),
            (False, sparse_factor_rows.get_parallel_limit_kernel(), sparse_fullwarp_group.get_prefix_kernel()),
        ):
            before = original.func.__closure__[0].cell_contents.native_snippet
            after = candidate.func.__closure__[0].cell_contents.native_snippet
            self.assertEqual(after, sparse_fullwarp_group._group_source(before, solve))
            self.assertNotIn("__syncthreads", after)
            self.assertLess(after.index("if(group>="), after.index("p.group_to_art.data[group]"))
            for collective in ("__shfl_sync", "__shfl_down_sync", "__ballot_sync", "__syncwarp"):
                self.assertEqual(before.count(collective), after.count(collective))
            if solve:
                for declaration in (
                    "du_storage[4][43]",
                    "lam_storage[4][100]",
                    "contact_cross_storage[4][100]",
                    "contact_ready_storage[4][4]",
                ):
                    self.assertIn(declaration, after)
        with self.assertRaises(RuntimeError):
            sparse_fullwarp_group._group_source("__syncthreads();", True)

    def test_owner_admission(self):
        """Admit both grouped owners together and leave default dispatch unchanged."""
        with patch.dict(os.environ, ENV):
            owner = fixture("cpu")["owner"]
        self.assertTrue(owner.fullwarp_group)
        self.assertIs(owner.kernels.solve, sparse_fullwarp_group.get_solve_kernel())
        self.assertIs(owner.kernels.prefix, sparse_fullwarp_group.get_prefix_kernel())
        with patch.dict(os.environ, {**ENV, "FEATHER_PGS_SPARSE_FULLWARP_GROUP": "0"}):
            ordinary = sparse_factor.SparseFactor(owner.solver, owner.plan, owner.host)
        self.assertFalse(ordinary.fullwarp_group)
        self.assertIs(ordinary.kernels.solve, sparse_spectral_tangents.get_solve_kernel())
        self.assertIs(ordinary.kernels.prefix, sparse_factor_rows.get_parallel_limit_kernel())
        for name, value in (
            ("FEATHER_PGS_SPARSE_FULLWARP_GROUP", "2"),
            ("FEATHER_PGS_SPARSE_SPECTRAL_TANGENTS", "0"),
            ("FEATHER_PGS_SPARSE_PARALLEL_LIMITS", "0"),
        ):
            with self.subTest(name=name), patch.dict(os.environ, {**ENV, name: value}):
                with self.assertRaises(ValueError):
                    sparse_factor.SparseFactor(owner.solver, owner.plan, owner.host)


def check_native(test, f, **kwargs):
    """Compare the grouped owner against the independently checked spectral owner."""
    s, owner = f["solver"], f["owner"]
    test.assertTrue(owner.fullwarp_group)
    test.assertIs(owner.kernels.solve, sparse_fullwarp_group.get_solve_kernel())
    incoming = s.impulses.numpy().copy()
    owner.fullwarp_group = False
    owner.kernels.solve = sparse_spectral_tangents.get_solve_kernel()
    try:
        expected_v, expected_lam, stats = spectral.check_native(test, f, **kwargs)
    finally:
        owner.fullwarp_group = True
        owner.kernels.solve = sparse_fullwarp_group.get_solve_kernel()
    s.impulses.assign(incoming)
    s.v_out.fill_(np.nan)
    owner.solve(s.rhs, kwargs.get("iterations", 8), kwargs.get("omega", 1.0), kwargs.get("friction_start", 0))
    owner.check()
    count = int(s.constraint_count.numpy()[0])
    actual_v, actual_lam = s.v_out.numpy(), s.impulses.numpy()[0, :count]
    np.testing.assert_allclose(actual_v, expected_v, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(actual_lam, expected_lam, rtol=2e-6, atol=2e-6)
    return actual_v.copy(), actual_lam.copy(), stats


def _plan_and_data(owner, worlds, device, capacity=100):
    """Replicate physical factor inputs with deliberately nonidentity group maps."""
    plan = sparse_factor.SparsePlan()
    for name in owner.plan._cls.vars:
        setattr(plan, name, getattr(owner.plan, name))
    groups = np.arange(worlds, dtype=np.int32)[::-1].copy()
    world_map = np.roll(np.arange(worlds, dtype=np.int32), 1)
    plan.group_to_art = wp.array(groups, dtype=int, device=device)
    plan.art_to_world = wp.array(world_map, dtype=int, device=device)
    plan.art_dof_start = wp.array(np.arange(worlds, dtype=np.int32) * 43, dtype=int, device=device)
    data = sparse_factor.SparseData()
    held = np.repeat(owner.data.W.numpy(), worlds, axis=0)
    held *= (1 + np.arange(worlds, dtype=np.float32) * 0.005)[:, None]
    data.W = wp.array(held, dtype=float, device=device)
    data.Z = wp.zeros((worlds, capacity, 18), device=device)
    data.support = wp.zeros((worlds, capacity), dtype=int, device=device)
    data.incident = wp.zeros((worlds, capacity), device=device)
    data.valid = wp.ones(worlds, dtype=int, device=device)
    data.status = wp.zeros(worlds, dtype=int, device=device)
    return plan, data, groups, world_map


@unittest.skipUnless(wp.is_cuda_available(), "Root owns native execution")
class TestSparseFullwarpGroupCUDA(unittest.TestCase):
    def test_native_current_held_graph_and_empty(self):
        """Retain independent current/held physical and graph lifetime controls."""
        with patch.dict(os.environ, ENV), patch.object(metric, "check_native", check_native):
            metric.TestSparseMetricTangentsCUDA.test_native_current_held_graph_and_empty(self)

    def test_native_saved_sixteen(self):
        """Retain saved16 independent J/H, cone, momentum and current-row checks."""
        with patch.dict(os.environ, ENV), patch.object(metric, "check_native", check_native):
            metric.TestSparseMetricTangentsCUDA.test_native_saved_sixteen_current_held_epochs(self)

    def test_native_world_guards_graph_and_tail(self):
        """Isolate full warps at sizes1/3/4/5 with empty, maximum and rejected worlds."""
        device = "cuda:0"
        with patch.dict(os.environ, ENV):
            f = fixture(device)
        owner = f["owner"]
        owner.refresh(f["solver"])
        template = int(np.flatnonzero(owner.host["support_count"] == 18)[0])
        for worlds in (1, 3, 4, 5):
            with self.subTest(worlds=worlds):
                plan, data, groups, world_map = _plan_and_data(owner, worlds, device)
                count_host = np.array([3, 0, 100, 7, 13][:worlds], np.int32)
                counts = wp.array(count_host, dtype=int, device=device)
                z = np.zeros((worlds, 100, 18), np.float32)
                types = np.full((worlds, 100), 3, np.int32)
                parents = np.full((worlds, 100), -1, np.int32)
                rhs = np.zeros((worlds, 100), np.float32)
                mu = np.zeros_like(rhs)
                for world, count in enumerate(count_host):
                    prefix = count % 3
                    for row in range(count):
                        component = (row - prefix) % 3 if row >= prefix else 0
                        z[world, row, component] = 0.8 + world * 0.01
                        z[world, row, 16:] = (0.13, -0.08)
                        rhs[world, row] = (-0.15, 0.07, -0.09)[component]
                        if row >= prefix:
                            types[world, row] = 0 if component == 0 else 2
                            if component:
                                parents[world, row] = row - component
                                mu[world, row] = 0.6
                cfm = np.full_like(rhs, 0.01)
                diagonal = np.sum(z * z, axis=2) + cfm
                data.Z.assign(z)
                data.support.fill_(template)
                row_arrays = [
                    wp.array(a, dtype=dtype, device=device)
                    for a, dtype in (
                        (rhs, float),
                        (diagonal, float),
                        (cfm, float),
                        (types, int),
                        (parents, int),
                        (mu, float),
                    )
                ]
                vhat = wp.array(np.linspace(-0.03, 0.02, worlds * 43, dtype=np.float32), device=device)
                outputs = [
                    (wp.zeros((worlds, 100), device=device), wp.zeros(worlds * 43, device=device)) for _ in range(2)
                ]

                def launch(
                    arm,
                    worlds=worlds,
                    plan=plan,
                    data=data,
                    counts=counts,
                    row_arrays=row_arrays,
                    outputs=outputs,
                    vhat=vhat,
                ):
                    wp.launch_tiled(
                        sparse_spectral_tangents.get_solve_kernel()
                        if arm == 0
                        else sparse_fullwarp_group.get_solve_kernel(),
                        dim=[worlds if arm == 0 else (worlds + 3) // 4],
                        inputs=[
                            plan,
                            data,
                            counts,
                            *row_arrays[:3],
                            outputs[arm][0],
                            *row_arrays[3:],
                            8,
                            1.0,
                            0,
                            vhat,
                            outputs[arm][1],
                        ],
                        block_dim=32 if arm == 0 else 128,
                        device=device,
                    )

                for arm in range(2):
                    launch(arm)
                graphs = []
                for arm in range(2):
                    with wp.ScopedCapture(device=device) as capture:
                        launch(arm)
                    graphs.append(capture.graph)
                for epoch in range(3):
                    counts.assign(np.zeros_like(count_host) if epoch == 1 else count_host)
                    for impulse, velocity in outputs:
                        impulse.zero_()
                        velocity.fill_(19.0)
                    for graph in graphs:
                        wp.capture_launch(graph)
                    for a, b in zip(outputs[0], outputs[1], strict=True):
                        np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=2e-6, atol=2e-6)
                    self.assertTrue(np.isfinite(outputs[1][1].numpy()).all())
                # One rejected warp cannot prevent another warp from publishing.
                rejected = count_host.copy()
                rejected[0] = 101
                counts.assign(rejected)
                valid, status = np.ones(worlds, np.int32), np.zeros(worlds, np.int32)
                if worlds > 1:
                    valid[1] = 0
                if worlds > 2:
                    status[2] = 8
                data.valid.assign(valid)
                for arm in range(2):
                    data.status.assign(status)
                    outputs[arm][0].zero_()
                    outputs[arm][1].fill_(19.0)
                    wp.capture_launch(graphs[arm])
                for a, b in zip(outputs[0], outputs[1], strict=True):
                    np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=2e-6, atol=2e-6)
                for art in groups:
                    world = world_map[art]
                    if rejected[world] <= 100 and valid[world] and not status[world]:
                        self.assertFalse(np.all(outputs[1][1].numpy()[art * 43 : (art + 1) * 43] == 19.0))

    def test_native_prefix_worlds_and_capacity(self):
        """Preserve stable limit order, uncapped counts and disjoint mapped worlds."""
        device = "cuda:0"
        with patch.dict(os.environ, ENV):
            f = fixture(device)
        owner, solver = f["owner"], f["solver"]
        owner.refresh(solver)
        for worlds in (1, 3, 4, 5):
            for capacity in (7, 100):
                with self.subTest(worlds=worlds, capacity=capacity):
                    plan, data, _, _ = _plan_and_data(owner, worlds, device, capacity)
                    local_qi = solver._joint_limit_q_index.numpy()
                    qi = np.concatenate([np.where(local_qi >= 0, local_qi + 44 * art, -1) for art in range(worlds)])
                    limit_q = wp.array(qi.astype(np.int32), dtype=int, device=device)
                    lower = wp.full(worlds * 43, -0.001, device=device)
                    upper = wp.full(worlds * 43, 0.001, device=device)
                    q = wp.zeros(worlds * 44, device=device)
                    vhat = wp.array(np.linspace(-0.2, 0.3, worlds * 43, dtype=np.float32), device=device)
                    counter = wp.empty(worlds, dtype=int, device=device)
                    fields = [
                        wp.empty((worlds, capacity), dtype=dtype, device=device)
                        for dtype in (int, int, float, float, float, float, float, float)
                    ]
                    phase = wp.empty((worlds, 4), dtype=int, device=device)
                    outputs = [data.Z, data.incident, data.support, counter, *fields, phase]
                    for enabled in (1, 0):
                        args = [
                            plan,
                            data,
                            limit_q,
                            lower,
                            upper,
                            q,
                            vhat,
                            enabled,
                            0.01,
                            0.2,
                            1e-6,
                            counter,
                            *fields,
                            phase,
                        ]
                        expected = None
                        for arm in range(2):
                            for output in outputs:
                                output.fill_(-777)
                            wp.launch_tiled(
                                sparse_factor_rows.get_parallel_limit_kernel()
                                if arm == 0
                                else sparse_fullwarp_group.get_prefix_kernel(),
                                dim=[worlds if arm == 0 else (worlds + 3) // 4],
                                inputs=args,
                                block_dim=32 if arm == 0 else 128,
                                device=device,
                            )
                            if arm == 0:
                                expected = [output.numpy() for output in outputs]
                            else:
                                for output, reference in zip(outputs, expected, strict=True):
                                    np.testing.assert_array_equal(output.numpy(), reference)
                        np.testing.assert_array_equal(counter.numpy(), np.full(worlds, 74 if enabled else 0))


if __name__ == "__main__":
    unittest.main()
