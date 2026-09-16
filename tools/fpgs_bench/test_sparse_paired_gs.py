# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare two half-warp worlds against the unchanged sparse metric law."""

import inspect
import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import sparse_paired_gs
from newton._src.solvers.feather_pgs.sparse_factor import SparseData, SparsePlan
from newton._src.solvers.feather_pgs.sparse_factor_rows import get_solve_kernel
from tools.fpgs_bench import test_sparse_metric_tangents as metric
from tools.fpgs_bench.test_sparse_contact_block import saved_records
from tools.fpgs_bench.test_sparse_factor import fixture, physical_rows

KEY = "sparse_paired_metric_tangent43_s18_c100"
ENV = {
    **metric.METRIC_ENV,
    "FEATHER_PGS_BODY_BASIS_ROWS": "0",
    "FEATHER_PGS_SPARSE_PAIRED_GS": "0",
}


class TestSparsePairedGS(unittest.TestCase):
    def test_factories(self):
        """Require the optional owner and unchanged native argument contract."""
        self.assertTrue(callable(sparse_paired_gs.install))
        original = get_solve_kernel(100, metric_tangents=True)
        candidate = sparse_paired_gs.get_solve_kernel()
        self.assertEqual(candidate.key, KEY)
        self.assertEqual(inspect.signature(candidate.func), inspect.signature(original.func))

    def test_support_tail_reduction(self):
        """Check both extra coefficients in the fixed sixteen-lane reduction."""
        random = np.random.default_rng(37)
        for length in (17, 18):
            values = np.zeros(32, np.float32)
            values[:length] = random.normal(size=length)
            original = values.copy()
            for shift in (16, 8, 4, 2, 1):
                original[: 32 - shift] += original[shift:].copy()
            half = values[:16] + values[16:]
            for shift in (8, 4, 2, 1):
                half[: 16 - shift] += half[shift:].copy()
            self.assertEqual(half[0], original[0])


@unittest.skipUnless(wp.is_cuda_available(), "Paired sparse native controls require CUDA")
class TestSparsePairedGSCUDA(unittest.TestCase):
    def test_native_saved_sixteen_original_same_rows(self):
        """Compare unchanged original rows/GS against paired GS on pinned current/held inputs."""
        replay, records = saved_records()
        tested = 0
        for gpu, record, inputs in records:
            for index, world in enumerate(inputs["worlds"]):
                with self.subTest(gpu=gpu, step=record["step"], world=int(world)), patch.dict(os.environ, ENV):
                    case = replay.bind_world(record, inputs, index, "cuda:0")
                    solver, owner = case["solver"], case["owner"]
                    lower = np.linalg.cholesky(case["H"][::-1, ::-1])
                    inverse = np.linalg.solve(lower, np.eye(43))
                    owner.data.W.assign(inverse[owner.host["row"], owner.host["col"]][None].astype(np.float32))
                    owner.data.valid.fill_(1)
                    # The ORIGINAL producer owns all candidate inputs. Neither
                    # builder nor allocator runs again between the two solves.
                    owner.build_rows(case["state"], solver, case["contacts"], 0.0025)
                    solver.check_constraint_capacity()
                    solver._stage4_compute_rhs_world(0.0025)
                    owner.restitution(0.0025)
                    solver.impulses.zero_()
                    original_v, original_lambda, _ = metric.check_native(self, case)
                    held, rows = owner.data.W.numpy().copy(), owner.data.Z.numpy().copy()
                    count = int(solver.constraint_count.numpy()[0])
                    self.assertTrue(sparse_paired_gs.install(owner))
                    self.assertTrue(owner.paired_gs)
                    self.assertEqual(owner.kernels.solve.key, KEY)
                    solver.impulses.zero_()
                    solver.v_out.fill_(np.nan)
                    owner.solve(solver.rhs, 8, 1.0, 0)
                    owner.check()
                    actual, impulse = solver.v_out.numpy(), solver.impulses.numpy()[0, :count]
                    np.testing.assert_allclose(actual, original_v, rtol=3e-4, atol=3e-5)
                    np.testing.assert_allclose(impulse, original_lambda, rtol=3e-4, atol=3e-5)
                    self.assertTrue(np.isfinite(actual).all() and np.isfinite(impulse).all())
                    J = physical_rows(case)
                    delta = actual.astype(float) - solver.v_hat.numpy()
                    force = J.T @ impulse
                    defect = np.linalg.norm(case["H"] @ delta - force, np.inf) / (
                        1
                        + np.linalg.norm(case["H"], np.inf) * np.linalg.norm(delta, np.inf)
                        + np.linalg.norm(force, np.inf)
                    )
                    self.assertLess(defect, 2e-6)
                    diagonal, rhs, types, parents, mu = (
                        getattr(solver, name).numpy()[0, :count]
                        for name in ("diag", "rhs", "row_type", "row_parent", "row_mu")
                    )
                    physics = metric.physical_metrics(
                        J, diagonal, rhs, types, parents, mu, solver.v_hat.numpy(), actual, impulse
                    )
                    self.assertLess(physics["cone"], 3e-5)
                    np.testing.assert_array_equal(owner.data.W.numpy(), held)
                    np.testing.assert_array_equal(owner.data.Z.numpy(), rows)
                    tested += 1
        self.assertEqual(tested, 16)

    def test_native_five_worlds_mixed_support_status_and_graph(self):
        """Exercise actual half-warp isolation, secondary support, different half exits and odd-tail reuse."""
        device = "cuda:0"
        with patch.dict(os.environ, ENV):
            f = fixture(device)
        owner, solver = f["owner"], f["solver"]
        owner.refresh(solver)
        plan = SparsePlan()
        for name in owner.plan._cls.vars:
            setattr(plan, name, getattr(owner.plan, name))
        worlds = 5
        groups = np.arange(worlds, dtype=np.int32)[::-1].copy()
        world_map = np.roll(np.arange(worlds, dtype=np.int32), 2)
        plan.group_to_art = wp.array(groups, dtype=int, device=device)
        plan.art_to_world = wp.array(world_map, dtype=int, device=device)
        plan.art_dof_start = wp.array(np.arange(worlds, dtype=np.int32) * 43, dtype=int, device=device)
        nodes = owner.plan.support_nodes.numpy().reshape(-1, 18)
        lengths = owner.plan.support_count.numpy()
        full = int(np.flatnonzero(lengths == 18)[0])
        short = len(lengths)
        extra = nodes[full].copy()
        extra[17] = -1
        nodes = np.concatenate((nodes, extra[None]))
        plan.support_nodes = wp.array(nodes, dtype=int, device=device)
        plan.support_count = wp.array(np.append(lengths, 17).astype(np.int32), dtype=int, device=device)
        counts_host = np.array([0, 3, 4, 7, 13], np.int32)
        counts = wp.array(counts_host, dtype=int, device=device)
        z = np.zeros((worlds, 100, 18), np.float32)
        support = np.zeros((worlds, 100), np.int32)
        types = np.full((worlds, 100), 3, np.int32)
        parents = np.full((worlds, 100), -1, np.int32)
        rhs = np.zeros((worlds, 100), np.float32)
        mu = np.zeros_like(rhs)
        # Same row topology is retained during count withdrawal/regrowth.
        # Nonzero slots16/17 make a truncated half-warp reduction observably wrong.
        for world, count in enumerate(counts_host):
            width = 17 if world % 2 else 18
            support[world] = short if width == 17 else full
            prefix = count % 3
            for row in range(count):
                component = (row - prefix) % 3 if row >= prefix else 0
                z[world, row, component] = 0.8 + 0.01 * world
                z[world, row, 16] = (0.13, -0.08, 0.04)[component]
                if width == 18:
                    z[world, row, 17] = (-0.09, 0.06, 0.11)[component]
                rhs[world, row] = (-0.15, 0.07, -0.09)[component]
                if row >= prefix:
                    types[world, row] = 0 if component == 0 else 2
                    if component:
                        parents[world, row] = row - component
                        mu[world, row] = 0.6
            if world == 1:
                rhs[world, :count] = 0.2  # Stationary halves next to loaded worlds.
        compliance = np.full((worlds, 100), 0.01, np.float32)
        for world in (4,):
            normal = int(counts_host[world] % 3)
            z[world, normal + 2] = z[world, normal + 1]
            compliance[world, normal + 1 : normal + 3] = 0
        diagonal = np.sum(z * z, axis=2) + compliance
        for world, count in enumerate(counts_host):
            z[world, count:] = np.nan
            if world % 2:
                z[world, :count, 17] = np.nan
        shared = [
            wp.array(value, dtype=dtype, device=device)
            for value, dtype in (
                (rhs, float),
                (diagonal, float),
                (types, int),
                (parents, int),
                (mu, float),
            )
        ]
        vhat_host = np.random.default_rng(52).normal(0, 0.02, worlds * 43).astype(np.float32)
        vhat = wp.array(vhat_host, dtype=float, device=device)
        held = np.repeat(owner.data.W.numpy(), worlds, axis=0)
        factor_scale = 1 + np.arange(worlds, dtype=np.float32) * 0.005
        held *= factor_scale[:, None]
        independent_lower = np.linalg.cholesky(f["H"][::-1, ::-1])
        data, outputs = [], []
        for _ in range(2):
            d = SparseData()
            d.W = wp.array(held, dtype=float, device=device)
            d.Z = wp.array(z, dtype=float, device=device)
            d.support = wp.array(support, dtype=int, device=device)
            d.incident = wp.zeros((worlds, 100), dtype=float, device=device)
            d.valid = wp.ones(worlds, dtype=int, device=device)
            d.status = wp.zeros(worlds, dtype=int, device=device)
            data.append(d)
            outputs.append((wp.zeros((worlds, 100), device=device), wp.zeros(worlds * 43, device=device)))
        kernels = (get_solve_kernel(100, metric_tangents=True), sparse_paired_gs.get_solve_kernel())

        def launch(arm):
            impulse, velocity = outputs[arm]
            wp.launch_tiled(
                kernels[arm],
                dim=[worlds if arm == 0 else (worlds + 1) // 2],
                inputs=[
                    plan,
                    data[arm],
                    counts,
                    shared[0],
                    shared[1],
                    impulse,
                    shared[2],
                    shared[3],
                    shared[4],
                    8,
                    1.0,
                    0,
                    vhat,
                    velocity,
                ],
                block_dim=32,
                device=device,
            )

        for arm in range(2):
            launch(arm)
        graphs = []
        for arm in range(2):
            with wp.ScopedCapture(device=device) as capture:
                launch(arm)
            graphs.append(capture.graph)
        # Preserve all input bytes; only status and original output arrays are writable.
        for epoch in range(3):
            if epoch == 1:
                counts.assign(np.where(np.arange(worlds) % 3 == 0, 0, counts_host).astype(np.int32))
                z *= np.float32(1.03)
                shared[1].assign(np.sum(np.nan_to_num(z) ** 2, axis=2) + compliance)
                shared[0].assign(rhs * np.float32(0.7))
                held *= np.float32(0.97)
                factor_scale *= np.float32(0.97)
            elif epoch == 2:
                counts.assign(counts_host)
            for d, (impulse, velocity) in zip(data, outputs, strict=True):
                d.Z.assign(z)
                d.W.assign(held)
                d.status.zero_()
                impulse.zero_()
                velocity.fill_(19.0)
            for graph in graphs:
                wp.capture_launch(graph)
            for index in range(2):
                np.testing.assert_allclose(outputs[1][index].numpy(), outputs[0][index].numpy(), rtol=3e-4, atol=3e-5)
            actual = outputs[1][1].numpy()
            impulse = outputs[1][0].numpy()
            current_counts = counts.numpy()
            current_diagonal, current_rhs = shared[1].numpy(), shared[0].numpy()
            for group, art in enumerate(groups):
                world = world_map[art]
                count = current_counts[world]
                full_z = np.zeros((count, 43))
                for row in range(count):
                    width = 17 if world % 2 else 18
                    full_z[row, nodes[support[world, row], :width]] = z[world, row, :width]
                W = np.zeros((43, 43))
                W[owner.host["row"], owner.host["col"]] = held[group]
                start = art * 43
                np.testing.assert_allclose(
                    actual[start : start + 43] - vhat_host[start : start + 43],
                    (W.T @ (full_z.T @ impulse[world, :count]))[::-1],
                    rtol=3e-5,
                    atol=3e-6,
                )
                physical_j = (full_z @ independent_lower.T)[:, ::-1] / factor_scale[group]
                held_h = f["H"] / float(factor_scale[group]) ** 2
                delta = actual[start : start + 43].astype(float) - vhat_host[start : start + 43]
                force = physical_j.T @ impulse[world, :count]
                defect = np.linalg.norm(held_h @ delta - force, np.inf) / (
                    1 + np.linalg.norm(held_h, np.inf) * np.linalg.norm(delta, np.inf) + np.linalg.norm(force, np.inf)
                )
                self.assertLess(defect, 2e-6)
                _, expected_impulse, stats = metric.reference(
                    full_z,
                    full_z,
                    current_diagonal[world, :count],
                    current_rhs[world, :count],
                    types[world, :count],
                    parents[world, :count],
                    mu[world, :count],
                    np.zeros(43),
                    templates=support[world, :count],
                    iterations=8,
                )
                np.testing.assert_allclose(impulse[world, :count], expected_impulse, rtol=3e-4, atol=3e-5)
                if world == 4 and count:
                    self.assertGreater(
                        stats["unsafe"], 0, "Singular tangent block must exercise scalar/sibling fallback"
                    )
                for row in np.flatnonzero(types[world, :count] == 0):
                    self.assertLessEqual(
                        np.linalg.norm(impulse[world, row + 1 : row + 3]),
                        mu[world, row + 1] * impulse[world, row] + 3e-5,
                    )
            for d in data:
                np.testing.assert_array_equal(d.Z.numpy(), z)
                np.testing.assert_array_equal(d.W.numpy(), held)
                np.testing.assert_array_equal(d.status.numpy(), np.zeros(worlds, np.int32))
        # A rejected half or odd-tail world must not prevent its neighbors publishing.
        guarded_counts = counts_host.copy()
        guarded_counts[0] = 101
        counts.assign(guarded_counts)
        bad_parents = parents.copy()
        bad_parents[1, 1] = 98
        shared[3].assign(bad_parents)
        for d, (impulse, velocity) in zip(data, outputs, strict=True):
            valid, status = np.ones(worlds, np.int32), np.zeros(worlds, np.int32)
            valid[2], status[3] = 0, 8
            d.valid.assign(valid)
            d.status.assign(status)
            impulse.zero_()
            velocity.fill_(19.0)
        for graph in graphs:
            wp.capture_launch(graph)
        np.testing.assert_array_equal(data[1].status.numpy(), data[0].status.numpy())
        self.assertEqual(data[1].status.numpy()[1], 4)
        for index in range(2):
            np.testing.assert_allclose(outputs[1][index].numpy(), outputs[0][index].numpy(), rtol=3e-4, atol=3e-5)
        active_art = int(np.flatnonzero(world_map == 4)[0])
        self.assertFalse(np.all(outputs[1][1].numpy()[active_art * 43 : (active_art + 1) * 43] == 19.0))


if __name__ == "__main__":
    unittest.main()
