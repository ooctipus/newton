# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check paired sparse metric worlds against the existing physical oracles."""

import importlib
import inspect
import os
import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs.sparse_factor import SparseData, SparsePlan
from newton._src.solvers.feather_pgs.sparse_factor_rows import get_solve_kernel
from tools.fpgs_bench import test_sparse_metric_tangents as metric
from tools.fpgs_bench.test_sparse_factor import fixture

KEY = "sparse_metric_paired43_s18_c100_w16"
ENV = {**metric.METRIC_ENV, "FEATHER_PGS_SPARSE_PAIRED_WORLDS": "1"}


def paired():
    """Load the actual optional runtime, never a test replacement."""
    return importlib.import_module("newton._src.solvers.feather_pgs.sparse_paired_worlds")


class TestSparsePairedWorldsCPU(unittest.TestCase):
    def test_factory_abi(self):
        """Require the new owner with the unchanged original argument order."""
        original = get_solve_kernel(100, metric_tangents=True)
        candidate = paired().get_solve_kernel()
        self.assertEqual(candidate.key, KEY)
        self.assertEqual(inspect.signature(candidate.func), inspect.signature(original.func))

    def test_support_reduction_and_ownership(self):
        """Cover both secondary lanes and forbid cross-world native collectives."""
        source = paired().native_source()
        self.assertEqual(source.count("__syncthreads();"), 1)
        self.assertNotIn("0xffffffff", source)
        self.assertIn("du_storage[16][43]", source)
        self.assertIn("lam_storage[16][100]", source)
        self.assertIn("group=group*16+permutation[slot]", source)
        self.assertLess(source.index("__syncthreads();"), source.index("if(group>=p.group_to_art.shape[0])return;"))
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

    def test_admission_and_cpu_default(self):
        """Reject unsupported owners before replacing the original solver."""
        with patch.dict(os.environ, ENV):
            owner = fixture("cpu")["owner"]
        self.assertFalse(owner.paired_worlds)
        self.assertEqual(owner.kernels.solve.key, "sparse_metric_tangent43_s18_c100")
        solver = SimpleNamespace(
            model=SimpleNamespace(device=SimpleNamespace(is_cuda=True)),
            dense_max_constraints=100,
            pgs_warmstart=False,
            _mf_warmstart_enabled=False,
            pgs_velocity_iterations=0,
            pgs_iterations=8,
            pgs_omega=1.0,
        )
        owner = SimpleNamespace(solver=solver, metric_tangents=True, packet_rows=False, block_contacts=False)
        self.assertTrue(paired().supported(owner))
        for name, value in (
            ("dense_max_constraints", 101),
            ("pgs_warmstart", True),
            ("_mf_warmstart_enabled", True),
            ("pgs_velocity_iterations", 1),
            ("pgs_iterations", 7),
            ("pgs_omega", 1.2),
        ):
            with self.subTest(name=name), patch.object(solver, name, value):
                self.assertFalse(paired().supported(owner))
        for name in ("packet_rows", "block_contacts", "present_ports", "zero_expiry"):
            with self.subTest(name=name), patch.object(owner, name, True, create=True):
                self.assertFalse(paired().supported(owner))


@unittest.skipUnless(wp.is_cuda_available(), "Paired sparse native controls require CUDA")
class TestSparsePairedWorldsCUDA(unittest.TestCase):
    def test_native_saved_sixteen_current_held_epochs(self):
        """Reuse all sixteen actual current/held physical cases and unchanged gates."""
        check = partial(metric.check_native, expected_key=KEY)
        with patch.dict(os.environ, ENV), patch.object(metric, "check_native", check):
            metric.TestSparseMetricTangentsCUDA.test_native_saved_sixteen_current_held_epochs(self)

    def test_native_current_held_graph_and_fallback(self):
        """Reuse current geometry, held factors, graph empties and scalar fallback."""
        check = partial(metric.check_native, expected_key=KEY)
        with patch.dict(os.environ, ENV), patch.object(metric, "check_native", check):
            metric.TestSparseMetricTangentsCUDA.test_native_current_held_graph_and_empty(self)
            metric.TestSparseMetricTangentsCUDA.test_native_stick_open_slip_and_guarded_fallback(self)

    def test_native_seventeen_worlds_mixed_support_and_status(self):
        """Exercise actual half-warp isolation, secondary support, sorting and tail reuse."""
        device = "cuda:0"
        with patch.dict(os.environ, {**ENV, "FEATHER_PGS_SPARSE_PAIRED_WORLDS": "0"}):
            f = fixture(device)
        owner, solver = f["owner"], f["solver"]
        owner.refresh(solver)
        plan = SparsePlan()
        for name in owner.plan._cls.vars:
            setattr(plan, name, getattr(owner.plan, name))
        worlds = 17
        groups = np.arange(worlds, dtype=np.int32)[::-1].copy()
        world_map = np.roll(np.arange(worlds, dtype=np.int32), 3)
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
        plan.support_nodes = wp.array(nodes.ravel(), dtype=int, device=device)
        plan.support_count = wp.array(np.append(lengths, 17).astype(np.int32), dtype=int, device=device)
        counts_host = np.array([0, 3, 4, 7, 13, 9, 6, 33, 66, 99, 100, 12, 3, 45, 8, 2, 7], np.int32)
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
            if world in (1, 9):
                rhs[world, :count] = 0.2  # Stationary halves next to loaded worlds.
        compliance = np.full((worlds, 100), 0.01, np.float32)
        for world in (7, 16):
            normal = int(counts_host[world] % 3)
            z[world, normal + 2] = z[world, normal + 1]
            compliance[world, normal + 1 : normal + 3] = 0
        diagonal = np.sum(z * z, axis=2) + compliance
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
        held *= 1 + np.arange(worlds, dtype=np.float32)[:, None] * 0.005
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
        kernels = (get_solve_kernel(100, metric_tangents=True), paired().get_solve_kernel())

        def launch(arm):
            impulse, velocity = outputs[arm]
            wp.launch_tiled(
                kernels[arm],
                dim=[worlds if arm == 0 else (worlds + 15) // 16],
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
                block_dim=32 if arm == 0 else 256,
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
                shared[1].assign(np.sum(z * z, axis=2) + compliance)
                shared[0].assign(rhs * np.float32(0.7))
                held *= np.float32(0.97)
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
        guarded_counts[4] = 101
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


if __name__ == "__main__":
    unittest.main()
