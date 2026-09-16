# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Component controls for field-major world-lane GS and its charged decode."""

import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import sparse_field_major
from newton._src.solvers.feather_pgs.sparse_factor import SparseData, SparsePlan
from newton._src.solvers.feather_pgs.sparse_factor_rows import get_solve_kernel
from tools.fpgs_bench import test_sparse_metric_tangents as metric
from tools.fpgs_bench.test_sparse_contact_block import saved_records
from tools.fpgs_bench.test_sparse_factor import fixture, physical_rows

ENV = {
    **metric.METRIC_ENV,
    "FEATHER_PGS_BODY_BASIS_ROWS": "0",
    "FEATHER_PGS_SPARSE_PAIRED_GS": "0",
    "FEATHER_PGS_PARALLEL_WORLD": "0",
    "FEATHER_PGS_PARALLEL_WORLD_SPLIT": "0",
    "FEATHER_PGS_SPARSE_SUPERNODAL": "0",
}


def synthetic(device):
    """Real held factor with 35 independently mapped worlds and support17/18."""
    with patch.dict(os.environ, ENV):
        original = fixture(device)
    owner, solver = original["owner"], original["solver"]
    owner.refresh(solver)
    p = SparsePlan()
    for name in owner.plan._cls.vars:
        setattr(p, name, getattr(owner.plan, name))
    worlds = 35
    groups = np.arange(worlds, dtype=np.int32)[::-1].copy()
    world_map = np.roll(np.arange(worlds, dtype=np.int32), 7)
    p.group_to_art = wp.array(groups, dtype=int, device=device)
    p.art_to_world = wp.array(world_map, dtype=int, device=device)
    p.art_dof_start = wp.array(np.arange(worlds, dtype=np.int32) * 43, dtype=int, device=device)
    nodes = p.support_nodes.numpy().reshape(-1, 18)
    lengths = p.support_count.numpy()
    full = int(np.flatnonzero(lengths == 18)[0])
    short = len(lengths)
    extra = nodes[full].copy()
    extra[17] = -1
    nodes = np.concatenate((nodes, extra[None]))
    lengths = np.append(lengths, 17).astype(np.int32)
    p.support_nodes = wp.array(nodes, dtype=int, device=device)
    p.support_count = wp.array(lengths, dtype=int, device=device)
    counts_host = np.resize(np.array([0, 3, 4, 7, 13], np.int32), worlds)
    z = np.zeros((worlds, 100, 18), np.float32)
    support = np.zeros((worlds, 100), np.int32)
    types = np.full((worlds, 100), 3, np.int32)
    parents = np.full((worlds, 100), -1, np.int32)
    rhs = np.zeros((worlds, 100), np.float32)
    mu = np.zeros_like(rhs)
    for world, count in enumerate(counts_host):
        support[world] = short if world % 2 else full
        prefix = count % 3
        for row in range(count):
            component = (row - prefix) % 3 if row >= prefix else 0
            z[world, row, component] = 0.8 + 0.002 * world
            z[world, row, 16] = (0.13, -0.08, 0.04)[component]
            if world % 2 == 0:
                z[world, row, 17] = (-0.09, 0.06, 0.11)[component]
            rhs[world, row] = (-0.15, 0.07, -0.09)[component]
            if row >= prefix:
                types[world, row] = 0 if component == 0 else 2
                if component:
                    parents[world, row] = row - component
                    mu[world, row] = 0.6
        if world % 5 == 1:
            rhs[world, :count] = 0.2
    # A valid scalar sibling may have a different sparse template; this also
    # forces transactional metric rejection before the scalar fallback.
    support[2, 3] = short
    compliance = np.full((worlds, 100), 0.01, np.float32)
    for world in range(4, worlds, 5):
        normal = int(counts_host[world] % 3)
        z[world, normal + 2] = z[world, normal + 1]
        compliance[world, normal + 1 : normal + 3] = 0
    diagonal = np.sum(z * z, axis=2) + compliance
    dense = np.zeros((worlds, 100, 43), np.float64)
    for world, count in enumerate(counts_host):
        for row in range(count):
            tpl = support[world, row]
            width = lengths[tpl]
            dense[world, row, nodes[tpl, :width]] = z[world, row, :width]
        z[world, count:] = np.nan
        if world % 2:
            z[world, :count, 17] = np.nan
    z[2, 3, 17] = np.nan
    d = SparseData()
    held = np.repeat(owner.data.W.numpy(), worlds, axis=0)
    held *= (1 + 0.002 * np.arange(worlds, dtype=np.float32))[:, None]
    d.W = wp.array(held, dtype=float, device=device)
    d.Z = wp.array(z, dtype=float, device=device)
    d.support = wp.array(support, dtype=int, device=device)
    d.incident = wp.zeros((worlds, 100), dtype=float, device=device)
    d.valid = wp.ones(worlds, dtype=int, device=device)
    d.status = wp.zeros(worlds, dtype=int, device=device)
    inputs = [
        wp.array(v, dtype=t, device=device)
        for v, t in ((rhs, float), (diagonal, float), (types, int), (parents, int), (mu, float))
    ]
    counts = wp.array(counts_host, dtype=int, device=device)
    vhat = wp.array(np.random.default_rng(52).normal(0, 0.02, worlds * 43).astype(np.float32), device=device)
    impulse = wp.zeros((worlds, 100), dtype=float, device=device)
    velocity = wp.empty(worlds * 43, dtype=float, device=device)
    packed = sparse_field_major.pack_rows(d, *inputs)
    return {
        "p": p,
        "d": d,
        "packed": packed,
        "inputs": inputs,
        "counts": counts,
        "counts_host": counts_host,
        "impulse": impulse,
        "velocity": velocity,
        "vhat": vhat,
        "dense": dense,
        "groups": groups,
        "world_map": world_map,
        "worlds": worlds,
        "held": held,
    }


def launch(case, *, iterations=8, omega=1.0, friction_start=0):
    p, d, f = case["p"], case["d"], case["packed"]
    device = case["impulse"].device
    # Deliberate overlaunch exercises guards before group mapping for both owners.
    wp.launch(
        sparse_field_major.get_solve_kernel(),
        dim=case["worlds"] + 3,
        inputs=[p, d, f, case["counts"], case["impulse"], iterations, omega, friction_start],
        block_dim=32,
        device=device,
    )
    wp.launch_tiled(
        sparse_field_major.get_decode_kernel(),
        dim=[case["worlds"] + 3],
        inputs=[p, d, f, case["counts"], case["impulse"], case["vhat"], case["velocity"]],
        block_dim=32,
        device=device,
    )


def check_synthetic(test, device):
    case = synthetic(device)
    rhs, diagonal, types, parents, mu = [a.numpy() for a in case["inputs"]]
    index = case["p"].index.numpy()
    vhat = case["vhat"].numpy()
    for iterations, omega, delayed in ((8, 1.0, 0), (2, 1.2, 0), (2, 1.0, 1), (0, 1.0, 0)):
        incoming = np.full((case["worlds"], 100), -17.0, np.float32)
        for world, count in enumerate(case["counts_host"]):
            incoming[world, :count] = np.resize(np.array([0.2, 0.01, -0.02], np.float32), count)
        case["impulse"].assign(incoming)
        case["velocity"].fill_(19.0)
        case["packed"].du.fill_(np.nan)
        case["packed"].cross.fill_(np.nan)
        launch(case, iterations=iterations, omega=omega, friction_start=delayed)
        actual, lam = case["velocity"].numpy(), case["impulse"].numpy()
        test.assertTrue(np.isfinite(actual).all())
        for group, art in enumerate(case["groups"]):
            world = case["world_map"][art]
            count = int(case["counts_host"][world])
            W = np.zeros((43, 43))
            mask = index >= 0
            W[mask] = case["held"][group, index[mask]]
            z = case["dense"][world, :count]
            du, expected, _ = metric.reference(
                z,
                z,
                diagonal[world, :count],
                rhs[world, :count],
                types[world, :count],
                parents[world, :count],
                mu[world, :count],
                np.zeros(43),
                iterations=iterations,
                omega=omega,
                friction_start=delayed,
                incoming=incoming[world, :count],
                templates=case["d"].support.numpy()[world, :count],
            )
            start = int(art) * 43
            np.testing.assert_allclose(
                actual[start : start + 43], vhat[start : start + 43] + (W.T @ du)[::-1], rtol=3e-4, atol=3e-5
            )
            np.testing.assert_allclose(lam[world, :count], expected, rtol=3e-4, atol=3e-5)
            np.testing.assert_array_equal(lam[world, count:], incoming[world, count:])
    # Veto stale private outputs after a previous successful call, including
    # invalid mapping-independent states and malformed friction metadata.
    status = np.zeros(case["worlds"], np.int32)
    status[2] = 8
    valid = np.ones(case["worlds"], np.int32)
    valid[3] = 0
    counts = case["counts_host"].copy()
    counts[4] = 101
    packed_types = case["packed"].row_type.numpy()
    packed_types[0, 1] = 9
    case["packed"].row_type.assign(packed_types)
    case["d"].status.assign(status)
    case["d"].valid.assign(valid)
    case["counts"].assign(counts)
    case["velocity"].fill_(19.0)
    before = case["impulse"].numpy().copy()
    if wp.get_device(device).is_cuda:
        with wp.ScopedCapture(device=device) as capture:
            launch(case)
        wp.capture_launch(capture.graph)
    else:
        launch(case)
    for art, world in enumerate(case["world_map"]):
        if world in (1, 2, 3, 4):
            np.testing.assert_array_equal(case["velocity"].numpy()[art * 43 : (art + 1) * 43], 19.0)
            np.testing.assert_array_equal(case["impulse"].numpy()[world], before[world])
    test.assertEqual(case["d"].status.numpy()[1], 4)
    np.testing.assert_array_equal(case["d"].W.numpy(), case["held"])


class TestSparseFieldMajor(unittest.TestCase):
    def test_factory_contract(self):
        self.assertEqual(sparse_field_major.get_solve_kernel().key, "sparse_field_major_metric43_s18_c100")
        self.assertEqual(sparse_field_major.get_decode_kernel().key, "sparse_field_major_decode43_s18_c100")
        source = sparse_field_major.native_source()
        self.assertIn("__shared__ float du[43][32]", source)
        self.assertNotIn("__shfl", source)
        self.assertNotIn("__syncwarp", source)
        self.assertNotIn("__syncthreads", source)

    def test_cpu_physical_transactions_and_guards(self):
        check_synthetic(self, "cpu")


@unittest.skipUnless(wp.is_cuda_available(), "Field-major native controls require CUDA")
class TestSparseFieldMajorCUDA(unittest.TestCase):
    def test_mapped_worlds_tail_fallback_incoming_and_graph(self):
        check_synthetic(self, "cuda:0")

    def test_saved_sixteen_original_same_rows(self):
        replay, records = saved_records()
        tested = 0
        for gpu, record, inputs in records:
            for i, world in enumerate(inputs["worlds"]):
                with self.subTest(gpu=gpu, step=record["step"], world=int(world)), patch.dict(os.environ, ENV):
                    case = replay.bind_world(record, inputs, i, "cuda:0")
                    s, owner = case["solver"], case["owner"]
                    lower = np.linalg.cholesky(case["H"][::-1, ::-1])
                    inverse = np.linalg.solve(lower, np.eye(43))
                    owner.data.W.assign(inverse[owner.host["row"], owner.host["col"]][None].astype(np.float32))
                    owner.data.valid.fill_(1)
                    owner.build_rows(case["state"], s, case["contacts"], 0.0025)
                    s.check_constraint_capacity()
                    s._stage4_compute_rhs_world(0.0025)
                    owner.restitution(0.0025)
                    s.impulses.zero_()
                    self.assertIs(owner.kernels.solve, get_solve_kernel(100, False, metric_tangents=True))
                    old_v, old_lam, _ = metric.check_native(self, case)
                    count = int(s.constraint_count.numpy()[0])
                    packed = sparse_field_major.pack_rows(owner.data, s.rhs, s.diag, s.row_type, s.row_parent, s.row_mu)
                    s.impulses.zero_()
                    s.v_out.fill_(np.nan)
                    run = {
                        "p": owner.plan,
                        "d": owner.data,
                        "packed": packed,
                        "counts": s.constraint_count,
                        "impulse": s.impulses,
                        "vhat": s.v_hat,
                        "velocity": s.v_out,
                        "worlds": 1,
                    }
                    launch(run)
                    owner.check()
                    actual, impulse = s.v_out.numpy(), s.impulses.numpy()[0, :count]
                    np.testing.assert_allclose(actual, old_v, rtol=3e-4, atol=3e-5)
                    np.testing.assert_allclose(impulse, old_lam, rtol=3e-4, atol=3e-5)
                    J = physical_rows(case)
                    delta = actual.astype(float) - s.v_hat.numpy()
                    force = J.T @ impulse
                    defect = np.linalg.norm(case["H"] @ delta - force, np.inf) / (
                        1
                        + np.linalg.norm(case["H"], np.inf) * np.linalg.norm(delta, np.inf)
                        + np.linalg.norm(force, np.inf)
                    )
                    self.assertLess(defect, 2e-6)
                    physics = metric.physical_metrics(
                        J,
                        s.diag.numpy()[0, :count],
                        s.rhs.numpy()[0, :count],
                        s.row_type.numpy()[0, :count],
                        s.row_parent.numpy()[0, :count],
                        s.row_mu.numpy()[0, :count],
                        s.v_hat.numpy(),
                        actual,
                        impulse,
                    )
                    self.assertLess(physics["cone"], 3e-5)
                    tested += 1
        self.assertEqual(tested, 16)


if __name__ == "__main__":
    unittest.main()
