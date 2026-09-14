# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check bounded local rows independently of the force prediction fragment."""

import os
import unittest
from functools import cache
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs.small_step_rows import cuda_source
from newton._src.solvers.feather_pgs.sparse_factor import SparseData, SparsePlan
from newton._src.solvers.feather_pgs.sparse_packet_rows import PacketInput
from tools.fpgs_bench.test_sparse_metric_tangents import physical_metrics, reference
from tools.fpgs_bench.test_sparse_packet_rows import prepared, problem


def fixture(device="cpu"):
    """Reuse current physical packets without enabling the old packet solve."""
    with patch.dict(
        os.environ,
        {"FEATHER_PGS_SPARSE_METRIC_TANGENTS": "0", "FEATHER_PGS_SPARSE_CONTACT_BLOCK": "0"},
    ):
        return prepared(device)


@cache
def row_kernel():
    """Inject only a known predictor to isolate the production row/decode source."""
    injection = """
    __shared__ float small_vhat[43];
    for(int k=lane;k<43;k+=32)small_vhat[k]=prediction.data[start+k];
    __syncwarp();
"""

    @wp.func_native(cuda_source(injection))
    def native(
        group: int,
        p: SparsePlan,
        d: SparseData,
        x: PacketInput,
        selected: wp.array[int],
        packets: wp.array2d[int],
        counts: wp.array[int],
        row_beta: wp.array2d[float],
        row_type: wp.array2d[int],
        parent: wp.array2d[int],
        mu: wp.array2d[float],
        iterations: int,
        omega: float,
        friction_start: int,
        contact_speculative_scale: float,
        impulses: wp.array2d[float],
        prediction: wp.array[float],
        vout: wp.array[float],
    ): ...

    def solve(
        p: SparsePlan,
        d: SparseData,
        x: PacketInput,
        selected: wp.array[int],
        packets: wp.array2d[int],
        counts: wp.array[int],
        row_beta: wp.array2d[float],
        row_type: wp.array2d[int],
        parent: wp.array2d[int],
        mu: wp.array2d[float],
        iterations: int,
        omega: float,
        friction_start: int,
        contact_speculative_scale: float,
        impulses: wp.array2d[float],
        prediction: wp.array[float],
        vout: wp.array[float],
    ):
        group, _ = wp.tid()
        native(
            group,
            p,
            d,
            x,
            selected,
            packets,
            counts,
            row_beta,
            row_type,
            parent,
            mu,
            iterations,
            omega,
            friction_start,
            contact_speculative_scale,
            impulses,
            prediction,
            vout,
        )

    return wp.kernel(enable_backward=False, module="unique")(solve)


def inputs(f, *, omega=1.0, friction_start=0):
    """Separate current packet identity from the retired support producer."""
    s, owner = f["solver"], f["owner"]
    selected = wp.ones(1, dtype=int, device=s.model.device)
    packets = wp.clone(owner.data.support)
    return [
        owner.plan,
        owner.data,
        owner.packet_input,
        selected,
        packets,
        s.constraint_count,
        s.row_beta,
        s.row_type,
        s.row_parent,
        s.row_mu,
        8,
        omega,
        friction_start,
        s.contact_speculative_scale,
        s.impulses,
        s.v_hat,
        s.v_out,
    ]


def check_rows(test, f, *, omega=1.0, friction_start=0):
    """Compare the metric law and held physical momentum, not old-eight coefficients."""
    s, owner = f["solver"], f["owner"]
    args = problem(f)
    J, Y, diagonal, rhs, types, parents, mu, before = args
    test.assertLessEqual(len(J), 32)
    initial = s.impulses.numpy()[0, : len(J)].astype(float)
    expected, expected_lam, _ = reference(
        *args,
        omega=omega,
        friction_start=friction_start,
        incoming=initial,
    )
    launch = inputs(f, omega=omega, friction_start=friction_start)
    # These buffers must not remain hidden selected-world dependencies.
    owner.data.support.fill_(-999)
    owner.data.Z.fill_(float("nan"))
    owner.data.incident.fill_(float("nan"))
    s.diag.fill_(float("nan"))
    s.rhs.fill_(float("nan"))
    wp.launch_tiled(row_kernel(), dim=[1], inputs=launch, block_dim=32, device=s.model.device)
    owner.check()
    actual = s.v_out.numpy().astype(float)
    lam = s.impulses.numpy()[0, : len(J)].astype(float)
    np.testing.assert_allclose(actual, expected, rtol=5e-4, atol=5e-5)
    np.testing.assert_allclose(lam, expected_lam, rtol=5e-4, atol=5e-5)
    if not np.any(initial):
        np.testing.assert_allclose(actual, before + Y.T @ lam, rtol=5e-5, atol=5e-6)
        momentum = f["H"] @ (actual - before) - J.T @ lam
        test.assertLess(np.linalg.norm(momentum) / (1 + np.linalg.norm(J.T @ lam)), 2e-5)
    metrics = physical_metrics(J, diagonal, rhs, types, parents, mu, before, actual, lam)
    baseline = physical_metrics(J, diagonal, rhs, types, parents, mu, before, expected, expected_lam)
    test.assertLess(metrics["cone"], 2e-5)
    for key in metrics:
        test.assertTrue(np.isfinite(metrics[key]))
        test.assertLess(abs(metrics[key] - baseline[key]), 3e-3 * (1 + baseline[key]))
    return launch


class TestSmallStepRowsCPU(unittest.TestCase):
    def test_local_storage_and_retired_consumers(self):
        """Require local32 rows, private prediction and the unchanged metric law."""
        source = cuda_source("    __shared__ float small_vhat[43];\n")
        self.assertIn("zrows[32*18]", source)
        self.assertIn("contact_cross[32]", source)
        self.assertIn("selected.data[world]", source)
        self.assertIn("packets.data[packet_base+r]", source)
        for retired in ("d.Z.data", "d.incident.data", "d.support.data", "rhs.data", "diagonal.data", "vhat.data"):
            self.assertNotIn(retired, source)
        self.assertIn("probe<16", source)
        self.assertIn("omega==1.0f", source)
        self.assertIn("iteration>=friction_start", source)

    def test_physical_current_rows_metric_and_held_momentum(self):
        """Check the independent law on actual contacts and every active unit limit."""
        f = fixture()
        s, owner = f["solver"], f["owner"]
        held = owner.data.W.numpy().copy()
        for friction in (True, False):
            s.enable_contact_friction = friction
            owner.build_rows(f["state"], s, f["contacts"], 0.0025)
            s._stage4_compute_rhs_world(0.0025)
            args = problem(f)
            self.assertLessEqual(len(args[0]), 32)
            self.assertTrue(np.any(args[4] == 3))
            after, lam, _ = reference(*args)
            np.testing.assert_allclose(f["H"] @ (after - args[7]), args[0].T @ lam, rtol=3e-5, atol=3e-6)
            metrics = physical_metrics(args[0], args[2], *args[3:7], args[7], after, lam)
            self.assertLess(metrics["cone"], 1e-12)
            self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
            np.testing.assert_array_equal(owner.data.W.numpy(), held)


@unittest.skipUnless(wp.is_cuda_available(), "Root owns the paired CUDA lease")
class TestSmallStepRowsCUDA(unittest.TestCase):
    def test_current_geometry_held_metric_and_fallback(self):
        """Keep current contacts, held response, delayed friction and scalar fallback."""
        f = fixture("cuda:0")
        s, owner = f["solver"], f["owner"]
        held = owner.data.W.numpy().copy()
        for omega, friction_start, friction in ((1.0, 0, True), (1.0, 2, True), (1.2, 0, True), (1.0, 0, False)):
            s.enable_contact_friction = friction
            points = f["contacts"].rigid_contact_point0.numpy()
            points[:3, 0] += 0.001
            f["contacts"].rigid_contact_point0.assign(points)
            s.v_hat.assign(np.linspace(0.07, -0.09, 43, dtype=np.float32))
            owner.build_rows(f["state"], s, f["contacts"], 0.0025)
            s._stage4_compute_rhs_world(0.0025)
            s.impulses.zero_()
            with self.subTest(omega=omega, delay=friction_start, friction=friction):
                check_rows(self, f, omega=omega, friction_start=friction_start)
            np.testing.assert_array_equal(owner.data.W.numpy(), held)
        # Exercise the original predicted end-gap restitution predicate using
        # the private predictor rather than a retained incident/RHS producer.
        s.enable_contact_friction = True
        owner.build_rows(f["state"], s, f["contacts"], 0.0025)
        s._stage4_compute_rhs_world(0.0025)
        args = problem(f)
        row = int(np.flatnonzero(args[4] == 0)[0])
        direction = args[0][row]
        s.v_hat.assign((-direction / (direction @ direction)).astype(np.float32))
        relative = float(direction @ s.v_hat.numpy() - s.target_velocity.numpy()[0, row])
        phi = s.phi.numpy()
        phi[0, row] = -0.0025 * relative + 0.5e-6
        s.phi.assign(phi)
        restitution = s.row_restitution.numpy()
        restitution[0, row] = 0.7
        s.row_restitution.assign(restitution)
        s._stage4_compute_rhs_world(0.0025)
        s.impulses.zero_()
        check_rows(self, f)

    def test_selection_capacity_and_layout_guards(self):
        """Solve32 unit limits and leave unselected/oversized/invalid outputs untouched."""
        f = fixture("cuda:0")
        s, owner = f["solver"], f["owner"]
        positions = f["state"].joint_q.numpy()
        indices = s._joint_limit_q_index.numpy()
        active = np.flatnonzero(indices >= 0)[:16]
        lower = np.full(43, -np.inf, np.float32)
        upper = np.full(43, np.inf, np.float32)
        lower[active] = positions[indices[active]] - 0.001
        upper[active] = positions[indices[active]] + 0.001
        f["model"].joint_limit_lower.assign(lower)
        f["model"].joint_limit_upper.assign(upper)
        f["contacts"].rigid_contact_count.zero_()
        owner.build_rows(f["state"], s, f["contacts"], 0.0025)
        s._stage4_compute_rhs_world(0.0025)
        self.assertEqual(int(s.constraint_count.numpy()[0]), 32)
        check_rows(self, f)
        launch = inputs(f)
        count = s.constraint_count.numpy().copy()
        s.v_out.fill_(123.0)
        saved = s.v_out.numpy()
        saved_impulses = s.impulses.numpy()
        for kind in ("unselected", "oversized", "invalid", "layout"):
            owner.data.valid.fill_(1)
            owner.data.status.zero_()
            s.constraint_count.assign(count)
            launch[3].fill_(1)
            if kind == "unselected":
                launch[3].zero_()
            elif kind == "oversized":
                s.constraint_count.fill_(33)
            elif kind == "invalid":
                owner.data.valid.zero_()
            else:
                s.row_type.fill_(7)
            wp.launch_tiled(row_kernel(), dim=[1], inputs=launch, block_dim=32, device=s.model.device)
            np.testing.assert_array_equal(s.v_out.numpy(), saved)
            np.testing.assert_array_equal(s.impulses.numpy(), saved_impulses)
        self.assertNotEqual(int(owner.data.status.numpy()[0]), 0)

    def test_graph_empty_and_regrowing_rows(self):
        """Reinitialize private rows and impulses for empty/regrowing captured launches."""
        f = fixture("cuda:0")
        s, owner = f["solver"], f["owner"]
        launch = inputs(f)
        count = s.constraint_count.numpy().copy()
        wp.launch_tiled(row_kernel(), dim=[1], inputs=launch, block_dim=32, device=s.model.device)
        expected = s.v_out.numpy().copy()
        expected_lam = s.impulses.numpy().copy()
        with wp.ScopedCapture(device=s.model.device) as capture:
            wp.launch_tiled(row_kernel(), dim=[1], inputs=launch, block_dim=32, device=s.model.device)
        s.constraint_count.zero_()
        s.impulses.zero_()
        s.v_out.fill_(123.0)
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(s.v_out.numpy(), s.v_hat.numpy())
        s.constraint_count.assign(count)
        s.impulses.zero_()
        wp.capture_launch(capture.graph)
        owner.check()
        np.testing.assert_allclose(s.v_out.numpy(), expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(s.impulses.numpy(), expected_lam, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
