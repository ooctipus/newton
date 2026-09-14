# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the actual composed force-to-velocity kernel, not full Solver.step."""

import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs.kernels import _FPGS_CONTACT_END_GAP_SLOP
from newton._src.solvers.feather_pgs.small_step_dispatch import classify
from newton._src.solvers.feather_pgs.small_step_rows import cuda_source, get_solve_kernel
from newton.tests.test_small_step_force import force_fixture, force_reference, predictor_reference
from tools.fpgs_bench.test_sparse_factor import physical_rows, unpack
from tools.fpgs_bench.test_sparse_metric_tangents import physical_metrics, reference


def integration_fixture(device="cpu"):
    """Bind original metadata and a held physical operator to the production kernel."""
    flags = {
        "FEATHER_PGS_SPARSE_SMALL_STEP": "0",
        "FEATHER_PGS_SPARSE_PACKETS": "1",
        "FEATHER_PGS_SPARSE_METRIC_TANGENTS": "0",
        "FEATHER_PGS_SPARSE_CONTACT_BLOCK": "0",
    }
    with patch.dict(os.environ, flags):
        case = force_fixture(device)
    owner = case["owner"]
    W = np.linalg.solve(np.linalg.cholesky(case["H"][::-1, ::-1]), np.eye(43))
    owner.data.W.assign(W[owner.host["row"], owner.host["col"]][None].astype(np.float32))
    owner.data.valid.fill_(1)
    owner.data.status.zero_()
    case["packets"] = wp.empty((1, 100), dtype=int, device=device)
    case["fallback_counts"] = wp.zeros(1, dtype=int, device=device)
    case["kernel"] = get_solve_kernel()
    build_current(case)
    return case


def build_current(case):
    """Rebuild current canonical identities before classifying this generation."""
    owner, solver = case["owner"], case["solver"]
    owner.build_rows(case["state"], solver, case["contacts"], case["force_input"].dt)
    wp.copy(case["packets"], owner.data.support)
    wp.launch(
        classify,
        dim=1,
        inputs=[solver.constraint_count, solver.row_type, owner.data.selected, case["fallback_counts"]],
        device=solver.model.device,
    )
    solver.impulses.zero_()


def physical_problem(case, predicted):
    """Build current J, original bias and held H response without private native rows."""
    owner, solver = case["owner"], case["solver"]
    count = int(solver.constraint_count.numpy()[0])
    J, W = physical_rows(case), unpack(owner)
    Z = (W @ J[:, ::-1].T).T
    Y = (Z @ W)[:, ::-1]
    types, parents, mu, phi, beta, cfm, target, restitution = (
        getattr(solver, name).numpy()[0, :count].copy()
        for name in (
            "row_type",
            "row_parent",
            "row_mu",
            "phi",
            "row_beta",
            "row_cfm",
            "target_velocity",
            "row_restitution",
        )
    )
    dt = case["force_input"].dt
    rhs = -target.astype(float)
    normals, limits = types == 0, types == 3
    rhs[normals] += (
        np.where(
            phi[normals] <= 0,
            beta[normals] * phi[normals],
            solver.contact_speculative_scale * phi[normals],
        )
        / dt
    )
    rhs[limits] += np.where(phi[limits] < 0, beta[limits] * phi[limits], phi[limits]) / dt
    relative = J @ predicted - target
    fires = (
        normals
        & (restitution > 0)
        & (relative < -solver._effective_restitution_velocity_threshold)
        & ((phi <= _FPGS_CONTACT_END_GAP_SLOP) | (phi + dt * relative <= _FPGS_CONTACT_END_GAP_SLOP))
    )
    rhs[fires] = -target[fires] + restitution[fires] * relative[fires]
    diagonal = np.sum(Z * Z, axis=1) + cfm
    return J, Y, diagonal, rhs, types, parents, mu, predicted


def launch_inputs(case):
    """Use the actual production factory and both current input structs."""
    owner, solver = case["owner"], case["solver"]
    return [
        owner.plan,
        owner.data,
        case["force_input"],
        owner.packet_input,
        owner.data.selected,
        case["packets"],
        solver.constraint_count,
        solver.row_beta,
        solver.row_type,
        solver.row_parent,
        solver.row_mu,
        8,
        1.0,
        0,
        solver.contact_speculative_scale,
        solver.impulses,
        solver.v_out,
    ]


def run(case):
    """Launch only the selected complete kernel under the root's device lease."""
    wp.launch_tiled(
        case["kernel"],
        dim=[1],
        inputs=launch_inputs(case),
        block_dim=32,
        device=case["model"].device,
    )


def expected(case):
    """Combine independent tree-force accumulation and the original metric law."""
    tau, _ = force_reference(case)
    predicted = predictor_reference(case, tau, unpack(case["owner"]))
    problem = physical_problem(case, predicted)
    velocity, impulses, _ = reference(*problem)
    return tau, problem, velocity, impulses


def check_output(test, case, target):
    """Check full velocity, physical momentum and endpoint contact metrics."""
    tau, problem, velocity, impulses = target
    J, _, diagonal, rhs, types, parents, mu, predicted = problem
    solver, owner = case["solver"], case["owner"]
    owner.check()
    actual = solver.v_out.numpy().astype(float)
    lam = solver.impulses.numpy()[0, : len(J)].astype(float)
    np.testing.assert_allclose(actual, velocity, rtol=7e-4, atol=8e-5)
    np.testing.assert_allclose(lam, impulses, rtol=7e-4, atol=8e-5)
    # This is the complete force+constraint momentum statement, with original
    # free-root transport kept separate from acceleration.
    dt = case["force_input"].dt
    qd = case["force_input"].stage3_qd.numpy().astype(float)
    transport = np.zeros(43)
    transport[:3] = dt * np.cross(qd[3:6], qd[:3])
    delivered = dt * tau + J.T @ lam
    defect = case["H"] @ (actual - qd - transport) - delivered
    test.assertLess(np.linalg.norm(defect) / (1 + np.linalg.norm(delivered)), 4e-5)
    metrics = physical_metrics(J, diagonal, rhs, types, parents, mu, predicted, actual, lam)
    baseline = physical_metrics(J, diagonal, rhs, types, parents, mu, predicted, velocity, impulses)
    test.assertLess(metrics["cone"], 3e-5)
    for name, value in metrics.items():
        test.assertTrue(np.isfinite(value))
        test.assertLess(abs(value - baseline[name]), 4e-3 * (1 + baseline[name]))


class TestSmallStepIntegrationCPU(unittest.TestCase):
    def test_selected_cold_initialization(self):
        """Ignore stale canonical impulses when selected cold-start preparation is retired."""
        source = cuda_source("    __shared__ float small_vhat[43];\n")
        self.assertNotIn("lam[r]=impulses.data[base+r]", source)
        self.assertIn("lam[r]=0.0f", source)

    def test_composed_api_and_independent_momentum(self):
        """Bind the actual force/row kernel and verify its complete physical reference."""
        case = integration_fixture()
        self.assertEqual(case["kernel"].key, "small_step_force_rows_metric43_s18_c32")
        self.assertEqual(int(case["owner"].data.selected.numpy()[0]), 1)
        tau, problem, velocity, lam = expected(case)
        dt = case["force_input"].dt
        qd = case["force_input"].stage3_qd.numpy().astype(float)
        transport = np.zeros(43)
        transport[:3] = dt * np.cross(qd[3:6], qd[:3])
        np.testing.assert_allclose(
            case["H"] @ (velocity - qd - transport),
            dt * tau + problem[0].T @ lam,
            rtol=3e-5,
            atol=3e-6,
        )


@unittest.skipUnless(wp.is_cuda_available(), "Root owns the paired CUDA lease")
class TestSmallStepIntegrationCUDA(unittest.TestCase):
    def test_native_complete_current_force_geometry_and_held_operator(self):
        """Run actual force+rows+decode with current inputs and a held physical mass."""
        case = integration_fixture("cuda:0")
        solver, owner, force = case["solver"], case["owner"], case["force_input"]
        held = owner.data.W.numpy().copy()
        u0 = force.u0.numpy().copy()
        for current_change in (False, True):
            if current_change:
                screws = force.joint_S_s.numpy()
                screws[6:] += np.random.default_rng(914934).normal(scale=0.005, size=(37, 6)).astype(np.float32)
                force.joint_S_s.assign(screws)
                points = case["contacts"].rigid_contact_point0.numpy()
                points[:3, 0] += 0.003
                case["contacts"].rigid_contact_point0.assign(points)
                bias = force.body_f_s.numpy()
                bias[:, 2] += 0.015
                force.body_f_s.assign(bias)
                solver.mass_update_mask.zero_()
                build_current(case)
            target = expected(case)
            # Retired selected intermediates are not even valid input values.
            solver.joint_qdd.fill_(float("nan"))
            solver.v_hat.fill_(float("nan"))
            solver.rhs.fill_(float("nan"))
            solver.diag.fill_(float("nan"))
            owner.data.support.fill_(-999)
            owner.data.Z.fill_(float("nan"))
            owner.data.incident.fill_(float("nan"))
            solver.impulses.fill_(0.125)
            run(case)
            check_output(self, case, target)
            np.testing.assert_array_equal(owner.data.W.numpy(), held)
            np.testing.assert_array_equal(force.u0.numpy(), u0)
            self.assertTrue(np.isnan(solver.joint_qdd.numpy()).all())
            self.assertTrue(np.isnan(solver.v_hat.numpy()).all())

    def test_native_selection_empty_regrow_and_graph(self):
        """Reclassify current reservations and avoid stale private state on graph replay."""
        case = integration_fixture("cuda:0")
        solver, owner = case["solver"], case["owner"]
        target = expected(case)
        run(case)
        check_output(self, case, target)
        with wp.ScopedCapture(device=case["model"].device) as capture:
            wp.launch(
                classify,
                dim=1,
                inputs=[solver.constraint_count, solver.row_type, owner.data.selected, case["fallback_counts"]],
                device=case["model"].device,
            )
            run(case)
        counts = solver.constraint_count.numpy().copy()
        solver.constraint_count.fill_(33)
        solver.v_out.fill_(123.0)
        solver.impulses.fill_(0.25)
        wp.capture_launch(capture.graph)
        self.assertEqual(int(owner.data.selected.numpy()[0]), 0)
        self.assertEqual(int(case["fallback_counts"].numpy()[0]), 33)
        np.testing.assert_array_equal(solver.v_out.numpy(), np.full(43, 123.0))
        # Empty worlds still execute their current force prediction; they are
        # not admitted as free or stationary work.
        solver.constraint_count.zero_()
        wp.capture_launch(capture.graph)
        self.assertEqual(int(owner.data.selected.numpy()[0]), 1)
        tau, _ = force_reference(case)
        predicted = predictor_reference(case, tau, unpack(owner))
        np.testing.assert_allclose(solver.v_out.numpy(), predicted, rtol=3e-5, atol=3e-6)
        solver.constraint_count.assign(counts)
        solver.impulses.fill_(0.375)
        wp.capture_launch(capture.graph)
        check_output(self, case, target)


if __name__ == "__main__":
    unittest.main()
