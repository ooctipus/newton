# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the default-off, transactional joint-limit-only Jacobi owner."""

import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import sparse_limit_jacobi as jacobi
from newton._src.solvers.feather_pgs import sparse_spectral_tangents as spectral
from tools.fpgs_bench import limit_jacobi_control as control
from tools.fpgs_bench import spectral_gs_control as ordered
from tools.fpgs_bench import test_sparse_metric_tangents as metric
from tools.fpgs_bench.test_sparse_contact_block import full_rows
from tools.fpgs_bench.test_sparse_factor import fixture, unpack

ENV = {
    **metric.METRIC_ENV,
    "FEATHER_PGS_SPARSE_SPECTRAL_TANGENTS": "1",
    "FEATHER_PGS_SPARSE_LIMIT_JACOBI": "1",
}


def check_native(test, f, *, iterations=8, omega=1.0, friction_start=0, fallback=False):
    """Compare the native translation and complete momentum publication."""
    s, owner = f["solver"], f["owner"]
    test.assertTrue(owner.limit_jacobi)
    test.assertIs(owner.kernels.solve, jacobi.get_solve_kernel())
    count = int(s.constraint_count.numpy()[0])
    z, templates = full_rows(owner, count)
    diagonal, rhs, types, parents, mu, incoming = (
        getattr(s, name).numpy()[0, :count].copy()
        for name in ("diag", "rhs", "row_type", "row_parent", "row_mu", "impulses")
    )
    seed = rhs.astype(float) + owner.data.incident.numpy()[0, :count]
    if count == 0:
        du, expected_lam, stats = np.zeros(43), np.zeros(0), {}
    elif omega != 1.0:
        du, expected_lam, stats = metric.scalar_reference(
            z,
            z,
            diagonal,
            seed,
            types,
            parents,
            mu,
            np.zeros(43),
            iterations=iterations,
            omega=omega,
            friction_start=friction_start,
            incoming=incoming,
            templates=templates,
            block=False,
        )
    else:
        solve = ordered.solve if fallback else control.solve
        result = solve(
            z,
            np.eye(43),
            diagonal,
            seed,
            types,
            parents,
            mu,
            np.zeros(43),
            iterations=iterations,
            incoming=incoming,
            early_stop=False,
        )
        du, expected_lam, stats = result.velocity, result.impulses, result.work
    vhat, W = s.v_hat.numpy().copy(), unpack(owner)
    owner.solve(s.rhs, iterations, omega, friction_start)
    owner.check()
    actual, lam = s.v_out.numpy(), s.impulses.numpy()[0, :count]
    np.testing.assert_allclose(lam, expected_lam, rtol=3e-4, atol=3e-5)
    np.testing.assert_allclose(actual, vhat + (W.T @ du)[::-1], rtol=3e-4, atol=3e-5)
    test.assertTrue(np.isfinite(actual).all() and np.isfinite(lam).all())
    if friction_start == 0 or not np.any(incoming):
        np.testing.assert_allclose(actual, vhat + (W.T @ (z.T @ (lam - incoming)))[::-1], rtol=3e-5, atol=3e-6)
    return actual.copy(), lam.copy(), stats


def seed_limits(f, count, *, contacts=False, incoming=False, ascending=False, signed=False):
    """Populate source-exact signed columns of the actual sparse-plan W."""
    s, owner = f["solver"], f["owner"]
    W = unpack(owner)
    if ascending:
        W = np.eye(43)
        W[42, :3] = 2.0
        owner.data.W.assign(W[owner.host["row"], owner.host["col"]][None].astype(np.float32))
    rows = count + (3 if contacts else 0)
    packed = np.zeros((1, 100, 18), np.float32)
    support = np.zeros((1, 100), np.int32)
    diagonal = np.zeros(rows, np.float32)
    cfm = np.full(rows, 1e-6, np.float32)
    rhs = np.ones(rows, np.float32)
    types = np.full(rows, 3, np.int32)
    parents = np.full(rows, -1, np.int32)
    mu = np.zeros(rows, np.float32)
    lam = np.zeros(rows, np.float32)
    for row in range(rows):
        column = row % 37
        sign = -1.0 if signed and row % 2 else 1.0
        template = int(owner.host["limit_support"][42 - column])
        if row >= count:
            column = row - count
            template = int(owner.host["limit_support"][42])
        nodes = owner.host["support_nodes"][template]
        live = nodes >= 0
        support[0, row] = template
        packed[0, row, live] = sign * W[nodes[live], column]
        diagonal[row] = float(np.sum(packed[0, row].astype(float) ** 2)) + cfm[row]
        rhs[row] = -0.07 * diagonal[row] if row % 3 == 0 else 0.11 * diagonal[row]
        if incoming:
            lam[row] = 0.025 if row % 4 == 0 else 0.0
    if ascending:
        assert count == 3 and not contacts
        rhs[:] = -5.0
        diagonal[:] = 5.0
        cfm[:] = 0.0
    if contacts:
        types[count:] = [0, 2, 2]
        parents[count + 1 :] = count
        mu[count + 1 :] = 0.5
        lam[count:] = 0.0
    owner.data.Z.assign(packed)
    owner.data.support.assign(support)
    owner.data.incident.zero_()
    s.constraint_count.assign(np.array([rows], np.int32))
    s.v_hat.assign(np.linspace(-0.02, 0.02, 43).astype(np.float32))
    s.v_out.fill_(np.nan)
    for name, values in (
        ("diag", diagonal),
        ("rhs", rhs),
        ("row_type", types),
        ("row_parent", parents),
        ("row_mu", mu),
        ("impulses", lam),
        ("row_cfm", cfm),
    ):
        array = getattr(s, name)
        data = np.zeros(array.shape, dtype=array.numpy().dtype)
        data[0, :rows] = values
        array.assign(data)


class TestSparseLimitJacobi(unittest.TestCase):
    def test_factory(self):
        """Require a separate factory and leave the old spectral owner intact."""
        self.assertEqual(jacobi.get_solve_kernel().key, "sparse_spectral_limit_jacobi43_s18_c100")
        self.assertIs(jacobi.get_solve_kernel(), jacobi.get_solve_kernel())

    def test_owner_and_unchanged_contact_source(self):
        """Keep off-mode ownership and the exact original contact fragment."""
        with patch.dict(os.environ, ENV):
            owner = fixture("cpu")["owner"]
        self.assertTrue(owner.limit_jacobi)
        self.assertIs(owner.kernels.solve, jacobi.get_solve_kernel())
        with patch.dict(os.environ, {**ENV, "FEATHER_PGS_SPARSE_LIMIT_JACOBI": "0"}):
            ordinary = fixture("cpu")["owner"]
        self.assertFalse(ordinary.limit_jacobi)
        self.assertIs(ordinary.kernels.solve, spectral.get_solve_kernel())
        for flags in (
            {"FEATHER_PGS_SPARSE_LIMIT_JACOBI": "bad"},
            {"FEATHER_PGS_SPARSE_SPECTRAL_TANGENTS": "0"},
            {"FEATHER_PGS_SPARSE_PAIRED_GS": "1"},
        ):
            with self.subTest(flags=flags), patch.dict(os.environ, {**ENV, **flags}):
                with self.assertRaises(ValueError):
                    fixture("cpu")
        setup, update = jacobi.get_fragments(100)
        old_setup, old_update = spectral.get_fragments(100)
        self.assertEqual(setup, old_setup + jacobi._SETUP)
        self.assertEqual(update, jacobi._UPDATE + old_update)
        self.assertEqual(jacobi.SCRATCH_BYTES, 688)
        self.assertIn("isfinite(trial)", update)
        self.assertIn("omega==1.0f", update)

    def test_fixed_cpu_map_and_transactional_rejection(self):
        """Check simultaneous reads, denominator-only CFM, and real ascent rejection."""
        z = np.array([[1.0, 0.0], [0.3, np.sqrt(0.91)]])
        value, _, accepted, _ = control.prefix_proposal(z, np.zeros(2), -np.ones(2), np.ones(2))
        np.testing.assert_array_equal(value, [1.0, 1.0])
        self.assertTrue(accepted)
        value, delta, accepted, _ = control.prefix_proposal(
            np.eye(2), np.array([0.25, 0.5]), np.array([-0.5, 0.25]), np.array([2.0, 1.0])
        )
        np.testing.assert_array_equal(value, [0.5, 0.25])
        np.testing.assert_array_equal(delta, [0.25, -0.25])
        self.assertTrue(accepted)
        z = np.c_[np.eye(3), np.full(3, 2.0)]
        _, _, accepted, info = control.prefix_proposal(z, np.zeros(3), np.full(3, -5.0), np.full(3, 5.0))
        self.assertFalse(accepted)
        self.assertEqual(info["energy"], 4.5)
        args = (
            z,
            np.eye(4),
            np.full(3, 5.0),
            np.full(3, -5.0),
            np.full(3, 3),
            np.full(3, -1),
            np.zeros(3),
            np.zeros(4),
        )
        result = control.solve(*args, iterations=1, early_stop=False)
        np.testing.assert_allclose(result.impulses, [1.0, 0.2, 0.04], atol=1e-15)
        self.assertEqual(result.work["sweeps"], 1)
        self.assertEqual(result.work["limit_fallbacks"], 1)

    def test_cpu_zero_incoming_and_nonfinite(self):
        """Preserve zero-budget, zero-step, incoming-delta and finite-admission semantics."""
        old = np.array([0.25, 0.5])
        _, delta, accepted, info = control.prefix_proposal(np.eye(2), np.zeros(2), np.ones(2), np.ones(2))
        self.assertTrue(accepted and info["zero_delta"])
        np.testing.assert_array_equal(delta, 0.0)
        _, _, accepted, _ = control.prefix_proposal(np.eye(2), old, np.array([np.nan, 0.0]), np.ones(2))
        self.assertFalse(accepted)
        args = (
            np.eye(2),
            np.eye(2),
            np.ones(2),
            np.array([-0.5, 0.25]),
            np.full(2, 3),
            np.full(2, -1),
            np.zeros(2),
            np.array([0.3, -0.1]),
        )
        result = control.solve(*args, iterations=1, incoming=old, early_stop=False)
        np.testing.assert_allclose(result.velocity - args[-1], result.impulses - old, atol=1e-15)
        result = control.solve(*args, iterations=0, incoming=old, early_stop=False)
        np.testing.assert_array_equal(result.velocity, args[-1])
        np.testing.assert_array_equal(result.impulses, old)


@unittest.skipUnless(wp.is_cuda_available(), "Root owns native execution")
class TestSparseLimitJacobiCUDA(unittest.TestCase):
    def test_native_prefix_tiles_and_energy_fallback(self):
        """Exercise actual W-column tiles, both signs, incoming data and genuine energy rejection."""
        with patch.dict(os.environ, ENV):
            f = fixture("cuda:0")
        f["owner"].refresh(f["solver"])
        saved_w = f["owner"].data.W.numpy().copy()
        for count in (0, 1, 13, 86, 87):
            with self.subTest(count=count):
                seed_limits(f, count, incoming=True, signed=True)
                check_native(self, f, iterations=2, fallback=count > 86)
        seed_limits(f, 13, contacts=True, incoming=True, signed=True)
        check_native(self, f)
        seed_limits(f, 3, ascending=True)
        _, lam, stats = check_native(self, f, iterations=1)
        np.testing.assert_allclose(lam, [1.0, 0.2, 0.04], rtol=3e-5, atol=3e-6)
        self.assertEqual(stats["limit_fallbacks"], 1)
        f["owner"].data.W.assign(saved_w)
        seed_limits(f, 13, incoming=True)
        check_native(self, f, iterations=2, omega=1.2)
        seed_limits(f, 13)
        check_native(self, f, iterations=0)
        seed_limits(f, 1)
        impulses = f["solver"].impulses.numpy()
        impulses[0, 0] = -0.01
        f["solver"].impulses.assign(impulses)
        check_native(self, f, iterations=1)
        seed_limits(f, 1)
        z = f["owner"].data.Z.numpy()
        z[0, 0, 0] *= 1.25
        f["owner"].data.Z.assign(z)
        check_native(self, f, iterations=1, fallback=True)

    def test_native_current_held_graph_and_empty(self):
        """Reuse actual current/held physical rows, graph replay and empty/regrow controls."""
        with patch.dict(os.environ, ENV), patch.object(metric, "check_native", check_native):
            metric.TestSparseMetricTangentsCUDA.test_native_current_held_graph_and_empty(self)

    def test_native_saved_sixteen(self):
        """Retain independent physical J/H momentum checks on the pinned current/held cases."""
        with patch.dict(os.environ, ENV), patch.object(metric, "check_native", check_native):
            metric.TestSparseMetricTangentsCUDA.test_native_saved_sixteen_current_held_epochs(self)


if __name__ == "__main__":
    unittest.main()
