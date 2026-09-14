# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the complete sparse packet owner without a second response producer."""

import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import sparse_packet_rows as packet
from newton._src.solvers.feather_pgs.kernels import _FPGS_CONTACT_END_GAP_SLOP
from tools.fpgs_bench.test_sparse_factor import fixture, original_eight, physical_rows, unpack


def prepared(device="cpu"):
    """Build actual current packet geometry and an independently seeded held W."""
    with patch.dict(os.environ, {"FEATHER_PGS_SPARSE_PACKETS": "1"}):
        f = fixture(device)
    s, o = f["solver"], f["owner"]
    W = np.linalg.solve(np.linalg.cholesky(f["H"][::-1, ::-1]), np.eye(43))
    o.data.W.assign(W[o.host["row"], o.host["col"]][None].astype(np.float32))
    o.data.valid.fill_(1)
    o.data.status.zero_()
    s.v_hat.assign(np.linspace(-0.11, 0.09, 43, dtype=np.float32))
    o.build_rows(f["state"], s, f["contacts"], 0.0025)
    s._stage4_compute_rhs_world(0.0025)
    o.restitution(0.0025)
    s.impulses.zero_()
    s.v_out.assign(s.v_hat)
    return f


def problem(f):
    """Reconstruct original physical rows and response without candidate Z."""
    s, o = f["solver"], f["owner"]
    count = int(s.constraint_count.numpy()[0])
    J = physical_rows(f)
    W = unpack(o)
    Z = (W @ J[:, ::-1].T).T
    Y = (Z @ W)[:, ::-1]
    diagonal = np.sum(Z * Z, axis=1) + s.row_cfm.numpy()[0, :count]
    rhs = s.rhs.numpy()[0, :count].astype(float)
    types = s.row_type.numpy()[0, :count]
    before = s.v_hat.numpy().astype(float)
    incident = J @ before
    target = s.target_velocity.numpy()[0, :count]
    phi = s.phi.numpy()[0, :count]
    restitution = s.row_restitution.numpy()[0, :count]
    relative = incident - target
    fires = (
        (types == 0)
        & (restitution > 0)
        & (relative < -s._effective_restitution_velocity_threshold)
        & ((phi <= _FPGS_CONTACT_END_GAP_SLOP) | (phi + 0.0025 * relative <= _FPGS_CONTACT_END_GAP_SLOP))
    )
    rhs[fires] = -target[fires] + restitution[fires] * relative[fires]
    return J, Y, diagonal, rhs, types, s.row_parent.numpy()[0, :count], s.row_mu.numpy()[0, :count], before


def original_warm(reference, initial, friction_start):
    """Apply the original sequential impulse deltas, including delayed clearing."""
    J, Y, diagonal, rhs, types, parent, mu, before = reference
    v, lam = before.copy(), initial.astype(float).copy()
    for iteration in range(8):
        for row in range(len(J)):
            if types[row] == 2 and iteration < friction_start:
                lam[row] = 0
                continue
            old = lam[row]
            new = old - (J[row] @ v + rhs[row]) / diagonal[row]
            if types[row] != 2:
                new = max(new, 0)
            else:
                par = parent[row]
                radius = max(mu[row] * lam[par], 0)
                if radius <= 0:
                    new = 0
                else:
                    sibling = par + 2 if row == par + 1 else par + 1
                    magnitude = np.hypot(new, lam[sibling])
                    if magnitude > radius:
                        scale = radius / magnitude
                        new *= scale
                        other = lam[sibling] * scale
                        v += Y[sibling] * (other - lam[sibling])
                        lam[sibling] = other
            lam[row] = new
            v += Y[row] * (new - old)
    return v, lam


def compare(f, *, reference=None, initial=None, friction_start=0):
    """Check original physical eight, final held action and current friction cones."""
    s, o = f["solver"], f["owner"]
    reference = problem(f) if reference is None else reference
    expected, _ = original_eight(*reference) if initial is None else original_warm(reference, initial, friction_start)
    o.solve(s.rhs, 8, 1.0, friction_start)
    o.check()
    actual = s.v_out.numpy().astype(float)
    lam = s.impulses.numpy()[0, : len(reference[0])].astype(float)
    error = np.max(np.abs(actual - expected)) / (1 + np.max(np.abs(expected)))
    if not np.isfinite(actual).all() or error > 3e-5:
        raise AssertionError(f"Packet physical-eight scaled velocity error {error:.9g}")
    if initial is None:
        np.testing.assert_allclose(actual, reference[7] + reference[1].T @ lam, rtol=3e-5, atol=3e-6)
    for row, kind in enumerate(reference[4]):
        if kind in (0, 3) and lam[row] < -1e-7:
            raise AssertionError("Negative unilateral impulse")
        if kind == 2 and row == reference[5][row] + 1:
            par = reference[5][row]
            if np.hypot(lam[row], lam[row + 1]) > reference[6][row] * lam[par] + 3e-6:
                raise AssertionError("Current friction disk violation")
    return actual


def check_packet_operator(f):
    """Reuse original current/held factor fixture with independently rebuilt rows."""
    s, o = f["solver"], f["owner"]
    if not o.packet_rows or o.kernels.solve.key != "sparse_packet_gs43_s18_c100":
        raise AssertionError("The actual packet solver is not active")
    s._stage4_compute_rhs_world(0.0025)
    o.restitution(0.0025)
    s.impulses.zero_()
    compare(f)
    q, qi = f["state"].joint_q.numpy(), s._joint_limit_q_index.numpy()
    lower = np.full(43, -np.inf, np.float32)
    upper = -lower
    lower[qi >= 0] = q[qi[qi >= 0]] - 0.001
    upper[qi >= 0] = q[qi[qi >= 0]] + 0.001
    f["model"].joint_limit_lower.assign(lower)
    f["model"].joint_limit_upper.assign(upper)
    for friction, expected in ((False, 77), (True, 83)):
        s.enable_contact_friction = friction
        if friction:
            s.contact_shared_anchor = True
            shapes = f["model"].shape_body.numpy()
            ids = f["contacts"].rigid_contact_shape1.numpy().copy()
            ids[0] = int(np.flatnonzero(shapes == 13)[0])
            f["contacts"].rigid_contact_shape1.assign(ids)
            s.contact_gap_gate = float("inf")
        o.build_rows(f["state"], s, f["contacts"], 0.0025)
        if int(s.constraint_count.numpy()[0]) != expected:
            raise AssertionError("Original finite prefix/reservation count changed")
        s._stage4_compute_rhs_world(0.0025)
        s.impulses.zero_()
        compare(f)
    saved = s.v_out.numpy().copy()
    o.data.status.fill_(4)
    o.solve(s.rhs, 8, 1.0, 0)
    np.testing.assert_array_equal(s.v_out.numpy(), saved)


class TestSparsePacketRows(unittest.TestCase):
    """Exercise the new owner API and generated source contracts."""

    def test_owner_api(self):
        """Require the packet producer and local-row solver factories."""
        self.assertTrue(callable(packet.install))
        self.assertTrue(callable(packet.get_solve_kernel))

    def test_unsupported_capacity_retains_sparse_rows(self):
        """Keep unfunded shared-memory capacities on the original sparse owner."""
        with patch.dict(os.environ, {"FEATHER_PGS_SPARSE_PACKETS": "1"}):
            f = fixture(getattr(self, "device", "cpu"), capacity=128)
        o = f["owner"]
        self.assertFalse(o.packet_rows)
        self.assertEqual(o.data.Z.shape, (1, 128, 18))
        self.assertEqual(o.data.incident.shape, (1, 128))
        self.assertEqual(o.data.support.shape, (1, 128))
        self.assertNotIn("packet", o.kernels.contacts.key)
        self.assertNotIn("packet", o.kernels.solve.key)

    def test_current_geometry_held_and_packet_lifetime(self):
        """Exercise actual current geometry, held reuse and changing packet prefixes."""
        f = prepared(getattr(self, "device", "cpu"))
        s, o = f["solver"], f["owner"]
        self.assertEqual(o.data.Z.shape, (1, 1, 1))
        self.assertEqual(o.data.incident.shape, (1, 1))
        self.assertEqual(o.data.support.shape, (1, 100))
        compare(f)
        held = o.data.W.numpy().copy()
        # Same held operator, changed current velocity and limit/contact activity.
        s.v_hat.assign(np.linspace(0.06, -0.08, 43, dtype=np.float32))
        s.enable_contact_friction = False
        o.build_rows(f["state"], s, f["contacts"], 0.0025)
        s._stage4_compute_rhs_world(0.0025)
        s.impulses.zero_()
        compare(f)
        np.testing.assert_array_equal(o.data.W.numpy(), held)
        # Both dynamic feet and shared contact anchors retain the 18-entry union.
        s.enable_contact_friction = True
        s.contact_shared_anchor = True
        shape_bodies = f["model"].shape_body.numpy()
        ids = f["contacts"].rigid_contact_shape1.numpy().copy()
        ids[0] = int(np.flatnonzero(shape_bodies == 13)[0])
        f["contacts"].rigid_contact_shape1.assign(ids)
        s.contact_gap_gate = float("inf")
        o.build_rows(f["state"], s, f["contacts"], 0.0025)
        s._stage4_compute_rhs_world(0.0025)
        s.impulses.zero_()
        compare(f)
        saved = s.v_out.numpy().copy()
        o.data.status.fill_(4)
        o.solve(s.rhs, 8, 1.0, 0)
        np.testing.assert_array_equal(s.v_out.numpy(), saved)

    def test_retired_global_rows_and_restitution_predicate(self):
        """Require local sparse row use and the original end-gap trigger."""
        source = packet.cuda_source(100)
        self.assertNotIn("d.Z.data", source)
        self.assertNotIn("d.incident.data", source)
        self.assertIn("zrows[row*18+lane]", source)
        self.assertIn("phi+x.dt*velocity<=1e-06f", source)
        self.assertIn("phi<=1e-06f", source)
        self.assertNotIn("#pragma unroll", source)

    def test_full_capacity_empty_and_delayed_siblings(self):
        """Exercise 100 actual-geometry packet rows, delayed siblings and empty reuse."""
        f = prepared(getattr(self, "device", "cpu"))
        s, o = f["solver"], f["owner"]
        old = problem(f)
        normal = int(np.flatnonzero(old[4] == 0)[0])
        raw = int(o.data.support.numpy()[0, normal] // 3)
        keys = np.empty(100, np.int32)
        types = np.full(100, 3, np.int32)
        parents = np.full(100, -1, np.int32)
        J = np.zeros((100, 43))
        for row in range(0, 99, 3):
            keys[row : row + 3] = 3 * raw + np.arange(3)
            types[row : row + 3] = (0, 2, 2)
            parents[row + 1 : row + 3] = row
            J[row : row + 3] = old[0][normal : normal + 3]
        keys[99] = -13  # Lower bound on physical DOF6.
        J[99, 6] = 1
        o.data.support.assign(keys[None])
        s.constraint_count.assign(np.array([100], np.int32))
        s.row_type.assign(types[None])
        s.row_parent.assign(parents[None])
        s.row_mu.fill_(0.6)
        s.row_cfm.fill_(0.01)
        s.row_restitution.zero_()
        rhs = np.tile(np.array([-0.2, 0.7, -0.8], np.float32), 34)[:100]
        s.rhs.assign(rhs[None])
        W = unpack(o)
        Z = (W @ J[:, ::-1].T).T
        Y = (Z @ W)[:, ::-1]
        ref = (
            J,
            Y,
            np.sum(Z * Z, axis=1) + 0.01,
            rhs.astype(float),
            types,
            parents,
            np.full(100, 0.6),
            s.v_hat.numpy().astype(float),
        )
        compare(f, reference=ref)
        initial = np.linspace(0.001, 0.02, 100, dtype=np.float32)
        s.impulses.assign(initial[None])
        compare(f, reference=ref, initial=initial, friction_start=2)
        s.constraint_count.zero_()
        s.v_out.fill_(123)
        o.solve(s.rhs, 8, 1.0, 0)
        np.testing.assert_array_equal(s.v_out.numpy(), s.v_hat.numpy())

    def test_restitution_reaches_surface_with_original_slop(self):
        """Keep the original predicted end-gap slop at the restitution threshold."""
        f = prepared(getattr(self, "device", "cpu"))
        s = f["solver"]
        ref = problem(f)
        row = int(np.flatnonzero(ref[4] == 0)[0])
        # Author a closing predictor and a gap inside the 1e-6 end-gap allowance.
        direction = ref[0][row]
        velocity = -direction / (direction @ direction)
        s.v_hat.assign(velocity.astype(np.float32))
        incident = ref[0] @ s.v_hat.numpy()
        target = s.target_velocity.numpy()[0]
        phi = s.phi.numpy()
        phi[0, row] = -0.0025 * (incident[row] - target[row]) + 0.5e-6
        s.phi.assign(phi)
        restitution = s.row_restitution.numpy()
        restitution[0, row] = 0.7
        s.row_restitution.assign(restitution)
        s.impulses.zero_()
        compare(f)


@unittest.skipUnless(wp.is_cuda_available(), "Root owns the paired CUDA lease")
class TestSparsePacketCUDA(TestSparsePacketRows):
    """Run the actual packet-to-local-row owner on current CUDA geometry."""

    device = "cuda:0"

    def test_full_packet_allowance(self):
        """Run full capacity, empty, delayed siblings and restitution slop on CUDA."""
        self.test_full_capacity_empty_and_delayed_siblings()
        self.test_restitution_reaches_surface_with_original_slop()


if __name__ == "__main__":
    unittest.main()
