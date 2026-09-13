"""Compare register residuals with the independent original physical eight."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import sparse_register_gram as gram
from tools.fpgs_bench.test_sparse_factor import fixture, original_eight, physical_rows, unpack


def seed_factor(f):
    """Seed the actual held tree operator independently of candidate GS."""
    owner = f["owner"]
    W = np.linalg.solve(np.linalg.cholesky(f["H"][::-1, ::-1]), np.eye(43))
    owner.data.W.assign(W[owner.host["row"], owner.host["col"]][None].astype(np.float32))
    owner.data.valid.fill_(1)
    owner.data.status.zero_()
    return unpack(owner)


def synthetic(f, count, *, stationary=False, incoming=False):
    """Bind sparse rows on the real held G1 operator without candidate oracles."""
    s, owner = f["solver"], f["owner"]
    W = seed_factor(f)
    rng = np.random.default_rng(101 + count)
    support = rng.integers(0, len(owner.host["support_count"]), count)
    z = np.zeros((100, 18), np.float32)
    full = np.zeros((count, 43), np.float64)
    for row, tpl in enumerate(support):
        n = owner.host["support_count"][tpl]
        z[row, :n] = rng.normal(0, 0.3, n)
        full[row, owner.host["support_nodes"][tpl, :n]] = z[row, :n]
    types = np.full(100, 3, np.int32)
    parent = np.full(100, -1, np.int32)
    mu = np.full(100, 0.6, np.float32)
    for r in range(0, count - 2, 3):
        types[r : r + 3] = (0, 2, 2)
        parent[r + 1 : r + 3] = r
    # Express original physical J/Y from the held operator and authored Z.
    J = np.linalg.solve(W, full.T).T[:, ::-1]
    Y = (full @ W)[:, ::-1]
    vhat = rng.normal(0, 0.01, 43).astype(np.float32)
    incident = (J @ vhat).astype(np.float32)
    rhs = rng.normal(-0.2, 0.1, count).astype(np.float32)
    rhs[types[:count] == 2] += rng.choice((-1, 1), np.sum(types[:count] == 2)) * 0.7
    if stationary:
        rhs[types[:count] != 2] = 1.0 - incident[types[:count] != 2]
    diagonal = (np.sum(full * full, axis=1) + 0.01).astype(np.float32)
    impulses = np.zeros(100, np.float32)
    if incoming:
        impulses[:count] = rng.uniform(0, 0.02, count)
    s.constraint_count.assign(np.array([count], np.int32))
    s.row_type.assign(types[None])
    s.row_parent.assign(parent[None])
    s.row_mu.assign(mu[None])
    s.diag.assign(np.pad(diagonal, (0, 100 - count))[None])
    s.rhs.assign(np.pad(rhs, (0, 100 - count))[None])
    s.impulses.assign(impulses[None])
    s.v_hat.assign(vhat)
    s.v_out.assign(vhat)
    owner.data.Z.assign(z[None])
    owner.data.support.assign(np.pad(support.astype(np.int32), (0, 100 - count))[None])
    owner.data.incident.assign(np.pad(incident, (0, 100 - count))[None])
    # Match the actually rounded incident while retaining independent J/Y action.
    physical_rhs = rhs.astype(float) + incident - J @ vhat
    return J, Y, diagonal, physical_rhs, types[:count], parent[:count], mu[:count], vhat, impulses[:count]


def launched(f, problem, *, incoming=False, friction_start=0):
    """Compare velocity/action and current cones, never exact impulse bits."""
    s, owner = f["solver"], f["owner"]
    gram.install(owner)
    if incoming:
        expected, expected_lam = original_warm(problem, friction_start)
    else:
        expected, expected_lam = original_eight(*problem[:8])
    owner.solve(s.rhs, 8, 1.0, friction_start)
    owner.check()
    actual = s.v_out.numpy().astype(float)
    lam = s.impulses.numpy()[0, : len(problem[0])].astype(float)
    scale = 1 + np.max(np.abs(expected))
    error = np.max(np.abs(actual - expected)) / scale
    if not np.isfinite(actual).all() or error > 3e-5:
        raise AssertionError(f"physical eight velocity error {error:.9g}")
    # Cold output must close against the actual held response action.
    if not incoming:
        np.testing.assert_allclose(actual, problem[7] + problem[1].T @ lam, rtol=3e-5, atol=3e-6)
    for r, kind in enumerate(problem[4]):
        if kind in (0, 3) and lam[r] < -1e-7:
            raise AssertionError("Negative unilateral impulse")
        if kind == 2 and r == problem[5][r] + 1:
            p = problem[5][r]
            if np.hypot(lam[r], lam[r + 1]) > problem[6][r] * lam[p] + 3e-6:
                raise AssertionError("Current friction disk violation")
    return actual, lam, expected_lam


def original_warm(problem, friction_start):
    """Preserve the original applied-delta law including pre-friction clearing."""
    J, Y, diag, rhs, types, parent, mu, vhat, initial = problem
    v, lam = vhat.astype(float).copy(), initial.astype(float).copy()
    for iteration in range(8):
        for row in range(len(J)):
            if types[row] == 2 and iteration < friction_start:
                lam[row] = 0
                continue
            old = lam[row]
            new = old - (J[row] @ v + rhs[row]) / diag[row]
            if types[row] != 2:
                new = max(new, 0)
            else:
                p = parent[row]
                radius = max(mu[row] * lam[p], 0)
                if radius <= 0:
                    new = 0
                else:
                    sibling = p + 2 if row == p + 1 else p + 1
                    mag = np.hypot(new, lam[sibling])
                    if mag > radius:
                        scale = radius / mag
                        new *= scale
                        other = lam[sibling] * scale
                        v += Y[sibling] * (other - lam[sibling])
                        lam[sibling] = other
            lam[row] = new
            v += Y[row] * (new - old)
    return v, lam


class TestSparseRegisterPhysical(unittest.TestCase):
    """Run independent small controls on the CPU native implementation."""

    def test_boundaries_stationarity_and_siblings(self):
        """Exercise0/31/32/33/64, stationary rows and cross-lane friction siblings."""
        f = fixture()
        for count in (0, 31, 32, 33, 64):
            for stationary in (False, True):
                with self.subTest(count=count, stationary=stationary):
                    problem = synthetic(f, count, stationary=stationary)
                    actual, lam, _ = launched(f, problem)
                    if stationary:
                        np.testing.assert_array_equal(actual, problem[7])
                        self.assertFalse(np.any(lam))
        for count in (33, 64):
            with self.subTest(count=count, incoming=True):
                launched(f, synthetic(f, count, incoming=True), incoming=True, friction_start=2)

    def test_current_geometry_and_stale_guard(self):
        """Keep actual current rows and held action while rejecting stale status."""
        f = fixture()
        s, o = f["solver"], f["owner"]
        W = seed_factor(f)
        s.v_hat.assign(np.linspace(-0.1, 0.1, 43, dtype=np.float32))
        o.build_rows(f["state"], s, f["contacts"], 0.0025)
        s._stage4_compute_rhs_world(0.0025)
        o.restitution(0.0025)
        count = int(s.constraint_count.numpy()[0])
        J = physical_rows(f)
        Z = (W @ J[:, ::-1].T).T
        Y = (Z @ W)[:, ::-1]
        # The inherited contact native body is CUDA-only. CPU controls bind
        # its mathematical held response from independent current geometry;
        # this is not a CPU claim about the unchanged contact producer.
        templates = o.data.support.numpy()[0].copy()
        shape_bodies = f["model"].shape_body.numpy()
        c = f["contacts"]
        shapes0 = c.rigid_contact_shape0.numpy()
        shapes1 = c.rigid_contact_shape1.numpy()
        slots = s.contact_slot.numpy()
        needed = s.contact_slots_needed.numpy()
        for raw in range(int(c.rigid_contact_count.numpy()[0])):
            a, b = shape_bodies[shapes0[raw]], shape_bodies[shapes1[raw]]
            ta = o.host["body_tag"][a] if a >= 0 else 0
            tb = o.host["body_tag"][b] if b >= 0 else 0
            templates[slots[raw] : slots[raw] + needed[raw]] = o.host["pair_support"][ta, tb]
        pack = np.zeros((100, 18), np.float32)
        for row in range(count):
            tpl = templates[row]
            n = o.host["support_count"][tpl]
            pack[row, :n] = Z[row, o.host["support_nodes"][tpl, :n]]
        o.data.support.assign(templates[None])
        o.data.Z.assign(pack[None])
        o.data.incident.assign(np.pad((J @ s.v_hat.numpy()).astype(np.float32), (0, 100 - count))[None])
        diag = np.zeros(100, np.float32)
        diag[:count] = np.sum(pack[:count] ** 2, axis=1) + s.row_cfm.numpy()[0, :count]
        s.diag.assign(diag[None])
        rhs = s.rhs.numpy()[0, :count].astype(float)
        rhs += o.data.incident.numpy()[0, :count] - J @ s.v_hat.numpy()
        problem = (
            J,
            Y,
            s.diag.numpy()[0, :count],
            rhs,
            s.row_type.numpy()[0, :count],
            s.row_parent.numpy()[0, :count],
            s.row_mu.numpy()[0, :count],
            s.v_hat.numpy(),
            np.zeros(count),
        )
        s.impulses.zero_()
        s.v_out.assign(s.v_hat)
        launched(f, problem)
        old = s.v_out.numpy().copy()
        o.data.status.fill_(4)
        o.solve(s.rhs, 8, 1.0, 0)
        np.testing.assert_array_equal(s.v_out.numpy(), old)


@unittest.skipUnless(wp.is_cuda_available(), "Root owns the paired CUDA lease")
class TestSparseRegisterCUDA(unittest.TestCase):
    """Exercise the actual CUDA owners and unmodified65-row fallback."""

    def test_boundaries_original_eight_and_fallback(self):
        """Compare0/31/32/33/64/65 and loaded sibling updates on the real GPU."""
        f = fixture("cuda:0")
        for count in (0, 31, 32, 33, 64, 65):
            for stationary in (False, True):
                with self.subTest(count=count, stationary=stationary):
                    launched(f, synthetic(f, count, stationary=stationary))
        for count in (33, 64, 65):
            launched(f, synthetic(f, count, incoming=True), incoming=True, friction_start=2)


if __name__ == "__main__":
    unittest.main()
