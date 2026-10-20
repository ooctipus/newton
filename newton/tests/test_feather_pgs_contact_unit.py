# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the exact local solve of one contact unit against the Coulomb conditions it must satisfy.

Each case builds a unit's Delassus block and residual, solves it with
:mod:`newton._src.solvers.feather_pgs.contact_unit` (the CPU binding of the one
C++ law) and checks the solution in float64: non-negative normal impulses whose
rows do not approach and are complementary, friction inside the disk of the
final load, and a sliding friction that opposes the sliding velocity on the
disk's boundary (maximal dissipation).
"""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import contact_unit
from newton._src.solvers.feather_pgs.device_preludes import cuda_native

wp.init()

_POINT_FIELDS = 13  # w_nn, w_t0, w_t01, w_t1, w_nt0, w_nt1, b_n, b_t0, b_t1, mu, ring_load, alpha, cap
_CPU = "cpu"


@wp.kernel
def _solve_points(problems: wp.array2d[float], out: wp.array2d[float]):
    i = wp.tid()
    r = contact_unit.contact_unit_solve(
        problems[i, 0],
        problems[i, 1],
        problems[i, 2],
        problems[i, 3],
        problems[i, 4],
        problems[i, 5],
        problems[i, 6],
        problems[i, 7],
        problems[i, 8],
        problems[i, 9],
        problems[i, 10],
        problems[i, 11],
        int(problems[i, 12]),
    )
    for j in range(5):
        out[i, j] = r[j]


@wp.kernel
def _latch(states: wp.array2d[float], out: wp.array2d[float]):
    i = wp.tid()
    r = contact_unit.contact_unit_latch(states[i, 0], states[i, 1], states[i, 2], states[i, 3])
    out[i, 0] = r[0]
    out[i, 1] = r[1]
    out[i, 2] = r[2]


@wp.kernel
def _solve_ring(
    block: wp.array[float],
    normals: int,
    anchors: int,
    mu0: float,
    mu1: float,
    evaluations: wp.array[int],
):
    evaluations[0] = contact_unit.contact_unit_ring(
        block,
        0,
        normals,
        anchors,
        mu0,
        mu1,
        contact_unit.CONTACT_UNIT_RING_PASSES,
        contact_unit.CONTACT_UNIT_NEWTON_CAP,
        contact_unit.CONTACT_UNIT_PROXIMAL,
    )


_NATIVE_BODY = r"""
    const float* p = problems.data + i * 13;
    const ContactUnitSolution s = ContactUnit::solve(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10],
                                                     p[11], static_cast<int>(p[12]));
    float* o = out.data + i * 5;
    o[0] = s.normal;
    o[1] = s.friction0;
    o[2] = s.friction1;
    o[3] = s.alpha;
    o[4] = static_cast<float>(s.evaluations);
#endif
"""


@wp.func_native(cuda_native(_NATIVE_BODY, preludes=(contact_unit.CONTACT_UNIT,)))
def _native_solve(problems: wp.array2d[float], out: wp.array2d[float], i: int): ...


@wp.kernel
def _solve_points_native(problems: wp.array2d[float], out: wp.array2d[float]):
    _native_solve(problems, out, wp.tid())


def solve_points(problems, device=_CPU):
    """Solve rows of point problems (``_POINT_FIELDS`` columns) and return (normal, f0, f1, alpha, evaluations)."""
    problems = np.atleast_2d(np.asarray(problems, np.float32))
    out = wp.zeros((len(problems), 5), dtype=float, device=device)
    wp.launch(
        _solve_points, dim=len(problems), inputs=[wp.array(problems, dtype=float, device=device), out], device=device
    )
    return out.numpy().astype(np.float64)


def point_problem(W, b, mu, ring_load=0.0, alpha=-1.0, cap=contact_unit.CONTACT_UNIT_NEWTON_CAP):
    """Pack a symmetric 3x3 block [normal, t0, t1] and its residual into one problem row."""
    W = np.asarray(W, np.float64)
    return [W[0, 0], W[1, 1], W[1, 2], W[2, 2], W[0, 1], W[0, 2], b[0], b[1], b[2], mu, ring_load, alpha, cap]


def disk_projection(f, radius):
    norm = np.linalg.norm(f)
    return f if norm <= radius else f * (max(radius, 0.0) / norm)


def point_residual(W, b, mu, x, ring_load=0.0):
    """Natural residual [m/s] of a point (or anchor) unit's local problem at ``x``: zero exactly at its solution."""
    W = np.asarray(W, np.float64)
    v = np.asarray(b, np.float64) + W @ x
    normal = abs(min(x[0] * W[0, 0], v[0]))
    d = max(W[1, 1], W[2, 2])
    f = x[1:]
    radius = mu * (x[0] + ring_load)
    friction = d * np.linalg.norm(f - disk_projection(f - v[1:] / d, radius))
    return max(normal, friction)


def random_block(rng, coupling=1.0, cfm=1.0e-6):
    """A random SPD contact block whose normal-tangent coupling is scaled by ``coupling``."""
    J = rng.normal(size=(3, 6))
    W = J @ J.T * rng.uniform(0.1, 4.0)
    W[1:, 0] *= coupling
    W[0, 1:] *= coupling
    w, V = np.linalg.eigh(W)
    W = V @ np.diag(np.maximum(w, 1.0e-3 * w.max())) @ V.T
    return W + cfm * np.eye(3)


class TestContactUnitPoint(unittest.TestCase):
    """The 3x3 solve of a point contact or a patch anchor: take-off, stick, slide."""

    def test_take_off_of_a_separating_contact(self):
        """Leave a contact that separates at zero impulse without impulse."""
        W = np.array([[2.0, 0.3, -0.2], [0.3, 1.5, 0.1], [-0.2, 0.1, 1.2]])
        out = solve_points(point_problem(W, [0.4, -0.3, 0.2], 0.8))[0]
        np.testing.assert_array_equal(out[:3], 0.0)
        self.assertEqual(out[3], -1.0)

    def test_stick_holds_the_contact_still(self):
        """Stick a contact whose friction cone admits the sticking force: zero velocity on every row."""
        W = np.array([[2.0, 0.3, -0.2], [0.3, 1.5, 0.1], [-0.2, 0.1, 1.2]])
        b = np.array([-1.0, 0.05, -0.04])
        out = solve_points(point_problem(W, b, 1.0))[0]
        x = out[:3]
        np.testing.assert_allclose(b + W @ x, 0.0, atol=1.0e-6)
        expected = np.linalg.solve(W, -b)
        np.testing.assert_allclose(x, expected, rtol=1.0e-5)
        self.assertLessEqual(np.linalg.norm(x[1:]), 1.0 * x[0])
        self.assertEqual(out[3], -1.0)

    def test_isotropic_slide_matches_the_closed_form(self):
        """Slide an uncoupled isotropic contact to the closed form: full normal stop, friction mu x_n against slip."""
        W = np.diag([2.0, 1.5, 1.5])
        b = np.array([-1.0, 3.0, -4.0])
        mu = 0.5
        out = solve_points(point_problem(W, b, mu))[0]
        normal = 1.0 / 2.0
        expected_friction = -mu * normal * b[1:] / np.linalg.norm(b[1:])
        self.assertAlmostEqual(out[0], normal, places=6)
        np.testing.assert_allclose(out[1:3], expected_friction, rtol=1.0e-5, atol=1.0e-7)
        # The sliding velocity opposes the friction: the root's alpha makes v_T = alpha * (-f).
        v_t = b[1:] + W[1:, 1:] @ out[1:3]
        self.assertGreater(out[3], 0.0)
        np.testing.assert_allclose(v_t, -out[3] * out[1:3], rtol=1.0e-4)

    def test_coupled_slide_satisfies_coulomb(self):
        """Slide contacts whose friction lifts or presses their normal and satisfy the local problem to 1e-5."""
        rng = np.random.default_rng(3)
        rows, blocks, residuals, mus = [], [], [], []
        for _ in range(400):
            W = random_block(rng)
            b = np.array([-rng.uniform(0.1, 2.0), *rng.normal(size=2) * rng.uniform(0.5, 5.0)])
            mu = rng.uniform(0.2, 1.2)
            rows.append(point_problem(W, b, mu))
            blocks.append((W, b))
            mus.append(mu)
        out = solve_points(rows)
        sliding = 0
        for (W, b), mu, o in zip(blocks, mus, out, strict=True):
            x = o[:3]
            scale = max(abs(b).max(), 1.0)
            residuals.append(point_residual(W, b, mu, x) / scale)
            self.assertGreaterEqual(x[0], 0.0)
            self.assertLessEqual(np.linalg.norm(x[1:]), mu * x[0] * (1.0 + 1.0e-6) + 1.0e-12)
            sliding += o[3] > 0.0
        self.assertGreater(sliding, 100)
        self.assertLess(max(residuals), 1.0e-5)

    def test_rank_one_tangent_block(self):
        """Solve a contact whose tangent block has rank one, sticking and sliding, to the local problem."""
        t = np.array([1.0, 0.0])
        W = np.zeros((3, 3))
        W[0, 0] = 2.0
        W[1:, 1:] = np.outer(t, t) * 1.5
        W[0, 1:] = W[1:, 0] = [0.4, 0.0]
        for b in ([-1.0, 0.1, 0.0], [-1.0, 3.0, 0.0], [-1.0, 0.2, 0.5], [-1.0, -2.0, 1.0]):
            with self.subTest(b=b):
                out = solve_points(point_problem(W, b, 0.6))[0]
                self.assertTrue(np.all(np.isfinite(out)))
                self.assertLess(point_residual(W, b, 0.6, out[:3]), 1.0e-5)
        # The stick of a residual in the block's range is exact.
        out = solve_points(point_problem(W, [-1.0, 0.1, 0.0], 0.6))[0]
        np.testing.assert_allclose((np.array([-1.0, 0.1, 0.0]) + W @ out[:3])[:2], 0.0, atol=1.0e-6)

    def test_rank_one_block_takes_the_closed_form(self):
        """Slide a rank-one 3x3 block, whose Schur complement vanishes, by the closed-form root in one evaluation."""
        u = np.array([1.0, 0.5, 0.25])
        W = np.outer(u, u)
        b = np.array([-1.0, 0.75, -2.0])
        mu = 0.5
        out = solve_points(point_problem(W, b, mu))[0]
        # Closed form: c = b_T - (b_n / W_nn) w, g = -w / W_nn, S0 = -b_n / W_nn, alpha = (|c| + mu g.c) / (mu S0).
        w = W[0, 1:]
        c = b[1:] - b[0] / W[0, 0] * w
        g = -w / W[0, 0]
        S0 = -b[0] / W[0, 0]
        alpha = (np.linalg.norm(c) + mu * g @ c) / (mu * S0)
        s = c / alpha
        normal = (w @ s - b[0]) / W[0, 0]
        np.testing.assert_allclose(out[0], normal, rtol=1.0e-5)
        np.testing.assert_allclose(out[1:3], -s, rtol=1.0e-5)
        np.testing.assert_allclose(out[3], alpha, rtol=1.0e-5)
        self.assertEqual(out[4], 2.0)
        self.assertLess(point_residual(W, b, mu, out[:3]), 1.0e-5)

    def test_anchor_take_off_keeps_friction_on_the_ring_load(self):
        """Lift an anchor off while its ring keeps load L: its friction minimizes the tangent energy on radius mu L."""
        W = np.array([[2.0, 0.2, 0.1], [0.2, 1.5, 0.3], [0.1, 0.3, 1.2]])
        mu, ring_load = 0.5, 0.4
        for b, sliding in (([1.0, 0.05, -0.02], False), ([1.0, 3.0, -2.0], True)):
            with self.subTest(sliding=sliding):
                out = solve_points(point_problem(W, b, mu, ring_load=ring_load))[0]
                self.assertEqual(out[0], 0.0)
                f = out[1:3]
                v_t = np.asarray(b[1:]) + W[1:, 1:] @ f
                radius = mu * ring_load
                if sliding:
                    self.assertAlmostEqual(np.linalg.norm(f), radius, places=6)
                    # Maximal dissipation: friction against the sliding velocity.
                    cosine = f @ v_t / (np.linalg.norm(f) * np.linalg.norm(v_t))
                    self.assertLess(cosine, -1.0 + 1.0e-5)
                else:
                    np.testing.assert_allclose(v_t, 0.0, atol=1.0e-6)
                    self.assertLess(np.linalg.norm(f), radius)
                self.assertLess(point_residual(W, b, mu, out[:3], ring_load=ring_load), 1.0e-5)

    def test_no_friction_coefficient_stops_the_normal_alone(self):
        """Solve a contact without a friction coefficient as its frictionless normal row."""
        W = np.array([[2.0, 0.4, -0.3], [0.4, 1.5, 0.1], [-0.3, 0.1, 1.2]])
        out = solve_points(point_problem(W, [-1.0, 2.0, 1.0], 0.0))[0]
        self.assertAlmostEqual(out[0], 0.5, places=6)
        np.testing.assert_array_equal(out[1:3], 0.0)

    def test_slide_root_lies_in_its_bracket(self):
        """Find every slide root inside the bracket [0, |c| (1 + mu |g|) / (mu S0) - e2] the Newton iteration keeps."""
        rng = np.random.default_rng(11)
        checked = 0
        rows, cases = [], []
        for _ in range(600):
            W = random_block(rng, coupling=rng.uniform(0.0, 2.0))
            b = np.array([-rng.uniform(0.05, 2.0), *rng.normal(size=2) * rng.uniform(0.1, 6.0)])
            mu = rng.uniform(0.1, 1.5)
            rows.append(point_problem(W, b, mu))
            cases.append((W, b, mu))
        out = solve_points(rows)
        for (W, b, mu), o in zip(cases, out, strict=True):
            if o[3] <= 0.0:
                continue
            w = W[0, 1:]
            A = W[1:, 1:] - np.outer(w, w) / W[0, 0]
            c = b[1:] - b[0] / W[0, 0] * w
            g = -w / W[0, 0]
            S0 = -b[0] / W[0, 0]
            e2 = np.linalg.eigvalsh(A)[0]
            high = np.linalg.norm(c) * (1.0 + mu * np.linalg.norm(g)) / (mu * S0) - e2

            def gap(alpha, A=A, c=c, g=g, S0=S0, mu=mu):
                s = np.linalg.solve(A + alpha * np.eye(2), c)
                return np.linalg.norm(s) - mu * (S0 - g @ s)

            # The bracket's ends straddle the root: slipping at 0, inside the cone at its upper end.
            self.assertGreater(gap(0.0), 0.0)
            self.assertLessEqual(gap(high), 1.0e-9 * max(1.0, mu * S0))
            self.assertGreaterEqual(o[3], 0.0)
            self.assertLessEqual(o[3], high * (1.0 + 1.0e-5) + 1.0e-6)
            self.assertLess(abs(gap(o[3])), 1.0e-4 * max(1.0, mu * S0))
            checked += 1
        self.assertGreater(checked, 150)

    def test_non_unique_problem_takes_the_first_branch(self):
        """Take off where the contact's own problem also has a loaded stick and a loaded slide.

        Friction that presses the normal down, with ``mu w.u / W_nn`` 1.5 here, lets a contact that separates at
        zero impulse hold load as well: this problem has three exact solutions. The solve takes the first of
        take-off, stick and slide, so a state built on another branch is not kept by a unit sweep.
        """
        W = np.array([[1.0, 1.5, 0.0], [1.5, 3.0, 0.0], [0.0, 0.0, 3.0]])
        b = np.array([0.1, 0.5, 0.0])
        mu = 1.0
        take_off = np.zeros(3)
        stick = np.array([0.6, -0.35 / 0.75, 0.0])
        slide = np.array([0.2, -0.2, 0.0])
        for x in (take_off, stick, slide):
            self.assertLess(point_residual(W, b, mu, x), 1.0e-12)
        out = solve_points(point_problem(W, b, mu))[0]
        np.testing.assert_array_equal(out[:3], take_off)
        self.assertEqual(out[3], -1.0)

    def test_capped_root_resumes_from_its_warm_start(self):
        """Resume a slide root that the 8-evaluation cap stopped short from its last iterate on the next visit.

        A strongly lifting sliding contact (kappa 1.3) needs 11 Newton evaluations from the isotropic estimate. The
        cap returns a feasible but unconverged force; the unit's next visit in the same solve starts from that root
        and converges.
        """
        W = np.array([[0.48623, -0.6174, -0.33833], [-0.6174, 6.20362, 0.46554], [-0.33833, 0.46554, 0.92675]])
        b = np.array([-0.12804, -1.34818, 0.27253])
        mu = 0.9
        first = solve_points(point_problem(W, b, mu))[0]
        self.assertEqual(first[4], 1 + contact_unit.CONTACT_UNIT_NEWTON_CAP)
        self.assertGreater(point_residual(W, b, mu, first[:3]), 0.05)
        self.assertLessEqual(np.linalg.norm(first[1:3]), mu * first[0] * (1.0 + 1.0e-6))
        second = solve_points(point_problem(W, b, mu, alpha=first[3]))[0]
        self.assertLess(second[4], 1 + contact_unit.CONTACT_UNIT_NEWTON_CAP)
        self.assertLess(point_residual(W, b, mu, second[:3]), 1.0e-6)
        np.testing.assert_allclose(second[:3], exact_contact(W, b, mu), rtol=1.0e-5)


def exact_contact(W, b, mu):
    """The exact solution of one isolated point contact in float64 (take-off, stick, or the slide root by bisection)."""
    if b[0] >= 0.0:
        return np.zeros(3)
    w = W[0, 1:]
    A = W[1:, 1:] - np.outer(w, w) / W[0, 0]
    c = b[1:] - b[0] / W[0, 0] * w
    g = -w / W[0, 0]
    S0 = -b[0] / W[0, 0]

    def solve(alpha):
        s = np.linalg.solve(A + alpha * np.eye(2), c)
        return s, np.linalg.norm(s) - mu * (S0 - g @ s)

    s, gap = solve(0.0)
    if gap > 0.0:
        lo, hi = 0.0, 1.0
        while solve(hi)[1] > 0.0:
            hi *= 2.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if solve(mid)[1] > 0.0 else (lo, mid)
        s = solve(hi)[0]
    return np.array([(w @ s - b[0]) / W[0, 0], *(-s)])


class TestContactUnitLatch(unittest.TestCase):
    """The alternation latch of a unit."""

    def test_latch_keeps_fixed_points(self):
        """Move no impulse for any latch state when the exact solution equals the impulses (x* = lambda)."""
        rng = np.random.default_rng(5)
        states = np.zeros((2000, 4), np.float32)
        states[:, 0] = np.where(rng.uniform(size=2000) < 0.5, 0.0, rng.uniform(0.1, 1.0, size=2000))
        states[:, 1] = rng.normal(size=2000)
        states[:, 2] = np.abs(rng.normal(size=2000))
        states[:, 3] = states[:, 2]
        out = wp.zeros((2000, 3), dtype=float, device=_CPU)
        wp.launch(_latch, dim=2000, inputs=[wp.array(states, dtype=float, device=_CPU), out], device=_CPU)
        result = out.numpy()
        np.testing.assert_array_equal(result[:, 2], 0.0)
        # The factor stays where it was and the weight stays positive, so x = lambda + w (x* - lambda) = lambda.
        np.testing.assert_array_equal(result[:, 1], states[:, 0])
        self.assertTrue(np.all(result[:, 0] >= 0.1))

    def test_latch_compounds_on_reversals(self):
        """Relax a unit by 1/(1+kappa) per reversal of at least 80% of its last change, down to 0.1."""
        cases = np.array(
            [
                [0.0, 1.0, 1.0, 0.1],  # reverses 0.9 of a change of 1: kappa 0.9
                [0.0, 1.0, 1.0, 0.5],  # reverses 0.5: below 80%, no latch
                [0.0, 1.0, 1.0, 2.0],  # same direction: no latch
                [0.5, -1.0, 1.0, 3.0],  # latched at 0.5, reverses 1.0 (= 0.5 * 2): kappa 1
                [0.12, 1.0, 10.0, 0.0],  # compounds down to the floor 0.1
                [0.0, 1.0, 100.0, 99.0],  # reversal below 5% of the load: no latch
            ],
            np.float32,
        )
        out = wp.zeros((len(cases), 3), dtype=float, device=_CPU)
        wp.launch(_latch, dim=len(cases), inputs=[wp.array(cases, dtype=float, device=_CPU), out], device=_CPU)
        weight, factor, change = out.numpy().T
        np.testing.assert_allclose(weight, [1.0 / 1.9, 1.0, 1.0, 0.25, 0.1, 1.0], rtol=1.0e-6)
        np.testing.assert_allclose(factor, [1.0 / 1.9, 0.0, 0.0, 0.25, 0.1, 0.0], rtol=1.0e-6)
        np.testing.assert_allclose(change, [-0.9 / 1.9, -0.5, 1.0, 0.5, -1.0, -1.0], rtol=1.0e-6)


def ring_block(W, b, old, alpha=(0.0, 0.0)):
    """Lay a ring unit out in the routine's block: block, residual, impulses, solution, slide roots."""
    nu = len(b)
    block = np.zeros(contact_unit.RING_BLOCK_SIZE, np.float32)
    block[contact_unit.RING_BLOCK : contact_unit.RING_BLOCK + nu * nu] = np.asarray(W, np.float32).reshape(-1)
    block[contact_unit.RING_RESIDUAL : contact_unit.RING_RESIDUAL + nu] = b
    block[contact_unit.RING_IMPULSE : contact_unit.RING_IMPULSE + nu] = old
    block[contact_unit.RING_ALPHA : contact_unit.RING_ALPHA + 2] = alpha
    return block


def solve_ring(W, b, old, normals, anchors, mu0, mu1=0.0, visits=3):
    """Visit an isolated ring unit ``visits`` times, each from the previous solution, as consecutive sweeps do.

    One visit solves the proximally regularized block, which leaves ``-eps W_ii (x_i - old_i)`` on a normal's
    velocity; revisits converge to the unit's exact solution.
    """
    nu = normals + 2 * anchors
    start = contact_unit.RING_SOLUTION
    x = np.asarray(old, np.float64)
    alpha = (0.0, 0.0)
    for _ in range(visits):
        block = wp.array(ring_block(W, b, x, alpha), dtype=float, device=_CPU)
        evaluations = wp.zeros(1, dtype=int, device=_CPU)
        wp.launch(_solve_ring, dim=1, inputs=[block, normals, anchors, mu0, mu1, evaluations], device=_CPU)
        result = block.numpy()
        x = result[start : start + nu].astype(np.float64)
        alpha = tuple(result[contact_unit.RING_ALPHA : contact_unit.RING_ALPHA + 2])
    return x


def box_face(corners, anchors, mass=2.0, inertia=(0.02, 0.03, 0.04), friction_rows=True):
    """Delassus block of a box face on fixed ground: normals at ``corners``, tangent pairs at the anchors' corners."""
    inverse = np.diag([1.0 / mass] * 3 + [1.0 / value for value in inertia])
    rows = []
    for x, y in corners:
        r = np.array([x, y, 0.0])
        rows.append(np.concatenate([[0.0, 0.0, 1.0], np.cross(r, [0.0, 0.0, 1.0])]))
    if friction_rows:
        for a in anchors:
            r = np.array([*corners[a], 0.0])
            for t in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]):
                rows.append(np.concatenate([t, np.cross(r, t)]))
    J = np.array(rows)
    return J @ inverse @ J.T


def ring_check(test, W, b, x, normals, anchors, mus, tolerance=1.0e-5):
    """Assert a ring unit's solution: complementary normals that do not approach, friction inside the final disk."""
    v = b + W @ x
    load = x[:normals].sum()
    scale = max(abs(b).max(), 1.0)
    test.assertTrue(np.all(x[:normals] >= 0.0))
    test.assertTrue(np.all(v[:normals] >= -tolerance * scale), v[:normals])
    test.assertLess(np.max(np.abs(x[:normals] * v[:normals])), tolerance * scale * max(load, 1.0))
    for j in range(anchors):
        f = x[normals + 2 * j : normals + 2 * j + 2]
        test.assertLessEqual(np.linalg.norm(f), mus[j] * load + 1.0e-6)
    return v, load


class TestContactUnitRing(unittest.TestCase):
    """A patch region's ring unit: its normals by an active set, its anchors' friction on the pooled load."""

    def test_coplanar_face_supports_its_weight(self):
        """Hold a box face on four coplanar corners, whose normal block has rank 3 of 4, without approach or slip."""
        corners = [(-0.1, -0.05), (0.1, -0.05), (0.1, 0.05), (-0.1, 0.05)]
        W = box_face(corners, anchors=(0, 2))
        self.assertEqual(np.linalg.matrix_rank(W[:4, :4], tol=1.0e-9), 3)
        W += 1.0e-6 * np.eye(len(W))
        dt, g, mass = 0.005, 9.81, 2.0
        b = np.zeros(len(W))
        b[:4] = -g * dt  # every corner approaches at the weight's velocity change
        for old in (np.zeros(len(W)), np.r_[np.full(4, 0.02), np.zeros(4)]):
            with self.subTest(warm=bool(old.any())):
                x = solve_ring(W, b, old, 4, 2, 0.5, 0.5)
                self.assertTrue(np.all(np.isfinite(x)))
                v, load = ring_check(self, W, b, x, 4, 2, (0.5, 0.5))
                self.assertAlmostEqual(load, mass * g * dt, places=4)
                np.testing.assert_allclose(x[4:], 0.0, atol=1.0e-7)
                np.testing.assert_allclose(v[4:], 0.0, atol=1.0e-5)

    def test_coplanar_face_slides_on_its_pooled_load(self):
        """Slide a pushed box face: each anchor's friction on the disk of its share of the pooled load."""
        corners = [(-0.1, -0.05), (0.1, -0.05), (0.1, 0.05), (-0.1, 0.05)]
        W = box_face(corners, anchors=(0, 2)) + 1.0e-6 * np.eye(8)
        b = np.zeros(8)
        b[:4] = -0.05
        b[4:] = [1.0, 0.0, 1.0, 0.0]  # sliding along +x at both anchors
        x = solve_ring(W, b, np.zeros(8), 4, 2, 0.25, 0.25)
        _, load = ring_check(self, W, b, x, 4, 2, (0.25, 0.25))
        for j in range(2):
            f = x[4 + 2 * j : 6 + 2 * j]
            self.assertAlmostEqual(np.linalg.norm(f), 0.25 * load, delta=1.0e-4 * load)
            self.assertLess(f[0], 0.0)

    def test_duplicate_normals_share_one_solution(self):
        """Solve K identical normal rows as one: their total is the single row's solution and none approaches."""
        u = np.array([1.0, 0.3, -0.2])
        for copies in (2, 3):
            with self.subTest(copies=copies):
                J = np.vstack([np.tile(u, (copies, 1)), [[0.2, 1.0, 0.0], [0.1, 0.0, 1.0]]])
                W = J @ J.T + 1.0e-6 * np.eye(copies + 2)
                b = np.r_[np.full(copies, -0.5), 0.02, -0.01]
                x = solve_ring(W, b, np.zeros(copies + 2), copies, 1, 0.8)
                _, load = ring_check(self, W, b, x, copies, 1, (0.8,))
                single = solve_points(point_problem(W[copies - 1 :, copies - 1 :], b[copies - 1 :], 0.8))[0]
                self.assertAlmostEqual(load, single[0], delta=1.0e-4 * single[0])
                np.testing.assert_allclose(x[copies:], single[1:3], rtol=1.0e-3, atol=1.0e-7)

    def test_frictionless_ring_stops_every_normal(self):
        """Solve a ring without anchors as its normals' complementarity problem."""
        corners = [(-0.1, -0.05), (0.1, -0.05), (0.0, 0.05)]
        W = box_face(corners, anchors=(), friction_rows=False) + 1.0e-6 * np.eye(3)
        b = np.array([-0.2, 0.1, -0.05])
        x = solve_ring(W, b, np.zeros(3), 3, 0, 0.0)
        ring_check(self, W, b, x, 3, 0, ())
        self.assertGreater(x.sum(), 0.0)

    def test_separated_ring_stays_at_zero(self):
        """Leave a ring whose normals all separate at zero impulse without impulse."""
        W = box_face([(-0.1, 0.0), (0.1, 0.0)], anchors=(0,)) + 1.0e-6 * np.eye(4)
        x = solve_ring(W, np.array([0.1, 0.2, 1.0, -1.0]), np.zeros(4), 2, 1, 0.5, visits=1)
        np.testing.assert_array_equal(x, 0.0)

    def test_one_visit_keeps_the_proximal_residual_only(self):
        """Leave a normal's velocity at -eps W_ii (x_i - old_i) after one visit: the proximal term, no other error."""
        corners = [(-0.1, -0.05), (0.1, -0.05), (0.0, 0.05)]
        W = box_face(corners, anchors=(), friction_rows=False) + 1.0e-6 * np.eye(3)
        b = np.array([-0.2, 0.1, -0.05])
        old = np.array([0.01, 0.0, 0.02])
        x = solve_ring(W, b, old, 3, 0, 0.0, visits=1)
        v = b + W @ x
        active = x > 0.0
        expected = -contact_unit.CONTACT_UNIT_PROXIMAL * np.diag(W) * (x - old)
        np.testing.assert_allclose(v[active], expected[active], rtol=1.0e-2, atol=1.0e-7)


@unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
class TestContactUnitCuda(unittest.TestCase):
    def test_cuda_prelude_matches_the_cpu_binding(self):
        """Solve point problems with the CUDA prelude as the CPU binding does, to float32 rounding."""
        rng = np.random.default_rng(17)
        rows = []
        for _ in range(4000):
            W = random_block(rng, coupling=rng.uniform(0.0, 2.0))
            b = np.array([rng.normal(), *rng.normal(size=2) * rng.uniform(0.1, 5.0)])
            rows.append(point_problem(W, b, rng.uniform(0.1, 1.2), ring_load=rng.choice([0.0, rng.uniform(0, 1)])))
        rows = np.asarray(rows, np.float32)
        cpu = solve_points(rows)
        device = "cuda:0"
        out = wp.zeros((len(rows), 5), dtype=float, device=device)
        wp.launch(
            _solve_points_native, dim=len(rows), inputs=[wp.array(rows, dtype=float, device=device), out], device=device
        )
        cuda = out.numpy().astype(np.float64)
        scale = np.maximum(np.abs(cpu[:, :3]).max(axis=1, keepdims=True), 1.0e-6)
        self.assertLess(np.quantile(np.abs(cuda[:, :3] - cpu[:, :3]) / scale, 0.99), 1.0e-5)
        self.assertLess(np.max(np.abs(cuda[:, :3] - cpu[:, :3]) / scale), 1.0e-3)
        cuda_twin = solve_points(rows, device=device)
        self.assertLess(np.max(np.abs(cuda_twin[:, :3] - cpu[:, :3]) / scale), 1.0e-3)


if __name__ == "__main__":
    unittest.main()
