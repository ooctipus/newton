# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Solve contact units exactly in the loop sweeps (``contact_update="unit"``) and keep the sequential law.

The sequential sweep steps a contact's normal row against the friction of its
previous update, then solves the friction pair at the held load. A sliding
contact whose friction lifts its normal then alternates with period two, and a
solve can end with no normal impulse on an approaching contact. The unit sweeps
solve each contact unit exactly against its own block (take-off, else stick,
else the slide root) and relax a unit whose load keeps reversing.

The cases run the sweep kernels the solver launches for each law on
synthesized blocks (the audit block of the cycle, an oversize patch ring, two
contacts of one free body whose loads alternate) and on stored legged-robot
replica systems of the CPU contact bench
(``golden_data/test_feather_pgs_contact_update/replica_systems.npz``: certified
references, upper triangles of the Delassus matrices, chosen for zero-normal
events, a 128-sweep cycle census, friction outside its final disk and the
cross-contact case of replica 79). Residuals follow the bench: a normal row's
``|min(d lambda, w)|`` and a friction pair's distance to the projection on the
disk of its ring's load, in velocity units.
"""

import os
import unittest
import warnings
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import contact_unit, environment_flags, path_selection, sweep_routes_split
from newton._src.solvers.feather_pgs import kernels as K
from newton._src.solvers.feather_pgs.kernels import BodyFactors
from newton.solvers import SolverFeatherPGS
from newton.tests.feather_pgs_row_builders import launched_kernels
from newton.tests.test_feather_pgs_contact_unit import slide_solutions

wp.init()

CONTACT = int(K.PGS_CONSTRAINT_TYPE_CONTACT)
FRICTION = int(K.PGS_CONSTRAINT_TYPE_FRICTION)
LIMIT = int(K.PGS_CONSTRAINT_TYPE_JOINT_LIMIT)
VLIMIT = int(K.PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT)
_CPU = "cpu"
_FIXTURE = os.path.join(
    os.path.dirname(__file__), "golden_data", "test_feather_pgs_contact_update", "replica_systems.npz"
)
_TAGS = ("zero_end", "census", "cone", "replica79")

# The audit block of the sliding cycle: a sphere foot's 3x3 Delassus block, its residual at zero impulse
# (approaching at 0.141 m/s, sliding along -t2) and mu 1, with lift coupling mu |W_nt| / W_nn = 1.0.
W_AUDIT = np.array([[0.1729, 0.0492, 0.1662], [0.0492, 0.3586, 0.0685], [0.1662, 0.0685, 0.2116]])
W0_AUDIT = np.array([-0.14089, -0.2, -1.0])

# Two sliding point contacts of one free body (rows over its 6 coordinates, identity mass) whose loads reverse
# between visits, so the alternation latch relaxes them within 8 sweeps: normal, tangent, tangent per contact.
J_ALTERNATING = np.array(
    [
        [-0.25, -1.43, -0.19, -0.02, 1.69, 0.62],
        [-1.53, 2.03, -0.4, -0.88, 1.47, -0.05],
        [-0.37, 0.22, 0.84, 0.99, -1.38, 2.0],
        [0.95, -0.38, -0.82, -0.97, 0.12, -0.65],
        [-0.76, 0.81, 0.36, -0.39, 0.73, 1.37],
        [-1.09, -0.6, 0.94, 0.72, 0.23, 1.16],
    ]
)
W0_ALTERNATING = np.array([-0.38, -0.44, -0.26, -0.09, -0.24, -0.15])

# A point contact whose friction presses its normal down: it separates at zero impulse (S0 = -0.02) and has take-off
# and two press slides, whose loads answer its approach with dS/dS0 = +35 (the larger load) and -29.
W_PRESSED = np.array([[1.0, -0.7, -0.9], [-0.7, 0.69, 0.63], [-0.9, 0.63, 2.11]])
B_PRESSED = np.array([0.02, -0.4, -0.2])
MU_PRESSED = 1.2

# The certified references that 256 sweeps started from them leave above an NCP residual of 1e-3 m/s, by update
# and friction law (test_sweeps_from_a_reference_leave_only_the_recorded_ones).
_LEFT_REFERENCES = {
    "sequential": {"point": set(), "patch": {2549}},
    "unit": {"point": set(), "patch": set()},
}


@contextmanager
def _without_solver_environment():
    """Remove every FEATHER_PGS_ and IL_NEWTON_FPGS_ variable for the block."""
    with mock.patch.dict(os.environ):
        for name in [name for name in os.environ if name.startswith(environment_flags.PREFIXES)]:
            del os.environ[name]
        yield


# -- systems ----------------------------------------------------------------------------------------------------


def contact_system(normals, rhs, *, anchors, mu=1.0, dofs=3, ring=True):
    """A dense contact system from Jacobian rows.

    Args:
        normals: Normal rows, one Jacobian row each, in ring order; a point contact is a ring of one.
        rhs: Each normal's residual at zero impulse; anchors add their two tangent residuals.
        anchors: ``{ring member: (tangent row 0, tangent row 1, rhs0, rhs1)}``; their pairs follow their normals.
        mu: Friction coefficient of the ring, divided among its anchors.
        dofs: Velocity coordinates of the rows (identity mass).
        ring: Link the normals as one patch region; a point contact's normal has no ring (parent -1).
    """
    J, b, rtype, parent, mus, normal_rows = [], [], [], [], [], []
    for j, row in enumerate(normals):
        normal_rows.append(len(J))
        J.append(row)
        b.append(rhs[j])
        rtype.append(CONTACT)
        parent.append(-1)
        mus.append(0.0)
        if j in anchors:
            t0, t1, r0, r1 = anchors[j]
            for t, r in ((t0, r0), (t1, r1)):
                J.append(t)
                b.append(r)
                rtype.append(FRICTION)
                parent.append(normal_rows[-1])
                mus.append(mu / len(anchors))
    if ring:
        # A patch region's normals link into a circular list; a one-member region points to itself.
        for j, row in enumerate(normal_rows):
            parent[row] = normal_rows[(j + 1) % len(normal_rows)]
    J = np.asarray(J, np.float64).reshape(len(J), dofs)
    C = J @ J.T
    return {
        "J": J,
        "C": C,
        "diag": np.diag(C).copy(),
        "rhs": np.asarray(b, np.float64),
        "rtype": np.asarray(rtype, np.int32),
        "parent": np.asarray(parent, np.int32),
        "mu": np.asarray(mus, np.float64),
        "group": np.where(np.asarray(rtype) == CONTACT, 0, -1),
    }


def point_unit_system(W, b, mu):
    """One point contact from its 3x3 block (normal, tangent, tangent), its residual at zero impulse and coefficient."""
    W = np.asarray(W, np.float64)
    return {
        "C": W,
        "diag": np.diag(W).copy(),
        "rhs": np.asarray(b, np.float64),
        "rtype": np.array([CONTACT, FRICTION, FRICTION], np.int32),
        "parent": np.array([-1, 0, 0], np.int32),
        "mu": np.array([0.0, mu, mu]),
        "group": np.array([0, -1, -1]),
    }


def audit_system(copies, law):
    """The audit block as ``copies`` identical contacts: point contacts, or one patch ring with 1 or 2 anchors."""
    L = np.linalg.cholesky(W_AUDIT)
    normals = [L[0]] * copies
    if law == "point":
        systems = [
            contact_system([L[0]], [W0_AUDIT[0]], anchors={0: (L[1], L[2], W0_AUDIT[1], W0_AUDIT[2])}, ring=False)
            for _ in range(copies)
        ]
        return concatenate(systems)
    count = 1 if law == "patch1" else 2
    anchors = dict.fromkeys(range(copies - count, copies), (L[1], L[2], W0_AUDIT[1], W0_AUDIT[2]))
    return contact_system(normals, [W0_AUDIT[0]] * copies, anchors=anchors)


def sticking_system():
    """The audit block's normal with a small tangential residual: it sticks, down to a row weight of 0.5."""
    L = np.linalg.cholesky(W_AUDIT)
    return contact_system([L[0]], [W0_AUDIT[0]], anchors={0: (L[1], L[2], -0.01, 0.02)}, ring=False)


def alternating_system():
    """Two sliding point contacts (mu 0.65) of one free body whose loads reverse between visits."""
    J, b = J_ALTERNATING, W0_ALTERNATING
    contacts = [
        contact_system(
            [J[n]], [b[n]], anchors={0: (J[n + 1], J[n + 2], b[n + 1], b[n + 2])}, mu=0.65, dofs=6, ring=False
        )
        for n in (0, 3)
    ]
    return concatenate(contacts)


def concatenate(systems):
    """Systems on one shared body as one system: rows stacked, Delassus from the stacked Jacobian."""
    offsets = np.cumsum([0] + [len(s["rhs"]) for s in systems])
    J = np.vstack([s["J"] for s in systems])
    C = J @ J.T
    parent = np.concatenate(
        [np.where(s["parent"] >= 0, s["parent"] + o, -1) for s, o in zip(systems, offsets, strict=False)]
    )
    return {
        "J": J,
        "C": C,
        "diag": np.diag(C).copy(),
        "rhs": np.concatenate([s["rhs"] for s in systems]),
        "rtype": np.concatenate([s["rtype"] for s in systems]),
        "parent": parent.astype(np.int32),
        "mu": np.concatenate([s["mu"] for s in systems]),
        "group": np.concatenate([s["group"] for s in systems]),
    }


def replica_systems(law):
    """The stored replica systems of one friction law, each a dict with its tags, source index and reference."""
    data = np.load(_FIXTURE)
    systems = []
    row = upper = 0
    for k, rows_of_system in enumerate(data[f"{law}_m"]):
        m = int(rows_of_system)
        C = np.zeros((m, m), np.float64)
        iu = np.triu_indices(m)
        C[iu] = data[f"{law}_C_upper"][upper : upper + len(iu[0])]
        C = C + np.triu(C, 1).T
        upper += len(iu[0])
        rows = slice(row, row + m)
        row += m
        tags = int(data[f"{law}_tags"][k])
        systems.append(
            {
                "source": int(data[f"{law}_source"][k]),
                "tags": {tag for bit, tag in enumerate(_TAGS) if tags >> bit & 1},
                "C": C,
                "diag": data[f"{law}_diag"][rows].astype(np.float64),
                "rhs": data[f"{law}_rhs"][rows].astype(np.float64),
                "rtype": data[f"{law}_rtype"][rows].astype(np.int32),
                "parent": data[f"{law}_parent"][rows].astype(np.int32),
                "mu": data[f"{law}_mu"][rows].astype(np.float64),
                "group": data[f"{law}_group"][rows].astype(np.int64),
                "lam_ref": data[f"{law}_lam_ref"][rows].astype(np.float64),
            }
        )
    return systems


def reverse_contacts(system):
    """The system with its contact blocks (a normal and its pair) in reverse order, other rows first."""
    rtype, parent = system["rtype"], system["parent"]
    m = len(rtype)
    fixed, blocks, i = [], [], 0
    while i < m:
        if rtype[i] == CONTACT:
            block = [i]
            j = i + 1
            while j < m and rtype[j] == FRICTION and parent[j] == i:
                block.append(j)
                j += 1
            blocks.append(block)
            i = j
        else:
            fixed.append(i)
            i += 1
    order = np.array(fixed + [r for block in blocks[::-1] for r in block], np.int64)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(m)
    out = dict(system)
    out["C"] = system["C"][np.ix_(order, order)]
    for key in ("diag", "rhs", "rtype", "mu", "group", "lam_ref"):
        if key in system:
            out[key] = np.asarray(system[key])[order]
    p = parent[order]
    out["parent"] = np.where(p >= 0, inverse[np.maximum(p, 0)], -1).astype(np.int32)
    return out


# -- sweeps -----------------------------------------------------------------------------------------------------


class World:
    """Systems in the solver's (world, row) layout on one device, swept by the kernel each law launches."""

    def __init__(self, systems, device=_CPU):
        self.systems = systems
        self.device = device
        self.D = D = max(len(s["rhs"]) for s in systems)
        W = len(systems)
        C = np.zeros((W, D, D), np.float32)
        arrays = {k: np.zeros((W, D), np.float32) for k in ("diag", "rhs", "mu")}
        rtype = np.zeros((W, D), np.int32)
        parent = -np.ones((W, D), np.int32)
        self.count = np.array([len(s["rhs"]) for s in systems], np.int32)
        for k, s in enumerate(systems):
            m = len(s["rhs"])
            C[k, :m, :m] = s["C"]
            for key, values in arrays.items():
                values[k, :m] = s[key]
            rtype[k, :m] = s["rtype"]
            parent[k, :m] = s["parent"]
        self.w_count = wp.array(self.count, dtype=int, device=device)
        self.w_C = wp.array(C, dtype=float, device=device)
        self.w = {k: wp.array(v, dtype=float, device=device) for k, v in arrays.items()}
        self.w_rtype = wp.array(rtype, dtype=int, device=device)
        self.w_parent = wp.array(parent, dtype=int, device=device)
        self.state = wp.zeros((W, D, contact_unit.CONTACT_UNIT_STATE_FIELDS), dtype=float, device=device)
        self.rows = wp.zeros((W, D), dtype=int, device=device)
        self.block = wp.zeros(W * contact_unit.RING_BLOCK_SIZE, dtype=float, device=device)

    def sweep(self, iterations, update, *, start=None, snapshots=False, ring_units=1, restart=False):
        """Sweep every system ``iterations`` times from ``start`` (cold by default) under one contact update.

        Returns the impulses per system, or with ``snapshots`` the impulses after every sweep (one launch per
        sweep that continues the solve, as the interleaved split driver launches). With ``restart`` each of those
        launches starts a new solve (``iteration_offset`` 0), which clears the unit sweep's latches and slide roots.
        """
        lam = np.zeros((len(self.systems), self.D), np.float32)
        if start is not None:
            for k, s in enumerate(start):
                lam[k, : len(s)] = s
        impulses = wp.array(lam, dtype=float, device=self.device)
        launches = [(iterations, 0)]
        if snapshots:
            launches = [(1, 0 if restart else it) for it in range(iterations)]
        history = []
        for count, offset in launches:
            common = [
                self.w_count,
                self.w["diag"],
                self.w_C,
                self.w["rhs"],
                impulses,
                count,
                1.0,
                self.w_rtype,
                self.w_parent,
                self.w["mu"],
                self.w["mu"],  # unread weights: no regularization
                0,
                0,
                offset,
            ]
            if update == "unit":
                wp.launch(
                    sweep_routes_split.loop_unit_kernel(),
                    dim=len(self.systems),
                    inputs=[*common, ring_units, self.state, self.rows, self.block],
                    device=self.device,
                )
            else:
                wp.launch(sweep_routes_split.loop_kernel(), dim=len(self.systems), inputs=common, device=self.device)
            if snapshots:
                history.append(impulses.numpy().astype(np.float64))
        if snapshots:
            stacked = np.stack(history)
            return [stacked[:, k, : self.count[k]] for k in range(len(self.systems))]
        out = impulses.numpy().astype(np.float64)
        return [out[k, : self.count[k]] for k in range(len(self.systems))]


class FreeBody:
    """A system as the free-body rows of one body (identity mass, rows over its first coordinates).

    Swept by the free-body kernel each law launches; the unit sweep's storage persists across sweeps, as the
    solver keeps it.
    """

    def __init__(self, system):
        self.system = system
        J = system["J"]
        self.m = m = J.shape[0]
        J6 = np.zeros((1, m, 6), np.float32)
        J6[0, :, : J.shape[1]] = J
        # The body's velocity gives each row its residual at zero impulse (the rows carry no bias).
        self.velocity = np.zeros(6, np.float32)
        self.velocity[: J.shape[1]] = np.linalg.lstsq(J, system["rhs"], rcond=None)[0]
        # Identity mass: the body's factor is the identity, so the whitened rows are the rows.
        factors = BodyFactors()
        factors.dof_factor = wp.array([0, -1, -1, -1, -1, -1], dtype=int, device=_CPU)
        factors.L = wp.array(np.eye(6, dtype=np.float32)[None], dtype=float, device=_CPU)
        self.arrays = [
            wp.array([m], dtype=int, device=_CPU),
            wp.zeros((1, m), dtype=int, device=_CPU),
            wp.full((1, m), -1, dtype=int, device=_CPU),
            wp.array(J6, dtype=float, device=_CPU),
            wp.zeros((1, m, 6), dtype=float, device=_CPU),
            factors,
            wp.array((1.0 / np.einsum("ij,ij->i", J, J))[None].astype(np.float32), dtype=float, device=_CPU),
            wp.zeros((1, m), dtype=float, device=_CPU),
            wp.array(system["rtype"][None], dtype=int, device=_CPU),
            wp.array(system["parent"][None], dtype=int, device=_CPU),
            wp.array(system["mu"][None].astype(np.float32), dtype=float, device=_CPU),
        ]
        self.state = wp.zeros((1, m, contact_unit.CONTACT_UNIT_STATE_FIELDS), dtype=float, device=_CPU)
        self.rows = wp.zeros((1, m), dtype=int, device=_CPU)
        self.block = wp.zeros(contact_unit.RING_BLOCK_SIZE, dtype=float, device=_CPU)

    def sweep(self, iterations, update, *, ring_units=1, row_weight=1.0, offsets=None):
        """Sweep from zero impulses ``iterations`` times under one contact update and return the impulses.

        Args:
            iterations: Sweeps of the solve.
            update: ``"sequential"`` or ``"unit"``.
            ring_units: 1 solves a patch ring as one ring unit, 0 as anchor units.
            row_weight: Contact rows' regularization weight ``1 / (1 + g)``; below 1 the sweeps regularize them.
            offsets: The ``iteration_offset`` of each launch of one sweep; ``None`` launches every sweep at once.
        """
        weights = np.where(self.system["rtype"] == CONTACT, row_weight, 1.0)[None].astype(np.float32)
        arrays = [
            *self.arrays,
            wp.array(weights, dtype=float, device=_CPU),
            wp.array([0], dtype=int, device=_CPU),
            wp.array([0], dtype=int, device=_CPU),
        ]
        impulses = wp.zeros((1, self.m), dtype=float, device=_CPU)
        velocity = wp.array(self.velocity, dtype=float, device=_CPU)
        regularize = int(row_weight < 1.0)
        launches = [(iterations, 0)] if offsets is None else [(1, int(offset)) for offset in offsets]
        for count, offset in launches:
            if update == "unit":
                inputs = [*arrays, count, 1.0, regularize, 0, offset, ring_units, self.state, self.rows, self.block]
                kernel = sweep_routes_split.mf_loop_unit_kernel()
            else:
                inputs = [*arrays, count, 1.0, regularize, 0, offset]
                kernel = sweep_routes_split.mf_loop_kernel()
            wp.launch(kernel, dim=1, inputs=inputs, outputs=[impulses, velocity], device=_CPU)
        return impulses.numpy()[0].astype(np.float64)


def sweep_free_body(system, iterations, update, **options):
    """Sweep a system as the free-body rows of one body with fresh unit storage (see :meth:`FreeBody.sweep`)."""
    return FreeBody(system).sweep(iterations, update, **options)


# -- metrics ----------------------------------------------------------------------------------------------------


def ring_of(system, n):
    members, row = [n], system["parent"][n]
    while row >= 0 and row != n and len(members) < len(system["rhs"]):
        members.append(int(row))
        row = system["parent"][row]
    return members


def pairs_of(system):
    rtype, parent = system["rtype"], system["parent"]
    m = len(rtype)
    return [i for i in range(m) if rtype[i] == FRICTION and i == parent[i] + 1 and i + 1 < m]


def residuals(system, lam):
    """Per-row natural residual [m/s] (a friction pair's on its first row) and the row velocities ``w``."""
    C, rhs, d, rtype = system["C"], system["rhs"], system["diag"], system["rtype"]
    w = C @ lam + rhs
    r = np.zeros(len(rhs))
    unilateral = (rtype == CONTACT) | (rtype == LIMIT) | (rtype == VLIMIT)
    r[unilateral] = np.abs(np.minimum(d[unilateral] * lam[unilateral], w[unilateral]))
    for p in pairs_of(system):
        load = lam[ring_of(system, int(system["parent"][p]))].sum()
        radius = system["mu"][p] * load
        dp = 0.5 * (d[p] + d[p + 1])
        f = lam[p : p + 2]
        z = f - w[p : p + 2] / dp
        norm = np.linalg.norm(z)
        projection = z if norm <= radius else z * (max(radius, 0.0) / norm)
        r[p] = dp * np.linalg.norm(f - projection)
    return r, w


def group_loads(system, lam):
    normals = np.flatnonzero(system["rtype"] == CONTACT)
    return np.bincount(system["group"][normals], weights=lam[normals], minlength=int(system["group"].max()) + 1)


def zero_normal_groups(system, lam):
    """Loaded groups that end below 5% of their reference load while approaching faster than 0.02 m/s."""
    _, w = residuals(system, lam)
    normals = np.flatnonzero(system["rtype"] == CONTACT)
    reference = group_loads(system, system["lam_ref"])
    loads = group_loads(system, lam)
    approach = np.zeros(len(reference))
    np.maximum.at(approach, system["group"][normals], np.maximum(-w[normals], 0.0))
    loaded = reference > 1.0e-3 * reference.max()
    return np.flatnonzero(loaded & (loads < 0.05 * reference) & (approach > 0.02))


def classify(system, history):
    """The bench census's class of a 128-sweep history: converged, a fixed wrong state, a cycle or other."""
    residual = residuals(system, history[-1])[0].max()
    scale = max(np.abs(history[-1]).max(), 1.0e-12)
    change = np.abs(history[-1] - history[-2]).max() / scale
    period = 0
    for p in range(2, 9):
        if np.abs(history[-1] - history[-1 - p]).max() / scale <= 1.0e-5 and change > 1.0e-6:
            period = p
            break
    if period:
        residual = max(residuals(system, history[-1 - j])[0].max() for j in range(period))
    if residual <= 1.0e-3:
        return "converged"
    if change <= 1.0e-6 and residual > 1.0e-2:
        return "fixed"
    if period and residual > 1.0e-2:
        return "cycle"
    return "other"


def exact_point_contact(W, b, mu):
    """The exact solution of one isolated point contact (float64): take-off, stick, or Daviet's slide root."""
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


# -- tests ------------------------------------------------------------------------------------------------------


class TestContactUpdateAuditBlock(unittest.TestCase):
    """The audit block: one sliding contact whose friction lifts its normal (kappa 1), as K identical copies."""

    @classmethod
    def setUpClass(cls):
        cls.load = exact_point_contact(W_AUDIT, W0_AUDIT, 1.0)[0]

    def cases(self):
        """Point contacts and one ring with one or two anchors, as 1 to 3 identical copies."""
        return (
            ("point", 1),
            ("point", 2),
            ("point", 3),
            ("patch1", 1),
            ("patch1", 2),
            ("patch1", 3),
            ("patch2", 2),
            ("patch2", 3),
        )

    @staticmethod
    def forms(law):
        """The unit sweep's forms of a law's rings: ring units (1) and, for patch rings, anchor units (0, Stage A)."""
        return (1,) if law == "point" else (1, 0)

    def check(self, system, lam):
        normals = np.flatnonzero(system["rtype"] == CONTACT)
        _, w = residuals(system, lam)
        self.assertAlmostEqual(lam[normals].sum(), self.load, delta=1.0e-4)
        self.assertGreaterEqual(w[normals].min(), -1.0e-5)

    def test_unit_loop_holds_the_audit_block(self):
        """Hold the audit block's exact load with no approach at 8 and 9 sweeps, for 1-3 copies under both laws.

        Patch rings run as ring units and as anchor units (patch Stage A).
        """
        self.assertAlmostEqual(self.load, 0.40878, places=4)
        for law, copies in self.cases():
            system = audit_system(copies, law)
            for ring_units in self.forms(law):
                for sweeps in (8, 9):
                    with self.subTest(law=law, copies=copies, ring_units=ring_units, sweeps=sweeps):
                        self.check(system, World([system]).sweep(sweeps, "unit", ring_units=ring_units)[0])

    def test_unit_free_body_loop_holds_the_audit_block(self):
        """Hold the audit block's exact load in the free-body loop sweep, for 1-3 copies under both laws.

        Patch rings run as ring units and as anchor units (patch Stage A).
        """
        for law, copies in self.cases():
            system = audit_system(copies, law)
            for ring_units in self.forms(law):
                for sweeps in (8, 9):
                    with self.subTest(law=law, copies=copies, ring_units=ring_units, sweeps=sweeps):
                        self.check(system, sweep_free_body(system, sweeps, "unit", ring_units=ring_units))

    def test_unit_free_body_loop_solves_the_regularized_law(self):
        """Solve regularized contact normals in the free-body loop sweep to the sequential w-form update's loads.

        A contact row of weight ``w = 1 / (1 + g)`` comes to rest where ``min(d lambda_n, v_n + (1/w - 1) d
        lambda_n) = 0``, ``d`` its diagonal; the unit solves its normal against ``d / w``, the same law. Without the
        weight it would converge to the rigid loads instead (the audit block's 0.4088 against 0.136-0.401 at these
        weights). The unit sweep is checked after 512 sweeps; the sequential w-form, which needs up to 2048 on
        three copies of the sliding ring at weight 0.9, after 2048.
        """
        systems = [("stick", 1, sticking_system())] + [(law, n, audit_system(n, law)) for law, n in self.cases()]
        for law, copies, system in systems:
            normals = np.flatnonzero(system["rtype"] == CONTACT)
            pairs = pairs_of(system)
            d = system["diag"]
            for weight in (0.9, 0.5, 0.2):
                sequential = sweep_free_body(system, 2048, "sequential", row_weight=weight)
                for ring_units in self.forms(law):
                    with self.subTest(law=law, copies=copies, weight=weight, ring_units=ring_units):
                        lam = sweep_free_body(system, 512, "unit", ring_units=ring_units, row_weight=weight)
                        residual, v = residuals(system, lam)
                        regularized = np.minimum(d * lam, v + (1.0 / weight - 1.0) * d * lam)[normals]
                        self.assertLessEqual(np.abs(regularized).max(), 1.0e-6)
                        self.assertLessEqual(residual[pairs].max(), 1.0e-6)
                        load = sequential[normals].sum()
                        self.assertAlmostEqual(lam[normals].sum(), load, delta=1.0e-4 * load)

    def test_sequential_loop_cycles_on_the_audit_block(self):
        """Keep the sequential law's period-two cycle: 0.043 then 0.772 at 8 and 9 sweeps, approaching at 0.126 m/s."""
        system = audit_system(1, "point")
        world = World([system])
        eight, nine = world.sweep(8, "sequential")[0], world.sweep(9, "sequential")[0]
        self.assertAlmostEqual(eight[0], 0.0434, delta=1.0e-3)
        self.assertAlmostEqual(nine[0], 0.7716, delta=1.0e-3)
        self.assertLess(residuals(system, eight)[1][0], -0.12)
        for law, copies in self.cases():
            system = audit_system(copies, law)
            with self.subTest(law=law, copies=copies), self.assertRaises(AssertionError):
                lams = [World([system]).sweep(sweeps, "sequential")[0] for sweeps in (8, 9)]
                for lam in lams:
                    self.check(system, lam)


class TestContactUpdateOversize(unittest.TestCase):
    """Rings beyond a ring unit's size take anchor units with the ring's other load held, never the sequential law."""

    def ring(self, members, anchor_rows):
        """The audit block's normal in a ring of ``members`` duplicates, anchors at ``anchor_rows`` (sliding)."""
        L = np.linalg.cholesky(W_AUDIT)
        anchors = dict.fromkeys(anchor_rows, (L[1], L[2], W0_AUDIT[1], W0_AUDIT[2]))
        return contact_system([L[0]] * members, [W0_AUDIT[0]] * members, anchors=anchors)

    def check_no_zero_normal(self, system, lam):
        """No normal ends without impulse while it approaches, and the ring holds its exact load."""
        normals = np.flatnonzero(system["rtype"] == CONTACT)
        _, w = residuals(system, lam)
        stopped = lam[normals] <= 0.0
        self.assertFalse(np.any(stopped & (w[normals] < -1.0e-5)), (lam[normals], w[normals]))
        self.assertGreaterEqual(w[normals].min(), -1.0e-5)
        self.assertAlmostEqual(lam[normals].sum(), exact_point_contact(W_AUDIT, W0_AUDIT, 1.0)[0], delta=1.0e-4)

    def test_oversize_ring_keeps_normal_impulse(self):
        """End a sliding 9-member ring, whose sequential sweep ends with a zero-impulse approaching anchor, loaded."""
        system = self.ring(9, (8,))
        self.assertGreater(len(ring_of(system, 0)), contact_unit.CONTACT_UNIT_RING_NORMALS)
        for sweeps in (8, 9):
            with self.subTest(sweeps=sweeps):
                self.check_no_zero_normal(system, World([system]).sweep(sweeps, "unit")[0])
                self.check_no_zero_normal(system, sweep_free_body(system, sweeps, "unit"))
        # The sequential law, the fallback this rule forbids, ends 8 sweeps with the anchor at zero, approaching.
        lam = World([system]).sweep(8, "sequential")[0]
        anchor = ring_of(system, 0)[-1]
        self.assertEqual(lam[anchor], 0.0)
        self.assertLess(residuals(system, lam)[1][anchor], -0.1)
        with self.assertRaises(AssertionError):
            self.check_no_zero_normal(system, lam)

    def test_more_anchors_than_a_ring_unit_take_anchor_units(self):
        """Solve a ring of three anchors, more than a ring unit holds, by anchor units to its exact load."""
        system = self.ring(3, (0, 1, 2))
        for sweeps in (8, 9):
            with self.subTest(sweeps=sweeps):
                self.check_no_zero_normal(system, World([system]).sweep(sweeps, "unit")[0])


class TestContactUpdateReplica(unittest.TestCase):
    """Stored legged-robot replica systems with certified references."""

    @classmethod
    def setUpClass(cls):
        cls.systems = {law: replica_systems(law) for law in ("point", "patch")}

    def tagged(self, tag):
        for law, systems in self.systems.items():
            for system in systems:
                if tag in system["tags"]:
                    yield law, system

    def tight(self, law):
        """The systems of a law whose reference is certified to 1e-5 m/s: all but one of the stored ones."""
        systems = self.systems[law]
        tight = [s for s in systems if residuals(s, s["lam_ref"])[0].max() <= 1.0e-5]
        self.assertGreaterEqual(len(tight), len(systems) - 1)
        return tight

    @staticmethod
    def updates(law):
        """The contact updates and ring forms a law's systems run under: (update, ring_units)."""
        forms = [("sequential", 1), ("unit", 1)]
        return [*forms, ("unit", 0)] if law == "patch" else forms

    def test_zero_normal_events_recover(self):
        """Recover every group the sequential sweep leaves unloaded and approaching at 8 sweeps.

        Patch rings run as ring units and as anchor units (patch Stage A).
        """
        cases = list(self.tagged("zero_end"))
        self.assertEqual(len(cases), 7)
        world = World([system for _, system in cases])
        sequential = world.sweep(8, "sequential")
        for ring_units in (1, 0):
            unit = world.sweep(8, "unit", ring_units=ring_units)
            for (law, system), before, after in zip(cases, sequential, unit, strict=True):
                if law == "point" and ring_units == 0:
                    continue
                with self.subTest(law=law, system=system["source"], ring_units=ring_units):
                    events = zero_normal_groups(system, before)
                    self.assertGreater(len(events), 0)
                    self.assertEqual(len(zero_normal_groups(system, after)), 0)
                    _, w = residuals(system, after)
                    self.assertLessEqual(np.maximum(-w[system["rtype"] == CONTACT], 0.0).max(), 1.0e-3)
                    reference = group_loads(system, system["lam_ref"])[events]
                    np.testing.assert_allclose(group_loads(system, after)[events], reference, rtol=0.02)

    def test_replica_79_converges_in_both_orders(self):
        """Converge the cross-contact case of replica 79 to NCP 1e-3 in 256 sweeps, under both laws and orders."""
        for law, system in self.tagged("replica79"):
            for name, case in (("recorded", system), ("reversed", reverse_contacts(system))):
                for update in ("sequential", "unit"):
                    with self.subTest(law=law, order=name, update=update):
                        lam = World([case]).sweep(256, update)[0]
                        self.assertLessEqual(residuals(case, lam)[0].max(), 1.0e-3)

    def test_cycle_census_at_most_sequential(self):
        """Leave no more replica systems cycling at 128 sweeps than the sequential sweep, under both laws.

        The census systems are every kind the full census found: cycling under the sequential sweep, cycling
        under exact units without the alternation latch, and cycling under the unit sweep.
        """
        for law in ("point", "patch"):
            systems = [system for owner, system in self.tagged("census") if owner == law]
            world = World(systems)
            counts = {}
            for update in ("sequential", "unit"):
                histories = world.sweep(128, update, snapshots=True)
                counts[update] = sum(classify(s, h) == "cycle" for s, h in zip(systems, histories, strict=True))
                fixed = sum(classify(s, h) == "fixed" for s, h in zip(systems, histories, strict=True))
                self.assertEqual(fixed, 0)
            with self.subTest(law=law, counts=counts):
                self.assertGreater(counts["sequential"], 0)
                self.assertLessEqual(counts["unit"], counts["sequential"])

    def test_ring_friction_inside_final_disk(self):
        """End every patch sweep's ring units with each anchor's friction inside mu_a times its ring's final load.

        Anchor units (patch Stage A) hold each anchor's friction on the load the other members had at its visit,
        so they keep the sequential law's defect, at no more than its count of pairs (4 against 4 here).
        """
        systems = self.systems["patch"]
        world = World(systems)
        outside, impulses = {}, {}
        for name, update, ring_units in (("sequential", "sequential", 1), ("ring", "unit", 1), ("anchor", "unit", 0)):
            outside[name] = 0
            impulses[name] = world.sweep(8, update, ring_units=ring_units)
            for system, lam in zip(systems, impulses[name], strict=True):
                for p in pairs_of(system):
                    load = lam[ring_of(system, int(system["parent"][p]))].sum()
                    outside[name] += np.hypot(lam[p], lam[p + 1]) > system["mu"][p] * load + 1.0e-6
        self.assertGreater(outside["sequential"], 0)
        self.assertEqual(outside["ring"], 0)
        self.assertLessEqual(outside["anchor"], outside["sequential"])
        # The forms are different laws on multi-member rings, so the switch reaches the sweep.
        self.assertTrue(
            any(not np.array_equal(a, b) for a, b in zip(impulses["ring"], impulses["anchor"], strict=True))
        )

    def test_reference_is_a_fixed_point_of_the_sequential_sweep(self):
        """Keep every certified reference in one sequential sweep, to 1e-5 of its largest impulse.

        References certified to 1e-5 m/s are used; a looser one moves under either law by its residual.
        """
        for law in self.systems:
            tight = self.tight(law)
            start = [s["lam_ref"] for s in tight]
            for system, lam in zip(tight, World(tight).sweep(1, "sequential", start=start), strict=True):
                with self.subTest(law=law, system=system["source"]):
                    self.assertLess(np.abs(lam - system["lam_ref"]).max(), 1.0e-5 * np.abs(system["lam_ref"]).max())

    def test_reference_is_a_fixed_point_of_the_unit_sweep(self):
        """Keep every certified reference in one unit sweep, to 1e-5 of its largest impulse, in both ring forms.

        Every solution of the whole problem solves each unit's problem, so the unit returns the impulses it holds where
        that problem has several solutions: sliding contacts whose friction presses their normal down (kappa 1.03 to
        1.46) also admit take-off or a stick, and a branch order of take-off, stick, slide left 6 of the 12 point and 7
        of the 14 patch references here. The slide roots of these contacts, which a negative ``S0 = -b_n / W_nn``
        puts outside the bracket of a positive one, and roots the evaluation cap stops short from scratch, are resumed
        from the held friction (a held slide of gain outside (0, 8] takes the split step, which keeps it); a ring's
        coinciding normals (patch 2549, whose velocities agree to 8e-6 m/s) keep their split, and a ring's anchors
        restart from their held friction after a wrong first basis (patch 1781). Patch rings also run as anchor units
        (patch Stage A).
        """
        for law in self.systems:
            tight = self.tight(law)
            world = World(tight)
            start = [s["lam_ref"] for s in tight]
            for ring_units in (1, 0) if law == "patch" else (1,):
                after = world.sweep(1, "unit", start=start, ring_units=ring_units)
                for system, lam in zip(tight, after, strict=True):
                    with self.subTest(law=law, ring_units=ring_units, system=system["source"]):
                        reference = system["lam_ref"]
                        self.assertLess(np.abs(lam - reference).max(), 1.0e-5 * np.abs(reference).max())

    def test_sweeps_from_a_reference_leave_only_the_recorded_ones(self):
        """Leave exactly the recorded certified references within 256 sweeps started from them, under each update.

        One sequential sweep keeps every reference, and the exit law stops it there, but for patch 2549, whose
        period-two cycle moves its coinciding normals above their floor and repels it within 256 sweeps (the bitwise
        exit kept sweeping point 2114 too, until the cycle repelled it). The unit sweeps keep every reference, also where a unit's problem has several
        solutions (test_reference_is_a_fixed_point_of_the_unit_sweep), and stay there. Exact solves of a slide whose
        load answers the contact's approach with the wrong sign (the middle one of several slides) or more than 8
        times its normal row's own response are driven away from it by the coupling of units: without the split step
        a unit holding such a slide takes, 256 sweeps leave point 276, 308, 2053, 2069 and 1477 (the last one by its
        gain) and patch 260, 276, 1647 and 1663; with that step only for frictions within 1e-4 of the slide instead of
        1e-3, point 1477; only where the held friction is the root to its convergence tolerance, point 276 and 1477 and
        patch 260, 1647 and 1663. On 1,500 bench systems per law of the same replica set, 256 unit sweeps from
        certified references leave 0 of 1,376 point and 1 of 1,112 patch ones (5 and 4 under the sequential sweep).
        Patch rings also run as anchor units. A change of the law that alters these sets fails here and must record
        its measured sets.
        """
        for law in self.systems:
            tight = self.tight(law)
            world = World(tight)
            for update, ring_units in self.updates(law):
                later = world.sweep(256, update, start=[s["lam_ref"] for s in tight], ring_units=ring_units)
                left = {s["source"] for s, lam in zip(tight, later, strict=True) if residuals(s, lam)[0].max() > 1.0e-3}
                with self.subTest(law=law, update=update, ring_units=ring_units):
                    self.assertEqual(left, _LEFT_REFERENCES[update][law])

    def test_stationary_states_solve_the_ncp(self):
        """Leave no stationary state after 256 sweeps that is not a solution (NCP above 1e-4), under each update.

        A state is stationary when its last sweep moves no impulse by more than 1e-6 of the largest: the float32
        resolution of a sweep and the census's bound for a fixed state. Patch rings also run as anchor units
        (patch Stage A). At least 5 systems are stationary per law and update (5 to 8 measured).
        """
        for law, systems in self.systems.items():
            world = World(systems)
            for update, ring_units in self.updates(law):
                histories = world.sweep(256, update, snapshots=True, ring_units=ring_units)
                stationary = 0
                for system, history in zip(systems, histories, strict=True):
                    scale = max(np.abs(history[-1]).max(), 1.0e-12)
                    if np.abs(history[-1] - history[-2]).max() <= 1.0e-6 * scale:
                        stationary += 1
                        with self.subTest(law=law, update=update, ring_units=ring_units, system=system["source"]):
                            self.assertLessEqual(residuals(system, history[-1])[0].max(), 1.0e-4)
                with self.subTest(law=law, update=update, ring_units=ring_units):
                    self.assertGreaterEqual(stationary, 5)


class TestContactUpdateSplitStep(unittest.TestCase):
    """The split step of a held slide whose gain lies outside (0, 8], in the loop sweep of one contact."""

    def test_lone_unit_lands_on_its_press_slides_and_stays(self):
        """Land a lone contact on its press slides in one sweep; inside the hold, step by the normal row's response.

        The pressed contact held on either press slide with its friction turned on its disk. Beyond the hold of
        CONTACT_UNIT_SPLIT_HOLD (1e-3 of the slide's friction; a 0.02 rad turn) the exact solve lands on the slide in
        one sweep (NCP below 1e-6). Inside it (2e-4 rad) the unit holds the slide, whose gain asks for the split step:
        a lone unit then follows the normal row's response, nearing the slide of gain +35 by about 1 - 1/35 per sweep
        (0.85 over 7 sweeps measured) and leaving the one of gain -29 by about 1 + 1/29 (1.23 over 7), until an exact
        solve returns it. Every start stays within NCP 1e-5 of a solution for 256 sweeps.
        """
        system = point_unit_system(W_PRESSED, B_PRESSED, MU_PRESSED)
        slides = slide_solutions(W_PRESSED, B_PRESSED, MU_PRESSED)
        self.assertEqual(len(slides), 2)
        for k, (_, x) in enumerate(slides):
            for turn in (0.02, 2.0e-4):
                rotation = np.array([[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]])
                start = np.array([x[0], *(rotation @ x[1:])])
                history = World([system]).sweep(256, "unit", start=[start], snapshots=True)[0]
                distance = np.abs(history - x).max(axis=1)
                ncp = np.array([residuals(system, lam)[0].max() for lam in history])
                with self.subTest(slide=k, turn=turn):
                    self.assertLess(ncp.max(), 1.0e-5)
                    if turn > contact_unit.CONTACT_UNIT_SPLIT_HOLD:
                        self.assertLess(ncp[0], 1.0e-6)
                        self.assertLess(distance[0], 0.01 * np.abs(start - x).max())
                    else:
                        ratio = distance[7] / distance[0]
                        if k == 0:
                            self.assertTrue(0.7 < ratio < 0.95, ratio)
                        else:
                            self.assertTrue(1.1 < ratio < 1.4, ratio)


class TestContactUpdateSolveState(unittest.TestCase):
    """The unit sweeps' latches and slide roots last one solve: a launch at iteration_offset 0 clears them."""

    def test_loop_unit_state_lasts_one_solve(self):
        """Continue a solve's unit state across one-sweep launches and clear it when a new solve starts.

        One launch of 8 or 64 sweeps equals as many one-sweep launches that continue it (iteration_offset 1, 2,
        ...), as the interleaved split driver launches them, to the exit law's floor: the launch stops at its first
        quiet sweep, the one-sweep launches sweep on by changes within it. A new solve equals a fresh one bitwise,
        whatever the previous solve left. Restarting the solve at every launch, as the prototype harness did, changes
        the result of 6 and 11 of the 27 stored systems at 8 and 64 sweeps by more than 1e-4 of their largest impulse
        (the others by less than 1e-6, at the floor): their latches restart, and the slide roots that a held friction
        does not resume. A change of the state's semantics that changes fewer fails here.
        """
        systems = replica_systems("point") + replica_systems("patch")
        fresh = {sweeps: World(systems).sweep(sweeps, "unit") for sweeps in (8, 64)}
        world = World(systems)
        for sweeps in (8, 64):
            continued = world.sweep(sweeps, "unit", snapshots=True)
            restarted = world.sweep(sweeps, "unit", snapshots=True, restart=True)
            again = world.sweep(sweeps, "unit")
            for k, system in enumerate(systems):
                with self.subTest(sweeps=sweeps, system=system["source"]):
                    scale = max(float(np.abs(fresh[sweeps][k]).max()), 1.0e-12)
                    np.testing.assert_allclose(continued[k][-1], fresh[sweeps][k], rtol=0.0, atol=1.0e-4 * scale)
                    np.testing.assert_array_equal(again[k], fresh[sweeps][k])
            changed = sum(
                np.abs(h[-1] - lam).max() > 1.0e-4 * max(float(np.abs(lam).max()), 1.0e-12)
                for h, lam in zip(restarted, fresh[sweeps], strict=True)
            )
            self.assertGreaterEqual(changed, {8: 6, 64: 11}[sweeps])

    def test_free_body_unit_state_lasts_one_solve(self):
        """Continue and clear the free-body unit sweep's state as the loop sweep's, on two contacts that alternate.

        The two contacts' loads reverse between visits, so their latches relax them: restarting the solve at every
        launch moves their loads by 5% after 8 sweeps. The one-sweep launches equal one launch to the exit law's
        floor, past which they sweep on.
        """
        system = alternating_system()
        body = FreeBody(system)
        normals = np.flatnonzero(system["rtype"] == CONTACT)
        for sweeps in (8, 64):
            single = sweep_free_body(system, sweeps, "unit")
            with self.subTest(sweeps=sweeps):
                scale = max(float(np.abs(single).max()), 1.0e-12)
                np.testing.assert_allclose(
                    body.sweep(sweeps, "unit", offsets=range(sweeps)), single, rtol=0.0, atol=1.0e-4 * scale
                )
                np.testing.assert_array_equal(body.sweep(sweeps, "unit"), single)
                restarted = body.sweep(sweeps, "unit", offsets=[0] * sweeps)
                if sweeps == 8:
                    self.assertGreater(np.abs(restarted[normals] - single[normals]).max(), 0.01 * single[normals].max())


def _free_box_model(**builder):
    """A one-joint arm and a free box resting on the ground in one CPU world."""
    model = newton.ModelBuilder()
    body = model.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    model.add_articulation([model.add_joint_revolute(-1, body)])
    free = model.add_body(xform=wp.transform((0.0, 0.0, 0.098), wp.quat_identity()))
    model.add_shape_box(free, hx=0.1, hy=0.1, hz=0.1)
    model.add_ground_plane()
    return model.finalize(device="cpu", **builder)


def _step(solver, model, steps):
    state, after = model.state(), model.state()
    control = model.control()
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    for _ in range(steps):
        pipeline.collide(state, contacts)
        solver.step(state, after, control, contacts, 1.0 / 240.0)
        state, after = after, state
    return state


class TestContactUpdateSolver(unittest.TestCase):
    """The ``contact_update`` option, its route record and launches."""

    def test_sequential_is_the_unchanged_law(self):
        """Run the unchanged sequential kernels for the default and an explicit "sequential" law."""
        states = []
        with _without_solver_environment():
            for options in ({}, {"contact_update": "sequential"}):
                model = _free_box_model()
                solver = SolverFeatherPGS(model, pgs_iterations=8, **options)
                sweep = solver.get_path_selection()["routes"]["sweep"]
                self.assertEqual(sweep["dense"], {"kernels": ["sweep_loop"]})
                self.assertEqual(sweep["free_body_matrix_free"], "sweep_mf_loop")
                self.assertEqual(sweep["contact_update"]["law"], "sequential")
                with launched_kernels() as keys:
                    states.append(_step(solver, model, 20))
                self.assertIn("sweep_loop", keys)
                self.assertNotIn("sweep_loop_unit", keys)
                self.assertNotIn("sweep_mf_loop_unit", keys)
        for state in states[1:]:
            for name in ("body_q", "body_qd", "joint_q", "joint_qd"):
                np.testing.assert_array_equal(getattr(state, name).numpy(), getattr(states[0], name).numpy())

    def test_sequential_dispatch_runs_the_unchanged_kernel_on_stored_systems(self):
        """Dispatch "sequential" to the loop sweep, bitwise, and "unit" to the unit loop sweep on stored systems."""
        systems = replica_systems("point") + replica_systems("patch")
        world = World(systems)
        for update, expected in (("sequential", world.sweep(8, "sequential")), ("unit", world.sweep(8, "unit"))):
            impulses = wp.zeros((len(systems), world.D), dtype=float, device=_CPU)
            unit = update == "unit"
            stub = SimpleNamespace(
                _dense_contact_unit=unit,
                _pgs_solve_loop_kernel=(
                    sweep_routes_split.loop_unit_kernel if unit else sweep_routes_split.loop_kernel
                )(),
                _solve_readout=SimpleNamespace(operands=lambda kernel: []),
                world_count=len(systems),
                constraint_count=world.w_count,
                dense_max_constraints=world.D,
                diag=world.w["diag"],
                C=world.w_C,
                rhs=world.w["rhs"],
                impulses=impulses,
                pgs_omega=1.0,
                row_type=world.w_rtype,
                row_parent=world.w_parent,
                row_mu=world.w["mu"],
                row_w=world.w["mu"],  # unread: no regularization
                _regularization_enabled=False,
                _contact_unit_patch="ring",
                _dense_unit_state=world.state,
                _dense_unit_rows=world.rows,
                _dense_unit_block=world.block,
                model=SimpleNamespace(device=wp.get_device(_CPU)),
            )
            with launched_kernels() as keys:
                SolverFeatherPGS._stage5_pgs_solve_world_loop(stub, 8, 0, 0)
            self.assertEqual(keys, ["sweep_loop_unit" if unit else "sweep_loop"])
            out = impulses.numpy()
            for k, lam in enumerate(expected):
                with self.subTest(update=update, system=k):
                    np.testing.assert_array_equal(out[k, : len(lam)].astype(np.float64), lam)

    def test_unit_option_runs_the_unit_loop_sweeps(self):
        """Run both CPU loop sweeps' unit variants for contact_update="unit"."""
        with _without_solver_environment():
            for options in ({"contact_update": "unit"},):
                with self.subTest(options=options):
                    model = _free_box_model()
                    with warnings.catch_warnings():
                        warnings.simplefilter("error")
                        solver = SolverFeatherPGS(model, pgs_iterations=8, **options)
                    selected = solver.get_path_selection()
                    sweep = selected["routes"]["sweep"]
                    self.assertEqual(sweep["dense"], {"kernels": ["sweep_loop_unit"]})
                    self.assertEqual(sweep["free_body_matrix_free"], "sweep_mf_loop_unit")
                    self.assertEqual(
                        sweep["contact_update"],
                        {
                            "requested": "unit",
                            "law": "unit",
                            "unit": ["sweep_loop_unit", "sweep_mf_loop_unit"],
                            "sequential": [],
                            "patch_units": "ring",
                            "reason": sweep["contact_update"]["reason"],
                        },
                    )
                    option = selected["options"]["contact_update"]
                    self.assertEqual(option["applied"], "unit")
                    self.assertEqual(option["source"], "argument")
                    with launched_kernels() as keys:
                        state = _step(solver, model, 20)
                    # The arm and the box share a world: one-iteration dense and free-body launches interleave.
                    self.assertEqual(keys.count("sweep_loop_unit"), 20 * 8)
                    self.assertEqual(keys.count("sweep_mf_loop_unit"), 20 * 8)
                    self.assertNotIn("sweep_loop", keys)
                    self.assertTrue(np.all(np.isfinite(state.body_q.numpy())))
                    # The box rests on the ground.
                    self.assertAlmostEqual(float(state.body_q.numpy()[1, 2]), 0.1, delta=2.0e-3)

    def test_invalid_contact_update_raises(self):
        """Reject a contact update other than "sequential" or "unit"."""
        with _without_solver_environment():
            with self.assertRaisesRegex(ValueError, "contact_update must be one of"):
                SolverFeatherPGS(_free_box_model(), contact_update="block")

    def test_sweeps_without_units_keep_the_sequential_law_and_warn(self):
        """Record the sweeps of a "unit" request that keep the sequential law, and warn once with them."""
        sweep = {
            "contact_update": {
                "requested": "unit",
                "law": "unit",
                "unit": ["sweep_loop_unit"],
                "sequential": ["pgs_solve_mf_cuda"],
                "patch_units": "ring",
                "reason": "Only the loop sweeps solve contact units.",
            }
        }
        messages = path_selection.contact_update_warnings(sweep)
        self.assertEqual(len(messages), 1)
        self.assertIn("pgs_solve_mf_cuda", messages[0])
        sweep["contact_update"]["sequential"] = []
        self.assertEqual(path_selection.contact_update_warnings(sweep), [])
        sweep["contact_update"].update(requested="sequential", sequential=["pgs_solve_mf_cuda"])
        self.assertEqual(path_selection.contact_update_warnings(sweep), [])

    def test_patch_unit_switch_selects_anchor_units(self):
        """Launch the anchor-unit form of patch regions when the internal switch says so."""
        with _without_solver_environment():
            model = _free_box_model()
            solver = SolverFeatherPGS(model, pgs_iterations=8, contact_update="unit")
        solver._contact_unit_patch = "anchor"
        self.assertEqual(solver.get_path_selection()["routes"]["sweep"]["contact_update"]["patch_units"], "anchor")
        launches = []
        launch = wp.launch

        def record(kernel, *args, **kwargs):
            if getattr(kernel, "key", None) in ("sweep_loop_unit", "sweep_mf_loop_unit"):
                launches.append(kwargs["inputs"][-4])
            return launch(kernel, *args, **kwargs)

        with mock.patch.object(wp, "launch", record):
            _step(solver, model, 2)
        self.assertTrue(launches)
        self.assertEqual(set(launches), {0})


@unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
class TestContactUpdateCuda(unittest.TestCase):
    def test_cuda_loop_sweep_matches_cpu(self):
        """Sweep the stored systems with the unit loop kernel on CUDA to the CPU's row velocities, within 1e-5.

        The row velocities ``C lambda + rhs`` after 8 sweeps, in one launch and in one-sweep launches that continue
        the solve, are compared to 1e-5 of each system's largest residual at zero impulse. They are unique where the
        impulses are not: two dependent ring normals (patch 2549) and two anchors' friction (patch 1309) split their
        load by rounding, by up to 1e-2 relative when the CPU's inputs change by one ulp, while the velocities of
        every stored system then move by at most 6e-7 of that scale. Before treating a failure as a defect, check
        whether a unit took another branch at a near-tie on one device.
        """
        systems = replica_systems("point") + replica_systems("patch")
        references = [s["lam_ref"] for s in systems]
        for snapshots, start in ((False, None), (True, None), (False, references)):
            cpu = World(systems).sweep(8, "unit", snapshots=snapshots, start=start)
            cuda = World(systems, device="cuda:0").sweep(8, "unit", snapshots=snapshots, start=start)
            if snapshots:
                cpu, cuda = [history[-1] for history in cpu], [history[-1] for history in cuda]
            for system, on_cpu, on_cuda in zip(systems, cpu, cuda, strict=True):
                with self.subTest(system=system["source"], snapshots=snapshots, reference=start is not None):
                    difference = system["C"] @ (on_cuda - on_cpu)
                    self.assertLessEqual(np.abs(difference).max(), 1.0e-5 * np.abs(system["rhs"]).max())

    def test_cuda_reference_is_a_fixed_point_of_the_unit_sweep(self):
        """Keep every certified reference in one CUDA unit loop sweep, to 1e-5 of its largest impulse plus its residual.

        The CUDA twin of test_reference_is_a_fixed_point_of_the_unit_sweep, in both ring forms. An exact unit solve
        corrects the reference's own residual, which the device's rounding moves: on the CPU patch 2549 moves 8.3e-6
        of its largest impulse, correcting its 7.8e-6 m/s residual, 0.83 of the flat 1e-5 bound. Each reference may
        therefore also move by its residual over its smallest contact-row diagonal (3.1e-5 of 2549's largest impulse,
        below 2e-5 for every other).
        """
        for law in ("point", "patch"):
            tight = [s for s in replica_systems(law) if residuals(s, s["lam_ref"])[0].max() <= 1.0e-5]
            world = World(tight, device="cuda:0")
            for ring_units in (1, 0) if law == "patch" else (1,):
                after = world.sweep(1, "unit", start=[s["lam_ref"] for s in tight], ring_units=ring_units)
                for system, lam in zip(tight, after, strict=True):
                    with self.subTest(law=law, ring_units=ring_units, system=system["source"]):
                        reference = system["lam_ref"]
                        contact = (system["rtype"] == CONTACT) | (system["rtype"] == FRICTION)
                        own = residuals(system, reference)[0].max() / system["diag"][contact].min()
                        self.assertLess(np.abs(lam - reference).max(), 1.0e-5 * np.abs(reference).max() + own)

    def test_cuda_audit_block_holds_in_the_loop_sweep(self):
        """Hold the audit block's exact load with the CUDA unit loop sweep at 8 and 9 sweeps, in each ring form."""
        load = exact_point_contact(W_AUDIT, W0_AUDIT, 1.0)[0]
        for law, copies in (("point", 2), ("patch1", 3), ("patch2", 2)):
            system = audit_system(copies, law)
            normals = np.flatnonzero(system["rtype"] == CONTACT)
            for ring_units in TestContactUpdateAuditBlock.forms(law):
                for sweeps in (8, 9):
                    with self.subTest(law=law, copies=copies, ring_units=ring_units, sweeps=sweeps):
                        lam = World([system], device="cuda:0").sweep(sweeps, "unit", ring_units=ring_units)[0]
                        self.assertAlmostEqual(lam[normals].sum(), load, delta=1.0e-4)
                        self.assertGreaterEqual(residuals(system, lam)[1][normals].min(), -1.0e-5)

    def test_cuda_routes_without_units_keep_the_sequential_law(self):
        """Keep the sequential law on CUDA sweeps without units, record them and warn; the loop sweep runs units."""
        with _without_solver_environment():
            builder = newton.ModelBuilder()
            body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
            builder.add_articulation([builder.add_joint_revolute(-1, body)])
            builder.add_shape_sphere(body, radius=0.1)
            builder.add_ground_plane()
            model = builder.finalize(device="cuda:0")
            with self.assertWarnsRegex(UserWarning, "contact_update='unit' applies only to some sweeps"):
                tiled = SolverFeatherPGS(model, contact_update="unit")
            update = tiled.get_path_selection()["routes"]["sweep"]["contact_update"]
            self.assertEqual(update["law"], "sequential")
            self.assertTrue(update["sequential"])
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter("always")
                with mock.patch.dict(SolverFeatherPGS._route_pins, {"pgs_kernel": "loop"}):
                    loop = SolverFeatherPGS(model, contact_update="unit")
            self.assertFalse([w for w in recorded if "contact_update" in str(w.message)])
            update = loop.get_path_selection()["routes"]["sweep"]["contact_update"]
            self.assertEqual(update["unit"], ["sweep_loop_unit"])
            self.assertEqual(update["sequential"], [])


if __name__ == "__main__":
    unittest.main()
