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
synthesized blocks (the audit block of the cycle, an oversize patch ring) and
on stored ANYmal-replica systems of the CPU contact bench
(``golden_data/test_feather_pgs_contact_update/replica_systems.npz``: certified
references, upper triangles of the Delassus matrices, chosen for zero-normal
events, a 128-sweep cycle census, friction outside its final disk and the
cross-contact case of replica 79). Residuals follow the bench: a normal row's
``|min(d lambda, w)|`` and a friction pair's distance to the projection on the
disk of its ring's load, in velocity units.
"""

import os
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import contact_unit, contact_unit_sweeps
from newton._src.solvers.feather_pgs import kernels as K

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
        self.state = wp.zeros((W, D, contact_unit_sweeps.CONTACT_UNIT_STATE_FIELDS), dtype=float, device=device)
        self.rows = wp.zeros((W, D), dtype=int, device=device)
        self.block = wp.zeros(W * contact_unit.RING_BLOCK_SIZE, dtype=float, device=device)

    def sweep(self, iterations, update, *, start=None, snapshots=False, ring_units=1):
        """Sweep every system ``iterations`` times from ``start`` (cold by default) under one contact update.

        Returns the impulses per system, or with ``snapshots`` the impulses after every sweep (one launch per
        sweep that continues the solve, as the interleaved split driver launches).
        """
        lam = np.zeros((len(self.systems), self.D), np.float32)
        if start is not None:
            for k, s in enumerate(start):
                lam[k, : len(s)] = s
        impulses = wp.array(lam, dtype=float, device=self.device)
        launches = [(iterations, 0)] if not snapshots else [(1, it) for it in range(iterations)]
        history = []
        for count, offset in launches:
            common = [
                self.w_count,
                self.D,
                self.w["diag"],
                self.w_C,
                self.w["rhs"],
                impulses,
                count,
                1.0,
                self.w_rtype,
                self.w_parent,
                self.w["mu"],
                0,
                offset,
            ]
            if update == "unit":
                wp.launch(
                    contact_unit_sweeps.pgs_solve_loop_unit,
                    dim=len(self.systems),
                    inputs=[*common, ring_units, self.state, self.rows, self.block],
                    device=self.device,
                )
            else:
                wp.launch(K.pgs_solve_loop, dim=len(self.systems), inputs=common, device=self.device)
            if snapshots:
                history.append(impulses.numpy().astype(np.float64))
        if snapshots:
            stacked = np.stack(history)
            return [stacked[:, k, : self.count[k]] for k in range(len(self.systems))]
        out = impulses.numpy().astype(np.float64)
        return [out[k, : self.count[k]] for k in range(len(self.systems))]


def sweep_free_body(system, iterations, update, ring_units=1):
    """Sweep a system as the free-body rows of one body (identity mass, rows over its first coordinates)."""
    J = system["J"]
    m, dofs = J.shape
    J6 = np.zeros((1, m, 6), np.float32)
    J6[0, :, :dofs] = J
    v = np.zeros(6, np.float32)
    # The body's velocity gives each row its residual at zero impulse (the rows carry no bias).
    v[:dofs] = np.linalg.lstsq(J, system["rhs"], rcond=None)[0]
    impulses = wp.zeros((1, m), dtype=float, device=_CPU)
    velocity = wp.array(v, dtype=float, device=_CPU)
    arrays = [
        wp.array([m], dtype=int, device=_CPU),
        wp.zeros((1, m), dtype=int, device=_CPU),
        wp.full((1, m), -1, dtype=int, device=_CPU),
        wp.array(J6, dtype=float, device=_CPU),
        wp.zeros((1, m, 6), dtype=float, device=_CPU),
        wp.array(J6, dtype=float, device=_CPU),
        wp.zeros((1, m, 6), dtype=float, device=_CPU),
        wp.array((1.0 / np.einsum("ij,ij->i", J, J))[None].astype(np.float32), dtype=float, device=_CPU),
        wp.zeros((1, m), dtype=float, device=_CPU),
        wp.array(system["rtype"][None], dtype=int, device=_CPU),
        wp.array(system["parent"][None], dtype=int, device=_CPU),
        wp.array(system["mu"][None].astype(np.float32), dtype=float, device=_CPU),
        wp.ones((1, m), dtype=float, device=_CPU),
        wp.array([0], dtype=int, device=_CPU),
        wp.array([0], dtype=int, device=_CPU),
        iterations,
        1.0,
        0,
    ]
    if update == "unit":
        inputs = [
            *arrays,
            0,
            0,
            ring_units,
            wp.zeros((1, m, contact_unit_sweeps.CONTACT_UNIT_STATE_FIELDS), dtype=float, device=_CPU),
            wp.zeros((1, m), dtype=int, device=_CPU),
            wp.zeros(contact_unit.RING_BLOCK_SIZE, dtype=float, device=_CPU),
        ]
        kernel = contact_unit_sweeps.pgs_solve_mf_loop_unit
    else:
        inputs = [*arrays, int(K.FRICTION_MODE_CURRENT), 0, 0]
        kernel = K.pgs_solve_mf_loop
    wp.launch(kernel, dim=1, inputs=inputs, outputs=[impulses, velocity], device=_CPU)
    return impulses.numpy()[0].astype(np.float64)


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


def first_moved_unit(system, lam, tolerance):
    """The rows, normals and anchor count of the first contact unit (in sweep order) that moved from the reference."""
    rtype, parent = system["rtype"], system["parent"]
    seen = set()
    for n in np.flatnonzero(rtype == CONTACT):
        if n in seen:
            continue
        normals = sorted(ring_of(system, int(n)))
        seen.update(normals)
        pairs = [r + 1 for r in normals if r + 2 < len(rtype) and rtype[r + 1] == FRICTION and parent[r + 1] == r]
        rows = normals + [x for p in pairs for x in (p, p + 1)]
        if np.abs(lam[rows] - system["lam_ref"][rows]).max() >= tolerance:
            return rows, normals, len(pairs)
    return None, None, 0


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

    def check(self, system, lam):
        normals = np.flatnonzero(system["rtype"] == CONTACT)
        _, w = residuals(system, lam)
        self.assertAlmostEqual(lam[normals].sum(), self.load, delta=1.0e-4)
        self.assertGreaterEqual(w[normals].min(), -1.0e-5)

    def test_unit_loop_holds_the_audit_block(self):
        """Hold the audit block's exact load with no approach at 8 and 9 sweeps, for 1-3 copies under both laws."""
        self.assertAlmostEqual(self.load, 0.40878, places=4)
        for law, copies in self.cases():
            system = audit_system(copies, law)
            for sweeps in (8, 9):
                with self.subTest(law=law, copies=copies, sweeps=sweeps):
                    self.check(system, World([system]).sweep(sweeps, "unit")[0])

    def test_unit_free_body_loop_holds_the_audit_block(self):
        """Hold the audit block's exact load in the free-body loop sweep, for 1-3 copies under both laws."""
        for law, copies in self.cases():
            system = audit_system(copies, law)
            for sweeps in (8, 9):
                with self.subTest(law=law, copies=copies, sweeps=sweeps):
                    self.check(system, sweep_free_body(system, sweeps, "unit"))

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
    """Stored ANYmal-replica systems with certified references."""

    @classmethod
    def setUpClass(cls):
        cls.systems = {law: replica_systems(law) for law in ("point", "patch")}

    def tagged(self, tag):
        for law, systems in self.systems.items():
            for system in systems:
                if tag in system["tags"]:
                    yield law, system

    def test_zero_normal_events_recover(self):
        """Recover every group the sequential sweep leaves unloaded and approaching at 8 sweeps."""
        cases = list(self.tagged("zero_end"))
        self.assertEqual(len(cases), 7)
        world = World([system for _, system in cases])
        sequential = world.sweep(8, "sequential")
        unit = world.sweep(8, "unit")
        for (law, system), before, after in zip(cases, sequential, unit, strict=True):
            with self.subTest(law=law, system=system["source"]):
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
        """End every patch sweep with each anchor's friction inside mu_a times its ring's final load."""
        systems = self.systems["patch"]
        world = World(systems)
        outside = {}
        for update in ("sequential", "unit"):
            outside[update] = 0
            for system, lam in zip(systems, world.sweep(8, update), strict=True):
                for p in pairs_of(system):
                    load = lam[ring_of(system, int(system["parent"][p]))].sum()
                    outside[update] += np.hypot(lam[p], lam[p + 1]) > system["mu"][p] * load + 1.0e-6
        self.assertGreater(outside["sequential"], 0)
        self.assertEqual(outside["unit"], 0)

    def test_reference_is_a_fixed_point(self):
        """Keep a certified reference in one sweep, except where a unit's own problem has another solution.

        References certified to 1e-5 m/s are used; a looser one moves under either law by its residual. The
        sequential sweep moves no impulse by 1e-5 of the largest. The unit sweep solves each unit's local problem
        in a fixed branch order (take-off, else stick, else slide), so where that problem has several solutions
        (a sliding contact whose friction presses its normal down also admits take-off or a stick) it can leave a
        reference built on another branch. The first unit a sweep moves sees the reference itself: a point unit
        must move from one exact solution of its local problem to another. A ring unit's split among dependent
        normals, proximal step and two-anchor alternation are transient: the sweeps return to a solution.
        """
        moved = 0
        for law, systems in self.systems.items():
            tight = [s for s in systems if residuals(s, s["lam_ref"])[0].max() <= 1.0e-5]
            self.assertGreaterEqual(len(tight), len(systems) - 1)
            world = World(tight)
            start = [s["lam_ref"] for s in tight]
            for system, lam in zip(tight, world.sweep(1, "sequential", start=start), strict=True):
                with self.subTest(law=law, update="sequential", system=system["source"]):
                    self.assertLess(np.abs(lam - system["lam_ref"]).max(), 1.0e-5 * np.abs(system["lam_ref"]).max())
            for system, lam in zip(tight, world.sweep(1, "unit", start=start), strict=True):
                reference = system["lam_ref"]
                tolerance = 1.0e-5 * np.abs(reference).max()
                if np.abs(lam - reference).max() < tolerance:
                    continue
                moved += 1
                with self.subTest(law=law, update="unit", system=system["source"]):
                    rows, normals, _ = first_moved_unit(system, lam, tolerance)
                    self.assertIsNotNone(rows)
                    if len(normals) == 1:
                        state = reference.copy()
                        state[rows] = lam[rows]
                        self.assertLessEqual(residuals(system, reference)[0][rows].max(), 1.0e-5)
                        self.assertLessEqual(residuals(system, state)[0][rows].max(), 1.0e-5)
                    else:
                        later = World([system]).sweep(32, "unit", start=[reference])[0]
                        self.assertLessEqual(residuals(system, later)[0].max(), 1.0e-5)
        self.assertGreater(moved, 0)

    def test_stationary_states_solve_the_ncp(self):
        """Leave no stationary state after 256 sweeps that is not a solution (NCP above 1e-4), under both laws."""
        for law, systems in self.systems.items():
            world = World(systems)
            for update in ("sequential", "unit"):
                histories = world.sweep(256, update, snapshots=True)
                for system, history in zip(systems, histories, strict=True):
                    scale = max(np.abs(history[-1]).max(), 1.0e-12)
                    if np.abs(history[-1] - history[-2]).max() <= 1.0e-7 * scale:
                        with self.subTest(law=law, update=update, system=system["source"]):
                            self.assertLessEqual(residuals(system, history[-1])[0].max(), 1.0e-4)


@unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
class TestContactUpdateCuda(unittest.TestCase):
    def test_cuda_loop_sweep_matches_cpu(self):
        """Sweep the stored systems with the unit loop kernel on CUDA as on CPU, to 1e-5 of the largest impulse."""
        systems = replica_systems("point") + replica_systems("patch")
        cpu = World(systems).sweep(8, "unit")
        cuda = World(systems, device="cuda:0").sweep(8, "unit")
        for k, (a, b) in enumerate(zip(cpu, cuda, strict=True)):
            with self.subTest(system=k):
                self.assertLessEqual(np.abs(a - b).max(), 1.0e-5 * max(np.abs(a).max(), 1.0e-6))

    def test_cuda_audit_block_holds_in_the_loop_sweep(self):
        """Hold the audit block's exact load with the CUDA unit loop sweep at 8 and 9 sweeps."""
        load = exact_point_contact(W_AUDIT, W0_AUDIT, 1.0)[0]
        for law, copies in (("point", 2), ("patch1", 3), ("patch2", 2)):
            system = audit_system(copies, law)
            normals = np.flatnonzero(system["rtype"] == CONTACT)
            for sweeps in (8, 9):
                with self.subTest(law=law, copies=copies, sweeps=sweeps):
                    lam = World([system], device="cuda:0").sweep(sweeps, "unit")[0]
                    self.assertAlmostEqual(lam[normals].sum(), load, delta=1.0e-4)
                    self.assertGreaterEqual(residuals(system, lam)[1][normals].min(), -1.0e-5)


if __name__ == "__main__":
    unittest.main()
