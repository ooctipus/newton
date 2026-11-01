# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The general placement's one writer (FL3.3): rows by the walk, the slips' frame and the leave in one frame.

- **The block's exactness** (FL3_design §1.4, ``exp_block``, W9's release): on random trees with a weld, a mimic or
  both and one to three slipping drives, the release then the closure (``R``, then the projector of ``R Z_B``'s
  orthonormal basis) gives the decided closed-loop operator ``J_c [A_F^-1 - A_F^-1 J_B^T (J_B A_F^-1 J_B^T)^+ J_B
  A_F^-1] J_c^T`` in float64, and in float32 with the split pivot at the level of a float32 refactor of ``A_F``.
- **The device release** (:func:`.tree_factor.walk_release_kernel`, W9's verdict): on a tree whose drives saturate,
  ``R^T R`` equals ``I + B C^-1 B^T`` of the float64 walk, ``R`` keeps every row's support (tree-lower), and the rows
  the writer stores give the decided operator ``J A_F^-1 J^T``, not the all-``K`` one: without the release a row on a
  slipping joint takes ``(I + K) / I`` its impulse.
- **One frame per placement** (general worlds, the CPU catalogue): every walked row against its physical ``J v_hat``
  (``Z_i^T L^T P v_hat`` against the incident velocity its walk summed), and the step's leave against the float64
  walk's.
"""

import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.kernels import PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT
from newton._src.solvers.feather_pgs.tree_factor import TreeWalk, local_frames
from newton.solvers import SolverFeatherPGS
from newton.tests.test_feather_pgs_walk import BOUND, ROUNDING, catalogue, run_route

F32 = np.float32


# ---------------------------------------------------------------------------------------------------------------------
# The block primitive in numpy: frame (slips), then closure (links, mimics)
# ---------------------------------------------------------------------------------------------------------------------


def random_tree(rng, bodies: int, device="cpu"):
    """A random tree of ``bodies`` links (revolute, prismatic and ball joints) with a drive on every scalar coordinate,
    no gravity: its model, built on ``device``."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    links, joints = [], []
    for b in range(bodies):
        parent = -1 if b == 0 else links[int(rng.integers(0, b))]
        inertia = np.diag(rng.uniform(0.02, 0.2, 3))
        link = builder.add_link(mass=float(rng.uniform(0.5, 2.0)), inertia=wp.mat33(inertia))
        offset = wp.transform(wp.vec3(*rng.normal(size=3) * 0.2), wp.quat_identity())
        axis = tuple(np.eye(3)[int(rng.integers(0, 3))])
        kind = int(rng.integers(0, 3)) if b > 0 else 0
        if kind == 0:
            joint = builder.add_joint_revolute(parent=parent, child=link, axis=axis, parent_xform=offset)
        elif kind == 1:
            joint = builder.add_joint_prismatic(parent=parent, child=link, axis=axis, parent_xform=offset)
        else:
            joint = builder.add_joint_ball(parent=parent, child=link, parent_xform=offset)
        links.append(link)
        joints.append(joint)
    builder.add_articulation(joints)
    model = builder.finalize(device=device)
    q = model.joint_q.numpy()
    q[:] = rng.normal(size=q.shape) * 0.3
    # Ball joints hold unit quaternions.
    types, q_start = model.joint_type.numpy(), model.joint_q_start.numpy()
    for j in np.flatnonzero(types == int(newton.JointType.BALL)):
        quat = rng.normal(size=4)
        q[q_start[j] : q_start[j] + 4] = quat / np.linalg.norm(quat)
    return model, q


def orth(M: np.ndarray, tol: float = 1.0e-5) -> np.ndarray:
    """Modified Gram-Schmidt over the columns with the block's relative pivot (BACKBONE §1.4)."""
    Q = []
    for j in range(M.shape[1]):
        v = M[:, j].copy()
        own = float(v @ v)
        for q in Q:
            v = v - q * (q @ v)
        rest = float(v @ v)
        if own > 0.0 and rest > tol * own:
            Q.append((v / np.sqrt(rest)).astype(M.dtype))
    return np.stack(Q, axis=1) if Q else np.zeros((M.shape[0], 0), M.dtype)


def release(walk, S: list[int], K: np.ndarray, pivots: np.ndarray, dtype) -> np.ndarray:
    """The slips' release ``R`` in ``dtype`` (an ``n x n`` matrix by coordinate), as :func:`.tree_factor.walk_release_kernel`
    forms it on ``walk`` (the all-``K`` walk): the slips ``S`` root-most first, each one's column ``b = W e_i`` (the walk
    of ``e_i`` with the earlier releases applied), then FL3.4's recurrence along its path, leaf first, with ``K`` the
    released stiffness and ``pivots`` each slip's split pivot (before its released ``K``)."""
    n = walk.s.shape[0]
    body = np.searchsorted(walk.first, np.arange(n), side="right") - 1
    d = (walk.ell * walk.ell).astype(dtype)
    steps = []

    def apply(x, recurrence):
        s = dtype(0.0)
        for c, b, a, g in recurrence:
            u = dtype(x[c] - b * s)
            s = dtype(s + a * u)
            x[c] = dtype(g * u)
        return x

    for j, i in enumerate(S):
        column = walk.up(covector=np.eye(n, dtype=walk.dtype)[i])[0].astype(dtype)
        for recurrence in steps:
            column = apply(column, recurrence)
        path, c, k = [], i, int(body[i])
        while k >= 0:
            path += list(range(c, int(walk.first[k]) - 1, -1))
            k = int(walk.parent[k])
            c = int(walk.first[k + 1]) - 1 if k >= 0 else -1
        alpha, recurrence = dtype(-K[j]), []
        for c in path:
            scale = dtype(np.sqrt(d[c]))
            b = column[c]
            dbar = dtype(pivots[j]) if c == i else dtype(d[c] + alpha * (b * scale) * (b * scale))
            beta = dtype(alpha * b * scale / dbar)
            alpha = dtype(alpha * d[c] / dbar)
            recurrence.append((c, b, dtype(beta * scale), dtype(scale / np.sqrt(dbar))))
            d[c] = dbar
        steps.append(recurrence)
    R = np.eye(n, dtype=dtype)
    for col in range(n):
        for recurrence in steps:
            R[:, col] = apply(R[:, col], recurrence)
    return R


def block_case(rng, kind: str, slips: int, kappa: float) -> dict:
    """One exp_block case on a random Newton tree: relative errors of the frame-then-closure operator against the exact
    decided closed-loop operator, in float64 and in float32 with the split pivot."""
    model, q = random_tree(rng, int(rng.integers(5, 10)))
    parent, first, columns, inertia, offset = local_frames(model, q, 0)
    n = columns.shape[0]
    bare = TreeWalk(parent, first, columns, inertia, offset, np.zeros(n))
    Linv, _ = bare.up(covector=np.eye(n))  # L^-1 of the leaf-first H, by coordinate
    L = np.linalg.inv(Linv)
    H = L @ L.T
    K = kappa * np.diag(H).copy()
    S = sorted(rng.choice(n, size=min(slips, n - 1), replace=False).tolist())
    K_F = K.copy()
    K_F[S] = 0.0
    A_F = H + np.diag(K_F)
    walk = TreeWalk(parent, first, columns, inertia, offset, K)
    walk32 = TreeWalk(parent, first, columns, inertia, offset, K, dtype=F32)
    bodies = walk.parent.size
    # The block: a weld of two bodies (six unit wrenches) and/or a mimic covector.
    rows_B = []
    if kind in ("weld", "both"):
        k, l = rng.choice(bodies, size=2, replace=False)
        for i in range(6):
            w = np.zeros(6)
            w[i] = 1.0
            rows_B.append(bare.up(int(k), w)[0] - bare.up(int(l), w)[0])
    if kind in ("mimic", "both"):
        a, b = rng.choice(n, size=2, replace=False)
        x = np.zeros(n)
        x[a], x[b] = 1.0, -rng.uniform(0.5, 1.5)
        rows_B.append(np.linalg.solve(L, x))  # L^-1 x: the walk of a covector on the bare factor
    # Physical rows: J^T = L Z (bare), so every row is in joint coordinates.
    JB = np.array([L @ z for z in rows_B])
    rows_c = []
    for _ in range(4):
        k = int(rng.integers(bodies))
        for _r in range(3):
            rows_c.append(L @ bare.up(k, rng.normal(size=6))[0])
    Jc = np.array(rows_c)
    AFi = np.linalg.inv(A_F)
    G = JB @ AFi @ JB.T
    W_ex = Jc @ (AFi - AFi @ JB.T @ np.linalg.pinv(G, rcond=1e-10) @ JB @ AFi) @ Jc.T
    # Relative to the open loop's operator: a block that locks the whole tree leaves W_ex at rounding.
    W_scale = np.linalg.norm(Jc @ AFi @ Jc.T)

    out = {}
    for name, w, dtype in (("f64", walk, np.float64), ("f32", walk32, F32)):
        Z_B = np.stack([w.up(covector=x.astype(dtype))[0] for x in JB], axis=1).astype(dtype)
        Z_c = np.stack([w.up(covector=x.astype(dtype))[0] for x in Jc], axis=1).astype(dtype)
        pivots = w.own_pivots[S].astype(dtype)  # P_i: the pivot before the slipping K is added (split pivot)
        R = release(w, S, K[S], pivots, dtype)
        Q = orth((R @ Z_B).astype(dtype))
        Z = (R @ Z_c).astype(dtype)
        Z = (Z - Q @ (Q.T @ Z)).astype(dtype)
        W = (Z.T @ Z).astype(np.float64)
        out[name] = float(np.linalg.norm(W - W_ex) / W_scale)
    return out


class TestBlockPrimitive(unittest.TestCase):
    """Release, then closure: exp_block (FL3_design §1.4, W9) on random Newton trees."""

    def test_release_then_closure_is_the_decided_closed_loop_operator(self):
        rng = np.random.default_rng(20261101)
        worst = {"f64": 0.0, "f32": 0.0}
        errors = {"f64": [], "f32": []}
        for trial in range(24):
            kind = ("weld", "mimic", "both")[trial % 3]
            kappa = (1.0, 1.0e2, 1.0e4)[(trial // 3) % 3]
            result = block_case(rng, kind, 1 + trial % 3, kappa)
            for key, value in worst.items():
                worst[key] = max(value, result[key])
                errors[key].append(result[key])
        # exp_block: float64 to 5e-15 (median) and 4.8e-9 (max); float32 with the split pivot 1.2e-7 / 3.8e-6, the naive
        # pivot 1.7e-3 at K/H_ii = 1e4.
        self.assertLess(np.median(errors["f64"]), 1.0e-12, worst)
        self.assertLess(worst["f64"], 1.0e-8, worst)
        self.assertLess(np.median(errors["f32"]), 1.0e-6, worst)
        self.assertLess(worst["f32"], 1.0e-4, worst)


# ---------------------------------------------------------------------------------------------------------------------
# The device frame on saturating drives
# ---------------------------------------------------------------------------------------------------------------------


def slipping_hinges(device="cpu"):
    """Three hinges, each a stiff PD drive held past its effort limit, each with a joint limit its rows hold."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    parent, joints = -1, []
    for inertia, (lo, hi) in zip((0.05, 0.01, 0.02), ((-0.2, 0.0), (-0.1, 0.0), (-0.3, 0.05)), strict=True):
        link = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * inertia))
        joints.append(
            builder.add_joint_revolute(
                parent=parent,
                child=link,
                axis=(0.0, 0.0, 1.0),
                target_ke=2.0e3,
                target_kd=20.0,
                effort_limit=1.0,
                armature=0.0,
                limit_lower=lo,
                limit_upper=hi,
                parent_xform=wp.transform(wp.vec3(0.3, 0.0, 0.0), wp.quat_identity()),
            )
        )
        parent = link
    builder.add_articulation(joints)
    model = builder.finalize(device=device)
    control = model.control()
    control.joint_target_q.assign(np.asarray([3.0, -3.0, 3.0], dtype=np.float32))
    return model, control


def device_release(view, art: int, n: int) -> np.ndarray:
    """The device release ``R`` of articulation ``art`` (:class:`.tree_factor.WalkRelease`) as an ``n x n`` float64
    matrix by coordinate: each slip's recurrence along its stored path, leaf first, the root-most slip first."""
    release = view.release
    count = int(release.count.numpy()[art])
    coordinate = release.coordinate.numpy()[art]
    b, a, g = (getattr(release, k).numpy()[art].astype(np.float64) for k in ("b", "a", "g"))
    R = np.eye(n)
    for j in range(count):
        s = np.zeros(n)
        for t in range(coordinate.shape[1]):
            c = int(coordinate[j, t])
            if c < 0:
                break
            u = R[c] - b[j, t] * s
            s = s + a[j, t] * u
            R[c] = g[j, t] * u
    return R


class TestSlipRelease(unittest.TestCase):
    """The release of H7's decided set on the walk: rows and leave of the decided operator."""

    def test_rows_take_the_decided_operator(self):
        for device in ("cpu", *(["cuda:0"] if wp.is_cuda_available() else [])):
            with self.subTest(device=device):
                model, control = slipping_hinges(device)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    solver = SolverFeatherPGS(model, angular_damping=0.0, pgs_iterations=30, enable_joint_limits=True)
                self.assertTrue(solver.placement.walks)
                state, out = model.state(), model.state()
                joint_q = state.joint_q.numpy().astype(np.float64)
                solver.step(state, out, control, None, 1.0 / 60.0)
                view = solver.factor.walk_view()
                self.assertEqual(int(view.release.count.numpy()[0]), 3)
                # R^T R = I + B C^-1 B^T against the float64 walk of the all-K factor at the refresh's coordinates.
                walk = solver.factor.walk(joint_q, 0)
                n = walk.s.shape[0]
                K = solver.factor.aug_row_K.numpy()[:n].astype(np.float64)
                Kd = solver.factor.aug_row_Kd.numpy()[:n].astype(np.float64)
                released = K - Kd
                B = np.stack([walk.up(covector=np.eye(n)[i])[0] for i in range(n)], axis=1)
                C = np.diag(1.0 / released) - B.T @ B
                expected = np.eye(n) + B @ np.linalg.solve(C, B.T)
                R = device_release(view, 0, n)
                # Tree-lower (lower triangular leaf first): on a chain, R_ab is nonzero only where a is b or one of
                # its ancestors, so every row keeps its support.
                self.assertEqual(np.count_nonzero(np.tril(R, -1)), 0)
                self.assertLess(np.linalg.norm(R.T @ R - expected) / np.linalg.norm(expected), 1.0e-5)
                # The rows (the hinges' limit rows): their Gram is the decided operator's. By coordinate the walk's
                # columns are B = Lc^-1 (A = Lc Lc^T, so A^-1 = B^T B), and a stored row is z = R Lc^-1 J^T.
                p = solver.placement
                count = int(p.constraint_count.numpy()[0])
                self.assertGreater(count, 0)
                Z = p.packed_Z.numpy()[0, :count, :n].astype(np.float64)
                J = Z @ np.linalg.inv(R).T @ np.linalg.inv(B).T
                A = np.linalg.inv(B.T @ B)
                W_rows = Z @ Z.T
                W_decided = J @ np.linalg.solve(A - np.diag(released), J.T)
                W_all = J @ np.linalg.solve(A, J.T)
                scale_W = np.linalg.norm(W_decided)
                self.assertLess(np.linalg.norm(W_rows - W_decided) / scale_W, 1.0e-5)
                # Without the release the rows would hold the all-K operator: (I + K) / I away.
                self.assertGreater(np.linalg.norm(W_all - W_decided) / scale_W, 1.0e-1)


# ---------------------------------------------------------------------------------------------------------------------
# One frame per placement: the walked rows and the leave of the general worlds
# ---------------------------------------------------------------------------------------------------------------------


def walked_frames(solver, joint_q: np.ndarray, capture) -> dict:
    """The general worlds' walked rows, the dense rows and the free bodies' (the bodies sink), against their physical
    ``J v_hat`` and the step's leave against the float64 walk's, as ratios to their rounding: ``n u |Z_i| |u_hat|``
    (and ``u`` of the incident velocity), and ``n u kappa |dv|`` (FL3.1's leave bound) with the publication's rounding
    of ``v_hat`` and ``v_out``. A free rigid body is a one-body tree of six coordinates. A tree whose drives slip this
    step (its rows carry the release, :class:`TestSlipRelease`), the closure's corrected rows (``test_feather_pgs_preelim``)
    and a world whose speed ball acted (its ``du`` is a law's, not ``sum Z lambda``) are left to their own checks."""
    forest, placement, factor = solver.forest, solver.placement, solver.factor
    out = {"rows": 0.0, "free_rows": 0.0, "leave": 0.0, "worlds": 0, "free_bodies": 0}
    sizes = forest.articulation_response_dof_count.numpy()
    offsets = forest.articulation_world_dof_offset.numpy()
    starts = forest.articulation_dof_start.numpy()
    art_world = forest.art_to_world.numpy()
    body_art = forest.body_to_articulation.numpy()
    slips = factor.walk_view().release.count.numpy()
    dense = bool(placement.packed_dense)
    incident = placement.row_incident.numpy().astype(np.float64)
    sides_all = placement.row_sides.numpy()
    counts = placement.constraint_count.numpy()
    first = placement.dense_first.numpy()
    walks, hats = {}, {}

    def walk_of(art):
        if art not in walks:
            walks[art] = factor.walk(joint_q, art)
        return walks[art]

    def u_hat_of(art):
        """``u_hat = L^T P v_hat`` by the float64 walk: the coordinates an articulation's rows read."""
        if art not in hats:
            n = int(sizes[art])
            inverse, _ = walk_of(art).up(covector=np.eye(n))
            hats[art] = np.linalg.solve(inverse.T, capture.v_hat[int(starts[art]) : int(starts[art]) + n])
        return hats[art]

    for world in range(int(forest.world_count)):
        line, count = int(first[world]), int(counts[world])
        if count == 0:
            continue
        for row in range(count):
            term, scale_z, scale_u, length = 0.0, 0.0, 0.0, 0
            at = 0
            skip = False
            for art in sides_all[line, row]:
                if art < 0:
                    continue
                n, offset = int(sizes[art]), int(offsets[art])
                entries = slice(offset if dense else at, (offset if dense else at) + n)
                at += n
                if slips[art] or solver.factor.eliminates:
                    skip = True
                    break
                u_hat = u_hat_of(art)
                Z = capture.Z_final[line, row, entries]
                term += Z @ u_hat
                scale_z, scale_u = scale_z + Z @ Z, scale_u + u_hat @ u_hat
                length = max(length, n)
            if skip or length == 0:
                continue
            bound = length * ROUNDING * np.sqrt(scale_z * scale_u) + ROUNDING * abs(incident[line, row])
            if bound > 0.0:
                out["rows"] = max(out["rows"], abs(term - incident[line, row]) / bound)
        out["worlds"] += 1
    # The free bodies' rows: each side a six-step walk, against the incident velocity the writer summed.
    ball = np.zeros(int(forest.world_count), dtype=bool)
    free_du = {}
    if capture.mf_final is not None:
        Z_a, Z_b, lam, mf_counts, body_a, body_b, _dof_a, _dof_b, kinds = capture.mf_final
        mf_incident = placement.mf_row_incident.numpy().astype(np.float64)
        for world, count in enumerate(mf_counts):
            line = int(capture.mf_first[world])
            for row in range(int(count)):
                if kinds[line, row] == PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT:
                    ball[world] |= lam[line, row] != 0.0
                    continue
                term, scale_z, scale_u = 0.0, 0.0, 0.0
                for body, Z in ((body_a, Z_a), (body_b, Z_b)):
                    if body[line, row] < 0:
                        continue
                    art = int(body_art[body[line, row]])
                    z, u_hat = Z[line, row], u_hat_of(art)
                    term += z @ u_hat
                    scale_z, scale_u = scale_z + z @ z, scale_u + u_hat @ u_hat
                    free_du[art] = free_du.get(art, 0.0) + z * lam[line, row]
                incident_row = mf_incident[line, row]
                bound = 6 * ROUNDING * np.sqrt(scale_z * scale_u) + ROUNDING * abs(incident_row)
                if bound > 0.0:
                    out["free_rows"] = max(out["free_rows"], abs(term - incident_row) / bound)
    # The leave: the published dv against the float64 walk's leave of the step's du (the final rows' Z lambda).
    for art in range(sizes.size):
        world, n = int(art_world[art]), int(sizes[art])
        if world < 0 or n == 0 or slips[art] or solver.factor.eliminates or ball[world]:
            continue
        line, count = int(first[world]), int(counts[world])
        du = np.zeros(n) + free_du.get(art, 0.0)
        for row in range(count):
            at = 0
            for side in sides_all[line, row]:
                if side < 0:
                    continue
                m = int(sizes[side])
                start = int(offsets[side]) if dense else at
                at += m
                if side == art:
                    du += capture.Z_final[line, row, start : start + m] * capture.impulses[line, row]
        if not np.any(du):
            continue
        out["free_bodies"] += int(art in free_du)
        dofs = int(starts[art]) + np.arange(n)
        dv = capture.v_out[dofs] - capture.v_hat[dofs]
        walk = walk_of(art)
        expected = walk.down(du)[0]
        inverse, _ = walk.up(covector=np.eye(n))
        kappa = float(np.linalg.cond(inverse))
        published = 2.0 * ROUNDING * (np.linalg.norm(capture.v_out[dofs]) + np.linalg.norm(capture.v_hat[dofs]))
        bound = n * ROUNDING * kappa * np.linalg.norm(expected) + published
        out["leave"] = max(out["leave"], np.linalg.norm(dv - expected) / bound)
    return out


class TestWalkedPlacement(unittest.TestCase):
    """Every walked row of the CPU catalogue's general worlds against its physical ``J``, and the leave in its frame."""

    def test_one_frame_on_the_cpu_catalogue(self):
        worst = {"rows": 0.0, "free_rows": 0.0, "leave": 0.0}
        worlds = free_bodies = 0
        for physics, route in catalogue():
            solver, joint_q, capture = run_route(physics, route, device="cpu", steps=4)
            if not solver.placement.walks:
                continue
            result = walked_frames(solver, joint_q, capture)
            worlds += result["worlds"]
            free_bodies += result["free_bodies"]
            for key, value in worst.items():
                worst[key] = max(value, result[key])
        # FL3.1 measured the walk's rows at 1.68 and its leave at 2.30 of their rounding (n u kappa).
        self.assertGreater(worlds, 0)
        # The free bodies walk too: their rows and leave are held to the same rounding.
        self.assertGreater(free_bodies, 0)
        self.assertLess(worst["rows"], BOUND, worst)
        self.assertLess(worst["free_rows"], BOUND, worst)
        self.assertLess(worst["leave"], BOUND, worst)


if __name__ == "__main__":
    unittest.main()
