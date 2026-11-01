# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The general placement's one writer (FL3.3): rows by the walk, the slips' frame and the leave in one frame.

- **The block's exactness** (FL3_design §1.4, ``exp_block``): on random trees with a weld, a mimic or both and one to
  three slipping drives, the frame then the closure (``F = I + Bh (Ln^T - I) Bh^T``, then the projector of ``F Z_B``'s
  orthonormal basis) gives the decided closed-loop operator ``J_c [A_F^-1 - A_F^-1 J_B^T (J_B A_F^-1 J_B^T)^+ J_B
  A_F^-1] J_c^T`` in float64, and in float32 with the split pivot at the level of a float32 refactor of ``A_F``.
- **The device frame** (:func:`.tree_factor.walk_frame_kernel`): on a tree whose drives saturate, ``F^T F`` equals
  ``I + B C^-1 B^T`` of the float64 walk, and the rows the writer stores give the decided operator ``J A_F^-1 J^T``,
  not the all-``K`` one: without the frame a row on a slipping joint takes ``(I + K) / I`` its impulse.
- **One frame per placement** (general worlds, the CPU catalogue): every walked row against its physical ``J v_hat``
  (``Z_i^T L^T P v_hat`` against the incident velocity its walk summed), and the step's leave against the float64
  walk's.
"""

import unittest
import warnings

import numpy as np
import warp as wp

import newton
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


def frame(B: np.ndarray, K: np.ndarray, own: np.ndarray, pivots: np.ndarray, dtype) -> np.ndarray:
    """The slips' frame ``F = I + Bh (Ln^T - I) Bh^T`` in ``dtype``, as :func:`.tree_factor.walk_frame_kernel` forms it:
    ``B`` the walk's columns of the slipping coordinates, ``K`` their released stiffness, ``own`` each one's coordinate
    and ``pivots`` its pivot without ``K`` (the split pivot: ``C_ii = P_i / (K_i D_i) - sum_ancestors B_ki^2``)."""
    n, s = B.shape
    B = B.astype(dtype)
    C = -(B.T @ B).astype(dtype)
    for i in range(s):
        ancestors = dtype(np.sum(np.delete(B[:, i], own[i]) ** 2))
        P = dtype(pivots[i])
        C[i, i] = dtype(P / (dtype(K[i]) * (P + dtype(K[i])))) - ancestors
    Bh, T = np.linalg.qr(B)
    sign = np.sign(np.diag(T))
    Bh, T = (Bh * sign).astype(dtype), (T * sign[:, None]).astype(dtype)
    N = (T @ np.linalg.solve(C, T.T)).astype(dtype)
    Ln = np.linalg.cholesky((np.eye(s, dtype=dtype) + N).astype(dtype)).astype(dtype)
    return (np.eye(n, dtype=dtype) + Bh @ (Ln.T - np.eye(s, dtype=dtype)) @ Bh.T).astype(dtype)


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
        B = np.stack([w.up(covector=np.eye(n, dtype=dtype)[i])[0] for i in S], axis=1).astype(dtype)
        pivots = w.own_pivots[S].astype(dtype)  # P_i: the pivot before the slipping K is added (split pivot)
        F = frame(B, K[S], np.array(S), pivots, dtype)
        Q = orth((F @ Z_B).astype(dtype))
        Z = (F @ Z_c).astype(dtype)
        Z = (Z - Q @ (Q.T @ Z)).astype(dtype)
        W = (Z.T @ Z).astype(np.float64)
        out[name] = float(np.linalg.norm(W - W_ex) / W_scale)
    return out


class TestBlockPrimitive(unittest.TestCase):
    """Frame, then closure: exp_block (FL3_design §1.4) on random Newton trees."""

    def test_frame_then_closure_is_the_decided_closed_loop_operator(self):
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


class TestSlipFrame(unittest.TestCase):
    """The frame of H7's decided set on the walk: rows and leave of the decided operator."""

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
                slips = int(view.slips.numpy()[0])
                self.assertEqual(slips, 3)
                # F^T F = I + B C^-1 B^T against the float64 walk of the all-K factor at the refresh's coordinates.
                walk = solver.factor.walk(joint_q, 0)
                n = walk.s.shape[0]
                K = solver.factor.aug_row_K.numpy()[:n].astype(np.float64)
                Kd = solver.factor.aug_row_Kd.numpy()[:n].astype(np.float64)
                released = K - Kd
                B = np.stack([walk.up(covector=np.eye(n)[i])[0] for i in range(n)], axis=1)
                C = np.diag(1.0 / released) - B.T @ B
                expected = np.eye(n) + B @ np.linalg.solve(C, B.T)
                basis = view.basis.numpy()[0, :slips, :n].astype(np.float64).T
                scale = np.tril(view.scale.numpy()[0, :slips, :slips].astype(np.float64))
                F = np.eye(n) + basis @ (scale.T - np.eye(slips)) @ basis.T
                self.assertLess(np.linalg.norm(F.T @ F - expected) / np.linalg.norm(expected), 1.0e-4)
                # The rows (the hinges' limit rows): their Gram is the decided operator's. By coordinate the walk's
                # columns are B = Lc^-1 (A = Lc Lc^T, so A^-1 = B^T B), and a stored row is z = F Lc^-1 J^T.
                p = solver.placement
                count = int(p.constraint_count.numpy()[0])
                self.assertGreater(count, 0)
                Z = p.packed_Z.numpy()[0, :count, :n].astype(np.float64)
                J = Z @ np.linalg.inv(F).T @ np.linalg.inv(B).T
                A = np.linalg.inv(B.T @ B)
                W_rows = Z @ Z.T
                W_decided = J @ np.linalg.solve(A - np.diag(released), J.T)
                W_all = J @ np.linalg.solve(A, J.T)
                scale_W = np.linalg.norm(W_decided)
                self.assertLess(np.linalg.norm(W_rows - W_decided) / scale_W, 1.0e-4)
                # Without the frame the rows would hold the all-K operator: (I + K) / I away.
                self.assertGreater(np.linalg.norm(W_all - W_decided) / scale_W, 1.0e-1)


# ---------------------------------------------------------------------------------------------------------------------
# One frame per placement: the walked rows and the leave of the general worlds
# ---------------------------------------------------------------------------------------------------------------------


def walked_frames(solver, joint_q: np.ndarray, capture) -> dict:
    """The general worlds' walked rows against their physical ``J v_hat`` and the step's leave against the float64
    walk's, as ratios to their rounding: ``n u |Z_i| |u_hat|`` (and ``u`` of the incident velocity), and ``n u kappa
    |dv|`` (FL3.1's leave bound) with the publication's rounding of ``v_hat`` and ``v_out``. A free rigid body (its own 6x6 factor until the free
    bodies convert), a tree whose drives slip this step (its rows carry the frame, :class:`TestSlipFrame`) and the
    closure's corrected rows (``test_feather_pgs_preelim``) are left to their own checks."""
    forest, placement, factor = solver.forest, solver.placement, solver.factor
    out = {"rows": 0.0, "leave": 0.0, "worlds": 0}
    sizes = forest.articulation_response_dof_count.numpy()
    offsets = forest.articulation_world_dof_offset.numpy()
    starts = forest.articulation_dof_start.numpy()
    art_world = forest.art_to_world.numpy()
    free = forest.is_free_rigid_host if forest.is_free_rigid is not None else np.zeros(sizes.size, dtype=np.int32)
    slips = factor.walk_view().slips.numpy()
    dense = bool(placement.packed_dense)
    incident = placement.row_incident.numpy().astype(np.float64)
    sides_all = placement.row_sides.numpy()
    counts = placement.constraint_count.numpy()
    first = placement.dense_first.numpy()
    walks = {}

    def walk_of(art):
        if art not in walks:
            walks[art] = factor.walk(joint_q, art)
        return walks[art]

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
                if free[art] or slips[art] or solver.factor.eliminates:
                    skip = True
                    break
                walk = walk_of(art)
                v_hat = capture.v_hat[int(starts[art]) : int(starts[art]) + n]
                # u_hat = L^T P v_hat by the float64 walk: the coordinates its rows read.
                inverse, _ = walk.up(covector=np.eye(n))
                u_hat = np.linalg.solve(inverse.T, v_hat)
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
    # The leave: the published dv against the float64 walk's leave of the step's du (the final rows' Z lambda).
    for art in range(sizes.size):
        world, n = int(art_world[art]), int(sizes[art])
        if world < 0 or n == 0 or free[art] or slips[art] or solver.factor.eliminates:
            continue
        line, count = int(first[world]), int(counts[world])
        du = np.zeros(n)
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
        worst = {"rows": 0.0, "leave": 0.0}
        worlds = 0
        for physics, route in catalogue():
            solver, joint_q, capture = run_route(physics, route, device="cpu", steps=4)
            if not solver.placement.walks:
                continue
            result = walked_frames(solver, joint_q, capture)
            worlds += result["worlds"]
            for key, value in worst.items():
                worst[key] = max(value, result[key])
        # FL3.1 measured the walk's rows at 1.68 and its leave at 2.30 of their rounding (n u kappa).
        self.assertGreater(worlds, 0)
        self.assertLess(worst["rows"], BOUND, worst)
        self.assertLess(worst["leave"], BOUND, worst)


if __name__ == "__main__":
    unittest.main()
