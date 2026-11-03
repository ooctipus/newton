# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The one writer (FL3.3-FL3.5): rows by the walk, the slips' release and the leave in one frame.

- **The block's exactness** (FL3_design §1.4, ``exp_block``, W9's release): on random trees with a weld, a mimic or
  both and one to three slipping drives, the release then the closure (``R``, then the projector of ``R Z_B``'s
  orthonormal basis) gives the decided closed-loop operator ``J_c [A_F^-1 - A_F^-1 J_B^T (J_B A_F^-1 J_B^T)^+ J_B
  A_F^-1] J_c^T`` in float64, and in float32 with the split pivot at the level of a float32 refactor of ``A_F``.
- **The device release** (:func:`.tree_kinetic_state.border_solve`, the predictor's border; W9's verdict): on a tree whose drives saturate,
  ``R^T R`` equals ``I + B C^-1 B^T`` of the float64 walk, ``R`` keeps every row's support (tree-lower), and the rows
  the writer stores give the decided operator ``J A_F^-1 J^T``, not the all-``K`` one: without the release a row on a
  slipping joint takes ``(I + K) / I`` its impulse.
- **One frame per placement** (general worlds, the CPU catalogue): every walked row against its physical ``J v_hat``
  (``Z_i^T L^T P v_hat`` against the incident velocity its walk summed), and the step's leave against the float64
  walk's.
- **The sparse diagonal** (FL3.5, CUDA): its arm's rows of a slipping tree, released along the slips' paths (W9's
  ``R``), give the decided operator, and the keyboard's rows (arm and keys) and its arm's leave hold the same invariant
  on each of its routes.
- **The Jacobi tiers' release** (FL3.7, CUDA): the tiers walk their keyed rows in the kernel, or read the writer's, and
  leave by the release's transpose; on saturating drives their step is the decided operator's, as the general sweep's.
"""

import unittest
import warnings
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.kernels import PGS_CONSTRAINT_TYPE_SPEED
from newton._src.solvers.feather_pgs.tree_factor import TreeWalk, local_frames
from newton.solvers import SolverFeatherPGS
from newton.tests.test_feather_pgs_response_diagonal import _build_mixed_response_model
from newton.tests.test_feather_pgs_walk import BOUND, ROUNDING, Capture, catalogue, run_route

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
    """The slips' release ``R`` in ``dtype`` (an ``n x n`` matrix by coordinate), as the predictor's border
    (:func:`.tree_kinetic_state.border_solve`) forms it on ``walk`` (the all-``K`` walk): the slips ``S`` root-most
    first, each one's column ``b = W e_i`` (the walk of ``e_i`` with the earlier releases applied), then FL3.4's
    recurrence along its path, leaf first, with ``K`` the released stiffness and ``pivots`` each slip's split pivot
    (before its released ``K``)."""
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
    steps = release.step.numpy()[art]
    coordinate = np.ascontiguousarray(steps[..., 0]).view(np.int32)
    b, a, g = (steps[..., k].astype(np.float64) for k in (1, 2, 3))
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


def slipping_hinges_beside_a_body(device):
    """:func:`slipping_hinges` with limits only near their upper sides (one row each at the joint limit activation gap
    of 0.1) and a free box apart in the same world: a small tree beside a body, the local owners' class (case K3)."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    parent, joints = -1, []
    for inertia, (lo, hi) in zip((0.05, 0.01, 0.02), ((-0.2, 0.0), (-0.15, 0.0), (-0.3, 0.05)), strict=True):
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
    builder.add_body(xform=wp.transform(wp.vec3(5.0, 0.0, 0.0), wp.quat_identity()), mass=1.0)
    model = builder.finalize(device=device)
    control = model.control()
    target = control.joint_target_q.numpy()
    target[:3] = (3.0, -3.0, 3.0)
    control.joint_target_q.assign(target)
    return model, control


@unittest.skipUnless(wp.is_cuda_available(), "The local owners are a CUDA placement")
class TestLocalSlipFrame(unittest.TestCase):
    """The local owners (case K3, FL3.6) take the decided operator: the packets' rows walked in the kernel and the one
    writer's rows both carry the release ``R`` of the slipping drives along their paths, and the local leave ``R^T``
    (W9's verdict), so each step equals the general placement's (:class:`TestSlipRelease`'s decided operator)."""

    def test_the_local_kernel_takes_the_decided_operator(self):
        from unittest import mock  # noqa: PLC0415

        model, control = slipping_hinges_beside_a_body("cuda:0")
        steps = {}
        # The packets' seeds walked in the kernel; the one writer's rows the kernel reads (no packets); the general
        # placement.
        routes = (("packets", {}), ("stored", {"producer": "compact_jacobian"}), ("general", {"placement": "general"}))
        for name, pins in routes:
            with mock.patch.object(SolverFeatherPGS, "_route_pins", pins), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                solver = SolverFeatherPGS(
                    model,
                    angular_damping=0.0,
                    pgs_iterations=30,
                    enable_joint_limits=True,
                    joint_limit_activation_gap=0.1,
                )
            state, out = model.state(), model.state()
            solver.step(state, out, control, None, 1.0 / 60.0)
            steps[name] = (solver, out.joint_qd.numpy()[:3].astype(np.float64))
        general = steps["general"][0]
        self.assertEqual(general.routes.case, "K6")
        qd_general = steps["general"][1]
        self.assertGreater(np.linalg.norm(qd_general), 1.0e-3)
        for name, packets in (("packets", True), ("stored", False)):
            with self.subTest(route=name):
                local, qd_local = steps[name]
                self.assertEqual(local.routes.case, "K3")
                self.assertEqual(local.placement.row_packets is not None, packets)
                self.assertTrue(local.placement.walks)
                self.assertEqual(int(local.placement.local_solve_owner.numpy()[0]), 1)
                # Every hinge slips this step: both row sources carry the tree's release.
                self.assertEqual(int(local.factor.walk_view().release.count.numpy()[0]), 3)
                self.assertGreater(int(local.placement.constraint_count.numpy()[0]), 0)
                self.assertLess(np.linalg.norm(qd_local - qd_general) / np.linalg.norm(qd_general), 1.0e-5)


# ---------------------------------------------------------------------------------------------------------------------
# The release (case K1): the decided factor's rows from the all-K walk, each inside its support
# ---------------------------------------------------------------------------------------------------------------------


def release_path(path: np.ndarray, n: int, dof_body: np.ndarray, i: int) -> list:
    """Coordinate ``i``'s path in its tree, leaf first: its body's coordinates from ``i`` down, then each ancestor's
    from its last (``path``: each body's parent and first coordinate)."""
    out, body, c = [], int(dof_body[i]), i
    while body >= 0:
        while c >= int(path[body][1]):
            out.append(c)
            c -= 1
        body = int(path[body][0])
        if body >= 0:
            c = (int(path[body + 1][1]) if body + 1 < path.shape[0] else n) - 1
    return out


def release_rows(release, art: int, n: int, path: np.ndarray, dof_body: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """The release of :func:`.walk_rows.walk_release_kernel` (its device data, by path step) applied to the columns of
    ``Z`` (by tree coordinate), slip by slip as the writer applies it: ``u = z - b s``, ``s += a u``, ``z = g u``."""
    Z = Z.astype(np.float64).copy()
    records = release.step.numpy()[art]
    b, a, g = (records[..., k].astype(np.float64) for k in (1, 2, 3))
    for j in range(int(release.count.numpy()[art])):
        steps = release_path(path, n, dof_body, int(release.dof.numpy()[art, j]))
        s = np.zeros(Z.shape[1])
        for t, c in enumerate(steps):
            u = Z[c] - b[j, t] * s
            s = s + a[j, t] * u
            Z[c] = g[j, t] * u
    return Z


def released(solver, joint_q: np.ndarray, art: int = 0):
    """The solver's release of ``art``'s slipping drives this step (the factor's :attr:`.tree_factor.WalkTrees.release`)
    on its walk, and the float64 walk's columns ``B = L^-1`` (by coordinate), the all-``K`` operator ``A = (B^T B)^-1``
    and the released stiffness of each coordinate."""
    factor, forest = solver.factor, solver.forest
    n = int(forest.articulation_dof_count_host[art])
    release = factor.walk_view().release
    status = int(factor.walk_release_status.numpy()[0])
    walk = factor.walk(joint_q, art)
    B = np.stack([walk.up(covector=np.eye(n)[i])[0] for i in range(n)], axis=1)
    K = factor.aug_row_K.numpy()[:n].astype(np.float64)
    Kd = factor.aug_row_Kd.numpy()[:n].astype(np.float64)
    j0 = int(solver.model.articulation_start.numpy()[art])
    path = factor.walk_path_host[j0 : j0 + int(forest.articulation_joint_end_host[art]) - j0]
    d0 = int(forest.articulation_dof_start_host[art])
    dof_body = factor.walk_view().dof_body.numpy()[d0 : d0 + n]
    return release, status, B, np.linalg.inv(B.T @ B), K - Kd, path, dof_body


class TestSupportSlipRelease(unittest.TestCase):
    """The release of H7's decided set on the all-``K`` walk (case K1's slips): ``R L^-1`` is the decided factor's
    inverse, its Gram the decided operator, and it keeps every column inside its path (the support sink's rows)."""

    def test_release_is_the_decided_factor(self):
        model, control = slipping_hinges("cpu")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            solver = SolverFeatherPGS(model, angular_damping=0.0, pgs_iterations=30, enable_joint_limits=True)
        state, out = model.state(), model.state()
        joint_q = state.joint_q.numpy().astype(np.float64)
        solver.step(state, out, control, None, 1.0 / 60.0)
        release, status, B, A, slip_K, path, dof_body = released(solver, joint_q)
        self.assertEqual(status, 0)
        self.assertEqual(int(release.count.numpy()[0]), 3)
        n = B.shape[0]
        B_F = release_rows(release, 0, n, path, dof_body, B)
        decided = np.linalg.inv(A - np.diag(slip_K))
        scale = np.linalg.norm(decided)
        self.assertLess(np.linalg.norm(B_F.T @ B_F - decided) / scale, 1.0e-5)
        self.assertGreater(np.linalg.norm(B.T @ B - decided) / scale, 1.0e-1)
        # The release keeps each column inside its path: R L^-1 is the decided factor's L_F^-1, of L's pattern.
        self.assertTrue(np.all(B_F[np.abs(B) == 0.0] == 0.0))
        # The step's one leave takes the release's transpose (tree_factor.walk_leave_kernel): dv = L^-T R^T du, the
        # decided factor's leave, against the float64 walk of R^T du.
        du = np.random.default_rng(5).normal(size=(1, n)).astype(np.float32)
        walk_view = solver.factor.walk_view()
        d0 = int(solver.forest.articulation_dof_start_host[0])
        values = np.zeros(walk_view.du.shape[0], dtype=np.float32)
        values[d0 : d0 + n] = du[0]
        walk_view.du.assign(values)
        v_out = wp.zeros(walk_view.du.shape[0], dtype=float, device="cpu")
        solver.factor.leave(v_out)
        v_out = v_out.numpy()[d0 : d0 + n]
        R = release_rows(release, 0, n, path, dof_body, np.eye(n))
        walk = solver.factor.walk(joint_q, 0)
        expected = walk.down(R.T @ du[0].astype(np.float64))[0]
        self.assertLess(np.linalg.norm(v_out - expected) / np.linalg.norm(expected), 1.0e-5)


def _hinges_step(pins: dict, iterations: int = 30, **options) -> tuple:
    """One step of :func:`slipping_hinges` on CUDA under route ``pins``: the solver and its published velocity."""
    model, control = slipping_hinges("cuda:0")
    with patch.dict(SolverFeatherPGS._route_pins, pins), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver = SolverFeatherPGS(
            model, angular_damping=0.0, pgs_iterations=iterations, enable_joint_limits=True, **options
        )
    state, out = model.state(), model.state()
    solver.step(state, out, control, None, 1.0 / 60.0)
    return solver, out.joint_qd.numpy().astype(np.float64)


@unittest.skipUnless(wp.is_cuda_available(), "the Jacobi tiers run on CUDA")
class TestTierSlipFrame(unittest.TestCase):
    """The Jacobi tiers on the walk (FL3.7): their rows and leave carry the release ``R`` of the decided operator (the
    keyed rows walked in the kernel and the one writer's rows take ``R``, the leave ``R^T``), so their step is the
    decided operator's."""

    def test_tiers_take_the_decided_operator(self):
        # The general sweep's step on the walk, whose rows and leave take the release (TestSlipRelease).
        general, reference = _hinges_step({"tiers": 0})
        self.assertFalse(general.sweep.tiers)
        self.assertEqual(int(general.factor.walk_view().release.count.numpy()[0]), 3)
        scale = float(np.linalg.norm(reference))
        for producer in (None, "general"):
            # The conformance routes' pins: the tiers on any class whose rows they hold (keyed_parallel, mf_parallel).
            pins = {"tiers": 32, **({"producer": producer} if producer else {})}
            with self.subTest(producer=producer or "keyed"):
                solver, qd = _hinges_step(pins)
                self.assertTrue(solver.sweep.tiers)
                self.assertEqual(solver.placement.keyed_rows is not None, producer is None)
                # The keyed rows and the writer's rows release their slips along their paths (R).
                self.assertEqual(int(solver.factor.walk_view().release.count.numpy()[0]), 3)
                self.assertLess(np.linalg.norm(qd - reference) / scale, 1.0e-4, f"{qd} {reference}")

    def test_keyed_general_worlds_take_the_decided_operator(self):
        """A keyed class's world of more rows than its tier holds goes to the general sweep through the one writer:
        its rows carry the release ``R`` as the tiers' do, so its step is the decided operator's (the all-K factor's
        rows missed it while the predictor took the decided step)."""
        general, reference = _hinges_step({"tiers": 0}, enable_joint_velocity_limits=True)
        scale = float(np.linalg.norm(reference))
        solver, qd = _hinges_step({"tiers": 4}, enable_joint_velocity_limits=True)
        self.assertIsNotNone(solver.placement.keyed_rows)
        self.assertGreater(
            int(solver.placement.constraint_count.numpy()[0]), solver.placement.keyed_rows.layout.parallel_rows
        )
        self.assertEqual(int(solver.factor.walk_view().release.count.numpy()[0]), 3)
        self.assertLess(np.linalg.norm(qd - reference) / scale, 1.0e-4, f"{qd} {reference}")


# ---------------------------------------------------------------------------------------------------------------------
# One frame per placement: the walked rows and the leave of the general worlds
# ---------------------------------------------------------------------------------------------------------------------


def walked_frames(solver, joint_q: np.ndarray, capture) -> dict:
    """The general worlds' walked rows, the dense rows and the free bodies' (the bodies sink), against their physical
    ``J v_hat`` and the step's leave against the float64 walk's, as ratios to their rounding: ``n u |Z_i| |u_hat|``
    (and ``u`` of the incident velocity), and ``n u kappa |dv|`` (FL3.1's leave bound) with the publication's rounding
    of ``v_hat`` and ``v_out``. A free rigid body is a one-body tree of six coordinates. A tree whose drives slip this
    step leaves ``R^T du`` (its rows carry the release and are checked by :class:`TestSlipRelease`); the closure's
    corrected rows (``test_feather_pgs_preelim``), a world whose speed ball acted (its ``du`` is a law's, not
    ``sum Z lambda``) and a world the local owners' packets hold (its rows are walked in the kernel, no store holds
    them: ``test_feather_pgs_walk_device.placement_frames``) are left to their own checks."""
    forest, placement, factor = solver.forest, solver.placement, solver.factor
    owned = np.zeros(int(forest.world_count), dtype=bool)
    if placement.row_packets is not None:
        owned = placement.local_solve_owner.numpy() != 0
    out = {"rows": 0.0, "free_rows": 0.0, "leave": 0.0, "worlds": 0, "free_bodies": 0, "slipping": 0}
    sizes = forest.articulation_response_dof_count.numpy()
    offsets = forest.articulation_world_dof_offset.numpy()
    starts = forest.articulation_dof_start.numpy()
    art_world = forest.art_to_world.numpy()
    free = forest.is_free_rigid_host if forest.is_free_rigid is not None else np.zeros(sizes.size, dtype=np.int32)
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
        if count == 0 or owned[world]:
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
        Z_a, Z_b, lam, mf_counts, side_a, side_b, _dof_a, _dof_b, kinds = capture.mf_final
        mf_incident = placement.row_incident.numpy()[:, placement.body_base :].astype(np.float64)
        for world, count in enumerate(mf_counts):
            line = int(capture.mf_first[world])
            for row in range(int(count)):
                if kinds[line, row] == PGS_CONSTRAINT_TYPE_SPEED:
                    ball[world] |= lam[line, row] != 0.0
                    continue
                term, scale_z, scale_u = 0.0, 0.0, 0.0
                for side, Z in ((side_a, Z_a), (side_b, Z_b)):
                    if side[line, row] < 0:
                        continue
                    art = int(side[line, row])
                    z, u_hat = Z[line, row], u_hat_of(art)
                    term += z @ u_hat
                    scale_z, scale_u = scale_z + z @ z, scale_u + u_hat @ u_hat
                    free_du[art] = free_du.get(art, 0.0) + z * lam[line, row]
                incident_row = mf_incident[line, row]
                bound = 6 * ROUNDING * np.sqrt(scale_z * scale_u) + ROUNDING * abs(incident_row)
                if bound > 0.0:
                    out["free_rows"] = max(out["free_rows"], abs(term - incident_row) / bound)
    # The leave: the published dv against the float64 walk's leave of the step's du (the final rows' Z lambda), a
    # slipping tree's after its rows' release, R^T du (the device's release, the predictor's border).
    view = factor.walk_view()
    for art in range(sizes.size):
        world, n = int(art_world[art]), int(sizes[art])
        if world < 0 or n == 0 or solver.factor.eliminates or ball[world] or owned[world]:
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
        out["free_bodies"] += int(free[art] != 0)
        if slips[art]:
            # R^T du: the last slip first, each along its path root first (the step's one leave, walk_leave_kernel).
            release = view.release
            steps = release.step.numpy()[art]
            coordinate = np.ascontiguousarray(steps[..., 0]).view(np.int32)
            b, a, g = (steps[..., k] for k in (1, 2, 3))
            du = du.copy()
            for j in reversed(range(int(slips[art]))):
                t = 0.0
                for k in reversed(range(coordinate.shape[1])):
                    c = int(coordinate[j, k])
                    if c < 0:
                        continue
                    x = float(g[j, k]) * du[c] + float(a[j, k]) * t
                    du[c] = x
                    t -= float(b[j, k]) * x
            out["slipping"] += 1
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


def slot_frames(solver, joint_q: np.ndarray, capture) -> dict:
    """The sparse diagonal's walked rows (FL3.5) against their physical ``J v_hat`` and its arms' leave against the
    float64 walk's, as :func:`walked_frames` reads the general worlds': a row is its arm's dense part (the walk's
    coordinates, ``u_hat = L^T P v_hat`` by the float64 walk) and its keys' slots (``u_hat = v_hat / s``, ``s =
    m^-1/2``). An arm whose drives slip this step is left to :class:`TestSlotFrame`."""
    forest, placement, factor = solver.forest, solver.placement, solver.factor
    out = {"rows": 0.0, "leave": 0.0, "worlds": 0}
    n = int(placement.sparse_diagonal_dense_size)
    Z = placement.J_by_size[n].numpy().astype(np.float64)
    slot_dof = placement.sparse_diagonal_row_dof.numpy()
    slot_z = placement.sparse_diagonal_row_z.numpy().astype(np.float64)
    groups = placement.sparse_diagonal_dense_groups.numpy()
    arts = placement.sparse_diagonal_dense_arts_host
    world_dofs = forest.world_dof_indices.numpy()
    inverse_mass = factor.diagonal_inverse_mass.numpy().astype(np.float64)
    starts = forest.articulation_dof_start.numpy()
    slips = factor.walk_view().release.count.numpy()
    incident = placement.row_incident.numpy().astype(np.float64)
    first = placement.dense_first.numpy()
    for world in range(int(forest.world_count)):
        art, group, line, count = int(arts[world]), int(groups[world]), int(first[world]), int(capture.counts[world])
        if count == 0 or slips[art]:
            continue
        walk = factor.walk(joint_q, art)
        inverse, _ = walk.up(covector=np.eye(n))
        dofs = int(starts[art]) + np.arange(n)
        u_hat = np.linalg.solve(inverse.T, capture.v_hat[dofs])
        du = np.zeros(n)
        for row in range(count):
            dense = Z[group, row, :n]
            term, scale_z, scale_u = dense @ u_hat, dense @ dense, u_hat @ u_hat
            for s in range(2):
                coordinate = int(slot_dof[line, row, s])
                if coordinate < 0:
                    continue  # no key, or an independent chain's link
                dof = int(world_dofs[world, coordinate])
                u = capture.v_hat[dof] / np.sqrt(inverse_mass[dof])
                z = slot_z[line, row, s]
                term, scale_z, scale_u = term + z * u, scale_z + z * z, scale_u + u * u
            bound = n * ROUNDING * np.sqrt(scale_z * scale_u) + ROUNDING * abs(incident[line, row])
            if bound > 0.0:
                out["rows"] = max(out["rows"], abs(term - incident[line, row]) / bound)
            du += dense * capture.impulses[line, row]
        out["worlds"] += 1
        if not np.any(du):
            continue
        # The arm's leave: its published dv against the float64 walk's leave of du.
        dv = capture.v_out[dofs] - capture.v_hat[dofs]
        expected = walk.down(du)[0]
        kappa = float(np.linalg.cond(inverse))
        published = 2.0 * ROUNDING * (np.linalg.norm(capture.v_out[dofs]) + np.linalg.norm(capture.v_hat[dofs]))
        bound = n * ROUNDING * kappa * np.linalg.norm(expected) + published
        out["leave"] = max(out["leave"], np.linalg.norm(dv - expected) / bound)
    return out


def star_beside_hinges(device):
    """Case K2 (a tree beside a star): four limited prismatic keys on a fixed frame, and :func:`slipping_hinges`' three
    hinges, each held past its effort limit, as the dense arm."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    frame = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * 0.01))
    keys = [builder.add_joint_fixed(-1, frame)]
    for k in range(4):
        key = builder.add_link(mass=0.01, inertia=wp.mat33(np.eye(3) * 1.0e-5))
        keys.append(
            builder.add_joint_prismatic(
                frame,
                key,
                axis=(0.0, 0.0, 1.0),
                parent_xform=wp.transform(wp.vec3(0.03 * k, 1.0, 0.0), wp.quat_identity()),
                limit_lower=-0.008,
                limit_upper=0.002,
            )
        )
    builder.add_articulation(keys)
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
    targets = np.zeros(model.joint_dof_count, dtype=np.float32)
    targets[-3:] = (3.0, -3.0, 3.0)
    control.joint_target_q.assign(targets)
    return model, control


@unittest.skipUnless(wp.is_cuda_available(), "the sparse diagonal is a CUDA placement")
class TestSlotFrame(unittest.TestCase):
    """The sparse diagonal on the walk (FL3.5): the arm's rows of H7's decided set, the keyboard's rows and leave."""

    def test_arm_rows_take_the_decided_operator(self):
        """The arm's limit rows, released along the slips' paths (W9's ``R``), give the decided operator's Gram
        ``J A_F^-1 J^T``; without the release they would give the all-``K`` one."""
        model, control = star_beside_hinges("cuda:0")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            solver = SolverFeatherPGS(model, angular_damping=0.0, pgs_iterations=30, enable_joint_limits=True)
        self.assertEqual(solver.routes.case, "K2")
        self.assertTrue(solver.placement.walks)
        state, out = model.state(), model.state()
        joint_q = state.joint_q.numpy().astype(np.float64)
        solver.step(state, out, control, None, 1.0 / 60.0)
        art = int(solver.placement.sparse_diagonal_dense_arts_host[0])
        self.assertEqual(int(solver.factor.walk_view().release.count.numpy()[art]), 3)
        walk = solver.factor.walk(joint_q, art)
        n = walk.s.shape[0]
        start = int(solver.forest.articulation_dof_start.numpy()[art])
        drive_rows = solver.factor.drive_row_by_dof.numpy()[start + np.arange(n)]
        K = solver.factor.aug_row_K.numpy()[drive_rows].astype(np.float64)
        Kd = solver.factor.aug_row_Kd.numpy()[drive_rows].astype(np.float64)
        released = K - Kd
        # The rows' physical J from their unit sources: each a joint limit, +-e_dof.
        p = solver.placement
        count = int(p.constraint_count.numpy()[0])
        self.assertGreater(count, 0)
        line = int(p.dense_first.numpy()[0])
        units = p.unit_source.numpy()[line, :count]
        J = np.zeros((count, n))
        for row, source in enumerate(units):
            self.assertEqual(source >> 28, 2)  # UNIT_LIMIT
            index = source & ((1 << 28) - 1)
            J[row, index // 2 - start] = 1.0 - 2.0 * (index % 2)
        group = int(p.sparse_diagonal_dense_groups.numpy()[0])
        Z = p.J_by_size[n].numpy()[group, :count, :n].astype(np.float64)
        B = np.stack([walk.up(covector=np.eye(n)[i])[0] for i in range(n)], axis=1)
        A = np.linalg.inv(B.T @ B)
        W_rows = Z @ Z.T
        W_decided = J @ np.linalg.solve(A - np.diag(released), J.T)
        W_all = J @ np.linalg.solve(A, J.T)
        scale_W = np.linalg.norm(W_decided)
        self.assertLess(np.linalg.norm(W_rows - W_decided) / scale_W, 1.0e-5)
        self.assertGreater(np.linalg.norm(W_all - W_decided) / scale_W, 1.0e-1)

    def test_one_frame_on_the_keyboard(self):
        """The keyboard on each sparse-diagonal route: its rows against ``J v_hat`` and its arms' leave against the
        float64 walk's, within ``BOUND`` of their rounding."""
        from newton.tests import test_feather_pgs_conformance as conformance  # noqa: PLC0415

        names = {route.name for route in (conformance.SPARSE_DIAGONAL, conformance.SPARSE_DIAGONAL_TREE)}
        names.add(conformance.COMPACT_BOUNDARY.name)
        done = set()
        for group in conformance.cuda_groups():
            for route in (group.reference, *group.routes):
                if route.name not in names or route.name in done:
                    continue
                with self.subTest(problem=group.physics.name, route=route.name):
                    solver, joint_q, capture = run_route(group.physics, route, device="cuda:0")
                    self.assertTrue(solver.placement.walks)
                    result = slot_frames(solver, joint_q, capture)
                    self.assertGreater(result["worlds"], 0)
                    self.assertLess(result["rows"], BOUND, result)
                    self.assertLess(result["leave"], BOUND, result)
                    done.add(route.name)
        self.assertEqual(done, names)


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


def paired_arm_and_box(device, drives: bool):
    """Two worlds of a 23-DOF arm and a free box sliding down onto it (case K5 on CUDA: the paired factor sweep), with
    ``drives`` stiff PD drives on the arm held past their effort limits, so they slip."""
    model = _build_mixed_response_model(device, world_count=2, dof_count=23, friction=0.7)
    qd = model.joint_qd.numpy()
    q_start = model.joint_qd_start.numpy()
    free = model.joint_type.numpy() == int(newton.JointType.FREE)
    for joint in np.flatnonzero(free):
        qd[q_start[joint] : q_start[joint] + 6] = (0.3, 0.0, -1.0, 0.0, 0.0, 0.0)
    model.joint_qd.assign(qd)
    control = model.control()
    if drives:
        arm = np.flatnonzero(~np.repeat(free, np.diff(q_start)))
        for name, value in (("joint_target_ke", 50.0), ("joint_target_kd", 1.0), ("joint_effort_limit", 1.0e-3)):
            values = getattr(model, name).numpy()
            values[arm] = value
            getattr(model, name).assign(values)
        targets = control.joint_target_q.numpy()
        targets[arm] = 1.5
        control.joint_target_q.assign(targets)
    return model, control


class TestPairedPlacement(unittest.TestCase):
    """Case K5 walks (FL3.3 stage two C): its rows by the one writer, the paired factor sweep's leave by the walk."""

    @unittest.skipUnless(wp.is_cuda_available(), "The paired factor sweep requires CUDA")
    def test_one_frame_on_the_paired_class(self):
        for drives in (False, True):
            with self.subTest(drives=drives):
                model, control = paired_arm_and_box("cuda:0", drives)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    solver = SolverFeatherPGS(
                        model,
                        pgs_iterations=16,
                        enable_joint_limits=True,
                        dense_max_constraints=96,
                        mf_max_constraints=32,
                    )
                self.assertEqual(solver.routes.case, "K5")
                self.assertTrue(solver.placement.walks)
                self.assertIsNotNone(solver.sweep.paired_factor_kernel)
                capture = Capture(solver)
                pipeline = newton.CollisionPipeline(model, broad_phase="nxn", reduce_contacts=False, deterministic=True)
                contacts = pipeline.contacts()
                state, out = model.state(), model.state()
                newton.eval_fk(model, state.joint_q, state.joint_qd, state)
                worst = {"rows": 0.0, "free_rows": 0.0, "leave": 0.0}
                seen = {"free_bodies": 0, "slipping": 0, "rows": 0}
                for _ in range(6):
                    state.clear_forces()
                    pipeline.collide(state, contacts)
                    solver.step(state, out, control, contacts, 1.0 / 240.0)
                    joint_q = solver.cut.mirror(state).joint_q.numpy().astype(np.float64)
                    capture.finish()
                    seen["rows"] += int(solver.placement.constraint_count.numpy().sum())
                    result = walked_frames(solver, joint_q, capture)
                    for key in worst:
                        worst[key] = max(worst[key], result[key])
                    for key in ("free_bodies", "slipping"):
                        seen[key] += result[key]
                    state, out = out, state
                self.assertGreater(seen["rows"], 0, "no row reached the paired sweep")
                self.assertGreater(seen["free_bodies"], 0)
                if drives:
                    self.assertGreater(seen["slipping"], 0, "no drive slipped")
                self.assertLess(worst["rows"], BOUND, worst)
                self.assertLess(worst["free_rows"], BOUND, worst)
                self.assertLess(worst["leave"], BOUND, worst)


# ---------------------------------------------------------------------------------------------------------------------
# The support placement (case K1, CUDA): its rows from the support sink, one frame with its leave, the release
# ---------------------------------------------------------------------------------------------------------------------


def support_rows(solver) -> dict:
    """Case K1's rows from its support sink, by ``(world, row)``: each side's footprints by tree coordinate
    ``{art: z}``, read through the support table or the explicit coordinates (factor coordinates, the plan's
    ``permutation`` back to the tree's)."""
    forest, placement, factor = solver.forest, solver.placement, solver.factor
    perm = np.asarray(factor.sparse_pattern.permutation)
    size = int(placement.sparse_mass_matrix_size)
    Z = placement.sparse_row_factor.numpy().astype(np.float64)
    table = bool(placement.sparse_row_support.enabled)
    if table:
        row_class = placement.sparse_row_support.row_class.numpy()
        class_dof = placement.sparse_row_support.class_dof.numpy()
    else:
        row_dof = placement.sparse_row_dof.numpy()
    offsets = forest.articulation_world_dof_offset.numpy()
    art_world = forest.art_to_world.numpy()
    arts = forest.group_to_art_host[size]
    counts, first = placement.constraint_count.numpy(), placement.dense_first.numpy()
    out = {}
    for world in range(int(forest.world_count)):
        line = int(first[world])
        mine = [int(a) for a in arts if art_world[a] == world]
        for row in range(min(int(counts[world]), Z.shape[1])):
            sides = {}
            for e in range(Z.shape[2]):
                coordinate = int(class_dof[row_class[line, row], e] if table else row_dof[line, row, e])
                if coordinate < 0:
                    continue
                for art in mine:
                    local = coordinate - int(offsets[art])
                    if 0 <= local < size:
                        sides.setdefault(art, np.zeros(size))[perm[local]] = Z[line, row, e]
            out[(world, row)] = sides
    return out


def support_frames(solver, joint_q: np.ndarray) -> dict:
    """Case K1's rows against their physical ``J v_hat`` and its leave against the float64 walk's, as
    :func:`walked_frames` reads the general placement's. A slipping tree's rows are the release ``R`` of the walk's
    (the predictor's border, its device data): ``z . R^-T u_hat`` is ``J v_hat`` and the leave is the
    walk down of ``R^T du``."""
    forest, placement, factor = solver.forest, solver.placement, solver.factor
    size = int(placement.sparse_mass_matrix_size)
    starts = forest.articulation_dof_start.numpy()
    joint_start = solver.model.articulation_start.numpy()
    release = solver.factor.walk_view().release
    incident = placement.row_incident.numpy().astype(np.float64)
    impulses = placement.impulses.numpy().astype(np.float64)
    first = placement.dense_first.numpy()
    dof_body = factor.walk_view().dof_body.numpy()
    v_hat, v_out = solver.v_hat.numpy().astype(np.float64), solver.v_out.numpy().astype(np.float64)
    trees, out = {}, {"rows": 0.0, "leave": 0.0, "rows_checked": 0, "trees": 0, "rows_seen": 0, "slipping": 0}
    du = {}

    def tree_of(art):
        """The float64 walk at the step's coordinates, its ``u_hat = L^T P v_hat`` and the release ``R``."""
        if art not in trees:
            walk = factor.walk(joint_q, art)
            inverse, _ = walk.up(covector=np.eye(size))
            u_hat = np.linalg.solve(inverse.T, v_hat[int(starts[art]) : int(starts[art]) + size])
            j0, d0 = int(joint_start[art]), int(starts[art])
            path = factor.walk_path_host[j0 : j0 + int(forest.articulation_joint_end_host[art]) - j0]
            R = release_rows(release, art, size, path, dof_body[d0 : d0 + size], np.eye(size))
            trees[art] = (walk, u_hat, R)
            out["slipping"] += int(release.count.numpy()[art] > 0)
        return trees[art]

    for (world, row), sides in support_rows(solver).items():
        line = int(first[world])
        out["rows_seen"] += 1
        term, zz, uu = 0.0, 0.0, 0.0
        for art, z in sides.items():
            _, u_hat, R = tree_of(art)
            u = np.linalg.solve(R.T, u_hat)
            term += z @ u
            zz, uu = zz + z @ z, uu + u @ u
            du[art] = du.get(art, np.zeros(size)) + z * impulses[line, row]
        bound = size * ROUNDING * np.sqrt(zz * uu) + ROUNDING * abs(incident[line, row])
        if sides and bound > 0.0:
            out["rows"] = max(out["rows"], abs(term - incident[line, row]) / bound)
            out["rows_checked"] += 1
    for art, delta in du.items():
        if not np.any(delta):
            continue
        walk, _, R = tree_of(art)
        dofs = int(starts[art]) + np.arange(size)
        expected = walk.down(R.T @ delta)[0]
        inverse, _ = walk.up(covector=np.eye(size))
        kappa = float(np.linalg.cond(inverse))
        published = 2.0 * ROUNDING * (np.linalg.norm(v_out[dofs]) + np.linalg.norm(v_hat[dofs]))
        bound = size * ROUNDING * kappa * np.linalg.norm(expected) + published
        out["leave"] = max(out["leave"], np.linalg.norm(v_out[dofs] - v_hat[dofs] - expected) / bound)
        out["trees"] += 1
    return out


@unittest.skipUnless(wp.is_cuda_available(), "native sparse factors (case K1) run on CUDA")
class TestSupportPlacement(unittest.TestCase):
    """Case K1 on the walk (FL3.4): the support sink's rows and the walk's leave in one frame, and a slipping drive's
    release on its rows."""

    def test_one_frame_on_the_support_placement(self):
        from newton.tests import feather_pgs_sparse_scene as scene  # noqa: PLC0415

        worst, rows, trees, seen = {"rows": 0.0, "leave": 0.0}, 0, 0, []
        # Every step refreshes, so the float64 walk at the step's coordinates is the device's factor.
        for shape, options in (
            ({}, {"update_mass_matrix_interval": 1}),
            ({"foot": "box"}, {"friction_anchor_beta": 0.2, "update_mass_matrix_interval": 1}),
            ({"floating": False, "depth": 3}, {"update_mass_matrix_interval": 1}),
        ):
            with self.subTest(shape=shape, options=options):
                case = scene.make_case(True, options=options, **shape)
                solver = case[1]
                self.assertEqual(solver.routes.case, "K1")
                # Every step: a bouncing foot holds rows on some steps only.
                for _ in range(6):
                    joint_q = solver.cut.mirror(case[2]).joint_q.numpy().astype(np.float64)
                    scene.step(case)
                    result = support_frames(solver, joint_q)
                    rows, trees = rows + result["rows_checked"], trees + result["trees"]
                    seen.append((result["rows_seen"], result["slipping"]))
                    for key, value in worst.items():
                        worst[key] = max(value, result[key])
        self.assertGreater(rows, 0, seen)
        self.assertGreater(trees, 0, seen)
        self.assertLess(worst["rows"], BOUND, worst)
        self.assertLess(worst["leave"], BOUND, worst)

    def test_rows_take_the_decided_operator(self):
        model, control = slipping_hinges("cuda:0")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Velocity rows admit native sparse factors on a chain.
            solver = SolverFeatherPGS(
                model,
                angular_damping=0.0,
                pgs_iterations=30,
                enable_joint_limits=True,
                enable_joint_velocity_limits=True,
            )
        self.assertEqual(solver.routes.case, "K1")
        state, out = model.state(), model.state()
        joint_q = state.joint_q.numpy().astype(np.float64)
        solver.step(state, out, control, None, 1.0 / 60.0)
        release = solver.factor.walk_view().release
        self.assertEqual(int(release.count.numpy()[0]), 3)
        walk = solver.factor.walk(joint_q, 0)
        n = walk.s.shape[0]
        B = np.stack([walk.up(covector=np.eye(n)[i])[0] for i in range(n)], axis=1)
        K = solver.factor.aug_row_K.numpy()[:n].astype(np.float64)
        Kd = solver.factor.aug_row_Kd.numpy()[:n].astype(np.float64)
        path = solver.factor.walk_path_host[:3]
        dof_body = solver.factor.walk_view().dof_body.numpy()[:n]
        R = release_rows(release, 0, n, path, dof_body, np.eye(n))
        sides = [z for (_, _), z in sorted(support_rows(solver).items())]
        Z = np.array([z[0] for z in sides if 0 in z])
        self.assertGreater(Z.shape[0], 0)
        # A stored row is z = R B J^T: J = Z R^-T B^-T; its Gram is the decided operator's, not the all-K one's.
        J = Z @ np.linalg.inv(R).T @ np.linalg.inv(B).T
        A = np.linalg.inv(B.T @ B)
        W_rows = Z @ Z.T
        W_decided = J @ np.linalg.solve(A - np.diag(K - Kd), J.T)
        W_all = J @ np.linalg.solve(A, J.T)
        scale_W = np.linalg.norm(W_decided)
        self.assertLess(np.linalg.norm(W_rows - W_decided) / scale_W, 1.0e-4)
        self.assertGreater(np.linalg.norm(W_all - W_decided) / scale_W, 1.0e-1)


if __name__ == "__main__":
    unittest.main()
