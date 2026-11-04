# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The walk on the device (FL3.1): the tree factor's walk as the native functions FL3's one writer will call.

:func:`.factor.walk.walk_prelude` declares the walk for both devices: the class's path table in constant memory, a lane
per seed walking its rows up (``WALK_UP``, ``WALK_SEED``) into the caller's sink, with the step's ``J`` entry each
footprint whitens; :func:`.factor.walk.walk_down_prelude` emits a class's walk down (the leave, ``WALK_DOWN``), a
lane per tree, straight-line up to ``WALK_UNROLL`` coordinates and the loop form past it. These tests run them on
the CPU and, where there is one, on a CUDA device, against the float64 walk (:class:`.factor.walk.TreeWalk`, which
:mod:`.test_feather_pgs_walk` holds to the factor), and hold every placement to one frame:

- **F1, the device walk**, on every catalogue model: rows of every unit (1 to 4 rows a lane) from wrench and covector
  seeds, their ``J`` entries (a row's incident is their sum against ``v_hat``), the leave and its twists, within
  ``BOUND n u kappa`` of the float64 walk;
- **F2 on the device**: ``long_chain(N)`` without armature, N = 8 to 1,000: a tip impulse's response, rows up and leave
  down in float32, against the float64 truth;
- **one frame per placement**: whitening and the leave of ``du`` share one factorization, so ``|J dv - Z^T du|`` stays
  at rounding: within ``n u kappa`` of ``|Z| |du|``, and as measured within ``n u``. The device walk's rows and leave
  hold the tighter bound; on ``long_chain(106)`` a leave by main's factor under the walk's rows breaks it, so the test
  tells one frame from two. Every placement's rows and leave are held to the walk's on every CPU catalogue route and
  on the CUDA catalogue's routes (:func:`placement_frames`);
- **F1 on CUDA**: Z0's F1 on the CUDA placements: the one writer's rows and leave, and the row packets' worlds (FL3.6)
  through the walk's response of their impulses to the packets' seeds. Native sparse factors, the compact boundary
  and the keyed rows store no row this check reads: :data:`UNSTORED` names their routes.

``python -m newton.tests.test_feather_pgs_walk_device report`` prints the measured tables.
"""

import functools
import hashlib
import sys
import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import tree_scan
from newton._src.solvers.feather_pgs.device_preludes import DevicePrelude, substitute
from newton._src.solvers.feather_pgs.factor import walk as factor_walk
from newton._src.solvers.feather_pgs.factor.walk import (
    WALK_COORDINATE,
    WALK_FRAME,
    WALK_LINK,
    WALK_NOW,
    TreeWalk,
    local_frames,
    unpack_walk_records,
)
from newton._src.solvers.feather_pgs.kernel_template import THREADS, KernelTemplate, params
from newton._src.solvers.feather_pgs.row_kernels import PGS_CONSTRAINT_TYPE_SPEED
from newton._src.solvers.feather_pgs.sweep import PGS_LOCAL_SOLVE_OWNER_GENERAL
from newton.solvers import SolverFeatherPGS
from newton.tests import feather_pgs_scenes as scenes
from newton.tests import test_feather_pgs_conformance as conformance
from newton.tests.feather_pgs_law_harness import dense_factor32
from newton.tests.feather_pgs_mass_reference import MassReference
from newton.tests.test_feather_pgs_walk import (
    BOUND,
    CHAINS,
    ROUNDING,
    catalogue,
    contact_wrenches,
    factor_matrix,
    ground_contact,
    jacobian,
    newton_state_body_q,
    relative,
    run_route,
)
from newton.tests.test_feather_pgs_walk_rows import random_tree

UNITS = (1, 2, 3, 4)
"""Rows a lane walks: a limit, a velocity limit's pair, a point contact, a contact with spin."""
UNSTORED = ("sparse", "sparse_triplets", "keyed_parallel")
"""The CUDA catalogue's routes whose rows no store holds as ``J``, packed ``Z`` or the walk's slots: native sparse
factors (rows in factor coordinates, whitened from a ``J`` in shared scratch) and the keyed rows;
:func:`placement_frames` also counts the row packets' worlds as unstored."""


def devices() -> list:
    return ["cpu", *(["cuda:0"] if wp.is_cuda_available() else [])]


# ---------------------------------------------------------------------------------------------------------------------
# The device walk of one topology
# ---------------------------------------------------------------------------------------------------------------------

_UP = """    if (tid >= body.shape[0]) return;
    const int n = z.shape[2], bodies = link.shape[1];
    const float* d = data.data + (size_t)tree.data[tid] * data.shape[1] * $coordinate;
    const float* l = link.data + (size_t)tree.data[tid] * bodies * $link;
    const float* s = now.data + (size_t)tree.data[tid] * now.shape[1] * $now;
    const float* f = now_link.data + (size_t)tree.data[tid] * bodies * $link;
    float* out = z.data + (size_t)tid * $rows * n;
    float* entries = jz.data + (size_t)tid * $rows * n;
    const auto sink = [&](int r, int c, float value, float j) {
        out[(size_t)r * n + c] = value;
        entries[(size_t)r * n + c] = j;
    };
    float p[6 * $rows], w[6 * $rows];
    for (int k = 0; k < 6 * $rows; ++k) p[k] = w[k] = wrench.data[(size_t)tid * 6 * $rows + k];
    int start = top.data[tid];
    if (seeded.data[tid] != 0) {
        float x[$rows];
        for (int r = 0; r < $rows; ++r) x[r] = covector.data[(size_t)tid * $rows + r];
        WALK_SEED(d, start, x, p, sink);
        start -= 1;
        s = nullptr;
    }
    // The fixture's frames are one geometry's: no origin shift between the factor's and the step's.
    const float shift[3] = {0.0f, 0.0f, 0.0f};
    WALK_UP(d, l, s, f, shift, body.data[tid], start, p, w, sink);
"""

_TREE = """    if (tid >= u.shape[0]) return;
    const int n = u.shape[1], bodies = link.shape[1];
    const float* d = data.data + (size_t)tree.data[tid] * data.shape[1] * $coordinate;
    const float* l = link.data + (size_t)tree.data[tid] * bodies * $link;
    const float* in = u.data + (size_t)tid * n;
    float* out = qd.data + (size_t)tid * n;
    float* twists = twist.data + (size_t)tid * bodies * 6;
    WALK_DOWN(
        0, d, l, [&](const int c) -> float { return in[c]; }, [](const int) -> float { return 0.0f; },
        [&](const int c, const float q, const float) { out[c] = q; },
        [&](const int b, const float* V) {
            for (int k = 0; k < 6; ++k) twists[6 * b + k] = V[k];
        });
"""

_UP_PARAMS = params(
    ("data", wp.array3d[float]),
    ("link", wp.array3d[float]),
    ("now", wp.array3d[float]),
    ("now_link", wp.array3d[float]),
    ("tree", wp.array[int]),
    ("body", wp.array[int]),
    ("top", wp.array[int]),
    ("seeded", wp.array[int]),
    ("wrench", wp.array3d[float]),
    ("covector", wp.array2d[float]),
    ("z", wp.array3d[float]),
    ("jz", wp.array3d[float]),
)
_TREE_PARAMS = params(
    ("data", wp.array3d[float]),
    ("link", wp.array3d[float]),
    ("tree", wp.array[int]),
    ("u", wp.array2d[float]),
    ("qd", wp.array2d[float]),
    ("twist", wp.array3d[float]),
)


@functools.cache
def walk_kernels(table: tuple, rows: int) -> tuple:
    """The rows kernel (a thread per seed) and the tree kernel (a thread per tree, the class's emitted walk down) of
    one path table."""
    prelude = factor_walk.walk_prelude(*table, rows)
    tag = hashlib.sha256(repr(table).encode()).hexdigest()[:12]
    sizes = {"coordinate": WALK_COORDINATE, "link": WALK_LINK, "now": WALK_NOW}
    up = KernelTemplate(
        f"walk_up_{rows}_{tag}", _UP_PARAMS, substitute(_UP, rows=rows, **sizes), THREADS, (prelude,), dual=True
    )
    tree = KernelTemplate(
        f"walk_tree_{tag}",
        _TREE_PARAMS,
        substitute(_TREE, coordinate=WALK_COORDINATE, link=WALK_LINK),
        THREADS,
        (factor_walk.walk_down_prelude((tree_scan.WalkClass(*table),)),),
        dual=True,
    )
    return up.kernel(), tree.kernel()


def frame_points(walk: TreeWalk) -> np.ndarray:
    """Each body's frame point of ``walk``, its root's at the origin: the parent's less the body's offset."""
    frames = np.zeros((walk.parent.size, 3))
    for b in range(walk.parent.size):
        if walk.parent[b] >= 0:
            frames[b] = frames[walk.parent[b]] - np.asarray(walk.offset[b], dtype=np.float64)
    return frames


class DeviceWalk:
    """The device walk of trees of one topology (one path table), their data from their float32 walks."""

    def __init__(self, walks: list, device, rows: int = 3, step: list | None = None, frames: list | None = None):
        self.walks, self.device, self.rows = walks, device, rows
        self.n, self.bodies = walks[0].s.shape[0], walks[0].parent.size
        self.up_kernel, self.tree_kernel = walk_kernels(walks[0].path_table, rows)
        data, link = zip(*(walk.device_data() for walk in walks), strict=True)
        link = np.stack(link)
        # Each body's frame point (the trees' roots at the origin), at the refresh's poses: the link's frame.
        self.frames = np.stack(frames if frames is not None else [frame_points(walk) for walk in walks])
        link[:, :, WALK_FRAME : WALK_FRAME + 3] = self.frames
        self.data = wp.array(np.stack(data), dtype=float, device=device)
        self.link = wp.array(link, dtype=float, device=device)
        # The step's geometry (WalkTrees.now, .now_link): the columns at each body's frame and the links of the walks'
        # own poses, or of ``step``'s walks, the same trees at a step's poses after the refresh's (its frames at the
        # step's).
        step = walks if step is None else step
        self.step_frames = np.stack([frame_points(walk) for walk in step])
        now_link = np.stack([walk.device_data()[1] for walk in step])
        now_link[:, :, WALK_FRAME : WALK_FRAME + 3] = self.step_frames
        self.now = wp.array(
            np.stack([np.asarray(walk.s, dtype=np.float32) for walk in step]), dtype=float, device=device
        )
        self.now_link = wp.array(now_link, dtype=float, device=device)

    def up(self, tree, body, top, seeded, wrench, covector, entries: bool = False):
        """Footprints ``[seeds, rows, n]`` of seeds of ``rows`` rows: wrenches ``[seeds, rows, 6]`` at their bodies'
        frames from coordinate ``top``, plus covectors ``[seeds, rows]`` at ``top`` where ``seeded``; with ``entries``,
        also the step's ``J`` entries the sink received beside each footprint."""
        count = len(body)
        # The wrenches at the bodies' frames at the step's poses, moved to their frames in the walk's links.
        wrench = np.array(np.reshape(wrench, (count, self.rows, 6)), dtype=np.float64)
        for i in range(count):
            moved = self.step_frames[tree[i], body[i]] - self.frames[tree[i], body[i]]
            wrench[i, :, 3:6] += np.cross(moved, wrench[i, :, 0:3])
        z = wp.zeros((count, self.rows, self.n), dtype=float, device=self.device)
        jz = wp.zeros((count, self.rows, self.n), dtype=float, device=self.device)

        def array(values, dtype):
            return wp.array(
                np.asarray(values, dtype=np.float32 if dtype is float else np.int32), dtype=dtype, device=self.device
            )

        wp.launch(
            self.up_kernel,
            dim=count,
            inputs=[
                self.data,
                self.link,
                self.now,
                self.now_link,
                array(tree, int),
                array(body, int),
                array(top, int),
                array(seeded, int),
                array(np.reshape(wrench, (count, self.rows, 6)), float),
                array(np.reshape(covector, (count, self.rows)), float),
                z,
                jz,
            ],
            device=self.device,
        )
        if entries:
            return z.numpy().astype(np.float64), jz.numpy().astype(np.float64)
        return z.numpy().astype(np.float64)

    def down(self, tree, u) -> tuple:
        """The leave of ``u`` ``[cases, n]``: joint velocities and body twists, a thread per case of tree ``tree``."""
        count = len(tree)
        outputs = (
            wp.zeros((count, self.n), dtype=float, device=self.device),
            wp.zeros((count, self.bodies, 6), dtype=float, device=self.device),
        )
        wp.launch(
            self.tree_kernel,
            dim=count,
            inputs=[
                self.data,
                self.link,
                wp.array(np.asarray(tree, dtype=np.int32), dtype=int, device=self.device),
                wp.array(np.asarray(u, dtype=np.float32), dtype=float, device=self.device),
                outputs[0],
                outputs[1],
            ],
            device=self.device,
        )
        return tuple(array.numpy().astype(np.float64) for array in outputs)


def held_step(device) -> dict:
    """A step between refreshes (``update_mass_matrix_interval > 1``) on random trees of revolute, prismatic and ball
    joints: the device walk with the refresh's factor (``q_r``) and the step's geometry (``q_t``, every coordinate moved),
    its contact rows seeded at the step's frames, against ``L_r^-1 P J_t^T`` in float64 (the dense rows' whitening by the
    held factor), as worst ratios to ``n u kappa``; ``stale`` the same rows whitened with the refresh pose's ``J``, what
    the walk gave before it took the step's geometry (a root body's rows are the same in both)."""
    rng = np.random.default_rng(1102)
    out = {"held": 0.0, "stale": 0.0, "seeds": 0}
    for bodies in (3, 6, 9, 12):
        model, q_r = random_tree(rng, bodies)
        q_t = q_r + rng.normal(size=q_r.shape) * 0.2
        types, q_start = model.joint_type.numpy(), model.joint_q_start.numpy()
        for j in np.flatnonzero(types == int(newton.JointType.BALL)):
            quat = q_t[q_start[j] : q_start[j] + 4]
            q_t[q_start[j] : q_start[j] + 4] = quat / np.linalg.norm(quat)
        n = int(model.joint_dof_count)
        diagonal = rng.uniform(0.0, 0.05, n)
        walk_r = TreeWalk(*local_frames(model, q_r, 0), diagonal)
        walk_t = TreeWalk(*local_frames(model, q_t, 0), diagonal)
        walk32 = TreeWalk(*local_frames(model, q_r, 0), diagonal, dtype=np.float32)
        kappa = float(np.linalg.cond(factor_matrix(walk_r)))
        seeds = [(b, contact_wrenches(rng.normal(scale=0.05, size=3), rng.normal(size=3))) for b in range(bodies)]
        z = DeviceWalk([walk32], device, 3, step=[walk_t]).up(
            [0] * bodies,
            [b for b, _ in seeds],
            [int(walk_t.first[b + 1]) - 1 for b, _ in seeds],
            [0] * bodies,
            [w.T for _, w in seeds],
            np.zeros((bodies, 3)),
        )
        for i, (b, w) in enumerate(seeds):
            held, _ = walk_r.up(covector=jacobian(walk_t, b, w))
            stale, _ = walk_r.up(covector=jacobian(walk_r, b, w))
            scale = n * ROUNDING * kappa
            out["held"] = max(out["held"], relative(z[i].T, held) / scale)
            out["stale"] = max(out["stale"], relative(stale, held) / scale)
        out["seeds"] += bodies
    return out


def seed_rows(walk: TreeWalk, rows: int, rng) -> list:
    """Seeds of every unit a lane walks: at every body, ``rows`` wrenches (a contact's normal and tangents at a random
    point, then random); at every coordinate, ``rows`` covectors. Each is ``(body, top, seeded, wrench, covector)``."""
    seeds = []
    for b in range(walk.parent.size):
        top = int(walk.first[b + 1]) - 1
        wrench = contact_wrenches(rng.normal(scale=0.05, size=3), rng.normal(size=3))
        wrench = np.concatenate((wrench, rng.normal(size=(6, max(rows - 3, 0)))), axis=1)[:, :rows]
        seeds.append((b, top, 0, wrench.T, np.zeros(rows)))
        for c in range(int(walk.first[b]), int(walk.first[b + 1])):
            seeds.append((b, c, 1, np.zeros((rows, 6)), rng.choice((-1.0, 1.0), rows) * rng.uniform(0.5, 2.0, rows)))
    return seeds


@functools.cache
def catalogue_walks() -> tuple:
    """Every catalogue model once (loops cut at a body), at its first pose: per articulation, the float64 walk of its
    factor, the float32 one and the factor's condition number."""
    out, seen = [], set()
    for physics, _ in catalogue():
        if physics.scene.name in seen:
            continue
        seen.add(physics.scene.name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            solver = SolverFeatherPGS(physics.scene.build("cpu"))
        joint_q = solver.model.joint_q.numpy()
        for art in range(solver.model.articulation_count):
            walk = solver.factor.walk(joint_q, art)
            if walk.s.shape[0] == 0:
                continue
            kappa = float(np.linalg.cond(factor_matrix(walk)))
            out.append((physics.scene.name, art, walk, solver.factor.walk(joint_q, art, dtype=np.float32), kappa))
    return tuple(out)


def device_f1(device, units=UNITS, rng=None) -> dict:
    """F1 of the device walk on every catalogue model: each unit's rows and their ``J`` entries (``entries``: what a
    row's incident sums against ``v_hat``), the leave and its twists against the float64 walk, as worst ratios to ``n u
    kappa``, and its own rows and leave in one frame, ``|J dv - Z^T du|`` over ``n u |Z| |du|`` (pass: at most
    ``BOUND``), with the counts behind them."""
    rng = np.random.default_rng(1101) if rng is None else rng
    worst = {"rows": 0.0, "entries": 0.0, "leave": 0.0, "twist": 0.0, "frame": 0.0, "seeds": 0, "trees": 0}
    for _scene, _art, walk, walk32, kappa in catalogue_walks():
        n = walk.s.shape[0]
        scale = n * ROUNDING * kappa
        for rows in units:
            device_walk = DeviceWalk([walk32], device, rows)
            seeds = seed_rows(walk, rows, rng)
            body, top, seeded, wrench, covector = zip(*seeds, strict=True)
            z, j = device_walk.up([0] * len(seeds), body, top, seeded, wrench, covector, entries=True)
            for i, (b, c, is_covector, w, x) in enumerate(seeds):
                if is_covector:
                    X = np.zeros((n, rows))
                    X[c] = x
                    z64, _ = walk.up(covector=X)
                    j64 = X
                else:
                    z64, _ = walk.up(b, w.T)
                    j64 = jacobian(walk, b, w.T)
                worst["rows"] = max(worst["rows"], relative(z[i].T, z64) / scale)
                worst["entries"] = max(worst["entries"], relative(j[i].T, j64) / scale)
            worst["seeds"] += len(seeds)
        u = rng.normal(size=(2, n))
        reference = [walk.down(case) for case in u]
        qd, twist = DeviceWalk([walk32], device).down([0, 0], u)
        for case, (qd64, twist64) in enumerate(reference):
            worst["leave"] = max(worst["leave"], relative(qd[case], qd64) / scale)
            worst["twist"] = max(worst["twist"], relative(twist[case], twist64) / scale)
        # One frame: the device walk's rows of the contact seeds and its leave of a du, J dv against Z^T du.
        contacts = [seed for seed in seed_rows(walk, 3, rng) if not seed[2]]
        body, top, seeded, wrench, covector = zip(*contacts, strict=True)
        Z = DeviceWalk([walk32], device, 3).up([0] * len(contacts), body, top, seeded, wrench, covector)
        J = np.stack([jacobian(walk, b, w.T).T for b, _, _, w, _ in contacts])
        du = rng.normal(size=n)
        dv = DeviceWalk([walk32], device).down([0], du[None])[0][0]
        worst["frame"] = max(
            worst["frame"], frame_residual(J.reshape(-1, n), Z.reshape(-1, n), du, dv) / (n * ROUNDING)
        )
        worst["trees"] += 1
    return worst


def frame_residual(J: np.ndarray, Z: np.ndarray, du: np.ndarray, dv: np.ndarray) -> float:
    """``max_i |J_i dv - Z_i^T du| / (|Z_i| |du|)`` of rows ``J`` and their footprints ``Z`` ``[rows, n]``, ``dv`` the
    leave of ``du``. One factorization leaves the rounding of its two triangular passes, at most ``n u kappa`` and
    measured at ``n u``; two leave their factors' difference times ``|dv|``, which grows with ``kappa``."""
    residual = np.abs(J @ dv - Z @ du)
    scale = np.linalg.norm(Z, axis=1) * np.linalg.norm(du)
    keep = scale > 0.0
    return float(np.max(residual[keep] / scale[keep], initial=0.0))


# ---------------------------------------------------------------------------------------------------------------------
# F2 on the device, and one frame against two
# ---------------------------------------------------------------------------------------------------------------------


def chain_on_device(links: int, device) -> dict:
    """``long_chain(links, armature=0)``: a tip contact's three rows up and their responses down through the device
    walk (float32 data in local frames, no armature), against the float64 truth: ``Z``, ``dv = H^-1 J^T`` and the
    float32 walk on the host beside them."""
    model = scenes.long_chain(links=links, armature=0.0).build("cpu")
    joint_q = model.joint_q.numpy().astype(np.float64)
    n = int(model.joint_dof_count)
    parent, first, columns, inertia, offset = local_frames(model, joint_q, 0)
    # The device walk holds each body's float32 frame point and forms its offset as the difference of two (WALK_LINK):
    # the truth and the host's float32 walk take the chain of those offsets, so the bar measures the walk on its data.
    frames = frame_points(TreeWalk(parent, first, columns, inertia, offset, np.zeros(n))).astype(np.float32)
    offset = np.array([frames[up] - frames[b] if up >= 0 else np.zeros(3, np.float32) for b, up in enumerate(parent)])
    offset = offset.astype(np.float64)
    truth = TreeWalk(parent, first, columns, inertia, offset, np.zeros(n))
    walk32 = TreeWalk(parent, first, columns, inertia, offset, np.zeros(n), dtype=np.float32)
    body, W = ground_contact(model, newton_state_body_q(model, joint_q), links)
    z64, _ = truth.up(body, W)
    dv64 = np.stack([truth.down(z64[:, r])[0] for r in range(3)], axis=1)
    device_walk = DeviceWalk([walk32], device, frames=[frames])
    z = device_walk.up([0], [body], [int(truth.first[body + 1]) - 1], [0], [W.T], [np.zeros(3)])[0].T
    dv = device_walk.down([0, 0, 0], z.T)[0].T
    host32, _ = walk32.up(body, W.astype(np.float32))
    return {
        "links": links,
        "Z": relative(z, z64),
        "dv": relative(dv, dv64),
        "host_dv": relative(np.stack([walk32.down(host32[:, r])[0] for r in range(3)], axis=1), dv64),
        "pivots_positive": bool(np.all(walk32.ell > 0.0)),
    }


def frames_on_chain(links: int, device) -> dict:
    """One frame against two on ``long_chain(links)`` without armature (the solver adds none, so the walk and main's
    ``L`` factor one matrix): the device walk's rows of a tip contact with the device walk's leave, and with a leave by
    main's float32 ``L``: ``|J dv - Z^T du|`` over ``n u |Z| |du|``. Main's float32 factor fails on this chain without
    the armature it added (a non-finite leave), which is not within the bound either."""
    model = scenes.long_chain(links=links, armature=0.0).build("cpu")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver = SolverFeatherPGS(model, dense_max_constraints=8)
    state, out, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    joint_q = state.joint_q.numpy().astype(np.float64)
    solver.step(state, out, control, None, 1.0 / 240.0)
    n = int(model.joint_dof_count)
    H32 = MassReference(model, np.float32).mass_matrix(joint_q, 0).astype(np.float64)
    L32 = dense_factor32(H32 + np.diag(solver.factor.diagonal(0))).astype(np.float64)
    walk = solver.factor.walk(joint_q, 0)
    walk32 = solver.factor.walk(joint_q, 0, dtype=np.float32)
    body, W = ground_contact(model, newton_state_body_q(model, joint_q), links)
    J = jacobian(walk, body, W).T
    device_walk = DeviceWalk([walk32], device)
    Z = device_walk.up([0], [body], [int(walk.first[body + 1]) - 1], [0], [W.T], [np.zeros(3)])[0]
    du = Z.sum(axis=0)
    dv_walk = device_walk.down([0], du[None])[0][0]
    from scipy.linalg import solve_triangular

    dv_main = solve_triangular(L32, du[::-1], lower=True, trans=1, check_finite=False)[::-1]
    return {
        "links": links,
        "kappa": float(np.linalg.cond(factor_matrix(walk))),
        "one_frame": frame_residual(J, Z, du, dv_walk) / (n * ROUNDING),
        "two_frames": frame_residual(J, Z, du, dv_main) / (n * ROUNDING),
    }


# ---------------------------------------------------------------------------------------------------------------------
# The step's placements: one frame each, and F1 on CUDA
# ---------------------------------------------------------------------------------------------------------------------


def placement_frames(solver, joint_q: np.ndarray, capture) -> dict:
    """One run's placements against the walk, as ratios to their rounding, ``n u kappa`` and the publication's (pass:
    at most ``BOUND``). Every class with response coordinates walks its rows (the one writer, FL3.3; the sparse
    diagonal's, FL3.5), so no store keeps a row's ``J``:

    - ``rows`` and ``leave``: the one writer's worlds, their rows against their physical ``J v_hat`` and their leave
      against the float64 walk's (:func:`.test_feather_pgs_walk_rows.walked_frames`, ``slot_frames``; Z0's F1);
    - ``kernel``: the row packets' worlds (FL3.6), whose kernel walks each row's seeds and leaves by the walk: their
      published ``dv`` against the float64 walk's response of their impulses to the same seeds.

    The worlds a speed ball acted in are left out (their ``du`` is a law's, not ``sum Z lambda``), and so are the
    worlds whose rows no store holds (``unstored``: the keyed rows'; and every world of the support placement, case K1,
    whose rows are :class:`TestSupportPlacement`'s)."""
    forest, placement = solver.forest, solver.placement
    out = {"rows": 0.0, "leave": 0.0, "kernel": 0.0, "general": 0, "in_kernel": 0, "unstored": 0}
    if not getattr(placement, "walks", False) or solver.routes.case == "K1":
        out["unstored"] = int(forest.world_count)
        return out
    # The one writer of the general placement (FL3.3) and of the sparse diagonal (FL3.5): its rows are walked from
    # seeds. Its rows against their physical J v_hat and its leave against the float64 walk's
    # (test_feather_pgs_walk_rows).
    from newton.tests.test_feather_pgs_walk_rows import slot_frames, walked_frames  # noqa: PLC0415

    frames = slot_frames if placement.sparse_diagonal_contact_solve else walked_frames
    walked = frames(solver, joint_q, capture)
    out.update(rows=walked["rows"], leave=walked["leave"], general=walked["worlds"])
    packets = placement.row_packets
    if packets is None:
        return out
    sizes = forest.articulation_response_dof_count.numpy()
    offsets = forest.articulation_world_dof_offset.numpy()
    starts = forest.articulation_dof_start.numpy()
    art_world = forest.art_to_world.numpy()
    owner = placement.local_solve_owner.numpy()
    mf_counts = placement.mf_constraint_count.numpy()
    skip_rows = int(placement.ink_skip_rows())
    joint_q = np.asarray(joint_q, dtype=np.float64)
    worlds, D = int(forest.world_count), max(int(forest.max_world_dofs), 1)
    trees = {}

    def walk_of(art):
        if art not in trees:
            walk = solver.factor.walk(joint_q, art)
            trees[art] = (walk, float(np.linalg.cond(factor_matrix(walk))))
        return trees[art]

    # The step's published dv in world coordinates; the speed balls' worlds.
    v_out, v_hat = (np.zeros((worlds, D)) for _ in range(2))
    for art in range(sizes.size):
        world, n = int(art_world[art]), int(sizes[art])
        if world >= 0 and n > 0:
            dofs = int(starts[art]) + np.arange(n)
            at = slice(int(offsets[art]), int(offsets[art]) + n)
            v_out[world, at], v_hat[world, at] = capture.v_out[dofs], capture.v_hat[dofs]
    dv = v_out - v_hat
    ball = np.zeros(worlds, dtype=bool)
    if capture.mf_final is not None:
        _, _, lam, counts, _, _, _, _, kinds = capture.mf_final
        for world, live in enumerate(counts):
            line = int(capture.mf_first[world])
            for row in range(int(live)):
                if kinds[line, row] == PGS_CONSTRAINT_TYPE_SPEED:
                    ball[world] |= lam[line, row] != 0.0

    primaries = solver.sweep.local_primary_articulation.numpy()
    slips = solver.factor.walk_view().release.count.numpy()
    seeds, codes = packets.packet.seed.numpy().astype(np.float64), packets.packet.seed_body.numpy()
    contacts = packets.packet.row_contact.numpy()
    prefix_dofs, prefix_weights = packets._packets.dofs.numpy(), packets._packets.weights.numpy()
    for world, live in enumerate(capture.counts):
        count = int(live)
        if count == 0 or ball[world] or owner[world] == PGS_LOCAL_SOLVE_OWNER_GENERAL:
            continue  # The one writer's worlds: walked_frames above.
        if skip_rows > 0 and count <= skip_rows and mf_counts[world] == 0 and placement.keyed_rows is not None:
            out["unstored"] += 1  # The keyed rows keep their worlds' rows out of the store.
            continue
        # The row packets (FL3.6): the kernel walks each row's seeds and leaves by the walk. Their published dv
        # against the float64 walk's response of the impulses to the same seeds: a covector at a coordinate, a
        # wrench at a body's frame. A tree that slips this step carries its release (TestLocalSlipFrame).
        art = int(primaries[world])
        line = int(capture.dense_first[world])
        if art < 0 or slips[art]:
            continue
        walk, kappa = walk_of(art)
        n = int(sizes[art])
        response = np.zeros(n)
        for row in range(count):
            lam = float(capture.impulses[line, row])
            if lam == 0.0:
                continue
            if contacts[world, row] < 0:
                for dof, weight in zip(prefix_dofs[world, row], prefix_weights[world, row], strict=True):
                    if dof >= 0 and weight != 0.0:
                        x = np.zeros(n)
                        x[dof] = weight
                        response += walk.up(covector=x)[0] * lam
                continue
            for side in range(2):
                if codes[world, row, side] >= 0:
                    response += (
                        walk.up(int(codes[world, row, side]), seeds[world, row, 6 * side : 6 * side + 6])[0] * lam
                    )
        at = slice(int(offsets[art]), int(offsets[art]) + n)
        if np.any(dv[world, at]):
            published = 2.0 * ROUNDING * (np.linalg.norm(v_out[world, at]) + np.linalg.norm(v_hat[world, at]))
            scale = n * ROUNDING * kappa + published / np.linalg.norm(dv[world, at])
            out["kernel"] = max(out["kernel"], relative(dv[world, at], walk.down(response)[0]) / scale)
        out["in_kernel"] += 1
    return out


def cuda_placements() -> list:
    """The CUDA catalogue's problems under point friction on the routes whose rows a store holds, ``(physics,
    route)`` in catalogue order: a route is checked on its first problem with articulated rows (:func:`checked`)."""
    return [
        (group.physics, route)
        for group in conformance.cuda_groups()
        if group.physics.law != "patch"
        for route in (group.reference, *group.routes)
        if route.name not in UNSTORED
    ]


# ---------------------------------------------------------------------------------------------------------------------
# The class emitter in both directions
# ---------------------------------------------------------------------------------------------------------------------

_BOTH = """    if (tid >= weights.shape[0]) return;
    const int n = values.shape[1], bodies = weights.shape[1];
    WALK_BOTH(weights.data + (size_t)tid * bodies, values.data + (size_t)tid * n, path.data + (size_t)tid * n,
              subtree.data + (size_t)tid * bodies);
"""
_BOTH_PARAMS = params(
    ("weights", wp.array2d[float]),
    ("values", wp.array2d[float]),
    ("path", wp.array2d[float]),
    ("subtree", wp.array2d[float]),
)


@functools.cache
def both_ways_kernel(table: tuple):
    """A thread per case of one class: root first, each coordinate's sum of the weights of the bodies and the values of
    the coordinates on its root path up to it (``path``); leaf first, each body's sum over its subtree (``subtree``)."""
    walk = tree_scan.WalkClass(*table)

    down = walk.walk("$V[0] = $P[0];", "$V[0] += x[$c];\ndown[$c] = $V[0];", begin="$V[0] += w[$b];", width=1)
    up = walk.walk(
        "$P[0] += $V[0];", "$V[0] += x[$c];", begin="$V[0] += w[$b];", end="up[$b] = $V[0];", root_first=False, width=1
    )
    text = f"""    const auto WALK_BOTH = [&](const float* w, const float* x, float* down, float* up) {{
        {{
{tree_scan.indent_text(down, 12)}
        }}
        {{
{tree_scan.indent_text(up, 12)}
        }}
    }};"""

    # The loop form's tables take the one access's WALK_CONSTANT.
    prelude = DevicePrelude(f"walk_both_{walk.tag}", text, cpu=text)
    preludes = (factor_walk.walk_loads_prelude(), prelude)
    return KernelTemplate(f"walk_both_{walk.tag}", _BOTH_PARAMS, _BOTH, THREADS, preludes, dual=True).kernel()


def both_ways(table: tuple, device, rng) -> tuple:
    """Integer weights and values (sums exact in float32) walked both ways by the emitted class, against numpy."""
    parent, first = table
    bodies, n = len(parent), first[-1]
    cases = 4
    weights = rng.integers(-8, 9, size=(cases, bodies)).astype(np.float32)
    values = rng.integers(-8, 9, size=(cases, n)).astype(np.float32)
    path, subtree = np.zeros((cases, n), np.float32), np.zeros((cases, bodies), np.float32)
    for case in range(cases):
        acc = np.zeros(bodies)
        for b in range(bodies):
            acc[b] = (acc[parent[b]] if parent[b] >= 0 else 0.0) + weights[case, b]
            for c in range(first[b], first[b + 1]):
                acc[b] += values[case, c]
                path[case, c] = acc[b]
        sub = weights[case].astype(np.float64) + [values[case, first[b] : first[b + 1]].sum() for b in range(bodies)]
        for b in reversed(range(bodies)):
            if parent[b] >= 0:
                sub[parent[b]] += sub[b]
        subtree[case] = sub
    outputs = (wp.zeros((cases, n), dtype=float, device=device), wp.zeros((cases, bodies), dtype=float, device=device))
    wp.launch(
        both_ways_kernel(table),
        dim=cases,
        inputs=[wp.array(weights, dtype=float, device=device), wp.array(values, dtype=float, device=device), *outputs],
        device=device,
    )
    return outputs[0].numpy(), path, outputs[1].numpy(), subtree


def emitter_tables(rng) -> list:
    """Classes on both sides of ``WALK_UNROLL``: random trees (free and scalar joints, several roots, joins whose spans
    nest and follow each other), a chain and a star."""
    tables = []
    for bodies, roots in ((5, 1), (12, 2), (30, 1), (48, 3)):
        parent = [-1] + [int(rng.integers(-1, b)) if b < roots else int(rng.integers(0, b)) for b in range(1, bodies)]
        counts = [int(rng.choice((1, 1, 1, 3, 6))) for _ in range(bodies)]
        tables.append((tuple(parent), tuple(np.concatenate(([0], np.cumsum(counts))).tolist())))
    tables.append((tuple(range(-1, 105)), tuple(range(107))))
    tables.append(((-1, *([0] * 69)), (0, *range(6, 76))))
    return tables


def checked(result: dict) -> bool:
    """Whether a run held rows the check could read: worlds the general sweep solves from the store, or worlds
    whitened in a kernel."""
    return result["general"] + result["in_kernel"] > 0


def refresh_error(device, bodies: int, rng) -> float:
    """One refresh of a random tree (its joins included) by walk_step_kernel's lane feed against the float64 walk: the
    worst relative error of ``s``, ``ut``, ``1/ell`` and ``P0``."""
    model, q = random_tree(rng, bodies, device)
    model.joint_q.assign(q)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver = SolverFeatherPGS(model, pgs_iterations=4)
    state = model.state()
    factor = solver.factor
    if factor.walk_arts is None:
        factor.plan_walk(np.arange(model.articulation_count, dtype=np.int32))
    # The held tree's repair publishes the state's poses and the walk's step geometry, as a step's would.
    factor.last_step_dt = 1.0 / 240.0
    factor.fk_id(state, model.state(), wp.zeros_like(state.joint_qd))
    factor.mass_update_mask.fill_(1)
    walk = factor.walk(q, 0)
    n = walk.s.shape[0]
    pivots = walk.own_pivots + factor.armature[:n]
    factor.walk_step(state, 1.0 / 240.0)
    d = unpack_walk_records(factor.walk_data.numpy(), n).astype(np.float64)
    return max(
        np.abs(d[:, 0:6] - walk.s).max(),
        np.abs(d[:, 6:12] - walk.ut).max() / max(np.abs(walk.ut).max(), 1.0e-30),
        np.abs(1.0 / d[:, 12] - walk.ell).max() / max(walk.ell.max(), 1.0e-30),
        np.abs(d[:, 13] - pivots).max() / max(np.abs(walk.own_pivots).max(), 1.0e-30),
    )


# ---------------------------------------------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------------------------------------------


class TestWalkDevice(unittest.TestCase):
    """F1 and F2 of the device walk, and one frame per placement, on the CPU and the CUDA device."""

    def test_f1_device_walk_on_the_catalogue(self):
        """Every unit's rows and their ``J`` entries, the leave and its twists within ``BOUND n u kappa`` of the float64
        walk, and the device walk's own rows and leave in one frame."""
        for device in devices():
            with self.subTest(device=device):
                worst = device_f1(device)
                self.assertGreater(worst["seeds"], 0)
                for name in ("rows", "entries", "leave", "twist", "frame"):
                    self.assertLessEqual(worst[name], BOUND, f"{name}: {worst}")

    def test_f2_on_the_device(self):
        """A tip impulse's response on ``long_chain(N)`` without armature: the device walk's float32 within 1e-5 of the
        float64 truth at every length, as the host's float32 walk."""
        for device in devices():
            for links in CHAINS:
                with self.subTest(device=device, links=links):
                    chain = chain_on_device(links, device)
                    self.assertTrue(chain["pivots_positive"])
                    self.assertLess(chain["dv"], 1.0e-5, f"{chain}")
                    self.assertLess(chain["dv"], 10.0 * chain["host_dv"] + 1.0e-6, f"{chain}")

    def test_a_held_factor_whitens_the_steps_J(self):
        """Between refreshes the factor is the refresh's and ``J`` the step's: the device walk with the step's
        geometry gives ``L_r^-1 P J_t^T`` within ``BOUND n u kappa``, and the refresh pose's ``J`` misses it by far."""
        for device in devices():
            with self.subTest(device=device):
                result = held_step(device)
                self.assertGreater(result["seeds"], 0)
                self.assertLessEqual(result["held"], BOUND, f"{result}")
                self.assertGreater(result["stale"], 1.0e3 * BOUND, f"{result}")

    def test_one_frame_tells_two_apart(self):
        """``long_chain(106)`` without armature: the walk's rows with the walk's leave stay within ``BOUND n u`` of
        ``|Z| |du|``; with a leave by main's float32 factor they do not (a failed factor, NaN, is not within). The
        invariant guards a placement that converts its rows without its leave."""
        for device in devices():
            with self.subTest(device=device):
                frames = frames_on_chain(106, device)
                self.assertLessEqual(frames["one_frame"], BOUND, f"{frames}")
                self.assertFalse(frames["two_frames"] <= BOUND, f"{frames}")

    def test_one_frame_per_placement_cpu(self):
        """Every CPU catalogue route: its rows and its leave within ``BOUND n u kappa`` of the walk's."""
        for physics, route in catalogue():
            with self.subTest(problem=physics.name, route=route.name):
                solver, joint_q, capture = run_route(physics, route)
                result = placement_frames(solver, joint_q, capture)
                for name in ("rows", "leave", "kernel"):
                    self.assertLessEqual(result[name], BOUND, f"{name}: {result}")

    def test_the_refresh_meets_the_float64_walk(self):
        """walk_step_kernel's refresh (the lane feed) within float32 rounding of the float64 walk, on random trees with
        joins."""
        for device in devices():
            for bodies in (3, 8, 14):
                with self.subTest(device=device, bodies=bodies):
                    error = refresh_error(device, bodies, np.random.default_rng(bodies))
                    self.assertLess(error, 1.0e-4)

    def test_the_class_emitter_walks_both_ways(self):
        """The class emitter's walk root first and leaf first, unrolled and in the loop form, on trees with joins and
        several roots: exact sums along each root path and over each subtree."""
        tables = emitter_tables(np.random.default_rng(1102))
        self.assertTrue(any(first[-1] <= tree_scan.WALK_UNROLL for _, first in tables))
        self.assertTrue(any(first[-1] > tree_scan.WALK_UNROLL for _, first in tables))
        for device in devices():
            for table in tables:
                with self.subTest(device=device, coordinates=table[1][-1]):
                    down, path, up, subtree = both_ways(table, device, np.random.default_rng(len(table[0])))
                    np.testing.assert_array_equal(down, path)
                    np.testing.assert_array_equal(up, subtree)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA placements")
    def test_f1_on_the_cuda_placements(self):
        """Z0's F1 on the CUDA placements: the one writer's worlds' rows and leave, and the row packets' worlds, whose
        kernel walks their seeds, through the walk's response of their impulses."""
        covered, done = {"general": 0, "in_kernel": 0}, set()
        for physics, route in cuda_placements():
            if route.name in done:
                continue
            with self.subTest(problem=physics.name, route=route.name):
                solver, joint_q, capture = run_route(physics, route, device="cuda:0")
                result = placement_frames(solver, joint_q, capture)
                for name in ("rows", "leave", "kernel"):
                    self.assertLessEqual(result[name], BOUND, f"{name}: {result}")
                if checked(result):
                    done.add(route.name)
                for name in covered:
                    covered[name] += result[name]
        # Both forms were checked: the one writer's worlds, and worlds whose kernel walks their rows' seeds.
        self.assertGreater(covered["general"], 0)
        self.assertGreater(covered["in_kernel"], 0)


def report() -> None:
    """Print the measured tables."""
    for device in devices():
        print(f"== {device}: F1 of the device walk (worst ratio to n u kappa)", device_f1(device))
        for links in CHAINS:
            print(f"== {device}: F2", chain_on_device(links, device))
        for links in (32, 106, 300):
            print(f"== {device}: frames", frames_on_chain(links, device))
    for physics, route in catalogue():
        solver, joint_q, capture = run_route(physics, route)
        print(f"== cpu {physics.name} {route.name}", placement_frames(solver, joint_q, capture))
    if wp.is_cuda_available():
        done = set()
        for physics, route in cuda_placements():
            if route.name in done:
                continue
            solver, joint_q, capture = run_route(physics, route, device="cuda:0")
            result = placement_frames(solver, joint_q, capture)
            print(f"== cuda {physics.name} {route.name}", result)
            if checked(result):
                done.add(route.name)


if __name__ == "__main__":
    if sys.argv[1:] == ["report"]:
        report()
    else:
        unittest.main()
