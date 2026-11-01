# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The walk on the device (FL3.1): the tree factor's walk as the native functions FL3's one writer will call.

:func:`.tree_factor.walk_prelude` declares the walk for both devices: the class's path table in constant memory, a lane
per seed walking its rows up (``WALK_UP``, ``WALK_SEED``) into the caller's sink, and a lane per tree walking down (the
leave, ``WALK_DOWN``, and the twists of known coordinates, ``WALK_TWISTS``). No step launches it. These tests run it on
the CPU and, where there is one, on a CUDA device, against the float64 walk (:class:`.tree_factor.TreeWalk`, which
:mod:`.test_feather_pgs_walk` holds to the factor), and hold every placement to one frame:

- **F1, the device walk**, on every catalogue model: rows of every unit (1 to 4 rows a lane) from wrench and covector
  seeds, the leave and the twists, within ``BOUND n u kappa`` of the float64 walk;
- **F2 on the device**: ``long_chain(N)`` without armature, N = 8 to 1,000: a tip impulse's response, rows up and leave
  down in float32, against the float64 truth;
- **one frame per placement**: whitening and the leave of ``du`` share one factorization, so ``|J dv - Z^T du|`` stays
  at rounding: within ``n u kappa`` of ``|Z| |du|``, and as measured within ``n u``. The device walk's rows and leave
  hold the tighter bound; on ``long_chain(106)`` a leave by main's factor under the walk's rows breaks it, so the test
  tells one frame from two. Every placement whose rows the step stores holds both bounds, on every CPU catalogue
  route and on the CUDA catalogue's routes;
- **F1 on CUDA**: Z0's F1 on the CUDA placements: the general rows' footprints and leave, and the worlds whose sweep
  whitens their rows in the kernel (the local owners, the Jacobi tiers, the paired factor), whose store keeps ``J``,
  through the walk's response of their impulses. Native sparse factors, the sparse diagonal, the compact boundary,
  the keyed rows and the row packets store neither ``J`` nor a packed ``Z``: :data:`UNSTORED` names their routes.

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
from newton._src.solvers.feather_pgs import tree_factor
from newton._src.solvers.feather_pgs.device_preludes import substitute
from newton._src.solvers.feather_pgs.kernel_template import THREADS, KernelTemplate, params
from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT,
    PGS_LOCAL_SOLVE_OWNER_GENERAL,
)
from newton._src.solvers.feather_pgs.tree_factor import WALK_COORDINATE, WALK_LINK, TreeWalk, local_frames
from newton.solvers import SolverFeatherPGS
from newton.tests import feather_pgs_scenes as scenes
from newton.tests import test_feather_pgs_conformance as conformance
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

UNITS = (1, 2, 3, 4)
"""Rows a lane walks: a limit, a velocity limit's pair, a point contact, a contact with spin."""
UNSTORED = ("sparse", "compact", "keyed")
"""Prefixes of the CUDA catalogue's routes whose rows no store holds as ``J`` or packed ``Z``: native sparse factors
(rows in factor coordinates, whitened from a ``J`` in shared scratch), the sparse diagonal and the compact boundary,
and the keyed rows; :func:`placement_frames` also counts the row packets' worlds as unstored."""


def devices() -> list:
    return ["cpu", *(["cuda:0"] if wp.is_cuda_available() else [])]


# ---------------------------------------------------------------------------------------------------------------------
# The device walk of one topology
# ---------------------------------------------------------------------------------------------------------------------

_UP = """    if (tid >= body.shape[0]) return;
    const int n = data.shape[1], bodies = link.shape[1];
    const float* d = data.data + (size_t)tree.data[tid] * n * $coordinate;
    const float* l = link.data + (size_t)tree.data[tid] * bodies * $link;
    float* out = z.data + (size_t)tid * $rows * n;
    const auto sink = [&](int r, int c, float value) { out[(size_t)r * n + c] = value; };
    float p[6 * $rows];
    for (int k = 0; k < 6 * $rows; ++k) p[k] = wrench.data[(size_t)tid * 6 * $rows + k];
    int start = top.data[tid];
    if (seeded.data[tid] != 0) {
        float x[$rows];
        for (int r = 0; r < $rows; ++r) x[r] = covector.data[(size_t)tid * $rows + r];
        WALK_SEED(d, start, x, p, sink);
        start -= 1;
    }
    WALK_UP(d, l, body.data[tid], start, p, sink);
"""

_TREE = """    if (tid >= u.shape[0]) return;
    const int n = data.shape[1], bodies = link.shape[1];
    const float* d = data.data + (size_t)tree.data[tid] * n * $coordinate;
    const float* l = link.data + (size_t)tree.data[tid] * bodies * $link;
    WALK_DOWN(d, l, u.data + (size_t)tid * n, qd.data + (size_t)tid * n, twist.data + (size_t)tid * bodies * 6);
    WALK_TWISTS(d, l, known.data + (size_t)tid * n, known_twist.data + (size_t)tid * bodies * 6);
"""

_UP_PARAMS = params(
    ("data", wp.array3d[float]),
    ("link", wp.array3d[float]),
    ("tree", wp.array[int]),
    ("body", wp.array[int]),
    ("top", wp.array[int]),
    ("seeded", wp.array[int]),
    ("wrench", wp.array3d[float]),
    ("covector", wp.array2d[float]),
    ("z", wp.array3d[float]),
)
_TREE_PARAMS = params(
    ("data", wp.array3d[float]),
    ("link", wp.array3d[float]),
    ("tree", wp.array[int]),
    ("u", wp.array2d[float]),
    ("qd", wp.array2d[float]),
    ("twist", wp.array3d[float]),
    ("known", wp.array2d[float]),
    ("known_twist", wp.array3d[float]),
)


@functools.cache
def walk_kernels(table: tuple, rows: int) -> tuple:
    """The rows kernel (a thread per seed) and the tree kernel (a thread per tree) of one path table."""
    prelude = tree_factor.walk_prelude(*table, rows)
    tag = hashlib.sha256(repr(table).encode()).hexdigest()[:12]
    sizes = {"coordinate": WALK_COORDINATE, "link": WALK_LINK}
    up = KernelTemplate(
        f"walk_up_{rows}_{tag}", _UP_PARAMS, substitute(_UP, rows=rows, **sizes), THREADS, (prelude,), dual=True
    )
    tree = KernelTemplate(f"walk_tree_{tag}", _TREE_PARAMS, substitute(_TREE, **sizes), THREADS, (prelude,), dual=True)
    return up.kernel(), tree.kernel()


class DeviceWalk:
    """The device walk of trees of one topology (one path table), their data from their float32 walks."""

    def __init__(self, walks: list, device, rows: int = 3):
        self.walks, self.device, self.rows = walks, device, rows
        self.n, self.bodies = walks[0].s.shape[0], walks[0].parent.size
        self.up_kernel, self.tree_kernel = walk_kernels(walks[0].path_table, rows)
        data, link = zip(*(walk.device_data() for walk in walks), strict=True)
        self.data = wp.array(np.stack(data), dtype=float, device=device)
        self.link = wp.array(np.stack(link), dtype=float, device=device)

    def up(self, tree, body, top, seeded, wrench, covector) -> np.ndarray:
        """Footprints ``[seeds, rows, n]`` of seeds of ``rows`` rows: wrenches ``[seeds, rows, 6]`` at their bodies'
        frames from coordinate ``top``, plus covectors ``[seeds, rows]`` at ``top`` where ``seeded``."""
        count = len(body)
        z = wp.zeros((count, self.rows, self.n), dtype=float, device=self.device)

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
                array(tree, int),
                array(body, int),
                array(top, int),
                array(seeded, int),
                array(np.reshape(wrench, (count, self.rows, 6)), float),
                array(np.reshape(covector, (count, self.rows)), float),
                z,
            ],
            device=self.device,
        )
        return z.numpy().astype(np.float64)

    def down(self, tree, u, known) -> tuple:
        """The leave of ``u`` ``[cases, n]`` (joint velocities and body twists) and the twists of ``known`` coordinates,
        a thread per case of tree ``tree``."""
        count = len(tree)
        outputs = (
            wp.zeros((count, self.n), dtype=float, device=self.device),
            wp.zeros((count, self.bodies, 6), dtype=float, device=self.device),
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
                wp.array(np.asarray(known, dtype=np.float32), dtype=float, device=self.device),
                outputs[2],
            ],
            device=self.device,
        )
        return tuple(array.numpy().astype(np.float64) for array in outputs)


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
    """F1 of the device walk on every catalogue model: each unit's rows, the leave and the twists against the float64
    walk, as worst ratios to ``n u kappa``, and its own rows and leave in one frame, ``|J dv - Z^T du|`` over ``n u |Z|
    |du|`` (pass: at most ``BOUND``), with the counts behind them."""
    rng = np.random.default_rng(1101) if rng is None else rng
    worst = {"rows": 0.0, "leave": 0.0, "twist": 0.0, "known": 0.0, "frame": 0.0, "seeds": 0, "trees": 0}
    for _scene, _art, walk, walk32, kappa in catalogue_walks():
        n = walk.s.shape[0]
        scale = n * ROUNDING * kappa
        for rows in units:
            device_walk = DeviceWalk([walk32], device, rows)
            seeds = seed_rows(walk, rows, rng)
            body, top, seeded, wrench, covector = zip(*seeds, strict=True)
            z = device_walk.up([0] * len(seeds), body, top, seeded, wrench, covector)
            for i, (b, c, is_covector, w, x) in enumerate(seeds):
                if is_covector:
                    X = np.zeros((n, rows))
                    X[c] = x
                    z64, _ = walk.up(covector=X)
                else:
                    z64, _ = walk.up(b, w.T)
                worst["rows"] = max(worst["rows"], relative(z[i].T, z64) / scale)
            worst["seeds"] += len(seeds)
        u = rng.normal(size=(2, n))
        reference = [walk.down(case) for case in u]
        known = np.stack([qd for qd, _ in reference])
        qd, twist, known_twist = DeviceWalk([walk32], device).down([0, 0], u, known)
        for case, (qd64, twist64) in enumerate(reference):
            worst["leave"] = max(worst["leave"], relative(qd[case], qd64) / scale)
            worst["twist"] = max(worst["twist"], relative(twist[case], twist64) / scale)
            worst["known"] = max(worst["known"], relative(known_twist[case], twist64) / scale)
        # One frame: the device walk's rows of the contact seeds and its leave of a du, J dv against Z^T du.
        contacts = [seed for seed in seed_rows(walk, 3, rng) if not seed[2]]
        body, top, seeded, wrench, covector = zip(*contacts, strict=True)
        Z = DeviceWalk([walk32], device, 3).up([0] * len(contacts), body, top, seeded, wrench, covector)
        J = np.stack([jacobian(walk, b, w.T).T for b, _, _, w, _ in contacts])
        du = rng.normal(size=n)
        dv = DeviceWalk([walk32], device).down([0], du[None], du[None])[0][0]
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
    truth = TreeWalk(*local_frames(model, joint_q, 0), np.zeros(n))
    walk32 = TreeWalk(*local_frames(model, joint_q, 0), np.zeros(n), dtype=np.float32)
    body, W = ground_contact(model, newton_state_body_q(model, joint_q), links)
    z64, _ = truth.up(body, W)
    dv64 = np.stack([truth.down(z64[:, r])[0] for r in range(3)], axis=1)
    device_walk = DeviceWalk([walk32], device)
    z = device_walk.up([0], [body], [int(truth.first[body + 1]) - 1], [0], [W.T], [np.zeros(3)])[0].T
    dv = device_walk.down([0, 0, 0], z.T, z.T)[0].T
    host32, _ = walk32.up(body, W.astype(np.float32))
    return {
        "links": links,
        "Z": relative(z, z64),
        "dv": relative(dv, dv64),
        "host_dv": relative(np.stack([walk32.down(host32[:, r])[0] for r in range(3)], axis=1), dv64),
        "pivots_positive": bool(np.all(walk32.ell > 0.0)),
    }


def frames_on_chain(links: int, device) -> dict:
    """One frame against two on ``long_chain(links)`` as main runs it (its armature in the walk's diagonal, so the walk
    and main's ``L`` factor one matrix): the device walk's rows of a tip contact with the device walk's leave, and with
    a leave by main's float32 ``L``: ``|J dv - Z^T du|`` over ``n u |Z| |du|``."""
    model = scenes.long_chain(links=links, armature=0.0).build("cpu")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver = SolverFeatherPGS(model, dense_max_constraints=8)
    state, out, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    joint_q = state.joint_q.numpy().astype(np.float64)
    solver.step(state, out, control, None, 1.0 / 240.0)
    n = int(model.joint_dof_count)
    L32 = np.tril(solver.factor.L_by_size[n].numpy()[0]).astype(np.float64)
    walk = solver.factor.walk(joint_q, 0)
    walk32 = solver.factor.walk(joint_q, 0, dtype=np.float32)
    body, W = ground_contact(model, newton_state_body_q(model, joint_q), links)
    J = jacobian(walk, body, W).T
    device_walk = DeviceWalk([walk32], device)
    Z = device_walk.up([0], [body], [int(walk.first[body + 1]) - 1], [0], [W.T], [np.zeros(3)])[0]
    du = Z.sum(axis=0)
    dv_walk = device_walk.down([0], du[None], du[None])[0][0]
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
    """One run's placements against the walk, by the form the store keeps a world's rows in
    (:func:`.packed_rows._skipped`), as ratios to their rounding, ``n u kappa`` and the publication's (pass: at most
    ``BOUND``):

    - ``rows``: the general worlds' footprints against the float64 walk's of their ``J`` (Z0's F1);
    - ``frame`` and ``tight``: their ``|J_i dv - Z_i^T du|`` against ``n u kappa |Z_i| |du|`` and against ``n u |Z_i|
      |du|``, each with ``|J_i|`` times the publication's rounding of ``dv``, ``du`` the step's (its final rows' ``Z
      lambda``, the free bodies' included) and ``dv`` its published velocity change: whitening and leave in one
      frame;
    - ``leave``: their published ``dv`` against the walk's leave of ``du`` (Z0's F1);
    - ``kernel``: the worlds whose sweep whitens their rows in the kernel, whose store keeps ``J``: their published
      ``dv`` against the walk's response of their impulses.

    The worlds a speed ball acted in are left out (their ``du`` is a law's, not ``sum Z lambda``), and so are the
    worlds whose rows no store holds (``unstored``: the keyed rows', the row packets'; and every world of native sparse
    factors and of the sparse diagonal)."""
    forest, placement = solver.forest, solver.placement
    out = {"rows": 0.0, "frame": 0.0, "tight": 0.0, "leave": 0.0, "kernel": 0.0, "general": 0, "in_kernel": 0}
    out["unstored"] = 0
    if getattr(placement, "walks", False):
        # The general placement's one writer (FL3.3): its rows are walked from seeds, no J is stored. Its rows against
        # their physical J v_hat and its leave against the float64 walk's (test_feather_pgs_walk_rows).
        from newton.tests.test_feather_pgs_walk_rows import walked_frames  # noqa: PLC0415

        walked = walked_frames(solver, joint_q, capture)
        out.update(rows=walked["rows"], leave=walked["leave"], general=walked["worlds"])
        return out
    if capture.J is None or solver.routes.case in ("K1", "K2"):
        out["unstored"] = int(forest.world_count)
        return out
    sizes = forest.articulation_response_dof_count.numpy()
    offsets = forest.articulation_world_dof_offset.numpy()
    starts = forest.articulation_dof_start.numpy()
    art_world = forest.art_to_world.numpy()
    owner = placement.local_solve_owner.numpy()
    free = forest.is_free_rigid_host if forest.is_free_rigid is not None else np.zeros(sizes.size, dtype=np.int32)
    mf_counts = placement.mf_constraint_count.numpy()
    skip_rows = int(placement.ink_skip_rows())
    dense = bool(placement.packed_dense)
    joint_q = np.asarray(joint_q, dtype=np.float64)
    worlds, D = int(forest.world_count), max(int(forest.max_world_dofs), 1)
    walks = {}

    def walk_of(art):
        if art not in walks:
            walk = solver.factor.walk(joint_q, art)
            walks[art] = (walk, float(np.linalg.cond(factor_matrix(walk))))
        return walks[art]

    def sides(line, row):
        """The row's sides: ``(art, entries in the store, world coordinates)``."""
        first = 0
        for art in capture.sides[line, row]:
            if art < 0:
                continue
            n, offset = int(sizes[art]), int(offsets[art])
            at = offset if dense else first
            first += n
            yield int(art), slice(at, at + n), slice(offset, offset + n)

    # The step's du and published dv in world coordinates; the speed balls' worlds.
    du, dv, v_out, v_hat = (np.zeros((worlds, D)) for _ in range(4))
    for art in range(sizes.size):
        world, n = int(art_world[art]), int(sizes[art])
        if world >= 0 and n > 0:
            dofs = int(starts[art]) + np.arange(n)
            at = slice(int(offsets[art]), int(offsets[art]) + n)
            v_out[world, at], v_hat[world, at] = capture.v_out[dofs], capture.v_hat[dofs]
    dv = v_out - v_hat
    for world, live in enumerate(capture.counts):
        line = int(capture.dense_first[world])
        for row in range(int(live)):
            for _art, entries, coordinates in sides(line, row):
                du[world, coordinates] += capture.Z_final[line, row, entries] * capture.impulses[line, row]
    ball = np.zeros(worlds, dtype=bool)
    if capture.mf_final is not None:
        Z_a, Z_b, lam, counts, _, _, dof_a, dof_b, kinds = capture.mf_final
        for world, live in enumerate(counts):
            line = int(capture.mf_first[world])
            for row in range(int(live)):
                if kinds[line, row] == PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT:
                    ball[world] |= lam[line, row] != 0.0
                    continue
                for dof, Z in ((dof_a, Z_a), (dof_b, Z_b)):
                    if dof[line, row] >= 0:
                        du[world, dof[line, row] : dof[line, row] + 6] += Z[line, row] * lam[line, row]

    for world, live in enumerate(capture.counts):
        count = int(live)
        if count == 0 or ball[world]:
            continue
        tier = skip_rows > 0 and count <= skip_rows and mf_counts[world] == 0
        local = owner[world] != PGS_LOCAL_SOLVE_OWNER_GENERAL
        if (tier and placement.keyed_rows is not None) or (local and placement.row_packets is not None):
            out["unstored"] += 1  # The keyed rows and the row packets keep their worlds' rows out of the store.
            continue
        line = int(capture.dense_first[world])
        if (tier or local) and solver.factor.eliminates:
            out["unstored"] += 1  # The closure's corrected rows live in the kernel; the store keeps the raw J.
            continue
        if tier or local:
            # The kernel's whitening and leave against the walk's: its articulations' response to the rows' impulses.
            response = np.zeros(D)
            for row in range(count):
                for art, entries, coordinates in sides(line, row):
                    J = capture.J[line, row, entries].astype(np.float64)
                    response[coordinates] += walk_of(art)[0].up(covector=J)[0] * capture.impulses[line, row]
            for art in {art for row in range(count) for art, _, _ in sides(line, row)}:
                walk, kappa = walk_of(art)
                at = slice(int(offsets[art]), int(offsets[art]) + int(sizes[art]))
                # A free body's velocity change also takes its free-body rows, which the kernel solves beside these.
                if free[art] or not np.any(dv[world, at]):
                    continue
                published = 2.0 * ROUNDING * (np.linalg.norm(v_out[world, at]) + np.linalg.norm(v_hat[world, at]))
                scale = int(sizes[art]) * ROUNDING * kappa + published / np.linalg.norm(dv[world, at])
                out["kernel"] = max(out["kernel"], relative(dv[world, at], walk.down(response[at])[0]) / scale)
            out["in_kernel"] += 1
            continue
        for row in range(count):
            # J_i dv - Z_i^T du over the row's sides; its rounding: n u kappa of |Z_i| |du|, and |J_i| times the
            # publication's rounding of dv.
            term, zz, uu, factor, published, length = 0.0, 0.0, 0.0, 0.0, 0.0, 0
            for art, entries, coordinates in sides(line, row):
                walk, kappa = walk_of(art)
                n = int(sizes[art])
                J = capture.J[line, row, entries].astype(np.float64)
                Z = capture.Z[line, row, entries].astype(np.float64)
                z64, _ = walk.up(covector=J)
                if np.any(z64):
                    out["rows"] = max(out["rows"], relative(Z, z64) / (n * ROUNDING * kappa))
                term += J @ dv[world, coordinates] - Z @ du[world, coordinates]
                zz, uu = zz + Z @ Z, uu + du[world, coordinates] @ du[world, coordinates]
                factor = max(factor, n * ROUNDING * kappa)
                length = max(length, n)
                velocity = np.linalg.norm(v_out[world, coordinates]) + np.linalg.norm(v_hat[world, coordinates])
                published += np.linalg.norm(J) * 2.0 * ROUNDING * velocity
            scale = factor * np.sqrt(zz * uu) + published
            if scale > 0.0:
                out["frame"] = max(out["frame"], abs(term) / scale)
                tight = length * ROUNDING * np.sqrt(zz * uu) + published
                out["tight"] = max(out["tight"], abs(term) / tight)
        for art in {art for row in range(count) for art, _, _ in sides(line, row)}:
            walk, kappa = walk_of(art)
            at = slice(int(offsets[art]), int(offsets[art]) + int(sizes[art]))
            if not np.any(du[world, at]):
                continue
            published = 2.0 * ROUNDING * (np.linalg.norm(v_out[world, at]) + np.linalg.norm(v_hat[world, at]))
            scale = int(sizes[art]) * ROUNDING * kappa + published / np.linalg.norm(dv[world, at])
            out["leave"] = max(out["leave"], relative(dv[world, at], walk.down(du[world, at])[0]) / scale)
        out["general"] += 1
    return out


def cuda_placements() -> list:
    """The CUDA catalogue's problems under point friction on the routes whose rows a store holds, ``(physics,
    route)`` in catalogue order: a route is checked on its first problem with articulated rows (:func:`checked`)."""
    return [
        (group.physics, route)
        for group in conformance.cuda_groups()
        if group.physics.law != "patch"
        for route in (group.reference, *group.routes)
        if not route.name.startswith(UNSTORED)
    ]


def checked(result: dict) -> bool:
    """Whether a run held rows the check could read: worlds the general sweep solves from the store, or worlds
    whitened in a kernel."""
    return result["general"] + result["in_kernel"] > 0


# ---------------------------------------------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------------------------------------------


class TestWalkDevice(unittest.TestCase):
    """F1 and F2 of the device walk, and one frame per placement, on the CPU and the CUDA device."""

    def test_f1_device_walk_on_the_catalogue(self):
        """Every unit's rows, the leave and the twists within ``BOUND n u kappa`` of the float64 walk, and the device
        walk's own rows and leave in one frame."""
        for device in devices():
            with self.subTest(device=device):
                worst = device_f1(device)
                self.assertGreater(worst["seeds"], 0)
                for name in ("rows", "leave", "twist", "known", "frame"):
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

    def test_one_frame_tells_two_apart(self):
        """``long_chain(106)`` as main runs it: the walk's rows with the walk's leave stay within ``BOUND n u`` of
        ``|Z| |du|``; with a leave by main's float32 factor they do not. The invariant guards a placement that converts
        its rows without its leave."""
        for device in devices():
            with self.subTest(device=device):
                frames = frames_on_chain(106, device)
                self.assertLessEqual(frames["one_frame"], BOUND, f"{frames}")
                self.assertGreater(frames["two_frames"], BOUND, f"{frames}")

    def test_one_frame_per_placement_cpu(self):
        """Every CPU catalogue route: its rows within ``BOUND n u kappa`` of the walk's, and ``J dv`` against ``Z^T du``
        and its leave against the walk's within their rounding."""
        for physics, route in catalogue():
            with self.subTest(problem=physics.name, route=route.name):
                solver, joint_q, capture = run_route(physics, route)
                result = placement_frames(solver, joint_q, capture)
                for name in ("rows", "frame", "tight", "leave", "kernel"):
                    self.assertLessEqual(result[name], BOUND, f"{name}: {result}")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA placements")
    def test_f1_on_the_cuda_placements(self):
        """Z0's F1 and one frame on the CUDA placements whose rows a store holds: the general worlds' footprints, their
        ``J dv`` against ``Z^T du`` and their leave, and the worlds whitened in the kernel through the walk's response
        of their impulses."""
        covered, done = {"general": 0, "in_kernel": 0}, set()
        for physics, route in cuda_placements():
            if route.name in done:
                continue
            with self.subTest(problem=physics.name, route=route.name):
                solver, joint_q, capture = run_route(physics, route, device="cuda:0")
                result = placement_frames(solver, joint_q, capture)
                for name in ("rows", "frame", "tight", "leave", "kernel"):
                    self.assertLessEqual(result[name], BOUND, f"{name}: {result}")
                if checked(result):
                    done.add(route.name)
                for name in covered:
                    covered[name] += result[name]
        # Both forms were checked: worlds the general sweep solves from the store, and worlds whitened in a kernel.
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
