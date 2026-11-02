# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The walk is the tree factor (Z_UNION §1.2): its falsification tests on the CPU (Z_UNION §5, Z0).

:class:`.tree_factor.TreeWalk` walks a six-number state up the tree, one coordinate at a time, in local frames. Its
data ``(s, ut, ell)`` is the leaf-first factor; its footprints are the whitened rows, the walk run down is the leave,
and two walks meeting at their lowest common ancestor give the Gram. No step calls it; these tests hold it to the
factor and to the claims built on it:

- **identities**, float64: the walk against the factor of the independent float64 mass matrix
  (:mod:`.feather_pgs_mass_reference`) on every catalogue model: ``L^-1``, the leave, a row from its seed, the meeting;
- **F1**: on every catalogue problem and both CPU routes, the walk's ``Z`` against the step's packed ``Z`` (and its
  free bodies' rows), and the walk's leave of the step's ``du`` against the step's velocity change;
- **F2**: on ``long_chain(N)`` without armature, the float32 walk against the float64 truth, beside main's float32
  factor with the conditioning armature it adds;
- **F2b**: main's factor on the same chains with that armature off;
- **F4, F5**: row entries per seed in a frame cut every 32 coordinates, and merged across seed-free stretches.

``python -m newton.tests.test_feather_pgs_walk report`` prints the measured tables.
"""

import dataclasses
import sys
import unittest
import warnings
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import tree_factor
from newton._src.solvers.feather_pgs.conditioning import ArmatureFloor
from newton._src.solvers.feather_pgs.kernels import PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT
from newton._src.solvers.feather_pgs.placement import Placement
from newton._src.solvers.feather_pgs.tree_factor import TreeWalk, local_frames
from newton.solvers import SolverFeatherPGS
from newton.tests import feather_pgs_scenes as scenes
from newton.tests import test_feather_pgs_conformance as conformance
from newton.tests.feather_pgs_law_harness import dense_factor32
from newton.tests.feather_pgs_mass_reference import MassReference

ROUNDING = 2.0**-24
BOUND = 16.0
"""The rounding bounds' multiple of ``n u`` (as :mod:`.test_feather_pgs_tree_invariants`)."""
STEPS = 8
CHAINS = (8, 32, 64, 106, 200, 300, 1000)
BISECT = (72, 80, 88, 96)
"""F2b's lengths between the last that main factors without its armature and the first that it does not."""
SEGMENT = 32
REVERSED = ("gripper_point", "hand_point", "legged_point", "pinch_twist_point_torsion")
"""F1's problems of several worlds, dense and free bodies' rows, run again with every row table reversed."""


# ---------------------------------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------------------------------


def jacobian(walk: TreeWalk, body: int, wrench: np.ndarray) -> np.ndarray:
    """``J^T`` of rows seeded by ``wrench`` at ``body`` (``[6]`` or ``[6, m]``): ``s_c^T w`` at every coordinate of
    the body's root path, the wrench carried rigidly between frames (no joint takes any of it)."""
    w = np.asarray(wrench, dtype=np.float64)
    x = np.zeros((walk.s.shape[0], *w.shape[1:]))
    for b in walk.ancestors(body):
        x[walk.first[b] : walk.first[b + 1]] = walk.s[walk.first[b] : walk.first[b + 1]].astype(np.float64) @ w
        d = walk.offset[b].astype(np.float64)
        w = np.concatenate((w[:3], w[3:] - np.cross(d, w[:3], axis=0)))
    return x


def factor_matrix(walk: TreeWalk) -> np.ndarray:
    """``P^T L P`` by coordinate, float64: the inverse of the walk's footprints of every unit covector."""
    inverse, _ = walk.up(covector=np.eye(walk.s.shape[0]))
    return np.linalg.inv(inverse.astype(np.float64))


def contact_wrenches(frame_offset: np.ndarray, normal=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Unit forces along ``normal`` and two tangents at a contact ``frame_offset`` from a body's frame: the wrenches
    ``(f, r x f)`` at the frame, one column each."""
    n = np.asarray(normal, dtype=np.float64)
    n /= np.linalg.norm(n)
    tangents = np.linalg.svd(n[None, :])[2][1:]
    forces = np.stack((n, *tangents), axis=1)
    return np.concatenate((forces, np.cross(frame_offset[:, None], forces, axis=0)), axis=0)


def relative(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, np.float64) - b) / max(np.linalg.norm(b), 1e-300))


# ---------------------------------------------------------------------------------------------------------------------
# Identities: the walk is the factor (float64, against the independent mass matrix)
# ---------------------------------------------------------------------------------------------------------------------


def identities(model, joint_q: np.ndarray, rng) -> dict:
    """The walk against the dense float64 factor of ``H + D`` (``D`` random armature) of every articulation: its
    ``L^-1`` (footprints of unit covectors), its leave, a body seed's row and the meeting of two rows."""
    reference = MassReference(model)
    worst = dict.fromkeys(("Linv", "leave", "seed", "meet", "twist"), 0.0)
    for art in range(model.articulation_count):
        H = reference.mass_matrix(joint_q, art)
        n = H.shape[0]
        if n == 0:
            continue
        D = rng.uniform(0.0, 1.0e-3, n)
        walk = TreeWalk(*local_frames(model, joint_q, art), D)
        L = np.linalg.cholesky((H + np.diag(D))[::-1, ::-1])
        footprints, _ = walk.up(covector=np.eye(n))
        expected = np.linalg.solve(L, np.eye(n)[::-1])[::-1]
        worst["Linv"] = max(worst["Linv"], relative(footprints, expected))
        u = rng.normal(size=(n, 3))
        qd, twist = walk.down(u)
        worst["leave"] = max(worst["leave"], relative(qd, np.linalg.solve(L.T, u[::-1])[::-1]))
        bodies = walk.parent.size
        k, l = int(rng.integers(bodies)), int(rng.integers(bodies))
        w_k, w_l = rng.normal(size=(6, 2)), rng.normal(size=(6, 3))
        x = jacobian(walk, k, w_k)
        z_k, _ = walk.up(k, w_k)
        worst["seed"] = max(worst["seed"], relative(z_k, np.linalg.solve(L, x[::-1])[::-1]))
        z_l, _ = walk.up(l, w_l)
        worst["meet"] = max(worst["meet"], relative(walk.meet(k, w_k, l, w_l), z_k.T @ z_l))
        # The leave's state at a body is its twist: Gamma_k^T u.
        gamma, _ = walk.up(k, np.eye(6))
        worst["twist"] = max(worst["twist"], relative(twist[k], gamma.T @ u))
    return worst


# ---------------------------------------------------------------------------------------------------------------------
# F1: the catalogue's rows and leaves
# ---------------------------------------------------------------------------------------------------------------------


def catalogue() -> list:
    """Every problem of the CPU catalogue (:func:`.test_feather_pgs_conformance.cpu_groups`) with both CPU routes:
    the reference's (dense storage, the general placement) and the default's (the group's options)."""
    return [(group.physics, route) for group in conformance.cpu_groups() for route in (group.reference, *group.routes)]


class Capture:
    """The rows a step builds, around their whitening (:meth:`.Placement.pack_rows`, the free bodies'
    :meth:`.Placement.whiten_free_bodies`): each call keeps its last values."""

    def __init__(self, solver):
        self.solver = solver
        self.J = self.Z = self.mf = None
        placement = solver.placement
        pack, free = placement.pack_rows, placement.whiten_free_bodies

        def pack_rows(*args, **kwargs):
            J = placement.packed_Z.numpy().copy()
            result = pack(*args, **kwargs)
            self.J, self.Z = J, placement.packed_Z.numpy().copy()
            self.sides = placement.row_sides.numpy().copy()
            self.counts = placement.constraint_count.numpy().copy()
            return result

        def whiten_free_bodies(*args, **kwargs):
            J = (placement.mf_Z_a.numpy().copy(), placement.mf_Z_b.numpy().copy())
            result = free(*args, **kwargs)
            Z = (placement.mf_Z_a.numpy().copy(), placement.mf_Z_b.numpy().copy())
            self.mf = (J, Z)
            return result

        placement.pack_rows = pack_rows
        placement.whiten_free_bodies = whiten_free_bodies

    def finish(self):
        """The step's end: its final rows and impulses (after any elimination), the predictor and the velocity."""
        placement, solver = self.solver.placement, self.solver
        self.Z_final = placement.packed_Z.numpy().astype(np.float64)
        self.impulses = placement.impulses.numpy().astype(np.float64)
        # The general placement's one writer (FL3.3) runs no pack: its rows' sides and counts are the step's.
        self.sides = placement.row_sides.numpy().copy()
        self.counts = placement.constraint_count.numpy().copy()
        self.mf_final = None
        if self.mf is not None:
            self.mf_final = (
                placement.mf_Z_a.numpy().astype(np.float64),
                placement.mf_Z_b.numpy().astype(np.float64),
                placement.mf_impulses.numpy().astype(np.float64),
                placement.mf_constraint_count.numpy().copy(),
                placement.mf_body_a.numpy().copy(),
                placement.mf_body_b.numpy().copy(),
                placement.mf_dof_a.numpy().copy(),
                placement.mf_dof_b.numpy().copy(),
                placement.mf_row_type.numpy().copy(),
            )
        self.v_hat = solver.v_hat.numpy().astype(np.float64)
        self.v_out = solver.v_out.numpy().astype(np.float64)
        # The line of each family's store where a world's rows start (its row tables), as every step read takes it.
        self.dense_first = placement.dense_first.numpy().copy()
        self.mf_first = placement.mf_first.numpy().copy()


def pipeline_options(scene) -> dict:
    """The collision pipeline's options a scene asks for (as :func:`.test_feather_pgs_conformance.simulate`)."""
    options = {}
    if scene.rigid_contact_max is not None:
        options["rigid_contact_max"] = scene.rigid_contact_max
    if scene.contact_matching is not None:
        options["contact_matching"] = scene.contact_matching
    if scene.deterministic_contacts:
        options["deterministic"] = True
    return options


def run_route(physics, route, device="cpu", steps=STEPS):
    """Step ``physics`` through ``route``; return the solver, the last step's input coordinates (as the forest holds
    them) and its :class:`Capture`."""
    scene = physics.scene
    model = scene.build(device)
    pipeline = newton.CollisionPipeline(model, **pipeline_options(scene))
    contacts = pipeline.contacts()
    options = {**conformance.BASE_OPTIONS, **scene.solver_options, **physics.options(physics.iterations)}
    options.update(route.options)
    # The rows' material law is the sweep's: these contacts hold no per-contact material.
    options.pop("contact_compliance", None)
    with conformance.route_context(route), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver = SolverFeatherPGS(model, **options)
        capture = Capture(solver)
        state, out, control = model.state(), model.state(), model.control()
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        joint_q_in = None
        for step in range(steps):
            state.clear_forces()
            if scene.control is not None:
                scene.control(model, control, state, step)
            pipeline.collide(state, contacts)
            solver.step(state, out, control, contacts, scene.dt)
            joint_q_in = solver.cut.mirror(state).joint_q.numpy().copy()
            state, out = out, state
        capture.finish()
    return solver, joint_q_in, capture


@dataclasses.dataclass
class RowCheck:
    """F1 on one run: the worst of each measure, and the counts behind them.

    A float32 whitening of a row is exact for a factor perturbed by ``n u`` of its norm, so it is within rounding of the
    walk's when ``|Z_step - Z_walk| / |Z_walk| <= BOUND n u kappa``, ``kappa`` the walk's factor's condition number
    (two-norms); the leave likewise, with the publication's rounding of ``v_hat + dv`` added. A measure's ratio to its
    rounding passes at ``BOUND``."""

    rows: int = 0
    sides: int = 0
    seeded: int = 0
    free_rows: int = 0
    kappa: float = 0.0
    """The largest condition number of an articulation's factor."""
    forward_step: float = 0.0
    """Worst ``|Z_step - Z_walk64| / |Z_walk64|`` of a side."""
    forward_walk32: float = 0.0
    """The same of the float32 walk."""
    ratio: float = 0.0
    """Worst ``|Z_step - Z_walk64| / |Z_walk64|`` over ``n u kappa`` (pass: at most ``BOUND``)."""
    writer: float = 0.0
    """Worst ``|J(seed) - J| / |J|`` of a row's recovered seed: how near the writer's float32 row is a body seed."""
    seed: float = 0.0
    """Worst row walked from its seed against the walk of the covector that seed writes (float64)."""
    meet: float = 0.0
    """Worst meeting at the LCA against the seeds' footprints' dot product (float64)."""
    leave: float = 0.0
    """Worst ``|dv_walk - dv_step| / |dv_walk|`` of an articulation's leave."""
    leave_ratio: float = 0.0
    """The same over its rounding, ``n u kappa`` and the publication's (pass: at most ``BOUND``)."""
    ball_worlds: int = 0
    """Worlds whose speed ball acted: their ``du`` is a law's, not ``sum Z lambda``, so their leave is not compared."""
    operators: tuple = ()


def _seed(walk: TreeWalk, x: np.ndarray):
    """The body seed of a row ``x``: the deepest body it reaches, the wrench there nearest to writing it and how near
    (``|J(w) - x| / |x|``), or ``None`` where no wrench at one body writes it (a joint's covector; a row of two bodies of
    one tree)."""
    touched = np.flatnonzero(x != 0.0)
    if not touched.size:
        return None
    body = int(np.searchsorted(walk.first, touched[-1], side="right") - 1)
    G = jacobian(walk, body, np.eye(6))
    path = np.flatnonzero(np.any(G != 0.0, axis=1))
    if np.any(x[np.setdiff1d(np.arange(x.size), path)] != 0.0):
        return None
    w, *_ = np.linalg.lstsq(G[path], x[path], rcond=None)
    near = np.linalg.norm(G @ w - x) / np.linalg.norm(x)
    if near > 1.0e-5:
        return None
    return body, w, near


def check_rows(solver, joint_q: np.ndarray, capture: Capture, rng) -> RowCheck:
    """F1 on one run: every dense row's sides and every free-body row, then the leave of the step's ``du``."""
    out = RowCheck(operators=tuple(type(group.sparse).__name__ for group in solver.factor.held.groups))
    forest, placement = solver.forest, solver.placement
    sizes = forest.articulation_response_dof_count.numpy()
    offsets = forest.articulation_world_dof_offset.numpy()
    starts = forest.articulation_dof_start.numpy()
    art_world = forest.art_to_world.numpy()
    dense = bool(placement.packed_dense)
    walks = {}

    def walk_of(art):
        if art not in walks:
            walk = solver.factor.walk(joint_q, art)
            kappa = float(np.linalg.cond(factor_matrix(walk)))
            walks[art] = (walk, solver.factor.walk(joint_q, art, dtype=np.float32), kappa)
            out.kappa = max(out.kappa, kappa)
        return walks[art]

    def compare(art, x, z_step, seeds):
        walk, walk32, kappa = walk_of(art)
        z64, _ = walk.up(covector=x)
        norm = np.linalg.norm(z64)
        out.sides += 1
        if norm == 0.0:
            out.forward_step = max(out.forward_step, float(np.abs(z_step).max()))
            return
        z32, _ = walk32.up(covector=x.astype(np.float32))
        forward = np.linalg.norm(z_step - z64) / norm
        out.forward_step = max(out.forward_step, forward)
        out.forward_walk32 = max(out.forward_walk32, np.linalg.norm(z32 - z64) / norm)
        out.ratio = max(out.ratio, forward / (x.size * ROUNDING * kappa))
        seed = _seed(walk, x)
        if seed is not None:
            body, w, near = seed
            out.seeded += 1
            out.writer = max(out.writer, near)
            z_seed, _ = walk.up(body, w)
            written, _ = walk.up(covector=jacobian(walk, body, w))
            out.seed = max(out.seed, np.linalg.norm(z_seed - written) / np.linalg.norm(written))
            seeds.setdefault(art, []).append((body, w, z_seed))

    if capture.J is not None:
        for world, count in enumerate(capture.counts):
            seeds, line = {}, int(capture.dense_first[world])
            for row in range(int(count)):
                out.rows += 1
                first = 0
                for art in capture.sides[line, row]:
                    if art < 0:
                        continue
                    n, offset = int(sizes[art]), int(offsets[art])
                    at = offset if dense else first
                    first += n
                    entries = slice(at, at + n)
                    x = capture.J[line, row, entries].astype(np.float64)
                    compare(art, x, capture.Z[line, row, entries].astype(np.float64), seeds)
            # Two walks meet at their LCA: pairs of seeded sides of one tree.
            for art, sided in seeds.items():
                walk = walks[art][0]
                for _ in range(min(8, len(sided) ** 2)):
                    (k, w_k, z_k), (l, w_l, z_l) = sided[rng.integers(len(sided))], sided[rng.integers(len(sided))]
                    met = walk.meet(k, w_k, l, w_l)
                    out.meet = max(out.meet, abs(met - z_k @ z_l) / (np.linalg.norm(z_k) * np.linalg.norm(z_l)))
    body_art = forest.body_to_articulation.numpy()
    # The walked free-body rows form no J (walk_rows): test_feather_pgs_walk_rows holds them to their J v_hat.
    if capture.mf is not None and not getattr(solver.placement, "walks", False):
        (J_a, J_b), (Z_a, Z_b) = capture.mf
        _, _, _, counts, body_a, body_b, _, _, kinds = capture.mf_final
        for world, count in enumerate(counts):
            line = int(capture.mf_first[world])
            for row in range(int(count)):
                if kinds[line, row] == PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT:
                    continue
                out.free_rows += 1
                for body, J, Z in ((body_a, J_a, Z_a), (body_b, J_b, Z_b)):
                    if body[line, row] >= 0:
                        art = int(body_art[body[line, row]])
                        compare(art, J[line, row].astype(np.float64), Z[line, row].astype(np.float64), {})
    # The leave: the step's du is sum Z lambda over its final rows; each articulation leaves it once.
    D = int(forest.max_world_dofs)
    du = np.zeros((int(forest.world_count), max(D, 1)))
    # Every stored row's footprints, the pack's or the one writer's (FL3.3, whose rows form no J).
    if capture.J is not None or getattr(solver.placement, "walks", False):
        for world, count in enumerate(capture.counts):
            line = int(capture.dense_first[world])
            for row in range(int(count)):
                lam = capture.impulses[line, row]
                if dense:
                    du[world, :D] += capture.Z_final[line, row, :D] * lam
                    continue
                first = 0
                for art in capture.sides[line, row]:
                    if art < 0:
                        continue
                    n, offset = int(sizes[art]), int(offsets[art])
                    du[world, offset : offset + n] += capture.Z_final[line, row, first : first + n] * lam
                    first += n
    ball = np.zeros(du.shape[0], dtype=bool)
    if capture.mf_final is not None:
        Z_a, Z_b, lam, counts, body_a, body_b, dof_a, dof_b, kinds = capture.mf_final
        for world, count in enumerate(counts):
            line = int(capture.mf_first[world])
            for row in range(int(count)):
                if kinds[line, row] == PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT:
                    # A free body's speed row is its speed ball's: a magnitude against the body's velocity as each
                    # visit finds it, a law and not a row, so its world's du is not sum Z lambda.
                    ball[world] |= lam[line, row] != 0.0
                    continue
                for dof, Z in ((dof_a, Z_a), (dof_b, Z_b)):
                    if dof[line, row] >= 0:
                        du[world, dof[line, row] : dof[line, row] + 6] += Z[line, row] * lam[line, row]
    out.ball_worlds = int(np.count_nonzero(ball))
    for art in range(solver.model.articulation_count):
        n, world = int(sizes[art]), int(art_world[art])
        if n <= 0 or world < 0 or ball[world] or not np.any(du[world, offsets[art] : offsets[art] + n]):
            continue
        walk, _, kappa = walk_of(art)
        dv, _ = walk.down(du[world, offsets[art] : offsets[art] + n])
        dofs = int(starts[art]) + np.arange(n)
        step = capture.v_out[dofs] - capture.v_hat[dofs]
        norm = np.linalg.norm(dv)
        relative_error = np.linalg.norm(dv - step) / norm
        published = 2.0 * ROUNDING * (np.linalg.norm(capture.v_out[dofs]) + np.linalg.norm(capture.v_hat[dofs])) / norm
        out.leave = max(out.leave, relative_error)
        out.leave_ratio = max(out.leave_ratio, relative_error / (n * ROUNDING * kappa + published))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# F2, F2b: long chains in float32
# ---------------------------------------------------------------------------------------------------------------------


def ground_contact(model, poses: np.ndarray, joint: int) -> tuple[int, np.ndarray]:
    """The ground's contact under the last sphere of ``joint``'s child (``long_chain``'s links and head): the walk's
    body and the wrenches of a normal and two tangents at its frame."""
    body = int(model.joint_child.numpy()[joint])
    shape = int(np.flatnonzero(model.shape_body.numpy() == body)[-1])
    centre = transform_point(poses[body], model.shape_transform.numpy()[shape, :3].astype(np.float64))
    point = centre - np.array([0.0, 0.0, float(model.shape_scale.numpy()[shape, 0])])
    parent = int(model.joint_parent.numpy()[joint])
    if int(model.joint_type.numpy()[joint]) == int(newton.JointType.FREE):
        frame = transform_point(poses[body], model.body_com.numpy()[body].astype(np.float64))
    else:
        anchor = model.joint_X_p.numpy()[joint, :3].astype(np.float64)
        frame = transform_point(poses[parent], anchor) if parent >= 0 else anchor
    return joint - int(model.articulation_start.numpy()[0]), contact_wrenches(point - frame)


def newton_state_body_q(model, joint_q: np.ndarray) -> np.ndarray:
    """The body poses at ``joint_q`` (Newton's forward kinematics, float32, as float64)."""
    state = model.state()
    state.joint_q.assign(np.asarray(joint_q, dtype=np.float32))
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return state.body_q.numpy().astype(np.float64)


def transform_point(pose: np.ndarray, point: np.ndarray) -> np.ndarray:
    x, y, z, w = pose[3:]
    q = np.array([x, y, z])
    t = 2.0 * np.cross(q, point)
    return pose[:3] + point + w * t + np.cross(q, t)


@dataclasses.dataclass
class Chain:
    """F2 at one length: errors against the float64 truth (no armature) of a tip row's ``Z``, a leave and the tip
    impulse's response ``dv = H^-1 J^T``."""

    links: int
    coordinates: int
    condition: float
    """``cond(H)``, float64."""
    armature_coordinates: int
    armature_largest: float
    armature_alone: float
    """The armature's own change to ``dv``, in float64: main's physics against the truth."""
    truth_floor: float
    """The float64 walk against the float64 Cholesky of the independent mass matrix (the truth's own spread)."""
    main_factor: float
    """Main's float32 factor's backward error ``|L L^T - P (H + R + K) P^T| / |H + R + K|``."""
    main: dict
    """Main's float32 factor, applied in float32 (and ``"dv64"``: applied in float64)."""
    walk32: dict
    walk32_poses: dict
    """The float32 walk whose local frames come from Newton's float32 world poses, main's forward kinematics."""
    walk32_pivots_positive: bool
    walk32_pivots: float
    """The float32 walk's worst pivot ``ell^2`` against the truth's, relative."""
    main_finite: bool


def chain_precision(links: int, rng) -> Chain:
    """Main's float32 factor of ``long_chain(links, armature=0)``, with the conditioning armature it adds, against the
    float32 walk in local frames with none, both against the float64 walk with none (the truth)."""
    scene = scenes.long_chain(links=links, armature=0.0)
    model = scene.build("cpu")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver = SolverFeatherPGS(model, dense_max_constraints=8)
    state, out, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    joint_q = state.joint_q.numpy().astype(np.float64)
    solver.step(state, out, control, None, scene.dt)
    n = links + 6
    added = solver.factor.armature[:n].astype(np.float64) - model.joint_armature.numpy()[:n]
    H = MassReference(model).mass_matrix(joint_q, 0)
    L32 = dense_factor32(H + np.diag(solver.factor.diagonal(0)))
    truth = TreeWalk(*local_frames(model, joint_q, 0), np.zeros(n))
    walk32 = TreeWalk(*local_frames(model, joint_q, 0), np.zeros(n), dtype=np.float32)
    body, W = ground_contact(model, newton_state_body_q(model, joint_q), links)
    x = jacobian(truth, body, W)
    z64, _ = truth.up(body, W)
    dv64, _ = truth.down(z64)
    u = rng.normal(size=n)
    v64, _ = truth.down(u)
    floor = relative(dv64, np.linalg.solve(H, x))
    physics = solver.factor.walk(joint_q, 0)  # main's own physics: its armature and drives, in float64
    dv_physics, _ = physics.down(physics.up(body, W)[0])
    A = (H + np.diag(solver.factor.diagonal(0)))[::-1, ::-1]
    L64 = L32.astype(np.float64)

    from scipy.linalg import solve_triangular

    def solve(lower, rhs, transpose=False):
        return solve_triangular(lower, rhs, lower=True, trans=1 if transpose else 0, check_finite=False)

    with np.errstate(all="ignore"):
        x32 = x.astype(np.float32)[::-1]
        z_main = solve(L32, x32)
        main = {
            "Z": relative(z_main[::-1], z64),
            "leave": relative(solve(L32, u.astype(np.float32)[::-1], True)[::-1], v64),
            "dv": relative(solve(L32, z_main, True)[::-1], dv64),
            "dv64": relative(solve(L64, solve(L64, x[::-1]), True)[::-1], dv64),
        }

        def errors(walk):
            z32, _ = walk.up(body, W.astype(np.float32))
            return {
                "Z": relative(z32, z64),
                "leave": relative(walk.down(u.astype(np.float32))[0], v64),
                "dv": relative(walk.down(z32)[0], dv64),
            }

        mine = errors(walk32)
        poses = newton_state_body_q(model, joint_q)
        from_poses = errors(TreeWalk(*local_frames(model, joint_q, 0, poses), np.zeros(n), dtype=np.float32))
    return Chain(
        links=links,
        coordinates=n,
        condition=float(np.linalg.cond(H)),
        armature_coordinates=int(np.count_nonzero(added > 0.0)),
        armature_largest=float(added.max(initial=0.0)),
        armature_alone=relative(dv_physics, dv64),
        truth_floor=floor,
        main_factor=float(np.abs(L64 @ L64.T - A).max() / np.abs(A).max()) if np.all(np.isfinite(L64)) else np.inf,
        main=main,
        walk32=mine,
        walk32_poses=from_poses,
        walk32_pivots_positive=bool(np.all(np.isfinite(walk32.ell)) and np.all(walk32.ell > 0.0)),
        walk32_pivots=float(np.max(np.abs(walk32.pivots.astype(np.float64) / truth.pivots - 1.0))),
        main_finite=bool(np.all(np.isfinite(L32))),
    )


def _no_conditioning(model, armature, joint_forest):
    return np.array(armature, dtype=np.float32, copy=True), ArmatureFloor(0, 0, 0.0, 0)


@dataclasses.dataclass
class Unconditioned:
    """F2b at one length: main's factor with the conditioning armature off."""

    links: int
    first_pose_finite: bool
    failed_pivots: int
    """Main's non-finite float32 pivots at the first pose."""
    least_pivot_ratio: float
    """Main's least finite float32 pivot ``L_ii^2`` over its diagonal at the first pose (the floor is ``4 n u``)."""
    steps_finite: int
    """Steps of the scene stepped before the factor or the velocity went non-finite (all: none did)."""
    steps: int


def unconditioned(links: int, steps: int = 30) -> Unconditioned:
    """Step ``long_chain(links, armature=0)`` on the ground with the conditioning armature off; record where the
    float32 factor fails."""
    scene = scenes.long_chain(links=links, armature=0.0)
    model = scene.build("cpu")
    with mock.patch.object(tree_factor, "conditioning_armature", _no_conditioning), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver = SolverFeatherPGS(model, dense_max_constraints=3 * links + 32)
    np.testing.assert_array_equal(solver.factor.armature, model.joint_armature.numpy())
    pipeline = newton.CollisionPipeline(model, **pipeline_options(scene))
    contacts = pipeline.contacts()
    state, out, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    first, finite = None, 0
    for step in range(steps):
        state.clear_forces()
        pipeline.collide(state, contacts)
        H = MassReference(model).mass_matrix(state.joint_q.numpy(), 0)
        solver.step(state, out, control, contacts, scene.dt)
        L = dense_factor32(H + np.diag(solver.factor.diagonal(0))).astype(np.float64)
        ok = bool(np.all(np.isfinite(np.tril(L))) and np.all(np.isfinite(out.joint_qd.numpy())))
        if first is None:
            pivots = np.diag(L) ** 2
            ratios = pivots / np.diag(H)[::-1]
            first = (
                ok,
                int(np.count_nonzero(~np.isfinite(pivots))),
                float(np.nanmin(np.where(np.isfinite(ratios), ratios, np.nan))),
            )
        if not ok:
            break
        finite = step + 1
        state, out = out, state
    return Unconditioned(links, first[0], first[1], first[2], finite, steps)


# ---------------------------------------------------------------------------------------------------------------------
# F4, F5: row entries per seed in the segmented and the merged frames
# ---------------------------------------------------------------------------------------------------------------------


def segments(walk: TreeWalk, size: int = SEGMENT) -> list:
    """A chain's coordinates cut at bodies every ``size`` coordinates from the root: ``(coordinates, last body)``."""
    cuts, coordinates, last = [], [], -1
    for b in range(walk.parent.size):
        own = list(range(walk.first[b], walk.first[b + 1]))
        if coordinates and len(coordinates) + len(own) > size:
            cuts.append((coordinates, last))
            coordinates = []
        coordinates += own
        last = b
    if coordinates:
        cuts.append((coordinates, last))
    return cuts


def segmented_frame(walk: TreeWalk, size: int = SEGMENT) -> np.ndarray:
    """The orthonormal frame that cuts the chain every ``size`` coordinates (Z_UNION §2.3c): each segment rotated by the
    complete QR of the footprints of its last body's six unit wrenches, interface first."""
    n = walk.s.shape[0]
    O = np.zeros((n, n))
    for coordinates, last in segments(walk, size):
        gamma, _ = walk.up(last, np.eye(6))
        Q, _ = np.linalg.qr(gamma[coordinates].astype(np.float64), mode="complete")
        O[np.ix_(coordinates, coordinates)] = Q
    return O


def entries(Z: np.ndarray, threshold: float) -> np.ndarray:
    """Each row's entries above ``threshold`` of its largest (rows are columns of ``Z``)."""
    return np.sum(np.abs(Z) > threshold * np.abs(Z).max(axis=0), axis=0)


@dataclasses.dataclass
class Counts:
    """F4 and F5 on ``long_chain(links)``."""

    links: int
    tip_rows: int
    tip_static: dict
    """Tip rows' entries in the frame cut every 32 coordinates, ``{threshold: (mean, max)}``."""
    structure: int
    """Rows with more entries than the frame allows (their own segment, then six per segment above): none."""
    tip_merged: tuple
    """Tip rows' entries with the seed-free stretch merged: ``(mean, max)``, the window and the stretch's rank."""
    tip_residue: float
    """A tip row's footprints above the merged stretch against ``Gamma_c`` times its carried state (two walks of one
    push): the walk's own rounding, below which entries are residue."""
    tip_gram: float
    ground_rows: int
    ground_main: tuple
    """Main's entries per ground row: ``(mean support, stored width)``."""
    ground_segmented: tuple
    """Segmented entries per ground row (1e-12): ``(mean, max)``."""
    ground_gram: float


def counts(links: int = 300) -> Counts:
    scene = scenes.long_chain(links=links)
    model = scene.build("cpu")
    joint_q = model.joint_q.numpy().astype(np.float64)
    walk = TreeWalk(*local_frames(model, joint_q, 0), model.joint_armature.numpy()[: links + 6])
    poses = newton_state_body_q(model, joint_q)

    def rows(joints):
        seeds = [ground_contact(model, poses, j) for j in joints]
        Z = np.concatenate([walk.up(b, W)[0] for b, W in seeds], axis=1)
        return seeds, Z

    O = segmented_frame(walk)
    cuts = segments(walk)
    segment_of = np.zeros(walk.parent.size, dtype=np.int64)
    for index, (coordinates, _) in enumerate(cuts):
        segment_of[np.searchsorted(walk.first, coordinates, side="right") - 1] = index

    def beyond(seeds, Zs):
        """Rows whose entries pass their own segment's coordinates plus six for each segment above it."""
        allowed = np.repeat([len(cuts[segment_of[b]][0]) + 6 * segment_of[b] for b, _ in seeds], 3)
        return int(np.count_nonzero(entries(Zs, 1e-12) > allowed))

    last = walk.parent.size - 1
    tip_seeds, Z_tip = rows([last - 1, last])
    tip_static = {
        t: (float(entries(O.T @ Z_tip, t).mean()), int(entries(O.T @ Z_tip, t).max())) for t in (1e-12, 1e-10)
    }
    # Merged: the window below the shallowest seeded body's parent c, plus the six numbers that cross c.
    c = min(b for b, _ in tip_seeds) - 1
    stretch = [k for b in walk.ancestors(c) for k in range(walk.first[b], walk.first[b + 1])]
    window = np.setdiff1d(np.arange(walk.s.shape[0]), stretch)
    gamma, _ = walk.up(c, np.eye(6))
    _, R = np.linalg.qr(gamma[stretch])
    merged = np.concatenate(
        [np.concatenate((walk.up(b, W)[0][window], R @ walk.carry(b, W, c)), axis=0) for b, W in tip_seeds], axis=1
    )
    merged_counts = entries(merged, 1e-12)
    rank = int(np.linalg.matrix_rank(gamma[stretch], tol=1e-10 * np.abs(gamma[stretch]).max()))
    b, W = tip_seeds[0]
    residue = relative(walk.up(b, W)[0][stretch], gamma[stretch] @ walk.carry(b, W, c))
    gram = np.abs(merged.T @ merged - Z_tip.T @ Z_tip).max() / np.abs(Z_tip.T @ Z_tip).max()
    # F5: a contact under every link's sphere and the head's.
    ground, Z_ground = rows(range(walk.parent.size))
    support = np.array(
        [sum(walk.first[a + 1] - walk.first[a] for a in walk.ancestors(b)) for b, W in ground for _ in range(3)]
    )
    Zs = O.T @ Z_ground
    seg = entries(Zs, 1e-12)
    return Counts(
        links=links,
        tip_rows=Z_tip.shape[1],
        tip_static=tip_static,
        structure=beyond(tip_seeds, O.T @ Z_tip) + beyond(ground, Zs),
        tip_merged=(float(merged_counts.mean()), int(merged_counts.max()), int(window.size), rank),
        tip_residue=residue,
        tip_gram=float(gram),
        ground_rows=Z_ground.shape[1],
        ground_main=(float(support.mean()), walk.s.shape[0]),
        ground_segmented=(float(seg.mean()), int(seg.max())),
        ground_gram=float(np.abs(Zs.T @ Zs - Z_ground.T @ Z_ground).max() / np.abs(Z_ground.T @ Z_ground).max()),
    )


# ---------------------------------------------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------------------------------------------


class TestWalkIdentities(unittest.TestCase):
    """The walk against the independent float64 factor, on every catalogue model at its first pose."""

    def test_catalogue_models(self):
        rng = np.random.default_rng(1101)
        seen = set()
        for physics, _ in catalogue():
            if physics.scene.name in seen:
                continue
            seen.add(physics.scene.name)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = SolverFeatherPGS(physics.scene.build("cpu")).model  # loops cut at a body
            with self.subTest(scene=physics.scene.name):
                worst = identities(model, model.joint_q.numpy(), rng)
                for name, value in worst.items():
                    self.assertLess(value, 1.0e-9, f"{name}")


_construct = Placement.__init__


def _reversed_lines(self, *args, **kwargs):
    """A placement whose row tables put every world's rows on another world's line
    (:mod:`.test_feather_pgs_row_lines`)."""
    _construct(self, *args, **kwargs)
    for table in (self.dense_first, self.mf_first):
        table.assign(np.arange(table.shape[0] - 1, -1, -1, dtype=np.int32))


class TestWalkF1(unittest.TestCase):
    """F1: the walk's rows and leave against the step's, on every catalogue problem and both CPU routes."""

    def check(self, result):
        self.assertLessEqual(result.ratio, BOUND, f"Z: {result}")
        self.assertLess(result.seed, 1.0e-9, f"seed: {result}")
        self.assertLess(result.meet, 1.0e-9, f"meet: {result}")
        self.assertLessEqual(result.leave_ratio, BOUND, f"leave: {result}")

    def test_rows_read_through_their_tables(self):
        """F1 with every row table reversed: the identity tables would hide a read that takes its line from the world,
        and a world's leave would then sum another world's rows."""
        for physics, route in catalogue():
            if physics.name not in REVERSED or route.name != "cpu":
                continue
            with self.subTest(problem=physics.name):
                with mock.patch.object(Placement, "__init__", _reversed_lines):
                    solver, joint_q, capture = run_route(physics, route)
                worlds = capture.dense_first.size
                self.assertGreater(worlds, 1)
                np.testing.assert_array_equal(capture.dense_first, np.arange(worlds)[::-1])
                self.check(check_rows(solver, joint_q, capture, np.random.default_rng(7)))


def _add_f1_tests():
    for physics, route in catalogue():

        def test(self, physics=physics, route=route):
            solver, joint_q, capture = run_route(physics, route)
            self.check(check_rows(solver, joint_q, capture, np.random.default_rng(7)))

        test.__doc__ = f"F1 on {physics.name} through {route.name}."
        setattr(TestWalkF1, f"test_{physics.name}_{route.name}", test)


_add_f1_tests()


class TestWalkF2(unittest.TestCase):
    """F2 and F2b on ``long_chain(N)`` without armature."""

    def test_f2_float32_walk_needs_no_armature(self):
        """The float32 walk builds every chain without armature, and from 32 links it answers a tip impulse at least
        twice as close to the float64 truth as main's float32 factor with the armature main adds."""
        rng = np.random.default_rng(3)
        for links in CHAINS:
            with self.subTest(links=links):
                chain = chain_precision(links, rng)
                self.assertTrue(chain.walk32_pivots_positive)
                self.assertLess(chain.walk32["dv"], 1.0e-5)
                if links >= 32:
                    self.assertGreater(chain.main["dv"], 2.0 * chain.walk32["dv"])

    def test_f2b_main_fails_without_its_armature(self):
        """Without the conditioning armature, main's float32 factor of the chain fails at its first pose from 80 links
        and holds below."""
        for links in (8, 64, 80, 300):
            with self.subTest(links=links):
                result = unconditioned(links, steps=2)
                self.assertEqual(result.first_pose_finite, links < 80)


class TestWalkCounts(unittest.TestCase):
    """F4 and F5 on ``long_chain(300)``."""

    def test_rows_per_seed(self):
        """Every row keeps the frame's structure; merged, a tip row has at most 10 entries; the chain on the ground
        has at most 0.45 of main's entries per row."""
        result = counts(300)
        self.assertEqual(result.structure, 0)
        self.assertLessEqual(result.tip_merged[1], 10)
        self.assertLessEqual(result.ground_segmented[0], 0.45 * result.ground_main[0])
        self.assertLess(result.tip_gram, 1.0e-12)
        self.assertLess(result.ground_gram, 1.0e-12)


def report() -> None:
    """Print every falsifier's measured table."""
    rng = np.random.default_rng(3)
    print("F2  long_chain(N), armature 0: errors against the float64 walk without armature (the truth); dv = H^-1 J^T")
    print(
        "     N   cond(H)  main armature: coords largest  dv alone | main factor | Z: main / walk32 | leave: main / "
        "walk32 | dv: main (f64 solves) / walk32  ratio / walk32 on float32 poses | walk32 pivots"
    )
    for links in CHAINS:
        c = chain_precision(links, rng)
        ratio = c.main["dv"] / c.walk32["dv"] if c.walk32["dv"] > 0 else float("nan")
        print(
            f"  {c.links:5d}  {c.condition:8.1e}  {c.armature_coordinates:5d} {c.armature_largest:9.2e} "
            f"{c.armature_alone:8.1e} | {c.main_factor:8.1e} | {c.main['Z']:8.1e} / {c.walk32['Z']:8.1e} | "
            f"{c.main['leave']:8.1e} / {c.walk32['leave']:8.1e} | {c.main['dv']:8.1e} ({c.main['dv64']:8.1e}) / "
            f"{c.walk32['dv']:8.1e} {ratio:8.1f} / {c.walk32_poses['dv']:8.1e} | positive {c.walk32_pivots_positive} "
            f"{c.walk32_pivots:8.1e} (truth floor {c.truth_floor:.0e})",
            flush=True,
        )
    print("F2b main without the conditioning armature")
    for links in sorted({*CHAINS, *BISECT}):
        u = unconditioned(links)
        print(
            f"  {u.links:5d}: first pose finite {u.first_pose_finite} ({u.failed_pivots} non-finite pivots), least "
            f"finite pivot/diagonal {u.least_pivot_ratio:9.2e} (floor {4 * (u.links + 6) * ROUNDING:8.1e}), finite "
            f"steps {u.steps_finite} of {u.steps}",
            flush=True,
        )
    c = counts(300)
    print(
        f"F4  tip of long_chain(300), {c.tip_rows} rows: static s=32 (mean, max) by threshold {c.tip_static}; rows past "
        f"the frame's structure {c.structure}; merged (mean, max, window, rank) {c.tip_merged}; residue "
        f"{c.tip_residue:.1e}; Gram {c.tip_gram:.1e}"
    )
    print(
        f"F5  long_chain(300) on the ground, {c.ground_rows} rows: main support mean {c.ground_main[0]:.1f} "
        f"(stored {c.ground_main[1]}); segmented mean {c.ground_segmented[0]:.1f}, max {c.ground_segmented[1]}; "
        f"ratio {c.ground_segmented[0] / c.ground_main[0]:.3f}; Gram {c.ground_gram:.1e}",
        flush=True,
    )
    print(f"F1  catalogue (Z and leave against their rounding n u kappa: pass at <= {BOUND:g})")
    for physics, route in catalogue():
        solver, joint_q, capture = run_route(physics, route)
        r = check_rows(solver, joint_q, capture, np.random.default_rng(7))
        print(
            f"  {physics.name:44s} {route.name:16s} rows {r.rows:4d} sides {r.sides:4d} seeded {r.seeded:4d} "
            f"free {r.free_rows:3d} kappa {r.kappa:8.1e} | Z: step {r.forward_step:8.1e} walk32 {r.forward_walk32:8.1e} "
            f"ratio {r.ratio:6.2f} | writer {r.writer:7.1e} seed {r.seed:7.1e} meet {r.meet:7.1e} | leave {r.leave:8.1e} "
            f"ratio {r.leave_ratio:6.2f} ball {r.ball_worlds} | {','.join(sorted(set(r.operators)))}",
            flush=True,
        )


if __name__ == "__main__":
    if sys.argv[1:2] == ["report"]:
        wp.config.quiet = True
        report()
    else:
        unittest.main()
