# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare the persistent patch mechanism with its full-capacity reference.

Random contact frames cover static and missing shapes, flipped pair orientation,
several contacts per pair, empty tails and frames past the arrays. The reference
copies the patch kernels as of 98401301 (:mod:`feather_pgs_patch_reference`);
pair keys compare through their decoded body and shape codes. A solver-level
check covers the route tables and warm-start scales the solver passes to the
patch state.

On CUDA the patch kernels and the reference are separate Warp modules, and NVRTC
contracts their float multiply-adds into fused multiply-adds. Which product of a
sum it leaves unfused can follow the order in which Warp emits a module's
kernels, and that order differs between processes that compile the same module,
so two caches can hold different code under one module hash. The projection
``error -= n * wp.dot(error, n)`` in ``_build_pair`` rounds either ``n.x * e.x``
or ``n.y * e.y`` on its own; over the CUDA differentials (NVRTC 12.9, sm_120)
the two choices moved phi and carried displacements by at most 1.12 * 2**-23
times the field's largest magnitude and changed no other field. On CUDA,
computed float payloads therefore compare with the reference within
``_FMA_ULPS`` float32 epsilons of the magnitude each row is computed from
(:func:`_assert_payload_equal`), so a contraction choice anywhere in the patch
arithmetic passes. The bound does not tell contraction from other codegen
changes of a few rounding errors: a fast-math build of the patch module exceeds
it only in phi, in one sequence. Integers and floats the kernels compute
without a product compare bitwise on every device, and every field compares
bitwise on the CPU. A twin state on another grid of the same compiled module
steps alongside on CUDA and must match the state under test bitwise
(:func:`_twin`), so races do not hide below the payload bound.

Captured builds prepare through ``_prepare_frame`` and eager builds through
``_prepare``, two kernels whose float code NVRTC chooses separately. Solver
graph tests therefore compare with an always-sort chain that prepares through
``_prepare_frame`` as well (:func:`_sort_every_build`), which tests frame
reuse bitwise on any device, and :func:`run_sorting_prepare_parity` compares
the two kernels on their own.
"""

import contextlib
import gc
import inspect
import itertools
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.core.live_sort import MAX_MERGES, SORT_CHUNK
from newton._src.sim import contacts as contacts_module
from newton._src.sim.contacts import note_contact_frame_write
from newton._src.solvers.feather_pgs import environment_flags as fpgs_flags
from newton._src.solvers.feather_pgs import friction_patches, path_selection
from newton._src.solvers.feather_pgs.friction_patches import (
    _BUILD_FIELDS,
    FrictionPatches,
    _FrictionPatchState,
    _PatchFrame,
    link_patch_routes,
    patch_routes,
)
from newton._src.solvers.feather_pgs.solver_feather_pgs import SolverFeatherPGS
from newton.tests.feather_pgs_law_harness import row_lines
from newton.tests.feather_pgs_patch_reference import SENTINEL, ReferencePatches, pair_codes, store_frame
from newton.tests.feather_pgs_patch_reference import link_patch_rows as reference_link_patch_rows
from newton.tests.test_feather_pgs_mixed_tree_free_body import _build as _arm_over_box

_PIECE_OFFSET = 0.04
# Float32 epsilons of its scale by which a computed payload may differ on CUDA
# (module docstring). The largest measured difference is 1.12 epsilons at this
# revision and 2.87, in phi, with the anchor choice of the patch-determinism
# work; 8 leaves room for a compile that fuses carried arithmetic differently in
# more places.
_FMA_ULPS = 8
_EPS32 = float(np.finfo(np.float32).eps)
# The twin's (threads, build_threads): grids of one block, or two of _build's
# 32-thread blocks, that stride over every slot, where the states under test
# launch many blocks.
_TWIN_THREADS = (97, 61)


def _rotation(rng, angle):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    return np.concatenate((np.sin(0.5 * angle) * axis, [np.cos(0.5 * angle)]))


def _compose(a, b):
    """Quaternion product a * b in xyzw order."""
    av, aw, bv, bw = a[:3], a[3], b[:3], b[3]
    return np.concatenate((aw * bv + bw * av + np.cross(av, bv), [aw * bw - np.dot(av, bv)]))


def _rotate(q, v):
    u, w = q[:3], q[3]
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


class _Scene:
    """Bodies of two convex pieces in a few worlds, static shapes, and random contact frames.

    Candidate pairs keep a local contact footprint from frame to frame while the
    bodies move slightly, so anchors are carried, released and re-created.
    """

    def __init__(self, device, seed, *, worlds=3, bodies=4, contacts=96, shapeless_pairs=True):
        self.rng = rng = np.random.default_rng(seed)
        self.device = device
        self.worlds = worlds
        self.body_count = worlds * bodies
        # Each world lists its ground, its body pieces, then a ledge. Interleaved
        # static shapes spread the pair codes over the whole key range.
        shape_body, transforms, radii, materials = [], [], [], []
        grounds, ledges, pieces = [], [], []

        def add_shape(body, transform, radius, mu=0.5):
            shape_body.append(body)
            transforms.append(transform)
            radii.append(radius)
            materials.append(mu)
            return len(shape_body) - 1

        for world in range(worlds):
            grounds.append(add_shape(-1, [10.0 * world, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], 5.0))
            for body in range(world * bodies, (world + 1) * bodies):
                for side in (-1.0, 1.0):
                    mu = 0.8 if body % bodies == 1 and side > 0.0 else 0.5
                    pieces.append(add_shape(body, [side * _PIECE_OFFSET, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], 0.05, mu))
            ledges.append(add_shape(-1, [10.0 * world + 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], 0.3))
        shape_count = len(shape_body)
        self.body_world = np.repeat(np.arange(worlds), bodies).astype(np.int32)
        self.model = SimpleNamespace(
            device=wp.get_device(device),
            body_count=self.body_count,
            body_world=wp.array(self.body_world, dtype=int, device=device),
            shape_body=wp.array(shape_body, dtype=int, device=device),
            shape_collision_radius=wp.array(radii, dtype=float, device=device),
            shape_transform=wp.array(transforms, dtype=wp.transform, device=device),
            shape_scale=wp.ones(shape_count, dtype=wp.vec3, device=device),
            shape_type=wp.zeros(shape_count, dtype=int, device=device),
            shape_source_ptr=wp.zeros(shape_count, dtype=wp.uint64, device=device),
            shape_margin=wp.zeros(shape_count, dtype=float, device=device),
            shape_gap=wp.full(shape_count, 0.002, dtype=float, device=device),
            shape_is_solid=wp.ones(shape_count, dtype=bool, device=device),
            shape_material_mu=wp.array(materials, dtype=float, device=device),
        )
        self.shape_body = np.asarray(shape_body)
        self.poses = np.zeros((self.body_count, 7))
        for body in range(self.body_count):
            world, slot = divmod(body, bodies)
            self.poses[body, :3] = (10.0 * world + 0.3 * slot, 0.0, 0.05)
            self.poses[body, 3:] = _rotation(rng, rng.uniform(0.0, 0.2))
        self.state = SimpleNamespace(body_q=wp.array(self.poses, dtype=wp.transform, device=device))
        # Candidate pairs: piece-ground, piece-ledge, neighbouring pieces, and pieces
        # against a missing shape. Footprints live in shape0's body frame.
        self.pairs = []
        for world in range(worlds):
            first = world * bodies
            for body in range(first, first + bodies):
                for piece in (pieces[2 * body], pieces[2 * body + 1]):
                    self._add_pair(piece, grounds[world], (0.0, 0.0, -1.0))
                if body + 1 < first + bodies:
                    self._add_pair(pieces[2 * body + 1], pieces[2 * body + 2], (1.0, 0.0, 0.0))
            self._add_pair(pieces[2 * first], ledges[world], (0.0, -1.0, 0.0))
            # Static pairs have static codes on both sides, the largest keys.
            self._add_pair(grounds[world], ledges[world], (0.0, 0.0, 1.0))
            if shapeless_pairs:
                self._add_pair(pieces[2 * first + 1], -1, (0.0, 0.0, -1.0))
        if shapeless_pairs:
            self._add_pair(-1, -1, (0.0, 0.0, -1.0))
        self.active = rng.random(len(self.pairs)) < 0.6
        self.capacity = contacts
        self.generation = 0
        self.contacts = SimpleNamespace(
            rigid_contact_max=contacts,
            contact_generation=wp.zeros(1, dtype=wp.int32, device=device),
            rigid_contact_count=wp.zeros(1, dtype=int, device=device),
            rigid_contact_shape0=wp.full(contacts, -1, dtype=int, device=device),
            rigid_contact_shape1=wp.full(contacts, -1, dtype=int, device=device),
            rigid_contact_point0=wp.zeros(contacts, dtype=wp.vec3, device=device),
            rigid_contact_point1=wp.zeros(contacts, dtype=wp.vec3, device=device),
            rigid_contact_normal=wp.zeros(contacts, dtype=wp.vec3, device=device),
            rigid_contact_margin0=wp.zeros(contacts, dtype=float, device=device),
            rigid_contact_margin1=wp.zeros(contacts, dtype=float, device=device),
        )

    def _add_pair(self, shape0, shape1, normal):
        rng = self.rng
        normal = np.asarray(normal, dtype=float)
        tangent0 = np.cross(normal, [0.3, 0.5, 0.8])
        tangent0 /= np.linalg.norm(tangent0)
        tangent1 = np.cross(normal, tangent0)
        size = int(rng.integers(1, 7))
        footprint = rng.uniform(-0.02, 0.02, size=(6, 2))
        if rng.random() < 0.3:
            footprint[:, 0] = 0.0  # A narrow footprint.
        if rng.random() < 0.2:
            footprint[1] = footprint[0]  # Coincident witnesses.
        offsets = footprint[:, :1] * tangent0 + footprint[:, 1:] * tangent1
        body = self.shape_body[shape0] if shape0 >= 0 else -1
        anchor = np.zeros(3) if body < 0 else rng.uniform(-0.01, 0.01, size=3)
        self.pairs.append(
            {"shapes": (shape0, shape1), "normal": normal, "offsets": offsets, "size": size, "base": anchor}
        )

    def _world(self, body, point):
        if body < 0:
            return point
        return self.poses[body, :3] + _rotate(self.poses[body, 3:], point)

    def _local(self, body, point):
        if body < 0:
            return point
        inverse = self.poses[body, 3:] * np.array([-1.0, -1.0, -1.0, 1.0])
        return _rotate(inverse, point - self.poses[body, :3])

    def move(self, scale=1.0):
        """Advance every body by a small rigid motion; a few jump far enough to release anchors."""
        rng = self.rng
        for body in range(self.body_count):
            step = 2.0e-4 * scale
            if rng.random() < 0.03:
                step = 0.05
            self.poses[body, :3] += rng.normal(scale=step, size=3)
            self.poses[body, 3:] = _compose(_rotation(rng, rng.normal(scale=1.0e-3 * scale)), self.poses[body, 3:])
            self.poses[body, 3:] /= np.linalg.norm(self.poses[body, 3:])
        self.state.body_q.assign(self.poses)

    def frame(self, mode="live", churn=True):
        """Write a random contact frame and return the live contact count.

        ``mode`` is ``"live"`` (the active pairs), ``"sparse"`` (a single pair),
        ``"empty"`` (no contacts) or ``"overflow"`` (more contacts than the arrays
        hold: as a collision pass does, the frame keeps the rows the arrays hold
        and its count is clamped to them, so every slot below it is live).
        ``churn`` toggles a few pairs and footprint sizes first. The contact
        generation advances and the host notes the write, as in a collision pass.
        """
        self.generation += 1
        note_contact_frame_write(self.contacts.contact_generation)
        self.contacts.contact_generation.assign([self.generation])
        rng = self.rng
        if churn:
            self.active ^= rng.random(len(self.pairs)) < 0.1
            for pair in self.pairs:
                if rng.random() < 0.2:
                    pair["size"] = int(np.clip(pair["size"] + rng.integers(-1, 2), 1, 6))
        chosen = np.flatnonzero(self.active)
        if mode == "sparse":
            chosen = chosen[:1]
        rows = []
        for index in chosen:
            pair = self.pairs[index]
            shape0, shape1 = pair["shapes"]
            body0 = self.shape_body[shape0] if shape0 >= 0 else -1
            body1 = self.shape_body[shape1] if shape1 >= 0 else -1
            normal = pair["normal"] if body0 < 0 else _rotate(self.poses[body0, 3:], pair["normal"])
            if rng.random() < 0.05:
                normal = normal + rng.normal(scale=0.2, size=3)  # An incompatible normal.
                normal /= np.linalg.norm(normal)
            for offset in pair["offsets"][: pair["size"]]:
                point = self._world(body0, pair["base"] + offset) + rng.normal(scale=1.0e-4, size=3)
                margin = (rng.random() < 0.1) * rng.uniform(0.0, 0.003, size=2)
                record = [shape0, shape1, self._local(body0, point), self._local(body1, point), normal, margin]
                if rng.random() < 0.5:
                    record = [shape1, shape0, record[3], record[2], -normal, margin[::-1]]
                rows.append(record)
        rows = [rows[i] for i in rng.permutation(len(rows))][: self.capacity]
        count = 0 if mode == "empty" else len(rows)
        if rows:
            shape0, shape1, point0, point1, normal, margin = (list(column) for column in zip(*rows, strict=True))
            n = len(rows)
            self.contacts.rigid_contact_shape0.assign(np.concatenate((shape0, np.full(self.capacity - n, -1))))
            self.contacts.rigid_contact_shape1.assign(np.concatenate((shape1, np.full(self.capacity - n, -1))))
            for name, values in (("point0", point0), ("point1", point1), ("normal", normal)):
                padded = np.concatenate((np.asarray(values), np.zeros((self.capacity - n, 3))))
                getattr(self.contacts, f"rigid_contact_{name}").assign(padded)
            margin = np.concatenate((np.asarray(margin), np.zeros((self.capacity - n, 2))))
            self.contacts.rigid_contact_margin0.assign(margin[:, 0])
            self.contacts.rigid_contact_margin1.assign(margin[:, 1])
        self.contacts.rigid_contact_count.assign([count])
        return count


def _pair(scene, capacity):
    """Return the patch state under test and its reference, both built from ``scene``."""
    phi = [wp.zeros(capacity, dtype=wp.vec2, device=scene.device) for _ in range(2)]
    return _FrictionPatchState(scene.model, capacity, phi[0]), ReferencePatches(scene.model, capacity, phi[1])


def _set_threads(patches, threads):
    """Stride ``patches``' light launches and builds with ``threads``; the live sort's merges stride as builds."""
    patches.threads, patches.build_threads = threads
    patches.merge_threads = patches.build_threads


def _twin(patches, scene):
    """Return a second patch state on another grid, which CUDA tests step alongside ``patches``, or None on the CPU.

    The twin runs the module under test on the same inputs, with
    ``_TWIN_THREADS`` and one sort block, so it must match ``patches`` bitwise
    (:func:`assert_same_state`): a race or any other grid-dependent result makes
    them differ, however small. The CPU compares with the reference bitwise.
    """
    if not patches.device.is_cuda:
        return None
    twin = _FrictionPatchState(
        scene.model,
        patches.capacity,
        wp.zeros_like(patches.view.phi),
        reuse_frames=patches.reuse_frames,
    )
    _set_threads(twin, _TWIN_THREADS)
    twin.sort_blocks = 1
    twin.live_sort = patches.live_sort
    return twin


def _magnitude(values):
    """Return each row's largest absolute component: rows run along the first axis."""
    values = np.abs(np.asarray(values, dtype=np.float64))
    return values.max(axis=tuple(range(1, values.ndim)), initial=0.0)


def _assert_payload_equal(actual, desired, err_msg, cuda, scale):
    """Compare a computed float payload: bitwise on the CPU, on CUDA within ``_FMA_ULPS`` epsilons of its scale.

    ``scale`` holds the magnitude each row is computed from, broadcast over the
    row's trailing components; a float applies to every row. An element's own
    magnitude also counts, so a zero scale bounds the difference relative to
    the values themselves.
    """
    if not cuda:
        np.testing.assert_array_equal(actual, desired, err_msg)
        return
    actual, desired = np.asarray(actual, dtype=np.float64), np.asarray(desired, dtype=np.float64)
    if actual.shape != desired.shape:
        raise AssertionError(f"{err_msg}\nshapes {actual.shape} and {desired.shape} differ")
    scale = np.asarray(scale, dtype=np.float64)
    scale = scale.reshape(scale.shape + (1,) * (desired.ndim - scale.ndim))
    bound = _FMA_ULPS * _EPS32 * np.maximum(scale, np.maximum(np.abs(actual), np.abs(desired)))
    difference = np.abs(actual - desired)
    close = (actual == desired) | (difference <= bound) | (np.isnan(actual) & np.isnan(desired))
    if not close.all():
        far = np.argwhere(~close)
        with np.errstate(divide="ignore", invalid="ignore"):
            excess = np.where(np.isnan(difference), np.inf, difference / bound)[~close]
        worst = tuple(int(i) for i in far[np.argmax(excess)])
        raise AssertionError(
            f"{err_msg}\n{len(far)} / {close.size} elements differ by more than {_FMA_ULPS} float32 epsilons of"
            f" their scale; worst {worst}: {float(actual[worst])!r} vs {float(desired[worst])!r},"
            f" bound {float(bound[worst]):.3e}"
        )


def _largest(values):
    return float(np.abs(values).max(initial=0.0))


# Integer fields, and floats the kernels copy, negate, add without a product or
# take a min or max of, compare bitwise on every device: contraction only fuses a
# product into an add. The others are computed payloads, listed with the scale
# they compare at: "own", each row's largest component; "world", the largest
# coordinate of the contact or the stored bodies, for points that body poses map
# from world coordinates; "field", the field's largest magnitude, for carried
# displacements, which combine two stored anchors through world-coordinate lever
# arms: their rows moved by up to 114 epsilons of their own magnitude.
_CURRENT_EXACT = (
    "normal",
    "mu",
    "radius",
    "body_a",
    "body_b",
    "shape_a",
    "shape_b",
    "flipped",
    "support_gap_limit",
    "eligible",
)
_CURRENT_COMPUTED = (
    ("center", "own"),
    ("surface_a", "world"),
    ("surface_b", "world"),
    ("displacement", "field"),
)
# Current anchors compare at the "world" scale on valid contacts, the only ones
# whose anchors _build writes and later kernels read.
_CURRENT_ANCHORS = ("anchor_a", "anchor_b")
_PREVIOUS_EXACT = ("mu", "body_a", "body_b", "owner", "shape_a", "shape_b")
_PREVIOUS_COMPUTED = (
    ("normal", "own"),
    ("displacement", "field"),
    ("anchor_a", "world"),
    ("anchor_b", "world"),
    ("surface_a", "world"),
    ("surface_b", "world"),
)


def _scale(kind, values, world):
    """Return the scale of ``kind`` for the reference's ``values``; ``world`` holds each row's world magnitude."""
    return {"own": _magnitude(values), "world": world, "field": _largest(values)}[kind]


def assert_same_patches(patches, reference, live, context="", twin=None):
    """Require the reference's patch state on every index a patch kernel reads.

    Sorted keys, validity, ownership, sources, the patch view and stored worlds
    compare everywhere; sorted indices where a pair key is live; contact payload
    on the live contacts and anchors on the valid ones; stored payload on valid
    anchors; history poses on the bodies of valid anchors. Scratch arrays
    (``members``, ``used``) and the sort's second halves are excluded. Integer
    fields and floats computed without a product compare bitwise on every
    device; computed float payloads compare bitwise on the CPU and within
    :func:`_assert_payload_equal`'s bound on CUDA. A ``twin`` must match
    ``patches`` bitwise in every array (:func:`assert_same_state`).
    """
    if twin is not None:
        assert_same_state(twin, patches, f"{context} twin")
    n = reference.capacity
    stride = patches.key_stride
    cuda = reference.view.weight.device.is_cuda
    for name in ("current", "previous"):
        mine, theirs = getattr(patches, name), getattr(reference, name)
        codes = pair_codes(theirs.keys.numpy()[:n])
        np.testing.assert_array_equal(pair_codes(mine.keys.numpy()[:n], stride), codes, f"{context} {name}.keys")
        keyed = codes[:, 0] >= 0
        np.testing.assert_array_equal(
            mine.indices.numpy()[:n][keyed], theirs.indices.numpy()[:n][keyed], f"{context} {name}.indices"
        )
        np.testing.assert_array_equal(mine.valid.numpy(), theirs.valid.numpy(), f"{context} {name}.valid")
    for name in ("owner", "source"):
        np.testing.assert_array_equal(
            getattr(patches.current, name).numpy(),
            getattr(reference.current, name).numpy(),
            f"{context} current.{name}",
        )
    for name in ("weight", "next_contact"):
        np.testing.assert_array_equal(
            getattr(patches.view, name).numpy(), getattr(reference.view, name).numpy(), f"{context} view.{name}"
        )
    # Friction locations are world coordinates; phi projects carried displacements.
    for name, kind in (("phi", "field"), ("point", "own")):
        theirs = getattr(reference.view, name).numpy()
        _assert_payload_equal(
            getattr(patches.view, name).numpy(), theirs, f"{context} view.{name}", cuda, _scale(kind, theirs, None)
        )
    np.testing.assert_array_equal(
        patches.previous_world.numpy(), reference.previous_world.numpy(), f"{context} previous_world"
    )
    for name in _CURRENT_EXACT:
        np.testing.assert_array_equal(
            getattr(patches.current, name).numpy()[:live],
            getattr(reference.current, name).numpy()[:live],
            f"{context} current.{name}",
        )
    # A live contact's body poses lie within a body radius of its center.
    world = _magnitude(reference.current.center.numpy()[:live])
    for name, kind in _CURRENT_COMPUTED:
        theirs = getattr(reference.current, name).numpy()[:live]
        _assert_payload_equal(
            getattr(patches.current, name).numpy()[:live],
            theirs,
            f"{context} current.{name}",
            cuda,
            _scale(kind, theirs, world),
        )
    anchors = reference.current.valid.numpy()[:live] != 0
    for name in _CURRENT_ANCHORS:
        _assert_payload_equal(
            getattr(patches.current, name).numpy()[:live][anchors],
            getattr(reference.current, name).numpy()[:live][anchors],
            f"{context} current.{name}",
            cuda,
            world[anchors],
        )
    valid = reference.previous.valid.numpy() != 0
    for name in _PREVIOUS_EXACT:
        np.testing.assert_array_equal(
            getattr(patches.previous, name).numpy()[valid],
            getattr(reference.previous, name).numpy()[valid],
            f"{context} previous.{name}",
        )
    # The last store saved the poses that mapped each stored anchor's points; a
    # static side keeps world coordinates, which its own magnitude covers.
    poses = _magnitude(reference.previous_q.numpy()[:, :3])
    stored_world = np.zeros(np.count_nonzero(valid))
    for side in ("body_a", "body_b"):
        body = getattr(reference.previous, side).numpy()[valid]
        stored_world = np.maximum(stored_world, np.where(body >= 0, poses[body], 0.0))
    for name, kind in _PREVIOUS_COMPUTED:
        theirs = getattr(reference.previous, name).numpy()[valid]
        _assert_payload_equal(
            getattr(patches.previous, name).numpy()[valid],
            theirs,
            f"{context} previous.{name}",
            cuda,
            _scale(kind, theirs, stored_world),
        )
    bodies = np.concatenate((reference.previous.body_a.numpy()[valid], reference.previous.body_b.numpy()[valid]))
    bodies = np.unique(bodies[bodies >= 0])
    np.testing.assert_array_equal(
        patches.previous_q.numpy()[bodies], reference.previous_q.numpy()[bodies], f"{context} previous_q"
    )


def assert_empty_tail(patches, context=""):
    """Require Invariant I: slots past the stored and built counts hold the empty record.

    Bounded launches never visit those slots, so they must already hold what a
    full-capacity frame writes there.
    """
    n = patches.capacity
    stored = int(patches.stored.numpy()[0])
    built = int(patches.built.numpy()[0])
    for name, count in (("previous", stored), ("current", built)):
        frame = getattr(patches, name)
        np.testing.assert_array_equal(frame.keys.numpy()[count:n], SENTINEL, f"{context} {name}.keys tail")
        np.testing.assert_array_equal(frame.indices.numpy()[count:n], np.arange(count, n), f"{context} {name}.indices")
        np.testing.assert_array_equal(frame.valid.numpy()[count:], 0, f"{context} {name}.valid tail")
    for name, empty in (("owner", -1), ("source", -1)):
        np.testing.assert_array_equal(getattr(patches.current, name).numpy()[built:], empty, f"{context} {name} tail")
    for name, empty in (("weight", 0.0), ("next_contact", -1), ("phi", 0.0)):
        np.testing.assert_array_equal(getattr(patches.view, name).numpy()[built:], empty, f"{context} {name} tail")
    np.testing.assert_array_equal(patches.previous_world.numpy()[stored:], -1, f"{context} previous_world tail")


def assert_same_state(twin, patches, context="", records=True):
    """Require bitwise-equal arrays in two patch states, scratch included.

    ``records`` compares the frame records as well: the sorted frame, the sort
    flag and the frame status.
    """
    for frame in ("current", "previous"):
        for name in _PatchFrame.vars:
            if frame == "previous" and name in _BUILD_FIELDS:
                continue  # the stored frame holds no field only a build reads
            np.testing.assert_array_equal(
                getattr(getattr(twin, frame), name).numpy(),
                getattr(getattr(patches, frame), name).numpy(),
                f"{context} {frame}.{name}",
            )
    for name in ("weight", "next_contact", "point", "phi"):
        np.testing.assert_array_equal(
            getattr(twin.view, name).numpy(), getattr(patches.view, name).numpy(), f"{context} view.{name}"
        )
    names = ["previous_world", "previous_q", "stored", "built", "extent"]
    if records:
        names += ["sorted_frame", "sort_pairs", "frame_status"]
    for name in names:
        np.testing.assert_array_equal(getattr(twin, name).numpy(), getattr(patches, name).numpy(), f"{context} {name}")


class _Rows:
    """Synthetic contact routing and row arrays of routes 0, 1 and 2.

    Live contacts get world, route, slot and row values the way the allocator
    and row builders lay them out; slots past the live count keep stale values
    that no patch kernel may read. :meth:`arrays` returns a fresh device copy
    per state, because linking, seeding and finishing write the row arrays.
    """

    def __init__(self, scene, capacity, rng):
        self.rng = rng
        self.scene = scene
        self.capacity = capacity
        self.rows = 3 * capacity + 8

    def generate(self, reference, live, routes=(0, 1, 2)):
        rng, n, rows, worlds = self.rng, self.capacity, self.rows, self.scene.worlds
        world = rng.integers(0, worlds, size=n)
        path = rng.integers(0, 3, size=n)
        needed = rng.choice([1, 3], size=n)
        slot = rng.integers(0, rows - 3, size=n)
        parents = rng.integers(-1, rows, size=(3, worlds, rows))
        mu = rng.uniform(0.0, 1.0, size=(3, worlds, rows))
        impulses = rng.uniform(-1.0, 1.0, size=(3, worlds, rows))
        body_a, body_b = reference.current.body_a.numpy(), reference.current.body_b.numpy()
        weight = reference.view.weight.numpy()
        counters = np.zeros((3, worlds), dtype=int)
        for c in rng.permutation(live):
            a, b = int(body_a[c]), int(body_b[c])
            body = a if a >= 0 else b
            world[c] = self.scene.body_world[body] if body >= 0 else 0
            # Contacts of one pair mostly share a route, as the allocator assigns them.
            path[c] = (max(a, b) + 7) % 3 if rng.random() < 0.9 else rng.integers(0, 3)
            needed[c] = 3 if weight[c] > 0.0 and rng.random() < 0.9 else 1
            if rng.random() < 0.05 or path[c] not in routes:
                slot[c] = -1  # A capacity rejection, or a route without rows.
                continue
            r, w = path[c], world[c]
            s = counters[r, w]
            counters[r, w] += needed[c]
            slot[c] = s
            load = 0.0 if rng.random() < 0.15 else rng.uniform(0.1, 2.0)
            parents[r, w, s] = -1
            impulses[r, w, s] = load
            mu[r, w, s : s + needed[c]] = rng.uniform(0.2, 1.0)
            if needed[c] == 3:
                parents[r, w, s + 1 : s + 3] = s
                tangent = rng.normal(size=2)
                tangent *= mu[r, w, s + 1] * load * rng.uniform(0.0, 3.0) / max(np.linalg.norm(tangent), 1.0e-9)
                impulses[r, w, s + 1 : s + 3] = tangent
        self.maps = (world, slot, path, needed)
        self.values = (parents, mu, impulses)
        self.routes = routes
        self.live = live

    def arrays(self):
        """Return device copies: contact maps and ``(route, parents, mu, impulses, first)`` rows."""
        device = self.scene.device
        maps = [wp.array(values.astype(np.int32), dtype=int, device=device) for values in self.maps]
        parents, mu, impulses = self.values
        routes = [
            (
                r,
                wp.array(parents[r].astype(np.int32), dtype=int, device=device),
                wp.array(mu[r].astype(np.float32), dtype=float, device=device),
                wp.array(impulses[r].astype(np.float32), dtype=float, device=device),
                row_lines(device),
            )
            for r in self.routes
        ]
        return maps, routes


def _assert_same_rows(mine, theirs, context=""):
    """Require the reference's row arrays bitwise: parents, friction budgets and impulses.

    Linking divides copied budgets by an anchor count, which matches bitwise on
    every device; no patch launch writes an impulse.
    """
    for (route, *arrays), (_, *expected) in zip(mine, theirs, strict=True):
        for name, array, reference in zip(("parents", "mu", "impulses"), arrays[:3], expected[:3], strict=True):
            np.testing.assert_array_equal(array.numpy(), reference.numpy(), f"{context} route {route} {name}")


def _solved_state(scene, rng):
    """Return a moved output state and centers of mass for the release test."""
    poses = scene.poses.copy()
    poses[:, :3] += rng.normal(scale=1.0e-4, size=(scene.body_count, 3))
    velocity = rng.normal(scale=0.05, size=(scene.body_count, 6))
    device = scene.device
    out = SimpleNamespace(
        body_q=wp.array(poses, dtype=wp.transform, device=device),
        body_qd=wp.array(velocity, dtype=wp.spatial_vector, device=device),
    )
    return out, wp.array(rng.normal(scale=0.01, size=(scene.body_count, 3)), dtype=wp.vec3, device=device)


# Random sequence events: "grow" raises the active pair count, "sparse" keeps one
# pair, "empty" leaves no live contact, "overflow" keeps the rows the arrays hold
# of more contacts, "none" passes no contacts,
# "noframe" passes contacts without capacity, "rebuild" builds twice before the
# rows, "unstored" resets all worlds between an unstored sparse build and the
# step, "gates" filters friction, "geometry" edits a shape, and "reset" and
# "masked" reset all or some worlds after the step.
_PLAN = (
    "grow",
    "grow",
    "grow",
    "live",
    "sparse",
    "live",
    "unstored",
    "empty",
    "grow",
    "overflow",
    "live",
    "none",
    "live",
    "rebuild",
    "live",
    "noframe",
    "live",
    "masked",
    "live",
    "geometry",
    "live",
    "gates",
    "grow",
    "sparse",
    "reset",
    "live",
    "overflow",
    "sparse",
    "live",
)


def _gates(scene, rng):
    """Return build gates that filter friction or normals, with articulation maps."""
    articulation = np.where(rng.random(scene.body_count) < 0.8, np.arange(scene.body_count), -1)
    free = (rng.random(scene.body_count) < 0.5).astype(np.int32)
    device = scene.device
    gates = {
        "body_to_articulation": wp.array(articulation.astype(np.int32), dtype=int, device=device),
        "is_free_rigid": wp.array(free, dtype=int, device=device),
    }
    if rng.random() < 0.5:
        gates["friction_gap"] = -1.0
    else:
        gates.update(contact_gap_gate=1.0e-3, articulation_pair_gap_gate=5.0e-4, same_articulation_gap_gate=2.0e-4)
    return gates


def run_reference_differential(
    device, seed, *, worlds=3, bodies=4, contacts=96, capacity=128, threads=None, live_sort=None, plan=_PLAN
):
    """Step a patch state and the reference through ``plan``, comparing after every launch group.

    ``live_sort`` overrides the capacity's choice of pair sort. On CUDA a twin
    steps alongside (:func:`_twin`). Returns counts of carried anchors, of
    anchors released by finishing and of stored anchors past an unstored
    build's live count, and the largest live count, so callers can require the
    sequence to exercise them.
    """
    rng = np.random.default_rng(1000 + seed)
    scene = _Scene(device, seed, worlds=worlds, bodies=bodies, contacts=contacts)
    patches, reference = _pair(scene, capacity)
    if threads is not None:
        _set_threads(patches, threads)
    if live_sort is not None:
        patches.live_sort = live_sort
    twin = _twin(patches, scene)
    live_states = [state for state in (patches, twin) if state is not None]
    states = [*live_states, reference]
    rows = _Rows(scene, capacity, rng)
    coverage = {"carried": 0, "released": 0, "unstored": 0, "live": 0}
    fraction = 0.1
    dt = 1.0 / 240.0
    for step, event in enumerate(plan):
        context = f"seed {seed} step {step} {event}"
        scene.move()
        if event == "grow":
            fraction = min(1.0, fraction + 0.3)
            scene.active = scene.rng.random(len(scene.pairs)) < fraction
        if event == "geometry":
            transforms = scene.model.shape_transform.numpy()
            transforms[int(rng.integers(0, len(transforms))), 0] += 1.0e-3
            scene.model.shape_transform.assign(transforms)
            for state in states:
                state.update_geometry(scene.model)
            assert_same_patches(patches, reference, 0, f"{context} invalidate", twin)
        if event == "none":
            for state in states:
                state.previous.valid.zero_()
            assert_same_patches(patches, reference, 0, context, twin)
            assert_empty_tail(patches, context)
            continue
        if event == "noframe":
            # No frame this step: the reference's row kernels launch over no slots
            # and its store copies the last frame again; the state stores it alone.
            empty = SimpleNamespace(rigid_contact_max=0, rigid_contact_count=scene.contacts.rigid_contact_count)
            out, com = _solved_state(scene, rng)
            maps, routes = rows.arrays()
            reference.finish(empty, scene.state, out, com, *maps, routes, dt)
            for state in live_states:
                store_frame(state, scene.state)
            assert_same_patches(patches, reference, 0, context, twin)
            assert_empty_tail(patches, context)
            continue
        mode = {"sparse": "sparse", "empty": "empty", "overflow": "overflow"}.get(event, "live")
        gates = _gates(scene, rng) if event == "gates" else {}
        if event == "rebuild":
            scene.frame("live")
            for state in states:
                state.build(scene.model, scene.state, scene.contacts)
            mode = "sparse"
        if event == "unstored":
            # The reset must reach every stored anchor, also past the live count
            # of a build that no store followed.
            sparse = scene.frame("sparse")
            for state in states:
                state.build(scene.model, scene.state, scene.contacts)
            assert_same_patches(patches, reference, sparse, f"{context} unstored build", twin)
            coverage["unstored"] += int(np.count_nonzero(reference.previous.valid.numpy()[sparse:]))
            for state in states:
                state.reset(None)
            assert_same_patches(patches, reference, sparse, f"{context} unstored reset", twin)
            assert_empty_tail(patches, f"{context} unstored reset")
        live = scene.frame(mode)
        for state in states:
            state.build(scene.model, scene.state, scene.contacts, **gates)
        assert_same_patches(patches, reference, live, f"{context} build", twin)
        assert_empty_tail(patches, f"{context} build")
        coverage["live"] = max(coverage["live"], live)
        coverage["carried"] += int(np.count_nonzero(reference.current.source.numpy()[:live] >= 0))
        routes_present = (0, 1, 2) if rng.random() < 0.7 else (0, 1)
        rows.generate(reference, live, routes_present)
        mine = [rows.arrays() for _ in live_states]
        their_maps, their_routes = rows.arrays()
        for state, (maps, routes) in zip(live_states, mine, strict=True):
            state.link(*maps, patch_routes(routes))
        reference.link(scene.contacts, *their_maps, their_routes)
        _assert_live_rows(mine, their_routes, f"{context} link")
        out, com = _solved_state(scene, rng)
        before = int(np.count_nonzero(reference.current.valid.numpy()[:live]))
        for state, (maps, routes) in zip(live_states, mine, strict=True):
            state.finish(scene.state, out, com, *maps, patch_routes(routes), dt)
        reference.finish(scene.contacts, scene.state, out, com, *their_maps, their_routes, dt)
        coverage["released"] += before - int(np.count_nonzero(reference.current.valid.numpy()[:live]))
        assert_same_patches(patches, reference, live, f"{context} finish", twin)
        assert_empty_tail(patches, f"{context} finish")
        if event in ("reset", "masked"):
            mask = None
            if event == "masked":
                mask = wp.array(rng.random(scene.worlds) < 0.5, dtype=wp.bool, device=device)
            for state in states:
                state.reset(mask)
            assert_same_patches(patches, reference, live, f"{context} reset", twin)
            assert_empty_tail(patches, f"{context} reset")
    return coverage


def _assert_live_rows(mine, theirs, context):
    """Compare the state under test's rows, ``mine[0]``, with the reference's, and a twin's with them bitwise."""
    _assert_same_rows(mine[0][1], theirs, context)
    for _, routes in mine[1:]:
        _assert_same_rows(routes, mine[0][1], f"{context} twin")


class TestPatchSortKeys(unittest.TestCase):
    def test_host_sort_honors_end_bit(self):
        """Ignore key bits at and above ``end_bit`` and keep equal keys in input order on the host."""
        rng = np.random.default_rng(3)
        count = 1000
        low = rng.integers(0, 1 << 12, size=count)
        keys = (rng.integers(0, 1 << 20, size=count) << 40) | low
        array = wp.array(np.concatenate((keys, keys)), dtype=wp.int64, device="cpu")
        values = wp.array(np.tile(np.arange(count, dtype=np.int32), 2), dtype=int, device="cpu")
        wp.utils.radix_sort_pairs(array, values, count, end_bit=12)
        expected = np.argsort(low, kind="stable")
        np.testing.assert_array_equal(values.numpy()[:count], expected)
        np.testing.assert_array_equal(array.numpy()[:count], keys[expected])

    def test_compressed_keys_keep_the_reference_order(self):
        """Sort compressed pair keys into the shifted-key permutation and build identical frames."""
        for seed, capacity in ((0, 96), (1, 128), (2, 160)):
            with self.subTest(seed=seed, capacity=capacity):
                scene = _Scene("cpu", seed)
                patches, reference = _pair(scene, capacity)
                patches.live_sort = False
                self.assertLessEqual(patches.key_bits, 62)
                self.assertLess(patches.key_bits, 64 - 8, "the sort must skip at least one byte")
                carried = 0
                for step, mode in enumerate(
                    ("live", "live", "live", "sparse", "live", "empty", "live", "overflow", "live", "live")
                ):
                    scene.move()
                    live = scene.frame(mode)
                    for state in (patches, reference):
                        state.build(scene.model, scene.state, scene.contacts)
                    assert_same_patches(patches, reference, live, f"step {step} build")
                    carried += int(np.count_nonzero(patches.current.source.numpy()[:live] >= 0))
                    store_frame(patches, scene.state)
                    reference.store(scene.state)
                    assert_same_patches(patches, reference, live, f"step {step} store")
                    assert_empty_tail(patches, f"step {step} store")
                self.assertGreater(carried, 0, "the frames must carry anchor history")

    def test_key_stride_fits_a_kernel_int(self):
        """Admit pair-key strides up to the largest 32-bit int, the type _prepare receives them as."""
        self.assertEqual(friction_patches._pair_key_layout(12, 18), (32, 10))
        self.assertEqual(friction_patches._pair_key_layout(2**31 - 3, 0), (2**31 - 1, 62))
        self.assertEqual(friction_patches._pair_key_layout(2**30, 2**30 - 3), (2**31 - 1, 62))
        for bodies, shapes in ((2**31 - 2, 0), (0, 2**31 - 2), (2**31, 2**31)):
            with self.subTest(bodies=bodies, shapes=shapes):
                with self.assertRaisesRegex(ValueError, r"at most 2\*\*31 - 3 bodies and shapes"):
                    friction_patches._pair_key_layout(bodies, shapes)

    def test_build_rejects_another_model(self):
        """Refuse body poses or shapes of another model than the one the pair keys were sized for."""
        scene = _Scene("cpu", 0)
        patches = _FrictionPatchState(scene.model, 96, wp.zeros(96, dtype=wp.vec2, device="cpu"))
        scene.frame()
        fewer_bodies = SimpleNamespace(body_q=wp.array(scene.poses[:-1], dtype=wp.transform, device="cpu"))
        more_shapes = SimpleNamespace(
            **{**vars(scene.model), "shape_body": wp.zeros(len(scene.shape_body) + 1, dtype=int, device="cpu")}
        )
        for model, state in ((scene.model, fewer_bodies), (more_shapes, scene.state)):
            with self.assertRaisesRegex(ValueError, "sized for another model"):
                patches.build(model, state, scene.contacts)
        patches.build(scene.model, scene.state, scene.contacts)

    def test_shapeless_pair_ties_the_sentinel_bits(self):
        """Keep a pair whose key equals the sentinel's sorted bits ahead of the empty slots, with either sort."""
        devices = ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])
        for device, live_sort in itertools.product(devices, (False, True)):
            with self.subTest(device=device, live_sort=live_sort):
                scene = _Scene(device, 5, worlds=1, bodies=4, contacts=16)
                stride = scene.body_count + len(scene.shape_body) + 2
                self.assertEqual(stride & (stride - 1), 0, "the fixture needs a power-of-two key stride")
                patches, reference = _pair(scene, 24)
                patches.live_sort = live_sort
                self.assertEqual(stride * stride - 1, (1 << patches.key_bits) - 1)
                scene.active[:] = False
                scene.active[-1] = True  # The (-1, -1) pair has the largest key, stride**2 - 1.
                scene.active[0] = True
                live = scene.frame(churn=False)
                for state in (patches, reference):
                    state.build(scene.model, scene.state, scene.contacts)
                codes = pair_codes(patches.current.keys.numpy()[:live], stride)
                self.assertTrue(np.any(codes[:, 0] == stride - 1))
                self.assertTrue(np.all(patches.current.keys.numpy()[live : patches.capacity] == SENTINEL))
                assert_same_patches(patches, reference, live)


class TestPatchLiveExtent(unittest.TestCase):
    def test_live_extent_matches_the_full_capacity_reference(self):
        """Match the full-capacity mechanism bitwise through rising, dropping, overflowing and missing frames."""
        totals = {"carried": 0, "released": 0, "unstored": 0}
        # Small capacities take the live sort; the radix sort runs where forced.
        for seed, contacts, capacity, threads, live_sort in (
            (0, 96, 96, None, None),
            (1, 96, 128, (7, 5), False),
            (2, 64, 160, (1, 3), None),
            (3, 128, 128, (32, 64), False),
        ):
            with self.subTest(seed=seed, capacity=capacity, threads=threads, live_sort=live_sort):
                coverage = run_reference_differential(
                    "cpu", seed, contacts=contacts, capacity=capacity, threads=threads, live_sort=live_sort
                )
                for name in totals:
                    totals[name] += coverage[name]
        self.assertGreater(totals["carried"], 0, "the sequences must carry anchor history")
        self.assertGreater(totals["released"], 0, "the sequences must release finished anchors")
        self.assertGreater(totals["unstored"], 0, "a reset must find stored anchors past an unstored build")

    def test_unstored_builds_clear_their_slots(self):
        """Clear the slots of a build that no store followed, as a full-capacity frame does."""
        run_unstored_builds("cpu", 8, capacity=96, threads=(5, 3))

    @unittest.skipUnless(wp.is_cuda_available(), "multi-block grids need CUDA")
    def test_unstored_builds_clear_their_slots_on_many_blocks(self):
        """Clear an unstored build's slots on many blocks when the next frame is smaller and nothing is stored."""
        run_unstored_builds("cuda:0", 18, worlds=48, contacts=2048, capacity=3072)

    @unittest.skipUnless(wp.is_cuda_available(), "multi-block grids need CUDA")
    def test_live_extent_matches_the_reference_on_many_blocks(self):
        """Publish the live extent without a race while many blocks stride over sharp drops."""
        for seed, threads in ((10, None), (11, (1000, 600))):
            with self.subTest(seed=seed, threads=threads):
                coverage = run_reference_differential(
                    "cuda:0", seed, worlds=48, contacts=2048, capacity=3072, threads=threads
                )
                self.assertGreater(coverage["carried"], 0)

    @unittest.skipUnless(wp.is_cuda_available(), "graph capture needs CUDA")
    def test_graph_replay_follows_changing_counts(self):
        """Replay one captured patch step over rising, empty, overflowing and sharply dropping counts."""
        run_graph_replay("cuda:0", live_sort=False)


def run_unstored_builds(device, seed, *, worlds=3, contacts=96, capacity=96, threads=None):
    """Build full, overflowing, sparse and empty frames that no store follows, comparing every build.

    Every pair is active, so each full frame leaves more built slots than the
    next frame holds and no store ever records them.
    """
    scene = _Scene(device, seed, worlds=worlds, contacts=contacts)
    patches, reference = _pair(scene, capacity)
    if threads is not None:
        _set_threads(patches, threads)
    twin = _twin(patches, scene)
    scene.active[:] = True
    for mode in ("live", "overflow", "live", "sparse", "empty"):
        scene.move()
        live = scene.frame(mode)
        for state in (patches, twin, reference):
            if state is not None:
                state.build(scene.model, scene.state, scene.contacts)
        assert_same_patches(patches, reference, live, mode, twin)
        assert_empty_tail(patches, mode)


def run_graph_replay(device, *, worlds=32, contacts=1536, capacity=2048, live_sort=None):
    """Capture one patch step, replay it over changing counts, and compare each replay with the reference.

    On CUDA a twin steps eagerly alongside the replays (:func:`_twin`).
    """
    rng = np.random.default_rng(7)
    scene = _Scene(device, 21, worlds=worlds, contacts=contacts)
    patches, reference = _pair(scene, capacity)
    if live_sort is not None:
        patches.live_sort = live_sort
    twin = _twin(patches, scene)
    rows = _Rows(scene, capacity, rng)
    dt = 1.0 / 240.0

    def prepare(mode):
        scene.move()
        live = scene.frame(mode)
        reference.build(scene.model, scene.state, scene.contacts)
        rows.generate(reference, live)
        return live

    def assign(fixed, out, com):
        (maps, routes), (fresh_maps, fresh_routes) = fixed, rows.arrays()
        for array, values in zip(maps, fresh_maps, strict=True):
            array.assign(values)
        for row, values in zip(routes, fresh_routes, strict=True):
            for array, fresh in zip(row[1:4], values[1:4], strict=True):
                array.assign(fresh)
        solved, centers = _solved_state(scene, rng)
        out.body_q.assign(solved.body_q)
        out.body_qd.assign(solved.body_qd)
        com.assign(centers)

    live = prepare("live")
    fixed = rows.arrays()
    out, com = _solved_state(scene, rng)
    maps, routes = fixed
    linked = patch_routes(routes)

    def step():
        patches.build(scene.model, scene.state, scene.contacts)
        patches.link(*maps, linked)
        patches.finish(scene.state, out, com, *maps, linked, dt)

    def reference_step():
        their_maps, their_routes = rows.arrays()
        reference.link(scene.contacts, *their_maps, their_routes)
        reference.finish(scene.contacts, scene.state, out, com, *their_maps, their_routes, dt)
        return their_routes

    def twin_step(context):
        """Step the twin on fresh copies of the rows and compare its rows with the replayed ones."""
        if twin is None:
            return
        twin_maps, twin_routes = rows.arrays()
        twin.build(scene.model, scene.state, scene.contacts)
        twin.link(*twin_maps, patch_routes(twin_routes))
        twin.finish(scene.state, out, com, *twin_maps, patch_routes(twin_routes), dt)
        _assert_same_rows(twin_routes, routes, f"{context} twin")

    step()
    twin_step("eager")
    reference_step()
    assert_same_patches(patches, reference, live, "eager", twin)
    with wp.ScopedCapture(device=device) as capture:
        step()
    for index, mode in enumerate(("live", "sparse", "empty", "live", "overflow", "live", "sparse", "live")):
        scene.active = scene.rng.random(len(scene.pairs)) < (0.9 if index == 3 else 0.6)
        live = prepare(mode)
        assign(fixed, out, com)
        wp.capture_launch(capture.graph)
        twin_step(f"replay {index} {mode}")
        their_routes = reference_step()
        assert_same_patches(patches, reference, live, f"replay {index} {mode}", twin)
        assert_empty_tail(patches, f"replay {index} {mode}")
        _assert_same_rows(routes, their_routes, f"replay {index} {mode}")


@contextlib.contextmanager
def _record_patch_routes(patches):
    """Record ``(method, routes)`` for every link and finish call the solver makes on ``patches``."""
    calls = []
    with contextlib.ExitStack() as stack:
        for name in ("link", "finish"):
            method = getattr(patches, name)

            def record(*args, _name=name, _method=method, **kwargs):
                calls.append((_name, inspect.signature(_method).bind(*args, **kwargs).arguments["routes"]))
                return _method(*args, **kwargs)

            stack.enter_context(mock.patch.object(patches, name, record))
        yield calls


class TestPatchSolverRoutes(unittest.TestCase):
    def test_solver_passes_every_patch_route(self):
        """Link and finish every route that holds patch rows, warm-started or not: no step seeds friction."""
        # Both modes route the fixed-base arm's contacts and the free box's through the line's rows (a free body is a
        # one-node tree; only colored worlds keep a free bodies' run); no route builds the other families.
        configs = [("cpu", {}, (1, 0, 0))]
        if wp.is_cuda_available():
            configs.append(("cuda:0", {}, (1, 0, 0)))
        dts = (1.0 / 240.0, 1.0 / 240.0, 1.0 / 120.0, 1.0 / 480.0)
        for (device, mode, enabled), warmstart in itertools.product(configs, (False, True)):
            with self.subTest(device=device, warmstart=warmstart):
                model = _arm_over_box(wp.get_device(device), worlds=2)
                # The routes hold rows on the complete path: the box's speculative contacts separate at first.
                solver = newton.solvers.SolverFeatherPGS(
                    model,
                    pgs_iterations=4,
                    pgs_warmstart=warmstart,
                    pgs_warmstart_decay=0.8,
                    pins=newton.solvers.SolverFeatherPGS.Pins(claims=()),
                    **mode,
                )
                self.assertGreater(solver.laws.friction_anchor_beta, 0.0)
                arrays = (
                    (solver.placement.row_parent, solver.placement.row_mu, solver.placement.impulses),
                    (solver.placement.row_parent, solver.placement.row_mu, solver.placement.impulses),
                )
                pipeline = newton.CollisionPipeline(model, contact_matching="latest")
                contacts = pipeline.contacts()
                state, out = model.state(), model.state()
                newton.eval_fk(model, state.joint_q, state.joint_qd, state)
                control = model.control()
                paths = set()
                with _record_patch_routes(solver.laws.friction_patches) as calls:
                    for step, dt in enumerate(dts):
                        pipeline.collide(state, contacts)
                        state.clear_forces()
                        solver.step(state, out, control, contacts, dt)
                        state, out = out, state
                        count = int(contacts.rigid_contact_count.numpy()[0])
                        paths.update(solver.placement.contact_path.numpy()[:count].tolist())
                        self.assertEqual([name for name, _ in calls], ["link", "finish"], f"step {step}")
                        for name, routes in calls:
                            self.assertEqual(tuple(routes.enabled), enabled, f"step {step} {name}")
                            for route in np.flatnonzero(enabled):
                                for field, array in zip(("parents", "mu", "impulses"), arrays[route], strict=True):
                                    self.assertEqual(getattr(routes, f"{field}{route}").ptr, array.ptr, field)
                        calls.clear()
                    # A step without a frame finishes no route; a step without contacts
                    # drops the stored anchors without calling the patch state.
                    empty = newton.Contacts(0, 0, device=device, contact_matching=True)
                    for step_contacts, expected in ((empty, [("finish", (0, 0, 0))]), (None, [])):
                        solver.step(state, out, control, step_contacts, dts[0])
                        state, out = out, state
                        self.assertEqual([(name, tuple(routes.enabled)) for name, routes in calls], expected)
                        calls.clear()
                    self.assertFalse(np.any(solver.laws.friction_patches.previous.valid.numpy()))
                if device == "cpu":
                    self.assertLessEqual({0}, paths, "contacts must use the line's rows")
                    self.assertNotIn(1, paths, "a single free body keeps no run of its own")


def _fill_sort_case(patches, live, rng, codes):
    """Write a frame as _prepare leaves it: ``live`` pair keys drawn from ``codes``, then empty slots.

    The second halves hold unrelated values, as the radix sort's scratch would.
    Returns the first half of the keys.
    """
    n = patches.capacity
    keys = np.full(2 * n, SENTINEL, dtype=np.int64)
    keys[:live] = rng.choice(codes, size=live)
    keys[n:] = rng.integers(0, 1 << 40, size=n)
    indices = np.concatenate((np.arange(n), rng.integers(0, n, size=n))).astype(np.int32)
    patches.current.keys.assign(keys)
    patches.current.indices.assign(indices)
    patches.extent.assign(_patch_extent(live, live, patches.extent.shape[0]))
    return keys[:n]


def _patch_extent(end, live, size):
    """Return the patch extent a prepare publishes: the slots to visit, then the live sort's extent."""
    extent = np.zeros(size, dtype=np.int32)
    rounds = 0
    while (int(SORT_CHUNK) << rounds) < live and rounds < int(MAX_MERGES):
        rounds += 1
    extent[:3] = end, live, -(-live // int(SORT_CHUNK))
    extent[3 : 3 + rounds] = live
    return extent


def run_live_sort_cases(test, device, capacities, extra_counts=()):
    """Sort live prefixes of several sizes and compare with numpy's stable argsort and the radix sort."""
    rng = np.random.default_rng(17)
    scene = _Scene(device, 0)
    # Pair codes of several worlds, including the smallest and largest ones a
    # model with 2**25 + 1 bodies and shapes can produce; few codes, many repeats.
    stride = (1 << 25) + 3
    first = np.concatenate(([1, 2, stride - 2, stride - 1], rng.integers(1, stride, size=40)))
    second = np.concatenate(([stride - 1, 3, stride - 1, stride - 1], rng.integers(1, stride, size=40)))
    codes = np.minimum(first, second).astype(np.int64) * stride + np.maximum(first, second)
    test.assertLess(int(codes.max()), 1 << 51)
    for capacity in capacities:
        patches = _FrictionPatchState(scene.model, capacity, wp.zeros(capacity, dtype=wp.vec2, device=device))
        counts = {0, 1, 2047, 2048, 2049, capacity // 2, capacity - 1, capacity, *extra_counts}
        for live in sorted(count for count in counts if 0 <= count <= capacity):
            with test.subTest(capacity=capacity, live=live):
                keys = _fill_sort_case(patches, live, rng, codes)
                # CPU arrays expose their memory; copies keep what the slots held.
                before = (patches.current.keys.numpy().copy(), patches.current.indices.numpy().copy())
                patches._sort_live()
                if live == 0:
                    np.testing.assert_array_equal(patches.current.keys.numpy(), before[0], "no slot changes")
                    np.testing.assert_array_equal(patches.current.indices.numpy(), before[1], "no slot changes")
                sorted_keys = patches.current.keys.numpy()[:capacity]
                sorted_indices = patches.current.indices.numpy()[:capacity]
                order = np.argsort(keys[:live], kind="stable")
                np.testing.assert_array_equal(sorted_indices[:live], order)
                np.testing.assert_array_equal(sorted_keys[:live], keys[order])
                radix_keys = wp.array(np.tile(keys, 2), dtype=wp.int64, device=device)
                radix_indices = wp.array(np.tile(np.arange(capacity, dtype=np.int32), 2), dtype=int, device=device)
                wp.utils.radix_sort_pairs(radix_keys, radix_indices, capacity)
                np.testing.assert_array_equal(sorted_keys, radix_keys.numpy()[:capacity])
                np.testing.assert_array_equal(sorted_indices, radix_indices.numpy()[:capacity])


class TestPatchLiveSort(unittest.TestCase):
    def test_live_sort_matches_stable_argsort_and_radix_sort(self):
        """Sort the live prefix exactly as a stable sort, from no contacts, which no slot sees, through merges."""
        run_live_sort_cases(self, "cpu", (128, 2047, 3000, 4096, 10000, 16_384))

    def test_key_width_selects_the_live_sort(self):
        """Sort only the live prefix at every capacity while the pair keys leave room for the slot bits."""
        scene = _Scene("cpu", 0)
        phi = wp.zeros(96, dtype=wp.vec2, device="cpu")
        self.assertTrue(_FrictionPatchState(scene.model, 96, phi).live_sort)
        self.assertTrue(_FrictionPatchState(scene.model, 64, phi[:64]).live_sort)
        self.assertTrue(friction_patches._uses_live_sort(51))
        # Pair keys of 52 bits and more fall back to the radix sort: their 11-bit
        # slot would reach the chunk keys' pad bit.
        self.assertFalse(friction_patches._uses_live_sort(52))

    def test_live_sort_builds_match_the_full_capacity_reference(self):
        """Build, link, seed, finish and store bitwise as the reference with the live sort."""
        for seed, threads in ((4, None), (5, (7, 5))):
            with self.subTest(seed=seed, threads=threads):
                run_reference_differential("cpu", seed, capacity=160, threads=threads, live_sort=True)
        # More than one chunk of live contacts runs the merges.
        plan = ("grow", "grow", "grow", "live", "sparse", "grow", "overflow", "live", "rebuild", "masked", "live")
        coverage = run_reference_differential(
            "cpu", 6, worlds=110, contacts=4096, capacity=4600, threads=(97, 61), live_sort=True, plan=plan
        )
        self.assertGreater(coverage["live"], 2048)

    @unittest.skipUnless(wp.is_cuda_available(), "the device sort needs CUDA")
    def test_live_sort_matches_on_the_device(self):
        """Sort live prefixes on the device, from capacities below one chunk to hundreds of chunks and nine merges."""
        capacities = (128, 3000, 10000, 16_384, 1_000_000)
        run_live_sort_cases(self, "cuda:0", capacities, (300_000,))

    @unittest.skipUnless(wp.is_cuda_available(), "multi-block grids need CUDA")
    def test_live_sort_builds_match_on_many_blocks(self):
        """Match the reference with the live sort on multi-block grids, through merges and graph replay."""
        run_reference_differential("cuda:0", 12, worlds=160, contacts=8192, capacity=8192, live_sort=True)
        run_graph_replay("cuda:0", worlds=160, contacts=8192, capacity=8192, live_sort=True)


def _shape(capacity, sms, *, light, build, block, merge=4, sort=2):
    """Return the launch attributes of a patch state with ``capacity`` slots on ``sms`` SMs.

    ``light``, ``build`` and ``merge`` count 256-thread blocks per SM of the light
    launches, ``_build`` and ``_merge_runs``, ``block`` is ``_build``'s block size,
    and ``sort`` counts the live sort's chunk blocks per SM.
    """
    return {
        "threads": max(1, min(capacity, light * sms * 256)),
        "build_threads": max(1, min(capacity, build * sms * 256)),
        "build_block_dim": block,
        "merge_threads": max(1, min(capacity, merge * sms * 256)),
        "sort_blocks": max(1, min(-(-capacity // 2048), sort * sms)),
    }


# Launch shapes before and since one-warp _build blocks on twice the grid and
# light launches on three times theirs.
_FORMER_SHAPE = {"light": 2, "build": 4, "block": 256}
_SHAPE = {"light": 6, "build": 8, "block": 32}
# An odd shape, far below the live counts, on any device.
_ODD_SHAPE = {"threads": 7, "build_threads": 5, "build_block_dim": 32, "merge_threads": 3, "sort_blocks": 1}

# Collision frames of the launch-shape differential and the substeps that solve
# each: "grow" activates more pairs, "sparse" keeps one pair, "empty" leaves no
# live contact, "overflow" keeps the rows the arrays hold of more contacts,
# "unstored" first builds a sparse frame that
# no store follows and resets every world, "store" stores the frame again
# without finishing it, and "masked" resets some worlds after its substeps.
_SHAPE_PLAN = (
    ("grow", 2),
    ("grow", 1),
    ("grow", 2),
    ("live", 2),
    ("sparse", 1),
    ("unstored", 2),
    ("live", 1),
    ("empty", 1),
    ("store", 1),
    ("overflow", 2),
    ("live", 2),
    ("masked", 2),
    ("live", 1),
)


class _TiledScene:
    """Copies of a :class:`_Scene` side by side in one model, for live contact counts of device scale.

    Copy k holds the base scene's worlds, bodies and shapes from k times their
    counts on, at the base poses. Each frame repeats the base frame in every
    copy, with all copies' contacts in one random order, so a pair's contacts
    differ between copies only in their order. The base has no shapeless
    pairs: the pair of two missing shapes would join every copy's contacts in
    one run.
    """

    # Per-contact arrays of a frame.
    _FIELDS = ("shape0", "shape1", "point0", "point1", "normal", "margin0", "margin1")

    def __init__(self, device, seed, copies, *, worlds, contacts):
        self.base = base = _Scene("cpu", seed, worlds=worlds, contacts=contacts, shapeless_pairs=False)
        self.copies = copies
        self.device = device
        self.worlds = base.worlds * copies
        self.body_count = base.body_count * copies
        self.capacity = base.capacity * copies
        self.shape_count = base.model.shape_body.shape[0]
        copy = np.arange(copies)[:, None]
        self.body_world = (base.body_world + base.worlds * copy).ravel().astype(np.int32)
        arrays = {}
        for name, array in vars(base.model).items():
            if isinstance(array, wp.array):
                values = array.numpy()
                tiled = np.tile(values, (copies,) + (1,) * (values.ndim - 1))
                arrays[name] = wp.array(tiled, dtype=array.dtype, device=device)
        shape_body = base.model.shape_body.numpy()
        shape_body = np.where(shape_body >= 0, shape_body + base.body_count * copy, -1).ravel()
        arrays["shape_body"] = wp.array(shape_body, dtype=int, device=device)
        arrays["body_world"] = wp.array(self.body_world, dtype=int, device=device)
        self.model = SimpleNamespace(device=wp.get_device(device), body_count=self.body_count, **arrays)
        self.poses = np.tile(base.poses, (copies, 1))
        self.state = SimpleNamespace(body_q=wp.array(self.poses, dtype=wp.transform, device=device))
        self.contacts = SimpleNamespace(
            rigid_contact_max=self.capacity,
            contact_generation=wp.zeros(1, dtype=wp.int32, device=device),
            rigid_contact_count=wp.zeros(1, dtype=int, device=device),
        )
        for name in self._FIELDS:
            array = getattr(base.contacts, f"rigid_contact_{name}")
            setattr(self.contacts, f"rigid_contact_{name}", wp.zeros(self.capacity, dtype=array.dtype, device=device))

    @property
    def rng(self):
        return self.base.rng

    @property
    def pairs(self):
        return self.base.pairs

    @property
    def active(self):
        return self.base.active

    @active.setter
    def active(self, value):
        self.base.active = value

    def move(self, scale=1.0):
        """Move the base scene's bodies and every copy of them alike."""
        self.base.move(scale)
        self.poses = np.tile(self.base.poses, (self.copies, 1))
        self.state.body_q.assign(self.poses)

    def frame(self, mode="live", churn=True):
        """Repeat a base frame (:meth:`_Scene.frame`) in every copy and return the live contact count."""
        base = self.base
        base.frame(mode, churn)
        count = int(base.contacts.rigid_contact_count.numpy()[0])
        self.contacts.contact_generation.assign([base.generation])
        live = count * self.copies
        copy, contact = np.divmod(self.rng.permutation(live), max(count, 1))
        for name in self._FIELDS:
            values = getattr(base.contacts, f"rigid_contact_{name}").numpy()[contact]
            empty = 0
            if name.startswith("shape"):
                values = np.where(values >= 0, values + self.shape_count * copy, -1)
                empty = -1
            tiled = np.full((self.capacity, *values.shape[1:]), empty, dtype=values.dtype)
            tiled[:live] = values
            getattr(self.contacts, f"rigid_contact_{name}").assign(tiled)
        self.contacts.rigid_contact_count.assign([live])
        return live


class _WideRows(_Rows):
    """:class:`_Rows` for frames of device scale, with as many rows per world as the fullest world needs.

    Live contacts get the world, route, slot and row values :class:`_Rows`
    gives them, from array operations: each route reserves a world's slots in
    one random order of the live contacts. Slots past the live count keep
    stale values, and the row arrays need not hold three rows per slot of the
    capacity in every world. A contact without a body (a static pair) keeps a
    random world: world 0 would gather every copy's static contacts, and the
    row arrays' width would grow with the copies.
    """

    def generate(self, reference, live, routes=(0, 1, 2)):
        rng, n, worlds = self.rng, self.capacity, self.scene.worlds
        body_a = reference.current.body_a.numpy()[:live]
        body_b = reference.current.body_b.numpy()[:live]
        weight = reference.view.weight.numpy()[:live]
        world = rng.integers(0, worlds, size=n)
        path = rng.integers(0, 3, size=n)
        needed = rng.choice([1, 3], size=n)
        body = np.where(body_a >= 0, body_a, body_b)
        world[:live] = np.where(body >= 0, self.scene.body_world[np.maximum(body, 0)], world[:live])
        # Contacts of one pair mostly share a route, as the allocator assigns them.
        shared = rng.random(live) < 0.9
        path[:live] = np.where(shared, (np.maximum(body_a, body_b) + 7) % 3, rng.integers(0, 3, size=live))
        needed[:live] = np.where((weight > 0.0) & (rng.random(live) < 0.9), 3, 1)
        # Capacity rejections and routes without rows get no slot.
        order = rng.permutation(live)
        order = order[(rng.random(live) >= 0.05) & np.isin(path[order], routes)]
        order = order[np.argsort(path[order] * worlds + world[order], kind="stable")]
        group = path[order] * worlds + world[order]
        end = np.cumsum(needed[order])
        first = (end - needed[order])[np.searchsorted(group, group)]
        start = end - needed[order] - first
        self.rows = rows = int((end - first).max(initial=0)) + 8
        slot = rng.integers(0, rows - 3, size=n)
        slot[:live] = -1
        slot[order] = start
        parents = rng.integers(-1, rows, size=(3, worlds, rows))
        mu = rng.uniform(0.0, 1.0, size=(3, worlds, rows))
        impulses = rng.uniform(-1.0, 1.0, size=(3, worlds, rows))
        r, w, s = path[order], world[order], start
        load = np.where(rng.random(len(order)) < 0.15, 0.0, rng.uniform(0.1, 2.0, size=len(order)))
        budget = rng.uniform(0.2, 1.0, size=len(order))
        parents[r, w, s] = -1
        impulses[r, w, s] = load
        mu[r, w, s] = budget
        friction = needed[order] == 3
        r, w, s, budget, load = r[friction], w[friction], s[friction], budget[friction], load[friction]
        tangent = rng.normal(size=(len(s), 2))
        tangent *= (budget * load * rng.uniform(0.0, 3.0, size=len(s)) / np.maximum(np.hypot(*tangent.T), 1.0e-9))[
            :, None
        ]
        for k in (1, 2):
            parents[r, w, s + k] = s
            mu[r, w, s + k] = budget
            impulses[r, w, s + k] = tangent[:, k - 1]
        self.maps = (world, slot, path, needed)
        self.values = (parents, mu, impulses)
        self.routes = routes
        self.live = live


def run_launch_shapes(
    device, seed, shapes, *, worlds=3, contacts=96, capacity=128, live_sort=None, reuse=False, copies=1, stages=None
):
    """Step one patch state per launch shape on identical inputs; every array must match the first state's bitwise.

    ``shapes`` holds attribute overrides (:func:`_shape`), ``None`` for the
    constructor's own shape. Each frame is built, linked, seeded and finished
    over its substeps, by states built with frame reuse under ``reuse``.
    After each launch group every array of every state, sort halves and scratch
    included, and every row array must equal the first state's (frames follow
    ``_SHAPE_PLAN``); ``stages`` limits the comparisons to the named launch
    groups ("build", "link", "finish", "store" and "reset"). With
    ``copies`` above one the scene is that many copies of a scene of
    ``worlds`` and ``contacts`` (:class:`_TiledScene`), and ``capacity`` counts
    slots per copy. Returns the largest live count and the carried anchors.
    """
    rng = np.random.default_rng(4000 + seed)
    if copies == 1:
        scene = _Scene(device, seed, worlds=worlds, contacts=contacts)
        rows = _Rows(scene, capacity, rng)
    else:
        scene = _TiledScene(device, seed, copies, worlds=worlds, contacts=contacts)
        capacity *= copies
        rows = _WideRows(scene, capacity, rng)
    states = []
    for shape in shapes:
        phi = wp.zeros(capacity, dtype=wp.vec2, device=device)
        state = _FrictionPatchState(scene.model, capacity, phi, reuse_frames=reuse)
        for name, value in (shape or {}).items():
            setattr(state, name, value)
        if live_sort is not None:
            state.live_sort = live_sort
        states.append(state)
    first = states[0]
    coverage = {"live": 0, "carried": 0}
    dt = 1.0 / 240.0
    fraction = 0.2

    def check(group, context, routes=None):
        if stages is not None and group not in stages:
            return
        for index, state in enumerate(states[1:], start=1):
            assert_same_state(state, first, f"{context} shape {index}")
            if routes is not None:
                _assert_same_rows(routes[index], routes[0], f"{context} shape {index}")

    def each(method, *args):
        for state in states:
            getattr(state, method)(*args)

    for step, (event, substeps) in enumerate(_SHAPE_PLAN):
        context = f"seed {seed} step {step} {event}"
        scene.move()
        if event == "grow":
            fraction = min(1.0, fraction + 0.4)
            scene.active = scene.rng.random(len(scene.pairs)) < fraction
        if event == "unstored":
            scene.frame("sparse")
            each("build", scene.model, scene.state, scene.contacts)
            check("build", f"{context} unstored build")
            each("reset", None)
            check("reset", f"{context} unstored reset")
        if event == "store":
            for state in states:
                store_frame(state, scene.state)
            check("store", f"{context} store")
            continue
        live = scene.frame({"sparse": "sparse", "empty": "empty", "overflow": "overflow"}.get(event, "live"))
        coverage["live"] = max(coverage["live"], live)
        for substep in range(substeps):
            stage = f"{context} substep {substep}"
            if substep > 0:
                scene.move()
            each("build", scene.model, scene.state, scene.contacts)
            check("build", f"{stage} build")
            coverage["carried"] += int(np.count_nonzero(first.current.source.numpy()[:live] >= 0))
            rows.generate(first, live)
            arrays = [rows.arrays() for _ in states]
            routes = [row for _, row in arrays]
            for state, (maps, row) in zip(states, arrays, strict=True):
                state.link(*maps, patch_routes(row))
            check("link", f"{stage} link", routes)
            out, com = _solved_state(scene, rng)
            for state, (maps, row) in zip(states, arrays, strict=True):
                state.finish(scene.state, out, com, *maps, patch_routes(row), dt)
            check("finish", f"{stage} finish", routes)
        if event == "masked":
            each("reset", wp.array(rng.random(scene.worlds) < 0.5, dtype=wp.bool, device=device))
            check("reset", f"{context} reset")
    return coverage


@contextlib.contextmanager
def _patch_variants_compiled_here(device, block_dims=(256, friction_patches._BUILD_BLOCK_DIM)):
    """Run the body on block-size variants of the patch module that this process compiled.

    Which product of a sum NVRTC leaves unfused can follow the order in which
    Warp emits a module's kernels, and that order differs between processes
    (module docstring). _build launches from a variant of its own block size,
    so two cached variants that two processes compiled may round carried
    displacements differently. Variants compiled by one process share the
    emission order and must match bitwise. The variants of ``block_dims``
    compile into a temporary kernel cache before the body runs, and every
    other module loads from the kernel cache as usual; the body must launch
    the patch module from these variants only.
    """
    module = friction_patches._build.module
    cache = wp.config.kernel_cache_dir
    module.unload()
    with tempfile.TemporaryDirectory() as directory:
        wp.config.kernel_cache_dir = directory
        try:
            for block_dim in block_dims:
                wp.load_module(module, device=device, block_dim=block_dim)
        finally:
            wp.config.kernel_cache_dir = cache
    compiled = dict(module.execs)
    try:
        yield
        if module.execs != compiled:
            raise AssertionError(f"the patch module loaded variants other than block sizes {block_dims}")
    finally:
        module.unload()


class TestPatchLaunchShapes(unittest.TestCase):
    def test_builds_launch_one_warp_blocks_on_a_wider_grid(self):
        """Launch _build in one-warp blocks over twice the former grid and light passes over three times theirs.

        The live sort's merges and chunk blocks keep their grids.
        """
        scene = _Scene("cpu", 1, worlds=110, contacts=4096)
        scene.active[:] = True
        for capacity in (96, 3000, 4600, 10000):
            with self.subTest(capacity=capacity):
                phi = wp.zeros(capacity, dtype=wp.vec2, device="cpu")
                state = _FrictionPatchState(scene.model, capacity, phi)
                # The CPU counts as one SM.
                expected = _shape(capacity, 1, **_SHAPE)
                self.assertEqual({name: getattr(state, name) for name in expected}, expected)
        state.live_sort = True
        launched = set()
        original = wp.launch

        def record(kernel, **kwargs):
            # Each launch strides over the slots by its own thread count.
            arguments = [*kwargs.get("inputs", ()), *kwargs.get("outputs", ())]
            stride = dict(zip((arg.label for arg in kernel.adj.args), arguments, strict=True))["threads"]
            launched.add((kernel.key, kwargs["dim"], stride, kwargs.get("block_dim")))
            return original(kernel, **kwargs)

        self.assertGreater(scene.frame(), 2048, "more than one chunk of live contacts runs the merges")
        with mock.patch.object(wp, "launch", record):
            state.build(scene.model, scene.state, scene.contacts)
        self.assertEqual(
            launched, {("_prepare", 1536, 1536, None), ("_merge_runs", 1024, 1024, None), ("_build", 2048, 2048, 32)}
        )

    def test_launch_shapes_match_bitwise(self):
        """Build, link, seed, finish, store and reset bitwise on every grid, with each sort and each prepare.

        CPU launches run one thread at a time, so they check how each grid
        strides over the slots; the device tests run every combination of sort
        and frame reuse.
        """
        shapes = (None, _shape(4600, 1, **_FORMER_SHAPE), _ODD_SHAPE)
        for live_sort, reuse in ((False, False), (True, True)):
            with self.subTest(live_sort=live_sort, reuse=reuse):
                coverage = run_launch_shapes(
                    "cpu", 2, shapes, worlds=110, contacts=4096, capacity=4600, live_sort=live_sort, reuse=reuse
                )
                self.assertGreater(coverage["live"], 2048, "the live sort must merge chunks")
                self.assertGreater(coverage["carried"], 0, "the frames must carry anchor history")

    @unittest.skipUnless(wp.is_cuda_available(), "launch shapes need CUDA")
    def test_launch_shapes_match_bitwise_on_the_device(self):
        """Match the former 256-thread _build blocks and grids bitwise with one-warp blocks on the device.

        The device shapes differ only in _build's block size at this capacity; the
        one-SM shapes and the odd shape also stride over the live slots. Both
        block-size variants of the patch module compile in this process
        (:func:`_patch_variants_compiled_here`).
        """
        device = wp.get_device("cuda:0")
        capacity = 4600
        default = _shape(capacity, device.sm_count, **_SHAPE)
        shapes = (
            None,
            _shape(capacity, device.sm_count, **_FORMER_SHAPE),
            _shape(capacity, 1, **_FORMER_SHAPE),
            _shape(capacity, 1, **_SHAPE),
            _ODD_SHAPE,
        )
        with _patch_variants_compiled_here(device):
            for live_sort, reuse in itertools.product((False, True), (False, True)):
                with self.subTest(live_sort=live_sort, reuse=reuse):
                    coverage = run_launch_shapes(
                        device,
                        3,
                        shapes,
                        worlds=110,
                        contacts=4096,
                        capacity=capacity,
                        live_sort=live_sort,
                        reuse=reuse,
                    )
                    self.assertGreater(coverage["live"], 2048, "the live sort must merge chunks")
                    self.assertGreater(coverage["carried"], 0, "the frames must carry anchor history")
        state = _FrictionPatchState(_Scene(device, 0).model, capacity, wp.zeros(capacity, dtype=wp.vec2, device=device))
        self.assertEqual({name: getattr(state, name) for name in default}, default)

    @unittest.skipUnless(wp.is_cuda_available(), "launch shapes need CUDA")
    def test_launch_shapes_match_bitwise_at_device_scale(self):
        """Match the former device grids bitwise where the live contacts outnumber every grid's threads.

        A full frame of the scene's copies (:class:`_TiledScene`) holds a
        quarter more live contacts than _build's grid has threads, eight
        256-thread blocks per SM, so no grid is capped by the capacity and each
        launch strides over the live slots. The states compare after each
        finish, store and reset, with and without frame reuse.
        """
        device = wp.get_device("cuda:0")
        widest = _SHAPE["build"] * device.sm_count * 256
        worlds, contacts, capacity = 64, 2048, 2304
        copies = -(-5 * widest // (4 * contacts))
        shapes = (None, _shape(capacity * copies, device.sm_count, **_FORMER_SHAPE))
        with _patch_variants_compiled_here(device):
            for reuse in (False, True):
                with self.subTest(reuse=reuse):
                    coverage = run_launch_shapes(
                        device,
                        4,
                        shapes,
                        worlds=worlds,
                        contacts=contacts,
                        capacity=capacity,
                        reuse=reuse,
                        copies=copies,
                        stages=("finish", "store", "reset"),
                    )
                    self.assertGreater(
                        coverage["live"], widest, "the live contacts must outnumber every grid's threads"
                    )
                    self.assertGreater(coverage["carried"], 0, "the frames must carry anchor history")

    def test_wide_rows_stay_narrow_with_static_pairs(self):
        """Keep the device-scale row arrays as wide as one copy's fullest world, whatever the copy count.

        Static pairs have no body and so no world of their own. Gathering every
        copy's static contacts in world 0 made the rows grow with the copies,
        past the device's memory at device scale.
        """
        copies, worlds, per_world = 64, 4, 16
        live = copies * worlds * per_world
        scene = SimpleNamespace(worlds=copies * worlds, body_world=np.arange(copies * worlds, dtype=np.int32))
        static = np.full(live, -1, dtype=np.int32)
        reference = SimpleNamespace(
            current=SimpleNamespace(body_a=wp.array(static, device="cpu"), body_b=wp.array(static, device="cpu")),
            view=SimpleNamespace(weight=wp.ones(live, dtype=float, device="cpu")),
        )
        rows = _WideRows(scene, live, np.random.default_rng(7))
        rows.generate(reference, live)
        self.assertLessEqual(rows.rows, 8 * 3 * per_world + 8, "a world's rows must not grow with the copies")


def _ring_case(device, seed, *, contacts=600, worlds=3):
    """Return random patch rings, their contact maps and route rows for one link.

    Most rings hold one contact: anchors with three slots, carry-only contacts
    (zero weight), anchors with a normal row only, and contacts without rows.
    The others hold two to five members, some of which sit on another route or
    world than their ring or have no rows. The last contacts belong to no ring.
    Returns the patch view, the live count, the contact maps, numpy parents and
    friction budgets per route, and the counts of one-contact rings with rows.
    """
    rng = np.random.default_rng(seed)
    live = contacts - 24
    order = rng.permutation(live)
    sizes = []
    while sum(sizes) < live:
        sizes.append(int(rng.choice([1, 1, 1, 1, 1, 2, 3, 5])))
    sizes[-1] -= sum(sizes) - live
    next_contact = np.full(contacts, -1, dtype=np.int32)
    weight = np.zeros(contacts, dtype=np.float32)
    world = rng.integers(0, worlds, size=contacts)
    path = rng.integers(0, 3, size=contacts)
    start = 0
    for size in sizes:
        ring = order[start : start + size]
        start += size
        next_contact[ring] = np.roll(ring, -1)
        anchors = min(size, int(rng.integers(0, 3)))
        weight[ring[:anchors]] = 1.0 / max(anchors, 1)
        world[ring] = rng.integers(0, worlds)
        path[ring] = rng.integers(0, 3)
        stray = ring[rng.random(size) < 0.1]
        world[stray] = rng.integers(0, worlds, size=len(stray))
        path[stray] = rng.integers(0, 3, size=len(stray))
    needed = np.where((weight > 0.0) & (rng.random(contacts) < 0.8), 3, 1)
    rows = 3 * contacts + 8
    slot = np.full(contacts, -1, dtype=np.int64)
    counters = np.zeros((3, worlds), dtype=np.int64)
    for c in range(contacts):
        if rng.random() < 0.08:
            continue  # A capacity rejection, or a route without rows.
        slot[c] = counters[path[c], world[c]]
        counters[path[c], world[c]] += needed[c]
    parents = rng.integers(-1, rows, size=(3, worlds, rows)).astype(np.int32)
    mu = rng.uniform(0.2, 1.0, size=(3, worlds, rows)).astype(np.float32)
    view = FrictionPatches()
    view.weight = wp.array(weight, dtype=float, device=device)
    view.next_contact = wp.array(next_contact, dtype=int, device=device)
    view.point = wp.zeros(contacts, dtype=wp.vec3, device=device)
    view.phi = wp.zeros(contacts, dtype=wp.vec2, device=device)
    view.spin_phi = wp.zeros(contacts, dtype=float, device=device)
    view.spin_arm = wp.zeros(contacts, dtype=float, device=device)
    maps = [wp.array(values.astype(np.int32), dtype=int, device=device) for values in (world, slot, path, needed)]
    single = (next_contact == np.arange(contacts)) & (slot >= 0)
    singles = {
        "anchors with three slots": int(np.count_nonzero(single & (weight > 0.0) & (needed == 3))),
        "carry-only contacts": int(np.count_nonzero(single & (weight == 0.0))),
        "anchors with one slot": int(np.count_nonzero(single & (weight > 0.0) & (needed == 1))),
    }
    return view, live, maps, (parents, mu), singles


def run_ring_links(test, device, seed):
    """Link random rings with the patch link and with the reference's per-route kernel; require equal rows.

    ``link_patch_routes`` (all routes in one launch, one thread per contact or
    seven striding threads) must write the reference's parents and divided
    friction budgets bitwise.
    """
    view, live, maps, (parents, mu), singles = _ring_case(device, seed)
    for name, count in singles.items():
        test.assertGreater(count, 0, f"one-contact rings: {name}")
    contacts = view.weight.shape[0]
    count = wp.array([live], dtype=int, device=device)

    def rows():
        return [
            (
                route,
                wp.array(parents[route], dtype=int, device=device),
                wp.array(mu[route], dtype=float, device=device),
                wp.zeros(mu[route].shape, dtype=float, device=device),
                row_lines(device),
            )
            for route in range(3)
        ]

    def assert_rows(mine, context):
        for (route, *arrays), (_, *reference) in zip(mine, expected, strict=True):
            for name, array, desired in zip(("parents", "mu"), arrays[:2], reference[:2], strict=True):
                np.testing.assert_array_equal(
                    array.numpy(), desired.numpy(), f"seed {seed} {context} route {route} {name}"
                )

    expected = rows()
    for route, route_parents, route_mu, _, _ in expected:
        wp.launch(
            reference_link_patch_rows,
            dim=contacts,
            inputs=[count, view, *maps, route, route_parents, route_mu],
            device=device,
        )
    for threads in (contacts, 7):
        joint = rows()
        wp.launch(
            link_patch_routes,
            dim=threads,
            inputs=[wp.array([live, live], dtype=int, device=device), view, *maps, patch_routes(joint), threads],
            device=device,
        )
        assert_rows(joint, f"link_patch_routes on {threads} threads")


class TestPatchRingLinks(unittest.TestCase):
    def test_links_match_the_reference_on_random_rings(self):
        """Link one-contact and longer rings on every route as the reference does, bitwise."""
        for seed in range(3):
            with self.subTest(seed=seed):
                run_ring_links(self, "cpu", seed)

    @unittest.skipUnless(wp.is_cuda_available(), "the device link needs CUDA")
    def test_links_match_the_reference_on_the_device(self):
        """Link one-contact and longer rings on every route as the reference does on the device, bitwise."""
        for seed in range(3):
            with self.subTest(seed=seed):
                run_ring_links(self, "cuda:0", 10 + seed)


def _bare_solver(**attributes):
    """Return a ``SolverFeatherPGS`` that skipped construction, with ``attributes`` and a deviceless model."""
    solver = object.__new__(SolverFeatherPGS)
    solver.model = SimpleNamespace(device=SimpleNamespace(is_cuda=False))
    for name, value in attributes.items():
        setattr(solver, name, value)
    return solver


# The capture scope that host-emulated captures report.
_HOST_SCOPE = ("host", None)


_CONTACT_FIELDS = (
    "rigid_contact_shape0",
    "rigid_contact_shape1",
    "rigid_contact_point0",
    "rigid_contact_point1",
    "rigid_contact_normal",
    "rigid_contact_margin0",
    "rigid_contact_margin1",
)


def _swapped_stream(scene, live, rng, stream=None):
    """Return another contact stream of equal generation and count, holding the live contacts reordered.

    ``stream`` receives the copy in place, for a graph captured over it.
    """
    if stream is None:
        names = ("contact_generation", "rigid_contact_count", *_CONTACT_FIELDS)
        stream = SimpleNamespace(
            rigid_contact_max=scene.capacity, **{name: wp.empty_like(getattr(scene.contacts, name)) for name in names}
        )
    order = np.concatenate((rng.permutation(live), np.arange(live, scene.capacity)))
    for name in _CONTACT_FIELDS:
        getattr(stream, name).assign(getattr(scene.contacts, name).numpy()[order])
    stream.contact_generation.assign(scene.contacts.contact_generation)
    stream.rigid_contact_count.assign(scene.contacts.rigid_contact_count)
    return stream


def _frame_identity(patches, contacts):
    """Return the host's view of a contact frame: stream token, generation and raw count."""
    generation = contacts.contact_generation
    return (
        patches.stream_token(generation),
        int(generation.numpy()[0]),
        int(contacts.rigid_contact_count.numpy()[0]),
    )


def _assert_record(test, patches, sorted_frame, context):
    """Require the device record to name ``sorted_frame``, or to hold the sentinel generation for ``None``."""
    record = patches.sorted_frame.numpy()
    if sorted_frame is None:
        test.assertEqual(int(record[0]), -1, f"{context} record")
    else:
        np.testing.assert_array_equal(record, sorted_frame[1:] + sorted_frame[:1], f"{context} record")


@wp.kernel(enable_backward=False)
def _nudge_poses(q: wp.array[wp.transform], step: float):
    """Translate each body by a small body-dependent offset, as integrating a substep would."""
    b = wp.tid()
    pose = q[b]
    offset = step * wp.vec3(float(b % 3) - 1.0, float(b % 5) - 2.0, 1.0)
    q[b] = wp.transform(wp.transform_get_translation(pose) + offset, wp.transform_get_rotation(pose))


class TestPatchFrameReuse(unittest.TestCase):
    def test_patches_without_reuse_keep_the_original_chain(self):
        """Launch the unconditional chain when frame reuse is off or the contacts carry no generation."""
        scene = _Scene("cpu", 7)
        scene.frame()
        launched = []
        original = wp.launch

        def record(kernel, *args, **kwargs):
            launched.append(kernel.key)
            return original(kernel, *args, **kwargs)

        anonymous = SimpleNamespace(**{k: v for k, v in vars(scene.contacts).items() if k != "contact_generation"})
        phi = wp.zeros(128, dtype=wp.vec2, device="cpu")
        for reuse, contacts in ((False, scene.contacts), (True, anonymous)):
            with self.subTest(reuse=reuse):
                patches = _FrictionPatchState(scene.model, 128, phi, reuse_frames=reuse)
                launched.clear()
                with mock.patch.object(wp, "launch", record):
                    patches.build(scene.model, scene.state, contacts)
                self.assertEqual(launched, ["_prepare", "_build"])
                np.testing.assert_array_equal(patches.sorted_frame.numpy(), -1)


def _reused_build(patches, scene, contacts, mode=friction_patches._FRAME_KEEP):
    """Build ``contacts`` as an admitted reuse does: _prepare_frame in ``mode``, no pair sort, then the pairs."""
    generation = contacts.contact_generation
    inputs = patches._prepare_inputs(scene.model, scene.state, contacts)
    patches._launch_prepare_frame(inputs, generation, patches.stream_token(generation), mode)
    patches._launch_build(scene.model, scene.state)


def _captured(patches, scope=_HOST_SCOPE):
    """Make ``patches`` build as captured in ``scope`` (a callable returning the scope varies it) and return it."""
    patches._capture_scope = scope if callable(scope) else lambda: scope
    return patches


def _sort_every_build(patches):
    """Make every later build of ``patches`` sort through _prepare_frame, as a captured build that keeps nothing does.

    Each build reports a capture scope of its own, so none keeps the previous
    build's pairs. A chain of such builds prepares through the kernel that
    captured builds launch, so a graph must match it bitwise whatever code the
    compiler gives _prepare_frame and _prepare; :func:`run_sorting_prepare_parity`
    compares those two.
    """
    if not patches.reuse_frames:
        raise AssertionError("sorting every build through _prepare_frame needs frame reuse")
    serials = itertools.count()
    return _captured(patches, lambda: ("always sort", next(serials)))


def _counted_sorts(patches):
    """Return the list each pair sort of ``patches`` appends its live count to."""
    sorts = []
    order = patches._order_pairs

    def counted(**kwargs):
        sorts.append(int(patches.extent.numpy()[1]))
        order(**kwargs)

    patches._order_pairs = counted
    return sorts


def _recording_pair(scene, capacity):
    """Return a patch state without frame reuse and a frame-reusing one that builds as captured."""
    phi = [wp.zeros(capacity, dtype=wp.vec2, device=scene.device) for _ in range(2)]
    plain = _FrictionPatchState(scene.model, capacity, phi[0])
    recording = _captured(_FrictionPatchState(scene.model, capacity, phi[1], reuse_frames=True))
    return plain, recording


def run_sorting_prepare_parity(test, device, *, worlds=3, contacts=96, capacity=128):
    """Prepare and store new frames bitwise as _prepare does, through _prepare_frame in sort mode.

    The frame-reusing state builds as captured, so every new frame takes
    _prepare_frame in sort mode, which records the frame in the same launch,
    followed by the same sort as the state without frame reuse. Frames cover live, sparse, overflowing, empty and gated
    contacts, including static and missing shapes and flipped pairs.
    """
    rng = np.random.default_rng(41)
    scene = _Scene(device, 41, worlds=worlds, contacts=contacts)
    plain, recording = _recording_pair(scene, capacity)
    launched = []
    original = wp.launch

    def record(kernel, *args, **kwargs):
        launched.append(kernel.key)
        return original(kernel, *args, **kwargs)

    for step, mode in enumerate(("live", "live", "sparse", "overflow", "live", "empty", "gates", "live")):
        context = f"{device} step {step} {mode}"
        scene.move()
        scene.frame("live" if mode == "gates" else mode)
        gates = _gates(scene, rng) if mode == "gates" else {}
        plain.build(scene.model, scene.state, scene.contacts, **gates)
        launched.clear()
        with mock.patch.object(wp, "launch", record):
            recording.build(scene.model, scene.state, scene.contacts, **gates)
        test.assertEqual(launched.count("_prepare_frame"), 1, context)
        assert_same_state(recording, plain, f"{context} build", records=False)
        _assert_record(test, recording, _frame_identity(recording, scene.contacts), context)
        for state in (plain, recording):
            store_frame(state, scene.state)
        assert_same_state(recording, plain, f"{context} store", records=False)


# Capture scopes of the admission table: a graph's main body, a conditional body of
# it, and another graph whose Python object reuses the first one's id.
_MAIN, _BODY, _OTHER_GRAPH = (1, None), (1, 0xB0D1), (2, None)
# Admission sequences: ("build", scope, stream) builds a stream's frame in a scope
# (None eagerly), ("write", stream) notes a write of a stream, ("forget",) drops
# the recorded frame. Each lists the outcome of its builds.
_ADMISSIONS = {
    "collide, step, step": ((("write", "a"), ("build", _MAIN, "a"), ("build", _MAIN, "a")), ("sort", "keep")),
    "step, step": ((("build", _MAIN, "a"), ("build", _MAIN, "a")), ("sort", "keep")),
    "every later substep": (
        (("build", _MAIN, "a"), ("build", _MAIN, "a"), ("build", _MAIN, "a")),
        ("sort", "keep", "keep"),
    ),
    "write between steps": ((("build", _MAIN, "a"), ("write", "a"), ("build", _MAIN, "a")), ("sort", "sort")),
    "build in a body": ((("build", _MAIN, "a"), ("build", _BODY, "a")), ("sort", "sort")),
    "build after a body": ((("build", _BODY, "a"), ("build", _MAIN, "a")), ("sort", "sort")),
    "another stream": (
        (("build", _MAIN, "a"), ("build", _MAIN, "b"), ("build", _MAIN, "a")),
        ("sort", "sort", "sort"),
    ),
    "forget": ((("build", _MAIN, "a"), ("forget",), ("build", _MAIN, "a")), ("sort", "sort")),
    "another graph": ((("build", _MAIN, "a"), ("build", _OTHER_GRAPH, "a")), ("sort", "sort")),
    "eager": ((("build", None, "a"), ("build", None, "a")), ("sort", "sort")),
    "eager between steps": (
        (("build", _MAIN, "a"), ("build", None, "a"), ("build", _MAIN, "a")),
        ("sort", "sort", "sort"),
    ),
}


def _admitted_modes(patches):
    """Return the list each build of ``patches`` appends its outcome to: "keep" or "sort"."""
    outcomes = []
    launch, plain = patches._launch_prepare_frame, patches._prepare_and_sort
    names = {int(friction_patches._FRAME_KEEP): "keep", int(friction_patches._FRAME_SORT): "sort"}

    def launched(prepare_inputs, generation, stream, mode):
        outcomes.append(names[int(mode)])
        launch(prepare_inputs, generation, stream, mode)

    def sorted_plainly(prepare_inputs):
        outcomes.append("sort")
        plain(prepare_inputs)

    patches._launch_prepare_frame = launched
    patches._prepare_and_sort = sorted_plainly
    return outcomes


class TestPatchFrameModes(unittest.TestCase):
    def test_admission_keeps_only_unwritten_frames_of_the_previous_build_scope(self):
        """Keep a build's pairs only after a build captured in its scope over its stream with no noted write since.

        Both streams hold the same live contacts, and neither has a noted write,
        so only the stream identity tells them apart; no kept build may find
        another frame than the recorded one.
        """
        rng = np.random.default_rng(45)
        for name, (events, expected) in _ADMISSIONS.items():
            with self.subTest(name=name):
                scene = _Scene("cpu", 45)
                scene.active[:] = True
                live = scene.frame()
                streams = {"a": _swapped_stream(scene, live, rng), "b": _swapped_stream(scene, live, rng)}
                _, patches = _recording_pair(scene, 128)
                scope = [None]
                _captured(patches, lambda scope=scope: scope[0])
                outcomes = _admitted_modes(patches)
                for event in events:
                    if event[0] == "build":
                        scope[0] = event[1]
                        patches.build(scene.model, scene.state, streams[event[2]])
                    elif event[0] == "write":
                        note_contact_frame_write(streams[event[1]].contact_generation)
                    else:
                        patches.forget_frame()
                self.assertEqual(tuple(outcomes), expected)
                np.testing.assert_array_equal(patches.frame_status.numpy(), [0])

    def test_admission_follows_the_generation_array_through_contacts_wrappers(self):
        """Admit builds through wrappers of a contact stream as builds of the stream, keyed by its generation array.

        A fresh object around the stream's arrays, as a caller may build per
        substep, keeps the pairs that the previous build of the stream sorted in
        the same scope, and a write noted on the stream makes it sort. Contacts
        without a generation array sort and end the chain. The wrappers carry
        none of the private state of :class:`~newton.Contacts`, and every build
        matches a chain that sorts each frame. The noted write reorders the
        stream's contacts and keeps its generation and count, and the contacts
        without a generation hold another order, so a build that kept its pairs
        after either would group contacts by stale pairs that the device check
        cannot see.
        """
        rng = np.random.default_rng(46)
        scene = _Scene("cpu", 46)
        scene.active[:] = True
        live = scene.frame()
        stream = _swapped_stream(scene, live, rng)
        other = _swapped_stream(scene, live, rng)
        anonymous = SimpleNamespace(**{k: v for k, v in vars(other).items() if k != "contact_generation"})
        sources = {
            "stream": lambda: stream,
            "wrapper": lambda: SimpleNamespace(**vars(stream)),
            "anonymous": lambda: anonymous,
        }
        plain, patches = _recording_pair(scene, 128)
        outcomes = _admitted_modes(patches)
        for event in ("stream", "wrapper", "write", "wrapper", "stream", "anonymous", "wrapper", "wrapper"):
            if event == "write":
                note_contact_frame_write(stream.contact_generation)
                _swapped_stream(scene, live, rng, stream=stream)
                continue
            contacts = sources[event]()
            for state in (plain, patches):
                state.build(scene.model, scene.state, contacts)
            assert_same_state(patches, plain, event, records=False)
        self.assertEqual(outcomes, ["sort", "keep", "sort", "keep", "sort", "sort", "keep"])
        np.testing.assert_array_equal(patches.frame_status.numpy(), [0])

    def test_capture_kernels_load_at_construction(self):
        """Load the module of the kernel that captured builds launch first when a CUDA patch state is built.

        Without it, the first capture would load that module. A stand-in
        CUDA device lets the CPU check which modules load; a CPU state loads
        none, since CPU captures build as eager ones do.
        """
        scene = _Scene("cpu", 49)
        phi = wp.zeros(128, dtype=wp.vec2, device="cpu")
        cuda = SimpleNamespace(is_cuda=True)
        for reuse, expected in ((False, []), (True, ["_prepare_frame"])):
            with self.subTest(reuse=reuse):
                loaded = []
                kernel = friction_patches._prepare_frame
                with mock.patch.object(
                    kernel.module,
                    "load",
                    side_effect=lambda device, key=kernel.key, loaded=loaded: loaded.append((key, device)),
                ):
                    patches = _FrictionPatchState(scene.model, 128, phi, reuse_frames=reuse)
                    self.assertEqual(loaded, [], "a CPU state loads no module")
                    patches.device = cuda
                    patches._load_capture_modules()
                self.assertEqual([key for key, _ in loaded], expected)
                self.assertTrue(all(device is cuda for _, device in loaded))

    @unittest.skipUnless(wp.is_cuda_available(), "module loads per device need CUDA")
    def test_solver_loads_the_frame_reuse_prepare_before_capture(self):
        """Load _prepare_frame's module on the solver's CUDA device at construction, so no capture loads it."""
        module = friction_patches._prepare_frame.module
        with mock.patch.object(module, "load", wraps=module.load) as load:
            twin = _SolverTwin("cuda:0")
        self.assertTrue(twin.patches.reuse_frames)
        load.assert_any_call(twin.patches.device)
        self.assertIn((twin.patches.device.context, module.options["block_dim"]), module.execs)

    def test_forget_frame_makes_the_next_build_sort_without_a_launch(self):
        """Make the next build sort after forget_frame() and leave the device record, which no build trusts alone.

        Isaac Lab notifies model changes eagerly before most graph launches, so
        frame reuse must not add a launch there.
        """
        scene = _Scene("cpu", 47)
        scene.frame()
        phi = wp.zeros(128, dtype=wp.vec2, device="cpu")
        patches = _FrictionPatchState(scene.model, 128, phi, reuse_frames=True)
        _captured(patches)
        patches.build(scene.model, scene.state, scene.contacts)
        identity = _frame_identity(patches, scene.contacts)
        _assert_record(self, patches, identity, "sorted")
        self.assertIsNotNone(patches._last_build)
        with mock.patch.object(wp.array, "fill_", autospec=True, side_effect=wp.array.fill_) as fill:
            patches.forget_frame()
        self.assertIsNone(patches._last_build)
        self.assertEqual(fill.call_count, 0)
        _assert_record(self, patches, identity, "forgotten")

    def test_sorting_prepare_matches_prepare_and_records_the_frame(self):
        """Prepare bitwise as _prepare where the sort follows, and record the frame in the same launch."""
        run_sorting_prepare_parity(self, "cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "the CUDA build of the prepare kernels needs CUDA")
    def test_sorting_prepare_matches_prepare_on_the_device(self):
        """Prepare bitwise as _prepare on CUDA, where each kernel's float code is NVRTC's choice.

        This decides whether the default's captured builds, which prepare through
        _prepare_frame, step bitwise as builds without frame reuse, which prepare
        through _prepare. The graph tests compare with a chain that prepares
        through _prepare_frame too (:func:`_sort_every_build`), so their
        admission check holds either way.
        """
        run_sorting_prepare_parity(self, "cuda:0", worlds=48, contacts=2048, capacity=3072)

    def test_keep_mode_reuses_the_sorted_pairs_of_the_recorded_frame(self):
        """Keep a recorded frame's sorted pairs on the substeps that share it, bitwise as builds that sort it anew."""
        scene = _Scene("cpu", 42)
        plain, recording = _recording_pair(scene, 128)
        sorts = _counted_sorts(recording)
        kept = 0
        for step, mode in enumerate(("live", "sparse", "live", "empty", "live", "overflow", "live")):
            scene.move()
            live = scene.frame(mode)
            for substep in range(3):
                context = f"step {step} {mode} substep {substep}"
                if substep:
                    scene.move()
                before = len(sorts)
                for state in (plain, recording):
                    state.build(scene.model, scene.state, scene.contacts)
                self.assertEqual(len(sorts) - before, int(substep == 0), f"{context} sorts")
                assert_same_state(recording, plain, context, records=False)
                _assert_record(self, recording, _frame_identity(recording, scene.contacts), context)
                np.testing.assert_array_equal(recording.frame_status.numpy(), [0], context)
                assert_empty_tail(recording, context)
                kept += live if substep else 0
                for state in (plain, recording):
                    store_frame(state, scene.state)
                assert_same_state(recording, plain, f"{context} store", records=False)
        self.assertGreater(kept, 0, "substeps must reuse live pairs")

    def test_keep_mode_preserves_the_sorted_pairs_of_live_slots(self):
        """Preserve the pair keys and indices that a recorded frame's live slots hold.

        Values cannot tell a keep that never writes these slots from one that
        reads and restores them, which costs about 50 us per kept build at
        keyboard scale; the GPU probe's keep-mode prepare time gates that.
        """
        scene = _Scene("cpu", 44)
        _, patches = _recording_pair(scene, 128)
        scene.active[:] = True
        live = scene.frame()
        self.assertGreater(live, 0)
        patches.build(scene.model, scene.state, scene.contacts)
        # CPU arrays expose their memory; copies keep what the slots held.
        keys, indices = patches.current.keys.numpy().copy(), patches.current.indices.numpy().copy()
        # Values no prepare writes: a kept slot must hold them afterwards.
        keys[:live] = np.arange(live) + 12345
        indices[:live] = np.arange(live)[::-1] + 1000
        patches.current.keys.assign(keys)
        patches.current.indices.assign(indices)
        generation = scene.contacts.contact_generation
        inputs = patches._prepare_inputs(scene.model, scene.state, scene.contacts)
        patches._launch_prepare_frame(
            inputs, generation, patches.stream_token(generation), friction_patches._FRAME_KEEP
        )
        np.testing.assert_array_equal(patches.current.keys.numpy(), keys)
        np.testing.assert_array_equal(patches.current.indices.numpy(), indices)
        np.testing.assert_array_equal(patches.extent.numpy()[:2], [live, live])

    def test_keep_mode_empties_a_frame_rewritten_since_its_record(self):
        """Leave a kept build without live contacts and latch the status when its frame is not the recorded one.

        Writing the generation or the count directly, unnoted, breaks the capture
        contract and is admitted as a reuse; another stream or a dropped record
        is not, and is forced here.
        """
        rng = np.random.default_rng(43)
        for change in ("generation", "count", "stream", "record"):
            with self.subTest(change=change):
                scene = _Scene("cpu", 43)
                _, patches = _recording_pair(scene, 128)
                sorts = _counted_sorts(patches)
                scene.active[:] = True
                live = scene.frame()
                self.assertGreater(live, 1)
                patches.build(scene.model, scene.state, scene.contacts)
                store_frame(patches, scene.state)
                self.assertTrue(np.any(patches.previous.valid.numpy()), "the frame must store anchors")
                contacts = scene.contacts
                if change == "generation":
                    contacts.contact_generation.assign([scene.generation + 1])
                elif change == "count":
                    contacts.rigid_contact_count.assign([live - 1])
                elif change == "stream":
                    contacts = _swapped_stream(scene, live, rng)
                else:
                    patches.sorted_frame.fill_(-1)
                if change in ("generation", "count"):
                    patches.build(scene.model, scene.state, contacts)
                    self.assertEqual(len(sorts), 1, "the unnoted rewrite is admitted as a reuse")
                else:
                    _reused_build(patches, scene, contacts)
                np.testing.assert_array_equal(patches.extent.numpy()[:2], [live, 0])
                np.testing.assert_array_equal(patches.frame_status.numpy(), [1])
                self.assertEqual(int(patches.sorted_frame.numpy()[0]), -1, "the record is dropped")
                np.testing.assert_array_equal(patches.current.keys.numpy()[: patches.capacity], SENTINEL)
                assert_empty_tail(patches, change)
                store_frame(patches, scene.state)
                self.assertFalse(np.any(patches.previous.valid.numpy()), "the empty frame releases every anchor")
                assert_empty_tail(patches, f"{change} store")
                # A noted write lets the next build sort and record again; only the host clears the status.
                note_contact_frame_write(contacts.contact_generation)
                patches.build(scene.model, scene.state, contacts)
                _assert_record(self, patches, _frame_identity(patches, contacts), f"{change} sort")
                np.testing.assert_array_equal(patches.frame_status.numpy(), [1])

    def test_keep_mode_misses_unnoted_rewrites_of_equal_generation_and_count(self):
        """Keep the pairs through unnoted in-place rewrites that keep the generation and count, as documented.

        The device guard compares the generation and the raw count only. A
        rewrite of points, normals and margins alone keeps every pair, so the
        kept build equals a sorting build bitwise. Reordering the contacts in
        place goes unseen: the kept build groups them by the previous order's
        pairs, with indices still below the live count, and reports nothing.
        """
        rng = np.random.default_rng(48)
        for rewrite in ("geometry", "order"):
            with self.subTest(rewrite=rewrite):
                scene = _Scene("cpu", 48)
                scene.active[:] = True
                plain, recording = _recording_pair(scene, 128)
                sorts = _counted_sorts(recording)
                live = scene.frame()
                self.assertGreater(live, 2)
                for state in (plain, recording):
                    state.build(scene.model, scene.state, scene.contacts)
                kept = recording.current.indices.numpy()[:live].copy()
                contacts = scene.contacts
                if rewrite == "geometry":
                    for name in ("rigid_contact_point0", "rigid_contact_point1", "rigid_contact_normal"):
                        values = getattr(contacts, name).numpy()
                        values[:live] += rng.normal(scale=1.0e-3, size=(live, 3))
                        getattr(contacts, name).assign(values)
                    margins = contacts.rigid_contact_margin0.numpy()
                    margins[:live] = rng.uniform(0.0, 2.0e-3, size=live)
                    contacts.rigid_contact_margin0.assign(margins)
                else:
                    _swapped_stream(scene, live, rng, contacts)
                for state in (plain, recording):
                    state.build(scene.model, scene.state, contacts)
                self.assertEqual(len(sorts), 1, "the unnoted rewrite is admitted as a reuse")
                np.testing.assert_array_equal(recording.frame_status.numpy(), [0], "the guard cannot see it")
                self.assertEqual(int(recording.extent.numpy()[1]), live)
                if rewrite == "geometry":
                    assert_same_state(recording, plain, rewrite, records=False)
                    continue
                indices = recording.current.indices.numpy()[:live]
                np.testing.assert_array_equal(indices, kept, "the kept build holds the previous order")
                self.assertFalse(np.array_equal(indices, plain.current.indices.numpy()[:live]), "the order is stale")
                np.testing.assert_array_equal(np.sort(indices), np.arange(live), "indices stay below the live count")


def _expected_pairs(model, contacts, capacity):
    """Return the sorted pair keys and slot order of the live contacts, computed on the host."""
    bodies, shapes = model.body_count, model.shape_body.shape[0]
    stride = bodies + shapes + 2
    count = int(contacts.rigid_contact_count.numpy()[0])
    live = min(count, capacity) if 0 <= count <= contacts.rigid_contact_max else 0
    shape_body = model.shape_body.numpy()

    def codes(shape):
        body = np.where(shape >= 0, shape_body[np.maximum(shape, 0)], -1)
        return np.where(body >= 0, body + 1, bodies + np.where(shape >= 0, shape, shapes) + 1).astype(np.int64)

    a = codes(contacts.rigid_contact_shape0.numpy()[:live])
    b = codes(contacts.rigid_contact_shape1.numpy()[:live])
    keys = np.minimum(a, b) * stride + np.maximum(a, b)
    order = np.argsort(keys, kind="stable")
    return keys[order], order


def _permute_contacts(contacts, target, order):
    """Copy ``contacts`` into ``target`` with its slots reordered, keeping the generation and the count."""
    for name, value in vars(contacts).items():
        if (
            name.startswith("rigid_contact_")
            and isinstance(value, wp.array)
            and value.ndim == 1
            and value.shape[0] == contacts.rigid_contact_max
        ):
            getattr(target, name).assign(value.numpy()[order])
    target.rigid_contact_count.assign(contacts.rigid_contact_count)
    target.contact_generation.assign(contacts.contact_generation)


def _free(device):
    """Collect dropped arrays and let ``device`` return their memory to its pool."""
    gc.collect()
    wp.synchronize_device(device)


def _replace_first_stream(twin, order):
    """Drop a twin's first contact stream and copy its frame, reordered by ``order``, into a new one.

    The new ``Contacts`` takes a generation counter that aliases the dropped counter's memory, which stays alive on
    the twin: a new stream at a freed stream's counter address, the case a memory pool that hands the block to the
    next ``Contacts`` produces, built directly instead of left to where the pool places the allocation. Returns
    whether the new counter sits at the dropped counter's address, which holds by construction.
    """
    stream = twin.streams[0]
    twin.dropped_counter = counter = stream.contact_generation
    frame = SimpleNamespace(
        rigid_contact_max=stream.rigid_contact_max,
        **{name: wp.clone(value, device="cpu") for name, value in vars(stream).items() if isinstance(value, wp.array)},
    )
    twin.streams[0] = stream = None
    _free(twin.model.device)
    replacement = twin.pipeline.contacts()
    replacement.contact_generation = wp.array(
        ptr=counter.ptr, dtype=counter.dtype, shape=counter.shape, device=counter.device
    )
    twin.streams[0] = replacement
    _permute_contacts(frame, replacement, order)
    return replacement.contact_generation.ptr == counter.ptr


def _falling_boxes(device):
    """Free boxes in three worlds, released just above a ground plane."""
    world = newton.ModelBuilder(up_axis=newton.Axis.Z)
    world.rigid_gap = 0.005
    box = world.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.06), wp.quat_rpy(0.05, 0.02, 0.1)))
    world.add_shape_box(box, hx=0.02, hy=0.03, hz=0.02, cfg=newton.ModelBuilder.ShapeConfig(density=500.0, mu=0.6))
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.rigid_gap = 0.005
    builder.replicate(world, 3)
    builder.add_ground_plane()
    with wp.ScopedDevice(device):
        return builder.finalize()


@contextlib.contextmanager
def _flags(*, reuse):
    """Set ``FEATHER_PGS_PATCH_FRAME_REUSE`` to ``reuse`` for the duration; ``None`` unsets it."""
    with mock.patch.dict(os.environ):
        os.environ.pop("FEATHER_PGS_PATCH_FRAME_REUSE", None)
        if reuse is not None:
            os.environ["FEATHER_PGS_PATCH_FRAME_REUSE"] = reuse
        yield


class _SolverTwin:
    """A FeatherPGS scene under default patch friction with two contact streams and double-buffered states.

    ``reuse=False`` sorts every patch build (``FEATHER_PGS_PATCH_FRAME_REUSE=0``);
    by default captured builds keep reused frames, and :meth:`sort_every_build`
    makes the twin the always-sort chain that graph twins compare with, which
    prepares through the same kernel as their captured builds.
    ``jaws`` selects the squeezing gripper of the friction-anchor tests, whose
    graphs capture two solver steps each, and ``boxes`` free boxes above a
    ground plane, whose contact frames are empty until they land, instead of the
    arms over boxes. Both pipelines sort the contacts, so twins collide into the
    same slots: the CUDA narrow phase assigns them in a run-dependent order.
    """

    def __init__(self, device, *, reuse=True, jaws=False, boxes=False):
        if jaws:
            from newton.tests.test_feather_pgs_friction_anchors import _SQUEEZE_SOLVER, _build_v_jaws  # noqa: PLC0415

            with wp.ScopedDevice(device):
                self.model, _, _ = _build_v_jaws(5.0)
            options = dict(_SQUEEZE_SOLVER)
            pipeline_options = {"rigid_contact_max": 256, "broad_phase": "nxn", "deterministic": True}
        elif boxes:
            self.model = _falling_boxes(device)
            options = {"pgs_iterations": 16}
            pipeline_options = {"rigid_contact_max": 128, "broad_phase": "nxn", "deterministic": True}
        else:
            self.model = _arm_over_box(wp.get_device(device), worlds=2)
            options, pipeline_options = {"pgs_iterations": 8}, {}
        with contextlib.ExitStack() as stack:
            if not reuse:
                stack.enter_context(mock.patch.dict(os.environ, {"FEATHER_PGS_PATCH_FRAME_REUSE": "0"}))
            self.solver = SolverFeatherPGS(self.model, **options)
        self.patches = self.solver.laws.friction_patches
        self.pipeline = newton.CollisionPipeline(self.model, **pipeline_options)
        self.streams = [self.pipeline.contacts(), self.pipeline.contacts()]
        self.states = [self.model.state(), self.model.state()]
        newton.eval_fk(self.model, self.states[0].joint_q, self.states[0].joint_qd, self.states[0])
        self.control = self.model.control()

    def substep(self, stream=0, collide=False):
        """Step once from ``states[0]`` into ``states[1]`` and swap them back, as a graph would not."""
        state, following = self.states
        contacts = self.streams[stream]
        if collide:
            self.pipeline.collide(state, contacts)
        state.clear_forces()
        self.solver.step(state, following, self.control, contacts, 1.0 / 240.0)
        self.states.reverse()

    def pair(self, stream=0, collide=False):
        """Two substeps that return to ``states[0]``, collided first when ``collide``."""
        self.substep(stream, collide)
        self.substep(stream)

    def sort_every_build(self):
        """Sort every later patch build through _prepare_frame (:func:`_sort_every_build`) and return the twin.

        Call it after the eager steps that precede a graph's capture, which a
        graph twin also prepares through _prepare.
        """
        _sort_every_build(self.patches)
        return self

    def throw(self, speed):
        """Give every free body of ``states[0]`` an upward velocity."""
        state = self.states[0]
        qd = state.joint_qd.numpy().reshape(-1, 6)
        qd[:, 2] = speed
        state.joint_qd.assign(qd.reshape(-1))
        body_qd = state.body_qd.numpy()
        body_qd[:, 2] = speed
        state.body_qd.assign(body_qd)

    def same_steps(self, other):
        """Whether two twins hold bitwise-equal bodies, joints and stored patch history."""
        try:
            _assert_same_twins(self, other, "")
        except AssertionError:
            return False
        return True

    def assert_sorted(self, test, stream, context):
        """Require the slots to hold the host-sorted pairs of a stream's live contacts."""
        keys, order = _expected_pairs(self.model, self.streams[stream], self.patches.capacity)
        live = len(keys)
        np.testing.assert_array_equal(
            pair_codes(self.patches.current.keys.numpy()[:live], self.patches.key_stride),
            pair_codes(keys, self.patches.key_stride),
            f"{context} keys",
        )
        np.testing.assert_array_equal(self.patches.current.indices.numpy()[:live], order, f"{context} indices")
        np.testing.assert_array_equal(
            self.patches.current.keys.numpy()[live : self.patches.capacity], SENTINEL, f"{context} empty keys"
        )
        test.assertGreater(live, 0, f"{context}: the box must touch its support")


def _change(twins, event, rng):
    """Apply a frame change between substeps to every twin; returns the contact stream the next substeps use."""
    if event in ("truncate", "write"):
        for twin in twins:
            count = int(twin.streams[0].rigid_contact_count.numpy()[0])
            if event == "write":
                note_contact_frame_write(twin.streams[0].contact_generation)
            twin.streams[0].rigid_contact_count.assign([count - 1])
    elif event == "notify":
        for twin in twins:
            twin.solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
    elif event == "swap":
        source = twins[0].streams[0]
        count = int(source.rigid_contact_count.numpy()[0])
        order = np.concatenate((rng.permutation(count), np.arange(count, source.rigid_contact_max)))
        for twin in twins:
            _permute_contacts(twin.streams[0], twin.streams[1], order)
        return 1
    return 0


# Substep pairs of the solver-level reuse tests: "collide" opens a collision frame,
# "step" shares the last one, "truncate" lowers the count and "notify" reports a
# model change before sharing it, "swap" moves to a reordered stream of equal
# generation and count, and "back" returns to the first stream.
_SOLVER_PLAN = (
    "collide",
    "collide",
    "step",
    "step",
    "collide",
    "truncate",
    "step",
    "notify",
    "swap",
    "back",
    "collide",
)


# Substep pairs of the static frame-reuse tests: as _SOLVER_PLAN, plus "write",
# which lowers the count as "truncate" does but notes the write, and "scope",
# which builds in another capture scope.
_STATIC_PLAN = (
    "collide",
    "step",
    "step",
    "write",
    "step",
    "notify",
    "swap",
    "back",
    "step",
    "scope",
    "step",
    "collide",
)


class TestPatchSolverFrameReuse(unittest.TestCase):
    def test_solver_records_its_frame_reuse(self):
        """Record frame reuse (on unless =0) with its reason."""
        devices = ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])
        for device in devices:
            model = _arm_over_box(wp.get_device(device), worlds=2)
            cases = (
                ({}, None, True),
                ({}, "0", False),
                ({}, "1", True),
                ({}, "", True),
            )
            for options, reuse, reused in cases:
                with self.subTest(device=device, options=options, reuse=reuse):
                    with _flags(reuse=reuse):
                        solver = SolverFeatherPGS(model, **options)
                    friction = solver.get_path_selection()["routes"]["friction"]
                    self.assertEqual(friction["frame_reuse"]["enabled"], reused)
                    self.assertEqual(solver.laws.friction_patches.reuse_frames, reused)
                    if reuse == "0":
                        self.assertIn("FEATHER_PGS_PATCH_FRAME_REUSE=0", friction["frame_reuse"]["reason"])
                    else:
                        self.assertIn("previous build captured in the same graph", friction["frame_reuse"]["reason"])
                    # The live sort does not depend on frame reuse.
                    pair_sort = friction["pair_sort"]
                    self.assertTrue(pair_sort["live"])
                    self.assertEqual(pair_sort["live"], solver.laws.friction_patches.live_sort)
                    self.assertEqual(pair_sort["capacity"], solver.laws.friction_patches.capacity)
                    self.assertEqual(pair_sort["key_bits"], solver.laws.friction_patches.key_bits)
                    self.assertIn("only the live slots", pair_sort["reason"])

    def test_frame_reuse_opt_out_changes_only_frame_reuse(self):
        """Record under FEATHER_PGS_PATCH_FRAME_REUSE=0 the same routes, the live pair sort included, but frame reuse."""
        self.assertEqual(fpgs_flags.FLAGS["FEATHER_PGS_PATCH_FRAME_REUSE"].kind, "policy")
        routes = {}
        for reuse in (None, "0"):
            with _flags(reuse=reuse):
                solver = SolverFeatherPGS(_arm_over_box(wp.get_device("cpu"), worlds=2))
            routes[reuse] = solver.get_path_selection()
            self.assertTrue(solver.laws.friction_patches.live_sort)
        self.assertIn("FEATHER_PGS_PATCH_FRAME_REUSE=0", routes["0"]["routes"]["friction"]["frame_reuse"]["reason"])
        for selected in routes.values():
            del selected["routes"]["friction"]["frame_reuse"]
            selected.pop("environment", None)
        self.assertEqual(routes[None], routes["0"])

    def test_route_records_the_pair_sort(self):
        """Record the live sort, or the radix sort with the reason when the pair keys are too wide."""
        cases = (
            # (live_sort, capacity, key_bits, reason)
            (True, 16_384, 51, "only the live slots"),
            (True, 2_000_000, 51, "only the live slots"),
            (False, 16_384, 52, "52-bit pair keys"),
        )
        for live, capacity, key_bits, reason in cases:
            with self.subTest(capacity=capacity, key_bits=key_bits):
                patches = SimpleNamespace(live_sort=live, capacity=capacity, key_bits=key_bits)
                self.assertEqual(friction_patches._uses_live_sort(key_bits), live)
                record = path_selection._pair_sort(patches)
                self.assertEqual(
                    {name: record[name] for name in ("live", "capacity", "key_bits")},
                    {"live": live, "capacity": capacity, "key_bits": key_bits},
                )
                self.assertIn(reason, record["reason"])

    def test_static_reuse_steps_bitwise_as_the_unconditional_chain(self):
        """Keep the sorted pairs on captured substeps without a noted write, bitwise as the always-sort chain.

        Builds run as captured in one scope, so every build after a collision
        pass, a noted contact write, a model change, another contact stream or
        another capture scope sorts, and every other build keeps its pairs; no
        frame is left empty.
        """
        rng = np.random.default_rng(15)
        reuse, plain = _SolverTwin("cpu"), _SolverTwin("cpu", reuse=False)
        self.assertTrue(reuse.patches.reuse_frames)
        scope = [_HOST_SCOPE]
        _captured(reuse.patches, lambda: scope[0])
        sorts = _counted_sorts(reuse.patches)
        for twin in (reuse, plain):
            twin.pair(collide=True)
        for index, event in enumerate(_STATIC_PLAN):
            context = f"pair {index} {event}"
            stream = _change((reuse, plain), event, rng)
            if event == "scope":
                scope[0] = ("host", index)
            for substep in range(2):
                expected = substep == 0 and event != "step"
                before = len(sorts)
                for twin in (reuse, plain):
                    twin.substep(stream, collide=substep == 0 and event == "collide")
                self.assertEqual(len(sorts) - before, int(expected), f"{context} substep {substep} sorts")
                reuse.assert_sorted(self, stream, f"{context} substep {substep}")
                _assert_same_twins(reuse, plain, f"{context} substep {substep}")
        self.assertFalse(reuse.solver.constraint_capacity_status()["patch_frames"])

    def test_unnoted_contact_writes_are_reported(self):
        """Solve a kept step whose contacts were rewritten unnoted without patch rows, and report it until cleared."""
        twin = _SolverTwin("cpu", boxes=True)
        _captured(twin.patches)
        for _ in range(40):
            twin.pair(collide=True)
            if np.any(twin.patches.previous.valid.numpy()):
                break
        self.assertGreater(int(twin.streams[0].rigid_contact_count.numpy()[0]), 1)
        self.assertTrue(np.any(twin.patches.previous.valid.numpy()), "the landed boxes must store anchors")
        twin.solver.check_constraint_capacity()
        self.assertFalse(twin.solver.constraint_capacity_status()["patch_frames"])
        # A caller's own kernel lowers the count between captured steps without a noted write.
        count = int(twin.streams[0].rigid_contact_count.numpy()[0])
        twin.streams[0].rigid_contact_count.assign([count - 1])
        twin.substep()
        self.assertTrue(np.all(np.isfinite(twin.states[0].body_qd.numpy())))
        self.assertFalse(np.any(twin.patches.previous.valid.numpy()), "the empty frame releases every anchor")
        with self.assertRaisesRegex(RuntimeError, "without patch friction"):
            twin.solver.check_constraint_capacity()
        self.assertTrue(twin.solver.constraint_capacity_status(clear=True)["patch_frames"])
        self.assertFalse(twin.solver.constraint_capacity_status()["patch_frames"])
        twin.solver.check_constraint_capacity()

    def test_frame_status_is_read_only_with_frame_reuse(self):
        """Copy the patch frame status to the host only where frame reuse can set it.

        Under FEATHER_PGS_PATCH_FRAME_REUSE=0 no build keeps its pairs, so status
        reads and checks copy no fifth integer and report the flag clear.
        """
        for reuse in (None, "0"):
            with self.subTest(reuse=reuse):
                with _flags(reuse=reuse):
                    solver = SolverFeatherPGS(_arm_over_box(wp.get_device("cpu"), worlds=2))
                real, calls = solver.laws.friction_patches.frame_status, []
                solver.laws.friction_patches.frame_status = SimpleNamespace(
                    numpy=lambda real=real, calls=calls: calls.append("read") or real.numpy(),
                    zero_=lambda real=real, calls=calls: calls.append("clear") or real.zero_(),
                )
                status = solver.constraint_capacity_status(clear=True)
                solver.check_constraint_capacity()
                self.assertFalse(status["patch_frames"])
                self.assertEqual(calls, ["read", "clear", "read"] if reuse is None else [])

    @unittest.skipUnless(wp.is_cuda_available(), "graph capture needs CUDA")
    def test_solver_graphs_reuse_sorted_frames(self):
        """Replay collide and substep graphs bitwise as eager always-sort steps, sorting once per graph.

        Each graph sorts its first build and keeps the second. A probe captured
        between the builds shows on every replay which of them sorted
        (:class:`_SortProbe`).
        """
        rng = np.random.default_rng(6)
        graph = _graph_twin(jaws=True)
        eager = _SolverTwin("cuda:0", jaws=True)
        for _ in range(3):
            for twin in (graph, eager):
                twin.pair(collide=True)
        eager.sort_every_build()
        captured = {}
        probe = _SortProbe(graph.patches, 2)
        outcomes, sorts = _admitted_modes(graph.patches), _counted_sort_calls(graph.patches)
        for name, stream, collide in (("collide", 0, True), ("step", 0, False), ("swap", 1, False)):
            with _capture() as capture:
                probe.start()
                probe.mark()
                graph.substep(stream, collide)
                probe.check(0)
                probe.mark()
                graph.substep(stream)
                probe.check(1)
            captured[name] = capture.graph
        self.assertEqual(outcomes, ["sort", "keep"] * 3)
        # A sorting build records its sort; a kept build records none.
        self.assertEqual(len(sorts), 3, "sorts the graphs record")
        for index, event in enumerate(_SOLVER_PLAN):
            context = f"pair {index} {event}"
            stream = _change((graph, eager), event, rng)
            wp.capture_launch(captured[event if event in ("collide", "swap") else "step"])
            eager.pair(stream, collide=event == "collide")
            np.testing.assert_array_equal(
                probe.intact.numpy(), [0, 1], f"{context}: the first build sorts and the second keeps its pairs"
            )
            graph.assert_sorted(self, stream, context)
            _assert_same_twins(graph, eager, context)
        self.assertFalse(graph.solver.constraint_capacity_status()["patch_frames"])

    @unittest.skipUnless(wp.is_cuda_available(), "graph capture needs CUDA")
    def test_graphs_sort_a_stream_allocated_at_a_freed_address(self):
        """Sort the first captured build over a new Contacts whose generation counter sits where a dropped one was."""
        rng = np.random.default_rng(8)
        graph = _graph_twin(jaws=True)
        eager = _SolverTwin("cuda:0", jaws=True)
        # Eager steps leave the slots holding the first stream's frame, recorded as sorted.
        for twin in (graph, eager):
            twin.pair(collide=True)
        eager.sort_every_build()
        count = int(graph.streams[0].rigid_contact_count.numpy()[0])
        order = np.concatenate((rng.permutation(count), np.arange(count, graph.streams[0].rigid_contact_max)))
        # The replacement holds the frame reordered, with the same generation and count.
        reused = [_replace_first_stream(twin, order) for twin in (graph, eager)]
        self.assertTrue(reused[0], "the replacement's generation counter must sit at the freed address")
        with _capture() as capture:
            graph.pair()
        _poison_sort_scratch(graph)
        wp.capture_launch(capture.graph)
        eager.pair()
        self.assertFalse(_sort_scratch_intact(graph), "the replacement must sort")
        graph.assert_sorted(self, 0, "replacement")
        _assert_same_twins(graph, eager, "replacement")

    @unittest.skipUnless(wp.is_cuda_available(), "graph capture needs CUDA")
    def test_graphs_with_a_caller_fork_across_steps_sort_their_first_build(self):
        """Step bitwise as eager always-sort steps while a stream the caller forked is unjoined.

        The first build sorts and the second keeps its pairs, as without the fork.
        """
        graph = _graph_twin(jaws=True)
        eager = _SolverTwin("cuda:0", jaws=True)
        for _ in range(3):
            for twin in (graph, eager):
                twin.pair(collide=True)
        eager.sort_every_build()
        side = wp.Stream("cuda:0")
        overlapped = wp.zeros(1024, dtype=float, device="cuda:0")
        outcomes = _admitted_modes(graph.patches)
        with wp.ScopedCapture(device="cuda:0") as capture:
            main = wp.get_stream("cuda:0")
            side.wait_event(main.record_event())
            with wp.ScopedStream(side, sync_enter=False):
                overlapped.fill_(1.0)
            graph.pair()
            main.wait_event(side.record_event())
        self.assertEqual(outcomes, ["sort", "keep"])
        _poison_sort_scratch(graph)
        wp.capture_launch(capture.graph)
        eager.pair()
        self.assertFalse(_sort_scratch_intact(graph), "the first build must sort")
        np.testing.assert_array_equal(overlapped.numpy(), 1.0)
        graph.assert_sorted(self, 0, "caller fork")
        _assert_same_twins(graph, eager, "caller fork")

    @unittest.skipUnless(wp.is_cuda_available(), "graph capture needs CUDA")
    def test_two_solvers_share_a_graph(self):
        """Capture two solvers in one graph."""
        graphs = (_graph_twin(jaws=True), _graph_twin(boxes=True))
        eagers = (_SolverTwin("cuda:0", jaws=True), _SolverTwin("cuda:0", boxes=True))
        # The boxes land before the captures.
        for _ in range(12):
            for twin in (*graphs, *eagers):
                twin.pair(collide=True)
        for twin in eagers:
            twin.sort_every_build()
        with _capture() as capture:
            for twin in graphs:
                twin.pair(collide=True)
        for replay in range(4):
            wp.capture_launch(capture.graph)
            for twin in eagers:
                twin.pair(collide=True)
            for name, graph, eager in zip(("jaws", "boxes"), graphs, eagers, strict=True):
                _assert_same_twins(graph, eager, f"replay {replay} {name}")
        self.assertGreater(int(graphs[1].streams[0].rigid_contact_count.numpy()[0]), 0, "the boxes must touch")

    @unittest.skipUnless(wp.is_cuda_available(), "graph capture needs CUDA")
    def test_solver_graphs_replay_empty_contact_frames(self):
        """Replay graphs bitwise as eager always-sort steps over empty, live and again empty contact frames.

        Boxes fall and rest on the ground; a Contacts.clear() empties the frame of a
        replay without a collision pass while their anchors are stored; they collide
        again, are thrown up and land again. A second eager twin is an A/A control:
        the graph must match bitwise for as long as the eager twins do, which must
        cover every transition. With frame reuse the boxes' 128 contact slots take
        the live sort, which skips empty frames.
        """
        graph = _graph_twin(boxes=True, radix=False)
        self.assertTrue(graph.patches.live_sort)
        eager, control = (_SolverTwin("cuda:0", boxes=True) for _ in range(2))
        twins = (graph, eager, control)
        for twin in twins:
            twin.pair(collide=True)
        for twin in (eager, control):
            twin.sort_every_build()
        captured = {}
        outcomes = _admitted_modes(graph.patches)
        for collide in (True, False):
            with _capture() as capture:
                graph.pair(collide=collide)
            captured[collide] = capture.graph
        self.assertEqual(outcomes, ["sort", "keep"] * 2)
        live, cleared = [], None
        for replay in range(120):
            collide = replay != 40
            if not collide:
                for twin in twins:
                    twin.streams[0].clear()
                cleared = int(np.count_nonzero(graph.patches.previous.valid.numpy()))
            if replay == 60:
                for twin in twins:
                    twin.throw(1.2)
            wp.capture_launch(captured[collide])
            for twin in (eager, control):
                twin.pair(collide=collide)
            if not eager.same_steps(control):
                break
            _assert_same_twins(graph, eager, f"replay {replay}")
            live.append(int(graph.streams[0].rigid_contact_count.numpy()[0]))
        # Covered: empty frames before landing, live frames, an empty frame while
        # anchors are stored, empty frames in the air after contact, and landing again.
        self.assertGreater(len(live), 100, f"the eager twins must agree long enough: {len(live)} replays")
        self.assertEqual(live[0], 0, f"the boxes must start in the air: {live}")
        self.assertGreater(cleared, 0, "anchors must be stored when the frame is cleared")
        self.assertEqual(live[40], 0, "the cleared frame must hold no contacts")
        flights = [i for i in range(61, len(live)) if live[i] == 0 and live[i - 1] > 0]
        self.assertTrue(flights, f"the thrown boxes must lose contact: {live}")
        self.assertGreater(max(live[flights[0] :]), 0, f"the boxes must land again: {live}")


def _poison_sort_scratch(twin):
    """Fill the second half of the radix sort's keys, which only the sort writes."""
    keys = twin.patches.current.keys.numpy()
    keys[twin.patches.capacity :] = -12345
    twin.patches.current.keys.assign(keys)


def _sort_scratch_intact(twin):
    """Whether no radix sort ran since :func:`_poison_sort_scratch`."""
    return bool(np.all(twin.patches.current.keys.numpy()[twin.patches.capacity :] == -12345))


@wp.kernel(enable_backward=False)
def _fill_flags(flags: wp.array[wp.int32], value: int):
    flags[wp.tid()] = value


@wp.kernel(enable_backward=False)
def _poison_scratch(keys: wp.array[wp.int64], capacity: int, poison: wp.int64):
    """Fill the second half of the pair keys, which only the radix sort writes."""
    keys[capacity + wp.tid()] = poison


@wp.kernel(enable_backward=False)
def _check_scratch(keys: wp.array[wp.int64], capacity: int, poison: wp.int64, intact: wp.array[wp.int32], slot: int):
    """Clear ``intact[slot]`` if a radix sort overwrote the poisoned second half of the pair keys."""
    if keys[capacity + wp.tid()] != poison:
        intact[slot] = 0


class _SortProbe:
    """Record on the device whether the radix sort ran between the builds a graph captures.

    :meth:`mark` poisons the second half of the pair keys, which only the radix
    sort writes, and :meth:`check` records in ``intact[slot]`` whether it still
    holds the poison. Both launch on the capturing stream, so each replay
    evaluates them between the same builds; :meth:`start` resets ``intact``.
    """

    def __init__(self, patches, slots):
        self.keys, self.capacity, self.device = patches.current.keys, patches.capacity, patches.device
        self.intact = wp.zeros(slots, dtype=wp.int32, device=self.device)
        self.poison = wp.int64(-12345)

    def start(self):
        wp.launch(_fill_flags, dim=self.intact.shape[0], inputs=[self.intact, 1], device=self.device)

    def mark(self):
        wp.launch(
            _poison_scratch, dim=self.capacity, inputs=[self.keys, self.capacity, self.poison], device=self.device
        )

    def check(self, slot):
        wp.launch(
            _check_scratch,
            dim=self.capacity,
            inputs=[self.keys, self.capacity, self.poison, self.intact, slot],
            device=self.device,
        )


def _counted_sort_calls(patches):
    """Return the list each pair sort that ``patches`` launches or captures appends to, reading no device array."""
    calls = []
    order = patches._order_pairs

    def counted(**kwargs):
        calls.append(1)
        order(**kwargs)

    patches._order_pairs = counted
    return calls


def _assert_same_twins(graph, eager, context):
    """Require bitwise-equal bodies, joints and stored patch history of two solver twins."""
    for name in ("body_q", "body_qd", "joint_q", "joint_qd"):
        np.testing.assert_array_equal(
            getattr(graph.states[0], name).numpy(), getattr(eager.states[0], name).numpy(), f"{context} {name}"
        )
    for name in ("valid", "displacement"):
        np.testing.assert_array_equal(
            getattr(graph.patches.previous, name).numpy(),
            getattr(eager.patches.previous, name).numpy(),
            f"{context} previous.{name}",
        )


_TEST_CAPTURE_SERIALS = itertools.count(1)


@contextlib.contextmanager
def _cpu_capture_scopes():
    """Give CPU graph captures a capture scope for the duration, as Warp's registry gives CUDA captures.

    Warp records a CPU capture into a byte stream that :func:`warp.capture_launch`
    replays. Newton counts it as eager; these tests name it the way
    ``capture_scope`` names a CUDA capture. They capture no conditional bodies.
    """
    from warp._src import context  # noqa: PLC0415

    real = contacts_module.capture_scope

    def scope(device):
        graph = context.runtime._apic_graph
        if device.is_cpu and graph is not None:
            serial = getattr(graph, "_test_capture_serial", None)
            if serial is None:
                serial = graph._test_capture_serial = next(_TEST_CAPTURE_SERIALS)
            return serial, None
        return real(device)

    with (
        mock.patch.object(friction_patches, "capture_scope", scope),
        mock.patch.object(contacts_module, "capture_scope", scope),
    ):
        yield


@contextlib.contextmanager
def _capture(device="cuda:0"):
    """Capture a graph on ``device``; CPU captures get a capture scope (:func:`_cpu_capture_scopes`)."""
    with contextlib.ExitStack() as stack:
        if not wp.get_device(device).is_cuda:
            stack.enter_context(_cpu_capture_scopes())
        yield stack.enter_context(wp.ScopedCapture(device=device))


def _graph_twin(*, radix=True, **options):
    """Return a CUDA twin with frame reuse, to capture graphs of.

    ``radix`` keeps the radix sort, whose poisoned scratch half shows whether a
    replay sorted (:func:`_poison_sort_scratch`).
    """
    with _flags(reuse=None):
        twin = _SolverTwin("cuda:0", **options)
    if not twin.patches.reuse_frames:
        raise AssertionError("the twin must reuse frames")
    if radix:
        twin.patches.live_sort = False
    return twin


@wp.kernel(enable_backward=False)
def _reverse_live_contacts(
    count: wp.array[wp.int32],
    shape0: wp.array[wp.int32],
    shape1: wp.array[wp.int32],
    point_id: wp.array[wp.int32],
    tids: wp.array[wp.int32],
    point0: wp.array[wp.vec3],
    point1: wp.array[wp.vec3],
    offset0: wp.array[wp.vec3],
    offset1: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
    margin0: wp.array[float],
    margin1: wp.array[float],
):
    """Reverse the order of the live contacts, so that slot order no longer follows pair order."""
    i = wp.tid()
    j = wp.min(count[0], shape0.shape[0]) - 1 - i
    if i >= j:
        return
    s0 = shape0[i]
    shape0[i] = shape0[j]
    shape0[j] = s0
    s1 = shape1[i]
    shape1[i] = shape1[j]
    shape1[j] = s1
    p = point_id[i]
    point_id[i] = point_id[j]
    point_id[j] = p
    t = tids[i]
    tids[i] = tids[j]
    tids[j] = t
    a0 = point0[i]
    point0[i] = point0[j]
    point0[j] = a0
    a1 = point1[i]
    point1[i] = point1[j]
    point1[j] = a1
    o0 = offset0[i]
    offset0[i] = offset0[j]
    offset0[j] = o0
    o1 = offset1[i]
    offset1[i] = offset1[j]
    offset1[j] = o1
    n = normal[i]
    normal[i] = normal[j]
    normal[j] = n
    m0 = margin0[i]
    margin0[i] = margin0[j]
    margin0[j] = m0
    m1 = margin1[i]
    margin1[i] = margin1[j]
    margin1[j] = m1


def _reverse_contacts(contacts):
    """Reverse the order of a contact stream's live contacts in place."""
    wp.launch(
        _reverse_live_contacts,
        dim=max(1, contacts.rigid_contact_max // 2),
        inputs=[
            contacts.rigid_contact_count,
            contacts.rigid_contact_shape0,
            contacts.rigid_contact_shape1,
            contacts.rigid_contact_point_id,
            contacts.rigid_contact_tids,
            contacts.rigid_contact_point0,
            contacts.rigid_contact_point1,
            contacts.rigid_contact_offset0,
            contacts.rigid_contact_offset1,
            contacts.rigid_contact_normal,
            contacts.rigid_contact_margin0,
            contacts.rigid_contact_margin1,
        ],
        device=contacts.device,
    )


def _step_passes(twin, passes, probe=None):
    """Step ``twin`` through collision passes, each a substep count, colliding on each pass's first substep.

    Each pass reverses its live contacts after colliding, so that the slots hold
    pairs out of order and a kept build that lost its sorted pairs shows. A
    :class:`_SortProbe` records whether pass ``p`` sorted in its first substep
    in ``intact[2 * p]`` and in its later substeps in ``intact[2 * p + 1]``.
    """
    if probe is not None:
        probe.start()
    for index, substeps in enumerate(passes):
        if probe is not None:
            probe.mark()
        for substep in range(substeps):
            if substep == 0:
                twin.pipeline.collide(twin.states[0], twin.streams[0])
                _reverse_contacts(twin.streams[0])
            twin.substep()
            if probe is not None and substep == 0:
                probe.check(2 * index)
                probe.mark()
        if probe is not None:
            probe.check(2 * index + 1)


# Collision passes captured in one graph, each a substep count: one pass of two
# substeps, as most tasks capture per launch; four such passes, as tasks whose
# whole step is graphable capture per launch; and two passes of one substep.
_COLLISION_PASSES = {
    "collide, step, step": (2,),
    "four passes of two steps": (2, 2, 2, 2),
    "two passes of one step": (1, 1),
}


def run_captured_collision_passes(test, device, replays):
    """Replay graphs of collision passes bitwise as the eager always-sort chain, sorting once per pass.

    The graph twin takes the radix sort, which also records into CPU captures: a probe captured between the builds
    shows on every replay that each pass's first substep sorts and its later
    substeps do not (:class:`_SortProbe`), as the sorts the capture records do.
    """
    cuda = wp.get_device(device).is_cuda
    # The squeezing jaws need CUDA; on the CPU, boxes landed on the ground keep their contacts.
    scene = {"jaws": True} if cuda else {"boxes": True}
    for name, passes in _COLLISION_PASSES.items():
        with test.subTest(device=device, passes=name):
            graph = _SolverTwin(device, **scene)
            eager = _SolverTwin(device, **scene)
            test.assertTrue(graph.patches.reuse_frames)
            for _ in range(2 if cuda else 12):
                for twin in (graph, eager):
                    twin.pair(collide=True)
            eager.sort_every_build()
            graph.patches.live_sort = False
            probe = _SortProbe(graph.patches, 2 * len(passes))
            outcomes, sorts = _admitted_modes(graph.patches), _counted_sort_calls(graph.patches)
            with _capture(device) as capture:
                _step_passes(graph, passes, probe)
            test.assertEqual(outcomes, [mode for count in passes for mode in ("sort", *("keep",) * (count - 1))])
            test.assertEqual(len(sorts), len(passes), "the graph records one sort per collision pass")
            for replay in range(replays):
                wp.capture_launch(capture.graph)
                _step_passes(eager, passes)
                np.testing.assert_array_equal(
                    probe.intact.numpy(), [0, 1] * len(passes), f"{name} replay {replay}: sorts by pass and substep"
                )
                _assert_same_twins(graph, eager, f"{name} replay {replay}")
            test.assertGreater(int(graph.streams[0].rigid_contact_count.numpy()[0]), 0, "the bodies must touch")
            test.assertTrue(np.any(graph.patches.previous.valid.numpy()), "the passes must carry anchors")
            test.assertFalse(graph.solver.constraint_capacity_status()["patch_frames"])


def run_captured_steps_without_a_collision_pass(test, device, replays):
    """Sort the first build of a graph without a collision pass and keep the second, whatever ran in between.

    Collision passes, a model change and a Contacts.clear() run eagerly between
    replays, as they may around a graph of substeps.
    """
    graph = _SolverTwin(device, boxes=True)
    eager = _SolverTwin(device, boxes=True)
    for _ in range(12):
        for twin in (graph, eager):
            twin.pair(collide=True)
    eager.sort_every_build()
    outcomes = _admitted_modes(graph.patches)
    with _capture(device) as capture:
        graph.pair()
    test.assertEqual(outcomes, ["sort", "keep"])
    for replay in range(replays):
        for twin in (graph, eager):
            if replay % 3 == 0:
                twin.pipeline.collide(twin.states[0], twin.streams[0])
            elif replay % 3 == 1:
                twin.solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
            elif replay == 5:
                twin.streams[0].clear()
        wp.capture_launch(capture.graph)
        eager.pair()
        _assert_same_twins(graph, eager, f"replay {replay}")
    test.assertFalse(graph.solver.constraint_capacity_status()["patch_frames"])


@wp.kernel(enable_backward=False)
def _lower_contact_count(count: wp.array[wp.int32]):
    count[0] = count[0] - 1


@wp.kernel(enable_backward=False)
def _advance_contact_generation(generation: wp.array[wp.int32]):
    generation[0] = generation[0] + 1


def run_captured_contact_writes(test, device):
    """Solve a kept substep that a caller's kernel rewrote without patch rows and report it; a cleared one sorts.

    The caller's kernel runs between the two captured substeps of a collision
    pass: lowering the count or advancing the generation without a noted write
    breaks the capture contract, while Contacts.clear() followed by restoring the
    count is a noted write, after which the substep sorts and steps bitwise as
    the always-sort chain.
    """
    for write in ("count", "generation", "clear"):
        with test.subTest(device=device, write=write):
            graph = _SolverTwin(device, boxes=True)
            eager = _SolverTwin(device, boxes=True)
            for _ in range(12):
                for twin in (graph, eager):
                    twin.pair(collide=True)
            eager.sort_every_build()
            test.assertTrue(np.any(graph.patches.previous.valid.numpy()), "the landed boxes must store anchors")
            saved = {twin: wp.zeros(1, dtype=wp.int32, device=device) for twin in (graph, eager)}

            def rewrite(twin, write=write, saved=saved):
                contacts = twin.streams[0]
                if write == "count":
                    wp.launch(_lower_contact_count, dim=1, inputs=[contacts.rigid_contact_count], device=device)
                elif write == "generation":
                    wp.launch(_advance_contact_generation, dim=1, inputs=[contacts.contact_generation], device=device)
                else:
                    wp.copy(saved[twin], contacts.rigid_contact_count)
                    contacts.clear()
                    wp.copy(contacts.rigid_contact_count, saved[twin])

            def collision_pass(twin, rewrite=rewrite):
                twin.substep(collide=True)
                rewrite(twin)
                twin.substep()

            outcomes = _admitted_modes(graph.patches)
            with _capture(device) as capture:
                collision_pass(graph)
            test.assertEqual(outcomes, ["sort", "sort" if write == "clear" else "keep"])
            wp.capture_launch(capture.graph)
            collision_pass(eager)
            test.assertTrue(np.all(np.isfinite(graph.states[0].body_qd.numpy())))
            if write == "clear":
                _assert_same_twins(graph, eager, write)
                test.assertFalse(graph.solver.constraint_capacity_status()["patch_frames"])
                continue
            test.assertFalse(np.any(graph.patches.previous.valid.numpy()), "the empty frame releases every anchor")
            test.assertTrue(np.any(eager.patches.previous.valid.numpy()), "the sorted frame keeps its anchors")
            with test.assertRaisesRegex(RuntimeError, "without patch friction"):
                graph.solver.check_constraint_capacity()
            test.assertTrue(graph.solver.constraint_capacity_status(clear=True)["patch_frames"])
            graph.solver.check_constraint_capacity()


def run_captured_entry_filter(test, device, replays):
    """Keep patch pairs across substeps of a coupled entry's filtered contacts, sorting after each filter pass.

    A graph holds two collision passes. Each collides, filters the contacts for
    a coupled entry and builds the patch frame of the filtered buffer twice, as
    an entry's substeps do. The collision pass notes a write of its own buffer
    only, so the filter's note alone makes the second pass sort. Replays match
    an always-sort patch state bitwise.
    """
    from newton.solvers import SolverSemiImplicit  # noqa: PLC0415
    from newton.solvers.experimental.coupled import SolverCoupled  # noqa: PLC0415

    model = _falling_boxes(device)
    coupled = SolverCoupled(
        model=model,
        entries=[
            SolverCoupled.Entry(
                name="boxes",
                solver=SolverSemiImplicit,
                bodies=list(range(model.body_count)),
                shapes=list(range(model.shape_count)),
            )
        ],
    )
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=128, broad_phase="nxn", deterministic=True)
    contacts = pipeline.contacts()
    state = model.state()
    poses = state.body_q.numpy()
    poses[:, 2] = 0.019
    state.body_q.assign(poses)
    phi = [wp.zeros(128, dtype=wp.vec2, device=device) for _ in range(2)]
    plain = _FrictionPatchState(model, 128, phi[0], reuse_frames=True)
    patches = _FrictionPatchState(model, 128, phi[1], reuse_frames=True)

    def entry_passes(states, passes=2):
        """Collide, filter and build twice per pass; the poses move up in a pass and back in the next."""
        for index in range(passes):
            pipeline.collide(state, contacts)
            filtered = coupled.entry_contacts("boxes", contacts)
            for substep in range(2):
                for patch_state in states:
                    patch_state.build(model, state, filtered)
                    store_frame(patch_state, state)
                if substep == 0:
                    step = 1.0e-4 if index % 2 == 0 else -1.0e-4
                    wp.launch(_nudge_poses, dim=model.body_count, inputs=[state.body_q, step], device=device)
        return filtered

    filtered = entry_passes((plain, patches), passes=1)
    _sort_every_build(plain)
    test.assertIsNot(filtered, contacts)
    test.assertGreater(int(filtered.rigid_contact_count.numpy()[0]), 0, "the boxes must touch the ground")
    outcomes = _admitted_modes(patches)
    with _capture(device) as capture:
        entry_passes((patches,))
    test.assertEqual(outcomes, ["sort", "keep", "sort", "keep"])
    for replay in range(replays):
        before = wp.clone(state.body_q)
        wp.capture_launch(capture.graph)
        state.body_q.assign(before)
        entry_passes((plain,))
        assert_same_state(patches, plain, f"replay {replay}", records=False)
        test.assertGreater(int(filtered.rigid_contact_count.numpy()[0]), 0, f"replay {replay}: the boxes must touch")
    test.assertEqual(int(patches.frame_status.numpy()[0]), 0)


class TestPatchCapturedFrameReuse(unittest.TestCase):
    """Graphs captured as Isaac Lab captures them, replayed on the CPU and on CUDA."""

    def test_collision_pass_graphs_step_as_the_always_sort_chain(self):
        """Replay [collide, step, ...] graphs bitwise as eager always-sort steps, sorting once per collision pass."""
        run_captured_collision_passes(self, "cpu", replays=4)

    def test_graphs_without_a_collision_pass_sort_their_first_build(self):
        """Sort the first build of a substep graph each replay and keep the second, bitwise as eager steps."""
        run_captured_steps_without_a_collision_pass(self, "cpu", replays=6)

    def test_unnoted_contact_writes_between_captured_steps_are_reported(self):
        """Report a captured kept substep rewritten by a caller's kernel; Contacts.clear() makes it sort instead."""
        run_captured_contact_writes(self, "cpu")

    def test_coupled_entry_filters_start_new_frames(self):
        """Sort after each coupled entry filter pass and keep the pairs on the entry's later substeps."""
        run_captured_entry_filter(self, "cpu", replays=4)

    @unittest.skipUnless(wp.is_cuda_available(), "graph capture needs CUDA")
    def test_collision_pass_graphs_step_as_the_always_sort_chain_on_the_device(self):
        """Replay collision-pass graphs on CUDA bitwise as eager always-sort steps, with one sort per pass."""
        run_captured_collision_passes(self, "cuda:0", replays=100)

    @unittest.skipUnless(wp.is_cuda_available(), "graph capture needs CUDA")
    def test_graphs_without_a_collision_pass_sort_their_first_build_on_the_device(self):
        """Sort the first build of a CUDA substep graph each replay and keep the second, bitwise as eager steps."""
        run_captured_steps_without_a_collision_pass(self, "cuda:0", replays=100)

    @unittest.skipUnless(wp.is_cuda_available(), "graph capture needs CUDA")
    def test_unnoted_contact_writes_between_captured_steps_are_reported_on_the_device(self):
        """Report a CUDA-captured kept substep rewritten by a caller's kernel; a cleared one sorts."""
        run_captured_contact_writes(self, "cuda:0")

    @unittest.skipUnless(wp.is_cuda_available(), "graph capture needs CUDA")
    def test_coupled_entry_filters_start_new_frames_on_the_device(self):
        """Sort after each coupled entry filter pass in a CUDA graph and keep the pairs on later substeps."""
        run_captured_entry_filter(self, "cuda:0", replays=100)


if __name__ == "__main__":
    unittest.main()
