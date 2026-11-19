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
``error -= n * wp.dot(error, n)`` in ``_settle_region`` rounds either ``n.x * e.x``
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

Every build prepares through ``_prepare``, which keeps the pairs of the frame
the slots hold sorted, eager or captured. Frame reuse tests compare with an
always-sort chain that forgets that frame before each build
(:func:`_sort_every_build`) and so launches the same kernels, which tests frame
reuse bitwise on any device.
"""

import contextlib
import gc
import inspect
import itertools
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.geometry.contact_sort import body_pair_codes
from newton._src.solvers.feather_pgs import friction_patches
from newton._src.solvers.feather_pgs.friction_patches import (
    _BUILD_FIELDS,
    _STORED_FIELDS,
    FrictionPatches,
    _FrictionPatchState,
    _PatchFrame,
    link_patch_routes,
    patch_routes,
)
from newton._src.solvers.feather_pgs.solver_feather_pgs import SolverFeatherPGS
from newton.tests.feather_pgs_law_harness import row_lines
from newton.tests.feather_pgs_patch_reference import SENTINEL, ReferencePatches, frame_order, pair_codes, store_frame
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
            # Each slot's identity in its pass, which orders a pair's contacts in the frame.
            rigid_contact_key=wp.array(np.arange(contacts), dtype=wp.int64, device=device),
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
        generation advances, as in a collision pass.
        """
        self.generation += 1
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
        _pipeline_order(self.model, self.contacts, count)
        return count


def _pipeline_order(model, contacts, live):
    """Put the ``live`` contacts in the collision pipeline's order, by body pair then key (``rigid_contact_key``, each
    written row's index here), as every pass leaves them (:meth:`.ContactSorter.sort_full`)."""
    order = frame_order(model, contacts)[:live]
    for name in _CONTACT_FIELDS:
        if name == "rigid_contact_key":
            continue
        array = getattr(contacts, name)
        values = array.numpy()
        values[:live] = values[order]
        array.assign(values)


def _pair(scene, capacity):
    """Return the patch state under test and its reference, both built from ``scene``."""
    phi = [wp.zeros(capacity, dtype=wp.vec2, device=scene.device) for _ in range(2)]
    return _FrictionPatchState(scene.model, capacity, phi[0]), ReferencePatches(scene.model, capacity, phi[1])


def _set_threads(patches, threads):
    """Stride ``patches``' light launches and builds with ``threads``."""
    patches.threads, patches.build_threads = threads


def _twin(patches, scene):
    """Return a second patch state on another grid, which CUDA tests step alongside ``patches``, or None on the CPU.

    The twin runs the module under test on the same inputs, with
    ``_TWIN_THREADS``, so it must match ``patches`` bitwise
    (:func:`assert_same_state`): a race or any other grid-dependent result makes
    them differ, however small. The CPU compares with the reference bitwise.
    """
    if not patches.device.is_cuda:
        return None
    twin = _FrictionPatchState(scene.model, patches.capacity, wp.zeros_like(patches.view.phi))
    _set_threads(twin, _TWIN_THREADS)
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


# Scratch of the stored frame that a build writes for itself and no later kernel reads: a build that makes a frame's
# regions leaves each pair's index of the stored history there, and a build that carries the regions only each
# region's own.
_BUILD_SCRATCH = ("members", "centres", "used")
# Scratch of a new frame's placement (friction_patches._place): each region's pair history, claim and proposal, which a
# build that carries the regions neither writes nor reads.
_PLACEMENT_SCRATCH = ("history", "claim", "proposal")


def assert_same_state(twin, patches, context="", records=True):
    """Require bitwise-equal arrays in two patch states, the frames' state and what the builds place, and the same
    listed regions. The builds' own scratch (:data:`_BUILD_SCRATCH`, :data:`_PLACEMENT_SCRATCH`) differs between a
    build that makes a frame's regions and one that carries them, so it is not compared.

    ``records`` compares the frame records as well: the sorted frame, and the
    sort's extent, which a kept build leaves empty.
    """
    single, others = (int(count) for count in patches.regions.numpy())
    live = int(patches.extent.numpy()[1])
    np.testing.assert_array_equal(twin.regions.numpy(), patches.regions.numpy(), f"{context} regions")
    for part, listed in (("of one contact", slice(0, single)), ("of several", slice(live - others, live))):
        np.testing.assert_array_equal(
            np.sort(twin.current.regions.numpy()[listed]),
            np.sort(patches.current.regions.numpy()[listed]),
            f"{context} current.regions {part}",
        )
    for frame in ("current", "previous"):
        for name in _PatchFrame.vars:
            if frame == "previous" and name in _BUILD_FIELDS:
                continue  # the stored frame holds no field only a build reads
            if frame == "current" and name in _STORED_FIELDS:
                continue  # and the built frame none of the stored frame's scratch
            if name == "regions":
                continue  # the listed regions, compared above: threads list them in any order
            if frame == "previous" and name in _BUILD_SCRATCH:
                continue
            if frame == "current" and name in _PLACEMENT_SCRATCH:
                continue
            np.testing.assert_array_equal(
                getattr(getattr(twin, frame), name).numpy(),
                getattr(getattr(patches, frame), name).numpy(),
                f"{context} {frame}.{name}",
            )
    for name in ("weight", "next_contact", "point", "phi"):
        np.testing.assert_array_equal(
            getattr(twin.view, name).numpy(), getattr(patches.view, name).numpy(), f"{context} view.{name}"
        )
    names = ["previous_world", "previous_q", "stored", "built"]
    if records:
        names += ["extent", "built_frame"]
    for name in names:
        np.testing.assert_array_equal(getattr(twin, name).numpy(), getattr(patches, name).numpy(), f"{context} {name}")
    visits = friction_patches._BUILD_EXTENT
    np.testing.assert_array_equal(twin.extent.numpy()[:visits], patches.extent.numpy()[:visits], f"{context} extent")


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
    device, seed, *, worlds=3, bodies=4, contacts=96, capacity=128, threads=None, plan=_PLAN
):
    """Step a patch state and the reference through ``plan``, comparing after every launch group.

    On CUDA a twin steps alongside (:func:`_twin`). Returns counts of carried anchors, of
    anchors released by finishing and of stored anchors past an unstored
    build's live count, and the largest live count, so callers can require the
    sequence to exercise them.
    """
    rng = np.random.default_rng(1000 + seed)
    scene = _Scene(device, seed, worlds=worlds, bodies=bodies, contacts=contacts)
    patches, reference = _pair(scene, capacity)
    if threads is not None:
        _set_threads(patches, threads)
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
        """Keep a pair whose key has every bit of the stride's square below it ahead of the empty slots."""
        devices = ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])
        for device in devices:
            with self.subTest(device=device):
                scene = _Scene(device, 5, worlds=1, bodies=4, contacts=16)
                stride = scene.body_count + len(scene.shape_body) + 2
                self.assertEqual(stride & (stride - 1), 0, "the fixture needs a power-of-two key stride")
                patches, reference = _pair(scene, 24)
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
        for seed, contacts, capacity, threads in (
            (0, 96, 96, None),
            (1, 96, 128, (7, 5)),
            (2, 64, 160, (1, 3)),
            (3, 128, 128, (32, 64)),
        ):
            with self.subTest(seed=seed, capacity=capacity, threads=threads):
                coverage = run_reference_differential(
                    "cpu", seed, contacts=contacts, capacity=capacity, threads=threads
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
        run_graph_replay("cuda:0")


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


def run_graph_replay(device, *, worlds=32, contacts=1536, capacity=2048):
    """Capture one patch step, replay it over changing counts, and compare each replay with the reference.

    On CUDA a twin steps eagerly alongside the replays (:func:`_twin`).
    """
    rng = np.random.default_rng(7)
    scene = _Scene(device, 21, worlds=worlds, contacts=contacts)
    patches, reference = _pair(scene, capacity)
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
        """Finish every route that holds patch rows, warm-started or not: no step seeds friction. The row writers form
        each normal row's ring parent with its kind, so no step links apart (only the free bodies' regions do, whose
        rows their fill writes after the listing), and the link's rule leaves the rows' parents as the step did."""
        # Both modes route the fixed-base arm's contacts and the free box's through the line's rows (a free body is a
        # one-node tree; only colored worlds keep a free bodies' run); no route builds the other families.
        configs = [("cpu", {}, (1, 0, 0))]
        if wp.is_cuda_available():
            configs.append(("cuda:0", {}, (1, 0, 0)))
        dts = (1.0 / 240.0, 1.0 / 240.0, 1.0 / 120.0, 1.0 / 480.0)
        for (device, mode, enabled), warmstart in itertools.product(configs, (False, True)):
            with self.subTest(device=device, warmstart=warmstart):
                model = _arm_over_box(wp.get_device(device), worlds=2)
                # The routes write their rows on the complete path: the box's speculative contacts separate at first.
                solver = newton.solvers.SolverFeatherPGS(
                    model,
                    pgs_iterations=4,
                    pgs_warmstart=warmstart,
                    pgs_warmstart_decay=0.8,
                    pins=newton.solvers.SolverFeatherPGS.Pins(separated_worlds=False),
                    **mode,
                )
                self.assertGreater(solver.laws.friction_anchor_beta, 0.0)
                arrays = (
                    (solver.placement.row_parent, solver.placement.row_mu, solver.placement.impulses),
                    (solver.placement.row_parent, solver.placement.row_mu, solver.placement.impulses),
                )
                pipeline = newton.CollisionPipeline(model, contact_matching="latest", growable_contacts=True)
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
                        self.assertEqual([name for name, _ in calls], ["finish"], f"step {step}")
                        for name, routes in calls:
                            self.assertEqual(tuple(routes.enabled), enabled, f"step {step} {name}")
                            for route in np.flatnonzero(enabled):
                                for field, array in zip(("parents", "mu", "impulses"), arrays[route], strict=True):
                                    self.assertEqual(getattr(routes, f"{field}{route}").ptr, array.ptr, field)
                        parents = solver.placement.row_parent.numpy().copy()
                        solver.laws.friction_patches.link(
                            solver.placement.contact_world,
                            solver.placement.contact_slot,
                            solver.placement.contact_path,
                            solver.placement.contact_slots_needed,
                            solver.sweep.patch_routes(),
                        )
                        np.testing.assert_array_equal(
                            solver.placement.row_parent.numpy(), parents, f"step {step}: the writers' ring parents"
                        )
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


class TestPatchManyContacts(unittest.TestCase):
    def test_builds_match_the_full_capacity_reference(self):
        """Build, link, seed, finish and store bitwise as the reference, also past a thousand live contacts."""
        for seed, threads in ((4, None), (5, (7, 5))):
            with self.subTest(seed=seed, threads=threads):
                run_reference_differential("cpu", seed, capacity=160, threads=threads)
        plan = ("grow", "grow", "grow", "live", "sparse", "grow", "overflow", "live", "rebuild", "masked", "live")
        coverage = run_reference_differential(
            "cpu", 6, worlds=110, contacts=4096, capacity=4600, threads=(97, 61), plan=plan
        )
        self.assertGreater(coverage["live"], 2048)

    @unittest.skipUnless(wp.is_cuda_available(), "multi-block grids need CUDA")
    def test_builds_match_on_many_blocks(self):
        """Match the reference on multi-block grids, and through graph replay."""
        run_reference_differential("cuda:0", 12, worlds=160, contacts=8192, capacity=8192)
        run_graph_replay("cuda:0", worlds=160, contacts=8192, capacity=8192)


def _shape(capacity, sms, *, light, build, block):
    """Return the launch attributes of a patch state with ``capacity`` slots on ``sms`` SMs.

    ``light`` and ``build`` count 256-thread blocks per SM of the light launches and ``_build``, and ``block`` is
    ``_build``'s block size.
    """
    return {
        "threads": max(1, min(capacity, light * sms * 256)),
        "build_threads": max(1, min(capacity, build * sms * 256)),
        "build_block_dim": block,
    }


# Launch shapes before and since one-warp _build blocks on twice the grid and
# light launches on three times theirs.
_FORMER_SHAPE = {"light": 2, "build": 4, "block": 256}
_SHAPE = {"light": 6, "build": 8, "block": 32}
# An odd shape, far below the live counts, on any device.
_ODD_SHAPE = {"threads": 7, "build_threads": 5, "build_block_dim": 32}

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
            rigid_contact_key=wp.array(np.arange(self.capacity), dtype=wp.int64, device=device),
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
        _pipeline_order(self.model, self.contacts, live)
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


def run_launch_shapes(device, seed, shapes, *, worlds=3, contacts=96, capacity=128, copies=1, stages=None):
    """Step one patch state per launch shape on identical inputs; every array must match the first state's bitwise.

    ``shapes`` holds attribute overrides (:func:`_shape`), ``None`` for the
    constructor's own shape. Each frame is built, linked, seeded and finished
    over its substeps, whose later builds keep its pairs. After each launch group every array of every state, scratch
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
        state = _FrictionPatchState(scene.model, capacity, phi)
        for name, value in (shape or {}).items():
            setattr(state, name, value)
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
        launched = set()
        original = wp.launch

        def record(kernel, **kwargs):
            # Each launch strides over the slots by its own thread count.
            arguments = [*kwargs.get("inputs", ()), *kwargs.get("outputs", ())]
            stride = dict(zip((arg.label for arg in kernel.adj.args), arguments, strict=True))["threads"]
            launched.add((kernel.key, kwargs["dim"], stride, kwargs.get("block_dim")))
            return original(kernel, **kwargs)

        self.assertGreater(scene.frame(), 2048, "many live contacts")
        with mock.patch.object(wp, "launch", record):
            state.build(scene.model, scene.state, scene.contacts)
        self.assertEqual(
            launched,
            {
                ("_prepare", 1536, 1536, None),
                ("_prepare_contacts", 1536, 1536, None),
                ("_build", 2048, 2048, 32),
                ("_place", 2048, 2048, 32),
            },
        )

    def test_launch_shapes_match_bitwise(self):
        """Build, link, seed, finish, store and reset bitwise on every grid.

        CPU launches run one thread at a time, so they check how each grid
        strides over the slots.
        """
        shapes = (None, _shape(4600, 1, **_FORMER_SHAPE), _ODD_SHAPE)
        coverage = run_launch_shapes("cpu", 2, shapes, worlds=110, contacts=4096, capacity=4600)
        self.assertGreater(coverage["live"], 2048, "many live contacts")
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
            coverage = run_launch_shapes(device, 3, shapes, worlds=110, contacts=4096, capacity=capacity)
            self.assertGreater(coverage["live"], 2048, "many live contacts")
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
        finish, store and reset.
        """
        device = wp.get_device("cuda:0")
        widest = _SHAPE["build"] * device.sm_count * 256
        worlds, contacts, capacity = 64, 2048, 2304
        copies = -(-5 * widest // (4 * contacts))
        shapes = (None, _shape(capacity * copies, device.sm_count, **_FORMER_SHAPE))
        with _patch_variants_compiled_here(device):
            coverage = run_launch_shapes(
                device,
                4,
                shapes,
                worlds=worlds,
                contacts=contacts,
                capacity=capacity,
                copies=copies,
                stages=("finish", "store", "reset"),
            )
        self.assertGreater(coverage["live"], widest, "the live contacts must outnumber every grid's threads")
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
    view.normal = wp.zeros(contacts, dtype=wp.vec3, device=device)
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


_CONTACT_FIELDS = (
    "rigid_contact_shape0",
    "rigid_contact_shape1",
    "rigid_contact_point0",
    "rigid_contact_point1",
    "rigid_contact_normal",
    "rigid_contact_margin0",
    "rigid_contact_margin1",
    "rigid_contact_key",
)


def _copied_stream(scene, stream=None):
    """Return another contact stream of equal generation, count and contacts, in the pipeline's order.

    ``stream`` receives the copy in place, for a graph captured over it.
    """
    if stream is None:
        names = ("contact_generation", "rigid_contact_count", *_CONTACT_FIELDS)
        stream = SimpleNamespace(
            rigid_contact_max=scene.capacity, **{name: wp.empty_like(getattr(scene.contacts, name)) for name in names}
        )
    for name in _CONTACT_FIELDS:
        getattr(stream, name).assign(getattr(scene.contacts, name).numpy())
    stream.contact_generation.assign(scene.contacts.contact_generation)
    stream.rigid_contact_count.assign(scene.contacts.rigid_contact_count)
    return stream


def _frame_identity(patches, contacts):
    """Return a contact frame as the device records it: stream token, generation and live count."""
    generation = contacts.contact_generation
    return (
        patches.stream_token(generation),
        int(generation.numpy()[0]),
        max(int(contacts.rigid_contact_count.numpy()[0]), 0),
    )


def _assert_record(test, patches, built_frame, context):
    """Require the device record to name ``built_frame``, or to hold the sentinel generation for ``None``."""
    record = patches.built_frame.numpy()
    if built_frame is None:
        test.assertEqual(int(record[0]), -1, f"{context} record")
    else:
        np.testing.assert_array_equal(record, built_frame[1:] + built_frame[:1], f"{context} record")


@wp.kernel(enable_backward=False)
def _nudge_poses(q: wp.array[wp.transform], step: float):
    """Translate each body by a small body-dependent offset, as integrating a substep would."""
    b = wp.tid()
    pose = q[b]
    offset = step * wp.vec3(float(b % 3) - 1.0, float(b % 5) - 2.0, 1.0)
    q[b] = wp.transform(wp.transform_get_translation(pose) + offset, wp.transform_get_rotation(pose))


@wp.kernel(enable_backward=False)
def _lower_contact_count(count: wp.array[wp.int32]):
    count[0] = count[0] - 1


@wp.kernel(enable_backward=False)
def _advance_contact_generation(generation: wp.array[wp.int32]):
    generation[0] = generation[0] + 1


def _decision(extent):
    """Return how the build that published ``extent`` prepared its frame: "sort" (a new frame, its keys written),
    "keep", or "empty" when no contact is live, where keeping and a new frame are one."""
    if extent[1] == 0:
        return "empty"
    return "keep" if extent[friction_patches._KEPT_EXTENT] != 0 else "sort"


def _decisions(patches):
    """Return the list each later build of ``patches`` appends its decision to (:func:`_decision`), read on the host."""
    decisions = []
    build = patches.build

    def recorded(*args, **kwargs):
        build(*args, **kwargs)
        decisions.append(_decision(patches.extent.numpy()))

    patches.build = recorded
    return decisions


@wp.kernel(enable_backward=False)
def _record_decision(extent: wp.array[int], kept: int, decisions: wp.array[wp.int32], slot: int):
    """Write how the last build prepared its frame in ``decisions[slot]``: 1 new, 0 kept, -1 no live contact."""
    decision = wp.int32(-1)
    if extent[1] > 0:
        decision = wp.int32(0)
        if extent[kept] == 0:
            decision = wp.int32(1)
    decisions[slot] = decision


class _DecisionProbe:
    """Record on the device how a run's patch builds prepared their frames, in every replay of a graph that holds it.

    :meth:`mark` launches on the current stream right after a build, so a
    capture replays it after the same build; :meth:`read` names the decisions
    as :func:`_decision` does.
    """

    def __init__(self, patches, slots):
        self.patches = patches
        self.decisions = wp.full(slots, -2, dtype=wp.int32, device=patches.device)

    def mark(self, slot):
        kept = int(friction_patches._KEPT_EXTENT)
        wp.launch(
            _record_decision,
            dim=1,
            inputs=[self.patches.extent, kept, self.decisions, slot],
            device=self.patches.device,
        )

    def read(self):
        names = {1: "sort", 0: "keep", -1: "empty", -2: "unmarked"}
        return [names[int(value)] for value in self.decisions.numpy()]


def _sort_every_build(patches):
    """Make every later build of ``patches`` sort, as the chain that kept builds must match bitwise does.

    Each build first forgets the recorded frame, a device write that a graph
    captures as well, and prepares through the kernel that keeps a frame's pairs.
    """
    build = patches.build

    def sorted_build(*args, **kwargs):
        patches.forget_frame()
        build(*args, **kwargs)

    patches.build = sorted_build
    return patches


def _recording_pair(scene, capacity):
    """Return a patch state that sorts every build and one that keeps the pairs of the frame its slots hold."""
    phi = [wp.zeros(capacity, dtype=wp.vec2, device=scene.device) for _ in range(2)]
    plain = _sort_every_build(_FrictionPatchState(scene.model, capacity, phi[0]))
    return plain, _FrictionPatchState(scene.model, capacity, phi[1])


def run_sorting_prepare_parity(test, device, *, worlds=3, contacts=96, capacity=128):
    """Prepare and store new frames bitwise as the always-sort chain, and record each frame the build sorted.

    Every frame is new, so both states sort it. Frames cover live, sparse,
    overflowing, empty and gated contacts, including static and missing shapes
    and flipped pairs.
    """
    rng = np.random.default_rng(41)
    scene = _Scene(device, 41, worlds=worlds, contacts=contacts)
    plain, recording = _recording_pair(scene, capacity)
    decisions = _decisions(recording)
    for step, mode in enumerate(("live", "live", "sparse", "overflow", "live", "empty", "gates", "live")):
        context = f"{device} step {step} {mode}"
        scene.move()
        live = scene.frame("live" if mode == "gates" else mode)
        gates = _gates(scene, rng) if mode == "gates" else {}
        for state in (plain, recording):
            state.build(scene.model, scene.state, scene.contacts, **gates)
        test.assertEqual(decisions[-1], "sort" if live else "empty", context)
        assert_same_state(recording, plain, f"{context} build", records=False)
        _assert_record(test, recording, _frame_identity(recording, scene.contacts), context)
        for state in (plain, recording):
            store_frame(state, scene.state)
        assert_same_state(recording, plain, f"{context} store", records=False)


class TestPatchFrameReuse(unittest.TestCase):
    def test_contacts_without_a_generation_sort_every_build(self):
        """Sort every build of contacts that carry no generation, which no record matches, as the always-sort chain."""
        scene = _Scene("cpu", 7)
        scene.active[:] = True
        self.assertGreater(scene.frame(), 0)
        anonymous = SimpleNamespace(**{k: v for k, v in vars(scene.contacts).items() if k != "contact_generation"})
        plain, patches = _recording_pair(scene, 128)
        decisions = _decisions(patches)
        for build in range(3):
            for state in (plain, patches):
                state.build(scene.model, scene.state, anonymous)
            assert_same_state(patches, plain, f"build {build}", records=False)
        self.assertEqual(decisions, ["sort"] * 3)
        self.assertEqual(int(patches.built_frame.numpy()[0]), -1, "the record names no stream's generation")


# Build sequences: ("build", stream) builds a stream's frame, ("write", stream)
# advances its generation as Contacts.clear() does, ("count", stream) lowers
# its count with a caller's kernel, and ("forget",) drops the record. Each
# lists the decisions of its builds.
_DECISION_PLANS = {
    "write, step, step": ((("write", "a"), ("build", "a"), ("build", "a")), ("sort", "keep")),
    "step, step": ((("build", "a"), ("build", "a")), ("sort", "keep")),
    "every later substep": ((("build", "a"), ("build", "a"), ("build", "a")), ("sort", "keep", "keep")),
    "write between steps": ((("build", "a"), ("write", "a"), ("build", "a")), ("sort", "sort")),
    "count between steps": ((("build", "a"), ("count", "a"), ("build", "a")), ("sort", "sort")),
    "another stream": (
        (("build", "a"), ("build", "b"), ("build", "a")),
        ("sort", "sort", "sort"),
    ),
    "forget": ((("build", "a"), ("forget",), ("build", "a")), ("sort", "sort")),
}


class TestHeldRegions(unittest.TestCase):
    """A contact frame's regions are made at its first build and held for the builds that share it."""

    @staticmethod
    def _states(seed):
        """A scene with every pair in contact, a state that sorts every build and one that holds its frames."""
        scene = _Scene("cpu", seed)
        scene.active[:] = True
        plain, held = _recording_pair(scene, 128)
        return scene, plain, held, _decisions(held)

    def test_held_builds_keep_the_frames_regions(self):
        """Carry a frame's regions through builds whose poses would group its contacts otherwise.

        The bodies move a centimetre between the builds of a frame, far past the plane distance that joins two
        contacts: a build that sorts groups the contacts by the poses it finds, and a held build keeps the owner
        and the ring of the frame's first build.
        """
        scene, plain, held, decisions = self._states(61)
        regrouped = 0
        for step in range(4):
            scene.move()
            live = scene.frame()
            for state in (plain, held):
                state.build(scene.model, scene.state, scene.contacts)
                store_frame(state, scene.state)
            owner = held.current.owner.numpy()[:live].copy()
            ring = held.view.next_contact.numpy()[:live].copy()
            for substep in range(1, 4):
                context = f"step {step} substep {substep}"
                scene.move(scale=50.0)
                for state in (plain, held):
                    state.build(scene.model, scene.state, scene.contacts)
                self.assertEqual(decisions[-1], "keep", context)
                np.testing.assert_array_equal(held.current.owner.numpy()[:live], owner, f"{context} owner")
                np.testing.assert_array_equal(held.view.next_contact.numpy()[:live], ring, f"{context} ring")
                regrouped += int(np.any(plain.current.owner.numpy()[:live] != owner))
                for state in (plain, held):
                    store_frame(state, scene.state)
        self.assertGreater(regrouped, 0, "the moves must regroup the contacts of a build that sorts")

    def test_a_build_without_a_store_makes_the_regions_anew(self):
        """Make a kept frame's regions anew when no store followed its last build: the stored frame is not its own."""
        scene, plain, held, decisions = self._states(62)
        scene.move()
        live = scene.frame()
        for state in (plain, held):
            state.build(scene.model, scene.state, scene.contacts)
        scene.move(scale=50.0)
        for state in (plain, held):
            state.build(scene.model, scene.state, scene.contacts)
        self.assertEqual(decisions, ["sort", "keep"])
        self.assertGreater(live, 0)
        assert_same_state(held, plain, "second build without a store", records=False)

    def test_rewritten_points_keep_the_held_regions(self):
        """Keep a held frame's regions when a writer moves contact points under one generation, until it forgets.

        The frame reads its contacts' points at every build, and makes its regions at its first: they stay the
        frame's until the writer advances the generation or forgets the frame, and the next build then groups the
        contacts as a new frame's build does.
        """
        scene, plain, held, decisions = self._states(63)
        rng = np.random.default_rng(63)
        scene.move()
        live = scene.frame()
        for state in (plain, held):
            state.build(scene.model, scene.state, scene.contacts)
            store_frame(state, scene.state)
        owner = held.current.owner.numpy()[:live].copy()
        center = held.current.center.numpy()[:live].copy()
        points = scene.contacts.rigid_contact_point0.numpy()
        points[:live] += rng.normal(scale=0.01, size=(live, 3)).astype(points.dtype)
        scene.contacts.rigid_contact_point0.assign(points)
        for state in (plain, held):
            state.build(scene.model, scene.state, scene.contacts)
        self.assertEqual(decisions[-1], "keep")
        np.testing.assert_array_equal(held.current.owner.numpy()[:live], owner, "the held frame's regions")
        np.testing.assert_array_equal(
            held.current.center.numpy()[:live], plain.current.center.numpy()[:live], "every build reads the points"
        )
        self.assertTrue(np.any(plain.current.center.numpy()[:live] != center), "the points moved")
        sorted_owner = plain.current.owner.numpy()[:live].copy()
        self.assertTrue(np.any(sorted_owner != owner), "the rewrite must regroup the contacts of a build that sorts")
        held.forget_frame()
        held.build(scene.model, scene.state, scene.contacts)
        self.assertEqual(decisions[-1], "sort")
        np.testing.assert_array_equal(held.current.owner.numpy()[:live], sorted_owner, "the forgotten frame's regions")

    def test_a_reset_drops_the_history_and_keeps_the_frames_regions(self):
        """Carry a held frame's regions through a reset, which drops the anchors they stored."""
        scene, plain, held, decisions = self._states(64)
        scene.move()
        live = scene.frame()
        for state in (plain, held):
            state.build(scene.model, scene.state, scene.contacts)
            store_frame(state, scene.state)
        self.assertTrue(np.any(held.previous.valid.numpy()), "the frame must store anchors")
        owner = held.current.owner.numpy()[:live].copy()
        held.reset(None)
        held.build(scene.model, scene.state, scene.contacts)
        self.assertEqual(decisions[-1], "keep")
        np.testing.assert_array_equal(held.current.owner.numpy()[:live], owner, "the held frame's regions")
        np.testing.assert_array_equal(held.current.source.numpy()[:live], -1, "no region carries a dropped anchor")
        self.assertTrue(np.any(held.current.valid.numpy()[:live]), "the regions place their anchors anew")


class TestPatchFrameModes(unittest.TestCase):
    def test_builds_keep_only_the_frame_their_slots_hold(self):
        """Keep a build's pairs only where the slots hold its stream's frame sorted, bitwise as the always-sort chain.

        Both streams hold the same live contacts at the same generation and
        count, so only the stream identity tells them apart.
        """
        for name, (events, expected) in _DECISION_PLANS.items():
            with self.subTest(name=name):
                scene = _Scene("cpu", 45)
                scene.active[:] = True
                scene.frame()
                streams = {"a": _copied_stream(scene), "b": _copied_stream(scene)}
                plain, patches = _recording_pair(scene, 128)
                decisions = _decisions(patches)
                for index, event in enumerate(events):
                    if event[0] == "build":
                        for state in (plain, patches):
                            state.build(scene.model, scene.state, streams[event[1]])
                        assert_same_state(patches, plain, f"event {index}", records=False)
                    elif event[0] == "write":
                        wp.launch(
                            _advance_contact_generation,
                            dim=1,
                            inputs=[streams[event[1]].contact_generation],
                            device="cpu",
                        )
                    elif event[0] == "count":
                        wp.launch(
                            _lower_contact_count, dim=1, inputs=[streams[event[1]].rigid_contact_count], device="cpu"
                        )
                    else:
                        patches.forget_frame()
                self.assertEqual(tuple(decisions), expected)

    def test_builds_follow_the_generation_array_through_contacts_wrappers(self):
        """Key builds through wrappers of a contact stream by the stream's generation array.

        A fresh object around the stream's arrays, as a caller may build per
        substep, keeps the pairs that the previous build of the stream sorted,
        and a write that advances the stream's generation makes it sort.
        Contacts without a generation sort, and so does the next build of the
        stream. The wrappers carry none of the private state of
        :class:`~newton.Contacts`, and every build matches a chain that sorts
        each frame. The write reorders the stream's contacts and keeps its
        count, and the contacts without a generation hold another order, so a
        build that kept its pairs after either would group contacts by stale
        pairs.
        """
        scene = _Scene("cpu", 46)
        scene.active[:] = True
        scene.frame()
        stream = _copied_stream(scene)
        other = _copied_stream(scene)
        anonymous = SimpleNamespace(**{k: v for k, v in vars(other).items() if k != "contact_generation"})
        sources = {
            "stream": lambda: stream,
            "wrapper": lambda: SimpleNamespace(**vars(stream)),
            "anonymous": lambda: anonymous,
        }
        plain, patches = _recording_pair(scene, 128)
        decisions = _decisions(patches)
        for event in ("stream", "wrapper", "write", "wrapper", "stream", "anonymous", "wrapper", "wrapper"):
            if event == "write":
                _copied_stream(scene, stream=stream)
                wp.launch(_advance_contact_generation, dim=1, inputs=[stream.contact_generation], device="cpu")
                continue
            contacts = sources[event]()
            for state in (plain, patches):
                state.build(scene.model, scene.state, contacts)
            assert_same_state(patches, plain, event, records=False)
        self.assertEqual(decisions, ["sort", "keep", "sort", "keep", "sort", "sort", "keep"])

    def test_forget_frame_makes_the_next_build_sort(self):
        """Drop the record on the device, so that the next build sorts the frame the slots hold."""
        scene = _Scene("cpu", 47)
        scene.active[:] = True
        scene.frame()
        patches = _FrictionPatchState(scene.model, 128, wp.zeros(128, dtype=wp.vec2, device="cpu"))
        decisions = _decisions(patches)
        identity = _frame_identity(patches, scene.contacts)
        for event in ("build", "build", "forget", "build"):
            if event == "forget":
                patches.forget_frame()
                _assert_record(self, patches, None, "forgotten")
                continue
            patches.build(scene.model, scene.state, scene.contacts)
            _assert_record(self, patches, identity, f"build {len(decisions)}")
        self.assertEqual(decisions, ["sort", "keep", "sort"])

    def test_sorting_prepare_matches_the_always_sort_chain(self):
        """Prepare and store new frames bitwise as the always-sort chain, and record each frame."""
        run_sorting_prepare_parity(self, "cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "the CUDA build of the prepare kernel needs CUDA")
    def test_sorting_prepare_matches_the_always_sort_chain_on_the_device(self):
        """Prepare bitwise as the always-sort chain on CUDA, whose builds launch the same kernels."""
        run_sorting_prepare_parity(self, "cuda:0", worlds=48, contacts=2048, capacity=3072)

    def test_kept_builds_match_builds_that_sort(self):
        """Keep a frame's sorted pairs on the substeps that share it, bitwise as builds that sort it anew."""
        scene = _Scene("cpu", 42)
        plain, recording = _recording_pair(scene, 128)
        decisions = _decisions(recording)
        kept = 0
        for step, mode in enumerate(("live", "sparse", "live", "empty", "live", "overflow", "live")):
            scene.move()
            live = scene.frame(mode)
            for substep in range(3):
                context = f"step {step} {mode} substep {substep}"
                if substep:
                    scene.move()
                for state in (plain, recording):
                    state.build(scene.model, scene.state, scene.contacts)
                expected = "empty" if live == 0 else ("keep" if substep else "sort")
                self.assertEqual(decisions[-1], expected, context)
                assert_same_state(recording, plain, context, records=False)
                _assert_record(self, recording, _frame_identity(recording, scene.contacts), context)
                assert_empty_tail(recording, context)
                kept += live if substep else 0
                for state in (plain, recording):
                    store_frame(state, scene.state)
                assert_same_state(recording, plain, f"{context} store", records=False)
        self.assertGreater(kept, 0, "substeps must reuse live pairs")

    def test_kept_builds_preserve_the_sorted_pairs_of_live_slots(self):
        """Preserve the pair keys and indices that the recorded frame's live slots hold, and publish no sort.

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
        patches._launch_prepare([*inputs, generation, patches.stream_token(generation), patches.built_frame])
        np.testing.assert_array_equal(patches.current.keys.numpy(), keys)
        np.testing.assert_array_equal(patches.current.indices.numpy(), indices)
        np.testing.assert_array_equal(patches.extent.numpy()[:2], [live, live])
        self.assertEqual(int(patches.extent.numpy()[friction_patches._KEPT_EXTENT]), 1)

    def test_a_frame_changed_since_its_record_sorts(self):
        """Sort a frame whose generation, count or stream differs from the record, or whose record is forgotten.

        Each build matches the always-sort chain bitwise and keeps the anchors
        the frame stored; an advanced generation with the same contacts sorts
        them into the order the slots held.
        """
        for change in ("generation", "count", "stream", "record"):
            with self.subTest(change=change):
                scene = _Scene("cpu", 43)
                plain, patches = _recording_pair(scene, 128)
                decisions = _decisions(patches)
                scene.active[:] = True
                live = scene.frame()
                self.assertGreater(live, 1)
                for state in (plain, patches):
                    state.build(scene.model, scene.state, scene.contacts)
                    store_frame(state, scene.state)
                self.assertTrue(np.any(patches.previous.valid.numpy()), "the frame must store anchors")
                contacts = scene.contacts
                if change == "generation":
                    wp.launch(_advance_contact_generation, dim=1, inputs=[contacts.contact_generation], device="cpu")
                elif change == "count":
                    wp.launch(_lower_contact_count, dim=1, inputs=[contacts.rigid_contact_count], device="cpu")
                elif change == "stream":
                    contacts = _copied_stream(scene)
                else:
                    patches.forget_frame()
                for state in (plain, patches):
                    state.build(scene.model, scene.state, contacts)
                self.assertEqual(decisions, ["sort", "sort"])
                _assert_record(self, patches, _frame_identity(patches, contacts), change)
                assert_same_state(patches, plain, change, records=False)
                for state in (plain, patches):
                    store_frame(state, scene.state)
                assert_same_state(patches, plain, f"{change} store", records=False)
                self.assertTrue(np.any(patches.previous.valid.numpy()), "the sorted frame keeps its anchors")


def _expected_pairs(model, contacts, capacity):
    """Return the sorted pair keys and slot order (by pair key, then contact key) of the live contacts, computed on
    the host."""
    bodies, shapes = model.body_count, model.shape_body.shape[0]
    stride = bodies + shapes + 2
    count = int(contacts.rigid_contact_count.numpy()[0])
    live = min(count, capacity) if 0 <= count <= contacts.rigid_contact_max else 0
    shape_body = model.shape_body.numpy()
    # A body's code (friction_patches.body_codes); index -1 reads the padding of a missing body.
    body_code = np.append(body_pair_codes(model.body_world.numpy()), 0) if bodies else np.zeros(1, np.int64)

    def codes(shape):
        body = np.where(shape >= 0, shape_body[np.maximum(shape, 0)], -1)
        return np.where(body >= 0, body_code[body], bodies + np.where(shape >= 0, shape, shapes) + 1).astype(np.int64)

    a = codes(contacts.rigid_contact_shape0.numpy()[:live])
    b = codes(contacts.rigid_contact_shape1.numpy()[:live])
    keys = np.minimum(a, b) * stride + np.maximum(a, b)
    # A pair's contacts follow their keys, the identities the collision gives them.
    order = np.lexsort((contacts.rigid_contact_key.numpy()[:live], keys))
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


class _SolverTwin:
    """A FeatherPGS scene under default patch friction with two contact streams and double-buffered states.

    :meth:`sort_every_build` makes the twin the always-sort chain that frame
    reuse must match bitwise (:func:`_sort_every_build`). ``jaws`` selects the
    squeezing gripper of the friction-anchor tests, whose graphs capture two
    solver steps each, and ``boxes`` free boxes above a ground plane, whose
    contact frames are empty until they land, instead of the arms over boxes.
    Both pipelines sort the contacts, so twins collide into the same slots:
    the CUDA narrow phase assigns them in a run-dependent order.
    """

    def __init__(self, device, *, jaws=False, boxes=False):
        if jaws:
            from newton.tests.test_feather_pgs_friction_anchors import _SQUEEZE_SOLVER, _build_v_jaws  # noqa: PLC0415

            with wp.ScopedDevice(device):
                self.model, _, _ = _build_v_jaws(5.0)
            options = dict(_SQUEEZE_SOLVER)
            pipeline_options = {"rigid_contact_max": 256, "broad_phase": "nxn"}
        elif boxes:
            self.model = _falling_boxes(device)
            options = {"pgs_iterations": 16}
            pipeline_options = {"rigid_contact_max": 128, "broad_phase": "nxn"}
        else:
            self.model = _arm_over_box(wp.get_device(device), worlds=2)
            options, pipeline_options = {"pgs_iterations": 8}, {}
        self.solver = SolverFeatherPGS(self.model, **options)
        self.patches = self.solver.laws.friction_patches
        self.pipeline = newton.CollisionPipeline(self.model, **pipeline_options, growable_contacts=True)
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
        """Sort every later patch build (:func:`_sort_every_build`) and return the twin."""
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
        """Require the slots to hold the pairs of a stream's live contacts, which lie in the pipeline's order, each at
        its own index."""
        keys, order = _expected_pairs(self.model, self.streams[stream], self.patches.capacity)
        live = len(keys)
        np.testing.assert_array_equal(
            pair_codes(self.patches.current.keys.numpy()[:live], self.patches.key_stride),
            pair_codes(keys, self.patches.key_stride),
            f"{context} keys",
        )
        np.testing.assert_array_equal(order, np.arange(live), f"{context} the pipeline's order")
        np.testing.assert_array_equal(
            self.patches.current.indices.numpy()[:live], np.arange(live), f"{context} indices"
        )
        np.testing.assert_array_equal(
            self.patches.current.keys.numpy()[live : self.patches.capacity], SENTINEL, f"{context} empty keys"
        )
        test.assertGreater(live, 0, f"{context}: the box must touch its support")


def _change(twins, event):
    """Apply a frame change between substeps to every twin; returns the contact stream the next substeps use."""
    if event == "truncate":
        for twin in twins:
            count = int(twin.streams[0].rigid_contact_count.numpy()[0])
            twin.streams[0].rigid_contact_count.assign([count - 1])
    elif event == "write":
        for twin in twins:
            generation = twin.streams[0].contact_generation
            wp.launch(_advance_contact_generation, dim=1, inputs=[generation], device=generation.device)
    elif event in ("notify", "shape"):
        flags = newton.ModelFlags.SHAPE_PROPERTIES if event == "shape" else newton.ModelFlags.BODY_INERTIAL_PROPERTIES
        for twin in twins:
            twin.solver.notify_model_changed(flags)
    elif event == "swap":
        order = np.arange(twins[0].streams[0].rigid_contact_max)
        for twin in twins:
            _permute_contacts(twin.streams[0], twin.streams[1], order)
        return 1
    return 0


# Substep pairs of the solver-level reuse tests, each with its first substep's
# decision; the second keeps. "collide" opens a collision frame, "step" shares
# the last one, "truncate" lowers the count, "write" advances the generation
# as Contacts.clear() does, "notify" reports a body change and "shape" a shape
# change before sharing it, "swap" moves to another stream of equal
# generation, count and contacts, and "back" returns to the first stream.
_STEP_PLAN = (
    ("collide", "sort"),
    ("collide", "sort"),
    ("step", "keep"),
    ("step", "keep"),
    ("truncate", "sort"),
    ("step", "keep"),
    ("write", "sort"),
    ("notify", "keep"),
    ("shape", "sort"),
    ("swap", "sort"),
    ("back", "sort"),
    ("step", "keep"),
    ("collide", "sort"),
)


class TestPatchSolverFrameReuse(unittest.TestCase):
    def test_solver_steps_keep_frames_bitwise_as_the_always_sort_chain(self):
        """Keep the sorted pairs on substeps that share a frame, bitwise as the always-sort chain.

        Every first substep after a collision pass, a lowered count, an
        advanced generation, a shape change, another contact stream or a return
        to the first sorts, and every other substep keeps its pairs, a body
        change included.
        """
        twin, plain = _SolverTwin("cpu"), _SolverTwin("cpu").sort_every_build()
        decisions = _decisions(twin.patches)
        for each in (twin, plain):
            each.pair(collide=True)
        for index, (event, first) in enumerate(_STEP_PLAN):
            context = f"pair {index} {event}"
            stream = _change((twin, plain), event)
            for substep in range(2):
                for each in (twin, plain):
                    each.substep(stream, collide=substep == 0 and event == "collide")
                self.assertEqual(decisions[-1], "keep" if substep else first, f"{context} substep {substep}")
                twin.assert_sorted(self, stream, f"{context} substep {substep}")
                _assert_same_twins(twin, plain, f"{context} substep {substep}")

    def test_caller_count_writes_make_the_next_step_sort(self):
        """Sort a step whose contact count a caller lowered since the last build, as a chain that sorts it does.

        Both twins hold their frames while the boxes land: a landing box has two regions, and a held region takes
        its own centre of pressure where a sorted frame's takes the nearest of its pair, so a chain that sorted
        every landing substep would part from the twin before the step under test.
        """
        twin = _SolverTwin("cpu", boxes=True)
        plain = _SolverTwin("cpu", boxes=True)
        for _ in range(40):
            for each in (twin, plain):
                each.pair(collide=True)
            if np.any(twin.patches.previous.valid.numpy()):
                break
        self.assertGreater(int(twin.streams[0].rigid_contact_count.numpy()[0]), 1)
        self.assertTrue(np.any(twin.patches.previous.valid.numpy()), "the landed boxes must store anchors")
        plain.sort_every_build()
        decisions = _decisions(twin.patches)
        for each in (twin, plain):
            count = int(each.streams[0].rigid_contact_count.numpy()[0])
            each.streams[0].rigid_contact_count.assign([count - 1])
            each.substep()
        self.assertEqual(decisions, ["sort"])
        self.assertTrue(np.any(twin.patches.previous.valid.numpy()), "the sorted frame keeps its anchors")
        _assert_same_twins(twin, plain, "lowered count")
        twin.solver.check_constraint_capacity()

    @unittest.skipUnless(wp.is_cuda_available(), "graph capture needs CUDA")
    def test_solver_graphs_reuse_built_frames(self):
        """Replay collide and substep graphs bitwise as eager always-sort steps, sorting only new frames.

        A probe captured after each build shows on every replay how it prepared
        its frame (:class:`_DecisionProbe`): a graph's second build keeps its
        frame, and its first makes only a frame the slots do not hold new.
        """
        graph = _graph_twin(jaws=True)
        eager = _SolverTwin("cuda:0", jaws=True)
        for _ in range(3):
            for twin in (graph, eager):
                twin.pair(collide=True)
        eager.sort_every_build()
        captured = {}
        probe = _DecisionProbe(graph.patches, 2)
        for name, stream, collide in (("collide", 0, True), ("step", 0, False), ("swap", 1, False)):
            with wp.ScopedCapture(device="cuda:0") as capture:
                graph.substep(stream, collide)
                probe.mark(0)
                graph.substep(stream)
                probe.mark(1)
            captured[name] = capture.graph
        for index, (event, first) in enumerate(_STEP_PLAN):
            context = f"pair {index} {event}"
            stream = _change((graph, eager), event)
            wp.capture_launch(captured[event if event in ("collide", "swap") else "step"])
            eager.pair(stream, collide=event == "collide")
            self.assertEqual(probe.read(), [first, "keep"], context)
            graph.assert_sorted(self, stream, context)
            _assert_same_twins(graph, eager, context)

    @unittest.skipUnless(wp.is_cuda_available(), "graph capture needs CUDA")
    def test_graphs_sort_a_stream_allocated_at_a_freed_address(self):
        """Sort the first captured build over a new Contacts whose generation counter sits where a dropped one was."""
        graph = _graph_twin(jaws=True)
        eager = _SolverTwin("cuda:0", jaws=True)
        # Eager steps leave the slots holding the first stream's frame, recorded as sorted.
        for twin in (graph, eager):
            twin.pair(collide=True)
        eager.sort_every_build()
        order = np.arange(graph.streams[0].rigid_contact_max)
        # The replacement holds the frame, with the same generation and count.
        reused = [_replace_first_stream(twin, order) for twin in (graph, eager)]
        self.assertTrue(reused[0], "the replacement's generation counter must sit at the freed address")
        probe = _DecisionProbe(graph.patches, 2)
        with wp.ScopedCapture(device="cuda:0") as capture:
            graph.substep()
            probe.mark(0)
            graph.substep()
            probe.mark(1)
        wp.capture_launch(capture.graph)
        eager.pair()
        self.assertEqual(probe.read(), ["sort", "keep"], "the replacement must sort")
        graph.assert_sorted(self, 0, "replacement")
        _assert_same_twins(graph, eager, "replacement")

    @unittest.skipUnless(wp.is_cuda_available(), "graph capture needs CUDA")
    def test_graphs_with_a_caller_fork_across_steps_step_as_eager_steps(self):
        """Step bitwise as eager always-sort steps while a stream the caller forked is unjoined.

        Both builds keep the frame that the last eager step left sorted, as without the fork.
        """
        graph = _graph_twin(jaws=True)
        eager = _SolverTwin("cuda:0", jaws=True)
        for _ in range(3):
            for twin in (graph, eager):
                twin.pair(collide=True)
        eager.sort_every_build()
        side = wp.Stream("cuda:0")
        overlapped = wp.zeros(1024, dtype=float, device="cuda:0")
        probe = _DecisionProbe(graph.patches, 2)
        with wp.ScopedCapture(device="cuda:0") as capture:
            main = wp.get_stream("cuda:0")
            side.wait_event(main.record_event())
            with wp.ScopedStream(side, sync_enter=False):
                overlapped.fill_(1.0)
            graph.substep()
            probe.mark(0)
            graph.substep()
            probe.mark(1)
            main.wait_event(side.record_event())
        wp.capture_launch(capture.graph)
        eager.pair()
        self.assertEqual(probe.read(), ["keep", "keep"])
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
        with wp.ScopedCapture(device="cuda:0") as capture:
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
        """Replay graphs bitwise as eager steps over empty, live and again empty contact frames.

        Boxes fall and rest on the ground; a Contacts.clear() empties the frame of a
        replay without a collision pass while their anchors are stored; they collide
        again, are thrown up and land again. A second eager twin is an A/A control:
        the graph must match bitwise for as long as the eager twins do, which must
        cover every transition. The boxes' 128 contact slots take the live sort,
        which skips empty frames. The eager twins hold their frames as the graph
        does: a resting box has two regions, and a held region takes its own centre
        of pressure where a chain that sorted every build would take the nearest of
        its pair.
        """
        graph = _graph_twin(boxes=True)
        eager, control = (_SolverTwin("cuda:0", boxes=True) for _ in range(2))
        twins = (graph, eager, control)
        for twin in twins:
            twin.pair(collide=True)
        captured = {}
        for collide in (True, False):
            with wp.ScopedCapture(device="cuda:0") as capture:
                graph.pair(collide=collide)
            captured[collide] = capture.graph
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


def _graph_twin(**options):
    """Return a CUDA twin to capture graphs of."""
    return _SolverTwin("cuda:0", **options)


def _step_passes(twin, passes, probe=None):
    """Step ``twin`` through collision passes, each a substep count, colliding on each pass's first substep.

    A :class:`_DecisionProbe` records each substep's decision in order.
    """
    slot = 0
    for substeps in passes:
        for substep in range(substeps):
            if substep == 0:
                twin.pipeline.collide(twin.states[0], twin.streams[0])
            twin.substep()
            if probe is not None:
                probe.mark(slot)
            slot += 1


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

    A probe captured after each build shows on every replay that each pass's
    first substep sorts and its later substeps keep their pairs
    (:class:`_DecisionProbe`).
    """
    cuda = wp.get_device(device).is_cuda
    # The squeezing jaws need CUDA; on the CPU, boxes landed on the ground keep their contacts.
    scene = {"jaws": True} if cuda else {"boxes": True}
    for name, passes in _COLLISION_PASSES.items():
        with test.subTest(device=device, passes=name):
            graph = _SolverTwin(device, **scene)
            eager = _SolverTwin(device, **scene)
            for _ in range(2 if cuda else 12):
                for twin in (graph, eager):
                    twin.pair(collide=True)
            eager.sort_every_build()
            probe = _DecisionProbe(graph.patches, sum(passes))
            with wp.ScopedCapture(device=device) as capture:
                _step_passes(graph, passes, probe)
            expected = [decision for count in passes for decision in ("sort", *("keep",) * (count - 1))]
            for replay in range(replays):
                wp.capture_launch(capture.graph)
                _step_passes(eager, passes)
                test.assertEqual(probe.read(), expected, f"{name} replay {replay}: decisions by pass and substep")
                _assert_same_twins(graph, eager, f"{name} replay {replay}")
            test.assertGreater(int(graph.streams[0].rigid_contact_count.numpy()[0]), 0, "the bodies must touch")
            test.assertTrue(np.any(graph.patches.previous.valid.numpy()), "the passes must carry anchors")


def run_captured_steps_without_a_collision_pass(test, device, replays):
    """Sort a substep graph's first build only for a new frame and keep the second, whatever ran in between.

    Collision passes, a body change and a Contacts.clear() run eagerly between
    replays, as they may around a graph of substeps. A replay after a collision
    pass sorts its first build; one after a body change or after nothing keeps
    both builds' pairs, and the cleared frame holds no live contact.
    """
    graph = _SolverTwin(device, boxes=True)
    eager = _SolverTwin(device, boxes=True)
    for _ in range(12):
        for twin in (graph, eager):
            twin.pair(collide=True)
    eager.sort_every_build()
    probe = _DecisionProbe(graph.patches, 2)
    with wp.ScopedCapture(device=device) as capture:
        graph.substep()
        probe.mark(0)
        graph.substep()
        probe.mark(1)
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
        if int(graph.streams[0].rigid_contact_count.numpy()[0]) == 0:
            expected = ["empty", "empty"]
        else:
            expected = ["sort" if replay % 3 == 0 else "keep", "keep"]
        test.assertEqual(probe.read(), expected, f"replay {replay}")
        _assert_same_twins(graph, eager, f"replay {replay}")


def run_captured_contact_writes(test, device):
    """Sort a captured substep whose contacts a caller's kernel rewrote since the previous one, as the chain does.

    The caller's kernel runs between the two captured substeps of a collision
    pass: it lowers the count, advances the generation, or clears the contacts
    and restores their count. Each makes the second substep sort, bitwise as
    the always-sort chain, with its anchors kept.
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

            def collision_pass(twin, rewrite=rewrite, probe=None):
                twin.substep(collide=True)
                if probe is not None:
                    probe.mark(0)
                rewrite(twin)
                twin.substep()
                if probe is not None:
                    probe.mark(1)

            probe = _DecisionProbe(graph.patches, 2)
            with wp.ScopedCapture(device=device) as capture:
                collision_pass(graph, probe=probe)
            wp.capture_launch(capture.graph)
            collision_pass(eager)
            test.assertEqual(probe.read(), ["sort", "sort"])
            _assert_same_twins(graph, eager, write)
            test.assertTrue(np.any(graph.patches.previous.valid.numpy()), "the sorted frame keeps its anchors")


def run_captured_entry_filter(test, device, replays):
    """Keep patch pairs across substeps of a coupled entry's filtered contacts, sorting after each filter pass.

    A graph holds two collision passes. Each collides, filters the contacts for
    a coupled entry and builds the patch frame of the filtered buffer twice, as
    an entry's substeps do. The collision pass advances its own buffer's
    generation only, so the filter's advance of the entry buffer's generation
    alone makes each pass's first build sort. Replays match an always-sort
    patch state bitwise.
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
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=128, broad_phase="nxn", growable_contacts=True)
    contacts = pipeline.contacts()
    state = model.state()
    poses = state.body_q.numpy()
    poses[:, 2] = 0.019
    state.body_q.assign(poses)
    phi = [wp.zeros(128, dtype=wp.vec2, device=device) for _ in range(2)]
    plain = _FrictionPatchState(model, 128, phi[0])
    patches = _FrictionPatchState(model, 128, phi[1])

    def entry_passes(states, passes=2, probe=None):
        """Collide, filter and build twice per pass; the poses move up in a pass and back in the next."""
        for index in range(passes):
            pipeline.collide(state, contacts)
            filtered = coupled.entry_contacts("boxes", contacts)
            for substep in range(2):
                for patch_state in states:
                    patch_state.build(model, state, filtered)
                    store_frame(patch_state, state)
                if probe is not None:
                    probe.mark(2 * index + substep)
                if substep == 0:
                    step = 1.0e-4 if index % 2 == 0 else -1.0e-4
                    wp.launch(_nudge_poses, dim=model.body_count, inputs=[state.body_q, step], device=device)
        return filtered

    filtered = entry_passes((plain, patches), passes=1)
    _sort_every_build(plain)
    test.assertIsNot(filtered, contacts)
    test.assertGreater(int(filtered.rigid_contact_count.numpy()[0]), 0, "the boxes must touch the ground")
    probe = _DecisionProbe(patches, 4)
    with wp.ScopedCapture(device=device) as capture:
        entry_passes((patches,), probe=probe)
    for replay in range(replays):
        before = wp.clone(state.body_q)
        wp.capture_launch(capture.graph)
        state.body_q.assign(before)
        entry_passes((plain,))
        test.assertEqual(probe.read(), ["sort", "keep"] * 2, f"replay {replay}")
        assert_same_state(patches, plain, f"replay {replay}", records=False)
        test.assertGreater(int(filtered.rigid_contact_count.numpy()[0]), 0, f"replay {replay}: the boxes must touch")


def run_replay_past_the_captured_rows(test, device, replays=3):
    """Replay a build captured at 2,048 slots after the pool grew past them.

    Each replay of more live slots builds them bitwise as an eager build at the grown rows; so do the stale state's
    eager builds. The pool asks for no recapture: every launch of the build follows the live rows.
    """
    rows = 16_384
    scene = _Scene(device, 31, worlds=200, contacts=rows)
    states = [
        _FrictionPatchState(scene.model, size, wp.zeros(rows, dtype=wp.vec2, device=device)) for size in (2048, rows)
    ]
    stale, fresh = states
    if not stale.pool.bindable:
        test.skipTest("bound captures need CUDA virtual memory")
    scene.frame("sparse")
    for state in states:
        state.build(scene.model, scene.state, scene.contacts)
    with wp.ScopedCapture(device=device) as capture:
        stale.build(scene.model, scene.state, scene.contacts)
    test.assertFalse(stale.pool.needs_recapture)
    stale.pool.grow(rows)
    test.assertFalse(stale.pool.needs_recapture, "the captured build follows the grown rows")
    scene.active[:] = True
    for replay in range(replays + 1):
        scene.move()
        live = scene.frame("live", churn=False)
        test.assertGreater(live, 4 * 2048, "the live slots must outgrow the captured rows several times")
        if replay < replays:
            wp.capture_launch(capture.graph)
        else:
            stale.build(scene.model, scene.state, scene.contacts)
        fresh.build(scene.model, scene.state, scene.contacts)
        context = f"replay {replay}" if replay < replays else "eager"
        np.testing.assert_array_equal(stale.regions.numpy(), fresh.regions.numpy(), f"{context} regions")
        single, others = (int(count) for count in fresh.regions.numpy())
        for name in _PatchFrame.vars:
            if name in _STORED_FIELDS:
                continue  # the built frame holds none of the stored frame's scratch
            arrays = [getattr(state.current, name).numpy()[:live] for state in (stale, fresh)]
            if name == "regions":
                # The listed regions compare as sets, those of one contact and the others: threads list them in
                # any order.
                arrays = [
                    np.concatenate((np.sort(array[:single]), np.sort(array[live - others :]))) for array in arrays
                ]
            np.testing.assert_array_equal(*arrays, f"{context} {name}")
        for name in ("weight", "next_contact", "point", "phi"):
            np.testing.assert_array_equal(
                getattr(stale.view, name).numpy()[:live], getattr(fresh.view, name).numpy()[:live], f"{context} {name}"
            )
        np.testing.assert_array_equal(stale.built.numpy(), fresh.built.numpy(), f"{context} built")


class TestPatchCapturedFrameReuse(unittest.TestCase):
    """Graphs captured as Isaac Lab captures them, replayed on the CPU and on CUDA."""

    def test_collision_pass_graphs_step_as_the_always_sort_chain(self):
        """Replay [collide, step, ...] graphs bitwise as eager always-sort steps, sorting once per collision pass."""
        run_captured_collision_passes(self, "cpu", replays=4)

    def test_graphs_without_a_collision_pass_sort_only_new_frames(self):
        """Sort a substep graph's first build only after a collision pass and keep the rest, bitwise as eager steps."""
        run_captured_steps_without_a_collision_pass(self, "cpu", replays=6)

    def test_caller_writes_between_captured_steps_make_them_sort(self):
        """Sort a captured substep whose count or generation a caller's kernel changed, bitwise as eager steps."""
        run_captured_contact_writes(self, "cpu")

    def test_coupled_entry_filters_start_new_frames(self):
        """Sort after each coupled entry filter pass and keep the pairs on the entry's later substeps."""
        run_captured_entry_filter(self, "cpu", replays=4)

    @unittest.skipUnless(wp.is_cuda_available(), "graph capture needs CUDA")
    def test_collision_pass_graphs_step_as_the_always_sort_chain_on_the_device(self):
        """Replay collision-pass graphs on CUDA bitwise as eager always-sort steps, with one sort per pass."""
        run_captured_collision_passes(self, "cuda:0", replays=100)

    @unittest.skipUnless(wp.is_cuda_available(), "graph capture needs CUDA")
    def test_graphs_without_a_collision_pass_sort_only_new_frames_on_the_device(self):
        """Sort a CUDA substep graph's first build only after a collision pass and keep the rest, as eager steps."""
        run_captured_steps_without_a_collision_pass(self, "cuda:0", replays=100)

    @unittest.skipUnless(wp.is_cuda_available(), "graph capture needs CUDA")
    def test_caller_writes_between_captured_steps_make_them_sort_on_the_device(self):
        """Sort a CUDA-captured substep whose count or generation a caller's kernel changed, as eager steps do."""
        run_captured_contact_writes(self, "cuda:0")

    @unittest.skipUnless(wp.is_cuda_available(), "graph capture needs CUDA")
    def test_coupled_entry_filters_start_new_frames_on_the_device(self):
        """Sort after each coupled entry filter pass in a CUDA graph and keep the pairs on later substeps."""
        run_captured_entry_filter(self, "cuda:0", replays=100)

    @unittest.skipUnless(wp.is_cuda_available(), "graph capture needs CUDA")
    def test_replays_past_the_captured_rows_build_exactly(self):
        """Build bitwise as eager builds in replays whose live slots outgrew the captured rows."""
        run_replay_past_the_captured_rows(self, "cuda:0")


if __name__ == "__main__":
    unittest.main()
