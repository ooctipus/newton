# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare the persistent patch mechanism with its full-capacity reference.

Random contact frames cover static and missing shapes, flipped pair orientation,
several contacts per pair, empty tails and overflowing counts. The reference
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
"""

import contextlib
import inspect
import itertools
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import friction_patches
from newton._src.solvers.feather_pgs.friction_patches import _FrictionPatchState, patch_routes
from newton.tests.feather_pgs_patch_reference import SENTINEL, ReferencePatches, pair_codes
from newton.tests.test_feather_pgs_mixed_tree_free_body import _build as _arm_over_box

_PIECE_OFFSET = 0.04
# Float32 epsilons of its scale by which a computed payload may differ on CUDA
# (module docstring). The largest measured difference is 1.12 epsilons at this
# revision and 2.87, in phi, with the anchor choice of the patch-determinism
# work; 8 leaves room for a compile that fuses carried arithmetic differently in
# more places.
_FMA_ULPS = 8
_EPS32 = float(np.finfo(np.float32).eps)
# The twin's (threads, build_threads): one block that strides over every slot,
# where the states under test launch many blocks.
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
        self.contacts = SimpleNamespace(
            rigid_contact_max=contacts,
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
        ``"empty"`` (no contacts) or ``"overflow"`` (a count beyond the arrays).
        ``churn`` toggles a few pairs and footprint sizes first.
        """
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
        if mode == "overflow":
            count = self.capacity + int(rng.integers(1, 6))
        self.contacts.rigid_contact_count.assign([count])
        return count if count <= self.capacity else 0


def _pair(scene, capacity):
    """Return the patch state under test and its reference, both built from ``scene``."""
    phi = [wp.zeros(capacity, dtype=wp.vec2, device=scene.device) for _ in range(2)]
    return _FrictionPatchState(scene.model, capacity, True, phi[0]), ReferencePatches(scene.model, capacity, phi[1])


def _twin(patches, scene):
    """Return a second patch state on another grid, which CUDA tests step alongside ``patches``, or None on the CPU.

    The twin runs the module under test on the same inputs, with
    ``_TWIN_THREADS`` and one sort block, so it must match ``patches`` bitwise
    (:func:`assert_same_state`): a race or any other grid-dependent result makes
    them differ, however small. The CPU compares with the reference bitwise.
    """
    if not patches.device.is_cuda:
        return None
    twin = _FrictionPatchState(scene.model, patches.capacity, True, wp.zeros_like(patches.view.phi))
    twin.threads, twin.build_threads = _TWIN_THREADS
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
    ("tangent_impulse", "own"),
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
    ("tangent_impulse", "own"),
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
    for name, kind in (("phi", "field"), ("point_a", "own"), ("point_b", "own")):
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


def assert_same_state(twin, patches, context=""):
    """Require bitwise-equal arrays in two patch states, scratch included."""
    for frame in ("current", "previous"):
        for name in _PatchFrame.vars:
            np.testing.assert_array_equal(
                getattr(getattr(twin, frame), name).numpy(),
                getattr(getattr(patches, frame), name).numpy(),
                f"{context} {frame}.{name}",
            )
    for name in ("weight", "next_contact", "point_a", "point_b", "phi"):
        np.testing.assert_array_equal(
            getattr(twin.view, name).numpy(), getattr(patches.view, name).numpy(), f"{context} view.{name}"
        )
    for name in ("previous_world", "previous_q", "stored", "built", "extent"):
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
        self.decay = np.where(rng.random(3) < 0.3, 0.0, rng.uniform(0.3, 1.0, size=3))

    def impulse_scale(self, route, impulses):
        """Return the magnitude each row impulse is computed from: its friction pair's largest, else its own.

        Seeding writes both friction rows of a contact from one tangent impulse.
        """
        world, slot, path, needed = (values[: self.live] for values in self.maps)
        scale = np.abs(np.asarray(impulses, dtype=np.float64))
        friction = (path == route) & (slot >= 0) & (needed == 3)
        w, s = world[friction], slot[friction]
        pair = np.maximum(scale[w, s + 1], scale[w, s + 2])
        scale[w, s + 1] = pair
        scale[w, s + 2] = pair
        return scale

    def arrays(self):
        """Return device copies: contact maps and ``(route, parents, mu, impulses, decay)`` rows."""
        device = self.scene.device
        maps = [wp.array(values.astype(np.int32), dtype=int, device=device) for values in self.maps]
        parents, mu, impulses = self.values
        routes = [
            (
                r,
                wp.array(parents[r].astype(np.int32), dtype=int, device=device),
                wp.array(mu[r].astype(np.float32), dtype=float, device=device),
                wp.array(impulses[r].astype(np.float32), dtype=float, device=device),
                float(self.decay[r]),
            )
            for r in self.routes
        ]
        return maps, routes


def _assert_same_rows(mine, theirs, rows, context="", exact=False):
    """Require the reference's row arrays: parents and friction budgets bitwise, impulses as computed payload.

    Linking divides copied budgets by an anchor count, which matches bitwise on
    every device; seeding computes impulses (:func:`_assert_payload_equal`, at
    the scale of ``rows``). ``exact`` compares impulses bitwise too, as twins do.
    """
    for (route, *arrays), (_, *expected) in zip(mine, theirs, strict=True):
        for name, array, reference in zip(("parents", "mu"), arrays[:2], expected[:2], strict=True):
            np.testing.assert_array_equal(array.numpy(), reference.numpy(), f"{context} route {route} {name}")
        impulses = expected[2].numpy()
        _assert_payload_equal(
            arrays[2].numpy(),
            impulses,
            f"{context} route {route} impulses",
            arrays[2].device.is_cuda and not exact,
            rows.impulse_scale(route, impulses),
        )


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
# pair, "empty" and "overflow" leave no live contact, "none" passes no contacts,
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
        patches.threads, patches.build_threads = threads
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
                state.store(scene.state)
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
        _assert_live_rows(mine, their_routes, rows, f"{context} link")
        for state, (maps, routes) in zip(live_states, mine, strict=True):
            seeded = [row for row in routes if row[4] > 0.0]
            if seeded:
                state.seed(scene.state, *maps, patch_routes(seeded))
        reference.seed(scene.contacts, scene.state, *their_maps, their_routes)
        _assert_live_rows(mine, their_routes, rows, f"{context} seed")
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


def _assert_live_rows(mine, theirs, rows, context):
    """Compare the state under test's rows, ``mine[0]``, with the reference's, and a twin's with them bitwise."""
    _assert_same_rows(mine[0][1], theirs, rows, context)
    for _, routes in mine[1:]:
        _assert_same_rows(routes, mine[0][1], rows, f"{context} twin", exact=True)


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
                    for state in (patches, reference):
                        state.store(scene.state)
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
        patches = _FrictionPatchState(scene.model, 96, True, wp.zeros(96, dtype=wp.vec2, device="cpu"))
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
        patches.threads, patches.build_threads = threads
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
    decay = np.array([0.8, 0.0, 0.6])

    def prepare(mode):
        scene.move()
        live = scene.frame(mode)
        reference.build(scene.model, scene.state, scene.contacts)
        rows.generate(reference, live)
        rows.decay = decay
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
    linked, seeded = patch_routes(routes), patch_routes(row for row in routes if row[4] > 0.0)

    def step():
        patches.build(scene.model, scene.state, scene.contacts)
        patches.link(*maps, linked)
        patches.seed(scene.state, *maps, seeded)
        patches.finish(scene.state, out, com, *maps, linked, dt)

    def reference_step():
        their_maps, their_routes = rows.arrays()
        reference.link(scene.contacts, *their_maps, their_routes)
        reference.seed(scene.contacts, scene.state, *their_maps, their_routes)
        reference.finish(scene.contacts, scene.state, out, com, *their_maps, their_routes, dt)
        return their_routes

    def twin_step(context):
        """Step the twin on fresh copies of the rows and compare its rows with the replayed ones."""
        if twin is None:
            return
        twin_maps, twin_routes = rows.arrays()
        twin.build(scene.model, scene.state, scene.contacts)
        twin.link(*twin_maps, patch_routes(twin_routes))
        twin.seed(scene.state, *twin_maps, patch_routes(row for row in twin_routes if row[4] > 0.0))
        twin.finish(scene.state, out, com, *twin_maps, patch_routes(twin_routes), dt)
        _assert_same_rows(twin_routes, routes, rows, f"{context} twin", exact=True)

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
        _assert_same_rows(routes, their_routes, rows, f"replay {index} {mode}")


@contextlib.contextmanager
def _record_patch_routes(patches):
    """Record ``(method, routes)`` for every link, seed and finish call the solver makes on ``patches``."""
    calls = []
    with contextlib.ExitStack() as stack:
        for name in ("link", "seed", "finish"):
            method = getattr(patches, name)

            def record(*args, _name=name, _method=method, **kwargs):
                calls.append((_name, inspect.signature(_method).bind(*args, **kwargs).arguments["routes"]))
                return _method(*args, **kwargs)

            stack.enter_context(mock.patch.object(patches, name, record))
        yield calls


class TestPatchSolverRoutes(unittest.TestCase):
    def test_solver_passes_every_patch_route(self):
        """Link, seed and finish every route that holds patch rows; seed with its decay times dt over the last dt."""
        # Split mode on the CPU routes the fixed-base arm's contacts through dense
        # rows and the free box's through matrix-free rows; matrix-free mode with
        # propagation rows on CUDA enables all three routes.
        configs = [("cpu", {"pgs_mode": "split"}, (1, 1, 0))]
        if wp.is_cuda_available():
            propagation = {"pgs_mode": "matrix_free", "articulated_contact_response": "propagation"}
            configs.append(("cuda:0", propagation, (1, 1, 1)))
        dts = (1.0 / 240.0, 1.0 / 240.0, 1.0 / 120.0, 1.0 / 480.0)
        for (device, mode, enabled), warmstart in itertools.product(configs, (False, True)):
            with self.subTest(device=device, warmstart=warmstart):
                model = _arm_over_box(wp.get_device(device), worlds=2)
                solver = newton.solvers.SolverFeatherPGS(
                    model, pgs_iterations=4, pgs_warmstart=warmstart, pgs_warmstart_decay=0.8, **mode
                )
                self.assertGreater(solver.friction_anchor_beta, 0.0)
                arrays = (
                    (solver.row_parent, solver.row_mu, solver.impulses),
                    (solver.mf_row_parent, solver.mf_row_mu, solver.mf_impulses),
                    (solver.propagation_row_parent, solver.propagation_row_mu, solver.propagation_impulses),
                )
                pipeline = newton.CollisionPipeline(model, contact_matching="latest")
                contacts = pipeline.contacts()
                state, out = model.state(), model.state()
                newton.eval_fk(model, state.joint_q, state.joint_qd, state)
                control = model.control()
                paths = set()
                with _record_patch_routes(solver._friction_patches) as calls:
                    for step, dt in enumerate(dts):
                        pipeline.collide(state, contacts)
                        state.clear_forces()
                        solver.step(state, out, control, contacts, dt)
                        state, out = out, state
                        count = int(contacts.rigid_contact_count.numpy()[0])
                        paths.update(solver.contact_path.numpy()[:count].tolist())
                        expected = ["link", "seed", "finish"] if warmstart else ["link", "finish"]
                        self.assertEqual([name for name, _ in calls], expected, f"step {step}")
                        for name, routes in calls:
                            self.assertEqual(tuple(routes.enabled), enabled, f"step {step} {name}")
                            for route in np.flatnonzero(enabled):
                                for field, array in zip(("parents", "mu", "impulses"), arrays[route], strict=True):
                                    self.assertEqual(getattr(routes, f"{field}{route}").ptr, array.ptr, field)
                        if warmstart:
                            ratio = dt / dts[step - 1] if step else 1.0
                            seeded = dict(calls)["seed"]
                            np.testing.assert_allclose(seeded.scale, 0.8 * ratio * np.asarray(enabled), rtol=1.0e-6)
                        calls.clear()
                    # A step without a frame finishes no route; a step without contacts
                    # drops the stored anchors without calling the patch state.
                    empty = newton.Contacts(0, 0, device=device, contact_matching=True)
                    for step_contacts, expected in ((empty, [("finish", (0, 0, 0))]), (None, [])):
                        solver.step(state, out, control, step_contacts, dts[0])
                        state, out = out, state
                        self.assertEqual([(name, tuple(routes.enabled)) for name, routes in calls], expected)
                        calls.clear()
                    self.assertFalse(np.any(solver._friction_patches.previous.valid.numpy()))
                if device == "cpu":
                    self.assertLessEqual({0, 1}, paths, "contacts must use the dense and matrix-free routes")


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
    patches.extent.assign([live, live])
    return keys[:n]


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
        patches = _FrictionPatchState(scene.model, capacity, True, wp.zeros(capacity, dtype=wp.vec2, device=device))
        counts = {0, 1, 2047, 2048, 2049, capacity // 2, capacity - 1, capacity, *extra_counts}
        for live in sorted(count for count in counts if 0 <= count <= capacity):
            with test.subTest(capacity=capacity, live=live):
                keys = _fill_sort_case(patches, live, rng, codes)
                patches._sort_live()
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
        """Sort the live prefix exactly as a stable sort, from no contacts through several merges."""
        run_live_sort_cases(self, "cpu", (2047, 3000, 4096, 10000))

    def test_capacity_threshold_selects_the_live_sort(self):
        """Sort only the live prefix from the capacity threshold on, while chunk keys fit."""
        scene = _Scene("cpu", 0)
        phi = wp.zeros(96, dtype=wp.vec2, device="cpu")
        self.assertFalse(_FrictionPatchState(scene.model, 96, True, phi).live_sort)
        with mock.patch.object(friction_patches, "_LIVE_SORT_CAPACITY", 96):
            self.assertTrue(_FrictionPatchState(scene.model, 96, True, phi).live_sort)
            self.assertFalse(_FrictionPatchState(scene.model, 95, True, phi[:95]).live_sort)
            # Pair keys of 52 bits and more fall back to the radix sort: their 11-bit
            # slot would reach the chunk keys' pad bit.
            self.assertTrue(friction_patches._uses_live_sort(96, 51))
            self.assertFalse(friction_patches._uses_live_sort(96, 52))

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
        """Sort live prefixes on the device, up to hundreds of chunks and nine merges."""
        run_live_sort_cases(self, "cuda:0", (3000, 10000, 1_000_000), (300_000,))

    @unittest.skipUnless(wp.is_cuda_available(), "multi-block grids need CUDA")
    def test_live_sort_builds_match_on_many_blocks(self):
        """Match the reference with the live sort on multi-block grids, through merges and graph replay."""
        run_reference_differential("cuda:0", 12, worlds=160, contacts=8192, capacity=8192, live_sort=True)
        run_graph_replay("cuda:0", worlds=160, contacts=8192, capacity=8192, live_sort=True)


if __name__ == "__main__":
    unittest.main()
