# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare the persistent patch mechanism with its full-capacity reference.

Random contact frames cover static and missing shapes, flipped pair orientation,
several contacts per pair, empty tails and overflowing counts. The reference
copies the patch kernels as of 98401301 (:mod:`feather_pgs_patch_reference`);
pair keys compare through their decoded body and shape codes.
"""

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs.friction_patches import _FrictionPatchState
from newton.tests.feather_pgs_patch_reference import SENTINEL, ReferencePatches, pair_codes

_PIECE_OFFSET = 0.04


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


def assert_same_patches(patches, reference, live, context=""):
    """Require bitwise-equal patch state on every index a patch kernel reads.

    Sorted keys and indices, validity, ownership, sources, the patch view and
    stored worlds compare everywhere; contact payload on the live contacts;
    stored payload on valid anchors; history poses on the bodies of valid
    anchors. Scratch arrays (``members``, ``used``) and the sort's second halves
    are excluded.
    """
    n = reference.capacity
    stride = patches.key_stride
    for name in ("current", "previous"):
        mine, theirs = getattr(patches, name), getattr(reference, name)
        np.testing.assert_array_equal(
            pair_codes(mine.keys.numpy()[:n], stride), pair_codes(theirs.keys.numpy()[:n]), f"{context} {name}.keys"
        )
        np.testing.assert_array_equal(mine.indices.numpy()[:n], theirs.indices.numpy()[:n], f"{context} {name}.indices")
        np.testing.assert_array_equal(mine.valid.numpy(), theirs.valid.numpy(), f"{context} {name}.valid")
    for name in ("owner", "source"):
        np.testing.assert_array_equal(
            getattr(patches.current, name).numpy(),
            getattr(reference.current, name).numpy(),
            f"{context} current.{name}",
        )
    for name in ("weight", "next_contact", "phi", "point_a", "point_b"):
        np.testing.assert_array_equal(
            getattr(patches.view, name).numpy(), getattr(reference.view, name).numpy(), f"{context} view.{name}"
        )
    np.testing.assert_array_equal(
        patches.previous_world.numpy(), reference.previous_world.numpy(), f"{context} previous_world"
    )
    for name in (
        "center",
        "normal",
        "mu",
        "radius",
        "body_a",
        "body_b",
        "shape_a",
        "shape_b",
        "flipped",
        "surface_a",
        "surface_b",
        "support_gap_limit",
        "eligible",
        "displacement",
        "anchor_a",
        "anchor_b",
        "tangent_impulse",
    ):
        np.testing.assert_array_equal(
            getattr(patches.current, name).numpy()[:live],
            getattr(reference.current, name).numpy()[:live],
            f"{context} current.{name}",
        )
    valid = reference.previous.valid.numpy() != 0
    for name in (
        "normal",
        "displacement",
        "mu",
        "body_a",
        "body_b",
        "owner",
        "shape_a",
        "shape_b",
        "anchor_a",
        "anchor_b",
        "surface_a",
        "surface_b",
        "tangent_impulse",
    ):
        np.testing.assert_array_equal(
            getattr(patches.previous, name).numpy()[valid],
            getattr(reference.previous, name).numpy()[valid],
            f"{context} previous.{name}",
        )
    bodies = np.concatenate((reference.previous.body_a.numpy()[valid], reference.previous.body_b.numpy()[valid]))
    bodies = np.unique(bodies[bodies >= 0])
    np.testing.assert_array_equal(
        patches.previous_q.numpy()[bodies], reference.previous_q.numpy()[bodies], f"{context} previous_q"
    )


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
                self.assertGreater(carried, 0, "the frames must carry anchor history")

    def test_shapeless_pair_ties_the_sentinel_bits(self):
        """Keep a pair whose key equals the sentinel's sorted bits ahead of the empty slots."""
        scene = _Scene("cpu", 5, worlds=1, bodies=4, contacts=16)
        stride = scene.body_count + len(scene.shape_body) + 2
        self.assertEqual(stride & (stride - 1), 0, "the fixture needs a power-of-two key stride")
        patches, reference = _pair(scene, 24)
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


if __name__ == "__main__":
    unittest.main()
