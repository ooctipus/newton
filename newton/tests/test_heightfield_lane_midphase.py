# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Split long terrain cell windows over lanes without changing a single emitted triangle.

The fixture is the terrain probe's synthetic rough terrain: smooth waves plus
noise, a shelf of equal heights, a step, resting primitives, convex hulls,
barrel cylinders and ellipsoids (generic fallback), boxes hovering above every
cell under them and boxes covering many cells.
"""

import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.core.environment_flags import NEWTON_FLAGS
from newton._src.geometry.heightfield_geometric import PairProjection
from newton._src.geometry.heightfield_lanes import CellWindow, DeferredPairs
from newton._src.geometry.narrow_phase import NarrowPhase
from newton.tests.test_heightfield_packed_pairs import logical
from newton.tests.test_heightfield_row_gate import raw_multiset, rough_terrain_model

KINDS = ("sphere", "capsule", "box", "cylinder", "barrel", "hull", "wedge", "ellipsoid", "airborne", "big_box")
ROUTES = {
    # Cell rejection alone: every pair emits the triangles of its cells above none of its search box.
    "cell": {"NEWTON_HEIGHTFIELD_GEOMETRIC_CULL": "0", "NEWTON_HEIGHTFIELD_PRIMITIVE_QUERY": "1"},
    # The geometric cull also rejects separated triangles of supported shapes.
    "geometric": {"NEWTON_HEIGHTFIELD_GEOMETRIC_CULL": "1", "NEWTON_HEIGHTFIELD_PRIMITIVE_QUERY": "1"},
    # The pair-CSR route tags every triangle with its pair.
    "pair_csr": {
        "NEWTON_HEIGHTFIELD_GEOMETRIC_CULL": "1",
        "NEWTON_HEIGHTFIELD_FINITE_QUERY": "1",
        "NEWTON_HEIGHTFIELD_PAIR_CSR": "1",
        "NEWTON_HEIGHTFIELD_PRIMITIVE_QUERY": "0",
    },
}
# Deferral and lane counts forced on top of a pipeline's own (CPU: none deferred, one lane).
SCHEDULES = ((0, 1), (0, 4), (6, 16), (2, 32))


def pipeline(model, route, lanes, *, max_triangle_pairs=1 << 16, rigid_contact_max=1 << 14):
    """Build one terrain route with the lane midphase on or off."""
    flags = {
        "NEWTON_HEIGHTFIELD_CELL_REJECT": "1",
        "NEWTON_NARROW_PHASE_PACKED_HEIGHTFIELD_PAIRS": "1",
        "NEWTON_HEIGHTFIELD_FINITE_QUERY": "0",
        "NEWTON_HEIGHTFIELD_PAIR_CSR": "0",
        **ROUTES[route],
        "NEWTON_HEIGHTFIELD_LANE_MIDPHASE": str(int(lanes)),
    }
    with patch.dict(os.environ, flags):
        owner = newton.CollisionPipeline(
            model, reduce_contacts=True, rigid_contact_max=rigid_contact_max, max_triangle_pairs=max_triangle_pairs
        )
    narrow = owner.narrow_phase
    if route == "pair_csr":
        assert narrow._heightfield_pair_csr
    assert narrow._heightfield_geometric_cull == (route != "cell")
    assert narrow._heightfield_lane_midphase == lanes
    return owner


def stream(narrow):
    """Return the triangle stream in emission order and the pair-CSR tags, if any."""
    count = int(narrow.triangle_pairs_count.numpy()[0])
    count = min(count, narrow.triangle_pairs.shape[0])
    triples = narrow.triangle_pairs.numpy()[:count].copy()
    tags = narrow._pair_csr.triangle_pair.numpy()[:count].copy() if narrow._heightfield_pair_csr else None
    return triples, tags


def multiset(triples):
    """Sort (heightfield, shape, triangle) triples: equal results mean equal per-pair triangle multisets."""
    return triples[np.lexsort(triples.T[::-1])] if len(triples) else triples


class TestHeightfieldLaneMidphase(unittest.TestCase):
    def assert_tags_name_pairs(self, narrow):
        """Every pair-CSR tag must index the shape pair that emitted the triangle."""
        triples, tags = stream(narrow)
        if tags is None:
            return
        pairs = narrow.shape_pairs_mesh.numpy()[: int(narrow.shape_pairs_mesh_count.numpy()[0])]
        self.assertTrue(np.all((tags >= 0) & (tags < len(pairs))))
        np.testing.assert_array_equal(pairs[tags], triples[:, :2])

    def collide(self, owner, state, contacts=None):
        contacts = owner.contacts() if contacts is None else contacts
        owner.collide(state, contacts)
        owner.narrow_phase.check_buffer_capacity()
        return contacts

    def assert_same_stream(self, reference, actual, ordered):
        """Require the same triangles (and order when the schedule keeps it) and the same raw contacts."""
        expected, _ = stream(reference.narrow_phase)
        result, _ = stream(actual.narrow_phase)
        self.assertEqual(
            int(reference.narrow_phase.triangle_pairs_count.numpy()[0]),
            int(actual.narrow_phase.triangle_pairs_count.numpy()[0]),
        )
        if ordered:
            np.testing.assert_array_equal(result, expected)
        else:
            np.testing.assert_array_equal(multiset(result), multiset(expected))
        self.assertEqual(logical(reference.narrow_phase), logical(actual.narrow_phase))
        self.assert_tags_name_pairs(actual.narrow_phase)
        np.testing.assert_array_equal(raw_multiset(actual.narrow_phase), raw_multiset(reference.narrow_phase))

    def check_original_triangles(self, device, model, schedules, **capacities):
        """Emit the original triangle stream on every route and under every deferral schedule."""
        state = model.state()
        cpu = wp.get_device(device).is_cpu
        for route in ROUTES:
            with self.subTest(device=device, route=route):
                reference = pipeline(model, route, False, **capacities)
                actual = pipeline(model, route, True, **capacities)
                before, after = self.collide(reference, state), self.collide(actual, state)
                self.assertGreater(int(reference.narrow_phase.triangle_pairs_count.numpy()[0]), 0)
                # The device's own schedule: on CPU nothing is deferred and the stream keeps its order.
                self.assert_same_stream(reference, actual, ordered=cpu)
                if cpu:
                    np.testing.assert_array_equal(
                        before.rigid_contact_point0.numpy(), after.rigid_contact_point0.numpy()
                    )
                    self.assertEqual(
                        int(before.rigid_contact_count.numpy()[0]), int(after.rigid_contact_count.numpy()[0])
                    )
                narrow = actual.narrow_phase
                for small_cells, lanes in schedules:
                    with self.subTest(small_cells=small_cells, lanes=lanes):
                        narrow._heightfield_lane_small_cells = small_cells
                        narrow._heightfield_lane_width = lanes
                        self.collide(actual, state, after)
                        self.assertGreater(int(narrow._heightfield_lane_deferred.count.numpy()[0]), 0)
                        self.assert_same_stream(reference, actual, ordered=False)

    def test_original_triangles_cpu(self):
        """Match the original midphase per pair on CPU, in order without deferral."""
        model = rough_terrain_model("cpu", KINDS, per_kind=3, samples=32, half=1.2, terrain_yaw=0.4)
        self.check_original_triangles("cpu", model, SCHEDULES)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_original_triangles_cuda(self):
        """Match the original midphase per pair with warp lanes on a many-world rough terrain."""
        model = rough_terrain_model("cuda:0", KINDS, per_kind=4, samples=64, half=2.0, terrain_yaw=0.4, worlds=128)
        capacities = {"max_triangle_pairs": 1 << 20, "rigid_contact_max": 1 << 18}
        self.check_original_triangles("cuda:0", model, ((0, 16), (6, 32), (16, 8)), **capacities)

    def check_current_geometry(self, device):
        """Read current elevations, poses and dimensions every pass, including non-finite heights."""
        model = rough_terrain_model(device, KINDS, per_kind=2, samples=24, half=1.0)
        state = model.state()
        for route in ("cell", "geometric"):
            reference, actual = pipeline(model, route, False), pipeline(model, route, True)
            actual.narrow_phase._heightfield_lane_small_cells = 3
            actual.narrow_phase._heightfield_lane_width = 8
            heights = model.heightfield_elevations.numpy().copy()
            poses = state.body_q.numpy().copy()
            scales = model.shape_scale.numpy().copy()
            graph = None
            if wp.get_device(device).is_cuda:
                contacts = actual.contacts()
                with wp.ScopedCapture(device=device) as capture:
                    actual.collide(state, contacts)
                graph = capture.graph
            for epoch in range(4):
                with self.subTest(device=device, route=route, epoch=epoch):
                    current = heights.copy()
                    current[(7 * epoch + 3) % len(current)] = (np.nan, np.inf, -np.inf, 0.2)[epoch]
                    current[(11 * epoch + 5) % len(current)] += 0.05
                    model.heightfield_elevations.assign(current)
                    moved = poses.copy()
                    moved[:, 0] += 0.03 * epoch
                    # The last bodies leave the grid in x: part of their window falls outside it.
                    moved[-3:, 0] += 0.9 * epoch
                    state.body_q.assign(moved)
                    sized = scales.copy()
                    sized[1:, :2] *= 1.0 + 0.1 * epoch
                    model.shape_scale.assign(sized)
                    self.collide(reference, state)
                    if graph is None:
                        self.collide(actual, state)
                    else:
                        wp.capture_launch(graph)
                    self.assert_same_stream(reference, actual, ordered=False)
            model.heightfield_elevations.assign(heights)
            state.body_q.assign(poses)
            model.shape_scale.assign(scales)

    def test_current_geometry_cpu(self):
        """Follow geometry edits between CPU passes with deferred pairs."""
        self.check_current_geometry("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_current_geometry_cuda(self):
        """Follow geometry edits inside a replayed CUDA graph with deferred pairs."""
        self.check_current_geometry("cuda:0")

    def test_deferred_storage_overflow_cpu(self):
        """Walk pairs in place once the deferred storage is full."""
        model = rough_terrain_model("cpu", KINDS, per_kind=2, samples=32, half=1.2)
        state = model.state()
        reference, actual = pipeline(model, "geometric", False), pipeline(model, "geometric", True)
        narrow = actual.narrow_phase
        small = DeferredPairs()
        small.windows = wp.zeros(3, dtype=CellWindow, device="cpu")
        small.projections = wp.zeros(3, dtype=PairProjection, device="cpu")
        small.count = narrow._heightfield_lane_deferred.count
        narrow._heightfield_lane_deferred = small
        narrow._heightfield_lane_small_cells = 0
        narrow._heightfield_lane_width = 4
        self.collide(reference, state)
        self.collide(actual, state)
        self.assertGreater(int(small.count.numpy()[0]), 3)
        self.assert_same_stream(reference, actual, ordered=False)

    def test_triangle_capacity_overflow_cpu(self):
        """Count every triangle past the stream capacity and fill the same prefix without deferral."""
        model = rough_terrain_model("cpu", KINDS, per_kind=2, samples=32, half=1.2)
        state = model.state()
        reference = pipeline(model, "geometric", False, max_triangle_pairs=64)
        actual = pipeline(model, "geometric", True, max_triangle_pairs=64)
        for owner in (reference, actual):
            owner.collide(state, owner.contacts())
        total = int(reference.narrow_phase.triangle_pairs_count.numpy()[0])
        self.assertGreater(total, 64)
        self.assertEqual(int(actual.narrow_phase.triangle_pairs_count.numpy()[0]), total)
        np.testing.assert_array_equal(
            actual.narrow_phase.triangle_pairs.numpy(), reference.narrow_phase.triangle_pairs.numpy()
        )
        narrow = actual.narrow_phase
        narrow._heightfield_lane_small_cells, narrow._heightfield_lane_width = 0, 4
        actual.collide(state, actual.contacts())
        self.assertEqual(int(narrow.triangle_pairs_count.numpy()[0]), total)
        everything = pipeline(model, "geometric", False)
        everything.collide(state, everything.contacts())
        written = set(map(tuple, narrow.triangle_pairs.numpy().tolist()))
        self.assertTrue(written <= set(map(tuple, stream(everything.narrow_phase)[0].tolist())))

    def test_admission_and_record(self):
        """Admit the lane midphase with packed cell rejection and report it in the collision record."""

        def create(**options):
            options = {"has_heightfields": True, "has_meshes": False, **options}
            return NarrowPhase(max_candidate_pairs=8, max_triangle_pairs=16, device="cpu", **options)

        flags = {"NEWTON_HEIGHTFIELD_CELL_REJECT": "1", "NEWTON_NARROW_PHASE_PACKED_HEIGHTFIELD_PAIRS": "1"}
        with patch.dict(os.environ, flags):
            os.environ.pop("NEWTON_HEIGHTFIELD_LANE_MIDPHASE", None)
            narrow = create()
            self.assertTrue(narrow._heightfield_lane_midphase)
            # CPU kernels run one lane per block: nothing is deferred and the order is the original one.
            self.assertEqual((narrow._heightfield_lane_small_cells, narrow._heightfield_lane_width), (-1, 1))
            self.assertEqual(narrow._heightfield_lane_deferred.windows.shape[0], 4)
            self.assertEqual(create(max_heightfield_pairs=3)._heightfield_lane_deferred.windows.shape[0], 2)
            self.assertFalse(create(has_meshes=True)._heightfield_lane_midphase)
            with self.assertRaises(ValueError):
                create(max_heightfield_pairs=-1)
        for name, value in (
            ("NEWTON_HEIGHTFIELD_LANE_MIDPHASE", "0"),
            ("NEWTON_NARROW_PHASE_PACKED_HEIGHTFIELD_PAIRS", "0"),
            ("NEWTON_HEIGHTFIELD_CELL_REJECT", "0"),
        ):
            with patch.dict(os.environ, {**flags, name: value}):
                narrow = create()
                self.assertFalse(narrow._heightfield_lane_midphase)
                self.assertIsNone(narrow._heightfield_lane_deferred)
        with (
            patch.dict(os.environ, {**flags, "NEWTON_HEIGHTFIELD_LANE_MIDPHASE": "yes"}),
            self.assertRaises(ValueError),
        ):
            create()
        self.assertEqual(NEWTON_FLAGS["NEWTON_HEIGHTFIELD_LANE_MIDPHASE"].kind, "policy")

        model = rough_terrain_model("cpu", ("sphere", "capsule"), per_kind=2)
        lanes, original = pipeline(model, "geometric", True), pipeline(model, "geometric", False)
        record = lanes.get_collision_selection()["heightfield"]["lane_midphase"]
        self.assertTrue(record["enabled"])
        self.assertEqual((record["small_cells"], record["lanes"]), (-1, 1))
        # Two heightfield pairs per kind and world: the storage holds half of them.
        self.assertEqual(record["deferred_capacity"], 2)
        self.assertIn("lanes", record["reason"])
        record = original.get_collision_selection()["heightfield"]["lane_midphase"]
        self.assertFalse(record["enabled"])
        self.assertIn("NEWTON_HEIGHTFIELD_LANE_MIDPHASE=0", record["reason"])


if __name__ == "__main__":
    unittest.main()
