# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Split long terrain cell windows over lanes without changing a single emitted triangle.

The fixture is the terrain probe's synthetic rough terrain: smooth waves plus
noise, a shelf of equal heights, a step, resting primitives, convex hulls,
barrel cylinders and ellipsoids (generic fallback), boxes hovering above every
cell under them and boxes covering many cells.
"""

import ctypes
import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.core.environment_flags import NEWTON_FLAGS
from newton._src.geometry.heightfield_geometric import PairProjection
from newton._src.geometry.heightfield_lanes import CellWindow
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


def deferred_stored(narrow):
    """Return how many pairs the last pass stored for the lane pass (the rest walked in place)."""
    deferred = narrow._heightfield_lane_deferred
    return min(int(deferred.count.numpy()[0]), deferred.windows.shape[0])


def stored_windows(narrow):
    """Return the windows the last pass stored for the lane pass as {(heightfield, shape): (row, col, rows, cols)}."""
    windows = narrow._heightfield_lane_deferred.windows.numpy()[: deferred_stored(narrow)]
    return {(int(w["a"]), int(w["b"])): (int(w["row"]), int(w["col"]), int(w["rows"]), int(w["cols"])) for w in windows}


# The cells around an elevation sample, as (row, column) offsets, and their triangles that use the sample as a vertex.
_CORNER_CELLS = ((0, 0, (0, 1)), (0, -1, (0,)), (-1, -1, (0, 1)), (-1, 0, (1,)))


def corner_cells(sample, samples):
    """Yield (row, column, triangle) for every grid triangle with this elevation sample as a vertex."""
    sample_row, sample_col = divmod(sample, samples)
    for row_offset, col_offset, subs in _CORNER_CELLS:
        row, col = sample_row + row_offset, sample_col + col_offset
        if 0 <= row < samples - 1 and 0 <= col < samples - 1:
            for sub in subs:
                yield row, col, (row * (samples - 1) + col) * 2 + sub


def corner_triangles(sample, samples):
    """Return the grid triangles with this elevation sample as a vertex."""
    return {triangle for _row, _col, triangle in corner_cells(sample, samples)}


def touching_windows(sample, windows, pairs, samples):
    """Return the (heightfield, shape, triangle) triples of these pairs' window cells with the sample as a vertex."""
    found = set()
    for pair in pairs:
        first_row, first_col, rows, cols = windows[pair]
        for row, col, triangle in corner_cells(sample, samples):
            if first_row <= row < first_row + rows and first_col <= col < first_col + cols:
                found.add((*pair, triangle))
    return found


def edited_corner(windows, pairs, emitted, samples, exclude=()):
    """Pick the window corner of these pairs whose triangles the original midphase rejected most often."""
    candidates = {
        row * samples + col
        for pair in pairs
        for row in range(windows[pair][0], windows[pair][0] + windows[pair][2] + 1)
        for col in range(windows[pair][1], windows[pair][1] + windows[pair][3] + 1)
    }
    return max(
        sorted(candidates - set(exclude)), key=lambda s: len(touching_windows(s, windows, pairs, samples) - emitted)
    )


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
                # The device's own schedule: on CPU nothing is deferred and the stream keeps its
                # order; on CUDA, windows of more than 16 cells take 16 lanes each.
                narrow = actual.narrow_phase
                schedule = (narrow._heightfield_lane_small_cells, narrow._heightfield_lane_width)
                self.assertEqual(schedule, (-1, 1) if cpu else (16, 16))
                if cpu:
                    self.assertEqual(int(narrow._heightfield_lane_deferred.count.numpy()[0]), 0)
                else:
                    self.assertGreater(deferred_stored(narrow), 0)
                self.assert_same_stream(reference, actual, ordered=cpu)
                if cpu:
                    np.testing.assert_array_equal(
                        before.rigid_contact_point0.numpy(), after.rigid_contact_point0.numpy()
                    )
                    self.assertEqual(
                        int(before.rigid_contact_count.numpy()[0]), int(after.rigid_contact_count.numpy()[0])
                    )
                for small_cells, lanes in schedules:
                    with self.subTest(small_cells=small_cells, lanes=lanes):
                        narrow._set_lane_schedule(small_cells, lanes)
                        self.collide(actual, state, after)
                        self.assertGreater(deferred_stored(narrow), 0)
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
        """Read current elevations, poses and dimensions every pass, and keep cells with non-finite corners.

        Every epoch moves, grows and partly removes the shapes, then writes
        non-finite values into three samples: a corner of a window cell of a pair
        that pass 2 splits over lanes, a corner of a window cell of a pair that
        pass 1 walks in place, and, as a control, a sample outside every window.
        The two window samples are chosen where the original midphase rejected
        triangles before the edit. A cell with a non-finite corner is never
        rejected, so every window triangle that has the edited corner as a vertex
        must then be emitted by both midphases.
        """
        # Windows of up to 9 cells stay in place: hovering boxes then supply rejected corners to both passes.
        samples, small_cells, lanes = 24, 9, 8
        model = rough_terrain_model(device, KINDS, per_kind=2, samples=samples, half=1.0)
        state = model.state()
        heights = model.heightfield_elevations.numpy().copy()
        poses = state.body_q.numpy().copy()
        scales = model.shape_scale.numpy().copy()
        for route in ("cell", "geometric"):
            reference, actual = pipeline(model, route, False), pipeline(model, route, True)
            narrow = actual.narrow_phase
            bound = narrow.max_heightfield_pairs
            # Room for every long pair: which pairs pass 2 takes then follows from the window sizes alone.
            narrow._set_lane_schedule(small_cells, lanes, capacity=bound)
            # The probe hands every non-empty window to pass 2, so its storage lists every pair's window.
            probe = pipeline(model, route, True)
            probe.narrow_phase._set_lane_schedule(0, lanes, capacity=bound)
            graph = None
            if wp.get_device(device).is_cuda:
                contacts = actual.contacts()
                actual.collide(state, contacts)  # Load every module before the capture.
                with wp.ScopedCapture(device=device) as capture:
                    actual.collide(state, contacts)
                graph = capture.graph
            for epoch in range(4):
                with self.subTest(device=device, route=route, epoch=epoch):
                    moved = poses.copy()
                    moved[:, 0] += 0.03 * epoch
                    # The last bodies leave the grid in x: part of their window falls outside it.
                    moved[-3:, 0] += 0.9 * epoch
                    state.body_q.assign(moved)
                    sized = scales.copy()
                    sized[1:, :2] *= 1.0 + 0.1 * epoch
                    model.shape_scale.assign(sized)
                    model.heightfield_elevations.assign(heights)
                    self.collide(probe, state)
                    self.collide(reference, state)
                    windows = stored_windows(probe.narrow_phase)
                    self.assertEqual(len(windows), int(probe.narrow_phase._heightfield_lane_deferred.count.numpy()[0]))
                    emitted = set(map(tuple, stream(reference.narrow_phase)[0].tolist()))
                    split = {pair for pair, window in windows.items() if window[2] * window[3] > small_cells}
                    walked = set(windows) - split
                    self.assertTrue(split and walked)
                    deferred_corner = edited_corner(windows, split, emitted, samples)
                    in_place_corner = edited_corner(windows, walked, emitted, samples, exclude={deferred_corner})
                    control = next(
                        s for s in range(len(heights)) if not touching_windows(s, windows, set(windows), samples)
                    )
                    current = heights.copy()
                    current[deferred_corner] = (np.nan, np.inf, -np.inf, np.nan)[epoch]
                    current[in_place_corner] = (-np.inf, np.nan, np.inf, np.inf)[epoch]
                    current[control] = (np.inf, -np.inf, np.nan, -np.inf)[epoch]
                    model.heightfield_elevations.assign(current)
                    self.collide(reference, state)
                    if graph is None:
                        self.collide(actual, state)
                    else:
                        wp.capture_launch(graph)
                    self.assert_same_stream(reference, actual, ordered=False)
                    self.assertEqual(set(stored_windows(narrow)), split)
                    original = set(map(tuple, stream(reference.narrow_phase)[0].tolist()))
                    lanes_stream = set(map(tuple, stream(narrow)[0].tolist()))
                    for corner, pairs in ((deferred_corner, split), (in_place_corner, walked)):
                        expected = touching_windows(corner, windows, pairs, samples)
                        # The edit uncovered triangles the original midphase had rejected.
                        self.assertTrue(expected - emitted)
                        self.assertLessEqual(expected, original)
                        self.assertLessEqual(expected, lanes_stream)
                    self.assertFalse({tri for _a, _b, tri in original} & corner_triangles(control, samples))
            model.heightfield_elevations.assign(heights)
            state.body_q.assign(poses)
            model.shape_scale.assign(scales)

    def test_current_geometry_cpu(self):
        """Follow geometry edits and non-finite corners between CPU passes in both lane passes."""
        self.check_current_geometry("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_current_geometry_cuda(self):
        """Follow geometry edits and non-finite corners inside a replayed CUDA graph in both lane passes."""
        self.check_current_geometry("cuda:0")

    def test_deferred_storage_overflow_cpu(self):
        """Walk pairs in place once the deferred storage is full."""
        model = rough_terrain_model("cpu", KINDS, per_kind=2, samples=32, half=1.2)
        state = model.state()
        reference, actual = pipeline(model, "geometric", False), pipeline(model, "geometric", True)
        narrow = actual.narrow_phase
        narrow._set_lane_schedule(0, 4, capacity=3)
        self.collide(reference, state)
        self.collide(actual, state)
        self.assertGreater(int(narrow._heightfield_lane_deferred.count.numpy()[0]), 3)
        self.assertEqual(deferred_stored(narrow), 3)
        self.assert_same_stream(reference, actual, ordered=False)
        # A schedule that defers but has no storage walks every pair in place.
        narrow._set_lane_schedule(0, 4, capacity=0)
        self.collide(actual, state)
        self.assertGreater(int(narrow._heightfield_lane_deferred.count.numpy()[0]), 0)
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
        narrow._set_lane_schedule(0, 4)
        actual.collide(state, actual.contacts())
        self.assertGreater(deferred_stored(narrow), 0)
        self.assertEqual(int(narrow.triangle_pairs_count.numpy()[0]), total)
        everything = pipeline(model, "geometric", False)
        everything.collide(state, everything.contacts())
        written = set(map(tuple, narrow.triangle_pairs.numpy().tolist()))
        self.assertTrue(written <= set(map(tuple, stream(everything.narrow_phase)[0].tolist())))

    def test_admission_and_record(self):
        """Admit the lane midphase with packed cell rejection and report it in the collision record."""

        def create(**options):
            options = {"max_candidate_pairs": 8, "has_heightfields": True, "has_meshes": False, **options}
            return NarrowPhase(max_triangle_pairs=16, device="cpu", **options)

        flags = {"NEWTON_HEIGHTFIELD_CELL_REJECT": "1", "NEWTON_NARROW_PHASE_PACKED_HEIGHTFIELD_PAIRS": "1"}
        with patch.dict(os.environ, flags):
            os.environ.pop("NEWTON_HEIGHTFIELD_LANE_MIDPHASE", None)
            narrow = create()
            self.assertTrue(narrow._heightfield_lane_midphase)
            # CPU kernels run one lane per block: nothing is deferred, the order is the
            # original one and no deferred storage is allocated.
            self.assertEqual((narrow._heightfield_lane_small_cells, narrow._heightfield_lane_width), (-1, 1))
            self.assertEqual(narrow._heightfield_lane_deferred.windows.shape[0], 0)
            self.assertEqual(narrow._heightfield_lane_deferred.projections.shape[0], 0)
            # A deferring schedule stores half the routed-pair bound.
            self.assertEqual(narrow._heightfield_lane_capacity, 4)
            narrow._set_lane_schedule(16, 16)
            self.assertEqual(narrow._heightfield_lane_deferred.windows.shape[0], 4)
            self.assertEqual(create(max_heightfield_pairs=3)._heightfield_lane_capacity, 2)
            # Without a bound, a standalone NarrowPhase stores at most 1 << 16 deferred pairs.
            many = 1 << 18
            self.assertEqual(create(max_candidate_pairs=many)._heightfield_lane_capacity, 1 << 16)
            self.assertEqual(
                create(max_candidate_pairs=many, max_heightfield_pairs=many)._heightfield_lane_capacity, many // 2
            )
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
        self.assertEqual((record["deferred_capacity"], record["deferred_bytes"]), (0, 0))
        self.assertIn("lanes", record["reason"])
        # Two heightfield pairs per kind and world bound the routed pairs; a deferring
        # schedule stores half of them, each with its window and enclosing-box projection.
        lanes.narrow_phase._set_lane_schedule(16, 16)
        record = lanes.get_collision_selection()["heightfield"]["lane_midphase"]
        per_pair = ctypes.sizeof(CellWindow.ctype) + ctypes.sizeof(PairProjection.ctype)
        self.assertEqual((record["deferred_capacity"], record["deferred_bytes"]), (2, 2 * per_pair))
        record = original.get_collision_selection()["heightfield"]["lane_midphase"]
        self.assertFalse(record["enabled"])
        self.assertIsNone(record["deferred_bytes"])
        self.assertIn("NEWTON_HEIGHTFIELD_LANE_MIDPHASE=0", record["reason"])


if __name__ == "__main__":
    unittest.main()
