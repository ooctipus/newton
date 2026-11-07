# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Split long terrain cell windows over lanes without changing a single emitted triangle.

The fixture is the terrain probe's synthetic rough terrain: smooth waves plus
noise, a shelf of equal heights, a step, resting primitives, convex hulls,
barrel cylinders and ellipsoids (generic fallback), boxes hovering above every
cell under them and boxes covering many cells.
"""

import ast
import copy
import ctypes
import inspect
import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.core.environment_flags import NEWTON_FLAGS
from newton._src.geometry import heightfield_cells, heightfield_geometric, heightfield_lanes
from newton._src.geometry.heightfield_geometric import PairProjection
from newton._src.geometry.heightfield_lanes import CellWindow, _cell_window, _reserve_triangles
from newton._src.geometry.narrow_phase import NarrowPhase
from newton._src.utils.heightfield import HeightfieldData
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


def pipeline(model, route, lanes, *, max_triangle_pairs=1 << 16):
    """Build one terrain route with the lane midphase on or off."""
    flags = {
        "NEWTON_HEIGHTFIELD_CELL_REJECT": "1",
        "NEWTON_NARROW_PHASE_PACKED_HEIGHTFIELD_PAIRS": "1",
        "NEWTON_HEIGHTFIELD_FINITE_QUERY": "0",
        "NEWTON_HEIGHTFIELD_PAIR_CSR": "0",
        **ROUTES[route],
        "NEWTON_HEIGHTFIELD_LANE_MIDPHASE": str(int(lanes)),
        # The lane midphase is the terrain route's: keep the route for every heightfield pair.
        "NEWTON_TERRAIN_LAW": "none",
    }
    with patch.dict(os.environ, flags):
        owner = newton.CollisionPipeline(model, reduce_contacts=True, max_triangle_pairs=max_triangle_pairs)
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


def function_source(module, name):
    """Return the (possibly nested) function definition of this name in the module's source."""
    tree = ast.parse(inspect.getsource(module))
    return next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)


class _Rename(ast.NodeTransformer):
    """Rename variables; with ``inline``, also substitute earlier assignments and write vec3(x) as vec3(x, x, x)."""

    def __init__(self, names, inline=None):
        self.names, self.inline = names, inline

    def visit_Name(self, node):
        if self.inline is not None and node.id in self.inline:
            return copy.deepcopy(self.inline[node.id])
        return ast.Name(id=self.names.get(node.id, node.id), ctx=node.ctx)

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if self.inline is not None and ast.unparse(node.func) == "wp.vec3" and len(node.args) == 1:
            node.args = [copy.deepcopy(node.args[0]) for _ in range(3)]
        return node


def statements_until(function, last, names=None):
    """Return the function's leading assignments, up to the one that assigns ``last``, with variables renamed."""
    found = []
    for statement in function.body:
        if isinstance(statement, ast.Assign):
            found.append(ast.dump(_Rename(names or {}).visit(copy.deepcopy(statement))))
            if any(isinstance(t, ast.Name) and t.id == last for t in statement.targets):
                return found
    raise AssertionError(f"{function.name} never assigns {last}")


def dataflow(statements, outputs, names=None):
    """Return each output's expression with every earlier assignment substituted, as AST dumps."""
    values = {}
    for statement in statements:
        if not isinstance(statement, ast.Assign):
            continue
        target, value = statement.targets[0], statement.value
        pairs = zip(target.elts, value.elts, strict=True) if isinstance(target, ast.Tuple) else ((target, value),)
        for name, expression in pairs:
            if isinstance(name, ast.Name):  # Field stores into the window struct are not intermediates.
                values[name.id] = _Rename(names or {}, dict(values)).visit(copy.deepcopy(expression))
    return [ast.dump(values[output]) for output in outputs]


def boundary_model(device):
    """One tilted, turned box over a plateau of a translated, turned heightfield.

    The plateau top lies at world height 0, where a float32 step of the body's
    height is a fraction of an ulp of the bounds in the heightfield frame. A corner
    spike keeps the heightfield's broad-phase box above the shape and a low corner
    keeps its height range wide; the box stays over plateau cells.
    """
    heights = np.full((7, 7), 0.35, dtype=np.float32)
    heights[0, 0], heights[-1, -1] = 1.0, 0.0
    builder = newton.ModelBuilder()
    cfg = builder.ShapeConfig(margin=0.0025, gap=0.01)
    terrain = newton.Heightfield(data=heights, nrow=7, ncol=7, hx=0.3, hy=0.3)
    pose = wp.transform((0.3, -0.2, -0.35), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.2))
    builder.add_shape_heightfield(heightfield=terrain, xform=pose, cfg=cfg)
    turn = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.5) * wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.15)
    body = builder.add_body(xform=wp.transform_multiply(pose, wp.transform((0.02, 0.05, 0.5), turn)))
    builder.add_shape_box(body=body, hx=0.07, hy=0.04, hz=0.03, cfg=cfg)
    return builder.finalize(device=device)


def height(bits):
    """Return the float32 with this bit pattern."""
    return np.int32(bits).view(np.float32)


def emitted_at(owner, state, poses, z):
    """Collide with the body at this float32 world height and return the emitted triangles."""
    current = poses.copy()
    current[0, 2] = z
    state.body_q.assign(current)
    owner.collide(state, owner.contacts())
    # The broad phase keeps the pair at every height, so only the midphase decides.
    assert int(owner.narrow_phase.shape_pairs_mesh_count.numpy()[0]) == 1
    return set(map(tuple, stream(owner.narrow_phase)[0].tolist()))


def emits_triangle(owner, state, poses, triangle=None):
    """Return a height predicate: whether the owner emits this triangle, or any triangle when None."""

    def holds(z):
        triangles = emitted_at(owner, state, poses, z)
        return bool(triangles) if triangle is None else triangle in triangles

    return holds


def last_height(holds, low, high):
    """Bisect positive float32 heights: the bit pattern of the highest one at which ``holds`` is still true."""
    low_bits, high_bits = (int(np.float32(z).view(np.int32)) for z in (low, high))
    assert holds(height(low_bits)) and not holds(height(high_bits))
    while high_bits - low_bits > 1:
        middle = (low_bits + high_bits) // 2
        if holds(height(middle)):
            low_bits = middle
        else:
            high_bits = middle
    return low_bits


@wp.kernel(enable_backward=False)
def window_kernel(
    heightfield: HeightfieldData,
    transforms: wp.array[wp.transform],
    lower: wp.array[wp.vec3],
    upper: wp.array[wp.vec3],
    data: wp.array[wp.vec4],
    gaps: wp.array[float],
    windows: wp.array[CellWindow],
):
    """Build the lane window of the pair (shape 0, shape 1)."""
    windows[0] = _cell_window(0, 0, 1, heightfield, transforms, lower, upper, data, gaps, True)


def reservation_calls(tid):
    """Yield (call, amount) for every reservation this thread makes in ``reserve_kernel``."""
    if tid % 7 == 3:
        return
    for call in range(tid % 5 + (tid // 32) % 3):
        if (tid + call) % 4 == 1:
            continue
        yield call, 1 + (tid * 3 + call) % 2
        if (tid * 5 + call) % 11 == 0:
            return


@wp.kernel(enable_backward=False)
def reserve_kernel(count: wp.array[int], slots: wp.array[wp.vec3i]):
    """Reserve one or two slots per call from lanes with diverging trip counts, skips and early exits."""
    tid = wp.tid()
    if tid % 7 == 3:
        return
    for call in range(tid % 5 + (tid / 32) % 3):
        if (tid + call) % 4 == 1:
            continue
        amount = 1 + (tid * 3 + call) % 2
        base = _reserve_triangles(count, amount)
        for sub in range(amount):
            if base + sub < slots.shape[0]:
                slots[base + sub] = wp.vec3i(tid, call, sub)
        if (tid * 5 + call) % 11 == 0:
            return


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
        capacities = {"max_triangle_pairs": 1 << 20}
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

    def test_window_source_matches_original_midphases(self):
        """Pin the lane window and cell vertices to the original cell and geometric midphases' source.

        The lane passes run the original per-cell tests on a window that
        ``_cell_window`` rebuilds; any drift between the copies would change which
        cells a pair tests. The statements up to ``padding`` must equal the
        original cell midphase's after renaming its shape indices (it alone also
        assigns ``cols``), and the window bounds, height bound and padding must
        equal the original geometric midphase's expressions after substituting
        every intermediate variable.
        """
        window = function_source(heightfield_lanes, "_cell_window")
        cell = function_source(heightfield_cells, "_heightfield_cell_midphase")
        geometric = function_source(heightfield_geometric, "_geometric_midphase")
        shapes = {"hfield_shape": "a", "other_shape": "b"}
        expected = statements_until(cell, "padding", shapes)
        cols = ast.dump(ast.parse("cols = hfd.ncol - 1").body[0])
        self.assertEqual(expected.count(cols), 1)
        expected.remove(cols)
        self.assertEqual(statements_until(window, "padding"), expected)

        outputs = ("col_min", "col_max", "row_min", "row_max", "aabb_lower", "padding")
        arrays = {
            "transforms": "shape_transform",
            "lower": "shape_collision_aabb_lower",
            "upper": "shape_collision_aabb_upper",
            "data": "shape_data",
            "gaps": "shape_gap",
        }
        lane_window = dataflow(window.body, outputs)
        self.assertEqual(dataflow(cell.body, outputs, shapes), lane_window)
        self.assertEqual(
            dataflow(geometric.body, ("cmin", "cmax", "rmin", "rmax", "lo", "padding"), arrays), lane_window
        )

        # Cell vertices and spacing: the geometric midphase's loop body against _cell_survivors and both lane kernels.
        loop = geometric.body[-1].body[0].body
        vertices = ("v0", "v10", "v11", "v01")
        self.assertEqual(
            dataflow(function_source(heightfield_lanes, "_cell_survivors").body, vertices),
            dataflow(loop, vertices, {"r": "row", "c": "col"}),
        )
        spacing = ("dx", "dy", "height_range")

        def assigning(statements):
            # Only the spacing statements: the kernels look hfd up per pair, the original takes it as an argument.
            return [
                statement
                for statement in statements
                if isinstance(statement, ast.Assign)
                and {node.id for node in ast.walk(statement.targets[0]) if isinstance(node, ast.Name)} & set(spacing)
            ]

        expected = dataflow(assigning(geometric.body), spacing)
        for name in ("heightfield_pairs", "heightfield_lanes"):
            kernel = function_source(heightfield_lanes, name)
            body = next(node for node in kernel.body if isinstance(node, ast.For)).body
            self.assertEqual(dataflow(assigning(body), spacing), expected)

    def check_window_boundaries(self, device):
        """Match the original midphase within a few ulps of a cell height bound and of a separating-axis bound.

        Bisection over the body's float32 height finds the highest height at which
        the original cell midphase still emits a triangle, where the shape's
        lower bound meets ``top + padding + 16 ulp * |top|`` of a cell. On the
        geometric route it finds the highest height at which the original
        midphase emits one triangle that its enclosing-box test separates just
        above, while the cell midphase keeps that triangle's cell on both sides.
        Over 24 float32 steps of the body's height on each side of each bound,
        about 3 to 6 ulps of the bound in the heightfield frame, the lane midphase
        with the device's schedule and with the window deferred to pass 2 must
        emit the same triangles and raw contacts. On CUDA this checks that the
        separately compiled passes round the bounds as the original kernel does.
        """
        model = boundary_model(device)
        state = model.state()
        poses = state.body_q.numpy().copy()
        # The box penetrates the plateau at height 0 and clears every cell at 0.25 m.
        low, high = 0.0, 0.25
        cell_original = pipeline(model, "cell", False)
        cell_bound = last_height(emits_triangle(cell_original, state, poses), low, high)
        for route in ("cell", "geometric"):
            with self.subTest(device=device, route=route):
                original = pipeline(model, route, False)
                device_schedule, deferred = pipeline(model, route, True), pipeline(model, route, True)
                deferred.narrow_phase._set_lane_schedule(0, 4)
                bound, triangle = cell_bound, None
                if route == "geometric":
                    # Just below the cell bound every window cell passes the height test; pick a triangle
                    # that the enclosing-box test separates there but keeps 4 cm lower.
                    above, below = height(cell_bound) - 0.0005, height(cell_bound) - 0.04
                    kept = emitted_at(cell_original, state, poses, above)
                    separated = kept - emitted_at(original, state, poses, above)
                    candidates = sorted(separated & emitted_at(original, state, poses, below))
                    self.assertTrue(candidates)
                    triangle = candidates[0]
                    bound = last_height(emits_triangle(original, state, poses, triangle), below, above)
                    for offset in (0, 1):
                        self.assertIn(triangle, emitted_at(cell_original, state, poses, height(bound + offset)))
                holds = emits_triangle(original, state, poses, triangle)
                self.assertTrue(holds(height(bound)))
                self.assertFalse(holds(height(bound + 1)))
                # A body step is 1/8 to 1/4 ulp of the bounds: this scans about 3-6 bound ulps on each side.
                for offset in range(-24, 25):
                    with self.subTest(steps=offset):
                        z = height(bound + offset)
                        emitted_at(original, state, poses, z)
                        for owner in (device_schedule, deferred):
                            emitted_at(owner, state, poses, z)
                            self.assert_same_stream(original, owner, ordered=False)
                        self.assertEqual(deferred_stored(deferred.narrow_phase), 1)

    def test_window_boundaries_cpu(self):
        """Match the original midphase at ulp-level cell and separating-axis bounds on CPU."""
        self.check_window_boundaries("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_window_boundaries_cuda(self):
        """Match the original midphase at ulp-level bounds on CUDA, where each pass is compiled separately."""
        self.check_window_boundaries("cuda:0")

    def check_reserve_triangles(self, device):
        """Partition the triangle counter exactly with warp-aggregated reservations.

        Every thread reserves one or two slots per call, with thread-dependent
        trip counts, skipped calls, early returns before and after reserving and
        a partial last warp, so the lanes that call together change from call
        to call. The reserved ranges must cover ``[0, total)`` exactly once and
        each call's two slots must be consecutive.
        """
        threads = 1000
        count = wp.zeros(1, dtype=int, device=device)
        slots = wp.full(4 * threads, wp.vec3i(-1, -1, -1), dtype=wp.vec3i, device=device)
        wp.launch(reserve_kernel, threads, inputs=[count, slots], device=device, block_dim=128)
        expected = [
            (tid, call, sub)
            for tid in range(threads)
            for call, amount in reservation_calls(tid)
            for sub in range(amount)
        ]
        total = int(count.numpy()[0])
        self.assertEqual(total, len(expected))
        written = [tuple(slot) for slot in slots.numpy()[:total].tolist()]
        self.assertEqual(sorted(written), sorted(expected))
        self.assertTrue(np.all(slots.numpy()[total:] == -1))
        position = {slot: index for index, slot in enumerate(written)}
        for tid, call, sub in expected:
            if sub == 1:
                self.assertEqual(position[(tid, call, 1)], position[(tid, call, 0)] + 1)

    def test_reserve_triangles_cpu(self):
        """Partition the triangle counter with one atomic per reservation on CPU."""
        self.check_reserve_triangles("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_reserve_triangles_cuda(self):
        """Partition the triangle counter with ballots, one leader atomic and a shuffle per CUDA call group."""
        self.check_reserve_triangles("cuda:0")

    def test_launch_arguments_follow_bound_arrays_cpu(self):
        """Build the lane kernels' arguments once and rebuild them when a bound array is replaced."""
        model = rough_terrain_model("cpu", ("sphere", "airborne"), per_kind=2)
        state = model.state()
        reference, actual = pipeline(model, "geometric", False), pipeline(model, "geometric", True)
        narrow = actual.narrow_phase
        self.collide(actual, state)
        built = narrow._heightfield_lane_launch
        self.collide(actual, state)
        self.assertIs(narrow._heightfield_lane_launch, built)
        before = len(stream(narrow)[0])
        # Replace the elevation array with one at the top height, which reaches the hovering boxes.
        model.heightfield_elevations = wp.full_like(model.heightfield_elevations, 1.0)
        self.collide(reference, state)
        self.collide(actual, state)
        self.assertGreater(len(stream(narrow)[0]), before)
        self.assert_same_stream(reference, actual, ordered=True)
        self.assertIsNot(narrow._heightfield_lane_launch, built)

    def test_inverted_window_is_empty_cpu(self):
        """Keep a window empty when an inverted, huge local box saturates its cell bounds.

        aarch64 and CUDA saturate float-to-int conversions, so the first row and
        column become INT_MAX and the last ones INT_MIN. The original range is
        then empty, but ``last - first + 1`` wraps around to 2 in int32: such a
        window would hand pass 2 cells far outside the elevation array. (x86-64
        converts both to INT_MIN, which does not wrap.)
        """
        heightfield = HeightfieldData()
        heightfield.data_offset, heightfield.nrow, heightfield.ncol = 0, 8, 8
        heightfield.hx, heightfield.hy, heightfield.min_z, heightfield.max_z = 0.5, 0.5, 0.0, 1.0
        huge = 1.0e30
        inputs = [
            heightfield,
            wp.array([wp.transform_identity(), wp.transform_identity()], dtype=wp.transform, device="cpu"),
            wp.array([(-1.0, -1.0, -1.0), (huge, huge, huge)], dtype=wp.vec3, device="cpu"),
            wp.array([(1.0, 1.0, 1.0), (-huge, -huge, -huge)], dtype=wp.vec3, device="cpu"),
            wp.zeros(2, dtype=wp.vec4, device="cpu"),
            wp.zeros(2, dtype=float, device="cpu"),
        ]
        windows = wp.zeros(1, dtype=CellWindow, device="cpu")
        wp.launch(window_kernel, 1, inputs=inputs, outputs=[windows], device="cpu")
        window = windows.numpy()[0]
        self.assertEqual((int(window["rows"]), int(window["cols"])), (0, 0))

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
