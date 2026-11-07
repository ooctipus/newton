# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Gate the typed heightfield query rows by their current pairs without changing any contact."""

import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.geometry.heightfield_row_gate import HEIGHTFIELD_ROW_COUNT, HEIGHTFIELD_TYPED_ROWS

FLAGS = {
    "NEWTON_HEIGHTFIELD_CELL_REJECT": "1",
    "NEWTON_HEIGHTFIELD_GEOMETRIC_CULL": "1",
    "NEWTON_HEIGHTFIELD_PRIMITIVE_QUERY": "1",
    "NEWTON_HEIGHTFIELD_FINITE_QUERY": "0",
    "NEWTON_HEIGHTFIELD_PAIR_CSR": "0",
    # The typed rows are the terrain route's: keep the route for every heightfield pair.
    "NEWTON_TERRAIN_LAW": "none",
}
ROW_NAMES = ("sphere", "cylinder", "capsule", "box", "fallback")


def _cube_mesh():
    vertices = np.array([(x, y, z) for x in (-0.25, 0.25) for y in (-0.25, 0.25) for z in (-0.25, 0.25)], np.float32)
    faces = np.array(
        [
            (0, 1, 3),
            (0, 3, 2),
            (4, 6, 7),
            (4, 7, 5),
            (0, 4, 5),
            (0, 5, 1),
            (2, 3, 7),
            (2, 7, 6),
            (0, 2, 6),
            (0, 6, 4),
            (1, 5, 7),
            (1, 7, 3),
        ],
        dtype=np.int32,
    )
    return newton.Mesh(vertices=vertices, indices=faces.ravel())


def _wedge_mesh():
    vertices = 0.3 * np.array([(-1, -1, -1), (1, -1, -1), (0, 1, -1), (0, 0, 1)], dtype=np.float32)
    faces = np.array([(0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)], dtype=np.int32)
    return newton.Mesh(vertices=vertices, indices=faces.ravel())


def rough_terrain_model(
    device,
    kinds,
    *,
    per_kind=3,
    samples=24,
    half=1.0,
    seed=3,
    sink=0.02,
    terrain_yaw=0.0,
    worlds=1,
):
    """Rest shapes of each kind on a rough heightfield with a flat shelf and a step.

    The surface mixes smooth waves with noise, a shelf of exactly equal heights and
    a three-column step. ``airborne`` boxes hover inside the heightfield bounds but
    above every cell under them, and ``big_box`` covers many cells, so the scene
    exercises cell rejection, triangle rejection and long per-pair cell ranges.
    Each world holds its own heightfield and shapes.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 6.0 * np.pi, samples)
    heights = 0.06 * np.sin(x)[:, None] * np.cos(0.7 * x)[None, :] + 0.03 * rng.random((samples, samples))
    heights[samples // 3 : samples // 2, :] = 0.05
    heights[:, samples // 4 : samples // 4 + 3] += 0.15
    heights = heights.astype(np.float32)
    terrain_rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(terrain_yaw))
    terrain_pose = wp.transform((0.5, -0.25, 0.0), terrain_rotation)
    hull, wedge = _cube_mesh(), _wedge_mesh()

    def surface(px, py):
        col = int(np.clip((px + half) / (2.0 * half) * (samples - 1), 0, samples - 1))
        row = int(np.clip((py + half) / (2.0 * half) * (samples - 1), 0, samples - 1))
        return float(heights[row, col])

    builder = newton.ModelBuilder()
    for _world in range(worlds):
        builder.begin_world()
        cfg = builder.ShapeConfig(margin=0.0025, gap=0.01, density=500.0)
        terrain = newton.Heightfield(data=heights, nrow=samples, ncol=samples, hx=half, hy=half)
        builder.add_shape_heightfield(heightfield=terrain, xform=terrain_pose, cfg=cfg)
        for kind in kinds:
            for _ in range(per_kind):
                px, py = rng.uniform(-0.8 * half, 0.8 * half, size=2)
                pz = surface(px, py) + (0.12 if kind == "airborne" else 0.02) - sink
                yaw = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(rng.uniform(0.0, 2.0 * np.pi)))
                if kind in ("box", "hull", "big_box", "capsule") and rng.random() < 0.5:
                    yaw = yaw * wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), float(rng.uniform(-0.4, 0.4)))
                local = wp.transform((float(px), float(py), float(pz)), yaw)
                body = builder.add_body(xform=wp.transform_multiply(terrain_pose, local))
                if kind == "sphere":
                    builder.add_shape_sphere(body=body, radius=0.04, cfg=cfg)
                elif kind == "capsule":
                    builder.add_shape_capsule(body=body, radius=0.03, half_height=0.08, cfg=cfg)
                elif kind in ("box", "airborne"):
                    builder.add_shape_box(body=body, hx=0.06, hy=0.04, hz=0.03, cfg=cfg)
                elif kind == "cylinder":
                    builder.add_shape_cylinder(body=body, radius=0.05, half_height=0.03, cfg=cfg)
                elif kind == "barrel":
                    builder.add_shape_cylinder(body=body, radius=0.05, half_height=0.03, barrel_radius=0.2, cfg=cfg)
                elif kind == "hull":
                    builder.add_shape_convex_hull(body=body, mesh=hull, scale=(0.2, 0.16, 0.12), cfg=cfg)
                elif kind == "wedge":
                    builder.add_shape_convex_hull(body=body, mesh=wedge, scale=(0.17, 0.17, 0.13), cfg=cfg)
                elif kind == "ellipsoid":
                    builder.add_shape_ellipsoid(body=body, rx=0.05, ry=0.04, rz=0.03, cfg=cfg)
                elif kind == "big_box":
                    builder.add_shape_box(body=body, hx=0.35, hy=0.2, hz=0.05, cfg=cfg)
                else:
                    raise ValueError(f"Unknown shape kind {kind!r}")
        builder.end_world()
    return builder.finalize(device=device)


def pipeline(model, gate, *, convex_collision=None, environment=None):
    """Build the stock typed-table pipeline with the row gate on or off."""
    flags = {**FLAGS, **(environment or {}), "NEWTON_HEIGHTFIELD_ROW_GATE": str(int(gate))}
    with patch.dict(os.environ, flags):
        return newton.CollisionPipeline(
            model,
            reduce_contacts=True,
            rigid_contact_max=1 << 14,
            max_triangle_pairs=1 << 16,
            convex_collision=convex_collision,
        )


def raw_buffer(narrow):
    """Return the reducer's raw contacts in buffer order, as exact bit patterns."""
    reducer = narrow.global_contact_reducer
    count = int(reducer.contact_count.numpy()[0])
    assert 0 <= count < reducer.capacity

    def bits(array):
        values = array.numpy()[1 : count + 1]
        return values.view(np.uint32).reshape(count, int(np.prod(values.shape[1:], dtype=int)))

    return np.concatenate(
        [bits(array) for array in (reducer.shape_pairs, reducer.contact_fingerprints, reducer.position_depth)]
        + [bits(reducer.normal)],
        axis=1,
    )


def raw_multiset(narrow):
    """Return the raw contacts sorted by every column, independent of allocation order."""
    rows = raw_buffer(narrow)
    return rows[np.lexsort(rows.T[::-1])] if len(rows) else rows


def exported(contacts):
    """Return the exported contacts in output order, as exact bit patterns."""
    count = int(contacts.rigid_contact_count.numpy()[0])
    names = ("rigid_contact_shape0", "rigid_contact_shape1", "rigid_contact_point0", "rigid_contact_point1")
    names += ("rigid_contact_offset0", "rigid_contact_offset1", "rigid_contact_normal")
    return [getattr(contacts, name).numpy()[:count].view(np.uint32).copy() for name in names]


def row_kernels(narrow):
    """Return the kernels each typed row launches, in row order (the cylinder split replaces row 1)."""
    rows = []
    for row, kernel in enumerate(narrow._heightfield_primitive_kernels):
        split = narrow._heightfield_cylinder_split and row == HEIGHTFIELD_TYPED_ROWS.index(int(newton.GeoType.CYLINDER))
        rows.append(tuple(narrow._heightfield_cylinder_split_kernels) if split else (kernel,))
    return rows


def collide_observed(owner, state, contacts):
    """Collide and record, per typed row, every launched row kernel's stream length input and its value at launch."""
    narrow = owner.narrow_phase
    rows = row_kernels(narrow)
    launched = [[] for _ in rows]
    launch = wp.launch

    def observe(*args, **kwargs):
        kernel = kwargs.get("kernel", args[0] if args else None)
        for row, kernels in enumerate(rows):
            if any(kernel is candidate for candidate in kernels):
                length = kwargs["inputs"][9]  # The row kernels' triangle_pairs_count argument.
                launched[row].append((kernel, length, int(length.numpy()[0])))
        return launch(*args, **kwargs)

    with patch.object(wp, "launch", side_effect=observe):
        owner.collide(state, contacts)
    return launched


class TestHeightfieldRowGate(unittest.TestCase):
    def assert_row_launches(self, narrow, launched, reached):
        """Every row kernel must read its gated length: the stream length if a current pair reaches the row, else 0."""
        triangles = int(narrow.triangle_pairs_count.numpy()[0])
        for row, (name, kernels) in enumerate(zip(ROW_NAMES, row_kernels(narrow), strict=True)):
            with self.subTest(row=name):
                self.assertEqual([entry[0] for entry in launched[row]], list(kernels))
                for _kernel, length, value in launched[row]:
                    self.assertIs(length, narrow._heightfield_row_count_views[row])
                    self.assertEqual(value, triangles if name in reached else 0)

    def collide_pair(self, gated, reference, state, reached=None):
        """Collide both pipelines and require identical raw and exported contacts.

        With ``reached``, also require that the gated rows launch with the gated stream lengths.
        """
        outputs = []
        for owner in (gated, reference):
            contacts = owner.contacts()
            if owner is gated and reached is not None:
                launched = collide_observed(owner, state, contacts)
                self.assert_row_launches(owner.narrow_phase, launched, reached)
            else:
                owner.collide(state, contacts)
            owner.narrow_phase.check_buffer_capacity()
            outputs.append(contacts)
        device = wp.get_device(state.body_q.device)
        if device.is_cpu:
            # CPU kernels run lanes in order, so the gate must leave every buffer bitwise equal.
            np.testing.assert_array_equal(raw_buffer(gated.narrow_phase), raw_buffer(reference.narrow_phase))
            for left, right in zip(exported(outputs[0]), exported(outputs[1]), strict=True):
                np.testing.assert_array_equal(left, right)
        else:
            np.testing.assert_array_equal(raw_multiset(gated.narrow_phase), raw_multiset(reference.narrow_phase))
        return outputs

    def row_lengths(self, owner):
        return dict(zip(ROW_NAMES, owner.narrow_phase._heightfield_row_counts.numpy().tolist(), strict=True))

    def check_rows_follow_current_pairs(self, device):
        """Scan only the rows the current shapes reach, through type and dimension edits."""
        model = rough_terrain_model(device, ("sphere",), per_kind=3)
        state = model.state()
        gated, reference = pipeline(model, True), pipeline(model, False)
        self.assertTrue(gated.narrow_phase._heightfield_row_gate)
        self.assertFalse(reference.narrow_phase._heightfield_row_gate)
        self.collide_pair(gated, reference, state, reached={"sphere"})
        triangles = int(gated.narrow_phase.triangle_pairs_count.numpy()[0])
        self.assertGreater(triangles, 0)
        self.assertGreater(int(gated.narrow_phase.global_contact_reducer.contact_count.numpy()[0]), 0)
        self.assertEqual(self.row_lengths(gated), {**dict.fromkeys(ROW_NAMES, 0), "sphere": triangles})

        graph = None
        if wp.get_device(device).is_cuda:
            contacts = gated.contacts()
            with wp.ScopedCapture(device=device) as capture:
                gated.collide(state, contacts)
            graph = capture.graph

        types = model.shape_type.numpy().copy()
        scales = model.shape_scale.numpy().copy()
        edited = 1  # The first sphere; shape 0 is the heightfield.
        kinds = newton.GeoType
        cases = (
            # A barrel cylinder is served by the generic fallback, a regular one by its typed row.
            ("barrel", kinds.CYLINDER, (0.05, 0.03, 0.2), {"sphere", "fallback"}),
            ("regular", kinds.CYLINDER, (0.05, 0.03, 0.0), {"sphere", "cylinder"}),
            ("capsule", kinds.CAPSULE, (0.03, 0.05, 0.0), {"sphere", "capsule"}),
            ("box", kinds.BOX, (0.04, 0.03, 0.02), {"sphere", "box"}),
            ("ellipsoid", kinds.ELLIPSOID, (0.05, 0.04, 0.03), {"sphere", "fallback"}),
            ("restored", int(types[edited]), tuple(scales[edited]), {"sphere"}),
        )
        for name, kind, scale, reached in cases:
            with self.subTest(device=device, case=name):
                current_types, current_scales = types.copy(), scales.copy()
                current_types[edited] = int(kind)
                current_scales[edited] = scale
                model.shape_type.assign(current_types)
                model.shape_scale.assign(current_scales)
                if graph is None:
                    self.collide_pair(gated, reference, state, reached=reached)
                else:
                    wp.capture_launch(graph)
                    reference.collide(state, reference.contacts())
                    np.testing.assert_array_equal(
                        raw_multiset(gated.narrow_phase), raw_multiset(reference.narrow_phase)
                    )
                triangles = int(gated.narrow_phase.triangle_pairs_count.numpy()[0])
                expected = {row: triangles if row in reached else 0 for row in ROW_NAMES}
                self.assertEqual(self.row_lengths(gated), expected)

    def test_rows_follow_current_pairs_cpu(self):
        """Gate rows by current types and dimensions with bitwise-identical CPU contacts."""
        self.check_rows_follow_current_pairs("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_rows_follow_current_pairs_cuda(self):
        """Gate rows inside a replayed CUDA graph through type and dimension edits."""
        self.check_rows_follow_current_pairs("cuda:0")

    def check_mixed_scene(self, device):
        """Keep every reached row and the closed-form hull row exact."""
        kinds = ("sphere", "capsule", "box", "cylinder", "barrel", "hull", "wedge", "ellipsoid", "airborne", "big_box")
        model = rough_terrain_model(device, kinds, per_kind=2, terrain_yaw=0.3)
        state = model.state()
        gated, reference = pipeline(model, True), pipeline(model, False)
        self.collide_pair(gated, reference, state, reached=set(ROW_NAMES))
        triangles = int(gated.narrow_phase.triangle_pairs_count.numpy()[0])
        self.assertEqual(self.row_lengths(gated), dict.fromkeys(ROW_NAMES, triangles))

        # The staged cylinder query replaces the cylinder row with three kernels that read its gated length.
        split = {"NEWTON_HEIGHTFIELD_CYLINDER_SPLIT": "1", "NEWTON_HEIGHTFIELD_ANALYTIC_PRIMITIVES": "0"}
        spheres = rough_terrain_model(device, ("sphere", "capsule"), per_kind=2)
        for scene, reached in ((model, set(ROW_NAMES)), (spheres, {"sphere", "capsule"})):
            with self.subTest(device=device, cylinder_split=sorted(reached)):
                gated = pipeline(scene, True, environment=split)
                reference = pipeline(scene, False, environment=split)
                self.assertTrue(gated.narrow_phase._heightfield_cylinder_split)
                self.collide_pair(gated, reference, scene.state(), reached=reached)

        hulls = rough_terrain_model(device, ("hull",), per_kind=3)
        state = hulls.state()
        with self.subTest(device=device, convex_collision="auto"):
            gated = pipeline(hulls, True, convex_collision="auto")
            reference = pipeline(hulls, False, convex_collision="auto")
            self.collide_pair(gated, reference, state, reached={"fallback"})
            triangles = int(gated.narrow_phase.triangle_pairs_count.numpy()[0])
            self.assertGreater(triangles, 0)
            self.assertGreater(int(gated.narrow_phase.global_contact_reducer.contact_count.numpy()[0]), 0)
            self.assertEqual(self.row_lengths(gated), {**dict.fromkeys(ROW_NAMES, 0), "fallback": triangles})

    def test_mixed_scene_cpu(self):
        """Gate a mixed terrain scene and the convex-hull routes on CPU."""
        self.check_mixed_scene("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_mixed_scene_cuda(self):
        """Gate a mixed terrain scene and the convex-hull routes on CUDA."""
        self.check_mixed_scene("cuda:0")

    def test_row_order_and_record(self):
        """Keep the row table in gate order and report the gate in the collision record."""
        model = rough_terrain_model("cpu", ("sphere",), per_kind=1)
        gated, reference = pipeline(model, True), pipeline(model, False)
        kernels = gated.narrow_phase._heightfield_primitive_kernels
        self.assertEqual(len(kernels), HEIGHTFIELD_ROW_COUNT)
        for kernel, kind in zip(kernels, HEIGHTFIELD_TYPED_ROWS, strict=False):
            self.assertTrue(kernel.key.startswith(f"heightfield_{newton.GeoType(kind).name.lower()}_"))
        self.assertIn("fallback", kernels[-1].key)
        record = gated.get_collision_selection()["heightfield"]["primitive_query"]["row_gate"]
        self.assertTrue(record["enabled"])
        self.assertIn("current pair", record["reason"])
        record = reference.get_collision_selection()["heightfield"]["primitive_query"]["row_gate"]
        self.assertFalse(record["enabled"])
        self.assertIn("NEWTON_HEIGHTFIELD_ROW_GATE=0", record["reason"])
        self.assertIsNone(reference.narrow_phase._heightfield_row_counts)
        with patch.dict(os.environ, {**FLAGS, "NEWTON_HEIGHTFIELD_PRIMITIVE_QUERY": "0"}):
            plain = newton.CollisionPipeline(model, reduce_contacts=True, rigid_contact_max=256)
        self.assertFalse(plain.narrow_phase._heightfield_row_gate)
        self.assertIn(
            "table is off", plain.get_collision_selection()["heightfield"]["primitive_query"]["row_gate"]["reason"]
        )
        with patch.dict(os.environ, {**FLAGS, "NEWTON_HEIGHTFIELD_ROW_GATE": "yes"}), self.assertRaises(ValueError):
            newton.CollisionPipeline(model, reduce_contacts=True, rigid_contact_max=256)


if __name__ == "__main__":
    unittest.main()
