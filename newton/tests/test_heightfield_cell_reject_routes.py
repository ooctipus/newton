# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check conservative terrain rejection against the original triangle pipeline."""

import ast
import inspect
import os
import unittest
from contextlib import nullcontext
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.geometry import heightfield_cells
from newton._src.geometry.heightfield_cells import _cell_is_above
from newton._src.geometry.narrow_phase import NarrowPhase
from newton._src.utils import heightfield as original_heightfield
from newton._src.utils.heightfield import (
    HeightfieldData,
    _heightfield_cell_below_query,
    heightfield_vs_convex_midphase,
)

TERRAIN_ROUTE = {"NEWTON_TERRAIN_LAW": "none"}
"""Cell rejection is the terrain route's: keep the route for every heightfield pair (the terrain law's opt-out)."""


def _create_heightfield_midphase(*, upstream_default):
    """Exercise the upstream call contract or an unculled reference at the launch boundary."""

    @wp.kernel(enable_backward=False)
    def midphase(
        shape_types: wp.array[int],
        shape_transform: wp.array[wp.transform],
        shape_source: wp.array[wp.uint64],
        shape_mesh_properties: wp.array[wp.int32],
        shape_gap: wp.array[float],
        shape_data: wp.array[wp.vec4],
        shape_collision_radius: wp.array[float],
        shape_collision_aabb_lower: wp.array[wp.vec3],
        shape_collision_aabb_upper: wp.array[wp.vec3],
        shape_interior_point: wp.array[wp.vec3],
        shape_heightfield_index: wp.array[wp.int32],
        heightfield_data: wp.array[HeightfieldData],
        heightfield_elevations: wp.array[wp.float32],
        shape_pairs_mesh: wp.array[wp.vec2i],
        shape_pairs_mesh_count: wp.array[int],
        total_num_threads: int,
        triangle_pairs: wp.array[wp.vec3i],
        triangle_pairs_count: wp.array[int],
    ):
        tid, lane = wp.tid()
        if lane != 0:
            return
        for i in range(tid, shape_pairs_mesh_count[0], total_num_threads):
            pair = shape_pairs_mesh[i]
            hfd = heightfield_data[shape_heightfield_index[pair[0]]]
            if wp.static(upstream_default):
                heightfield_vs_convex_midphase(
                    pair[0],
                    pair[1],
                    hfd,
                    heightfield_elevations,
                    shape_transform,
                    shape_collision_aabb_lower,
                    shape_collision_aabb_upper,
                    shape_data,
                    shape_gap,
                    triangle_pairs,
                    triangle_pairs_count,
                )
            else:
                heightfield_vs_convex_midphase(
                    pair[0],
                    pair[1],
                    hfd,
                    heightfield_elevations,
                    shape_transform,
                    shape_collision_aabb_lower,
                    shape_collision_aabb_upper,
                    shape_data,
                    shape_gap,
                    triangle_pairs,
                    triangle_pairs_count,
                    False,
                )

    return midphase


_unculled_heightfield_midphase = _create_heightfield_midphase(upstream_default=False)
_default_heightfield_midphase = _create_heightfield_midphase(upstream_default=True)


@wp.kernel
def _check_cells(lower: wp.array[float], heights: wp.array[float], result: wp.array[int]):
    i = wp.tid()
    terrain = HeightfieldData()
    terrain.data_offset = i * 4
    terrain.nrow = 2
    terrain.ncol = 2
    terrain.min_z = -2.0
    terrain.max_z = 2.0
    result[i] = int(_heightfield_cell_below_query(lower[i], 0.001, terrain, heights, 0, 0))


def _model(
    *,
    device="cpu",
    z=0.2,
    rotation=None,
    reverse=False,
    shape="box",
    scale=(1.0, 1.0, 1.0),
    shape_rotation=None,
    mixed_mesh=False,
):
    builder = newton.ModelBuilder()
    cfg = builder.ShapeConfig(margin=0.01, gap=0.01)
    heights = np.zeros((9, 9), dtype=np.float32)
    heights[0, 0] = 1.0  # Keep the broad-phase terrain box above the distant shape.
    terrain = newton.Heightfield(data=heights, nrow=9, ncol=9, hx=0.4, hy=0.4)
    q = wp.quat_identity() if rotation is None else rotation
    terrain_pose = wp.transform((20.0, -40.0, 0.0), q)
    body_pose = terrain_pose * wp.transform(
        (0.0, 0.0, z), wp.quat_identity() if shape_rotation is None else shape_rotation
    )

    def add_terrain():
        builder.add_shape_heightfield(heightfield=terrain, xform=terrain_pose, scale=scale, cfg=cfg)

    def add_shape():
        body = builder.add_body(xform=body_pose)
        if shape == "box":
            builder.add_shape_box(body, hx=0.10155461, hy=0.03273462, hz=0.00925394, cfg=cfg)
        elif shape == "sphere":
            builder.add_shape_sphere(body, radius=0.035, cfg=cfg)
        elif shape == "capsule":
            builder.add_shape_capsule(body, radius=0.025, half_height=0.04, cfg=cfg)
        else:
            builder.add_shape_cylinder(body, radius=0.035, half_height=0.015, cfg=cfg)

    for add in (add_shape, add_terrain) if reverse else (add_terrain, add_shape):
        add()
    if mixed_mesh:
        mesh = newton.Mesh(
            vertices=np.array([[-0.1, -0.1, 0.0], [0.1, -0.1, 0.0], [0.0, 0.1, 0.0]], dtype=np.float32),
            indices=np.array([0, 1, 2], dtype=np.int32),
        )
        builder.add_shape_mesh(-1, mesh=mesh, xform=wp.transform((30.0, -40.0, 10.0), wp.quat_identity()))
    return builder.finalize(device=device)


def _collide(
    model,
    *,
    legacy=False,
    speculative=False,
    packed=None,
    midphase=None,
    reduce_contacts=False,
    lanes=None,
    schedule=None,
    launches=None,
):
    """Collide once and return the pipeline and the physical contact geometry.

    ``lanes=False`` builds the pipeline with the one-pass packed midphase,
    ``schedule`` forces the lane midphase's ``(small_cells, lanes)`` and
    ``launches`` collects the ``(kernel key, dim)`` of every midphase launch.
    """
    reference = {"NEWTON_HEIGHTFIELD_CELL_REJECT": "0"} if legacy or midphase is not None else {}
    reference.update(TERRAIN_ROUTE)
    if lanes is not None:
        reference["NEWTON_HEIGHTFIELD_LANE_MIDPHASE"] = str(int(lanes))
    with patch.dict(os.environ, reference):
        pipeline = newton.CollisionPipeline(
            model,
            reduce_contacts=reduce_contacts,
            max_triangle_pairs=4096,
            speculative_contact_gap_max=0.5 if speculative else None,
        )
    if legacy:
        midphase = _unculled_heightfield_midphase
    if packed is not None:
        pipeline.narrow_phase._heightfield_packed_pairs = packed
    if midphase is not None:
        pipeline.narrow_phase._heightfield_packed_pairs = False
    if schedule is not None:
        pipeline.narrow_phase._set_lane_schedule(*schedule)
    contacts = pipeline.contacts()
    state = model.state()
    if speculative:
        velocity = np.zeros((model.body_count, 6), dtype=np.float32)
        velocity[:, 2] = -20.0
        state.body_qd.assign(velocity)
    override = (
        patch("newton._src.geometry.narrow_phase.narrow_phase_find_mesh_triangle_overlaps_kernel", midphase)
        if midphase is not None
        else nullcontext()
    )
    launch = wp.launch

    def observe(*args, **kwargs):
        kernel = kwargs.get("kernel", args[0] if args else None)
        key = getattr(kernel, "key", "")
        if "overlaps" in key:
            launches.append((key, kwargs.get("dim")))
        return launch(*args, **kwargs)

    spy = patch.object(wp, "launch", side_effect=observe) if launches is not None else nullcontext()
    with override, spy:
        pipeline.collide(state, contacts, dt=0.01 if speculative else None)
    count = int(contacts.rigid_contact_count.numpy()[0])
    assert count <= contacts.rigid_contact_max
    assert int(pipeline.narrow_phase.triangle_pairs_count.numpy()[0]) <= pipeline.narrow_phase.max_triangle_pairs
    distance = wp.empty(contacts.rigid_contact_max, dtype=float, device=model.device)
    point = wp.empty(contacts.rigid_contact_max, dtype=wp.vec3, device=model.device)
    newton.eval_rigid_contact_kinematics(model, state, contacts, out_distance=distance, out_point0_world=point)
    geometry = (distance.numpy()[:count], contacts.rigid_contact_normal.numpy()[:count], point.numpy()[:count])
    return pipeline, geometry


@wp.kernel
def check_cells(lower: wp.array[float], heights: wp.array[float], result: wp.array[int]):
    """Exercise finite/top-surface admission without full pipeline setup."""
    i = wp.tid()
    hfd = HeightfieldData()
    hfd.data_offset = i * 4
    hfd.nrow = 2
    hfd.ncol = 2
    hfd.hx = 1.0
    hfd.hy = 1.0
    hfd.min_z = 0.0
    hfd.max_z = 1.0
    result[i] = int(_cell_is_above(lower[i], 0.001, hfd, heights, 0, 0))


def make_model(
    device="cpu",
    *,
    z=0.2,
    heights=None,
    rotation=None,
    reverse=False,
    margin=0.01,
    gap=0.01,
    heightfield_scale=(1.0, 1.0, 1.0),
):
    """Create one nonuniform cuboid and the exact finite heightfield surface."""
    builder = newton.ModelBuilder()
    config = builder.ShapeConfig(margin=margin, gap=gap)
    values = np.zeros((9, 9), dtype=np.float32) if heights is None else np.asarray(heights, dtype=np.float32)
    if heights is None:
        # A distant hill keeps the whole terrain AABB overlapping an airborne
        # cuboid; only current per-cell geometry can reject this pair's region.
        values[0, 0] = 1.0
    heightfield = newton.Heightfield(data=values, nrow=9, ncol=9, hx=0.4, hy=0.4)
    q = wp.quat_identity() if rotation is None else rotation
    terrain_x = wp.transform((20.0, -40.0, 0.0), q)
    body_x = wp.transform_multiply(terrain_x, wp.transform((0.0, 0.0, z), wp.quat_identity()))

    def terrain():
        builder.add_shape_heightfield(heightfield=heightfield, xform=terrain_x, cfg=config, scale=heightfield_scale)

    def box():
        body = builder.add_body(xform=body_x)
        builder.add_shape_box(body=body, hx=0.10155461, hy=0.03273462, hz=0.00925394, cfg=config)

    if reverse:
        box()
        terrain()
    else:
        terrain()
        box()
    return builder.finalize(device=device)


def collide(model, enabled, *, reduce_contacts=False, speculative=False):
    """Run the complete collision pipeline with one constructor flag."""
    with patch.dict(os.environ, {"NEWTON_HEIGHTFIELD_CELL_REJECT": str(int(enabled)), **TERRAIN_ROUTE}):
        pipeline = newton.CollisionPipeline(
            model,
            reduce_contacts=reduce_contacts,
            max_triangle_pairs=4096,
            speculative_contact_gap_max=0.5 if speculative else None,
        )
    contacts = pipeline.contacts()
    state = model.state()
    if speculative:
        velocity = np.zeros((model.body_count, 6), dtype=np.float32)
        velocity[:, 2] = -20.0
        state.body_qd.assign(velocity)
    pipeline.collide(state, contacts, dt=0.01 if speculative else None)
    pipeline.narrow_phase.check_buffer_capacity()
    count = int(contacts.rigid_contact_count.numpy()[0])
    distance = wp.empty(contacts.rigid_contact_max, dtype=float, device=model.device)
    point = wp.empty(contacts.rigid_contact_max, dtype=wp.vec3, device=model.device)
    newton.eval_rigid_contact_kinematics(model, state, contacts, out_distance=distance, out_point0_world=point)
    return (
        pipeline,
        contacts,
        (distance.numpy()[:count], contacts.rigid_contact_normal.numpy()[:count], point.numpy()[:count]),
    )


class TestHeightfieldCellRejectRoutes(unittest.TestCase):
    device = "cpu"

    def assert_geometry_equal(self, before, after):
        """Compare complete physical contact geometry independently of row ordering."""
        self.assertTrue(all(np.isfinite(a).all() for values in (before, after) for a in values))
        self.assertEqual(len(before[0]), len(after[0]))
        for source, target in ((before, after), (after, before)):
            error = np.maximum.reduce(
                (
                    np.abs(source[0][:, None] - target[0][None, :]) / 2.0e-4,
                    np.linalg.norm(source[1][:, None] - target[1][None, :], axis=2) / 2.0e-3,
                    np.linalg.norm(source[2][:, None] - target[2][None, :], axis=2) / 2.0e-4,
                )
            )
            self.assertLess(float(np.max(np.min(error, axis=1))), 1.0)

    def test_capability_selection(self):
        """Pack only heightfield-only scenes and retain the mixed-mesh route."""
        for meshes, heightfields, expected in ((False, True, True), (True, True, False), (False, False, False)):
            narrow = NarrowPhase(
                max_candidate_pairs=4,
                max_triangle_pairs=16,
                device=self.device,
                has_meshes=meshes,
                has_heightfields=heightfields,
                reduce_contacts=False,
            )
            self.assertEqual(narrow._heightfield_packed_pairs, expected)

    def test_remove_separated_triangle_work(self):
        """Reject airborne cells even while the terrain broad-phase AABB overlaps."""
        model = _model(device=self.device)
        before, a = _collide(model, legacy=True)
        after, b = _collide(model)
        self.assertGreater(int(before.narrow_phase.triangle_pairs_count.numpy()[0]), 0)
        self.assertEqual(int(after.narrow_phase.triangle_pairs_count.numpy()[0]), 0)
        self.assertEqual(len(a[0]), 0)
        self.assertEqual(len(b[0]), 0)

    def test_upstream_default_call_contract(self):
        """Compile the upstream eleven-argument call and retain default cell rejection."""
        model = _model(device=self.device)
        reference, _ = _collide(model, legacy=True)
        default, geometry = _collide(model, midphase=_default_heightfield_midphase)
        self.assertGreater(int(reference.narrow_phase.triangle_pairs_count.numpy()[0]), 0)
        self.assertEqual(int(default.narrow_phase.triangle_pairs_count.numpy()[0]), 0)
        self.assertEqual(len(geometry[0]), 0)

    def collide_arms(self, model, **options):
        """Collide on every midphase launch route and check which kernels each route launched.

        Returns ``{arm: (pipeline, geometry)}`` for the lane passes with the device's
        schedule and with every window deferred, the one-pass packed launch and the
        tiled launch.
        """
        arms = {
            "lanes": {},
            "lanes_deferred": {"schedule": (0, 4)},
            "packed": {"lanes": False},
            "tiled": {"packed": False},
        }
        results = {}
        for arm, choice in arms.items():
            launches = []
            pipeline, geometry = _collide(model, launches=launches, **choice, **options)
            narrow = pipeline.narrow_phase
            stem = "heightfield_cell_overlaps"
            if arm.startswith("lanes"):
                expected = [f"{stem}_pairs"]
                if narrow._heightfield_lane_small_cells >= 0:
                    expected.append(f"{stem}_lanes")
                self.assertEqual([key for key, _dim in launches], expected)
            else:
                dim = (
                    [narrow.total_num_threads, 1]
                    if arm == "packed"
                    else [narrow.num_tile_blocks, narrow.tile_size_mesh_convex]
                )
                self.assertEqual(launches, [(f"{stem}_kernel", dim)])
            if arm == "lanes_deferred":
                # Every pair with a cell was handed to the lane pass.
                deferred = narrow._heightfield_lane_deferred
                stored = min(int(deferred.count.numpy()[0]), deferred.windows.shape[0])
                self.assertEqual(stored, int(deferred.count.numpy()[0]))
                self.assertGreater(stored, 0)
            results[arm] = (pipeline, geometry)
        return results

    def test_packed_and_tiled_rejection(self):
        """Use identical cell rejection and contact geometry in every launch route."""
        for z in (0.2, 0.02):
            with self.subTest(z=z):
                model = _model(device=self.device, z=z)
                arms = self.collide_arms(model)
                counts = {
                    arm: int(owner.narrow_phase.triangle_pairs_count.numpy()[0]) for arm, (owner, _) in arms.items()
                }
                self.assertEqual(len(set(counts.values())), 1, counts)
                _, a = arms["packed"]
                for arm, (_, b) in arms.items():
                    with self.subTest(arm=arm):
                        if z == 0.2:
                            self.assertEqual(counts[arm], 0)
                            self.assertEqual(len(b[0]), 0)
                        else:
                            self.assertGreater(len(b[0]), 0)
                            self.assert_geometry_equal(a, b)

    def test_mixed_mesh_route(self):
        """Select the tiled route in a real mixed scene without changing terrain results."""
        for z in (0.2, 0.02):
            with self.subTest(z=z):
                packed, a = _collide(_model(device=self.device, z=z))
                mixed, b = _collide(_model(device=self.device, z=z, mixed_mesh=True))
                self.assertTrue(packed.narrow_phase._heightfield_packed_pairs)
                self.assertFalse(mixed.narrow_phase._heightfield_packed_pairs)
                self.assertEqual(
                    int(packed.narrow_phase.triangle_pairs_count.numpy()[0]),
                    int(mixed.narrow_phase.triangle_pairs_count.numpy()[0]),
                )
                if z == 0.2:
                    self.assertEqual(int(mixed.narrow_phase.triangle_pairs_count.numpy()[0]), 0)
                    self.assertEqual(len(a[0]), 0)
                    self.assertEqual(len(b[0]), 0)
                else:
                    self.assertGreater(len(a[0]), 0)
                    self.assert_geometry_equal(a, b)

    def test_near_below_transformed_and_scaled(self):
        """Keep margin contacts, downward prisms, scaled terrain and reversed endpoints."""
        for z, rotation, reverse, scale in (
            (0.035, None, False, (1.0, 1.0, 1.0)),
            (-0.02, None, True, (1.0, 1.0, 1.0)),
            (0.005, wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.45), True, (1.5, 0.7, 2.0)),
        ):
            with self.subTest(z=z):
                model = _model(
                    device=self.device,
                    z=z,
                    rotation=rotation,
                    reverse=reverse,
                    scale=scale,
                    shape_rotation=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.3)
                    if rotation is not None
                    else None,
                )
                before, a = _collide(model, legacy=True)
                after, b = _collide(model)
                self.assertGreater(len(a[0]), 0)
                self.assert_geometry_equal(a, b)
                self.assertEqual(
                    int(before.narrow_phase.triangle_pairs_count.numpy()[0]),
                    int(after.narrow_phase.triangle_pairs_count.numpy()[0]),
                )

    def test_primitive_contact_geometry(self):
        """Preserve contacts for sphere, capsule and cylinder terrain queries."""
        for shape in ("sphere", "capsule", "cylinder"):
            with self.subTest(shape=shape):
                model = _model(device=self.device, z=0.02, shape=shape)
                _, a = _collide(model, legacy=True)
                _, b = _collide(model)
                self.assertGreater(len(a[0]), 0)
                self.assert_geometry_equal(a, b)

    def test_speculative_search_gap(self):
        """Keep approaching contacts admitted by the current expanded search gap."""
        model = _model(device=self.device, z=0.15)
        _, a = _collide(model, legacy=True, speculative=True)
        pipeline, b = _collide(model, speculative=True)
        self.assertGreater(float(pipeline._shape_search_gap.numpy().max()), float(model.shape_gap.numpy().max()))
        self.assertGreater(len(a[0]), 0)
        self.assert_geometry_equal(a, b)

    def test_large_vertical_plane(self):
        """Retain plane candidates whose cached unit AABB does not bound the surface."""
        builder = newton.ModelBuilder()
        terrain = newton.Heightfield(data=np.zeros((5, 5), dtype=np.float32), nrow=5, ncol=5, hx=0.4, hy=0.4)
        builder.add_shape_heightfield(heightfield=terrain)
        plane_body = builder.add_body(mass=1.0)
        builder.add_shape_plane(
            body=plane_body,
            xform=wp.transform((0.0, 0.0, 2.0), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), np.pi / 2)),
            width=10.0,
            length=10.0,
        )
        model = builder.finalize(device=self.device)
        before, a = _collide(model, legacy=True)
        self.assertGreater(int(before.narrow_phase.triangle_pairs_count.numpy()[0]), 0)
        self.assertGreater(len(a[0]), 0)
        for arm, (after, b) in self.collide_arms(model).items():
            with self.subTest(arm=arm):
                self.assertEqual(
                    int(before.narrow_phase.triangle_pairs_count.numpy()[0]),
                    int(after.narrow_phase.triangle_pairs_count.numpy()[0]),
                )
                self.assert_geometry_equal(a, b)

        # Research query fallbacks must retain main's per-plane exemption too.
        before, a = _collide(model, legacy=True, reduce_contacts=True)
        for pair_csr in (False, True):
            with (
                self.subTest(pair_csr=pair_csr),
                patch.dict(
                    os.environ,
                    {
                        "NEWTON_HEIGHTFIELD_FINITE_QUERY": "1",
                        "NEWTON_HEIGHTFIELD_GEOMETRIC_CULL": "1",
                        "NEWTON_HEIGHTFIELD_PAIR_CSR": str(int(pair_csr)),
                    },
                ),
            ):
                after, b = _collide(model, reduce_contacts=True)
                self.assertTrue(after.narrow_phase._heightfield_geometric_cull)
                self.assertEqual(after.narrow_phase._heightfield_pair_csr, pair_csr)
                self.assertGreater(len(a[0]), 0)
                self.assert_geometry_equal(a, b)
                self.assertEqual(
                    int(before.narrow_phase.triangle_pairs_count.numpy()[0]),
                    int(after.narrow_phase.triangle_pairs_count.numpy()[0]),
                )

    def test_current_height_updates(self):
        """Read current elevations instead of retaining a cached empty-cell decision."""
        model = _model(device=self.device, z=0.2)
        before, _ = _collide(model)
        self.assertEqual(int(before.narrow_phase.triangle_pairs_count.numpy()[0]), 0)
        values = model.heightfield_elevations.numpy()
        values.reshape(9, 9)[2:7, 2:7] = 0.2
        model.heightfield_elevations.assign(values)
        after, geometry = _collide(model)
        self.assertGreater(int(after.narrow_phase.triangle_pairs_count.numpy()[0]), 0)
        self.assertGreater(len(geometry[0]), 0)

    def test_replay_height_withdrawal_and_regrowth(self):
        """Refresh live heights in one pipeline through captured withdrawal and regrowth."""
        model = _model(device=self.device, z=0.2)
        with patch.dict(os.environ, TERRAIN_ROUTE):
            pipeline = newton.CollisionPipeline(model, reduce_contacts=False, max_triangle_pairs=4096)
        contacts = pipeline.contacts()
        state = model.state()
        pipeline.collide(state, contacts)
        graph = None
        if model.device.is_cuda:
            with wp.ScopedCapture(device=model.device) as capture:
                pipeline.collide(state, contacts)
            graph = capture.graph
        initial = model.heightfield_elevations.numpy().copy()
        for elevated in (True, False, True, False):
            values = initial.copy()
            if elevated:
                values.reshape(9, 9)[2:7, 2:7] = 0.2
            model.heightfield_elevations.assign(values)
            if graph is None:
                pipeline.collide(state, contacts)
            else:
                wp.capture_launch(graph)
            triangle_count = int(pipeline.narrow_phase.triangle_pairs_count.numpy()[0])
            contact_count = int(contacts.rigid_contact_count.numpy()[0])
            self.assertLessEqual(triangle_count, pipeline.narrow_phase.max_triangle_pairs)
            self.assertLessEqual(contact_count, contacts.rigid_contact_max)
            self.assertEqual(triangle_count > 0, elevated)
            self.assertEqual(contact_count > 0, elevated)

    def test_negative_heights_ties_nonfinite_and_current_corners(self):
        """Retain boundary ties, deep prisms and unknown corners at negative heights."""
        lower = wp.array([-1.999, -1.998, -10.0, 0.5, 0.5, 0.5], dtype=float, device=self.device)
        values = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.65],
                [0.0, np.nan, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        heights = wp.array(values.ravel(), dtype=float, device=self.device)
        result = wp.empty(6, dtype=int, device=self.device)
        wp.launch(_check_cells, 6, inputs=[lower, heights, result], device=self.device)
        np.testing.assert_array_equal(result.numpy(), [0, 1, 0, 0, 0, 1])
        values[5, 3] = 0.7
        heights.assign(values.ravel())
        wp.launch(_check_cells, 6, inputs=[lower, heights, result], device=self.device)
        self.assertEqual(int(result.numpy()[5]), 0)


class TestHeightfieldCellRejectRoutesResearch(unittest.TestCase):
    """Keep constructor admission and complete surface geometry independently visible."""

    device = "cpu"

    def test_default_admission_and_explicit_opt_out(self):
        """Enable the shared heightfield owner by default without admitting mixed meshes."""
        with patch.dict(os.environ):
            os.environ.pop("NEWTON_HEIGHTFIELD_CELL_REJECT", None)
            os.environ.pop("NEWTON_NARROW_PHASE_PACKED_HEIGHTFIELD_PAIRS", None)
            pipeline = newton.CollisionPipeline(make_model(self.device))
            self.assertTrue(pipeline.narrow_phase._heightfield_cell_reject)
            self.assertTrue(pipeline.narrow_phase._heightfield_packed_pairs)
            for has_meshes, has_heightfields in ((True, True), (False, False)):
                narrow = NarrowPhase(
                    max_candidate_pairs=4,
                    max_triangle_pairs=16,
                    device=self.device,
                    has_meshes=has_meshes,
                    has_heightfields=has_heightfields,
                    reduce_contacts=False,
                )
                self.assertFalse(narrow._heightfield_cell_reject)
                self.assertFalse(narrow._heightfield_packed_pairs)
            os.environ["NEWTON_HEIGHTFIELD_CELL_REJECT"] = "0"
            pipeline = newton.CollisionPipeline(make_model(self.device))
            self.assertFalse(pipeline.narrow_phase._heightfield_cell_reject)
            self.assertFalse(pipeline.narrow_phase._heightfield_packed_pairs)

    def test_missing_heightfield_arrays_rejected_before_launch(self):
        """Reject missing heightfield inputs before either midphase dereferences them."""
        model = make_model(self.device, z=0.015)
        for enabled in (False, True):
            pipeline, contacts, _ = collide(model, enabled)
            original_write = pipeline.narrow_phase.launch_custom_write
            for missing in ("shape_heightfield_index", "heightfield_data", "heightfield_elevations"):
                with self.subTest(enabled=enabled, missing=missing):

                    def without_array(*args, _missing=missing, _write=original_write, **kwargs):
                        kwargs.pop(_missing, None)
                        # Guard before native dispatch so the pre-fix null
                        # dereference fails this regression, not the process.
                        with patch.object(wp, "launch", side_effect=AssertionError("Missing input reached a kernel")):
                            return _write(*args, **kwargs)

                    with (
                        patch.object(pipeline.narrow_phase, "launch_custom_write", side_effect=without_array),
                        self.assertRaisesRegex(ValueError, missing),
                    ):
                        pipeline.collide(model.state(), contacts)

    def test_nonheightfield_call_may_omit_heightfield_arrays(self):
        """Keep optional heightfield inputs optional for primitive-only pipelines."""
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        body = builder.add_body(xform=wp.transform((0.0, 0.0, 0.02), wp.quat_identity()))
        builder.add_shape_sphere(body=body, radius=0.05)
        model = builder.finalize(device=self.device)
        pipeline = newton.CollisionPipeline(model)
        self.assertFalse(pipeline.narrow_phase._heightfield_cell_reject)
        original_write = pipeline.narrow_phase.launch_custom_write

        def without_arrays(*args, **kwargs):
            for name in ("shape_heightfield_index", "heightfield_data", "heightfield_elevations"):
                kwargs.pop(name, None)
            return original_write(*args, **kwargs)

        contacts = pipeline.contacts()
        with patch.object(pipeline.narrow_phase, "launch_custom_write", side_effect=without_arrays):
            pipeline.collide(model.state(), contacts)
        pipeline.narrow_phase.check_buffer_capacity()
        self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)

    def _assert_geometry_coverage(self, a, b):
        """Require bidirectional full-contact geometry coverage, independent of ordering."""
        self.assertTrue(all(np.isfinite(value).all() for values in (a, b) for value in values))
        for source, target in ((a, b), (b, a)):
            distance = np.abs(source[0][:, None] - target[0][None, :])
            normal = np.linalg.norm(source[1][:, None] - target[1][None, :], axis=2)
            position = np.linalg.norm(source[2][:, None] - target[2][None, :], axis=2)
            score = np.maximum.reduce((distance / 2e-4, normal / 2e-3, position / 2e-4))
            self.assertLess(float(np.max(np.min(score, axis=1))), 1.0)

    def test_reject_above_cells_and_keep_default(self):
        """Remove separated triangle work on both routes: upstream's midphase rejects above cells too."""
        model = make_model(self.device)
        old, _, old_geometry = collide(model, False)
        new, _, new_geometry = collide(model, True)
        self.assertFalse(old.narrow_phase._heightfield_cell_reject)
        self.assertTrue(new.narrow_phase._heightfield_cell_reject)
        self.assertEqual(int(old.narrow_phase.triangle_pairs_count.numpy()[0]), 0)
        self.assertEqual(int(new.narrow_phase.triangle_pairs_count.numpy()[0]), 0)
        self.assertEqual(len(old_geometry[0]), 0)
        self.assertEqual(len(new_geometry[0]), 0)

    def test_near_below_rotated_and_reversed_keep_physical_contacts(self):
        """Retain surface distances and normals across penetration and transformed endpoints."""
        for z, rotation, reverse in (
            (0.015, None, False),
            (-0.02, None, True),
            (0.005, wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.45), True),
        ):
            with self.subTest(z=z, reverse=reverse):
                model = make_model(self.device, z=z, rotation=rotation, reverse=reverse)
                old, _, a = collide(model, False)
                new, _, b = collide(model, True)
                self.assertGreater(len(a[0]), 0)
                self.assertGreater(len(b[0]), 0)
                self._assert_geometry_coverage(a, b)
                self.assertAlmostEqual(float(a[0].min()), float(b[0].min()), delta=2e-5)
                self.assertLess(float(np.min(b[0])), 0.05)
                self.assertEqual(
                    int(old.narrow_phase.triangle_pairs_count.numpy()[0]),
                    int(new.narrow_phase.triangle_pairs_count.numpy()[0]),
                )
                # No dependence on scheduling/contact ordering in this geometry control.
                self.assertLess(
                    float(np.max(np.min(np.linalg.norm(b[1][:, None] - a[1][None, :], axis=2), axis=1))), 2e-4
                )

    def test_threshold_nonflat_nonfinite_and_current_height(self):
        """Retain ties, negative-depth prisms and unknown corner heights."""
        lower = wp.array([0.001, 0.002, -0.5, 0.5, 0.5, 0.5], dtype=float, device=self.device)
        values = np.array(
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0.6], [0, np.nan, 0, 0], [0, 0, 0, 0]],
            dtype=np.float32,
        )
        heights = wp.array(values.reshape(-1), dtype=float, device=self.device)
        out = wp.empty(6, dtype=int, device=self.device)
        wp.launch(check_cells, dim=6, inputs=[lower, heights, out], device=self.device)
        np.testing.assert_array_equal(out.numpy(), [0, 1, 0, 0, 0, 1])
        values[5, 3] = 0.7
        heights.assign(values.reshape(-1))
        wp.launch(check_cells, dim=6, inputs=[lower, heights, out], device=self.device)
        self.assertEqual(int(out.numpy()[5]), 0)

    def test_speculative_current_search_gap_is_retained(self):
        """Do not cull an approaching shape using the smaller base contact shell."""
        model = make_model(self.device, z=0.15)
        old, _, a = collide(model, False, speculative=True)
        new, _, b = collide(model, True, speculative=True)
        self.assertGreater(float(new._shape_search_gap.numpy().max()), float(model.shape_gap.numpy().max()))
        self.assertGreater(len(a[0]), 0)
        self.assertGreater(len(b[0]), 0)
        self._assert_geometry_coverage(a, b)
        self.assertEqual(
            int(old.narrow_phase.triangle_pairs_count.numpy()[0]), int(new.narrow_phase.triangle_pairs_count.numpy()[0])
        )
        self.assertAlmostEqual(float(a[0].min()), float(b[0].min()), delta=2e-5)

    def test_mixed_meshes_fallback_and_invalid_flag(self):
        """Keep the original mixed-scene owner and reject misspelled feature values."""
        with patch.dict(os.environ, {"NEWTON_HEIGHTFIELD_CELL_REJECT": "1"}):
            narrow = NarrowPhase(
                max_candidate_pairs=4,
                max_triangle_pairs=16,
                device=self.device,
                has_meshes=True,
                has_heightfields=True,
                reduce_contacts=False,
            )
        self.assertTrue(narrow._heightfield_cell_reject_requested)
        self.assertFalse(narrow._heightfield_cell_reject)
        with patch.dict(os.environ, {"NEWTON_HEIGHTFIELD_CELL_REJECT": "yes"}), self.assertRaises(ValueError):
            newton.CollisionPipeline(make_model(self.device))

    def test_scaled_heightfield_and_full_margin_gap(self):
        """Use baked nonuniform heightfield scale and both endpoint detection shells."""
        model = make_model(self.device, z=0.12, margin=0.025, gap=0.05, heightfield_scale=(1.3, 0.7, 2.0))
        old, _, a = collide(model, False)
        new, _, b = collide(model, True)
        self.assertGreater(len(a[0]), 0)
        self.assertGreater(len(b[0]), 0)
        self._assert_geometry_coverage(a, b)
        self.assertEqual(
            int(old.narrow_phase.triangle_pairs_count.numpy()[0]), int(new.narrow_phase.triangle_pairs_count.numpy()[0])
        )
        self.assertAlmostEqual(float(a[0].min()), float(b[0].min()), delta=2e-5)

    def test_original_aabb_source_recovery(self):
        """Preserve every original range operation; the triangle writes follow each midphase's own append."""

        def function(module, name):
            return next(
                n
                for n in ast.walk(ast.parse(inspect.getsource(module)))
                if isinstance(n, ast.FunctionDef) and n.name == name
            )

        old = function(original_heightfield, "heightfield_vs_convex_midphase")
        new = function(heightfield_cells, "_heightfield_cell_midphase")

        def bounds(fn):
            statements = []
            for node in fn.body:
                if isinstance(node, ast.Assign):
                    statements.append(ast.dump(node))
                    if isinstance(node.targets[0], ast.Name) and node.targets[0].id == "cols":
                        break
            return statements

        self.assertEqual(bounds(old), bounds(new))


@unittest.skipUnless(wp.is_cuda_available(), "CUDA device unavailable")
class TestHeightfieldCellRejectRoutesCUDA(TestHeightfieldCellRejectRoutes):
    device = "cuda:0"


if __name__ == "__main__":
    unittest.main()
