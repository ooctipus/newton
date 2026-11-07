# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare whole collision passes of every global-reducer route against recorded outputs.

The archives in ``golden_data/test_contact_reduction_reference`` record these
passes on the CPU with the slot-major reducer of 894a41b7, before each key's
winners moved into one entry-major row. A change to how the reducer stores or
addresses its values must reproduce them bitwise: the same contacts, in the
same order, with the same buffered and key counts.

The fixtures hold many near-tied candidates, so a last-bit change in the
generated geometry can change winners and counts. The archives are therefore
exact only for the platform that recorded them; elsewhere the comparison is
skipped and only platform-independent properties are checked: a repeated
pose reproduces its output after the reducer clears, and no insert fails.

Set ``NEWTON_UPDATE_CONTACT_REDUCTION_REFERENCE=1`` to record the archives
again, only for an intended change in contact selection or output order.
"""

import os
import platform
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.geometry.contact_data import ContactData
from newton._src.geometry.contact_reduction import get_slot
from newton._src.geometry.contact_reduction_global import (
    PREDICTIVE_BIN_ID,
    GlobalContactReducerData,
    make_contact_key,
)
from newton._src.geometry.contact_reduction_hydroelastic import (
    HydroelasticContactReduction,
    HydroelasticReductionConfig,
    export_hydroelastic_contact_to_buffer,
)
from newton._src.geometry.hashtable import hashtable_find_or_insert

_GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "golden_data", "test_contact_reduction_reference")
_UPDATE = os.environ.get("NEWTON_UPDATE_CONTACT_REDUCTION_REFERENCE") == "1"
_PLATFORM = f"{platform.system()}-{platform.machine()}"

_TERRAIN_FLAGS = {
    "NEWTON_HEIGHTFIELD_CELL_REJECT": "1",
    "NEWTON_HEIGHTFIELD_GEOMETRIC_CULL": "1",
    "NEWTON_HEIGHTFIELD_FINITE_QUERY": "0",
    "NEWTON_HEIGHTFIELD_PAIR_CSR": "0",
    # The recorded terrain outputs are the terrain route's: keep it for every heightfield pair.
    "NEWTON_TERRAIN_LAW": "none",
}
_CONTACT_ARRAYS = (
    "rigid_contact_shape0",
    "rigid_contact_shape1",
    "rigid_contact_point0",
    "rigid_contact_point1",
    "rigid_contact_offset0",
    "rigid_contact_offset1",
    "rigid_contact_normal",
    "rigid_contact_margin0",
    "rigid_contact_margin1",
)


def _add_terrain(builder, cfg, *, x=0.0):
    """Add a tilted, sloped heightfield with noise and a bump under the middle bodies."""
    rng = np.random.default_rng(7)
    grid = np.linspace(-1.0, 1.0, 21, dtype=np.float32)
    heights = 0.06 * grid[None, :] + 0.04 * grid[:, None] + rng.uniform(-0.02, 0.02, size=(21, 21))
    heights[10, 10] += 0.08
    terrain = newton.Heightfield(data=heights.astype(np.float32), nrow=21, ncol=21, hx=1.0, hy=1.0)
    rotation = wp.quat_from_axis_angle(wp.normalize(wp.vec3(0.3, 0.8, -0.2)), 0.12)
    builder.add_shape_heightfield(heightfield=terrain, xform=wp.transform((x, 0.0, 0.0), rotation), cfg=cfg)


def _terrain_model():
    """Primitives, a barrel, a hull and a box resting on the terrain."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    cfg = builder.ShapeConfig(margin=0.001, gap=0.01)
    _add_terrain(builder, cfg)
    rng = np.random.default_rng(11)
    kinds = ("sphere", "capsule", "cylinder", "box", "hull", "barrel", "small_sphere", "box")
    for index, kind in enumerate(kinds):
        rotation = wp.quat_from_axis_angle(wp.normalize(wp.vec3(*rng.normal(size=3))), float(rng.uniform(0.0, 1.0)))
        body = builder.add_body(
            xform=wp.transform((-0.75 + 0.3 * (index % 6), -0.5 + 0.45 * (index // 6), 0.03), rotation)
        )
        if kind == "sphere":
            builder.add_shape_sphere(body, radius=0.05, cfg=cfg)
        elif kind == "small_sphere":
            builder.add_shape_sphere(body, radius=0.02, cfg=cfg)
        elif kind == "capsule":
            builder.add_shape_capsule(body, radius=0.03, half_height=0.08, cfg=cfg)
        elif kind == "cylinder":
            builder.add_shape_cylinder(body, radius=0.04, half_height=0.06, cfg=cfg)
        elif kind == "barrel":
            builder.add_shape_cylinder(body, radius=0.04, half_height=0.06, barrel_radius=0.2, cfg=cfg)
        elif kind == "box":
            builder.add_shape_box(body, hx=0.07, hy=0.05, hz=0.04, cfg=cfg)
        else:
            hull = newton.Mesh.create_box(0.06, 0.05, 0.04, duplicate_vertices=False)
            builder.add_shape_convex_hull(body, mesh=hull, cfg=cfg)
    return builder.finalize(device="cpu")


def _mesh_pile_model():
    """Stacked mesh boxes on a plane (mesh-plane and mesh-mesh writers) and two on terrain."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    cfg = builder.ShapeConfig(margin=0.001, gap=0.01)
    builder.add_ground_plane(cfg=cfg)
    _add_terrain(builder, cfg, x=3.0)
    box = newton.Mesh.create_box(0.1, 0.08, 0.06, duplicate_vertices=False)
    rng = np.random.default_rng(5)
    poses = (
        (0.0, 0.0, 0.058),
        (0.15, 0.02, 0.059),
        (0.05, 0.01, 0.17),
        (0.12, -0.03, 0.175),
        (0.08, 0.0, 0.29),
        (3.0, 0.0, 0.07),
        (3.2, 0.1, 0.06),
    )
    for pose in poses:
        rotation = wp.quat_from_axis_angle(wp.normalize(wp.vec3(*rng.normal(size=3))), float(rng.uniform(0.0, 0.3)))
        body = builder.add_body(xform=wp.transform(pose, rotation))
        builder.add_shape_mesh(body, mesh=box, cfg=cfg)
    body = builder.add_body(xform=wp.transform((0.3, 0.0, 0.05), wp.quat_identity()))
    builder.add_shape_sphere(body, radius=0.051, cfg=cfg)
    return builder.finalize(device="cpu")


def _approaching_model():
    """Bodies falling fast onto terrain and a plane, and one mesh onto another.

    Within the collision horizon they come closer than the speculative
    extension but stay beyond their gaps, so the reducer keeps a predictive
    manifold (the reserved bin) for each pair, besides its regular bins.
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    cfg = builder.ShapeConfig(margin=0.001, gap=0.01)
    rng = np.random.default_rng(7)
    grid = np.linspace(-1.0, 1.0, 11, dtype=np.float32)
    heights = 0.06 * grid[None, :] + 0.04 * grid[:, None] + rng.uniform(-0.02, 0.02, size=(11, 11))
    terrain = newton.Heightfield(data=heights.astype(np.float32), nrow=11, ncol=11, hx=1.0, hy=1.0)
    builder.add_shape_heightfield(heightfield=terrain, cfg=cfg)
    builder.add_ground_plane(cfg=cfg)
    for index, kind in enumerate(("box", "hull", "mesh")):
        rotation = wp.quat_from_axis_angle(wp.normalize(wp.vec3(*rng.normal(size=3))), float(rng.uniform(0.0, 0.5)))
        body = builder.add_body(xform=wp.transform((-0.5 + 0.4 * index, 0.1, 0.08), rotation))
        builder.body_qd[body] = (0.0, 0.0, -4.0, 0.0, 0.0, 0.0)
        if kind == "box":
            builder.add_shape_box(body, hx=0.07, hy=0.05, hz=0.04, cfg=cfg)
        else:
            mesh = newton.Mesh.create_box(0.06, 0.05, 0.04, duplicate_vertices=False)
            if kind == "hull":
                builder.add_shape_convex_hull(body, mesh=mesh, cfg=cfg)
            else:
                builder.add_shape_mesh(body, mesh=mesh, cfg=cfg)
    box = newton.Mesh.create_box(0.1, 0.08, 0.06, duplicate_vertices=False)
    for pose, speed in (((3.0, 0.0, 0.058), 0.0), ((3.02, 0.01, 0.19), 4.0)):
        body = builder.add_body(xform=wp.transform(pose, wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.2)))
        builder.body_qd[body] = (0.0, 0.0, -speed, 0.0, 0.0, 0.0)
        builder.add_shape_mesh(body, mesh=box, cfg=cfg)
    return builder.finalize(device="cpu")


def _pair_csr_model():
    """Boxes and spheres resting on a flat heightfield, for the pair-CSR terrain route."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    cfg = builder.ShapeConfig(density=0.0, mu=0.4, margin=0.0025, gap=0.02)
    terrain = newton.Heightfield(data=np.zeros((9, 9), dtype=np.float32), nrow=9, ncol=9, hx=1.0, hy=1.0)
    builder.add_shape_heightfield(heightfield=terrain, cfg=cfg)
    for kind, x in (("box", -0.3), ("sphere", 0.3), ("box", 0.0), ("sphere", -0.6)):
        link = builder.add_link(
            xform=wp.transform(wp.vec3(x, 0.1 * x, 0.054), wp.quat_identity()),
            mass=1.0,
            inertia=wp.mat33(0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01),
            lock_inertia=True,
        )
        builder.add_articulation([builder.add_joint_free(child=link)])
        if kind == "sphere":
            builder.add_shape_sphere(body=link, radius=0.05, cfg=cfg)
        else:
            builder.add_shape_box(body=link, hx=0.08, hy=0.07, hz=0.05, cfg=cfg)
    return builder.finalize(device="cpu")


def _poses(model):
    """Initial, shifted, lifted clear of the ground, then initial again."""
    initial = model.state().body_q.numpy().copy()
    shifted = initial.copy()
    shifted[:, 0] += 0.013
    shifted[:, 2] -= 0.004
    lifted = initial.copy()
    lifted[:, 2] += 5.0
    return initial, shifted, lifted, initial


def _record(out, name, passes):
    """Store one array per field, the passes' outputs concatenated in order."""
    for field in passes[0]:
        out[f"{name}/{field}"] = np.concatenate([record[field] for record in passes])


def _check_repeat(name, first, repeated):
    """Require a repeated input to reproduce its first pass exactly."""
    for field, value in first.items():
        np.testing.assert_array_equal(repeated[field], value, err_msg=f"{name}/{field} of the repeated pass")


def _run_passes(test, out, name, model, flags, *, dt=None, **options):
    """Collide four poses with one pipeline and record each pass's contacts and reducer counts.

    ``reducer_counts`` holds the buffered contacts, active keys and insert
    failures of each pass. The last pass repeats the first pose, so it must
    reproduce the first pass exactly.
    """
    with patch.dict(os.environ, flags):
        pipeline = newton.CollisionPipeline(model, broad_phase="nxn", reduce_contacts=True, **options)
    contacts = pipeline.contacts()
    reducer = pipeline.narrow_phase.global_contact_reducer
    # The model's state keeps the authored body velocities the speculative pass reads.
    state = model.state()
    passes = []
    for body_q in _poses(model):
        state.body_q.assign(body_q)
        pipeline.collide(state, contacts, **({} if dt is None else {"dt": dt}))
        count = int(contacts.rigid_contact_count.numpy()[0])
        record = {array: getattr(contacts, array).numpy()[:count].copy() for array in _CONTACT_ARRAYS}
        record["count"] = np.array([count])
        record["reducer_counts"] = np.array(
            [
                [
                    int(reducer.contact_count.numpy()[0]),
                    int(reducer.hashtable.active_slots.numpy()[reducer.hashtable.capacity]),
                    int(reducer.ht_insert_failures.numpy()[0]),
                ]
            ]
        )
        passes.append(record)
    _check_repeat(name, passes[0], passes[-1])
    test.assertFalse(any(record["reducer_counts"][0, 2] for record in passes), f"{name}: insert failures")
    _record(out, name, passes)
    return pipeline


@wp.struct
class _HydroWriter:
    count: wp.array[int]
    pair: wp.array[wp.vec2i]
    point: wp.array[wp.vec3]
    normal: wp.array[wp.vec3]
    distance: wp.array[float]
    stiffness: wp.array[float]
    friction_scale: wp.array[float]
    key: wp.array[int]


@wp.func
def _write_hydro_contact(contact: ContactData, writer: _HydroWriter, output_index: int):
    index = wp.atomic_add(writer.count, 0, 1)
    if index < writer.point.shape[0]:
        writer.pair[index] = wp.vec2i(contact.shape_a, contact.shape_b)
        writer.point[index] = contact.contact_point_center
        writer.normal[index] = contact.contact_normal_a_to_b
        writer.distance[index] = contact.contact_distance
        writer.stiffness[index] = contact.contact_stiffness
        writer.friction_scale[index] = contact.contact_friction_scale
        writer.key[index] = contact.sort_sub_key


@wp.kernel(enable_backward=False)
def _buffer_hydro_faces(
    data: GlobalContactReducerData,
    pairs: wp.array[wp.vec2i],
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    separations: wp.array[float],
    areas: wp.array[float],
    pressures: wp.array[float],
    deterministic: int,
):
    """Buffer faces and their unreduced aggregates in order, as the SDF generator does."""
    for index in range(pairs.shape[0]):
        pair = pairs[index]
        normal = normals[index]
        separation = separations[index]
        if separation < 0.0 and deterministic == 0:
            key = make_contact_key(pair[0], pair[1], get_slot(normal))
            entry = hashtable_find_or_insert(key, data.ht_keys, data.ht_active_slots)
            if entry >= 0:
                force = areas[index] * pressures[index]
                wp.atomic_add(data.agg_force, entry, force * normal)
                wp.atomic_add(data.weighted_pos_sum, entry, force * points[index])
                wp.atomic_add(data.weight_sum, entry, force)
                wp.atomic_add(data.agg_depth_volume, entry, (areas[index] * (-separation)) * normal)
        export_hydroelastic_contact_to_buffer(
            pair[0], pair[1], points[index], normal, separation, areas[index], pressures[index], index, data
        )


def _hydro_faces():
    """Random penetrating and speculative faces between four stacked shapes."""
    rng = np.random.default_rng(23)
    faces = []
    for pair, z in (((0, 1), 0.0), ((1, 2), 0.2), ((2, 3), 0.4), ((1, 3), 0.3)):
        for _ in range(400):
            point = (rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1), z + rng.uniform(-0.005, 0.005))
            normal = rng.normal(scale=0.15, size=3) + np.array([0.0, 0.0, 1.0])
            normal /= np.linalg.norm(normal)
            separation = rng.uniform(-0.01, 0.006)
            faces.append((pair, point, normal, separation, rng.uniform(1e-5, 4e-4), rng.uniform(0.0, 5e5)))
    return faces


def _run_hydroelastic(out, name, faces, *, deterministic, moment_matching):
    """Reduce and export three passes of buffered faces through the hydroelastic reducer."""
    shapes = 4
    capacity = 4096
    reduction = HydroelasticContactReduction(
        capacity=capacity,
        device="cpu",
        writer_func=_write_hydro_contact,
        config=HydroelasticReductionConfig(moment_matching=moment_matching),
        deterministic=deterministic,
    )
    writer = _HydroWriter()
    writer.count = wp.zeros(1, dtype=int, device="cpu")
    writer.pair = wp.zeros(capacity, dtype=wp.vec2i, device="cpu")
    writer.point = wp.zeros(capacity, dtype=wp.vec3, device="cpu")
    writer.normal = wp.zeros(capacity, dtype=wp.vec3, device="cpu")
    writer.distance = wp.zeros(capacity, dtype=float, device="cpu")
    writer.stiffness = wp.zeros(capacity, dtype=float, device="cpu")
    writer.friction_scale = wp.zeros(capacity, dtype=float, device="cpu")
    writer.key = wp.zeros(capacity, dtype=int, device="cpu")
    k_hydro = wp.full(shapes, 1.0e9, dtype=float, device="cpu")
    transforms = wp.array(
        [wp.transform((0.0, 0.0, 0.1 * i), wp.quat_identity()) for i in range(shapes)], dtype=wp.transform, device="cpu"
    )
    lower = wp.full(shapes, wp.vec3(-0.12, -0.12, -0.12), dtype=wp.vec3, device="cpu")
    upper = wp.full(shapes, wp.vec3(0.12, 0.12, 0.12), dtype=wp.vec3, device="cpu")
    voxels = wp.full(shapes, wp.vec3i(4, 4, 4), dtype=wp.vec3i, device="cpu")
    gaps = wp.full(shapes, 0.01, dtype=float, device="cpu")
    # The second pass drops the first 300 faces and the third repeats the
    # first, so both also check the clear.
    passes = []
    for subset in (faces, faces[300:], faces):
        reduction.clear()
        writer.count.zero_()
        columns = [
            wp.array([face[column] for face in subset], dtype=dtype, device="cpu")
            for column, dtype in enumerate((wp.vec2i, wp.vec3, wp.vec3, float, float, float))
        ]
        wp.launch(
            _buffer_hydro_faces,
            dim=1,
            inputs=[reduction.get_data_struct(), *columns, int(deterministic)],
            device="cpu",
        )
        reduction.reduce_and_export(k_hydro, transforms, lower, upper, voxels, gaps, writer, 64)
        count = int(writer.count.numpy()[0])
        record = {
            field: getattr(writer, field).numpy()[:count].copy()
            for field in ("pair", "point", "normal", "distance", "stiffness", "friction_scale", "key")
        }
        record["count"] = np.array([count])
        record["reducer_counts"] = np.array(
            [
                [
                    int(reduction.contact_count.numpy()[0]),
                    int(reduction.reducer.hashtable.active_slots.numpy()[reduction.reducer.hashtable.capacity]),
                ]
            ]
        )
        passes.append(record)
    _check_repeat(name, passes[0], passes[-1])
    _record(out, name, passes)


class TestContactReductionReference(unittest.TestCase):
    """Whole-pass outputs of each global-reducer route on the CPU."""

    def _check(self, archive, out):
        """Compare recorded passes bitwise, or record them when updating the reference."""
        path = os.path.join(_GOLDEN_DIR, f"{archive}.npz")
        out = {**out, "meta/platform": np.array([_PLATFORM]), "meta/warp": np.array([wp.config.version])}
        if _UPDATE:
            os.makedirs(_GOLDEN_DIR, exist_ok=True)
            np.savez_compressed(path, **out)
            return
        with np.load(path) as reference, self.subTest("recorded outputs"):
            recorded = str(reference["meta/platform"][0])
            if recorded != _PLATFORM:
                self.skipTest(f"recorded on {recorded}; this platform's rounding can change near-tied winners")
            self.assertEqual(sorted(out), sorted(reference.files))
            warp_version = str(reference["meta/warp"][0])
            for key in reference.files:
                if not key.startswith("meta/"):
                    np.testing.assert_array_equal(
                        out[key], reference[key], err_msg=f"{key} (recorded with Warp {warp_version})"
                    )

    def test_terrain_routes(self):
        """Reproduce the terrain routes: typed rows, generic and finite queries, cleanup and determinism."""
        model = _terrain_model()
        options = {"max_triangle_pairs": 8192}
        out = {}
        typed = {**_TERRAIN_FLAGS, "NEWTON_HEIGHTFIELD_PRIMITIVE_QUERY": "1"}
        generic = {
            **_TERRAIN_FLAGS,
            "NEWTON_HEIGHTFIELD_PRIMITIVE_QUERY": "0",
            "NEWTON_HEIGHTFIELD_ANALYTIC_PRIMITIVES": "0",
        }
        pipeline = _run_passes(self, out, "typed", model, typed, **options)
        self.assertTrue(pipeline.narrow_phase._heightfield_primitive_query)
        _run_passes(self, out, "generic", model, generic, **options)
        pipeline = _run_passes(self, out, "cleanup", model, {**typed, "NEWTON_REDUCER_EXPORT_CLEANUP": "1"}, **options)
        self.assertTrue(pipeline.narrow_phase._reducer_export_cleanup)
        _run_passes(self, out, "deterministic", model, typed, deterministic=True, **options)
        _run_passes(self, out, "finite", model, {**_TERRAIN_FLAGS, "NEWTON_HEIGHTFIELD_FINITE_QUERY": "1"}, **options)
        self._check("terrain", out)

    def test_mesh_pile(self):
        """Reproduce the mesh-plane and mesh-mesh writers in both reduction modes."""
        model = _mesh_pile_model()
        options = {"max_triangle_pairs": 16384}
        out = {}
        _run_passes(self, out, "plain", model, _TERRAIN_FLAGS, **options)
        _run_passes(self, out, "deterministic", model, _TERRAIN_FLAGS, deterministic=True, **options)
        self._check("mesh_pile", out)

    def test_speculative(self):
        """Reproduce the regular bins and predictive manifolds of fast-approaching pairs."""
        model = _approaching_model()
        out = {}
        pipeline = _run_passes(
            self,
            out,
            "speculative",
            model,
            {**_TERRAIN_FLAGS, "NEWTON_HEIGHTFIELD_PRIMITIVE_QUERY": "1"},
            dt=0.02,
            max_triangle_pairs=16384,
            speculative_contact_gap_max=0.25,
        )
        # The last pass repeats the first; its keys include predictive manifolds.
        reducer = pipeline.narrow_phase.global_contact_reducer
        active = reducer.hashtable.active_slots.numpy()
        keys = reducer.hashtable.keys.numpy()[active[: active[reducer.hashtable.capacity]]]
        self.assertGreater(int(np.sum((keys >> np.uint64(55)) & np.uint64(0xFF) == PREDICTIVE_BIN_ID)), 0)
        self._check("speculative", out)

    def test_pair_csr(self):
        """Reproduce the pair-CSR terrain route and its fallback export."""
        model = _pair_csr_model()
        flags = {
            "NEWTON_HEIGHTFIELD_PAIR_CSR": "1",
            "NEWTON_HEIGHTFIELD_ADAPTIVE_MANIFOLD": "0",
            "NEWTON_HEIGHTFIELD_CELL_REJECT": "1",
            "NEWTON_HEIGHTFIELD_FINITE_QUERY": "1",
            "NEWTON_HEIGHTFIELD_GEOMETRIC_CULL": "1",
            "NEWTON_TERRAIN_LAW": "none",
        }
        out = {}
        pipeline = _run_passes(self, out, "pair_csr", model, flags, max_triangle_pairs=256)
        self.assertTrue(pipeline.narrow_phase._heightfield_pair_csr)
        self._check("pair_csr", out)

    def test_hydroelastic(self):
        """Reproduce the hydroelastic reducer with and without determinism and moment matching."""
        faces = _hydro_faces()
        out = {}
        for deterministic in (False, True):
            for moment_matching in (False, True):
                name = f"deterministic{int(deterministic)}_moments{int(moment_matching)}"
                _run_hydroelastic(out, name, faces, deterministic=deterministic, moment_matching=moment_matching)
        self._check("hydroelastic", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
