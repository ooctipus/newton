# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for velocity-expanded rigid-contact candidate generation."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.geometry.contact_reduction import get_slot
from newton._src.geometry.contact_reduction_global import (
    EXPORT_REDUCED_CONTACTS_BLOCK_DIM,
    GlobalContactReducer,
    GlobalContactReducerData,
    create_export_reduced_contacts_kernel,
    export_contact_to_buffer,
    reclaim_contact_id,
    reduce_buffered_contacts_speculative_kernel,
    resolve_reduced_contacts_kernel,
)
from newton._src.geometry.narrow_phase import (
    ContactWriterData,
    NarrowPhase,
    create_prepare_convex_pair,
    write_contact_simple,
)
from newton._src.geometry.types import GeoType
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices

_prepare_speculative_convex_pair = create_prepare_convex_pair(
    external_aabb=True,
    speculative=True,
)


@wp.kernel
def _mark_contact_detected(contact_count: wp.array[wp.int32], detected: wp.array[wp.int32]):
    """Persist whether any scheduled collision pass emitted a rigid contact."""
    if contact_count[0] > 0:
        detected[0] = 1


@wp.kernel
def _increment_counter(counter: wp.array[wp.int32]):
    """Increment a scalar callback counter for schedule tests."""
    counter[0] += 1


@wp.kernel
def _measure_rod_clearance(body_q: wp.array[wp.transform], minimum: wp.array[float]):
    corner = wp.tid()
    point = wp.vec3(
        0.5 if (corner & 1) != 0 else -0.5,
        0.04 if (corner & 2) != 0 else -0.04,
        0.04 if (corner & 4) != 0 else -0.04,
    )
    height = wp.transform_point(body_q[0], point)[2]
    wp.atomic_min(minimum, 0, height - 0.02)


@wp.kernel
def _extract_speculative_plane_proxy_scale(
    shape_types: wp.array[wp.int32],
    shape_data: wp.array[wp.vec4],
    shape_transform: wp.array[wp.transform],
    shape_source: wp.array[wp.uint64],
    shape_gap: wp.array[wp.float32],
    shape_collision_radius: wp.array[wp.float32],
    shape_aabb_lower: wp.array[wp.vec3],
    shape_aabb_upper: wp.array[wp.vec3],
    shape_collision_aabb_lower: wp.array[wp.vec3],
    shape_collision_aabb_upper: wp.array[wp.vec3],
    proxy_scale: wp.array[wp.vec3],
    valid_result: wp.array[wp.int32],
):
    valid, query = wp.static(_prepare_speculative_convex_pair)(
        wp.vec2i(0, 1),
        shape_types,
        shape_data,
        shape_transform,
        shape_source,
        shape_gap,
        shape_collision_radius,
        shape_aabb_lower,
        shape_aabb_upper,
        shape_collision_aabb_lower,
        shape_collision_aabb_upper,
    )
    valid_result[0] = int(valid)
    proxy_scale[0] = query.geom_a.scale


@wp.kernel
def _buffer_one_contact(reducer_data: GlobalContactReducerData):
    export_contact_to_buffer(
        0,
        1,
        wp.vec3(0.25, -0.5, 0.75),
        wp.vec3(1.0, 0.0, 0.0),
        -0.01,
        7,
        reducer_data,
    )


@wp.kernel
def _buffer_separated_axial_contact(reducer_data: GlobalContactReducerData):
    """Buffer one separated axial-shape contact produced by a mesh triangle."""
    export_contact_to_buffer(
        0,
        1,
        wp.vec3(0.0),
        wp.vec3(1.0, 0.0, 0.0),
        0.05,
        7,
        reducer_data,
    )


@wp.kernel
def _reclaim_and_allocate_contact_ids(
    reducer_data: GlobalContactReducerData,
    stride: int,
    allocated_ids: wp.array[wp.int32],
):
    """Reclaim one ID per ``stride`` threads while every thread contends for one."""
    tid = wp.tid()
    if tid % stride == 0:
        reclaim_contact_id(tid / stride + 1, reducer_data)
    allocated_ids[tid] = export_contact_to_buffer(
        0,
        1,
        wp.vec3(float(tid), 0.0, 0.0),
        wp.vec3(1.0, 0.0, 0.0),
        0.0,
        tid,
        reducer_data,
    )


def _expected_contended_clearances(clearance_order: int) -> list[float]:
    """Return the nearest clearance in each fingerprint shard plus the impact guard."""
    nearest_by_shard: dict[int, float] = {}
    for candidate_idx in range(255):
        if clearance_order == 0:
            clearance_rank = candidate_idx
        elif clearance_order == 1:
            clearance_rank = 254 - candidate_idx
        else:
            clearance_rank = (candidate_idx * 73) % 255
        clearance = 0.01 + clearance_rank * 0.0001
        fingerprint = (candidate_idx << 2) | ((candidate_idx & 1) << 1)
        shard_hash = fingerprint
        shard_hash ^= shard_hash >> 16
        shard_hash = (shard_hash * 0x7FEB352D) & 0xFFFFFFFF
        shard_hash ^= shard_hash >> 15
        shard_hash = (shard_hash * 0x846CA68B) & 0xFFFFFFFF
        shard_hash ^= shard_hash >> 16
        shard = shard_hash % 6
        nearest_by_shard[shard] = min(clearance, nearest_by_shard.get(shard, float("inf")))
    return sorted([*nearest_by_shard.values(), 0.08])


_export_reduced_contacts = create_export_reduced_contacts_kernel(write_contact_simple)


def _make_predictive_reducer(capacity, device):
    """Create reducer storage for predictive reservation tests."""
    return GlobalContactReducer(capacity=capacity, device=device, enable_contact_reclamation=True)


def _build_spheres(device, velocity: float, separation: float = 0.3, gap: float = 0.0):
    """Build two spheres separated along X with the first sphere moving."""
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    builder.rigid_gap = gap
    body_a = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0)))
    builder.add_shape_sphere(body_a, radius=0.1)
    builder.body_qd[body_a] = (velocity, 0.0, 0.0, 0.0, 0.0, 0.0)
    body_b = builder.add_body(xform=wp.transform(wp.vec3(separation, 0.0, 0.0)))
    builder.add_shape_sphere(body_b, radius=0.1)
    model = builder.finalize(device=device)
    return model, model.state()


def _build_opposing_capsules(device, speed_a: float, speed_b: float, gap: float, margin: float):
    """Build two side-on capsules moving toward one another along X."""
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    builder.rigid_gap = 0.0
    cfg = newton.ModelBuilder.ShapeConfig(gap=gap, margin=margin, mu=0.0)

    body_a = builder.add_body(xform=wp.transform(wp.vec3(-0.5, 0.0, 0.0)))
    builder.add_shape_capsule(body_a, radius=0.05, half_height=0.1, cfg=cfg)
    builder.body_qd[body_a] = (speed_a, 0.0, 0.0, 0.0, 0.0, 0.0)

    body_b = builder.add_body(xform=wp.transform(wp.vec3(0.5, 0.0, 0.0)))
    builder.add_shape_capsule(body_b, radius=0.05, half_height=0.1, cfg=cfg)
    builder.body_qd[body_b] = (-speed_b, 0.0, 0.0, 0.0, 0.0, 0.0)

    return builder.finalize(device=device), body_a, body_b


def _step_opposing_capsules(model, body_a, body_b, frame_dt, substeps, max_speculative_extension):
    """Advance one frame with collision detection before every equal substep."""
    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        speculative_contact_gap_max=max_speculative_extension,
    )
    contacts = pipeline.contacts()
    solver = newton.solvers.SolverXPBD(model, iterations=5)
    state_in = model.state()
    state_out = model.state()
    substep_dt = frame_dt / substeps
    detected_contact = False
    for _ in range(substeps):
        state_in.clear_forces()
        pipeline.collide(state_in, contacts, dt=substep_dt)
        detected_contact |= int(contacts.rigid_contact_count.numpy()[0]) > 0
        solver.step(state_in, state_out, None, contacts, substep_dt)
        state_in, state_out = state_out, state_in

    positions = state_in.body_q.numpy()
    return float(positions[body_a, 0]), float(positions[body_b, 0]), detected_contact


def _collide(model, state, speculative: bool):
    """Run one collision pass and return the populated contact buffer."""
    speculative_contact_gap_max = 0.25 if speculative else None
    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        speculative_contact_gap_max=speculative_contact_gap_max,
    )
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts, dt=0.02)
    return contacts


def _export_reducer_contacts(reducer: GlobalContactReducer, device):
    """Name every slot's winner (a speculative pass's resolve), then export them; return their count and positions."""
    resolve_threads = 64
    wp.launch(
        resolve_reduced_contacts_kernel,
        dim=resolve_threads,
        inputs=[reducer.get_data_struct(), resolve_threads],
        device=device,
    )
    contact_count = wp.zeros(1, dtype=wp.int32, device=device)
    contact_position = wp.zeros(8, dtype=wp.vec3, device=device)
    writer_data = ContactWriterData()
    writer_data.contact_max = 8
    writer_data.contact_count = contact_count
    writer_data.contact_pair = wp.zeros(8, dtype=wp.vec2i, device=device)
    writer_data.contact_position = contact_position
    writer_data.contact_normal = wp.zeros(8, dtype=wp.vec3, device=device)
    writer_data.contact_penetration = wp.zeros(8, dtype=wp.float32, device=device)
    writer_data.contact_tangent = wp.empty(0, dtype=wp.vec3, device=device)
    writer_data.contact_sort_key = wp.empty(0, dtype=wp.int64, device=device)
    shape_gap = wp.full(2, 0.1, dtype=wp.float32, device=device)
    writer_data.shape_gap = shape_gap
    writer_data.shape_transform = wp.empty(0, dtype=wp.transform, device=device)
    writer_data.shape_linear_velocity = wp.empty(0, dtype=wp.vec3, device=device)
    writer_data.shape_angular_velocity = wp.empty(0, dtype=wp.vec3, device=device)
    writer_data.collision_update_dt = 0.0
    writer_data.max_speculative_extension = 0.0
    total_blocks = 128
    wp.launch_tiled(
        _export_reduced_contacts,
        dim=total_blocks,
        inputs=[
            reducer.hashtable.keys,
            reducer.ht_values,
            reducer.hashtable.active_slots,
            reducer.position_depth,
            reducer.normal,
            reducer.shape_pairs,
            reducer.contact_fingerprints,
            reducer.contact_slots,
            wp.zeros(2, dtype=wp.int32, device=device),
            wp.zeros(2, dtype=wp.vec4, device=device),
            shape_gap,
            writer_data,
            total_blocks,
            int(not device.is_cpu),
        ],
        device=device,
        block_dim=EXPORT_REDUCED_CONTACTS_BLOCK_DIM,
    )
    return int(contact_count.numpy()[0]), contact_position.numpy()


def test_speculative_candidates_are_opt_in(test, device):
    """Verify separated approaching shapes only emit a candidate when enabled."""
    model, state = _build_spheres(device, velocity=10.0)
    test.assertEqual(int(_collide(model, state, speculative=False).rigid_contact_count.numpy()[0]), 0)
    test.assertGreater(int(_collide(model, state, speculative=True).rigid_contact_count.numpy()[0]), 0)


def test_speculative_candidates_require_approach(test, device):
    """Verify separated stationary and diverging shapes emit no candidate."""
    for velocity in (0.0, -10.0):
        model, state = _build_spheres(device, velocity=velocity)
        contacts = _collide(model, state, speculative=True)
        test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 0)


def test_speculative_candidates_require_dt(test, device):
    """Require a current horizon and suppress candidates that cannot reach it."""
    model, state = _build_spheres(device, velocity=10.0)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", speculative_contact_gap_max=0.25)

    contacts = pipeline.contacts()
    with test.assertRaisesRegex(ValueError, "dt must be provided"):
        pipeline.collide(state, contacts)

    pipeline.collide(state, contacts, dt=0.02)
    test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)

    pipeline.collide(state, contacts, dt=0.005)
    test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 0)


def test_speculative_gap_uses_larger_fixed_or_velocity_distance(test, device):
    """Use the larger fixed or velocity-based gap without adding them."""
    model, state = _build_spheres(device, velocity=1.0, separation=0.33, gap=0.05)
    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        speculative_contact_gap_max=0.25,
    )

    contacts = pipeline.contacts()
    pipeline.collide(state, contacts, dt=0.05)
    test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 0)

    pipeline.collide(state, contacts, dt=0.15)
    test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)


def test_speculative_candidates_reject_invalid_dt_override(test, device):
    """Reject negative and non-finite per-call speculative horizons."""
    model, state = _build_spheres(device, velocity=10.0)
    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        speculative_contact_gap_max=0.1,
    )
    contacts = pipeline.contacts()
    for dt in (-0.01, float("nan"), float("inf"), float("-inf")):
        with test.subTest(dt=dt), test.assertRaisesRegex(ValueError, "dt must be a non-negative finite number"):
            pipeline.collide(state, contacts, dt=dt)


def test_speculative_candidates_reject_invalid_gap_max(test, device):
    """Reject negative and non-finite speculative gap limits."""
    model, _state = _build_spheres(device, velocity=10.0)
    for value in (-0.01, float("nan"), float("inf"), float("-inf")):
        with (
            test.subTest(value=value),
            test.assertRaisesRegex(
                ValueError, "speculative_contact_gap_max must be a non-negative finite number or None"
            ),
        ):
            newton.CollisionPipeline(model, broad_phase="nxn", speculative_contact_gap_max=value)


def _build_moving_pair(device, velocity):
    """Two separated spheres, 0.2 m apart along x, both moving at ``velocity``."""
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    builder.rigid_gap = 0.0
    body_a = builder.add_body(xform=wp.transform_identity())
    builder.add_shape_sphere(body_a, radius=0.1)
    builder.body_qd[body_a] = (*velocity, 0.0, 0.0, 0.0)
    body_b = builder.add_body(xform=wp.transform(wp.vec3(0.4, 0.0, 0.0)))
    builder.add_shape_sphere(body_b, radius=0.1)
    builder.body_qd[body_b] = (*velocity, 0.0, 0.0, 0.0)
    return builder.finalize(device=device)


def _collide_moving_pair(model, device, broad_phase):
    pipeline = newton.CollisionPipeline(
        model,
        broad_phase=broad_phase,
        shape_pairs_filtered=(
            wp.array([wp.vec2i(0, 1)], dtype=wp.vec2i, device=device) if broad_phase == "explicit" else None
        ),
        speculative_contact_gap_max=0.25,
    )
    contacts = pipeline.contacts()
    pipeline.collide(model.state(), contacts, dt=0.1)
    return int(pipeline.broad_phase_pair_count.numpy()[0]), int(contacts.rigid_contact_count.numpy()[0])


def test_speculative_candidates_reject_tangential_common_motion(test, device):
    """Reject separated shapes whose common motion is across their separation: no stop of either closes it."""
    model = _build_moving_pair(device, (0.0, 20.0, 0.0))
    for broad_phase in ("nxn", "sap", "explicit"):
        with test.subTest(broad_phase=broad_phase):
            pairs, count = _collide_moving_pair(model, device, broad_phase)
            if broad_phase != "explicit":
                test.assertEqual(pairs, 0)
            test.assertEqual(count, 0)


def test_speculative_candidates_cover_a_stopped_leader(test, device):
    """Keep separated shapes whose common motion is along their separation: the leader may stop in the window."""
    model = _build_moving_pair(device, (20.0, 0.0, 0.0))
    for broad_phase in ("nxn", "sap", "explicit"):
        with test.subTest(broad_phase=broad_phase):
            pairs, count = _collide_moving_pair(model, device, broad_phase)
            test.assertEqual(pairs, 1)
            test.assertGreater(count, 0)


def test_speculative_window_holds_a_landing_stack(test, device):
    """A stack dropped as one onto the ground, one collision pass held over eight substeps: the lower box lands
    mid-window and the upper one keeps falling onto it, so the pass before must hold their contact (1.2 mm apart,
    closing speed zero at that pass). FeatherPGS then stops the upper box on the lower one, not 1.9 mm inside it."""
    builder = newton.ModelBuilder()
    builder.rigid_gap = 0.0003
    half = 0.01
    lower = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, half + 0.005)))
    builder.add_shape_box(lower, hx=half, hy=half, hz=half)
    upper = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 3.0 * half + 0.0062)))
    builder.add_shape_box(upper, hx=half, hy=half, hz=half)
    builder.add_ground_plane()
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model, broad_phase="sap", speculative_contact_gap_max=0.015)
    contacts = pipeline.contacts()
    solver = newton.solvers.SolverFeatherPGS(model)
    state_0, state_1 = model.state(), model.state()
    window, substeps = 0.01, 8
    worst = np.inf
    for _ in range(6):
        pipeline.collide(state_0, contacts, dt=window)
        for _ in range(substeps):
            state_0.clear_forces()
            solver.step(state_0, state_1, None, contacts, window / substeps)
            state_0, state_1 = state_1, state_0
            z = state_0.body_q.numpy()[:, 2]
            worst = min(worst, float(z[upper] - z[lower] - 2.0 * half))
    test.assertGreater(worst, -2.0e-4)


def test_speculative_candidates_preserve_physical_geometry(test, device):
    """Verify candidate generation does not enlarge stored physical margins."""
    model, state = _build_spheres(device, velocity=10.0)
    contacts = _collide(model, state, speculative=True)
    test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
    test.assertAlmostEqual(float(contacts.rigid_contact_margin0.numpy()[0]), 0.1, places=6)
    test.assertAlmostEqual(float(contacts.rigid_contact_margin1.numpy()[0]), 0.1, places=6)


def test_speculative_candidates_include_angular_motion(test, device):
    """Verify an offset shape's angular motion expands and filters candidates."""
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    builder.rigid_gap = 0.0
    body = builder.add_body(
        xform=wp.transform_identity(),
        mass=1.0,
        inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        lock_inertia=True,
    )
    builder.add_shape_sphere(body, radius=0.1, xform=wp.transform(wp.vec3(0.0, 1.0, 0.0)))
    builder.body_qd[body] = (0.0, 0.0, 0.0, 0.0, 0.0, -10.0)
    builder.add_shape_sphere(-1, radius=0.1, xform=wp.transform(wp.vec3(0.3, 1.0, 0.0)))
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        speculative_contact_gap_max=0.25,
    )
    contacts = pipeline.contacts()
    pipeline.collide(model.state(), contacts, dt=0.02)
    test.assertGreater(int(pipeline.broad_phase_pair_count.numpy()[0]), 0)
    test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)


def test_speculative_cone_reaches_infinite_plane(test, device):
    """Retain a swept GJK candidate before its current bounds reach an infinite plane."""
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    builder.rigid_gap = 0.0
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.5)))
    builder.add_shape_cone(body, radius=0.1, half_height=0.1)
    builder.body_qd[body] = (0.0, 0.0, -20.0, 0.0, 0.0, 0.0)
    builder.add_shape_plane(width=0.0, length=0.0)
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        speculative_contact_gap_max=0.75,
    )
    contacts = pipeline.contacts()
    pipeline.collide(model.state(), contacts, dt=0.03)

    test.assertGreater(int(pipeline.broad_phase_pair_count.numpy()[0]), 0)
    test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)


def test_speculative_plane_proxy_adds_gap_once(test, device):
    """Size an infinite-plane proxy from the base radius plus one pair gap."""
    shape_types = wp.array([int(GeoType.PLANE), int(GeoType.CONE)], dtype=wp.int32, device=device)
    shape_data = wp.array(
        [wp.vec4(0.0), wp.vec4(0.1, 0.1, 0.0, 0.0)],
        dtype=wp.vec4,
        device=device,
    )
    shape_transform = wp.array(
        [wp.transform_identity(), wp.transform(wp.vec3(0.0, 0.0, 0.5))],
        dtype=wp.transform,
        device=device,
    )
    shape_aabb_lower = wp.array([wp.vec3(0.0), wp.vec3(-0.1, -0.1, 0.4)], dtype=wp.vec3, device=device)
    shape_aabb_upper = wp.array([wp.vec3(0.0), wp.vec3(0.1, 0.1, 0.6)], dtype=wp.vec3, device=device)
    shape_collision_aabb_lower = wp.array(
        [wp.vec3(0.0), wp.vec3(-0.1)],
        dtype=wp.vec3,
        device=device,
    )
    shape_collision_aabb_upper = wp.array(
        [wp.vec3(0.0), wp.vec3(0.1)],
        dtype=wp.vec3,
        device=device,
    )
    proxy_scale = wp.zeros(1, dtype=wp.vec3, device=device)
    valid_result = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(
        _extract_speculative_plane_proxy_scale,
        dim=1,
        inputs=[
            shape_types,
            shape_data,
            shape_transform,
            wp.zeros(2, dtype=wp.uint64, device=device),
            wp.array([0.2, 0.3], dtype=wp.float32, device=device),
            wp.zeros(2, dtype=wp.float32, device=device),
            shape_aabb_lower,
            shape_aabb_upper,
            shape_collision_aabb_lower,
            shape_collision_aabb_upper,
        ],
        outputs=[proxy_scale, valid_result],
        device=device,
    )

    test.assertEqual(int(valid_result.numpy()[0]), 1)
    base_radius = np.linalg.norm([0.1, 0.1, 0.1])
    expected_half_extent = 10.0 * (base_radius + 0.2 + 0.3)
    np.testing.assert_allclose(proxy_scale.numpy()[0], expected_half_extent, rtol=1.0e-6, atol=1.0e-6)


def test_stationary_contacts_match_non_speculative_pipeline(test, device):
    """Match contacts for non-moving shapes with speculative generation on and off."""
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    builder.rigid_gap = 0.0
    body_a = builder.add_body()
    builder.add_shape_box(body_a, hx=0.1, hy=0.1, hz=0.1)
    body_b = builder.add_body(xform=wp.transform(wp.vec3(0.15, 0.0, 0.0)))
    builder.add_shape_box(
        body_b,
        hx=0.1,
        hy=0.1,
        hz=0.1,
    )
    model = builder.finalize(device=device)
    state = model.state()

    pipelines = (
        newton.CollisionPipeline(model, broad_phase="nxn"),
        newton.CollisionPipeline(
            model,
            broad_phase="nxn",
            speculative_contact_gap_max=0.25,
        ),
    )
    outputs = []
    for pipeline in pipelines:
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts, dt=0.03)
        count = int(contacts.rigid_contact_count.numpy()[0])
        test.assertGreater(count, 0)
        outputs.append(
            (
                contacts.rigid_contact_shape0.numpy()[:count],
                contacts.rigid_contact_shape1.numpy()[:count],
                contacts.rigid_contact_point0.numpy()[:count],
                contacts.rigid_contact_point1.numpy()[:count],
                contacts.rigid_contact_normal.numpy()[:count],
                contacts.rigid_contact_margin0.numpy()[:count],
                contacts.rigid_contact_margin1.numpy()[:count],
            )
        )

    for regular, speculative in zip(*outputs, strict=True):
        np.testing.assert_allclose(speculative, regular, rtol=0.0, atol=1.0e-6)


def test_speculative_contacts_prevent_dynamic_tunneling(test, device):
    """Prevent a fast dynamic sphere from crossing an infinite plane in one XPBD step."""
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    builder.rigid_gap = 0.0
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.5)))
    builder.add_shape_sphere(body, radius=0.05)
    builder.body_qd[body] = (0.0, 0.0, -20.0, 0.0, 0.0, 0.0)
    builder.add_shape_plane(width=0.0, length=0.0)
    model = builder.finalize(device=device)
    dt = 0.03

    def step(speculative):
        speculative_contact_gap_max = 0.75 if speculative else None
        pipeline = newton.CollisionPipeline(
            model,
            broad_phase="nxn",
            speculative_contact_gap_max=speculative_contact_gap_max,
        )
        contacts = pipeline.contacts()
        state_in = model.state()
        state_out = model.state()
        pipeline.collide(state_in, contacts, dt=dt)
        newton.solvers.SolverXPBD(model, iterations=5).step(state_in, state_out, None, contacts, dt)
        return float(state_out.body_q.numpy()[body, 2])

    test.assertLess(step(False), 0.0)
    test.assertGreaterEqual(step(True), 0.04)


def test_adaptive_collision_schedule_prevents_capsule_tunneling(test, device, external_capture=False):
    """Refresh collisions often enough to prevent opposing capsules from tunneling."""
    if external_capture and not wp.is_conditional_graph_supported():
        test.skipTest("CUDA graph capture requires conditional graph support")
    frame_dt = 0.01
    substeps = 10
    max_speculative_extension = 0.2
    speed_a = 100.0
    speed_b = 100.0

    fixed_model, fixed_a, fixed_b = _build_opposing_capsules(device, speed_a, speed_b, 0.0, 0.0)
    fixed_x_a, fixed_x_b, fixed_detected = _step_opposing_capsules(
        fixed_model,
        fixed_a,
        fixed_b,
        frame_dt,
        1,
        max_speculative_extension,
    )
    test.assertFalse(fixed_detected)
    test.assertGreater(fixed_x_a, fixed_x_b)

    adaptive_model, adaptive_a, adaptive_b = _build_opposing_capsules(device, speed_a, speed_b, 0.0, 0.0)
    states = (adaptive_model.state(), adaptive_model.state())
    pipeline = newton.CollisionPipeline(
        adaptive_model,
        broad_phase="nxn",
        speculative_contact_gap_max=max_speculative_extension,
    )
    contacts = pipeline.contacts()
    solver = newton.solvers.SolverXPBD(adaptive_model, iterations=5)
    detected = wp.zeros(1, dtype=wp.int32, device=device)
    collision_calls = wp.zeros(1, dtype=wp.int32, device=device)
    substep_calls = wp.zeros(1, dtype=wp.int32, device=device)

    def collide(state, dt):
        state.clear_forces()
        pipeline.collide(state, contacts, dt=dt)
        wp.launch(_increment_counter, dim=1, inputs=[collision_calls], device=device)
        wp.launch(
            _mark_contact_detected,
            dim=1,
            inputs=[contacts.rigid_contact_count, detected],
            device=device,
        )

    def substep(state_in, state_out, dt):
        solver.step(state_in, state_out, None, contacts, dt)
        wp.launch(_increment_counter, dim=1, inputs=[substep_calls], device=device)

    scheduler = newton.CollisionSubstepScheduler(
        pipeline,
        states,
        collision_callback=collide,
        substep_callback=substep,
        frame_dt=frame_dt,
        substeps=substeps,
    )
    if external_capture:
        with wp.ScopedCapture(device=device) as capture:
            scheduler.step()
        wp.capture_launch(capture.graph)
    else:
        scheduler.step()

    test.assertEqual(int(scheduler.interval_overflow.numpy()[0]), 0)
    test.assertGreaterEqual(int(collision_calls.numpy()[0]), 2)
    test.assertLessEqual(int(collision_calls.numpy()[0]), substeps)
    test.assertEqual(int(substep_calls.numpy()[0]), substeps)
    test.assertEqual(int(detected.numpy()[0]), 1)
    positions = states[0].body_q.numpy()
    adaptive_x_a = float(positions[adaptive_a, 0])
    adaptive_x_b = float(positions[adaptive_b, 0])
    test.assertLess(adaptive_x_a, adaptive_x_b)


def test_adaptive_collision_schedule_reacts_to_acceleration(test, device):
    """Refresh collision detection after observed speed grows during the frame."""
    frame_dt = 1.0
    substeps = 10
    travel_budget = 2.0

    builder = newton.ModelBuilder(gravity=wp.vec3(0.0, -10.0, 0.0))
    body = builder.add_body()
    builder.add_shape_sphere(body, radius=0.1)
    model = builder.finalize(device=device)
    states = (model.state(), model.state())
    pipeline = newton.CollisionPipeline(model, speculative_contact_gap_max=travel_budget)
    solver = newton.solvers.SolverXPBD(model, iterations=1)
    collision_calls = wp.zeros(1, dtype=wp.int32, device=device)

    def collide(state, dt):
        del state, dt
        wp.launch(_increment_counter, dim=1, inputs=[collision_calls], device=device)

    def substep(state_in, state_out, dt):
        state_in.clear_forces()
        solver.step(state_in, state_out, None, None, dt)

    scheduler = newton.CollisionSubstepScheduler(
        pipeline,
        states,
        collision_callback=collide,
        substep_callback=substep,
        frame_dt=frame_dt,
        substeps=substeps,
    )
    scheduler.step()

    test.assertGreater(int(collision_calls.numpy()[0]), 1)
    # Refreshing after each observed speed increase keeps travel within budget.
    test.assertEqual(int(scheduler.interval_overflow.numpy()[0]), 0)


def test_adaptive_collision_schedule_limits_refresh_interval(test, device):
    """Refresh collisions at least as often as the configured time limit."""
    frame_dt = 1.0 / 60.0
    substeps = 12

    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    body = builder.add_body()
    builder.add_shape_sphere(body, radius=0.1)
    model = builder.finalize(device=device)
    states = (model.state(), model.state())
    pipeline = newton.CollisionPipeline(model, speculative_contact_gap_max=0.1)
    collision_calls = wp.zeros(1, dtype=wp.int32, device=device)

    def collide(state, dt):
        del state, dt
        wp.launch(_increment_counter, dim=1, inputs=[collision_calls], device=device)

    def substep(state_in, state_out, dt):
        del state_in, state_out, dt

    scheduler = newton.CollisionSubstepScheduler(
        pipeline,
        states,
        collision_callback=collide,
        substep_callback=substep,
        frame_dt=frame_dt,
        substeps=substeps,
        collision_dt_max=1.0 / 120.0,
    )
    scheduler.step()

    test.assertEqual(int(collision_calls.numpy()[0]), 2)


def test_adaptive_collision_schedule_rejects_unsupported_inputs(test, device):
    """Reject particles, aliased states, and undersized rigid-body buffers."""

    def noop(*args):
        del args

    builder = newton.ModelBuilder()
    body = builder.add_body()
    builder.add_shape_sphere(body, radius=0.1)
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model, speculative_contact_gap_max=0.1)
    state = model.state()

    with test.assertRaisesRegex(ValueError, "at least the solver substep duration"):
        newton.CollisionSubstepScheduler(
            pipeline,
            (state, model.state()),
            collision_callback=noop,
            substep_callback=noop,
            frame_dt=0.01,
            substeps=2,
            collision_dt_max=0.004,
        )

    with test.assertRaisesRegex(ValueError, "distinct ping-pong buffers"):
        newton.CollisionSubstepScheduler(
            pipeline,
            (state, state),
            collision_callback=noop,
            substep_callback=noop,
            frame_dt=0.01,
            substeps=2,
        )

    short_state = model.state()
    short_state.body_q = wp.empty(0, dtype=wp.transform, device=device)
    short_state.body_qd = wp.empty(0, dtype=wp.spatial_vector, device=device)
    with test.assertRaisesRegex(ValueError, "must contain all bodies"):
        newton.CollisionSubstepScheduler(
            pipeline,
            (short_state, model.state()),
            collision_callback=noop,
            substep_callback=noop,
            frame_dt=0.01,
            substeps=2,
        )

    particle_builder = newton.ModelBuilder()
    particle_builder.add_particle(pos=wp.vec3(), vel=wp.vec3(), mass=1.0)
    particle_model = particle_builder.finalize(device=device)
    particle_pipeline = newton.CollisionPipeline(particle_model, speculative_contact_gap_max=0.1)
    with test.assertRaisesRegex(ValueError, "rigid contacts only"):
        newton.CollisionSubstepScheduler(
            particle_pipeline,
            (particle_model.state(), particle_model.state()),
            collision_callback=noop,
            substep_callback=noop,
            frame_dt=0.01,
            substeps=2,
        )


def test_speculative_narrow_phase_launch(test, device):
    """Verify the public narrow-phase convenience API performs exact admission."""
    shape_transform = wp.array(
        [wp.transform_identity(), wp.transform(wp.vec3(0.3, 0.0, 0.0))],
        dtype=wp.transform,
        device=device,
    )
    shape_aabb_lower = wp.full(2, wp.vec3(-0.1), dtype=wp.vec3, device=device)
    shape_aabb_upper = wp.full(2, wp.vec3(0.1), dtype=wp.vec3, device=device)
    narrow_phase = NarrowPhase(
        max_candidate_pairs=1,
        reduce_contacts=False,
        device=device,
        shape_aabb_lower=shape_aabb_lower,
        shape_aabb_upper=shape_aabb_upper,
        shape_voxel_resolution=wp.full(2, wp.vec3i(1), dtype=wp.vec3i, device=device),
        has_meshes=False,
        contact_max=4,
        verify_buffers=False,
        speculative=True,
    )

    candidate_pair = wp.array([wp.vec2i(0, 1)], dtype=wp.vec2i, device=device)
    candidate_pair_count = wp.array([1], dtype=wp.int32, device=device)
    shape_types = wp.array([int(GeoType.SPHERE), int(GeoType.SPHERE)], dtype=wp.int32, device=device)
    shape_data = wp.array(
        [wp.vec4(0.1, 0.1, 0.1, 0.0), wp.vec4(0.1, 0.1, 0.1, 0.0)],
        dtype=wp.vec4,
        device=device,
    )
    shape_gap = wp.array([0.2, 0.0], dtype=wp.float32, device=device)
    shape_base_gap = wp.zeros(2, dtype=wp.float32, device=device)
    shape_angular_velocity = wp.zeros(2, dtype=wp.vec3, device=device)

    for velocity, expected_count in ((10.0, 1), (-10.0, 0)):
        contact_count = wp.zeros(1, dtype=wp.int32, device=device)
        narrow_phase.launch(
            candidate_pair=candidate_pair,
            candidate_pair_count=candidate_pair_count,
            shape_types=shape_types,
            shape_data=shape_data,
            shape_transform=shape_transform,
            shape_source=wp.zeros(2, dtype=wp.uint64, device=device),
            shape_sdf_index=wp.full(2, -1, dtype=wp.int32, device=device),
            shape_gap=shape_gap,
            shape_base_gap=shape_base_gap,
            shape_collision_radius=wp.full(2, 0.1, dtype=wp.float32, device=device),
            shape_flags=wp.zeros(2, dtype=wp.int32, device=device),
            shape_collision_aabb_lower=shape_aabb_lower,
            shape_collision_aabb_upper=shape_aabb_upper,
            shape_voxel_resolution=wp.full(2, wp.vec3i(1), dtype=wp.vec3i, device=device),
            contact_pair=wp.zeros(4, dtype=wp.vec2i, device=device),
            contact_position=wp.zeros(4, dtype=wp.vec3, device=device),
            contact_normal=wp.zeros(4, dtype=wp.vec3, device=device),
            contact_penetration=wp.zeros(4, dtype=wp.float32, device=device),
            contact_count=contact_count,
            contact_tangent=wp.empty(0, dtype=wp.vec3, device=device),
            shape_linear_velocity=wp.array([wp.vec3(velocity, 0.0, 0.0), wp.vec3(0.0)], dtype=wp.vec3, device=device),
            shape_angular_velocity=shape_angular_velocity,
            collision_update_dt=0.02,
            max_speculative_extension=0.25,
            device=device,
        )
        test.assertEqual(int(contact_count.numpy()[0]), expected_count)


def test_speculative_narrow_phase_rejects_hydroelastic(test, device):
    """Verify indexed hydroelastic writers cannot silently bypass exact admission."""
    with test.assertRaisesRegex(NotImplementedError, "does not yet support hydroelastic"):
        NarrowPhase(
            max_candidate_pairs=1,
            device=device,
            hydroelastic_sdf=object(),
            speculative=True,
        )


def test_speculative_narrow_phase_rejects_unmarked_custom_writer(test, device):
    """Reject custom writers that do not explicitly implement speculative admission."""
    with test.assertRaisesRegex(ValueError, "contact_writer_supports_speculative=True"):
        NarrowPhase(
            max_candidate_pairs=1,
            device=device,
            contact_writer_warp_func=write_contact_simple,
            speculative=True,
        )


def test_speculative_pipeline_rejects_hydroelastic_before_sdf_construction(test, device):
    """Reject speculative hydroelastic pairs before constructing their SDF pipeline."""
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    body_a = builder.add_body()
    builder.add_shape_sphere(body_a, radius=0.1)
    body_b = builder.add_body(xform=wp.transform(wp.vec3(0.15, 0.0, 0.0)))
    builder.add_shape_sphere(body_b, radius=0.1)
    model = builder.finalize(device=device)
    shape_flags = model.shape_flags.numpy()
    shape_flags |= int(newton.ShapeFlags.HYDROELASTIC)
    model.shape_flags.assign(shape_flags)

    with test.assertRaisesRegex(NotImplementedError, "does not yet support hydroelastic"):
        newton.CollisionPipeline(
            model,
            speculative_contact_gap_max=0.1,
        )


def test_speculative_pipeline_allows_missing_explicit_pairs(test, device):
    """Allow non-explicit speculative pipelines when the model pair array is absent."""
    model, _state = _build_spheres(device, velocity=0.0)
    model.shape_contact_pairs = None
    newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        speculative_contact_gap_max=0.1,
    )


def test_speculative_mesh_sdf_candidates(test, device):
    """Verify separated approaching mesh SDFs retain leading candidates."""
    projectile = newton.Mesh.create_box(0.05, compute_normals=False, compute_uvs=False)
    projectile.build_sdf(device=device, max_resolution=32)
    wall = newton.Mesh.create_box(0.02, 0.3, 0.3, compute_normals=False, compute_uvs=False)
    wall.build_sdf(device=device, max_resolution=32)

    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    builder.rigid_gap = 0.0
    body = builder.add_body(xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0)))
    builder.add_shape_mesh(body, mesh=projectile)
    builder.body_qd[body] = (20.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    builder.add_shape_mesh(-1, mesh=wall)
    model = builder.finalize(device=device)

    contacts = _collide(model, model.state(), speculative=True)
    test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)


def test_speculative_axial_shapes_reach_triangle_mesh(test, device):
    """Retain separated sphere and capsule contacts against a triangle mesh."""
    for shape_type in ("sphere", "capsule"):
        with test.subTest(shape_type=shape_type):
            wall = newton.Mesh.create_box(0.02, 0.3, 0.3, compute_normals=False, compute_uvs=False)
            builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
            builder.rigid_gap = 0.0
            body = builder.add_body(xform=wp.transform(wp.vec3(-0.25, 0.0, 0.0)))
            if shape_type == "sphere":
                builder.add_shape_sphere(body, radius=0.1)
            else:
                builder.add_shape_capsule(body, radius=0.1, half_height=0.1)
            builder.body_qd[body] = (10.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            builder.add_shape_mesh(-1, mesh=wall)
            model = builder.finalize(device=device)

            contacts = _collide(model, model.state(), speculative=True)
            test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)


def test_speculative_mesh_sdf_predicted_manifold(test, device):
    """A separated mesh-SDF pair predicted to touch keeps its manifold in the normal bins and voxels, as a touching
    pair does, not a bounded handful of points."""
    projectile = newton.Mesh.create_sphere(
        0.2,
        num_latitudes=32,
        num_longitudes=32,
        compute_normals=False,
        compute_uvs=False,
    )
    projectile.build_sdf(device=device, max_resolution=64)
    wall = newton.Mesh.create_box(0.02, 0.5, 0.5, compute_normals=False, compute_uvs=False)
    wall.build_sdf(device=device, max_resolution=64)

    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    builder.rigid_gap = 0.0
    body = builder.add_body(xform=wp.transform(wp.vec3(-0.35, 0.0, 0.0)))
    builder.add_shape_mesh(body, mesh=projectile)
    builder.body_qd[body] = (10.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    builder.add_shape_mesh(-1, mesh=wall)
    model = builder.finalize(device=device)

    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        speculative_contact_gap_max=0.25,
    )
    contacts = pipeline.contacts()
    pipeline.collide(model.state(), contacts, dt=0.03)

    count = int(contacts.rigid_contact_count.numpy()[0])
    test.assertGreater(count, 7)


def test_speculative_mesh_sdf_retains_rotating_leading_feature(test, device, scheduled=False):
    """Verify an inner SDF contact cannot hide the rod end rotating toward the board."""
    rod = newton.Mesh.create_box(0.5, 0.04, 0.04, compute_normals=False, compute_uvs=False)
    rod.build_sdf(device=device, max_resolution=64)
    board = newton.Mesh.create_box(0.7, 0.3, 0.02, compute_normals=False, compute_uvs=False)
    board.build_sdf(device=device, max_resolution=64)

    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    builder.rigid_gap = 0.0
    rod_body = builder.add_body(
        xform=wp.transform(
            wp.vec3(0.0, 0.0, 0.09 if scheduled else 0.095),
            wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -0.05 if scheduled else -0.1),
        )
    )
    builder.add_shape_mesh(rod_body, mesh=rod)
    builder.body_qd[rod_body] = (0.0, 0.0, 0.0, 0.0, 10.0, 0.0)
    builder.add_shape_mesh(-1, mesh=board)
    model = builder.finalize(device=device)

    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        speculative_contact_gap_max=0.15,
    )
    contacts = pipeline.contacts()
    states = (model.state(), model.state())
    first_points = []
    if scheduled:
        solver = newton.solvers.SolverXPBD(model, iterations=20)
        minimum = wp.full(1, 1.0, dtype=float, device=device)

        def collide(state, horizon):
            pipeline.collide(state, contacts, dt=horizon)
            if not first_points:
                count = int(contacts.rigid_contact_count.numpy()[0])
                first_points.append(contacts.rigid_contact_point0.numpy()[:count])

        def step(state_in, state_out, dt):
            solver.step(state_in, state_out, None, contacts, dt)
            wp.launch(_measure_rod_clearance, dim=8, inputs=[state_out.body_q, minimum], device=device)

        schedule = newton.CollisionSubstepScheduler(
            pipeline, states, collision_callback=collide, substep_callback=step, frame_dt=0.03, substeps=30
        )
        schedule.step()
        test.assertEqual(int(schedule.interval_overflow.numpy()[0]), 0)
        # Millimeter-scale SDF error is acceptable; losing the leading end
        # allows centimeter-scale penetration even with 1 ms substeps.
        test.assertGreater(float(minimum.numpy()[0]), -0.002)
        rod_points = first_points[0]
    else:
        pipeline.collide(states[0], contacts, dt=0.03)
        count = int(contacts.rigid_contact_count.numpy()[0])
        rod_points = contacts.rigid_contact_point0.numpy()[:count]

    test.assertGreater(len(rod_points), 1)
    test.assertLess(float(rod_points[:, 0].min()), -0.3)
    test.assertGreater(float(rod_points[:, 0].max()), 0.3)


def test_speculative_buffered_reducer_uses_one_based_ids(test, device):
    """Verify buffered speculative reduction processes every real contact and skips reserved ID zero."""
    reducer = GlobalContactReducer(capacity=8, device=device)
    wp.launch(_buffer_one_contact, dim=1, inputs=[reducer.get_data_struct()], device=device)

    shape_transform = wp.array([wp.transform_identity(), wp.transform_identity()], dtype=wp.transform, device=device)
    wp.launch(
        reduce_buffered_contacts_speculative_kernel,
        dim=1,
        inputs=[
            reducer.get_data_struct(),
            wp.full(2, int(GeoType.BOX), dtype=wp.int32, device=device),
            wp.zeros(2, dtype=wp.vec4, device=device),
            wp.zeros(2, dtype=wp.float32, device=device),
            shape_transform,
            wp.zeros(2, dtype=wp.vec3, device=device),
            wp.zeros(2, dtype=wp.vec3, device=device),
            wp.full(2, wp.vec3(-1.0), dtype=wp.vec3, device=device),
            wp.full(2, wp.vec3(1.0), dtype=wp.vec3, device=device),
            wp.full(2, wp.vec3i(1), dtype=wp.vec3i, device=device),
            0.1,
            0.1,
            1,
        ],
        device=device,
    )

    test.assertEqual(int(reducer.contact_count.numpy()[0]), 1)
    exported_count, positions = _export_reducer_contacts(reducer, device)
    test.assertEqual(exported_count, 1)
    for actual, expected in zip(positions[0], (0.25, -0.5, 0.75), strict=True):
        test.assertAlmostEqual(float(actual), expected, places=6)


def test_speculative_buffered_axial_contacts_use_normal_bins(test, device):
    """A separated sphere or capsule mesh contact predicted within the window competes in its normal bin."""
    for shape_type in (GeoType.SPHERE, GeoType.CAPSULE):
        with test.subTest(shape_type=shape_type):
            reducer = GlobalContactReducer(capacity=8, device=device)
            wp.launch(_buffer_separated_axial_contact, dim=1, inputs=[reducer.get_data_struct()], device=device)

            wp.launch(
                reduce_buffered_contacts_speculative_kernel,
                dim=1,
                inputs=[
                    reducer.get_data_struct(),
                    wp.array([int(shape_type), int(GeoType.MESH)], dtype=wp.int32, device=device),
                    wp.array([wp.vec4(0.1, 0.1, 0.1, 0.0), wp.vec4(0.0)], dtype=wp.vec4, device=device),
                    wp.zeros(2, dtype=wp.float32, device=device),
                    wp.array([wp.transform_identity(), wp.transform_identity()], dtype=wp.transform, device=device),
                    wp.array([wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0)], dtype=wp.vec3, device=device),
                    wp.zeros(2, dtype=wp.vec3, device=device),
                    wp.full(2, wp.vec3(-1.0), dtype=wp.vec3, device=device),
                    wp.full(2, wp.vec3(1.0), dtype=wp.vec3, device=device),
                    wp.full(2, wp.vec3i(1), dtype=wp.vec3i, device=device),
                    0.1,
                    0.1,
                    1,
                ],
                device=device,
            )

            # Predicted to touch within the window, it competes in its normal bin like a present contact.
            bins = (reducer.hashtable.keys.numpy() >> np.uint64(55)) & np.uint64(0xFF)
            test.assertIn(get_slot(wp.vec3(1.0, 0.0, 0.0)), bins)


def test_predictive_reducer_reclamation_storage_is_opt_in(test, device):
    """Avoid allocating predictive reservation storage in the default reducer."""
    reducer = GlobalContactReducer(capacity=32, device=device)
    test.assertEqual(reducer.reclaimed_contact_bits.shape[0], 0)
    test.assertEqual(reducer.reclaimed_contact_count.shape[0], 0)


def test_predictive_reducer_reclaims_ids_without_duplicates(test, device):
    """Reuse every concurrently reclaimed contact ID exactly once."""
    # Enough threads to outlast one resident wave, so pops race later reclaims.
    is_cuda = wp.get_device(device).is_cuda
    num_threads = 1 << 17 if is_cuda else 1024
    allocated_ids = wp.empty(num_threads, dtype=wp.int32, device=device)

    # Stride 1 races pops against reclaims behind their scan; stride 8 keeps
    # the pool near empty so failed pops race the few reclaims.
    for stride in (1, 8):
        capacity = num_threads // stride
        reducer = _make_predictive_reducer(capacity, device)
        for trial in range(4 if is_cuda else 1):
            with test.subTest(stride=stride, trial=trial):
                reducer.clear()
                reducer.contact_count.fill_(capacity)
                allocated_ids.fill_(-2)
                wp.launch(
                    _reclaim_and_allocate_contact_ids,
                    dim=num_threads,
                    inputs=[reducer.get_data_struct(), stride, allocated_ids],
                    device=device,
                )

                # Each reclaim precedes a pop on the same thread, so none may remain.
                ids = allocated_ids.numpy()
                np.testing.assert_array_equal(np.sort(ids[ids > 0]), np.arange(1, capacity + 1, dtype=np.int32))
                test.assertEqual(int(np.count_nonzero(ids == -1)), num_threads - capacity)
                test.assertEqual(int(reducer.contact_count.numpy()[0]), capacity)
                test.assertEqual(int(np.count_nonzero(reducer.reclaimed_contact_bits.numpy())), 0)
                test.assertEqual(int(reducer.reclaimed_contact_count.numpy()[0]), 0)


class TestSpeculativeContacts(unittest.TestCase):
    """Test inexpensive speculative-candidate behavior."""


class TestSpeculativeMeshContacts(unittest.TestCase):
    """Test speculative mesh/SDF candidate generation on CUDA."""


for _name, _test in (
    ("test_speculative_candidates_are_opt_in", test_speculative_candidates_are_opt_in),
    ("test_speculative_candidates_require_approach", test_speculative_candidates_require_approach),
    ("test_speculative_candidates_require_dt", test_speculative_candidates_require_dt),
    (
        "test_speculative_gap_uses_larger_fixed_or_velocity_distance",
        test_speculative_gap_uses_larger_fixed_or_velocity_distance,
    ),
    ("test_speculative_candidates_reject_invalid_dt_override", test_speculative_candidates_reject_invalid_dt_override),
    ("test_speculative_candidates_reject_invalid_gap_max", test_speculative_candidates_reject_invalid_gap_max),
    (
        "test_speculative_candidates_reject_tangential_common_motion",
        test_speculative_candidates_reject_tangential_common_motion,
    ),
    ("test_speculative_candidates_cover_a_stopped_leader", test_speculative_candidates_cover_a_stopped_leader),
    ("test_speculative_window_holds_a_landing_stack", test_speculative_window_holds_a_landing_stack),
    ("test_speculative_candidates_preserve_physical_geometry", test_speculative_candidates_preserve_physical_geometry),
    ("test_speculative_candidates_include_angular_motion", test_speculative_candidates_include_angular_motion),
    ("test_speculative_cone_reaches_infinite_plane", test_speculative_cone_reaches_infinite_plane),
    ("test_speculative_plane_proxy_adds_gap_once", test_speculative_plane_proxy_adds_gap_once),
    (
        "test_stationary_contacts_match_non_speculative_pipeline",
        test_stationary_contacts_match_non_speculative_pipeline,
    ),
    ("test_speculative_contacts_prevent_dynamic_tunneling", test_speculative_contacts_prevent_dynamic_tunneling),
    (
        "test_adaptive_collision_schedule_prevents_capsule_tunneling",
        test_adaptive_collision_schedule_prevents_capsule_tunneling,
    ),
    (
        "test_adaptive_collision_schedule_reacts_to_acceleration",
        test_adaptive_collision_schedule_reacts_to_acceleration,
    ),
    (
        "test_adaptive_collision_schedule_limits_refresh_interval",
        test_adaptive_collision_schedule_limits_refresh_interval,
    ),
    (
        "test_adaptive_collision_schedule_rejects_unsupported_inputs",
        test_adaptive_collision_schedule_rejects_unsupported_inputs,
    ),
    ("test_speculative_narrow_phase_launch", test_speculative_narrow_phase_launch),
    (
        "test_speculative_narrow_phase_rejects_hydroelastic",
        test_speculative_narrow_phase_rejects_hydroelastic,
    ),
    (
        "test_speculative_narrow_phase_rejects_unmarked_custom_writer",
        test_speculative_narrow_phase_rejects_unmarked_custom_writer,
    ),
    (
        "test_speculative_pipeline_rejects_hydroelastic_before_sdf_construction",
        test_speculative_pipeline_rejects_hydroelastic_before_sdf_construction,
    ),
    (
        "test_speculative_pipeline_allows_missing_explicit_pairs",
        test_speculative_pipeline_allows_missing_explicit_pairs,
    ),
    (
        "test_predictive_reducer_reclamation_storage_is_opt_in",
        test_predictive_reducer_reclamation_storage_is_opt_in,
    ),
    (
        "test_speculative_buffered_axial_contacts_use_normal_bins",
        test_speculative_buffered_axial_contacts_use_normal_bins,
    ),
    (
        "test_predictive_reducer_reclaims_ids_without_duplicates",
        test_predictive_reducer_reclaims_ids_without_duplicates,
    ),
    ("test_speculative_buffered_reducer_uses_one_based_ids", test_speculative_buffered_reducer_uses_one_based_ids),
):
    add_function_test(TestSpeculativeContacts, _name, _test, devices=get_test_devices())

add_function_test(
    TestSpeculativeContacts,
    "test_adaptive_collision_schedule_external_capture",
    test_adaptive_collision_schedule_prevents_capsule_tunneling,
    devices=get_cuda_test_devices(),
    external_capture=True,
)

add_function_test(
    TestSpeculativeMeshContacts,
    "test_speculative_axial_shapes_reach_triangle_mesh",
    test_speculative_axial_shapes_reach_triangle_mesh,
    devices=get_test_devices(),
)
add_function_test(
    TestSpeculativeMeshContacts,
    "test_speculative_mesh_sdf_candidates",
    test_speculative_mesh_sdf_candidates,
    devices=get_cuda_test_devices(),
)
add_function_test(
    TestSpeculativeMeshContacts,
    "test_speculative_mesh_sdf_predicted_manifold",
    test_speculative_mesh_sdf_predicted_manifold,
    devices=get_cuda_test_devices(),
)
add_function_test(
    TestSpeculativeMeshContacts,
    "test_speculative_mesh_sdf_retains_rotating_leading_feature",
    test_speculative_mesh_sdf_retains_rotating_leading_feature,
    devices=get_cuda_test_devices(),
)
add_function_test(
    TestSpeculativeMeshContacts,
    "test_scheduled_mesh_sdf_retains_rotating_leading_feature",
    test_speculative_mesh_sdf_retains_rotating_leading_feature,
    devices=get_cuda_test_devices(),
    scheduled=True,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
