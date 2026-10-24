# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the growable rigid contact pool: policy, growth, overflow, capture, and its consumers."""

import contextlib
import unittest
import warnings
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.core import contact_pool
from newton._src.core.contact_pool import ContactField, ContactPool, grown_rows, initial_rows
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices

WORLDS = 4
BOXES = 3
# Four contacts per box on the ground.
CONTACTS = 4 * WORLDS * BOXES


def _small_pool_policy(rows_per_world: int, quantum: int):
    """Shrink the pool policy so a small scene starts below its bound."""
    return mock.patch.multiple(contact_pool, READY_ROWS_PER_WORLD=rows_per_world, ROW_QUANTUM=quantum)


def _build_boxes(device, worlds=WORLDS, boxes=BOXES, *, color=False):
    """Build ``worlds`` worlds of ``boxes`` free boxes resting on one global ground plane."""
    template = newton.ModelBuilder(up_axis=newton.Axis.Z)
    for i in range(boxes):
        body = template.add_body(xform=wp.transform(wp.vec3(1.0 * i, 0.0, 0.1), wp.quat_identity()))
        template.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.add_ground_plane()
    builder.replicate(template, worlds, spacing=(4.0, 4.0, 0.0))
    if color:
        builder.color()
    return builder.finalize(device=device)


def _records(contacts, *extra):
    """Return the live rigid contacts, with ``extra`` per-contact arrays, as one sorted float table."""
    count = int(contacts.rigid_contact_count.numpy()[0])
    arrays = (
        contacts.rigid_contact_shape0,
        contacts.rigid_contact_shape1,
        contacts.rigid_contact_point0,
        contacts.rigid_contact_point1,
        contacts.rigid_contact_normal,
        *extra,
    )
    table = np.concatenate([array.numpy()[:count].reshape(count, -1).astype(np.float64) for array in arrays], axis=1)
    return table[np.lexsort(table.T[::-1])]


class TestContactPoolPolicy(unittest.TestCase):
    def test_initial_rows_follow_worlds_and_bound(self):
        """Start at 64 rows per world, rounded to the row quantum and at most the bound."""
        self.assertEqual(initial_rows(25_231_360, 16384), 1_048_576)
        self.assertEqual(initial_rows(4_000_000, 4096), 524_288)
        self.assertEqual(initial_rows(589_824, 16384), 589_824)
        self.assertEqual(initial_rows(1000, 1), 1000)

    def test_growth_doubles_the_high_water_past_half_of_ready(self):
        """Grow to twice the high-water mark past half the ready rows, rounded and bounded; never shrink."""
        ready, bound = 1_048_576, 25_231_360
        self.assertEqual(grown_rows(0, ready, bound), ready)
        self.assertEqual(grown_rows(ready // 2, ready, bound), ready)
        self.assertEqual(grown_rows(ready // 2 + 1, ready, bound), 1_572_864)
        self.assertEqual(grown_rows(3 * ready, ready, bound), 6_291_456)
        self.assertEqual(grown_rows(20_000_000, ready, bound), bound)
        with _small_pool_policy(64, 8):
            self.assertEqual(grown_rows(48, 64, 1000), 96)


def test_pool_rows_keep_contents_and_fill_new_rows(test, device):
    """Keep every field's first rows through growth and give new and scratch rows their fill.

    Growing past one 2 MiB granule of every field maps new pages on CUDA.
    """
    grown = 300_000
    pool = ContactPool(1_000_000, device=device, rows=100)
    fields = (
        ContactField("index", wp.int32, -1),
        ContactField("point", wp.vec3),
        ContactField("key", wp.int64, 0x7FFFFFFFFFFFFFFF, per_contact=2),
        ContactField("slot", wp.int32, None),
        ContactField("force", wp.spatial_vector, extra=7),
    )
    target = type("Target", (), {})()
    published = []

    def on_rows(rows):
        rows.publish(target)
        published.append(rows.rows)

    rows = pool.rows(fields, on_rows=on_rows)
    test.assertEqual(published, [100])
    test.assertEqual(target.key.shape, (200,))
    test.assertEqual(target.force.shape, (107,))
    np.testing.assert_array_equal(target.slot.numpy(), np.arange(100))
    test.assertTrue(np.all(target.index.numpy() == -1))
    target.index.assign(np.arange(100, dtype=np.int32))
    keys = np.full(200, 5, dtype=np.int64)
    target.key.assign(keys)
    old_index = target.index
    test.assertTrue(pool.grow(grown))
    test.assertEqual(published, [100, grown])
    test.assertEqual(rows.rows, grown)
    test.assertEqual(pool.ready, grown)
    test.assertEqual(pool.growth_count, 1)
    index = target.index.numpy()
    np.testing.assert_array_equal(index[:100], np.arange(100))
    test.assertTrue(np.all(index[100:] == -1))
    key = target.key.numpy()
    test.assertEqual(key.shape, (2 * grown,))
    test.assertTrue(np.all(key[:100] == 5))
    test.assertTrue(np.all(key[100:] == 0x7FFFFFFFFFFFFFFF), "scratch and new rows take the fill")
    np.testing.assert_array_equal(target.slot.numpy(), np.arange(grown))
    test.assertEqual(target.force.shape, (grown + 7,))
    test.assertTrue(np.all(target.point.numpy() == 0.0))
    test.assertFalse(pool.grow(10))
    if wp.get_device(device).is_cuda:
        # Virtual memory keeps addresses: the old view is a valid prefix of the new one.
        test.assertEqual(old_index.ptr, target.index.ptr)
        np.testing.assert_array_equal(old_index.numpy(), np.arange(100))
    with test.assertRaises(ValueError):
        pool.grow(2_000_000)


def test_pool_follower_grows_and_respects_its_bound(test, device):
    """Grow a following pool with its leader, and refuse to grow it past its bound."""
    leader = ContactPool(10_000, device=device, rows=100)
    follower = ContactPool(10_000, device=device, rows=50)
    target = type("Target", (), {})()
    # Owners hold their rows; the pool holds them weakly.
    rows = follower.rows((ContactField("value", wp.float32, 1.0),), on_rows=lambda rows: rows.publish(target))
    leader.follow(follower)
    test.assertEqual(follower.ready, 100)
    test.assertEqual(target.value.shape, (100,))
    leader.grow(400)
    test.assertEqual(follower.ready, 400)
    test.assertEqual(rows.rows, 400)
    test.assertTrue(np.all(target.value.numpy() == 1.0))
    small = ContactPool(200, device=device, rows=200)
    with test.assertRaises(ValueError):
        leader.follow(small)


def test_pipeline_grows_after_overflow_and_warns(test, device):
    """Flag an overflowing collision pass, grow on service and warn, then match a fixed pipeline."""
    model = _build_boxes(device)
    fixed = newton.CollisionPipeline(model, deterministic=True)
    test.assertFalse(fixed._contact_pool.growable)
    expected = fixed.contacts()
    state = model.state()
    fixed.collide(state, expected)
    test.assertEqual(int(expected.rigid_contact_count.numpy()[0]), CONTACTS)
    with _small_pool_policy(1, 8):
        _pipeline_grows_after_overflow(test, device, model, state, expected)


def _pipeline_grows_after_overflow(test, device, model, state, expected):
    fixed_bound = expected.rigid_contact_max
    pipeline = newton.CollisionPipeline(model, deterministic=True)
    contacts = pipeline.contacts()
    test.assertEqual(contacts.rigid_contact_max, 8)
    test.assertEqual(pipeline.rigid_contact_ready, 8)
    test.assertEqual(pipeline.rigid_contact_max, fixed_bound)
    pipeline.collide(state, contacts)
    test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), CONTACTS, "the count keeps the demand")
    test.assertTrue(pipeline.narrow_phase.buffer_capacity_status()["contacts"])
    with test.assertRaisesRegex(RuntimeError, "contacts"):
        pipeline.narrow_phase.check_buffer_capacity()
    wp.synchronize_device(device)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        test.assertTrue(pipeline.service_capacity())
    test.assertTrue(any("reached the 8 ready rows" in str(w.message) for w in caught))
    test.assertEqual(contacts.rigid_contact_max, 2 * CONTACTS)
    test.assertEqual(contacts.rigid_contact_shape0.shape, (2 * CONTACTS,))
    test.assertEqual(pipeline._sort_key_array.shape, (2 * CONTACTS,))
    test.assertTrue(pipeline.narrow_phase.buffer_capacity_status(clear=True)["contacts"], "the flag stays")
    pipeline.collide(state, contacts)
    test.assertFalse(pipeline.narrow_phase.buffer_capacity_status()["contacts"])
    np.testing.assert_array_equal(_records(contacts), _records(expected))
    wp.synchronize_device(device)
    test.assertFalse(pipeline.service_capacity(), "half the ready rows or less: no growth")


def test_pipeline_growth_without_overflow_matches_fixed(test, device):
    """Grow past half the ready rows without dropping contacts; every pass equals the fixed pipeline's."""
    model = _build_boxes(device)
    fixed = newton.CollisionPipeline(model)
    expected = fixed.contacts()
    with _small_pool_policy(16, 8):
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        test.assertEqual(contacts.rigid_contact_max, 64)
        _collide_like_fixed(test, device, model, fixed, expected, pipeline, contacts)
    test.assertEqual(contacts.rigid_contact_max, 96)


def _collide_like_fixed(test, device, model, fixed, expected, pipeline, contacts):
    state = model.state()
    for step in range(3):
        fixed.collide(state, expected)
        pipeline.collide(state, contacts)
        np.testing.assert_array_equal(_records(contacts), _records(expected))
        test.assertFalse(pipeline.narrow_phase.buffer_capacity_status()["contacts"])
        wp.synchronize_device(device)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            test.assertEqual(pipeline.service_capacity(), step == 0)


def _feather_pgs_run(device, steps, *, policy=None, warmstart=False, after_collide=None):
    """Step boxes on the ground with FeatherPGS; return the final state, contacts, pipeline and solver."""
    model = _build_boxes(device)
    matching = {"contact_matching": "latest"} if warmstart else {}
    with policy if policy is not None else contextlib.nullcontext():
        pipeline = newton.CollisionPipeline(model, **matching)
        solver = newton.solvers.SolverFeatherPGS(model, pgs_iterations=8, pgs_warmstart=warmstart)
        contacts = pipeline.contacts()
        state_0, state_1, control = model.state(), model.state(), model.control()
        for step in range(steps):
            pipeline.collide(state_0, contacts)
            if after_collide is not None:
                after_collide(step, pipeline, contacts)
            state_0.clear_forces()
            solver.step(state_0, state_1, control, contacts, 1.0 / 120.0)
            state_0, state_1 = state_1, state_0
            wp.synchronize_device(device)
            pipeline.service_capacity()
    return state_0, contacts, pipeline, solver


def test_feather_pgs_grows_mid_run(test, device):
    """Grow FeatherPGS's per-contact buffers with the pipeline mid-run and keep the physics of a fixed run."""
    reference, _, _, _ = _feather_pgs_run(device, 6)
    state, contacts, pipeline, solver = _feather_pgs_run(device, 6, policy=_small_pool_policy(16, 8))
    test.assertEqual(pipeline._contact_pool.growth_count, 1)
    test.assertEqual(solver._contact_pool.ready, 96)
    test.assertEqual(solver.contact_slot.shape, (96,))
    test.assertEqual(solver._friction_patches.capacity, 96)
    test.assertEqual(contacts.rigid_contact_max, 96)
    solver.check_constraint_capacity()
    np.testing.assert_allclose(state.body_q.numpy(), reference.body_q.numpy(), rtol=0.0, atol=1.0e-5)
    np.testing.assert_allclose(state.body_qd.numpy(), reference.body_qd.numpy(), rtol=0.0, atol=1.0e-4)


def test_feather_pgs_overflow_is_flagged_and_raised(test, device):
    """Flag the overflowing step in the solver's capacity status, then grow and keep stepping."""
    state, _, _, solver = _feather_pgs_run(device, 1, policy=_small_pool_policy(1, 8))
    with test.assertRaisesRegex(RuntimeError, "capacity exceeded"):
        solver.check_constraint_capacity()
    test.assertEqual(solver._contact_pool.ready, 2 * CONTACTS)
    test.assertTrue(np.all(np.isfinite(state.body_q.numpy())))


def test_matching_and_warm_start_survive_growth(test, device):
    """Match the frame after a growth to the frame before it, so warm starts keep their history."""
    matched = {}

    def after_collide(step, pipeline, contacts):
        count = int(contacts.rigid_contact_count.numpy()[0])
        matched[step] = (pipeline._contact_pool.growth_count, contacts.rigid_contact_match_index.numpy()[:count])

    _feather_pgs_run(device, 3, policy=_small_pool_policy(16, 8), warmstart=True, after_collide=after_collide)
    growths, match = matched[1]
    test.assertEqual(growths, 1, "the pool grew after the first step")
    test.assertEqual(match.shape, (CONTACTS,))
    test.assertTrue(np.all(match >= 0), f"resting contacts must match across growth: {match}")


def test_graph_capture_recapture_after_growth(test, device):
    """Replay a graph captured before a growth safely, then recapture to use the new rows."""
    with _small_pool_policy(1, 8):
        _graph_capture_recapture_after_growth(test, device)


def _graph_capture_recapture_after_growth(test, device):
    model = _build_boxes(device)
    pipeline = newton.CollisionPipeline(model)
    solver = newton.solvers.SolverFeatherPGS(model, pgs_iterations=8)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()

    def simulate():
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, 1.0 / 120.0)
        wp.copy(state_0.body_q, state_1.body_q)
        wp.copy(state_0.body_qd, state_1.body_qd)

    def capture_step():
        # FeatherPGS seeds its double-buffer events in each capture, after the queued clears finish.
        wp.synchronize_device(device)
        with wp.ScopedCapture(device=device) as capture:
            solver.seed_double_buffer_events()
            simulate()
        return capture.graph

    simulate()  # warm up outside capture: module loads and lazy setup
    old_graph = capture_step()
    wp.capture_launch(old_graph)
    wp.synchronize_device(device)
    test.assertTrue(pipeline.narrow_phase.buffer_capacity_status(clear=True)["contacts"])
    test.assertTrue(pipeline.service_capacity())
    test.assertEqual(contacts.rigid_contact_max, 2 * CONTACTS)
    # A stale replay stays within mapped memory at its old extents: still flagged, still finite.
    wp.capture_launch(old_graph)
    wp.synchronize_device(device)
    test.assertTrue(pipeline.narrow_phase.buffer_capacity_status(clear=True)["contacts"])
    graph = capture_step()
    del old_graph
    for _ in range(3):
        wp.capture_launch(graph)
    wp.synchronize_device(device)
    test.assertFalse(pipeline.narrow_phase.buffer_capacity_status()["contacts"])
    test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), CONTACTS)
    test.assertFalse(pipeline.service_capacity())
    test.assertTrue(np.all(np.isfinite(state_0.body_q.numpy())))


def test_reserve_grows_now_or_at_the_next_service(test, device):
    """Grow to a reservation outside capture; defer it, with a warning, during capture; report each growth once."""
    with _small_pool_policy(1, 8):
        pool = ContactPool(1000, device=device, rows=8)
        target = type("Target", (), {})()
        rows = pool.rows((ContactField("value", wp.int32, -1),), on_rows=lambda rows: rows.publish(target))
        test.assertFalse(pool.reserve(8))
        test.assertTrue(pool.reserve(100))
        test.assertEqual(pool.ready, 104, "rounded up to the row quantum")
        test.assertEqual(rows.rows, 104)
        test.assertTrue(pool.service(), "a growth outside the service is reported once")
        test.assertFalse(pool.service())
        with test.assertRaisesRegex(ValueError, "cannot hold"):
            pool.reserve(1001)
        if not wp.get_device(device).is_cuda:
            return
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with wp.ScopedCapture(device=device):
                test.assertFalse(pool.reserve(300))
        test.assertTrue(any("reserved during CUDA graph capture" in str(w.message) for w in caught))
        test.assertEqual(pool.ready, 104, "no growth during capture")
        test.assertTrue(pool.service())
        test.assertEqual(pool.ready, 304)
        test.assertEqual(target.value.shape, (304,))
        test.assertTrue(np.all(target.value.numpy() == -1))
        test.assertFalse(pool.service())


def _mujoco(model, **kwargs):
    """Construct SolverMuJoCo for the boxes, with room for 32 contacts per world."""
    try:
        return newton.solvers.SolverMuJoCo(model, nconmax=32, njmax=192, iterations=20, ls_iterations=10, **kwargs)
    except ImportError as error:
        raise unittest.SkipTest(f"MuJoCo Warp is not available: {error}") from error


def _mujoco_records(contacts):
    """Return the live contacts MuJoCo wrote, with their forces, as one sorted table."""
    return _records(contacts, contacts.force)


def test_mujoco_contacts_grow_a_pipeline_buffer(test, device):
    """Write MJWarp's own contacts into a pipeline buffer below naconmax: grow it eagerly, or after a capture.

    MJWarp -> Newton: ``update_contacts`` declares ``naconmax`` rows. Outside capture the buffer grows at once
    and holds what a fixed ``naconmax`` buffer holds; during capture the replay keeps the count, drops the rows
    past the ready ones and flags it, and the service grows the buffer for the recapture.
    """
    model = _build_boxes(device)
    model.request_contact_attributes("force")
    solver = _mujoco(model, use_mujoco_contacts=True)
    naconmax = solver.mjw_data.naconmax
    reference = newton.Contacts(naconmax, 0, device=device, requested_attributes={"force"})
    with _small_pool_policy(1, 8):
        _mujoco_contacts_grow_a_pipeline_buffer(test, device, model, solver, reference)


def _mujoco_contacts_grow_a_pipeline_buffer(test, device, model, solver, reference):
    naconmax = solver.mjw_data.naconmax
    state_0, state_1, control = model.state(), model.state(), model.control()
    for _ in range(3):
        solver.step(state_0, state_1, control, None, 1.0 / 240.0)
        state_0, state_1 = state_1, state_0
    nacon = int(solver.mjw_data.nacon.numpy()[0])
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=4 * naconmax)
    contacts = pipeline.contacts()
    test.assertEqual(contacts.rigid_contact_max, 8)
    test.assertGreater(naconmax, 8)
    test.assertGreater(nacon, 8, "MJWarp must hold more contacts than the initial rows")
    solver.update_contacts(reference, state_0)
    solver.update_contacts(contacts, state_0)
    test.assertEqual(contacts.rigid_contact_max, naconmax)
    test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), nacon)
    np.testing.assert_array_equal(_mujoco_records(contacts), _mujoco_records(reference))
    test.assertFalse(any(solver.contact_capacity_status().values()))
    wp.synchronize_device(device)
    test.assertTrue(pipeline.service_capacity(), "the growth needs graphs recaptured")
    test.assertFalse(pipeline.service_capacity())

    if not wp.get_device(device).is_cuda:
        return
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=4 * naconmax)
    contacts = pipeline.contacts()

    def simulate():
        solver.step(state_0, state_1, control, None, 1.0 / 240.0)
        solver.update_contacts(contacts, state_1)
        state_0.assign(state_1)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with wp.ScopedCapture(device=device) as capture:
            simulate()
    test.assertTrue(any("reserved during CUDA graph capture" in str(w.message) for w in caught))
    test.assertEqual(contacts.rigid_contact_max, 8)
    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    nacon = int(solver.mjw_data.nacon.numpy()[0])
    test.assertGreater(nacon, 8)
    test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), nacon, "the count keeps MJWarp's count")
    test.assertTrue(solver.contact_capacity_status(clear=True)["source_contacts"])
    test.assertTrue(pipeline.service_capacity())
    test.assertEqual(contacts.rigid_contact_max, naconmax)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with wp.ScopedCapture(device=device) as capture:
            simulate()
    test.assertFalse(any("reserved during CUDA graph capture" in str(w.message) for w in caught))
    for _ in range(2):
        wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    test.assertFalse(any(solver.contact_capacity_status().values()))
    solver.update_contacts(reference, state_1)
    np.testing.assert_array_equal(_mujoco_records(contacts), _mujoco_records(reference))
    test.assertFalse(pipeline.service_capacity())


def test_newton_contacts_feed_mujoco_below_naconmax(test, device):
    """Feed MJWarp from a pipeline buffer below naconmax and write its contacts back without a growth.

    Newton -> MJWarp: the buffer grows from its demand only; ``update_contacts`` declares the rows it converted,
    which the buffer already holds, and the run matches one on a fixed buffer, eagerly and under capture.
    """
    model = _build_boxes(device)
    model.request_contact_attributes("force")
    reference = newton.CollisionPipeline(model, rigid_contact_max=512)
    test.assertFalse(reference._contact_pool.growable)
    with _small_pool_policy(16, 8):
        _newton_contacts_feed_mujoco_below_naconmax(test, device, model, reference)


def _newton_contacts_feed_mujoco_below_naconmax(test, device, model, reference):
    runs = [
        (pipeline, pipeline.contacts(), _mujoco(model, use_mujoco_contacts=False), model.state(), model.state())
        for pipeline in (reference, newton.CollisionPipeline(model, rigid_contact_max=512))
    ]
    (_, fixed, _, fixed_state, _), (pipeline, contacts, solver, state, _) = runs
    naconmax = solver.mjw_data.naconmax
    test.assertEqual(contacts.rigid_contact_max, 64)
    test.assertGreater(naconmax, 64)
    control = model.control()
    for step in range(4):
        for run_pipeline, run_contacts, run_solver, state_0, state_1 in runs:
            run_pipeline.collide(state_0, run_contacts)
            run_solver.step(state_0, state_1, control, run_contacts, 1.0 / 240.0)
            run_solver.update_contacts(run_contacts, state_1)
            state_0.assign(state_1)
            wp.synchronize_device(device)
            test.assertEqual(run_pipeline.service_capacity(), run_pipeline is pipeline and step == 0)
    test.assertEqual(contacts.rigid_contact_max, 96, "grown from the demand of 48 contacts, not to naconmax")
    test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), int(fixed.rigid_contact_count.numpy()[0]))
    for _, _, run_solver, _, _ in runs:
        test.assertFalse(any(run_solver.contact_capacity_status().values()))
    np.testing.assert_allclose(state.body_q.numpy(), fixed_state.body_q.numpy(), rtol=0.0, atol=1.0e-5)
    np.testing.assert_allclose(_mujoco_records(contacts), _mujoco_records(fixed), rtol=1.0e-4, atol=1.0e-4)

    if not wp.get_device(device).is_cuda:
        return
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=512)
    contacts = pipeline.contacts()
    state_0 = model.state()
    state_0.assign(state)
    state_1 = model.state()

    def simulate():
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
        solver.update_contacts(contacts, state_1)
        state_0.assign(state_1)

    pipeline.collide(state_0, contacts)  # warm up outside capture
    graphs = []
    for _ in range(2):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with wp.ScopedCapture(device=device) as capture:
                simulate()
        test.assertFalse(any("reserved during CUDA graph capture" in str(w.message) for w in caught))
        graphs.append(capture.graph)
        for _ in range(2):
            wp.capture_launch(graphs[-1])
        wp.synchronize_device(device)
        if not pipeline.service_capacity():
            break
    test.assertEqual(len(graphs), 2, "one growth from demand, then one recapture")
    test.assertEqual(contacts.rigid_contact_max, 96)
    test.assertFalse(any(solver.contact_capacity_status().values()))
    test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), int(solver.mjw_data.nacon.numpy()[0]))
    test.assertTrue(np.all(np.isfinite(state_0.body_q.numpy())))


def test_xpbd_update_contacts_after_growth(test, device):
    """Report the forces of the last step from a buffer that grew since it; rows the step did not solve get zero."""
    model = _build_boxes(device)
    model.request_contact_attributes("force")
    solver = newton.solvers.SolverXPBD(model, iterations=4)
    with _small_pool_policy(1, 8):
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        state_0, state_1, control = model.state(), model.state(), model.control()
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
        solver.update_contacts(contacts)
        forces = contacts.force.numpy()[:8]
        test.assertTrue(np.any(forces != 0.0))
        wp.synchronize_device(device)
        test.assertTrue(pipeline.service_capacity())
        test.assertEqual(contacts.rigid_contact_max, 2 * CONTACTS)
        solver.update_contacts(contacts)
    np.testing.assert_array_equal(contacts.force.numpy()[:8], forces)
    test.assertTrue(np.all(contacts.force.numpy()[8 : 2 * CONTACTS] == 0.0))


def test_kamino_contacts_grow_a_pipeline_buffer(test, device):
    """Write Kamino's own contacts into a pipeline buffer below Kamino's capacity: it grows to hold them."""
    from newton.tests.utils import basics  # noqa: PLC0415

    source = newton.ModelBuilder(up_axis=newton.Axis.Z)
    newton.solvers.SolverKamino.register_custom_attributes(source)
    basics.build_box_on_plane(builder=source, z_offset=0.0)
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    newton.solvers.SolverKamino.register_custom_attributes(builder)
    builder.replicate(source, world_count=3)
    model = builder.finalize(device=device, skip_validation_joints=True)
    solver = newton.solvers.SolverKamino(model, config=newton.solvers.SolverKamino.Config(use_collision_detector=True))
    capacity = solver._contacts_kamino.model_max_contacts_host
    with _small_pool_policy(1, 1):
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        test.assertEqual(pipeline.rigid_contact_max, capacity)
        test.assertLess(contacts.rigid_contact_max, capacity)
        state_0, state_1 = model.state(), model.state()
        solver.step(state_0, state_1, None, None, 1.0 / 240.0)
        solver.update_contacts(contacts, state_1)
    test.assertEqual(contacts.rigid_contact_max, capacity)
    active = int(solver._contacts_kamino.model_active_contacts.numpy()[0])
    test.assertGreater(active, 0)
    test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), active)
    test.assertTrue(np.all(contacts.rigid_contact_shape0.numpy()[:active] >= 0))


def test_vbd_contact_buffers_hold_the_bound(test, device):
    """Size VBD's per-contact buffers by the contacts' bound, so a growth needs no resize (none in a capture)."""
    model = _build_boxes(device, color=True)
    solver = newton.solvers.SolverVBD(model, iterations=2, rigid_compliant_alm=True)
    with _small_pool_policy(1, 8):
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=512)
        contacts = pipeline.contacts()
        state_0, state_1, control = model.state(), model.state(), model.control()
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
        penalty = solver.body_body_contact_penalty_k
        wp.synchronize_device(device)
        test.assertTrue(pipeline.service_capacity())
        test.assertEqual(contacts.rigid_contact_max, 2 * CONTACTS)
        pipeline.collide(state_1, contacts)
        solver.step(state_1, state_0, control, contacts, 1.0 / 240.0)
    test.assertIs(solver.body_body_contact_penalty_k, penalty)
    test.assertEqual(penalty.shape, (512,))
    test.assertEqual(contacts.rigid_contact_bound, 512)
    test.assertTrue(np.all(np.isfinite(state_0.body_q.numpy())))


def test_coupled_entry_contacts_grow_with_their_source(test, device):
    """Grow a coupled entry's filtered buffer with its source's pool, and keep reusing it."""
    from newton.solvers.experimental.coupled import SolverCoupled  # noqa: PLC0415

    model = _build_boxes(device)
    coupled = SolverCoupled(
        model=model,
        entries=[
            SolverCoupled.Entry(
                name="A",
                solver=newton.solvers.SolverSemiImplicit,
                bodies=list(range(model.body_count)),
                shapes=list(range(model.shape_count)),
            )
        ],
    )
    with _small_pool_policy(1, 8):
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        filtered = coupled.entry_contacts("A", contacts)
        test.assertEqual(filtered.rigid_contact_max, 8)
        state = model.state()
        pipeline.collide(state, contacts)
        wp.synchronize_device(device)
        test.assertTrue(pipeline.service_capacity())
    test.assertEqual(contacts.rigid_contact_max, 2 * CONTACTS)
    test.assertEqual(filtered.rigid_contact_max, 2 * CONTACTS)
    pipeline.collide(state, contacts)
    test.assertIs(coupled.entry_contacts("A", contacts), filtered)
    np.testing.assert_array_equal(_records(filtered), _records(contacts))


def test_contact_kinematics_outputs_sized_by_the_bound(test, device):
    """Evaluate contact kinematics into outputs sized by the bound, before and after a growth."""
    model = _build_boxes(device)
    state = model.state()
    with _small_pool_policy(16, 8):
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=512)
        contacts = pipeline.contacts()
        distance = wp.empty(pipeline.rigid_contact_max, dtype=float, device=device)
        pipeline.collide(state, contacts)
        newton.eval_rigid_contact_kinematics(model, state, contacts, out_distance=distance)
        before = np.sort(distance.numpy()[:CONTACTS])
        wp.synchronize_device(device)
        test.assertTrue(pipeline.service_capacity())
        test.assertEqual(contacts.rigid_contact_max, 2 * CONTACTS)
        pipeline.collide(state, contacts)
        newton.eval_rigid_contact_kinematics(model, state, contacts, out_distance=distance)
    np.testing.assert_allclose(np.sort(distance.numpy()[:CONTACTS]), before, rtol=0.0, atol=1.0e-7)
    test.assertEqual(contacts.rigid_contact_bound, pipeline.rigid_contact_max)
    with test.assertRaisesRegex(ValueError, "expected"):
        newton.eval_rigid_contact_kinematics(
            model, state, contacts, out_distance=wp.empty(64, dtype=float, device=device)
        )


class TestContactPool(unittest.TestCase):
    pass


devices = get_test_devices()
cuda_devices = get_cuda_test_devices()
# Overflowing passes print the narrow phase's capacity warning.
for name, function, check_output in (
    ("test_pool_rows_keep_contents_and_fill_new_rows", test_pool_rows_keep_contents_and_fill_new_rows, True),
    ("test_pool_follower_grows_and_respects_its_bound", test_pool_follower_grows_and_respects_its_bound, True),
    ("test_pipeline_grows_after_overflow_and_warns", test_pipeline_grows_after_overflow_and_warns, False),
    ("test_pipeline_growth_without_overflow_matches_fixed", test_pipeline_growth_without_overflow_matches_fixed, True),
    ("test_feather_pgs_grows_mid_run", test_feather_pgs_grows_mid_run, True),
    ("test_feather_pgs_overflow_is_flagged_and_raised", test_feather_pgs_overflow_is_flagged_and_raised, False),
    ("test_matching_and_warm_start_survive_growth", test_matching_and_warm_start_survive_growth, True),
    ("test_reserve_grows_now_or_at_the_next_service", test_reserve_grows_now_or_at_the_next_service, True),
    ("test_mujoco_contacts_grow_a_pipeline_buffer", test_mujoco_contacts_grow_a_pipeline_buffer, False),
    ("test_newton_contacts_feed_mujoco_below_naconmax", test_newton_contacts_feed_mujoco_below_naconmax, True),
    ("test_xpbd_update_contacts_after_growth", test_xpbd_update_contacts_after_growth, False),
    ("test_kamino_contacts_grow_a_pipeline_buffer", test_kamino_contacts_grow_a_pipeline_buffer, True),
    ("test_vbd_contact_buffers_hold_the_bound", test_vbd_contact_buffers_hold_the_bound, False),
    ("test_coupled_entry_contacts_grow_with_their_source", test_coupled_entry_contacts_grow_with_their_source, False),
    ("test_contact_kinematics_outputs_sized_by_the_bound", test_contact_kinematics_outputs_sized_by_the_bound, True),
):
    add_function_test(TestContactPool, name, function, devices=devices, check_output=check_output)
add_function_test(
    TestContactPool,
    "test_graph_capture_recapture_after_growth",
    test_graph_capture_recapture_after_growth,
    devices=cuda_devices,
    check_output=False,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
