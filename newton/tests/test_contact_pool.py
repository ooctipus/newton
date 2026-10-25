# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the growable rigid contact pool: policy, growth, overflow, capture, and its consumers."""

import contextlib
import math
import sys
import unittest
import warnings
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.core import contact_pool
from newton._src.core.contact_pool import (
    INDEX_LIMIT,
    ContactField,
    ContactPool,
    device_row_limit,
    grown_rows,
    model_contact_pool,
    sized_rows,
)
from newton._src.sim.collide import _estimate_rigid_contact_max
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices

WORLDS = 4
BOXES = 3
# Four contacts per box on the ground.
CONTACTS = 4 * WORLDS * BOXES
# The rows a pool is sized to for those contacts, at the row quantum of 8 the small pool policy sets.
GROWN = math.ceil(contact_pool.HEADROOM * CONTACTS / 8) * 8


def _small_pool_policy(rows: int, quantum: int):
    """Start new pools at ``rows`` rows and round sizes to ``quantum``, so a small scene starts below its demand."""
    return mock.patch.multiple(contact_pool, MIN_READY_ROWS=rows, ROW_QUANTUM=quantum)


@contextlib.contextmanager
def _without_gpu_components():
    """Make importing ``gpu_components`` fail, as in an environment without the optional package."""
    hidden = dict.fromkeys(
        ("gpu_components", "gpu_components.backing", "gpu_components.fields", "gpu_components.field_data")
    )
    with mock.patch.dict(sys.modules, hidden), mock.patch.object(contact_pool, "_VIRTUAL_MEMORY", []):
        yield


def _without_virtual_memory(function):
    """Run a test function with ``gpu_components`` hidden."""

    def run(test, device):
        with _without_gpu_components():
            function(test, device)

    return run


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
    def test_pools_start_at_the_floor(self):
        """Start every pool at the floor, whatever the model's world count."""
        self.assertEqual(contact_pool.MIN_READY_ROWS, 4096)
        self.assertEqual(ContactPool(device="cpu").ready, 4096)
        model = _build_boxes("cpu", worlds=128, boxes=1)
        self.assertEqual(newton.CollisionPipeline(model).contacts().rigid_contact_max, 4096)

    def test_sizing_holds_the_headroom_per_contact(self):
        """Size for a demand at HEADROOM rows per contact, rounded up and bounded; never shrink."""
        headroom, ready, bound = contact_pool.HEADROOM, 4096, 25_231_360
        self.assertEqual(sized_rows(0, ready, bound), ready)
        self.assertEqual(sized_rows(int(ready / headroom), ready, bound), ready)
        self.assertEqual(sized_rows(100_003, ready, bound), math.ceil(headroom * 100_003 / 64) * 64)
        self.assertEqual(sized_rows(20_000_000, ready, bound), bound)
        self.assertEqual(sized_rows(10, 8 * ready, bound), 8 * ready)

    def test_replay_growth_sizes_the_mark_past_the_threshold(self):
        """Grow from a replay's high-water mark only past GROW_THRESHOLD of the rows, then as for a measured demand."""
        ready, bound = 1_048_576, 25_231_360
        threshold = int(contact_pool.GROW_THRESHOLD * ready)
        self.assertEqual(grown_rows(0, ready, bound), ready)
        self.assertEqual(grown_rows(threshold, ready, bound), ready)
        self.assertEqual(grown_rows(threshold + 1, ready, bound), sized_rows(threshold + 1, ready, bound))
        self.assertEqual(grown_rows(20_000_000, ready, bound), bound)
        # The pool grows again only once the mark rises by HEADROOM * GROW_THRESHOLD.
        grown = grown_rows(threshold + 1, ready, bound)
        self.assertGreater(grown, ready)
        self.assertEqual(grown_rows(int(contact_pool.GROW_THRESHOLD * grown), grown, bound), grown)

    def test_device_row_limit_divides_the_device_memory(self):
        """Bound rows by the int32 index limit everywhere, and by the device memory over the row bytes on CUDA."""
        self.assertEqual(device_row_limit("cpu", 100), INDEX_LIMIT)
        self.assertEqual(device_row_limit("cpu", 100, 1000), 1000)
        for device in get_cuda_test_devices():
            memory = int(wp.get_device(device).total_memory)
            self.assertEqual(device_row_limit(device, 1000), memory // 1000)
            self.assertEqual(device_row_limit(device, 1), INDEX_LIMIT)


def test_pool_rows_keep_contents_and_fill_new_rows(test, device):
    """Keep every field's first rows through growth and give new and scratch rows their fill.

    Growing past one 2 MiB granule of every field maps new pages on CUDA.
    """
    grown = 300_000
    pool = ContactPool(device=device, rows=100)
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
    if rows._mode == "virtual":
        # Virtual memory keeps addresses: the old view is a valid prefix of the new one.
        test.assertEqual(old_index.ptr, target.index.ptr)
    else:
        # Ordinary arrays are reallocated; the superseded ones stay alive for stale graph replays.
        test.assertNotEqual(old_index.ptr, target.index.ptr)
        test.assertTrue(any(array is old_index for array in rows._superseded))
    np.testing.assert_array_equal(old_index.numpy(), np.arange(100))
    with test.assertRaisesRegex(MemoryError, "cannot back"):
        pool.grow(pool.bound + 1)
    test.assertEqual(pool.ready, grown)


def test_pool_bound_is_the_hardware_limit(test, device):
    """Bound a pool by the rows the device could back for its live fields; reservations follow the bound.

    Warp arrays index int32 elements; on CUDA, ``gpu_components`` also views a field's reservation as int32 bytes.
    """
    pool = ContactPool(device=device, rows=64)
    test.assertEqual(pool.bound, INDEX_LIMIT)
    narrow = pool.rows((ContactField("a", wp.int32), ContactField("b", wp.vec3)))
    virtual = narrow._mode == "virtual"
    test.assertEqual(virtual, wp.get_device(device).is_cuda and contact_pool._virtual_memory() is not None)
    narrow_limit = INDEX_LIMIT // 12 if virtual else INDEX_LIMIT
    test.assertEqual(pool.row_bytes, 16)
    test.assertEqual(pool.bound, device_row_limit(device, 16, narrow_limit))
    test.assertEqual(narrow.reserve, pool.bound)
    wide = pool.rows((ContactField("keys", wp.int64, per_contact=2),))
    test.assertEqual(pool.row_bytes, 32)
    test.assertEqual(
        pool.bound, device_row_limit(device, 32, min(narrow_limit, INDEX_LIMIT // (8 if virtual else 1) // 2))
    )
    test.assertEqual(wide.reserve, pool.bound)
    with test.assertRaisesRegex(MemoryError, "cannot back"):
        pool.reserve(pool.bound + 1)
    test.assertEqual(pool.ready, 64)
    # A collected member, such as a measurement's scratch rows, gives its bytes back.
    del wide
    test.assertEqual(pool.row_bytes, 16)
    test.assertEqual(pool.bound, device_row_limit(device, 16, narrow_limit))


def test_reservation_holds_the_device_bound(test, device):
    """Reserve addresses for the pool's device bound in every field, which the driver accepts, and map only the rows.

    Without ``gpu_components`` nothing is reserved: the rows are ordinary arrays of the ready rows.
    """
    model = _build_boxes(device)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    pool = contacts._rigid_contact_pool
    members = pool._live_members()
    virtual = contact_pool._virtual_memory() is not None
    test.assertTrue(all(member._mode == ("virtual" if virtual else "realloc") for member in members))
    memory = int(wp.get_device(device).total_memory)
    reserved = pool.reserved_bytes()
    mapped = pool.mapped_bytes()
    print(
        f"{wp.get_device(device).name}: {memory / 2**30:.1f} GiB device; contact pool of {len(members)} members, "
        f"{sum(len(member.fields) for member in members)} fields, {pool.row_bytes} B/row, bound {pool.bound} rows, "
        f"reserves {reserved / 2**30:.1f} GiB of addresses, maps {mapped / 2**20:.1f} MiB for {pool.ready} rows"
    )
    index_limit = min(field.row_limit(virtual) for member in members for field in member.fields)
    test.assertEqual(pool.bound, device_row_limit(device, pool.row_bytes, index_limit))
    if virtual:
        test.assertGreaterEqual(reserved, pool.bound * pool.row_bytes)
    else:
        test.assertEqual(reserved, mapped)
        test.assertEqual(mapped, pool.ready * pool.row_bytes)
    for member in members:
        test.assertGreaterEqual(member.reserve, pool.bound)
        for field in member.fields:
            size = wp.types.type_size_in_bytes(field.dtype) if virtual else 1
            test.assertLessEqual(field.length(member.reserve) * size, INDEX_LIMIT)
    test.assertLess(mapped, memory // 64, "only the ready rows are mapped")
    test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 0)
    pipeline.collide(model.state(), contacts)
    test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), CONTACTS)


def test_rows_fall_back_without_gpu_components(test, device):
    """Hold CUDA rows in ordinary arrays without ``gpu_components``: growth reallocates, copies and replaces the views."""
    with _without_gpu_components():
        with test.assertLogs("newton._src.core.contact_pool", level="INFO") as logs:
            pool = ContactPool(device=device, rows=64)
            target = type("Target", (), {})()
            rows = pool.rows((ContactField("value", wp.int32, -1),), on_rows=lambda rows: rows.publish(target))
            test.assertEqual(rows._mode, "realloc")
            # The second rows of the process look the package up no more.
            pool.rows((ContactField("other", wp.vec3),))
        test.assertEqual(sum("gpu_components is not installed" in line for line in logs.output), 1)
        target.value.assign(np.arange(64, dtype=np.int32))
        first = target.value
        test.assertTrue(pool.grow(4096))
        test.assertIsNot(target.value, first)
        test.assertEqual(target.value.shape, (4096,))
        np.testing.assert_array_equal(target.value.numpy()[:64], np.arange(64))
        test.assertTrue(np.all(target.value.numpy()[64:] == -1))
        test.assertEqual(pool.generation, 1, "a growth requests a recapture")
        # A collision pass outside capture grows the fallback rows as it grows virtual memory.
        model = _build_boxes(device)
        with _small_pool_policy(8, 8):
            pipeline = newton.CollisionPipeline(model)
            contacts = pipeline.contacts()
            test.assertEqual(contacts._rigid_rows._mode, "realloc")
            pipeline.collide(model.state(), contacts)
        test.assertEqual(contacts.rigid_contact_max, GROWN)
        test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), CONTACTS)


def test_growth_past_free_memory_raises_before_changing(test, device):
    """Refuse a growth the free device memory cannot back, before mapping anything, and keep the rows."""
    pool = ContactPool(device=device, rows=64)
    target = type("Target", (), {})()
    # Owners hold their rows; the pool holds them weakly.
    rows = pool.rows((ContactField("value", wp.vec4, 1.0),), on_rows=lambda rows: rows.publish(target))
    view = target.value
    free = mock.PropertyMock(return_value=64 << 20)
    with mock.patch.object(type(wp.get_device(device)), "free_memory", new=free):
        with test.assertRaisesRegex(MemoryError, r"to 67108864 rows needs [0-9.]+ MiB more device memory; 64\.0 MiB"):
            pool.grow(64 << 20)
    test.assertEqual(pool.ready, 64)
    test.assertIs(target.value, view)
    test.assertTrue(pool.grow(128))
    test.assertEqual(rows.rows, 128)


def test_pool_follower_grows_with_its_leader(test, device):
    """Grow a following pool with its leader."""
    leader = ContactPool(device=device, rows=100)
    follower = ContactPool(device=device, rows=50)
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


def test_compatibility_budgets_are_silent_and_size_nothing(test, device):
    """Accept Newton main's rigid contact budget silently, resolved as on main; it changes no pool's rows or bound.

    ``CollisionPipeline(rigid_contact_max=...)``, ``CollisionPipeline.rigid_contact_max``,
    ``Model.rigid_contact_max`` and ``ModelBuilder.num_rigid_contacts_per_world`` keep Newton main's behavior.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        builder = newton.ModelBuilder()
        builder.num_rigid_contacts_per_world = 3
        test.assertEqual(builder.num_rigid_contacts_per_world, 3)
        # A model of its own, sized by the estimate as on Newton main, for the pool to compare with.
        estimated = _build_boxes(device)
        reference = newton.CollisionPipeline(estimated)
        test.assertEqual(reference.rigid_contact_max, _estimate_rigid_contact_max(estimated))
        test.assertEqual(estimated.rigid_contact_max, reference.rigid_contact_max, "written to the model")
        reference_contacts = reference.contacts()

        model = _build_boxes(device)
        test.assertEqual(model.rigid_contact_max, 0)
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=5)
        test.assertEqual(pipeline.rigid_contact_max, 5)
        test.assertEqual(model.rigid_contact_max, 5)
        contacts = pipeline.contacts()
        pool = contacts._rigid_contact_pool
        test.assertEqual(contacts.rigid_contact_max, contact_pool.MIN_READY_ROWS)
        test.assertEqual(pool.ready, reference_contacts._rigid_contact_pool.ready)
        test.assertEqual(pool.bound, reference_contacts._rigid_contact_pool.bound)
        model.rigid_contact_max = 7
        later = newton.CollisionPipeline(model)
        test.assertEqual(later.rigid_contact_max, 7, "the model's value without an argument")
        test.assertEqual(later.contacts().rigid_contact_max, contacts.rigid_contact_max)
        pipeline.collide(model.state(), contacts)
    test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), CONTACTS)
    test.assertEqual(contacts.rigid_contact_max, contact_pool.MIN_READY_ROWS, "a budget of 5 holds all 48 contacts")


def test_collide_outside_capture_grows_before_reading(test, device):
    """Grow and rerun a collision pass outside capture that overflows its rows: no contact dropped, no flag."""
    model = _build_boxes(device, worlds=1, boxes=40)
    # The reference holds its own model, and so its own contact pool.
    reference = newton.CollisionPipeline(_build_boxes(device, worlds=1, boxes=40))
    expected = newton.Contacts(1024, 0, device=device)
    reference.collide(model.state(), expected)
    demand = int(expected.rigid_contact_count.numpy()[0])
    test.assertGreater(demand, 64)
    with _small_pool_policy(64, 64):
        pipeline = newton.CollisionPipeline(model, deterministic=True)
        contacts = pipeline.contacts()
        test.assertEqual(contacts.rigid_contact_max, 64)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            pipeline.collide(model.state(), contacts)
        rows = sized_rows(demand, 64, pipeline._contact_pool.bound)
    test.assertEqual(contacts.rigid_contact_max, rows)
    test.assertEqual(pipeline._sort_key_array.shape, (rows,))
    test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), demand)
    test.assertFalse(any(pipeline.narrow_phase.buffer_capacity_status().values()))
    np.testing.assert_array_equal(_records(contacts), _records(expected))
    test.assertTrue(pipeline.service_capacity(), "graphs captured before the growth must be recaptured")
    test.assertFalse(pipeline.service_capacity())


def test_service_measures_a_state_before_capture(test, device):
    """Measure a state's demand outside capture: grow to it without writing any Contacts or matching history."""
    model = _build_boxes(device)
    resting, lifted = model.state(), model.state()
    body_q = lifted.body_q.numpy()
    body_q[:, 2] += 1.0
    lifted.body_q.assign(body_q)
    with _small_pool_policy(8, 8):
        reference = newton.CollisionPipeline(_build_boxes(device), contact_matching="latest")
        expected = reference.contacts()
        pipeline = newton.CollisionPipeline(model, contact_matching="latest")
        contacts = pipeline.contacts()
        test.assertEqual(contacts.rigid_contact_max, 8)
        generation = contacts.contact_generation.numpy().copy()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            test.assertTrue(pipeline.service_capacity(resting), "graphs captured before must be recaptured")
        test.assertEqual(contacts.rigid_contact_max, GROWN)
        test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 0, "the measurement writes no Contacts")
        np.testing.assert_array_equal(contacts.contact_generation.numpy(), generation)
        test.assertFalse(any(pipeline.narrow_phase.buffer_capacity_status().values()))
        test.assertFalse(pipeline.service_capacity(resting), "the rows hold the measured demand")
        # Measuring a contact-free state between two frames keeps the second frame's matches.
        for state in (resting, None, resting):
            if state is None:
                pipeline.service_capacity(lifted)
                continue
            for run_pipeline, run_contacts in ((reference, expected), (pipeline, contacts)):
                run_pipeline.collide(state, run_contacts)
    test.assertEqual(contacts.rigid_contact_max, expected.rigid_contact_max)
    match = contacts.rigid_contact_match_index
    np.testing.assert_array_equal(_records(contacts, match), _records(expected, expected.rigid_contact_match_index))
    test.assertTrue(np.all(match.numpy()[:CONTACTS] >= 0), "resting contacts match their previous frame")


def test_capture_after_measuring_holds_every_contact(test, device):
    """Capture a collision pass after measuring the state it starts from: its first replay drops nothing."""
    model = _build_boxes(device)
    resting, lifted = model.state(), model.state()
    body_q = lifted.body_q.numpy()
    body_q[:, 2] += 1.0
    lifted.body_q.assign(body_q)
    with _small_pool_policy(8, 8):
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        # Load the modules outside capture with a contact-free pass: no growth.
        pipeline.collide(lifted, contacts)
        test.assertEqual(contacts.rigid_contact_max, 8)
        test.assertTrue(pipeline.service_capacity(resting))
        with wp.ScopedCapture(device=device) as capture:
            pipeline.collide(resting, contacts)
        wp.capture_launch(capture.graph)
        wp.synchronize_device(device)
    test.assertFalse(pipeline.narrow_phase.buffer_capacity_status()["contacts"])
    test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), CONTACTS)
    test.assertFalse(pipeline.service_capacity())


def test_example_runner_measures_and_recaptures(test, device):
    """Size an example's buffers for its starting state before the first frame; recapture before it only on overflow.

    An example captures its graph before the runner measures, and its first capture may allocate buffers that only
    its first replay makes valid: a graph whose buffers hold the starting state's contacts is recaptured after the
    first frame instead.
    """
    from newton.examples import _service_contact_capacity  # noqa: PLC0415

    def example_with_graph():
        example = type("Example", (), {})()
        model = _build_boxes(device)
        example.collision_pipeline = newton.CollisionPipeline(model)
        example.contacts = example.collision_pipeline.contacts()
        example.state_0 = model.state()
        example.graph = object()
        example.captures = []
        example.capture = lambda: example.captures.append(example.contacts.rigid_contact_max)
        return example

    with _small_pool_policy(8, 8):
        overflowing = example_with_graph()
        test.assertFalse(_service_contact_capacity(overflowing, overflowing.state_0))
        test.assertEqual(overflowing.captures, [GROWN], "8 rows cannot hold the 48 starting contacts")
        test.assertFalse(_service_contact_capacity(overflowing))
        test.assertFalse(_service_contact_capacity(overflowing, overflowing.state_0))
    test.assertEqual(overflowing.captures, [GROWN], "no growth, no recapture")
    with _small_pool_policy(64, 8):
        holding = example_with_graph()
        test.assertTrue(_service_contact_capacity(holding, holding.state_0), "64 rows hold the 48 starting contacts")
        test.assertEqual(holding.contacts.rigid_contact_max, GROWN)
        test.assertEqual(holding.captures, [])
        test.assertFalse(_service_contact_capacity(holding, recapture=True))
    test.assertEqual(holding.captures, [GROWN], "recaptured after the first frame")


def test_measurement_keeps_the_floor_or_sizes_to_the_demand(test, device):
    """Size a model's buffers before capture: no contacts keep the floor, contacts get HEADROOM rows each.

    The solver, constructed before the pipeline, shares the model's pool, so the measurement sizes its per-contact
    buffers too; a graph captured afterwards holds every contact from its first replay and never grows.
    """
    floor = contact_pool.MIN_READY_ROWS
    model = _build_boxes(device)
    solver = newton.solvers.SolverFeatherPGS(model, pgs_iterations=4)
    pipeline = newton.CollisionPipeline(model)
    lifted = model.state()
    body_q = lifted.body_q.numpy()
    body_q[:, 2] += 1.0
    lifted.body_q.assign(body_q)
    test.assertFalse(pipeline.service_capacity(lifted), "a contact-free scene keeps the floor")
    test.assertEqual(pipeline.contacts().rigid_contact_max, floor)
    test.assertEqual(solver.contact_slot.shape, (floor,))

    worlds, boxes = 128, 8
    demand = 4 * worlds * boxes
    model = _build_boxes(device, worlds=worlds, boxes=boxes)
    solver = newton.solvers.SolverFeatherPGS(model, pgs_iterations=4)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        test.assertTrue(pipeline.service_capacity(state_0), "graphs captured before the sizing must be recaptured")
    rows = sized_rows(demand, floor, pipeline._contact_pool.bound)
    test.assertGreater(rows, floor)
    test.assertEqual(contacts.rigid_contact_max, rows)
    test.assertIs(solver._contact_pool, pipeline._contact_pool)
    test.assertEqual(solver.contact_slot.shape, (rows,))
    test.assertEqual(solver._friction_patches.capacity, rows)

    def simulate():
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, 1.0 / 120.0)
        wp.copy(state_0.body_q, state_1.body_q)
        wp.copy(state_0.body_qd, state_1.body_qd)

    if wp.get_device(device).is_cuda:
        simulate()  # load the modules outside capture
        # FeatherPGS seeds its double-buffer events in each capture, after the queued clears finish.
        wp.synchronize_device(device)
        with wp.ScopedCapture(device=device) as capture:
            solver.seed_double_buffer_events()
            simulate()
        for _ in range(3):
            wp.capture_launch(capture.graph)
    else:
        for _ in range(3):
            simulate()
    wp.synchronize_device(device)
    test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), demand)
    test.assertFalse(pipeline.narrow_phase.buffer_capacity_status()["contacts"])
    solver.check_constraint_capacity()
    test.assertFalse(pipeline.service_capacity(), "no growth after the capture")
    test.assertEqual(contacts.rigid_contact_max, rows)


def test_pipelines_of_a_model_share_its_pool(test, device):
    """Share one pool among a model's pipelines and solvers; report a growth to each pipeline once."""
    model = _build_boxes(device)
    with _small_pool_policy(8, 8):
        first = newton.CollisionPipeline(model)
        solver = newton.solvers.SolverFeatherPGS(model, pgs_iterations=4)
        second = newton.CollisionPipeline(model)
        test.assertIs(first._contact_pool, model_contact_pool(model))
        test.assertIs(second._contact_pool, first._contact_pool)
        test.assertIs(solver._contact_pool, first._contact_pool)
        other = newton.CollisionPipeline(_build_boxes(device))
        test.assertIsNot(other._contact_pool, first._contact_pool, "another model has a pool of its own")
        test.assertTrue(first.service_capacity(model.state()))
        test.assertEqual(second.contacts().rigid_contact_max, GROWN)
        test.assertEqual(solver.contact_slot.shape, (GROWN,))
        test.assertTrue(second.service_capacity(), "the second pipeline's graphs must be recaptured too")
        test.assertFalse(first.service_capacity())
        test.assertFalse(second.service_capacity())
        test.assertEqual(other.contacts().rigid_contact_max, 8)


def test_pipeline_growth_without_overflow_matches_fixed(test, device):
    """Grow past half the ready rows without dropping contacts; every pass equals a larger buffer's."""
    model = _build_boxes(device)
    reference = newton.CollisionPipeline(_build_boxes(device))
    expected = reference.contacts()
    with _small_pool_policy(64, 8):
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        test.assertEqual(contacts.rigid_contact_max, 64)
        _collide_like_fixed(test, device, model, reference, expected, pipeline, contacts)
    test.assertEqual(contacts.rigid_contact_max, GROWN)


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
    """Grow FeatherPGS's per-contact buffers with the pipeline mid-run and keep the physics of a larger run."""
    reference, _, _, _ = _feather_pgs_run(device, 6)
    state, contacts, pipeline, solver = _feather_pgs_run(device, 6, policy=_small_pool_policy(64, 8))
    test.assertEqual(pipeline._contact_pool.growth_count, 1)
    test.assertEqual(solver._contact_pool.ready, GROWN)
    test.assertEqual(solver.contact_slot.shape, (GROWN,))
    test.assertEqual(solver._friction_patches.capacity, GROWN)
    test.assertEqual(contacts.rigid_contact_max, GROWN)
    solver.check_constraint_capacity()
    np.testing.assert_allclose(state.body_q.numpy(), reference.body_q.numpy(), rtol=0.0, atol=1.0e-5)
    np.testing.assert_allclose(state.body_qd.numpy(), reference.body_qd.numpy(), rtol=0.0, atol=1.0e-4)


def test_feather_pgs_follows_a_pass_that_grew(test, device):
    """Hand FeatherPGS the rows a collision pass grew to before it reads them, so no constraint is lost."""
    state, contacts, _, solver = _feather_pgs_run(device, 2, policy=_small_pool_policy(8, 8))
    solver.check_constraint_capacity()
    test.assertEqual(contacts.rigid_contact_max, GROWN)
    test.assertEqual(solver._contact_pool.ready, GROWN)
    test.assertTrue(np.all(np.isfinite(state.body_q.numpy())))


def test_matching_and_warm_start_survive_growth(test, device):
    """Match the frame after a growth to the frame before it, so warm starts keep their history."""
    matched = {}

    def after_collide(step, pipeline, contacts):
        count = int(contacts.rigid_contact_count.numpy()[0])
        matched[step] = (pipeline._contact_pool.growth_count, contacts.rigid_contact_match_index.numpy()[:count])

    _feather_pgs_run(device, 3, policy=_small_pool_policy(64, 8), warmstart=True, after_collide=after_collide)
    growths, match = matched[1]
    test.assertEqual(growths, 1, "the pool grew after the first step")
    test.assertEqual(match.shape, (CONTACTS,))
    test.assertTrue(np.all(match >= 0), f"resting contacts must match across growth: {match}")


def test_graph_capture_recapture_after_growth(test, device):
    """Flag a replayed overflow, grow on service, replay the stale graph safely, then recapture."""
    with _small_pool_policy(8, 8):
        _graph_capture_recapture_after_growth(test, device)


def _graph_capture_recapture_after_growth(test, device):
    model = _build_boxes(device)
    pipeline = newton.CollisionPipeline(model)
    solver = newton.solvers.SolverFeatherPGS(model, pgs_iterations=8)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()
    resting = state_0.body_q.numpy()

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

    # Warm up outside capture (module loads, lazy setup) with the boxes in the air: no contacts, no growth.
    lifted = resting.copy()
    lifted[:, 2] += 1.0
    state_0.body_q.assign(lifted)
    simulate()
    test.assertEqual(contacts.rigid_contact_max, 8)
    state_0.body_q.assign(resting)
    state_0.body_qd.zero_()
    old_graph = capture_step()
    wp.capture_launch(old_graph)
    wp.synchronize_device(device)
    test.assertTrue(pipeline.narrow_phase.buffer_capacity_status(clear=True)["contacts"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        test.assertTrue(pipeline.service_capacity())
    test.assertTrue(any("reached the 8 ready rows" in str(w.message) for w in caught))
    test.assertTrue(any("service_capacity(state)" in str(w.message) for w in caught), "says how to size before")
    test.assertEqual(contacts.rigid_contact_max, GROWN)
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
    """Grow to a reservation outside capture; defer it, with a warning, during capture; count each growth once."""
    with _small_pool_policy(8, 8):
        pool = ContactPool(device=device, rows=8)
        target = type("Target", (), {})()
        rows = pool.rows((ContactField("value", wp.int32, -1),), on_rows=lambda rows: rows.publish(target))
        test.assertFalse(pool.reserve(8))
        test.assertTrue(pool.reserve(100))
        test.assertEqual(pool.ready, 104, "rounded up to the row quantum")
        test.assertEqual(rows.rows, 104)
        test.assertEqual(pool.generation, 1, "graphs captured before the growth must be recaptured")
        test.assertFalse(pool.service(), "nothing left to grow")
        with test.assertRaisesRegex(MemoryError, "cannot back"):
            pool.reserve(pool.bound + 1)
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
        test.assertEqual(pool.generation, 2)
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
    with _small_pool_policy(8, 8):
        _mujoco_contacts_grow_a_pipeline_buffer(test, device)


def _mujoco_model_and_solver(device):
    """Boxes with contact forces, and MJWarp with its own collision three steps after they landed."""
    model = _build_boxes(device)
    model.request_contact_attributes("force")
    solver = _mujoco(model, use_mujoco_contacts=True)
    state_0, state_1, control = model.state(), model.state(), model.control()
    for _ in range(3):
        solver.step(state_0, state_1, control, None, 1.0 / 240.0)
        state_0, state_1 = state_1, state_0
    return model, solver, state_0, state_1, control


def _mujoco_contacts_grow_a_pipeline_buffer(test, device):
    model, solver, state_0, state_1, control = _mujoco_model_and_solver(device)
    naconmax = solver.mjw_data.naconmax
    reference = newton.Contacts(naconmax, 0, device=device, requested_attributes={"force"})
    nacon = int(solver.mjw_data.nacon.numpy()[0])
    pipeline = newton.CollisionPipeline(model)
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
    # A model of its own, whose pool starts at the initial rows again.
    model, solver, state_0, state_1, control = _mujoco_model_and_solver(device)
    pipeline = newton.CollisionPipeline(model)
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
    reference = newton.CollisionPipeline(_force_boxes(device))
    with _small_pool_policy(64, 8):
        _newton_contacts_feed_mujoco_below_naconmax(test, device, reference)


def _force_boxes(device):
    """The boxes, with contact forces requested."""
    model = _build_boxes(device)
    model.request_contact_attributes("force")
    return model


def _newton_contacts_feed_mujoco_below_naconmax(test, device, reference):
    model = _force_boxes(device)
    runs = [
        (
            pipeline,
            pipeline.contacts(),
            _mujoco(pipeline.model, use_mujoco_contacts=False),
            pipeline.model.state(),
            pipeline.model.state(),
        )
        for pipeline in (reference, newton.CollisionPipeline(model))
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
    test.assertEqual(contacts.rigid_contact_max, GROWN, "grown from the demand of 48 contacts, not to naconmax")
    test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), int(fixed.rigid_contact_count.numpy()[0]))
    for _, _, run_solver, _, _ in runs:
        test.assertFalse(any(run_solver.contact_capacity_status().values()))
    np.testing.assert_allclose(state.body_q.numpy(), fixed_state.body_q.numpy(), rtol=0.0, atol=1.0e-5)
    np.testing.assert_allclose(_mujoco_records(contacts), _mujoco_records(fixed), rtol=1.0e-4, atol=1.0e-4)

    if not wp.get_device(device).is_cuda:
        return
    # A model of its own, whose pool starts at the initial rows again.
    model = _force_boxes(device)
    solver = _mujoco(model, use_mujoco_contacts=False)
    pipeline = newton.CollisionPipeline(model)
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
    test.assertEqual(contacts.rigid_contact_max, GROWN)
    test.assertFalse(any(solver.contact_capacity_status().values()))
    test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), int(solver.mjw_data.nacon.numpy()[0]))
    test.assertTrue(np.all(np.isfinite(state_0.body_q.numpy())))


def test_xpbd_update_contacts_after_growth(test, device):
    """Report the forces of the last step from a buffer that grew since it; rows the step did not solve get zero."""
    model = _build_boxes(device)
    model.request_contact_attributes("force")
    solver = newton.solvers.SolverXPBD(model, iterations=4)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()
    pipeline.collide(state_0, contacts)
    solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
    solver.update_contacts(contacts)
    rows = contacts.rigid_contact_max
    forces = contacts.force.numpy()[:rows]
    test.assertTrue(np.any(forces != 0.0))
    test.assertTrue(contacts.reserve_rigid_contacts(4 * rows))
    solver.update_contacts(contacts)
    np.testing.assert_array_equal(contacts.force.numpy()[:rows], forces)
    test.assertTrue(np.all(contacts.force.numpy()[rows : 4 * rows] == 0.0))


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
    with _small_pool_policy(3, 1):
        contacts = newton.CollisionPipeline(model).contacts()
        test.assertLess(contacts.rigid_contact_max, capacity)
        state_0, state_1 = model.state(), model.state()
        solver.step(state_0, state_1, None, None, 1.0 / 240.0)
        solver.update_contacts(contacts, state_1)
    test.assertEqual(contacts.rigid_contact_max, capacity)
    active = int(solver._contacts_kamino.model_active_contacts.numpy()[0])
    test.assertGreater(active, 0)
    test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), active)
    test.assertTrue(np.all(contacts.rigid_contact_shape0.numpy()[:active] >= 0))


def test_vbd_contact_buffers_follow_the_contacts(test, device):
    """Grow VBD's per-contact state with the contacts it consumes, keeping its contents."""
    model = _build_boxes(device, color=True)
    solver = newton.solvers.SolverVBD(model, iterations=2, rigid_compliant_alm=True)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()
    pipeline.collide(state_0, contacts)
    solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
    rows = contacts.rigid_contact_max
    test.assertEqual(solver.body_body_contact_penalty_k.shape, (rows,))
    penalty = solver.body_body_contact_penalty_k.numpy().copy()
    test.assertTrue(contacts.reserve_rigid_contacts(4 * rows))
    pipeline.collide(state_1, contacts)
    solver.step(state_1, state_0, control, contacts, 1.0 / 240.0)
    test.assertEqual(solver.body_body_contact_penalty_k.shape, (4 * rows,))
    test.assertEqual(solver._rigid_contact_body0.shape, (4 * rows,))
    test.assertTrue(np.all(np.isfinite(solver.body_body_contact_penalty_k.numpy()[:rows])))
    test.assertEqual(penalty.shape, (rows,))
    test.assertTrue(np.all(np.isfinite(state_0.body_q.numpy())))


def test_coupled_entry_contacts_grow_with_their_source(test, device):
    """Grow a coupled entry's filtered buffer with its source, and keep reusing it."""
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
    with _small_pool_policy(8, 8):
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        filtered = coupled.entry_contacts("A", contacts)
        test.assertEqual(filtered.rigid_contact_max, 8)
        state = model.state()
        pipeline.collide(state, contacts)
    test.assertEqual(contacts.rigid_contact_max, GROWN)
    test.assertEqual(filtered.rigid_contact_max, GROWN)
    test.assertIs(coupled.entry_contacts("A", contacts), filtered)
    np.testing.assert_array_equal(_records(filtered), _records(contacts))


def test_contact_kinematics_outputs_hold_the_rows(test, device):
    """Evaluate contact kinematics into outputs of at least the contacts' rows, before and after a growth."""
    model = _build_boxes(device)
    state = model.state()
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    rows = contacts.rigid_contact_max
    distance = wp.empty(4 * rows, dtype=float, device=device)
    pipeline.collide(state, contacts)
    newton.eval_rigid_contact_kinematics(model, state, contacts, out_distance=distance)
    before = np.sort(distance.numpy()[:CONTACTS])
    test.assertTrue(contacts.reserve_rigid_contacts(4 * rows))
    pipeline.collide(state, contacts)
    newton.eval_rigid_contact_kinematics(model, state, contacts, out_distance=distance)
    np.testing.assert_allclose(np.sort(distance.numpy()[:CONTACTS]), before, rtol=0.0, atol=1.0e-7)
    with test.assertRaisesRegex(ValueError, "expected"):
        newton.eval_rigid_contact_kinematics(
            model, state, contacts, out_distance=wp.empty(rows, dtype=float, device=device)
        )


class TestContactPool(unittest.TestCase):
    pass


devices = get_test_devices()
cuda_devices = get_cuda_test_devices()
# Every test runs as well with gpu_components hidden: CUDA rows then fall back to ordinary arrays.
# Overflowing passes print the narrow phase's capacity warning.
for name, function, check_output in (
    ("test_pool_rows_keep_contents_and_fill_new_rows", test_pool_rows_keep_contents_and_fill_new_rows, True),
    ("test_pool_bound_is_the_hardware_limit", test_pool_bound_is_the_hardware_limit, True),
    ("test_pool_follower_grows_with_its_leader", test_pool_follower_grows_with_its_leader, True),
    (
        "test_compatibility_budgets_are_silent_and_size_nothing",
        test_compatibility_budgets_are_silent_and_size_nothing,
        True,
    ),
    ("test_collide_outside_capture_grows_before_reading", test_collide_outside_capture_grows_before_reading, True),
    ("test_service_measures_a_state_before_capture", test_service_measures_a_state_before_capture, True),
    ("test_example_runner_measures_and_recaptures", test_example_runner_measures_and_recaptures, True),
    (
        "test_measurement_keeps_the_floor_or_sizes_to_the_demand",
        test_measurement_keeps_the_floor_or_sizes_to_the_demand,
        True,
    ),
    ("test_pipelines_of_a_model_share_its_pool", test_pipelines_of_a_model_share_its_pool, True),
    ("test_pipeline_growth_without_overflow_matches_fixed", test_pipeline_growth_without_overflow_matches_fixed, True),
    ("test_feather_pgs_grows_mid_run", test_feather_pgs_grows_mid_run, True),
    ("test_feather_pgs_follows_a_pass_that_grew", test_feather_pgs_follows_a_pass_that_grew, True),
    ("test_matching_and_warm_start_survive_growth", test_matching_and_warm_start_survive_growth, True),
    ("test_reserve_grows_now_or_at_the_next_service", test_reserve_grows_now_or_at_the_next_service, True),
    ("test_mujoco_contacts_grow_a_pipeline_buffer", test_mujoco_contacts_grow_a_pipeline_buffer, False),
    ("test_newton_contacts_feed_mujoco_below_naconmax", test_newton_contacts_feed_mujoco_below_naconmax, True),
    ("test_xpbd_update_contacts_after_growth", test_xpbd_update_contacts_after_growth, True),
    ("test_kamino_contacts_grow_a_pipeline_buffer", test_kamino_contacts_grow_a_pipeline_buffer, True),
    ("test_vbd_contact_buffers_follow_the_contacts", test_vbd_contact_buffers_follow_the_contacts, True),
    ("test_coupled_entry_contacts_grow_with_their_source", test_coupled_entry_contacts_grow_with_their_source, True),
    ("test_contact_kinematics_outputs_hold_the_rows", test_contact_kinematics_outputs_hold_the_rows, True),
):
    add_function_test(TestContactPool, name, function, devices=devices, check_output=check_output)
    add_function_test(
        TestContactPool,
        f"{name}_without_gpu_components",
        _without_virtual_memory(function),
        devices=devices,
        check_output=check_output,
    )
for name, function in (
    ("test_reservation_holds_the_device_bound", test_reservation_holds_the_device_bound),
    ("test_growth_past_free_memory_raises_before_changing", test_growth_past_free_memory_raises_before_changing),
    ("test_graph_capture_recapture_after_growth", test_graph_capture_recapture_after_growth),
    ("test_capture_after_measuring_holds_every_contact", test_capture_after_measuring_holds_every_contact),
):
    add_function_test(TestContactPool, name, function, devices=cuda_devices, check_output=False)
    add_function_test(
        TestContactPool,
        f"{name}_without_gpu_components",
        _without_virtual_memory(function),
        devices=cuda_devices,
        check_output=False,
    )
add_function_test(
    TestContactPool,
    "test_rows_fall_back_without_gpu_components",
    test_rows_fall_back_without_gpu_components,
    devices=cuda_devices,
    check_output=False,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
