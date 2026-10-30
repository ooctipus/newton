# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the growable rigid contact pool: policy, growth, overflow, capture, and its consumers."""

import contextlib
import math
import sys
import time
import unittest
import warnings
from types import SimpleNamespace
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


@contextlib.contextmanager
def _small_pool_policy(rows: int, quantum: int):
    """Start new pools at ``rows`` rows, without a first mapping, and round sizes to ``quantum``.

    A small scene then starts below its demand, which exercises growth past the mapped rows.
    """
    with (
        mock.patch.multiple(contact_pool, MIN_READY_ROWS=rows, ROW_QUANTUM=quantum),
        mock.patch.object(ContactPool, "plan_first_mapping", lambda pool, estimate: None),
    ):
        yield


@contextlib.contextmanager
def _without_gpu_components():
    """Make importing ``gpu_components`` fail, as in an environment without the optional package."""
    hidden = dict.fromkeys(
        ("gpu_components", "gpu_components.backing", "gpu_components.fields", "gpu_components.field_data")
    )
    with mock.patch.dict(sys.modules, hidden), mock.patch.object(contact_pool, "_VIRTUAL_MEMORY", []):
        yield


def _reallocates(pool):
    """Whether growth reallocates the pool's rows, so that every graph that captured them must be recaptured."""
    return not all(member.virtual for member in pool._live_members())


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
    def test_pipelines_map_the_model_estimate_and_never_a_budget(self):
        """A pool no pipeline planned holds the floor; a pipeline maps the model's shape-pair estimate, not a budget."""
        self.assertEqual(contact_pool.MIN_READY_ROWS, 4096)
        self.assertEqual(ContactPool(device="cpu").ready, 4096)
        model = _build_boxes("cpu", worlds=32, boxes=8)
        estimate = _estimate_rigid_contact_max(model)
        self.assertGreater(estimate, contact_pool.MIN_READY_ROWS)
        model.rigid_contact_max = 7
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=5)
        self.assertEqual(pipeline.rigid_contact_max, 5)
        self.assertEqual(pipeline._contact_pool.estimate, estimate)
        self.assertEqual(pipeline.contacts().rigid_contact_max, estimate, "the estimate, never the budget")
        small = _build_boxes("cpu", worlds=1, boxes=1)
        self.assertEqual(newton.CollisionPipeline(small).contacts().rigid_contact_max, contact_pool.MIN_READY_ROWS)

    def test_first_mapping_keeps_a_share_of_the_free_memory(self):
        """Cap the first mapping at FIRST_MAPPING_SHARE of the memory free at creation over every field's bytes."""
        pool = ContactPool(device="cpu")
        pool._estimate = 10_000_000
        self.assertEqual(pool._first_rows(100, INDEX_LIMIT), 10_000_000, "the CPU reports no free memory: no cap")
        pool.device = SimpleNamespace(is_cuda=True)
        pool.free_memory_at_creation = 8 * 2**30
        share = contact_pool.FIRST_MAPPING_SHARE
        self.assertEqual(pool._first_rows(1000, INDEX_LIMIT), int(share * 8 * 2**30) // 1000)
        self.assertEqual(pool._first_rows(10, INDEX_LIMIT), 10_000_000, "the estimate fits the share")
        self.assertEqual(pool._first_rows(10, 1000), 1000, "at most the bound")
        pool._estimate = 0
        self.assertEqual(pool._first_rows(10, INDEX_LIMIT), contact_pool.MIN_READY_ROWS, "at least the floor")

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

    def test_outdated_gpu_components_counts_as_absent(self):
        """Keep ordinary arrays, with a warning, under a gpu_components release without the API the pool uses.

        A release whose withdrawal takes no reader streams would let a trim unmap rows that a replay on another
        stream still writes.
        """

        def prepare_updates(enable_count, *, enable_count_maximum, binding_capacity, errors=None, library=None):
            pass

        def launch(updates, kernel, dim, *, inputs=None, extent_axis=None, extent_source=None, parameters=()):
            pass

        def withdraw(storage, rows):
            pass

        package = mock.MagicMock()
        package.graph.prepare_updates = prepare_updates
        package.graph.launch = launch
        package.fields.withdraw = withdraw
        package.graph.__file__ = "outdated/gpu_components/graph.py"
        modules = {
            "gpu_components": package,
            **{f"gpu_components.{name}": getattr(package, name) for name in ("backing", "fields", "graph")},
            **{f"gpu_components.{name}": mock.MagicMock() for name in ("field_data", "graph_data")},
        }
        with mock.patch.dict(sys.modules, modules), mock.patch.object(contact_pool, "_VIRTUAL_MEMORY", []):
            with self.assertLogs(contact_pool.logger, "WARNING") as logs:
                self.assertIsNone(contact_pool._virtual_memory())
        self.assertIn(
            "graph.launch(clamp_extent), graph.launch(extent_scale), fields.withdraw(streams), "
            "fields.withdraw(protected_count_host)",
            logs.output[0],
        )


def test_pool_rows_keep_contents_and_fill_new_rows(test, device):
    """Keep every field's first rows through growth and give new and scratch rows their fill.

    CUDA virtual memory maps whole 2 MiB granules: once the owner settles the pool, its rows are every row the pages
    of all fields cover, and growing past them maps new pages. Ordinary arrays hold exactly the rows asked for.
    """
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
    virtual = rows._mode == "virtual"
    test.assertEqual((rows.rows, published), (100, [100]))
    pool.settle()
    first = pool.ready
    test.assertEqual(rows.rows, first)
    if virtual:
        # The spatial-vector field's first granule holds the fewest rows: 2 MiB / 24 B less its 7 scratch rows.
        test.assertEqual(first, (2 << 20) // 24 - 7)
        test.assertEqual(published, [100, first])
        test.assertEqual(pool.growth_count, 1, "publishing the covered rows is a growth")
    else:
        test.assertEqual((first, published, pool.growth_count), (100, [100], 0))
    settled = pool.growth_count
    test.assertEqual(target.key.shape, (2 * first,))
    test.assertEqual(target.force.shape, (first + 7,))
    np.testing.assert_array_equal(target.slot.numpy(), np.arange(first))
    test.assertTrue(np.all(target.index.numpy() == -1))
    target.index.assign(np.arange(first, dtype=np.int32))
    target.key.assign(np.full(2 * first, 5, dtype=np.int64))
    old_index = target.index
    grown = 3 * first + 1000
    test.assertEqual(pool.grow(grown), not virtual, "only reallocated rows need graphs recaptured")
    test.assertGreaterEqual(pool.ready, grown)
    test.assertEqual(published[-1], pool.ready)
    test.assertEqual(rows.rows, pool.ready)
    test.assertEqual(pool.growth_count, settled + 1)
    held = pool.ready
    index = target.index.numpy()
    np.testing.assert_array_equal(index[:first], np.arange(first))
    test.assertTrue(np.all(index[first:] == -1))
    key = target.key.numpy()
    test.assertEqual(key.shape, (2 * held,))
    test.assertTrue(np.all(key[:first] == 5))
    test.assertTrue(np.all(key[first:] == 0x7FFFFFFFFFFFFFFF), "scratch and new rows take the fill")
    np.testing.assert_array_equal(target.slot.numpy(), np.arange(held))
    test.assertEqual(target.force.shape, (held + 7,))
    test.assertTrue(np.all(target.point.numpy() == 0.0))
    test.assertFalse(pool.grow(10))
    if virtual:
        # Virtual memory keeps addresses: the old view is a valid prefix of the new one.
        test.assertEqual(old_index.ptr, target.index.ptr)
    else:
        # Ordinary arrays are reallocated; the superseded ones stay alive for stale graph replays.
        test.assertNotEqual(old_index.ptr, target.index.ptr)
        test.assertTrue(any(array is old_index for array in rows._superseded))
    np.testing.assert_array_equal(old_index.numpy(), np.arange(first))
    with test.assertRaisesRegex(MemoryError, "cannot back"):
        pool.grow(pool.bound + 1)
    test.assertEqual(pool.ready, held)


def test_pool_bound_is_the_hardware_limit(test, device):
    """Bound a pool by the rows the device could back for its fields; reservations follow the bound, which never rises.

    Warp arrays index int32 elements; CUDA virtual memory views each field as rows of bytes, so it has the same limit.
    """
    pool = ContactPool(device=device, rows=64)
    test.assertEqual(pool.bound, INDEX_LIMIT)
    narrow = pool.rows((ContactField("a", wp.int32), ContactField("b", wp.vec3)))
    virtual = narrow._mode == "virtual"
    test.assertEqual(virtual, wp.get_device(device).is_cuda and contact_pool._virtual_memory() is not None)
    test.assertEqual(pool.row_bytes, 16)
    test.assertEqual(pool.bound, device_row_limit(device, 16))
    test.assertEqual(narrow.reserve, pool.bound)
    wide = pool.rows((ContactField("keys", wp.int64, per_contact=2),))
    test.assertEqual(pool.row_bytes, 32)
    test.assertEqual(pool.bound, device_row_limit(device, 32, INDEX_LIMIT // 2))
    test.assertEqual(wide.reserve, pool.bound)
    ready = pool.ready
    with test.assertRaisesRegex(MemoryError, "cannot back"):
        pool.reserve(pool.bound + 1)
    test.assertEqual(pool.ready, ready)
    # A graph captured at the bound covers every row the pool may grow to, so the bound stays when a member goes.
    bound = pool.bound
    del wide
    test.assertEqual(pool.row_bytes, 16)
    test.assertEqual(pool.bound, bound)


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
    index_limit = min(field.row_limit() for member in members for field in member.fields)
    test.assertEqual(pool.bound, device_row_limit(device, pool.row_bytes, index_limit))
    if virtual:
        test.assertGreaterEqual(reserved, pool.bound * pool.row_bytes)
    else:
        test.assertEqual(reserved, mapped)
        test.assertEqual(mapped, pool.ready * pool.row_bytes)
    for member in members:
        test.assertGreaterEqual(member.reserve, pool.bound)
        for field in member.fields:
            test.assertLessEqual(field.length(member.reserve), INDEX_LIMIT)
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
        test.assertEqual(pool.generation, 1)
        test.assertTrue(pool.needs_recapture, "reallocated rows need graphs recaptured")
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
    view, ready = target.value, pool.ready
    free = mock.PropertyMock(return_value=64 << 20)
    with mock.patch.object(type(wp.get_device(device)), "free_memory", new=free):
        with test.assertRaisesRegex(MemoryError, r"to 67108864 rows needs [0-9.]+ MiB more device memory; 64\.0 MiB"):
            pool.grow(64 << 20)
    test.assertEqual(rows.rows, ready)
    test.assertIs(target.value, view)
    pool.grow(2 * ready)
    test.assertGreaterEqual(rows.rows, 2 * ready)


def test_pool_follower_grows_with_its_leader(test, device):
    """Grow a following pool with its leader: its rows stay at least the leader's."""
    leader = ContactPool(device=device, rows=100)
    follower = ContactPool(device=device, rows=50)
    target = type("Target", (), {})()
    # Owners hold their rows; the pool holds them weakly.
    rows = follower.rows((ContactField("value", wp.float32, 1.0),), on_rows=lambda rows: rows.publish(target))
    leader.follow(follower)
    test.assertGreaterEqual(follower.ready, 100)
    test.assertEqual(target.value.shape, (follower.ready,))
    leader.grow(4 * follower.ready)
    test.assertGreaterEqual(follower.ready, leader.ready)
    test.assertEqual(rows.rows, follower.ready)
    test.assertTrue(np.all(target.value.numpy() == 1.0))


def test_compatibility_budgets_are_silent_and_size_nothing(test, device):
    """Accept Newton main's rigid contact budget silently, resolved as on main; it changes no pool's rows or bound.

    ``CollisionPipeline(rigid_contact_max=...)``, ``CollisionPipeline.rigid_contact_max``,
    ``Model.rigid_contact_max`` and ``ModelBuilder.num_rigid_contacts_per_world`` keep Newton main's behavior.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # Model finalization's own deprecation of the joint-target layout is unrelated to contact budgets.
        warnings.filterwarnings("ignore", message="The legacy DOF-shaped joint_target_q", category=DeprecationWarning)
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
        rows = contacts.rigid_contact_max
        test.assertGreaterEqual(rows, contact_pool.MIN_READY_ROWS)
        test.assertEqual(pool.ready, reference_contacts._rigid_contact_pool.ready)
        test.assertEqual(pool.bound, reference_contacts._rigid_contact_pool.bound)
        model.rigid_contact_max = 7
        later = newton.CollisionPipeline(model)
        test.assertEqual(later.rigid_contact_max, 7, "the model's value without an argument")
        test.assertEqual(later.contacts().rigid_contact_max, contacts.rigid_contact_max)
        pipeline.collide(model.state(), contacts)
    test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), CONTACTS)
    test.assertEqual(contacts.rigid_contact_max, rows, "a budget of 5 holds all 48 contacts")


def test_collide_outside_capture_grows_before_reading(test, device):
    """Grow and rerun a collision pass outside capture that overflows its rows: no contact dropped, no flag.

    Ordinary arrays start at the 64 rows asked for and grow; CUDA virtual memory holds every row its first pages
    cover, which hold the contacts already.
    """
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
        reallocates = _reallocates(pipeline._contact_pool)
        first = contacts.rigid_contact_max
        test.assertEqual(first == 64, reallocates)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            pipeline.collide(model.state(), contacts)
        rows = sized_rows(demand, 64, pipeline._contact_pool.bound) if reallocates else first
    test.assertEqual(contacts.rigid_contact_max, rows)
    test.assertEqual(pipeline._sort_key_array.shape, (rows,))
    test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), demand)
    test.assertFalse(any(pipeline.narrow_phase.buffer_capacity_status().values()))
    np.testing.assert_array_equal(_records(contacts), _records(expected))
    test.assertEqual(pipeline.service_capacity(), reallocates, "only reallocated rows need a recapture")
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
        reallocates = _reallocates(pipeline._contact_pool)
        first = contacts.rigid_contact_max
        test.assertEqual(first == 8, reallocates)
        generation = contacts.contact_generation.numpy().copy()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            test.assertEqual(pipeline.service_capacity(resting), reallocates)
        test.assertEqual(contacts.rigid_contact_max, GROWN if reallocates else first)
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
        reallocates = _reallocates(pipeline._contact_pool)
        test.assertEqual(contacts.rigid_contact_max == 8, reallocates)
        test.assertEqual(pipeline.service_capacity(resting), reallocates)
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
        reallocates = _reallocates(overflowing.collision_pipeline._contact_pool)
        test.assertFalse(_service_contact_capacity(overflowing, overflowing.state_0))
        # Graphs whose contact launches follow the rows need no recapture; reallocated rows do, at once on overflow.
        test.assertEqual(overflowing.captures, [GROWN] if reallocates else [], "8 rows cannot hold 48 contacts")
        test.assertFalse(_service_contact_capacity(overflowing))
        test.assertFalse(_service_contact_capacity(overflowing, overflowing.state_0))
    test.assertEqual(overflowing.captures, [GROWN] if reallocates else [], "no growth, no recapture")
    with _small_pool_policy(64, 8):
        holding = example_with_graph()
        test.assertEqual(
            _service_contact_capacity(holding, holding.state_0), reallocates, "64 rows hold the 48 starting contacts"
        )
        test.assertGreaterEqual(holding.contacts.rigid_contact_max, GROWN)
        test.assertEqual(holding.captures, [])
        test.assertFalse(_service_contact_capacity(holding, recapture=reallocates))
    test.assertEqual(holding.captures, [GROWN] if reallocates else [], "recaptured after the first frame")


def test_first_mapping_holds_the_demand_from_the_first_replay(test, device):
    """Map the model's estimate before anything runs: the first replay holds every contact, and nothing grows.

    The solver, constructed before the pipeline, shares the model's pool, so the pipeline's first mapping sizes its
    per-contact buffers too. Measuring a state the estimate covers grows nothing.
    """
    model = _build_boxes(device)
    solver = newton.solvers.SolverFeatherPGS(model, pgs_iterations=4)
    pipeline = newton.CollisionPipeline(model)
    lifted = model.state()
    body_q = lifted.body_q.numpy()
    body_q[:, 2] += 1.0
    lifted.body_q.assign(body_q)
    test.assertFalse(pipeline.service_capacity(lifted), "a contact-free scene keeps the first mapping")
    held = pipeline.contacts().rigid_contact_max
    test.assertGreaterEqual(held, max(contact_pool.MIN_READY_ROWS, _estimate_rigid_contact_max(model)))
    test.assertEqual(solver.contact_slot.shape, (held,))

    worlds, boxes = 128, 8
    demand = 4 * worlds * boxes
    model = _build_boxes(device, worlds=worlds, boxes=boxes)
    solver = newton.solvers.SolverFeatherPGS(model, pgs_iterations=4)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()
    rows = contacts.rigid_contact_max
    test.assertGreaterEqual(rows, _estimate_rigid_contact_max(model))
    test.assertGreater(rows, contact_pool.HEADROOM * demand, "the estimate holds the demand with headroom")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        test.assertFalse(pipeline.service_capacity(state_0), "the first mapping holds the measured demand")
    test.assertEqual(contacts.rigid_contact_max, rows)
    test.assertIs(solver._contact_pool, pipeline._contact_pool)
    test.assertEqual(solver.contact_slot.shape, (rows,))
    test.assertEqual(solver.laws.friction_patches.capacity, rows)

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
        pool = first._contact_pool
        recapture = _reallocates(pool)
        test.assertEqual(first.service_capacity(model.state()), recapture)
        rows = second.contacts().rigid_contact_max
        # Ordinary arrays grow from 8 rows; CUDA virtual memory's first pages already hold the contacts.
        test.assertEqual(rows, GROWN if recapture else pool.ready)
        test.assertEqual(solver.contact_slot.shape, (rows,))
        test.assertEqual(second.service_capacity(), recapture, "the second pipeline learns of the growth too")
        test.assertFalse(first.service_capacity())
        test.assertFalse(second.service_capacity())
        test.assertEqual(other.contacts().rigid_contact_max == 8, recapture)


def test_pipeline_growth_without_overflow_matches_fixed(test, device):
    """Grow past half the ready rows without dropping contacts; every pass equals a larger buffer's."""
    model = _build_boxes(device)
    reference = newton.CollisionPipeline(_build_boxes(device))
    expected = reference.contacts()
    with _small_pool_policy(64, 8):
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        reallocates = _reallocates(pipeline._contact_pool)
        first = contacts.rigid_contact_max
        test.assertEqual(first == 64, reallocates)
        _collide_like_fixed(test, device, model, reference, expected, pipeline, contacts)
    test.assertEqual(contacts.rigid_contact_max, GROWN if reallocates else first)


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
            test.assertEqual(pipeline.service_capacity(), step == 0 and _reallocates(pipeline._contact_pool))


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
    pool = pipeline._contact_pool
    # Ordinary arrays grow once, from 64 rows to the demand; CUDA virtual memory once, when the contacts are
    # created, to every row its first pages cover, which hold the 48 contacts.
    reallocates = _reallocates(pool)
    rows = GROWN if reallocates else pool.ready
    test.assertEqual(pool.growth_count, 1)
    test.assertEqual(solver._contact_pool.ready, rows)
    test.assertEqual(solver.contact_slot.shape, (rows,))
    test.assertEqual(solver.laws.friction_patches.capacity, rows)
    test.assertEqual(contacts.rigid_contact_max, rows)
    solver.check_constraint_capacity()
    np.testing.assert_allclose(state.body_q.numpy(), reference.body_q.numpy(), rtol=0.0, atol=1.0e-5)
    np.testing.assert_allclose(state.body_qd.numpy(), reference.body_qd.numpy(), rtol=0.0, atol=1.0e-4)


def test_feather_pgs_follows_a_pass_that_grew(test, device):
    """Hand FeatherPGS the rows a collision pass grew to before it reads them, so no constraint is lost."""
    state, contacts, _, solver = _feather_pgs_run(device, 2, policy=_small_pool_policy(8, 8))
    solver.check_constraint_capacity()
    pool = solver._contact_pool
    test.assertEqual(contacts.rigid_contact_max, GROWN if _reallocates(pool) else pool.ready)
    test.assertEqual(pool.ready, contacts.rigid_contact_max)
    test.assertTrue(np.all(np.isfinite(state.body_q.numpy())))


def test_matching_and_warm_start_survive_growth(test, device):
    """Match the frame after a growth to the frame before it, so warm starts keep their history."""
    matched = {}

    def after_collide(step, pipeline, contacts):
        pool = pipeline._contact_pool
        count = int(contacts.rigid_contact_count.numpy()[0])
        matched[step] = (pool.growth_count, contacts.rigid_contact_match_index.numpy()[:count])
        if step == 0:
            # Grow between the first frame and the second, past the rows the first frame used.
            pool.grow(pool.ready + 1)

    _feather_pgs_run(device, 3, policy=_small_pool_policy(64, 8), warmstart=True, after_collide=after_collide)
    growths, match = matched[1]
    test.assertGreater(growths, matched[0][0], "the pool grew between the frames")
    test.assertEqual(match.shape, (CONTACTS,))
    test.assertTrue(np.all(match >= 0), f"resting contacts must match across growth: {match}")


@wp.kernel(enable_backward=False)
def _write_counted_rows(count: wp.array[wp.int32], value: int, rows: wp.array[wp.int32]):
    row = wp.tid()
    if row < count[0]:
        rows[row] = value + row


def test_bound_launch_follows_growth_past_mapped_pages(test, device):
    """Capture a pool launch once: growth maps pages past the first ones, and the next replay writes them.

    The launch is captured at the pool's bound and each replay runs it at the device count; the service-time
    growth maps more pages without synchronizing, and no recapture is needed.
    """
    pool = ContactPool(device=device, rows=1024)
    target = type("Target", (), {})()
    rows = pool.rows((ContactField("value", wp.int32, -1),), on_rows=lambda rows: rows.publish(target))
    if not pool.bindable:
        test.skipTest("contact launches bind with gpu_components' graph bridge only")
    count = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(_write_counted_rows, 1, inputs=[count, 0, target.value], device=device)
    with wp.ScopedCapture(device=device) as capture:
        pool.begin_updates()
        pool.launch(_write_counted_rows, pool.launch_rows(), inputs=[count, 7, target.value], extent=count)
    test.assertEqual(len(pool._unbound_captures), 0)
    first_mapping = pool.mapped_bytes()
    for size in (1000, 600_000, 3_000_000, 17):
        if size > pool.ready:
            test.assertFalse(pool.grow(size), "a bound graph needs no recapture")
        count.fill_(size)
        target.value.fill_(-1)
        wp.capture_launch(capture.graph)
        values = target.value.numpy()
        np.testing.assert_array_equal(values[:size], 7 + np.arange(size, dtype=np.int32))
        test.assertTrue(np.all(values[size:] == -1))
    test.assertGreater(pool.mapped_bytes(), first_mapping, "growth mapped pages past the first ones")
    test.assertGreaterEqual(rows.rows, 3_000_000)
    test.assertFalse(pool.launch_failed(), pool.launch_status())


def test_trim_releases_pages_while_a_bound_graph_replays(test, device):
    """Map big, trim while a captured graph replays, then grow past the trimmed rows: never recapture.

    A pool planned for 4M contacts maps them up front. Contact-free replays trim nothing: the demand has not been
    observed. Then each replay records a steady demand; once the high-water mark stayed within TRIM_BAND for
    TRIM_AFTER_PASSES recorded passes, the service lowers the ready rows in stream order and a later service unmaps
    the pages past them. Throughout, every replay writes exactly its counted rows and flags nothing. A demand past
    the trimmed rows' growth threshold then grows them again.
    """
    pool = ContactPool(device=device)
    target = type("Target", (), {})()
    rows = pool.rows((ContactField("value", wp.int32, -1),), on_rows=lambda rows: rows.publish(target))
    if not pool.bindable:
        test.skipTest("only CUDA virtual memory trims")
    pool.plan_first_mapping(4_000_000)
    test.assertGreaterEqual(pool.ready, 4_000_000)
    first = pool.mapped_bytes()
    demand, count = wp.zeros(1, dtype=wp.int32, device=device), wp.zeros(1, dtype=wp.int32, device=device)
    status = wp.zeros(1, dtype=wp.int32, device=device)
    with wp.ScopedCapture(device=device) as capture:
        wp.copy(count, demand)
        pool.record_demand(count, pool.ready_rows, status)
        pool.begin_updates()
        pool.launch(_write_counted_rows, pool.launch_rows(), inputs=[count, 7, target.value], extent=count)
    ready, mapped = [], []
    with mock.patch.object(contact_pool, "TRIM_AFTER_PASSES", 4):
        for size in [0] * 6 + [100_000] * 8 + [400_000] * 2 + [900_000] * 2:
            demand.fill_(size)
            target.value.fill_(-1)
            wp.capture_launch(capture.graph)
            # Reading the rows joins the replay; the trim itself joins nothing.
            values = target.value.numpy()
            np.testing.assert_array_equal(values[:size], 7 + np.arange(size, dtype=np.int32))
            test.assertTrue(np.all(values[size:] == -1))
            test.assertFalse(pool.service(), "a bound graph needs no recapture")
            ready.append(pool.ready)
            mapped.append(pool.mapped_bytes())
    test.assertEqual(pool.trim_count, 1)
    test.assertEqual(ready[:6], [ready[0]] * 6, "no trim before the first contact")
    trimmed = min(ready)
    test.assertGreaterEqual(trimmed, contact_pool.HEADROOM * 100_000, "a trim keeps HEADROOM rows per contact")
    test.assertLess(trimmed, 4_000_000)
    test.assertLess(min(mapped), first, "the trim returned pages")
    test.assertEqual(ready.index(trimmed), 10, "the service after TRIM_AFTER_PASSES steady passes trims")
    test.assertGreater(mapped[10], mapped[11], "the next service unmaps, once the replays before completed")
    test.assertGreaterEqual(rows.rows, contact_pool.HEADROOM * 900_000, "growth after the trim")
    test.assertEqual(int(status.numpy()[0]), 0, "no replay dropped a contact")
    test.assertFalse(pool.launch_failed(), pool.launch_status())


@wp.kernel(enable_backward=False)
def _snapshot_count(source: wp.array[wp.int32], target: wp.array[wp.int32]):
    target[0] = source[0]


@wp.kernel(enable_backward=False)
def _spin(value: wp.array[float], iterations: wp.array[wp.int32]):
    x = value[0]
    for _ in range(iterations[0]):
        x = x * 0.999999 + 1.0e-6
    value[0] = x


def _spin_iterations(device, milliseconds: float) -> int:
    """Return the iterations of :func:`_spin` that keep the device busy for about ``milliseconds``."""
    value, iterations = wp.zeros(1, dtype=float, device=device), wp.full(1, 1 << 20, dtype=wp.int32, device=device)
    wp.launch(_spin, 1, [value, iterations], device=device)
    start, end = wp.Event(device, enable_timing=True), wp.Event(device, enable_timing=True)
    wp.get_stream(device).record_event(start)
    wp.launch(_spin, 1, [value, iterations], device=device)
    wp.get_stream(device).record_event(end)
    wp.synchronize_device(device)
    return int((1 << 20) * milliseconds / max(wp.get_event_elapsed_time(start, end), 1.0e-3))


def test_trim_waits_for_a_replay_on_another_stream(test, device):
    """A trim unmaps only once a replay on another stream, which read the higher count, has finished writing.

    The graph, captured on a second stream, records its pass's demand, reads the ready rows as a graph updater
    does and writes those rows half a second later. While a replay runs, the service lowers the ready rows on its
    own stream and on the replay's (which the capture registered), with an event on each: the replay's stream's
    event completes after the replay, so its late writes stay mapped. Joining only the service stream's event
    instead unmaps under the replay, which faults with an illegal address.
    """
    pool = ContactPool(device=device)
    target = type("Target", (), {})()
    rows = pool.rows((ContactField("value", wp.int32, -1),), on_rows=lambda rows: rows.publish(target))
    if not pool.bindable:
        test.skipTest("only CUDA virtual memory trims")
    pool.plan_first_mapping(4_000_000)
    full, first = pool.ready, pool.mapped_bytes()
    reader = wp.Stream(device)
    demand = wp.full(1, 100_000, dtype=wp.int32, device=device)
    count, seen = wp.zeros(1, dtype=wp.int32, device=device), wp.zeros(1, dtype=wp.int32, device=device)
    spin, iterations = wp.zeros(1, dtype=float, device=device), wp.zeros(1, dtype=wp.int32, device=device)
    slow = _spin_iterations(device, 500.0)
    with wp.ScopedStream(reader), wp.ScopedCapture(device=device) as capture:
        wp.copy(count, demand)
        pool.record_demand(count, pool.ready_rows)
        wp.launch(_snapshot_count, 1, [pool.ready_rows, seen], device=device)
        wp.launch(_spin, 1, [spin, iterations], device=device)
        wp.launch(_write_counted_rows, full, [seen, 7, target.value], device=device)
    wp.synchronize_device(device)
    with mock.patch.object(contact_pool, "TRIM_AFTER_PASSES", 2):
        for _ in range(2):
            wp.capture_launch(capture.graph, stream=reader)
            wp.synchronize_stream(reader)
            pool.service()
        test.assertEqual(pool.trim_count, 0)
        iterations.fill_(slow)
        wp.synchronize_device(device)
        wp.capture_launch(capture.graph, stream=reader)
        time.sleep(0.1)  # the replay recorded its pass and read the full rows; it now spins
        pool.service()
        test.assertIsNotNone(pool._trim, "the service withdrew the rows")
        events = pool._trim[0]
        test.assertEqual(len(events), 2, "an event on the service's stream and one on the replay's")
        test.assertFalse(all(event.is_complete for event in events), "the replay still writes the rows it read")
        pool.service()
        test.assertEqual(pool.trim_count, 0, "no unmap before the replay finished")
        test.assertEqual(pool.mapped_bytes(), first)
        while not all(event.is_complete for event in events):
            time.sleep(0.005)
        pool.service()
    test.assertEqual(pool.trim_count, 1)
    test.assertLess(pool.mapped_bytes(), first)
    wp.synchronize_device(device)
    test.assertEqual(int(seen.numpy()[0]), full, "the replay wrote the rows of the higher count")
    trimmed = rows.rows
    np.testing.assert_array_equal(target.value.numpy(), 7 + np.arange(trimmed, dtype=np.int32))
    iterations.fill_(0)
    target.value.fill_(-1)
    wp.synchronize_device(device)
    wp.capture_launch(capture.graph, stream=reader)
    wp.synchronize_device(device)
    test.assertEqual(int(seen.numpy()[0]), trimmed, "a later replay reads the lower count")
    np.testing.assert_array_equal(target.value.numpy(), 7 + np.arange(trimmed, dtype=np.int32))


def test_conditional_bodies_record_their_own_updaters(test, device):
    """Each scope of a capture follows an updater of its own: a launch after a conditional body binds as before it.

    A collision pass inside a conditional node's body (a solver's iteration loop) records an updater there, which
    binds only launches in that body; a launch after the body binds to an updater of the graph around it.
    """
    pool = ContactPool(device=device, rows=1024)
    target = type("Target", (), {})()
    fields = tuple(ContactField(name, wp.int32, -1) for name in ("before", "inside", "after"))
    pool_rows = pool.rows(fields, on_rows=lambda rows: rows.publish(target))
    if not pool.bindable:
        test.skipTest("contact launches bind with gpu_components' graph bridge only")
    count, always = wp.full(1, 100, dtype=wp.int32, device=device), wp.ones(1, dtype=wp.int32, device=device)
    wp.launch(_write_counted_rows, 1, inputs=[count, 0, target.before], device=device)

    def body():
        pool.begin_updates()
        pool.launch(_write_counted_rows, pool.launch_rows(), inputs=[count, 2, target.inside], extent=count)

    with wp.ScopedCapture(device=device) as capture:
        pool.begin_updates()
        pool.launch(_write_counted_rows, pool.launch_rows(), inputs=[count, 1, target.before], extent=count)
        wp.capture_if(always, on_true=body)
        pool.launch(_write_counted_rows, pool.launch_rows(), inputs=[count, 3, target.after], extent=count)
    wp.capture_launch(capture.graph)
    for name, value in (("before", 1), ("inside", 2), ("after", 3)):
        values = getattr(target, name).numpy()
        np.testing.assert_array_equal(values[:100], value + np.arange(100, dtype=np.int32), name)
        test.assertTrue(np.all(values[100:] == -1), name)
    test.assertFalse(pool.launch_failed(), pool.launch_status())
    del pool_rows


def test_failed_closed_launches_are_capacity_failures(test, device):
    """A launch whose binding fails after its capture emitted it leaves a graph that skips its updater's launches.

    Every replay disables them and reports -117 (gpu_components fails the updater closed); the pool, the narrow
    phase and FeatherPGS report the skipped launches as a capacity failure instead of a silent zero launch.
    """
    model = _build_boxes(device)
    solver = newton.solvers.SolverFeatherPGS(model, pgs_iterations=4)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    pool = pipeline._contact_pool
    if not pool.bindable:
        test.skipTest("contact launches bind with gpu_components' graph bridge only")
    state_0, state_1, control = model.state(), model.state(), model.control()
    extra = wp.zeros(pool.ready, dtype=wp.int32, device=device)

    def simulate():
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, 1.0 / 120.0)

    simulate()  # load the modules outside capture
    wp.synchronize_device(device)
    with wp.ScopedCapture(device=device) as capture:
        solver.seed_double_buffer_events()
        simulate()
        # The updater the next launch binds to (each loads its own handle of the bridge library).
        library = pool._updates_for(contact_pool._capture_owner(wp.get_device(device)))._library
        with mock.patch.object(library, "bind_node", return_value=-999), test.assertRaisesRegex(RuntimeError, "bind"):
            pool.launch(
                _write_counted_rows,
                pool.launch_rows(),
                inputs=[contacts.rigid_contact_count, 1, extra],
                extent=contacts.rigid_contact_count,
            )
    test.assertFalse(pool.launch_failed(), "nothing replayed yet")
    wp.capture_launch(capture.graph)
    status = pool.launch_status()
    test.assertGreater(status["failed_closed"], 0, status)
    test.assertTrue(np.all(extra.numpy() == 0), "the failed launch did no work")
    test.assertTrue(pipeline.narrow_phase.buffer_capacity_status()["contact_launches"])
    test.assertTrue(solver.constraint_capacity_status()["contact_launches"])
    with test.assertRaisesRegex(RuntimeError, "did not run in their graph's last replay"):
        solver.check_constraint_capacity()
    with test.assertRaisesRegex(RuntimeError, "contact_launches"):
        pipeline.narrow_phase.check_buffer_capacity()


def test_trim_and_growth_follow_a_feather_pgs_graph(test, device):
    """Capture collide and FeatherPGS once with the model's estimate mapped; trim it, then grow past it while it replays.

    Half the boxes start in the air: once the demand settled, the service trims the first mapping (the model's
    estimate) to HEADROOM rows per contact while the graph keeps replaying. Then every box is put on the ground: the
    demand doubles within the trimmed rows, the service grows them, and no replay drops a contact or needs a recapture.
    """
    worlds, boxes = 12288, 6
    model = _build_resting_boxes(device, worlds, boxes)
    solver = newton.solvers.SolverFeatherPGS(model, pgs_iterations=8)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    pool = pipeline._contact_pool
    if not pool.bindable:
        test.skipTest("only CUDA virtual memory trims")
    test.assertGreaterEqual(contacts.rigid_contact_max, _estimate_rigid_contact_max(model), "the estimate, mapped")
    state_0, state_1, control = model.state(), model.state(), model.control()

    def simulate():
        for source, target in ((state_0, state_1), (state_1, state_0)):
            pipeline.collide(source, contacts)
            solver.step(source, target, control, contacts, 1.0 / 120.0)

    resting = state_0.joint_q.numpy()
    lifted = resting.reshape(worlds, boxes, 7).copy()
    lifted[:, 1::2, 2] += 50.0

    def place(joint_q):
        state_0.joint_q.assign(joint_q.reshape(-1))
        state_0.joint_qd.zero_()
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

    place(lifted)
    simulate()  # load the modules outside capture
    place(lifted)
    wp.synchronize_device(device)
    with wp.ScopedCapture(device=device) as capture:
        solver.seed_double_buffer_events()
        simulate()
    first = pool.mapped_bytes()
    counts, recaptures, overflowed, trimmed_bytes = [], 0, [], None
    with mock.patch.object(contact_pool, "TRIM_AFTER_PASSES", 6):
        for frame in range(40):
            if frame == 20:
                test.assertEqual(pool.trim_count, 1, "trimmed once the demand settled")
                trimmed_bytes, trimmed_rows, growths = pool.mapped_bytes(), pool.ready, pool.growth_count
                place(resting)
            wp.capture_launch(capture.graph)
            wp.synchronize_device(device)
            counts.append(int(contacts.rigid_contact_count.numpy()[0]))
            if pipeline.narrow_phase.buffer_capacity_status(clear=True)["contacts"]:
                overflowed.append(frame)
            recaptures += int(pipeline.service_capacity())
    demand = 4 * worlds * boxes
    test.assertEqual(overflowed, [], "no replay dropped a contact")
    test.assertEqual(recaptures, 0)
    test.assertEqual(counts[19], demand // 2)
    test.assertEqual(counts[-1], demand)
    test.assertLess(trimmed_bytes, first, "the trim returned pages of the first mapping")
    test.assertGreaterEqual(trimmed_rows, contact_pool.HEADROOM * demand // 2)
    test.assertGreater(demand, contact_pool.GROW_THRESHOLD * trimmed_rows, "the doubled demand grows the rows")
    test.assertGreater(pool.growth_count, growths, "growth after the trim")
    test.assertGreaterEqual(contacts.rigid_contact_max, contact_pool.HEADROOM * demand)
    solver.check_constraint_capacity()
    body_q = state_0.body_q.numpy()
    test.assertTrue(np.all(np.isfinite(body_q)))
    np.testing.assert_allclose(body_q[:, 2], 0.1, atol=2.0e-3, err_msg="every box rests on the ground")


def _build_resting_boxes(device, worlds, boxes):
    """Build ``worlds`` worlds of ``boxes`` boxes resting side by side on one ground plane, four contacts each."""
    template = newton.ModelBuilder(up_axis=newton.Axis.Z)
    for i in range(boxes):
        body = template.add_body(xform=wp.transform(wp.vec3(1.0 * i, 0.0, 0.1), wp.quat_identity()))
        template.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.add_ground_plane()
    builder.replicate(template, worlds, spacing=(8.0, 8.0, 0.0))
    return builder.finalize(device=device)


def test_bound_graph_follows_growth_without_recapture(test, device):
    """Capture collide and FeatherPGS once; let the demand jump well past the first rows; never recapture.

    The graph is captured with every box in the air, then replayed with the boxes resting: the first replay's
    contacts outnumber the rows the first pages hold, so it drops the excess and flags it; the service maps more
    rows without synchronizing, and every later replay of the same graph holds every contact. Without CUDA virtual
    memory, rows are reallocated and the service asks for the recapture instead.
    """
    worlds, boxes = 12288, 6
    with _small_pool_policy(8, 8):
        model = _build_resting_boxes(device, worlds, boxes)
        pipeline = newton.CollisionPipeline(model)
        solver = newton.solvers.SolverFeatherPGS(model, pgs_iterations=8)
        contacts = pipeline.contacts()
    pool = pipeline._contact_pool
    state_0, state_1, control = model.state(), model.state(), model.control()

    def simulate():
        # Two substeps keep the newest state in state_0; FeatherPGS integrates joint coordinates.
        for source, target in ((state_0, state_1), (state_1, state_0)):
            pipeline.collide(source, contacts)
            solver.step(source, target, control, contacts, 1.0 / 120.0)

    def capture():
        # FeatherPGS seeds its double-buffer events in each capture, after the queued clears finish.
        wp.synchronize_device(device)
        with wp.ScopedCapture(device=device) as recording:
            solver.seed_double_buffer_events()
            simulate()
        return recording.graph

    # Warm up in the air (module loads, lazy setup): no contacts, so the rows stay at 8.
    joint_q = state_0.joint_q.numpy()
    lifted = joint_q.reshape(-1, 7).copy()
    lifted[:, 2] += 50.0
    state_0.joint_q.assign(lifted.reshape(-1))
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    simulate()
    state_0.joint_q.assign(joint_q)
    state_0.joint_qd.zero_()
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    first = contacts.rigid_contact_max
    graph = capture()
    reallocates = _reallocates(pool)
    test.assertEqual(first == 8, reallocates)
    test.assertGreater(4 * worlds * boxes, 2 * first, "the demand jumps well past the first rows")
    test.assertEqual(
        len(pool._unbound_captures), int(reallocates), "every contact launch of the capture follows the rows"
    )
    first_mapping = pool.mapped_bytes()
    growths = pool.growth_count
    overflowed, recaptures, counts = [], 0, []
    for frame in range(60):
        wp.capture_launch(graph)
        wp.synchronize_device(device)
        if pipeline.narrow_phase.buffer_capacity_status(clear=True)["contacts"]:
            overflowed.append(frame)
        counts.append(int(contacts.rigid_contact_count.numpy()[0]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if pipeline.service_capacity():
                recaptures += 1
                graph = capture()
    test.assertEqual(overflowed, [0], "only the replay where the demand first passed the rows drops contacts")
    test.assertGreater(pool.growth_count, growths)
    test.assertEqual(recaptures, pool.growth_count - growths if reallocates else 0, "reallocated rows: every growth")
    test.assertEqual(counts[-1], 4 * worlds * boxes, "every box rests on the ground, every contact held")
    test.assertTrue(all(count == 4 * worlds * boxes for count in counts[1:]), "no contact dropped after the first")
    test.assertGreater(pool.mapped_bytes(), first_mapping)
    test.assertGreaterEqual(contacts.rigid_contact_max, counts[-1])
    solver.check_constraint_capacity()
    body_q = state_0.body_q.numpy()
    test.assertTrue(np.all(np.isfinite(body_q)))
    np.testing.assert_allclose(body_q[:, 2], 0.1, atol=2.0e-3, err_msg="the pile settles on the ground")


def test_sorted_matched_graph_follows_growth(test, device):
    """Sort and match contacts in a graph captured once while the demand grows past the first rows.

    A deterministic pipeline that matches contacts is captured with every box in the air, then replayed with the
    boxes resting: the first replay drops the contacts past the first rows and flags it, the service maps more
    rows, and every later replay sorts them as an eager pipeline does and matches each to the replay before.
    Without CUDA virtual memory, growth reallocates the rows and the service asks for the recapture instead.
    """
    worlds, boxes = 12288, 6
    options = {"deterministic": True, "contact_matching": "latest"}
    with _small_pool_policy(8, 8):
        model = _build_resting_boxes(device, worlds, boxes)
        pipeline = newton.CollisionPipeline(model, **options)
        contacts = pipeline.contacts()
    reference = newton.CollisionPipeline(_build_resting_boxes(device, worlds, boxes), **options)
    expected = reference.contacts()
    pool = pipeline._contact_pool
    state, resting = model.state(), model.state()
    lifted = state.joint_q.numpy().reshape(-1, 7).copy()
    lifted[:, 2] += 50.0
    state.joint_q.assign(lifted.reshape(-1))
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    pipeline.collide(state, contacts)  # warm up in the air: no contacts

    def capture():
        with wp.ScopedCapture(device=device) as recording:
            pipeline.collide(state, contacts)
        return recording.graph

    graph = capture()
    reallocates = _reallocates(pool)
    test.assertEqual(len(pool._unbound_captures), int(reallocates), "sorting and matching follow the rows")
    first = contacts.rigid_contact_max
    state.assign(resting)
    total = 4 * worlds * boxes
    test.assertGreater(total, first, "the demand passes the first rows")
    overflowed, recaptures = [], 0
    for frame in range(3):
        wp.capture_launch(graph)
        wp.synchronize_device(device)
        if pipeline.narrow_phase.buffer_capacity_status(clear=True)["contacts"]:
            overflowed.append(frame)
        count = int(contacts.rigid_contact_count.numpy()[0])
        if frame > 0:
            test.assertEqual(count, total)
            reference.collide(resting, expected)
            for name in ("shape0", "shape1", "point0", "point1", "normal"):
                np.testing.assert_array_equal(
                    getattr(contacts, f"rigid_contact_{name}").numpy()[:count],
                    getattr(expected, f"rigid_contact_{name}").numpy()[:count],
                    err_msg=f"replay {frame} sorts {name} as an eager pipeline does",
                )
        if frame == 2:
            match = contacts.rigid_contact_match_index.numpy()[:count]
            np.testing.assert_array_equal(match, np.arange(count), "each contact matches itself a replay before")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if pipeline.service_capacity():
                recaptures += 1
                graph = capture()
    test.assertEqual(overflowed, [0], "only the replay where the demand first passed the rows drops contacts")
    test.assertEqual(recaptures, int(reallocates))
    test.assertFalse(pool.launch_failed(), pool.launch_status())


def test_debug_clear_follows_growth(test, device):
    """Clear every ready row of a conservatively cleared buffer, in a graph captured before the rows grew."""
    pool = ContactPool(device=device, rows=1024)
    contacts = newton.Contacts(0, 0, device=device, rigid_contact_pool=pool, clear_buffers=True)
    contacts.clear()
    if not pool.bindable:
        test.skipTest("a capture follows the rows with gpu_components' graph bridge only")
    with wp.ScopedCapture(device=device) as capture:
        contacts.clear()
    test.assertEqual(len(pool._unbound_captures), 0)
    pool.grow(3 * pool.ready + 1000)
    rows = contacts.rigid_contact_max
    contacts.rigid_contact_shape0.fill_(7)
    contacts.rigid_contact_force.fill_(wp.vec3(1.0))
    wp.capture_launch(capture.graph)
    test.assertTrue(np.all(contacts.rigid_contact_shape0.numpy()[:rows] == -1), "every ready row is cleared")
    test.assertTrue(np.all(contacts.rigid_contact_force.numpy()[:rows] == 0.0))


def test_reserve_grows_now_or_at_the_next_service(test, device):
    """Grow to a reservation outside capture; defer it, with a warning, during capture; count each growth once."""
    with _small_pool_policy(8, 8):
        pool = ContactPool(device=device, rows=8)
        target = type("Target", (), {})()
        rows = pool.rows((ContactField("value", wp.int32, -1),), on_rows=lambda rows: rows.publish(target))
        virtual = rows._mode == "virtual"
        pool.settle()
        first, settled = pool.ready, pool.generation
        # CUDA virtual memory holds every row its first pages cover; ordinary arrays the 8 asked for.
        test.assertEqual(first == 8, not virtual)
        test.assertEqual(settled, int(virtual))
        test.assertFalse(pool.reserve(first))
        wanted = first + 92
        test.assertEqual(pool.reserve(wanted), not virtual)
        test.assertGreaterEqual(pool.ready, contact_pool.reserved_rows(wanted, pool.bound), "rounded up")
        if not virtual:
            test.assertEqual(pool.ready, 104, "rounded up to the row quantum")
        test.assertEqual(rows.rows, pool.ready)
        test.assertEqual(pool.generation, settled + 1)
        test.assertFalse(pool.service(), "nothing left to grow")
        with test.assertRaisesRegex(MemoryError, "cannot back"):
            pool.reserve(pool.bound + 1)
        if not wp.get_device(device).is_cuda:
            return
        held = pool.ready
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with wp.ScopedCapture(device=device):
                test.assertFalse(pool.reserve(held + 200))
        test.assertTrue(any("reserved during CUDA graph capture" in str(w.message) for w in caught))
        test.assertEqual(pool.ready, held, "no growth during capture")
        # The writer's capture is gone, so only reallocated rows ask for a recapture.
        test.assertEqual(pool.service(), not virtual)
        test.assertGreaterEqual(pool.ready, held + 200)
        test.assertEqual(pool.generation, settled + 2)
        test.assertEqual(target.value.shape, (pool.ready,))
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
    # Ordinary arrays start at 8 rows; CUDA virtual memory's first pages hold naconmax already.
    reallocates = _reallocates(pipeline._contact_pool)
    first = contacts.rigid_contact_max
    test.assertEqual(first == 8, reallocates)
    test.assertGreater(naconmax, 8)
    test.assertGreater(nacon, 8, "MJWarp must hold more contacts than the initial rows")
    solver.update_contacts(reference, state_0)
    solver.update_contacts(contacts, state_0)
    test.assertEqual(contacts.rigid_contact_max, naconmax if reallocates else first)
    test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), nacon)
    np.testing.assert_array_equal(_mujoco_records(contacts), _mujoco_records(reference))
    test.assertFalse(any(solver.contact_capacity_status().values()))
    wp.synchronize_device(device)
    test.assertEqual(pipeline.service_capacity(), reallocates, "no graph sized the rows")
    test.assertFalse(pipeline.service_capacity())

    if not wp.get_device(device).is_cuda or not reallocates:
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
    test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 8, "the count holds the rows the buffer kept")
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
    reallocates = _reallocates(pipeline._contact_pool)
    first = contacts.rigid_contact_max
    # Ordinary arrays start at 64 rows; CUDA virtual memory holds every row its first pages cover.
    test.assertEqual(first == 64, reallocates)
    test.assertGreater(naconmax, 64)
    control = model.control()
    for step in range(4):
        for run_pipeline, run_contacts, run_solver, state_0, state_1 in runs:
            run_pipeline.collide(state_0, run_contacts)
            run_solver.step(state_0, state_1, control, run_contacts, 1.0 / 240.0)
            run_solver.update_contacts(run_contacts, state_1)
            state_0.assign(state_1)
            wp.synchronize_device(device)
            test.assertEqual(
                run_pipeline.service_capacity(),
                run_pipeline is pipeline and step == 0 and reallocates,
            )
    test.assertEqual(
        contacts.rigid_contact_max,
        GROWN if reallocates else first,
        "grown from the demand of 48 contacts, not naconmax",
    )
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
    test.assertEqual(len(graphs), 2 if reallocates else 1, "one growth from demand, then one recapture")
    test.assertEqual(contacts.rigid_contact_max == GROWN, reallocates)
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
    test.assertEqual(contacts.reserve_rigid_contacts(4 * rows), _reallocates(contacts._rigid_contact_pool))
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
        first = contacts.rigid_contact_max
        reallocates = _reallocates(contacts._rigid_contact_pool)
        # Ordinary arrays start at 3 rows; CUDA virtual memory holds every row its first pages cover.
        test.assertEqual(first < capacity, reallocates)
        state_0, state_1 = model.state(), model.state()
        solver.step(state_0, state_1, None, None, 1.0 / 240.0)
        solver.update_contacts(contacts, state_1)
    test.assertEqual(contacts.rigid_contact_max, capacity if reallocates else first)
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
    test.assertEqual(contacts.reserve_rigid_contacts(4 * rows), _reallocates(contacts._rigid_contact_pool))
    grown = contacts.rigid_contact_max
    test.assertGreaterEqual(grown, 4 * rows)
    pipeline.collide(state_1, contacts)
    solver.step(state_1, state_0, control, contacts, 1.0 / 240.0)
    test.assertEqual(solver.body_body_contact_penalty_k.shape, (grown,))
    test.assertEqual(solver._rigid_contact_body0.shape, (grown,))
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
        reallocates = _reallocates(contacts._rigid_contact_pool)
        first = contacts.rigid_contact_max
        test.assertEqual(first == 8, reallocates)
        test.assertEqual(filtered.rigid_contact_max, first)
        state = model.state()
        pipeline.collide(state, contacts)
    rows = GROWN if reallocates else first
    test.assertEqual(contacts.rigid_contact_max, rows)
    test.assertEqual(filtered.rigid_contact_max, rows)
    test.assertIs(coupled.entry_contacts("A", contacts), filtered)
    np.testing.assert_array_equal(_records(filtered), _records(contacts))


def test_contact_kinematics_outputs_hold_the_rows(test, device):
    """Evaluate contact kinematics into outputs of at least the contacts' rows, before and after a growth."""
    model = _build_boxes(device)
    state = model.state()
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    rows = contacts.rigid_contact_max
    distance = wp.empty(rows, dtype=float, device=device)
    pipeline.collide(state, contacts)
    newton.eval_rigid_contact_kinematics(model, state, contacts, out_distance=distance)
    before = np.sort(distance.numpy()[:CONTACTS])
    test.assertEqual(contacts.reserve_rigid_contacts(4 * rows), _reallocates(contacts._rigid_contact_pool))
    # The outputs must hold the rows the buffer holds now, at least the 4 * rows reserved.
    distance = wp.empty(contacts.rigid_contact_max, dtype=float, device=device)
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
        "test_first_mapping_holds_the_demand_from_the_first_replay",
        test_first_mapping_holds_the_demand_from_the_first_replay,
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
    ("test_bound_launch_follows_growth_past_mapped_pages", test_bound_launch_follows_growth_past_mapped_pages),
    ("test_trim_releases_pages_while_a_bound_graph_replays", test_trim_releases_pages_while_a_bound_graph_replays),
    ("test_trim_waits_for_a_replay_on_another_stream", test_trim_waits_for_a_replay_on_another_stream),
    ("test_conditional_bodies_record_their_own_updaters", test_conditional_bodies_record_their_own_updaters),
    ("test_failed_closed_launches_are_capacity_failures", test_failed_closed_launches_are_capacity_failures),
    ("test_trim_and_growth_follow_a_feather_pgs_graph", test_trim_and_growth_follow_a_feather_pgs_graph),
    ("test_bound_graph_follows_growth_without_recapture", test_bound_graph_follows_growth_without_recapture),
    ("test_sorted_matched_graph_follows_growth", test_sorted_matched_graph_follows_growth),
    ("test_debug_clear_follows_growth", test_debug_clear_follows_growth),
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
