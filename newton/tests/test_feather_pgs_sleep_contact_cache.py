# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Capacity and geometry ownership for the experimental sleeping contact cache."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.sleep_contact_cache import _GEOMETRY_FIELDS, SleepContactCache


@wp.kernel(enable_backward=False)
def _append_one(count: wp.array[int], point_id: wp.array[int]):
    slot = wp.atomic_add(count, 0, 1)
    if slot >= 0 and slot < point_id.shape[0]:
        point_id[slot] = 999


def _fixture(device):
    contacts = newton.Contacts(7, 0, device=device, clear_buffers=True, requested_attributes={"force"})
    cache = SleepContactCache(contacts)
    for index, name in enumerate(_GEOMETRY_FIELDS):
        array = getattr(contacts, "rigid_contact_" + name)
        values = array.numpy()
        values[:] = (np.arange(values.size).reshape(values.shape) + 10 * index).astype(values.dtype)
        array.assign(values)
    contacts.rigid_contact_shape0.assign(np.array((0, 0, 0, 1, 3, 0, 0), np.int32))
    contacts.rigid_contact_shape1.assign(np.array((1, 3, 4, 2, 4, 5, 99), np.int32))
    contacts.rigid_contact_count.fill_(7)
    cache.held_force.assign(np.arange(21, dtype=np.float32).reshape(7, 3))
    held = cache.held_force.numpy()
    held[1] = 0.0  # Valid zero-load geometry must not disappear.
    cache.held_force.assign(held)
    cache.held_valid.assign(np.array((1, 1, 0, 1, 1, 1, 1), np.int32))
    shape_body = wp.array((0, 1, 2, -1, 3, 4), dtype=wp.int32, device=device)
    body_component = wp.array((0, 1, 2, -1, -2), dtype=wp.int32, device=device)
    asleep = wp.array((1, 1, 0), dtype=wp.int32, device=device)
    return contacts, cache, shape_body, body_component, asleep


class TestSleepContactCacheCPU(unittest.TestCase):
    def test_exact_capacity(self):
        """Allocate the history and current metadata at the original raw capacity."""
        contacts = newton.Contacts(7, 0, device="cpu")
        cache = SleepContactCache(contacts)
        self.assertEqual(cache.capacity, 7)
        for array in (cache.held_force, cache.held_valid, cache.source_index, cache.history.point0):
            self.assertEqual(array.shape, (7,))

    def test_canonical_geometry_force_and_fresh_append(self):
        """Preserve every dormant record field and append fresh contacts in the same capacity."""
        contacts, cache, shape_body, components, asleep = _fixture("cpu")
        expected = {name: getattr(contacts, "rigid_contact_" + name).numpy().copy() for name in _GEOMETRY_FIELDS}
        force, valid = cache.held_force.numpy(), cache.held_valid.numpy()
        cache.snapshot(contacts)
        self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 7)
        np.testing.assert_array_equal(cache.held_valid.numpy(), 0)
        np.testing.assert_array_equal(cache.source_index.numpy(), -1)
        contacts.clear()
        contacts.force.fill_(wp.spatial_vector(17.0, 17.0, 17.0, 17.0, 17.0, 17.0))
        cache.emit(contacts, shape_body, components, asleep)
        self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 3)
        sources = cache.source_index.numpy()[:3]
        np.testing.assert_array_equal(np.sort(sources), (0, 1, 2))
        for name in _GEOMETRY_FIELDS:
            np.testing.assert_array_equal(
                getattr(contacts, "rigid_contact_" + name).numpy()[:3], expected[name][sources]
            )
        np.testing.assert_array_equal(cache.held_force.numpy()[:3], force[sources])
        np.testing.assert_array_equal(cache.held_valid.numpy()[:3], valid[sources])
        np.testing.assert_array_equal(contacts.force.numpy(), 17.0)
        wp.launch(
            _append_one, dim=1, inputs=[contacts.rigid_contact_count, contacts.rigid_contact_point_id], device="cpu"
        )
        self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 4)
        self.assertEqual(int(contacts.rigid_contact_point_id.numpy()[3]), 999)
        self.assertEqual(int(cache.source_index.numpy()[3]), -1)
        self.assertEqual(int(cache.held_valid.numpy()[3]), 0)
        cache.check_capacity(contacts)

    def test_empty_wake_and_regrow(self):
        """Refresh history every cycle without resurrecting awake or stale records."""
        contacts, cache, shape_body, components, asleep = _fixture("cpu")
        cache.snapshot(contacts)
        contacts.clear()
        asleep.zero_()
        cache.emit(contacts, shape_body, components, asleep)
        self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 0)
        cache.snapshot(contacts)
        contacts.clear()
        asleep.fill_(1)
        cache.emit(contacts, shape_body, components, asleep)
        self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 0)
        contacts.rigid_contact_shape0.fill_(0)
        contacts.rigid_contact_shape1.fill_(3)
        contacts.rigid_contact_count.fill_(1)
        cache.held_valid.fill_(1)
        cache.snapshot(contacts)
        contacts.clear()
        cache.emit(contacts, shape_body, components, asleep)
        self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 1)
        self.assertEqual(int(cache.held_valid.numpy()[0]), 1)
        cache.check_capacity(contacts)

    def test_invalid_history_and_combined_overflow(self):
        """Reject invalid counts without treating a stored prefix as a complete history."""
        for raw_count in (-1, 8):
            with self.subTest(raw_count=raw_count):
                contacts, cache, shape_body, components, asleep = _fixture("cpu")
                contacts.rigid_contact_count.fill_(raw_count)
                cache.snapshot(contacts)
                contacts.clear()
                cache.emit(contacts, shape_body, components, asleep)
                self.assertEqual(int(cache.status.numpy()[0]), 1)
                self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 0)
                with self.assertRaises(RuntimeError):
                    cache.check_capacity(contacts)
        contacts, cache, shape_body, components, asleep = _fixture("cpu")
        cache.snapshot(contacts)
        contacts.clear()
        contacts.rigid_contact_count.fill_(6)  # Adversarial competing reservations share the original bound.
        cache.emit(contacts, shape_body, components, asleep)
        self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 9)
        self.assertEqual(int(cache.status.numpy()[0]), 2)
        with self.assertRaises(RuntimeError):
            cache.check_capacity(contacts)
        cache.snapshot(contacts)
        contacts.clear()
        cache.emit(contacts, shape_body, components, asleep)
        self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 0)

    def test_unsupported_modes_and_zero_capacity(self):
        """Decline altered geometry contracts while allowing caller-owned extended forces."""
        for options in (
            {"requires_grad": True},
            {"per_contact_shape_properties": True},
            {"contact_matching": True},
            {"contact_matching": True, "contact_report": True},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                SleepContactCache(newton.Contacts(1, 0, device="cpu", **options))
        for attribute in (
            "rigid_contacts_pair_sorted",
            "rigid_contacts_body_pair_reduced",
            "rigid_contacts_body_pair_reduced_capture",
        ):
            contacts = newton.Contacts(1, 0, device="cpu")
            setattr(contacts, attribute, True)
            with self.subTest(attribute=attribute), self.assertRaises(ValueError):
                SleepContactCache(contacts)
        contacts = newton.Contacts(0, 0, device="cpu", requested_attributes={"force"})
        cache = SleepContactCache(contacts)
        empty = wp.zeros(0, dtype=wp.int32, device="cpu")
        cache.snapshot(contacts)
        contacts.clear()
        cache.emit(contacts, empty, empty, empty)
        cache.check_capacity(contacts)
        self.assertEqual(cache.history.point0.shape, (0,))


class TestSleepContactCacheCUDA(unittest.TestCase):
    def test_native_captured_history_refresh(self):
        """Replay snapshot-clear-emit with changing asleep membership and canonical inputs."""
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("Actual cache capture requires CUDA")
        for device in devices:
            with self.subTest(device=str(device)):
                contacts, cache, shape_body, components, asleep = _fixture(device)
                cache.snapshot(contacts)
                contacts.clear()
                cache.emit(contacts, shape_body, components, asleep)
                with wp.ScopedCapture(device=device) as capture:
                    cache.snapshot(contacts)
                    contacts.clear()
                    cache.emit(contacts, shape_body, components, asleep)
                wp.capture_launch(capture.graph)
                self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 3)
                cache.check_capacity(contacts)
                asleep.zero_()
                wp.capture_launch(capture.graph)
                self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 0)
                asleep.fill_(1)
                wp.capture_launch(capture.graph)
                self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 0)
                contacts.rigid_contact_shape0.fill_(0)
                contacts.rigid_contact_shape1.fill_(3)
                contacts.rigid_contact_count.fill_(1)
                contacts.rigid_contact_point_id.fill_(101)
                wp.capture_launch(capture.graph)
                self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 1)
                self.assertEqual(int(contacts.rigid_contact_point_id.numpy()[0]), 101)
                cache.check_capacity(contacts)


if __name__ == "__main__":
    unittest.main()
