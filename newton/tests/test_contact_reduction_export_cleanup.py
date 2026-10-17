# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check entry retirement without changing reduced contact publication."""

import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.geometry.contact_data import ContactData
from newton._src.geometry.contact_reduction_global import (
    EXPORT_REDUCED_CONTACTS_BLOCK_DIM,
    VALUES_PER_KEY,
    GlobalContactReducer,
    GlobalContactReducerData,
    create_export_reduced_contacts_kernel,
    export_contact_to_buffer,
    make_contact_key,
    reduction_value_index,
    reduction_value_index_np,
)
from newton._src.geometry.hashtable import hashtable_find_or_insert
from newton._src.geometry.narrow_phase import NarrowPhase, write_contact_simple
from newton._src.sim.collide import write_contact
from newton.tests import test_heightfield_primitive_query as primitive


@wp.func
def _count_contact(contact: ContactData, output: wp.array[int], output_index: int):
    wp.atomic_add(output, 0, 1)


@wp.struct
class _ContactOutput:
    count: wp.array[int]
    fingerprint: wp.array[int]
    pair: wp.array[wp.vec2i]
    position: wp.array[wp.vec3]
    normal: wp.array[wp.vec3]
    depth: wp.array[float]


@wp.func
def _record_contact(contact: ContactData, output: _ContactOutput, output_index: int):
    index = wp.atomic_add(output.count, 0, 1)
    if index < output.position.shape[0]:
        output.fingerprint[index] = contact.sort_sub_key
        output.pair[index] = wp.vec2i(contact.shape_a, contact.shape_b)
        output.position[index] = contact.contact_point_center
        output.normal[index] = contact.contact_normal_a_to_b
        output.depth[index] = contact.contact_distance


@wp.kernel
def _seed_entries(data: GlobalContactReducerData, enabled: wp.array[int]):
    if enabled[0] == 0:
        return
    for index in range(5):
        position = wp.vec3(float(index), 0.02, 0.03)
        fingerprint = 10 * (index + 1)
        if index == 0:
            position = wp.vec3(0.01, 0.02, 0.03)
            fingerprint = 30
        elif index == 1:
            position = wp.vec3(0.010000000707805157, 0.02, 0.03)
            fingerprint = 10
        elif index == 2:
            fingerprint = 20
        export_contact_to_buffer(0, 1, position, wp.vec3(0.0, 1.0, 0.0), -0.01, fingerprint, data)
    for bin_index in range(4):
        entry = hashtable_find_or_insert(make_contact_key(0, 1, bin_index), data.ht_keys, data.ht_active_slots)
        for slot in range(wp.static(VALUES_PER_KEY)):
            contact_id = int(0)
            if bin_index == 0:
                if slot == 0 or slot == 2:
                    contact_id = 1
                elif slot == 1:
                    contact_id = 2
                elif slot == 3:
                    contact_id = 3
                elif slot == 5:
                    contact_id = 4
            elif bin_index == 1:
                if slot < 4:
                    contact_id = 3 + slot % 2
            elif bin_index == 2:
                contact_id = 5
            value = wp.uint64(contact_id)
            if bin_index == 0 and slot == 6:
                # Nonzero winner with reserved ID zero must never reach the writer.
                value = wp.uint64(1) << wp.uint64(32)
            data.ht_values[reduction_value_index(entry, slot)] = value


@wp.kernel
def _overflow_buffer(data: GlobalContactReducerData):
    for index in range(5):
        export_contact_to_buffer(0, 1, wp.vec3(float(index), 0.0, 0.0), wp.vec3(0.0, 1.0, 0.0), -0.01, index, data)


class _Fixture:
    def __init__(self, device, *, deterministic=False, capacity=16):
        self.device = wp.get_device(device)
        self.reducer = GlobalContactReducer(capacity, device=device, deterministic=deterministic)
        self.enabled = wp.ones(1, dtype=int, device=device)
        self.output = _ContactOutput()
        self.output.count = wp.zeros(1, dtype=int, device=device)
        self.output.fingerprint = wp.zeros(32, dtype=int, device=device)
        self.output.pair = wp.zeros(32, dtype=wp.vec2i, device=device)
        self.output.position = wp.zeros(32, dtype=wp.vec3, device=device)
        self.output.normal = wp.zeros(32, dtype=wp.vec3, device=device)
        self.output.depth = wp.zeros(32, dtype=float, device=device)
        self.shape_types = wp.zeros(2, dtype=int, device=device)
        self.shape_data = wp.zeros(2, dtype=wp.vec4, device=device)
        self.shape_gap = wp.zeros(2, dtype=float, device=device)

    def seed(self):
        wp.launch(_seed_entries, dim=1, inputs=[self.reducer.get_data_struct(), self.enabled], device=self.device)

    def export(self, kernel, *, parallel=False, blocks=1):
        reducer = self.reducer
        wp.launch_tiled(
            kernel,
            dim=blocks,
            inputs=[
                reducer.hashtable.keys,
                reducer.ht_values,
                reducer.hashtable.active_slots,
                reducer.position_depth,
                reducer.normal,
                reducer.shape_pairs,
                reducer.contact_fingerprints,
                reducer.exported_flags,
                self.shape_types,
                self.shape_data,
                self.shape_gap,
                self.output,
                blocks,
                int(parallel),
                int(reducer.deterministic),
            ],
            block_dim=EXPORT_REDUCED_CONTACTS_BLOCK_DIM,
            device=self.device,
        )

    def contacts(self):
        count = int(self.output.count.numpy()[0])
        if count > self.output.position.shape[0]:
            raise AssertionError("The test writer overflowed")
        ids = self.output.fingerprint.numpy()[:count]
        geometry = np.column_stack(
            (
                self.output.position.numpy()[:count],
                self.output.normal.numpy()[:count],
                self.output.depth.numpy()[:count],
            )
        )
        return ids[np.argsort(ids)], geometry[np.argsort(ids)]


class TestContactReductionExportCleanup(unittest.TestCase):
    def test_consumed_entries_retire_cpu(self):
        """Publish a winning contact before retiring its key and every winner slot."""
        device = "cpu"
        reducer = GlobalContactReducer(8, device=device)
        capacity = reducer.hashtable.capacity
        entry = 17
        keys = reducer.hashtable.keys.numpy().copy()
        keys[entry] = 1
        reducer.hashtable.keys.assign(keys)
        active = np.zeros(capacity + 1, dtype=np.int32)
        active[0], active[capacity] = entry, 1
        reducer.hashtable.active_slots.assign(active)
        values = np.zeros(reducer.ht_values.shape[0], dtype=np.uint64)
        winners = reduction_value_index_np(entry, np.arange(VALUES_PER_KEY))
        values[winners] = 1
        reducer.ht_values.assign(values)
        reducer.contact_count.fill_(1)
        reducer.ht_insert_failures.fill_(3)
        reducer.buffer_overflows.fill_(5)
        output = wp.zeros(1, dtype=int, device=device)
        kernel = create_export_reduced_contacts_kernel(_count_contact, clear_after_export=True)
        wp.launch_tiled(
            kernel,
            dim=1,
            inputs=[
                reducer.hashtable.keys,
                reducer.ht_values,
                reducer.hashtable.active_slots,
                reducer.position_depth,
                reducer.normal,
                reducer.shape_pairs,
                reducer.contact_fingerprints,
                reducer.exported_flags,
                wp.zeros(2, dtype=int, device=device),
                wp.zeros(2, dtype=wp.vec4, device=device),
                wp.zeros(2, dtype=float, device=device),
                output,
                1,
                0,
                0,
            ],
            block_dim=EXPORT_REDUCED_CONTACTS_BLOCK_DIM,
            device=device,
        )
        self.assertEqual(int(output.numpy()[0]), 1)
        self.assertTrue(np.all(reducer.ht_values.numpy()[winners] == 0))
        self.assertEqual(int(reducer.hashtable.keys.numpy()[entry]), (1 << 64) - 1)
        self.assertEqual(int(reducer.hashtable.active_slots.numpy()[capacity]), 1)
        self.assertEqual(int(reducer.contact_count.numpy()[0]), 1)
        self.assertEqual(int(reducer.ht_insert_failures.numpy()[0]), 3)
        self.assertEqual(int(reducer.buffer_overflows.numpy()[0]), 5)

    def check_export_modes(self, device):
        """Preserve roundoff representatives and deduplicate winners across consumed entries."""
        for deterministic in (False, True):
            for parallel in (False, True) if wp.get_device(device).is_cuda else (False,):
                for blocks in (1, 3):
                    with self.subTest(deterministic=deterministic, parallel=parallel, blocks=blocks):
                        observations = []
                        for cleanup in (False, True):
                            fixture = _Fixture(device, deterministic=deterministic)
                            reducer = fixture.reducer
                            fixture.seed()
                            slots_before = reducer.hashtable.active_slots.numpy().copy()
                            payload = reducer.position_depth.numpy().copy()
                            keys_before = reducer.hashtable.keys.numpy().copy()
                            values_before = reducer.ht_values.numpy().copy()
                            reducer.ht_insert_failures.fill_(3)
                            reducer.buffer_overflows.fill_(5)
                            factory = create_export_reduced_contacts_kernel
                            # The ordinary standalone API must remain nondestructive.
                            kernel = (
                                factory(_record_contact, clear_after_export=True)
                                if cleanup
                                else factory(_record_contact)
                            )
                            fixture.export(kernel, parallel=parallel, blocks=blocks)
                            ids, geometry = fixture.contacts()
                            np.testing.assert_array_equal(ids, [10, 20, 40, 50])
                            self.assertTrue(np.isfinite(geometry).all())
                            observations.append(geometry)
                            np.testing.assert_array_equal(reducer.hashtable.active_slots.numpy(), slots_before)
                            np.testing.assert_array_equal(reducer.position_depth.numpy(), payload)
                            self.assertEqual(int(reducer.contact_count.numpy()[0]), 5)
                            self.assertEqual(int(reducer.ht_insert_failures.numpy()[0]), 3)
                            self.assertEqual(int(reducer.buffer_overflows.numpy()[0]), 5)
                            self.assertTrue(np.all(reducer.exported_flags.numpy()[[2, 3, 4, 5]] > 0))
                            if cleanup:
                                self.assertTrue(np.all(reducer.ht_values.numpy() == 0))
                                self.assertTrue(np.all(reducer.hashtable.keys.numpy() == np.uint64((1 << 64) - 1)))
                            else:
                                np.testing.assert_array_equal(reducer.hashtable.keys.numpy(), keys_before)
                                np.testing.assert_array_equal(reducer.ht_values.numpy(), values_before)
                        np.testing.assert_allclose(observations[0], observations[1], rtol=2e-6, atol=2e-7)

    def test_export_modes_cpu(self):
        """Retire serial fast and deterministic entries without changing exported geometry."""
        self.check_export_modes("cpu")

    def check_lifecycle(self, device, *, graph=False):
        """Reset only between collisions and reuse IDs after empty and regrown exports."""
        fixture = _Fixture(device)
        reducer = fixture.reducer
        kernel = create_export_reduced_contacts_kernel(_record_contact, clear_after_export=True)

        def collide():
            reducer.reset_after_export()
            fixture.output.count.zero_()
            fixture.seed()
            fixture.export(kernel, parallel=wp.get_device(device).is_cuda)

        collide()
        if graph:
            with wp.ScopedCapture(device=device) as capture:
                collide()
        for enabled in (1, 0, 0, 1, 0, 1):
            with self.subTest(enabled=enabled):
                fixture.enabled.fill_(enabled)
                reducer.ht_insert_failures.fill_(9)
                reducer.buffer_overflows.fill_(7)
                if graph:
                    wp.capture_launch(capture.graph)
                else:
                    collide()
                ids, geometry = fixture.contacts()
                np.testing.assert_array_equal(ids, [10, 20, 40, 50] if enabled else [])
                self.assertTrue(np.isfinite(geometry).all())
                self.assertEqual(int(reducer.contact_count.numpy()[0]), 5 * enabled)
                self.assertEqual(int(reducer.hashtable.active_slots.numpy()[-1]), 4 * enabled)
                self.assertEqual(int(reducer.ht_insert_failures.numpy()[0]), 0)
                self.assertEqual(int(reducer.buffer_overflows.numpy()[0]), 0)
                self.assertTrue(np.all(reducer.ht_values.numpy() == 0))
        reducer.clear()
        collide()
        np.testing.assert_array_equal(fixture.contacts()[0], [10, 20, 40, 50])

    def test_lifecycle_cpu(self):
        """Keep repeated empty/regrow and explicit full reset safe on CPU."""
        self.check_lifecycle("cpu")

    def check_overflow(self, device):
        """Retain real failed reservations until the next collision starts."""
        fixture = _Fixture(device, capacity=2)
        reducer = fixture.reducer
        kernel = create_export_reduced_contacts_kernel(_record_contact, clear_after_export=True)
        for _ in range(2):
            reducer.reset_after_export()
            self.assertEqual(int(reducer.buffer_overflows.numpy()[0]), 0)
            wp.launch(_overflow_buffer, dim=1, inputs=[reducer.get_data_struct()], device=device)
            fixture.export(kernel, parallel=wp.get_device(device).is_cuda)
            self.assertEqual(int(reducer.contact_count.numpy()[0]), 2)
            self.assertEqual(int(reducer.buffer_overflows.numpy()[0]), 3)
            self.assertEqual(int(fixture.output.count.numpy()[0]), 0)

    def test_overflow_cpu(self):
        """Preserve overflow evidence through CPU cleanup export."""
        self.check_overflow("cpu")

    def test_admission_cpu(self):
        """Keep unsupported owners and default standalone reduction on full clearing."""
        flags = {
            **primitive.FLAGS,
            "NEWTON_HEIGHTFIELD_PRIMITIVE_QUERY": "0",
            "NEWTON_HEIGHTFIELD_CYLINDER_SPLIT": "0",
        }

        def create(**options):
            options = {"has_heightfields": True, "has_meshes": False, **options}
            return NarrowPhase(max_candidate_pairs=4, max_triangle_pairs=16, device="cpu", **options)

        with patch.dict(os.environ, flags):
            os.environ.pop("NEWTON_REDUCER_EXPORT_CLEANUP", None)
            self.assertFalse(create()._reducer_export_cleanup)
        with patch.dict(os.environ, {**flags, "NEWTON_REDUCER_EXPORT_CLEANUP": "1"}):
            self.assertTrue(create()._reducer_export_cleanup)
            self.assertTrue(create(has_heightfields=False, has_meshes=True)._reducer_export_cleanup)
            for options in (
                {"has_heightfields": False},
                {"reduce_contacts": False},
                {"speculative": True},
                {"deterministic": True},
                {"contact_writer_warp_func": write_contact_simple},
            ):
                with self.subTest(options=options):
                    self.assertFalse(create(**options)._reducer_export_cleanup)
            with patch.dict(os.environ, {"NEWTON_HEIGHTFIELD_FINITE_QUERY": "1"}):
                self.assertFalse(create()._reducer_export_cleanup)
            with patch.dict(os.environ, {"NEWTON_HEIGHTFIELD_PAIR_CSR": "1"}):
                # An inactive CSR request still uses the ordinary buffered owner.
                owner = create()
                self.assertFalse(owner._heightfield_pair_csr_requested)
                self.assertTrue(owner._reducer_export_cleanup)
                with patch.dict(os.environ, {"NEWTON_HEIGHTFIELD_FINITE_QUERY": "1"}):
                    owner = create(contact_writer_warp_func=write_contact)
                    self.assertTrue(owner._heightfield_pair_csr_requested)
                    self.assertFalse(owner._reducer_export_cleanup)
        for option in ("store_hydroelastic_data", "enable_contact_reclamation"):
            reducer = GlobalContactReducer(16, device="cpu", **{option: True})
            reducer.contact_count.fill_(2)
            with self.assertRaises(ValueError):
                reducer.reset_after_export()
            self.assertEqual(int(reducer.contact_count.numpy()[0]), 2)
        with patch.dict(os.environ, {**flags, "NEWTON_REDUCER_EXPORT_CLEANUP": "invalid"}):
            with self.assertRaises(ValueError):
                create()

    def check_pipeline(self, device, *, graph=False):
        """Retire real terrain entries while retaining public contact geometry and reuse."""
        model = primitive.primitive_model(device, "cylinder", slope=True)
        owners = []
        for cleanup in (False, True):
            with patch.dict(os.environ, {"NEWTON_REDUCER_EXPORT_CLEANUP": str(int(cleanup))}):
                owners.append(primitive.pipeline(model, True))
        old, new = owners
        self.assertFalse(old.narrow_phase._reducer_export_cleanup)
        self.assertTrue(new.narrow_phase._reducer_export_cleanup)
        state = model.state()
        old_contacts, new_contacts = old.contacts(), new.contacts()
        poses = state.body_q.numpy().copy()
        for owner, contacts in zip(owners, (old_contacts, new_contacts), strict=True):
            owner.collide(state, contacts)
        if graph:
            with wp.ScopedCapture(device=device) as capture:
                new.collide(state, new_contacts)
        comparison = primitive.TestHeightfieldPrimitiveQuery()
        for loaded in (True, False, False, True, True):
            current = poses.copy()
            if not loaded:
                current[:, 2] += 20.0
            state.body_q.assign(current)
            old.collide(state, old_contacts)
            if graph:
                wp.capture_launch(capture.graph)
            else:
                new.collide(state, new_contacts)
            if loaded:
                comparison.assert_current_contacts(old, new, state, old_contacts, new_contacts)
            else:
                self.assertEqual(int(old_contacts.rigid_contact_count.numpy()[0]), 0)
                self.assertEqual(int(new_contacts.rigid_contact_count.numpy()[0]), 0)
            for owner in owners:
                owner.narrow_phase.check_buffer_capacity()
            reducer = new.narrow_phase.global_contact_reducer
            self.assertTrue(np.all(reducer.ht_values.numpy() == 0))
            self.assertTrue(np.all(reducer.hashtable.keys.numpy() == np.uint64((1 << 64) - 1)))

    def test_pipeline_cpu(self):
        """Compare real loaded terrain contacts and empty/regrow reuse on CPU."""
        self.check_pipeline("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_export_modes_cuda(self):
        """Retire parallel and serial entries with deterministic and fast packing on CUDA."""
        self.check_export_modes("cuda:0")

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_graph_lifecycle_cuda(self):
        """Replay cleanup, reset and changing entry populations in a CUDA graph."""
        self.check_lifecycle("cuda:0", graph=True)

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_overflow_cuda(self):
        """Preserve actual failed CUDA reservations until next-collision reset."""
        self.check_overflow("cuda:0")

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_pipeline_graph_cuda(self):
        """Retain loaded terrain geometry through captured empty/regrow collisions."""
        self.check_pipeline("cuda:0", graph=True)


if __name__ == "__main__":
    unittest.main()
