# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check local suppression before the reducer's global export claim."""

import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.geometry.contact_reduction_global import (
    _roundoff_duplicate_bit_for_slot_pair,
    create_export_reduced_contacts_kernel,
)
from newton._src.geometry.narrow_phase import NarrowPhase
from newton.tests import test_heightfield_primitive_query as primitive
from newton.tests.test_contact_reduction_export_cleanup import _Fixture, _record_contact


@wp.kernel
def _equal_id_pairs(
    values: wp.array[wp.uint64],
    positions: wp.array[wp.vec4],
    normals: wp.array[wp.vec2],
    fingerprints: wp.array[wp.int32],
    output: wp.array[int],
):
    pair = wp.tid()
    output[pair] = _roundoff_duplicate_bit_for_slot_pair(
        pair,
        0,
        values,
        positions,
        normals,
        fingerprints,
        0,
    )


class TestCompactContactExport(unittest.TestCase):
    def test_public_cpu_selection(self):
        """Honor explicit choices and reject unsupported forced compact export."""
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        model = builder.finalize(device="cpu")
        with patch.dict(os.environ, {"NEWTON_REDUCER_COMPACT_EXPORT": "invalid"}):
            for choice in ("auto", "tiled"):
                pipeline = newton.CollisionPipeline(model, contact_export=choice)
                selection = pipeline.get_collision_selection()["contact_export"]
                self.assertEqual(selection["requested"], choice)
                self.assertEqual(selection["method"], "disabled")
            with self.assertRaisesRegex(ValueError, "contact_export='compact'"):
                newton.CollisionPipeline(model, contact_export="compact")
        with self.assertRaisesRegex(ValueError, "contact_export must"):
            newton.CollisionPipeline(model, contact_export="unknown")

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_mode_admission_cuda(self):
        """Choose compact for supported reducers and retain incompatible export owners."""
        common = {"max_candidate_pairs": 16, "max_triangle_pairs": 32, "device": "cuda:0"}
        with patch.dict(os.environ, {"NEWTON_REDUCER_COMPACT_EXPORT": "invalid"}):
            for choice, expected in (("auto", True), ("compact", True), ("tiled", False)):
                narrow = NarrowPhase(**common, contact_export=choice)
                self.assertEqual(narrow._reducer_compact_export, expected)
            for options in (
                {"deterministic": True},
                {"reduce_contacts": False},
                {"has_meshes": False},
                {"speculative": True},
                {"contact_writer_warp_func": _record_contact},
            ):
                with self.subTest(options=options):
                    narrow = NarrowPhase(**common, **options, contact_export="auto")
                    self.assertFalse(narrow._reducer_compact_export)
                    with self.assertRaisesRegex(ValueError, "contact_export='compact'"):
                        NarrowPhase(**common, **options, contact_export="compact")

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_pipeline_replay_cuda(self):
        """Preserve terrain contacts across empty and loaded graph replays for both cleanup modes."""
        model = primitive.primitive_model("cuda:0", "cylinder", slope=True)
        state = model.state()
        initial = state.body_q.numpy().copy()
        comparison = primitive.TestHeightfieldPrimitiveQuery()
        for cleanup in (False, True):
            flags = {**primitive.FLAGS, "NEWTON_REDUCER_EXPORT_CLEANUP": str(int(cleanup))}
            with self.subTest(cleanup=cleanup), patch.dict(os.environ, flags):
                ordinary, compact = (
                    newton.CollisionPipeline(
                        model, contact_export=choice, rigid_contact_max=4096, max_triangle_pairs=4096
                    )
                    for choice in ("tiled", "auto")
                )
                self.assertEqual(compact.get_collision_selection()["contact_export"]["method"], "compact")
                old_contacts, new_contacts = ordinary.contacts(), compact.contacts()
                state.body_q.assign(initial)
                ordinary.collide(state, old_contacts)
                compact.collide(state, new_contacts)
                with wp.ScopedCapture(device=model.device) as capture:
                    compact.collide(state, new_contacts)
                for loaded in (True, False, False, True):
                    positions = initial.copy()
                    if not loaded:
                        positions[:, 2] += 20.0
                    state.body_q.assign(positions)
                    ordinary.collide(state, old_contacts)
                    wp.capture_launch(capture.graph)
                    if loaded:
                        comparison.assert_current_contacts(ordinary, compact, state, old_contacts, new_contacts)
                    else:
                        self.assertEqual(int(new_contacts.rigid_contact_count.numpy()[0]), 0)
                    ordinary.narrow_phase.check_buffer_capacity()
                    compact.narrow_phase.check_buffer_capacity()

    def test_repeated_ids_claim_once_cpu(self):
        """Suppress every later copy of a winner before touching payload or global flags."""
        values = wp.full(7, 1, dtype=wp.uint64, device="cpu")
        masks = wp.zeros(21, dtype=int, device="cpu")
        wp.launch(
            _equal_id_pairs,
            dim=21,
            inputs=[
                values,
                wp.empty(0, dtype=wp.vec4, device="cpu"),
                wp.empty(0, dtype=wp.vec2, device="cpu"),
                wp.empty(0, dtype=int, device="cpu"),
                masks,
            ],
            device="cpu",
        )
        expected = [1 << slot for slot in range(1, 7) for _ in range(slot)]
        np.testing.assert_array_equal(masks.numpy(), expected)

    @staticmethod
    def launch_compact(fixture, kernel, workers=64):
        """Launch the ordinary-thread exporter with its explicit stride."""
        r = fixture.reducer
        wp.launch(
            kernel,
            dim=workers,
            inputs=[
                r.hashtable.keys,
                r.ht_values,
                r.hashtable.active_slots,
                r.position_depth,
                r.normal,
                r.shape_pairs,
                r.contact_fingerprints,
                r.exported_flags,
                fixture.shape_types,
                fixture.shape_data,
                fixture.shape_gap,
                fixture.output,
                workers,
                0,
                int(r.deterministic),
            ],
            device=fixture.device,
            block_dim=64,
        )

    def check_contacts(self, device):
        """Compare winner sets and geometry with the serial export oracle."""
        for cleanup in (False, True):
            for workers in (1, 64):
                with self.subTest(cleanup=cleanup, workers=workers, device=device):
                    reference = _Fixture(device)
                    reference.seed()
                    reference.export(create_export_reduced_contacts_kernel(_record_contact), parallel=False)
                    expected = reference.contacts()
                    fixture = _Fixture(device)
                    fixture.seed()
                    before = fixture.reducer.ht_values.numpy().copy()
                    kernel = create_export_reduced_contacts_kernel(
                        _record_contact, clear_after_export=cleanup, compact=True
                    )
                    self.launch_compact(fixture, kernel, workers)
                    actual = fixture.contacts()
                    np.testing.assert_array_equal(actual[0], expected[0])
                    np.testing.assert_array_equal(actual[1], expected[1])
                    # One contact wins every slot of its key: claim it once.
                    self.assertEqual(int(fixture.reducer.exported_flags.numpy()[5]), 1)
                    np.testing.assert_array_equal(fixture.reducer.ht_values.numpy(), 0 if cleanup else before)

    def test_contacts_cpu(self):
        """Preserve serial geometry for sparse, repeated, zero and roundoff winners."""
        self.check_contacts("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_contacts_cuda(self):
        """Preserve the contact set with concurrent key workers and cross-key duplicates."""
        self.check_contacts("cuda:0")

    @unittest.skipUnless(
        wp.is_cuda_available() and hasattr(wp, "get_cuda_kernel_properties"), "Requires CUDA properties"
    )
    def test_fixed_winners_need_no_local_memory_cuda(self):
        """Keep the seven fixed winners out of dynamically indexed local storage."""
        fixture = _Fixture("cuda:0")
        fixture.seed()
        kernel = create_export_reduced_contacts_kernel(_record_contact, clear_after_export=True, compact=True)
        self.launch_compact(fixture, kernel)
        for concrete in kernel.overloads.values():
            properties = wp.get_cuda_kernel_properties(concrete, device="cuda:0", block_dim=64)
            self.assertEqual(properties["local_memory_size"], 0)

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_graph_lifecycle_cuda(self):
        """Replay empty and regrown collisions without stale claims or winner slots."""
        fixture = _Fixture("cuda:0")
        kernel = create_export_reduced_contacts_kernel(_record_contact, clear_after_export=True, compact=True)

        def collide():
            fixture.reducer.reset_after_export()
            fixture.output.count.zero_()
            fixture.seed()
            self.launch_compact(fixture, kernel)

        collide()
        with wp.ScopedCapture(device=fixture.device) as capture:
            collide()
        for enabled in (1, 0, 0, 1, 0, 1):
            fixture.enabled.fill_(enabled)
            wp.capture_launch(capture.graph)
            np.testing.assert_array_equal(fixture.contacts()[0], [10, 20, 40, 50] if enabled else [])
            np.testing.assert_array_equal(fixture.reducer.ht_values.numpy(), 0)
            self.assertEqual(int(fixture.reducer.contact_count.numpy()[0]), 5 * enabled)


if __name__ == "__main__":
    unittest.main()
