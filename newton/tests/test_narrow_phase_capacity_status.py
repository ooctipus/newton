# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test persistent history from the existing narrow-phase buffer verifier."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.geometry.narrow_phase import verify_narrow_phase_buffers
from newton.tests.unittest_utils import StdOutCapture

NAMES = (
    "broad_phase",
    "split_query",
    "gjk",
    "split_gjk",
    "split_manifold",
    "mesh",
    "triangle",
    "mesh_plane",
    "mesh_mesh",
    "sdf_sdf",
    "contacts",
    "reduction_hash_load",
    "reduction_hash_insert",
)


def make_pipeline(device="cpu", *, verify=True, capacity=1):
    builder = newton.ModelBuilder()
    for x in (0.0, 0.1, 0.2):
        body = builder.add_body(xform=wp.transform(wp.vec3(x, 0.0, 0.0)))
        builder.add_shape_sphere(body, radius=0.5)
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(
        model, broad_phase_output_max=capacity, rigid_contact_max=8, verify_buffers=verify
    )
    return model, pipeline


class TestNarrowPhaseCapacityStatus(unittest.TestCase):
    def test_overflow_survives_empty_frame(self):
        model, pipeline = make_pipeline()
        narrow, state, contacts = pipeline.narrow_phase, model.state(), pipeline.contacts()
        self.assertEqual(narrow.buffer_capacity_status(), dict.fromkeys(NAMES, False))
        capture = StdOutCapture()
        capture.begin()
        try:
            pipeline.collide(state, contacts)
            self.assertTrue(narrow.buffer_capacity_status()["broad_phase"])
        finally:
            message = capture.end()
        self.assertIn("Broad phase pair buffer overflowed 3 > 1", message)
        with self.assertRaisesRegex(RuntimeError, "broad_phase"):
            narrow.check_buffer_capacity()
        transforms = state.body_q.numpy()
        transforms[:, 0] = np.arange(3) * 10.0
        state.body_q.assign(transforms)
        pipeline.collide(state, contacts)
        self.assertEqual(int(pipeline.broad_phase_pair_count.numpy()[0]), 0)
        self.assertTrue(narrow.buffer_capacity_status(clear=True)["broad_phase"])
        self.assertFalse(any(narrow.buffer_capacity_status().values()))
        narrow.check_buffer_capacity()

    def test_each_existing_warning_has_an_independent_sticky_bit(self):
        _, pipeline = make_pipeline()
        narrow = pipeline.narrow_phase
        counts = [wp.zeros(1, dtype=int, device="cpu") for _ in range(11)]
        active = wp.zeros(3, dtype=int, device="cpu")
        failures = wp.zeros(1, dtype=int, device="cpu")
        inputs = [item for count in counts for item in (count, 2)]
        inputs += [active, 2, failures, 90, narrow._buffer_capacity_status]
        capture = StdOutCapture()
        capture.begin()
        try:
            for bit, name in enumerate(NAMES):
                with self.subTest(name=name):
                    narrow.buffer_capacity_status(clear=True)
                    if bit < 11:
                        counts[bit].assign([3])
                    elif bit == 11:
                        active.assign([0, 0, 2])
                    else:
                        failures.assign([1])
                    wp.launch(verify_narrow_phase_buffers, dim=1, inputs=inputs, device="cpu")
                    self.assertEqual(narrow.buffer_capacity_status(), {key: key == name for key in NAMES})
                    for count in counts:
                        count.zero_()
                    active.zero_()
                    failures.zero_()
                    wp.launch(verify_narrow_phase_buffers, dim=1, inputs=inputs, device="cpu")
                    self.assertTrue(narrow.buffer_capacity_status()[name])
        finally:
            capture.end()

    def test_disabled_split_and_absent_mesh_storage_are_not_failures(self):
        _, pipeline = make_pipeline()
        narrow = pipeline.narrow_phase
        zero = wp.zeros(1, dtype=int, device="cpu")
        excess = wp.array([3], dtype=int, device="cpu")
        inputs = [zero, 2, excess, -1, zero, 2, excess, -1, excess, -1]
        inputs += [item for _ in range(5) for item in (None, 0)]
        inputs += [zero, 2, zero, 0, excess, 90, narrow._buffer_capacity_status]
        wp.launch(verify_narrow_phase_buffers, dim=1, inputs=inputs, device="cpu")
        narrow.check_buffer_capacity()

    def test_disabled_verification_cannot_report_success(self):
        _, pipeline = make_pipeline(verify=False)
        with self.assertRaisesRegex(RuntimeError, "verify_buffers=True"):
            pipeline.narrow_phase.buffer_capacity_status()
        with self.assertRaisesRegex(RuntimeError, "verify_buffers=True"):
            pipeline.narrow_phase.check_buffer_capacity()

    def test_sufficient_capacity_preserves_contacts(self):
        model, pipeline = make_pipeline(capacity=3)
        contacts = pipeline.contacts()
        pipeline.collide(model.state(), contacts)
        self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 3)
        pipeline.narrow_phase.check_buffer_capacity()

    def test_graph_replay_retains_failure_history(self):
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("CUDA graph replay requires CUDA")
        device = devices[0]
        model, pipeline = make_pipeline(device)
        narrow, state, contacts = pipeline.narrow_phase, model.state(), pipeline.contacts()
        initial = state.body_q.numpy()
        pipeline.collide(state, contacts)
        wp.synchronize_device(device)
        narrow.buffer_capacity_status(clear=True)
        with wp.ScopedCapture(device=device) as capture:
            pipeline.collide(state, contacts)
            with self.assertRaisesRegex(RuntimeError, "outside CUDA graph capture"):
                narrow.buffer_capacity_status()
        narrow.buffer_capacity_status(clear=True)
        wp.capture_launch(capture.graph)
        self.assertTrue(narrow.buffer_capacity_status()["broad_phase"])
        separated = initial.copy()
        separated[:, 0] = np.arange(3) * 10.0
        state.body_q.assign(separated)
        wp.capture_launch(capture.graph)
        self.assertEqual(int(pipeline.broad_phase_pair_count.numpy()[0]), 0)
        self.assertTrue(narrow.buffer_capacity_status(clear=True)["broad_phase"])
        wp.capture_launch(capture.graph)
        narrow.check_buffer_capacity()
        state.body_q.assign(initial)
        wp.capture_launch(capture.graph)
        self.assertTrue(narrow.buffer_capacity_status()["broad_phase"])


if __name__ == "__main__":
    unittest.main()
