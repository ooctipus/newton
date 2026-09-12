# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Observe external-contact truncation independently of MJWarp's own flags."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverMuJoCo


def _fixture(*, device=None, native=False, public_capacity=4, mj_capacity=1):
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    for i in range(3):
        body = builder.add_body(xform=wp.transform((2.0 * i, 0.0, 0.05), wp.quat_identity()))
        builder.add_shape_sphere(body, radius=0.1)
    model = builder.finalize(device=device)
    solver = SolverMuJoCo(model, use_mujoco_contacts=native, nconmax=mj_capacity, njmax=32, iterations=8)
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=public_capacity)
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    return model, solver, state, pipeline, contacts


def _convert(model, solver, state, contacts, raw=None, *, new_generation=True):
    if raw is not None:
        contacts.rigid_contact_count.fill_(raw)
    if new_generation:
        contacts.contact_generation.assign(contacts.contact_generation.numpy() + 1)
    solver._convert_contacts_to_mjwarp(model, state, contacts)


class TestMuJoCoContactCapacity(unittest.TestCase):
    def test_truncated_external_prefix_has_independent_sticky_status(self):
        """Reproduce raw3/public4/MJ1 loss despite zero native MJWarp flags."""
        model, solver, state, pipeline, contacts = _fixture()
        self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 3)
        self.assertEqual(solver.mjw_data.naconmax, 1)
        _convert(model, solver, state, contacts)
        self.assertEqual(int(solver.mjw_data.nacon.numpy()[0]), 1)
        np.testing.assert_array_equal(solver.mjw_data.overflow.numpy(), 0)
        expected = {"negative_count": False, "source_contacts": False, "mjwarp_contacts": True}
        self.assertEqual(solver.contact_capacity_status(), expected)
        with self.assertRaisesRegex(RuntimeError, "nconmax"):
            solver.check_contact_capacity()
        # Cache hits still observe the raw prefix; no generation/count side effects.
        solver.contact_capacity_status(clear=True)
        _convert(model, solver, state, contacts, new_generation=False)
        self.assertEqual(solver.contact_capacity_status(), expected)
        self.assertEqual(int(solver._last_nacon_count.numpy()[0]), 1)
        np.testing.assert_array_equal(solver._last_contact_generation.numpy(), contacts.contact_generation.numpy())
        # Reset/collision-owner invalidation must not acknowledge old loss.
        solver.reset(state)
        solver.notify_model_changed(newton.ModelFlags.BODY_PROPERTIES)
        pipeline.collide(state, contacts)
        _convert(model, solver, state, contacts, 0)
        self.assertEqual(solver.contact_capacity_status(), expected)
        self.assertEqual(solver.contact_capacity_status(clear=True), expected)
        self.assertFalse(any(solver.contact_capacity_status().values()))
        solver.check_contact_capacity()

    def test_negative_source_overflow_and_zero_grid(self):
        """Latch invalid counts even with no converter thread and never normalize input."""
        model, solver, state, _, contacts = _fixture()
        for raw, expected in (
            (-1, {"negative_count": True, "source_contacts": False, "mjwarp_contacts": False}),
            (5, {"negative_count": False, "source_contacts": True, "mjwarp_contacts": True}),
            (1, {"negative_count": False, "source_contacts": False, "mjwarp_contacts": False}),
        ):
            solver.contact_capacity_status(clear=True)
            _convert(model, solver, state, contacts, raw)
            self.assertEqual(solver.contact_capacity_status(), expected)
            self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), raw)
        empty = newton.Contacts(rigid_contact_max=0, soft_contact_max=0, device=model.device)
        for raw in (-1, 1):
            solver.contact_capacity_status(clear=True)
            _convert(model, solver, state, empty, raw)
            status = solver.contact_capacity_status()
            self.assertEqual(status["negative_count"], raw < 0)
            self.assertEqual(status["source_contacts"], raw > 0)
            self.assertFalse(status["mjwarp_contacts"])
            self.assertEqual(int(solver._last_nacon_count.numpy()[0]), 0)

    def test_source_capacity_is_checked_independently_of_larger_mj_pool(self):
        """Reject raw5/public4/MJ8 even though the MJ pool has unused capacity."""
        model, solver, state, _, contacts = _fixture(mj_capacity=8)
        # The final materialized slot is intentionally not an active contact.
        # The invalid fifth slot is never indexed by the bounded launch.
        shapes = contacts.rigid_contact_shape0.numpy()
        shapes[3] = -1
        contacts.rigid_contact_shape0.assign(shapes)
        _convert(model, solver, state, contacts, 5)
        self.assertEqual(
            solver.contact_capacity_status(),
            {"negative_count": False, "source_contacts": True, "mjwarp_contacts": False},
        )
        with self.assertRaisesRegex(RuntimeError, "rigid_contact_max"):
            solver.check_contact_capacity()
        self.assertEqual(int(solver.mjw_data.nacon.numpy()[0]), 3)
        np.testing.assert_array_equal(solver.mjw_data.overflow.numpy(), 0)

    def test_native_contacts_have_no_converter_history(self):
        """The dedicated converter status does not reinterpret native collision flags."""
        _, solver, _, _, _ = _fixture(native=True)
        solver.mjw_data.overflow.fill_(4)
        self.assertEqual(
            solver.contact_capacity_status(),
            {"negative_count": False, "source_contacts": False, "mjwarp_contacts": False},
        )
        solver.check_contact_capacity()
        np.testing.assert_array_equal(solver.mjw_data.overflow.numpy(), 4)

    def test_cuda_graph_replay_sticky_reset_and_explicit_clear(self):
        """Reuse the same status owner across actual cached/full graph conversions."""
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("CUDA graph replay requires CUDA")
        model, solver, state, _, contacts = _fixture(device=devices[0])
        _convert(model, solver, state, contacts, 1)
        with wp.ScopedCapture(device=model.device) as capture:
            with self.assertRaisesRegex(RuntimeError, "outside CUDA graph capture"):
                solver.contact_capacity_status(clear=True)
            solver._convert_contacts_to_mjwarp(model, state, contacts)
        for raw, generation, clear, expected in (
            (3, False, False, True),
            (0, True, False, True),
            (0, True, True, False),
            (3, True, False, True),
        ):
            if clear:
                self.assertTrue(solver.contact_capacity_status(clear=True)["mjwarp_contacts"])
            contacts.rigid_contact_count.fill_(raw)
            if generation:
                contacts.contact_generation.assign(contacts.contact_generation.numpy() + 1)
            wp.capture_launch(capture.graph)
            self.assertEqual(solver.contact_capacity_status()["mjwarp_contacts"], expected)
            np.testing.assert_array_equal(solver._last_contact_generation.numpy(), contacts.contact_generation.numpy())
        # Neither later empty output nor generation reuse clears invalid-prefix history.
        solver.contact_capacity_status(clear=True)
        for raw in (5, -1, 0):
            contacts.rigid_contact_count.fill_(raw)
            contacts.contact_generation.assign(contacts.contact_generation.numpy() + 1)
            wp.capture_launch(capture.graph)
        self.assertTrue(all(solver.contact_capacity_status().values()))


if __name__ == "__main__":
    unittest.main()
