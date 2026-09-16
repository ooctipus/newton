# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Private rigid sleeping-owner collision lifecycle controls."""

import unittest

import numpy as np
import warp as wp

import newton


def _fixture():
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    for x in (0.0, 0.75):
        body = builder.add_body(xform=wp.transform(wp.vec3(x, 0.0, 2.0)))
        builder.add_shape_sphere(body, radius=0.5)
    builder.add_particle(pos=(0.0, 0.0, 0.025), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.05)
    model = builder.finalize(device="cpu")
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", rigid_contact_max=8, soft_contact_margin=0.1)
    return pipeline, model.state(), pipeline.contacts()


class TestSleepContactHooksCPU(unittest.TestCase):
    def test_hook_order_and_rigid_boundary(self):
        """Call hooks around current broad pairs and completed rigid geometry, before soft contacts."""
        pipeline, state, contacts = _fixture()
        self.assertIsNone(contacts._rigid_sleep_owner)
        contacts.rigid_contact_count.fill_(5)
        events = []
        test = self

        class Owner:
            def before_collision(self, actual_pipeline, actual_state, actual_contacts):
                test.assertIs(actual_pipeline, pipeline)
                test.assertIs(actual_state, state)
                test.assertIs(actual_contacts, contacts)
                test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 5)
                events.append("before")

            def after_broad_phase(self, actual_pipeline, actual_state, actual_contacts):
                test.assertIs(actual_pipeline, pipeline)
                test.assertIs(actual_state, state)
                test.assertIs(actual_contacts, contacts)
                test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 0)
                test.assertGreater(int(pipeline.broad_phase_pair_count.numpy()[0]), 0)
                events.append("broad")

            def after_collision(self, actual_pipeline, actual_state, actual_contacts):
                test.assertIs(actual_pipeline, pipeline)
                test.assertIs(actual_state, state)
                test.assertIs(actual_contacts, contacts)
                test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
                test.assertEqual(int(contacts.soft_contact_count.numpy()[0]), 0)
                events.append("rigid")

        contacts._rigid_sleep_owner = Owner()
        pipeline.collide(state, contacts)
        self.assertEqual(events, ["before", "broad", "rigid"])
        self.assertGreater(int(contacts.soft_contact_count.numpy()[0]), 0)
        count = int(contacts.rigid_contact_count.numpy()[0])
        expected = contacts.rigid_contact_point0.numpy()[:count].copy()
        contacts._rigid_sleep_owner = None
        pipeline.collide(state, contacts)
        self.assertEqual(events, ["before", "broad", "rigid"])
        self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), count)
        np.testing.assert_array_equal(contacts.rigid_contact_point0.numpy()[:count], expected)

    def test_owner_rejection_precedes_all_mutation(self):
        """Permit unsupported-mode or async ownership rejection before touching the buffer."""
        pipeline, state, contacts = _fixture()
        contacts.rigid_contact_count.fill_(5)
        contacts.contact_generation.fill_(9)
        contacts.rigid_contact_shape0.fill_(13)
        contacts._enable_rigid_soft_full_surface_contact = True
        pipeline.broad_phase_pair_count.fill_(42)

        class RejectingOwner:
            def before_collision(self, _pipeline, _state, _contacts):
                raise RuntimeError("unsupported asynchronous collision owner")

            def after_broad_phase(self, _pipeline, _state, _contacts):
                raise AssertionError("rejected broad phase")

            def after_collision(self, _pipeline, _state, _contacts):
                raise AssertionError("rejected narrow phase")

        contacts._rigid_sleep_owner = RejectingOwner()
        with self.assertRaisesRegex(RuntimeError, "unsupported asynchronous"):
            pipeline.collide(state, contacts)
        self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 5)
        np.testing.assert_array_equal(contacts.contact_generation.numpy(), 9)
        np.testing.assert_array_equal(contacts.rigid_contact_shape0.numpy(), 13)
        self.assertTrue(contacts._enable_rigid_soft_full_surface_contact)
        self.assertEqual(int(pipeline.broad_phase_pair_count.numpy()[0]), 42)


if __name__ == "__main__":
    unittest.main()
