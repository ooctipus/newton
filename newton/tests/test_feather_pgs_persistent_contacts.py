# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify physical invariants needed by persistent FeatherPGS contacts."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS
from newton.tests.test_contact_reduction_body_pairs import _cylinder_foot


class TestFeatherPGSPersistentContacts(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "requires CUDA graph capture")
    def test_combined_history_reset_replays_with_changed_world_mask(self):
        """Capture matching and solver resets of body-pair reduced contacts with a live device mask."""
        template = newton.ModelBuilder()
        _cylinder_foot(template, wp.vec3(0.0, 0.0, 0.015))
        builder = newton.ModelBuilder()
        builder.replicate(template, 2, spacing=(1.0, 0.0, 0.0))
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_cuda_device())
        pipeline = newton.CollisionPipeline(
            model,
            contact_matching="latest",
            body_pair_reduction=newton.CollisionPipeline.BodyPairReductionConfig(),
            growable_contacts=True,
        )
        solver = SolverFeatherPGS(model, pgs_warmstart=True)
        state, output = model.state(), model.state()
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        solver.step(state, output, model.control(), contacts, 1.0 / 240.0)
        # The carry is keyed by the stepped contacts, each in its allocator's world.
        stepped = int(contacts.rigid_contact_count.numpy()[0])
        carried_worlds = solver.placement.contact_world.numpy()[:stepped]
        mask = wp.array([False, False, False], dtype=wp.bool, device=model.device)
        solver_mask = mask[:2]
        # Compile both reset paths before capture.
        solver.reset(state, solver_mask)
        pipeline.reset_contact_matching(mask)
        with wp.ScopedCapture(device=model.device) as capture:
            solver.reset(state, solver_mask)
            pipeline.reset_contact_matching(mask)
            pipeline.collide(state, contacts)
        for selected in (0, 1):
            values = np.zeros(3, dtype=bool)
            values[selected] = True
            mask.assign(values)
            solver.placement.carry_impulse.fill_(1.0)
            wp.capture_launch(capture.graph)
            previous = solver.placement.carry_impulse.numpy()[:stepped]
            self.assertTrue(np.any(carried_worlds == selected) and np.any(carried_worlds == 1 - selected))
            self.assertTrue(np.all(previous[carried_worlds == selected] == 0.0))
            self.assertTrue(np.all(previous[carried_worlds == 1 - selected] == 1.0))
            count = int(contacts.rigid_contact_count.numpy()[0])
            shape0 = contacts.rigid_contact_shape0.numpy()[:count]
            shape1 = contacts.rigid_contact_shape1.numpy()[:count]
            worlds = np.maximum(model.shape_world.numpy()[shape0], model.shape_world.numpy()[shape1])
            matches = contacts.rigid_contact_match_index.numpy()[:count]
            self.assertTrue(np.any(worlds == selected) and np.any(worlds == 1 - selected))
            self.assertTrue(np.all(matches[worlds == selected] < 0))
            self.assertTrue(np.all(matches[worlds == 1 - selected] >= 0))

    def _carry_contact(self, *, rotated=False, device="cpu"):
        """Step a resting box contact once, give it a carried normal impulse, turn its normal, and step it again
        without a sweep: the rows hold what the warm start seeded."""
        builder = newton.ModelBuilder(up_axis=newton.Axis.X, gravity=wp.vec3(0.0))
        builder.add_ground_plane()
        body = builder.add_body(xform=wp.transform(wp.vec3(0.1, 0.0, 0.0), wp.quat_identity()))
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize(device=device)
        model.shape_material_mu.fill_(1.0)
        solver = SolverFeatherPGS(model, pgs_warmstart=True, pgs_iterations=0)
        pipeline = newton.CollisionPipeline(model, contact_matching="latest", growable_contacts=True)
        contacts = pipeline.contacts()
        state_in, state_out = model.state(), model.state()
        newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
        pipeline.collide(state_in, contacts)
        self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
        contacts.rigid_contact_count.fill_(1)
        old_normal = wp.normalize(wp.vec3(-1.0, 0.001, 0.0))
        new_normal = wp.normalize(wp.vec3(-1.0, -0.001, 0.0))
        if rotated:
            old_normal = wp.vec3(0.0, 0.0, 1.0)
            new_normal = wp.vec3(0.5 / np.sqrt(2.0), 0.5 / np.sqrt(2.0), np.sqrt(3.0) / 2.0)
        normals = contacts.rigid_contact_normal.numpy()
        normals[0] = old_normal
        contacts.rigid_contact_normal.assign(normals)
        solver.step(state_in, state_out, model.control(), contacts, 1.0 / 240.0)
        slot = int(solver.placement.contact_slot.numpy()[0])
        self.assertGreaterEqual(slot, 0)
        solver.placement.carry_impulse[:1].fill_(1.0)
        normals[0] = new_normal
        contacts.rigid_contact_normal.assign(normals)
        contacts.rigid_contact_match_index.fill_(-1)
        indices = contacts.rigid_contact_match_index.numpy()
        indices[0] = 0
        contacts.rigid_contact_match_index.assign(indices)
        solver.step(state_in, state_out, model.control(), contacts, 1.0 / 240.0)
        slot = int(solver.placement.contact_slot.numpy()[0])
        return solver.placement.impulses.numpy()[0, slot : slot + 3]

    def test_carried_contact_seeds_only_its_normal(self):
        """A matched contact starts from its carried normal impulse, across a basis seam and a 30-degree turn of its
        normal; its friction pair starts cold (the carry holds none): the anchor's displacement is the friction's
        history."""
        devices = ["cpu", wp.get_cuda_device()] if wp.is_cuda_available() else ["cpu"]
        for device in devices:
            for rotated in (False, True):
                with self.subTest(device=str(device), rotated=rotated):
                    impulses = self._carry_contact(rotated=rotated, device=device)
                    self.assertEqual(float(impulses[0]), 1.0)
                    np.testing.assert_array_equal(impulses[1:], np.zeros(2, dtype=np.float32))

    def test_matching_retains_world_distance_policy(self):
        """Moving beyond the existing world-distance threshold must cold-start."""
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        for z in (0.1, 0.29):
            body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, z), wp.quat_identity()))
            builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize(device="cpu")
        state = model.state()
        pipeline = newton.CollisionPipeline(model, contact_matching="latest", growable_contacts=True)
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        count = int(contacts.rigid_contact_count.numpy()[0])
        self.assertGreater(count, 0)
        poses = state.body_q.numpy()
        poses[:, 0] += 1.0
        state.body_q.assign(poses)
        pipeline.collide(state, contacts)
        self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), count)
        self.assertTrue(np.all(contacts.rigid_contact_match_index.numpy()[:count] < 0))

    def test_reduced_contacts_preserve_warmstart_identity(self):
        """Carry only retained contact identities through reduction and solver steps."""
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        _cylinder_foot(builder, wp.vec3(0.0, 0.0, 0.015))
        model = builder.finalize(device="cpu")
        solver = SolverFeatherPGS(model, pgs_warmstart=True, pgs_iterations=8)
        pipeline = newton.CollisionPipeline(
            model,
            contact_matching="latest",
            body_pair_reduction=newton.CollisionPipeline.BodyPairReductionConfig(),
            growable_contacts=True,
        )
        unreduced = newton.CollisionPipeline(model, growable_contacts=True)
        unreduced_contacts = unreduced.contacts()
        unreduced.collide(model.state(), unreduced_contacts)
        contacts = pipeline.contacts()
        state_in, state_out = model.state(), model.state()
        matched, counts = 0, []
        for _ in range(40):
            pipeline.collide(state_in, contacts)
            count = int(contacts.rigid_contact_count.numpy()[0])
            counts.append(count)
            indices = contacts.rigid_contact_match_index.numpy()[:count]
            valid = indices[indices >= 0]
            self.assertEqual(len(valid), len(np.unique(valid)), "a cached impulse was assigned more than once")
            matched += len(valid)
            self.assertEqual(int(pipeline._contact_matcher.prev_contact_count.numpy()[0]), count)
            state_in.clear_forces()
            solver.step(state_in, state_out, model.control(), contacts, 1.0 / 240.0)
            state_in, state_out = state_out, state_in
        self.assertGreater(int(unreduced_contacts.rigid_contact_count.numpy()[0]), counts[0])
        self.assertGreater(matched, 20)
        self.assertTrue(np.isfinite(state_in.body_q.numpy()).all())
        self.assertGreater(float(state_in.body_q.numpy()[0, 2]), 0.012)

    def test_matching_reset_preserves_other_worlds_under_reduction(self):
        """A partial matching reset of body-pair reduced contacts cold-starts only the selected world."""
        template = newton.ModelBuilder()
        _cylinder_foot(template, wp.vec3(0.0, 0.0, 0.015))
        builder = newton.ModelBuilder()
        builder.replicate(template, 2, spacing=(1.0, 0.0, 0.0))
        builder.add_ground_plane()
        model = builder.finalize(device="cpu")
        pipeline = newton.CollisionPipeline(
            model,
            contact_matching="latest",
            body_pair_reduction=newton.CollisionPipeline.BodyPairReductionConfig(),
            growable_contacts=True,
        )
        state, contacts = model.state(), pipeline.contacts()
        pipeline.collide(state, contacts)
        pipeline.collide(state, contacts)
        pipeline.reset_contact_matching(wp.array([True, False, False], dtype=wp.bool, device=model.device))
        pipeline.collide(state, contacts)
        count = int(contacts.rigid_contact_count.numpy()[0])
        shape0 = contacts.rigid_contact_shape0.numpy()[:count]
        shape1 = contacts.rigid_contact_shape1.numpy()[:count]
        worlds = np.maximum(model.shape_world.numpy()[shape0], model.shape_world.numpy()[shape1])
        matches = contacts.rigid_contact_match_index.numpy()[:count]
        self.assertTrue(np.any(worlds == 0) and np.any(worlds == 1))
        self.assertTrue(np.all(matches[worlds == 0] < 0))
        self.assertTrue(np.all(matches[worlds == 1] >= 0))


if __name__ == "__main__":
    unittest.main()
