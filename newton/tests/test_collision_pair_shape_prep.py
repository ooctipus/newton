# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise the explicit opt-in geometry preparation ownership contract."""

import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton


class TestCollisionPairShapePrep(unittest.TestCase):
    """Keep default full geometry and opt-in participant outputs distinct."""

    def _model(self, *, particle=False):
        builder = newton.ModelBuilder()
        for i in range(3):
            body = builder.add_body(xform=wp.transform(wp.vec3(i * 0.5, 0.0, 1.0)))
            builder.add_shape_sphere(body=body, radius=0.3)
        if particle:
            builder.add_particle(wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0), 1.0)
        return builder.finalize(device="cpu")

    def _pipeline(self, model, *, enabled=True, pairs=((0, 1),), **kwargs):
        pair_array = wp.array(np.asarray(pairs, dtype=np.int32).reshape(-1, 2), dtype=wp.vec2i, device=model.device)
        with patch.dict(os.environ, {"NEWTON_NARROW_PHASE_PAIR_SHAPE_PREP": str(int(enabled))}):
            return newton.CollisionPipeline(model, shape_pairs_filtered=pair_array, rigid_contact_max=100, **kwargs)

    @staticmethod
    def _geometry(pipeline):
        return (
            pipeline.narrow_phase.shape_aabb_lower,
            pipeline.narrow_phase.shape_aabb_upper,
            pipeline.geom_data,
            pipeline.geom_transform,
        )

    def test_active_outputs_and_live_inputs_exact(self):
        """Match all active output bytes while leaving unowned slots untouched."""
        model = self._model()
        full = self._pipeline(model, enabled=False)
        sparse = self._pipeline(model)
        np.testing.assert_array_equal(sparse.prepared_shape_indices.numpy(), [0, 1])
        state = model.state()
        full_contacts, sparse_contacts = full.contacts(), sparse.contacts()
        for array in self._geometry(sparse):
            array.assign(np.full_like(array.numpy(), 123.25))
        for repeat in range(2):
            if repeat:
                poses = state.body_q.numpy()
                poses[:, 0] += 0.123
                state.body_q.assign(poses)
                model.shape_margin.assign(np.full(3, 0.003, np.float32))
                model.shape_gap.assign(np.full(3, 0.007, np.float32))
                flags = model.shape_flags.numpy()
                flags[0] = 0
                model.shape_flags.assign(flags)
            full.collide(state, full_contacts)
            sparse.collide(state, sparse_contacts)
            for expected, actual in zip(self._geometry(full), self._geometry(sparse), strict=True):
                self.assertEqual(expected.numpy()[:2].tobytes(), actual.numpy()[:2].tobytes())
                np.testing.assert_array_equal(actual.numpy()[2], np.full_like(actual.numpy()[2], 123.25))
            self.assertEqual(
                full_contacts.contact_counters.numpy().tobytes(), sparse_contacts.contact_counters.numpy().tobytes()
            )
            self.assertEqual(
                full_contacts.contact_generation.numpy().tobytes(), sparse_contacts.contact_generation.numpy().tobytes()
            )
            self.assertEqual(
                full.broad_phase_pair_count.numpy().tobytes(), sparse.broad_phase_pair_count.numpy().tobytes()
            )
            count = int(full_contacts.rigid_contact_count.numpy()[0])
            for name in (
                "rigid_contact_shape0",
                "rigid_contact_shape1",
                "rigid_contact_point0",
                "rigid_contact_point1",
                "rigid_contact_normal",
            ):
                self.assertEqual(
                    getattr(full_contacts, name).numpy()[:count].tobytes(),
                    getattr(sparse_contacts, name).numpy()[:count].tobytes(),
                )

    def test_default_prepares_unpaired_shape(self):
        """Keep all default geometry entries current even without any pairs."""
        model = self._model()
        pipeline = self._pipeline(model, enabled=False, pairs=())
        self.assertIsNone(pipeline.prepared_shape_indices)
        pipeline.collide(model.state(), pipeline.contacts())
        self.assertGreater(float(pipeline.narrow_phase.shape_aabb_upper.numpy()[2, 0]), 1.0)

    def test_empty_pairs_maintain_counters_and_wrap(self):
        """Run counter ownership once even when the participant list is empty."""
        model = self._model()
        full = self._pipeline(model, enabled=False, pairs=())
        sparse = self._pipeline(model, pairs=())
        self.assertEqual(len(sparse.prepared_shape_indices), 0)
        for pipeline in (full, sparse):
            contacts = pipeline.contacts()
            contacts.contact_generation.fill_(2147483647)
            pipeline.collide(model.state(), contacts)
            self.assertEqual(int(contacts.contact_generation.numpy()[0]), 0)
            self.assertTrue(np.all(contacts.contact_counters.numpy() == 0))
            self.assertEqual(int(pipeline.broad_phase_pair_count.numpy()[0]), 0)

    def test_pair_union_not_flags_controls_ownership(self):
        """Retain explicitly listed shapes even when their collision flag is off."""
        model = self._model()
        model.shape_flags.fill_(0)
        pipeline = self._pipeline(model, pairs=((2, 0), (0, 2), (1, 2)))
        np.testing.assert_array_equal(pipeline.prepared_shape_indices.numpy(), [0, 1, 2])

    def test_reject_nonexplicit(self):
        """Reject broad phases whose pair set is not fixed at construction."""
        for mode in ("nxn", "sap"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "requires internal explicit"):
                self._pipeline(self._model(), broad_phase=mode)

    def test_reject_particles(self):
        """Reject particle scenes from the bounded initial ownership policy."""
        with self.assertRaisesRegex(ValueError, "requires internal explicit"):
            self._pipeline(self._model(particle=True))

    def test_reject_speculative(self):
        """Reject speculative preprocessing that writes all shape AABBs."""
        with self.assertRaisesRegex(ValueError, "requires internal explicit"):
            self._pipeline(self._model(), speculative_config=newton.CollisionPipeline.SpeculativeContactConfig())

    def test_reject_gradients(self):
        """Reject gradients from the initial experimental preparation mode."""
        with self.assertRaisesRegex(ValueError, "requires internal explicit"):
            self._pipeline(self._model(), requires_grad=True)

    def test_reject_invalid_selector(self):
        """Fail explicitly instead of silently ignoring a malformed selector."""
        with patch.dict(os.environ, {"NEWTON_NARROW_PHASE_PAIR_SHAPE_PREP": "yes"}):
            with self.assertRaisesRegex(ValueError, "must be 0 or 1"):
                newton.CollisionPipeline(self._model())


if __name__ == "__main__":
    unittest.main()
