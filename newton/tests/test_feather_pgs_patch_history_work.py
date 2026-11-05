# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise sparse patch history through live and empty frames."""

import unittest

import numpy as np
import warp as wp

from newton.tests.feather_pgs_patch_reference import store_frame
from newton.tests.test_feather_pgs_friction_patches import _patch_fixture


def _anchor_bodies(patches):
    """Return the bodies of valid stored anchors, the only history poses a build reads."""
    valid = patches.previous.valid.numpy() != 0
    bodies = np.concatenate((patches.previous.body_a.numpy()[valid], patches.previous.body_b.numpy()[valid]))
    return np.unique(bodies[bodies >= 0])


class TestPatchHistoryWork(unittest.TestCase):
    def test_invalid_payload_is_not_published(self):
        """Retire stale anchors without reading or copying unused geometric payload."""
        devices = ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])
        for device in devices:
            with self.subTest(device=device):
                model, state, contacts, patches = _patch_fixture([[-0.1, 0, 0], [0.1, 0, 0]], device=device)
                # The fills below stand for a stored frame that used every slot.
                patches.stored.fill_(patches.capacity)
                patches.build(model, state, contacts)
                valid = patches.current.valid.numpy().astype(bool)
                # The two contacts are one region: one anchor.
                self.assertEqual(valid.sum(), 1)
                patches.previous.valid.fill_(1)
                patches.previous_world.fill_(42)
                patches.previous.displacement.fill_(wp.vec3(17.0))
                patches.current.displacement.fill_(wp.vec3(3.0))
                store_frame(patches, state)
                np.testing.assert_array_equal(patches.previous.valid.numpy(), valid)
                np.testing.assert_array_equal(patches.previous_world.numpy(), np.where(valid, 0, -1))
                np.testing.assert_array_equal(patches.previous.displacement.numpy()[valid], 3.0)
                np.testing.assert_array_equal(patches.previous.displacement.numpy()[~valid], 17.0)
                n = patches.capacity
                np.testing.assert_array_equal(patches.previous.keys.numpy()[:n], patches.current.keys.numpy()[:n])
                np.testing.assert_array_equal(patches.previous.indices.numpy()[:n], patches.current.indices.numpy()[:n])
                bodies = _anchor_bodies(patches)
                self.assertEqual(len(bodies), 2)
                np.testing.assert_array_equal(patches.previous_q.numpy()[bodies], state.body_q.numpy()[bodies])

    def test_history_survives_population_changes(self):
        """Keep live anchors and retire empty frames through graph replay."""
        devices = ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])
        for device in devices:
            with self.subTest(device=device):
                model, state, contacts, patches = _patch_fixture(
                    [[-0.1, 0, 0], [0.1, 0, 0]], shape_bodies=(9, 10, 9), device=device
                )

                def step(model=model, state=state, contacts=contacts, patches=patches):
                    patches.build(model, state, contacts)
                    store_frame(patches, state)

                step()
                if device.startswith("cuda"):
                    with wp.ScopedCapture(device=device) as capture:
                        step()

                    def replay(graph=capture.graph):
                        wp.capture_launch(graph)

                else:
                    replay = step
                # A collision pass clamps its count to the rows it wrote, so no count exceeds them.
                # The two contacts are one region with one anchor, and either alone is one too.
                for count, anchors in ((2, 1), (1, 1), (0, 0), (2, 1), (2, 1)):
                    contacts.rigid_contact_count.assign([count])
                    replay()
                    self.assertEqual(np.count_nonzero(patches.previous.valid.numpy()), anchors)
                    if anchors == 0:
                        np.testing.assert_array_equal(patches.previous_world.numpy(), -1)
                    bodies = _anchor_bodies(patches)
                    self.assertEqual(len(bodies), 2 if anchors else 0)
                    np.testing.assert_array_equal(patches.previous_q.numpy()[bodies], state.body_q.numpy()[bodies])
                poses = state.body_q.numpy()
                poses[10, 0] += 0.003
                state.body_q.assign(poses)
                replay()
                live = patches.current.valid.numpy().astype(bool)
                self.assertTrue(np.all(patches.current.source.numpy()[live] >= 0))
                np.testing.assert_allclose(np.linalg.norm(patches.view.phi.numpy()[live], axis=1), 0.003, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
