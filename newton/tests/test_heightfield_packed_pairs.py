# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Keep complete heightfield pair ownership when retiring idle BVH tile lanes."""

import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.geometry.heightfield_cells import heightfield_cell_overlaps_kernel
from newton._src.geometry.narrow_phase import NarrowPhase
from newton.tests.test_heightfield_cell_reject import make_model


def create_pipeline(model, packed):
    """Select only the independent-pair mapping, preserving current geometry."""
    with patch.dict(
        os.environ,
        {
            "NEWTON_HEIGHTFIELD_CELL_REJECT": "1",
            "NEWTON_NARROW_PHASE_PACKED_HEIGHTFIELD_PAIRS": str(int(packed)),
        },
    ):
        return newton.CollisionPipeline(model, reduce_contacts=True, rigid_contact_max=128, max_triangle_pairs=128)


def logical(narrow):
    """Decode finite-query markers before comparing logical triangle ownership."""
    count = int(narrow.triangle_pairs_count.numpy()[0])
    if not 0 <= count <= narrow.triangle_pairs.shape[0]:
        raise AssertionError("Triangle capacity overflow")
    triples = narrow.triangle_pairs.numpy()[:count].copy()
    triples[:, 2] = np.where(triples[:, 2] < 0, ~triples[:, 2], triples[:, 2])
    return sorted(map(tuple, triples.tolist()))


class TestHeightfieldPackedPairs(unittest.TestCase):
    def check_mapping(self, device):
        """Verify actual dispatched dimensions, matching stride and complete output."""
        model = make_model(device, z=0.015)
        state = model.state()
        original = create_pipeline(model, False)
        packed = create_pipeline(model, True)
        self.assertTrue(packed.narrow_phase._heightfield_packed_pairs)
        old_contacts, new_contacts = original.contacts(), packed.contacts()
        launches = []
        bindings = []
        real_launch = wp.launch

        def observed(*args, **kwargs):
            kernel = kwargs.get("kernel", args[0] if args else None)
            if kernel is heightfield_cell_overlaps_kernel:
                launches.append((list(kwargs["dim"]), kwargs["inputs"][-1]))
                bindings.append(list(kwargs["inputs"]))
            return real_launch(*args, **kwargs)

        with patch.object(wp, "launch", side_effect=observed):
            original.collide(state, old_contacts)
            packed.collide(state, new_contacts)
        self.assertEqual(launches[0][0][0], original.narrow_phase.num_tile_blocks)
        workers = packed.narrow_phase.total_num_threads
        self.assertEqual(launches[1], ([workers, 1], workers))
        self.assertEqual(logical(original.narrow_phase), logical(packed.narrow_phase))
        self.assertGreater(len(logical(packed.narrow_phase)), 0)
        packed.narrow_phase.check_buffer_capacity()
        # A nonmultiple pair count exercises grid-stride ownership independently
        # of contact reduction. Allocate exactly the known repeated query count.
        repeat_count = 17
        inputs = bindings[0]
        first_pair = inputs[8].numpy()[:1]
        inputs[8] = wp.array(np.repeat(first_pair, repeat_count, axis=0), dtype=wp.vec2i, device=device)
        inputs[9] = wp.array([repeat_count], dtype=int, device=device)
        query_count = len(logical(original.narrow_phase)) * repeat_count
        outputs = []
        for count, second in ((3, 128 if wp.get_device(device).is_cuda else 1), (17, 1)):
            inputs[-1] = count
            triples = wp.empty(query_count, dtype=wp.vec3i, device=device)
            total = wp.zeros(1, dtype=int, device=device)
            wp.launch(
                heightfield_cell_overlaps_kernel,
                dim=(count, second),
                inputs=inputs,
                outputs=[triples, total],
                device=device,
                block_dim=128,
            )
            self.assertEqual(int(total.numpy()[0]), query_count)
            outputs.append(sorted(map(tuple, triples.numpy().tolist())))
        self.assertEqual(outputs[0], outputs[1])
        # Current geometry withdrawal/regrowth must overwrite the live prefix.
        poses = state.body_q.numpy().copy()
        moved = poses.copy()
        moved[:, 2] += 3.0
        state.body_q.assign(moved)
        packed.collide(state, new_contacts)
        self.assertEqual(logical(packed.narrow_phase), [])
        state.body_q.assign(poses)
        if wp.get_device(device).is_cuda:
            with wp.ScopedCapture(device=device) as capture:
                packed.collide(state, new_contacts)
            for _ in range(3):
                wp.capture_launch(capture.graph)
                self.assertEqual(logical(original.narrow_phase), logical(packed.narrow_phase))
        else:
            packed.collide(state, new_contacts)
            self.assertEqual(logical(original.narrow_phase), logical(packed.narrow_phase))

    def test_mapping_cpu(self):
        """Cover current prefix, dimensions and stride on the CPU backend."""
        self.check_mapping("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_mapping_cuda(self):
        """Cover real CUDA mapping and continuing collision graph replay."""
        self.check_mapping("cuda:0")

    def test_admission_and_invalid_flag(self):
        """Keep mixed meshes and default dispatch unchanged; reject invalid flags."""
        with patch.dict(
            os.environ, {"NEWTON_HEIGHTFIELD_CELL_REJECT": "1", "NEWTON_NARROW_PHASE_PACKED_HEIGHTFIELD_PAIRS": "1"}
        ):
            narrow = NarrowPhase(
                max_candidate_pairs=4,
                max_triangle_pairs=16,
                device="cpu",
                has_meshes=True,
                has_heightfields=True,
                reduce_contacts=False,
            )
        self.assertFalse(narrow._heightfield_packed_pairs)
        with (
            patch.dict(os.environ, {"NEWTON_NARROW_PHASE_PACKED_HEIGHTFIELD_PAIRS": "yes"}),
            self.assertRaises(ValueError),
        ):
            newton.CollisionPipeline(make_model())


if __name__ == "__main__":
    unittest.main()
