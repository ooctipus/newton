# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Keep compact contact metadata inside the allocator's recorded row reservation."""

import inspect
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs.kernels import prepare_world_contact_rows


def metadata_inputs(device, needed, gaps):
    """Bind adjacent packets whose current gaps straddle their prior admission."""

    def array(values, dtype=wp.int32):
        return wp.array(values, dtype=dtype, device=device)

    count = len(needed)
    slots = np.concatenate(([0], np.cumsum(needed[:-1]))).astype(np.int32)
    values = {
        "contact_count": array([count]),
        "total_num_threads": 1,
        "contact_point0": array([[gap, 0.0, 0.0] for gap in gaps], wp.vec3),
        "contact_point1": wp.zeros(count, dtype=wp.vec3, device=device),
        "contact_normal": array([[-1.0, 0.0, 0.0]] * count, wp.vec3),
        "contact_shape0": array([0] * count),
        "contact_shape1": array([1] * count),
        "contact_thickness0": wp.zeros(count, dtype=float, device=device),
        "contact_thickness1": wp.zeros(count, dtype=float, device=device),
        "contact_world": array([0] * count),
        "contact_slot": array(slots),
        "contact_art_a": array([0] * count),
        "contact_art_b": array([-1] * count),
        "contact_path": array([0] * count),
        "contact_slots_needed": array(needed),
        "shape_body": array([0, -1]),
        "body_q": array([wp.transform_identity()], wp.transform),
        "body_v_s": wp.zeros(1, dtype=wp.spatial_vector, device=device),
        "prescribed_articulation": array([0]),
        "articulation_origin": wp.zeros(1, dtype=wp.vec3, device=device),
        "shape_material_mu": array([0.6, 0.8], wp.float32),
        "shape_material_restitution": array([0.2, 0.4], wp.float32),
        "enable_friction": 1,
        "contact_friction_gap_threshold": 0.02,
        "contact_friction_shared_anchor": 0,
        "contact_friction_anchor_limit": 2,
        "contact_friction_articulation_pairs_only": 0,
        "is_free_rigid": array([0]),
        "contact_friction_scale": 1.0,
        "contact_shared_anchor": 0,
        "pgs_beta": 0.05,
        "pgs_cfm": 1.0e-6,
    }
    for name in ("type", "parent", "mu", "beta", "cfm", "restitution"):
        dtype = wp.int32 if name in ("type", "parent") else wp.float32
        values[f"world_row_{name}"] = wp.full((1, 8), -9, dtype=dtype, device=device)
    for name in ("world_phi", "world_target_velocity"):
        values[name] = wp.full((1, 8), -9.0, dtype=wp.float32, device=device)
    return values, slots


def launch_metadata(values, device):
    """Launch the actual shared producer using its declared argument names."""
    wp.launch(
        prepare_world_contact_rows,
        dim=values["total_num_threads"],
        inputs=[values[name] for name in inspect.signature(prepare_world_contact_rows.func).parameters],
        device=device,
    )


class TestContactReservations(unittest.TestCase):
    def check_reservations(self, device):
        """Preserve both one-to-three and three-to-one threshold disagreements."""
        below = np.nextafter(np.float32(0.02), np.float32(0.0))
        above = np.nextafter(np.float32(0.02), np.float32(1.0))
        for needed, gaps in (([1, 3], [below, above]), ([3, 1], [above, below])):
            with self.subTest(device=str(device), needed=needed):
                values, slots = metadata_inputs(device, needed, gaps)
                launch_metadata(values, device)
                kinds = values["world_row_type"].numpy()[0]
                parents = values["world_row_parent"].numpy()[0]
                expected_kinds, expected_parents = [], []
                for slot, width in zip(slots, needed, strict=True):
                    expected_kinds += [0] + [2] * (width - 1)
                    expected_parents += [-1] + [int(slot)] * (width - 1)
                np.testing.assert_array_equal(kinds[:4], expected_kinds)
                np.testing.assert_array_equal(parents[:4], expected_parents)
                np.testing.assert_array_equal(kinds[4:], -9)
                np.testing.assert_array_equal(parents[4:], -9)
                np.testing.assert_array_equal(values["world_phi"].numpy()[0, slots], gaps)
                np.testing.assert_allclose(values["world_row_mu"].numpy()[0, slots], 0.7, atol=1.0e-7)

    def test_reservations_cpu(self):
        """Keep CPU metadata inside each reserved packet despite opposite gap admission."""
        self.check_reservations("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA is required")
    def test_reservations_cuda(self):
        """Keep CUDA metadata inside each reserved packet despite opposite gap admission."""
        self.check_reservations("cuda:0")


if __name__ == "__main__":
    unittest.main()
