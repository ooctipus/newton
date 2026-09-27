# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Keep compact contact metadata inside the allocator's recorded row reservation."""

import inspect
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.friction_patches import FrictionPatches
from newton._src.solvers.feather_pgs.kernels import prepare_world_contact_rows
from newton._src.solvers.feather_pgs.row_build import contact_row_prelude, populate_world_J_masked


def metadata_inputs(device, needed, gaps):
    """Bind adjacent packets whose current gaps straddle their prior admission."""

    def array(values, dtype=wp.int32):
        return wp.array(values, dtype=dtype, device=device)

    count = len(needed)
    capacity = max(8, sum(needed) + 2)
    patches = FrictionPatches()
    patches.phi = wp.zeros(count, dtype=wp.vec2, device=device)
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
        "friction_patches": patches,
        "friction_anchor_beta": 0.0,
        "contact_friction_articulation_pairs_only": 0,
        "is_free_rigid": array([0]),
        "contact_friction_scale": 1.0,
        "contact_shared_anchor": 0,
        "pgs_beta": 0.05,
        "pgs_cfm": 1.0e-6,
    }
    for name in ("type", "parent", "mu", "beta", "cfm", "restitution"):
        dtype = wp.int32 if name in ("type", "parent") else wp.float32
        values[f"world_row_{name}"] = wp.full((1, capacity), -9, dtype=dtype, device=device)
    for name in ("world_phi", "world_target_velocity"):
        values[name] = wp.full((1, capacity), -9.0, dtype=wp.float32, device=device)
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
    def test_deprecated_anchor_limit_preserves_point_rows(self):
        """Fill every reserved point row without legacy anchor culling or friction scaling."""
        device = "cpu"
        builder = newton.ModelBuilder()
        body = builder.add_body()
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize(device=device)
        for limit in (0, 2):
            with self.subTest(limit=limit):
                if limit:
                    with self.assertWarns(DeprecationWarning):
                        solver = newton.solvers.SolverFeatherPGS(
                            model, contact_friction_anchor_limit=limit, friction_anchor_beta=0.0
                        )
                else:
                    solver = newton.solvers.SolverFeatherPGS(model, friction_anchor_beta=0.0)

                # Main's point allocator reserves all three rows for each eligible contact.
                values, _ = metadata_inputs(device, [3, 3, 3], [0.0, 0.0, 0.0])
                values.update(
                    total_threads=9,
                    world_constraint_count=wp.array([9], dtype=int, device=device),
                    mf_constraint_count=wp.zeros(1, dtype=int, device=device),
                    owned_rows_le=0,
                    size_a=18,
                    size_b=-1,
                    articulation_response_dof_count=wp.array([18], dtype=int, device=device),
                    contact_friction_anchor_limit=solver.contact_friction_anchor_limit,
                    pre_flags=wp.full(3, -9, dtype=int, device=device),
                )
                wp.launch(
                    contact_row_prelude,
                    dim=9,
                    inputs=[values[name] for name in inspect.signature(contact_row_prelude.func).parameters],
                    device=device,
                )
                axes = np.zeros((18, 6), dtype=np.float32)
                axes[:3, :3] = np.eye(3)
                jacobian = wp.full((1, 11, 18), -9.0, dtype=float, device=device)
                values.update(
                    total_threads=54,
                    art_group_idx=wp.zeros(1, dtype=int, device=device),
                    art_dof_start=wp.zeros(1, dtype=int, device=device),
                    body_response_dof_mask=wp.array([(1 << 18) - 1], dtype=wp.uint32, device=device),
                    joint_S_s=wp.array(axes, dtype=wp.spatial_vector, device=device),
                    J_group_a=jacobian,
                    J_group_b=jacobian,
                )
                wp.launch(
                    populate_world_J_masked,
                    dim=54,
                    inputs=[values[name] for name in inspect.signature(populate_world_J_masked.func).parameters],
                    device=device,
                )
                # Normal +X gives tangents +Z and -Y against the three unit translation axes.
                expected_jacobian = np.zeros((9, 18), dtype=np.float32)
                expected_jacobian[0::3, 0] = 1.0
                expected_jacobian[1::3, 2] = 1.0
                expected_jacobian[2::3, 1] = -1.0
                np.testing.assert_array_equal(jacobian.numpy()[0, :9], expected_jacobian)
                np.testing.assert_array_equal(jacobian.numpy()[0, 9:], -9.0)
                np.testing.assert_array_equal(values["world_row_type"].numpy()[0, :9], [0, 2, 2] * 3)
                np.testing.assert_array_equal(values["world_row_parent"].numpy()[0, :9], [-1, 0, 0, -1, 3, 3, -1, 6, 6])
                np.testing.assert_allclose(values["world_row_mu"].numpy()[0, :9], 0.7, rtol=0.0, atol=1.0e-7)
                for name, array in values.items():
                    if name.startswith("world_row_") or name in ("world_phi", "world_target_velocity"):
                        np.testing.assert_array_equal(array.numpy()[0, 9:], -9)

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
