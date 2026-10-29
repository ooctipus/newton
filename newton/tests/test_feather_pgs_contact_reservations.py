# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Keep every contact row builder inside the allocator's recorded row reservation."""

import inspect
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.dense_rows import MfRows
from newton._src.solvers.feather_pgs.friction_patches import FrictionPatches, PatchRowLaw
from newton._src.solvers.feather_pgs.kernels import (
    build_mf_contact_rows,
    populate_world_J_for_size,
    prepare_world_contact_rows,
)
from newton._src.solvers.feather_pgs.row_build import contact_row_prelude, populate_world_J_masked
from newton.tests.feather_pgs_law_harness import bind_struct, contact_input

# Normal +X gives tangents +Z and -Y: against three unit translation dofs, these are the
# Jacobian rows of a contact's normal and its two tangents.
ROW_DIRECTIONS = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=np.float32)


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


def translation_dofs(device):
    """Give the contacts' articulation three unit translation dofs, so each Jacobian row is its direction."""
    axes = np.zeros((3, 6), dtype=np.float32)
    axes[:, :3] = np.eye(3)
    return {
        "target_size": 3,
        "size_a": 3,
        "size_b": -1,
        "articulation_response_dof_count": wp.array([3], dtype=int, device=device),
        "art_group_idx": wp.zeros(1, dtype=int, device=device),
        "art_dof_start": wp.zeros(1, dtype=int, device=device),
        "body_to_joint": wp.zeros(1, dtype=int, device=device),
        "joint_ancestor": wp.array([-1], dtype=int, device=device),
        "joint_qd_start": wp.array([0, 3], dtype=int, device=device),
        "joint_S_s": wp.array(axes, dtype=wp.spatial_vector, device=device),
        "body_response_dof_mask": wp.array([0b111], dtype=wp.uint32, device=device),
        "shape_transform": wp.array([wp.transform_identity()] * 2, dtype=wp.transform, device=device),
    }


def launch(kernel, values, threads, device):
    """Launch an actual row producer using its declared argument names; ``data`` binds the contact input,
    ``mf`` the matrix-free rows and ``law`` the friction row law."""
    data, _arrays = contact_input(values, device)
    law = PatchRowLaw()
    law.patches = values["friction_patches"]
    law.beta = values["friction_anchor_beta"]
    arguments = {**values, "data": data, "mf": bind_struct(MfRows, values, device)[0], "law": law}
    wp.launch(
        kernel,
        dim=threads,
        inputs=[arguments[name] for name in inspect.signature(kernel.func).parameters],
        device=device,
    )


def launch_metadata(values, device):
    """Launch the shared metadata producer, which writes no Jacobian."""
    launch(prepare_world_contact_rows, values, values["total_num_threads"], device)
    return None, None


def launch_per_contact(values, device):
    """Accumulate the per-contact builder's rows into a cleared Jacobian, as the solver does."""
    jacobian = wp.zeros((1, values["world_row_type"].shape[1], 3), dtype=float, device=device)
    values["J_group"] = jacobian
    launch(populate_world_J_for_size, values, values["total_num_threads"], device)
    return jacobian.numpy()[0], 0.0


def launch_masked(values, device):
    """Launch the point-law masked prelude, three threads per contact, and its Jacobian pass, one per dof."""
    count = values["contact_path"].shape[0]
    jacobian = wp.full((1, values["world_row_type"].shape[1], 3), -9.0, dtype=float, device=device)
    values.update(
        total_threads=3 * count,
        pre_flags=wp.full(count, -9, dtype=int, device=device),
        J_group_a=jacobian,
        J_group_b=jacobian,
    )
    launch(contact_row_prelude, values, 3 * count, device)
    launch(populate_world_J_masked, values, 3 * count, device)
    return jacobian.numpy()[0], -9.0


def launch_matrix_free(values, device):
    """Route the contacts to the matrix-free builder, writing its row metadata into the same arrays."""
    count = values["contact_path"].shape[0]
    capacity = values["world_row_type"].shape[1]
    jacobian = wp.full((1, capacity, 6), -9.0, dtype=float, device=device)
    values.update(
        contact_path=wp.ones(count, dtype=int, device=device),
        has_target_velocity=0,
        mf_body_a=wp.full((1, capacity), -9, dtype=int, device=device),
        mf_body_b=wp.full((1, capacity), -9, dtype=int, device=device),
        mf_Z_a=jacobian,
        mf_Z_b=wp.full((1, capacity, 6), -9.0, dtype=float, device=device),
        mf_row_type=values["world_row_type"],
        mf_row_parent=values["world_row_parent"],
        mf_row_mu=values["world_row_mu"],
        mf_phi=values["world_phi"],
        mf_target_velocity=values["world_target_velocity"],
        mf_row_restitution=values["world_row_restitution"],
    )
    launch(build_mf_contact_rows, values, values["total_num_threads"], device)
    # The linear part of a matrix-free row is its direction.
    return jacobian.numpy()[0, :, :3], -9.0


# Every producer that writes a contact's rows from its allocator reservation.
ROW_BUILDERS = {
    "metadata": launch_metadata,
    "per_contact": launch_per_contact,
    "masked": launch_masked,
    "matrix_free": launch_matrix_free,
}


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
                self.assertEqual(solver.contact_friction_anchor_limit, 0)
                values.update(
                    total_threads=9,
                    size_a=18,
                    size_b=-1,
                    articulation_response_dof_count=wp.array([18], dtype=int, device=device),
                    pre_flags=wp.full(3, -9, dtype=int, device=device),
                )
                launch(contact_row_prelude, values, 9, device)
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
                launch(populate_world_J_masked, values, 54, device)
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
        """Preserve both one-to-three and three-to-one threshold disagreements in every row builder.

        Each contact's gap lies on the other side of the friction gap than its reservation, so a
        builder that recomputes eligibility writes tangent rows into the next contact's rows, or
        leaves two reserved rows unwritten.
        """
        below = np.nextafter(np.float32(0.02), np.float32(0.0))
        above = np.nextafter(np.float32(0.02), np.float32(1.0))
        for builder, launch_builder in ROW_BUILDERS.items():
            for needed, gaps in (([1, 3], [below, above]), ([3, 1], [above, below])):
                with self.subTest(device=str(device), builder=builder, needed=needed):
                    values, slots = metadata_inputs(device, needed, gaps)
                    values.update(translation_dofs(device))
                    jacobian, untouched = launch_builder(values, device)
                    kinds = values["world_row_type"].numpy()[0]
                    parents = values["world_row_parent"].numpy()[0]
                    expected_kinds, expected_parents = [], []
                    expected_jacobian = np.full((kinds.shape[0], 3), untouched, dtype=np.float32)
                    for slot, width in zip(slots, needed, strict=True):
                        expected_kinds += [0] + [2] * (width - 1)
                        expected_parents += [-1] + [int(slot)] * (width - 1)
                        expected_jacobian[slot : slot + width] = ROW_DIRECTIONS[:width]
                    np.testing.assert_array_equal(kinds[:4], expected_kinds)
                    np.testing.assert_array_equal(parents[:4], expected_parents)
                    np.testing.assert_array_equal(kinds[4:], -9)
                    np.testing.assert_array_equal(parents[4:], -9)
                    np.testing.assert_array_equal(values["world_phi"].numpy()[0, slots], gaps)
                    np.testing.assert_allclose(values["world_row_mu"].numpy()[0, slots], 0.7, atol=1.0e-7)
                    if jacobian is not None:
                        np.testing.assert_array_equal(jacobian, expected_jacobian)

    def test_reservations_cpu(self):
        """Keep CPU rows inside each reserved packet despite opposite gap admission."""
        self.check_reservations("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA is required")
    def test_reservations_cuda(self):
        """Keep CUDA rows inside each reserved packet despite opposite gap admission."""
        self.check_reservations("cuda:0")


if __name__ == "__main__":
    unittest.main()
