# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import dense_rows, field_group
from newton._src.solvers.feather_pgs.contact_input import ContactInput
from newton._src.solvers.feather_pgs.friction_patches import FrictionPatches
from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
    apply_world_contact_restitution_matrix_free,
    compute_world_contact_bias,
    populate_world_J_for_compact_size,
    prepare_world_contact_rows,
)
from newton._src.solvers.feather_pgs.prefix_rows import PrefixLaw
from newton._src.solvers.feather_pgs.response_builds_dense import cfm_worlds_kernel
from newton._src.solvers.feather_pgs.response_builds_rows import (
    factor_contact_rows_kernel,
    factor_limit_rows_kernel,
    factor_publish_kernel,
)
from newton._src.solvers.feather_pgs.sparse_contact import finalize_sparse_joint_limit_rows
from newton._src.solvers.feather_pgs.sparse_mass_matrix import _SparseRowSupport, factor_indices, row_supports
from newton._src.solvers.feather_pgs.tree_plan import TreePlan
from newton.tests.feather_pgs_law_harness import packed_world_rows, position_pass, world_restitution_layout
from newton.tests.feather_pgs_patch_parity import row_law
from newton.tests.test_feather_pgs_sparse_mass_matrix import _endpoint_support


def _array(value, dtype=float):
    return wp.array(value, dtype=dtype, device="cpu")


def _full(shape, value, dtype=float):
    return wp.full(shape, value, dtype=dtype, device="cpu")


def _contact_input(f, device="cpu", **fields):
    """The fixture's contacts and bodies as a ContactInput, with ``fields`` set too (host values go to ``device``)."""
    values = {
        "count": f["count"],
        "point0": f["point0"],
        "point1": f["point1"],
        "normal": f["normal"],
        "shape0": f["shape0"],
        "shape1": f["shape1"],
        "margin0": f["thickness0"],
        "margin1": f["thickness1"],
        "world": f["world"],
        "slot": f["slot"],
        "art0": f["art_a"],
        "art1": f["art_b"],
        "path": f["path"],
        "slots_needed": f["needed"],
        "dof_start": f["start"],
        "origin": f["origin"],
        "motion": f["motion"],
        "shape_body": f["shape_body"],
        "body_q": f["body_q"],
        "incident": f["velocity"],
    }
    values.update(fields)
    if "row_type" in values and "sides" not in values:
        # The contact rows record their sides (the packed placement's).
        values["sides"] = wp.zeros(values["row_type"].shape, dtype=wp.vec2i, device=device)
    return field_group.upload(ContactInput, device, **values)


def _fixture(device="cpu", joint_dof_count=(3, 2, 1, 0)):
    """Make a two-branch articulation with a multi-DOF root and fixed endpoint."""

    def array(value, dtype=float):
        return wp.array(value, dtype=dtype, device=device)

    tree = TreePlan.build([-1, 0, 0, 1], joint_dof_count)
    plan = tree.factor
    dofs = plan.dof_count
    rng = np.random.default_rng(916)
    lower = np.zeros((dofs, dofs))
    lower[plan.entry_rows, plan.columns] = rng.normal(scale=0.3, size=plan.nonzero_count)
    lower[np.diag_indices(dofs)] = 2.0
    inverse = np.linalg.inv(lower)
    patches = FrictionPatches()
    patches.enabled = 0
    patches.weight = array([1.0, 1.0, 0.0])
    patches.next_contact = array([-1, -1, -1], int)
    patches.point_a = array(rng.normal(size=(3, 3)), wp.vec3)
    patches.point_b = array(rng.normal(size=(3, 3)), wp.vec3)
    patches.phi = array(np.zeros((3, 2)), wp.vec2)
    normals = rng.normal(size=(3, 3))
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    physical_masks = [sum(1 << int(plan.permutation[i]) for i in _endpoint_support(plan, j)) for j in range(4)]
    return {
        "tree": tree,
        "plan": plan,
        "indices": factor_indices(plan, device),
        "inverse": inverse,
        "packed": array(inverse[plan.entry_rows, plan.columns][None, :]),
        "patches": patches,
        "count": array([3], int),
        "point0": array(rng.normal(size=(3, 3)), wp.vec3),
        "point1": array(rng.normal(size=(3, 3)), wp.vec3),
        "normal": array(normals, wp.vec3),
        "shape0": array([0, 0, 2], int),
        "shape1": array([3, 1, 3], int),
        "thickness0": array([0.01, 0.02, 0.03]),
        "thickness1": array([0.03, 0.01, 0.02]),
        "world": array([0, 0, 0], int),
        "slot": array([0, 3, 6], int),
        "art_a": array([0, 0, 0], int),
        "art_b": array([-1, 0, -1], int),
        "path": array([0, 0, 0], int),
        "needed": array([3, 3, 1], int),
        "group": array([0], int),
        "start": array([0], int),
        "origin": array([[0.3, -0.4, 0.7]], wp.vec3),
        "sparse_mask": array(plan.joint_ancestor_mask, wp.uint64),
        "physical_mask": array(physical_masks, wp.uint32 if dofs <= 32 else wp.uint64),
        "motion": array(rng.normal(size=(dofs, 6)), wp.spatial_vector),
        "shape_body": array([1, 2, 3, -1], int),
        "body_q": array([wp.transform(wp.vec3(*p), wp.quat_identity()) for p in rng.normal(size=(4, 3))], wp.transform),
        "velocity": array(rng.normal(size=dofs)),
    }


class TestSparseContacts(unittest.TestCase):
    def test_mixed_contact_response_matches_independent_mass_oracle(self):
        """Match physical mixed responses with arbitrary offsets and wide ancestry support."""
        for counts in ((1, 1, 1, 0), (3, 2, 1, 0), (6, 19, 18, 0), (6, 29, 29, 0)):
            with self.subTest(joint_dof_count=counts):
                self._check_mixed_contact_response("cpu", 1, counts)

    @unittest.skipUnless(wp.is_cuda_available(), "cooperative contact construction requires CUDA")
    def test_mixed_contact_response_cuda(self):
        """Preserve mixed responses across partial warps, group widths and grid strides."""
        for lanes in (8, 16, 32):
            counts_to_check = (
                ((1, 1, 1, 0), (3, 2, 1, 0), (6, 19, 18, 0), (6, 29, 29, 0))
                if lanes == 8
                else ((3, 2, 1, 0), (6, 29, 29, 0))
            )
            for counts in counts_to_check:
                for workers in (1, 2, 7) if lanes == 8 else (2,):
                    with self.subTest(joint_dof_count=counts, workers=workers, lanes=lanes):
                        self._check_mixed_contact_response("cuda:0", workers, counts, lanes=lanes)

    def _check_mixed_contact_response(self, device, workers, joint_dof_count=(3, 2, 1, 0), *, lanes=8):
        """Match robot/robot, robot/object, object/object, ground and self-contact at distinct offsets."""

        def array(value, dtype=float):
            return wp.array(value, dtype=dtype, device=device)

        def full(shape, value, dtype=float):
            return wp.full(shape, value, dtype=dtype, device=device)

        f = _fixture(device, joint_dof_count)
        p, indices = f["plan"], f["indices"]
        dofs = p.dof_count
        rng = np.random.default_rng(500)
        art_a = np.array([1, 0, 0, 2, 1, 1, 3, 3], dtype=np.int32)
        art_b = np.array([0, 1, 2, -1, 1, 3, 1, -1], dtype=np.int32)
        body_a = np.array([1, 4, 4, 5, 1, 1, 8, 9], dtype=np.int32)
        body_b = np.array([4, 1, 5, -1, 2, 8, 1, -1], dtype=np.int32)
        contact_count = len(art_a)
        articulated = np.array([False, True, False, True])
        start = np.array([1, 8, 8 + dofs + 6, 8 + dofs + 16])
        offset, group = np.array([0, 6, 6 + dofs, 12 + dofs]), np.array([1, 0, 0, 1])
        world_dofs = 2 * dofs + 12
        global_dofs = int(start[3] + dofs)
        second_lower = np.zeros((dofs, dofs))
        second_lower[p.entry_rows, p.columns] = rng.normal(scale=0.2, size=p.nonzero_count)
        second_lower[np.diag_indices(dofs)] = 1.7
        articulated_inverse = np.stack((f["inverse"], np.linalg.inv(second_lower)))
        free_lower = np.tril(rng.normal(scale=0.1, size=(2, 6, 6))) + 1.4 * np.eye(6)
        free_inverse = np.linalg.inv(free_lower)
        origin = rng.normal(size=(4, 3)).astype(np.float32)
        motion = rng.normal(size=(global_dofs, 6)).astype(np.float32)
        velocity = rng.normal(size=global_dofs).astype(np.float32)
        point_a, point_b = rng.normal(size=(2, contact_count, 3)).astype(np.float32)
        normal = rng.normal(size=(contact_count, 3))
        normal /= np.linalg.norm(normal, axis=1)[:, None]
        normal = normal.astype(np.float32)
        # Reversing either a mixed or two-articulation contact preserves its row.
        for first, reverse in ((0, 1), (5, 6)):
            point_a[reverse], point_b[reverse], normal[reverse] = point_b[first], point_a[first], -normal[first]
        transforms = [
            wp.transform(wp.vec3(*position), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), float(angle)))
            for position, angle in zip(rng.normal(size=(10, 3)), np.linspace(-0.6, 0.8, 10), strict=True)
        ]
        masks = np.concatenate((p.joint_ancestor_mask, np.array([63, 63], dtype=np.uint64), p.joint_ancestor_mask))
        row_shape = (1, contact_count, max(2 * dofs, 12))
        row_dof, row_factor = full(row_shape, -9, int), full(row_shape, -9.0)
        row_free_response = full(row_shape, -9.0)
        incident, diagonal = full(row_shape[:2], -9.0), full(row_shape[:2], -9.0)
        data = field_group.upload(
            ContactInput,
            device,
            count=array([contact_count], int),
            point0=array(point_a, wp.vec3),
            point1=array(point_b, wp.vec3),
            normal=array(normal, wp.vec3),
            shape0=array(body_a, int),
            shape1=array(np.where(body_b < 0, len(transforms), body_b), int),
            margin0=full(contact_count, 0.0),
            margin1=full(contact_count, 0.0),
            world=full(contact_count, 0, int),
            slot=array(np.arange(contact_count), int),
            art0=array(art_a, int),
            art1=array(art_b, int),
            path=full(contact_count, 0, int),
            slots_needed=full(contact_count, 1, int),
            response_dofs=array(np.where(articulated, dofs, 6), int),
            dof_start=array(start, int),
            origin=array(origin, wp.vec3),
            motion=array(motion, wp.spatial_vector),
            shape_body=array([*range(len(transforms)), -1], int),
            body_q=array(transforms, wp.transform),
            shared_anchor=0,
            friction_shared_anchor=0,
            incident=array(velocity),
            dt=0.01,
            diag=diagonal,
        )
        wp.launch(
            factor_contact_rows_kernel(dofs, lanes_per_contact=lanes),
            dim=workers * lanes,
            inputs=[
                data,
                workers,
                array(group, int),
                array(~articulated, int),
                array(offset, int),
                array(masks, wp.uint64),
                row_law(f["patches"]),
                indices.permutation,
                indices.row_offsets,
                array(articulated_inverse[:, p.entry_rows, p.columns]),
                array(free_inverse),
            ],
            outputs=[row_dof, row_factor, row_free_response, incident, _SparseRowSupport()],
            block_dim=128,
            device=device,
        )
        mass, transform = np.zeros((world_dofs, world_dofs)), np.zeros((world_dofs, world_dofs))
        compact_velocity = np.zeros(world_dofs)
        free_coordinates = np.zeros(world_dofs, dtype=bool)
        for art, is_articulated in enumerate(articulated):
            width = dofs if is_articulated else 6
            block = slice(offset[art], offset[art] + width)
            permutation = p.permutation if is_articulated else np.arange(6)
            inverse = articulated_inverse[group[art]] if is_articulated else free_inverse[group[art]]
            local_transform = inverse @ np.eye(width)[permutation]
            transform[block, block] = local_transform if is_articulated else np.eye(6)
            local_decode = np.linalg.inv(local_transform)
            mass[block, block] = local_decode @ local_decode.T
            compact_velocity[block] = velocity[start[art] : start[art] + width]
            free_coordinates[block] = not is_articulated
        jacobian = np.zeros((contact_count, world_dofs))
        for row in range(contact_count):
            for body, art, point, sign in (
                (body_a[row], art_a[row], point_a[row], 1.0),
                (body_b[row], art_b[row], point_b[row], -1.0),
            ):
                if body < 0:
                    continue
                position = np.asarray(wp.transform_point(transforms[body], wp.vec3(*point)))
                permutation = p.permutation if articulated[art] else np.arange(6)
                for node, physical in enumerate(permutation):
                    if not (int(masks[body]) & (1 << node)):
                        continue
                    axis = motion[start[art] + physical]
                    point_velocity = axis[:3] + np.cross(axis[3:], position - origin[art])
                    jacobian[row, offset[art] + physical] += sign * (-normal[row] @ point_velocity)
        actual = np.zeros((contact_count, world_dofs))
        actual_response = np.zeros_like(actual)
        for row, support in enumerate(row_dof.numpy()[0]):
            valid = support >= 0
            self.assertEqual(np.unique(support[valid]).size, np.count_nonzero(valid))
            actual[row, support[valid]] = row_factor.numpy()[0, row, valid]
            for entry, coordinate in enumerate(support):
                if coordinate < 0:
                    continue
                actual_response[row, coordinate] = (
                    row_free_response.numpy()[0, row, entry]
                    if free_coordinates[coordinate]
                    else row_factor.numpy()[0, row, entry]
                )
        expected = jacobian @ transform.T
        np.testing.assert_allclose(actual, expected, atol=3.0e-6, rtol=3.0e-6)
        np.testing.assert_allclose(actual[0], actual[1], atol=3.0e-6)
        np.testing.assert_allclose(actual[5], actual[6], atol=3.0e-6)
        np.testing.assert_allclose(incident.numpy()[0], jacobian @ compact_velocity, atol=6.0e-6, rtol=3.0e-6)
        gram = jacobian @ np.linalg.solve(mass, jacobian.T)
        expected_response = np.linalg.solve(mass, jacobian.T).T @ np.linalg.inv(transform)
        np.testing.assert_allclose(actual_response, expected_response, atol=3.0e-6, rtol=3.0e-6)
        np.testing.assert_allclose(actual @ actual_response.T, gram, atol=1.0e-5, rtol=4.0e-6)
        np.testing.assert_allclose(diagonal.numpy()[0], np.diag(gram), atol=1.0e-5, rtol=4.0e-6)

    def test_contact_response_matches_dense_jacobian(self):
        """Match dense contact rows for static, same-articulation, and fixed endpoints."""
        self._check_contact_response("cpu", 1)

    @unittest.skipUnless(wp.is_cuda_available(), "cooperative contact construction requires CUDA")
    def test_contact_response_cuda(self):
        """Preserve anchors, patch tangents and reserved row extents with cooperative writes."""
        for lanes in (8, 16, 32):
            for workers in (1, 2, 5) if lanes == 8 else (2,):
                with self.subTest(workers=workers, lanes=lanes):
                    self._check_contact_response("cuda:0", workers, lanes=lanes)

    def test_contact_preparation_matches_canonical_kernels(self):
        """Match solver rows, diagonal CFM and restitution RHS to canonical preparation."""
        self._check_contact_response("cpu", 1, prepare_rows=True)

    def test_support_table_contact_response(self):
        """Retain self contacts, fixed endpoints and prescribed motion with shared support indices."""
        for prepare_rows in (False, True):
            self._check_contact_response("cpu", 1, prepare_rows=prepare_rows, use_support_table=True)

    @unittest.skipUnless(wp.is_cuda_available(), "cooperative contact construction requires CUDA")
    def test_support_table_contact_response_cuda(self):
        """Use the same padded topology support across contact lane widths and partial launches."""
        for lanes in (8, 16, 32):
            for prepare_rows in (False, True):
                self._check_contact_response(
                    "cuda:0", 2, lanes=lanes, prepare_rows=prepare_rows, use_support_table=True
                )

    @unittest.skipUnless(wp.is_cuda_available(), "cooperative contact preparation requires CUDA")
    def test_contact_preparation_cuda(self):
        """Preserve prepared rows across lane groups, partial launches and grid strides."""
        for lanes in (8, 16, 32):
            for workers in (1, 2, 5) if lanes == 8 else (2,):
                with self.subTest(workers=workers, lanes=lanes):
                    self._check_contact_response("cuda:0", workers, lanes=lanes, prepare_rows=True)

    def _check_contact_response(self, device, workers, *, lanes=8, prepare_rows=False, use_support_table=False):
        """Match dense contact rows for static, same-articulation, and fixed endpoints."""

        def array(value, dtype=float):
            return wp.array(value, dtype=dtype, device=device)

        def full(shape, value, dtype=float):
            return wp.full(shape, value, dtype=dtype, device=device)

        f = _fixture(device)
        p = f["plan"]
        indices = f["indices"]
        if prepare_rows:
            # Body zero is not a responding endpoint in the original fixture.
            # Use it as a separately prescribed articulation for contact zero.
            f["shape_body"] = array([1, 2, 3, -1, 0], int)
            f["shape1"] = array([4, 1, 3], int)
            f["art_b"] = array([1, 0, -1], int)
            f["group"] = array([0, 0], int)
            f["start"] = array([0, 6], int)
            f["origin"] = array([[0.3, -0.4, 0.7], [-0.2, 0.1, 0.4]], wp.vec3)
            f["patches"].phi = array([[0.01, -0.02], [-0.03, 0.04], [0.0, 0.0]], wp.vec2)
            points = f["point1"].numpy()
            transforms = f["body_q"].numpy()
            for contact, body_a, body_b, gap in ((0, 1, 0, -0.005), (2, 3, -1, 0.02)):
                normal = -wp.vec3(*f["normal"].numpy()[contact])
                point_a = wp.transform_point(wp.transform(*transforms[body_a]), wp.vec3(*f["point0"].numpy()[contact]))
                point_a -= float(f["thickness0"].numpy()[contact]) * normal
                point_b = point_a - (gap + float(f["thickness1"].numpy()[contact])) * normal
                if body_b >= 0:
                    point_b = wp.transform_point(wp.transform_inverse(wp.transform(*transforms[body_b])), point_b)
                points[contact] = np.asarray(point_b)
            f["point1"] = array(points, wp.vec3)
        response_dofs = array([6, 0] if prepare_rows else [6], int)
        for shared, friction_shared, patch in [(0, 0, 0), (0, 1, 0), (1, 0, 0), (0, 0, 1), (1, 1, 1)]:
            with self.subTest(shared=shared, friction_shared=friction_shared, patch=patch):
                f["patches"].enabled = patch
                dense = full((1, 9, 6), 0.0)
                row_dof = full((1, 1, 1) if use_support_table else (1, 9, 6), -9, int)
                row_factor = full((1, 9, 6), -9.0)
                incident = full((1, 9), -9.0)
                diagonal = full((1, 9), -9.0)
                support = _SparseRowSupport()
                law = {}
                if use_support_table:
                    support_plan = f["tree"].support_classes(p.joint_ancestor_mask, 6)
                    self.assertIsNotNone(support_plan)
                    support = row_supports(support_plan, 1, 9, device)
                    support.row_class.fill_(-9)
                friction_beta = 0.0
                if prepare_rows:
                    velocity = np.zeros((4, 6), dtype=np.float32)
                    velocity[0, :3] = -0.2 * f["normal"].numpy()[0]
                    velocity[0, 3:] = [0.03, -0.04, 0.02]
                    friction_beta = 0.23
                    law = {
                        "body_v": array(velocity, wp.spatial_vector),
                        "prescribed": array([0, 1], int),
                        "material_mu": array([0.8, 0.5, 0.3, 0.6, 0.4]),
                        "material_restitution": array([0.4, 0.2, 0.5, 0.8, 0.6]),
                        "beta": 0.05,
                        "cfm": 0.03,
                        "friction_scale": 0.8,
                        "speculative_scale": 0.65,
                        "restitution_threshold": 0.1,
                    }
                    for name in ("row_type", "row_parent"):
                        law[name] = full((1, 9), -9, int)
                    for name in ("row_mu", "row_beta", "row_cfm", "phi", "target", "restitution", "rhs"):
                        law[name] = full((1, 9), -9.0)
                compact = _contact_input(
                    f,
                    device,
                    response_dofs=response_dofs,
                    body_mask=f["physical_mask"],
                    shared_anchor=shared,
                    friction_shared_anchor=friction_shared,
                )
                wp.launch(
                    populate_world_J_for_compact_size,
                    dim=(1, 32),
                    inputs=[compact, 1, 6, f["group"], row_law(f["patches"])],
                    outputs=[dense],
                    device=device,
                )
                if prepare_rows:
                    normal_jacobian = dense.numpy()[0, 0]
                    f["velocity"] = array(-1.7 * normal_jacobian / np.dot(normal_jacobian, normal_jacobian))
                data = _contact_input(
                    f,
                    device,
                    response_dofs=response_dofs,
                    shared_anchor=shared,
                    friction_shared_anchor=friction_shared,
                    dt=0.01,
                    diag=diagonal,
                    **law,
                )
                wp.launch(
                    factor_contact_rows_kernel(
                        6, lanes_per_contact=lanes, prepare_rows=prepare_rows, use_support_table=use_support_table
                    ),
                    dim=workers * lanes,
                    inputs=[
                        data,
                        workers,
                        f["group"],
                        full(response_dofs.shape, 0, int),
                        full(response_dofs.shape, 0, int),
                        f["sparse_mask"],
                        row_law(f["patches"], friction_beta),
                        indices.permutation,
                        indices.row_offsets,
                        f["packed"],
                        array(np.eye(6)[None]),
                    ],
                    outputs=[row_dof, row_factor, full((1, 9, 12), -9.0), incident, support],
                    block_dim=128,
                    device=device,
                )
                jacobian = dense.numpy()[0, :7]
                expected = jacobian[:, p.permutation] @ f["inverse"].T
                actual = np.zeros_like(expected)
                dof_values = row_dof.numpy()
                if use_support_table:
                    np.testing.assert_array_equal(dof_values, -9)
                    classes = support.row_class.numpy()
                    self.assertTrue(np.all(classes[0, :7] >= 0))
                    np.testing.assert_array_equal(classes[0, 7:], -9)
                    dof_values = np.full((1, 9, 6), -9, dtype=np.int32)
                    dof_values[0, :7] = support_plan.class_dof[classes[0, :7]]
                for row in range(7):
                    dofs = dof_values[0, row]
                    valid = dofs >= 0
                    self.assertEqual(len(set(dofs[valid])), int(np.count_nonzero(valid)))
                    actual[row, dofs[valid]] = row_factor.numpy()[0, row, valid]
                np.testing.assert_allclose(actual, expected, rtol=3.0e-6, atol=2.0e-6)
                np.testing.assert_allclose(
                    incident.numpy()[0, :7], jacobian @ f["velocity"].numpy(), rtol=3.0e-6, atol=2.0e-6
                )
                np.testing.assert_allclose(
                    diagonal.numpy()[0, :7],
                    np.sum(expected**2, axis=1) + (law["cfm"] if prepare_rows else 0.0),
                    rtol=3.0e-6,
                    atol=2.0e-6,
                )
                np.testing.assert_array_equal(dof_values[0, 7:], -9)
                np.testing.assert_array_equal(row_factor.numpy()[0, 7:], -9.0)
                if prepare_rows:
                    reference = [full((1, 9), -9, int) for _ in range(2)] + [full((1, 9), -9.0) for _ in range(6)]
                    reference_data = _contact_input(
                        f,
                        device,
                        response_dofs=response_dofs,
                        shared_anchor=shared,
                        friction_shared_anchor=friction_shared,
                        **{
                            **law,
                            **dict(
                                zip(
                                    ("row_type", "row_parent", "row_mu", "row_beta", "row_cfm", "phi", "target"),
                                    reference[:7],
                                    strict=True,
                                )
                            ),
                            "restitution": reference[7],
                        },
                    )
                    wp.launch(
                        prepare_world_contact_rows,
                        dim=workers,
                        inputs=[reference_data, workers, row_law(f["patches"], friction_beta)],
                        device=device,
                    )
                    count = array([7], int)
                    reference_diag = array(
                        np.pad(np.sum(expected**2, axis=1)[None], ((0, 0), (0, 2)), constant_values=-9)
                    )
                    wp.launch(
                        cfm_worlds_kernel(),
                        dim=1,
                        inputs=[count, reference[4], array([0], int), 0],
                        outputs=[reference_diag],
                        device=device,
                    )
                    reference_rhs = full((1, 9), -9.0)
                    reference_rows = field_group.upload(
                        dense_rows.DenseRows,
                        device,
                        **{field.name: value for field, value in zip(dense_rows.RECORDS, reference, strict=False)},
                        rhs=reference_rhs,
                        row_w=full((1, 9), -9.0),
                    )
                    wp.launch(
                        compute_world_contact_bias,
                        dim=9,
                        inputs=[
                            *(reference_rows, count, 9, 0.01, 1.0, law["speculative_scale"], 1.0, 1.0),
                            *position_pass(device),
                        ],
                        device=device,
                    )
                    geometric_rhs = reference_rhs.numpy().copy()
                    wp.launch(
                        apply_world_contact_restitution_matrix_free,
                        dim=9,
                        inputs=[
                            reference_rows,
                            count,
                            9,
                            f["velocity"],
                            array([np.arange(6)], int),
                            *packed_world_rows(dense.numpy(), device),
                            *world_restitution_layout(device),
                            0.01,
                            law["restitution_threshold"],
                            0,
                        ],
                        device=device,
                    )
                    self.assertNotAlmostEqual(float(geometric_rhs[0, 0]), float(reference_rhs.numpy()[0, 0]))
                    self.assertGreater(float(reference[5].numpy()[0, 6]), 0.0)
                    self.assertNotEqual(float(reference[6].numpy()[0, 0]), 0.0)
                    # Prepared rows consume bias/CFM/restitution internally;
                    # compare their final solver inputs, not unused scratch.
                    for name, expected_metadata in zip(
                        ("row_type", "row_parent", "row_mu"), reference[:3], strict=True
                    ):
                        np.testing.assert_allclose(
                            law[name].numpy(), expected_metadata.numpy(), atol=2.0e-6, rtol=3.0e-6
                        )
                    np.testing.assert_allclose(diagonal.numpy(), reference_diag.numpy(), atol=3.0e-6, rtol=3.0e-6)
                    np.testing.assert_allclose(law["rhs"].numpy(), reference_rhs.numpy(), atol=5.0e-6, rtol=3.0e-6)

    def test_contact_metadata_honors_allocated_extent(self):
        """Keep reserved one-row contacts from writing newly eligible friction rows."""
        f = _fixture()
        # Friction would pass a recomputed geometric/material predicate. The
        # allocator's one-row extent is authoritative despite that discrepancy.
        f["count"] = _array([1], int)
        f["needed"] = _array([1, 3, 1], int)
        row_type = _full((1, 9), -9, int)
        row_parent = _full((1, 9), -9, int)
        other = [_full((1, 9), -9.0) for _ in range(6)]
        data = _contact_input(
            f,
            body_v=_full(4, wp.spatial_vector(0.0), wp.spatial_vector),
            prescribed=_array([0], int),
            material_mu=_array([0.8] * 4),
            material_restitution=_array([0.4] * 4),
            shared_anchor=0,
            friction_shared_anchor=0,
            friction_scale=1.0,
            beta=0.05,
            cfm=1.0e-6,
            row_type=row_type,
            row_parent=row_parent,
            **dict(zip(("row_mu", "row_beta", "row_cfm", "phi", "target", "restitution"), other, strict=True)),
        )
        wp.launch(prepare_world_contact_rows, dim=1, inputs=[data, 1, row_law(f["patches"])], device="cpu")
        self.assertEqual(row_type.numpy()[0, 0], PGS_CONSTRAINT_TYPE_CONTACT)
        for output in [row_type, row_parent, *other]:
            np.testing.assert_array_equal(output.numpy()[0, 1:], -9)

    def test_joint_limit_finalization_matches_canonical_kernels(self):
        """Finish positive, zero and violated limits without revisiting completed contact rows."""
        self._check_joint_limit_finalization("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "cooperative prefix preparation requires CUDA")
    def test_joint_limit_finalization_cuda(self):
        """Respect both the finalized count and the pre-contact prefix on CUDA."""
        self._check_joint_limit_finalization("cuda:0")

    def _check_joint_limit_finalization(self, device):
        """Compare prefix-only preparation with the canonical row kernels."""

        def array(value, dtype=float):
            return wp.array(value, dtype=dtype, device=device)

        shape = (3, 6)
        initial_diag = np.arange(18, dtype=np.float32).reshape(shape) + 1.0
        rows = field_group.upload(
            dense_rows.DenseRows,
            device,
            row_type=np.full(shape, PGS_CONSTRAINT_TYPE_JOINT_LIMIT),
            phi=np.tile([-0.02, 0.0, 0.03, -0.1, 0.2, -0.3], (3, 1)),
            row_beta=np.linspace(0.01, 0.2, 18).reshape(shape),
            row_cfm=np.linspace(0.02, 0.3, 18).reshape(shape),
            target=np.linspace(-0.3, 0.4, 18).reshape(shape),
            rhs=np.full(shape, -9.0),
            diag=initial_diag,
        )
        diagonal = rows.diag
        reference_diag = array(initial_diag)
        reference_rhs = array(np.full(shape, -9.0))
        prefix_count = array([4, 2, 0], int)
        wp.launch(
            finalize_sparse_joint_limit_rows,
            dim=3,
            inputs=[array([[0, 4], [0, 3], [0, 0]], int), array([6, 2, 5], int), rows, 0.0123],
            device=device,
        )
        wp.launch(
            cfm_worlds_kernel(),
            dim=3,
            inputs=[prefix_count, rows.row_cfm, array([0, 0, 0], int), 0],
            outputs=[reference_diag],
            device=device,
        )
        wp.launch(
            compute_world_contact_bias,
            dim=18,
            inputs=[
                field_group.replace(rows, rhs=reference_rhs, row_w=array(np.full(shape, -9.0))),
                prefix_count,
                6,
                0.0123,
                1.0,
                1.0,
                1.0,
                1.0,
                *position_pass(device),
            ],
            device=device,
        )
        np.testing.assert_array_equal(diagonal.numpy(), reference_diag.numpy())
        np.testing.assert_allclose(rows.rhs.numpy(), reference_rhs.numpy(), atol=2.0e-7, rtol=2.0e-6)

    def test_joint_limit_rows_and_capacity(self):
        """Match signed inverse-factor columns and preserve limit allocation order."""
        f = _fixture()
        plan, indices = f["plan"], f["indices"]
        expected_rows = [(0, 1.0, -0.02), (1, -1.0, 0.01), (3, -1.0, -0.2), (4, 1.0, 0.0), (4, -1.0, 0.0)]
        for capacity in (6, 3):
            with self.subTest(capacity=capacity):
                count = _array([1], int)
                row_type, parent = [_full((1, capacity), -9, int) for _ in range(2)]
                mu, beta, cfm, phi, target, incident, diagonal = [_full((1, capacity), -9.0) for _ in range(7)]
                dof = _full((1, capacity, 6), -9, int)
                factor = _full((1, capacity, 6), -9.0)
                wp.launch(
                    factor_limit_rows_kernel,
                    dim=32,
                    inputs=[
                        f["group"],
                        _array([0], int),
                        _array([0], int),
                        f["start"],
                        field_group.upload(
                            PrefixLaw,
                            "cpu",
                            limit_q_index=_array([0, 1, -1, 3, 4, 5], int),
                            lower=_array([-1.0, -1.0, -np.inf, -0.5, 0.0, -2.0]),
                            upper=_array([1.0, 1.0, np.inf, 0.5, 0.0, 2.0]),
                            q=_array([-1.02, 0.99, 0.0, 0.7, 0.0, -1.0]),
                            activation_gap=0.02,
                            beta=0.05,
                            cfm=1.0e-6,
                            slot_counter=count,
                            row_type=row_type,
                            row_parent=parent,
                            row_mu=mu,
                            row_beta=beta,
                            row_cfm=cfm,
                            phi=phi,
                            target=target,
                            diag=diagonal,
                        ),
                        indices.ancestor_mask,
                        indices.inverse_permutation,
                        indices.lookup,
                        f["packed"],
                        f["velocity"],
                    ],
                    outputs=[dof, factor, incident, _SparseRowSupport()],
                    block_dim=128,
                    device="cpu",
                )
                self.assertEqual(count.numpy()[0], 6)
                self.assertEqual(row_type.numpy()[0, 0], -9)
                for row, (physical, sign, gap) in enumerate(expected_rows[: capacity - 1], 1):
                    expected = sign * f["inverse"][:, plan.inverse_permutation[physical]]
                    actual = np.zeros(6)
                    active = dof.numpy()[0, row] >= 0
                    actual[dof.numpy()[0, row, active]] = factor.numpy()[0, row, active]
                    np.testing.assert_allclose(actual, expected, atol=1.0e-7)
                    self.assertEqual(row_type.numpy()[0, row], PGS_CONSTRAINT_TYPE_JOINT_LIMIT)
                    self.assertEqual(parent.numpy()[0, row], -1)
                    self.assertAlmostEqual(float(phi.numpy()[0, row]), gap, places=6)
                    self.assertAlmostEqual(
                        float(incident.numpy()[0, row]), sign * f["velocity"].numpy()[physical], places=6
                    )
                    self.assertAlmostEqual(float(diagonal.numpy()[0, row]), float(expected @ expected), places=6)

    def _check_joint_limit_parallel_scan(self, device, capacity, *, use_support_table=False):
        tree = TreePlan.build([-1, 0, 0], [2, 31, 31])
        plan = tree.factor
        dofs, worlds = plan.dof_count, 3
        rng = np.random.default_rng(923)
        lower_factor = np.zeros((dofs, dofs))
        lower_factor[plan.entry_rows, plan.columns] = rng.normal(scale=0.02, size=plan.nonzero_count)
        lower_factor[np.diag_indices(dofs)] = 1.5
        inverse = np.linalg.inv(lower_factor)
        packed = np.stack([(1.0 + group * 0.1) * inverse[plan.entry_rows, plan.columns] for group in range(worlds)])
        group_to_art = np.array([2, 0, 1], dtype=np.int32)
        art_to_world = np.array([1, 2, 0], dtype=np.int32)
        start = np.arange(worlds, dtype=np.int32) * dofs
        q_index = np.arange(worlds * dofs, dtype=np.int32)
        lower = np.full((worlds, dofs), -1.0, dtype=np.float32)
        upper = np.full((worlds, dofs), 1.0, dtype=np.float32)
        position = np.zeros((worlds, dofs), dtype=np.float32)
        # Articulation zero has no active rows. The others cross warp and 64-bit mask boundaries.
        position[1, [0, 15, 32]] = -1.0
        position[1, [16, 47]] = 1.0
        lower[1, [31, 63]] = upper[1, [31, 63]] = 0.0
        lower[2] = upper[2] = 0.0
        q_index[2 * dofs + 5] = -1
        lower[2, 7] = -np.inf
        upper[2, 8] = np.inf
        lower[2, 9] = np.nan
        velocity = rng.normal(size=worlds * dofs).astype(np.float32)
        initial_count = np.array([1, 2, 3], dtype=np.int32)

        def array(value, dtype=float):
            return wp.array(value, dtype=dtype, device=device)

        def full(shape, value, dtype=float):
            return wp.full(shape, value, dtype=dtype, device=device)

        indices = factor_indices(plan, device)
        counter = array(initial_count, int)
        row_type, parent = [full((worlds, capacity), -9, int) for _ in range(2)]
        mu, beta, cfm, phi, target, incident, diagonal = [full((worlds, capacity), -9.0) for _ in range(7)]
        row_dof = full((1, 1, 1) if use_support_table else (worlds, capacity, 34), -9, int)
        row_factor = full((worlds, capacity, 34), -9.0)
        support = _SparseRowSupport()
        if use_support_table:
            support_plan = tree.support_classes(np.zeros(0, dtype=np.uint64), 34)
            self.assertIsNotNone(support_plan)
            support = row_supports(support_plan, worlds, capacity, device)
            support.row_class.fill_(-9)
        outputs = [counter, row_type, parent, mu, beta, cfm, phi, target, row_dof, row_factor, incident, diagonal]
        wp.launch(
            factor_limit_rows_kernel,
            dim=worlds * 32,
            inputs=[
                array(group_to_art, int),
                array(art_to_world, int),
                array([0, 0, 0], int),
                array(start, int),
                field_group.upload(
                    PrefixLaw,
                    device,
                    limit_q_index=array(q_index, int),
                    lower=array(lower.reshape(-1)),
                    upper=array(upper.reshape(-1)),
                    q=array(position.reshape(-1)),
                    activation_gap=0.01,
                    beta=0.05,
                    cfm=1.0e-6,
                    slot_counter=counter,
                    row_type=row_type,
                    row_parent=parent,
                    row_mu=mu,
                    row_beta=beta,
                    row_cfm=cfm,
                    phi=phi,
                    target=target,
                    diag=diagonal,
                ),
                indices.ancestor_mask,
                indices.inverse_permutation,
                indices.lookup,
                array(packed),
                array(velocity),
            ],
            outputs=[row_dof, row_factor, incident, support],
            block_dim=128,
            device=device,
        )
        actual = [value.numpy() for value in outputs]
        if use_support_table:
            np.testing.assert_array_equal(actual[8], -9)
            classes = support.row_class.numpy()
            actual[8] = np.full((worlds, capacity, 34), -9, dtype=np.int32)
            selected = classes >= 0
            actual[8][selected] = support_plan.class_dof[classes[selected]]
        for group, art in enumerate(group_to_art):
            world = art_to_world[art]
            first = initial_count[world]
            expected = []
            for local_dof in range(dofs):
                if q_index[start[art] + local_dof] < 0:
                    continue
                for side in (0, 1):
                    bound = (lower if side == 0 else upper)[art, local_dof]
                    q = position[art, local_dof]
                    active = q <= bound + 0.01 if side == 0 else q >= bound - 0.01
                    if np.isfinite(bound) and active:
                        expected.append((local_dof, 1.0 if side == 0 else -1.0, q - bound))
            self.assertEqual(actual[0][world], first + len(expected))
            for output in actual[1:]:
                np.testing.assert_array_equal(output[world, :first], -9)
                np.testing.assert_array_equal(output[world, first + len(expected) :], -9)
            for row, (local_dof, sign, gap) in enumerate(expected[: capacity - first], first):
                support = np.flatnonzero([int(plan.ancestor_mask[local_dof]) & (1 << bit) for bit in range(dofs)])
                np.testing.assert_array_equal(actual[8][world, row, : len(support)], support)
                np.testing.assert_array_equal(actual[8][world, row, len(support) :], -1)
                np.testing.assert_array_equal(actual[9][world, row, len(support) :], 0.0)
                expected_factor = sign * (1.0 + group * 0.1) * inverse[support, plan.inverse_permutation[local_dof]]
                np.testing.assert_allclose(actual[9][world, row, : len(support)], expected_factor, atol=1.0e-7)
                self.assertEqual(actual[1][world, row], PGS_CONSTRAINT_TYPE_JOINT_LIMIT)
                self.assertEqual(actual[2][world, row], -1)
                self.assertEqual(actual[3][world, row], 0.0)
                self.assertAlmostEqual(actual[4][world, row], 0.05, places=7)
                self.assertAlmostEqual(actual[5][world, row], 1.0e-6, places=9)
                self.assertAlmostEqual(actual[6][world, row], sign * gap, places=7)
                self.assertEqual(actual[7][world, row], 0.0)
                self.assertAlmostEqual(actual[10][world, row], sign * velocity[start[art] + local_dof], places=7)
                self.assertAlmostEqual(actual[11][world, row], expected_factor @ expected_factor, places=6)

    def test_joint_limit_parallel_scan_cpu(self):
        """Retain ordered rows, full overflow demand and zero-row worlds across 64 DOFs."""
        for capacity in (8, 140):
            with self.subTest(capacity=capacity):
                self._check_joint_limit_parallel_scan("cpu", capacity)

    def test_support_table_joint_limits(self):
        """Keep full multi-DOF ancestor blocks and limit overflow demand without row-index storage."""
        for capacity in (8, 140):
            self._check_joint_limit_parallel_scan("cpu", capacity, use_support_table=True)

    @unittest.skipUnless(wp.is_cuda_available(), "cooperative limit construction requires CUDA")
    def test_support_table_joint_limits_cuda(self):
        """Publish topology classes for wide limits while leaving unused rows and index dummy untouched."""
        for capacity in (8, 140):
            self._check_joint_limit_parallel_scan("cuda:0", capacity, use_support_table=True)

    @unittest.skipUnless(wp.is_cuda_available(), "cooperative limit construction requires CUDA")
    def test_joint_limit_parallel_scan_cuda(self):
        """Match cooperative ballot ordering and responses to the scalar CPU oracle."""
        for capacity in (8, 140):
            with self.subTest(capacity=capacity):
                self._check_joint_limit_parallel_scan("cuda:0", capacity)

    def test_restitution_selects_impacts_from_the_incident_projection(self):
        """Select impacts and subtract targets with the matrix-free incident-projection law."""
        row_count = 7
        kinds = _array([[PGS_CONSTRAINT_TYPE_CONTACT] * 5 + [PGS_CONSTRAINT_TYPE_FRICTION] * 2], int)
        phi = _array([[-0.01, 0.01, 0.5, -0.01, -0.01, -0.01, -0.01]])
        incident = _array([[-2.0, -2.0, -2.0, -0.05, 0.5, -3.0, -4.0]])
        target = _array([[0.0, 0.2, 0.0, 0.0, 1.0, 0.0, 0.0]])
        restitution = _array([[0.4] * row_count])
        count = _array([6], int)
        rhs = _full((1, row_count), -7.0)
        wp.launch(
            apply_world_contact_restitution_matrix_free,
            dim=row_count,
            inputs=[
                field_group.upload(
                    dense_rows.DenseRows,
                    "cpu",
                    row_type=kinds,
                    phi=phi,
                    target=target,
                    restitution=restitution,
                    rhs=rhs,
                    row_w=_full((1, row_count), 1.0),
                ),
                count,
                row_count,
                _array(incident.numpy()[0]),
                _array([np.arange(row_count)], int),
                *packed_world_rows(np.eye(row_count)[None], "cpu"),
                *world_restitution_layout("cpu"),
                0.01,
                0.1,
                0,
            ],
            device="cpu",
        )
        np.testing.assert_allclose(rhs.numpy()[0], [-0.8, -1.08, -7.0, -7.0, -1.2, -7.0, -7.0], atol=1.0e-6)

    def test_decode_factor_velocity(self):
        """Decode accumulated factor impulses into the original physical DOF order."""
        f = _fixture()
        delta = np.arange(6, dtype=np.float32) * 0.17 - 0.4
        out = _full(6, -9.0)
        wp.launch(
            factor_publish_kernel,
            dim=(1, 6),
            inputs=[
                f["group"],
                _array([0], int),
                _array([0], int),
                f["start"],
                f["indices"].permutation,
                f["indices"].lookup,
                f["packed"],
                _array(delta[None]),
                f["velocity"],
            ],
            outputs=[out],
            device="cpu",
        )
        expected = f["velocity"].numpy().copy()
        expected[f["plan"].permutation] += f["inverse"].T @ delta
        np.testing.assert_allclose(out.numpy(), expected, rtol=2.0e-6, atol=2.0e-7)


if __name__ == "__main__":
    unittest.main()
