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
    compute_world_contact_bias,
    prepare_world_contact_rows,
)
from newton._src.solvers.feather_pgs.tree_plan import TreePlan
from newton.tests.feather_pgs_law_harness import row_lines
from newton.tests.feather_pgs_patch_parity import row_law
from newton.tests.test_feather_pgs_sparse_mass_matrix import _endpoint_support


def _array(value, dtype=float):
    return wp.array(value, dtype=dtype, device="cpu")


def _cfm_added(diag, cfm, counts) -> np.ndarray:
    """``diag`` with each world's first ``counts`` rows' cfm added, in float32: the canonical diagonal's cfm."""
    out = np.array(diag, dtype=np.float32, copy=True)
    cfm = np.asarray(cfm, dtype=np.float32)
    for world, count in enumerate(counts):
        out[world, :count] += cfm[world, :count]
    return out


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
        "max_depenetration_velocity": wp.full(f["body_q"].shape[0], np.inf, dtype=float, device=device),
        # The rigid law's contact weight: no regularization, so the producers write no row weight.
        "contact_w": 1.0,
    }
    values.update(fields)
    if "row_type" in values and "sides" not in values:
        # The contact rows record their sides (the packed placement's).
        values["sides"] = wp.zeros(values["row_type"].shape, dtype=wp.vec2i, device=device)
    return field_group.upload(ContactInput, device, **values, first=row_lines(device))


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
    patches.point = array(rng.normal(size=(3, 3)), wp.vec3)
    patches.phi = array(np.zeros((3, 2)), wp.vec2)
    normals = rng.normal(size=(3, 3))
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    physical_masks = [sum(1 << int(plan.permutation[i]) for i in _endpoint_support(plan, j)) for j in range(4)]
    return {
        "tree": tree,
        "plan": plan,
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
    def test_contact_metadata_honors_allocated_extent(self):
        """Keep reserved one-row contacts from writing newly eligible friction rows."""
        f = _fixture()
        # Friction would pass a recomputed geometric/material predicate. The
        # allocator's one-row extent is authoritative despite that discrepancy.
        f["count"] = _array([1], int)
        f["needed"] = _array([1, 3, 1], int)
        row_type = _full((1, 9), -9, int)
        row_parent = _full((1, 9), -9, int)
        other = [_full((1, 9), -9.0) for _ in range(5)]
        data = _contact_input(
            f,
            body_v=_full(4, wp.spatial_vector(0.0), wp.spatial_vector),
            response_dofs=_array([f["plan"].dof_count], int),
            prescribed=_array([0], int),
            material_mu=_array([0.8] * 4),
            material_restitution=_array([0.4] * 4),
            beta=0.05,
            cfm=1.0e-6,
            row_type=row_type,
            row_parent=row_parent,
            **dict(zip(("row_mu", "row_beta", "phi", "target", "restitution"), other, strict=True)),
        )
        wp.launch(prepare_world_contact_rows, dim=1, inputs=[data, 1, row_law(f["patches"])], device="cpu")
        self.assertEqual(row_type.numpy()[0, 0], PGS_CONSTRAINT_TYPE_CONTACT)
        for output in [row_type, row_parent, *other]:
            np.testing.assert_array_equal(output.numpy()[0, 1:], -9)

    def test_restitution_selects_impacts_from_the_incident_projection(self):
        """Select impacts and subtract targets with the matrix-free incident-projection law."""
        row_count = 7
        kinds = _array([[PGS_CONSTRAINT_TYPE_CONTACT] * 5 + [PGS_CONSTRAINT_TYPE_FRICTION] * 2], int)
        phi = _array([[-0.01, 0.01, 0.5, -0.01, -0.01, -0.01, -0.01]])
        incident = _array([[-2.0, -2.0, -2.0, -0.05, 0.5, -3.0, -4.0]])
        target = _array([[0.0, 0.2, 0.0, 0.0, 1.0, 0.0, 0.0]])
        count = _array([6], int)

        def right_hand_sides(coefficient):
            rhs = _full((1, row_count), -7.0)
            wp.launch(
                compute_world_contact_bias,
                dim=(1, row_count),
                inputs=[
                    field_group.upload(
                        dense_rows.DenseRows,
                        "cpu",
                        row_type=kinds,
                        phi=phi,
                        row_beta=_full((1, row_count), 0.0),
                        target=target,
                        restitution=_array([[coefficient] * row_count]),
                        rhs=rhs,
                        row_w=_full((1, row_count), 1.0),
                        first=row_lines("cpu"),
                    ),
                    incident,
                    count,
                    row_count,
                    row_count,
                    0.01,
                    1.0,
                    1.0,
                    0.1,
                    0,
                    0,
                ],
                device="cpu",
            )
            return rhs.numpy()[0]

        bias, bounced = right_hand_sides(0.0), right_hand_sides(0.4)
        np.testing.assert_allclose(bounced[[0, 1, 4]], [-0.8, -1.08, -1.2], atol=1.0e-6)
        np.testing.assert_array_equal(bounced[[2, 3, 5]], bias[[2, 3, 5]])
        self.assertEqual(bounced[6], -7.0)


if __name__ == "__main__":
    unittest.main()
