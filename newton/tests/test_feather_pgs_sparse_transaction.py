# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the block records the sparse contact builder writes for a single-stage contact transaction."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs.kernels import populate_world_J_for_compact_size
from newton._src.solvers.feather_pgs.sparse_contact import (
    CONTACT_RECORD_FLOATS,
    SparseContactRows,
    _get_sparse_contact_response_kernel,
)
from newton._src.solvers.feather_pgs.sparse_mass_matrix import _SparseRowSupportPlan
from newton.tests import test_feather_pgs_sparse_contact_triplets as triplets
from newton.tests.test_feather_pgs_sparse_contacts import _fixture


class TestContactRecordBuilder(unittest.TestCase):
    """The sparse contact builder writes each contact triplet's block record beside its rows."""

    def _build(self, device, records, shared, friction_shared, patch):
        """Run the prepared support-table builder on the two-branch fixture; return its outputs."""

        def array(value, dtype=float):
            return wp.array(value, dtype=dtype, device=device)

        def full(shape, value, dtype=float):
            return wp.full(shape, value, dtype=dtype, device=device)

        f = _fixture(device)
        plan = f["plan"]
        # A separately prescribed articulation on contact zero, as in the builder's preparation test.
        f["shape_body"] = array([1, 2, 3, -1, 0], int)
        f["shape1"] = array([4, 1, 3], int)
        f["art_b"] = array([1, 0, -1], int)
        f["group"] = array([0, 0], int)
        f["start"] = array([0, 6], int)
        f["origin"] = array([[0.3, -0.4, 0.7], [-0.2, 0.1, 0.4]], wp.vec3)
        f["patches"].phi = array([[0.01, -0.02], [-0.03, 0.04], [0.0, 0.0]], wp.vec2)
        f["patches"].enabled = patch
        response_dofs = array([6, 0], int)
        geometry = [f[name] for name in ("point0", "point1", "normal", "shape0", "shape1", "thickness0", "thickness1")]
        rows = SparseContactRows()
        rows.body_velocity = array(np.zeros((4, 6), dtype=np.float32), wp.spatial_vector)
        rows.prescribed_articulation = array([0, 1], int)
        rows.material_friction = array([0.8, 0.5, 0.3, 0.6, 0.4])
        rows.material_restitution = array([0.4, 0.2, 0.5, 0.8, 0.6])
        rows.beta, rows.cfm = 0.05, 0.03
        rows.friction_scale, rows.friction_beta = 0.8, 0.23
        rows.speculative_scale, rows.restitution_velocity_threshold = 0.65, 0.1
        for name in ("type", "parent"):
            setattr(rows, name, full((1, 9), -9, int))
        for name in ("mu", "beta_row", "cfm_row", "phi", "target_velocity", "restitution", "rhs"):
            setattr(rows, name, full((1, 9), -9.0))
        support_plan = _SparseRowSupportPlan.build(plan, plan.joint_ancestor_mask, max_row_dofs=6)
        support = support_plan.to_device(1, 9, device)
        dense = full((1, 9, 6), 0.0)
        wp.launch(
            populate_world_J_for_compact_size,
            dim=(1, 32),
            inputs=[
                f["count"],
                1,
                *geometry,
                f["slot"],
                f["art_a"],
                f["art_b"],
                f["path"],
                f["needed"],
                6,
                response_dofs,
                f["group"],
                f["start"],
                f["origin"],
                f["physical_mask"],
                f["motion"],
                f["shape_body"],
                f["body_q"],
                friction_shared,
                f["patches"],
                shared,
            ],
            outputs=[dense],
            device=device,
        )
        options = {"prepare_rows": True, "use_support_table": True}
        if records:
            options["contact_records"] = True
        record = full((1, 3, CONTACT_RECORD_FLOATS), -7.0)
        outputs = {
            "row_factor": full((1, 9, 6), -9.0),
            "incident": full((1, 9), -9.0),
            "diagonal": full((1, 9), -9.0),
        }
        wp.launch(
            _get_sparse_contact_response_kernel(6, **options),
            dim=8,
            inputs=[
                f["count"],
                1,
                *geometry,
                f["world"],
                f["slot"],
                f["art_a"],
                f["art_b"],
                f["path"],
                f["needed"],
                f["group"],
                response_dofs,
                full(response_dofs.shape, 0, int),
                full(response_dofs.shape, 0, int),
                f["start"],
                f["origin"],
                f["sparse_mask"],
                f["motion"],
                f["shape_body"],
                f["body_q"],
                friction_shared,
                f["patches"],
                shared,
                f["indices"].permutation,
                f["indices"].row_offsets,
                f["packed"],
                array(np.eye(6)[None]),
                f["velocity"],
                rows,
                0.01,
            ],
            outputs=[
                full((1, 1, 1), -9, int),
                outputs["row_factor"],
                record if records else full((1, 9, 12), -9.0),
                outputs["incident"],
                outputs["diagonal"],
                support,
            ],
            block_dim=128,
            device=device,
        )
        # The independent oracle: dense contact rows through the inverse factor, with the rows' CFM.
        expected = dense.numpy()[0, :7][:, plan.permutation] @ f["inverse"].T
        results = {name: value.numpy() for name, value in outputs.items()}
        results["rhs"] = rows.rhs.numpy()
        return results, record.numpy(), expected, float(rows.cfm)

    def test_records_hold_each_triplets_block_cpu(self):
        """Write W_nt1, W_nt2, W_t1t2, 1/W_nn and the pair law's tangent eigenbasis for every full triplet.

        Contacts at rows 0 and 3 carry three rows, records 0 and 1; the one-row contact at
        row 6 writes nothing. The rows the builder writes are those of the plain builder.
        """
        for shared, friction_shared, patch in ((0, 0, 0), (0, 1, 0), (1, 0, 0), (0, 0, 1), (1, 1, 1)):
            with self.subTest(shared=shared, friction_shared=friction_shared, patch=patch):
                results, record, expected, cfm = self._build("cpu", True, shared, friction_shared, patch)
                plain, _, _, _ = self._build("cpu", False, shared, friction_shared, patch)
                for name, value in results.items():
                    np.testing.assert_array_equal(value, plain[name], err_msg=name)
                for triplet, row in ((0, 0), (1, 3)):
                    block = expected[row : row + 3] @ expected[row : row + 3].T + cfm * np.eye(3)
                    np.testing.assert_allclose(
                        record[0, triplet, :4],
                        (block[0, 1], block[0, 2], block[1, 2], 1.0 / block[0, 0]),
                        rtol=2.0e-5,
                        atol=2.0e-6,
                    )
                    # The eigenbasis relative to the larger tangent diagonal: largest along (vx, vy).
                    tangents = block[1:, 1:] / max(block[1, 1], block[2, 2])
                    largest, smallest, vx, vy = record[0, triplet, 4:8]
                    np.testing.assert_allclose(
                        (largest, smallest, vx, vy),
                        triplets.pair_metric(block[1, 1], block[1, 2], block[2, 2]),
                        rtol=2.0e-5,
                        atol=2.0e-6,
                    )
                    np.testing.assert_allclose(sorted((smallest, largest)), np.linalg.eigvalsh(tangents), atol=2e-6)
                    np.testing.assert_allclose(tangents @ (vx, vy), largest * np.array((vx, vy)), atol=2e-6)
                    self.assertGreater(abs(block[0, 1]) + abs(block[0, 2]), 1e-3)
                np.testing.assert_array_equal(record[0, 2], -7.0)

    def test_records_require_prepared_support_rows_cpu(self):
        """Build records only beside prepared rows with the support table, whose routes leave the slot free."""
        for options in ({"use_support_table": True}, {"prepare_rows": True}, {}):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "records"):
                _get_sparse_contact_response_kernel(6, contact_records=True, **options)
        kernel = _get_sparse_contact_response_kernel(6, prepare_rows=True, use_support_table=True, contact_records=True)
        self.assertTrue(kernel.key.endswith("_prepared_support_table_contact_records"))


if __name__ == "__main__":
    unittest.main()
