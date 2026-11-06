# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import field_group, row_records
from newton._src.solvers.feather_pgs.row_kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
)
from newton.tests.feather_pgs_law_harness import bias_rows, row_bias, row_lines


def _array(value, dtype=float):
    return wp.array(value, dtype=dtype, device="cpu")


def _full(shape, value, dtype=float):
    return wp.full(shape, value, dtype=dtype, device="cpu")


class TestSparseContacts(unittest.TestCase):
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
                bias_rows,
                dim=(1, row_count),
                inputs=[
                    field_group.upload(
                        row_records.DenseRows,
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
                    row_bias(0.01, 1.0, 1.0, 0.1, 0, 0),
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
