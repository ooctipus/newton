# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The sparse factor rows' contact record layout and the finish of their internal-constraint prefix."""

import warp as wp

from .dense_rows import DenseRows
from .row_laws import row_rhs

CONTACT_RECORD_FLOATS = 8
"""Floats of a contact triplet's block record: ``W_nt1, W_nt2, W_t1t2, 1/W_nn``, then the pair
law's tangent eigenbasis ``largest, smallest, vx, vy`` relative to the larger tangent diagonal."""
CONTACT_RING_FLOATS = 4
"""Floats a linked sparse sweep appends to each record: the anchor's other ring members, as int bits."""


@wp.kernel(enable_backward=False)
def finalize_sparse_joint_limit_rows(
    phase_bounds: wp.array2d[int],
    constraint_count: wp.array[int],
    rows: DenseRows,
    dt: float,
):
    """Finish the internal-constraint prefix with each row's law; contacts are already complete."""
    world = wp.tid()
    line = rows.first[world]
    for row in range(wp.min(phase_bounds[world, 1], constraint_count[world])):
        rhs, _row_w = row_rhs(
            rows.row_type[line, row],
            rows.phi[line, row],
            rows.row_beta[line, row],
            rows.target[line, row],
            1.0 / dt,
            1.0,
            0.0,
            1.0,
            1.0,
        )
        rows.diag[line, row] += rows.row_cfm[line, row]
        rows.rhs[line, row] = rhs
