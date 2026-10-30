# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Read a FeatherPGS solver's free-body rows back in physical coordinates.

The rows' build whitens each side, ``Z = L^-1 P J^T`` with the body's held leaf-first factor ``L`` (``L_by_size[6]``
at its articulation's group index, ``P`` the coordinates reversed; factor coordinate ``i`` of ``Z`` at position
``5 - i``), so a test that checks a row against physical velocities recovers ``J = P^T L Z``. A speed row keeps its
unwhitened axis. The module defines no tests.
"""

import numpy as np

from newton._src.solvers.feather_pgs.kernels import PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT


def free_body_jacobians(solver, Z_a=None, Z_b=None):
    """The physical Jacobians ``(J_a, J_b)`` of every world's free-body rows, ``[world, row, 6]`` (float64).

    ``Z_a``, ``Z_b`` default to the solver's whitened rows; pass a snapshot of them (and the factors must be the
    ones that whitened it, the step's).
    """
    Z_a = solver.mf_Z_a.numpy() if Z_a is None else np.asarray(Z_a)
    Z_b = solver.mf_Z_b.numpy() if Z_b is None else np.asarray(Z_b)
    factors = solver.L_by_size[6].numpy().astype(np.float64)
    articulation = solver.forest.body_to_articulation.numpy()
    group = solver.forest.art_group_idx.numpy()
    row_type = solver.mf_row_type.numpy()
    jacobians = []
    for Z, bodies in ((Z_a, solver.mf_body_a.numpy()), (Z_b, solver.mf_body_b.numpy())):
        J = Z.astype(np.float64).copy()
        worlds, rows = np.nonzero(
            (bodies[:, : Z.shape[1]] >= 0) & (row_type[:, : Z.shape[1]] != PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT)
        )
        for world, row in zip(worlds, rows, strict=True):
            J[world, row] = (factors[group[articulation[bodies[world, row]]]] @ J[world, row][::-1])[::-1]
        jacobians.append(J)
    return tuple(jacobians)
