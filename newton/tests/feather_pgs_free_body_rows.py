# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Read a FeatherPGS solver's free-body rows back in physical coordinates, and sweep given free-body rows.

The rows' build whitens each side, ``Z = L^-1 P J^T`` with the body's held leaf-first factor ``L`` (``L_by_size[6]``
at its articulation's group index, ``P`` the coordinates reversed; factor coordinate ``i`` of ``Z`` at position
``5 - i``), so a test that checks a row against physical velocities recovers ``J = P^T L Z``. Where the placement
walks (:mod:`.walk_rows`) a side is the walk of the body's seed, ``z_c = (J_c - sum_{k > c} (s_c . ut_k) z_k) /
ell_c`` over its six coordinates from the last, so ``J_c = ell_c z_c + sum_{k > c} (s_c . ut_k) z_k`` from the walk's
data. A speed row keeps its unwhitened axis. :func:`sweep_free_body_rows` runs rows a test writes through the general
sweep. The module defines no tests.
"""

import inspect

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import sweep_routes
from newton._src.solvers.feather_pgs.kernels import PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT, BodyFactors
from newton._src.solvers.feather_pgs.tree_factor import unpack_walk_records


def free_body_jacobians(solver, Z_a=None, Z_b=None):
    """The physical Jacobians ``(J_a, J_b)`` of every world's free-body rows, ``[world, row, 6]`` (float64).

    ``Z_a``, ``Z_b`` default to the solver's whitened rows; pass a snapshot of them (and the factors must be the
    ones that whitened it, the step's).
    """
    Z_a = solver.placement.mf_Z_a.numpy() if Z_a is None else np.asarray(Z_a)
    Z_b = solver.placement.mf_Z_b.numpy() if Z_b is None else np.asarray(Z_b)
    articulation = solver.forest.body_to_articulation.numpy()
    group = solver.forest.art_group_idx.numpy()
    row_type = solver.placement.mf_row_type.numpy()
    if getattr(solver.placement, "walks", False):
        records = solver.factor.walk_view().data.numpy()
        data = unpack_walk_records(records, records.shape[0]).astype(np.float64)
        starts = solver.forest.articulation_dof_start.numpy()

        def physical(art, z):
            d = data[starts[art] : starts[art] + 6]
            return np.array(
                [z[c] / d[c, 12] + sum(d[c, :6] @ d[k, 6:12] * z[k] for k in range(c + 1, 6)) for c in range(6)]
            )

    else:
        factors = solver.factor.L_by_size[6].numpy().astype(np.float64)

        def physical(art, z):
            return (factors[group[art]] @ z[::-1])[::-1]

    jacobians = []
    for Z, bodies in ((Z_a, solver.placement.mf_body_a.numpy()), (Z_b, solver.placement.mf_body_b.numpy())):
        J = Z.astype(np.float64).copy()
        worlds, rows = np.nonzero(
            (bodies[:, : Z.shape[1]] >= 0) & (row_type[:, : Z.shape[1]] != PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT)
        )
        for world, row in zip(worlds, rows, strict=True):
            J[world, row] = physical(articulation[bodies[world, row]], J[world, row])
        jacobians.append(J)
    return tuple(jacobians)


def _pack(dof_a: int, inverse: float, rhs: float, kind: int, parent: int) -> tuple[int, int, int, int]:
    """A free-body row's packed word (:meth:`.FreeBodyZ.pack`): its body's first coordinate (``dof_b`` none), the bits
    of its stored inverse diagonal and right-hand side, its type and its parent (0xFFFF for none)."""
    word = np.uint32(((int(parent) & 0xFFFF) << 16) | int(kind)).view(np.int32)
    as_int = np.array((inverse, rhs), np.float32).view(np.int32)
    return (dof_a << 16) | 0xFFFF, int(as_int[0]), int(as_int[1]), int(word)


def sweep_free_body_rows(
    device, Z, rhs, types, parents, mu, *, impulses=None, weights=None, regularize=False, iterations=300
):
    """Sweep one world's free-body rows on one free body through the general sweep and return their impulses.

    The body has a unit mass factor, so its rows' whitened entries ``Z`` (``[row, 6]``) act on its velocity directly
    and the rows' Delassus block is ``Z Z^T``; each row stores the inverse of its diagonal ``|Z_i|^2`` (zero for an
    empty row). The sweep starts from ``impulses`` (zero by default), seeded with their velocity change ``Z^T
    impulses``, so a row's residual is ``(Z Z^T impulses)_i + rhs_i``. ``weights`` are the rows' regularization weights
    (``regularize``). Contact rows have no parent; a friction row's parent is its normal row.
    """
    Z = np.asarray(Z, np.float32)
    rows, capacity = len(Z), 8
    impulses = np.zeros(rows, np.float32) if impulses is None else np.asarray(impulses, np.float32)
    diag = np.sum(Z.astype(np.float64) ** 2, axis=1)
    inverse = np.where(diag > 0.0, 1.0 / np.where(diag > 0.0, diag, 1.0), 0.0)
    meta = np.zeros((1, capacity * 4), np.int32)
    for row in range(rows):
        meta[0, 4 * row : 4 * row + 4] = _pack(0, inverse[row], rhs[row], types[row], parents[row])
    padded = np.zeros((1, capacity, 6), np.float32)
    padded[0, :rows] = Z

    def per_row(values, dtype=np.float32):
        out = np.zeros((1, capacity), dtype)
        out[0, :rows] = values
        return wp.array(out, device=device)

    factors = BodyFactors()
    factors.dof_factor = wp.array(np.array((0, -1, -1, -1, -1, -1), np.int32), device=device)
    factors.L = wp.array(np.eye(6, dtype=np.float32)[None], device=device)
    factors.first_art = wp.array(np.array((0, -1, -1, -1, -1, -1), np.int32), device=device)
    factors.size = wp.array(np.array((6,), np.int32), device=device)
    factors.world_offset = wp.zeros(1, dtype=wp.int32, device=device)
    factors.address = wp.array(np.array((factors.L.ptr,), np.uint64), dtype=wp.uint64, device=device)
    values = {
        "general_world_count": wp.array((1,), dtype=wp.int32, device=device),
        "general_worlds": wp.zeros(1, dtype=wp.int32, device=device),
        "general_world_grid_stride": 1,
        "use_general_world_queue": 0,
        "world_constraint_count": wp.zeros(1, dtype=wp.int32, device=device),
        "world_dof_indices": wp.array(np.arange(6, dtype=np.int32)[None], device=device),
        "mf_constraint_count": wp.array((rows,), dtype=wp.int32, device=device),
        "mf_contact_rows_end": wp.array((rows,), dtype=wp.int32, device=device),
        "mf_meta": wp.array(meta, device=device),
        "mf_impulses": per_row(impulses),
        "mf_Z_a": wp.array(padded, device=device),
        "mf_Z_b": wp.zeros((1, capacity, 6), dtype=wp.float32, device=device),
        "mf_row_mu": per_row(mu),
        "mf_row_w": per_row(np.ones(rows) if weights is None else weights),
        "factors": factors,
        "iterations": int(iterations),
        "omega": 1.0,
        "regularize": int(regularize),
        "v_out": wp.zeros(6, dtype=wp.float32, device=device),
    }
    kernel = sweep_routes.mf_gs_kernel(
        8,
        capacity,
        6,
        str(wp.get_device(device).arch),
        has_dense_velocity_limit_rows=False,
        shared_metadata=False,
        seeded=True,
    )
    arguments = {arg.label: arg for arg in kernel.adj.args}
    inputs = []
    for name in inspect.signature(kernel.func).parameters:
        if name in values:
            inputs.append(values[name])
        elif wp.types.is_array(arguments[name].type):
            inputs.append(wp.zeros((1,) * arguments[name].type.ndim, dtype=arguments[name].type.dtype, device=device))
        elif wp.types.type_is_struct(arguments[name].type):
            inputs.append(arguments[name].type())
        else:
            inputs.append(arguments[name].type(0))
    wp.launch_tiled(kernel, dim=[1], inputs=inputs, block_dim=32, device=device)
    return values["mf_impulses"].numpy()[0, :rows].astype(np.float64)
