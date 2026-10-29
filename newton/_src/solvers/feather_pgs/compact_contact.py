# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental complete contact owner for the existing dense-six + sparse solver.

Allocation, contact admission, scheduling, RHS and the solve remain authoritative
in the original pipeline. This module only changes the representation producer.
Persistent patch friction writes its tangent rows with the patch row law. The
patch link, ``_FrictionPatchState.link``, runs after this producer and alone
links the patch rings and divides the friction coefficient, in one
``link_patch_routes`` launch over every route.
"""

import warp as wp

from . import field_group
from .contact_input import ContactInput
from .friction_patches import PatchRowLaw, patch_row_law
from .response_builds_rows import CompactBoundary, sparse_pair_row_response
from .row_build import write_contact_row


def supported(solver) -> bool:
    """Require the inherited complete sparse contract and its dense-six triple or unit mode.

    The sparse-diagonal owner exists only with its fused wide-articulation limits, whose admission already requires
    augmented drives, no velocity-limit, mimic or connect rows, no debug rows and no pre-elimination: only the dense
    six position axes keep rows, at most lower + upper for each axis.
    """
    return bool(
        solver.model.device.is_cuda
        and solver._sparse_diagonal_contact_solve
        and solver._sparse_diagonal_dense_size == 6
        and solver._sparse_diagonal_response_size > 6
        and (solver._sparse_diagonal_contact_triples or solver._sparse_diagonal_contact_units)
        and solver.dense_max_constraints >= 12
        and 0 <= solver._dense_internal_max_rows <= 12
        and solver._compact_diagonal_mass_size == solver._sparse_diagonal_response_size
    )


@wp.kernel
def clear_limit_prefix(dense_groups: wp.array[int], J: wp.array3d[float], sparse_dof: wp.array3d[int]):
    """Clear every possible dense-six limit row before partial limit writers run."""
    world, row = wp.tid()
    for dof in range(6):
        J[dense_groups[world], row, dof] = 0.0
    sparse_dof[world, row, 0] = -1
    sparse_dof[world, row, 1] = -1


@wp.kernel
def clear_contact_response(
    counts: wp.array[int],
    bounds: wp.array2d[int],
    groups: wp.array[int],
    Z: wp.array3d[float],
):
    """Clear only current contact coefficients, with contiguous lanes within each world."""
    world, lane = wp.tid()
    start = wp.max(bounds[world, 1], 0)
    end = wp.min(counts[world], Z.shape[1])
    if end <= start:
        return
    group = groups[world]
    # The limit prefix already contains the producers' current J. Align the first coefficient tile down for
    # coalescing, but never write its prefix lanes.
    first = (start * 6 // 32) * 32
    for coefficient in range(first + lane, end * 6, 256):
        if coefficient >= start * 6:
            row = coefficient // 6
            dof = coefficient % 6
            Z[group, row, dof] = 0.0


@wp.kernel
def produce_contacts(data: ContactInput, boundary: CompactBoundary, law: PatchRowLaw):
    """Publish each contact's rows in their original collision order, one thread per row (3 per contact): its records
    (:func:`.row_build.write_contact_row`), then its whitened dense-six part and sparse slots.

    Tangent rows act at the friction location and follow the patch row law (``patch_tangent_row``): under
    persistent patches at the patch friction location with the anchor's tangential phi, ``law.beta`` and the
    undivided coefficient; under point friction (``law.patches.enabled`` 0, beta 0, a zero phi view) at the
    contact's friction points.
    """
    contact, row = wp.tid()
    if contact >= data.count[0]:
        return
    slot = data.slot[contact]
    if data.path[contact] != 0 or slot < 0 or row >= data.slots_needed[contact]:
        return
    world = data.world[contact]
    art0 = data.art0[contact]
    art1 = data.art1[contact]
    body0, body1, direction, p0, p1 = write_contact_row(data, law, contact, row, False)
    has_dense = False
    if art0 >= 0 and body0 >= 0:
        has_dense = data.response_dofs[art0] == 6
    if art1 >= 0 and body1 >= 0:
        has_dense = has_dense or data.response_dofs[art1] == 6
    group = boundary.dense_group[world]
    sparse_pair_row_response(
        data, boundary, world, slot + row, group, has_dense, art0, body0, art1, body1, direction, p0, p1
    )


def launch_contacts(solver, data, workers):
    """Produce the contact rows from the step's bound contact input; Warp capture retains these launch descriptors."""
    boundary = field_group.bind(CompactBoundary, solver=solver, factor=solver.L_by_size[6], Z=solver.J_by_size[6])
    # Allocation has completed, but constraint_count is finalized later. Keep
    # this clear and the following producer ordered on the current stream.
    wp.launch(
        clear_contact_response,
        dim=(solver.world_count, 256),
        inputs=[solver.slot_counter, solver.dense_phase_bounds, boundary.dense_group, boundary.Z],
        block_dim=256,
        device=solver.model.device,
    )
    # Persistent patches write their tangent rows with the patch row law. The
    # patch link (_FrictionPatchState.link, one link_patch_routes launch) still
    # runs after this producer: it alone links normal rows into patch rings and
    # divides the tangent coefficient.
    # A thread per (contact, row); in a capture whose launches bind, each replay runs the live count. The rows
    # take their own axis: the contact axis is captured at the pool's bound, which a scalar thread index times 3
    # would overflow.
    pool = solver._contact_pool
    count = data.count
    pool.launch(produce_contacts, (pool.launch_rows(), 3), inputs=[data, boundary, patch_row_law(solver)], extent=count)
    # These original kernels encode schedule links into the freshly published
    # sparse DOF slots. Never overwrite those fields after this point.
    pool.launch(
        solver._mark_independent_sparse_contact_candidates_kernel,
        workers,
        inputs=[data, workers],
        outputs=[solver._sparse_diagonal_row_dof],
        extent=count,
        clamp=True,
    )
