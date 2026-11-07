# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The bilateral closure: the exact Schur transform of the packed rows (:func:`close_block`, :func:`close_row`,
:func:`seed_blocks_kernel`), which the one writer's block phase applies to fold a world's bilateral block into the
responses of its other rows."""

from __future__ import annotations

import warp as wp

from ..projection_laws import BILATERAL_PIVOT
from ..row_laws import LINK_ROWS
from .fields import Preelim

# The bilateral rows of a linked component of trees (:mod:`.body_cut`), its mimics and its links, form a block ``B``:
# the block primitive's second phase, the closure, after the slips' release (FL3_design §1.4), in the one writer's
# block phase (:mod:`.walk_rows`). On the packed rows (:mod:`.packed_rows`, ``Z_i = L^-1 P J_i^T`` on each side), wide
# over the world's coordinates, the transform makes the block's rows orthonormal in place by modified Gram-Schmidt, the right-hand sides following:
# ``Q^T Q = I``, a linear combination of the same constraints. A row whose remainder falls to
# :data:`.projection_laws.BILATERAL_PIVOT` of its own norm depends on the rows before it (a planar loop's out-of-plane
# axis, a mimic a closure fixes): it leaves the block, inert. With ``Pi = I - Q Q^T`` the exact projector onto the
# closures:
#
# - correct: every other row touching the block becomes ``Pi Z_i = Z_i - Q Q^T Z_i``, which reaches every tree of the
#   block, its diagonal ``|Pi Z_i|^2 + cfm`` (the closed-loop effective mass); its right-hand side takes back
#   ``(Q^T Z_i)^T r_Q``, ``r_Q = J_Q v_hat + b_Q`` being the block rows' residual at zero change;
# - project: the velocity change starts at ``du = -Q r_Q``, the block rows' impulses at ``-r_Q`` (the sweeps' seed,
#   :func:`.sweep.impulse_seed`).
#
# Then ``Q^T (u_hat + du) = -b_Q`` and a sweep's impulse, ``du += Pi Z_i delta``, keeps it. A corrected row's residual
# at zero change is ``(Pi Z_i)^T u_hat`` and its right-hand side, which the transform folds: its ``J_i v_hat + b_i``
# less ``(Q^T Z_i)^T r_Q``. ``Pi`` is idempotent, so the
# contacts solve on the closed-loop Delassus ``Z^T Pi Z`` exactly (no compliance). The block's rows stay in the sweep:
# they hold in exact arithmetic, and their visits absorb the corrected rows' float32 leakage, without which collinear
# contact rows of a jammed loop drift in their null space and never meet the exit law. Being orthonormal, each absorbs
# its share in one visit, where rows that couple would pass it among themselves at their Gauss-Seidel rate.


@wp.func
def wide_product(Z: wp.array3d[float], world: int, row_a: int, row_b: int, entries: int) -> float:
    """``Z_a . Z_b`` over a world's ``entries`` wide entries, in order."""
    acc = float(0.0)
    for e in range(entries):
        acc += Z[world, row_a, e] * Z[world, row_b, e]
    return acc


@wp.func
def subtract_row(Z: wp.array3d[float], world: int, row: int, scale: float, other: int, entries: int):
    """``Z_row -= scale Z_other`` over a world's ``entries`` wide entries."""
    for e in range(entries):
        Z[world, row, e] -= scale * Z[world, other, e]


@wp.func
def scale_row(Z: wp.array3d[float], world: int, row: int, scale: float, entries: int):
    """``Z_row *= scale`` over a world's ``entries`` wide entries."""
    for e in range(entries):
        Z[world, row, e] *= scale


@wp.func
def bilateral_rows(
    block: int,
    base: int,
    preelim: Preelim,
    mimic_slot: wp.array[int],
    link_slot: wp.array[int],
) -> int:
    """Select the block's reserved rows, in its constraints' order, into ``preelim.slots[base:]`` and return their
    count."""
    n = int(0)
    for i in range(preelim.start[block], preelim.start[block + 1]):
        entry = preelim.entry[i]
        if entry >= 0:
            if mimic_slot[entry] >= 0:
                preelim.slots[base + n] = mimic_slot[entry]
                n += 1
        elif link_slot[-entry - 1] >= 0:
            for a in range(LINK_ROWS):
                preelim.slots[base + n] = link_slot[-entry - 1] + a
                n += 1
    return n


@wp.func
def close_block(
    pe: int,
    capacity: int,
    line: int,
    preelim: Preelim,
    mimic_slot: wp.array[int],
    link_slot: wp.array[int],
    cfm: float,
    packed_Z: wp.array3d[float],
    world_rhs: wp.array2d[float],
    world_incident: wp.array2d[float],
    world_diag: wp.array2d[float],
):
    """Block ``pe``'s closure (the one writer's block phase, once its rows are written): modified Gram-Schmidt over the
    block's rows, in its constraints' order, in place on line ``line``. Each row takes out the rows kept before it, then
    joins them unit, or leaves the block inert where nothing of it remains; its right-hand side follows, its incident
    velocity joined to it first, and its diagonal is ``|Q_i|^2 + cfm``. ``preelim.slots`` ends with the kept rows
    (``preelim.nB`` of them)."""
    base = pe * capacity
    entries = packed_Z.shape[2]
    m = bilateral_rows(pe, base, preelim, mimic_slot, link_slot)
    kept = int(0)
    for p in range(m):
        row = preelim.slots[base + p]
        # The row's incident velocity joins its right-hand side: both follow the row through the transform.
        world_rhs[line, row] += world_incident[line, row]
        world_incident[line, row] = 0.0
        own = wide_product(packed_Z, line, row, row, entries)
        for q in range(kept):
            other = preelim.slots[base + q]
            along = wide_product(packed_Z, line, other, row, entries)
            subtract_row(packed_Z, line, row, along, other, entries)
            world_rhs[line, row] -= along * world_rhs[line, other]
        rest = wide_product(packed_Z, line, row, row, entries)
        scale = float(0.0)
        if rest > BILATERAL_PIVOT * own:
            scale = 1.0 / wp.sqrt(rest)
            preelim.slots[base + kept] = row
            kept += 1
        scale_row(packed_Z, line, row, scale, entries)
        world_rhs[line, row] *= scale
        world_diag[line, row] = scale * scale * rest + cfm
    for p in range(kept, capacity):
        preelim.slots[base + p] = -1
    preelim.nB[pe] = kept


@wp.func
def close_row(
    line: int,
    row: int,
    sides: wp.vec2i,
    capacity: int,
    preelim: Preelim,
    cfm: float,
    packed_Z: wp.array3d[float],
    world_rhs: wp.array2d[float],
    world_diag: wp.array2d[float],
):
    """Correct row ``row`` (not a block row) on the blocks its sides touch, once they are closed: ``Pi Z_i = Z_i - Q
    Q^T Z_i``, its right-hand side taking back ``(Q^T Z_i) . r_Q``, its diagonal ``|Pi Z_i|^2 + cfm`` (the squares summed
    in entry order, the padding's zeros included). A row the block does not see (``Q^T Z_i`` exactly zero) keeps its
    entries. Blocks hold different trees, so their projectors commute."""
    entries = packed_Z.shape[2]
    touched = int(0)
    last = int(-1)
    for side in range(2):
        art = sides[side]
        if art < 0:
            continue
        pe = preelim.art_to_preelim[art]
        if pe < 0 or pe == last:
            continue
        last = pe
        touched = 1
        base = pe * capacity
        for p in range(preelim.nB[pe]):
            other = preelim.slots[base + p]
            along = wide_product(packed_Z, line, other, row, entries)
            if along != 0.0:
                subtract_row(packed_Z, line, row, along, other, entries)
                world_rhs[line, row] -= along * world_rhs[line, other]
    if touched != 0:
        world_diag[line, row] = wide_product(packed_Z, line, row, row, entries) + cfm


@wp.kernel(enable_backward=False)
def seed_blocks_kernel(
    dense_first: wp.array[int],
    capacity: int,
    preelim: Preelim,
    world_rhs: wp.array2d[float],
    # outputs
    world_impulses: wp.array2d[float],
):
    """The closure's seed, after the warm start's gather: the kept rows' impulses ``-r_Q``, their residual at zero
    change (the sweeps' :func:`.sweep.impulse_seed` installs them), a thread per block."""
    pe = wp.tid()
    base = pe * capacity
    line = dense_first[preelim.world[pe]]
    for p in range(preelim.nB[pe]):
        slot = preelim.slots[base + p]
        world_impulses[line, slot] = -world_rhs[line, slot]
