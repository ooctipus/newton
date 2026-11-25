# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The bilateral closure: the exact Schur transform of the packed rows (:func:`close_world`,
:func:`seed_blocks_kernel`), which the one writer's block phase applies to fold a world's bilateral blocks into the
responses of its other rows."""

from __future__ import annotations

from functools import cache

import warp as wp

from ....core.memory import Buffer, Place, Staging
from ..device_preludes import SHARED_STORAGE, dual_native, lane_group, substitute
from ..projection_laws import BILATERAL_PIVOT
from ..row_laws import LINK_ROWS
from ..row_records import UNIT_LINK, UNIT_MIMIC, UNIT_SHIFT
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


def closure_lanes(entries: int) -> int:
    """The lanes a world's closure holds a block's rows over: a lane an entry of the widest block's span (``entries``,
    :attr:`.tree_factor.TreeFactor.block_entries`), the smallest power of two at or above them, at most a warp
    (:func:`close_world`)."""
    return 1 << (max(min(int(entries), 32), 1) - 1).bit_length()


CLOSURE_ROWS = "closure_rows"
"""The closure's staged rows: a block's rows over its span, a lane group's (:func:`closure_staging`)."""


def closure_staging(capacity: int, entries: int, block: int) -> Staging:
    """The closure's working set in a block of ``block`` threads: each lane group's staged rows, ``capacity`` rows of
    the widest span's ``entries`` (:data:`CLOSURE_ROWS`), a group for each the block holds at :func:`closure_lanes`.
    Where the writer's stage declines them (:meth:`.walk_rows.WriterShape.stage`) they live at their alternative place,
    a line of global memory a world (:attr:`.tree_factor.TreeFactor.closure_lines`), read and written as the stage's."""
    groups = max(int(block) // closure_lanes(entries), 1)
    rows = Buffer(CLOSURE_ROWS, "float", int(capacity) * int(entries), alternative=Place.GLOBAL)
    return Staging((rows,), groups=groups, **({"layout": "arrays"} if groups > 1 else {}))


# A world's blocks, one after another, by the world's lanes. A block's rows are staged once over its span (the world
# coordinates of its trees, which hold every entry of its rows), each lane holding the entries it strides over (the
# stage, closure_staging, or the world's line where the writer's stage declines it), closed there by modified Gram-Schmidt, each
# dot product a lane sum, and written back once; a lane a row then corrects every other row whose sides touch the
# block against the staged kept rows over the span, its own sums in entry order (its norm over every entry).
_CLOSE = """    const int M = $capacity, W = $entries, S = Z.shape[2];
$stage
    float* const stage = $rows;
    float* const z = Z.data + (size_t)line * (Z.strides[0] / (int)sizeof(*Z.data));
    float* const r = rhs.data + (size_t)line * (rhs.strides[0] / (int)sizeof(*rhs.data));
    float* const v = incident.data + (size_t)line * (incident.strides[0] / (int)sizeof(*incident.data));
    float* const d = diag.data + (size_t)line * (diag.strides[0] / (int)sizeof(*diag.data));
    const int* const source = units.data + (size_t)line * (units.strides[0] / (int)sizeof(*units.data));
    const auto* const side = sides.data + (size_t)line * (sides.strides[0] / (int)sizeof(*sides.data));
    for (int pe = preelim.world_start.data[world]; pe < preelim.world_start.data[world + 1]; ++pe) {
        // The block's reserved rows in its constraints' order: a mimic's, a link's six.
        int row[M], m = 0;
        for (int i = preelim.start.data[pe]; i < preelim.start.data[pe + 1]; ++i) {
            const int entry = preelim.entry.data[i];
            if (entry >= 0) {
                if (mimic_slot.data[entry] >= 0) row[m++] = mimic_slot.data[entry];
            } else if (link_slot.data[-entry - 1] >= 0) {
                for (int a = 0; a < $link_rows; ++a) row[m++] = link_slot.data[-entry - 1] + a;
            }
        }
        const int first = preelim.span.data[pe][0], n = preelim.span.data[pe][1] - first;
        for (int p = 0; p < m; ++p)
            for (int e = LANE; e < n; e += STRIDE) stage[p * W + e] = z[(size_t)row[p] * S + first + e];
        // Each row takes out the rows kept before it, then joins them unit, or leaves the block inert where nothing of
        // it remains (BILATERAL_PIVOT of its own norm); its right-hand side follows, its incident velocity joined to
        // it first, and its diagonal is |Q_i|^2 + cfm. Every lane holds the same sums, so the same rows.
        int keep[M], kept = 0;
        float q[M];
        for (int p = 0; p < m; ++p) {
            float* const zp = stage + p * W;
            float rp = r[row[p]] + v[row[p]];
            float own = 0.0f;
            for (int e = LANE; e < n; e += STRIDE) own += zp[e] * zp[e];
            SUM(own);
            for (int k = 0; k < kept; ++k) {
                const float* const zk = stage + keep[k] * W;
                float along = 0.0f;
                for (int e = LANE; e < n; e += STRIDE) along += zk[e] * zp[e];
                SUM(along);
                for (int e = LANE; e < n; e += STRIDE) zp[e] -= along * zk[e];
                rp -= along * q[keep[k]];
            }
            float rest = 0.0f;
            for (int e = LANE; e < n; e += STRIDE) rest += zp[e] * zp[e];
            SUM(rest);
            float scale = 0.0f;
            if (rest > $pivot * own) {
                scale = 1.0f / wp::sqrt(rest);
                keep[kept++] = p;
            }
            for (int e = LANE; e < n; e += STRIDE) zp[e] *= scale;
            q[p] = rp * scale;
            if (LANE == 0) {
                r[row[p]] = q[p];
                v[row[p]] = 0.0f;
                d[row[p]] = scale * scale * rest + cfm;
            }
        }
        for (int p = 0; p < m; ++p)
            for (int e = LANE; e < n; e += STRIDE) z[(size_t)row[p] * S + first + e] = stage[p * W + e];
        if (LANE == 0) {
            for (int k = 0; k < M; ++k) preelim.slots.data[pe * M + k] = k < kept ? row[keep[k]] : -1;
            preelim.nB.data[pe] = kept;
        }
        SYNC();
        // Every other row whose sides touch the block: Pi Z_i = Z_i - Q Q^T Z_i, its right-hand side taking back
        // (Q^T Z_i) . r_Q, its diagonal |Pi Z_i|^2 + cfm (the squares summed in entry order, the padding's zeros
        // included). Q is zero past the block's span, so the products there are zeros and the span holds the rest. A
        // row the block does not see (Q^T Z_i exactly zero) keeps its entries. Blocks hold different trees, so their
        // projectors commute: a row touching two takes them in the world's block order.
        for (int i = LANE; i < count; i += STRIDE) {
            const int kind = source[i] >> $shift;
            if (kind == $link || ($mimics && kind == $mimic)) continue;
            const int a = side[i][0], b = side[i][1];
            if ((a < 0 || preelim.art_to_preelim.data[a] != pe) && (b < 0 || preelim.art_to_preelim.data[b] != pe))
                continue;
            float* const zi = z + (size_t)i * S;
            float* const span = zi + first;
            float ri = r[i];
            for (int k = 0; k < kept; ++k) {
                const float* const zk = stage + keep[k] * W;
                float along = 0.0f;
                for (int e = 0; e < n; ++e) along += zk[e] * span[e];
                if (along != 0.0f) {
                    for (int e = 0; e < n; ++e) span[e] -= along * zk[e];
                    ri -= along * q[keep[k]];
                }
            }
            r[i] = ri;
            float norm = 0.0f;
            for (int e = 0; e < S; ++e) norm += zi[e] * zi[e];
            d[i] = norm + cfm;
        }
        // The stage is the next block's.
        SYNC();
    }
"""


@cache
def close_world(
    capacity: int, entries: int, mimic_blocks: bool, block: int, place: Place = Place.SHARED
) -> wp.Function:
    """A world's closure, the one writer's block phase once every row of the world is written (``lanes`` lanes, at
    least :func:`closure_lanes` of ``entries``, of a block of ``block`` threads): each of the world's blocks in turn,
    modified Gram-Schmidt over its rows (at most ``capacity``; the links', and the mimics' with ``mimic_blocks``) in its
    constraints' order on line ``line``, in place on the packed rows, then every other of the world's ``count`` rows
    corrected on it. ``preelim.slots`` ends with each block's kept rows (``preelim.nB`` of them).

    The block's rows are staged once over its span (at most ``entries`` coordinates; the world's lanes over them, each
    dot product a lane sum) and written back once; the corrections read the kept rows from the stage, a lane a row.
    The stage (:func:`closure_staging`) is at ``place``: the block's shared memory, a stage for each lane group it
    holds (on the CPU, where one thread runs the world, its stack), or where the writer's stage declines it the world's
    line of global memory in ``lines``, with the same arithmetic in the same order."""
    staging = closure_staging(capacity, entries, block)
    if place is Place.SHARED:
        stage, preludes = staging.declarations(), (lane_group("lanes"), SHARED_STORAGE)
    else:
        stage = f"    float* const {CLOSURE_ROWS} = lines.data + (size_t)world * lines.shape[1];"
        preludes = (lane_group("lanes"),)
    source = substitute(
        _CLOSE,
        capacity=int(capacity),
        entries=int(entries),
        stage=stage,
        rows=CLOSURE_ROWS,
        link_rows=int(LINK_ROWS),
        shift=int(UNIT_SHIFT),
        link=int(UNIT_LINK),
        mimic=int(UNIT_MIMIC),
        mimics=int(bool(mimic_blocks)),
        pivot=f"{float(BILATERAL_PIVOT)!r}f",
    )

    def close(
        world: int,
        line: int,
        count: int,
        lanes: int,
        preelim: Preelim,
        mimic_slot: wp.array[int],
        link_slot: wp.array[int],
        units: wp.array2d[int],
        sides: wp.array2d[wp.vec2i],
        cfm: float,
        Z: wp.array3d[float],
        rhs: wp.array2d[float],
        incident: wp.array2d[float],
        diag: wp.array2d[float],
        lines: wp.array2d[float],
    ): ...

    name = f"close_world_{capacity}_{entries}_{staging.groups}" + ("_mimics" * mimic_blocks)
    close.__name__ = close.__qualname__ = name + ("" if place is Place.SHARED else "_global")
    return wp.func_native(dual_native(source, preludes=preludes))(close)


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
