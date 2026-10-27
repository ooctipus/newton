# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contact units, their colors and their body copies: the data a colored or mass-split sweep runs on.

A Gauss-Seidel sweep over a world's free-body rows is a chain: each row reads
the velocities the rows before it updated. Rows that share no body commute, so
the sweep may update them at once. This module groups a world's rows into
*units* (a contact: its normal row and the friction pair that follows it) and
gives every unit a *color*, its level in row order: one past the latest color
of an earlier unit on either of its bodies. No two units of a color share a
body, and units that share one keep their row order, so a sweep that visits
the colors in order and a color's units in parallel is the row-order
Gauss-Seidel sweep itself. The row order is a physics parameter (the iterate a
budget of sweeps reaches depends on it), and the colors keep it.

A body with many contacts (a tray under a coin tower) holds a unit in as many
colors as it has contacts, so the busiest body sets about how many colors a
world takes (measured: the coins' tray, 336 units, 347 colors; the hardware's
bin, 491 units, 495 colors). The *split budget* ``B`` trades that: a body with more than
``B`` units is *split* into ``ceil(units / B)`` copies, its units taken in row
order ``B`` per copy (a patch's units share their copies, so the contacts
whose normal impulses pool into one friction disk never run at once), and
leveled as separate bodies. A color then holds up to one unit per copy: the
``j`` units of a split body in one color run at once, each on its copy with
the body's mass divided by ``j`` (a row's response to it is ``j`` times the
body's: ``MiJt`` and its share of the diagonal scale by ``j``), and after the
color the copies merge, the body taking the mean of their changes, which
solves the copies' equal-velocity constraints (mass splitting, Tonge et al.
2012, merged after every color instead of every sweep). A body that is not
split, or one unit of a split body in its color, takes a plain Gauss-Seidel
step. ``B`` is the design dial: no body split (``B = 0``, or at least the
largest degree) is the row-order sweep, ``B = 1`` splits every body by contact
(every unit in one color: mass splitting), and values between are the hybrid.
With
:data:`AUTO_BUDGET` the data choose it per world: twice the mean degree of the
world's bodies, so only a hub (a body holding more than twice the units of an
average one) is split and every other body keeps one copy. A split body
converges like Jacobi over its copies, which the conformance gate's tray (a
hub under a grid of boxes) shows as a missed normal law, so sweeps do not
split unless asked to.

Everything is computed from the data in row order, one block per world: units
in row order, ranks and copies in row order, levels in row order, units of a
color in row order. Nothing depends on scheduling, so the colors, copies and
sweep are deterministic whenever the rows are.
"""

from __future__ import annotations

import dataclasses
from functools import cache

import warp as wp

from . import kernel_template
from .device_preludes import cuda_native, lane_group
from .field_group import Field, FieldGroup
from .kernel_template import KernelTemplate
from .kernels import PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_FRICTION
from .row_metadata import _mf_parent_expression
from .staging import STATIC_SHARED_BYTES, Buffer, Staging

AUTO_BUDGET = -1
"""The split budget the data choose per world: twice the mean degree of its bodies (``ceil(2 sum / bodies)``)."""

BUILD_WARPS = 8
"""Warps of a world's unit builder block: they share the parallel passes; the first levels the units."""

MAX_COLORS = 512
"""Colors a world's units may take at most: a body copy keeps its taken colors in up to eight 64-bit words."""

UNIT_FIELDS = FieldGroup(
    Field("unit_count", wp.int32, ("world",)),
    # First row and row count of each unit, in row order.
    Field("unit_row", wp.int32, ("world", "unit")),
    Field("unit_rows", wp.int32, ("world", "unit")),
    # Packed first coordinates of the unit's bodies (``dof_a << 16 | dof_b``, as the rows' packed word).
    Field("unit_dofs", wp.int32, ("world", "unit")),
    # Velocity offset of each side's body copy in the sweep's extended velocity (world coordinates, then the
    # extra copies' six coordinates each), or -1.
    Field("unit_offset_a", wp.int32, ("world", "unit")),
    Field("unit_offset_b", wp.int32, ("world", "unit")),
    Field("unit_color", wp.int32, ("world", "unit")),
    # Colors: count, offsets into color_slot (count + 1 of them), and the units of each color in row order, each as
    # the record the sweep reads: first row, packed shape (rows | j_a << 2 | j_b << 17, j a side's units in the
    # color: 1 unless its body is split), and the offsets.
    Field("color_count", wp.int32, ("world",)),
    Field("color_offset", wp.int32, ("world", "color")),
    Field("color_slot", wp.int32, ("world", "unit", 4)),
    # Scratch by row: the copies a patch's first contact took (copy_a | copy_b << 16), which its other contacts take.
    Field("row_copy", wp.int32, ("world", "unit")),
    # Split bodies: count, each one's first velocity coordinate, its copy count and the index of its first extra copy.
    Field("split_count", wp.int32, ("world",)),
    Field("split_dof", wp.int32, ("world", "unit")),
    Field("split_copies", wp.int32, ("world", "unit")),
    Field("split_first", wp.int32, ("world", "unit")),
    # Each split body's units in each color (its copies that took the color): what its merge divides by.
    Field("split_units", wp.int32, ("world", "split", "palette")),
    Field("extra_copies", wp.int32, ("world",)),
    # The split budget the world's units were split by (0: none).
    Field("unit_budget", wp.int32, ("world",)),
    # Bit 0: a unit found no free color at the budget (the last color then holds it and is swept in row order).
    # Bit 1: a body over the budget had no room for its copies and stayed whole.
    Field("color_overflow", wp.int32, ("world",)),
)
"""Per-world units, colors and body copies (``unit`` capacity: the world's free-body row capacity)."""

SLOT_ENTRIES = 72
"""Entries a color slot stages per unit: for each of its (up to three) rows ``r``, the six ``J_a``, ``J_b``,
``MiJt_a`` and ``MiJt_b`` entries, at ``r * 24 + array * 6 + k`` (``array`` in that order)."""

SLOT_MU, SLOT_WEIGHT = SLOT_ENTRIES, SLOT_ENTRIES + 1
"""The slot's scalars after its entries: the friction coefficient of its first tangent and its normal row's
regularization weight."""

SLOT_WORDS = 9
"""Words a color slot stages per unit: for each row ``r``, at ``r * 3``, the bits of its stored inverse diagonal,
of its right-hand side, and its type and parent word (``mf_meta``'s ``y``, ``z``, ``w``)."""

_SLOT_MU = wp.constant(SLOT_MU)
_SLOT_WEIGHT = wp.constant(SLOT_WEIGHT)

SLOT_FIELDS = FieldGroup(
    # Slot-major: field f of slot p at [world, f, p], so a warp's units read each field in one transaction.
    Field("slot_data", wp.float32, ("world", SLOT_ENTRIES + 2, "unit")),
    Field("slot_words", wp.int32, ("world", SLOT_WORDS, "unit")),
)
"""A world's units' rows as the colored sweep reads them: by color slot, field by field (:func:`stage_slots`)."""


@wp.kernel(enable_backward=False)
def stage_slots(
    unit_count: wp.array[wp.int32],
    color_slot: wp.array3d[wp.int32],
    mf_meta: wp.array2d[wp.int32],
    mf_J_a: wp.array3d[float],
    mf_J_b: wp.array3d[float],
    mf_MiJt_a: wp.array3d[float],
    mf_MiJt_b: wp.array3d[float],
    mf_row_mu: wp.array2d[float],
    mf_row_w: wp.array2d[float],
    slot_data: wp.array3d[float],
    slot_words: wp.array3d[wp.int32],
):
    """Gather each color slot's unit rows into the slot-major :data:`SLOT_FIELDS` (one thread per world and slot)."""
    world, p = wp.tid()
    if p >= unit_count[world]:
        return
    row = color_slot[world, p, 0]
    rows = color_slot[world, p, 1] & 3
    for r in range(3):
        if r < rows:
            for k in range(6):
                slot_data[world, r * 24 + k, p] = mf_J_a[world, row + r, k]
                slot_data[world, r * 24 + 6 + k, p] = mf_J_b[world, row + r, k]
                slot_data[world, r * 24 + 12 + k, p] = mf_MiJt_a[world, row + r, k]
                slot_data[world, r * 24 + 18 + k, p] = mf_MiJt_b[world, row + r, k]
            slot_words[world, r * 3, p] = mf_meta[world, (row + r) * 4 + 1]
            slot_words[world, r * 3 + 1, p] = mf_meta[world, (row + r) * 4 + 2]
            slot_words[world, r * 3 + 2, p] = mf_meta[world, (row + r) * 4 + 3]
    slot_data[world, _SLOT_MU, p] = wp.where(rows == 3, mf_row_mu[world, row + 1], 0.0)
    slot_data[world, _SLOT_WEIGHT, p] = mf_row_w[world, row]


@dataclasses.dataclass(frozen=True)
class UnitPlan:
    """Shapes of a world's unit data: free-body row capacity, world coordinates and extra copy capacity."""

    mf_rows: int
    dofs: int
    extra: int

    colors: int = 64

    @property
    def sizes(self) -> dict[str, int]:
        return {"unit": self.mf_rows, "color": self.colors + 1, "split": self.extra + 1, "palette": self.colors}


_BUILD_SOURCE = r"""
    // Shared scratch (declared by build_staging): a bitmap of the body coordinates the rows use and its word prefix
    // (compact node ids), per node (copies included) its unit count, split index, color mask and level, owner in a
    // leveling round and next-fit state, the split bodies, the colors' sizes, each warp's count and its colors'
    // slots, the current chunk of units and the block's counters.
SHARED_DECLARATIONS
#define split_count_s build_state[0]
#define split_dropped build_state[1]
#define split_short_s build_state[2]
#define budget_s build_state[3]
#define colors_s build_state[4]
#define overflow_s build_state[5]

    const int lane = LANE;
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int m_mf = min(mf_constraint_count.data[world], MF_ROWS);
    const int end = min(mf_contact_rows_end.data[world], m_mf);
    const int off_mf = world * MF_ROWS;
    const int off_meta = off_mf * 4;

    for (int w = tid; w < WORDS; w += THREADS) used_bits[w] = 0u;
    for (int n = tid; n < NODES; n += THREADS) {
        node_degree[n] = 0;
        node_fill[n] = 0;
        node_copy[n] = 0;
        node_split[n] = -1;
        node_owner[n] = 32;
        node_level[n] = -1;
    }
    for (int n = tid; n < NODES * MASK_WORDS; n += THREADS) node_colors[n] = 0ull;
    for (int c = tid; c < MAX_COLORS; c += THREADS) color_size[c] = 0;
    if (tid == 0) {
        split_dropped = 0;
        split_short_s = 0;
    }
    __syncthreads();

    // Units in row order: a contact normal row and the friction rows after it whose parent it is. Each warp takes
    // 32 rows of a chunk; the warps' counts give each its first slot.
    int units = 0;
    for (int base = 0; base < end; base += THREADS) {
        const int row = base + tid;
        bool head = false;
        if (row < end) head = (mf_meta.data[off_meta + row * 4 + 3] & 0xFFFF) == CONTACT;
        int counted = 0;
        const int rank = ORDERED(counted, head);
        if (lane == 0) warp_count[warp] = counted;
        __syncthreads();
        int first = units, total = 0;
        for (int w = 0; w < WARPS; ++w) {
            if (w < warp) first += warp_count[w];
            total += warp_count[w];
        }
        if (head) {
            const int unit = first + rank;
            int rows = 1;
            while (row + rows < end && rows < 3) {
                const int packed = mf_meta.data[off_meta + (row + rows) * 4 + 3];
                if ((packed & 0xFFFF) != FRICTION || (PARENT(packed)) != row) break;
                ++rows;
            }
            const int dofs = mf_meta.data[off_meta + row * 4];
            unit_row.data[off_mf + unit] = row;
            unit_rows.data[off_mf + unit] = rows;
            unit_dofs.data[off_mf + unit] = dofs;
            const int dof_a = dofs >> 16, dof_b = (dofs << 16) >> 16;
            if (dof_a >= 0) atomicOr(&used_bits[dof_a >> 5], 1u << (dof_a & 31));
            if (dof_b >= 0) atomicOr(&used_bits[dof_b >> 5], 1u << (dof_b & 31));
        }
        units += total;
        __syncthreads();
    }
    // Compact node ids: the rank of a body's first coordinate among the used ones (a scan of the bitmap's counts).
    if (warp == 0) {
        int carry = 0;
        for (int base = 0; base < WORDS; base += 32) {
            const int w = base + lane;
            const int count = w < WORDS ? __popc(used_bits[w]) : 0;
            const int below = SCAN(count);
            if (w < WORDS) used_prefix[w] = carry + below;
            carry += READ(below + count, 31);
        }
    }
    __syncthreads();
    const auto NODE = [&](int dof) -> int {
        return used_prefix[dof >> 5] + __popc(used_bits[dof >> 5] & ((1u << (dof & 31)) - 1u));
    };
    // Degrees: every unit counts once on each of its bodies (integer counts: the order does not matter).
    for (int unit = tid; unit < units; unit += THREADS) {
        const int dofs = unit_dofs.data[off_mf + unit];
        const int dof_a = dofs >> 16, dof_b = (dofs << 16) >> 16;
        if (dof_a >= 0) atomicAdd(&node_degree[NODE(dof_a)], 1);
        if (dof_b >= 0 && dof_b != dof_a) atomicAdd(&node_degree[NODE(dof_b)], 1);
    }
    __syncthreads();

    // The split budget: the caller's, none (0), or the data's (AUTO_BUDGET): twice the mean degree of the bodies.
    if (warp == 0) {
        int budget = split_budget > 0 ? split_budget : 0x7fffffff;
        if (split_budget < 0) {
            int degrees = 0, bodies = 0;
            for (int n = lane; n < NODES; n += 32) {
                degrees += node_degree[n];
                bodies += node_degree[n] > 0 ? 1 : 0;
            }
            degrees = __reduce_add_sync(0xffffffffu, degrees);
            bodies = __reduce_add_sync(0xffffffffu, bodies);
            if (bodies > 0) budget = max(1, (2 * degrees + bodies - 1) / bodies);
        }
        if (lane == 0) budget_s = budget;
    }
    __syncthreads();
    const int budget = budget_s;
    // Bodies over the budget, in coordinate order, are split.
    if (tid == 0) {
        unit_budget.data[world] = budget == 0x7fffffff ? 0 : budget;
        unit_count.data[world] = units;
        int splits = 0;
        for (int w = 0; w < WORDS; ++w) {
            unsigned bits = used_bits[w];
            while (bits) {
                const int bit = __ffs(bits) - 1;
                bits &= bits - 1u;
                const int node = NODE(w * 32 + bit);
                if (node_degree[node] <= budget) continue;
                if (splits > EXTRA) {
                    split_dropped = 1;
                    continue;
                }
                node_split[node] = splits;
                split_node_s[splits] = node;
                split_dof_s[splits] = w * 32 + bit;
                ++splits;
            }
        }
        split_count_s = splits;
    }
    __syncthreads();

    // Copies: a split body's patches take its copies in row order, next fit: a patch opens the next copy when the
    // current one cannot hold all of its contacts, so a copy holds at most `budget` units (or one patch larger
    // than that) and a patch's contacts share a copy. Every thread finds its unit's patch; the first thread fits
    // the chunk's patches in row order. Each unit keeps its sides' copy indices (in its offsets until the colors
    // place the copies).
    for (int base = 0; base < units; base += THREADS) {
        const int unit = base + tid;
        const bool valid = unit < units;
        const int dofs = valid ? unit_dofs.data[off_mf + unit] : 0;
        const int dof_a = valid ? dofs >> 16 : -1;
        int dof_b = valid ? (dofs << 16) >> 16 : -1;
        if (dof_b == dof_a) dof_b = -1;
        const int id_a = dof_a >= 0 ? NODE(dof_a) : -1;
        const int id_b = dof_b >= 0 ? NODE(dof_b) : -1;
        const bool touches_split = (id_a >= 0 && node_split[id_a] >= 0) || (id_b >= 0 && node_split[id_b] >= 0);
        if (__syncthreads_or(touches_split) == 0) {
            if (valid) {
                unit_offset_a.data[off_mf + unit] = 0;
                unit_offset_b.data[off_mf + unit] = 0;
            }
            continue;
        }
        // The unit's patch: its first contact (the lowest row of its normal rows' ring) and its contacts.
        const int row = valid ? unit_row.data[off_mf + unit] : -1;
        int head = row, size = 1;
        if (touches_split) {
            int hop = 0;
            for (int r = PARENT(mf_meta.data[off_meta + row * 4 + 3]); r >= 0 && r != row && hop < MF_ROWS;
                 r = PARENT(mf_meta.data[off_meta + r * 4 + 3]), ++hop) {
                head = min(head, r);
                ++size;
            }
        }
        batch_node[2 * tid] = touches_split ? id_a : -1;
        batch_node[2 * tid + 1] = touches_split ? id_b : -1;
        batch_row[tid] = row;
        batch_head[tid] = head;
        batch_size[tid] = size;
        __syncthreads();
        if (tid == 0) {
            const int count = min(THREADS, units - base);
            for (int k = 0; k < count; ++k) {
                int copy[2] = {0, 0};
                for (int side = 0; side < 2; ++side) {
                    const int n = batch_node[2 * k + side];
                    if (n < 0 || node_split[n] < 0) continue;
                    if (batch_head[k] == batch_row[k]) {
                        if (node_fill[n] > 0 && node_fill[n] + batch_size[k] > budget) {
                            ++node_copy[n];
                            node_fill[n] = 0;
                        }
                        node_fill[n] += batch_size[k];
                        copy[side] = node_copy[n];
                    } else {
                        const int taken = row_copy.data[off_mf + batch_head[k]];
                        copy[side] = side == 0 ? (taken & 0xffff) : (taken >> 16);
                    }
                }
                if (batch_head[k] == batch_row[k]) row_copy.data[off_mf + batch_row[k]] = copy[0] | (copy[1] << 16);
                batch_copy[2 * k] = copy[0];
                batch_copy[2 * k + 1] = copy[1];
            }
        }
        __syncthreads();
        if (valid) {
            unit_offset_a.data[off_mf + unit] = batch_copy[2 * tid];
            unit_offset_b.data[off_mf + unit] = batch_copy[2 * tid + 1];
        }
        __syncthreads();
    }

    // The split bodies' extra copies, in coordinate order; a body without room for its copies stays whole.
    if (tid == 0) {
        int splits = 0, extra = 0;
        bool split_short = split_dropped != 0;
        for (int s = 0; s < split_count_s; ++s) {
            const int node = split_node_s[s];
            const int copies = node_copy[node] + 1;
            if (copies < 2 || extra + copies - 1 > EXTRA) {
                split_short = split_short || copies >= 2;
                node_split[node] = -1;
                continue;
            }
            node_split[node] = splits;
            split_node_s[splits] = node;
            split_first_s[splits] = extra;
            split_copies_s[splits] = copies;
            split_dof.data[off_mf + splits] = split_dof_s[s];
            split_copies.data[off_mf + splits] = copies;
            split_first.data[off_mf + splits] = extra;
            extra += copies - 1;
            ++splits;
        }
        split_count_s = splits;
        split_short_s = split_short ? 1 : 0;
        split_count.data[world] = splits;
        extra_copies.data[world] = extra;
    }
    __syncthreads();

    // Levels in row order, by the first warp, a batch of 32 units at a time. A unit's color is one past the latest
    // color of its body copies, so units on a copy keep their row order. In each round every pending unit claims its
    // copies (the lowest lane wins); a unit that owns both commits its color: no earlier pending unit shares a copy
    // with it, so it sees every earlier unit's color, exactly as a pass in row order would.
    if (warp == 0) {
        bool overflow = false;     // a unit's level is past the colors: the last color holds it, swept in row order
        int colors = 0;
        for (int base = 0; base < units; base += 32) {
            const int unit = base + lane;
            const bool valid = unit < units;
            const int dofs = valid ? unit_dofs.data[off_mf + unit] : 0;
            const int dof_a = valid ? dofs >> 16 : -1;
            int dof_b = valid ? (dofs << 16) >> 16 : -1;
            if (dof_b == dof_a) dof_b = -1;
            const int id_a = dof_a >= 0 ? NODE(dof_a) : -1;
            const int id_b = dof_b >= 0 ? NODE(dof_b) : -1;
            const int copy_a = valid ? unit_offset_a.data[off_mf + unit] : 0;
            const int copy_b = valid ? unit_offset_b.data[off_mf + unit] : 0;
            int node[2] = {id_a, id_b};
            int offset[2] = {dof_a, dof_b};
            const int copy[2] = {copy_a, copy_b};
            for (int side = 0; side < 2; ++side) {
                const int id = node[side];
                if (id < 0) continue;
                const int split = node_split[id];
                if (split < 0 || copy[side] == 0) continue;
                const int index = split_first_s[split] + copy[side] - 1;
                node[side] = BODIES + index;
                offset[side] = DOFS + 6 * index;
            }
            SYNC();
            if (valid) {
                unit_offset_a.data[off_mf + unit] = offset[0];
                unit_offset_b.data[off_mf + unit] = (dofs << 16) >> 16 == dof_a ? offset[0] : offset[1];
            }
            unsigned pending = BALLOT(valid);
            int color = -1;
            while (pending != 0u) {
                const bool mine = ((pending >> lane) & 1u) != 0u;
                if (mine) {
                    if (node[0] >= 0) atomicMin(&node_owner[node[0]], lane);
                    if (node[1] >= 0) atomicMin(&node_owner[node[1]], lane);
                }
                SYNC();
                const bool ready = mine && (node[0] < 0 || node_owner[node[0]] == lane)
                    && (node[1] < 0 || node_owner[node[1]] == lane);
                if (ready) {
                    color = 0;
                    if (node[0] >= 0) color = max(color, node_level[node[0]] + 1);
                    if (node[1] >= 0) color = max(color, node_level[node[1]] + 1);
                    if (color >= MAX_COLORS) {
                        overflow = true;
                        color = MAX_COLORS - 1;
                    }
                    if (node[0] >= 0) node_level[node[0]] = color;
                    if (node[1] >= 0) node_level[node[1]] = color;
                    const unsigned long long bit = 1ull << (color & 63);
                    if (node[0] >= 0) node_colors[node[0] * MASK_WORDS + (color >> 6)] |= bit;
                    if (node[1] >= 0) node_colors[node[1] * MASK_WORDS + (color >> 6)] |= bit;
                    atomicAdd(&color_size[color], 1);
                }
                SYNC();
                if (mine) {
                    if (node[0] >= 0) node_owner[node[0]] = 32;
                    if (node[1] >= 0) node_owner[node[1]] = 32;
                }
                SYNC();
                pending &= ~BALLOT(ready);
            }
            if (valid) unit_color.data[off_mf + unit] = color;
            colors = max(colors, color + 1);
        }
        MAXIMUM(colors);
        overflow = ANY(overflow);
        // Color offsets.
        if (lane == 0) {
            int start = 0;
            for (int c = 0; c < MAX_COLORS; ++c) {
                color_offset.data[world * (MAX_COLORS + 1) + c] = start;
                const int size = color_size[c];
                color_size[c] = start;
                start += size;
            }
            color_offset.data[world * (MAX_COLORS + 1) + MAX_COLORS] = start;
            color_count.data[world] = overflow ? MAX_COLORS : colors;
            color_overflow.data[world] = (overflow ? 1 : 0) | (split_short_s ? 2 : 0);
            colors_s = colors;
            overflow_s = overflow ? 1 : 0;
        }
    }
    __syncthreads();
    const bool overflow = overflow_s != 0;
    // Each split body's units in each color: its copies that took the color (a copy takes a color once). The
    // overflow color's units run in row order on the bodies themselves: one.
    const int split_total = split_count_s;
    const int split_units_base = world * (EXTRA + 1) * MAX_COLORS;
    for (int e = tid; e < split_total * MAX_COLORS; e += THREADS) {
        const int s = e / MAX_COLORS, c = e - s * MAX_COLORS;
        int j = 0;
        for (int copy = 0; copy < split_copies_s[s]; ++copy) {
            const int node = copy == 0 ? split_node_s[s] : BODIES + split_first_s[s] + copy - 1;
            j += (int)((node_colors[node * MASK_WORDS + (c >> 6)] >> (c & 63)) & 1ull);
        }
        if (overflow && c == MAX_COLORS - 1) j = j > 0 ? 1 : 0;
        split_units.data[split_units_base + e] = j;
    }
    // Each color's units in row order: every warp places a contiguous range of units, after the units of its
    // colors in the ranges before it (the warps' color counts); in a range the lanes of one color take their slots
    // in lane order.
    const int span = ((units + WARPS - 1) / WARPS + 31) / 32 * 32;
    const int lo = min(units, warp * span), hi = min(units, lo + span);
    for (int c = lane; c < MAX_COLORS; c += 32) warp_color[warp * MAX_COLORS + c] = 0;
    SYNC();
    for (int unit = lo + lane; unit < hi; unit += 32)
        atomicAdd(&warp_color[warp * MAX_COLORS + unit_color.data[off_mf + unit]], 1);
    __syncthreads();
    for (int c = tid; c < MAX_COLORS; c += THREADS) {
        int start = color_size[c];
        for (int w = 0; w < WARPS; ++w) {
            const int count = warp_color[w * MAX_COLORS + c];
            warp_color[w * MAX_COLORS + c] = start;
            start += count;
        }
    }
    __syncthreads();
    for (int base = lo; base < hi; base += 32) {
        const int unit = base + lane;
        const int color = unit < hi ? unit_color.data[off_mf + unit] : -1;
        const unsigned same = __match_any_sync(0xffffffffu, color);
        const int rank = __popc(same & ((1u << lane) - 1u));
        int slot = 0;
        if (color >= 0) slot = warp_color[warp * MAX_COLORS + color] + rank;
        SYNC();
        if (color >= 0) {
            // A split side's units in the color scale its response; an overflow unit runs on the body itself.
            const bool serial = overflow && color == MAX_COLORS - 1;
            const int dofs = unit_dofs.data[off_mf + unit];
            const int dof_a = dofs >> 16, dof_b = (dofs << 16) >> 16;
            int offset_a = unit_offset_a.data[off_mf + unit], offset_b = unit_offset_b.data[off_mf + unit];
            int ja = 1, jb = 1;
            const int split_a = dof_a >= 0 ? node_split[NODE(dof_a)] : -1;
            const int split_b = dof_b >= 0 && dof_b != dof_a ? node_split[NODE(dof_b)] : -1;
            if (split_a >= 0) {
                ja = serial ? 1 : split_units.data[split_units_base + split_a * MAX_COLORS + color];
                if (serial) offset_a = dof_a;
            }
            if (split_b >= 0) {
                jb = serial ? 1 : split_units.data[split_units_base + split_b * MAX_COLORS + color];
                if (serial) offset_b = dof_b;
            }
            if (dof_b == dof_a) {
                offset_b = offset_a;
                jb = ja;
            }
            const int at = (off_mf + slot) * 4;
            color_slot.data[at] = unit_row.data[off_mf + unit];
            color_slot.data[at + 1] = unit_rows.data[off_mf + unit] | (ja << 2) | (jb << 17);
            color_slot.data[at + 2] = offset_a;
            color_slot.data[at + 3] = offset_b;
            if (rank == 0) warp_color[warp * MAX_COLORS + color] += __popc(same);
        }
        SYNC();
    }
#undef split_count_s
#undef split_dropped
#undef split_short_s
#undef budget_s
#undef colors_s
#undef overflow_s
#endif
"""


@cache
def build_staging(dofs: int, extra: int, colors: int) -> Staging:
    """The unit builder's shared working set for ``dofs`` coordinates, ``extra`` copies and ``colors`` colors."""
    nodes = dofs // 6 + 1 + extra
    words = (dofs + 31) // 32
    threads = 32 * BUILD_WARPS
    return Staging(
        (
            Buffer("used_bits", "unsigned", words),
            Buffer("used_prefix", "int", words),
            Buffer("node_degree", "int", nodes),
            Buffer("node_split", "int", nodes),
            Buffer("node_owner", "int", nodes),
            Buffer("node_fill", "int", nodes),
            Buffer("node_copy", "int", nodes),
            Buffer("node_colors", "unsigned long long", nodes * (colors // 64)),
            Buffer("node_level", "int", nodes),
            Buffer("split_first_s", "int", extra + 1),
            Buffer("split_copies_s", "int", extra + 1),
            Buffer("split_node_s", "int", extra + 1),
            Buffer("split_dof_s", "int", extra + 1),
            Buffer("color_size", "int", colors),
            Buffer("warp_color", "int", BUILD_WARPS * colors),
            Buffer("warp_count", "int", BUILD_WARPS),
            Buffer("batch_node", "int", 2 * threads),
            Buffer("batch_copy", "int", 2 * threads),
            Buffer("batch_row", "int", threads),
            Buffer("batch_head", "int", threads),
            Buffer("batch_size", "int", threads),
            Buffer("build_state", "int", 6),
        ),
        threads=threads,
    )


def build_units_kernel(
    mf_rows: int, dofs: int, extra: int, colors: int = 64, *, static_bytes: int = STATIC_SHARED_BYTES
) -> wp.Kernel:
    """The unit builder of worlds with ``mf_rows`` free-body rows, ``dofs`` coordinates and ``extra`` copy capacity.

    One block of :data:`BUILD_WARPS` warps per world: launch ``dim=[worlds]`` tiled with ``block_dim=32 *
    BUILD_WARPS``. Inputs: ``mf_constraint_count``,
    ``mf_contact_rows_end``, ``mf_meta`` and the split budget (0: no body is split; :data:`AUTO_BUDGET`: the
    data's); outputs: :data:`UNIT_FIELDS`
    (``row_copy`` is the builder's scratch). Its shared working set (:func:`build_staging`) must fit
    ``static_bytes`` (a device's :func:`.staging.static_budget`).
    """
    if colors % 64 or not 64 <= colors <= MAX_COLORS:
        raise ValueError(f"colors must be a multiple of 64 up to {MAX_COLORS}, got {colors}")
    staging = build_staging(dofs, extra, colors)
    staging.check("the unit builder", static_bytes)
    parent = _mf_parent_expression("packed", mf_rows).replace("packed", "PACKED_WORD")
    bodies = dofs // 6 + 1
    words = (dofs + 31) // 32
    source = (
        f"#define MF_ROWS {mf_rows}\n#define DOFS {dofs}\n#define EXTRA {extra}\n#define MAX_COLORS {colors}\n"
        f"#define MASK_WORDS {colors // 64}\n#define WARPS {BUILD_WARPS}\n#define THREADS {32 * BUILD_WARPS}\n"
        f"#define BODIES {bodies}\n#define NODES {bodies + extra}\n#define WORDS {words}\n"
        f"#define CONTACT {int(PGS_CONSTRAINT_TYPE_CONTACT)}\n#define FRICTION {int(PGS_CONSTRAINT_TYPE_FRICTION)}\n"
        f"#define PARENT(PACKED_WORD) ({parent})\n"
        + _BUILD_SOURCE.replace("SHARED_DECLARATIONS", staging.declarations()).replace(
            "#endif",
            "#undef MF_ROWS\n#undef DOFS\n#undef EXTRA\n#undef MAX_COLORS\n#undef MASK_WORDS\n#undef WARPS\n"
            "#undef THREADS\n#undef BODIES\n"
            "#undef NODES\n#undef WORDS\n"
            "#undef CONTACT\n#undef FRICTION\n#undef PARENT\n#endif",
        )
    )
    params = kernel_template.params(
        ("mf_constraint_count", wp.array[int]),
        ("mf_contact_rows_end", wp.array[int]),
        ("mf_meta", wp.array2d[int]),
        ("split_budget", int),
        *UNIT_FIELDS,
    )
    return KernelTemplate(
        f"sweep_units_{mf_rows}_{dofs}_{extra}_{colors}_w{BUILD_WARPS}",
        params,
        cuda_native(source, preludes=(lane_group(32),)),
        dispatch=kernel_template.WORLDS,
        composed=True,
        native_name="sweep_units_native",
        launch_bounds=(32 * BUILD_WARPS, 1),
    ).kernel()
