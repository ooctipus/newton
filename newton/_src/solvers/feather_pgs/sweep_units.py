# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contact units and their colors: the data the colored free-body sweep runs on.

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
bin, 491 units, 495 colors). Its units per color are at most its free bodies,
so only the worlds with enough free bodies for the rule (:func:`capable_worlds`)
hold that storage, a slot each; the others keep the one-warp mapping and hold
none. A slot's capacity is sized from the units a world takes
(:func:`unit_capacity`), not from its row capacity; a world with more units
keeps the one-warp mapping and sets the capacity status
(:meth:`ColoredUnits.capacity_status`). A world never takes more colors than
it has units, so the colors fit the slot whenever its units do.

Everything is computed from the data in row order, one block per world: units
in row order, levels in row order, units of a color in row order. Nothing
depends on scheduling, so the colors and the sweep are deterministic whenever
the rows are.
"""

from __future__ import annotations

import dataclasses
from functools import cache

import numpy as np
import warp as wp

from . import kernel_template
from .device_preludes import SHARED_STORAGE, cuda_native, lane_group
from .field_group import Field, FieldGroup
from .kernel_template import KernelTemplate
from .kernels import PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_FRICTION
from .row_metadata import _mf_parent_expression
from .staging import STATIC_SHARED_BYTES, Buffer, Staging, static_budget

BUILD_WARPS = 8
"""Warps of a world's unit builder block: they share the parallel passes; the first levels the units."""


@dataclasses.dataclass(frozen=True)
class ColoredRule:
    """Which worlds the colored mapping takes, decided per world at every pack from the unit builder's counts.

    A world takes it with at least ``min_units`` contact units and at least ``parallelism`` units per color on
    average (``units >= parallelism * colors``): the one-warp mapping visits a world's units one after another, the
    colored one its colors, each costing about ``parallelism`` units' time (its slowest unit, its barrier, its idle
    threads), so a world gains in proportion to its units per color past that.
    """

    min_units: int
    parallelism: float
    waves: int = 1
    """The waves the colored worlds take on the device (:meth:`at_scale`)."""

    @property
    def units_per_color(self) -> float:
        """The units per color a world needs: the parallelism of one wave of worlds, times the waves."""
        return float(self.parallelism) * int(self.waves)

    def at_scale(self, worlds: int, slots: int) -> ColoredRule:
        """The rule for ``worlds`` colored worlds on a device that holds ``slots`` colored blocks at once (one per
        SM, the colored sweep's launch bounds): past that the colored worlds run in waves while the one-warp mapping
        keeps its worlds resident, so a world needs as many times the units per color as there are waves."""
        return dataclasses.replace(self, waves=max(-(-int(worlds) // max(int(slots), 1)), 1))

    def condition(self, units: str, colors: str) -> str:
        """The rule as a native boolean of the ``units`` and ``colors`` expressions."""
        need = repr(self.units_per_color)
        return f"({units} >= {int(self.min_units)} && (float){units} >= {need}f * (float){colors})"

    @property
    def name(self) -> str:
        """The rule's kernel-name suffix."""
        return f"u{int(self.min_units)}_p" + repr(self.units_per_color).replace(".", "p")


COLORED_RULE = ColoredRule(min_units=32, parallelism=1.75)
"""The measured crossover of the two mappings (RTX PRO 6000, patch box pyramids of 2 to 14 rows, ms per frame): one
wave of worlds (1 and 64) ties at 1.25 to 1.6 units per color (20 to 49 units: 1.02, 0.97, 1.02x) and gains from 1.9
(1.20x) through 2.1 to 2.8 (1.17 to 1.40x) and 6.4 to 6.9 (2.5 to 2.8x) to 15 (5.2x); the coins and hardware
clutter, about 12 units per color, gain 5.7 and 2.4x at 128 worlds. At 1024 worlds, six waves, the colored worlds
lose at 2.3 and 6.6 units per color (0.35 and 0.62x), as :meth:`ColoredRule.at_scale` requires."""

WORLD_FIELDS = FieldGroup(
    # 1 where the colored mapping takes the world (:class:`ColoredRule`), else 0: the one-warp mapping keeps it.
    Field("world_colored", wp.int32, ("world",)),
    # The world's storage slot, -1 for a world that cannot take the colored mapping (:func:`capable_worlds`).
    Field("world_slot", wp.int32, ("world",)),
    Field("unit_count", wp.int32, ("world",)),
    Field("color_count", wp.int32, ("world",)),
)
"""Every world's mapping, slot and unit and color counts."""

UNIT_FIELDS = FieldGroup(
    # First row and row count of each unit, in row order.
    Field("unit_row", wp.int32, ("slot", "unit")),
    Field("unit_rows", wp.int32, ("slot", "unit")),
    # Packed first coordinates of the unit's bodies (``dof_a << 16 | dof_b``, as the rows' packed word).
    Field("unit_dofs", wp.int32, ("slot", "unit")),
    Field("unit_color", wp.int32, ("slot", "unit")),
    # The unit's place among its color's units, in row order (the builder's scratch).
    Field("unit_rank", wp.int32, ("slot", "unit")),
    # Colors: offsets into color_slot (count + 1 of them; at most one color per unit), and the units of each color in
    # row order, each as the record the sweep reads: first row, row count and its bodies' first coordinates.
    Field("color_offset", wp.int32, ("slot", "color")),
    Field("color_slot", wp.int32, ("slot", "unit", 4)),
)
"""The units and colors of each slot's world (``unit``: the slot's unit capacity, :func:`unit_capacity`; ``color``:
one more)."""

SLOT_ENTRIES = 36
"""Entries a color slot stages per unit: for each of its (up to three) rows ``r``, the six ``Z_a`` and ``Z_b``
entries, at ``r * 12 + side * 6 + k``."""

SLOT_MU, SLOT_WEIGHT = SLOT_ENTRIES, SLOT_ENTRIES + 1
"""The slot's scalars after its entries: the friction coefficient of its first tangent and its normal row's
regularization weight."""

RING_MEMBERS = 8
"""Other normals of a patch ring a slot stages for its friction pair's pooled load; a larger ring walks its links."""

SLOT_RING = 9
"""The slot words of a friction unit's patch ring: from ``SLOT_RING``, the other normal rows of its ring in ring order
(two 16-bit rows a word, 0xFFFF past the last), then their count (-1 for a ring larger than :data:`RING_MEMBERS`)."""

SLOT_WORDS = SLOT_RING + RING_MEMBERS // 2 + 1
"""Words a color slot stages per unit: for each of its (up to three) rows ``r``, at ``r * 3``, the bits of its stored
inverse diagonal, of its right-hand side, and its type and parent word (``mf_meta``'s ``y``, ``z``, ``w``); then its
patch ring (:data:`SLOT_RING`)."""

_SLOT_MU = wp.constant(SLOT_MU)
_SLOT_WEIGHT = wp.constant(SLOT_WEIGHT)
_SLOT_RING = wp.constant(SLOT_RING)
_RING_MEMBERS = wp.constant(RING_MEMBERS)

METRIC_FIELDS = 7
"""Floats of a friction unit's cached metric: its pair's metric (scale, the two eigenvalues, the larger's axis) and
the pair's Gram couplings with the normal row, ``Z_n . Z_t1`` and ``Z_n . Z_t2``."""

SLOT_FIELDS = FieldGroup(
    # Slot-major: field f of unit p at [slot, f, p], so a warp's units read each field in one transaction.
    Field("slot_data", wp.float32, ("slot", SLOT_ENTRIES + 2, "unit")),
    Field("slot_words", wp.int32, ("slot", SLOT_WORDS, "unit")),
    # The friction pair metric and Gram couplings of each staged friction unit (METRIC_FIELDS), formed by the
    # colored sweep once per launch.
    Field("slot_metric", wp.float32, ("slot", METRIC_FIELDS, "unit")),
)
"""A world's units' rows as the colored sweep reads them: by color slot, field by field (:func:`stage_slots`)."""


@wp.kernel(enable_backward=False)
def stage_slots(
    slot_world: wp.array[wp.int32],
    world_colored: wp.array[wp.int32],
    unit_count: wp.array[wp.int32],
    color_slot: wp.array3d[wp.int32],
    mf_meta: wp.array2d[wp.int32],
    mf_Z_a: wp.array3d[float],
    mf_Z_b: wp.array3d[float],
    mf_row_mu: wp.array2d[float],
    mf_row_w: wp.array2d[float],
    slot_data: wp.array3d[float],
    slot_words: wp.array3d[wp.int32],
):
    """Gather each colored world's unit rows into its slot's slot-major :data:`SLOT_FIELDS` (one thread per slot and
    unit)."""
    slot, p = wp.tid()
    world = slot_world[slot]
    if world_colored[world] == 0 or p >= unit_count[world]:
        return
    row = color_slot[slot, p, 0]
    rows = color_slot[slot, p, 1] & 3
    for r in range(3):
        if r < rows:
            for k in range(6):
                slot_data[slot, r * 12 + k, p] = mf_Z_a[world, row + r, k]
                slot_data[slot, r * 12 + 6 + k, p] = mf_Z_b[world, row + r, k]
            slot_words[slot, r * 3, p] = mf_meta[world, (row + r) * 4 + 1]
            slot_words[slot, r * 3 + 1, p] = mf_meta[world, (row + r) * 4 + 2]
            slot_words[slot, r * 3 + 2, p] = mf_meta[world, (row + r) * 4 + 3]
    slot_data[slot, _SLOT_MU, p] = wp.where(rows == 3, mf_row_mu[world, row + 1], 0.0)
    slot_data[slot, _SLOT_WEIGHT, p] = mf_row_w[world, row]
    if rows == 3:
        # The friction pair's patch ring: its normal's ring successors in ring order (the pooled load's walk).
        words = wp.vec4i(-1, -1, -1, -1)
        members = int(0)
        hop = int(0)
        rows_cap = mf_meta.shape[1] // 4
        link = (mf_meta[world, row * 4 + 3] >> 16) & 0xFFFF
        r = wp.where(link == 0xFFFF, -1, link)
        while r >= 0 and r != row and hop < rows_cap:
            if members == _RING_MEMBERS:
                members = -1
                break
            # Two 16-bit rows a word, low half first; an unused half stays 0xFFFF.
            if members % 2 == 0:
                words[members // 2] = (r & 0xFFFF) | (-65536)
            else:
                words[members // 2] = (words[members // 2] & 0xFFFF) | (r << 16)
            members += 1
            hop += 1
            link = (mf_meta[world, r * 4 + 3] >> 16) & 0xFFFF
            r = wp.where(link == 0xFFFF, -1, link)
        for k in range(_RING_MEMBERS // 2):
            slot_words[slot, _SLOT_RING + k, p] = words[k]
        slot_words[slot, _SLOT_RING + _RING_MEMBERS // 2, p] = members


UNIT_ROWS = 3
"""Free-body rows a slot's unit capacity counts per unit (:func:`unit_capacity`): a contact's normal row and its
friction pair."""


def unit_capacity(mf_rows: int) -> int:
    """Units a slot holds for worlds of ``mf_rows`` free-body rows: one per :data:`UNIT_ROWS` rows, every unit a
    contact with its friction pair (measured, 128 worlds: coins 4,446 units, 4,704 at most, of 5,328 a slot; the parts
    bin 5,075, 5,553 at most, of 7,509). A world with more keeps the one-warp mapping, and the build sets the units'
    capacity status."""
    return max(int(mf_rows) // UNIT_ROWS, 1)


@dataclasses.dataclass(frozen=True)
class UnitPlan:
    """Shapes of the unit data: every world's counts, and per slot (a world that can take the colored mapping) its
    units and colors, which the slot's unit capacity bounds (:func:`unit_capacity`)."""

    worlds: int
    slots: int
    units: int

    @property
    def sizes(self) -> dict[str, int]:
        return {"world": self.worlds, "slot": self.slots, "unit": self.units, "color": self.units + 1}


def capable_worlds(free_bodies, mf_rows: int, rule: ColoredRule = COLORED_RULE):
    """The worlds that can take the colored mapping, from the model at construction: a world's units per color are
    at most its free bodies (each color holds a unit of the busiest body, at least ``units / bodies`` units share
    one), so it needs ``rule.units_per_color`` of them, and a row capacity of ``rule.min_units`` (a unit has a row).

    Args:
        free_bodies: Free rigid bodies per world (array of the worlds).
        mf_rows: Free-body row capacity per world.
        rule: The colored rule.

    Returns:
        The capable worlds' indices, in world order.
    """
    if int(mf_rows) < int(rule.min_units):
        return np.zeros(0, dtype=np.int32)
    return np.flatnonzero(np.asarray(free_bodies) >= rule.units_per_color).astype(np.int32)


_BUILD_SOURCE = r"""
    // Shared scratch (declared by build_staging): a bitmap of the body coordinates the units use and its word prefix
    // (compact body ids), per body its latest level and its owner in a leveling round, and the warps' counts.
SHARED_DECLARATIONS
    const int lane = LANE;
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int m_mf = min(mf_constraint_count.data[world], MF_ROWS);
    const int end = min(mf_contact_rows_end.data[world], m_mf);
    const int off_meta = world * MF_ROWS * 4;
    // The world's slot holds its units and colors; a world without one, or with fewer contact rows than the rule's
    // units, keeps the one-warp sweep: no units.
    const int slot = world_slot.data[world];
    const int off_mf = slot * UNITS;
    const int off_color = slot * (UNITS + 1);
    if (slot < 0 || end < MIN_UNITS) {
        if (tid == 0) {
            world_colored.data[world] = 0;
            unit_count.data[world] = 0;
            color_count.data[world] = 0;
        }
        return;
    }

    for (int w = tid; w < WORDS; w += THREADS) used_bits[w] = 0u;
    for (int n = tid; n < NODES; n += THREADS) {
        node_owner[n] = 32;
        node_level[n] = -1;
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
        // The slot holds UNITS units: a world with more counts them and keeps the one-warp sweep.
        if (head && first + rank < UNITS) {
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
    if (units > UNITS) {
        // Past the slot's units: the one-warp sweep keeps the world, and the units' capacity status records it once.
        if (tid == 0) {
            world_colored.data[world] = 0;
            unit_count.data[world] = units;
            color_count.data[world] = 0;
            if (atomicOr(&unit_status.data[0], 1) == 0)
                printf("Warning: FeatherPGS colored sweep: world %d has %d contact units, over its slot's %d; the "
                       "one-warp mapping keeps it.\n", world, units, UNITS);
        }
        return;
    }
    // A unit's level is at most its index: the counts of colors [0, units] start at zero.
    for (int c = tid; c <= units; c += THREADS) color_offset.data[off_color + c] = 0;
    // Compact body ids: the rank of a body's first coordinate among the used ones (a scan of the bitmap's counts).
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

    // Levels in row order, by the first warp, a batch of 32 units at a time. A unit's color is one past the latest
    // color of its bodies, so units on a body keep their row order. In each round every pending unit claims its
    // bodies (the lowest lane wins); a unit that owns both commits its color: no earlier pending unit shares a body
    // with it, so it sees every earlier unit's color, exactly as a pass in row order would. After its round each
    // unit takes its rank in its color: the color's units of earlier rounds (color_offset counts them) and the
    // round's units of the color below it.
    if (warp == 0) {
        int colors = 0;
        for (int base = 0; base < units; base += 32) {
            const int unit = base + lane;
            const bool valid = unit < units;
            const int dofs = valid ? unit_dofs.data[off_mf + unit] : 0;
            const int dof_a = valid ? dofs >> 16 : -1;
            int dof_b = valid ? (dofs << 16) >> 16 : -1;
            if (dof_b == dof_a) dof_b = -1;
            const int node_a = dof_a >= 0 ? NODE(dof_a) : -1;
            const int node_b = dof_b >= 0 ? NODE(dof_b) : -1;
            unsigned pending = BALLOT(valid);
            int color = -1;
            while (pending != 0u) {
                const bool mine = ((pending >> lane) & 1u) != 0u;
                if (mine) {
                    if (node_a >= 0) atomicMin(&node_owner[node_a], lane);
                    if (node_b >= 0) atomicMin(&node_owner[node_b], lane);
                }
                SYNC();
                const bool ready = mine && (node_a < 0 || node_owner[node_a] == lane)
                    && (node_b < 0 || node_owner[node_b] == lane);
                if (ready) {
                    color = 0;
                    if (node_a >= 0) color = max(color, node_level[node_a] + 1);
                    if (node_b >= 0) color = max(color, node_level[node_b] + 1);
                    if (node_a >= 0) node_level[node_a] = color;
                    if (node_b >= 0) node_level[node_b] = color;
                }
                SYNC();
                if (mine) {
                    if (node_a >= 0) node_owner[node_a] = 32;
                    if (node_b >= 0) node_owner[node_b] = 32;
                }
                SYNC();
                pending &= ~BALLOT(ready);
            }
            const unsigned same = __match_any_sync(0xffffffffu, color);
            const int below = __popc(same & ((1u << lane) - 1u));
            const int earlier = valid ? color_offset.data[off_color + color] : 0;
            SYNC();
            if (valid && below == 0) color_offset.data[off_color + color] = earlier + __popc(same);
            SYNC();
            if (valid) {
                unit_color.data[off_mf + unit] = color;
                unit_rank.data[off_mf + unit] = earlier + below;
            }
            colors = max(colors, color + 1);
        }
        MAXIMUM(colors);
        if (lane == 0) {
            color_count.data[world] = colors;
            colors_s[0] = colors;
        }
    }
    __syncthreads();
    // Color offsets: an exclusive scan of the counts, in place, a block of colors at a time; offset `colors` is the
    // unit count.
    const int colors = colors_s[0];
    int carry = 0;
    for (int base = 0; base <= colors; base += THREADS) {
        const int c = base + tid;
        const int count = c < colors ? color_offset.data[off_color + c] : 0;
        const int below = SCAN(count);
        if (lane == 31) warp_count[warp] = below + count;
        __syncthreads();
        int before = carry, total = 0;
        for (int w = 0; w < WARPS; ++w) {
            if (w < warp) before += warp_count[w];
            total += warp_count[w];
        }
        if (c <= colors) color_offset.data[off_color + c] = before + below;
        carry += total;
        __syncthreads();
    }
    // Each color's units in row order: a unit's position is its color's offset plus its rank.
    for (int unit = tid; unit < units; unit += THREADS) {
        const int color = unit_color.data[off_mf + unit];
        const int position = color_offset.data[off_color + color] + unit_rank.data[off_mf + unit];
        const int dofs = unit_dofs.data[off_mf + unit];
        const int at = (off_mf + position) * 4;
        color_slot.data[at] = unit_row.data[off_mf + unit];
        color_slot.data[at + 1] = unit_rows.data[off_mf + unit];
        color_slot.data[at + 2] = dofs >> 16;
        color_slot.data[at + 3] = (dofs << 16) >> 16;
    }
    if (tid == 0) {
        unit_count.data[world] = units;
        world_colored.data[world] = COLORED(units, colors) ? 1 : 0;
    }
#endif
"""


@cache
def build_staging(dofs: int) -> Staging:
    """The unit builder's shared working set for a world of ``dofs`` coordinates."""
    nodes = dofs // 6 + 1
    words = (dofs + 31) // 32
    return Staging(
        (
            Buffer("used_bits", "unsigned", words),
            Buffer("used_prefix", "int", words),
            Buffer("node_owner", "int", nodes),
            Buffer("node_level", "int", nodes),
            Buffer("warp_count", "int", BUILD_WARPS),
            Buffer("colors_s", "int", 1),
        ),
        threads=32 * BUILD_WARPS,
    )


def build_units_kernel(
    mf_rows: int, dofs: int, units: int, *, rule: ColoredRule = COLORED_RULE, static_bytes: int = STATIC_SHARED_BYTES
) -> wp.Kernel:
    """The unit builder of worlds with ``mf_rows`` free-body rows and ``dofs`` coordinates, in slots of ``units``
    units (:func:`unit_capacity`).

    One block of :data:`BUILD_WARPS` warps per world: launch ``dim=[worlds]`` tiled with ``block_dim=32 *
    BUILD_WARPS``. Inputs: ``mf_constraint_count``, ``mf_contact_rows_end`` and ``mf_meta``; outputs:
    :data:`WORLD_FIELDS` (each world's mapping by ``rule``; ``world_slot`` is read) and its slot's
    :data:`UNIT_FIELDS`. A world without a slot, or with fewer contact rows than the rule's units, takes no units, and
    a world of more units than its slot holds sets ``unit_status``: the one-warp sweep keeps both. Its shared working set (:func:`build_staging`) must fit ``static_bytes`` (a device's
    :func:`.staging.static_budget`).
    """
    staging = build_staging(dofs)
    staging.check("the unit builder", static_bytes)
    parent = _mf_parent_expression("packed", mf_rows).replace("packed", "PACKED_WORD")
    words = (dofs + 31) // 32
    source = (
        f"#define MF_ROWS {mf_rows}\n#define UNITS {int(units)}\n#define MIN_UNITS {int(rule.min_units)}\n"
        f"#define WARPS {BUILD_WARPS}\n#define COLORED(N_UNITS, N_COLORS) {rule.condition('(N_UNITS)', '(N_COLORS)')}\n"
        f"#define THREADS {32 * BUILD_WARPS}\n#define NODES {dofs // 6 + 1}\n#define WORDS {words}\n"
        f"#define CONTACT {int(PGS_CONSTRAINT_TYPE_CONTACT)}\n#define FRICTION {int(PGS_CONSTRAINT_TYPE_FRICTION)}\n"
        f"#define PARENT(PACKED_WORD) ({parent})\n"
        + _BUILD_SOURCE.replace("SHARED_DECLARATIONS", staging.declarations()).replace(
            "#endif",
            "#undef MF_ROWS\n#undef UNITS\n#undef MIN_UNITS\n#undef COLORED\n#undef WARPS\n#undef THREADS\n#undef NODES\n"
            "#undef WORDS\n"
            "#undef CONTACT\n#undef FRICTION\n#undef PARENT\n#endif",
        )
    )
    params = kernel_template.params(
        ("mf_constraint_count", wp.array[int]),
        ("mf_contact_rows_end", wp.array[int]),
        ("mf_meta", wp.array2d[int]),
        *WORLD_FIELDS,
        *UNIT_FIELDS,
        ("unit_status", wp.array[int]),
    )
    return KernelTemplate(
        f"sweep_units_{mf_rows}_{dofs}_{int(units)}_{rule.name}_w{BUILD_WARPS}",
        params,
        cuda_native(source, preludes=(lane_group(32), SHARED_STORAGE)),
        dispatch=kernel_template.WORLDS,
        composed=True,
        native_name="sweep_units_native",
        launch_bounds=(32 * BUILD_WARPS, 1),
    ).kernel()


class ColoredUnits:
    """The units and colors of the worlds the colored sweep may take, built from the packed rows at every pack.

    Only the capable worlds (:func:`capable_worlds`) hold storage, a slot each; the others always keep the one-warp
    mapping. Every world holds its mapping, slot and counts (:data:`WORLD_FIELDS`). A slot holds
    :func:`unit_capacity` units; a world with more keeps the one-warp mapping and sets the capacity status
    (:meth:`capacity_status`).

    Args:
        world_count: Worlds of the solver.
        mf_rows: Free-body row capacity per world (it sizes a slot's units: :func:`unit_capacity`).
        dofs: Coordinates per world.
        device: The solver's CUDA device.
        worlds: The capable worlds, in world order: slot ``k`` holds world ``worlds[k]``.
        rule: Which of them take the colored mapping at a pack (:class:`ColoredRule`).
    """

    def __init__(self, world_count: int, mf_rows: int, dofs: int, device, *, worlds, rule: ColoredRule = COLORED_RULE):
        self.world_count, self.mf_rows = int(world_count), int(mf_rows)
        self.unit_capacity = unit_capacity(self.mf_rows)
        slot_world = np.asarray(worlds, dtype=np.int32)
        self.slot_count = int(slot_world.size)
        sizes = UnitPlan(self.world_count, self.slot_count, self.unit_capacity).sizes
        self.units = {
            field.name: wp.zeros(field.shape(sizes), dtype=field.dtype, device=device)
            for field in (*WORLD_FIELDS, *UNIT_FIELDS)
        }
        world_slot = np.full(self.world_count, -1, dtype=np.int32)
        world_slot[slot_world] = np.arange(self.slot_count, dtype=np.int32)
        self.units["world_slot"] = wp.array(world_slot, dtype=wp.int32, device=device)
        self.slot_world = wp.array(slot_world, dtype=wp.int32, device=device)
        self.slots = {
            field.name: wp.zeros(field.shape(sizes), dtype=field.dtype, device=device) for field in SLOT_FIELDS
        }
        # Sticky: 1 once a world's units passed its slot's (the build's device status).
        self.status = wp.zeros(1, dtype=wp.int32, device=device)
        self.rule = rule
        self.build_kernel = build_units_kernel(
            self.mf_rows, int(dofs), self.unit_capacity, rule=rule, static_bytes=static_budget(device.arch)
        )
        self.device = device

    def build(
        self,
        mf_constraint_count,
        mf_contact_rows_end,
        mf_meta,
        mf_Z_a,
        mf_Z_b,
        mf_row_mu,
        mf_row_w,
    ) -> None:
        """Build every capable world's units and colors and stage the colored worlds' rows by color slot."""
        units = self.units
        wp.launch_tiled(
            self.build_kernel,
            dim=[self.world_count],
            inputs=[mf_constraint_count, mf_contact_rows_end, mf_meta],
            outputs=[*(units[field.name] for field in (*WORLD_FIELDS, *UNIT_FIELDS)), self.status],
            block_dim=32 * BUILD_WARPS,
            device=self.device,
        )
        wp.launch(
            stage_slots,
            dim=(self.slot_count, self.unit_capacity),
            inputs=[
                self.slot_world,
                units["world_colored"],
                units["unit_count"],
                units["color_slot"],
                mf_meta,
                mf_Z_a,
                mf_Z_b,
                mf_row_mu,
                mf_row_w,
            ],
            outputs=[self.slots["slot_data"], self.slots["slot_words"]],
            device=self.device,
        )

    @property
    def world_colored(self) -> wp.array:
        """Each world's mapping (1: colored), which the one-warp sweep reads to leave those worlds."""
        return self.units["world_colored"]

    @property
    def nbytes(self) -> int:
        """Device memory of the units, colors and staged slots [B]."""
        arrays = (*self.units.values(), *self.slots.values(), self.slot_world, self.status)
        return sum(array.capacity for array in arrays)

    def capacity_status(self) -> bool:
        """Whether a world's units passed its slot's since construction: the one-warp mapping kept that world at that
        pack (a host readback)."""
        return bool(self.status.numpy()[0])

    def operands(self) -> list:
        """The colored sweep's operands (:data:`.sweep_routes.COLORED_ABI`, after the matrix-free sweep's)."""
        return [
            self.units["world_colored"],
            self.units["world_slot"],
            self.units["color_count"],
            self.units["color_offset"],
            self.units["color_slot"],
            self.slots["slot_data"],
            self.slots["slot_words"],
            self.slots["slot_metric"],
        ]

    def stats(self) -> list[dict]:
        """The last build's units, colors, largest color and mapping of each world that took units (a host
        readback)."""
        counts = self.units["unit_count"].numpy()
        colors = self.units["color_count"].numpy()
        offsets = self.units["color_offset"].numpy()
        colored = self.units["world_colored"].numpy()
        slots = self.units["world_slot"].numpy()
        worlds = []
        for world in range(self.world_count):
            if counts[world] == 0:
                continue
            slot = int(slots[world])
            sizes = offsets[slot, 1 : colors[world] + 1] - offsets[slot, : colors[world]]
            worlds.append(
                {
                    "world": world,
                    "slot": slot,
                    "units": int(counts[world]),
                    "colors": int(colors[world]),
                    "largest_color": int(sizes.max()) if colors[world] else 0,
                    "colored": bool(colored[world]),
                }
            )
        return worlds
