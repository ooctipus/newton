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
from typing import NamedTuple

import numpy as np
import warp as wp

from ...core.memory import STATIC_SHARED_BYTES, Buffer, Staging, static_budget
from . import kernel_template, sweep_levels
from .device_preludes import SHARED_STORAGE, cuda_native, lane_group
from .field_group import Field, FieldGroup
from .kernel_template import KernelTemplate
from .row_kernels import PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_FRICTION, PGS_CONSTRAINT_TYPE_TORSION
from .row_metadata import _mf_parent_expression

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

RING_MEMBERS = 8
"""Other normals of a patch ring a slot stages for its friction pair's pooled load; a larger ring walks its links."""


class SlotLayout(NamedTuple):
    """A color slot's fields per unit of at most ``rows`` rows: a contact's normal row and friction pair, and in a layout
    with spin rows its spin row where it has one (:func:`slot_layout`, :func:`.sweep.spin_row`; a friction unit without
    one stages a zero fourth row).

    ``fields`` floats: each row ``r``'s twelve ``Z`` entries at ``r * 12 + side * 6 + k`` (its packed row), then the first
    tangent's friction coefficient (at ``mu``), the normal row's regularization weight (``weight``) and a spin row's
    torsional coefficient (``spin_mu``). ``words`` words: each row ``r``'s bits of its stored inverse diagonal, of its
    right-hand side, and its type and parent word (``mf_meta``'s ``y``, ``z``, ``w``) at ``r * 3``, then from
    ``ring`` a friction unit's patch ring: the other normal rows of its ring in ring order (two 16-bit rows a word,
    0xFFFF past the last), then their count (-1 for a ring larger than :data:`RING_MEMBERS`). ``metric`` floats: a
    friction unit's cached metric and Gram couplings (:func:`.sweep_routes._colored_metrics`)."""

    rows: int
    fields: int
    words: int
    metric: int
    mu: int
    weight: int
    spin_mu: int
    ring: int


def slot_layout(torsion: bool) -> SlotLayout:
    """The slot layout of units that may have spin rows (``torsion``: :func:`.sweep.spin_row`) or have none."""
    rows = 4 if torsion else 3
    entries, ring = 12 * rows, 3 * rows
    words = ring + RING_MEMBERS // 2 + 1
    # The friction metric and the normal's couplings: the pair's metric (4) and two, or under torsion eight and three.
    return SlotLayout(rows, entries + 2 + torsion, words, 11 if torsion else 6, entries, entries + 1, entries + 2, ring)


_RING_MEMBERS = wp.constant(RING_MEMBERS)

SLOT_FIELDS = FieldGroup(
    # Slot-major: field f of unit p at [slot, f, p], so a warp's units read each field in one transaction.
    Field("slot_data", wp.float32, ("slot", "slot_field", "unit")),
    Field("slot_words", wp.int32, ("slot", "slot_word", "unit")),
    # The friction metric and Gram couplings of each staged friction unit (SlotLayout.metric), formed by the
    # colored sweep once per launch.
    Field("slot_metric", wp.float32, ("slot", "slot_metric", "unit")),
)
"""A world's units' rows as the colored sweep reads them: by color slot, field by field (:func:`stage_slots`), in a
:class:`SlotLayout`."""


@cache
def stage_slots(weighted: bool) -> wp.Kernel:
    """The slot staging kernel of the laws (``weighted``: contact rows step by their weights, a constant of the plan
    that every sweep compiles in, :class:`.sweep.RowLaws`)."""

    @wp.kernel(enable_backward=False, module="unique")
    def stage(
        mf_first: wp.array[int],
        slot_world: wp.array[wp.int32],
        world_colored: wp.array[wp.int32],
        unit_count: wp.array[wp.int32],
        color_slot: wp.array3d[wp.int32],
        mf_meta: wp.array2d[wp.int32],
        mf_Z: wp.array3d[float],
        mf_row_mu: wp.array2d[float],
        mf_row_w: wp.array2d[float],
        body_base: int,
        layout_rows: int,
        slot_data: wp.array3d[float],
        slot_words: wp.array3d[wp.int32],
    ):
        """Gather each colored world's unit rows into its slot's slot-major :data:`SLOT_FIELDS` (one thread per slot and
        unit), in the :class:`SlotLayout` of units of at most ``layout_rows`` rows.

        A unit stages only what its rows hold: its pair's friction coefficient when it has a friction pair, its spin row's
        when it has one, and its normal row's weight from ``mf_row_w`` only where the laws are weighted; otherwise the
        solver binds a one-element stand-in for the weights, and the row's weight is 1.
        The coefficients and weights are the row store's, the free bodies' row ``i`` at ``body_base + i`` of the line.
        """
        slot, p = wp.tid()
        world = slot_world[slot]
        if world_colored[world] == 0 or p >= unit_count[world]:
            return
        mf_line = mf_first[world]
        row = color_slot[slot, p, 0]
        rows = color_slot[slot, p, 1] & 7
        for r in range(4):
            if r < rows:
                for k in range(12):
                    slot_data[slot, r * 12 + k, p] = mf_Z[mf_line, row + r, k]
                slot_words[slot, r * 3, p] = mf_meta[mf_line, (row + r) * 4 + 1]
                slot_words[slot, r * 3 + 1, p] = mf_meta[mf_line, (row + r) * 4 + 2]
                slot_words[slot, r * 3 + 2, p] = mf_meta[mf_line, (row + r) * 4 + 3]
            elif r == 3 and rows == 3 and layout_rows > 3:
                # A friction contact without a spin row in a layout with spin rows: a zero row, whose terms and
                # couplings are zero (:func:`.sweep.spin_row`).
                for k in range(12):
                    slot_data[slot, r * 12 + k, p] = 0.0
        entries = layout_rows * 12
        mu = float(0.0)
        if rows >= 3:
            mu = mf_row_mu[mf_line, body_base + row + 1]
        slot_data[slot, entries, p] = mu
        weight = float(1.0)
        if wp.static(weighted):
            weight = mf_row_w[mf_line, body_base + row]
        slot_data[slot, entries + 1, p] = weight
        if layout_rows > 3:
            # The spin row's torsional coefficient (contact torsion).
            spin_mu = float(0.0)
            if rows > 3:
                spin_mu = mf_row_mu[mf_line, body_base + row + 3]
            slot_data[slot, entries + 2, p] = spin_mu
        if rows >= 3:
            # The friction pair's patch ring: its normal's ring successors in ring order (the pooled load's walk).
            words = wp.vec4i(-1, -1, -1, -1)
            members = int(0)
            hop = int(0)
            rows_cap = mf_meta.shape[1] // 4
            link = (mf_meta[mf_line, row * 4 + 3] >> 16) & 0xFFFF
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
                link = (mf_meta[mf_line, r * 4 + 3] >> 16) & 0xFFFF
                r = wp.where(link == 0xFFFF, -1, link)
            ring = layout_rows * 3
            for k in range(_RING_MEMBERS // 2):
                slot_words[slot, ring + k, p] = words[k]
            slot_words[slot, ring + _RING_MEMBERS // 2, p] = members

    return stage


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
    units and colors, which the slot's unit capacity bounds (:func:`unit_capacity`), in its ``layout``."""

    worlds: int
    slots: int
    units: int
    layout: SlotLayout

    @property
    def sizes(self) -> dict[str, int]:
        layout = self.layout
        return {
            "world": self.worlds,
            "slot": self.slots,
            "unit": self.units,
            "color": self.units + 1,
            "slot_field": layout.fields,
            "slot_word": layout.words,
            "slot_metric": layout.metric,
        }


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
    const int mf_line = mf_first.data[world];
    __builtin_assume(mf_line >= 0);
    const int off_meta = mf_line * MF_ROWS * 4;
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

    // Levels in row order (the level builder's pass): a unit's color is one past the latest color of its bodies, so
    // units on a body keep their row order. The counts of colors [0, units] start at zero; a unit's rank is its
    // place among its color's units in row order.
    if (warp == 0) {
LEVEL_PASS
        if (lane == 0) {
            color_count.data[world] = levels;
            colors_s[0] = levels;
        }
    }
    __syncthreads();
    // Color offsets: an exclusive scan of the counts, in place; offset `colors` is the unit count.
    const int colors = colors_s[0];
LEVEL_OFFSETS
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


_UNIT_RESOURCES = """            const int dofs = valid ? unit_dofs.data[off_mf + unit] : 0;
            const int dof_a = valid ? dofs >> 16 : -1;
            int dof_b = valid ? (dofs << 16) >> 16 : -1;
            if (dof_b == dof_a) dof_b = -1;
            const int node_a = dof_a >= 0 ? NODE(dof_a) : -1;
            const int node_b = dof_b >= 0 ? NODE(dof_b) : -1;"""
"""A unit's resources: its bodies, by the compact ids of their first coordinates."""

_LEVEL_PASS = sweep_levels.level_pass(
    units="units",
    resources=_UNIT_RESOURCES,
    counts="color_offset.data[off_color + {}]",
    store="unit_color.data[off_mf + unit] = level;\n                unit_rank.data[off_mf + unit] = rank;",
)
_LEVEL_OFFSETS = sweep_levels.level_offsets(levels="colors", counts="color_offset.data[off_color + {}]")


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


_UNIT_ROWS = """            while (row + rows < end && rows < 3) {
                const int packed = mf_meta.data[off_meta + (row + rows) * 4 + 3];
                if ((packed & 0xFFFF) != FRICTION || (PARENT(packed)) != row) break;"""
"""The unit builder's rows of a unit after its normal: its friction pair."""

_SPIN_UNIT_ROWS = """            while (row + rows < end && rows < 4) {
                const int packed = mf_meta.data[off_meta + (row + rows) * 4 + 3];
                if ((packed & 0xFFFF) != (rows < 3 ? FRICTION : TORSION) || (PARENT(packed)) != row) break;"""
"""Under contact torsion: its friction pair, then its spin row."""


def build_units_kernel(
    mf_rows: int,
    dofs: int,
    units: int,
    *,
    rule: ColoredRule = COLORED_RULE,
    static_bytes: int = STATIC_SHARED_BYTES,
    torsion: bool = False,
) -> wp.Kernel:
    """The unit builder of worlds with ``mf_rows`` free-body rows and ``dofs`` coordinates, in slots of ``units``
    units (:func:`unit_capacity`); under contact ``torsion`` a unit's spin row follows its pair.

    One block of :data:`BUILD_WARPS` warps per world: launch ``dim=[worlds]`` tiled with ``block_dim=32 *
    BUILD_WARPS``. Inputs: ``mf_constraint_count``, ``mf_contact_rows_end`` and ``mf_meta``; outputs:
    :data:`WORLD_FIELDS` (each world's mapping by ``rule``; ``world_slot`` is read) and its slot's
    :data:`UNIT_FIELDS`. A world without a slot, or with fewer contact rows than the rule's units, takes no units, and
    a world of more units than its slot holds sets ``unit_status``: the one-warp sweep keeps both. Its shared working set (:func:`build_staging`) must fit ``static_bytes`` (a device's
    :func:`.memory.static_budget`).
    """
    staging = build_staging(dofs)
    staging.check("the unit builder", static_bytes)
    parent = _mf_parent_expression("packed", mf_rows).replace("packed", "PACKED_WORD")
    words = (dofs + 31) // 32
    body = _BUILD_SOURCE
    if torsion:
        body = f"#define TORSION {int(PGS_CONSTRAINT_TYPE_TORSION)}\n" + body.replace(_UNIT_ROWS, _SPIN_UNIT_ROWS)
        body = body.replace("#endif", "#undef TORSION\n#endif")
    source = (
        f"#define MF_ROWS {mf_rows}\n#define UNITS {int(units)}\n#define MIN_UNITS {int(rule.min_units)}\n"
        f"#define WARPS {BUILD_WARPS}\n#define COLORED(N_UNITS, N_COLORS) {rule.condition('(N_UNITS)', '(N_COLORS)')}\n"
        f"#define THREADS {32 * BUILD_WARPS}\n#define NODES {dofs // 6 + 1}\n#define WORDS {words}\n"
        f"#define CONTACT {int(PGS_CONSTRAINT_TYPE_CONTACT)}\n#define FRICTION {int(PGS_CONSTRAINT_TYPE_FRICTION)}\n"
        f"#define PARENT(PACKED_WORD) ({parent})\n"
        + body.replace("SHARED_DECLARATIONS", staging.declarations())
        .replace("LEVEL_PASS", _LEVEL_PASS)
        .replace("LEVEL_OFFSETS", _LEVEL_OFFSETS)
        .replace(
            "#endif",
            "#undef MF_ROWS\n#undef UNITS\n#undef MIN_UNITS\n#undef COLORED\n#undef WARPS\n#undef THREADS\n#undef NODES\n"
            "#undef WORDS\n"
            "#undef CONTACT\n#undef FRICTION\n#undef PARENT\n#endif",
        )
    )
    params = kernel_template.params(
        ("mf_constraint_count", wp.array[int]),
        ("mf_first", wp.array[int]),
        ("mf_contact_rows_end", wp.array[int]),
        ("mf_meta", wp.array2d[int]),
        *WORLD_FIELDS,
        *UNIT_FIELDS,
        ("unit_status", wp.array[int]),
    )
    return KernelTemplate(
        f"sweep_units_{mf_rows}_{dofs}_{int(units)}_{rule.name}_w{BUILD_WARPS}" + ("_torsion" if torsion else ""),
        params,
        cuda_native(source, preludes=(lane_group(32), SHARED_STORAGE)),
        dispatch=kernel_template.items("world"),
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
        torsion: Contact torsion: a unit's spin row follows its friction pair (:func:`slot_layout`).
        weighted: The laws' weighted rows: a unit stages its normal row's weight (:func:`stage_slots`).
    """

    def __init__(
        self,
        world_count: int,
        mf_rows: int,
        dofs: int,
        device,
        *,
        worlds,
        rule: ColoredRule = COLORED_RULE,
        torsion: bool = False,
        weighted: bool = False,
    ):
        self.world_count, self.mf_rows = int(world_count), int(mf_rows)
        self.unit_capacity = unit_capacity(self.mf_rows)
        slot_world = np.asarray(worlds, dtype=np.int32)
        self.slot_count = int(slot_world.size)
        self.layout = slot_layout(torsion)
        sizes = UnitPlan(self.world_count, self.slot_count, self.unit_capacity, self.layout).sizes
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
        self.stage_kernel = stage_slots(bool(weighted))
        self.build_kernel = build_units_kernel(
            self.mf_rows,
            int(dofs),
            self.unit_capacity,
            rule=rule,
            static_bytes=static_budget(device.arch),
            torsion=torsion,
        )
        self.device = device

    def build(
        self,
        mf_constraint_count,
        mf_first,
        mf_contact_rows_end,
        mf_meta,
        mf_Z,
        mf_row_mu,
        mf_row_w,
        body_base: int,
    ) -> None:
        """Build every capable world's units and colors and stage the colored worlds' rows by color slot (the row
        weights ``mf_row_w`` only where the laws are weighted: :func:`stage_slots`)."""
        units = self.units
        wp.launch_tiled(
            self.build_kernel,
            dim=[self.world_count],
            inputs=[mf_constraint_count, mf_first, mf_contact_rows_end, mf_meta],
            outputs=[*(units[field.name] for field in (*WORLD_FIELDS, *UNIT_FIELDS)), self.status],
            block_dim=32 * BUILD_WARPS,
            device=self.device,
        )
        wp.launch(
            self.stage_kernel,
            dim=(self.slot_count, self.unit_capacity),
            inputs=[
                mf_first,
                self.slot_world,
                units["world_colored"],
                units["unit_count"],
                units["color_slot"],
                mf_meta,
                mf_Z,
                mf_row_mu,
                mf_row_w,
                int(body_base),
                self.layout.rows,
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

    def operands(self) -> dict:
        """The colored sweep's operands by name (:data:`.sweep_routes.COLORED_ABI`, after the matrix-free sweep's)."""
        names = ("world_colored", "world_slot", "color_count", "color_offset", "color_slot")
        return {**{name: self.units[name] for name in names}, **self.slots}

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
