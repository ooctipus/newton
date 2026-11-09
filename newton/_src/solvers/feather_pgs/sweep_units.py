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
colors as it has contacts. Its units per color are at most its free bodies, so
only the worlds with enough free bodies for the rule (:func:`capable_worlds`)
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
import re
from functools import cache
from typing import NamedTuple

import numpy as np
import warp as wp

from ...core.execution import Queue, appended
from ...core.memory import STATIC_SHARED_BYTES, Buffer, Staging, static_budget
from . import kernel_template, sweep_levels
from .device_preludes import SHARED_STORAGE, cuda_native, lane_group
from .field_group import Field, FieldGroup
from .kernel_template import KernelTemplate
from .row_laws import ROW_CONTACT, ROW_FRICTION, ROW_TORSION
from .row_metadata import HEADER_WORDS, record_parent

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
"""The crossover of the two mappings: a world takes the colored one from 32 units at 1.75 units a color, scaled with
the waves of worlds (:meth:`ColoredRule.at_scale`)."""

WORLD_FIELDS = FieldGroup(
    # The world's storage slot, -1 for a world that cannot take the colored mapping (:func:`capable_worlds`).
    Field("world_slot", wp.int32, ("world",)),
    Field("unit_count", wp.int32, ("world",)),
    Field("color_count", wp.int32, ("world",)),
)
"""Every world's slot and unit and color counts (a candidate's, as the unit builder last walked it)."""

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
    right-hand side, and its type and parent word (its header's words 1 to 3, :func:`.row_metadata.pack_row_record`) at ``r * 3``, then from
    ``ring`` a friction unit's patch ring: the other normal rows of its ring in ring order (two 16-bit rows a word,
    0xFFFF past the last), then their count (-1 for a ring larger than :data:`RING_MEMBERS`)."""

    rows: int
    fields: int
    words: int
    mu: int
    weight: int
    spin_mu: int
    ring: int


def slot_layout(torsion: bool) -> SlotLayout:
    """The slot layout of units that may have spin rows (``torsion``: :func:`.sweep.spin_row`) or have none."""
    rows = 4 if torsion else 3
    entries, ring = 12 * rows, 3 * rows
    words = ring + RING_MEMBERS // 2 + 1
    return SlotLayout(rows, entries + 2 + torsion, words, entries, entries + 1, entries + 2, ring)


_RING_MEMBERS = wp.constant(RING_MEMBERS)
_HEADER_WORDS = wp.constant(HEADER_WORDS)

SLOT_FIELDS = FieldGroup(
    # Slot-major: field f of unit p at [slot, f, p], so a warp's units read each field in one transaction.
    Field("slot_data", wp.float32, ("slot", "slot_field", "unit")),
    Field("slot_words", wp.int32, ("slot", "slot_word", "unit")),
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
        colored_count: wp.array[int],
        colored_worlds: wp.array[int],
        world_slot: wp.array[wp.int32],
        unit_count: wp.array[wp.int32],
        color_slot: wp.array3d[wp.int32],
        row_record: wp.array2d[wp.int32],
        free_records: int,
        packed_Z: wp.array3d[float],
        mf_row_mu: wp.array2d[float],
        mf_row_w: wp.array2d[float],
        body_base: int,
        layout_rows: int,
        slot_data: wp.array3d[float],
        slot_words: wp.array3d[wp.int32],
    ):
        """Gather each colored world's unit rows into its slot's slot-major :data:`SLOT_FIELDS` (one thread per item of
        the colored bucket, ``colored_count[0]`` worlds of ``colored_worlds``, and unit), in the :class:`SlotLayout` of
        units of at most ``layout_rows`` rows.

        A unit stages only what its rows hold: its pair's friction coefficient when it has a friction pair, its spin row's
        when it has one, and its normal row's weight from ``mf_row_w`` only where the laws are weighted; otherwise the
        solver binds a one-element stand-in for the weights, and the row's weight is 1.
        The coefficients, weights and packed rows are the line's, the free bodies' row ``i`` at ``body_base + i``; a
        row's twelve entries in the sides layout (:func:`.sweep_routes.free_body_rows`): a capable world holds two
        free bodies, so its line holds twelve (:func:`capable_worlds`, asserted at the plan by :class:`ColoredUnits`).
        """
        item, p = wp.tid()
        if item >= colored_count[0]:
            return
        world = colored_worlds[item]
        slot = world_slot[world]
        if p >= unit_count[world]:
            return
        mf_line = mf_first[world]
        row = color_slot[slot, p, 0]
        rows = color_slot[slot, p, 1] & 7
        for r in range(4):
            if r < rows:
                for k in range(12):
                    slot_data[slot, r * 12 + k, p] = packed_Z[mf_line, body_base + row + r, k]
                record = free_records + (row + r) * _HEADER_WORDS
                slot_words[slot, r * 3, p] = row_record[mf_line, record + 1]
                slot_words[slot, r * 3 + 1, p] = row_record[mf_line, record + 2]
                slot_words[slot, r * 3 + 2, p] = row_record[mf_line, record + 3]
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
            rows_cap = (row_record.shape[1] - free_records) // _HEADER_WORDS
            link = (row_record[mf_line, free_records + row * _HEADER_WORDS + 3] >> 16) & 0xFFFF
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
                link = (row_record[mf_line, free_records + r * _HEADER_WORDS + 3] >> 16) & 0xFFFF
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
    contact with its friction pair. A world with more keeps the one-warp mapping, and the build sets the units' capacity
    status."""
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
        }


def capable_worlds(free_bodies, mf_rows: int, rule: ColoredRule = COLORED_RULE):
    """The worlds that can take the colored mapping, from the model at construction: a world's units per color are
    at most its free bodies (each color holds a unit of the busiest body, at least ``units / bodies`` units share
    one), so it needs ``rule.units_per_color`` of them and at least two (a world of one runs a unit a color, never
    two at once), and a row capacity of ``rule.min_units`` (a unit has a row). Two free bodies hold twelve
    coordinates, so a capable world's line holds a unit's twelve entries (:class:`ColoredUnits`).

    Args:
        free_bodies: Free rigid bodies per world (array of the worlds).
        mf_rows: Free-body row capacity per world.
        rule: The colored rule.

    Returns:
        The capable worlds' indices, in world order.
    """
    if int(mf_rows) < int(rule.min_units):
        return np.zeros(0, dtype=np.int32)
    return np.flatnonzero(np.asarray(free_bodies) >= max(rule.units_per_color, 2)).astype(np.int32)


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
    // The free bodies' records (row_metadata.pack_row_record), from the line's word free_records.
    const int off_meta = mf_line * row_record.shape[1] + free_records;
    // The world's slot (a candidate's: the claim stage's) holds its units and colors; a world with fewer contact rows
    // than the rule's units keeps the one-warp sweep: no units, the general bucket.
    const int slot = world_slot.data[world];
    const int off_mf = slot * UNITS;
    const int off_color = slot * (UNITS + 1);
    if (end < MIN_UNITS) {
        if (tid == 0) {
            unit_count.data[world] = 0;
            color_count.data[world] = 0;
            APPEND_CLAIM(general_bucket)
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
        if (row < end) head = (row_record.data[off_meta + row * HEADER_WORDS + 3] & 0xFFFF) == CONTACT;
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
                const int packed = row_record.data[off_meta + (row + rows) * HEADER_WORDS + 3];
                if ((packed & 0xFFFF) != FRICTION || (PARENT(packed)) != row) break;
                ++rows;
            }
            const int dofs = row_record.data[off_meta + row * HEADER_WORDS];
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
            unit_count.data[world] = units;
            color_count.data[world] = 0;
            APPEND_CLAIM(general_bucket)
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
        APPEND_CLAIM(COLORED(units, colors) ? colored_bucket : general_bucket)
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
                const int packed = row_record.data[off_meta + (row + rows) * HEADER_WORDS + 3];
                if ((packed & 0xFFFF) != FRICTION || (PARENT(packed)) != row) break;"""
"""The unit builder's rows of a unit after its normal: its friction pair."""

_SPIN_UNIT_ROWS = """            while (row + rows < end && rows < 4) {
                const int packed = row_record.data[off_meta + (row + rows) * HEADER_WORDS + 3];
                if ((packed & 0xFFFF) != (rows < 3 ? FRICTION : TORSION) || (PARENT(packed)) != row) break;"""
"""Under contact torsion: its friction pair, then its spin row."""


def _append_claim(match: re.Match) -> str:
    """``APPEND_CLAIM(bucket)`` in the builder's source: the world appended to the claims' ``bucket``, the one append's
    statements (:func:`~newton._src.core.execution.appended`) in a block of their own."""
    pad = match.group(1)
    statements = appended("claim_items", "claim_count", "claim", "claim_region", "world").strip("\n")
    inner = "\n".join(
        f"{pad}    {line.strip()}" if not line.startswith("#") else line for line in statements.splitlines()
    )
    return f"{pad}{{\n{pad}    const int claim = {match.group(2)};\n{inner}\n{pad}}}\n"


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

    One block of :data:`BUILD_WARPS` warps per world of the claims' candidate bucket (a world with a slot,
    :func:`.sweep_levels.claim_worlds`): launch tiled over the bucket's capacity, the slots, with ``block_dim=32 *
    BUILD_WARPS``. Inputs: ``mf_constraint_count``, ``mf_contact_rows_end`` and the free bodies' records
    (``row_record`` from the line's word ``free_records``, :func:`.row_metadata.record_layout`); outputs:
    :data:`WORLD_FIELDS` (``world_slot`` is read) and its slot's :data:`UNIT_FIELDS`, and each world appended to the
    claims' colored bucket or their general one by ``rule`` (:func:`~newton._src.core.execution.append`'s
    statements). A world with fewer contact rows than the rule's units takes no units, and a world of more units than
    its slot holds sets ``unit_status``: the one-warp sweep keeps both. Its shared working set (:func:`build_staging`)
    must fit ``static_bytes`` (a device's :func:`.memory.static_budget`).
    """
    staging = build_staging(dofs)
    staging.check("the unit builder", static_bytes)
    parent = record_parent("PACKED_WORD", mf_rows)
    words = (dofs + 31) // 32
    body = _BUILD_SOURCE
    if torsion:
        body = f"#define TORSION {int(ROW_TORSION)}\n" + body.replace(_UNIT_ROWS, _SPIN_UNIT_ROWS)
        body = body.replace("#endif", "#undef TORSION\n#endif")
    source = (
        f"#define MF_ROWS {mf_rows}\n#define HEADER_WORDS {HEADER_WORDS}\n#define UNITS {int(units)}\n#define MIN_UNITS {int(rule.min_units)}\n"
        f"#define WARPS {BUILD_WARPS}\n#define COLORED(N_UNITS, N_COLORS) {rule.condition('(N_UNITS)', '(N_COLORS)')}\n"
        f"#define THREADS {32 * BUILD_WARPS}\n#define NODES {dofs // 6 + 1}\n#define WORDS {words}\n"
        f"#define CONTACT {int(ROW_CONTACT)}\n#define FRICTION {int(ROW_FRICTION)}\n"
        f"#define PARENT(PACKED_WORD) ({parent})\n"
        + body.replace("SHARED_DECLARATIONS", staging.declarations())
        .replace("LEVEL_PASS", _LEVEL_PASS)
        .replace("LEVEL_OFFSETS", _LEVEL_OFFSETS)
        .replace(
            "#endif",
            "#undef MF_ROWS\n#undef HEADER_WORDS\n#undef UNITS\n#undef MIN_UNITS\n#undef COLORED\n#undef WARPS\n#undef THREADS\n#undef NODES\n"
            "#undef WORDS\n"
            "#undef CONTACT\n#undef FRICTION\n#undef PARENT\n#endif",
        )
    )
    # The appends last: their statements hold a preprocessor branch of their own.
    source = re.sub(r"( *)APPEND_CLAIM\((.*)\)\n", _append_claim, source)
    params = kernel_template.params(
        ("mf_constraint_count", wp.array[int]),
        ("mf_first", wp.array[int]),
        ("mf_contact_rows_end", wp.array[int]),
        ("row_record", wp.array2d[int]),
        ("free_records", int),
        *WORLD_FIELDS,
        *UNIT_FIELDS,
        ("unit_status", wp.array[int]),
        # The claims (execution.Queue) the builder appends each world to: its colored bucket or its general one.
        ("claim_items", wp.array[int]),
        ("claim_count", wp.array[int]),
        ("claim_region", int),
        ("colored_bucket", int),
        ("general_bucket", int),
    )
    return KernelTemplate(
        f"sweep_units_{mf_rows}_{dofs}_{int(units)}_{rule.name}_w{BUILD_WARPS}" + ("_torsion" if torsion else ""),
        params,
        cuda_native(source, preludes=(lane_group(32), SHARED_STORAGE)),
        dispatch=kernel_template.items("world", queue=("candidate_count", "candidate_stride", "candidate_worlds")),
        composed=True,
        native_name="sweep_units_native",
        launch_bounds=(32 * BUILD_WARPS, 1),
    ).kernel()


class ColoredUnits:
    """The units and colors of the worlds the colored sweep may take, built from the packed rows at every pack.

    Only the capable worlds (:func:`capable_worlds`) hold storage, a slot each; the others always keep the one-warp
    mapping. Every world holds its slot and counts (:data:`WORLD_FIELDS`). A slot holds :func:`unit_capacity` units; a
    world with more keeps the one-warp mapping and sets the capacity status (:meth:`capacity_status`). The mapping is
    a claim: the builder walks the claims' candidate bucket (the worlds with rows and a slot,
    :func:`.sweep_levels.claim_worlds`) and appends each to the colored bucket or the general one.

    Args:
        world_count: Worlds of the solver.
        mf_rows: Free-body row capacity per world (it sizes a slot's units: :func:`unit_capacity`).
        dofs: Coordinates per world.
        device: The solver's CUDA device.
        worlds: The capable worlds, in world order: slot ``k`` holds world ``worlds[k]``.
        rule: Which of them take the colored mapping at a pack (:class:`ColoredRule`).
        torsion: Contact torsion: a unit's spin row follows its friction pair (:func:`slot_layout`).
        weighted: The laws' weighted rows: a unit stages its normal row's weight (:func:`stage_slots`).
        claims: The step's claims (:attr:`.sweep_plan.Sweep.claims`).
        buckets: Its candidate, colored and general buckets.
        support: The line's packed rows' entries (:func:`.packed_rows.support_width`): twelve or more, since a capable
            world holds two free bodies (:func:`capable_worlds`); :func:`stage_slots` reads twelve unguarded.
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
        claims: Queue,
        buckets: tuple[int, int, int],
        support: int = 12,
    ):
        if int(support) < 12:
            raise ValueError(f"a colored world's line holds two free bodies' twelve entries, not {support}")
        self.world_count, self.mf_rows = int(world_count), int(mf_rows)
        self.claims = claims
        self.candidate, self.colored, self.general = buckets
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
        row_record,
        free_records: int,
        packed_Z,
        mf_row_mu,
        mf_row_w,
        body_base: int,
    ) -> None:
        """Build each candidate world's units and colors, claim it for the colored mapping or the general one, and
        stage the colored worlds' rows by color slot (the row weights ``mf_row_w`` only where the laws are weighted:
        :func:`stage_slots`). Both launches walk their bucket on a grid of its capacity, the slots."""
        units, claims = self.units, self.claims
        candidates, candidate_worlds = claims.bucket(self.candidate)
        colored, colored_worlds = claims.bucket(self.colored)
        wp.launch_tiled(
            self.build_kernel,
            dim=[self.slot_count],
            inputs=[
                candidates,
                self.slot_count,
                candidate_worlds,
                mf_constraint_count,
                mf_first,
                mf_contact_rows_end,
                row_record,
                int(free_records),
            ],
            outputs=[
                *(units[field.name] for field in (*WORLD_FIELDS, *UNIT_FIELDS)),
                self.status,
                claims.items,
                claims.count,
                claims.region,
                self.colored,
                self.general,
            ],
            block_dim=32 * BUILD_WARPS,
            device=self.device,
        )
        wp.launch(
            self.stage_kernel,
            dim=(self.slot_count, self.unit_capacity),
            inputs=[
                mf_first,
                colored,
                colored_worlds,
                units["world_slot"],
                units["unit_count"],
                units["color_slot"],
                row_record,
                int(free_records),
                packed_Z,
                mf_row_mu,
                mf_row_w,
                int(body_base),
                self.layout.rows,
            ],
            outputs=[self.slots["slot_data"], self.slots["slot_words"]],
            device=self.device,
        )

    @property
    def nbytes(self) -> int:
        """Device memory of the units, colors and staged slots [B]."""
        arrays = (*self.units.values(), *self.slots.values(), self.status)
        return sum(array.capacity for array in arrays)

    def capacity_status(self) -> bool:
        """Whether a world's units passed its slot's since construction: the one-warp mapping kept that world at that
        pack (a host readback)."""
        return bool(self.status.numpy()[0])

    def operands(self) -> dict:
        """The colored sweep's operands by name (:data:`.sweep_routes.COLORED_ABI`, after the matrix-free sweep's)."""
        names = ("world_slot", "color_count", "color_offset", "color_slot")
        return {**{name: self.units[name] for name in names}, **self.slots}

    def stats(self) -> list[dict]:
        """The last build's units, colors, largest color and mapping of each candidate world that took units, in world
        order (a host readback)."""

        def bucket(b: int) -> np.ndarray:
            count, items = self.claims.bucket(b)
            return items.numpy()[: int(count.numpy()[0])]

        counts = self.units["unit_count"].numpy()
        colors = self.units["color_count"].numpy()
        offsets = self.units["color_offset"].numpy()
        slots = self.units["world_slot"].numpy()
        colored = set(bucket(self.colored).tolist())
        worlds = []
        for world in sorted(bucket(self.candidate).tolist()):
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
                    "colored": world in colored,
                }
            )
        return worlds
