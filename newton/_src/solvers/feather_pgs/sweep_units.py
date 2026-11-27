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
Gauss-Seidel sweep itself. Each unit also records its turn: the colors of the
latest earlier unit on each of its bodies. The sweep runs a unit as soon as
those units ran, without waiting for whole colors: the same order on every
body, so the same arithmetic. The row order is a physics parameter (the iterate a
budget of sweeps reaches depends on it), and the colors keep it.

A body with many contacts (a tray under a coin tower) holds a unit in as many
colors as it has contacts. Its units per color are at most its free bodies, so
only the worlds with enough free bodies for the rule (:func:`capable_worlds`)
hold that storage, a slot each; the others keep the one-warp mapping and hold
none. The slots share one pool: at every pack each world's units are counted
and the worlds take their units and their colors' offsets in slot order
(:func:`place_slots`), so a world past its share (a unit is a contact's normal
row, and under friction patches most contacts carry no friction pair) keeps
its units where lighter worlds leave room. A world past the pool's end keeps
the one-warp mapping and sets the capacity status
(:meth:`ColoredUnits.capacity_status`). A world never takes more colors than
it has units, so its colors fit wherever its units do.

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

from ...core.execution import Queue, appended, fill_lanes
from ...core.memory import REFERENCE_ARCH, Buffer, Staging, arch_budget, device
from . import kernel_template, sweep_levels
from .device_preludes import SHARED_STORAGE, cuda_native, lane_group
from .field_group import Field, FieldGroup
from .kernel_template import KernelTemplate
from .row_laws import ROW_CONTACT, ROW_FRICTION, ROW_TORSION
from .row_metadata import record_parent, record_words, split_word
from .sweep_levels import COST_CLASSES

BUILD_WARPS = 8
"""Warps of a world's unit builder block: they share the parallel passes and level the units in turn."""

GROUP_WARPS = 8
"""Warps of a grouped world's block (:func:`.sweep_routes.group_rows`, the colored mapping's block of
:data:`.sweep_routes.COLORED_THREADS`): a color's groups a warp each at once."""


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

SLOT_PLACES = FieldGroup(
    # Each slot's entries of the pool at the last pack: its world's units and one more (its colors' offsets), zero for
    # a slot whose world is no candidate or takes no units; and their exclusive scan in slot order, the slot's first
    # entry (:func:`count_units`, :func:`place_slots`).
    Field("slot_need", wp.int32, ("slot",)),
    Field("slot_first", wp.int32, ("slot",)),
)
"""Where each slot's world keeps its units in the pool: chosen once a world at every pack, from its counted units."""

UNIT_FIELDS = FieldGroup(
    # First row and row count of each unit, in row order.
    Field("unit_row", wp.int32, ("pool",)),
    Field("unit_rows", wp.int32, ("pool",)),
    # Packed first coordinates of the unit's bodies (``dof_a << 16 | dof_b``, as the rows' packed word).
    Field("unit_dofs", wp.int32, ("pool",)),
    Field("unit_color", wp.int32, ("pool",)),
    # The unit's place among its color's units in row order (first its place among its island's units of its block's
    # run), and the colors of its latest earlier unit on each of its bodies (-1 for none) (the builder's scratch).
    Field("unit_rank", wp.int32, ("pool",)),
    Field("unit_after", wp.int32, ("pool", 2)),
    # Each unit's island (its root body), then where a world's islands take the sweep's group of blocks its block; each
    # run's units of each color, then the units of each (block, color) and their first places (the builder's scratch).
    Field("unit_block", wp.int32, ("pool",)),
    Field("unit_keys", wp.int32, ("pool",)),
    # Each island's units in row order, the islands in root order (the builder's scratch: the level pass's order).
    Field("unit_order", wp.int32, ("pool",)),
    # Colors: offsets into color_slot (count + 1 of them; at most one color per unit), and the units of each color in
    # row order, each as the record the sweep reads: first row, row count and its bodies' first coordinates.
    Field("color_offset", wp.int32, ("pool",)),
    Field("color_slot", wp.int32, ("pool", 4)),
    # Each unit's turn on its bodies, by place: the colors of the latest earlier unit on its first and its second body
    # (-1 for none). The sweep runs a unit once its bodies' units of those colors ran, so one past the larger is its
    # color.
    Field("color_after", wp.int32, ("pool", 2)),
)
"""The units and colors of every slot's world, in one pool: a world's units and its colors' offsets from its slot's
first entry (:data:`SLOT_PLACES`; ``pool``: :func:`pool_capacity`)."""

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
_ROW_CONTACT = wp.constant(int(ROW_CONTACT))

SLOT_FIELDS = FieldGroup(
    # A world's block of each, from its slot's first entry: field f of its unit at place p at (first * fields) + f *
    # need + p, flat (need: its entries, SLOT_PLACES), so a warp's units read each field in one transaction and a
    # world's fields lie together, as a slot's did.
    Field("slot_data", wp.float32, ("slot_field", "pool")),
    Field("slot_words", wp.int32, ("slot_word", "pool")),
)
"""A world's units' rows as the colored sweep reads them: by color slot, field by field (:func:`stage_slots`), in a
:class:`SlotLayout`, and their friction records; each world's block from its slot's first entry, its fields strided by
its entries."""


@cache
def stage_slots(weighted: bool, split: bool = False) -> wp.Kernel:
    """The slot staging kernel of the laws (``weighted``: contact rows step by their weights, a constant of the plan
    that every sweep compiles in, :class:`.sweep.RowLaws`) and of the line's record form (``split``,
    :func:`.row_metadata.split_records`: a free row's record holds its parent in a word of its last quad)."""
    free_words = record_words(True, split)
    parent_word = split_word(free_words, 1) if split else 3

    @wp.func
    def ring_link(row_record: wp.array2d[wp.int32], mf_line: int, record: int) -> int:
        # A row's parent from its record (-1 for none): a packed record's kind word's high half (0xFFFF none), a split
        # record's word of its own.
        if wp.static(split):
            return row_record[mf_line, record + wp.static(parent_word)]
        link = (row_record[mf_line, record + 3] >> 16) & 0xFFFF
        return wp.where(link == 0xFFFF, -1, link)

    @wp.func
    def stage_unit(
        p: int,
        first: int,
        data: int,
        words_at: int,
        need: int,
        mf_line: int,
        color_slot: wp.array2d[wp.int32],
        row_record: wp.array2d[wp.int32],
        dense_record_words: int,
        packed_Z: wp.array3d[float],
        mf_row_mu: wp.array2d[float],
        mf_row_w: wp.array2d[float],
        body_bases: wp.array[int],
        layout_rows: int,
        slot_data: wp.array[float],
        slot_words: wp.array[wp.int32],
    ):
        # The unit at place p of its world: its color slot at the world's first entry plus p; field f of the world's
        # block at data (words_at for the words) + f * need + p. The line's free rows from its body base, their records
        # after its dense rows'.
        body_base = body_bases[mf_line]
        free_records = body_base * dense_record_words
        row = color_slot[first + p, 0]
        rows = color_slot[first + p, 1] & 7
        for r in range(4):
            if r < rows:
                for k in range(12):
                    slot_data[data + (r * 12 + k) * need + p] = packed_Z[mf_line, body_base + row + r, k]
                record = free_records + (row + r) * wp.static(free_words)
                slot_words[words_at + (r * 3) * need + p] = row_record[mf_line, record + 1]
                slot_words[words_at + (r * 3 + 1) * need + p] = row_record[mf_line, record + 2]
                slot_words[words_at + (r * 3 + 2) * need + p] = row_record[mf_line, record + 3]
            elif r == 3 and rows == 3 and layout_rows > 3:
                # A friction contact without a spin row in a layout with spin rows: a zero row, whose terms and
                # couplings are zero (:func:`.sweep.spin_row`).
                for k in range(12):
                    slot_data[data + (r * 12 + k) * need + p] = 0.0
        entries = layout_rows * 12
        mu = float(0.0)
        if rows >= 3:
            mu = mf_row_mu[mf_line, body_base + row + 1]
        slot_data[data + (entries) * need + p] = mu
        weight = float(1.0)
        if wp.static(weighted):
            weight = mf_row_w[mf_line, body_base + row]
        slot_data[data + (entries + 1) * need + p] = weight
        if layout_rows > 3:
            # The spin row's torsional coefficient (contact torsion).
            spin_mu = float(0.0)
            if rows > 3:
                spin_mu = mf_row_mu[mf_line, body_base + row + 3]
            slot_data[data + (entries + 2) * need + p] = spin_mu
        if rows >= 3:
            # The friction pair's patch ring: its normal's ring successors in ring order (the pooled load's walk). A
            # ring the slot cannot hold (more members than it has halves, or a member past a half's 16 bits) is
            # walked by the sweep (members -1).
            words = wp.vec4i(-1, -1, -1, -1)
            members = int(0)
            hop = int(0)
            rows_cap = (row_record.shape[1] - free_records) // wp.static(free_words)
            r = ring_link(row_record, mf_line, free_records + row * wp.static(free_words))
            while r >= 0 and r != row and hop < rows_cap:
                if members == _RING_MEMBERS or r >= 0xFFFF:
                    members = -1
                    break
                # Two 16-bit rows a word, low half first; an unused half stays 0xFFFF.
                if members % 2 == 0:
                    words[members // 2] = (r & 0xFFFF) | (-65536)
                else:
                    words[members // 2] = (words[members // 2] & 0xFFFF) | (r << 16)
                members += 1
                hop += 1
                r = ring_link(row_record, mf_line, free_records + r * wp.static(free_words))
            ring = layout_rows * 3
            for k in range(_RING_MEMBERS // 2):
                slot_words[words_at + (ring + k) * need + p] = words[k]
            slot_words[words_at + (ring + _RING_MEMBERS // 2) * need + p] = members

    @wp.kernel(enable_backward=False, module="unique")
    def stage(
        mf_first: wp.array[int],
        colored_count: wp.array[int],
        colored_worlds: wp.array[int],
        world_slot: wp.array[wp.int32],
        slot_first: wp.array[wp.int32],
        slot_need: wp.array[wp.int32],
        unit_count: wp.array[wp.int32],
        color_slot: wp.array2d[wp.int32],
        row_record: wp.array2d[wp.int32],
        dense_record_words: int,
        packed_Z: wp.array3d[float],
        mf_row_mu: wp.array2d[float],
        mf_row_w: wp.array2d[float],
        body_base: wp.array[int],
        layout_rows: int,
        fields: int,
        words: int,
        stride: int,
        slot_data: wp.array[float],
        slot_words: wp.array[wp.int32],
    ):
        """Gather each colored world's unit rows into its block of the pool's :data:`SLOT_FIELDS` (flat): from its slot's
        first entry ``first``, ``fields`` (``words``) fields of its ``need`` entries, field-major, so a world's fields lie
        together as a slot's did (one thread per item of the colored bucket, ``colored_count[0]`` worlds of
        ``colored_worlds``, and unit, the launch's second dimension, ``stride`` threads, striding a world's units), in
        the :class:`SlotLayout` of units of at most ``layout_rows`` rows.

        A unit stages only what its rows hold: its pair's friction coefficient when it has a friction pair, its spin row's
        when it has one, and its normal row's weight from ``mf_row_w`` only where the laws are weighted; otherwise the
        solver binds a one-element stand-in for the weights, and the row's weight is 1.
        The coefficients, weights and packed rows are the line's, the free bodies' row ``i`` at its body base plus
        ``i``; a
        row's twelve entries in the sides layout (:func:`.sweep_routes.free_body_rows`): a capable world holds two
        free bodies, so its line holds twelve (:func:`capable_worlds`, asserted at the plan by :class:`ColoredUnits`).
        """
        item, p = wp.tid()
        if item >= colored_count[0]:
            return
        world = colored_worlds[item]
        slot = world_slot[world]
        first = slot_first[slot]
        need = slot_need[slot]
        mf_line = mf_first[world]
        units = unit_count[world]
        while p < units:
            stage_unit(
                p,
                first,
                fields * first,
                words * first,
                need,
                mf_line,
                color_slot,
                row_record,
                dense_record_words,
                packed_Z,
                mf_row_mu,
                mf_row_w,
                body_base,
                layout_rows,
                slot_data,
                slot_words,
            )
            p += stride

    return stage


UNIT_ROWS = 3
"""Free-body rows a slot's share of the pool counts per unit (:func:`unit_capacity`): a contact's normal row and its
friction pair."""


def unit_capacity(mf_rows: int) -> int:
    """A slot's share of the unit pool for worlds of ``mf_rows`` free-body rows: one unit per :data:`UNIT_ROWS` rows. A
    unit is a contact's normal row, and under friction patches only a patch's anchor carries a friction pair, so a world
    may hold up to ``mf_rows`` units: a world past its share takes the entries lighter worlds leave
    (:func:`pool_capacity`)."""
    return max(int(mf_rows) // UNIT_ROWS, 1)


def pool_capacity(slots: int, mf_rows: int) -> int:
    """The unit pool's entries: every slot's share and its colors' one more, and one world's every free row past its
    share, so any one world may make every row a unit. Each world takes its counted units and one more, in slot order
    (:func:`place_slots`); a world past the pool's end keeps the one-warp mapping and sets the capacity status."""
    share = unit_capacity(mf_rows)
    return int(slots) * (share + 1) + max(int(mf_rows) - share, 0)


@dataclasses.dataclass(frozen=True)
class UnitPlan:
    """Shapes of the unit data: every world's counts, per slot (a world that can take the colored mapping) its place in
    the pool, and the pool of every slot's units and colors (:func:`pool_capacity`), in its ``layout``."""

    worlds: int
    slots: int
    pool: int
    layout: SlotLayout

    @property
    def sizes(self) -> dict[str, int]:
        layout = self.layout
        return {
            "world": self.worlds,
            "slot": self.slots,
            "pool": self.pool,
            "slot_field": layout.fields,
            "slot_word": layout.words,
        }


def grouped_worlds(model, forest, rows: int) -> tuple[np.ndarray, ColoredRule]:
    """The worlds that can take the dense line by colors of groups (:class:`.row_slots.RowGroups`) and their rule: on
    CUDA, a world of ``rows`` dense rows a line, no free rigid body (whose resting stacks rock under first-fit colors)
    and trees with coordinates that fill the grouped block: a color holds at most a group a tree, so a world needs
    a tree for each of the block's warps (:data:`GROUP_WARPS`; with fewer, every color leaves warps idle
    and one warp walking its rows serves it) and the colored mapping's units a color (:func:`capable_worlds`), the
    rule scaled by the waves of grouped blocks, one a multiprocessor (a block of :data:`.sweep_routes.COLORED_THREADS`
    at every register a thread may hold)."""
    rule = COLORED_RULE
    if not model.device.is_cuda:
        return np.zeros(0, dtype=np.int32), rule
    worlds = max(int(forest.world_count), 1)
    trees = np.bincount(forest.articulation_world[forest.response_dof_count > 0], minlength=worlds)[:worlds]
    free = np.asarray(forest.free_rigid_per_world)[:worlds]
    trees = np.where((free > 0) | (trees < GROUP_WARPS), 0, trees)
    capable = capable_worlds(trees, rows, rule)
    if capable.size > 0:
        rule = rule.at_scale(capable.size, device(model.device).sms)
        capable = capable_worlds(trees, rows, rule)
    return capable, rule


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
    // Shared scratch (declared by build_staging): the warps' counts and the level pass's turn word.
SHARED_DECLARATIONS
    const int lane = LANE;
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int m_mf = min(mf_constraint_count.data[world], MF_ROWS);
    const int end = min(mf_contact_rows_end.data[world], m_mf);
    const int mf_line = mf_first.data[world];
    __builtin_assume(mf_line >= 0);
    // The free bodies' records (row_metadata.pack_row_record), after the line's dense rows' (from its body base).
    const int off_meta = mf_line * (row_record.strides[0] / (int)sizeof(*row_record.data))
        + body_base.data[mf_line] * dense_record_words;
    // The world's slot (a candidate's: the claim stage's) places its units and its colors' offsets in the pool, from
    // the slot's first entry (place_slots, from its counted units); a world with fewer contact rows than the rule's
    // units keeps the one-warp sweep: no units, the general bucket.
    const int slot = world_slot.data[world];
    const int off_mf = slot_first.data[slot];
    const int off_color = off_mf;
    // The pool's entries from the world's first: its units and its colors' offsets, one more.
    const int room = unit_row.shape[0] - off_mf - 1;
    if (pass_words.data[0] != 0) {
        // A later step of the collision pass: the world's units and colors are the pass's (its rows' slots are), and
        // only its claim is the step's, as the build below claims it.
        if (rank == 0 && tid == 0) {
            const int units = unit_count.data[world];
            APPEND_CLAIM(end >= MIN_UNITS && units <= room && COLORED(units, color_count.data[world]) ? colored_bucket : GENERAL_CLAIM)
        }
        return;
    }
    if (end < MIN_UNITS) {
        if (rank == 0 && tid == 0) {
            unit_count.data[world] = 0;
            color_count.data[world] = 0;
            APPEND_CLAIM(GENERAL_CLAIM)
        }
        return;
    }

    // The world's group: as many of its blocks as its rows give a thread each, at most its ``group`` (``spread``),
    // each a run of the rows (the others return). They count their units, take their places in row order, then find
    // their islands, level them and place them. They meet through counts (``unit_meets``: the counts in and every
    // later pass, the units written, the units placed; then each block's count or share of an ordered pass, and its
    // largest value).
    const int spread = min(group, max((end + THREADS - 1) / THREADS, 1));
    if (rank >= spread) return;
    int* const meets = unit_meets.data + (size_t)slot * unit_meets.shape[1];
    const auto MEET = [&](const int word, const int count) {
        __syncthreads();
        if (tid == 0) {
            __threadfence();
            atomicAdd(&meets[word], 1);
            const volatile int* const met = &meets[word];
            while (*met < count) { }
            __threadfence();
        }
        __syncthreads();
    };
    const int share = (end + spread - 1) / spread;
    const int lo = min(rank * share, end), hi = min(lo + share, end);
    int units = 0, all = 0;
    if (spread > 1) {
        // Each block's units, then the units before its run.
        int counted = 0;
        for (int row = lo + tid; row < hi; row += THREADS)
            counted += (row_record.data[off_meta + row * FREE_WORDS + 3] & 0xFFFF) == CONTACT;
        counted = __reduce_add_sync(0xffffffffu, counted);
        if (lane == 0) warp_count[warp] = counted;
        __syncthreads();
        if (tid == 0) {
            int total = 0;
            for (int w = 0; w < WARPS; ++w) total += warp_count[w];
            meets[4 + rank] = total;
        }
        MEET(0, spread);
        int before = 0, total = 0;
        for (int r = lane; r < spread; r += 32) {
            const int count = reinterpret_cast<volatile int*>(meets)[4 + r];
            total += count;
            if (r < rank) before += count;
        }
        units = __reduce_add_sync(0xffffffffu, before);
        all = __reduce_add_sync(0xffffffffu, total);
        // A world past the pool's room: none written (the world's block records it below).
        if (all > room) {
            if (rank > 0) return;
            units = all;
        }
    }
    __syncthreads();

    // Units in row order: a contact normal row and the friction rows after it whose parent it is. Each warp takes
    // 32 rows of a chunk; the warps' counts give each its first slot.
    if (units <= room) {
    for (int base = lo; base < hi; base += THREADS) {
        const int row = base + tid;
        bool head = false;
        if (row < hi) head = (row_record.data[off_meta + row * FREE_WORDS + 3] & 0xFFFF) == CONTACT;
        int counted = 0;
        const int order = ORDERED(counted, head);
        if (lane == 0) warp_count[warp] = counted;
        __syncthreads();
        int first = units, total = 0;
        for (int w = 0; w < WARPS; ++w) {
            if (w < warp) first += warp_count[w];
            total += warp_count[w];
        }
        // The pool holds the world's units within its room: a world with more counts them and keeps the one-warp
        // sweep.
        if (head && first + order < room) {
            const int unit = first + order;
            int rows = 1;
            while (row + rows < end && rows < 3) {
                const int packed = row_record.data[off_meta + (row + rows) * FREE_WORDS + 3];
                if ((packed & 0xFFFF) != FRICTION || ROW_PARENT(row + rows, packed) != row) break;
                ++rows;
            }
            const int dofs = ROW_DOFS(row);
            unit_row.data[off_mf + unit] = row;
            unit_rows.data[off_mf + unit] = rows;
            unit_dofs.data[off_mf + unit] = dofs;
        }
        units += total;
        __syncthreads();
    }
    }
    // The world's units once every block wrote its run (the others returned past the pool's room).
    if (spread > 1) {
        if (all <= room) MEET(1, spread);
        units = all;
    }
    if (rank == 0 && units > room) {
        // Past the pool's end: the one-warp sweep keeps the world, and the units' capacity status records it once.
        if (tid == 0) {
            unit_count.data[world] = units;
            color_count.data[world] = 0;
            APPEND_CLAIM(GENERAL_CLAIM)
            if (atomicOr(&unit_status.data[0], 1) == 0)
                printf("Warning: FeatherPGS colored sweep: world %d has %d contact units, past the unit pool's %d "
                       "entries from its first; the one-warp mapping keeps it.\n", world, units, max(room, 0));
            for (int w = 0; w < 4 + spread; ++w) meets[w] = 0;
        }
        return;
    }
    // The passes over the units take as many of the group's blocks as the units give a thread each (``passes``; the
    // others count in and return): a block more pays the passes' meets. They meet at word 0 (which the units' count
    // met at once where the group is more than one block); a world of one block meets at its barrier.
    const int passes = min(spread, max((units + THREADS - 1) / THREADS, 1));
    if (rank >= passes) {
        __syncthreads();
        if (tid == 0) {
            __threadfence();
            atomicAdd(&meets[1], 1);
        }
        return;
    }
    int met = spread > 1 ? spread : 0;
    const auto GROUP_MEET = [&]() {
        if (passes > 1) MEET(0, met += passes);
        else __syncthreads();
    };
    const int first = rank * THREADS + tid, stride = passes * THREADS;
    // An exclusive scan in place of ``n`` values over the group, each block a run of them (its largest value too).
    const auto GROUP_SCAN = [&](int* const values, const int n) -> int {
        const int share = (n + passes - 1) / passes;
        const int lo = min(rank * share, n), hi = min(lo + share, n);
        const int chunk = (hi - lo + THREADS - 1) / THREADS;
        const int a = min(lo + tid * chunk, hi), b = min(a + chunk, hi);
        int sum = 0, most = 0;
        for (int k = a; k < b; ++k) {
            sum += values[k];
            most = max(most, values[k]);
        }
        const int below = SCAN(sum);
        MAXIMUM(most);
        if (lane == 31) warp_count[warp] = below + sum;
        if (lane == 0) warp_count[WARPS + warp] = most;
        __syncthreads();
        int carry = below, total = 0;
        most = 0;
        for (int w = 0; w < WARPS; ++w) {
            if (w < warp) carry += warp_count[w];
            total += warp_count[w];
            most = max(most, warp_count[WARPS + w]);
        }
        if (passes > 1) {
            if (tid == 0) {
                meets[4 + rank] = total;
                meets[4 + passes + rank] = most;
            }
            GROUP_MEET();
            int before = 0;
            most = 0;
            for (int r = lane; r < passes; r += 32) {
                const volatile int* const shares = meets;
                if (r < rank) before += shares[4 + r];
                most = max(most, shares[4 + passes + r]);
            }
            carry += __reduce_add_sync(0xffffffffu, before);
            most = __reduce_max_sync(0xffffffffu, most);
        }
        for (int k = a; k < b; ++k) {
            const int value = values[k];
            values[k] = carry;
            carry += value;
        }
        GROUP_MEET();
        return most;
    };
    // Each of a run's items ``lo`` to ``hi`` (in order) its rank among the run's items of its key, by the block's
    // warps in turn, a batch of 32 a turn (a warp finds its batch's peers, then in its turn reads the key's count so
    // far and leaves it with its batch's added): ``ranks[item]``, and the run's count of each key in ``counts`` (zero
    // before). The turns are the level pass's: named barriers of two warps, the counts relaxed at the block's scope.
    const auto RUN_RANKS = [&](const int lo, const int hi, const auto& key_of, int* const counts, int* const ranks) {
        const int batches = (hi - lo + 31) / 32;
        for (int g = warp; g < batches; g += WARPS) {
            const int item = lo + g * 32 + lane;
            const bool valid = item < hi;
            const int key = valid ? key_of(item) : -1;
            const unsigned peers = __match_any_sync(0xffffffffu, key);
            if (g > 0) asm volatile("bar.sync %0, 64;" ::"r"(1 + warp) : "memory");
            int held = 0;
            if (valid) asm volatile("ld.relaxed.cta.s32 %0, [%1];" : "=r"(held) : "l"(&counts[key]) : "memory");
            __syncwarp();
            if (valid && (peers >> lane) == 1u)
                asm volatile("st.relaxed.cta.s32 [%0], %1;" ::"l"(&counts[key]), "r"(held + __popc(peers)) : "memory");
            if (valid) ranks[item] = held + __popc(peers & ((1u << lane) - 1u));
            __threadfence_block();
            if (g + 1 < batches) asm volatile("bar.arrive %0, 64;" ::"r"(1 + (warp + 1) % WARPS) : "memory");
        }
        __syncthreads();
    };
    // The world's bodies by their first coordinate's slot (dof / 6): their island parents and sizes, then the islands'
    // first places; their latest levels; and each block's count of each island's units in its run (zero between
    // builds).
    int* const parent = unit_nodes.data + (size_t)slot * unit_nodes.shape[1];
    int* const sizes = parent + NODES;
    int* const node_level = sizes + NODES;
    int* const shares = node_level + NODES;
    int* const keys_at = unit_keys.data + off_mf;
    for (int n = first; n < NODES; n += stride) {
        parent[n] = n;
        sizes[n] = 0;
        node_level[n] = -1;
    }
    GROUP_MEET();
    // Islands: the units' bodies joined by union-find (the lowest root wins whatever order the unions run in), each
    // unit's island its root, each island's units.
    const auto FIND = [&](int x) -> int {
        volatile int* const up = parent;
        for (;;) {
            const int p = up[x];
            if (p == x) return x;
            const int g = up[p];
            if (g != p) up[x] = g;  // halving: a node only ever moves closer to its root
            x = g;
        }
    };
    for (int unit = first; unit < units; unit += stride) {
        const int dofs = unit_dofs.data[off_mf + unit];
        const int dof_a = dofs >> 16, dof_b = (dofs << 16) >> 16;
        if (dof_a < 0 || dof_b < 0 || dof_a == dof_b) continue;
        int a = dof_a / 6, b = dof_b / 6;
        for (;;) {
            a = FIND(a);
            b = FIND(b);
            if (a == b) break;
            if (a > b) { const int t = a; a = b; b = t; }
            if (atomicCAS(&parent[b], b, a) == b) break;
        }
    }
    GROUP_MEET();
    for (int unit = first; unit < units; unit += stride) {
        const int dofs = unit_dofs.data[off_mf + unit];
        const int dof_a = dofs >> 16, dof_b = (dofs << 16) >> 16;
        const int root = FIND((dof_a >= 0 ? dof_a : dof_b) / 6);
        unit_block.data[off_mf + unit] = root;
        atomicAdd(&sizes[root], 1);
    }
    GROUP_MEET();
    // The islands' first places in root order (the islands' order); the largest island's units.
    const int largest = GROUP_SCAN(sizes, NODES);
    const auto ISLAND_UNITS = [&](const int n) -> int { return (n + 1 < NODES ? sizes[n + 1] : units) - sizes[n]; };
    // Where one block takes the passes (``row_order``) it levels every unit in row order, counting each color's units
    // and ranking each unit among them as it goes (the islands share no body: any order of whole islands gives the same
    // levels). Over more blocks each island's units in row order (``unit_order``): each block ranks its run's units in
    // their islands (its counts its row of ``shares``); a unit's place is its island's first, then its island's units
    // in the runs before, then its rank. Each block levels the islands whose first places fall in its share of the
    // places (whole islands, balanced by units): the first and the end of its places (``meets``, the least first place
    // at or past it).
    const bool row_order = passes == 1;
    int span_lo = 0, span_hi = units;
    if (row_order) {
        for (int c = tid; c <= units; c += THREADS) color_offset.data[off_color + c] = 0;
        __syncthreads();
    } else {
        if (tid == 0) meets[4 + rank] = units;
        const int unit_lo = (int)((long long)rank * units / passes), unit_hi = (int)((long long)(rank + 1) * units / passes);
        RUN_RANKS(unit_lo, unit_hi, [&](const int unit) { return unit_block.data[off_mf + unit]; }, shares + rank * NODES,
                  unit_rank.data + off_mf);
        GROUP_MEET();
        for (int n = first; n < NODES; n += stride) {
            if (ISLAND_UNITS(n) == 0) continue;
            int before = 0;
            for (int r = 0; r < passes; ++r) {
                const int count = shares[r * NODES + n];
                shares[r * NODES + n] = before;
                before += count;
            }
            if (passes > 1) atomicMin(&meets[4 + (int)((long long)sizes[n] * passes / units)], sizes[n]);
        }
        GROUP_MEET();
        for (int unit = unit_lo + tid; unit < unit_hi; unit += THREADS) {
            const int root = unit_block.data[off_mf + unit];
            unit_order.data[off_mf + sizes[root] + shares[rank * NODES + root] + unit_rank.data[off_mf + unit]] = unit;
        }
        {
            int from = units, to = units;
            for (int r = lane; r < passes; r += 32) {
                const int at = reinterpret_cast<volatile int*>(meets)[4 + r];
                if (r >= rank) from = min(from, at);
                if (r > rank) to = min(to, at);
            }
            span_lo = __reduce_min_sync(0xffffffffu, from);
            span_hi = __reduce_min_sync(0xffffffffu, to);
        }
        GROUP_MEET();
        for (int n = first; n < NODES; n += stride)
            if (ISLAND_UNITS(n) != 0)
                for (int r = 0; r < passes; ++r) shares[r * NODES + n] = 0;
    }
    // Levels by islands (the level pass on the block's places, its warps in turn): a unit's color is one past the
    // latest color of its bodies, so units on a body keep their row order; islands share no body, so each island's
    // levels are the world's.
    const int span = span_hi - span_lo;
    int colors = 0;
    {
LEVEL_PASS
        if (tid == 0) meets[4 + passes + rank] = levels;
        colors = levels;
    }
    GROUP_MEET();
    if (passes > 1) {
        int most = 0;
        for (int r = lane; r < passes; r += 32) most = max(most, reinterpret_cast<volatile int*>(meets)[4 + passes + r]);
        colors = __reduce_max_sync(0xffffffffu, most);
    }
    if (rank == 0 && tid == 0) color_count.data[world] = colors;
    // Over more blocks, each unit's rank among its color's units in row order, as the islands' (``keys_at``, a row of
    // the colors a run, as many runs as the world's units hold rows of its colors); then the colors' first places.
    const int runs = row_order ? 0 : min(passes, units / max(colors, 1));
    if (!row_order) {
    for (int k = first; k < runs * colors; k += stride) keys_at[k] = 0;
    for (int c = first; c <= colors; c += stride) color_offset.data[off_color + c] = 0;
    GROUP_MEET();
    const int run_lo = (int)((long long)rank * units / max(runs, 1)), run_hi = (int)((long long)(rank + 1) * units / max(runs, 1));
    if (rank < runs)
        RUN_RANKS(run_lo, run_hi, [&](const int unit) { return unit_color.data[off_mf + unit]; }, keys_at + rank * colors,
                  unit_rank.data + off_mf);
    GROUP_MEET();
    for (int c = first; c < colors; c += stride) {
        int before = 0;
        for (int r = 0; r < runs; ++r) {
            const int count = keys_at[r * colors + c];
            keys_at[r * colors + c] = before;
            before += count;
        }
        color_offset.data[off_color + c] = before;
    }
    GROUP_MEET();
    if (rank < runs)
        for (int unit = run_lo + tid; unit < run_hi; unit += THREADS)
            unit_rank.data[off_mf + unit] += keys_at[rank * colors + unit_color.data[off_mf + unit]];
    }
    // Color offsets: an exclusive scan of the counts, in place; offset ``colors`` is the unit count.
    GROUP_SCAN(color_offset.data + off_color, colors + 1);
    const int world_colors = colors;
    // Islands to blocks, where the sweep takes a group of ``sweep_blocks`` blocks a world: where every island's units
    // fit a block's threads at each of the world's colors, each island a block by its first place (whole islands,
    // balanced by units) and the units placed by (block, color), each block's first place in ``island_first`` (its
    // last word: whether the world's islands take blocks).
    bool by_island = false;
    if (sweep_blocks > 1) {
        const int keys = sweep_blocks * world_colors;
        by_island = largest <= THREADS * world_colors && keys <= units;
        if (by_island) {
            for (int k = first; k < keys; k += stride) keys_at[k] = 0;
            GROUP_MEET();
            for (int unit = first; unit < units; unit += stride) {
                const int block = (int)((long long)sizes[unit_block.data[off_mf + unit]] * sweep_blocks / units);
                unit_block.data[off_mf + unit] = block;
                atomicAdd(&keys_at[block * world_colors + unit_color.data[off_mf + unit]], 1);
            }
            GROUP_MEET();
            // The (block, color) keys' first places; each block's first.
            GROUP_SCAN(keys_at, keys);
            int* const firsts = island_first.data + (size_t)world * island_first.shape[1];
            for (int b = first; b < sweep_blocks; b += stride) firsts[b] = keys_at[b * world_colors];
            if (rank == 0 && tid == 0) firsts[sweep_blocks] = units;
            // Every block's first read before a ticket moves it.
            GROUP_MEET();
        }
        if (rank == 0 && tid == 0) island_first.data[(size_t)world * island_first.shape[1] + sweep_blocks + 1] = by_island;
    }
    // Each color's units in row order: a unit's position is its color's offset plus its rank; where the world's islands
    // take blocks, its (block, color)'s first place and a ticket (units of one color share no body: their order is
    // free).
    for (int unit = first; unit < units; unit += stride) {
        const int color = unit_color.data[off_mf + unit];
        const int position = by_island
            ? atomicAdd(&keys_at[unit_block.data[off_mf + unit] * world_colors + color], 1)
            : color_offset.data[off_color + color] + unit_rank.data[off_mf + unit];
        const int dofs = unit_dofs.data[off_mf + unit];
        const int at = (off_mf + position) * 4;
        color_slot.data[at] = unit_row.data[off_mf + unit];
        color_slot.data[at + 1] = unit_rows.data[off_mf + unit];
        color_slot.data[at + 2] = dofs >> 16;
        color_slot.data[at + 3] = (dofs << 16) >> 16;
        color_after.data[(off_mf + position) * 2] = unit_after.data[(off_mf + unit) * 2];
        color_after.data[(off_mf + position) * 2 + 1] = unit_after.data[(off_mf + unit) * 2 + 1];
    }
    if (spread > 1) {
        // The others count in and return; the world's block waits for them and clears the meets.
        if (rank > 0) {
            __syncthreads();
            if (tid == 0) {
                __threadfence();
                atomicAdd(&meets[1], 1);
            }
            return;
        }
        __syncthreads();
        if (tid == 0) {
            const volatile int* const met = &meets[1];
            while (*met < 2 * spread - 1) { }
            __threadfence();
            for (int w = 0; w < 4 + spread; ++w) meets[w] = 0;
        }
    }
    if (tid == 0) {
        unit_count.data[world] = units;
        APPEND_CLAIM(COLORED(units, colors) ? colored_bucket : GENERAL_CLAIM)
    }
#endif
"""


_UNIT_RESOURCES = """            const int id = !valid ? 0 : row_order ? unit : unit_order.data[off_mf + span_lo + unit];
            const int dofs = valid ? unit_dofs.data[off_mf + id] : 0;
            const int dof_a = valid ? dofs >> 16 : -1;
            int dof_b = valid ? (dofs << 16) >> 16 : -1;
            if (dof_b == dof_a) dof_b = -1;
            const int node_a = dof_a >= 0 ? dof_a / 6 : -1;
            const int node_b = dof_b >= 0 ? dof_b / 6 : -1;"""
"""A unit's resources: the unit at its place among the block's island-ordered places, its bodies by their first
coordinates' slots."""

_LEVEL_PASS = sweep_levels.level_pass(
    units="span",
    resources=_UNIT_RESOURCES,
    counts="color_offset.data[off_color + {}]",
    store="unit_color.data[off_mf + id] = level;"
    "\n                if (row_order) unit_rank.data[off_mf + id] = rank;"
    "\n                unit_after.data[(off_mf + id) * 2] = after_a;\n                unit_after.data[(off_mf + id) * 2 + 1] = after_b;",
    after=True,
    turns="colors_s[0]",
    counted="row_order",
)


@cache
def build_staging(dofs: int) -> Staging:
    """The unit builder's shared working set for a world of ``dofs`` coordinates."""
    return Staging(
        (
            Buffer("warp_count", "int", 2 * BUILD_WARPS),
            Buffer("colors_s", "int", 1),
        ),
        threads=32 * BUILD_WARPS,
    )


_UNIT_ROWS = """            while (row + rows < end && rows < 3) {
                const int packed = row_record.data[off_meta + (row + rows) * FREE_WORDS + 3];
                if ((packed & 0xFFFF) != FRICTION || ROW_PARENT(row + rows, packed) != row) break;"""
"""The unit builder's rows of a unit after its normal: its friction pair."""

_SPIN_UNIT_ROWS = """            while (row + rows < end && rows < 4) {
                const int packed = row_record.data[off_meta + (row + rows) * FREE_WORDS + 3];
                if ((packed & 0xFFFF) != (rows < 3 ? FRICTION : TORSION) || ROW_PARENT(row + rows, packed) != row) break;"""
"""Under contact torsion: its friction pair, then its spin row."""


def _append_claim(match: re.Match) -> str:
    """``APPEND_CLAIM(bucket)`` in the builder's source: the world appended to the claims' ``bucket``, the one append's
    statements (:func:`~newton._src.core.execution.appended`) in a block of their own."""
    pad = match.group(1)
    statements = appended("claim_items", "claim_count", "claim", "claim_region", "world").strip("\n")
    inner = "\n".join(
        f"{pad}    {line.strip()}" if not line.startswith("#") else line for line in statements.splitlines()
    )
    # Only a world a row of which can act is claimed (its acts word; none: every world).
    acting = "acts.shape[0] == 0 || acts.data[world] != 0"
    return f"{pad}if ({acting}) {{\n{pad}    const int claim = {match.group(2)};\n{inner}\n{pad}}}\n"


def build_units_kernel(
    mf_rows: int,
    dofs: int,
    *,
    rule: ColoredRule = COLORED_RULE,
    device_arch=REFERENCE_ARCH,
    torsion: bool = False,
    split: bool = False,
) -> wp.Kernel:
    """The unit builder of worlds with ``mf_rows`` free-body rows and ``dofs`` coordinates, each world's units in the
    pool from its slot's first entry (:data:`SLOT_PLACES`); under contact ``torsion`` a unit's spin row follows its pair;
    the free rows' records packed or ``split`` (:func:`.row_metadata.split_records`).

    A group of ``unit_blocks`` blocks of :data:`BUILD_WARPS` warps per world of the claims' candidate bucket (a world
    with a slot, :func:`.sweep_levels.claim_worlds`; :func:`unit_blocks`): launch tiled over the bucket's capacity, the
    slots, times the group, with ``block_dim=32 * BUILD_WARPS``. A world takes as many of its group's blocks as its rows
    give a thread each: they count and write its units in row order, find their islands and place them; its first block levels them. Inputs: ``mf_constraint_count``, ``mf_contact_rows_end`` and the free bodies' records
    (``row_record`` after the line's dense rows' records, ``dense_record_words`` a row to its body base ``body_base``,
    :func:`.row_metadata.record_layout`); outputs:
    :data:`WORLD_FIELDS` (``world_slot`` is read) and its :data:`UNIT_FIELDS` in the pool, and each world appended to the
    claims' colored bucket or their general one by ``rule`` (:func:`~newton._src.core.execution.append`'s
    statements; the general one is the sweep's cost class of the world, ``world_cost``, from ``general_bucket`` on,
    :func:`.sweep_levels.cost_bucket`). A world with fewer contact rows than the rule's units takes no units, and a
    world of more units than the pool holds from its first entry sets ``unit_status``: the one-warp sweep keeps both. Its shared working set (:func:`build_staging`)
    must fit the stage budget of the architecture ``device_arch`` it is built for (:func:`.memory.arch_budget`).
    """
    staging = build_staging(dofs)
    staging.check("the unit builder", arch_budget(device_arch))
    free_words = record_words(True, split)
    if split:
        # A split record (row_metadata.pack_row_record): its first side's coordinate whole in its first word, the
        # second's and its parent in its last quad; a unit's sides take the packed halves the units hold.
        row_parent = f"row_record.data[off_meta + (ROW) * FREE_WORDS + {split_word(free_words, 1)}]"
        row_dofs = (
            f"((row_record.data[off_meta + (ROW) * FREE_WORDS] << 16)"
            f" | (row_record.data[off_meta + (ROW) * FREE_WORDS + {split_word(free_words, 0)}] & 0xFFFF))"
        )
    else:
        row_parent = f"({record_parent('PACKED_WORD', mf_rows)})"
        row_dofs = "row_record.data[off_meta + (ROW) * FREE_WORDS]"
    body = _BUILD_SOURCE
    if torsion:
        body = f"#define TORSION {int(ROW_TORSION)}\n" + body.replace(_UNIT_ROWS, _SPIN_UNIT_ROWS)
        body = body.replace("#endif", "#undef TORSION\n#endif")
    source = (
        f"#define MF_ROWS {mf_rows}\n#define FREE_WORDS {free_words}\n#define MIN_UNITS {int(rule.min_units)}\n"
        f"#define WARPS {BUILD_WARPS}\n#define COLORED(N_UNITS, N_COLORS) {rule.condition('(N_UNITS)', '(N_COLORS)')}\n"
        f"#define THREADS {32 * BUILD_WARPS}\n#define NODES {dofs // 6 + 1}\n"
        f"#define CONTACT {int(ROW_CONTACT)}\n#define FRICTION {int(ROW_FRICTION)}\n"
        f"#define ROW_PARENT(ROW, PACKED_WORD) {row_parent}\n#define ROW_DOFS(ROW) {row_dofs}\n"
        # The general sweep's cost class of the world (sweep_levels.cost_bucket), from the general one on.
        f"#define GENERAL_CLAIM (general_bucket + {COST_CLASSES - 1} - min(max(world_cost.data[world], 0), {COST_CLASSES - 1}))\n"
        + body.replace("SHARED_DECLARATIONS", staging.declarations())
        .replace("LEVEL_PASS", _LEVEL_PASS)
        .replace(
            "#endif",
            "#undef MF_ROWS\n#undef FREE_WORDS\n#undef MIN_UNITS\n#undef COLORED\n#undef WARPS\n#undef THREADS\n#undef NODES\n"
            "#undef CONTACT\n#undef FRICTION\n#undef ROW_PARENT\n#undef ROW_DOFS\n#undef GENERAL_CLAIM\n#endif",
        )
    )
    # The appends last: their statements hold a preprocessor branch of their own.
    source = re.sub(r"( *)APPEND_CLAIM\((.*)\)\n", _append_claim, source)
    params = kernel_template.params(
        ("mf_constraint_count", wp.array[int]),
        ("mf_first", wp.array[int]),
        ("mf_contact_rows_end", wp.array[int]),
        ("row_record", wp.array2d[int]),
        ("dense_record_words", int),
        ("body_base", wp.array[int]),
        *WORLD_FIELDS,
        ("slot_first", wp.array[wp.int32]),
        *UNIT_FIELDS,
        ("unit_status", wp.array[int]),
        # The claims (execution.Queue) the builder appends each world to: its colored bucket or its general one.
        ("claim_items", wp.array[int]),
        ("claim_count", wp.array[int]),
        ("claim_region", int),
        ("colored_bucket", int),
        ("general_bucket", int),
        ("world_cost", wp.array[wp.int32]),
        ("acts", wp.array[wp.int32]),
        ("pass_words", wp.array[wp.int32]),
        # A world's group of blocks (``unit_blocks`` a world): their meets, and by its bodies' first coordinates' slots
        # (``dof // 6``) their island parents, sizes and latest levels, then each block's count of each island's units.
        ("unit_meets", wp.array2d[wp.int32]),
        ("unit_nodes", wp.array2d[wp.int32]),
        # The sweep's blocks a world and, where its islands take them, each block's first place (island_first).
        ("sweep_blocks", int),
        ("island_first", wp.array2d[wp.int32]),
    )
    return KernelTemplate(
        f"sweep_units_{mf_rows}_{dofs}_{rule.name}_w{BUILD_WARPS}"
        + ("_torsion" if torsion else "")
        + ("_split" if split else ""),
        params,
        cuda_native(source, preludes=(lane_group(32), SHARED_STORAGE)),
        dispatch=kernel_template.items(
            "world", queue=("candidate_count", "candidate_stride", "candidate_worlds"), ranks="unit_blocks"
        ),
        composed=True,
        native_name="sweep_units_native",
        launch_bounds=(32 * BUILD_WARPS, 1),
        stage=staging.dynamic_bytes(device_arch),
    ).kernel()


def unit_blocks(slots: int, device) -> int:
    """Blocks a world of the unit builder (:func:`build_units_kernel`): as many as the device's multiprocessors hold
    over the slots' worlds, a block each, so a group's blocks are resident together; one on the CPU."""
    device = wp.get_device(device)
    if not device.is_cuda:
        return 1
    return max(int(device.sm_count) // max(int(slots), 1), 1)


PLACE_THREADS = 256
"""Threads of a block that counts as many of a world's rows, and of the one block that places the slots
(:func:`place_slots`)."""


@wp.kernel(enable_backward=False, module="unique")
def count_units(
    candidate_count: wp.array[int],
    candidate_worlds: wp.array[int],
    world_slot: wp.array[wp.int32],
    mf_constraint_count: wp.array[int],
    mf_first: wp.array[int],
    mf_contact_rows_end: wp.array[int],
    row_record: wp.array2d[int],
    dense_record_words: int,
    body_base: wp.array[int],
    free_words: int,
    mf_rows: int,
    min_units: int,
    chunks: int,
    pass_words: wp.array[wp.int32],
    slot_need: wp.array[wp.int32],
):
    """Each candidate world's units, as the unit builder counts them (a contact normal row among its free rows before
    their contact rows' end; none under ``min_units`` such rows), and one more: its entries of the pool, added at its
    slot (``slot_need``, zero beforehand). A candidate's ``chunks`` blocks of :data:`PLACE_THREADS` (launch tiled over
    the slots and the blocks that fill the device, :func:`.core.execution.fill_lanes`) stride over its live rows, a
    thread a row at a time, and each adds its count once, so a world of many rows counts them over the device, not on
    one block, and the launch follows the device, not the row capacity. A later step of the collision pass keeps the
    pass's counts (``pass_words[0]``, :data:`.placement.CONTACT_PASS`)."""
    item, chunk, t = wp.tid()
    if item >= candidate_count[0] or pass_words[0] != 0:
        return
    world = candidate_worlds[item]
    end = wp.min(mf_contact_rows_end[world], wp.min(mf_constraint_count[world], mf_rows))
    first = chunk * wp.block_dim()
    if end < min_units or first >= end:
        return
    head = int(0)
    row = first + t
    line = mf_first[world]
    free_records = body_base[line] * dense_record_words
    while row < end:
        if (row_record[line, free_records + row * free_words + 3] & 0xFFFF) == _ROW_CONTACT:
            head += 1
        row += chunks * wp.block_dim()
    total = wp.tile_sum(wp.tile(head))
    if t == 0:
        # The first block adds the world's colors' one more.
        wp.atomic_add(slot_need, world_slot[world], wp.int32(wp.tile_extract(total, 0) + wp.where(chunk == 0, 1, 0)))


@wp.kernel(enable_backward=False, module="unique")
def clear_need(pass_words: wp.array[wp.int32], slot_need: wp.array[wp.int32]):
    """Zero each slot's entries before a collision pass's first step counts them (:func:`count_units`); a later step
    of the pass keeps them, which the sweeps read."""
    if pass_words[0] == 0:
        slot_need[wp.tid()] = 0


@wp.kernel(enable_backward=False, module="unique")
def place_slots(
    slot_count: int, pass_words: wp.array[wp.int32], slot_need: wp.array[wp.int32], slot_first: wp.array[wp.int32]
):
    """Each slot's first entry of the pool: the exclusive scan of the slots' entries in slot order, the worlds' order
    (one block of :data:`PLACE_THREADS`, launch tiled over one item). A world thus keeps its units wherever lighter
    worlds before it leave room, and the same counts place it the same way on every run. A later step of the
    collision pass keeps the pass's places."""
    _item, t = wp.tid()
    if pass_words[0] != 0:
        return
    carry = int(0)
    chunk = int(0)
    while chunk < slot_count:
        i = chunk + t
        need = int(0)
        if i < slot_count:
            need = int(slot_need[i])
        tile = wp.tile(need)
        before = wp.untile(wp.tile_scan_exclusive(tile))
        if i < slot_count:
            slot_first[i] = wp.int32(carry + before)
        carry += wp.tile_extract(wp.tile_sum(tile), 0)
        chunk += wp.block_dim()


class ColoredUnits:
    """The units and colors of the worlds the colored sweep may take, built from the packed rows at every pack.

    Only the capable worlds (:func:`capable_worlds`) hold storage, a slot each; the others always keep the one-warp
    mapping. Every world holds its slot and counts (:data:`WORLD_FIELDS`). The slots' worlds keep their units in one
    pool of :func:`pool_capacity` entries, each from its slot's first entry (:data:`SLOT_PLACES`); a world past the
    pool's end keeps the one-warp mapping and sets the capacity status (:meth:`capacity_status`). The mapping is
    a claim: the builder walks the claims' candidate bucket (the worlds with rows and a slot,
    :func:`.sweep_levels.claim_worlds`) and appends each to the colored bucket or the general one.

    Args:
        world_count: Worlds of the solver.
        mf_rows: Free-body row capacity per world (it sizes the pool: :func:`pool_capacity`).
        dofs: Coordinates per world.
        device: The solver's CUDA device.
        worlds: The capable worlds, in world order: slot ``k`` holds world ``worlds[k]``.
        rule: Which of them take the colored mapping at a pack (:class:`ColoredRule`).
        torsion: Contact torsion: a unit's spin row follows its friction pair (:func:`slot_layout`).
        weighted: The laws' weighted rows: a unit stages its normal row's weight (:func:`stage_slots`).
        claims: The step's claims (:attr:`.sweep_plan.Sweep.claims`).
        buckets: Its candidate and colored buckets and the general sweep's first cost class.
        world_cost: The bits of each world's own cycles in the general sweep's last call (its cost class).
        support: The line's packed rows' entries (:func:`.packed_rows.support_width`): twelve or more, since a capable
            world holds two free bodies (:func:`capable_worlds`); :func:`stage_slots` reads twelve unguarded.
        group: The sweep's blocks a world.
        split: The free rows' records are split (:func:`.row_metadata.split_records`).
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
        world_cost: wp.array,
        support: int = 12,
        group: int = 1,
        split: bool = False,
    ):
        if int(support) < 12:
            raise ValueError(f"a colored world's line holds two free bodies' twelve entries, not {support}")
        self.world_count, self.mf_rows = int(world_count), int(mf_rows)
        self.claims = claims
        self.candidate, self.colored, self.general = buckets
        self.world_cost = world_cost
        self.unit_capacity = unit_capacity(self.mf_rows)
        slot_world = np.asarray(worlds, dtype=np.int32)
        self.slot_count = int(slot_world.size)
        # The count's blocks of a slot follow the device's fill, bounded by the row capacity, never sized by it: they
        # stride over the slot's live rows.
        rows = -(-max(self.mf_rows, 1) // PLACE_THREADS) * PLACE_THREADS
        self.count_chunks = fill_lanes(self.slot_count, device, least=PLACE_THREADS, most=rows) // PLACE_THREADS
        self.pool = pool_capacity(self.slot_count, self.mf_rows)
        self.layout = slot_layout(torsion)
        sizes = UnitPlan(self.world_count, self.slot_count, self.pool, self.layout).sizes
        self.units = {
            field.name: wp.zeros(field.shape(sizes), dtype=field.dtype, device=device)
            for field in (*WORLD_FIELDS, *SLOT_PLACES, *UNIT_FIELDS)
        }
        world_slot = np.full(self.world_count, -1, dtype=np.int32)
        world_slot[slot_world] = np.arange(self.slot_count, dtype=np.int32)
        self.units["world_slot"] = wp.array(world_slot, dtype=wp.int32, device=device)
        self.slots = {
            field.name: wp.zeros(field.shape(sizes), dtype=field.dtype, device=device) for field in SLOT_FIELDS
        }
        # Sticky: 1 once a world's units passed its slot's (the build's device status).
        self.status = wp.zeros(1, dtype=wp.int32, device=device)
        # Each slot's group of blocks (unit_blocks): their meets (four counts, then each block's units or share of an
        # ordered pass, then its largest value) and, by the world's bodies, their island parents, sizes and latest
        # levels, then each block's count of each island's units (zero between builds).
        self.blocks = unit_blocks(self.slot_count, device)
        self.meets = wp.zeros((self.slot_count, 4 + 2 * self.blocks), dtype=wp.int32, device=device)
        self.nodes = wp.zeros(
            (self.slot_count, (3 + self.blocks) * (int(dofs) // 6 + 1)), dtype=wp.int32, device=device
        )
        # The sweep's blocks a world (its group) and, where a world's islands take them, each block's first place and
        # whether they do (the builder's, the sweep reads it).
        self.group = int(group)
        self.island_first = wp.zeros((self.world_count, self.group + 2), dtype=wp.int32, device=device)
        self.rule = rule
        # The free rows' records, packed or split (row_metadata.split_records): the builder, the count and the stage
        # read either.
        self.free_words = record_words(True, bool(split))
        self.stage_kernel = stage_slots(bool(weighted), bool(split))
        self.build_kernel = build_units_kernel(
            self.mf_rows,
            int(dofs),
            rule=rule,
            device_arch=device.arch,
            torsion=torsion,
            split=bool(split),
        )
        self.device = device

    def build(
        self,
        mf_constraint_count,
        mf_first,
        mf_contact_rows_end,
        row_record,
        dense_record_words: int,
        packed_Z,
        mf_row_mu,
        mf_row_w,
        body_base,
        world_acts,
        pass_words,
    ) -> None:
        """Build each candidate world's units and colors, claim it for the colored mapping or the general one where a
        row of it can act (``world_acts``), and stage the colored worlds' rows by color slot (the row weights
        ``mf_row_w`` only where the laws are weighted: :func:`stage_slots`). Both launches walk their bucket on a grid
        of its capacity, the slots. The units and colors are a function of the contact set, a collision pass's: a later
        step of the pass (``pass_words[0]``, :data:`.placement.CONTACT_PASS`) keeps them and only claims and stages."""
        units, claims = self.units, self.claims
        candidates, candidate_worlds = claims.bucket(self.candidate)
        colored, colored_worlds = claims.bucket(self.colored)
        # Each candidate world's entries of the pool, then each slot's first entry (in slot order: the worlds').
        wp.launch(
            clear_need, dim=self.slot_count, inputs=[pass_words], outputs=[units["slot_need"]], device=self.device
        )
        wp.launch_tiled(
            count_units,
            dim=[self.slot_count, self.count_chunks],
            inputs=[
                candidates,
                candidate_worlds,
                units["world_slot"],
                mf_constraint_count,
                mf_first,
                mf_contact_rows_end,
                row_record,
                int(dense_record_words),
                body_base,
                self.free_words,
                self.mf_rows,
                int(self.rule.min_units),
                self.count_chunks,
                pass_words,
            ],
            outputs=[units["slot_need"]],
            block_dim=PLACE_THREADS,
            device=self.device,
        )
        wp.launch_tiled(
            place_slots,
            dim=[1],
            inputs=[self.slot_count, pass_words, units["slot_need"]],
            outputs=[units["slot_first"]],
            block_dim=PLACE_THREADS,
            device=self.device,
        )
        wp.launch_tiled(
            self.build_kernel,
            dim=[self.slot_count * self.blocks],
            inputs=[
                self.blocks,
                candidates,
                self.slot_count,
                candidate_worlds,
                mf_constraint_count,
                mf_first,
                mf_contact_rows_end,
                row_record,
                int(dense_record_words),
                body_base,
            ],
            outputs=[
                *(units[field.name] for field in WORLD_FIELDS),
                units["slot_first"],
                *(units[field.name] for field in UNIT_FIELDS),
                self.status,
                claims.items,
                claims.count,
                claims.region,
                self.colored,
                self.general,
                self.world_cost,
                world_acts,
                pass_words,
                self.meets,
                self.nodes,
                self.group,
                self.island_first,
            ],
            block_dim=32 * BUILD_WARPS,
            device=self.device,
        )
        # A thread per unit of a slot's share, striding past it: a unit's gather is a chain of loads, which a lane a
        # unit hides (2,048 lanes a slot, the device's fill, cost the clutter hardware's stage 12% at 5k units a world).
        wp.launch(
            self.stage_kernel,
            dim=(self.slot_count, self.unit_capacity),
            inputs=[
                mf_first,
                colored,
                colored_worlds,
                units["world_slot"],
                units["slot_first"],
                units["slot_need"],
                units["unit_count"],
                units["color_slot"],
                row_record,
                int(dense_record_words),
                packed_Z,
                mf_row_mu,
                mf_row_w,
                body_base,
                self.layout.rows,
                self.layout.fields,
                self.layout.words,
                self.unit_capacity,
            ],
            outputs=[self.slots["slot_data"].flatten(), self.slots["slot_words"].flatten()],
            device=self.device,
        )

    @property
    def nbytes(self) -> int:
        """Device memory of the units, colors and staged slots [B]."""
        arrays = (*self.units.values(), *self.slots.values(), self.status)
        return sum(array.capacity for array in arrays)

    def capacity_status(self) -> bool:
        """Whether a world's units passed the pool's end since construction: the one-warp mapping kept that world at
        that pack (a host readback)."""
        return bool(self.status.numpy()[0])

    def operands(self) -> dict:
        """The colored sweep's operands by name (:data:`.sweep_routes.COLORED_ABI`, after the matrix-free sweep's)."""
        names = ("world_slot", "slot_first", "slot_need", "color_count", "color_offset", "color_slot", "color_after")
        return {**{name: self.units[name] for name in names}, **self.slots, "island_first": self.island_first}

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
        firsts = self.units["slot_first"].numpy()
        colored = set(bucket(self.colored).tolist())
        worlds = []
        for world in sorted(bucket(self.candidate).tolist()):
            if counts[world] == 0:
                continue
            slot = int(slots[world])
            first = int(firsts[slot])
            sizes = offsets[first + 1 : first + colors[world] + 1] - offsets[first : first + colors[world]]
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
