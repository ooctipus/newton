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
bin, 491 units, 495 colors). A world never takes more colors than it has units,
so the color data are sized from the free-body row capacity and cannot
overflow.

Everything is computed from the data in row order, one block per world: units
in row order, levels in row order, units of a color in row order. Nothing
depends on scheduling, so the colors and the sweep are deterministic whenever
the rows are.
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
from .staging import STATIC_SHARED_BYTES, Buffer, Staging, static_budget

BUILD_WARPS = 8
"""Warps of a world's unit builder block: they share the parallel passes; the first levels the units."""

UNIT_FIELDS = FieldGroup(
    Field("unit_count", wp.int32, ("world",)),
    # First row and row count of each unit, in row order.
    Field("unit_row", wp.int32, ("world", "unit")),
    Field("unit_rows", wp.int32, ("world", "unit")),
    # Packed first coordinates of the unit's bodies (``dof_a << 16 | dof_b``, as the rows' packed word).
    Field("unit_dofs", wp.int32, ("world", "unit")),
    Field("unit_color", wp.int32, ("world", "unit")),
    # The unit's place among its color's units, in row order (the builder's scratch).
    Field("unit_rank", wp.int32, ("world", "unit")),
    # Colors: count, offsets into color_slot (count + 1 of them; at most one color per unit), and the units of each
    # color in row order, each as the record the sweep reads: first row, row count and its bodies' first coordinates.
    Field("color_count", wp.int32, ("world",)),
    Field("color_offset", wp.int32, ("world", "color")),
    Field("color_slot", wp.int32, ("world", "unit", 4)),
)
"""Per-world units and colors (``unit`` capacity: the world's free-body row capacity; ``color``: one more)."""

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
    """Shapes of a world's unit data: its free-body row capacity bounds its units and its colors."""

    mf_rows: int

    @property
    def sizes(self) -> dict[str, int]:
        return {"unit": self.mf_rows, "color": self.mf_rows + 1}


_BUILD_SOURCE = r"""
    // Shared scratch (declared by build_staging): a bitmap of the body coordinates the units use and its word prefix
    // (compact body ids), per body its latest level and its owner in a leveling round, and the warps' counts.
SHARED_DECLARATIONS
    const int lane = LANE;
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int m_mf = min(mf_constraint_count.data[world], MF_ROWS);
    const int end = min(mf_contact_rows_end.data[world], m_mf);
    const int off_mf = world * MF_ROWS;
    const int off_meta = off_mf * 4;
    const int off_color = world * (MF_ROWS + 1);
    // A world below the colored sweep's contact rows keeps the one-warp sweep: no units.
    if (end < MIN_ROWS) {
        if (tid == 0) {
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
    // Each color's units in row order: a unit's slot is its color's offset plus its rank.
    for (int unit = tid; unit < units; unit += THREADS) {
        const int color = unit_color.data[off_mf + unit];
        const int slot = color_offset.data[off_color + color] + unit_rank.data[off_mf + unit];
        const int dofs = unit_dofs.data[off_mf + unit];
        const int at = (off_mf + slot) * 4;
        color_slot.data[at] = unit_row.data[off_mf + unit];
        color_slot.data[at + 1] = unit_rows.data[off_mf + unit];
        color_slot.data[at + 2] = dofs >> 16;
        color_slot.data[at + 3] = (dofs << 16) >> 16;
    }
    if (tid == 0) unit_count.data[world] = units;
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


def build_units_kernel(mf_rows: int, dofs: int, *, min_rows: int, static_bytes: int = STATIC_SHARED_BYTES) -> wp.Kernel:
    """The unit builder of worlds with ``mf_rows`` free-body rows and ``dofs`` coordinates.

    One block of :data:`BUILD_WARPS` warps per world: launch ``dim=[worlds]`` tiled with ``block_dim=32 *
    BUILD_WARPS``. Inputs: ``mf_constraint_count``, ``mf_contact_rows_end`` and ``mf_meta``; outputs:
    :data:`UNIT_FIELDS`. A world with fewer than ``min_rows`` contact rows takes no units (the one-warp sweep
    keeps it). Its shared working set (:func:`build_staging`) must fit ``static_bytes`` (a device's
    :func:`.staging.static_budget`).
    """
    staging = build_staging(dofs)
    staging.check("the unit builder", static_bytes)
    parent = _mf_parent_expression("packed", mf_rows).replace("packed", "PACKED_WORD")
    words = (dofs + 31) // 32
    source = (
        f"#define MF_ROWS {mf_rows}\n#define MIN_ROWS {int(min_rows)}\n#define WARPS {BUILD_WARPS}\n"
        f"#define THREADS {32 * BUILD_WARPS}\n#define NODES {dofs // 6 + 1}\n#define WORDS {words}\n"
        f"#define CONTACT {int(PGS_CONSTRAINT_TYPE_CONTACT)}\n#define FRICTION {int(PGS_CONSTRAINT_TYPE_FRICTION)}\n"
        f"#define PARENT(PACKED_WORD) ({parent})\n"
        + _BUILD_SOURCE.replace("SHARED_DECLARATIONS", staging.declarations()).replace(
            "#endif",
            "#undef MF_ROWS\n#undef MIN_ROWS\n#undef WARPS\n#undef THREADS\n#undef NODES\n#undef WORDS\n"
            "#undef CONTACT\n#undef FRICTION\n#undef PARENT\n#endif",
        )
    )
    params = kernel_template.params(
        ("mf_constraint_count", wp.array[int]),
        ("mf_contact_rows_end", wp.array[int]),
        ("mf_meta", wp.array2d[int]),
        *UNIT_FIELDS,
    )
    return KernelTemplate(
        f"sweep_units_{mf_rows}_{dofs}_r{int(min_rows)}_w{BUILD_WARPS}",
        params,
        cuda_native(source, preludes=(lane_group(32),)),
        dispatch=kernel_template.WORLDS,
        composed=True,
        native_name="sweep_units_native",
        launch_bounds=(32 * BUILD_WARPS, 1),
    ).kernel()


class ColoredUnits:
    """The units and colors of the worlds the colored sweep takes, built from the packed rows at every pack.

    Args:
        world_count: Worlds of the solver.
        mf_rows: Free-body row capacity per world (it bounds a world's units and colors).
        dofs: Coordinates per world.
        device: The solver's CUDA device.
        min_rows: Free-body contact rows from which a world takes the colored sweep (:func:`build_units_kernel`).
    """

    def __init__(self, world_count: int, mf_rows: int, dofs: int, device, *, min_rows: int):
        self.world_count, self.mf_rows = int(world_count), int(mf_rows)
        sizes = {"world": self.world_count, **UnitPlan(self.mf_rows).sizes}
        self.units = {
            field.name: wp.zeros(field.shape(sizes), dtype=field.dtype, device=device) for field in UNIT_FIELDS
        }
        self.slots = {
            field.name: wp.zeros(field.shape(sizes), dtype=field.dtype, device=device) for field in SLOT_FIELDS
        }
        self.build_kernel = build_units_kernel(
            self.mf_rows, int(dofs), min_rows=min_rows, static_bytes=static_budget(device.arch)
        )
        self.device = device

    def build(
        self,
        mf_constraint_count,
        mf_contact_rows_end,
        mf_meta,
        mf_J_a,
        mf_J_b,
        mf_MiJt_a,
        mf_MiJt_b,
        mf_row_mu,
        mf_row_w,
    ) -> None:
        """Build every colored world's units and colors and stage their rows by color slot."""
        units = self.units
        wp.launch_tiled(
            self.build_kernel,
            dim=[self.world_count],
            inputs=[mf_constraint_count, mf_contact_rows_end, mf_meta],
            outputs=[units[field.name] for field in UNIT_FIELDS],
            block_dim=32 * BUILD_WARPS,
            device=self.device,
        )
        wp.launch(
            stage_slots,
            dim=(self.world_count, self.mf_rows),
            inputs=[
                units["unit_count"],
                units["color_slot"],
                mf_meta,
                mf_J_a,
                mf_J_b,
                mf_MiJt_a,
                mf_MiJt_b,
                mf_row_mu,
                mf_row_w,
            ],
            outputs=[self.slots["slot_data"], self.slots["slot_words"]],
            device=self.device,
        )

    def operands(self) -> list:
        """The colored sweep's operands (:data:`.sweep_routes.COLORED_ABI`, after the matrix-free sweep's)."""
        return [
            self.units["color_count"],
            self.units["color_offset"],
            self.units["color_slot"],
            self.slots["slot_data"],
            self.slots["slot_words"],
        ]

    def stats(self) -> list[dict]:
        """The last build's units, colors and largest color of each colored world (a host readback)."""
        counts = self.units["unit_count"].numpy()
        colors = self.units["color_count"].numpy()
        offsets = self.units["color_offset"].numpy()
        worlds = []
        for world in range(self.world_count):
            if counts[world] == 0:
                continue
            sizes = offsets[world, 1 : colors[world] + 1] - offsets[world, : colors[world]]
            worlds.append(
                {
                    "world": world,
                    "units": int(counts[world]),
                    "colors": int(colors[world]),
                    "largest_color": int(sizes.max()) if colors[world] else 0,
                }
            )
        return worlds
