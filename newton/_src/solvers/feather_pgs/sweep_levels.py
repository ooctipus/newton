# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The level builder: the schedules of the row order's trace, and the work lists the sweeps run over.

A world's units (a scalar row; a contact's normal with its pair and spin row) hold resources, the trees their rows
touch; two units conflict when they share one. Every linear extension of the row order's trace (the partial order of
conflicting pairs in row order) gives the same Gauss-Seidel iterate bit for bit, and the builder writes two of its
invariants:

- a unit's *level* (Cartier-Foata), one past the latest level of an earlier unit sharing a resource: a level's units
  share none (the colored units, :mod:`.sweep_units`);
- the *components* of the conflict graph, a *chain* where each unit holds one resource (case K2's scalar contacts on
  a coordinate no coupled contact holds, whose places in their chain are their levels; the tiers' worlds, which
  share nothing).

A schedule runs components side by side and a component's levels in order, so it is the row order's trace.
:func:`level_pass` levels a world's units and :func:`level_offsets` places each level's units; the work lists are
queues (:class:`~newton._src.core.execution.Queue`): :func:`list_worlds` buckets the Jacobi tiers' worlds.

Case K2 (the sparse diagonal): the row pass marks the contacts confined to one coordinate of the diagonal articulation,
and the builder levels them over their coordinates and lists the rest in row order, which the sweep reads.
"""

from functools import cache

import warp as wp

from ...core.execution import append
from ...core.memory import Buffer, Staging
from . import kernel_template
from .contact_input import ContactInput
from .device_preludes import SHARED_STORAGE, cuda_native, lane_group
from .kernel_template import KernelTemplate, items, python_kernel
from .row_kernels import PGS_CONSTRAINT_TYPE_FRICTION, PGS_CONSTRAINT_TYPE_TORSION


def level_pass(*, units: str, resources: str, counts: str, store: str) -> str:
    """Return CUDA source that levels a world's ``units`` units in row order (Cartier-Foata), by one warp, a batch of
    32 at a time.

    A unit's ``level`` is one past the latest level of an earlier unit on either of its resources, so units that share
    one keep their row order and a level's units share none. In each round every pending unit of the batch claims its
    resources (the lowest lane wins), and a unit that owns both commits its level: no earlier pending unit shares a
    resource with it, so it sees every earlier unit's level, as a pass in row order would. After its batch a unit takes
    its ``rank``, its place among its level's units in row order: the level's units of earlier batches and the batch's
    units of the level on lower lanes.

    ``resources`` declares, for lane's ``unit`` (``valid`` when it is below ``units``; ``resources`` may clear it for
    a unit that takes no level), its resources ``node_a`` and ``node_b``: compact ids into the warp's shared
    ``node_owner`` (32 at the start) and ``node_level`` (-1, or any other value no level overwrites), or -1 for
    none. ``counts`` formats an index into the units of each level so far: zero at the start for levels ``[0,
    units]``, and the batch's counts added. ``store`` keeps a valid unit's ``level`` and ``rank``. The fragment runs on
    every lane of the warp, uses the lane group, ``lane`` and the names ``resources`` and ``store`` read, and declares
    ``levels``, the world's level count, on every lane.
    """
    return f"""        int levels = 0;
        for (int base = 0; base < {units}; base += 32) {{
            const int unit = base + lane;
            bool valid = unit < {units};
{resources}
            unsigned pending = BALLOT(valid);
            int level = -1;
            while (pending != 0u) {{
                const bool mine = ((pending >> lane) & 1u) != 0u;
                if (mine) {{
                    if (node_a >= 0) atomicMin(&node_owner[node_a], lane);
                    if (node_b >= 0) atomicMin(&node_owner[node_b], lane);
                }}
                SYNC();
                const bool ready = mine && (node_a < 0 || node_owner[node_a] == lane)
                    && (node_b < 0 || node_owner[node_b] == lane);
                if (ready) {{
                    level = 0;
                    if (node_a >= 0) level = max(level, node_level[node_a] + 1);
                    if (node_b >= 0) level = max(level, node_level[node_b] + 1);
                    if (node_a >= 0) node_level[node_a] = level;
                    if (node_b >= 0) node_level[node_b] = level;
                }}
                SYNC();
                if (mine) {{
                    if (node_a >= 0) node_owner[node_a] = 32;
                    if (node_b >= 0) node_owner[node_b] = 32;
                }}
                SYNC();
                pending &= ~BALLOT(ready);
            }}
            const unsigned same = __match_any_sync(0xffffffffu, level);
            const int below = __popc(same & ((1u << lane) - 1u));
            const int earlier = valid ? {counts.format("level")} : 0;
            SYNC();
            if (valid && below == 0) {counts.format("level")} = earlier + __popc(same);
            SYNC();
            if (valid) {{
                const int rank = earlier + below;
                {store}
            }}
            levels = max(levels, level + 1);
        }}
        MAXIMUM(levels);"""


def level_offsets(*, levels: str, counts: str) -> str:
    """Return CUDA source that turns the units of each of ``levels`` levels into each level's first place, in place:
    an exclusive scan by the block, a block of levels at a time; place ``levels`` is the unit count.

    ``counts`` formats an index into the counts (``levels + 1`` of them). The fragment runs on every thread of a block
    of ``THREADS`` threads in ``WARPS`` warps, and uses the lane group, ``tid``, ``lane``, ``warp`` and a shared
    ``warp_count`` of a word a warp.
    """
    return f"""    int carry = 0;
    for (int base = 0; base <= {levels}; base += THREADS) {{
        const int c = base + tid;
        const int count = c < {levels} ? {counts.format("c")} : 0;
        const int below = SCAN(count);
        if (lane == 31) warp_count[warp] = below + count;
        __syncthreads();
        int before = carry, total = 0;
        for (int w = 0; w < WARPS; ++w) {{
            if (w < warp) before += warp_count[w];
            total += warp_count[w];
        }}
        if (c <= {levels}) {counts.format("c")} = before + below;
        carry += total;
        __syncthreads();
    }}"""


@wp.kernel
def list_worlds(
    rows: wp.array[int],
    free_body_rows: wp.array[int],
    bound: int,
    width: int,
    region: int,
    # outputs: a Queue's (execution.Queue), a bucket a tier
    items: wp.array[int],
    count: wp.array[int],
):
    """Bucket the worlds whose rows are dense only and at most ``bound`` by their rows: bucket ``(rows - 1) // width``
    of a queue of ``region`` slots a bucket.

    The Jacobi tiers' worlds share nothing, so each is a component of its own and the order inside a bucket is free
    (:func:`~newton._src.core.execution.append`'s, an atomic a warp a bucket); tier ``t`` serves ``(t * width, (t + 1) * width]`` rows and walks bucket ``t`` (an exiting block of a
    tier's grid still paid four dependent global loads: about 55 us a launch at 16k worlds).
    """
    world = wp.tid()
    m = rows[world]
    if m <= 0 or m > bound or free_body_rows[world] != 0:
        return
    append(items, count, (m - 1) // width, region, world)


@cache
def _get_mark_independent_sparse_contact_candidates_kernel(
    sparse_response_dofs: int,
    device_arch: str,
    *,
    torsion: bool = False,
) -> "wp.Kernel":
    """Mark contact units whose response is confined to one sparse coordinate: a normal row, followed by its two
    tangent rows only when the contact owns a friction anchor, and its spin row where it has one (four rows:
    :func:`.sweep.spin_row`).
    """
    del device_arch
    full = 3 + int(torsion)
    sparse_size = int(sparse_response_dofs)
    if sparse_size <= 0:
        raise ValueError("sparse response DOFs must be positive")

    def mark_independent_sparse_contact_units_template(
        dense_first: wp.array[int],
        data: ContactInput,
        total_num_workers: int,
        sparse_row_dof: wp.array3d[int],
    ):
        worker = wp.tid()
        total_contacts = data.count[0]
        for contact in range(worker, total_contacts, total_num_workers):
            slot = data.slot[contact]
            rows = data.slots_needed[contact]
            if data.path[contact] != 0 or slot < 0 or (rows != 1 and rows != 3 and rows != full):
                continue

            art_a = data.art0[contact]
            art_b = data.art1[contact]
            has_other_response = False
            if art_a >= 0:
                response_dofs = data.response_dofs[art_a]
                has_other_response = response_dofs > 0 and response_dofs != sparse_size
            if art_b >= 0:
                response_dofs = data.response_dofs[art_b]
                has_other_response = has_other_response or (response_dofs > 0 and response_dofs != sparse_size)
            if has_other_response:
                continue

            world = data.world[contact]
            line = dense_first[world]
            coord_0 = sparse_row_dof[line, slot, 0]
            coord_1 = sparse_row_dof[line, slot, 1]
            if coord_0 >= 0 and coord_1 < 0:
                sparse_row_dof[line, slot, 1] = -2
            elif coord_1 >= 0 and coord_0 < 0:
                sparse_row_dof[line, slot, 0] = -2

    suffix = "_torsion" if torsion else ""
    name = f"mark_independent_sparse_contact_candidates_{sparse_size}_units{suffix}"
    return python_kernel(mark_independent_sparse_contact_units_template, name)


@cache
def _get_sparse_contact_levels_kernel(
    max_constraints: int,
    max_world_dofs: int,
    device_arch: str,
    *,
    build_serial_contacts: bool = False,
    torsion: bool = False,
) -> "wp.Kernel":
    """The sparse-diagonal contact schedule over the contact units (:func:`_sparse_contact_levels_kernel`)."""
    del device_arch
    return _sparse_contact_levels_kernel(int(max_constraints), int(max_world_dofs), build_serial_contacts, torsion)


def _list_serial_contacts_source(*, count: str, stride: int, head: str, serial_capacity: int) -> str:
    """Return CUDA source that lists the contacts outside every level, in row order, across the warp.

    Contacts lie ``stride`` rows apart from ``contact_start``: position ``p`` is row ``contact_start + p * stride`` for
    ``p`` below the C expression ``count``, and the C condition ``head`` on ``normal`` tells whether that row heads a
    contact. A contact head without a mark (no slot left below ``-1``) joins ``serial_normals`` in row order (a
    :class:`~newton._src.core.execution.Queue`'s bucket a world, :attr:`~newton._src.core.execution.Order.ORDERED`),
    and lane 0 writes their count.

    The fragment runs on every lane of one world's warp and uses the builder's lane group, ``lane``,
    ``world``, ``contact_start``, ``sparse_world_base``, ``sparse_row_dof``, ``serial_normals`` and
    ``serial_count_out``, and the names ``head`` reads. It declares ``serial_count`` in the scope it
    is placed in.
    """
    return f"""
    int serial_count = 0;
    for (int base = 0; base < {count}; base += 32) {{
        const int normal = contact_start + (base + lane) * {stride};
        bool serial = false;
        if (base + lane < {count}) {{
            const int sparse_row = sparse_world_base + normal * 2;
            const int coord_0 = sparse_row_dof.data[sparse_row];
            const int coord_1 = sparse_row_dof.data[sparse_row + 1];
            serial = ({head}) && !(coord_0 < -1 || coord_1 < -1);
        }}
        const int serial_slot = ORDERED(serial_count, serial);
        if (serial) serial_normals.data[world * {serial_capacity} + serial_slot] = normal;
    }}
    if (lane == 0) serial_count_out.data[world] = serial_count;
"""


@cache
def levels_staging(max_constraints: int, max_world_dofs: int) -> Staging:
    """The contact schedule's shared working set, a world's warp: its coordinates' owner and level
    (:func:`level_pass`), each contact position's packed level and rank, and the offsets' warp sum."""
    return Staging(
        (
            Buffer("node_owner", "int", max_world_dofs),
            Buffer("node_level", "int", max_world_dofs),
            Buffer("unit_place", "int", max_constraints),
            Buffer("warp_count", "int", 1),
        ),
        threads=32,
    )


def _sparse_contact_levels_kernel(M: int, D: int, build_serial_contacts: bool, torsion: bool = False) -> "wp.Kernel":
    """Build the sparse-diagonal contact schedule over the contact units: the levels of the scalar contacts and the
    serial list of the rest.

    Under patch friction a contact's rows form a unit: its normal row, followed by its two tangent rows only when it
    owns an anchor (and its spin row, where it has one, under ``torsion``), each naming the normal as parent; every row
    of the contact segment other than such a member heads a unit (the row writer's layout). A unit the marker left
    with a marked slot (below ``-1``) is a *scalar contact*, on the coordinate of its other slot.

    The resources are the world's coordinates. A coupled unit (no marked slot) reserves its coordinates, and a scalar
    contact on a reserved coordinate loses its mark (``-1``): it stays in the serial order. The others are leveled in
    row order (:func:`level_pass`, one resource each): a coordinate's ``k``-th contact is level ``k``, so a level holds
    at most one contact a coordinate, and a coordinate's contacts keep their row order. Each level's contacts are
    listed in row order (:func:`level_offsets`): ``level_count[world]`` levels, level ``l`` the heads
    ``level_units[world, level_offsets[world, l]:level_offsets[world, l + 1]]``. The serial list
    (:func:`_list_serial_contacts_source`) holds the other unit heads in row order.

    One warp a world, a block each: launch ``dim=[worlds]`` tiled with ``block_dim=32``.
    """
    if M >= 1 << 15:
        raise ValueError(f"a world's contact schedule packs a unit's level and rank in 16 bits each: {M} rows")
    friction, spin = int(PGS_CONSTRAINT_TYPE_FRICTION), int(PGS_CONSTRAINT_TYPE_TORSION)
    kind = "world_row_type.data[row_base + normal]"
    # A unit's members (a tangent, or a spin row) follow its head; every other row of the segment heads a unit.
    unit_head = f"({kind} != {friction} && {kind} != {spin})" if torsion else f"{kind} != {friction}"
    staging = levels_staging(M, D)
    staging.check("the sparse-diagonal contact schedule")
    resources = f"""            const int normal = contact_start + unit;
            int node_a = -1;
            const int node_b = -1;
            if (valid) {{
                const int sparse_row = sparse_world_base + normal * 2;
                const int coord_0 = sparse_row_dof.data[sparse_row];
                const int coord_1 = sparse_row_dof.data[sparse_row + 1];
                valid = ({unit_head}) && (coord_0 < -1 || coord_1 < -1);
                if (valid) {{
                    const int coordinate = coord_0 >= 0 ? coord_0 : coord_1;
                    if (node_level[coordinate] == RESERVED) {{
                        sparse_row_dof.data[coord_0 < -1 ? sparse_row : sparse_row + 1] = -1;
                        valid = false;
                    }} else {{
                        node_a = coordinate;
                    }}
                }}
            }}
            if (unit < units) unit_place[unit] = -1;"""
    counts = f"level_offsets.data[world * {M + 1} + {{}}]"
    level = level_pass(
        units="units",
        resources=resources,
        counts=counts,
        store="unit_place[unit] = (level << 16) | rank;",
    )
    offsets = level_offsets(levels="levels", counts=counts)
    serial = (
        _list_serial_contacts_source(count="units", stride=1, head=unit_head, serial_capacity=M)
        if build_serial_contacts
        else ""
    )
    snippet = f"""    const int lane = LANE;
    const int tid = lane, warp = 0;
    const int contact_start = dense_phase_bounds.data[world * 2 + 1];
    const int contact_end = min(world_constraint_count.data[world], {M});
    const int units = contact_end - contact_start;
    const int line = dense_first.data[world];
    __builtin_assume(line >= 0);
    const int row_base = line * {M};
    const int sparse_world_base = line * {M} * 2;
{staging.declarations("    ")}
    for (int coordinate = lane; coordinate < {D}; coordinate += 32) {{
        node_owner[coordinate] = 32;
        node_level[coordinate] = -1;
    }}
    for (int l = lane; l <= units; l += 32) {counts.format("l")} = 0;
    SYNC();

    // A coupled unit reserves every coordinate it holds: a scalar contact sharing one stays in the serial order.
    for (int normal = contact_start + lane; normal < contact_end; normal += 32) {{
        const int sparse_row = sparse_world_base + normal * 2;
        const int coord_0 = sparse_row_dof.data[sparse_row];
        const int coord_1 = sparse_row_dof.data[sparse_row + 1];
        if (!({unit_head}) || coord_0 < -1 || coord_1 < -1) continue;
        if (coord_0 >= 0) node_level[coord_0] = RESERVED;
        if (coord_1 >= 0) node_level[coord_1] = RESERVED;
    }}
    SYNC();

{level}
    SYNC();
{offsets}
    for (int unit = lane; unit < units; unit += 32) {{
        const int place = unit_place[unit];
        if (place >= 0)
            level_units.data[world * {M} + {counts.format("(place >> 16)")} + (place & 0xFFFF)] = contact_start + unit;
    }}
    if (lane == 0) level_count.data[world] = levels;
{serial}
#endif
"""
    source = "#define THREADS 32\n#define WARPS 1\n#define RESERVED (-2)\n" + snippet.replace(
        "#endif", "#undef THREADS\n#undef WARPS\n#undef RESERVED\n#endif"
    )
    name = f"sparse_contact_levels_{M}_{D}"
    if build_serial_contacts:
        name += "_serial_contacts"
    name += "_units" + ("_torsion" if torsion else "")
    return KernelTemplate(
        name,
        kernel_template.params(
            ("world_constraint_count", wp.array[int]),
            ("dense_first", wp.array[int]),
            ("dense_phase_bounds", wp.array2d[int]),
            ("world_row_type", wp.array2d[int]),
            ("sparse_row_dof", wp.array3d[int]),
            ("level_count", wp.array[int]),
            ("level_offsets", wp.array2d[int]),
            ("level_units", wp.array2d[int]),
            ("serial_count_out", wp.array[int]),
            ("serial_normals", wp.array2d[int]),
        ),
        cuda_native(source, preludes=(lane_group(32), SHARED_STORAGE)),
        dispatch=items("world"),
        composed=True,
        native_name="sparse_contact_levels_native",
    ).kernel()
