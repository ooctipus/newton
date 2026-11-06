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
  a coordinate no coupled contact holds; the tiers' worlds, which share nothing).

A schedule runs components side by side and a component's levels in order, so it is the row order's trace.
:func:`level_pass` levels a world's units and :func:`level_offsets` places each level's units; the work lists are
queues (:class:`~newton._src.core.execution.Queue`): :func:`list_worlds` buckets the Jacobi tiers' worlds.

Case K2 (the sparse diagonal): the row pass marks the contacts confined to one coordinate of the diagonal articulation,
and the builder links them into chains per coordinate and lists the rest in row order, which the sweep reads.
"""

from functools import cache

import warp as wp

from ...core.execution import append
from . import kernel_template
from .contact_input import ContactInput
from .device_preludes import lane_group
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

    ``resources`` declares, for lane's ``unit`` (``valid`` when it is below ``units``), its resources ``node_a`` and
    ``node_b``: compact ids into the warp's shared ``node_owner`` (32 at the start) and ``node_level`` (-1), or -1 for
    none. ``counts`` formats an index into the units of each level so far: zero at the start for levels ``[0,
    units]``, and the batch's counts added. ``store`` keeps a valid unit's ``level`` and ``rank``. The fragment runs on
    every lane of the warp, uses the lane group, ``lane`` and the names ``resources`` and ``store`` read, and declares
    ``levels``, the world's level count, on every lane.
    """
    return f"""        int levels = 0;
        for (int base = 0; base < {units}; base += 32) {{
            const int unit = base + lane;
            const bool valid = unit < {units};
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
def _get_build_independent_sparse_contact_groups_kernel(
    max_constraints: int,
    max_world_dofs: int,
    device_arch: str,
    *,
    build_serial_contacts: bool = False,
    torsion: bool = False,
) -> "wp.Kernel":
    """Build exact independent and serial contact schedules for sparse diagonal response, over the contact units
    (:func:`_build_independent_sparse_contact_units_kernel`)."""
    del device_arch
    return _build_independent_sparse_contact_units_kernel(
        int(max_constraints), int(max_world_dofs), build_serial_contacts, torsion
    )


def _link_scalar_contacts_source(*, count: str, stride: int, head: str, dofs: int) -> str:
    """Return CUDA source that links each coordinate's scalar contacts across a world's warp.

    Contacts lie ``stride`` rows apart from ``contact_start``: position ``p`` is row
    ``contact_start + p * stride`` for ``p`` below the C expression ``count``, and the C
    condition ``head`` on ``normal`` tells whether that row heads a contact. A scalar contact
    (one marked coordinate) links to the next scalar contact of its coordinate in row order
    (``-(next + 3)``, ``-2`` at the chain end) unless a coupled contact reserved the coordinate
    (``-1``), and ``group_heads`` ends with each chain's first contact. These are the outputs of
    prepending the contacts to their chains in reverse row order. The warp visits chunks of 32
    positions from the last: a contact links to the next lane of its coordinate, found by one
    ballot per coordinate bit, else to the coordinate's running head, and the chunk's first
    contact of the coordinate becomes that head.

    The fragment runs on every lane of one world's warp and uses the builder's lane group, ``lane``,
    ``contact_start``, ``sparse_world_base``, ``head_base``, ``sparse_row_dof`` and ``group_heads``,
    the names ``head`` reads, and ``earlier_lanes``, which must equal ``(1u << lane) - 1u``. A scalar
    contact's unmarked slot must hold a coordinate in ``[0, dofs)``, as the marker leaves it: the
    fragment reads ``group_heads`` at that coordinate and matches peers on its low
    ``(dofs - 1).bit_length()`` bits only.
    """
    coordinate_bits = max(dofs - 1, 0).bit_length()
    return f"""
    for (int chunk = ({count} + 31) / 32 - 1; chunk >= 0; --chunk) {{
        const int normal = contact_start + (chunk * 32 + lane) * {stride};
        int coordinate = -1;
        int link_row = 0;
        int running = -1;
        if (chunk * 32 + lane < {count}) {{
            const int sparse_row = sparse_world_base + normal * 2;
            const int coord_0 = sparse_row_dof.data[sparse_row];
            const int coord_1 = sparse_row_dof.data[sparse_row + 1];
            if (({head}) && (coord_0 < -1 || coord_1 < -1)) {{
                const int scalar_coordinate = coord_0 >= 0 ? coord_0 : coord_1;
                running = group_heads.data[head_base + scalar_coordinate];
                if (running != -2) {{
                    coordinate = scalar_coordinate;
                    link_row = coord_0 < 0 ? sparse_row : sparse_row + 1;
                }} else {{
                    sparse_row_dof.data[coord_0 < -1 ? sparse_row : sparse_row + 1] = -1;
                }}
            }}
        }}
        unsigned peers = BALLOT(coordinate >= 0);
        for (int bit = 0; bit < {coordinate_bits}; ++bit) {{
            const unsigned ones = BALLOT((coordinate >> bit) & 1);
            peers &= ((coordinate >> bit) & 1) ? ones : ~ones;
        }}
        if (coordinate >= 0) {{
            const unsigned later = peers & ~earlier_lanes & ~(1u << lane);
            const int next_normal = later != 0u
                ? contact_start + (chunk * 32 + FIRST(later)) * {stride} : running;
            sparse_row_dof.data[link_row] = next_normal >= 0 ? -(next_normal + 3) : -2;
        }}
        SYNC();
        if (coordinate >= 0 && (peers & earlier_lanes) == 0u)
            group_heads.data[head_base + coordinate] = normal;
        SYNC();
    }}
"""


def _list_serial_contacts_source(*, count: str, stride: int, head: str, serial_capacity: int) -> str:
    """Return CUDA source that lists the contacts outside every chain, in row order, across the warp.

    Positions and ``head`` are those of :func:`_link_scalar_contacts_source`. A contact head
    without a link (no marked coordinate left below ``-1``) joins ``serial_normals``, compacted
    by one ballot per chunk of 32 positions, and lane 0 writes their count.

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


def _build_independent_sparse_contact_units_kernel(
    M: int, D: int, build_serial_contacts: bool, torsion: bool = False
) -> "wp.Kernel":
    """Build the independent and serial schedules over patch contact units.

    Under patch friction a contact's rows form a unit: its normal row, followed by
    its two tangent rows only when it owns an anchor. Every row of the contact
    segment other than a tangent row heads a unit, and a tangent row names its
    unit's head as parent. The links and the serial list hold unit heads. Members of a patch ring share
    their body pair, hence their coordinate and independence class.

    Two serial walks over the rows define the schedules. When the rows form units
    as the producers write them, both walks visit exactly the non-friction rows,
    and the warp builds the same links, heads and serial list in parallel; lane 0
    runs the walks for any other row layout. Where spin rows may appear (``torsion``)
    a unit's spin row, where it has one, follows its tangents and names its head too.
    """
    friction, spin = int(PGS_CONSTRAINT_TYPE_FRICTION), int(PGS_CONSTRAINT_TYPE_TORSION)

    def member(row: str) -> str:
        """Whether row ``row`` belongs to the unit of an earlier head (a tangent, or a spin row)."""
        kind = f"world_row_type.data[row_base + {row}]"
        return f"({kind} == {friction} || {kind} == {spin})" if torsion else f"{kind} == {friction}"

    unit_head = f"!{member('normal')}" if torsion else f"world_row_type.data[row_base + normal] != {friction}"
    # The serial walks that define the schedules, which lane 0 runs for any other row layout.
    serial_walks = f"""
    // Prepending candidates in reverse row order makes each negative link
    // reproduce the original ascending GS order. Each step lands on the last
    // row of the previous unit, which is its head or one of its tangent rows.
    if (lane == 0) {{
        int row = contact_end - 1;
        while (row >= contact_start) {{
            int normal = row;
            if ({member("row")}) {{
                const int head = world_row_parent.data[row_base + row];
                if (head >= contact_start && head < row) normal = head;
            }}
            const int sparse_row = sparse_world_base + normal * 2;
            const int coord_0 = sparse_row_dof.data[sparse_row];
            const int coord_1 = sparse_row_dof.data[sparse_row + 1];
            const bool scalar_candidate = coord_0 < -1 || coord_1 < -1;
            const int scalar_coordinate = coord_0 >= 0 ? coord_0 : coord_1;
            if (scalar_candidate) {{
                if (group_heads.data[head_base + scalar_coordinate] != -2) {{
                    const int next_normal = group_heads.data[head_base + scalar_coordinate];
                    const int link = next_normal >= 0 ? -(next_normal + 3) : -2;
                    if (sparse_row_dof.data[sparse_row] < 0)
                        sparse_row_dof.data[sparse_row] = link;
                    else
                        sparse_row_dof.data[sparse_row + 1] = link;
                    group_heads.data[head_base + scalar_coordinate] = normal;
                }} else if (coord_0 < -1) {{
                    sparse_row_dof.data[sparse_row] = -1;
                }} else {{
                    sparse_row_dof.data[sparse_row + 1] = -1;
                }}
            }}
            row = normal - 1;
        }}
    }}
    SYNC();
"""
    serial_schedule = (
        f"""
    // Compact the remaining units in original row order. The solve may
    // inspect up to four units concurrently, but any active batch is still
    // applied through this serial list in exact GS order.
    if (lane == 0) {{
        int serial_count = 0;
        int normal = contact_start;
        while (normal < contact_end) {{
            const int sparse_row = sparse_world_base + normal * 2;
            const bool scheduled_independently = sparse_row_dof.data[sparse_row] < -1
                || sparse_row_dof.data[sparse_row + 1] < -1;
            if (!scheduled_independently)
                serial_normals.data[world * {M} + serial_count++] = normal;
            const bool owns_tangents = normal + 2 < contact_end
                && world_row_type.data[row_base + normal + 1] == {friction}
                && world_row_parent.data[row_base + normal + 1] == normal;
            // A unit's spin row, where it has one, names its head after its tangents (sweep.spin_row).
            const bool owns_spin = {"true" if torsion else "false"} && owns_tangents && normal + 3 < contact_end
                && world_row_type.data[row_base + normal + 3] == {spin}
                && world_row_parent.data[row_base + normal + 3] == normal;
            normal += owns_tangents ? (owns_spin ? 4 : 3) : 1;
        }}
        serial_count_out.data[world] = serial_count;
    }}
    SYNC();
"""
        if build_serial_contacts
        else ""
    )
    parallel_serial_schedule = (
        _list_serial_contacts_source(count="contact_end - contact_start", stride=1, head=unit_head, serial_capacity=M)
        if build_serial_contacts
        else ""
    )
    parallel_links = _link_scalar_contacts_source(count="contact_end - contact_start", stride=1, head=unit_head, dofs=D)
    # A unit's rows: its head, then nothing or its tangents (and spin row) naming it, then the next head.
    layout = "if (tangent_1 && !(named_1 && tangent_2 && named_2 && !tangent_3)) rows_form_units = false;"
    if torsion:
        layout = f"""const bool spin_3 = normal + 3 < contact_end && world_row_type.data[row_base + normal + 3] == {spin};
        const bool named_3 = normal + 3 < contact_end && world_row_parent.data[row_base + normal + 3] == normal;
        const bool member_4 = normal + 4 < contact_end && {member("normal + 4")};
        const bool member_3 = normal + 3 < contact_end && {member("normal + 3")};
        const bool whole = spin_3 && named_3 ? !member_4 : !member_3;
        if (tangent_1 && !(named_1 && tangent_2 && named_2 && whole)) rows_form_units = false;"""

    snippet = f"""    const int lane = LANE;
    const int contact_start = dense_phase_bounds.data[world * 2 + 1];
    const int contact_end = world_constraint_count.data[world];
    const int head_base = world * {D};
    const int line = dense_first.data[world];
    __builtin_assume(line >= 0);
    const int row_base = line * {M};
    const int sparse_world_base = line * {M} * 2;
    const unsigned earlier_lanes = (1u << lane) - 1u;
    for (int coordinate = lane; coordinate < {D}; coordinate += 32)
        group_heads.data[head_base + coordinate] = -1;
    SYNC();

    // Reserve every diagonal coordinate used by a coupled unit. A scalar
    // unit sharing one must remain in the serial chain to preserve GS order.
    // Also check the layout: the segment starts at a head, and each head is
    // followed by the next head or the segment end, directly or after its two
    // tangent rows naming it. Both serial walks then visit exactly the heads.
    bool rows_form_units = contact_start >= contact_end
        || {f"!{member('contact_start')}" if torsion else f"world_row_type.data[row_base + contact_start] != {friction}"};
    for (int normal = contact_start + lane; normal < contact_end; normal += 32) {{
        const int sparse_row = sparse_world_base + normal * 2;
        const int coord_0 = sparse_row_dof.data[sparse_row];
        const int coord_1 = sparse_row_dof.data[sparse_row + 1];
        const bool unit_head = {unit_head};
        const bool tangent_1 = normal + 1 < contact_end && world_row_type.data[row_base + normal + 1] == {friction};
        const bool tangent_2 = normal + 2 < contact_end && world_row_type.data[row_base + normal + 2] == {friction};
        const bool tangent_3 = normal + 3 < contact_end && world_row_type.data[row_base + normal + 3] == {friction};
        const bool named_1 = normal + 1 < contact_end && world_row_parent.data[row_base + normal + 1] == normal;
        const bool named_2 = normal + 2 < contact_end && world_row_parent.data[row_base + normal + 2] == normal;
        if (!unit_head) continue;
        {layout}
        const bool scalar_candidate = coord_0 < -1 || coord_1 < -1;
        if (!scalar_candidate) {{
            if (coord_0 >= 0) atomicExch(&group_heads.data[head_base + coord_0], -2);
            if (coord_1 >= 0) atomicExch(&group_heads.data[head_base + coord_1], -2);
        }}
    }}
    rows_form_units = ALL(rows_form_units);
    SYNC();

    if (rows_form_units) {{
{parallel_links}
{parallel_serial_schedule}
    }} else {{
{serial_walks}
{serial_schedule}
    }}
    SYNC();

    int count = 0;
    for (int coordinate_base = 0; coordinate_base < {D}; coordinate_base += 32) {{
        const int coordinate = coordinate_base + lane;
        const int head = coordinate < {D} ? group_heads.data[head_base + coordinate] : -1;
        const int output = ORDERED(count, head >= 0);
        if (head >= 0) group_heads.data[head_base + output] = head;
        SYNC();
    }}
    if (lane == 0) group_count.data[world] = count;
#endif
"""

    name = f"build_independent_sparse_contact_groups_{M}_{D}"
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
            ("world_row_parent", wp.array2d[int]),
            ("sparse_row_dof", wp.array3d[int]),
            ("group_count", wp.array[int]),
            ("group_heads", wp.array2d[int]),
            ("serial_count_out", wp.array[int]),
            ("serial_normals", wp.array2d[int]),
        ),
        snippet,
        dispatch=items("world", lanes=32, flat=True),
        preludes=(lane_group(32),),
        native_name="build_independent_sparse_contact_units_native",
    ).kernel()
