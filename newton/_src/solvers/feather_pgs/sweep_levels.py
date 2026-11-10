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
queues (:class:`~newton._src.core.execution.Queue`): :func:`claim_worlds` writes each world into its claim's bucket.

Case K2 (the sparse diagonal): the row pass marks the contacts confined to one coordinate of the diagonal articulation,
and the builder levels them over their coordinates and lists the rest in row order, which the sweep reads.
"""

from functools import cache

import warp as wp

from ...core.execution import append, device_lanes
from ...core.memory import STATIC_SHARED_BYTES, Buffer, Staging
from . import kernel_template
from .contact_input import ContactInput
from .device_preludes import LANE_WORDS, SHARED_STORAGE, group_native, lane_group
from .kernel_template import KernelTemplate, items, python_kernel
from .packed_rows import CLASS_SIDES
from .row_laws import ROW_CONTACT, ROW_FRICTION, ROW_JOINT_LIMIT, ROW_JOINT_VELOCITY_LIMIT, ROW_TORSION


def level_pass(*, units: str, resources: str, counts: str, store: str, lanes: int = 32) -> str:
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
    ``levels``, the world's level count, on every lane. ``lanes`` is the group: a warp, or the CPU's one thread, a batch
    of one unit, whose words are the lane group's (``MATCH``, ``BELOW``, ``COUNT``; :data:`.device_preludes.LANE_WORDS`).
    """
    if lanes == 32:
        same, below, count = (
            "__match_any_sync(0xffffffffu, level)",
            "__popc(same & ((1u << lane) - 1u))",
            "__popc(same)",
        )
    elif lanes == 1:
        same, below, count = "MATCH(level)", "BELOW(same)", "COUNT(same)"
    else:
        raise ValueError(f"the level pass runs a warp or one thread, got {lanes}")
    return f"""        int levels = 0;
        for (int base = 0; base < {units}; base += {lanes}) {{
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
            const unsigned same = {same};
            const int below = {below};
            const int earlier = valid ? {counts.format("level")} : 0;
            SYNC();
            if (valid && below == 0) {counts.format("level")} = earlier + {count};
            SYNC();
            if (valid) {{
                const int rank = earlier + below;
                {store}
            }}
            levels = max(levels, level + 1);
        }}
        MAXIMUM(levels);"""


def level_offsets(*, levels: str, counts: str, lanes: int = 32, sync: str = "__syncthreads()") -> str:
    """Return CUDA source that turns the units of each of ``levels`` levels into each level's first place, in place:
    an exclusive scan by the group, a group of levels at a time; place ``levels`` is the unit count.

    ``counts`` formats an index into the counts (``levels + 1`` of them). The fragment runs on every thread of a group
    of ``THREADS`` threads in ``WARPS`` warps that ``sync`` synchronizes (a block's ``__syncthreads()``, or the lane
    group's ``SYNC()`` for a group of one warp or the CPU's one thread), and uses the lane group, ``tid``, ``lane``,
    ``warp`` and a shared ``warp_count`` of a word a warp. ``lanes`` is a warp's lanes, or the CPU's one thread.
    """
    last, barrier = f"{lanes - 1}", f"{sync};"
    return f"""    int carry = 0;
    for (int base = 0; base <= {levels}; base += THREADS) {{
        const int c = base + tid;
        const int count = c < {levels} ? {counts.format("c")} : 0;
        const int below = SCAN(count);
        if (lane == {last}) warp_count[warp] = below + count;
        {barrier}
        int before = carry, total = 0;
        for (int w = 0; w < WARPS; ++w) {{
            if (w < warp) before += warp_count[w];
            total += warp_count[w];
        }}
        if (c <= {levels}) {counts.format("c")} = before + below;
        carry += total;
        {barrier}
    }}"""


@wp.func
def tier_world(rows: int, free_body_rows: int, speed_first: wp.array[int], world: int, bound: int) -> bool:
    """Whether a world of ``rows`` dense rows is a Jacobi tier's (the tiers form its rows' response in their kernel):
    its rows are dense only, no free body's (``free_body_rows``, the free bodies' run, or a speed row of the line,
    from ``speed_first``), and at most the tiers' ``bound``. The claims (:func:`claim_worlds`) and the audit read this
    one predicate."""
    return free_body_rows == 0 and rows <= bound and rows <= speed_first[world]


@wp.kernel
def claim_worlds(
    rows: wp.array[int],
    free_body_rows: wp.array[int],
    speed_first: wp.array[int],
    slot: wp.array[wp.int32],
    bound: int,
    width: int,
    general: int,
    candidate: int,
    region: int,
    acts: wp.array[int],
    dense_first: wp.array[int],
    # outputs: a Queue's (execution.Queue), a bucket a tier, then the general sweep's and the colored candidates'
    items: wp.array[int],
    count: wp.array[int],
    impulses: wp.array2d[float],
):
    """Write each world with rows into its class's bucket of one queue of ``region`` slots a bucket: a world whose rows
    are dense only and at most ``bound`` into its Jacobi tier's, ``(rows - 1) // width``; any other world with a colored
    storage ``slot`` (``slot[world] >= 0``, :class:`.sweep_units.ColoredUnits`) into ``candidate``, which the unit
    builder walks and splits into the colored bucket and the general one (``candidate`` negative: no colored mapping,
    and ``slot`` is not read); the rest into ``general``, the general sweep's. A world without rows takes none, nor one
    none of whose rows can act (its ``acts`` word, :data:`.placement.WORLD_ACTS`). A free
    body's rows are not dense only: those in the free bodies' run (``free_body_rows``) and the line's speed rows (its
    rows from ``speed_first``).

    A claim's worlds share nothing with another's, so each is a component of its own and the order inside a bucket is
    free (:func:`~newton._src.core.execution.append`'s, an atomic a warp a bucket); tier ``t`` serves
    ``(t * width, (t + 1) * width]`` rows and walks bucket ``t``, the general sweep bucket ``general`` (an exiting block
    of a grid over every world paid four dependent global loads: about 55 us a launch at 16k worlds).
    """
    world = wp.tid()
    m = rows[world]
    mf = free_body_rows[world]
    if (m <= 0 and mf <= 0) or (acts.shape[0] > 0 and acts[world] == 0):
        # Where the solve group starts the impulses of its claim's worlds (``impulses`` not empty), a world left out
        # starts its active rows' here, as prepare_world_impulses would have (a cold solve).
        if impulses.shape[0] > 0:
            for i in range(m):
                impulses[dense_first[world], i] = 0.0
        return
    bucket = general
    if tier_world(m, mf, speed_first, world, bound):
        bucket = (m - 1) // width
    elif candidate >= 0:
        if slot[world] >= 0:
            bucket = candidate
    append(items, count, bucket, region, world)


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
    """The sparse-diagonal contact schedule over the contact units (:func:`_sparse_contact_levels_kernel`), a world a
    warp on CUDA and a thread on the CPU (:func:`.execution.device_lanes`)."""
    return _sparse_contact_levels_kernel(
        int(max_constraints), int(max_world_dofs), build_serial_contacts, torsion, device_lanes(device_arch, 32)
    )


def _list_serial_contacts_source(*, count: str, stride: int, head: str, serial_capacity: int, lanes: int = 32) -> str:
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
    for (int base = 0; base < {count}; base += {lanes}) {{
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


LEVELS_WORLDS = 8
"""Worlds a block of the sparse-diagonal contact schedule holds at most, a warp each: one warp a block held a
multiprocessor at its block limit (24 warps on sm_120, half its 48)."""


@cache
def levels_staging(max_world_dofs: int, lanes: int = 32) -> Staging:
    """The contact schedule's shared working set: a world's lane group's coordinates' owner and level
    (:func:`level_pass`) and the offsets' warp sum; on CUDA as many worlds a block, a warp each and up to
    :data:`LEVELS_WORLDS`, as the static shared budget holds (the CPU's one thread runs one). Each contact position's
    packed level and rank are its lane's (:func:`_sparse_contact_levels_kernel`), not shared."""
    buffers = (
        Buffer("node_owner", "int", max_world_dofs),
        Buffer("node_level", "int", max_world_dofs),
        Buffer("warp_count", "int", 1),
    )
    groups = max(1, min(LEVELS_WORLDS, STATIC_SHARED_BYTES // Staging(buffers).group_bytes)) if lanes > 1 else 1
    return Staging(buffers, groups=groups, layout="arrays")


def levels_launch(world_count: int, max_world_dofs: int, device_arch) -> tuple[int, int]:
    """The contact schedule's launch, ``(dim, block_dim)``: a world's lane group (a warp, or the CPU's one thread,
    :func:`.execution.device_lanes`), :func:`levels_staging`'s groups a block."""
    lanes = device_lanes(device_arch, 32)
    return int(world_count) * lanes, lanes * levels_staging(int(max_world_dofs), lanes).groups


def _levels_kernel(
    name: str,
    params: tuple,
    M: int,
    D: int,
    L: int,
    *,
    units: str,
    reserve: str,
    resources: str,
    tail: str = "",
    preludes: tuple = (),
) -> "wp.Kernel":
    """A world's units by level (:func:`level_pass`, a coordinate each), a world's lane group (a warp, several a block;
    the CPU's one thread: :func:`levels_launch`). ``units`` declares ``contact_start``, ``contact_end``, ``line`` and
    ``row_base``, ``reserve`` reserves the coordinates no level holds, ``resources`` a unit's coordinate, ``tail``
    follows. Level ``l`` lists its heads in row order, ``level_units[world, level_offsets[world, l]:level_offsets[world,
    l + 1]]`` of ``level_count[world]``; a unit's level and rank wait for the offsets in its lane's ``unit_place``."""
    staging = levels_staging(D, L)
    counts = f"level_offsets.data[world * {M + 1} + {{}}]"
    level = level_pass(
        units="units",
        resources=resources,
        counts=counts,
        store=f"unit_place[base / {L}] = (level << 16) | rank;",
        lanes=L,
    )
    offsets = level_offsets(levels="levels", counts=counts, lanes=L, sync="SYNC()")
    snippet = f"""    const int lane = LANE;
    const int tid = lane, warp = 0;
{units}
{staging.declarations("    ")}
    int unit_place[{-(-M // L)}];
    for (int coordinate = lane; coordinate < {D}; coordinate += {L}) {{
        node_owner[coordinate] = 32;
        node_level[coordinate] = -1;
    }}
    for (int l = lane; l <= units; l += {L}) {counts.format("l")} = 0;
    SYNC();

{reserve}
    SYNC();

{level}
    SYNC();
{offsets}
    for (int unit = lane; unit < units; unit += {L}) {{
        const int place = unit_place[unit / {L}];
        if (place >= 0)
            level_units.data[world * {M} + {counts.format("(place >> 16)")} + (place & 0xFFFF)] = contact_start + unit;
    }}
    if (lane == 0) level_count.data[world] = levels;
{tail}"""
    defines = f"#define THREADS {L}\n#define WARPS 1\n#define RESERVED (-2)\n"
    source = f"{defines}{snippet}\n#undef THREADS\n#undef WARPS\n#undef RESERVED\n"
    return KernelTemplate(
        name,
        kernel_template.params(*params),
        group_native(
            source, preludes=(lane_group(L), SHARED_STORAGE, *preludes, *(() if L == 32 else (LANE_WORDS,))), lanes=L
        ),
        dispatch=items("world", lanes=L, flat=True),
        composed=True,
        native_name="sparse_contact_levels_native",
    ).kernel()


_LEVELS_OUTPUTS = (
    ("level_count", wp.array[int]),
    ("level_offsets", wp.array2d[int]),
    ("level_units", wp.array2d[int]),
)


def _sparse_contact_levels_kernel(
    M: int, D: int, build_serial_contacts: bool, torsion: bool = False, lanes: int = 32
) -> "wp.Kernel":
    """Case K2's contact schedule over its slots (:func:`_levels_kernel`): the levels of the scalar contacts and the
    serial list of the rest.

    Under patch friction a contact's rows form a unit: its normal row, followed by its two tangent rows only when it
    owns an anchor (and its spin row under ``torsion``), each naming the normal as parent. A unit the marker left with a
    marked slot (below ``-1``) is a *scalar contact*, on its other slot's coordinate. A coupled unit reserves its
    coordinates, and a scalar contact on a reserved one loses its mark (``-1``): it stays in the serial order
    (:func:`_list_serial_contacts_source`, the other unit heads in row order). The others are leveled in row order: a
    coordinate's ``k``-th contact is level ``k``.
    """
    L = lanes
    if M >= 1 << 15:
        raise ValueError(f"a world's contact schedule packs a unit's level and rank in 16 bits each: {M} rows")
    friction, spin = int(ROW_FRICTION), int(ROW_TORSION)
    kind = "world_row_type.data[row_base + normal]"
    # A unit's members (a tangent, or a spin row) follow its head; every other row of the segment heads a unit.
    unit_head = f"({kind} != {friction} && {kind} != {spin})" if torsion else f"{kind} != {friction}"
    levels_staging(D, L).check("the sparse-diagonal contact schedule")
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
            unit_place[base / {L}] = -1;"""
    serial = (
        _list_serial_contacts_source(count="units", stride=1, head=unit_head, serial_capacity=M, lanes=L)
        if build_serial_contacts
        else ""
    )
    units = f"""    const int contact_start = dense_phase_bounds.data[world * 2 + 1];
    const int contact_end = min(world_constraint_count.data[world], {M});
    const int units = contact_end - contact_start;
    const int line = dense_first.data[world];
    __builtin_assume(line >= 0);
    const int row_base = line * {M};
    const int sparse_world_base = line * {M} * 2;"""
    reserve = f"""    // A coupled unit reserves every coordinate it holds: a scalar contact sharing one stays in the serial order.
    for (int normal = contact_start + lane; normal < contact_end; normal += {L}) {{
        const int sparse_row = sparse_world_base + normal * 2;
        const int coord_0 = sparse_row_dof.data[sparse_row];
        const int coord_1 = sparse_row_dof.data[sparse_row + 1];
        if (!({unit_head}) || coord_0 < -1 || coord_1 < -1) continue;
        if (coord_0 >= 0) node_level[coord_0] = RESERVED;
        if (coord_1 >= 0) node_level[coord_1] = RESERVED;
    }}"""
    name = f"sparse_contact_levels_{M}_{D}"
    if build_serial_contacts:
        name += "_serial_contacts"
    name += "_units" + ("_torsion" if torsion else "") + ("" if L == 32 else f"_lanes{L}")
    params = (
        ("world_constraint_count", wp.array[int]),
        ("dense_first", wp.array[int]),
        ("dense_phase_bounds", wp.array2d[int]),
        ("world_row_type", wp.array2d[int]),
        ("sparse_row_dof", wp.array3d[int]),
        *_LEVELS_OUTPUTS,
        ("serial_count_out", wp.array[int]),
        ("serial_normals", wp.array2d[int]),
    )
    return _levels_kernel(name, params, M, D, L, units=units, reserve=reserve, resources=resources, tail=serial)


@cache
def coordinate_levels_kernel(M: int, P: int, D: int, device_arch: str, *, torsion: bool = False) -> "wp.Kernel":
    """The levels of the units on one coordinate no wider unit holds, over a world's rows by support class
    (:func:`.packed_rows.row_classes`, ``P`` rows a line), which the general sweep steps first (``coordinate_levels``).

    A unit heads its contact's rows (its pair and spin row name it as parent) or is any other row but a velocity limit
    (the last word's). A contact or joint limit of one entry ahead of every other unit on its coordinate is leveled in
    row order (:func:`_levels_kernel`), so it commutes with every row stepped in order: the order's iterate.
    """
    L = device_lanes(device_arch, 32)
    if M >= 1 << 15:
        raise ValueError(f"a world's level schedule packs a unit's level and rank in 16 bits each: {M} rows")
    levels_staging(D, L).check("the coordinate levels")
    kind = "world_row_type.data[row_base + row]"
    head = f"{kind} != {int(ROW_FRICTION)}" + (f" && {kind} != {int(ROW_TORSION)}" if torsion else "")
    entries = """const int classes = row_classes.data[row_base + row];
            const int c1 = (classes >> 13) & 0x1FFF;
            const int entries = ((classes >> 26) & 0x3F) + (c1 == 0x1FFF ? 0 : class_count.data[c1]);"""
    unit = f"({head} && {kind} != {int(ROW_JOINT_VELOCITY_LIMIT)})"
    # A level holds contacts and joint limits (the sweep compiles their laws apart).
    kinds = f"({kind} == {int(ROW_CONTACT)} || {kind} == {int(ROW_JOINT_LIMIT)})"
    units = f"""    const int contact_start = 0;
    const int contact_end = min(world_constraint_count.data[world], {M});
    const int units = contact_end;
    const int line = dense_first.data[world];
    __builtin_assume(line >= 0);
    const int row_base = line * {P};"""
    # Each coordinate's first row of a unit no level holds (node_owner, reset for the level pass): a contact or joint
    # limit of one entry before it on its coordinate keeps that coordinate for the pass (unit_coordinate, 0 for none).
    reserve = f"""    for (int c = lane; c < {D}; c += {L}) node_owner[c] = {M};
    SYNC();
    for (int row = lane; row < contact_end; row += {L}) {{
        if (!{unit}) continue;
        {entries}
        if (entries < 2 && {kinds}) continue;
        const wp::vec2i sides = row_sides.data[row_base + row];
        for (int e = 0; e < entries; ++e) {{
            const int coordinate = CLASS_COORDINATE(sides, classes, e);
            if (coordinate >= 0) atomicMin(&node_owner[coordinate], row);
        }}
    }}
    SYNC();
    int unit_coordinate[{-(-M // L)}];
    for (int base = 0; base < units; base += {L}) {{
        const int row = base + lane;
        int leveled = 0;
        if (row < units && {unit} && {kinds}) {{
            {entries}
            const int coordinate = CLASS_COORDINATE(row_sides.data[row_base + row], classes, 0);
            if (entries == 1 && coordinate >= 0 && row < node_owner[coordinate]) leveled = coordinate + 1;
        }}
        unit_coordinate[base / {L}] = leveled;
    }}
    SYNC();
    for (int c = lane; c < {D}; c += {L}) node_owner[c] = 32;"""
    resources = f"""            int node_a = -1;
            const int node_b = -1;
            if (valid) valid = unit_coordinate[base / {L}] > 0;
            if (valid) node_a = unit_coordinate[base / {L}] - 1;
            unit_place[base / {L}] = -1;"""
    name = f"coordinate_levels_{M}_{P}_{D}" + ("_torsion" if torsion else "") + ("" if L == 32 else f"_lanes{L}")
    params = (
        ("world_constraint_count", wp.array[int]),
        ("dense_first", wp.array[int]),
        ("world_row_type", wp.array2d[int]),
        ("row_sides", wp.array2d[wp.vec2i]),
        ("row_classes", wp.array2d[int]),
        ("class_dof", wp.array2d[int]),
        ("class_count", wp.array[int]),
        ("articulation_world_dof_offset", wp.array[int]),
        *_LEVELS_OUTPUTS,
    )
    return _levels_kernel(
        name, params, M, D, L, units=units, reserve=reserve, resources=resources, preludes=(CLASS_SIDES,)
    )
