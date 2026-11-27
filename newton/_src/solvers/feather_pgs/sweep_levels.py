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
from ...core.memory import Buffer, Staging, arch_budget
from . import kernel_template
from .contact_input import ContactInput
from .device_preludes import LANE_WORDS, SHARED_STORAGE, group_native, lane_group
from .kernel_template import KernelTemplate, items, python_kernel
from .row_laws import ROW_FRICTION, ROW_TORSION
from .row_records import line_size

# FeatherPGS takes no gradients: its kernels compile no adjoints.
wp.set_module_options({"enable_backward": False})


def level_pass(
    *,
    units: str,
    resources: str,
    counts: str | None,
    store: str,
    lanes: int | str = 32,
    after: bool = False,
    ahead: tuple[str, str] = ("", ""),
    turns: str = "",
    counted: str = "",
) -> str:
    """Return CUDA source that levels a world's ``units`` units in row order (Cartier-Foata), by one warp, a batch of
    32 at a time.

    A unit's ``level`` is one past the latest level of an earlier unit on either of its resources, so units that share
    one keep their row order and a level's units share none. A batch's units read the levels their resources hold from
    earlier batches and find, on each resource, the batch's latest earlier unit that shares it and whether a later one
    does (one look at every lane of the batch). A unit's level is one past those units' and the held levels; the batch
    takes it in rounds, each unit reading its two predecessors' levels, until no level moves: a round fixes one more
    unit of every chain, so the levels are the pass's in row order. Each resource's last unit in the batch leaves its
    level. After its batch a unit takes its ``rank``, its place among its level's units in row order: the level's units
    of earlier batches and the batch's units of the level on lower lanes.

    ``resources`` declares, for lane's ``unit`` (``valid`` when it is below ``units``; ``resources`` may clear it for
    a unit that takes no level), its resources ``node_a`` and ``node_b``: compact ids into the warp's ``node_level``
    (-1, or any other value no level overwrites), or -1 for none. ``counts`` formats an index into the units of each
    level so far: zero at the start for levels ``[0, units]``, and the batch's counts added; with none the pass levels
    alone (no ``rank``), and with ``counted`` (a native condition, the same on every thread) only where it holds (else
    ``rank`` is 0). ``store`` keeps a valid unit's ``level`` and ``rank``; with ``after`` it may also read
    ``after_a`` and ``after_b``, the levels its resources held before it (its latest earlier unit's on each, -1 for
    none or no resource). The fragment runs on every lane of
    the warp, uses the lane group, ``lane`` and the names ``resources`` and ``store`` read, and declares ``levels``, the
    world's level count, on every lane. ``lanes`` is the group: a warp, the CPU's one thread (a batch of one unit), or a
    lane group of a native expression's lanes (``STRIDE``), whose words are the lane group's (``MATCH``, ``BELOW``,
    ``COUNT``, ``READ``; :data:`.device_preludes.LANE_WORDS`). ``ahead`` is a caller's pipeline of its resources'
    loads: statements before the batches, and statements each batch runs once its held levels are read, so the next
    batch's loads are in flight while the batch levels.

    With ``turns`` (a block's shared word, CUDA's warps; at most 15 warps) the block's warps take the batches in turn,
    warp ``w`` the batches ``w``, ``w + WARPS``, ...: a batch's resources and in-batch predecessors are its own, so its
    warp finds them while earlier batches level; then its turn (the warp before it arrives at its named barrier, ``1
    + w``, once its levels and counts are fenced) reads the held levels and the counts, runs the rounds, leaves its
    resources' levels and the counts, and passes the turn on; its ranks, stores and ``ahead``'s loads follow. A
    waiting warp issues nothing (a polled word would take the shuffle pipe from the turn). The batches level in row
    order, so the levels and ranks are the one warp's. ``units`` and ``node_level`` are the block's, ``levels`` its
    count on every thread (``turns``; a block barrier at each end).
    """
    if lanes == 32:
        same, below, count = (
            "__match_any_sync(0xffffffffu, level)",
            "__popc(same & ((1u << lane) - 1u))",
            "__popc(same)",
        )
        width, moved = "32", "__any_sync(0xffffffffu, moved)"

        def read(value: str, source: str = "j") -> str:
            return f"__shfl_sync(0xffffffffu, {value}, {source})"

    elif lanes == 1 or isinstance(lanes, str):
        # A lane group's words (the CPU's one thread, or a group of a native expression's lanes, ``STRIDE``).
        same, below, count = "MATCH(level)", "BELOW(same)", "COUNT(same)"
        width, moved = str(lanes), "ANY(moved)"

        def read(value: str, source: str = "j") -> str:
            return f"READ({value}, {source})"

    else:
        raise ValueError(f"the level pass runs a warp, a lane group or one thread, got {lanes}")
    if turns and lanes != 32:
        raise ValueError(f"the block's turns take warps of 32 lanes, got {lanes}")
    held = (
        """
            const int after_a = prev_a >= 0 ? via_a : held_a, after_b = prev_b >= 0 ? via_b : held_b;"""
        if after
        else ""
    )
    before, during = ahead
    count_at = counts.format("level") if counts is not None else ""
    if turns:
        # The block's warps in turn: a batch's warp finds its predecessors, then takes its turn from the warp before
        # it by a named barrier of the two warps (1 + its warp; 0 is the block's); the held levels and the counts are
        # the turns' (relaxed at the block's scope), ordered by the producer's block fence before the barrier.
        begin, step = "(int)(threadIdx.x & ~31u)", "(int)blockDim.x"
        start = f"""        const auto TURN_LOAD = [](const int* at) {{
            int value;
            asm volatile("ld.relaxed.cta.s32 %0, [%1];" : "=r"(value) : "l"(at) : "memory");
            return value;
        }};
        const auto TURN_STORE = [](int* at, int value) {{
            asm volatile("st.relaxed.cta.s32 [%0], %1;" ::"l"(at), "r"(value) : "memory");
        }};
        if (threadIdx.x == 0) {turns} = 0;
        __syncthreads();
"""
        wait = """            // The batch's turn: the warp before it has left its levels and counts and arrived at this warp's barrier.
            if (base != 0) asm volatile("bar.sync %0, 64;" ::"r"(1 + (int)(threadIdx.x >> 5)) : "memory");
"""
        release = f"""
            // The next batch's turn: the levels and counts before the next warp's barrier (a block fence).
            __threadfence_block();
            if (base + 32 < {units})
                asm volatile("bar.arrive %0, 64;" ::"r"(1 + (int)((threadIdx.x >> 5) + 1) % (int)(blockDim.x >> 5)) : "memory");
{during.rstrip()}"""
        end = f"""        MAXIMUM(levels);
        if (lane == 0) atomicMax(&{turns}, levels);
        __syncthreads();
        levels = {turns};"""

        def load(at: str) -> str:
            return f"TURN_LOAD(&{at})"

        def put(at: str, value: str) -> str:
            return f"TURN_STORE(&{at}, {value})"

    else:
        begin, step, start, wait, end = "0", f"{lanes}", "", "", "        MAXIMUM(levels);"
        release = "\n            SYNC();"

        def load(at: str) -> str:
            return at

        def put(at: str, value: str) -> str:
            return f"{at} = {value}"

    held_levels = f"""            // The levels the unit's resources hold from earlier batches.
            const int held_a = res_a >= 0 ? {load("node_level[res_a]")} : -1;
            const int held_b = res_b >= 0 ? {load("node_level[res_b]")} : -1;
"""
    # The batch's counts of each level and each unit's rank among its level's units in row order, where counted.
    if counts is not None and counted:
        ranked = f"""            int earlier = 0, below = 0;
            unsigned same = 0u;
            if ({counted}) {{
                same = {same};
                below = {below};
                earlier = valid ? {load(count_at)} : 0;
            }}
            SYNC();
            if (({counted}) && valid && below == 0) {put(count_at, f"earlier + {count}")};{release}
            if (valid) {{
                const int rank = earlier + below;
                {store}
            }}"""
    elif counts is not None:
        ranked = f"""            const unsigned same = {same};
            const int below = {below};
            const int earlier = valid ? {load(count_at)} : 0;
            SYNC();
            if (valid && below == 0) {put(count_at, f"earlier + {count}")};{release}
            if (valid) {{
                const int rank = earlier + below;
                {store}
            }}"""
    else:
        ranked = f"""{release.lstrip(chr(10))}
            if (valid) {{
                {store}
            }}"""
    # One warp reads the held levels first (their loads in flight with the batch's own work); a turn reads them last.
    # A turn's next loads follow its hand-off: the hand-off's fence would wait for a load in flight.
    early, late = ("", wait + held_levels) if turns else (held_levels + during, "")
    return f"""{start}{before}        int levels = 0;
        for (int base = {begin}; base < {units}; base += {step}) {{
            const int unit = base + lane;
            bool valid = unit < {units};
{resources}
            const int res_a = valid ? node_a : -1, res_b = valid ? node_b : -1;
{early}            // On each resource, the batch's latest earlier unit that shares it and whether a later one does.
            int prev_a = -1, prev_b = -1;
            bool last_a = true, last_b = true;
            for (int j = 0; j < {width}; ++j) {{
                const int other_a = {read("res_a")}, other_b = {read("res_b")};
                const bool on_a = res_a >= 0 && (other_a == res_a || other_b == res_a);
                const bool on_b = res_b >= 0 && (other_a == res_b || other_b == res_b);
                if (j < lane && on_a) prev_a = j;
                if (j < lane && on_b) prev_b = j;
                if (j > lane && on_a) last_a = false;
                if (j > lane && on_b) last_b = false;
            }}
{late}            // One past the held levels and the predecessors' levels, by rounds until no level moves.
            const int first = valid ? max(held_a, held_b) + 1 : -1;
            int level = first, via_a = -1, via_b = -1;
            for (bool moved = true; {moved};) {{
                via_a = {read("level", "prev_a >= 0 ? prev_a : lane")};
                via_b = {read("level", "prev_b >= 0 ? prev_b : lane")};
                int next = first;
                if (prev_a >= 0) next = max(next, via_a + 1);
                if (prev_b >= 0) next = max(next, via_b + 1);
                moved = next != level;
                level = next;
            }}{held}
            if (res_a >= 0 && last_a) {put("node_level[res_a]", "level")};
            if (res_b >= 0 && last_b) {put("node_level[res_b]", "level")};
{ranked}
            levels = max(levels, level + 1);
        }}
{end}"""


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
def tier_world(rows: int, free_body_rows: int, bound: int) -> bool:
    """Whether a world of ``rows`` dense rows is a Jacobi tier's (the tiers form its rows' response in their kernel):
    its rows are dense only, no free body's (``free_body_rows``, the free bodies' run), and at most the tiers'
    ``bound``. The claims (:func:`claim_worlds`) and the audit read this one predicate."""
    return free_body_rows == 0 and rows <= bound


COST_CLASSES = 32
"""The general sweep's cost classes, a bucket each: a world's class is the bit length of its own cycles in the sweep's
last call (``world_cost``, written by the sweep: 0 before one), heaviest first, so the classes read as one list order
the worlds by cost to a factor of two (:func:`cost_bucket`)."""


COST_MATES = 2
"""The classes above the middle world's that make a world heavy: two, about four times the middle world's cost. A
warp's worlds take about their sum where their paths part, so where the general sweep's list holds a heavy world its
warps pair the list's two ends (a heavy world with the lightest), and where it holds none, neighbours of like cost
share a warp, on like paths and near their data (:func:`~newton._src.core.execution.warp_mates`)."""


@wp.func
def cost_bucket(general: int, cost: int) -> int:
    """The general sweep's bucket of a world whose own cycles in the sweep's last call have ``cost`` bits: the classes
    from ``general`` on, heaviest first (:data:`COST_CLASSES`)."""
    return general + COST_CLASSES - 1 - wp.clamp(cost, 0, COST_CLASSES - 1)


@wp.kernel
def claim_worlds(
    rows: wp.array[int],
    free_body_rows: wp.array[int],
    slot: wp.array[wp.int32],
    world_cost: wp.array[wp.int32],
    bound: int,
    width: int,
    general: int,
    candidate: int,
    built: int,
    region: int,
    acts: wp.array[int],
    group_slot: wp.array[int],
    group_colors: wp.array[int],
    group_offsets: wp.array2d[int],
    group_rows: wp.array2d[int],
    topology_rows: wp.array[int],
    grouped: int,
    # outputs: a Queue's (execution.Queue), a bucket a tier, then the general sweep's and the colored candidates'
    items: wp.array[int],
    count: wp.array[int],
):
    """Write each world with rows into its class's bucket of one queue of ``region`` slots a bucket: a world whose rows
    are dense only and at most ``bound`` into its Jacobi tier's, ``(rows - 1) // width``; any other world with a colored
    storage ``slot`` (``slot[world] >= 0``, :class:`.sweep_units.ColoredUnits`) into ``candidate``, which the unit
    builder walks and splits into the colored bucket and the general one (``candidate`` negative: no colored mapping,
    and ``slot`` is not read); a world whose dense line its last reservation placed by colors of groups (its
    ``group_slot``'s ``group_colors``, :class:`.row_slots.RowGroups`) and holds them still (its topology rows end at
    its first group's row and its rows at its last group's end: a world with no contacts reserves none) into
    ``grouped``; the rest into the general sweep's cost classes from ``general`` on, by the bits of its
    own cycles in the sweep's last call (``world_cost``, :func:`cost_bucket`). A world without rows takes none, nor one
    none of whose rows can act (its ``acts`` word, :data:`.placement.WORLD_ACTS`) but a candidate: the unit builder
    builds every candidate's units, a function of the collision pass's contact set alone, and claims only those that
    act (:func:`.sweep_units.build_units_kernel`). A free body's rows in the free bodies' run (``free_body_rows``) are not dense only.

    A claim's worlds share nothing with another's, so each is a component of its own and the order inside a bucket is
    free (:func:`~newton._src.core.execution.append`'s, an atomic a warp a bucket); tier ``t`` serves
    ``(t * width, (t + 1) * width]`` rows and walks bucket ``t``, the general sweep its classes as one list, heaviest
    first (an exiting block of a grid over every world paid four dependent global loads: about 55 us a launch at 16k
    worlds). The order is a move: a world's sweep reads and writes its own rows and coordinates only.
    """
    world = wp.tid()
    m = rows[world]
    mf = free_body_rows[world]
    if m <= 0 and mf <= 0:
        return
    bucket = int(-1)
    group = int(-1)
    if group_slot.shape[0] > 0:
        group = group_slot[world]
    held = False
    if group >= 0:
        colors = group_colors[group]
        if colors > 0:
            held = group_rows[group, 0] == topology_rows[world] and group_rows[group, group_offsets[group, colors]] == m
    if held:
        bucket = grouped
    elif tier_world(m, mf, bound):
        bucket = (m - 1) // width
    elif candidate >= 0:
        if slot[world] >= 0:
            bucket = candidate
    if (built == 0 or bucket != candidate) and acts.shape[0] > 0 and acts[world] == 0:
        return
    if bucket < 0:
        bucket = cost_bucket(general, world_cost[world])
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
        int(max_constraints), int(max_world_dofs), build_serial_contacts, torsion, device_arch
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
def levels_staging(max_world_dofs: int, device_arch) -> Staging:
    """The contact schedule's shared working set: a world's lane group's coordinates' owner and level
    (:func:`level_pass`) and the offsets' warp sum; on CUDA as many worlds a block, a warp each and up to
    :data:`LEVELS_WORLDS`, as the device's stage budget holds (:func:`.memory.arch_budget`; the CPU's one thread runs
    one). Each contact position's packed level and rank are its lane's (:func:`_sparse_contact_levels_kernel`)."""
    buffers = (
        Buffer("node_owner", "int", max_world_dofs),
        Buffer("node_level", "int", max_world_dofs),
        Buffer("warp_count", "int", 1),
    )
    fit = arch_budget(device_arch) // Staging(buffers).group_bytes
    groups = max(1, min(LEVELS_WORLDS, fit)) if device_lanes(device_arch, 32) > 1 else 1
    return Staging(buffers, groups=groups, layout="arrays")


def levels_launch(world_count: int, max_world_dofs: int, device_arch) -> tuple[int, int]:
    """The contact schedule's launch, ``(dim, block_dim)``: a world's lane group (a warp, or the CPU's one thread,
    :func:`.execution.device_lanes`), :func:`levels_staging`'s groups a block."""
    lanes = device_lanes(device_arch, 32)
    return int(world_count) * lanes, lanes * levels_staging(int(max_world_dofs), device_arch).groups


def _levels_kernel(
    name: str,
    params: tuple,
    M: int,
    D: int,
    device_arch,
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
    staging, L = levels_staging(D, device_arch), device_lanes(device_arch, 32)
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
    for (int coordinate = lane; coordinate < {D}; coordinate += {L}) node_level[coordinate] = -1;
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
        stage=staging.dynamic_bytes(device_arch),
    ).kernel()


_LEVELS_OUTPUTS = (
    ("level_count", wp.array[int]),
    ("level_offsets", wp.array2d[int]),
    ("level_units", wp.array2d[int]),
)


def _sparse_contact_levels_kernel(
    M: int, D: int, build_serial_contacts: bool, torsion: bool = False, device_arch=None
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
    L = device_lanes(device_arch, 32)
    if M >= 1 << 15:
        raise ValueError(f"a world's contact schedule packs a unit's level and rank in 16 bits each: {M} rows")
    friction, spin = int(ROW_FRICTION), int(ROW_TORSION)
    kind = "world_row_type.data[row_base + normal]"
    # A unit's members (a tangent, or a spin row) follow its head; every other row of the segment heads a unit.
    unit_head = f"({kind} != {friction} && {kind} != {spin})" if torsion else f"{kind} != {friction}"
    levels_staging(D, device_arch).check("the sparse-diagonal contact schedule", arch_budget(device_arch))
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
    units = f"""    const int contact_start = topology_rows.data[world];
    const int contact_end = min(world_constraint_count.data[world], {M});
    const int units = contact_end - contact_start;
    const int line = dense_first.data[world];
    __builtin_assume(line >= 0);
    const int row_base = line * {line_size("world_row_type")};
    const int sparse_world_base = line * {line_size("sparse_row_dof")};"""
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
        ("topology_rows", wp.array[int]),
        ("world_row_type", wp.array2d[int]),
        ("sparse_row_dof", wp.array3d[int]),
        *_LEVELS_OUTPUTS,
        ("serial_count_out", wp.array[int]),
        ("serial_normals", wp.array2d[int]),
    )
    return _levels_kernel(
        name, params, M, D, device_arch, units=units, reserve=reserve, resources=resources, tail=serial
    )
