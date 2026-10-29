# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Device idioms of the native sources and the Warp kernels, and the one place that composes them.

A native body calls device helpers that it does not define itself: the paired
friction law, the solve readout, and the idioms below. The body declares them
as preludes, and :func:`cuda_native` places them after the body's one
architecture guard, in the declared order; :func:`dual_native` composes a body
that also compiles for the CPU, where one thread runs a whole lane group. Only
this module joins prelude text to a body.

No body compiles to nothing on a device. A body written for CUDA only
(:func:`cuda_native`) fails the CPU build with ``#error``, and a part of a body
that runs on one device only (:func:`device_branches`) states the other
device's statements, or an ``#error``, explicitly.

The idioms, one implementation each, serve native sources through the
preludes and Warp kernels through the ``wp.func_native`` functions built here
from the same text:

``lane_group(lanes)``
    A group of ``lanes`` threads that synchronize, vote and exchange values: a
    power of two up to 32 shares a warp with the warp's other groups, more
    lanes (whole warps) own the block. It defines ``LANES``, the shuffle
    ``WIDTH``, the group's lane ``MASK`` in its warp, this thread's ``LANE``,
    the ``STRIDE`` of a group-strided loop, the group's index ``SUB`` among its
    block's ``GROUPS``, this thread's ``WARP`` among the group's ``WARPS`` (one
    for a warp group), and ``SYNC()``, ``WARP_SYNC()`` (this thread's warp
    only), ``ALL(p)``, ``ANY(p)``, ``BALLOT(p)`` (the group's vote bits, lane 0
    first), ``BELOW(bits)`` (bits of lower lanes), ``COUNT(bits)``,
    ``FIRST(bits)`` (-1 for none), ``READ(value, lane)``, ``SUM(values...)``,
    ``SEGMENT_SUM(width, values...)`` (sums over aligned segments of ``width``
    lanes, a power of two up to ``WIDTH``, on every lane of the group),
    ``MINIMUM(values...)``, ``MAXIMUM(values...)``, ``ARGMAX(value, index)``,
    ``SCAN(value)`` (the exclusive prefix sum), ``RESERVE(counter, n)``, one
    atomic reservation of ``n`` slots for the group that returns its first slot
    to every lane (the lanes then take their slots in lane order, ``BELOW`` of
    the vote), and ``ORDERED(base, take)``, the ordered reservation of a group
    that owns its counter: the lanes that take a slot get ``base`` onwards in
    lane order, and ``base`` advances past them. The exchanges stay within a
    warp: ``WIDTH`` lanes. The reductions
    run a butterfly from the widest offset down, so every lane holds the result
    its lane 0 would get from a descending shuffle-down ladder, bit for bit;
    ``SUM`` reduces several channels in one ladder. On the CPU one thread runs
    the group: one lane, and every operation is that lane's own value.

``ORDERED_TILES``
    Ordered reservation across a launch's blocks (:mod:`newton._src.core.ordered_reserve`):
    ``ORDERED_TILE(tile)`` (one block per tile) and ``ORDERED_NEXT(site, block, live_tiles, blocks)`` (a grid
    striding over a device-side count) claim a tile in input order, ``ORDERED_SCAN(site, claim, count, last,
    counter)`` gives every thread of the block its first slot in input order after the ``counter`` slots
    already taken (a block scan and a decoupled look-back). Warp kernels call :func:`ordered_tile`,
    :func:`ordered_next` and :func:`ordered_scan`, built from the same text.

``ring``
    A patch ring: contact normal rows linked into a circular list by their
    parents (-1 for a row without a ring), whatever encodes the parent link:
    the walks take it, and the values, as accessors. ``RING_LOAD(load, first,
    hops, parent, value)`` adds the values of the rows after ``first`` to
    ``load`` in ring order, the order the pooled normal load of a friction
    pair takes; ``RING_ANY(first, hops, parent, predicate)`` tells whether one
    of them passes. Both stop after ``hops`` rows, so a broken ring cannot
    hang a sweep.
"""

from dataclasses import dataclass
from functools import cache
from string import Template

import warp as wp

# The ordered mode of the L0 reservation, shared with the collision pipeline's writers.
from ...core.ordered_reserve import (  # noqa: F401
    ORDERED_CPU,
    ORDERED_CUDA,
    OrderedTiles,
    ordered_next,
    ordered_scan,
    ordered_tile,
)

ARCHITECTURE_GUARD = "#if defined(__CUDA_ARCH__)\n"


@dataclass(frozen=True)
class DevicePrelude:
    """Device code that a native body declares ahead of its own statements."""

    name: str
    """Name of the prelude in tests and diagnostics."""
    source: str
    """Device code placed after the architecture guard, without a trailing newline."""
    cpu: str | None = None
    """The CPU twin of :attr:`source`, for bodies :func:`dual_native` composes; ``None`` for CUDA-only preludes."""


def substitute(text: str, vocabulary: dict | None = None, /, **values) -> str:
    """Fill the ``$NAME`` (or ``${NAME}``) slots of ``text``.

    Every slot needs a value, and every keyword value a slot; ``vocabulary`` offers values a text may or may not use
    (a layout's sizes). Values are converted with ``str`` and never searched for slots themselves.

    Args:
        text: A source template (``string.Template`` syntax; ``$$`` is a literal ``$``).
        vocabulary: Values available to every slot, by name; keyword values take precedence.
        values: The value of each slot.
    """
    template = Template(text)
    slots = set(template.get_identifiers())
    available = {**(vocabulary or {}), **values}
    missing, unused = sorted(slots - available.keys()), sorted(values.keys() - slots)
    if missing or unused:
        raise ValueError(f"source slots without values {missing}, values without slots {unused}")
    return template.substitute({name: str(available[name]) for name in slots})


CUDA_ONLY = '#error "a CUDA-only native body is built for the CPU"\n'
"""The CPU branch of a CUDA-only body: the CPU build fails instead of compiling the body to nothing."""


def cuda_native(body: str, *, preludes: tuple[DevicePrelude, ...] = (), lead: str = "\n") -> str:
    """Return a ``func_native`` snippet that opens with one architecture guard and the body's preludes.

    The snippet is ``lead``, the guard, each prelude followed by a newline in
    the declared order, then ``body`` up to the guard's closing ``#endif``, then
    the CPU branch :data:`CUDA_ONLY`: a CPU build of the body fails.

    Args:
        body: Native statements after the preludes, ending with the guard's closing ``#endif``.
        preludes: Device preludes the body declares, in the order they are defined.
        lead: Text before the architecture guard.

    Raises:
        ValueError: for a body that does not end with the guard's ``#endif``.
    """
    closed = body.rstrip()
    if not closed.endswith("#endif"):
        raise ValueError("a cuda_native body ends with its architecture guard's #endif")
    return (
        lead
        + ARCHITECTURE_GUARD
        + "".join(prelude.source + "\n" for prelude in preludes)
        + closed[: -len("#endif")]
        + "#else\n"
        + CUDA_ONLY
        + "#endif\n"
    )


def device_branches(cuda: str, cpu: str) -> str:
    """Return native statements that run ``cuda`` on a CUDA device and ``cpu`` on the CPU, under one guard.

    Args:
        cuda: Statements for the device, each line newline-terminated.
        cpu: Statements for the CPU, each line newline-terminated: never empty, :data:`CUDA_ONLY` for a part that
            exists on CUDA only.

    Raises:
        ValueError: for an empty CPU branch.
    """
    if not cpu.strip():
        raise ValueError("a device branch states its CPU statements, or CUDA_ONLY")
    return ARCHITECTURE_GUARD + cuda + "#else\n" + cpu + "#endif\n"


def dual_native(body: str, *, preludes: tuple[DevicePrelude, ...]) -> str:
    """Return a ``func_native`` snippet for both devices: the preludes for CUDA, their CPU twins otherwise, then ``body``.

    The body runs on both devices; it may keep architecture guards of its own.

    Args:
        body: Native statements after the preludes.
        preludes: Device preludes with CPU twins, in the order they are defined.
    """
    for prelude in preludes:
        if prelude.cpu is None:
            raise ValueError(f"prelude {prelude.name} has no CPU twin")
    cuda = "".join(prelude.source + "\n" for prelude in preludes)
    cpu = "".join(prelude.cpu + "\n" for prelude in preludes)
    return "\n" + ARCHITECTURE_GUARD + cuda + "#else\n" + cpu + "#endif\n" + body


# ---------------------------------------------------------------------------------------------------------------------
# Lane groups
# ---------------------------------------------------------------------------------------------------------------------

_LANE_GROUP = """    const int LANES = $lanes, WIDTH = LANES < 32 ? LANES : 32;
    const unsigned MASK = (0xffffffffu >> (32 - WIDTH)) << (threadIdx.x & (31 & ~(WIDTH - 1)));
    const int LANE = LANES <= 32 ? (int)(threadIdx.x & (LANES - 1)) : (int)threadIdx.x;
    const int STRIDE = LANES <= 32 ? LANES : (int)blockDim.x;
    const int SUB = LANES <= 32 ? (int)threadIdx.x / LANES : 0, GROUPS = LANES <= 32 ? (int)blockDim.x / LANES : 1;
    const int WARP = LANES <= 32 ? 0 : (int)(threadIdx.x >> 5), WARPS = LANES <= 32 ? 1 : (int)(blockDim.x >> 5);
    const auto SYNC = [&]() { if (LANES <= 32) __syncwarp(MASK); else __syncthreads(); };
    const auto WARP_SYNC = [&]() { __syncwarp(MASK); };
    const auto ALL = [&](bool p) -> bool { return LANES <= 32 ? __all_sync(MASK, p) : __syncthreads_and(p); };
    const auto ANY = [&](bool p) -> bool { return LANES <= 32 ? __any_sync(MASK, p) : __syncthreads_or(p); };
    const auto BALLOT = [&](bool p) -> unsigned {
        return __ballot_sync(MASK, p) >> (threadIdx.x & (31 & ~(WIDTH - 1)));
    };
    const auto BELOW = [&](unsigned bits) -> int { return __popc(bits & ((1u << (threadIdx.x & (WIDTH - 1))) - 1u)); };
    const auto COUNT = [](auto bits) -> int {
        if constexpr (sizeof(bits) == 8) return __popcll(bits); else return __popc(bits);
    };
    const auto FIRST = [](auto bits) -> int {
        if constexpr (sizeof(bits) == 8) return __ffsll(bits) - 1; else return __ffs(bits) - 1;
    };
    const auto READ = [&](auto value, int source) { return __shfl_sync(MASK, value, source, WIDTH); };
    const auto SEGMENT_SUM = [&](int width, auto&... values) {
        for (int offset = width >> 1; offset > 0; offset >>= 1)
            ((values += __shfl_xor_sync(MASK, values, offset, width)), ...);
    };
    const auto SUM = [&](auto&... values) { SEGMENT_SUM(WIDTH, values...); };
    const auto MINIMUM = [&](auto&... values) {
        for (int offset = WIDTH >> 1; offset > 0; offset >>= 1)
            ([&](auto& value) {
                const auto other = __shfl_xor_sync(MASK, value, offset, WIDTH);
                if (other < value) value = other;
            }(values), ...);
    };
    const auto MAXIMUM = [&](auto&... values) {
        for (int offset = WIDTH >> 1; offset > 0; offset >>= 1)
            ([&](auto& value) {
                const auto other = __shfl_xor_sync(MASK, value, offset, WIDTH);
                if (other > value) value = other;
            }(values), ...);
    };
    const auto ARGMAX = [&](auto& value, int& index) {
        for (int offset = WIDTH >> 1; offset > 0; offset >>= 1) {
            const auto other = __shfl_xor_sync(MASK, value, offset, WIDTH);
            const int other_index = __shfl_xor_sync(MASK, index, offset, WIDTH);
            if (other > value || (other == value && other_index > index)) {
                value = other;
                index = other_index;
            }
        }
    };
    const auto SCAN = [&](int value) -> int {
        int sum = value;
        for (int offset = 1; offset < WIDTH; offset <<= 1) {
            const int lower = __shfl_up_sync(MASK, sum, offset, WIDTH);
            if ((int)(threadIdx.x & (WIDTH - 1)) >= offset) sum += lower;
        }
        return sum - value;
    };
    const auto RESERVE = [&](int* counter, int n) -> int {
        int first = 0;
        if (LANE == 0 && n != 0) first = atomicAdd(counter, n);
        return READ(first, 0);
    };
    const auto ORDERED = [&](int& base, bool take) -> int {
        const unsigned votes = BALLOT(take);
        const int slot = base + BELOW(votes);
        base += COUNT(votes);
        return slot;
    };"""

# One thread runs the whole group on the CPU.
_LANE_GROUP_CPU = """    const int LANES = 1, WIDTH = 1;
    const unsigned MASK = 1u;
    const int LANE = 0, STRIDE = 1, SUB = 0, GROUPS = 1, WARP = 0, WARPS = 1;
    const auto SYNC = []() {};
    const auto WARP_SYNC = []() {};
    const auto ALL = [](bool p) -> bool { return p; };
    const auto ANY = [](bool p) -> bool { return p; };
    const auto BALLOT = [](bool p) -> unsigned { return p ? 1u : 0u; };
    const auto BELOW = [](unsigned bits) -> int { return 0; };
    const auto COUNT = [](auto bits) -> int {
        int count = 0;
        for (; bits != 0; bits &= bits - 1) ++count;
        return count;
    };
    const auto FIRST = [](auto bits) -> int {
        if (bits == 0) return -1;
        int first = 0;
        for (; (bits & 1) == 0; bits >>= 1) ++first;
        return first;
    };
    const auto READ = [](auto value, int source) { return value; };
    const auto SEGMENT_SUM = [](int width, auto&... values) {};
    const auto SUM = [](auto&... values) {};
    const auto MINIMUM = [](auto&... values) {};
    const auto MAXIMUM = [](auto&... values) {};
    const auto ARGMAX = [](auto& value, int& index) {};
    const auto SCAN = [](int value) -> int { return 0; };
    const auto RESERVE = [](int* counter, int n) -> int {
        const int first = *counter;
        *counter += n;
        return first;
    };
    const auto ORDERED = [](int& base, bool take) -> int {
        const int slot = base;
        base += take ? 1 : 0;
        return slot;
    };"""


@cache
def lane_group(lanes: int | str) -> DevicePrelude:
    """Return the lane-group prelude of ``lanes`` threads.

    Args:
        lanes: The group's lane count: a power of two up to 32 for a group within a warp, or the block size, in
            whole warps, for a block group; an integer or a native expression (such as an argument name) of one,
            evaluated at run time.
    """
    if isinstance(lanes, int) and (lanes < 1 or (lanes & (lanes - 1) if lanes <= 32 else lanes % 32)):
        raise ValueError(f"a lane group needs a power of two up to 32 lanes or whole warps, got {lanes}")
    return DevicePrelude(f"lane_group_{lanes}", substitute(_LANE_GROUP, lanes=lanes), _LANE_GROUP_CPU)


SHARED_STORAGE = DevicePrelude("shared_storage", "    #define SHARED __shared__", "    #define SHARED")
"""``SHARED``, the storage of a block's working set (:meth:`.staging.Staging.declarations`): CUDA's shared memory, or
the stack of the CPU's one thread, which runs the block's lane groups."""


# ---------------------------------------------------------------------------------------------------------------------
# Patch rings
# ---------------------------------------------------------------------------------------------------------------------

_RING = """    // A patch ring: rows linked into a circular list by parent(row), negative for a row without one. The walks
    // visit the rows after first in ring order, at most hops of them: RING_LOAD adds their value(row) to load,
    // RING_ANY tells whether predicate(row) holds for one of them.
    const auto RING_LOAD = [](float load, int first, int hops, auto parent, auto value) {
        int hop = 0;
        for (int row = parent(first); row >= 0 && row != first && hop < hops; row = parent(row), ++hop)
            load += value(row);
        return load;
    };
    const auto RING_ANY = [](int first, int hops, auto parent, auto predicate) {
        int hop = 0;
        for (int row = parent(first); row >= 0 && row != first && hop < hops; row = parent(row), ++hop)
            if (predicate(row)) return true;
        return false;
    };"""

RING = DevicePrelude("ring", _RING, _RING)
"""The patch ring's walks, ``RING_LOAD(load, first, hops, parent, value)`` and ``RING_ANY``, for native sources."""


ORDERED_TILES = DevicePrelude("ordered_tiles", ORDERED_CUDA, ORDERED_CPU)
"""Ordered reservation across a launch's blocks (``ORDERED_TILE``, ``ORDERED_NEXT``, ``ORDERED_SCAN``), for native
sources; :class:`OrderedTiles` holds a call site's claim counter, epoch and tile states."""


# ---------------------------------------------------------------------------------------------------------------------
# Slot reservations
# ---------------------------------------------------------------------------------------------------------------------


@wp.func
def reserve(counter: wp.array[int], index: int, n: int, capacity: int, first_rejected: wp.array[int]) -> int:
    """Reserve ``n`` consecutive slots of ``counter[index]`` below ``capacity``, all or none: the first slot, or -1.

    The overflow contract of every row family: counters only grow, so the counter keeps the demand; a reservation
    that does not fit lowers ``first_rejected[index]`` (preset to a large value) to its first slot, and
    :func:`reserved_count` truncates the count there and raises the family's status, so no accepted reservation
    holds a slot at or past the count and none is partial. Slots are reserved in one of three modes under this
    contract:

    - *atomic*: this function, one atomic per reservation; slots follow the order in which threads reach the
      counter;
    - *warp-aggregated*: the lane group's ``RESERVE``, one atomic for a group's one-slot reservations, which take
      their slots in lane order;
    - *ordered*: slots in input order. A group that owns its counter places its lanes' slots with the lane
      group's ``ORDERED(base, take)`` and writes the counter; blocks that share a counter claim tiles in input
      order and scan them (:func:`ordered_tile` or :func:`ordered_next`, then :func:`ordered_scan`; natively
      ``ORDERED_TILES``), so no output order depends on the GPU's scheduling.

    A one-slot reservation past capacity starts at or past it, so the last two need no first-rejected slot; an
    ordered reservation rejects the inputs last in input order.
    """
    slot = wp.atomic_add(counter, index, n)
    if slot + n > capacity:
        wp.atomic_min(first_rejected, index, slot)
        return -1
    return slot


@wp.func
def reserved_count(
    counter: wp.array[int],
    index: int,
    capacity: int,
    first_rejected: wp.array[int],
    status: wp.array[int],
    family: int,
) -> int:
    """Return the slots of ``counter[index]`` its reservations hold, raising ``status[family]`` when demand passed capacity.

    The status is sticky: only the caller that owns it clears it.
    """
    count = counter[index]
    if count > capacity:
        wp.atomic_max(status, family, 1)
        count = capacity
    return wp.min(count, first_rejected[index])


# ---------------------------------------------------------------------------------------------------------------------
# Warp functions from the same text
# ---------------------------------------------------------------------------------------------------------------------


@wp.func_native(dual_native("    MINIMUM(value);\n    return value;\n", preludes=(lane_group("lanes"),)))
def lane_minimum(value: float, lanes: int) -> float:
    """Return the least ``value`` of this thread's group of ``lanes`` lanes (``value`` on the CPU).

    Every lane of the group must call it with the same power-of-two ``lanes`` of at most 32.
    """


@wp.func_native(dual_native("    SUM(value);\n    return value;\n", preludes=(lane_group(32),)))
def warp_sum(value: int) -> int:
    """Return the sum of ``value`` over this thread's warp (``value`` on the CPU, where a thread is alone).

    Every lane of the warp must call it.
    """


@wp.func_native(dual_native("    return SCAN(value);\n", preludes=(lane_group(32),)))
def warp_scan(value: int) -> int:
    """Return the sum of ``value`` over the lower lanes of this thread's warp (0 on the CPU).

    Every lane of the warp must call it.
    """


@wp.func_native(dual_native("    SYNC();\n", preludes=(lane_group(32),)))
def warp_sync():
    """Synchronize this thread's warp (nothing on the CPU)."""


@wp.func_native(dual_native("    return LANE;\n", preludes=(lane_group(32),)))
def warp_lane() -> int:
    """Return this thread's lane in its warp (0 on the CPU)."""


@wp.func_native(dual_native("    return STRIDE;\n", preludes=(lane_group(32),)))
def warp_stride() -> int:
    """Return the stride of a warp-strided loop: 32, or 1 on the CPU, where one thread runs the warp's lanes."""


@wp.func_native(
    dual_native(
        "    return wp::vec3(READ(value[0], 0), READ(value[1], 0), READ(value[2], 0));\n", preludes=(lane_group(32),)
    )
)
def warp_broadcast(value: wp.vec3) -> wp.vec3:
    """Return lane 0's ``value`` on every lane of this thread's warp (``value`` on the CPU).

    Every lane of the warp must call it.
    """
