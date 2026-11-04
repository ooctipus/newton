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

    Its further words are preludes of their own, declared by the bodies that
    call them: ``LANE_WORDS`` (``WARP_LANE``, ``MATCH``, ``READ_DOWN`` and the
    block group's ``BLOCK_SUM``, ``BLOCK_MAXIMUM``, ``BLOCK_ARGMAX``), ``HELD``
    (a distributed array over the lanes: ``HELD_PICK``, ``HELD_ROW``,
    ``HELD_LOAD``), ``LANE_FEED`` (``FEED``, a held batch feeding one chain)
    and ``MEMORY_WORDS`` (``LDG``, ``ST``, ``ADD``). Warp functions over the
    words take their snippet from :func:`lane_word`.

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


def block_groups(per_block: int, count: str, body: str) -> str:
    """Return native statements that run ``body`` for this block's lane groups below ``count``, ``per_block`` a block.

    On CUDA each lane group runs its own group, ``block * per_block + SUB``, and the body's ``return`` ends the thread;
    on the CPU one thread runs the block's groups in order, each body in a lambda whose ``return`` ends that group.
    The body reads its group as ``group``. After a lane-group prelude, in a body whose dispatch passes ``block``
    (:data:`.kernel_template.BLOCKS`).

    Args:
        per_block: Lane groups a block.
        count: The native count of groups.
        body: The group's statements, each line newline-terminated.
    """
    cuda = f"    const int group = block * {per_block} + SUB;\n    if (group >= {count}) return;\n"
    cpu = (
        "    // One CPU thread runs the block's groups in order.\n"
        f"    for (int group = block * {per_block}; group < (block + 1) * {per_block} && group < {count}; ++group) [&] {{\n"
    )
    return device_branches(cuda, cpu) + body + device_branches("", "    }();\n")


_LANE_FEED = """    // FEED(first, last, step, ahead, build, chain): a lane group feeding one serial chain over the records
    // [first, last] of a tree, with no global scratch. step -1 walks from last down to first (leaves to root), +1 from
    // first up to last (root to leaves). First the group strides over every record calling ahead(index) (its lines'
    // prefetch: every load of the chain's inputs leaves the chain). Then, in batches of LANES records in the walk's
    // order, lane k calls build(index, valid) for the batch's k-th record (valid false past the tree: return zeros);
    // build loads and forms what the chain needs of that record and returns it as a struct whose one member is
    // float w[N] (ints as their bits). Every lane then calls chain(index, record) for each record of the batch in
    // order, the record read from its lane (a batch is a held array of one record a lane, HELD_ROW), so every lane
    // runs the same chain and its state stays in registers. On the CPU one thread is the group: batches of one record.
    const auto FEED = [&](int first, int last, int step, auto ahead, auto build, auto chain) {
        const int total = last - first + 1;
        const int start = step < 0 ? last : first;
        for (int k = LANE; k < total; k += LANES) ahead(start + step * k);
        for (int done = 0; done < total; done += LANES) {
            const int base = start + step * done;
            const auto mine = build(base + step * LANE, LANE < total - done);
            using Words = decltype(mine.w);
            const int count = total - done < LANES ? total - done : LANES;
            for (int i = 0; i < count; ++i) {
                auto record = mine;
                HELD_ROW(reinterpret_cast<const Words (&)[1]>(mine.w), i, record.w);
                chain(base + step * i, record);
            }
        }
    };"""

_HELD = """    // HELD: a distributed array over the group's lanes, item x on lane x % WIDTH in word x / WIDTH of the lane's
    // array (ceil(n / WIDTH) words for n items; on the CPU the one lane holds every item). x is the same on every lane:
    // HELD_PICK(v, x) returns item x of the held array v, HELD_ROW(v, x, out) writes item x's row of a held array of
    // rows (v[word][m]) into out, and HELD_LOAD(v, n, load) calls load(word, x, x < n) for each word of the lane's
    // array v and its item x, so the load fills each word (zeros past the n items).
    const auto HELD_PICK = [&](const auto& v, const int x) {
        constexpr int words = sizeof(v) / sizeof(v[0]);
        auto t = v[0];
        #pragma unroll
        for (int k = 1; k < words; ++k)
            if (x >= k * WIDTH) t = v[k];
        return READ(t, x);
    };
    const auto HELD_ROW = [&](const auto& v, const int x, auto* out) {
        constexpr int words = sizeof(v) / sizeof(v[0]), row = sizeof(v[0]) / sizeof(v[0][0]);
        #pragma unroll
        for (int m = 0; m < row; ++m) {
            auto t = v[0][m];
            #pragma unroll
            for (int k = 1; k < words; ++k)
                if (x >= k * WIDTH) t = v[k][m];
            out[m] = READ(t, x);
        }
    };
    const auto HELD_LOAD = [&](const auto& v, const int n, auto load) {
        constexpr int words = sizeof(v) / sizeof(v[0]);
        const int lane = LANE & (WIDTH - 1);
        #pragma unroll
        for (int k = 0; k < words; ++k) load(k, lane + k * WIDTH, lane + k * WIDTH < n);
    };"""

HELD = DevicePrelude("held", _HELD, _HELD)
"""``HELD_PICK(v, x)``, ``HELD_ROW(v, x, out)`` and ``HELD_LOAD(v, n, load)``: a distributed array over a lane group,
item ``x`` on lane ``x % WIDTH`` in the lane's word ``x / WIDTH``, read on every lane by shuffle from the lane that
holds it; on the CPU the one lane holds every item. After a lane-group prelude; :data:`LANE_FEED` carries it."""

LANE_FEED = DevicePrelude("lane_feed", _HELD + "\n" + _LANE_FEED, _HELD + "\n" + _LANE_FEED)
"""``FEED(first, last, step, ahead, build, chain)``, a lane group feeding one serial chain (the walk's refresh leaves
to root, its walk down root to leaves): its records' loads leave the chain, each lane builds one record of a batch, and
every lane runs the chain on the batch's records in order, read from the batch held over the lanes. It carries
:data:`HELD` (a body declares one of them). After a lane-group prelude."""


_LANE_WORDS = """    // WARP_LANE: this thread's lane in its warp. MATCH(key): the bits of the group's lanes holding the same key,
    // lane 0 first. READ_DOWN(value, delta): the value of the lane delta above (its own past the group's top).
    // BLOCK_SUM(scratch, value), BLOCK_MAXIMUM(scratch, value), BLOCK_ARGMAX(scratch, indices, value, index): the
    // group's reduction on every lane, for a block group through the block's per-warp scratch (WARPS entries, the
    // body's Staging) and two barriers.
    const int WARP_LANE = (int)(threadIdx.x & 31);
    const auto MATCH = [&](auto key) -> unsigned {
        return __match_any_sync(MASK, key) >> (threadIdx.x & (31 & ~(WIDTH - 1)));
    };
    const auto READ_DOWN = [&](auto value, int delta) { return __shfl_down_sync(MASK, value, delta, WIDTH); };
    const auto BLOCK_SUM = [&](auto* scratch, auto& value) {
        SUM(value);
        if (LANES > 32) {
            if (WARP_LANE == 0) scratch[WARP] = value;
            __syncthreads();
            value = 0;
            if (WARP_LANE < WARPS) value = scratch[WARP_LANE];
            SUM(value);
            __syncthreads();
        }
    };
    const auto BLOCK_MAXIMUM = [&](auto* scratch, auto& value) {
        MAXIMUM(value);
        if (LANES > 32) {
            if (WARP_LANE == 0) scratch[WARP] = value;
            __syncthreads();
            value = scratch[WARP_LANE < WARPS ? WARP_LANE : 0];
            MAXIMUM(value);
            __syncthreads();
        }
    };
    const auto BLOCK_ARGMAX = [&](auto* scratch, int* indices, auto& value, int& index) {
        ARGMAX(value, index);
        if (LANES > 32) {
            if (WARP_LANE == 0) {
                scratch[WARP] = value;
                indices[WARP] = index;
            }
            __syncthreads();
            const int from = WARP_LANE < WARPS ? WARP_LANE : 0;
            value = scratch[from];
            index = indices[from];
            ARGMAX(value, index);
            __syncthreads();
        }
    };"""

_LANE_WORDS_CPU = """    const int WARP_LANE = 0;
    const auto MATCH = [](auto key) -> unsigned { return 1u; };
    const auto READ_DOWN = [](auto value, int delta) { return value; };
    const auto BLOCK_SUM = [](auto* scratch, auto& value) {};
    const auto BLOCK_MAXIMUM = [](auto* scratch, auto& value) {};
    const auto BLOCK_ARGMAX = [](auto* scratch, int* indices, auto& value, int& index) {};"""

LANE_WORDS = DevicePrelude("lane_words", _LANE_WORDS, _LANE_WORDS_CPU)
"""The lane group's further words: ``WARP_LANE``, ``MATCH(key)``, ``READ_DOWN(value, delta)`` and the block group's
reductions ``BLOCK_SUM``, ``BLOCK_MAXIMUM`` and ``BLOCK_ARGMAX`` (each warp's, then every warp's through a scratch of
``WARPS`` entries the body declares). On the CPU the one thread holds the group's every value. Their own prelude, not
the lane group's, as :data:`STATUS_CLOCK`'s. After a lane-group prelude."""


MEMORY_WORDS = DevicePrelude(
    "memory_words",
    """    const auto LDG = [](const auto* p) { return __ldg(p); };
    const auto ST = [](auto* p, const auto v) {
        __builtin_assume(__isGlobal(p));
        *p = v;
    };
    const auto ADD = [](auto& word, auto n) { return atomicAdd(&word, n); };""",
    """    const auto LDG = [](const auto* p) { return *p; };
    const auto ST = [](auto* p, const auto v) { *p = v; };
    const auto ADD = [](auto& word, auto n) {
        const auto old = word;
        word += n;
        return old;
    };""",
)
"""``LDG(p)``, a read-only load through the non-coherent path, ``ST(p, v)``, a store to global memory (no generic
store), and ``ADD(word, n)``, an add to a word that any thread may write at once (CUDA's ``atomicAdd``; the CPU's one
thread adds), returning the word before. The CPU's are plain loads, stores and adds."""


STATUS_CLOCK = DevicePrelude(
    "status_clock",
    """    const auto RAISE = [](auto& word, auto bits) { atomicOr(&word, bits); };
    const auto CLOCK = []() -> long long { return clock64(); };""",
    """    const auto RAISE = [](auto& word, auto bits) { word |= bits; };
    const auto CLOCK = []() -> long long { return 0; };""",
)
"""``RAISE(word, bits)`` sets ``bits`` in a status word that any thread may write at once (CUDA's ``atomicOr``, a plain
``|=`` of the CPU's one thread), and ``CLOCK()`` reads the cycle counter (0 on the CPU). Its own prelude, not the lane
group's: a body that declares words it does not call can still schedule differently."""


SHARED_STORAGE = DevicePrelude("shared_storage", "    #define SHARED __shared__", "    #define SHARED")
"""``SHARED``, the storage of a block's working set (:meth:`.staging.Staging.declarations`): CUDA's shared memory, or
the stack of the CPU's one thread, which runs the block's lane groups."""


CPU_MATH = DevicePrelude(
    "cpu_math",
    "    // The CUDA intrinsics the bodies call: the device's own.",
    """    // The CUDA intrinsics the bodies call, on the CPU, whose build declares none of them: Warp's own minimum and
    // maximum, the bit casts, the first set bit, the one thread's plain bit or, the 128-bit word.
    const auto fmaxf = [](float a, float b) -> float { return wp::max(a, b); };
    const auto fminf = [](float a, float b) -> float { return wp::min(a, b); };
    const auto __int_as_float = [](int i) -> float { float f; memcpy(&f, &i, sizeof(f)); return f; };
    const auto __float_as_int = [](float f) -> int { int i; memcpy(&i, &f, sizeof(i)); return i; };
    const auto __ffs = [](unsigned bits) -> int { return __builtin_ffs(static_cast<int>(bits)); };
    const auto atomicOr = [](unsigned* word, unsigned bits) -> unsigned {
        const unsigned old = *word;
        *word = old | bits;
        return old;
    };
    struct int4 { int x, y, z, w; };""",
)
"""The CUDA intrinsics the bodies call, for a body that also runs on the CPU (:func:`dual_native`): ``fminf`` and
``fmaxf`` are Warp's ``wp::min`` and ``wp::max`` there, the same up to the sign of a zero; ``atomicOr`` is a plain
``|=`` of the one thread that runs the block; ``int4`` is a struct of four ints. CUDA has its own."""


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


def lane_word(body: str, lanes: int | str = 32, preludes: tuple[DevicePrelude, ...] = ()) -> str:
    """The ``func_native`` snippet of a Warp function over a lane group's words: ``body`` after the group of ``lanes``
    lanes (:func:`lane_group`) and ``preludes``, on both devices (:func:`dual_native`).

    Args:
        body: The function's native statements.
        lanes: The group's lanes: a count, or the name of the function's argument that holds it.
        preludes: Further preludes the body reads, after the lane group.
    """
    return dual_native(body, preludes=(lane_group(lanes), *preludes))


@wp.func_native(lane_word("    MINIMUM(value);\n    return value;\n", "lanes"))
def lane_minimum(value: float, lanes: int) -> float:
    """Return the least ``value`` of this thread's group of ``lanes`` lanes (``value`` on the CPU).

    Every lane of the group must call it with the same power-of-two ``lanes`` of at most 32.
    """


@wp.func_native(lane_word("    SUM(value);\n    return value;\n"))
def warp_sum(value: int) -> int:
    """Return the sum of ``value`` over this thread's warp (``value`` on the CPU, where a thread is alone).

    Every lane of the warp must call it.
    """


@wp.func_native(lane_word("    return SCAN(value);\n"))
def warp_scan(value: int) -> int:
    """Return the sum of ``value`` over the lower lanes of this thread's warp (0 on the CPU).

    Every lane of the warp must call it.
    """


@wp.func_native(lane_word("    SYNC();\n"))
def warp_sync():
    """Synchronize this thread's warp (nothing on the CPU)."""


@wp.func_native(lane_word("    return LANE;\n"))
def warp_lane() -> int:
    """Return this thread's lane in its warp (0 on the CPU)."""


@wp.func_native(lane_word("    return STRIDE;\n"))
def warp_stride() -> int:
    """Return the stride of a warp-strided loop: 32, or 1 on the CPU, where one thread runs the warp's lanes."""


@wp.func_native(lane_word("    return wp::vec3(READ(value[0], 0), READ(value[1], 0), READ(value[2], 0));\n"))
def warp_broadcast(value: wp.vec3) -> wp.vec3:
    """Return lane 0's ``value`` on every lane of this thread's warp (``value`` on the CPU).

    Every lane of the warp must call it.
    """
