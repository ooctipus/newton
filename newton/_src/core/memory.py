# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The memory toolkit: where a kernel's operands live, and the words that reach them.

The math and physics of a kernel never name a cache detail. A kernel declares each operand's :class:`Place` (a lane's
registers, a block's shared stage, a constant table, global memory read-only, coherent, streamed ahead or kept in
L2) and its body reads and writes the operand through that place's words (:func:`read`):
the place is data, chosen by a rule, and the body's text does not change with it. This module is
the one place every such concept lives:

- the device: :class:`Device`, its memory hierarchy and occupancy, read once (:func:`device`);
- the shared stage: :class:`Staging` (a block's working set, declared buffer by buffer) under the device's static
  budget (:func:`static_budget`), the rule that stages what pays (:func:`stage`) and the carveout (:func:`carveout`);
- the access words, one text each with its CPU twin: ``LDG`` (L1's non-coherent path), ``LD``/``ST``/``ADD``
  (coherent global loads, stores and adds), ``TOUCH`` and ``FETCH`` (loads ahead: one L1 hint a line, and a lane
  group's fetch of a byte range), ``LDCG``/``STCG`` (L2 past L1), ``HELD`` (a distributed array over a lane group's
  registers) and ``SHARED`` (the stage's storage);
- constant tables (:func:`table`): one text, ``__constant__`` on CUDA and ``const`` on the CPU.

The texts are plain native statements; :mod:`newton._src.solvers.feather_pgs.device_preludes` composes them into a
body's preludes, as it does the ordered reservation's.
"""

from __future__ import annotations

import dataclasses
import enum
import functools
from collections.abc import Callable

import warp as wp

__all__ = [
    "CONSTANT_STORAGE",
    "CPU",
    "CPU_STAGING_BYTES",
    "HELD",
    "HIDING_WARPS",
    "L1_LINE_BYTES",
    "REFERENCE_ARCH",
    "REGISTERS_PER_THREAD",
    "REGISTER_GRANULE",
    "REGISTER_PARTITIONS",
    "RESERVED_SHARED_BYTES",
    "SHARED_ALLOCATION_BYTES",
    "SHARED_STORAGE",
    "STATIC_SHARED_BUDGETS",
    "STATIC_SHARED_BYTES",
    "TILE_BYTES_PER_THREAD",
    "Buffer",
    "Device",
    "Operand",
    "Place",
    "Staging",
    "arch_budget",
    "built_for",
    "carveout",
    "device",
    "granules",
    "read",
    "stage",
    "static_budget",
    "table",
    "words",
]

# ---------------------------------------------------------------------------------------------------------------------
# Device facts
# ---------------------------------------------------------------------------------------------------------------------

STATIC_SHARED_BYTES = 48 * 1024
"""Static shared memory a block may declare in an offline build [B]: ptxas 13.2 rejects more on every architecture
(0xc000)."""

CPU_STAGING_BYTES = 4 * 1024 * 1024
"""The working set a CPU block may declare [B]: the CPU's one thread runs the block and keeps it on its stack (an
automatic array), so no shared-memory budget applies; this bound keeps a block well inside a thread's stack."""

RESERVED_SHARED_BYTES = 1024
"""Shared memory the driver reserves in every block [B] (sm_80 and later), out of the opt-in maximum per block."""

STATIC_SHARED_BUDGETS: dict[int, int] = {}
"""Static shared memory budgets [B] by architecture (``Device.arch``) that a probe measured below the opt-in
maximum less the driver's reservation; :func:`static_budget` reads them first. None is recorded: the RTX PRO 6000
(sm_120, opt-in 101,376 B) loads a kernel declaring 100,352 B and refuses one declaring 101,376 B, the GB300 (sm_103)
loads 231,424 B and refuses 232,448 B (measured 2026-10-27)."""

REFERENCE_ARCH = 120
"""The architecture an offline build that names none is sized for (:func:`built_for`): the RTX PRO 6000's (sm_120),
the smallest of the suite's GPUs."""

SHARED_ALLOCATION_BYTES = 128
"""Granule in which an SM allocates a block's shared memory [B] (``cuda_occupancy.h``, sm_80 and later)."""

REGISTERS_PER_THREAD = 255
"""The most registers a thread holds (every architecture since sm_35): the budget of a kernel without a bound."""

REGISTER_PARTITIONS = 4
"""Sub-partitions a multiprocessor's register file divides into (every architecture since sm_70): a warp takes its
registers from one, so a register line keeps whole warps a partition (``cuda_occupancy.h``)."""

REGISTER_GRANULE = 256
"""Registers a warp is allocated in (``cuda_occupancy.h``, sm_70 and later)."""

_CARVEOUTS_128 = (0, 8, 16, 32, 64, 100)
_CARVEOUTS_192 = (0, 8, 16, 32, 64, 100, 132, 164)
_CARVEOUTS_256 = (0, 8, 16, 32, 64, 100, 132, 164, 196, 228)

_ARCHITECTURES: dict[int, tuple[int, int, tuple[int, ...], int]] = {
    75: (1024, 16, (0, 32, 64), 96),
    80: (2048, 32, _CARVEOUTS_192, 192),
    86: (1536, 16, _CARVEOUTS_128, 128),
    87: (2048, 16, _CARVEOUTS_192, 192),
    89: (1536, 24, _CARVEOUTS_128, 128),
    90: (2048, 32, _CARVEOUTS_256, 256),
    100: (2048, 32, _CARVEOUTS_256, 256),
    103: (2048, 32, _CARVEOUTS_256, 256),
    120: (1536, 24, _CARVEOUTS_128, 128),
    121: (1536, 24, _CARVEOUTS_128, 128),
}
"""A multiprocessor's occupancy limits by architecture (CUDA's occupancy tables): resident threads, resident blocks,
the driver's shared-memory carveout steps [KiB] and the unified L1/shared memory they divide [KiB]. Warp reports
neither; every architecture keeps 65,536 registers a multiprocessor."""


@dataclasses.dataclass(frozen=True)
class Device:
    """What the execution layer and the memory toolkit read of a device, once (:func:`device`).

    Args:
        arch: The CUDA architecture (``120`` for sm_120), 0 for the CPU.
        sms: Multiprocessors.
        threads_per_sm: Threads a multiprocessor keeps resident.
        blocks_per_sm: Blocks a multiprocessor keeps resident.
        shared_per_block: Static shared memory a block may declare [B] (:func:`static_budget`).
        carveouts_kb: The driver's shared-memory carveout steps [KiB], ascending; the last is a multiprocessor's
            shared memory.
        unified_kb: The unified L1 and shared memory a multiprocessor divides between them [KiB].
        registers_per_sm: Registers a multiprocessor holds.
    """

    arch: int
    sms: int
    threads_per_sm: int
    blocks_per_sm: int
    shared_per_block: int = STATIC_SHARED_BYTES
    carveouts_kb: tuple[int, ...] = (0,)
    unified_kb: int = 0
    registers_per_sm: int = 65536

    @property
    def cuda(self) -> bool:
        """Whether the device is a GPU (the CPU runs a lane group on one thread)."""
        return self.arch > 0

    @property
    def shared_per_sm(self) -> int:
        """Shared memory a multiprocessor gives its resident blocks [B]: the largest carveout."""
        return self.carveouts_kb[-1] * 1024

    @classmethod
    def of(cls, arch: int, sms: int, opt_in_shared: int) -> Device:
        """The device of a CUDA architecture with ``sms`` multiprocessors and an opt-in maximum of ``opt_in_shared``
        bytes of shared memory a block (Warp's ``max_shared_memory_per_block``), the rest from the architecture's
        occupancy table."""
        if arch not in _ARCHITECTURES:
            raise ValueError(f"no occupancy table for sm_{arch}: add the architecture to execution._ARCHITECTURES")
        threads, blocks, carveouts, unified = _ARCHITECTURES[arch]
        budget = STATIC_SHARED_BUDGETS.get(arch, int(opt_in_shared) - RESERVED_SHARED_BYTES)
        return cls(arch, int(sms), threads, blocks, budget, carveouts, unified)


CPU = Device(0, 1, 1, 1, CPU_STAGING_BYTES)
"""The CPU: one thread runs a lane group and a block, never timed or tuned."""


@functools.cache
def _read(alias: str) -> Device:
    found = wp.get_device(alias)
    if not found.is_cuda:
        return CPU
    return Device.of(int(found.arch), int(found.sm_count), int(found.max_shared_memory_per_block))


def device(identifier=None) -> Device:
    """The :class:`Device` of a Warp device (by default the current one), read once."""
    return _read(wp.get_device(identifier).alias)


def _architecture(device_arch) -> int:
    """The architecture number of a kernel factory's ``device_arch`` (a number or an offline ``"sm_XX"``), or -1 for
    one that names none (an offline build without an architecture)."""
    try:
        return int(str(device_arch).removeprefix("sm_"))
    except (TypeError, ValueError):
        return -1


def _of_architecture(arch: int) -> Device | None:
    """The first CUDA device of the architecture at hand, or ``None`` (an offline build, the CPU)."""
    for found in wp.get_cuda_devices() if arch > 0 else ():
        if found.arch == arch:
            return device(found)
    return None


def built_for(device_arch) -> Device:
    """The device a kernel factory's ``device_arch`` (a number or an offline ``"sm_XX"``) builds for: a CUDA device of
    the architecture at hand, else the architecture's occupancy table (an offline build; one that names no
    architecture is sized for :data:`REFERENCE_ARCH`); :data:`CPU` for architecture 0."""
    arch = _architecture(device_arch)
    if arch == 0:
        return CPU
    arch = arch if arch > 0 else REFERENCE_ARCH
    return _of_architecture(arch) or Device.of(arch, 1, STATIC_SHARED_BYTES + RESERVED_SHARED_BYTES)


def static_budget(arch: int) -> int:
    """Static shared memory a block of a kernel built for the CUDA architecture ``arch`` may declare [B].

    The one place a device's static budget is read: a probe's measured budget of the architecture
    (:data:`STATIC_SHARED_BUDGETS`), else the opt-in maximum per block that Warp reports for a CUDA device of the
    architecture less the driver's reservation (:attr:`Device.shared_per_block`), else, with no such device (an
    offline build, the CPU), :data:`STATIC_SHARED_BYTES`.
    """
    if arch in STATIC_SHARED_BUDGETS:
        return int(STATIC_SHARED_BUDGETS[arch])
    found = _of_architecture(arch)
    return found.shared_per_block if found is not None else STATIC_SHARED_BYTES


def arch_budget(device_arch) -> int:
    """The working set a block of a kernel factory's ``device_arch`` may declare [B]: :func:`static_budget` of a CUDA
    architecture number, :data:`CPU_STAGING_BYTES` for the CPU (architecture 0) and :data:`STATIC_SHARED_BYTES` for an
    offline build that names none."""
    arch = _architecture(device_arch)
    if arch < 0:
        return STATIC_SHARED_BYTES
    return static_budget(arch) if arch > 0 else CPU_STAGING_BYTES


def granules(block_bytes: int) -> int:
    """Shared memory a block takes of its multiprocessor [B]: its bytes and the driver's reservation, in granules."""
    granules = -(-(int(block_bytes) + RESERVED_SHARED_BYTES) // SHARED_ALLOCATION_BYTES)
    return granules * SHARED_ALLOCATION_BYTES


def carveout(device: Device, block: int, shared: int) -> int:
    """The driver's carveout step [KiB] for blocks of ``block`` threads and ``shared`` bytes: the smallest step that
    holds the blocks the thread and block limits allow (H8: the PRO keeps 64 KiB for 32-thread blocks of 1,664 B,
    takes 100 KiB at 2,200 B); 0 on the CPU."""
    if not device.cuda:
        return 0
    need = min(device.blocks_per_sm, device.threads_per_sm // block) * granules(shared)
    for step in device.carveouts_kb:
        if step * 1024 >= need:
            return step
    return device.carveouts_kb[-1]


# ---------------------------------------------------------------------------------------------------------------------
# Places and their words
# ---------------------------------------------------------------------------------------------------------------------

L1_LINE_BYTES = 128
"""Bytes of an L1 line (every architecture since sm_50): the unit ``TOUCH`` asks for and ``FETCH`` strides by."""


class Place(enum.Enum):
    """Where an operand lives: the plan's choice, which sets the words a body reaches it through."""

    REGISTERS = "registers"
    """A lane's registers; across a lane group, a distributed array (``HELD_PICK``, ``HELD_ROW``, ``HELD_LOAD``)."""
    SHARED = "shared"
    """A block's shared stage (:class:`Staging`, declared in ``SHARED`` storage), read and written by name."""
    CONSTANT = "constant"
    """A table every thread reads at the same index, folded from the spec (:func:`table`)."""
    READ_ONLY = "read_only"
    """Global memory no thread writes while the kernel reads it: L1's non-coherent path (``LDG``)."""
    GLOBAL = "global"
    """Global memory read coherently, stored and added to (``LD``, ``ST``, ``ADD``): data the kernel also writes, or
    reads where the non-coherent path measured slower (the K2 writer, 1.4% on sm_120)."""
    STREAMED = "streamed"
    """Global memory a chain reads in order: loaded ahead of its use, its lines asked of L1 first (``LDG`` with
    ``TOUCH`` a line, ``FETCH`` a lane group's byte range)."""
    PERSISTING = "persisting"
    """Global memory kept in L2 past L1 (``LDCG``, ``STCG``). Warp sets no L2 access-policy window, so this is the
    load and store kind only; like a carveout move, it is refused without a measurement (none uses it)."""


# Each word's lambda: its CUDA text and its CPU twin (the CPU's one thread: plain loads and stores, no hint), declared
# under its own name (words) or as an operand's (read).
_WORDS = {
    "LDG": ("[](const auto* p) { return __ldg(p); }", "[](const auto* p) { return *p; }"),
    "LD": (
        """[](const auto* p) {
        __builtin_assume(__isGlobal(p));
        return *p;
    }""",
        "[](const auto* p) { return *p; }",
    ),
    "ST": (
        """[](auto* p, const auto v) {
        __builtin_assume(__isGlobal(p));
        *p = v;
    }""",
        "[](auto* p, const auto v) { *p = v; }",
    ),
    "ADD": (
        "[](auto& word, auto n) { return atomicAdd(&word, n); }",
        """[](auto& word, auto n) {
        const auto old = word;
        word += n;
        return old;
    }""",
    ),
    "TOUCH": ('[](const auto p) { asm volatile("prefetch.global.L1 [%0];" ::"l"(p)); }', "[](const auto) {}"),
    "FETCH": (
        f"""[&](const void* start, const int bytes, const int lane, const int lanes) {{
        const size_t end = reinterpret_cast<size_t>(start) + (size_t)bytes;
        for (size_t line = (reinterpret_cast<size_t>(start) & ~(size_t){L1_LINE_BYTES - 1}) + (size_t){L1_LINE_BYTES} * lane;
             line < end; line += (size_t){L1_LINE_BYTES} * lanes)
            TOUCH(line);
    }}""",
        "[](const void*, const int, const int, const int) {}",
    ),
    "LDCG": ("[](const auto* p) { return __ldcg(p); }", "[](const auto* p) { return *p; }"),
    "STCG": ("[](auto* p, const auto v) { __stcg(p, v); }", "[](auto* p, const auto v) { *p = v; }"),
}

_LOAD = {Place.READ_ONLY: "LDG", Place.GLOBAL: "LD", Place.STREAMED: "LDG", Place.PERSISTING: "LDCG"}
_STORE = {Place.GLOBAL: "ST", Place.PERSISTING: "STCG"}


def _declare(declared) -> tuple[str, str]:
    # Each (name, word) of declared as the native statement const auto name = word's lambda, for CUDA and the CPU.
    declared = tuple(declared)
    return tuple("\n".join(f"    const auto {n} = {_WORDS[w][cpu]};" for n, w in declared) for cpu in (0, 1))


@dataclasses.dataclass(frozen=True)
class Operand:
    """An operand of a kernel in global memory, as the read rule takes it (:func:`read`).

    Args:
        name: The operand's name in the body: it loads ``LOAD_<name>(p)`` and, where the kernel writes it, stores
            ``STORE_<name>(p, v)``.
        writes: Whether a thread writes it while the kernel runs.
        measured: The global place a measurement chose over the rule's, or ``None``.
    """

    name: str
    writes: bool = False
    measured: Place | None = None

    def __post_init__(self):
        if self.measured is not None and self.measured not in (_STORE if self.writes else _LOAD):
            raise ValueError(f"operand {self.name}: {self.measured} is not a global place for it")


def words(*names: str) -> tuple[str, str]:
    """The named words alone, in one order, as ``(cuda, cpu)`` native statements: a body that calls a few words
    declares only those (a declared word it does not call can still move its schedule)."""
    unknown = set(names) - set(_WORDS)
    if unknown:
        raise ValueError(f"no memory words {sorted(unknown)}")
    return _declare((name, name) for name in _WORDS if name in names)


def read(*operands: Operand) -> tuple[str, str]:
    """The read rule: each operand's words by its place, ``LDG`` (:attr:`Place.READ_ONLY`) where no thread writes it
    while the kernel runs, ``LD`` and ``ST`` (:attr:`Place.GLOBAL`) where one does, or the place a measurement chose
    (``LDCG`` and ``STCG`` for data read once). As ``(cuda, cpu)`` native statements: each operand's ``LOAD_<name>``
    and, where the kernel writes it, ``STORE_<name>``, so its text does not change with its place."""
    declared = []
    for operand in operands:
        place = operand.measured or (Place.GLOBAL if operand.writes else Place.READ_ONLY)
        declared.append((f"LOAD_{operand.name}", _LOAD[place]))
        if operand.writes:
            declared.append((f"STORE_{operand.name}", _STORE[place]))
    return _declare(declared)


SHARED_STORAGE = ("    #define SHARED __shared__", "    #define SHARED")
"""``SHARED``, the storage of a block's working set (:meth:`Staging.declarations`): CUDA's shared memory, or the stack
of the CPU's one thread, which runs the block's lane groups."""


HELD = """    // HELD: a distributed array over the group's lanes, item x on lane x % WIDTH in word x / WIDTH of the lane's
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
"""``HELD_PICK(v, x)``, ``HELD_ROW(v, x, out)`` and ``HELD_LOAD(v, n, load)``: a lane group's registers as one
distributed array, item ``x`` on lane ``x % WIDTH`` in the lane's word ``x / WIDTH``, read on every lane by shuffle
from the lane that holds it; on the CPU the one lane holds every item. The same text on both devices, after a lane
group's words."""


CONSTANT_STORAGE = ("    #define CONSTANT __constant__", "    #define CONSTANT const")
"""``CONSTANT``, the storage of a constant table (:func:`table`): CUDA's constant memory, or the CPU's constant data."""


def table(name: str, values, ctype: str = "int") -> str:
    """A constant table ``name`` of ``values`` (one entry at least) that every thread reads at one index, in ``CONSTANT``
    storage (:data:`CONSTANT_STORAGE`): one declaration for both devices."""
    entries = [str(int(value) if ctype == "int" else value) for value in values]
    return f"static CONSTANT {ctype} {name}[{max(len(entries), 1)}] = {{{', '.join(entries)}}};"


# ---------------------------------------------------------------------------------------------------------------------
# The shared stage
# ---------------------------------------------------------------------------------------------------------------------

TILE_BYTES_PER_THREAD = 4
"""Shared memory Warp's tiled launches declare per block thread [B] (``tile_shared_storage_t``'s offsets)."""

_SIZES = {
    "float": 4,
    "int": 4,
    "unsigned": 4,
    "unsigned int": 4,
    "short": 2,
    "unsigned short": 2,
    "char": 1,
    "signed char": 1,
    "unsigned char": 1,
    "double": 8,
    "long long": 8,
    "unsigned long long": 8,
    "int2": 8,
    "float2": 8,
    "int4": 16,
    "float4": 16,
}
_ALIGNMENTS = {"int2": 8, "float2": 8, "int4": 16, "float4": 16}


@dataclasses.dataclass(frozen=True)
class Buffer:
    """One shared array of a lane group's working set.

    Args:
        name: The array's name in the body.
        ctype: Its CUDA element type (one of the scalar or vector types of :data:`_SIZES`).
        count: Its element count (a compile-time constant).
        align: Byte alignment of the array (default: its element type's).
        alternative: The place the body reads it at where the stage declines it (:func:`stage`); ``None``: staged.
    """

    name: str
    ctype: str
    count: int
    align: int | None = None
    alternative: Place | None = None

    def __post_init__(self):
        if self.ctype not in _SIZES:
            raise ValueError(f"buffer {self.name}: unknown element type {self.ctype!r}")
        if int(self.count) < 1:
            raise ValueError(f"buffer {self.name}: needs at least one element, got {self.count}")
        object.__setattr__(self, "count", int(self.count))

    @property
    def alignment(self) -> int:
        """Byte alignment of the array."""
        return int(self.align or _ALIGNMENTS.get(self.ctype, _SIZES[self.ctype]))

    @property
    def bytes(self) -> int:
        """Bytes of the array's elements."""
        return self.count * _SIZES[self.ctype]

    def declaration(self) -> str:
        """The array's C declaration, without storage class."""
        align = f"__align__({self.align}) " if self.align else ""
        return f"{align}{self.ctype} {self.name}[{self.count}];"


def _layout(buffers) -> int:
    """Bytes of ``buffers`` laid out in order, each at its alignment (the C rules of a struct's members)."""
    offset, widest = 0, 1
    for buffer in buffers:
        alignment = buffer.alignment
        widest = max(widest, alignment)
        offset = (offset + alignment - 1) // alignment * alignment + buffer.bytes
    return (offset + widest - 1) // widest * widest


@dataclasses.dataclass(frozen=True)
class Staging:
    """A block's shared-memory working set: the buffers of each lane group, declared once.

    Args:
        buffers: Each lane group's buffers, in declaration order.
        groups: Lane groups per block; each holds its own copy of the buffers.
        threads: Threads per block, for the tile storage of a tiled launch (0: the launch is not tiled).
        name: Name of the per-group struct when ``groups > 1``.
        layout: How several groups' copies lie: ``"struct"``, one struct of the buffers per group, or
            ``"arrays"``, one array per buffer that holds every group's copy in group order (a buffer's
            elements of the groups that share a warp then fall in distinct banks).
        declined: The buffers :func:`stage` left at their alternative place.
    """

    buffers: tuple[Buffer, ...] = ()
    groups: int = 1
    threads: int = 0
    name: str = "Scratch"
    layout: str = "struct"
    declined: tuple[Buffer, ...] = ()

    def __post_init__(self):
        names = [buffer.name for buffer in self.buffers]
        if len(set(names)) != len(names):
            raise ValueError(f"staging buffers need distinct names: {names}")
        if self.groups < 1:
            raise ValueError("a staging needs at least one lane group")
        if self.layout not in ("struct", "arrays"):
            raise ValueError(f"staging layout must be 'struct' or 'arrays', got {self.layout!r}")
        object.__setattr__(self, "buffers", tuple(self.buffers))

    def __iter__(self):
        return iter(self.buffers)

    def __contains__(self, name: str) -> bool:
        return any(buffer.name == name for buffer in self.buffers)

    def plus(self, *buffers: Buffer) -> Staging:
        """This staging with ``buffers`` appended."""
        return dataclasses.replace(self, buffers=(*self.buffers, *buffers))

    def only(self, names) -> Staging:
        """This staging with the buffers ``names`` names, the others declined."""
        kept = tuple(buffer for buffer in self.buffers if buffer.name in names)
        out = tuple(buffer for buffer in self.buffers if buffer.name not in names)
        return dataclasses.replace(self, buffers=kept, declined=(*self.declined, *out))

    def place(self, name: str) -> Place:
        """Where buffer ``name`` lives: the stage, or the alternative place of a declined one."""
        if name in self:
            return Place.SHARED
        return next(buffer.alternative for buffer in self.declined if buffer.name == name)

    @property
    def group_bytes(self) -> int:
        """Shared memory of one lane group's buffers [B]: separate arrays or a struct's members, each at its own
        alignment."""
        return _layout(self.buffers)

    @property
    def tile_bytes(self) -> int:
        """Shared memory Warp's tiled launch declares for this block [B]."""
        return TILE_BYTES_PER_THREAD * int(self.threads)

    @property
    def bytes(self) -> int:
        """Static shared memory the block declares [B]: every group's buffers and the tile storage."""
        if self.layout == "arrays" and self.groups > 1:
            arrays = (dataclasses.replace(buffer, count=buffer.count * self.groups) for buffer in self.buffers)
            return _layout(arrays) + self.tile_bytes
        return self.groups * self.group_bytes + self.tile_bytes

    def fits(self, limit: int = STATIC_SHARED_BYTES) -> bool:
        """Whether the block's static shared memory fits the budget ``limit`` [B] (by default an offline build's;
        a device's is :func:`static_budget`)."""
        return self.bytes <= int(limit)

    def check(self, what: str, limit: int = STATIC_SHARED_BYTES) -> None:
        """Raise ``ValueError`` naming ``what`` when the block does not fit the budget ``limit``."""
        if not self.fits(limit):
            raise ValueError(
                f"{what} declares {self.bytes} bytes of static shared memory per block "
                f"({self.groups} x {self.group_bytes} + {self.tile_bytes} tile storage), over the "
                f"{int(limit)}-byte limit"
            )

    def declarations(self, indent: str = "    ", *, group: str = "SUB") -> str:
        """The declarations of the working set in ``SHARED`` storage: CUDA's shared memory, or the stack of the CPU's
        one thread. A body that declares them declares the prelude :data:`SHARED_STORAGE`.

        One array per buffer; with several lane groups per block, one struct of
        the buffers per group, indexed by ``group``, and an alias per buffer
        (:meth:`undefine` removes the aliases at the end of the body), or in the
        ``arrays`` layout one array of every group's copies per buffer
        (``name_storage``) and a pointer ``name`` to the group's copy.
        """
        if not self.buffers:
            return ""
        if self.groups == 1:
            return "\n".join(f"{indent}SHARED {buffer.declaration()}" for buffer in self.buffers)
        if self.layout == "arrays":
            arrays = (
                dataclasses.replace(buffer, name=f"{buffer.name}_storage", count=buffer.count * self.groups)
                for buffer in self.buffers
            )
            views = (
                f"{indent}{buffer.ctype}* {buffer.name} = {buffer.name}_storage + {group} * {buffer.count};"
                for buffer in self.buffers
            )
            return "\n".join((*(f"{indent}SHARED {array.declaration()}" for array in arrays), *views))
        members = "\n".join(f"{indent}    {buffer.declaration()}" for buffer in self.buffers)
        aliases = "\n".join(
            f"#define {buffer.name} {self.name.lower()}[{group}].{buffer.name}" for buffer in self.buffers
        )
        return (
            f"{indent}struct {self.name} {{\n{members}\n{indent}}};\n"
            f"{indent}SHARED {self.name} {self.name.lower()}[{self.groups}];\n{aliases}"
        )

    def undefine(self) -> str:
        """Remove the aliases :meth:`declarations` defines for several lane groups in a struct."""
        if self.groups == 1 or self.layout == "arrays":
            return ""
        return "\n".join(f"#undef {buffer.name}" for buffer in self.buffers)


HIDING_WARPS = 4
"""Resident warps a scheduler the general sweep's chain needs (:func:`stage`, STAGE1): a stage pays at 16 to 28 warps
(g1, kuka, franka-reach, go2-flat, allegro on the GB300), costs at 4 to 14 (franka-cabinet, keyboard, PRO allegro)."""


def stage(staging: Staging, warps: Callable[[int], int], budget: int, hiding: int, filled: Callable) -> Staging:
    """The staging rule: buffers without an alternative place, then each with one in order of reuse while the stage fits
    ``budget`` [B] and leaves (``warps(bytes)``) at least the fewer of the warps without it and ``hiding`` a scheduler;
    one that streams otherwise pays only in throughput: only where the live items fill them (``filled``)."""
    names = {buffer.name for buffer in staging if buffer.alternative is None}
    kept = staging.only(names)
    for buffer in (buffer for buffer in staging if buffer.alternative is not None):
        trial = staging.only(names | {buffer.name})
        if not trial.fits(budget) or warps(trial.bytes) < min(warps(kept.bytes), hiding * REGISTER_PARTITIONS):
            break
        if buffer.alternative is Place.STREAMED and not filled(trial.bytes):
            break
        names.add(buffer.name)
        kept = trial
    return kept
