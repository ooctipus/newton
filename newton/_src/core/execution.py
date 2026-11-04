# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The execution layer: a kernel declares its work, and the layer picks its launch from the data and the device.

A launch covers *items* (worlds, trees, units, contacts) with a lane group of one or more threads an item. What
differs between launches is the work, not the arithmetic that sizes them, so the layer keeps that arithmetic once:

- :class:`Device` is what the layer reads of a device, once (:func:`device`): its multiprocessors, its occupancy
  limits (threads, blocks and registers a multiprocessor), its static shared-memory budget a block and the driver's
  shared-memory carveout steps. The CPU is :data:`CPU`: one thread runs a lane group, and it is never tuned;
- :class:`Work` is what a kernel declares: its items (a host count or a device :class:`Queue`), how an item's lanes
  share its work (:class:`Use`), its width, its register line, its shared memory and whether its grid strides;
- :func:`shape` turns a work and a device into a :class:`Shape`: lanes an item, block, resident blocks a
  multiprocessor, the launch bound, the register budget a lane, the grid, and the carveout step with the L1 it leaves;
- :func:`launch` runs a kernel at a shape.

The one rule (``SP/core/FL/HANDOFF_ZEX.md`` §3.2):

- resident threads a multiprocessor are the fewer of its thread limit and its register file over the line;
- lanes an item: a *scan* takes the item's width; a *split* fills at least one wave of resident threads (the
  ceiling), a *feed* at most one (the floor); each within ``[least, min(most, the power of two over the width)]``;
- resident blocks are the fewest the block limit, the threads, the register line and the shared memory allow; a
  declared line is the launch bound ``(block, those blocks)``;
- the grid covers the items, at most ``waves`` waves of resident blocks when the kernel strides over them;
- the carveout is the driver's smallest step that holds the blocks the thread and block limits allow (H8).

The shared-memory facts that kernel factories read by architecture (:func:`static_budget`, :func:`arch_budget`,
:func:`resident_blocks`) live here too: offline (no device of the architecture at hand, as in an offline SASS build)
they fall back to the offline compiler's budget and the smallest suite GPU's multiprocessor.
"""

from __future__ import annotations

import dataclasses
import enum
import functools

import warp as wp

__all__ = [
    "CPU",
    "CPU_STAGING_BYTES",
    "RESERVED_SHARED_BYTES",
    "SHARED_ALLOCATION_BYTES",
    "SM_SHARED_BYTES",
    "STATIC_SHARED_BUDGETS",
    "STATIC_SHARED_BYTES",
    "Device",
    "Order",
    "Queue",
    "Shape",
    "Use",
    "Work",
    "arch_budget",
    "carveout",
    "device",
    "device_lanes",
    "fill_lanes",
    "launch",
    "resident_blocks",
    "shape",
    "static_budget",
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

SM_SHARED_BYTES = 100 * 1024
"""Shared memory one SM gives its resident blocks [B] when no device of the architecture is at hand (an offline
build, the CPU): the RTX PRO 6000's (sm_120), the smallest of the suite's GPUs."""

SHARED_ALLOCATION_BYTES = 128
"""Granule in which an SM allocates a block's shared memory [B] (``cuda_occupancy.h``, sm_80 and later)."""

REGISTERS_PER_THREAD = 255
"""The most registers a thread holds (every architecture since sm_35): the budget of a kernel without a bound."""

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
    """What the layer reads of a device, once (:func:`device`).

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


def _granules(block_bytes: int) -> int:
    """Shared memory a block takes of its multiprocessor [B]: its bytes and the driver's reservation, in granules."""
    granules = -(-(int(block_bytes) + RESERVED_SHARED_BYTES) // SHARED_ALLOCATION_BYTES)
    return granules * SHARED_ALLOCATION_BYTES


def resident_blocks(block_bytes: int, device_arch=None) -> int:
    """Blocks of ``block_bytes`` static shared memory each that one SM of ``device_arch`` keeps resident.

    A block takes its bytes and the driver's reservation, in 128-byte granules, out of the SM's shared memory
    (:attr:`Device.shared_per_sm` of a CUDA device of the architecture), else :data:`SM_SHARED_BYTES`.
    """
    found = _of_architecture(max(_architecture(device_arch), 0))
    sm_bytes = found.shared_per_sm if found is not None else SM_SHARED_BYTES
    return sm_bytes // _granules(block_bytes)


# ---------------------------------------------------------------------------------------------------------------------
# Lanes of today's launch sites (until they declare their work)
# ---------------------------------------------------------------------------------------------------------------------


def fill_lanes(items: int, device, least: int = 32, most: int = 1024) -> int:
    """Lanes an item of a launch whose items stride over their own work (a world's rows, no lane waiting on another):
    the smallest power of two from ``least`` at which ``items`` items cover a block of ``most`` threads on each SM, at
    most ``most``; ``least`` on the CPU.

    A latency-bound map over a world's rows at a few worlds idles the device at a warp or a block a world (coins'
    128 worlds of 9k free-body rows on 188 SMs); many worlds keep ``least``.
    """
    device = wp.get_device(device)
    if not device.is_cuda:
        return least
    need = -(-int(device.sm_count) * most // max(int(items), 1))
    lanes = least
    while lanes < min(need, most):
        lanes *= 2
    return min(lanes, most)


def device_lanes(device_arch, lanes: int = 32) -> int:
    """Threads of a group of ``lanes`` lanes on ``device_arch``: ``lanes`` on a CUDA architecture (a number, or an
    offline build's ``"sm_XX"``), one on the CPU (architecture 0), which runs a tiled launch's block on one thread."""
    try:
        arch = int(str(device_arch).removeprefix("sm_"))
    except (TypeError, ValueError):
        return lanes
    return lanes if arch > 0 else 1


# ---------------------------------------------------------------------------------------------------------------------
# Work and its shape
# ---------------------------------------------------------------------------------------------------------------------


class Use(enum.Enum):
    """How an item's lanes share its work, declared by the body (it sets which way the fill rounds)."""

    SPLIT = "split"
    """Lanes stride over the item's independent sub-items (rows): a short wave costs throughput, so fill at least one
    wave of resident threads (the ceiling)."""
    FEED = "feed"
    """Lanes feed one serial chain every lane runs: a second wave repeats the chain, so fill at most one wave (the
    floor)."""
    SCAN = "scan"
    """Lanes shorten a parallel scan over the item: as many lanes as its width."""


class Order(enum.Enum):
    """How a :class:`Queue`'s appends take their slots (the overflow contract of :func:`.device_preludes.reserve`)."""

    ATOMIC = "atomic"
    """One atomic an append: slots in the order threads reach the counter."""
    GROUP = "group"
    """One atomic a lane group (the lane group's ``RESERVE``): a group's appends in lane order."""
    ORDERED = "ordered"
    """Slots in input order (the lane group's ``ORDERED``, or ``ORDERED_TILES`` across blocks)."""


class Queue:
    """A device work list: appended where its items are reserved, consumed by a persistent launch over its count.

    ``items`` holds ``buckets`` regions of the list's slots; ``count`` one counter a kind a bucket (``count[kinds * b
    + k]``), zeroed where a new list starts. A :class:`Work` over a queue is sized at :attr:`capacity`, its host bound;
    the device's count is the launch's extent.

    Args:
        capacity: The most items the list holds.
        device: The device of its arrays.
        kinds: Item kinds kept apart (the unit list's wide and one-row units).
        buckets: Counters each kind spreads its appends over.
        order: How appends take their slots.
    """

    def __init__(self, capacity: int, device=None, *, kinds: int = 1, buckets: int = 1, order: Order = Order.ATOMIC):
        if int(capacity) < 0 or kinds < 1 or buckets < 1:
            raise ValueError(
                f"a queue needs a capacity and at least one kind and bucket: {capacity}, {kinds}, {buckets}"
            )
        self.capacity, self.kinds, self.buckets, self.order = int(capacity), int(kinds), int(buckets), order
        self.items = wp.empty(max(self.capacity, 1), dtype=int, device=device)
        self.count = wp.zeros(self.kinds * self.buckets, dtype=int, device=device)


@dataclasses.dataclass(frozen=True)
class Work:
    """What a kernel declares of its launch.

    Args:
        items: The items: a host count, or a :class:`Queue` (its capacity is the host bound).
        use: How an item's lanes share its work.
        width: Sub-items an item's lanes can share (rows, bodies, a path's coordinates).
        least: The fewest lanes an item.
        most: The most lanes an item; past 32, whole warps of a block group.
        lanes: A measured fixed lane count, in place of the rule.
        block: Threads a block (by default 256, or the lanes of a block group).
        registers: The register line a lane, which sets the launch bound (``None``: unbounded).
        shared: Static shared memory a block [B] (a :class:`.staging.Staging`'s bytes).
        persistent: The kernel strides over its items, so its grid is capped at ``waves`` waves of resident blocks.
        waves: Waves of resident blocks of a persistent grid.
        grid_per_sm: A measured persistent grid in blocks a multiprocessor, in place of ``waves``.
    """

    items: int | Queue
    use: Use = Use.SPLIT
    width: int = 1
    least: int = 1
    most: int = 32
    lanes: int | None = None
    block: int | None = None
    registers: int | None = None
    shared: int = 0
    persistent: bool = False
    waves: float = 1.0
    grid_per_sm: int | None = None

    def __post_init__(self):
        for name in ("least", "most", "lanes"):
            value = getattr(self, name)
            if value is not None and (value < 1 or (value & (value - 1) if value <= 32 else value % 32)):
                raise ValueError(f"work {name} must be a power of two up to 32 or whole warps, got {value}")
        if self.least > self.most:
            raise ValueError(f"work needs least <= most lanes, got {self.least} > {self.most}")
        if self.registers is not None and not 1 <= self.registers <= REGISTERS_PER_THREAD:
            raise ValueError(f"a register line is 1 to {REGISTERS_PER_THREAD} registers, got {self.registers}")

    @property
    def count(self) -> int:
        """The host bound of the items."""
        return self.items.capacity if isinstance(self.items, Queue) else int(self.items)


@dataclasses.dataclass(frozen=True)
class Shape:
    """What the layer picks for a :class:`Work` on a :class:`Device`.

    Args:
        lanes: Threads an item.
        groups: Items a block.
        block: Threads a block.
        blocks_per_sm: Blocks a multiprocessor keeps resident: the block limit, the threads, the register line and
            the shared memory.
        bounds: The launch bound ``(block, blocks_per_sm)`` of a declared register line, else ``None``.
        registers: The register budget a lane: the line the bound leaves, else :data:`REGISTERS_PER_THREAD`. A
            body's unroll, hold and loads ahead read it.
        grid: Blocks of the launch.
        carveout_kb: The driver's shared-memory carveout step [KiB].
        l1_kb: The L1 the carveout leaves [KiB].
    """

    lanes: int
    groups: int
    block: int
    blocks_per_sm: int
    bounds: tuple[int, int] | None
    registers: int
    grid: int
    carveout_kb: int
    l1_kb: int


def _pow2_ceil(x: float) -> int:
    return 1 << max(int(x) - 1, 0).bit_length()


def _pow2_floor(x: float) -> int:
    x = int(x)
    return 1 << (x.bit_length() - 1) if x >= 1 else 1


def _resident(device: Device, block: int, registers: int | None, shared: int) -> int:
    """Blocks of ``block`` threads a multiprocessor keeps: the block limit, the threads, the register line, the
    shared memory; at least one."""
    limit = min(device.blocks_per_sm, device.threads_per_sm // block)
    if registers:
        limit = min(limit, device.registers_per_sm // (registers * block))
    if shared:
        limit = min(limit, device.shared_per_sm // _granules(shared))
    return max(limit, 1)


def carveout(device: Device, block: int, shared: int) -> int:
    """The driver's carveout step [KiB] for blocks of ``block`` threads and ``shared`` bytes: the smallest step that
    holds the blocks the thread and block limits allow (H8: the PRO keeps 64 KiB for 32-thread blocks of 1,664 B,
    takes 100 KiB at 2,200 B); 0 on the CPU."""
    if not device.cuda:
        return 0
    need = min(device.blocks_per_sm, device.threads_per_sm // block) * _granules(shared)
    for step in device.carveouts_kb:
        if step * 1024 >= need:
            return step
    return device.carveouts_kb[-1]


def _lanes(work: Work, device: Device, resident_threads: int) -> int:
    if work.lanes is not None:
        return work.lanes
    cap = min(work.most, _pow2_ceil(max(work.width, 1)))
    if work.use is Use.SCAN:
        lanes = cap
    else:
        fit = resident_threads * device.sms / max(work.count, 1)
        lanes = _pow2_ceil(fit) if work.use is Use.SPLIT else _pow2_floor(fit)
    return max(work.least, min(lanes, cap))


def shape(work: Work, device: Device) -> Shape:
    """The launch of ``work`` on ``device`` by the one rule (module docstring). The CPU takes one lane and one thread
    a block, its grid an item."""
    if not device.cuda:
        return Shape(1, 1, 1, 1, None, REGISTERS_PER_THREAD, work.count, 0, 0)
    line = work.registers
    resident_threads = min(device.threads_per_sm, device.registers_per_sm // line if line else device.threads_per_sm)
    lanes = _lanes(work, device, resident_threads)
    block = work.block or (256 if lanes <= 32 else lanes)
    per_sm = _resident(device, block, line, work.shared)
    registers = min(REGISTERS_PER_THREAD, device.registers_per_sm // (block * per_sm)) if line else REGISTERS_PER_THREAD
    groups = max(block // lanes, 1)
    grid = -(-work.count // groups)
    if work.persistent:
        cap = (work.grid_per_sm or round(work.waves * per_sm)) * device.sms
        grid = min(grid, max(cap, 1))
    step = carveout(device, block, work.shared)
    return Shape(
        lanes, groups, block, per_sm, (block, per_sm) if line else None, registers, grid, step, device.unified_kb - step
    )


def launch(
    kernel,
    shape: Shape,
    *,
    inputs=(),
    outputs=(),
    device=None,
    tiled: bool = True,
    extent: wp.array | None = None,
    pool=None,
    stream=None,
) -> None:
    """Launch ``kernel`` at ``shape``: ``shape.grid`` blocks of ``shape.block`` threads.

    Args:
        kernel: The Warp kernel.
        shape: Its shape (:func:`shape`).
        inputs: Kernel inputs.
        outputs: Kernel outputs.
        device: The launch device.
        tiled: A tiled launch (the kernel reads ``block, lane = wp.tid()``) or a plain grid of ``grid * block``
            threads.
        extent: A device count of the live items (a :class:`Queue`'s count): with ``pool`` each graph replay follows
            it.
        pool: The :class:`~newton._src.core.contact_pool.ContactPool` that sizes contact launches.
        stream: The launch stream (default: the device's current stream).
    """
    if shape.grid == 0:
        return
    if pool is not None:
        if extent is None:
            raise ValueError("a pool launch needs the device extent it follows")
        pool.launch(
            kernel,
            [shape.grid] if tiled else shape.grid * shape.block,
            inputs=inputs,
            outputs=outputs,
            extent=extent,
            scale=1 if tiled else shape.block,
            tiled=tiled,
            block_dim=shape.block,
        )
        return
    common = {"inputs": list(inputs), "outputs": list(outputs), "block_dim": shape.block, "device": device}
    if tiled:
        wp.launch_tiled(kernel, dim=[shape.grid], stream=stream, **common)
    else:
        wp.launch(kernel, dim=shape.grid * shape.block, stream=stream, **common)
