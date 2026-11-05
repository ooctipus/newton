# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The execution layer: a kernel declares its work, and the layer picks its launch from the data and the device.

A launch covers *items* (worlds, trees, units, contacts) with a lane group of one or more threads an item. What
differs between launches is the work, not the arithmetic that sizes them, so the layer keeps that arithmetic once:

- :class:`~.memory.Device` is what the layer reads of a device, once (:func:`~.memory.device`, the memory
  toolkit's): its multiprocessors, its occupancy limits, its shared memory and carveout steps. The CPU is
  :data:`~.memory.CPU`: one thread runs a lane group, and it is never tuned;
- :class:`Work` is what a kernel declares: its items (a host count or a device :class:`Queue`), how an item's lanes
  share its work (:class:`Use`), its width, its register line, its shared memory and whether its grid strides;
- :func:`shape` turns a work and a device into a :class:`Shape`: lanes an item, block, resident blocks a
  multiprocessor, the launch bound, the register budget a lane, the grid, and the carveout step with the L1 it leaves;
- :func:`launch` runs a kernel at a shape.

The one rule (``SP/core/FL/HANDOFF_ZEX.md`` §3.2):

- resident threads a multiprocessor are the fewer of its thread limit and the warps its register file keeps at the
  line (whole warps a sub-partition);
- lanes an item: a *scan* takes the item's width; a *split* fills at least one wave of resident threads (the
  ceiling), a *feed* at most one (the floor); each within ``[least, min(most, the power of two over the width)]``;
- resident blocks are the fewest the block limit, the threads, the register line and the shared memory allow; a
  declared line is the launch bound ``(block, those blocks)``, or, for a line that is no cap, only where the shared
  memory keeps no more blocks than the line;
- the grid covers the items, at most ``waves`` waves of resident blocks when the kernel strides over them;
- the carveout is the driver's smallest step that holds the blocks the thread and block limits allow (H8,
  :func:`~.memory.carveout`).
"""

from __future__ import annotations

import dataclasses
import enum
import math

import warp as wp

from .memory import REGISTER_GRANULE, REGISTER_PARTITIONS, REGISTERS_PER_THREAD, Device, carveout, granules

__all__ = [
    "Beside",
    "Order",
    "Queue",
    "Shape",
    "Use",
    "Work",
    "device_lanes",
    "fill_lanes",
    "launch",
    "shape",
]

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
        cap: The line caps the kernel's registers. Without, it is the least a bound must leave: the kernel is bound
            only where shared memory keeps no more blocks than the line would, so a bound gives registers and never
            takes them (a kernel whose variants compile on both sides of its line, and spill under a cap).
        shared: Static shared memory a block [B] (a :class:`~.memory.Staging`'s bytes).
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
    cap: bool = True
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
    return 1 << max(math.ceil(x) - 1, 0).bit_length()


def _pow2_floor(x: float) -> int:
    x = int(x)
    return 1 << (x.bit_length() - 1) if x >= 1 else 1


def _line_warps(device: Device, line: int) -> int:
    """Warps a multiprocessor's register file keeps at ``line`` registers a lane: a warp's registers, in granules,
    from one sub-partition (24 at a line of 80, 16 at 128, 8 at 255)."""
    warp = -(-line * 32 // REGISTER_GRANULE) * REGISTER_GRANULE
    return REGISTER_PARTITIONS * (device.registers_per_sm // REGISTER_PARTITIONS // warp)


def _resident(device: Device, block: int, registers: int | None, shared: int) -> int:
    """Blocks of ``block`` threads a multiprocessor keeps: the block limit, the threads, the register line, the
    shared memory; at least one."""
    limit = min(device.blocks_per_sm, device.threads_per_sm // block)
    if registers:
        limit = min(limit, _line_warps(device, registers) // -(-block // 32))
    if shared:
        limit = min(limit, device.shared_per_sm // granules(shared))
    return max(limit, 1)


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
    resident_threads = min(device.threads_per_sm, 32 * _line_warps(device, line) if line else device.threads_per_sm)
    lanes = _lanes(work, device, resident_threads)
    block = work.block or (256 if lanes <= 32 else lanes)
    per_sm = _resident(device, block, line, work.shared)
    if line and not work.cap and (not work.shared or device.shared_per_sm // granules(work.shared) > per_sm):
        line = None  # shared memory would keep more blocks than the line: the kernel takes the registers it needs
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


# ---------------------------------------------------------------------------------------------------------------------
# Work beside the step's stream
# ---------------------------------------------------------------------------------------------------------------------


class Beside:
    """Work that runs on a stream of its own, from where the step starts it to where the step joins it.

    Work whose inputs are ready early and whose outputs are read late need not wait in the step's one stream: it runs
    beside the work in between, which leaves the device's multiprocessors part idle. :meth:`start` forks it from the
    current stream (an event, so a graph capture records the edge), :meth:`join` makes the current stream wait for it.
    On the CPU it runs in line.

    Args:
        device: The device of the work.
    """

    def __init__(self, device=None):
        cuda = wp.get_device(device).is_cuda
        self.device = device
        self.stream = wp.Stream(device) if cuda else None
        self.started = wp.Event(device) if cuda else None
        self.done = wp.Event(device) if cuda else None

    def start(self, work) -> None:
        """Run ``work()`` (its launches) beside the current stream, after what the current stream queued so far."""
        if self.stream is None:
            work()
            return
        wp.get_stream(self.device).record_event(self.started)
        self.stream.wait_event(self.started)
        with wp.ScopedStream(self.stream, sync_enter=False):
            work()
        self.stream.record_event(self.done)

    def join(self) -> None:
        """Make the current stream wait for the work :meth:`start` ran."""
        if self.stream is not None:
            wp.get_stream(self.device).wait_event(self.done)
