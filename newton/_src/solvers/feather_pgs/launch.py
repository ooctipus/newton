# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""One way to size a launch: items, a thread mapping, and the device's SM count or a device-published extent.

A FeatherPGS kernel covers *items* (worlds, candidates, contact rows, colors' units) with a :class:`Mapping`:
``lanes`` threads per item, ``items_per_block`` items per block. :func:`grid` turns an item count into blocks:

- a direct launch takes one block per ``items_per_block`` items (the ceiling, once, here);
- a persistent launch (``persistent=True``: the kernel strides over its items) takes at most ``blocks_per_sm`` blocks
  per SM, about one resident wave, and never more blocks than items;
- an item count that only the device knows (a live contact count) is an *extent*: the launch is captured at the
  host-side bound and, when it goes through a :class:`~newton._src.core.contact_pool.ContactPool`, each graph replay
  runs it at the extent (the pool's device-updatable launch).

:func:`launch` makes the launch: tiled (``block_dim`` threads per block, the kernel's native mapping reads ``LANE``
and ``SUB``) or a plain thread grid. No launch site writes a ceiling division or an SM-count rule of its own.
"""

from __future__ import annotations

import dataclasses

import warp as wp

__all__ = ["Mapping", "device_lanes", "grid", "launch"]


@dataclasses.dataclass(frozen=True)
class Mapping:
    """How a launch's threads cover its items.

    Args:
        lanes: Threads per item (1 for a thread per item, 8/16/32 for a lane group, a block size for a block).
        items_per_block: Items each block holds.
        persistent: The kernel strides over its items, so the grid is capped near one resident wave.
        blocks_per_sm: Blocks per SM of a persistent launch.
    """

    lanes: int = 32
    items_per_block: int = 1
    persistent: bool = False
    blocks_per_sm: int = 1

    def __post_init__(self):
        if self.lanes < 1 or self.items_per_block < 1 or self.blocks_per_sm < 1:
            raise ValueError(f"a mapping needs positive lanes, items per block and blocks per SM: {self}")

    @property
    def block_dim(self) -> int:
        """Threads per block."""
        return self.lanes * self.items_per_block


def device_lanes(device_arch, lanes: int = 32) -> int:
    """Threads of a group of ``lanes`` lanes on ``device_arch``: ``lanes`` on a CUDA architecture (a number, or an
    offline build's ``"sm_XX"``), one on the CPU (architecture 0), which runs a tiled launch's block on one thread."""
    try:
        arch = int(str(device_arch).removeprefix("sm_"))
    except (TypeError, ValueError):
        return lanes
    return lanes if arch > 0 else 1


def grid(items: int, mapping: Mapping, device=None) -> int:
    """Blocks of a launch over ``items`` items with ``mapping`` on ``device``.

    A direct launch covers every item; a persistent one takes at most ``blocks_per_sm`` blocks per SM of the device
    (and at least one block), never more than the items need.
    """
    blocks = (max(int(items), 0) + mapping.items_per_block - 1) // mapping.items_per_block
    if mapping.persistent:
        device = wp.get_device(device)
        sms = int(getattr(device, "sm_count", 1) or 1) if device.is_cuda else 1
        blocks = min(blocks, max(1, mapping.blocks_per_sm * sms))
    return max(blocks, 1) if items else 0


def launch(
    kernel,
    items: int,
    mapping: Mapping,
    *,
    inputs=(),
    outputs=(),
    device=None,
    tiled: bool = True,
    extent: wp.array | None = None,
    pool=None,
    stream=None,
) -> int:
    """Launch ``kernel`` over ``items`` items with ``mapping``; return the blocks launched.

    Args:
        kernel: The Warp kernel.
        items: The item count (the host-side bound when ``extent`` counts the live items on the device).
        mapping: The thread mapping.
        inputs: Kernel inputs.
        outputs: Kernel outputs.
        device: The launch device.
        tiled: A tiled launch (``block_dim`` threads per block, one tile per block) or a plain grid of
            ``blocks * block_dim`` threads.
        extent: A device count of the live items: with ``pool`` the launch binds so replays follow it.
        pool: The :class:`~newton._src.core.contact_pool.ContactPool` that sizes contact launches.
        stream: The launch stream (default: the device's current stream).
    """
    blocks = grid(items, mapping, device)
    if blocks == 0:
        return 0
    if pool is not None:
        if extent is None:
            raise ValueError("a pool launch needs the device extent it follows")
        pool.launch(
            kernel,
            [blocks] if tiled else blocks * mapping.block_dim,
            inputs=inputs,
            outputs=outputs,
            extent=extent,
            scale=mapping.block_dim if not tiled else 1,
            tiled=tiled,
            block_dim=mapping.block_dim,
        )
        return blocks
    if tiled:
        wp.launch_tiled(
            kernel,
            dim=[blocks],
            inputs=list(inputs),
            outputs=list(outputs),
            block_dim=mapping.block_dim,
            device=device,
            stream=stream,
        )
    else:
        wp.launch(
            kernel,
            dim=blocks * mapping.block_dim,
            inputs=list(inputs),
            outputs=list(outputs),
            block_dim=mapping.block_dim,
            device=device,
            stream=stream,
        )
    return blocks
