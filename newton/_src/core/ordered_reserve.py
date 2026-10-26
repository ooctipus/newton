# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Ordered reservation: output slots in data order from one pass, with a block scan and a decoupled look-back.

A kernel that appends a variable number of items per input (a broad phase's
overlapping pairs, a narrow phase's contacts, a world's rows) usually reserves
its slots with an atomic counter, so the output order follows the GPU's thread
scheduling and changes from run to run. Ordered reservation gives every input
its slots in input order instead, in the same kernel and the same pass:

- the inputs are cut into *tiles*, one block's share each, in input order;
- a block counts its tile's items, scans the counts (warp shuffles, then the
  warps' totals), and publishes the tile's *aggregate* at once;
- one warp then looks back over the earlier tiles, 32 at a time, summing their
  aggregates until it meets a tile that has published its *inclusive prefix*,
  and publishes its own (Merrill and Garland, "Single-pass Parallel Prefix Scan
  with Decoupled Look-back", 2016);
- every thread's first slot is the tile's exclusive prefix plus its own place in
  the block scan, so the items land in input order.

A launch *appends* to a list: tile 0 starts from the list's count, ``counter[0]``
(slots taken before this launch, by earlier launches or other writers), and the
last tile writes the new count back. Writers that run one after another in a
stream therefore fill one list in segments, each in its own input order; the
list's owner zeroes the count where a new list starts (a frame's first kernel).

A call site keeps one ``uint64`` array (:class:`OrderedTiles`): ``site[0]``
counts the tiles claimed by the running launch, ``site[1]`` holds the launch's
*epoch*, and tile ``t``'s state is ``site[16 + t]``, one word
``epoch << 33 | inclusive << 32 | value`` written whole. A launch reads only
states of its own epoch, so no launch clears them, and no launch takes an
extra kernel or memset:

- *fixed* tiles (:func:`ordered_tile`): a launch of one block per tile, the
  block's index in the grid being its tile, as in CUB's single-pass scan: a
  block only waits for blocks of lower index, which the GPU dispatched before
  it. No atomics; the grid may follow a device-side count (a contact-sized
  launch);
- *dynamic* tiles (:func:`ordered_next`): a fixed grid that strides over a
  device-side count of tiles. The first ``active = min(blocks, tiles)`` blocks
  claim the next tile with one atomic on ``site[0]`` until the tiles run out,
  and the others leave at once. Every active block makes exactly one claim that
  fails, so the launch's last claim, the one numbered ``tiles + active - 1``,
  resets the counter for the next launch. A tile waits only for tiles that
  running blocks claimed before it, and a block claims a tile only when it is
  ready to work on it.

A claim (``wp.vec3i``: tile, epoch, start) carries what the scan needs: thread 0
reads the epoch and tile 0's list count when it claims, so the reads overlap the
tile's own work.

The last tile of a launch of several opens the next epoch once its look-back
ends, when every tile of the launch has read the current one. A launch of a
single tile scans its block and neither publishes nor looks back, so its floor
is the block scan alone. Epochs take 31 bits and skip 0, the zeroed states'
epoch; a tile state left unwritten for exactly 2**31 - 1 launches would be
misread when its tile is next used.

On the CPU a launch runs its blocks one at a time in order, one thread per
block (``block_dim`` 1), so a tile's exclusive prefix is the previous tile's
inclusive prefix.

Kernels call :func:`ordered_tile` or :func:`ordered_next`, then
:func:`ordered_scan` for every claimed tile, with every thread of the block
(block sizes are whole warps); native sources take the same operations from
:data:`ORDERED_CUDA` and :data:`ORDERED_CPU` (``ORDERED_TILE``,
``ORDERED_NEXT``, ``ORDERED_SCAN``).
"""

from __future__ import annotations

import numpy as np
import warp as wp

__all__ = [
    "ORDERED_CPU",
    "ORDERED_CUDA",
    "ORDERED_STATES",
    "OrderedTiles",
    "ordered_also",
    "ordered_next",
    "ordered_scan",
    "ordered_tile",
]

ORDERED_STATES = 16
"""Offset of tile 0's state in a site: one 128-byte line after the claim counter and the epoch."""

_COMMON = r"""
    // A claim: the tile (-1 past the launch's tiles), the launch's epoch (0: the scan reads it) and tile 0's list
    // count before the launch (thread 0; -1: the scan reads it).
    struct OrderedClaim { int tile; int epoch; int start; };
    const auto ORDERED_PACK = [](unsigned epoch, unsigned inclusive, unsigned value) -> unsigned long long {
        return ((unsigned long long)epoch << 33) | ((unsigned long long)inclusive << 32) | (unsigned long long)value;
    };
    const auto ORDERED_NEXT_EPOCH = [](unsigned epoch) -> unsigned long long {
        return epoch >= 0x7fffffffu ? 1ull : (unsigned long long)(epoch + 1u);
    };"""

ORDERED_CUDA = (
    r"""    // ---- ordered reservation: tiles in input order, decoupled look-back with epoch-tagged states ----"""
    + _COMMON
    + r"""
    const auto ORDERED_EPOCH = [](unsigned long long* site) -> int {
        return (int)(*((volatile unsigned long long*)(site + 1)) & 0x7fffffffull);
    };
    // Fixed tiles: the block's own tile. Thread 0 reads the epoch (a launch of several tiles) and tile 0's start.
    const auto ORDERED_TILE = [&](unsigned long long* site, int tile, int tiles, int* counter) -> OrderedClaim {
        OrderedClaim claim{tile, 0, 0};
        if (threadIdx.x == 0) {
            if (tiles > 1) claim.epoch = ORDERED_EPOCH(site);
            if (tile == 0) claim.start = *((volatile int*)counter);
        }
        return claim;
    };
    // Dynamic tiles: the next unclaimed tile, or tile -1 once the launch's tiles are claimed (the block stops).
    // Only the first min(blocks, live_tiles) blocks claim; each makes exactly one failing claim, so the claim
    // numbered live_tiles + active - 1 is the launch's last and resets the counter. Thread 0 reads the epoch and
    // the list count with the claim. A block claims only when it is ready for the tile: a claimed tile that waits
    // for its block would hold up the look-backs of the tiles after it.
    const auto ORDERED_NEXT = [&](unsigned long long* site, int block, int live_tiles, int blocks,
                                  int* counter) -> OrderedClaim {
        __shared__ long long s_ordered_claim;
        const int active = blocks < live_tiles ? blocks : live_tiles;
        OrderedClaim claim{-1, 0, 0};
        if (block >= active) return claim;
        if (threadIdx.x == 0) {
            const unsigned tile = atomicAdd((unsigned*)site, 1u);
            const int epoch = ORDERED_EPOCH(site);
            const int start = *((volatile int*)counter);
            if ((int)tile < live_tiles) {
                claim.start = tile == 0u ? start : 0;
                s_ordered_claim = ((long long)epoch << 32) | (long long)tile;
            } else {
                if (tile == (unsigned)(live_tiles + active - 1)) *((volatile unsigned*)site) = 0u;
                s_ordered_claim = -1;
            }
        }
        __syncthreads();
        const long long word = s_ordered_claim;
        if (word >= 0) {
            claim.tile = (int)(word & 0xffffffffll);
            claim.epoch = (int)(word >> 32);
        }
        return claim;
    };
    // Warp 0's look-back for a tile after the first: publishes the tile's aggregate, sums earlier tiles back to
    // an inclusive prefix, publishes its own; returns the tile's exclusive prefix on every lane of warp 0.
    const auto ORDERED_LOOK_BACK = [&](unsigned long long* states, int tile, unsigned epoch,
                                       unsigned aggregate) -> unsigned {
        const int lane = threadIdx.x & 31;
        // States go out by atomic exchange, performed at L2 where the look-backs read them.
        if (lane == 0) atomicExch(states + tile, ORDERED_PACK(epoch, 0u, aggregate));
        unsigned exclusive = 0u;
        int window_end = tile - 1;
        while (true) {
            const int t = window_end - lane;
            unsigned inclusive = 1u, value = 0u;
            if (t >= 0) {
                unsigned long long s;
                do {
                    s = *((volatile const unsigned long long*)(states + t));
                } while ((unsigned)(s >> 33) != epoch);
                inclusive = (unsigned)(s >> 32) & 1u;
                value = (unsigned)s;
            }
            const unsigned prefixes = __ballot_sync(0xffffffffu, inclusive != 0u);
            const int stop = prefixes ? __ffs(prefixes) - 1 : 31;
            unsigned contribution = lane <= stop ? value : 0u;
            for (int offset = 16; offset > 0; offset >>= 1)
                contribution += __shfl_xor_sync(0xffffffffu, contribution, offset);
            exclusive += contribution;
            if (prefixes) break;
            window_end -= 32;
        }
        if (lane == 0) atomicExch(states + tile, ORDERED_PACK(epoch, 1u, exclusive + aggregate));
        return exclusive;
    };
    // Every thread of the block: its first slot in input order, after the *counter slots taken before the launch.
    // The last tile writes the new count and, in a launch of several tiles, opens the next epoch.
    const auto ORDERED_SCAN = [&](unsigned long long* site, OrderedClaim claim, int count, bool last,
                                  int* counter) -> int {
        __shared__ unsigned s_ordered_warp[32];
        const unsigned full = 0xffffffffu;
        const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
        const int tile = claim.tile;
        const bool alone = tile == 0 && last;
        // The reads a claim of another list did not make (ordered_also): thread 0 issues them ahead of the scan.
        int epoch_word = claim.epoch, start_word = claim.start;
        if (threadIdx.x == 0) {
            if (epoch_word == 0 && !alone) epoch_word = ORDERED_EPOCH(site);
            if (tile == 0 && start_word < 0) start_word = *((volatile int*)counter);
        }
        unsigned inclusive = (unsigned)count;
        for (int offset = 1; offset < 32; offset <<= 1) {
            const unsigned lower = __shfl_up_sync(full, inclusive, offset);
            if (lane >= offset) inclusive += lower;
        }
        if (lane == 31) s_ordered_warp[warp] = inclusive;
        __syncthreads();
        if (warp == 0) {
            const unsigned own = lane < warps ? s_ordered_warp[lane] : 0u;
            unsigned below = own;
            for (int offset = 1; offset < 32; offset <<= 1) {
                const unsigned lower = __shfl_up_sync(full, below, offset);
                if (lane >= offset) below += lower;
            }
            const unsigned aggregate = __shfl_sync(full, below, 31);
            const unsigned epoch = (unsigned)__shfl_sync(full, epoch_word, 0);
            const unsigned prefix = tile == 0 ? (unsigned)__shfl_sync(full, start_word, 0)
                                              : ORDERED_LOOK_BACK(site + 16, tile, epoch, aggregate);
            if (tile == 0 && !alone && lane == 0) atomicExch(site + 16, ORDERED_PACK(epoch, 1u, prefix + aggregate));
            if (lane < warps) s_ordered_warp[lane] = prefix + below - own;
            if (lane == 0 && last) {
                *((volatile int*)counter) = (int)(prefix + aggregate);
                if (!alone) *((volatile unsigned long long*)(site + 1)) = ORDERED_NEXT_EPOCH(epoch);
            }
        }
        __syncthreads();
        const int first = (int)(s_ordered_warp[warp] + inclusive - (unsigned)count);
        __syncthreads();
        return first;
    };"""
)
"""CUDA text of the ordered reservation (statements inside a native body, ahead of the body's own)."""

ORDERED_CPU = (
    r"""    // ---- ordered reservation, CPU: blocks run one at a time in order, one thread each ----"""
    + _COMMON
    + r"""
    const auto ORDERED_EPOCH = [](unsigned long long* site) -> int { return (int)(site[1] & 0x7fffffffull); };
    const auto ORDERED_TILE = [&](unsigned long long* site, int tile, int tiles, int* counter) -> OrderedClaim {
        return OrderedClaim{tile, tiles > 1 ? ORDERED_EPOCH(site) : 0, tile == 0 ? *counter : 0};
    };
    const auto ORDERED_NEXT = [&](unsigned long long* site, int block, int live_tiles, int blocks,
                                  int* counter) -> OrderedClaim {
        const int active = blocks < live_tiles ? blocks : live_tiles;
        if (block >= active) return OrderedClaim{-1, 0, 0};
        const unsigned tile = (unsigned)site[0];
        site[0] = (unsigned long long)(tile + 1u);
        if ((int)tile < live_tiles) return OrderedClaim{(int)tile, ORDERED_EPOCH(site), tile == 0u ? *counter : 0};
        if (tile == (unsigned)(live_tiles + active - 1)) site[0] = 0ull;
        return OrderedClaim{-1, 0, 0};
    };
    const auto ORDERED_SCAN = [&](unsigned long long* site, OrderedClaim claim, int count, bool last,
                                  int* counter) -> int {
        unsigned long long* states = site + 16;
        const int tile = claim.tile;
        const bool alone = tile == 0 && last;
        const unsigned epoch = (unsigned)(claim.epoch == 0 && !alone ? ORDERED_EPOCH(site) : claim.epoch);
        const unsigned prefix = tile > 0 ? (unsigned)states[tile - 1]
                                         : (unsigned)(claim.start < 0 ? *counter : claim.start);
        if (!alone) states[tile] = ORDERED_PACK(epoch, 1u, prefix + (unsigned)count);
        if (last) {
            *counter = (int)(prefix + (unsigned)count);
            if (!alone) site[1] = ORDERED_NEXT_EPOCH(epoch);
        }
        return (int)prefix;
    };"""
)
"""CPU twin of :data:`ORDERED_CUDA`."""


def _snippet(body: str) -> str:
    return "\n#if defined(__CUDA_ARCH__)\n" + ORDERED_CUDA + "\n#else\n" + ORDERED_CPU + "\n#endif\n" + body


_TO_VEC = "    return wp::vec3i(claim.tile, claim.epoch, claim.start);\n"
_FROM_VEC = "    const OrderedClaim from{claim[0], claim[1], claim[2]};\n"


@wp.func_native(
    _snippet("    const OrderedClaim claim = ORDERED_TILE(site.data, tile, tiles, counter.data);\n" + _TO_VEC)
)
def ordered_tile(site: wp.array[wp.uint64], tile: int, tiles: int, counter: wp.array[int]) -> wp.vec3i:
    """Claim tile ``tile``, the block's index in a launch of ``tiles`` blocks, of a list counted by ``counter``.

    Every thread of the block calls it once, before the tile's work: thread 0 starts the reads the scan needs.
    """


@wp.func
def ordered_also(claim: wp.vec3i) -> wp.vec3i:
    """The same tile, for a list of another site: :func:`ordered_scan` reads that site's epoch and list count.

    A kernel that appends to several lists per tile claims from one site and scans the others with this claim;
    every list it appends to is scanned for every claimed tile.
    """
    return wp.vec3i(claim[0], 0, -1)


@wp.func_native(
    _snippet(
        "    const OrderedClaim claim = ORDERED_NEXT(site.data, block, live_tiles, blocks, counter.data);\n" + _TO_VEC
    )
)
def ordered_next(
    site: wp.array[wp.uint64], block: int, live_tiles: int, blocks: int, counter: wp.array[int]
) -> wp.vec3i:
    """Claim the next of ``live_tiles`` tiles for block ``block`` of a ``blocks``-block grid, for a list counted by
    ``counter``.

    Every thread of the block calls it, once the block is ready for the tile; a tile of -1 means the launch's tiles
    are taken and the block must stop. Every block of the grid calls it until then, and scans every tile it claims.
    """


@wp.func_native(_snippet(_FROM_VEC + "    return ORDERED_SCAN(site.data, from, count, last != 0, counter.data);\n"))
def ordered_scan(site: wp.array[wp.uint64], claim: wp.vec3i, count: int, last: int, counter: wp.array[int]) -> int:
    """Return this thread's first slot, in input order, for its ``count`` items of the claimed tile.

    Every thread of the block calls it once per claimed tile. Slots follow the ``counter[0]`` slots taken
    before the launch; ``last`` marks the launch's last tile, which writes the new count to ``counter[0]``.
    """


class OrderedTiles:
    """A call site's ordered-reservation state: its claim counter, its epoch and its tile states.

    Fixed and dynamic launches may share a site; launches that share one run in stream order.

    Args:
        tiles: The most tiles one launch reserves over.
        device: Device of the array.
    """

    def __init__(self, tiles: int, device):
        self.tiles = max(int(tiles), 1)
        site = np.zeros(ORDERED_STATES + self.tiles, dtype=np.uint64)
        site[1] = 1  # zeroed states carry epoch 0, which no launch uses
        self.site = wp.array(site, dtype=wp.uint64, device=device)
