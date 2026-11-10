# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sort the live slots of a contact stream by int64 key, with launches that follow the live count.

The slots live in two halves of buffers of ``2 * capacity + SORT_CHUNK``
elements (keys and slot indices): the first half holds the live slots, the
second is scratch, and the chunk past it lets a chunk's tile load stay in the
buffer. A kernel that knows the live count writes the sort's extent with
:func:`publish_sort_extent`: ``[live slots, chunks, work of each merge]``.
:func:`launch_live_sort` then sorts each chunk of :data:`SORT_CHUNK` live slots
in shared memory and merges sorted runs pairwise, its last round every run
left, until one run holds every live slot, ending in the first half.

Order: by key; a merge keeps the slots of its earlier run first among equal
keys. Chunks of *slot keys* (``slot_keys``, keys of at most 51 bits) sort by
(key, slot), so the whole sort is a stable sort by key; other keys sort by key
alone within a chunk.

Every launch strides over its extent entry and goes through
:meth:`~newton._src.core.contact_pool.ContactPool.launch`: in a CUDA graph
capture whose contact launches bind, each replay runs it at the live work only,
and the half offset follows the ready rows.
"""

from __future__ import annotations

import warp as wp

SORT_CHUNK = wp.constant(2048)
"""Slots a block of the chunk sort orders in shared memory."""
SLOT_BITS = wp.constant(11)
"""Bits of a slot within its chunk (``SORT_CHUNK == 1 << SLOT_BITS``)."""
MAX_MERGES = wp.constant(21)
"""Merge rounds the extent holds: ``SORT_CHUNK << 21`` slots exceed any int32 count."""
EXTENT_SIZE = 2 + MAX_MERGES
"""Entries of a sort extent: the live slots, their chunks, and the work of each merge."""
_PAD = wp.constant(wp.int64(1 << 62))
_KEY_SENTINEL = wp.constant(wp.int64(0x7FFFFFFFFFFFFFFF))
_BLOCK_THREADS = 256
# Each chunk-sort block sorts SORT_CHUNK slots; two such blocks per SM fill the GPU. The merges stride
# over the live slots with four 256-thread blocks per SM.
_SORT_BLOCKS_PER_SM = 2
_MERGE_BLOCKS_PER_SM = 4


def slot_keys_fit(key_bits: int) -> bool:
    """Whether keys of ``key_bits`` bits sort by (key, slot) within a chunk: the key and slot bits stay below the pad bit."""
    return key_bits + int(SLOT_BITS) <= 62


@wp.func
def merge_rounds(live: int):
    """Return the pairwise merges that join sorted chunks into one run of ``live`` slots."""
    rounds = int(0)
    width = int(SORT_CHUNK)
    while width < live and rounds < MAX_MERGES:
        width *= 2
        rounds += 1
    return rounds


@wp.func
def publish_sort_extent(extent: wp.array[int], base: int, live: int):
    """Write the sort's extent of ``live`` slots at ``extent[base:base + EXTENT_SIZE]``.

    The live slots, their chunks, and for each merge the live slots when it has
    runs to join, else 0: a captured sort follows them and launches no work past
    the live slots.
    """
    extent[base] = live
    extent[base + 1] = (live + SORT_CHUNK - 1) // SORT_CHUNK
    rounds = merge_rounds(live)
    for merge in range(MAX_MERGES):
        work = int(0)
        if merge < rounds:
            work = live
        extent[base + 2 + merge] = work


@wp.kernel(enable_backward=False)
def _publish_count(count: wp.array[int], rows: wp.array[wp.int32], extent: wp.array[int]):
    live = count[0]
    if rows.shape[0] > 0:
        live = wp.min(live, int(rows[0]))
    publish_sort_extent(extent, 0, wp.max(live, 0))


@wp.func
def _slot_order(key: wp.int64, local: int, start: int, live: int):
    """Order a chunk's slots by (key, slot); slots past the live count sort last, in slot order."""
    if start + local < live:
        return (key << wp.int64(SLOT_BITS)) | wp.int64(local)
    return _PAD + wp.int64(local)


@wp.func
def _slot_key(order: wp.int64):
    if order >= _PAD:
        return _KEY_SENTINEL
    return order >> wp.int64(SLOT_BITS)


@wp.func
def _key_order(key: wp.int64, local: int, start: int, live: int):
    """Order a chunk's slots by key; slots past the live count sort last."""
    if start + local < live:
        return key
    return _KEY_SENTINEL


@wp.func
def _key(order: wp.int64):
    return order


# Below SORT_CHUNK // 2 slots per half, a chunk's tile reaches past the 2 * capacity
# keys and indices of ordinary arrays: those capacities rely on wp.tile_store's
# bounds check, which drops the stores past the arrays' end. Such slots only hold
# padding past the live slots; fields hold SORT_CHUNK extra slots to avoid it.
def _chunk_sort(order_key, stored_key):
    def sort_chunks(
        slot_keys: wp.array[wp.int64],
        slot_indices: wp.array[int],
        keys: wp.array[wp.int64],
        indices: wp.array[int],
        extent: wp.array[int],
        capacity: int,
        blocks: int,
        merges: int,
    ):
        """Sort each chunk of the live slots in shared memory, into the half where ``merges`` rounds end in the first.

        Chunks load through ``slot_keys`` and ``slot_indices``; a block stores
        only the chunk it loaded, so no block reads what another writes.
        """
        block = wp.tid()
        live = extent[0]
        output = capacity * (wp.min(merge_rounds(live), merges) % 2)
        for start in range(block * SORT_CHUNK, live, blocks * SORT_CHUNK):
            loaded = wp.tile_load(slot_keys, shape=SORT_CHUNK, offset=start)
            local = wp.tile_arange(SORT_CHUNK, dtype=int)
            first = wp.tile_full(SORT_CHUNK, start, dtype=int)
            count = wp.tile_full(SORT_CHUNK, live, dtype=int)
            order = wp.tile_zeros(shape=SORT_CHUNK, dtype=wp.int64, storage="shared")
            # Warp hashes a function passed to a builtin only through wp.static();
            # otherwise edits to the key maps would reuse a stale cached kernel.
            wp.tile_assign(order, wp.tile_map(wp.static(order_key), loaded, local, first, count))
            values = wp.tile_load(slot_indices, shape=SORT_CHUNK, offset=start, storage="shared")
            wp.tile_sort(order, values)
            wp.tile_store(keys, wp.tile_map(wp.static(stored_key), order), offset=output + start)
            wp.tile_store(indices, values, offset=output + start)

    return sort_chunks


# Two chunk sorts that differ in their key maps, each a kernel of this module.
_sort_slot_chunks = wp.Kernel(
    _chunk_sort(_slot_order, _slot_key), key="live_sort_slot_chunks", options={"enable_backward": False}
)
_sort_key_chunks = wp.Kernel(
    _chunk_sort(_key_order, _key), key="live_sort_key_chunks", options={"enable_backward": False}
)


@wp.kernel(enable_backward=False)
def _merge_runs(
    keys: wp.array[wp.int64],
    indices: wp.array[int],
    extent: wp.array[int],
    capacity: int,
    merge: int,
    merges: int,
    threads: int,
):
    """Merge sorted runs pairwise, in the last of the ``merges`` rounds every run left; slots move past smaller keys.

    Among equal keys an earlier run's slots go first: a slot also counts an earlier run's equal keys, so every target
    position is distinct. Launches past the needed merges return.
    """
    thread = wp.tid()
    live = extent[0]
    rounds = wp.min(merge_rounds(live), merges)
    if merge >= rounds:
        return
    source = capacity * ((rounds + merge) % 2)
    target = capacity * ((rounds + merge + 1) % 2)
    width = int(SORT_CHUNK) << merge
    runs = (live - 1) // width + 1
    last = merge == rounds - 1
    for i in range(thread, live, threads):
        key = keys[source + i]
        index = indices[source + i]
        run = i // width
        group = int(0)
        end = runs
        if not last:
            group = run - run % 2
            end = wp.min(group + 2, runs)
        position = i - (run - group) * width
        for other in range(group, end):
            if other != run:
                lo = other * width
                hi = wp.min(lo + width, live)
                first = lo
                while lo < hi:
                    mid = (lo + hi) // 2
                    value = keys[source + mid]
                    if value < key or (other < run and value == key):
                        lo = mid + 1
                    else:
                        hi = mid
                position += lo - first
        keys[target + position] = key
        indices[target + position] = index


def sort_grid(device, rows: int) -> tuple[int, int, int]:
    """Return the grids of a live sort over at most ``rows`` slots: chunk-sort blocks, merge rounds, merge threads.

    The grids are fixed so CUDA graphs replay them, and launches stride over
    their extent entry, so a grid only has to fill the device; the merge rounds
    are those ``rows`` slots may need, at least the one that can join every run.
    """
    device = wp.get_device(device)
    sms = device.sm_count if device.is_cuda else 1
    rows = max(int(rows), 1)
    blocks = max(1, min(-(-rows // int(SORT_CHUNK)), _SORT_BLOCKS_PER_SM * sms))
    merge_threads = max(1, min(rows, _MERGE_BLOCKS_PER_SM * sms * _BLOCK_THREADS))
    merges = 1
    while int(SORT_CHUNK) << merges < rows:
        merges += 1
    return blocks, merges, merge_threads


def launch_live_sort(
    pool, keys, indices, extent, capacity: int, *, blocks: int, merges: int, merge_threads: int, slot_keys: bool
) -> None:
    """Sort the live slots of ``keys`` and ``indices`` (first halves) by key, through ``pool``'s launches.

    A kernel published ``extent`` before, followed by a device graph updater
    (:meth:`~newton._src.core.contact_pool.ContactPool.begin_updates`), so a
    captured sort follows the extent of its own replay (:func:`publish_live`).

    Args:
        pool: The :class:`~newton._src.core.contact_pool.ContactPool` whose rows the buffers hold.
        keys: Keys of ``2 * capacity + SORT_CHUNK`` elements, the live slots first; views of the whole
            reservation in CUDA virtual memory, so tile loads past a capture's rows stay in bounds.
        indices: Slot indices of the same layout, ``indices[i] == i`` for the live slots.
        extent: The sort extent (:data:`EXTENT_SIZE` entries) a kernel wrote with :func:`publish_sort_extent`.
        capacity: The rows of each half; in a capture whose launches bind, the ready rows rewrite it.
        blocks: Blocks of the chunk sort (:func:`sort_grid`).
        merges: Merge rounds to launch; a capture notes the slots they pair (:meth:`ContactPool.note_cover`).
        merge_threads: Threads of each merge.
        slot_keys: Keys of at most 51 bits, which sort stably (:func:`slot_keys_fit`).
    """
    kernel = _sort_slot_chunks if slot_keys else _sort_key_chunks
    pool.note_cover(int(SORT_CHUNK) << merges)
    pool.launch(
        kernel,
        blocks,
        inputs=[keys, indices, keys, indices, extent, capacity, blocks, merges],
        extent=extent[1:2],
        clamp=True,
        row_args=(6,),
        tiled=True,
        block_dim=_BLOCK_THREADS,
    )
    for merge in range(merges):
        pool.launch(
            _merge_runs,
            merge_threads,
            inputs=[keys, indices, extent, capacity, merge, merges, merge_threads],
            extent=extent[2 + merge : 3 + merge],
            clamp=True,
            row_args=(4,),
        )


def publish_live(pool, count: wp.array, extent: wp.array) -> None:
    """Write the sort extent of ``count[0]`` live slots, at most ``pool``'s ready rows, then record an updater.

    ``extent[0:1]`` then holds the live slots, a count the launches that sort
    and move them can follow. The updater
    (:meth:`~newton._src.core.contact_pool.ContactPool.begin_updates`) makes
    the launches that follow in a capture read this extent.
    """
    rows = pool.ready_rows if pool.ready_rows is not None else wp.empty(0, dtype=wp.int32, device=extent.device)
    wp.launch(_publish_count, dim=1, inputs=[count, rows, extent], device=extent.device, record_tape=False)
    pool.begin_updates()
