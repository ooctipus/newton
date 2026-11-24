# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sort the live slots of a contact stream by int64 key, with launches that follow the live count.

The slots live in two halves of buffers of ``2 * capacity + SORT_CHUNK``
elements (keys and slot indices): the first half holds the live slots, the
second is scratch, and the chunk past it lets a chunk's tile load stay in the
buffer. A kernel that knows the live count writes the sort's extent with
:func:`publish_sort_extent`: ``[live slots, chunks, work of each merge, chunks
of each sized chunk sort]``. :func:`launch_live_sort` then sorts each chunk of
live slots in shared memory and merges sorted runs pairwise, its last round
every run left, until one run holds every live slot, ending in the first half.
A sort of slot keys takes chunks the live slots size (:func:`sort_chunk`): the
smallest of :data:`SORT_CHUNKS` that holds them, else :data:`SORT_CHUNK`, so a
few hundred live slots sort in one small tile; its order values are unique, so
the order is the same at every chunk.

Order: by key; a merge keeps the slots of its earlier run first among equal
keys. Chunks of *slot keys* (``slot_keys``, keys of at most 51 bits) sort by
(key, slot), so the whole sort is a stable sort by key; other keys sort by key
alone within a chunk. Slot keys with *ties* (``ties``: a unique int64 per slot
index, read at the slot's index) sort by (key, tie) instead, in chunks and in
merges: the order is then a function of the keys and ties alone, whatever
order the slots arrived in.

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
SORT_CHUNKS = (256, 512, 1024, int(SORT_CHUNK))
"""The chunk sizes a sort of slot keys is built for (:func:`sort_chunk` picks one by the live slots)."""
SMALLEST_CHUNK = wp.constant(SORT_CHUNKS[0])
"""The smallest chunk a sort of slot keys takes."""
EXTENT_SIZE = 3 + MAX_MERGES
"""Entries of a sort extent: the live slots, their chunks, the work of each merge, and the chunks of the size
:func:`sort_chunk` picks (the one chunk sort of slot keys)."""
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
def sort_chunk(live: int):
    """Return the chunk a sort of ``live`` slot keys takes: the smallest of :data:`SORT_CHUNKS` that holds every live
    slot, else :data:`SORT_CHUNK`. Below :data:`SORT_CHUNK` the live slots fit one chunk, so no merge runs."""
    chunk = int(SMALLEST_CHUNK)
    while chunk < live and chunk < SORT_CHUNK:
        chunk *= 2
    return chunk


@wp.func
def publish_sort_extent(extent: wp.array[int], base: int, live: int):
    """Write the sort's extent of ``live`` slots at ``extent[base:base + EXTENT_SIZE]``.

    The live slots, their chunks, for each merge the live slots when it has
    runs to join, else 0, and the chunks of the size :func:`sort_chunk` picks:
    a captured sort follows them and launches no work past the live slots.
    """
    extent[base] = live
    extent[base + 1] = (live + SORT_CHUNK - 1) // SORT_CHUNK
    rounds = merge_rounds(live)
    for merge in range(MAX_MERGES):
        work = int(0)
        if merge < rounds:
            work = live
        extent[base + 2 + merge] = work
    picked = sort_chunk(live)
    extent[base + 2 + MAX_MERGES] = (live + picked - 1) // picked


@wp.kernel(enable_backward=False)
def _publish_count(count: wp.array[int], rows: wp.array[wp.int32], limit: int, extent: wp.array[int]):
    live = count[0]
    if rows.shape[0] > 0:
        live = wp.min(live, int(rows[0]))
    if limit >= 0:
        live = wp.min(live, limit)
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


@wp.func
def _tie_order(tie: wp.int64, local: int, start: int, live: int):
    """Order a chunk's live slots by tie; slots past the live count sort last."""
    if start + local < live:
        return tie
    return _KEY_SENTINEL


@wp.func
def _ranked_order(key: wp.int64, rank: int, local: int, start: int, live: int):
    """Order a chunk's slots by (key, the rank of their tie in the chunk); slots past the live count sort last, in
    slot order."""
    if start + local < live:
        return (key << wp.int64(SLOT_BITS)) | wp.int64(rank)
    return _PAD + wp.int64(local)


# Below SORT_CHUNK // 2 slots per half, a chunk's tile reaches past the 2 * capacity
# keys and indices of ordinary arrays: those capacities rely on wp.tile_store's
# bounds check, which drops the stores past the arrays' end. Such slots only hold
# padding past the live slots; fields hold SORT_CHUNK extra slots to avoid it.
def _chunks(order_key, stored_key, chunk: int, label: str):
    """The chunk sort's work of one block: each of its chunks of ``chunk`` slots, ordered by ``order_key`` in shared
    memory and stored by ``stored_key`` from ``output`` on (the half where the merges end in the first).

    Chunks load through ``slot_keys`` and ``slot_indices``; a block stores only the chunk it loaded, so no block reads
    what another writes."""

    def sort_block(
        block: int,
        slot_keys: wp.array[wp.int64],
        slot_indices: wp.array[int],
        keys: wp.array[wp.int64],
        indices: wp.array[int],
        live: int,
        output: int,
        blocks: int,
    ):
        for start in range(block * wp.static(chunk), live, blocks * wp.static(chunk)):
            loaded = wp.tile_load(slot_keys, shape=wp.static(chunk), offset=start)
            local = wp.tile_arange(wp.static(chunk), dtype=int)
            first = wp.tile_full(wp.static(chunk), start, dtype=int)
            count = wp.tile_full(wp.static(chunk), live, dtype=int)
            order = wp.tile_zeros(shape=wp.static(chunk), dtype=wp.int64, storage="shared")
            # Warp hashes a function passed to a builtin only through wp.static();
            # otherwise edits to the key maps would reuse a stale cached kernel.
            wp.tile_assign(order, wp.tile_map(wp.static(order_key), loaded, local, first, count))
            values = wp.tile_load(slot_indices, shape=wp.static(chunk), offset=start, storage="shared")
            wp.tile_sort(order, values)
            wp.tile_store(keys, wp.tile_map(wp.static(stored_key), order), offset=output + start)
            wp.tile_store(indices, values, offset=output + start)

    return wp.func(sort_block, name=f"live_sort_{label}_chunks_{chunk}")


def _tied_chunks(chunk: int):
    """The chunk sort by (slot key, tie) of one block, each of its chunks of ``chunk`` slots (:func:`_chunks`).

    The chunk's slots are first ordered by tie (``ties`` at each slot's index, which the chunk's slots hold as loaded: a
    new frame's slots are their indices), then by (slot key, tie rank): the rank takes the slot's bits of
    :func:`_slot_order`, so the chunk sort stays one key of at most 62 bits. The live slots' ranks are their places in
    tie order whatever the chunk, so every chunk that holds them orders them alike."""

    def sort_block(
        block: int,
        slot_keys: wp.array[wp.int64],
        slot_indices: wp.array[int],
        ties: wp.array[wp.int64],
        keys: wp.array[wp.int64],
        indices: wp.array[int],
        live: int,
        output: int,
        blocks: int,
    ):
        for start in range(block * wp.static(chunk), live, blocks * wp.static(chunk)):
            local = wp.tile_arange(wp.static(chunk), dtype=int)
            first = wp.tile_full(wp.static(chunk), start, dtype=int)
            count = wp.tile_full(wp.static(chunk), live, dtype=int)
            tied = wp.tile_load(ties, shape=wp.static(chunk), offset=start)
            by_tie = wp.tile_zeros(shape=wp.static(chunk), dtype=wp.int64, storage="shared")
            wp.tile_assign(by_tie, wp.tile_map(_tie_order, tied, local, first, count))
            perm = wp.tile_arange(wp.static(chunk), dtype=int, storage="shared")
            wp.tile_sort(by_tie, perm)
            loaded = wp.tile_load_indexed(slot_keys, perm, shape=(wp.static(chunk),), offset=(start,), axis=0)
            values = wp.tile_load_indexed(
                slot_indices, perm, shape=(wp.static(chunk),), offset=(start,), axis=0, storage="shared"
            )
            order = wp.tile_zeros(shape=wp.static(chunk), dtype=wp.int64, storage="shared")
            wp.tile_assign(order, wp.tile_map(_ranked_order, loaded, local, perm, first, count))
            wp.tile_sort(order, values)
            wp.tile_store(keys, wp.tile_map(_slot_key, order), offset=output + start)
            wp.tile_store(indices, values, offset=output + start)

    return wp.func(sort_block, name=f"live_sort_tied_chunks_{chunk}")


def _kernel(function, key: str) -> wp.Kernel:
    return wp.Kernel(function, key=key, options={"enable_backward": False})


_SLOT_CHUNKS = tuple(_chunks(_slot_order, _slot_key, chunk, "slot") for chunk in SORT_CHUNKS)
_TIED_CHUNKS = tuple(_tied_chunks(chunk) for chunk in SORT_CHUNKS)
_KEY_CHUNKS = _chunks(_key_order, _key, int(SORT_CHUNK), "key")


def _sort_slot_chunks_kernel(
    slot_keys: wp.array[wp.int64],
    slot_indices: wp.array[int],
    keys: wp.array[wp.int64],
    indices: wp.array[int],
    extent: wp.array[int],
    capacity: int,
    blocks: int,
    merges: int,
):
    """Sort each chunk of the live slot keys in shared memory, in chunks of the size :func:`sort_chunk` picks (one
    launch for every size: its replay runs the picked size's chunks), into the half where ``merges`` rounds end in the
    first."""
    block = wp.tid()
    live = extent[0]
    output = capacity * (wp.min(merge_rounds(live), merges) % 2)
    picked = sort_chunk(live)
    if picked == wp.static(SORT_CHUNKS[0]):
        wp.static(_SLOT_CHUNKS[0])(block, slot_keys, slot_indices, keys, indices, live, output, blocks)
    elif picked == wp.static(SORT_CHUNKS[1]):
        wp.static(_SLOT_CHUNKS[1])(block, slot_keys, slot_indices, keys, indices, live, output, blocks)
    elif picked == wp.static(SORT_CHUNKS[2]):
        wp.static(_SLOT_CHUNKS[2])(block, slot_keys, slot_indices, keys, indices, live, output, blocks)
    else:
        wp.static(_SLOT_CHUNKS[3])(block, slot_keys, slot_indices, keys, indices, live, output, blocks)


def _sort_tied_chunks_kernel(
    slot_keys: wp.array[wp.int64],
    slot_indices: wp.array[int],
    ties: wp.array[wp.int64],
    keys: wp.array[wp.int64],
    indices: wp.array[int],
    extent: wp.array[int],
    capacity: int,
    blocks: int,
    merges: int,
):
    """Sort each chunk of the live slot keys by (slot key, tie) (:func:`_tied_chunks`), in chunks of the size
    :func:`sort_chunk` picks (one launch for every size), into the half where ``merges`` rounds end in the first."""
    block = wp.tid()
    live = extent[0]
    output = capacity * (wp.min(merge_rounds(live), merges) % 2)
    picked = sort_chunk(live)
    if picked == wp.static(SORT_CHUNKS[0]):
        wp.static(_TIED_CHUNKS[0])(block, slot_keys, slot_indices, ties, keys, indices, live, output, blocks)
    elif picked == wp.static(SORT_CHUNKS[1]):
        wp.static(_TIED_CHUNKS[1])(block, slot_keys, slot_indices, ties, keys, indices, live, output, blocks)
    elif picked == wp.static(SORT_CHUNKS[2]):
        wp.static(_TIED_CHUNKS[2])(block, slot_keys, slot_indices, ties, keys, indices, live, output, blocks)
    else:
        wp.static(_TIED_CHUNKS[3])(block, slot_keys, slot_indices, ties, keys, indices, live, output, blocks)


def _sort_key_chunks_kernel(
    slot_keys: wp.array[wp.int64],
    slot_indices: wp.array[int],
    keys: wp.array[wp.int64],
    indices: wp.array[int],
    extent: wp.array[int],
    capacity: int,
    blocks: int,
    merges: int,
):
    """Sort each SORT_CHUNK chunk of the live keys in shared memory (whose equal keys a smaller chunk could order
    otherwise), into the half where ``merges`` rounds end in the first."""
    block = wp.tid()
    live = extent[0]
    output = capacity * (wp.min(merge_rounds(live), merges) % 2)
    wp.static(_KEY_CHUNKS)(block, slot_keys, slot_indices, keys, indices, live, output, blocks)


# The chunk sorts, each one kernel of this module: of slot keys (by slot, or by tie) at the chunk size the live slots
# pick, of other keys at SORT_CHUNK.
_sort_slot_chunks = _kernel(_sort_slot_chunks_kernel, "live_sort_slot_chunks")
_sort_tied_chunks = _kernel(_sort_tied_chunks_kernel, "live_sort_tied_chunks")
_sort_key_chunks = _kernel(_sort_key_chunks_kernel, "live_sort_key_chunks")


@wp.func
def _placed_before(
    keys: wp.array[wp.int64],
    indices: wp.array[int],
    ties: wp.array[wp.int64],
    source: int,
    lo: int,
    hi: int,
    key: wp.int64,
    tie: wp.int64,
    earlier: bool,
):
    """Count the slots of the sorted run [lo, hi) placed before ``key``: smaller keys, and among equal ones those of
    a smaller tie (with ``ties``) and, at an equal tie or without ties, an ``earlier`` run's."""
    first = lo
    tied = ties.shape[0] > 0
    while lo < hi:
        mid = (lo + hi) // 2
        value = keys[source + mid]
        before = value < key
        if value == key:
            before = earlier
            if tied:
                other = ties[indices[source + mid]]
                before = other < tie or (other == tie and earlier)
        if before:
            lo = mid + 1
        else:
            hi = mid
    return lo - first


@wp.kernel(enable_backward=False)
def _merge_runs(
    keys: wp.array[wp.int64],
    indices: wp.array[int],
    ties: wp.array[wp.int64],
    extent: wp.array[int],
    capacity: int,
    merge: int,
    merges: int,
    threads: int,
):
    """Merge sorted runs pairwise, in the last of the ``merges`` rounds every run left; slots move past smaller keys.

    Among equal keys the smaller tie goes first (with ``ties``), else an earlier run's slots; launches past the
    needed merges return.
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
        tie = wp.int64(0)
        if ties.shape[0] > 0:
            tie = ties[index]
        run = i // width
        group = int(0)
        end = runs
        if not last:
            group = run - run % 2
            end = wp.min(group + 2, runs)
        position = i - (run - group) * width
        # The runs before this slot's are full; the last run may hold fewer slots.
        for other in range(group, run):
            position += _placed_before(
                keys, indices, ties, source, other * width, other * width + width, key, tie, True
            )
        for other in range(run + 1, end):
            position += _placed_before(
                keys, indices, ties, source, other * width, wp.min(other * width + width, live), key, tie, False
            )
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
    pool,
    keys,
    indices,
    extent,
    capacity: int,
    *,
    blocks: int,
    merges: int,
    merge_threads: int,
    slot_keys: bool,
    ties: wp.array | None = None,
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
        ties: With slot keys, an int64 per slot index (the slots' indices name them): equal keys sort by tie, so
            with unique ties the order is the keys' and ties' alone. The live slots' indices must be the slots
            themselves. Like ``keys``, a view of the whole reservation: the chunk sort loads it in tiles, which stop
            at the array's shape.
    """
    if ties is not None and not slot_keys:
        raise ValueError("ties order slot keys only")
    pool.note_cover(int(SORT_CHUNK) << merges)
    if slot_keys:
        # One launch at every chunk size: its replay runs the chunks of the size sort_chunk picks.
        tied = ties is not None
        entry = 2 + int(MAX_MERGES)
        pool.launch(
            _sort_tied_chunks if tied else _sort_slot_chunks,
            blocks,
            inputs=[keys, indices, *((ties,) if tied else ()), keys, indices, extent, capacity, blocks, merges],
            extent=extent[entry : entry + 1],
            clamp=True,
            row_args=(7,) if tied else (6,),
            tiled=True,
            block_dim=_BLOCK_THREADS,
        )
    else:
        pool.launch(
            _sort_key_chunks,
            blocks,
            inputs=[keys, indices, keys, indices, extent, capacity, blocks, merges],
            extent=extent[1:2],
            clamp=True,
            row_args=(6,),
            tiled=True,
            block_dim=_BLOCK_THREADS,
        )
    no_ties = ties if ties is not None else wp.empty(0, dtype=wp.int64, device=keys.device)
    for merge in range(merges):
        pool.launch(
            _merge_runs,
            merge_threads,
            inputs=[keys, indices, no_ties, extent, capacity, merge, merges, merge_threads],
            extent=extent[2 + merge : 3 + merge],
            clamp=True,
            row_args=(5,),
        )


def publish_live(pool, count: wp.array, extent: wp.array, limit: int = -1) -> None:
    """Write the sort extent of ``count[0]`` live slots, at most ``pool``'s ready rows and ``limit`` (when not
    negative: the slots the sorted arrays hold), then record an updater.

    ``extent[0:1]`` then holds the live slots, a count the launches that sort
    and move them can follow. The updater
    (:meth:`~newton._src.core.contact_pool.ContactPool.begin_updates`) makes
    the launches that follow in a capture read this extent.
    """
    rows = pool.ready_rows if pool.ready_rows is not None else wp.empty(0, dtype=wp.int32, device=extent.device)
    wp.launch(_publish_count, dim=1, inputs=[count, rows, limit, extent], device=extent.device, record_tape=False)
    pool.begin_updates()
