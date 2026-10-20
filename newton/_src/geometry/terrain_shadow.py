# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run the terrain law beside the narrow phase, into a side buffer that nothing in the step reads (shadow mode).

``NEWTON_TERRAIN_LAW_SHADOW`` lists partner classes (``point``: spheres and capsules; ``box``; ``cylinder``:
cylinders and cones; ``hull``: convex hulls and ellipsoids; or ``all``, every class with a kernel), empty by default.
Only ``point`` has kernels yet; naming a class without one raises. With classes listed, a
collision pipeline builds :class:`TerrainLawShadow` once, at construction: the terrain features
(:class:`~newton._src.geometry.terrain_contacts.TerrainFeatures`), the static pair table
(:class:`~newton._src.geometry.terrain_contacts.TerrainPairTable`, from the pipeline's pairs) and per listed pair a
static range of row slots. Every collide then launches, after today's terrain route, three kernels per listed class
with dimensions fixed at construction: no host read, no allocation, no scan, so it runs inside CUDA graph capture.
Today's route still produces every contact; the shadow's rows are read only by :meth:`TerrainLawShadow.rows`.

The per-type kernels take ``(begin, slot_begin, inputs, data, out)``
(:class:`~newton._src.geometry.terrain_law.TerrainPairInputs`, :class:`~newton._src.geometry.terrain_law.TerrainFeatureData`,
:class:`~newton._src.geometry.terrain_law.TerrainLawRows`) and a launch argument: table pair ``begin + i`` writes its
rows to slots ``slot_begin + i * K``, ``K`` its class's row limit
(:data:`~newton._src.geometry.terrain_contacts.TERRAIN_PARTNER_ROWS`), its row count and ``TERRAIN_PAIR_*`` state.
Per class (:data:`TERRAIN_LAW_KERNELS` names the kernels; a class without an entry has no kernel yet):

1. The one-thread kernel, one thread per pair of the class (both lane classes: the table keeps them contiguous), with
   the class's :class:`TerrainPairWork`. It walks the pairs whose windows hold at most
   :data:`~newton._src.geometry.terrain_contacts.TERRAIN_LANE_CELLS` cells; a pair with a larger window it only takes
   through the whole-pair rejects (stale features, the pair policy, a window off the heightfield, the tile-max exit),
   and queues it in the work list when it reaches its window.
2. The tiled kernel, ``wp.launch_tiled`` with :data:`TERRAIN_LAW_BLOCK` lanes per pair and the work list's fixed grid
   (:attr:`TerrainPairWork.blocks` blocks), which strides over the queued pairs: a pair that leaves at the tile max
   never occupies a tile.
3. The full-finish kernel (:data:`TERRAIN_LAW_KERNELS`).

The kernels apply the pipeline's pair policy: the table holds the pairs its collision filter admits (worlds,
collision groups, filter pairs, or its explicit pair list), and each kernel drops the pairs its broad phase's
immovable-pair test drops (``include_immobile_pairs``, ``include_static_kinematic_pairs``). Features older than
the heights (an edit reported with ``mark_elevations_changed`` and not yet rebuilt) give no row and raise the
narrow phase's sticky ``terrain_stale`` bit; a pair whose rows overflow its slots keeps each plane group's deepest
row first and raises ``terrain_rows``; ``NarrowPhase.check_buffer_capacity`` raises on either.
"""

from __future__ import annotations

import importlib
import time

import numpy as np
import warp as wp

from .terrain_contacts import (
    TERRAIN_LANE_THREAD,
    TERRAIN_LANE_TILE,
    TERRAIN_PARTNER_BOX,
    TERRAIN_PARTNER_CLASSES,
    TERRAIN_PARTNER_CYLINDER,
    TERRAIN_PARTNER_HULL,
    TERRAIN_PARTNER_POINT,
    TERRAIN_PARTNER_ROWS,
    TerrainFeatures,
    TerrainPairTable,
)
from .terrain_law import (
    TERRAIN_PAIR_CANDIDATE_OVERFLOW,
    TERRAIN_PAIR_FILTERED,
    TERRAIN_PAIR_OUTSIDE,
    TERRAIN_PAIR_ROW_OVERFLOW,
    TERRAIN_PAIR_STALE,
    TERRAIN_PAIR_TILE_EXIT,
    TERRAIN_PAIR_WIDE,
    TERRAIN_ROW_CANDIDATES,
    TerrainLawRows,
    TerrainPairInputs,
    TerrainRows,
)

TERRAIN_LAW_BLOCK = 32
"""Threads per block of every terrain-law launch on CUDA: lanes per pair of the tiled kernels (one warp), and the
one-thread kernel's block (several blocks per SM, so that its warps spread evenly), one module build per device."""
TERRAIN_LAW_TILE_BLOCKS = 2048
"""Blocks of a tiled launch at most, whatever the class's size: one per queued pair up to this many (about one
resident wave of one-warp blocks on a 188-SM GPU), more queued pairs taking turns."""
TERRAIN_LAW_FULL_BLOCKS = 64
"""Blocks of a full-finish launch (:data:`TERRAIN_LAW_BLOCK` lanes each), whatever the class's size: they take the
class's deferred pairs in turn."""


@wp.struct
class TerrainPairWork:
    """A partner class's pairs with larger windows that reach their windows, which its one-thread kernel queues for its
    tiled kernel, in any order (a pair's rows depend on the pair alone)."""

    first: int
    """First table pair of the class's tiled segment: the one-thread kernel walks the pairs before it and queues
    those from it on."""
    pairs: wp.array[wp.int32]
    """The queued table pairs, room for every pair of the tiled segment."""
    count: wp.array[wp.int32]
    """The queued pairs (one entry, zeroed before every launch)."""
    blocks: int
    """Blocks of the tiled launch: block ``b`` takes the queued pairs ``b``, ``b + blocks``, ..."""


def terrain_pair_work(device, first: int, pairs: int, *, count: wp.array | None = None) -> TerrainPairWork:
    """Allocate the :class:`TerrainPairWork` of a tiled segment of ``pairs`` pairs from table pair ``first``.

    Args:
        device: The device.
        first: First table pair of the tiled segment; the one-thread kernel walks the pairs before it.
        pairs: Pairs of the tiled segment.
        count: The one-entry counter of queued pairs (zeroed before every launch), or ``None`` for one of its own.
    """
    work = TerrainPairWork()
    work.first = first
    work.pairs = wp.zeros(max(pairs, 1), dtype=wp.int32, device=device)
    work.count = count if count is not None else wp.zeros(1, dtype=wp.int32, device=device)
    work.blocks = max(min(pairs, TERRAIN_LAW_TILE_BLOCKS), 1)
    return work


def terrain_law_block_dim(device) -> int:
    """Return the block size of every terrain-law launch on ``device``: :data:`TERRAIN_LAW_BLOCK` on CUDA, one on the
    CPU (where ``wp.launch_tiled`` runs one lane per pair whatever it is given, and one size builds the module once)."""
    return TERRAIN_LAW_BLOCK if wp.get_device(device).is_cuda else 1


@wp.struct
class TerrainFinishScratch:
    """Global scratch of a full-finish launch: per block, the candidate rows of the pair it finishes.

    Block ``b`` holds candidate ``j`` at ``b * TERRAIN_ROW_CANDIDATES + j`` of ``q``, ``k``, ``n``, ``d`` and
    ``feature`` (window frame; the fields of :class:`~newton._src.geometry.terrain_law.TerrainRows`) and the finish's
    input at ``rows[b]``, off the per-thread stack: the largest stack a context launches sets the driver's
    local-memory reservation for every thread the device can hold.
    """

    blocks: int
    """Blocks of the launch: its grid stride over the deferred list."""
    q: wp.array[wp.vec3]
    """Terrain point of every candidate [m]."""
    k: wp.array[wp.vec3]
    """Core point of every candidate [m]."""
    n: wp.array[wp.vec3]
    """Unit normal of every candidate."""
    d: wp.array[float]
    """Core distance of every candidate [m]."""
    feature: wp.array[wp.int32]
    """Feature id of every candidate."""
    rows: wp.array[TerrainRows]
    """Per thread, the finish's input (:func:`~newton._src.geometry.terrain_law.terrain_finish`)."""


def terrain_finish_scratch(device, blocks: int = TERRAIN_LAW_FULL_BLOCKS) -> TerrainFinishScratch:
    """Allocate the :class:`TerrainFinishScratch` of a full-finish launch of ``blocks`` blocks."""
    scratch = TerrainFinishScratch()
    scratch.blocks = blocks
    candidates = blocks * TERRAIN_ROW_CANDIDATES
    scratch.q = wp.zeros(candidates, dtype=wp.vec3, device=device)
    scratch.k = wp.zeros(candidates, dtype=wp.vec3, device=device)
    scratch.n = wp.zeros(candidates, dtype=wp.vec3, device=device)
    scratch.d = wp.zeros(candidates, dtype=wp.float32, device=device)
    scratch.feature = wp.zeros(candidates, dtype=wp.int32, device=device)
    scratch.rows = wp.zeros(blocks, dtype=TerrainRows, device=device)
    return scratch


TERRAIN_LAW_CLASS_NAMES = {
    "point": TERRAIN_PARTNER_POINT,
    "box": TERRAIN_PARTNER_BOX,
    "cylinder": TERRAIN_PARTNER_CYLINDER,
    "hull": TERRAIN_PARTNER_HULL,
}
"""``NEWTON_TERRAIN_LAW_SHADOW`` names of the partner classes."""

TERRAIN_LAW_KERNELS: dict[int, tuple[str, str, str, str | None]] = {
    TERRAIN_PARTNER_POINT: (
        "newton._src.geometry.terrain_law_point",
        "terrain_point_rows",
        "terrain_point_rows_tiled",
        "terrain_point_rows_full",
    ),
}
"""Per partner class: the module, its one-thread and tiled kernels, and its full-finish kernel or ``None``.

The module is imported only when the class is listed. The one-thread kernel runs over both lane segments of the class
and the tiled one over the queued pairs of its tiled segment, each with the class's :class:`TerrainPairWork` as sixth
argument (``begin`` and ``slot_begin`` are the class's first pair and slot for the one-thread kernel, the tiled
segment's for the tiled one). The full-finish kernel, when the class has one, runs after them and finishes the pairs
they marked :data:`~newton._src.geometry.terrain_law.TERRAIN_PAIR_FULL`: ``wp.launch_tiled`` with
:data:`TERRAIN_LAW_FULL_BLOCKS` blocks of :data:`TERRAIN_LAW_BLOCK` lanes whatever the class's size, striding over its
deferred list, and a :class:`TerrainFinishScratch` as sixth argument (``begin`` and ``slot_begin`` are the class's), so
that an empty list costs one near-empty launch.

A class's module and kernels follow one protocol (the helpers live in :mod:`~newton._src.geometry.terrain_law`):

* The module compiles without floating-point contraction (``wp.set_module_options({"fuse_fp": False})``), so that the
  CPU and CUDA round every float32 operation alike and decide every predicate boundary on the same bits: a CPU test or
  census then counts what the GPU does.
* A pair that has no row (stale features, the pair policy, a window off the heightfield, the tile-max exit) is
  recorded by ``terrain_pair_exit``.
* While its window is walked, a pair may use its ``K`` slots as scratch, in the window frame. The point kernels hold
  up to ``K - 2`` distinct rows there, merged as the finish merges (``terrain_hold_row``), and keep the best exit of
  each core end inside the solid in the last two slots (with both ends outside, the last slot holds the capsule
  axis's deepest crossing for its G3); an exit is read before a row may land in its slot.
* When the finish keeps the held rows as they are (no plane group of three, ``terrain_plane_group_of_three``; at most
  ``K`` rows; the candidates fit the finish's list), the kernel writes them in the world frame (``terrain_store_row``)
  with the pair's count and state. Otherwise it defers the pair (``terrain_defer_pair``), and the class's full-finish
  kernel walks it again into the full candidate list and runs ``terrain_finish``.
"""

TERRAIN_LANE_NAMES = {TERRAIN_LANE_THREAD: "thread", TERRAIN_LANE_TILE: "tiled"}
"""Names of the pair table's lane classes, by id (the table orders a class's pairs by lane class id)."""

TERRAIN_PAIR_STATES = {
    "tile_exit": int(TERRAIN_PAIR_TILE_EXIT),
    "stale": int(TERRAIN_PAIR_STALE),
    "filtered": int(TERRAIN_PAIR_FILTERED),
    "outside": int(TERRAIN_PAIR_OUTSIDE),
    "row_overflow": int(TERRAIN_PAIR_ROW_OVERFLOW),
    "candidate_overflow": int(TERRAIN_PAIR_CANDIDATE_OVERFLOW),
    "wide": int(TERRAIN_PAIR_WIDE),
}
"""Names of the ``TERRAIN_PAIR_*`` state bits."""


def terrain_law_class_name(partner_class: int) -> str:
    """Return the ``NEWTON_TERRAIN_LAW_SHADOW`` name of a partner class."""
    return next(name for name, value in TERRAIN_LAW_CLASS_NAMES.items() if value == partner_class)


def parse_terrain_law_shadow(value: str | None) -> tuple[int, ...]:
    """Return the partner classes a ``NEWTON_TERRAIN_LAW_SHADOW`` value lists, sorted (empty for none).

    Args:
        value: Comma-separated names of :data:`TERRAIN_LAW_CLASS_NAMES`, or ``all`` for every class with a
            kernel; ``None`` or blank for none.

    Raises:
        ValueError: On an unknown name, or a named class without a kernel yet.
    """
    if value is None:
        return ()
    classes = set()
    for name in (item.strip() for item in value.split(",")):
        if not name:
            continue
        if name == "all":
            classes.update(TERRAIN_LAW_KERNELS)
            continue
        if name not in TERRAIN_LAW_CLASS_NAMES:
            raise ValueError(
                f"NEWTON_TERRAIN_LAW_SHADOW names unknown partner class {name!r}; "
                f"use a comma-separated list of {sorted(TERRAIN_LAW_CLASS_NAMES)} or 'all'"
            )
        partner_class = TERRAIN_LAW_CLASS_NAMES[name]
        if partner_class not in TERRAIN_LAW_KERNELS:
            raise ValueError(f"NEWTON_TERRAIN_LAW_SHADOW: the terrain law has no kernel for {name!r} yet")
        classes.add(partner_class)
    return tuple(sorted(classes))


class TerrainLawShadow:
    """The terrain law's kernels for some partner classes, run after the narrow phase into a side buffer.

    Built once, outside graph capture: the terrain features, the pair table, the side buffer (per listed pair
    ``K`` slots of a terrain point, a normal, a core distance and a feature id, plus a row count and a state), per
    class its :class:`TerrainPairWork` and, with a full-finish kernel, its :class:`TerrainFinishScratch`.
    :meth:`launch` then runs each class's kernels with dimensions fixed here; the narrow phase's contacts are
    untouched.

    Args:
        model: A finalized model.
        classes: Partner classes to run (:func:`parse_terrain_law_shadow`); each must have a kernel.
        pairs: The collision pipeline's explicit shape pairs, or ``None`` for the model's filter.
        status: The narrow phase's sticky buffer status word, or ``None`` for one of the shadow's own.
    """

    def __init__(self, model, classes, *, pairs=None, status: wp.array | None = None):
        started = time.perf_counter()
        self.device = model.device
        self.classes = tuple(sorted(int(c) for c in classes))
        missing = [c for c in self.classes if c not in TERRAIN_LAW_KERNELS]
        if missing:
            raise ValueError(f"The terrain law has no kernel for partner classes {missing}")
        self.table = TerrainPairTable(model, pairs=pairs)
        """The static (heightfield, partner) pair table."""
        self._launches = []
        slot_first = np.full(self.table.pair_count, -1, np.int64)
        slot = 0
        device = self.device
        # Per class its deferred pairs, then per class its queued pairs: zeroed by one memset before every launch.
        self._counters = wp.zeros(2 * TERRAIN_PARTNER_CLASSES, dtype=wp.int32, device=device)
        for partner_class in self.classes:
            module_name, thread_name, tiled_name, full_name = TERRAIN_LAW_KERNELS[partner_class]
            module = importlib.import_module(module_name)
            rows = TERRAIN_PARTNER_ROWS[partner_class]
            # The class's lane segments are contiguous in the table (lane classes by id); its slots follow them.
            begin = self.table.segment(partner_class, TERRAIN_LANE_THREAD)[0]
            tiled_begin, end = self.table.segment(partner_class, TERRAIN_LANE_TILE)
            if end == begin:
                continue
            slot_first[begin:end] = slot + rows * np.arange(end - begin)
            counter = TERRAIN_PARTNER_CLASSES + partner_class
            work = terrain_pair_work(
                device, tiled_begin, end - tiled_begin, count=self._counters[counter : counter + 1]
            )
            # One thread per pair of the class: it walks the small windows and queues the larger ones it does not
            # reject whole, which the tiled kernel's fixed grid then takes in turn.
            self._launches.append(("thread", getattr(module, thread_name), end - begin, begin, slot, work))
            if end > tiled_begin:
                tiled_slot = slot + rows * (tiled_begin - begin)
                self._launches.append(
                    ("tiled", getattr(module, tiled_name), work.blocks, tiled_begin, tiled_slot, work)
                )
            if full_name is not None:
                # A fixed grid over the class's deferred list (the pairs are found on the device, in any order).
                scratch = terrain_finish_scratch(device)
                self._launches.append(("full", getattr(module, full_name), scratch.blocks, begin, slot, scratch))
            slot += rows * (end - begin)
        self.slot_count = slot
        """Row slots of the side buffer: every listed pair times its class's row limit."""
        self.features = TerrainFeatures(model) if self._launches else None
        """The heightfields' terrain features (``None`` without a listed pair); report height edits to it and rebuild
        outside capture."""
        self._slot_first = slot_first
        self._listed = np.flatnonzero(slot_first >= 0)
        out = TerrainLawRows()
        slots = max(slot, 1)
        pair_count = max(self.table.pair_count, 1)
        out.q = wp.zeros(slots, dtype=wp.vec3, device=device)
        out.n = wp.zeros(slots, dtype=wp.vec3, device=device)
        out.d = wp.zeros(slots, dtype=wp.float32, device=device)
        out.feature = wp.zeros(slots, dtype=wp.int32, device=device)
        out.count = wp.zeros(pair_count, dtype=wp.int32, device=device)
        out.state = wp.zeros(pair_count, dtype=wp.int32, device=device)
        out.status = status if status is not None else wp.zeros(1, dtype=wp.int32, device=device)
        out.overflow = wp.zeros(2, dtype=wp.int32, device=device)
        out.deferred = wp.zeros(pair_count, dtype=wp.int32, device=device)
        out.deferred_first = wp.array(
            [self.table.segment(c, 0)[0] for c in range(TERRAIN_PARTNER_CLASSES)], dtype=wp.int32, device=device
        )
        out.deferred_count = self._counters[:TERRAIN_PARTNER_CLASSES]
        self._out = out
        self._no_body_flags = wp.empty(0, dtype=wp.int32, device=device)
        self.build_seconds = time.perf_counter() - started
        """Host time of the construction [s]: features, pair table, kernel modules imported and buffers."""

    # ---- per collide -------------------------------------------------------------------------------------------
    def launch(
        self,
        shape_transform: wp.array,
        shape_data: wp.array,
        *,
        shape_type: wp.array,
        shape_gap: wp.array,
        shape_heightfield_index: wp.array,
        shape_body: wp.array,
        body_flags: wp.array | None,
        include_static_kinematic_pairs: bool,
    ) -> None:
        """Evaluate every listed pair into the side buffer (graph-capture safe: fixed dimensions, no host read).

        Args:
            shape_transform: World transform of every shape, current for this collide.
            shape_data: Per shape, scale ``xyz`` and margin ``w``, current for this collide.
            shape_type: Geometry type of every shape.
            shape_gap: Contact gap of every shape [m].
            shape_heightfield_index: Heightfield index of every shape.
            shape_body: The broad phase's immovable-pair test: body of every shape.
            body_flags: The broad phase's immovable-pair test: body flags, or ``None``.
            include_static_kinematic_pairs: The broad phase's immovable-pair test: keep pairs of two immovable
                shapes.
        """
        if not self._launches:
            return
        inputs = TerrainPairInputs()
        inputs.pairs = self.table.pairs
        inputs.shape_type = shape_type
        inputs.shape_transform = shape_transform
        inputs.shape_data = shape_data
        inputs.shape_gap = shape_gap
        inputs.shape_heightfield_index = shape_heightfield_index
        inputs.shape_body = shape_body
        inputs.body_flags = body_flags if body_flags is not None else self._no_body_flags
        inputs.include_static_kinematic_pairs = int(bool(include_static_kinematic_pairs))
        data = self.features.data()
        self._counters.zero_()
        block = terrain_law_block_dim(self.device)
        for kind, kernel, dim, begin, slot, extra in self._launches:
            args = [begin, slot, inputs, data, self._out, extra]
            if kind == "thread":
                wp.launch(kernel, dim=dim, inputs=args, block_dim=block, device=self.device, record_tape=False)
            else:
                wp.launch_tiled(kernel, dim=dim, inputs=args, block_dim=block, device=self.device, record_tape=False)

    # ---- host reads --------------------------------------------------------------------------------------------
    def rows(self) -> dict[str, np.ndarray]:
        """Return the listed pairs and their rows of the last launch (host copies; call outside graph capture).

        Returns:
            Per listed pair, in table order: ``pairs`` (heightfield shape, partner shape), ``partner_class``,
            ``lane_class``, ``world``, ``count`` (rows) and ``state`` (``TERRAIN_PAIR_*`` bits, see
            :data:`TERRAIN_PAIR_STATES`); per row, pair by pair: ``row_pair`` (index into the listed pairs), ``q``
            (terrain point [m]), ``n`` (unit normal, terrain to partner), ``d`` (core distance [m]) and ``feature``
            (window-local feature id), all in the world frame.
        """
        if self.device.is_capturing:
            raise RuntimeError("Terrain-law shadow rows must be read outside graph capture")
        listed = self._listed
        out = self._out
        count = out.count.numpy()[listed].astype(np.int64) if len(listed) else np.zeros(0, np.int64)
        state = out.state.numpy()[listed] if len(listed) else np.zeros(0, np.int32)
        first = self._slot_first[listed]
        row_pair = np.repeat(np.arange(len(listed)), count)
        offsets = np.concatenate([[0], np.cumsum(count)])[:-1]
        slots = np.repeat(first, count) + (np.arange(int(count.sum())) - np.repeat(offsets, count))
        q, n, d, feature = (array.numpy()[slots] for array in (out.q, out.n, out.d, out.feature))
        return {
            "pairs": self.table.pairs_host[listed],
            "partner_class": self.table.partner_class[listed],
            "lane_class": self.table.lane_class[listed],
            "world": self.table.world[listed],
            "count": count.astype(np.int32),
            "state": state,
            "row_pair": row_pair,
            "q": q,
            "n": n,
            "d": d,
            "feature": feature,
        }

    def overflow_counts(self) -> tuple[int, int]:
        """Return the pairs whose rows and whose candidates overflowed, summed over every launch so far."""
        if self.device.is_capturing:
            raise RuntimeError("Terrain-law shadow counters must be read outside graph capture")
        rows, candidates = (int(v) for v in self._out.overflow.numpy())
        return rows, candidates

    def describe(self) -> dict:
        """Return a host-only record of the shadow: classes, pairs and launches per lane class, memory."""
        pairs = {}
        for partner_class in range(TERRAIN_PARTNER_CLASSES):
            if partner_class not in self.classes:
                continue
            per_lane = {}
            for lane, name in TERRAIN_LANE_NAMES.items():
                begin, end = self.table.segment(partner_class, lane)
                per_lane[name] = end - begin
            pairs[terrain_law_class_name(partner_class)] = per_lane
        out = self._out
        side = sum(
            array.capacity
            for array in (out.q, out.n, out.d, out.feature, out.count, out.state, out.overflow, out.deferred)
        )
        scratch = sum(
            sum(array.capacity for array in (s.q, s.k, s.n, s.d, s.feature, s.rows))
            for kind, *_launch, s in self._launches
            if kind == "full"
        )
        work = sum(w.pairs.capacity for kind, *_launch, w in self._launches if kind == "thread")
        return {
            "classes": [terrain_law_class_name(c) for c in self.classes],
            "pairs": pairs,
            "launches": len(self._launches),
            "slots": self.slot_count,
            "bytes": {
                "side_buffer": int(side + self._counters.capacity),
                "work_lists": int(work),
                "finish_scratch": int(scratch),
                "pair_table": int(self.table.memory_bytes()["pairs"]),
                "features": 0 if self.features is None else int(sum(self.features.memory_bytes().values())),
            },
            "build_seconds": round(self.build_seconds, 3),
        }
