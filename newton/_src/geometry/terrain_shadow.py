# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run the terrain law beside the narrow phase, into a side buffer that nothing in the step reads (shadow mode).

``NEWTON_TERRAIN_LAW_SHADOW`` lists partner classes (``point``: spheres and capsules; ``box``; ``cylinder``:
cylinders and cones; ``hull``: convex hulls and ellipsoids; or ``all``, every class with a kernel), empty by default.
With classes listed, a collision pipeline builds :class:`TerrainLawShadow` once, at construction: the terrain features
(:class:`~newton._src.geometry.terrain_contacts.TerrainFeatures`), the static pair table
(:class:`~newton._src.geometry.terrain_contacts.TerrainPairTable`, from the pipeline's pairs), per listed pair a static
range of walk slots (:data:`~newton._src.geometry.terrain_contacts.TERRAIN_PARTNER_WALK_SLOTS`, the fast path's
layout, which bounds no row) and one spill region for all the rows of pairs that have more
(:class:`~newton._src.geometry.terrain_law.TerrainLawRows`), sized at the scene level, as ``rigid_contact_max`` sizes
contacts: by default from the listed pairs' own bounds on their rows (:meth:`TerrainLawShadow.default_spill_rows`);
``NEWTON_TERRAIN_LAW_SHADOW_ROWS`` sets it. Every collide then launches, after today's terrain route, up to four
kernels per listed class (the tiled kernel only for a class with a tiled segment, the exit kernel only for a solid
class) and a check of the spill region, with dimensions fixed at construction (:meth:`TerrainLawShadow.describe`
records them): no host read, no allocation, no scan, so it runs inside CUDA graph capture.
Today's route still produces every contact; the shadow's rows are read only by :meth:`TerrainLawShadow.rows`.

The per-type kernels take ``(begin, slot_begin, inputs, data, out)``
(:class:`~newton._src.geometry.terrain_law.TerrainPairInputs`,
:class:`~newton._src.geometry.terrain_law.TerrainFeatureData`,
:class:`~newton._src.geometry.terrain_law.TerrainLawRows`) and a launch argument, the solid classes' also their
:class:`~newton._src.geometry.terrain_law_solid.TerrainSolidData` (hull tables, the one-thread kernels' lane choice
and rule X's exit lists; built once when a solid class is listed): table pair ``begin + i`` walks its window in its
walk slots ``slot_begin + i * K``, ``K`` its class's walk slots, and writes its rows there, or when they are more, to
rows of the spill region; its first row, row count and ``TERRAIN_PAIR_*`` state. :data:`TERRAIN_LAW_KERNELS` names the
kernels per class (a class without an entry has no kernel yet). Every class launches them alike, in blocks of
:func:`~newton._src.geometry.terrain_law.terrain_law_block_dim` threads:

1. The one-thread kernel, one thread per pair of the class (both lane classes: the table keeps them contiguous), with
   the class's :class:`~newton._src.geometry.terrain_law.TerrainPairWork`. It walks the pairs whose windows hold at
   most :data:`~newton._src.geometry.terrain_contacts.TERRAIN_LANE_CELLS` cells over every orientation; a pair with a
   larger window it takes through the whole-pair rejects (stale features, the pair policy, a window off the
   heightfield, the tile-max exit), and when it reaches its window queues it in the work list: the point class every
   such pair, a solid class those whose windows, as their poses give them, hold more than
   :data:`~newton._src.geometry.terrain_law_solid.TERRAIN_SOLID_THREAD_VERTICES` vertices (it walks the others).
2. The tiled kernel, ``wp.launch_tiled`` with :data:`~newton._src.geometry.terrain_law.TERRAIN_LAW_BLOCK` lanes per
   pair and the work list's fixed grid (:attr:`~newton._src.geometry.terrain_law.TerrainPairWork.blocks` blocks),
   which strides over the queued pairs: a pair that leaves at the tile max never occupies a tile.
3. A solid class's exit kernel, one thread per pair of the class, on the pairs its walks left to rule X with their
   rows held (a core reaching below its window's highest vertex): their deepest exit and their finish.
4. The full-finish kernel, ``wp.launch_tiled`` on the fixed grid of its
   :class:`~newton._src.geometry.terrain_law.TerrainFinishScratch`
   (:func:`~newton._src.geometry.terrain_law.terrain_finish_blocks` blocks, each holding every distinct candidate a
   window of the class can give), which strides over the class's deferred list.

The kernels apply the pipeline's pair policy: the table holds the pairs its collision filter admits (worlds,
collision groups, filter pairs, or its explicit pair list), and each kernel drops the pairs its broad phase's
immovable-pair test drops (``include_immobile_pairs``, ``include_static_kinematic_pairs``). Features older than
the heights (an edit reported with ``mark_elevations_changed`` and not yet rebuilt) give no row and raise the
narrow phase's sticky ``terrain_stale`` bit; a pair whose rows the spill region has no room for keeps the first of
them by priority in its walk slots (each plane group's deepest first) and raises ``terrain_rows`` (the spill check
prints the rows asked for against the region's), as does a pair whose window outgrew the bound its shape data gave at
construction; ``NarrowPhase.check_buffer_capacity`` raises on either.
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
    TERRAIN_PARTNER_WALK_SLOTS,
    TerrainFeatures,
    TerrainPairTable,
)
from .terrain_law import (
    TERRAIN_FINISH_ROW_BYTES,
    TERRAIN_LAW_FULL_BLOCKS,
    TERRAIN_PAIR_CANDIDATE_OVERFLOW,
    TERRAIN_PAIR_FILTERED,
    TERRAIN_PAIR_OUTSIDE,
    TERRAIN_PAIR_ROW_OVERFLOW,
    TERRAIN_PAIR_STALE,
    TERRAIN_PAIR_TILE_EXIT,
    TERRAIN_PAIR_UNSUPPORTED,
    TERRAIN_PAIR_WIDE,
    TerrainPairInputs,
    terrain_finish_scratch,
    terrain_law_block_dim,
    terrain_law_rows,
    terrain_pair_work,
)

TERRAIN_LAW_CLASS_NAMES = {
    "point": TERRAIN_PARTNER_POINT,
    "box": TERRAIN_PARTNER_BOX,
    "cylinder": TERRAIN_PARTNER_CYLINDER,
    "hull": TERRAIN_PARTNER_HULL,
}
"""``NEWTON_TERRAIN_LAW_SHADOW`` names of the partner classes."""

TERRAIN_LAW_KERNELS: dict[int, tuple[str, str, str, str | None, str | None]] = {
    TERRAIN_PARTNER_POINT: (
        "newton._src.geometry.terrain_law_point",
        "terrain_point_rows",
        "terrain_point_rows_tiled",
        None,
        "terrain_point_rows_full",
    ),
    TERRAIN_PARTNER_BOX: (
        "newton._src.geometry.terrain_law_box",
        "terrain_box_rows",
        "terrain_box_rows_tiled",
        "terrain_box_rows_exit",
        "terrain_box_rows_full",
    ),
    TERRAIN_PARTNER_CYLINDER: (
        "newton._src.geometry.terrain_law_cylinder",
        "terrain_cylinder_rows",
        "terrain_cylinder_rows_tiled",
        "terrain_cylinder_rows_exit",
        "terrain_cylinder_rows_full",
    ),
    TERRAIN_PARTNER_HULL: (
        "newton._src.geometry.terrain_law_hull",
        "terrain_hull_rows",
        "terrain_hull_rows_tiled",
        "terrain_hull_rows_exit",
        "terrain_hull_rows_full",
    ),
}
"""Per partner class: the module, its one-thread and tiled kernels, its exit kernel (rule X of the solid classes) or
``None``, and its full-finish kernel or ``None``.

The module is imported only when the class is listed. Every class's kernels take one launch mechanism, with
dimensions fixed at construction. The one-thread kernel runs over both lane segments of the class and the tiled one
over the queued pairs of its tiled segment, each with the class's
:class:`~newton._src.geometry.terrain_law.TerrainPairWork` as sixth argument (``begin`` and ``slot_begin`` are the
class's first pair and walk slot for the one-thread kernel, the tiled segment's for the tiled one). A solid class's
exit kernel runs next, one thread per pair of the class (``begin`` and ``slot_begin`` the class's), on the pairs its
walks left to rule X with their rows held (:data:`~newton._src.geometry.terrain_law.TERRAIN_PAIR_EXIT`, the list in
the solid classes' data). The full-finish kernel, when the class has one, runs last and finishes the pairs they marked
:data:`~newton._src.geometry.terrain_law.TERRAIN_PAIR_FULL`: ``wp.launch_tiled`` with
:func:`~newton._src.geometry.terrain_law.terrain_finish_blocks` blocks of
:data:`~newton._src.geometry.terrain_law.TERRAIN_LAW_BLOCK` lanes whatever the class's size, striding over its
deferred list, and a :class:`~newton._src.geometry.terrain_law.TerrainFinishScratch` as sixth argument (``begin`` and
``slot_begin`` are the class's), so that an empty list costs one near-empty launch. A solid class's kernels take the
solid classes' data as seventh argument (:data:`TERRAIN_LAW_SOLID_CLASSES`).

A class's module and kernels follow one protocol (the helpers live in :mod:`~newton._src.geometry.terrain_law`):

* The module compiles without floating-point contraction (``wp.set_module_options({"fuse_fp": False})``), so that the
  CPU and CUDA round every float32 operation alike and decide every predicate boundary on the same bits: a CPU test or
  census then counts what the GPU does.
* A pair that has no row (stale features, the pair policy, a window off the heightfield, the tile-max exit) is
  recorded by ``terrain_pair_exit``.
* While its window is walked, a pair may use its ``K`` walk slots as scratch, in the window frame, its distinct rows
  merged as the finish merges (``terrain_hold_row``). The point kernels hold up to ``K - 2`` there and keep the best
  exit of each core end inside the solid in the last two slots (with both ends outside, the last slot holds the
  capsule axis's deepest crossing for its G3); an exit is read before a row may land in its slot. The solid kernels
  hold up to ``K``.
* When the finish keeps the held rows as they are (no plane group of three, ``terrain_plane_group_of_three``; at most
  ``K`` rows), the point kernels write them in the world frame (``terrain_store_row``) with the pair's count and state;
  the solid kernels run the finish in place on any held rows (``terrain_finish_slots``) and write the kept ones. A pair
  beyond that (for the solid kernels: more distinct rows than ``K``) is deferred (``terrain_defer_pair``), and the
  class's full-finish kernel walks it again into its distinct candidates (``terrain_hold_candidate``), runs the finish
  on all of them (``terrain_finish_list``) and writes every row it keeps, in the pair's walk slots when they fit, else
  in rows of the spill region it reserves (``terrain_spill_rows``; no room: the first ``K`` of them by priority in
  the walk slots, ``terrain_finish_overflow``, state ``TERRAIN_PAIR_ROW_OVERFLOW``, ``terrain_rows``).
"""

TERRAIN_LAW_SOLID_CLASSES = (TERRAIN_PARTNER_BOX, TERRAIN_PARTNER_CYLINDER, TERRAIN_PARTNER_HULL)
"""Partner classes whose kernels take a seventh argument, the solid classes' data
(:class:`~newton._src.geometry.terrain_law_solid.TerrainSolidData`: hull tables, the one-thread kernels' lane
choice and rule X's exit lists), built once when one of them is listed."""

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
    "unsupported": int(TERRAIN_PAIR_UNSUPPORTED),
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


TERRAIN_LAW_SPILL_SHARE = 256
"""The default spill region holds the sum of the listed pairs' bounds on their rows over this, beyond the largest
pair's bound (:meth:`TerrainLawShadow.default_spill_rows`): the worst case of one pair in 256 at once. On the recorded
suite scenes (16,384 worlds) that is 11 to 25% of the walk slots, while at most one pair spilled (30 rows)."""

_DEFERRED = 0
"""Offset of the per-class deferred-pair counters in the shadow's counters."""
_QUEUED = TERRAIN_PARTNER_CLASSES
"""Offset of the per-class tiled-queue counters."""
_SPILLED = 2 * TERRAIN_PARTNER_CLASSES
"""Offset of the spill rows asked for."""
_RESERVED = _SPILLED + 1
"""Offset of the spill rows reserved."""
_EXITS = _RESERVED + 1
"""Offset of the per-class counters of the pairs left to rule X's exit kernels."""
_EXIT_WORK = _EXITS + TERRAIN_PARTNER_CLASSES
"""Offset of rule X's exit searches and the window vertices they covered."""
_COUNTERS = _EXIT_WORK + 2
"""The shadow's counters, all zeroed by one memset before every launch."""


@wp.kernel(enable_backward=False)
def terrain_law_spill_check(spilled: wp.array[wp.int32], spill: int):
    """After a launch: warn when the spill region overflowed, with the spill rows asked for against its rows (the pairs
    that found no room have raised ``terrain_rows``)."""
    if spilled[0] > spill:
        wp.printf(
            "Warning: Terrain-law spill region overflowed %d > %d. Set NEWTON_TERRAIN_LAW_SHADOW_ROWS to at least the "
            "first number.\n",
            spilled[0],
            spill,
        )


def parse_terrain_law_shadow_rows(value: str | None) -> int | None:
    """Return the spill region's rows a ``NEWTON_TERRAIN_LAW_SHADOW_ROWS`` value sets, or ``None`` for the default.

    Args:
        value: A non-negative row count; ``None`` or blank for the default
            (:meth:`TerrainLawShadow.default_spill_rows`).

    Raises:
        ValueError: On a value that is not a non-negative integer.
    """
    if value is None or not value.strip():
        return None
    try:
        rows = int(value.strip())
    except ValueError:
        rows = -1
    if rows < 0:
        raise ValueError(f"NEWTON_TERRAIN_LAW_SHADOW_ROWS must be a non-negative row count, not {value!r}")
    return rows


class TerrainLawShadow:
    """The terrain law's kernels for some partner classes, run after the narrow phase into a side buffer.

    Built once, outside graph capture: the terrain features, the pair table, the side buffer (per listed pair ``K``
    walk slots and one spill region of terrain points, normals, core distances and feature ids, and per listed pair its
    first row, row count and state), per class its :class:`~newton._src.geometry.terrain_law.TerrainPairWork` and,
    with a full-finish kernel, its
    :class:`~newton._src.geometry.terrain_law.TerrainFinishScratch` (sized from the class's pairs' windows), and with a
    solid class listed the solid classes' data (:class:`~newton._src.geometry.terrain_law_solid.TerrainSolidData`).
    :meth:`launch` then runs each class's kernels with dimensions fixed here; the narrow phase's contacts are untouched.

    Args:
        model: A finalized model.
        classes: Partner classes to run (:func:`parse_terrain_law_shadow`); each must have a kernel.
        pairs: The collision pipeline's explicit shape pairs, or ``None`` for the model's filter.
        status: The narrow phase's sticky buffer status word, or ``None`` for one of the shadow's own.
        spill_rows: Rows of the spill region, or ``None`` for :meth:`default_spill_rows` (from the listed pairs'
            bounds on their rows).
    """

    def __init__(self, model, classes, *, pairs=None, status: wp.array | None = None, spill_rows: int | None = None):
        started = time.perf_counter()
        self.device = model.device
        self.classes = tuple(sorted(int(c) for c in classes))
        missing = [c for c in self.classes if c not in TERRAIN_LAW_KERNELS]
        if missing:
            raise ValueError(f"The terrain law has no kernel for partner classes {missing}")
        self.table = TerrainPairTable(model, pairs=pairs)
        """The static (heightfield, partner) pair table."""
        self._launches = []
        self._launch_shapes = {}
        block = terrain_law_block_dim(model.device)
        slot_first = np.full(self.table.pair_count, -1, np.int64)
        slot = 0
        device = self.device
        # Per class its deferred pairs, per class its queued pairs, the spill rows asked for and reserved, per class its
        # pairs left to rule X, and rule X's exit searches and their window vertices: zeroed by one memset before every
        # launch.
        self._counters = wp.zeros(_COUNTERS, dtype=wp.int32, device=device)
        solid_pairs = []
        solid_module = None
        for partner_class in self.classes:
            module_name, thread_name, tiled_name, exit_name, full_name = TERRAIN_LAW_KERNELS[partner_class]
            module = importlib.import_module(module_name)
            walk = int(TERRAIN_PARTNER_WALK_SLOTS[partner_class])
            # The class's lane segments are contiguous in the table (lane classes by id); its walk slots follow them.
            begin = self.table.segment(partner_class, TERRAIN_LANE_THREAD)[0]
            tiled_begin, end = self.table.segment(partner_class, TERRAIN_LANE_TILE)
            if end == begin:
                continue
            slot_first[begin:end] = slot + walk * np.arange(end - begin)
            # A solid class's kernels also take the solid classes' data (built below, once).
            solid = partner_class in TERRAIN_LAW_SOLID_CLASSES
            if solid:
                if solid_module is None:
                    solid_module = importlib.import_module("newton._src.geometry.terrain_law_solid")
                solid_pairs.append((begin, end - begin))
            counter = _QUEUED + partner_class
            work = terrain_pair_work(
                device, tiled_begin, end - tiled_begin, count=self._counters[counter : counter + 1]
            )
            # One thread per pair of the class: it walks the small windows and queues the larger ones it does not
            # reject whole (a solid class: those its pose makes large), which the tiled kernel's fixed grid then takes
            # in turn.
            self._launches.append(("thread", getattr(module, thread_name), end - begin, begin, slot, solid, work))
            shapes = {
                "block_dim": block,
                "thread_dim": end - begin,
                "tiled_blocks": 0,
                "exit_dim": 0,
                "full_blocks": 0,
                "full_candidates": 0,
                "full_limited": False,
                "walk_slots": walk,
            }
            if end > tiled_begin:
                tiled_slot = slot + walk * (tiled_begin - begin)
                self._launches.append(
                    ("tiled", getattr(module, tiled_name), work.blocks, tiled_begin, tiled_slot, solid, work)
                )
                shapes["tiled_blocks"] = work.blocks
            if exit_name is not None:
                # Rule X of the pairs the walks left with their rows held, one thread per pair of the class (the
                # list's room), off the walks' warps and stacks.
                self._launches.append(("exit", getattr(module, exit_name), end - begin, begin, slot, solid, work))
                shapes["exit_dim"] = end - begin
            if full_name is not None:
                # A fixed grid over the class's deferred list (the pairs are found on the device, in any order), each
                # block holding every distinct candidate a window of the class can give.
                scratch = terrain_finish_scratch(device, int(self.table.candidates[begin:end].max()))
                self._launches.append(("full", getattr(module, full_name), scratch.blocks, begin, slot, solid, scratch))
                shapes["full_blocks"] = scratch.blocks
                shapes["full_candidates"] = scratch.capacity
                # A class whose largest pair's window can give many candidates takes fewer blocks, its scratch kept
                # near TERRAIN_LAW_FULL_SCRATCH_BYTES: the rare deferred path's parallelism, never a row.
                shapes["full_limited"] = scratch.blocks < TERRAIN_LAW_FULL_BLOCKS
            self._launch_shapes[terrain_law_class_name(partner_class)] = shapes
            slot += walk * (end - begin)
        self.slot_count = slot
        """Walk slots of the side buffer: every listed pair times its class's walk slots."""
        self.features = TerrainFeatures(model) if self._launches else None
        """The heightfields' terrain features (``None`` without a listed pair); report height edits to it and rebuild
        outside capture."""
        self._slot_first = slot_first
        self._listed = np.flatnonzero(slot_first >= 0)
        self.spill_rows = self.default_spill_rows() if spill_rows is None else int(spill_rows)
        """Rows of the spill region (:meth:`default_spill_rows` unless set)."""
        if self.spill_rows < 0:
            raise ValueError(f"The terrain law's spill region needs a non-negative row count, not {self.spill_rows}")
        self._out = terrain_law_rows(
            device,
            self.table.pair_count,
            self.slot_count,
            self.spill_rows,
            [self.table.segment(c, 0)[0] for c in range(TERRAIN_PARTNER_CLASSES)],
            status=status,
            spilled=self._counters[_SPILLED : _SPILLED + 1],
            reserved=self._counters[_RESERVED : _RESERVED + 1],
            deferred_count=self._counters[_DEFERRED : _DEFERRED + TERRAIN_PARTNER_CLASSES],
        )
        self._solids = None
        self._unsupported_pairs = 0
        if solid_pairs:
            partners = np.concatenate([self.table.pairs_host[b : b + c, 1] for b, c in solid_pairs])
            self._solids = solid_module.terrain_solid_data(
                model,
                partners,
                pairs=self.table.pair_count,
                exit_count=self._counters[_EXITS : _EXITS + TERRAIN_PARTNER_CLASSES],
                exit_work=self._counters[_EXIT_WORK : _EXIT_WORK + 2],
            )
            classes = np.concatenate([self.table.partner_class[b : b + c] for b, c in solid_pairs])
            self._unsupported_pairs = int(
                np.count_nonzero(solid_module.terrain_solid_unsupported(classes, model.shape_type.numpy()[partners]))
            )
        self._no_body_flags = wp.empty(0, dtype=wp.int32, device=device)
        self.build_seconds = time.perf_counter() - started
        """Host time of the construction [s]: features, pair table, kernel modules imported and buffers."""

    def default_spill_rows(self) -> int:
        """Return the spill region's rows by default, from the listed pairs' own bounds on their rows
        (:func:`~newton._src.geometry.terrain_contacts.terrain_pair_candidates`, from each pair's shape extent, reach
        and cells): the largest pair's bound, so that every single pair fits, and
        :data:`TERRAIN_LAW_SPILL_SHARE` of the sum of every pair's, the rows the scene's pairs could ask for together
        at most. A pair spills only with more rows than its walk slots; one that finds no room keeps the first of its
        rows by priority in its walk slots and raises ``terrain_rows``, and the spill check prints the rows the launch
        asked for, which ``NEWTON_TERRAIN_LAW_SHADOW_ROWS`` sets."""
        if not len(self._listed):
            return 0
        bounds = self.table.candidates[self._listed]
        return int(bounds.max()) + -(-int(bounds.sum()) // TERRAIN_LAW_SPILL_SHARE)

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
        for kind, kernel, dim, begin, slot, solid, extra in self._launches:
            args = [begin, slot, inputs, data, self._out, extra]
            if solid:
                args.append(self._solids)
            if kind in ("thread", "exit"):
                wp.launch(kernel, dim=dim, inputs=args, block_dim=block, device=self.device, record_tape=False)
            else:
                wp.launch_tiled(kernel, dim=dim, inputs=args, block_dim=block, device=self.device, record_tape=False)
        wp.launch(
            terrain_law_spill_check,
            dim=1,
            inputs=[self._out.spilled, self.spill_rows],
            device=self.device,
            record_tape=False,
        )

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
        first = out.first.numpy()[listed].astype(np.int64) if len(listed) else np.zeros(0, np.int64)
        row_pair = np.repeat(np.arange(len(listed)), count)
        offsets = np.concatenate([[0], np.cumsum(count)])[:-1]
        rows = np.repeat(first, count) + (np.arange(int(count.sum())) - np.repeat(offsets, count))
        q, n, d, feature = (array.numpy()[rows] for array in (out.q, out.n, out.d, out.feature))
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
        """Return the pairs whose rows found no room in the spill region and whose windows outgrew their full-finish
        scratch, summed over every launch so far."""
        if self.device.is_capturing:
            raise RuntimeError("Terrain-law shadow counters must be read outside graph capture")
        rows, candidates = (int(v) for v in self._out.overflow.numpy())
        return rows, candidates

    def exit_work(self) -> tuple[int, int]:
        """Return rule X's exit searches of the last launch and the window vertices they covered (0 without a solid
        class)."""
        if self.device.is_capturing:
            raise RuntimeError("Terrain-law shadow counters must be read outside graph capture")
        searches, vertices = (int(v) for v in self._counters.numpy()[_EXIT_WORK : _EXIT_WORK + 2])
        return searches, vertices

    def rows_spilled(self) -> int:
        """Return the spill rows the last launch asked for (past :attr:`spill_rows`: the rows it lacked room for)."""
        if self.device.is_capturing:
            raise RuntimeError("Terrain-law shadow counters must be read outside graph capture")
        return int(self._out.spilled.numpy()[0])

    def describe(self) -> dict:
        """Return a host-only record of the shadow: classes, pairs per lane class, launches, walk slots and the spill
        region, memory, and
        with a solid class listed the one-thread kernels' lane choice (``thread_vertices``) and the pairs whose partner
        has no core yet (``unsupported_pairs``, no row).

        ``launches`` counts every launch of a collide, the spill check included. ``launch_shapes`` holds, per class
        with pairs, the dimensions fixed at construction: ``block_dim`` (every launch's), ``thread_dim`` (the
        one-thread kernel's threads, one per pair of the class), ``tiled_blocks`` (the tiled kernel's grid),
        ``exit_dim`` (a solid class's exit kernel's threads, one per pair of the class) and ``full_blocks`` (the
        full-finish kernel's grid), each 0 for a kernel the class does not launch, ``full_candidates`` (the distinct
        candidates a full-finish block holds: the most any pair of the class can give), ``full_limited`` (whether the
        full-finish grid has fewer than :data:`~newton._src.geometry.terrain_law.TERRAIN_LAW_FULL_BLOCKS` blocks: its
        blocks, each sized for the class's largest pair, would exceed
        :data:`~newton._src.geometry.terrain_law.TERRAIN_LAW_FULL_SCRATCH_BYTES`) and ``walk_slots`` (a pair's walk
        slots). ``slots`` and ``spill_rows`` are the side buffer's walk slots and spill rows.
        """
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
            for array in (out.q, out.n, out.d, out.feature, out.first, out.count, out.state, out.overflow, out.deferred)
        )
        scratch = sum(
            s.blocks * s.capacity * TERRAIN_FINISH_ROW_BYTES for kind, *_launch, s in self._launches if kind == "full"
        )
        work = sum(w.pairs.capacity for kind, *_launch, w in self._launches if kind == "thread")
        solid = 0
        if self._solids is not None:
            hulls = self._solids.hulls
            solid = sum(
                array.capacity
                for array in (
                    hulls.points,
                    hulls.planes,
                    hulls.face_vertices,
                    hulls.face_begin,
                    hulls.vertex_faces,
                    hulls.vertex_face_begin,
                    hulls.neighbors,
                    hulls.neighbor_begin,
                    hulls.triangles,
                    hulls.hull_points,
                    hulls.hull_faces,
                    hulls.hull_triangles,
                    hulls.shape_hull,
                    self._solids.exits,
                )
            )
        solids = {}
        if self._solids is not None:
            solids = {
                "thread_vertices": int(self._solids.thread_vertices),
                "unsupported_pairs": self._unsupported_pairs,
            }
        return {
            "classes": [terrain_law_class_name(c) for c in self.classes],
            "pairs": pairs,
            "launches": len(self._launches) + 1,
            **solids,
            "launch_shapes": {name: dict(shapes) for name, shapes in self._launch_shapes.items()},
            "slots": self.slot_count,
            "spill_rows": self.spill_rows,
            "bytes": {
                "side_buffer": int(side + self._counters.capacity),
                "work_lists": int(work),
                "finish_scratch": int(scratch),
                "solid_data": int(solid),
                "pair_table": int(self.table.memory_bytes()["pairs"]),
                "features": 0 if self.features is None else int(sum(self.features.memory_bytes().values())),
            },
            "build_seconds": round(self.build_seconds, 3),
        }
