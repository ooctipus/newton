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
``NEWTON_TERRAIN_LAW_SHADOW_ROWS`` sets it. A listed solid class also has its jobs
(:class:`~newton._src.geometry.terrain_law_solid.TerrainSolidJobs`): job lists and a candidate pool sized at the
scene level from its pairs' own bounds (:func:`~newton._src.geometry.terrain_law_solid.terrain_solid_job_capacities`),
which ``NEWTON_TERRAIN_LAW_SHADOW_JOBS`` scales. Every collide then launches, after today's terrain route, each listed
class's kernels (:data:`TERRAIN_LAW_KERNELS`) and a check of the spill region and the job lists, with dimensions fixed
at construction (:meth:`TerrainLawShadow.describe` records them): no host read, no allocation, no scan, so it runs
inside CUDA graph capture. Today's route still produces every contact; the shadow's rows are read only by
:meth:`TerrainLawShadow.rows`.

The per-type kernels take ``(begin, slot_begin, inputs, data, out)``
(:class:`~newton._src.geometry.terrain_law.TerrainPairInputs`,
:class:`~newton._src.geometry.terrain_law.TerrainFeatureData`,
:class:`~newton._src.geometry.terrain_law.TerrainLawRows`) and their launch arguments: table pair ``begin + i`` writes
its rows in its walk slots ``slot_begin + i * K``, ``K`` its class's walk slots, or when they are more, to rows of the
spill region; its first row, row count and ``TERRAIN_PAIR_*`` state. Every launch takes blocks of
:func:`~newton._src.geometry.terrain_law.terrain_law_block_dim` threads.

The point class (spheres and capsules) walks a pair's window in one thread:

1. The one-thread kernel, one thread per pair of the class (both lane classes: the table keeps them contiguous), with
   the class's :class:`~newton._src.geometry.terrain_law.TerrainPairWork`. It walks the pairs whose windows hold at
   most :data:`~newton._src.geometry.terrain_contacts.TERRAIN_LANE_CELLS` cells over every orientation; a pair with a
   larger window it takes through the whole-pair rejects (stale features, the pair policy, a window off the
   heightfield, the tile-max exit), and when it reaches its window queues it in the work list.
2. The tiled kernel, ``wp.launch_tiled`` with :data:`~newton._src.geometry.terrain_law.TERRAIN_LAW_BLOCK` lanes per
   pair and the work list's fixed grid (:attr:`~newton._src.geometry.terrain_law.TerrainPairWork.blocks` blocks),
   which strides over the queued pairs.
3. The full-finish kernel (below).

A solid class (boxes; cylinders and cones; convex hulls and ellipsoids) takes the shape of today's terrain route: a
cheap scan that lists the surviving features, one thread per listed feature, and a per-pair close that holds their rows
in window order (:data:`~newton._src.geometry.terrain_law_solid.TERRAIN_SOLID_LAUNCHES`, each kernel with the class's
jobs and the solid classes' :class:`~newton._src.geometry.terrain_law_solid.TerrainSolidData`):

1. The whole-pair rejects, one thread per pair of the class (stale features, the pair policy, a partner without a core,
   a window off the heightfield, the tile-max exit): a pair past them is listed live, its window in runs of
   :data:`~newton._src.geometry.terrain_law_solid.TERRAIN_SOLID_SCAN_VERTICES` window vertices.
2. The scan, a fixed grid of the class's job threads striding over the runs: a run's features through their cheapest
   rejects, the survivors listed (rule F's faces, one per cluster plane and run; rule E's edges; rule V's vertices).
3. The job kernels, the job threads striding over a list: faces, edges (whose creases with an arc that can hold a row
   are listed for ``theta*``), the creases' ``theta*``, their rows, vertices. A job's rows go to the class's candidate
   pool with their place in window order.
4. The close, the job threads over the live pairs: a pair's candidates held in window order (the walk's held rows),
   then rule R's kernel over the pairs without a row, and the finish, or for a core reaching below its window's
   highest vertex rule X's items.
5. Rule X's items (a core point, or a window vertex that can give a candidate), valleys and the listed candidates'
   near-bound descents, the job threads again: they list the candidates whose bounds pass the pair's target; then the
   exit kernel, one thread per pair of the class (its exit list's room), runs rule X's passes over the listed
   candidates (exact: any other candidate is a no-op in both passes), and the settle kernel finishes the pair.
6. The full-finish kernel (below).

The full-finish kernel of a class, ``wp.launch_tiled`` on the fixed grid of its
:class:`~newton._src.geometry.terrain_law.TerrainFinishScratch`
(:func:`~newton._src.geometry.terrain_law.terrain_finish_blocks` blocks, each holding every distinct candidate a window
of the class can give), strides over the class's deferred list: pairs with more distinct rows than their walk slots,
and solid pairs whose work found no room in a job list or the candidate pool (the job check prints the entries asked
for against the room).

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


def _solid_launches(name: str) -> tuple[tuple[str, str], ...]:
    """A solid class's launches, ``(kind, kernel)`` in order: those of
    :data:`~newton._src.geometry.terrain_law_solid.TERRAIN_SOLID_LAUNCHES`, named here so that the solid modules are
    imported only when a solid class is listed."""
    kernels = (
        "rows",
        "scan",
        "faces",
        "edges",
        "thetas",
        "creases",
        "vertices",
        "close",
        "recovery",
        "xitems",
        "xvalleys",
        "xdescents",
        "exit",
        "settle",
        "full",
    )
    kinds = ("pairs", *("jobs",) * 11, "pairs", "pairs", "full")
    return tuple(
        (kind, f"terrain_{name}_rows" if kernel == "rows" else f"terrain_{name}_rows_{kernel}")
        for kind, kernel in zip(kinds, kernels, strict=True)
    )


TERRAIN_LAW_KERNELS: dict[int, tuple[str, tuple[tuple[str, str], ...]]] = {
    TERRAIN_PARTNER_POINT: (
        "newton._src.geometry.terrain_law_point",
        (("thread", "terrain_point_rows"), ("tiled", "terrain_point_rows_tiled"), ("full", "terrain_point_rows_full")),
    ),
    TERRAIN_PARTNER_BOX: ("newton._src.geometry.terrain_law_box", _solid_launches("box")),
    TERRAIN_PARTNER_CYLINDER: ("newton._src.geometry.terrain_law_cylinder", _solid_launches("cylinder")),
    TERRAIN_PARTNER_HULL: ("newton._src.geometry.terrain_law_hull", _solid_launches("hull")),
}
"""Per partner class: the module and its launches in order, ``(kind, kernel)``.

The module is imported only when the class is listed. Launch kinds, every one with dimensions fixed at construction:

* ``thread``: one thread per pair of the class, with the class's
  :class:`~newton._src.geometry.terrain_law.TerrainPairWork` as sixth argument (the point class's walk);
* ``tiled``: ``wp.launch_tiled`` with the work list's
  :attr:`~newton._src.geometry.terrain_law.TerrainPairWork.blocks` blocks, ``begin`` and ``slot_begin`` the tiled
  segment's (the point class's large windows);
* ``pairs``: one thread per pair of the class, with the class's
  :class:`~newton._src.geometry.terrain_law_solid.TerrainSolidJobs` and the solid classes'
  :class:`~newton._src.geometry.terrain_law_solid.TerrainSolidData` (a solid class's whole-pair rejects, exit and
  settle kernels);
* ``jobs``: the class's job threads (:attr:`~newton._src.geometry.terrain_law_solid.TerrainSolidJobs.threads`), the same
  arguments (a solid class's scan, job, close and rule R kernels, each striding over its list);
* ``full``: ``wp.launch_tiled`` with :func:`~newton._src.geometry.terrain_law.terrain_finish_blocks` blocks of
  :data:`~newton._src.geometry.terrain_law.TERRAIN_LAW_BLOCK` lanes whatever the class's size, striding over its
  deferred list, and a :class:`~newton._src.geometry.terrain_law.TerrainFinishScratch` as sixth argument (a solid
  class's also the solid classes' data and its jobs), so that an empty list costs one near-empty launch.

``begin`` and ``slot_begin`` are the class's first pair and walk slot but for the ``tiled`` launch.

A class's module and kernels follow one protocol (the helpers live in :mod:`~newton._src.geometry.terrain_law`):

* The module compiles without floating-point contraction (``wp.set_module_options({"fuse_fp": False})``), so that the
  CPU and CUDA round every float32 operation alike and decide every predicate boundary on the same bits: a CPU test or
  census then counts what the GPU does.
* A pair that has no row (stale features, the pair policy, a window off the heightfield, the tile-max exit) is
  recorded by ``terrain_pair_exit``.
* While its window is walked, a pair may use its ``K`` walk slots as scratch, in the window frame, its distinct rows
  merged as the finish merges (``terrain_hold_row``). The point kernels hold up to ``K - 2`` there and keep the best
  exit of each core end inside the solid in the last two slots (with both ends outside, the last slot holds the
  capsule axis's deepest crossing for its G3); an exit is read before a row may land in its slot. A solid pair's close
  holds up to ``K`` there, its job rows taken in window order.
* When the finish keeps the held rows as they are (no plane group of three, ``terrain_plane_group_of_three``; at most
  ``K`` rows), the point kernels write them in the world frame (``terrain_store_row``) with the pair's count and state;
  the solid kernels run the finish in place on any held rows (``terrain_finish_slots``) and write the kept ones. A pair
  beyond that (for the solid kernels: more distinct rows than ``K``, or work without room) is deferred
  (``terrain_defer_pair``), and the class's full-finish kernel walks it again into its distinct candidates
  (``terrain_hold_candidate``), runs the finish on all of them (``terrain_finish_list``) and writes every row it keeps,
  in the pair's walk slots when they fit, else in rows of the spill region it reserves (``terrain_spill_rows``; no
  room: the first ``K`` of them by priority in the walk slots, ``terrain_finish_overflow``, state
  ``TERRAIN_PAIR_ROW_OVERFLOW``, ``terrain_rows``).
"""

TERRAIN_LAW_SOLID_CLASSES = (TERRAIN_PARTNER_BOX, TERRAIN_PARTNER_CYLINDER, TERRAIN_PARTNER_HULL)
"""Partner classes whose kernels take the solid classes' data
(:class:`~newton._src.geometry.terrain_law_solid.TerrainSolidData`: hull tables and rule X's exit lists, built once
when one of them is listed) and their class's jobs."""

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

TERRAIN_LAW_JOB_COUNTERS = 11
"""Counters of a solid class's jobs: its six job lists, its candidate pool, its live pairs, its pairs left to rule R,
its scan's runs and its listed rule X candidates' descents
(:data:`~newton._src.geometry.terrain_law_solid.TERRAIN_SOLID_COUNTERS`)."""

TERRAIN_LAW_JOB_NAMES = (
    "faces",
    "edges",
    "thetas",
    "vertices",
    "xitems",
    "xvalleys",
    "candidates",
    "live",
    "recovery",
    "scans",
    "xdescents",
)
"""Names of a solid class's job counters, in order (:meth:`TerrainLawShadow.job_counts`)."""

_JOB_COUNTERS = wp.constant(TERRAIN_LAW_JOB_COUNTERS)

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
_JOBS = _EXIT_WORK + 2
"""Offset of the per-class job counters (:data:`TERRAIN_LAW_JOB_COUNTERS` per class)."""
_COUNTERS = _JOBS + TERRAIN_PARTNER_CLASSES * TERRAIN_LAW_JOB_COUNTERS
"""The shadow's counters, all zeroed by one memset before every launch."""


@wp.kernel(enable_backward=False)
def terrain_law_spill_check(
    counters: wp.array[wp.int32], spilled: int, spill: int, jobs: int, capacities: wp.array[wp.int32]
):
    """After a launch: warn when the spill region overflowed, with the spill rows asked for against its rows (the pairs
    that found no room have raised ``terrain_rows``), and when a solid class's job list or candidate pool overflowed,
    with the entries asked for against its room (the pairs that found none took the full-finish kernel)."""
    if counters[spilled] > spill:
        wp.printf(
            "Warning: Terrain-law spill region overflowed %d > %d. Set NEWTON_TERRAIN_LAW_SHADOW_ROWS to at least the "
            "first number.\n",
            counters[spilled],
            spill,
        )
    for k in range(capacities.shape[0]):
        if capacities[k] > 0 and counters[jobs + k] > capacities[k]:
            wp.printf(
                "Warning: Terrain-law job list %d of partner class %d overflowed %d > %d: those pairs took the "
                "full-finish kernel. Raise NEWTON_TERRAIN_LAW_SHADOW_JOBS (a factor on every job list).\n",
                k % _JOB_COUNTERS,
                k // _JOB_COUNTERS,
                counters[jobs + k],
                capacities[k],
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


def parse_terrain_law_shadow_jobs(value: str | None) -> float | None:
    """Return the factor on the solid classes' job capacities a ``NEWTON_TERRAIN_LAW_SHADOW_JOBS`` value sets, or
    ``None`` for the default (1).

    Args:
        value: A positive number; ``None`` or blank for the default
            (:func:`~newton._src.geometry.terrain_law_solid.terrain_solid_job_capacities`).

    Raises:
        ValueError: On a value that is not a positive finite number.
    """
    if value is None or not value.strip():
        return None
    try:
        scale = float(value.strip())
    except ValueError:
        scale = -1.0
    if not (np.isfinite(scale) and scale > 0.0):
        raise ValueError(f"NEWTON_TERRAIN_LAW_SHADOW_JOBS must be a positive factor, not {value!r}")
    return scale


class TerrainLawShadow:
    """The terrain law's kernels for some partner classes, run after the narrow phase into a side buffer.

    Built once, outside graph capture: the terrain features, the pair table, the side buffer (per listed pair ``K``
    walk slots and one spill region of terrain points, normals, core distances and feature ids, and per listed pair its
    first row, row count and state), per class its launch arguments (the point class's
    :class:`~newton._src.geometry.terrain_law.TerrainPairWork`, a solid class's
    :class:`~newton._src.geometry.terrain_law_solid.TerrainSolidJobs`, every class's
    :class:`~newton._src.geometry.terrain_law.TerrainFinishScratch`, sized from the class's pairs' windows) and with a
    solid class listed the solid classes' data (:class:`~newton._src.geometry.terrain_law_solid.TerrainSolidData`).
    :meth:`launch` then runs each class's kernels with dimensions fixed here; the narrow phase's contacts are untouched.

    Args:
        model: A finalized model.
        classes: Partner classes to run (:func:`parse_terrain_law_shadow`); each must have a kernel.
        pairs: The collision pipeline's explicit shape pairs, or ``None`` for the model's filter.
        status: The narrow phase's sticky buffer status word, or ``None`` for one of the shadow's own.
        spill_rows: Rows of the spill region, or ``None`` for :meth:`default_spill_rows` (from the listed pairs'
            bounds on their rows).
        job_scale: Factor on every solid class's job capacities, or ``None`` for 1
            (:func:`~newton._src.geometry.terrain_law_solid.terrain_solid_job_capacities`).
    """

    def __init__(
        self,
        model,
        classes,
        *,
        pairs=None,
        status: wp.array | None = None,
        spill_rows: int | None = None,
        job_scale: float | None = None,
    ):
        started = time.perf_counter()
        self.device = model.device
        self.classes = tuple(sorted(int(c) for c in classes))
        missing = [c for c in self.classes if c not in TERRAIN_LAW_KERNELS]
        if missing:
            raise ValueError(f"The terrain law has no kernel for partner classes {missing}")
        self.table = TerrainPairTable(model, pairs=pairs)
        """The static (heightfield, partner) pair table."""
        self.job_scale = 1.0 if job_scale is None else float(job_scale)
        """Factor on every solid class's job capacities."""
        self._launches = []
        self._launch_shapes = {}
        block = terrain_law_block_dim(model.device)
        slot_first = np.full(self.table.pair_count, -1, np.int64)
        slot = 0
        device = self.device
        # Per class its deferred pairs, per class its queued pairs, the spill rows asked for and reserved, per class its
        # pairs left to rule X, rule X's exit searches and their window vertices, and per class its jobs: zeroed by one
        # memset before every launch.
        self._counters = wp.zeros(_COUNTERS, dtype=wp.int32, device=device)
        capacities = np.zeros(TERRAIN_PARTNER_CLASSES * TERRAIN_LAW_JOB_COUNTERS, np.int32)
        solid_pairs = []
        solid_module = None
        if any(c in TERRAIN_LAW_SOLID_CLASSES for c in self.classes):
            solid_module = importlib.import_module("newton._src.geometry.terrain_law_solid")
            assert solid_module.TERRAIN_SOLID_COUNTERS == TERRAIN_LAW_JOB_COUNTERS
            assert (
                TERRAIN_LAW_JOB_NAMES[: len(solid_module.TERRAIN_SOLID_JOB_LISTS)]
                == solid_module.TERRAIN_SOLID_JOB_LISTS
            )
        self._jobs = {}
        for partner_class in self.classes:
            module_name, launches = TERRAIN_LAW_KERNELS[partner_class]
            module = importlib.import_module(module_name)
            walk = int(TERRAIN_PARTNER_WALK_SLOTS[partner_class])
            # The class's lane segments are contiguous in the table (lane classes by id); its walk slots follow them.
            begin = self.table.segment(partner_class, TERRAIN_LANE_THREAD)[0]
            tiled_begin, end = self.table.segment(partner_class, TERRAIN_LANE_TILE)
            if end == begin:
                continue
            slot_first[begin:end] = slot + walk * np.arange(end - begin)
            solid = partner_class in TERRAIN_LAW_SOLID_CLASSES
            shapes = {"block_dim": block, "walk_slots": walk}
            scratch = terrain_finish_scratch(device, int(self.table.candidates[begin:end].max()))
            if solid:
                solid_pairs.append((begin, end - begin))
                partners = self.table.pairs_host[begin:end, 1]
                points, takes = solid_module.terrain_solid_exit_items(model, partners)
                capacity = solid_module.terrain_solid_job_capacities(
                    self.table.vertices[begin:end],
                    points,
                    takes,
                    self.table.candidates[begin:end],
                    walk,
                    scale=self.job_scale,
                )
                offset = _JOBS + TERRAIN_LAW_JOB_COUNTERS * partner_class
                jobs = solid_module.terrain_solid_jobs(
                    device,
                    partner_class,
                    begin,
                    end - begin,
                    capacity,
                    counts=self._counters[offset : offset + TERRAIN_LAW_JOB_COUNTERS],
                )
                self._jobs[partner_class] = jobs
                names = (*solid_module.TERRAIN_SOLID_JOB_LISTS, "candidates", "scans", "xdescents")
                # the live and rule R lists hold every pair of the class: no check
                for name in names:
                    capacities[offset - _JOBS + TERRAIN_LAW_JOB_NAMES.index(name)] = capacity[name]
                for kind, kernel_name in launches:
                    kernel = getattr(module, kernel_name)
                    if kind == "pairs":
                        self._launches.append(("thread", kernel, end - begin, begin, slot, (jobs, True)))
                    elif kind == "jobs":
                        self._launches.append(("thread", kernel, jobs.threads, begin, slot, (jobs, True)))
                    else:
                        self._launches.append(("tiled", kernel, scratch.blocks, begin, slot, (scratch, True, jobs)))
                shapes.update(
                    {
                        "pair_dim": end - begin,
                        "job_threads": jobs.threads,
                        "jobs": {name: int(capacity[name]) for name in names},
                    }
                )
            else:
                counter = _QUEUED + partner_class
                work = terrain_pair_work(
                    device, tiled_begin, end - tiled_begin, count=self._counters[counter : counter + 1]
                )
                shapes.update({"thread_dim": end - begin, "tiled_blocks": 0})
                for kind, kernel_name in launches:
                    kernel = getattr(module, kernel_name)
                    if kind == "thread":
                        # One thread per pair of the class: it walks the small windows and queues the larger ones it
                        # does not reject whole, which the tiled kernel's fixed grid then takes in turn.
                        self._launches.append(("thread", kernel, end - begin, begin, slot, (work, False)))
                    elif kind == "tiled":
                        if end > tiled_begin:
                            tiled_slot = slot + walk * (tiled_begin - begin)
                            self._launches.append(
                                ("tiled", kernel, work.blocks, tiled_begin, tiled_slot, (work, False))
                            )
                            shapes["tiled_blocks"] = work.blocks
                    else:
                        self._launches.append(("tiled", kernel, scratch.blocks, begin, slot, (scratch, False)))
            # A fixed grid over the class's deferred list (the pairs are found on the device, in any order), each block
            # holding every distinct candidate a window of the class can give. A class whose largest pair's window can
            # give many candidates takes fewer blocks, its scratch kept near TERRAIN_LAW_FULL_SCRATCH_BYTES: the rare
            # deferred path's parallelism, never a row.
            shapes["full_blocks"] = scratch.blocks
            shapes["full_candidates"] = scratch.capacity
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
        self._capacities = wp.array(capacities, dtype=wp.int32, device=device)
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
        for kind, kernel, dim, begin, slot, extra in self._launches:
            args = [begin, slot, inputs, data, self._out, extra[0]]
            if extra[1]:
                args.append(self._solids)
                args.extend(extra[2:])
            if kind == "thread":
                wp.launch(kernel, dim=dim, inputs=args, block_dim=block, device=self.device, record_tape=False)
            else:
                wp.launch_tiled(kernel, dim=dim, inputs=args, block_dim=block, device=self.device, record_tape=False)
        wp.launch(
            terrain_law_spill_check,
            dim=1,
            inputs=[self._counters, _SPILLED, self.spill_rows, _JOBS, self._capacities],
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

    def job_counts(self) -> dict[str, dict[str, int]]:
        """Return, per listed solid class with pairs, the entries its last launch asked of every job list, of its
        candidate pool and of its scan's runs (past a capacity, :meth:`describe`'s ``launch_shapes``: entries without
        room, whose pairs took the full-finish kernel), its live pairs and its pairs left to rule R, by
        :data:`TERRAIN_LAW_JOB_NAMES`."""
        if self.device.is_capturing:
            raise RuntimeError("Terrain-law shadow counters must be read outside graph capture")
        if not self._jobs:
            return {}
        counters = self._counters.numpy()
        out = {}
        for partner_class in self._jobs:
            offset = _JOBS + TERRAIN_LAW_JOB_COUNTERS * partner_class
            out[terrain_law_class_name(partner_class)] = {
                name: int(counters[offset + k]) for k, name in enumerate(TERRAIN_LAW_JOB_NAMES)
            }
        return out

    def rows_spilled(self) -> int:
        """Return the spill rows the last launch asked for (past :attr:`spill_rows`: the rows it lacked room for)."""
        if self.device.is_capturing:
            raise RuntimeError("Terrain-law shadow counters must be read outside graph capture")
        return int(self._out.spilled.numpy()[0])

    def describe(self) -> dict:
        """Return a host-only record of the shadow: classes, pairs per lane class, launches, walk slots and the spill
        region, memory, the solid classes' job scale and with a solid class listed the pairs whose partner has no core
        yet (``unsupported_pairs``, no row).

        ``launches`` counts every launch of a collide, the spill and job check included. ``launch_shapes`` holds, per
        class with pairs, the dimensions fixed at construction: ``block_dim`` (every launch's), ``walk_slots`` (a
        pair's walk slots), ``full_blocks`` (the full-finish kernel's grid), ``full_candidates`` (the distinct
        candidates a full-finish block holds: the most any pair of the class can give) and ``full_limited`` (whether
        the full-finish grid has fewer than :data:`~newton._src.geometry.terrain_law.TERRAIN_LAW_FULL_BLOCKS` blocks:
        its blocks, each sized for the class's largest pair, would exceed
        :data:`~newton._src.geometry.terrain_law.TERRAIN_LAW_FULL_SCRATCH_BYTES`); for the point class ``thread_dim``
        (the one-thread kernel's threads, one per pair of the class) and ``tiled_blocks`` (the tiled kernel's grid, 0
        without a tiled segment); for a solid class ``pair_dim`` (its whole-pair rejects', exit and settle kernels'
        threads, one per pair of the class), ``job_threads`` (its scan, job, close and rule R kernels' threads) and
        ``jobs`` (the entries of its job lists, its candidate pool and its scan's runs). ``slots`` and ``spill_rows``
        are the side buffer's walk slots and spill rows.
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
            extra[0].blocks * extra[0].capacity * TERRAIN_FINISH_ROW_BYTES
            for kind, kernel, *_launch, extra in self._launches
            if kernel.key.endswith("_full")
        )
        work = sum(
            extra[0].pairs.capacity for kind, *_launch, extra in self._launches if kind == "thread" and not extra[1]
        )
        jobs = 0
        if self._jobs:
            from .terrain_law_solid import terrain_solid_jobs_bytes  # noqa: PLC0415

            jobs = sum(terrain_solid_jobs_bytes(j) for j in self._jobs.values())
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
            solids = {"job_scale": self.job_scale, "unsupported_pairs": self._unsupported_pairs}
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
                "jobs": int(jobs),
                "finish_scratch": int(scratch),
                "solid_data": int(solid),
                "pair_table": int(self.table.memory_bytes()["pairs"]),
                "features": 0 if self.features is None else int(sum(self.features.memory_bytes().values())),
            },
            "build_seconds": round(self.build_seconds, 3),
        }
