# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run the terrain law after the narrow phase: as the source of its partners' heightfield contacts, or in shadow.

The partner classes are ``point`` (spheres and capsules), ``box``, ``cylinder`` (cylinders and cones) and ``hull``
(convex hulls and ellipsoids). By default (``NEWTON_TERRAIN_LAW`` unset) a collision pipeline with heightfield pairs
builds :class:`TerrainLawShadow` for every class with a kernel, writes the rows of its pairs as their contacts
(:meth:`TerrainLawShadow.write_contacts`, :func:`terrain_law_contact_kernel`), and the narrow phase drops those pairs
(:meth:`TerrainLawShadow.written_pairs`); a configuration the law does not support yet keeps the terrain route instead,
and the pipeline's route record says why (``get_collision_selection()["heightfield"]["law"]``). ``NEWTON_TERRAIN_LAW``
lists classes, or ``all``, to request the law strictly (an unsupported configuration raises), or ``none`` to keep the
terrain route, a temporary opt-out until that route is removed. ``NEWTON_TERRAIN_LAW_SHADOW`` lists classes whose
kernels run instead beside the terrain route, which writes every contact, into a side buffer that nothing in the step
reads (shadow mode, for comparisons).

A collision pipeline builds :class:`TerrainLawShadow` once, at construction: the terrain features
(:class:`~newton._src.geometry.terrain_contacts.TerrainFeatures`), the static pair table
(:class:`~newton._src.geometry.terrain_contacts.TerrainPairTable`, from the pipeline's pairs), per listed pair a static
range of walk slots (:data:`~newton._src.geometry.terrain_contacts.TERRAIN_PARTNER_WALK_SLOTS`, the fast path's
layout, which bounds no row) and one spill region for all the rows of pairs that have more
(:class:`~newton._src.geometry.terrain_law.TerrainLawRows`), sized at the scene level, as ``rigid_contact_max`` sizes
contacts: by default from the listed pairs' own bounds on their rows (:meth:`TerrainLawShadow.default_spill_rows`);
``NEWTON_TERRAIN_LAW_SHADOW_ROWS`` sets it. A listed solid class also has its jobs
(:class:`~newton._src.geometry.terrain_law_solid.TerrainSolidJobs`): job lists and a candidate pool sized at the
scene level from its pairs' own bounds (:func:`~newton._src.geometry.terrain_law_solid.terrain_solid_job_capacities`),
which ``NEWTON_TERRAIN_LAW_SHADOW_JOBS`` scales. Every collide then launches, after the narrow phase, each listed
class's kernels (:data:`TERRAIN_LAW_KERNELS`) and a check of the spill region and the job lists, with dimensions fixed
at construction (:meth:`TerrainLawShadow.describe` records them): no host read, no allocation, no scan, so it runs
inside CUDA graph capture. When the law writes, its contact kernel follows; in shadow, the terrain route produces every
contact and the rows are read only by :meth:`TerrainLawShadow.rows`.

The terrain features are built from the heights at construction. Heights edited in place afterwards must be reported
to them (:meth:`~newton._src.geometry.terrain_contacts.TerrainFeatures.mark_elevations_changed`, which makes the pairs
over them give no row and raise ``terrain_stale``, then
:meth:`~newton._src.geometry.terrain_contacts.TerrainFeatures.rebuild_terrain_features`), or the pipeline rebuilt.

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
3. The class's finish of its deferred pairs (below), the tiled kernel's walk into the scratch.

A solid class (boxes; cylinders and cones; convex hulls and ellipsoids) takes the shape of today's terrain route: a
cheap scan that lists the surviving features, one thread per listed feature, and a per-pair close that holds their rows
in window order (:data:`~newton._src.geometry.terrain_law_solid.TERRAIN_SOLID_LAUNCHES`, each kernel with the class's
jobs and the solid classes' :class:`~newton._src.geometry.terrain_law_solid.TerrainSolidData`):

1. The whole-pair rejects, one thread per pair of the class (stale features, the pair policy, a partner without a core,
   a window off the heightfield, the tile-max exit): a pair past them is listed live, its window in runs of its class's
   :data:`~newton._src.geometry.terrain_contacts.TERRAIN_PARTNER_SCAN_RUN` window vertices.
2. The scan, a fixed grid of the class's job threads striding over the runs: a run's features through their cheapest
   rejects, the survivors listed (rule F's faces, one per cluster plane and run; rule E's edges; rule V's vertices).
3. The job kernels, the job threads striding over a list: faces, edges (whose creases with an arc that can hold a row
   are listed for ``theta*``), the creases' ``theta*``, their rows, vertices. A job's rows go to the class's candidate
   pool with their place in window order.
4. The close, the job threads over the live pairs: a pair's candidates held in window order (the rows a walk of its
   window holds), then rule R's kernel over the pairs without a row; a pair's held rows are written as they are when
   the finish keeps them so (no plane group of three), else held for the small-pair finish, one thread per pair.
5. The class's finish of its deferred pairs (below), from their candidates in the pool.

Work past a solid class's job lists or candidate pool (scene-level buffers) takes their reserve, the same kernels
running it, with its pair's rows unchanged: the narrow phase's sticky ``terrain_jobs`` bit is raised, and the job
check prints the entries asked for against the room, by list and class; a pair some of whose work found no room even
there keeps no row.

The finish of a class's deferred pairs (pairs whose rows need more than their walk slots, or for the point class the
general finish), ``wp.launch_tiled`` on the fixed grid of its
:class:`~newton._src.geometry.terrain_law.TerrainFinishScratch`
(:func:`~newton._src.geometry.terrain_law.terrain_finish_blocks` blocks, each holding every candidate a window of the
class can give), strides over the class's deferred list, a block of lanes per pair: it collects the pair's candidates
in the block's scratch (a solid pair's from the pool, a point pair's by the tiled walk), the lanes run the finish on
them, every step split among them, and write every row it keeps, in the pair's walk slots when they fit, else in rows
of the spill region.

The kernels apply the pipeline's pair policy: the table holds the pairs its collision filter admits (worlds,
collision groups, filter pairs, or its explicit pair list), and each kernel drops the pairs its broad phase's
immovable-pair test drops (``include_immobile_pairs``, ``include_static_kinematic_pairs``). Features older than
the heights (an edit reported with ``mark_elevations_changed`` and not yet rebuilt) give no row and raise the
narrow phase's sticky ``terrain_stale`` bit; a pair whose rows the spill region has no room for keeps the first of
them by priority in its walk slots (each plane group's deepest first) and raises ``terrain_rows`` (the spill check
prints the rows asked for against the region's), as does a pair whose window outgrew the bound its shape data gave at
construction; a solid pair whose work found no room in its class's job lists raises ``terrain_jobs`` (its rows are
unchanged while the lists' reserve takes the work; past it, it keeps no row); ``NarrowPhase.check_buffer_capacity``
raises on each.

A solid class's kernels pass 4.2 to 4.7 KB of launch arguments (its jobs and the solid classes' data by value): CUDA
launches them on Volta or newer GPUs with a CUDA 12.1 or newer driver (older drivers take at most 4 KB).
"""

from __future__ import annotations

import importlib
import time
from typing import Any

import numpy as np
import warp as wp

from .contact_data import ContactData
from .terrain_contacts import (
    TERRAIN_LANE_THREAD,
    TERRAIN_LANE_TILE,
    TERRAIN_PARTNER_BOX,
    TERRAIN_PARTNER_CLASSES,
    TERRAIN_PARTNER_CYLINDER,
    TERRAIN_PARTNER_HULL,
    TERRAIN_PARTNER_POINT,
    TERRAIN_PARTNER_SCAN_RUN,
    TERRAIN_PARTNER_WALK_SLOTS,
    TerrainFeatures,
    TerrainPairTable,
)
from .terrain_law import (
    TERRAIN_FINISH_ROW_BYTES,
    TERRAIN_LAW_DEFERRED_BLOCKS,
    TERRAIN_PAIR_CANDIDATE_OVERFLOW,
    TERRAIN_PAIR_FILTERED,
    TERRAIN_PAIR_INCOMPLETE,
    TERRAIN_PAIR_OUTSIDE,
    TERRAIN_PAIR_ROW_OVERFLOW,
    TERRAIN_PAIR_STALE,
    TERRAIN_PAIR_TILE_EXIT,
    TERRAIN_PAIR_WIDE,
    TerrainLawRows,
    TerrainPairInputs,
    terrain_finish_scratch,
    terrain_law_block_dim,
    terrain_law_rows,
    terrain_pair_work,
)
from .types import GeoType

TERRAIN_LAW_CLASS_NAMES = {
    "point": TERRAIN_PARTNER_POINT,
    "box": TERRAIN_PARTNER_BOX,
    "cylinder": TERRAIN_PARTNER_CYLINDER,
    "hull": TERRAIN_PARTNER_HULL,
}
"""``NEWTON_TERRAIN_LAW`` and ``NEWTON_TERRAIN_LAW_SHADOW`` names of the partner classes."""


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
        "border",
        "close",
        "recovery",
        "small",
        "finish",
    )
    kinds = tuple(
        "pairs"
        if kernel == "rows"
        else "deferred"
        if kernel == "finish"
        else "pair_tiles"
        if kernel == "recovery"
        else "pair_jobs"
        if kernel in ("close", "small")
        else "border_jobs"
        if kernel == "border"
        else "jobs"
        for kernel in kernels
    )
    return tuple(
        (kind, f"terrain_{name}_rows" if kernel == "rows" else f"terrain_{name}_rows_{kernel}")
        for kind, kernel in zip(kinds, kernels, strict=True)
    )


TERRAIN_LAW_KERNELS: dict[int, tuple[str, tuple[tuple[str, str], ...]]] = {
    TERRAIN_PARTNER_POINT: (
        "newton._src.geometry.terrain_law_point",
        (
            ("thread", "terrain_point_rows"),
            ("tiled", "terrain_point_rows_tiled"),
            ("deferred", "terrain_point_rows_finish"),
        ),
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
  :class:`~newton._src.geometry.terrain_law_solid.TerrainSolidData` (a solid class's whole-pair rejects);
* ``jobs``: the class's job threads (:attr:`~newton._src.geometry.terrain_law_solid.TerrainSolidJobs.threads`), the same
  arguments (a solid class's scan and job kernels, each striding over its list);
* ``border_jobs``: the same with the border kernel's threads
  (:attr:`~newton._src.geometry.terrain_law_solid.TerrainSolidJobs.border_threads`, a smaller grid: the border's jobs
  are rare);
* ``pair_jobs``: the same over a list of the class's pairs, with at most one thread per pair
  (:attr:`~newton._src.geometry.terrain_law_solid.TerrainSolidJobs.pair_threads`: a solid class's close and the
  small-pair finish);
* ``pair_tiles``: ``wp.launch_tiled`` with
  :attr:`~newton._src.geometry.terrain_law_solid.TerrainSolidJobs.finish_blocks` blocks of
  :data:`~newton._src.geometry.terrain_law.TERRAIN_LAW_BLOCK` lanes, a pair per block at a time over a list of the
  class's pairs, the same arguments (a solid class's rule R kernel);
* ``deferred``: ``wp.launch_tiled`` with :func:`~newton._src.geometry.terrain_law.terrain_finish_blocks` blocks of
  :data:`~newton._src.geometry.terrain_law.TERRAIN_LAW_BLOCK` lanes whatever the class's size, striding over its
  deferred list, and a :class:`~newton._src.geometry.terrain_law.TerrainFinishScratch` as sixth argument (a solid
  class's also the solid classes' data and its jobs), so that an empty list costs one near-empty launch: the class's
  finish of its deferred pairs.

``begin`` and ``slot_begin`` are the class's first pair and walk slot but for the ``tiled`` launch. The cylinder
class's kernels compile its barrel cylinders' paths out; a class whose table holds a barrel cylinder takes
:data:`TERRAIN_LAW_BARREL_KERNELS` instead.

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
  beyond that (for the solid kernels: more distinct rows than ``K``) is deferred (``terrain_defer_pair``), and the
  class's finish of its deferred pairs places its candidates in its scratch, unmerged (a solid pair's from the pool,
  a point pair's by its walk, ``terrain_place_candidate``), runs the finish on all of them with a block's lanes, every
  step split among them (``terrain_finish_list``), and writes every row it keeps, in the pair's walk slots when they
  fit, else in rows of the spill region it reserves (``terrain_finish_deferred``, ``terrain_spill_rows``; no room: the
  first ``K`` of them by priority in the walk slots, ``terrain_finish_overflow``, state ``TERRAIN_PAIR_ROW_OVERFLOW``,
  ``terrain_rows``).
"""

TERRAIN_LAW_BARREL_KERNELS: tuple[str, tuple[tuple[str, str], ...]] = (
    "newton._src.geometry.terrain_law_barrel",
    _solid_launches("barrel"),
)
"""The cylinder class's module and launches when its table holds a barrel cylinder
(:attr:`~newton._src.geometry.terrain_contacts.TerrainPairTable.barrels`): its kernels with the barrels' paths, which
:data:`TERRAIN_LAW_KERNELS`'s compile out (every pair of the class takes them; a flat-sided cylinder or cone gets the
same rows from both)."""

TERRAIN_LAW_SOLID_CLASSES = (TERRAIN_PARTNER_BOX, TERRAIN_PARTNER_CYLINDER, TERRAIN_PARTNER_HULL)
"""Partner classes whose kernels take the solid classes' data
(:class:`~newton._src.geometry.terrain_law_solid.TerrainSolidData`: the hull tables, built once when one of them is
listed) and their class's jobs."""

TERRAIN_LANE_NAMES = {TERRAIN_LANE_THREAD: "thread", TERRAIN_LANE_TILE: "tiled"}
"""Names of the pair table's lane classes, by id (the table orders a class's pairs by lane class id)."""

TERRAIN_PAIR_STATES = {
    "tile_exit": int(TERRAIN_PAIR_TILE_EXIT),
    "stale": int(TERRAIN_PAIR_STALE),
    "filtered": int(TERRAIN_PAIR_FILTERED),
    "outside": int(TERRAIN_PAIR_OUTSIDE),
    "row_overflow": int(TERRAIN_PAIR_ROW_OVERFLOW),
    "candidate_overflow": int(TERRAIN_PAIR_CANDIDATE_OVERFLOW),
    "incomplete": int(TERRAIN_PAIR_INCOMPLETE),
    "wide": int(TERRAIN_PAIR_WIDE),
}
"""Names of the ``TERRAIN_PAIR_*`` state bits."""


def terrain_law_class_name(partner_class: int) -> str:
    """Return the ``NEWTON_TERRAIN_LAW_SHADOW`` name of a partner class."""
    return next(name for name, value in TERRAIN_LAW_CLASS_NAMES.items() if value == partner_class)


def parse_terrain_law_shadow(value: str | None, flag: str = "NEWTON_TERRAIN_LAW_SHADOW") -> tuple[int, ...]:
    """Return the partner classes a ``NEWTON_TERRAIN_LAW_SHADOW`` (or ``NEWTON_TERRAIN_LAW``) value lists, sorted
    (empty for none).

    Args:
        value: Comma-separated names of :data:`TERRAIN_LAW_CLASS_NAMES`, or ``all`` for every class with a
            kernel; ``None`` or blank for none.
        flag: The variable's name, for the errors.

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
                f"{flag} names unknown partner class {name!r}; "
                f"use a comma-separated list of {sorted(TERRAIN_LAW_CLASS_NAMES)} or 'all'"
            )
        partner_class = TERRAIN_LAW_CLASS_NAMES[name]
        if partner_class not in TERRAIN_LAW_KERNELS:
            raise ValueError(f"{flag}: the terrain law has no kernel for {name!r} yet")
        classes.add(partner_class)
    return tuple(sorted(classes))


def parse_terrain_law(value: str | None) -> tuple[str, tuple[int, ...]]:
    """Return what a ``NEWTON_TERRAIN_LAW`` value requests and the partner classes it lists.

    Args:
        value: ``None`` or blank for the default; ``none`` to keep the terrain route (a temporary opt-out until that
            route is removed); else comma-separated names of :data:`TERRAIN_LAW_CLASS_NAMES`, or ``all``, to request
            the law for them (:func:`parse_terrain_law_shadow`).

    Returns:
        ``("default", ())``: the law writes every class with a kernel (:data:`TERRAIN_LAW_KERNELS`) where the
        collision pipeline's configuration is one it supports, else the terrain route writes; ``("none", ())``; or
        ``("explicit", classes)``: the law writes the listed classes, and an unsupported configuration raises.

    Raises:
        ValueError: On an unknown name, a named class without a kernel yet, or ``none`` listed with other names.
    """
    names = [item.strip() for item in (value or "").split(",") if item.strip()]
    if not names:
        return "default", ()
    if "none" in names:
        if names != ["none"]:
            raise ValueError(f"NEWTON_TERRAIN_LAW={value!r}: 'none' opts out of the terrain law and stands alone")
        return "none", ()
    return "explicit", parse_terrain_law_shadow(value, flag="NEWTON_TERRAIN_LAW")


TERRAIN_LAW_SPILL_SHARE = 256
"""The default spill region holds the sum of the listed pairs' bounds on their rows over this, beyond the largest
pair's bound (:meth:`TerrainLawShadow.default_spill_rows`): the worst case of one pair in 256 at once. On the recorded
suite scenes (16,384 worlds) that is 11 to 25% of the walk slots, while at most one pair spilled (30 rows)."""

TERRAIN_LAW_JOB_COUNTERS = 10
"""Counters of a solid class's jobs: its five job lists, its candidate pool, its live pairs, its pairs left to rule R,
its scan's runs and its pairs finished by the small-pair finish kernel
(:data:`~newton._src.geometry.terrain_law_solid.TERRAIN_SOLID_COUNTERS`)."""

TERRAIN_LAW_JOB_NAMES = (
    "faces",
    "edges",
    "thetas",
    "vertices",
    "border",
    "candidates",
    "live",
    "recovery",
    "scans",
    "small",
)
"""Names of a solid class's job counters, in order (:meth:`TerrainLawShadow.job_counts`)."""

_JOB_COUNTERS = wp.constant(TERRAIN_LAW_JOB_COUNTERS)
_JOB_SCANS = wp.constant(TERRAIN_LAW_JOB_NAMES.index("scans"))
_BOX = wp.constant(TERRAIN_PARTNER_BOX)
_CYLINDER = wp.constant(TERRAIN_PARTNER_CYLINDER)

_DEFERRED = 0
"""Offset of the per-class deferred-pair counters in the shadow's counters."""
_QUEUED = TERRAIN_PARTNER_CLASSES
"""Offset of the per-class tiled-queue counters."""
_SPILLED = 2 * TERRAIN_PARTNER_CLASSES
"""Offset of the spill rows asked for."""
_RESERVED = _SPILLED + 1
"""Offset of the spill rows reserved."""
_JOBS = _RESERVED + 1
"""Offset of the per-class job counters (:data:`TERRAIN_LAW_JOB_COUNTERS` per class)."""
_COUNTERS = _JOBS + TERRAIN_PARTNER_CLASSES * TERRAIN_LAW_JOB_COUNTERS
"""The shadow's counters, all zeroed by one memset before every launch."""


@wp.func
def _print_solid_class(partner_class: int):
    """Print a solid class's ``NEWTON_TERRAIN_LAW_SHADOW`` name (no newline)."""
    if partner_class == _BOX:
        wp.printf("box")
    elif partner_class == _CYLINDER:
        wp.printf("cylinder")
    else:
        wp.printf("hull")


@wp.func
def _print_job_list(k: int):
    """Print the name of a solid class's job counter ``k`` (:data:`TERRAIN_LAW_JOB_NAMES`) that has a room (no
    newline)."""
    if k == wp.static(TERRAIN_LAW_JOB_NAMES.index("faces")):
        wp.printf("rule F face job list")
    elif k == wp.static(TERRAIN_LAW_JOB_NAMES.index("edges")):
        wp.printf("rule E edge job list")
    elif k == wp.static(TERRAIN_LAW_JOB_NAMES.index("thetas")):
        wp.printf("rule E crease (theta*) job list")
    elif k == wp.static(TERRAIN_LAW_JOB_NAMES.index("vertices")):
        wp.printf("rule V vertex job list")
    elif k == wp.static(TERRAIN_LAW_JOB_NAMES.index("border")):
        wp.printf("border wall and clip job list")
    else:
        wp.printf("candidate pool")


@wp.kernel(enable_backward=False)
def terrain_law_spill_check(
    counters: wp.array[wp.int32],
    spilled: int,
    spill: int,
    jobs: int,
    capacities: wp.array[wp.int32],
    reserves: wp.array[wp.int32],
):
    """After a launch: warn when the spill region overflowed, with the spill rows asked for against its rows (the pairs
    that found no room have raised ``terrain_rows``), and when a solid class's job list, candidate pool or scan's runs
    outgrew their room, with the entries asked for against it, by class and list (the pairs whose work took the reserve
    beyond it, their rows unchanged, have raised ``terrain_jobs``), and again when the reserve overflowed too (the pairs
    whose work found no room there keep no row). The scan's runs are sized exactly from the pairs' windows: they
    overflow only when a window outgrew the bound its shape data gave at construction."""
    if counters[spilled] > spill:
        wp.printf(
            "Warning: Terrain-law spill region overflowed %d > %d. Set NEWTON_TERRAIN_LAW_SHADOW_ROWS to at least the "
            "first number.\n",
            counters[spilled],
            spill,
        )
    for k in range(capacities.shape[0]):
        if capacities[k] > 0 and counters[jobs + k] > capacities[k]:
            kind = k % _JOB_COUNTERS
            wp.printf("Warning: Terrain-law ")
            _print_solid_class(k // _JOB_COUNTERS)
            if kind == _JOB_SCANS:
                wp.printf(
                    " class: its scan's runs overflowed %d > %d: windows outgrew the bounds their shapes gave when the "
                    "collision pipeline was built (their runs took the reserve). Rebuild the collision pipeline.\n",
                    counters[jobs + k],
                    capacities[k],
                )
            else:
                wp.printf(" class: its ")
                _print_job_list(kind)
                wp.printf(
                    " overflowed %d > %d: those pairs' work took its reserve (rows unchanged). Raise "
                    "NEWTON_TERRAIN_LAW_SHADOW_JOBS, a factor on every job list's room, by at least the first number "
                    "over the second.\n",
                    counters[jobs + k],
                    capacities[k],
                )
            if counters[jobs + k] > capacities[k] + reserves[k]:
                wp.printf("Warning: Terrain-law ")
                _print_solid_class(k // _JOB_COUNTERS)
                wp.printf(
                    " class: the reserve of %d overflowed too: the pairs whose work found no room there kept no row.\n",
                    reserves[k],
                )


_CONTACT_KERNELS = {}

_CONTACTS_OVERFLOWED = wp.constant(1024)
"""The ``contacts`` bit of the narrow phase's buffer status word (``verify_narrow_phase_buffers``), which checks the
contact buffer before the law writes."""


def terrain_law_contact_kernel(writer_func: Any):
    """Return the kernel that writes the law's rows as contacts through a contact writer (the default; see
    ``NEWTON_TERRAIN_LAW``).

    One thread per written pair (``written``, table indices): its rows pass the writer's gap admission (the surface
    distance ``d - r`` beyond both margins within both gaps, ``r`` the radius of a sphere or capsule partner, else 0),
    one reservation takes the admitted rows' contacts, and the writer writes each at its index. A contact's shape A is
    the heightfield, its normal the row's (terrain to partner), its points the terrain point ``q`` and the partner's
    core point ``q + d n`` (a sphere or capsule partner's ``radius_eff_b`` is ``r``) and its ``sort_sub_key`` the row's
    window-local feature id.

    Args:
        writer_func: Contact writer ``(ContactData, writer_data, output_index)`` that writes at a given index
            ``output_index >= 0`` without admission.
    """
    kernel = _CONTACT_KERNELS.get(writer_func)
    if kernel is not None:
        return kernel

    @wp.kernel(enable_backward=False, module=f"terrain_law_contacts_{writer_func.key}")
    def terrain_law_write_contacts(
        written: wp.array[wp.int32],
        pairs: wp.array[wp.vec2i],
        rows: TerrainLawRows,
        shape_type: wp.array[wp.int32],
        shape_data: wp.array[wp.vec4],
        writer_data: Any,
    ):
        pair = written[wp.tid()]
        count = rows.count[pair]
        if count <= 0:
            return
        first = rows.first[pair]
        hfield = pairs[pair][0]
        partner = pairs[pair][1]
        radius = float(0.0)
        kind = shape_type[partner]
        if kind == GeoType.SPHERE or kind == GeoType.CAPSULE:
            radius = shape_data[partner][0]
        margin_a = shape_data[hfield][3]
        margin_b = shape_data[partner][3]
        gap = writer_data.shape_gap[hfield] + writer_data.shape_gap[partner]
        admitted = int(0)
        for i in range(count):
            if rows.d[first + i] - radius - margin_a - margin_b <= gap:
                admitted += 1
        if admitted == 0:
            return
        index = wp.atomic_add(writer_data.contact_count, 0, admitted)
        if index + admitted > writer_data.contact_max:
            # The writer drops the contacts past the buffer; raise the narrow phase's contacts bit, warn once.
            wp.atomic_or(rows.status, 0, _CONTACTS_OVERFLOWED)
            if index <= writer_data.contact_max:
                wp.printf(
                    "Warning: Contact buffer overflowed by the terrain law's contacts: %d > %d.\n",
                    index + admitted,
                    writer_data.contact_max,
                )
        contact = ContactData()
        contact.shape_a = hfield
        contact.shape_b = partner
        contact.radius_eff_a = 0.0
        contact.radius_eff_b = radius
        contact.margin_a = margin_a
        contact.margin_b = margin_b
        contact.gap_sum = gap
        for i in range(count):
            d = rows.d[first + i]
            if d - radius - margin_a - margin_b > gap:
                continue
            n = rows.n[first + i]
            contact.contact_normal_a_to_b = n
            contact.contact_distance = d - radius
            contact.contact_point_center = rows.q[first + i] + (0.5 * (d - radius)) * n
            contact.sort_sub_key = rows.feature[first + i]
            writer_func(contact, writer_data, index)
            index += 1

    _CONTACT_KERNELS[writer_func] = terrain_law_write_contacts
    return terrain_law_write_contacts


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
        if job_scale is not None and not (np.isfinite(float(job_scale)) and float(job_scale) > 0.0):
            raise ValueError(f"The terrain law's job capacities need a positive finite factor, not {job_scale!r}")
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
        # Per class its deferred pairs, per class its queued pairs, the spill rows asked for and reserved, and per class
        # its jobs: zeroed by one memset before every launch.
        self._counters = wp.zeros(_COUNTERS, dtype=wp.int32, device=device)
        capacities = np.zeros(TERRAIN_PARTNER_CLASSES * TERRAIN_LAW_JOB_COUNTERS, np.int32)
        reserves = np.zeros_like(capacities)
        solid_pairs = []
        solid_module = None
        if any(c in TERRAIN_LAW_SOLID_CLASSES for c in self.classes):
            solid_module = importlib.import_module("newton._src.geometry.terrain_law_solid")
            assert solid_module.TERRAIN_SOLID_COUNTERS == TERRAIN_LAW_JOB_COUNTERS
            assert [kind for kind, _kernel in _solid_launches("box")] == [
                kind for kind, _kernel in solid_module.TERRAIN_SOLID_LAUNCHES
            ]
            assert (
                TERRAIN_LAW_JOB_NAMES[: len(solid_module.TERRAIN_SOLID_JOB_LISTS)]
                == solid_module.TERRAIN_SOLID_JOB_LISTS
            )
        self._jobs = {}
        for partner_class in self.classes:
            module_name, launches = TERRAIN_LAW_KERNELS[partner_class]
            if partner_class == TERRAIN_PARTNER_CYLINDER and self.table.barrels:
                module_name, launches = TERRAIN_LAW_BARREL_KERNELS
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
                run = int(TERRAIN_PARTNER_SCAN_RUN[partner_class])
                capacity = solid_module.terrain_solid_job_capacities(
                    self.table.vertices[begin:end],
                    self.table.candidates[begin:end],
                    walk,
                    scan_run=run,
                    sides=self.table.sides[begin:end],
                    scale=self.job_scale,
                )
                reserve = solid_module.terrain_solid_job_reserves(
                    self.table.vertices[begin:end],
                    self.table.candidates[begin:end],
                    scan_run=run,
                    sides=self.table.sides[begin:end],
                )
                offset = _JOBS + TERRAIN_LAW_JOB_COUNTERS * partner_class
                jobs = solid_module.terrain_solid_jobs(
                    device,
                    partner_class,
                    begin,
                    end - begin,
                    capacity,
                    reserve,
                    counts=self._counters[offset : offset + TERRAIN_LAW_JOB_COUNTERS],
                )
                self._jobs[partner_class] = jobs
                names = (*solid_module.TERRAIN_SOLID_JOB_LISTS, "candidates", "scans")
                # the live, rule R and small lists hold every pair of the class: no check
                for name in names:
                    capacities[offset - _JOBS + TERRAIN_LAW_JOB_NAMES.index(name)] = capacity[name]
                    reserves[offset - _JOBS + TERRAIN_LAW_JOB_NAMES.index(name)] = reserve[name]
                for kind, kernel_name in launches:
                    kernel = getattr(module, kernel_name)
                    if kind == "pairs":
                        self._launches.append(("thread", kernel, end - begin, begin, slot, (jobs, True)))
                    elif kind == "pair_jobs":
                        self._launches.append(("thread", kernel, jobs.pair_threads, begin, slot, (jobs, True)))
                    elif kind == "pair_tiles":
                        self._launches.append(("tiled", kernel, jobs.finish_blocks, begin, slot, (jobs, True)))
                    elif kind == "jobs":
                        self._launches.append(("thread", kernel, jobs.threads, begin, slot, (jobs, True)))
                    elif kind == "border_jobs":
                        self._launches.append(("thread", kernel, jobs.border_threads, begin, slot, (jobs, True)))
                    else:
                        self._launches.append(("tiled", kernel, scratch.blocks, begin, slot, (scratch, True, jobs)))
                shapes.update(
                    {
                        "pair_dim": end - begin,
                        "job_threads": jobs.threads,
                        "border_threads": jobs.border_threads,
                        "pair_job_threads": jobs.pair_threads,
                        "finish_blocks": jobs.finish_blocks,
                        "jobs": {name: int(capacity[name]) for name in names},
                        "jobs_reserve": {name: int(reserve[name]) for name in names},
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
            # The finish of the class's deferred pairs: a fixed grid over its deferred list (the pairs are found on the
            # device, in any order), each block holding every candidate a window of the class can give. A
            # class whose largest pair's window can give many candidates takes fewer blocks, its scratch kept near
            # TERRAIN_LAW_SCRATCH_BYTES: the rare deferred path's parallelism, never a row.
            shapes["deferred_blocks"] = scratch.blocks
            shapes["deferred_candidates"] = scratch.capacity
            shapes["deferred_limited"] = scratch.blocks < TERRAIN_LAW_DEFERRED_BLOCKS
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
        self._reserves = wp.array(reserves, dtype=wp.int32, device=device)
        self._solids = None
        if solid_pairs:
            partners = np.concatenate([self.table.pairs_host[b : b + c, 1] for b, c in solid_pairs])
            self._solids = solid_module.terrain_solid_data(model, partners)
        self._no_body_flags = wp.empty(0, dtype=wp.int32, device=device)
        self.build_seconds = time.perf_counter() - started
        """Host time of the construction [s]: features, pair table, kernel modules imported and buffers."""

    def written_pairs(self, model) -> tuple[np.ndarray, np.ndarray]:
        """Return what the law writes when it feeds the solver (the default; see ``NEWTON_TERRAIN_LAW``): per shape, 1
        for a partner whose heightfield pairs it writes (else 0), and the table indices of those pairs: every listed
        pair (every partner of a listed class has a core).

        Args:
            model: The finalized model the law was built for.
        """
        listed = self._listed
        partner = np.zeros(model.shape_count, np.int32)
        partner[self.table.pairs_host[listed, 1].astype(np.int64)] = 1
        return partner, listed.astype(np.int32)

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
            inputs=[self._counters, _SPILLED, self.spill_rows, _JOBS, self._capacities, self._reserves],
            device=self.device,
            record_tape=False,
        )

    def write_contacts(
        self, written: wp.array, shape_type: wp.array, shape_data: wp.array, writer_func: Any, writer_data: Any
    ) -> None:
        """Write the rows of the last :meth:`launch` as contacts (graph-capture safe), for the pairs
        :meth:`written_pairs` returns (:func:`terrain_law_contact_kernel`).

        Args:
            written: Table indices of the pairs to write (int32).
            shape_type: Geometry type of every shape.
            shape_data: Per shape, scale ``xyz`` and margin ``w``, current for this collide.
            writer_func: Contact writer that writes at a given index without admission.
            writer_data: The writer's data (its contact counter, gaps and output arrays).
        """
        if written.shape[0] == 0:
            return
        wp.launch(
            terrain_law_contact_kernel(writer_func),
            dim=written.shape[0],
            inputs=[written, self.table.pairs, self._out, shape_type, shape_data, writer_data],
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

    def overflow_counts(self) -> tuple[int, int, int]:
        """Return the pairs whose rows found no room in the spill region (``terrain_rows``), whose windows outgrew their
        scratch (``terrain_rows``) and whose work found no room in their solid class's job lists or candidate pool
        (``terrain_jobs``: the reserve took it, their rows unchanged, or past the reserve they keep no row), summed over
        every launch so far."""
        if self.device.is_capturing:
            raise RuntimeError("Terrain-law shadow counters must be read outside graph capture")
        rows, candidates, jobs = (int(v) for v in self._out.overflow.numpy())
        return rows, candidates, jobs

    def job_counts(self) -> dict[str, dict[str, int]]:
        """Return, per listed solid class with pairs, the entries its last launch asked of every job list, of its
        candidate pool and of its scan's runs (past a room, :meth:`describe`'s ``launch_shapes``: entries that took the
        reserve, ``jobs_reserve``, or found no room), its live pairs, its pairs left to rule R and its pairs given to
        the small-pair finish, by :data:`TERRAIN_LAW_JOB_NAMES`."""
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
        region, memory and with a solid class listed the solid classes' job scale.

        ``launches`` counts every launch of a collide, the spill and job check included. ``launch_shapes`` holds, per
        class with pairs, the dimensions fixed at construction: ``block_dim`` (every launch's), ``walk_slots`` (a
        pair's walk slots), ``deferred_blocks`` (the grid of the finish of its deferred pairs), ``deferred_candidates``
        (the candidates a block of it holds: the most any pair of the class can give) and ``deferred_limited``
        (whether that grid has fewer than :data:`~newton._src.geometry.terrain_law.TERRAIN_LAW_DEFERRED_BLOCKS` blocks:
        its blocks, each sized for the class's largest pair, would exceed
        :data:`~newton._src.geometry.terrain_law.TERRAIN_LAW_SCRATCH_BYTES`); for the point class ``thread_dim`` (the
        one-thread kernel's threads, one per pair of the class) and ``tiled_blocks`` (the tiled kernel's grid, 0
        without a tiled segment); for a solid class ``pair_dim`` (its whole-pair rejects' threads, one per pair of the
        class), ``job_threads`` (its scan and job kernels' threads), ``pair_job_threads`` (its close and small-pair
        finish kernels' threads, at most one per pair of the class), ``finish_blocks`` (its rule R kernel's grid),
        ``jobs`` (the room of its job lists, its candidate pool and its scan's runs) and ``jobs_reserve`` (their
        reserves). ``slots`` and ``spill_rows`` are the side buffer's walk slots and spill rows.
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
            if kernel.key.endswith("_finish")
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
                )
            )
        solids = {}
        if self._solids is not None:
            solids = {"job_scale": self.job_scale}
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
