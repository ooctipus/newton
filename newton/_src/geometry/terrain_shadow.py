# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run the terrain law beside the narrow phase, into a side buffer that nothing in the step reads (shadow mode).

``NEWTON_TERRAIN_LAW_SHADOW`` lists partner classes (``point``: spheres and capsules; ``box``; ``cylinder``:
cylinders and cones; ``hull``: convex hulls and ellipsoids; or ``all``), empty by default. With classes listed, a
collision pipeline builds :class:`TerrainLawShadow` once, at construction: the terrain features
(:class:`~newton._src.geometry.terrain_contacts.TerrainFeatures`), the static pair table
(:class:`~newton._src.geometry.terrain_contacts.TerrainPairTable`, from the pipeline's pairs) and per listed pair a
static range of row slots. Every collide then launches, after today's terrain route, one kernel per listed class and
lane class over its table segment, with dimensions fixed at construction: no host read, no allocation, no scan, so
it runs inside CUDA graph capture. Today's route still produces every contact; the shadow's rows are read only by
:meth:`TerrainLawShadow.rows`.

The per-type kernels share one signature, ``(begin, slot_begin, inputs, data, out)``
(:class:`~newton._src.geometry.terrain_law.TerrainPairInputs`, :class:`~newton._src.geometry.terrain_law.TerrainFeatureData`,
:class:`~newton._src.geometry.terrain_law.TerrainLawRows`), the solid classes' with a sixth argument, their
:class:`~newton._src.geometry.terrain_law_solid.TerrainSolidData` (hull tables, the full kernels' scratch rows; built
once when a solid class is listed): a kernel launched over ``count`` pairs handles table
pairs ``begin + i`` and writes their rows to slots ``slot_begin + i * K``, ``K`` its class's row limit
(:data:`~newton._src.geometry.terrain_contacts.TERRAIN_PARTNER_ROWS`), their row counts and ``TERRAIN_PAIR_*``
states. The one-thread kernel of a class takes pairs whose windows hold at most
:data:`~newton._src.geometry.terrain_contacts.TERRAIN_LANE_CELLS` cells; the tiled one is launched with
``wp.launch_tiled`` and :data:`~newton._src.geometry.terrain_contacts.TERRAIN_TILE_LANES` lanes per pair.
:data:`TERRAIN_LAW_KERNELS` names them per class; a class without an entry has no kernel yet.

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
    TERRAIN_LANE_CLASSES,
    TERRAIN_LANE_THREAD,
    TERRAIN_PARTNER_BOX,
    TERRAIN_PARTNER_CLASSES,
    TERRAIN_PARTNER_CYLINDER,
    TERRAIN_PARTNER_HULL,
    TERRAIN_PARTNER_POINT,
    TERRAIN_PARTNER_ROWS,
    TERRAIN_TILE_LANES,
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
    TERRAIN_PAIR_UNSUPPORTED,
    TERRAIN_PAIR_WIDE,
    TerrainLawRows,
    TerrainPairInputs,
)

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
    TERRAIN_PARTNER_BOX: (
        "newton._src.geometry.terrain_law_box",
        "terrain_box_rows",
        "terrain_box_rows_tiled",
        "terrain_box_rows_full",
    ),
    TERRAIN_PARTNER_CYLINDER: (
        "newton._src.geometry.terrain_law_cylinder",
        "terrain_cylinder_rows",
        "terrain_cylinder_rows_tiled",
        "terrain_cylinder_rows_full",
    ),
    TERRAIN_PARTNER_HULL: (
        "newton._src.geometry.terrain_law_hull",
        "terrain_hull_rows",
        "terrain_hull_rows_tiled",
        "terrain_hull_rows_full",
    ),
}
"""Per partner class: the module, its one-thread and tiled kernels, and its full-finish kernel or ``None``.

The module is imported only when the class is listed. The one-thread kernel runs over the class's one-thread lane
segment, the tiled one over its tiled segment; the full-finish kernel, when the class has one, runs after them and
finishes the pairs they marked :data:`~newton._src.geometry.terrain_law.TERRAIN_PAIR_FULL`: the point class's over
the whole class (both segments, which the table keeps contiguous), a solid class's on a fixed grid of
:data:`~newton._src.geometry.terrain_law_solid.TERRAIN_SOLID_FULL_THREADS` threads that strides over the class's
deferred list (``begin`` and ``slot_begin`` are the class's first pair and slot)."""

TERRAIN_LAW_SOLID_CLASSES = (TERRAIN_PARTNER_BOX, TERRAIN_PARTNER_CYLINDER, TERRAIN_PARTNER_HULL)
"""Partner classes whose kernels take a sixth argument, the solid classes' data
(:class:`~newton._src.geometry.terrain_law_solid.TerrainSolidData`: hull tables and the full kernels' scratch rows),
built once when one of them is listed."""

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


class TerrainLawShadow:
    """The terrain law's kernels for some partner classes, run after the narrow phase into a side buffer.

    Built once, outside graph capture: the terrain features, the pair table and the side buffer (per listed pair
    ``K`` slots of a terrain point, a normal, a core distance and a feature id, plus a row count and a state).
    :meth:`launch` then runs one kernel per listed class and lane class with dimensions fixed here; the narrow
    phase's contacts are untouched.

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
        solid_pairs = []
        solid_module = None
        for partner_class in self.classes:
            module_name, thread_name, tiled_name, full_name = TERRAIN_LAW_KERNELS[partner_class]
            module = importlib.import_module(module_name)
            kernels = (getattr(module, thread_name), getattr(module, tiled_name))
            rows = TERRAIN_PARTNER_ROWS[partner_class]
            solid = partner_class in TERRAIN_LAW_SOLID_CLASSES
            if solid and solid_module is None:
                solid_module = importlib.import_module("newton._src.geometry.terrain_law_solid")
            class_begin, class_slot = None, slot
            for lane in range(TERRAIN_LANE_CLASSES):
                begin, end = self.table.segment(partner_class, lane)
                if end > begin:
                    self._launches.append((kernels[lane], lane, begin, end - begin, slot, solid))
                    slot_first[begin:end] = slot + rows * np.arange(end - begin)
                    slot += rows * (end - begin)
                    class_begin = begin if class_begin is None else class_begin
            if full_name is not None and class_begin is not None:
                count = (slot - class_slot) // rows
                # A solid class's full kernel is a fixed grid striding over the class's deferred list.
                threads = solid_module.TERRAIN_SOLID_FULL_THREADS if solid else count
                self._launches.append(
                    (getattr(module, full_name), TERRAIN_LANE_THREAD, class_begin, threads, class_slot, solid)
                )
                if solid:
                    solid_pairs.append((class_begin, count))
        self.slot_count = slot
        """Row slots of the side buffer: every listed pair times its class's row limit."""
        self.features = TerrainFeatures(model) if self._launches else None
        """The heightfields' terrain features (``None`` without a listed pair); report height edits to it and rebuild
        outside capture."""
        self._slot_first = slot_first
        self._listed = np.flatnonzero(slot_first >= 0)
        device = self.device
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
        out.deferred_count = wp.zeros(TERRAIN_PARTNER_CLASSES, dtype=wp.int32, device=device)
        self._out = out
        self._solids = None
        self._unsupported_pairs = 0
        if solid_pairs:
            partners = np.concatenate([self.table.pairs_host[b : b + c, 1] for b, c in solid_pairs])
            # One scratch list per thread of the full kernels' grid, which the solid classes' launches share in turn.
            self._solids = solid_module.terrain_solid_data(model, partners)
            classes = np.concatenate([self.table.partner_class[b : b + c] for b, c in solid_pairs])
            self._unsupported_pairs = int(
                np.count_nonzero(solid_module.terrain_solid_unsupported(classes, model.shape_type.numpy()[partners]))
            )
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
        self._out.deferred_count.zero_()
        for kernel, lane, begin, count, slot, solid in self._launches:
            args = [begin, slot, inputs, data, self._out]
            if solid:
                args.append(self._solids)
            if lane == TERRAIN_LANE_THREAD:
                wp.launch(kernel, dim=count, inputs=args, device=self.device, record_tape=False)
            else:
                wp.launch_tiled(
                    kernel, dim=count, inputs=args, block_dim=TERRAIN_TILE_LANES, device=self.device, record_tape=False
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
        """Return a host-only record of the shadow: classes, pairs per lane class, launches, memory, and with a solid
        class listed the threads of the full kernels' grid and the pairs whose partner has no core yet (no row)."""
        pairs = {}
        for partner_class in range(TERRAIN_PARTNER_CLASSES):
            if partner_class not in self.classes:
                continue
            per_lane = {}
            for lane, name in ((0, "thread"), (1, "tiled")):
                begin, end = self.table.segment(partner_class, lane)
                per_lane[name] = end - begin
            pairs[terrain_law_class_name(partner_class)] = per_lane
        out = self._out
        side = sum(
            array.capacity
            for array in (out.q, out.n, out.d, out.feature, out.count, out.state, out.overflow, out.deferred)
        )
        solid = 0
        if self._solids is not None:
            hulls = self._solids.hulls
            solid = sum(
                array.capacity
                for array in (
                    self._solids.scratch_q,
                    self._solids.scratch_n,
                    self._solids.scratch_d,
                    self._solids.scratch_feature,
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
            solids = {"full_threads": int(self._solids.full_threads), "unsupported_pairs": self._unsupported_pairs}
        return {
            "classes": [terrain_law_class_name(c) for c in self.classes],
            "pairs": pairs,
            "launches": len(self._launches),
            **solids,
            "slots": self.slot_count,
            "bytes": {
                "side_buffer": int(side),
                "solid_data": int(solid),
                "pair_table": int(self.table.memory_bytes()["pairs"]),
                "features": 0 if self.features is None else int(sum(self.features.memory_bytes().values())),
            },
            "build_seconds": round(self.build_seconds, 3),
        }
