# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Collision topology summaries and on-demand explicit shape pairs.

A replicated scene's worlds are copies of a few templates. :class:`_ShapeContactPairs` groups the worlds once by an
exact signature of their local pair structure (:class:`_WorldTemplates`) and computes every per-world quantity, pair
counts and pairs, on one representative world of each group; the other worlds take the representative's result,
shifted by their first shape (and first body). A shape mask or per-shape values (categories, types) refine the groups
by the values they take on each world's shapes (:meth:`_WorldTemplates.refine`), so worlds that differ are still
computed exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from ..core.host_unique import row_ids
from ..geometry.flags import ShapeFlags

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from ..geometry.broad_phase_nxn import ShapePairTemplates
    from ..solvers.coupled.model_view import ModelView
    from . import Model

# Bound enumeration and replicated replay scratch independently of scene size.
_PAIR_CHUNK_SIZE = 262144

_HASH_SEED = 0x5A17_C0DE
"""Seed of the position weights that fingerprint a world's signature (verified exactly after grouping)."""


def _segments(sizes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each element's segment and rank within it, for consecutive segments of ``sizes`` elements."""
    sizes = np.asarray(sizes, dtype=np.int64)
    total = int(sizes.sum())
    starts = np.zeros(sizes.size + 1, dtype=np.int64)
    np.cumsum(sizes, out=starts[1:])
    marks = np.zeros(total + 1, dtype=np.int64)
    np.add.at(marks, starts[:-1][sizes > 0], 1)
    segment = np.cumsum(marks[:total]) - 1
    # Empty segments own no element: skip their ids.
    segment = np.flatnonzero(sizes)[segment] if total else segment
    return segment, np.arange(total, dtype=np.int64) - starts[segment]


@dataclass(frozen=True, eq=False)
class _Part:
    """Elements of each world compared position by position: ``values`` (one column per field) of the world's
    elements, which are consecutive (``offsets``) in world order."""

    values: tuple[np.ndarray, ...]
    offsets: np.ndarray
    world: np.ndarray
    rank: np.ndarray

    @classmethod
    def of(cls, values: tuple[np.ndarray, ...], sizes: np.ndarray) -> _Part:
        offsets = np.zeros(sizes.size + 1, dtype=np.int64)
        np.cumsum(sizes, out=offsets[1:])
        world, rank = _segments(sizes)
        return cls(tuple(np.asarray(v, dtype=np.int64) for v in values), offsets, world, rank)

    def columns(self) -> list[np.ndarray]:
        """Each world's element count and a 64-bit fingerprint of its elements (position-weighted field sums)."""
        rng = np.random.default_rng(_HASH_SEED + len(self.values))
        longest = int(np.max(np.diff(self.offsets), initial=0))
        total = np.zeros(self.rank.size, dtype=np.uint64)
        for column in self.values:
            weights = rng.integers(1, 2**63, size=longest, dtype=np.uint64) | np.uint64(1)
            total += column.view(np.uint64) * weights[self.rank]
        running = np.zeros(self.rank.size + 1, dtype=np.uint64)
        np.cumsum(total, out=running[1:])
        fingerprint = (running[self.offsets[1:]] - running[self.offsets[:-1]]).view(np.int64)
        return [np.diff(self.offsets), fingerprint]

    def mismatched(self, representative: np.ndarray) -> np.ndarray:
        """The worlds whose elements differ from those of their ``representative`` world (same counts)."""
        source = self.offsets[representative[self.world]] + self.rank
        differs = np.zeros(self.rank.size, dtype=bool)
        for column in self.values:
            differs |= column != column[source]
        return np.unique(self.world[differs])


@dataclass(frozen=True, eq=False)
class _FilterBlocks:
    """Each world's exclusions with a first shape in the world: consecutive packed codes ``[lo, hi)`` of the sorted
    exclusions. Two worlds' blocks are equal when each code differs by the worlds' shape offset in its first shape,
    and in its second unless that is global (at or past ``tail``)."""

    codes: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    world_start: np.ndarray
    world_end: np.ndarray
    tail: int

    _SAMPLES = 8

    def _normalized(self, world: np.ndarray, rank: np.ndarray) -> np.ndarray:
        code = self.codes[self.lo[world] + rank]
        start = self.world_start[world]
        return code - start * (1 << 32) - start * ((code & 0xFFFFFFFF) < self.tail)

    def columns(self) -> list[np.ndarray]:
        """Each world's block length and a few of its codes, normalized to the world's first shape."""
        lengths = self.hi - self.lo
        worlds = np.flatnonzero(lengths)
        columns = [lengths]
        for sample in range(self._SAMPLES):
            column = np.zeros(lengths.size, dtype=np.int64)
            column[worlds] = self._normalized(worlds, lengths[worlds] * sample // self._SAMPLES)
            columns.append(column)
        return columns

    def mismatched(self, representative: np.ndarray) -> np.ndarray:
        """The worlds whose block differs from their ``representative`` world's (equal lengths)."""
        found = []
        shift = self.world_start - self.world_start[representative]
        for first in np.unique(representative).tolist():
            members = np.flatnonzero((representative == first) & (np.arange(representative.size) != first))
            count = int(self.hi[first] - self.lo[first])
            if not members.size or not count:
                continue
            reference = self.codes[self.lo[first] : self.hi[first]]
            second = reference & 0xFFFFFFFF
            # A local second shape moves with the world; a global one stays.
            weight = (1 << 32) + (second < self.tail).astype(np.int64)
            # A second shape in a later world is no exclusion of the world; it must stay off the global shapes.
            foreign = np.flatnonzero((second < self.tail) & (second >= self.world_end[first]))
            for begin in range(0, members.size, max(1, _PAIR_CHUNK_SIZE // count)):
                chunk = members[begin : begin + max(1, _PAIR_CHUNK_SIZE // count)]
                block = self.codes[self.lo[chunk, None] + np.arange(count)]
                block -= reference
                differs = np.any(block != shift[chunk, None] * weight, axis=1)
                if foreign.size:
                    later = self.codes[self.lo[chunk, None] + foreign] & 0xFFFFFFFF
                    differs |= np.any(later >= self.tail, axis=1)
                found.append(chunk[differs])
        return np.concatenate(found) if found else np.zeros(0, dtype=np.int64)


def _group_worlds(world_count: int, keys: list[np.ndarray], parts: list) -> tuple[np.ndarray, np.ndarray]:
    """Group the ``world_count`` worlds whose ``keys`` and ``parts`` are equal, exactly.

    Worlds are grouped by their keys and their parts' summary columns, then each world is compared with its group's
    first world element by element; a world that differs moves to a new group, until every group verifies.

    Returns:
        ``(group, first)``: each world's group, numbered by first world, and each group's first world.
    """
    if world_count == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    columns = [np.zeros(world_count, dtype=np.int64)] + [np.asarray(k, dtype=np.int64) for k in keys]
    for part in parts:
        columns += part.columns()
    group = row_ids(np.column_stack(columns))
    while True:
        first = np.full(int(group.max()) + 1, world_count, dtype=np.int64)
        np.minimum.at(first, group, np.arange(world_count))
        representative = first[group]
        moved = np.unique(np.concatenate([np.zeros(0, dtype=np.int64)] + [p.mismatched(representative) for p in parts]))
        if not moved.size:
            break
        group[moved] += first.size
        group = np.unique(group, return_inverse=True)[1].reshape(-1)
    # Number the groups by their first world.
    order = np.argsort(first, kind="stable")
    rank = np.empty(first.size, dtype=np.int64)
    rank[order] = np.arange(first.size)
    return rank[group], first[order]


@dataclass(frozen=True, eq=False)
class _WorldGroups:
    """Worlds grouped so that one representative world (its group's first) stands for the others."""

    group: np.ndarray
    """Each local world's group, numbered by first world."""
    first: np.ndarray
    """Each group's first (representative) world."""
    template: np.ndarray
    """Each group's template (:attr:`_WorldTemplates.templates`)."""

    @cached_property
    def sizes(self) -> np.ndarray:
        """The number of worlds of each group."""
        return np.bincount(self.group, minlength=self.first.size)


@dataclass(frozen=True, eq=False)
class _WorldTemplates:
    """Each local world's exact local pair structure, as templates.

    The shapes are laid out as :attr:`~newton.Model.shape_world_start` describes: global shapes first and last, each
    world's shapes contiguous, in world order. A world's signature is its collidable shapes, each by its index, body
    and collision group (relative to the world's first shape and first local body; a body some global shape is on by
    its index), and its authored exclusions among its shapes and with global shapes (relative the same way). Worlds
    of equal signatures have equal pairs, shifted by their first shapes.
    """

    world_start: np.ndarray
    """Each world's first shape."""
    body_start: np.ndarray
    """Each world's first body among its collidable shapes' bodies that no global shape is on (0 for none)."""
    global_bodies: np.ndarray
    """The bodies of collidable global shapes, ascending."""
    globals_: np.ndarray
    """The collidable global shapes, ascending."""
    shapes: _Part
    """Each world's collidable shapes, ascending; :attr:`local_shapes` holds their indices."""
    local_shapes: np.ndarray
    templates: _WorldGroups
    template_filters: tuple[np.ndarray, ...]
    """Each template's first world's exclusions (packed codes) that involve its shapes."""
    global_filters: np.ndarray
    """The exclusions between global shapes (packed codes)."""

    @classmethod
    def of(cls, topology: _ShapeContactPairs) -> _WorldTemplates:
        world_count = topology.world_count
        shape_world, shape_body = topology.shape_world, topology.shape_body
        count = shape_world.size
        local = shape_world >= 0
        sizes = np.bincount(shape_world[local], minlength=world_count)
        front = int(np.argmax(local)) if local.any() else count
        tail = front + int(sizes.sum())
        block = shape_world[front:tail]
        if np.any(block < 0) or np.any(block[1:] < block[:-1]) or np.any(shape_world[tail:] >= 0):
            raise ValueError(
                "Shapes must be laid out as Model.shape_world_start describes: global shapes first and last, each "
                "world's shapes contiguous, in world order"
            )
        world_start = front + np.cumsum(sizes) - sizes
        world_end = world_start + sizes

        active = topology._active()
        globals_ = np.flatnonzero(active & ~local)
        global_bodies = np.unique(shape_body[globals_])
        global_bodies = global_bodies[global_bodies >= 0]
        indices = front + np.flatnonzero(active[front:tail])
        world = shape_world[indices]
        bodies = shape_body[indices].astype(np.int64)
        on_global = np.isin(bodies, global_bodies)
        own = (bodies >= 0) & ~on_global
        body_start = np.full(world_count, np.iinfo(np.int64).max, dtype=np.int64)
        np.minimum.at(body_start, world[own], bodies[own])
        body_start[body_start == np.iinfo(np.int64).max] = 0
        body_code = np.where(own, bodies - body_start[world], np.where(on_global, -2 - bodies, -1))
        shapes = _Part.of(
            (indices - world_start[world], body_code, topology.shape_group[indices]),
            np.bincount(world, minlength=world_count),
        )

        # Exclusions by the world they belong to: a block of those whose first shape is in the world, and those
        # of a leading global shape with a shape of the world.
        codes = topology.filter_pairs
        lo, hi = np.searchsorted(codes, world_start << 32), np.searchsorted(codes, world_end << 32)
        leading = codes[: np.searchsorted(codes, front << 32)]
        second = leading & 0xFFFFFFFF
        owner = np.where((second >= front) & (second < tail), np.searchsorted(world_end, second, side="right"), -1)
        global_filters = np.concatenate((leading[owner < 0], codes[np.searchsorted(codes, tail << 32) :]))
        crossing = np.flatnonzero(owner >= 0)
        crossing = crossing[np.argsort(owner[crossing], kind="stable")]
        cross = _Part.of(
            (leading[crossing] - world_start[owner[crossing]],),
            np.bincount(owner[crossing], minlength=world_count),
        )
        blocks = _FilterBlocks(codes, lo, hi, world_start, world_end, tail)
        group, first_world = _group_worlds(world_count, [], [shapes, blocks, cross])
        template_filters = tuple(
            np.concatenate((codes[lo[w] : hi[w]], leading[crossing[cross.offsets[w] : cross.offsets[w + 1]]]))
            for w in first_world.tolist()
        )
        templates = _WorldGroups(group, first_world, np.arange(first_world.size))
        return cls(
            world_start,
            body_start,
            global_bodies,
            globals_,
            shapes,
            indices,
            templates,
            template_filters,
            global_filters,
        )

    def refine(self, *values: np.ndarray) -> _WorldGroups:
        """The templates split by the per-shape ``values`` on each world's collidable shapes."""
        if not values:
            return self.templates
        shapes = self.shapes
        part = _Part(
            tuple(np.asarray(v)[self.local_shapes].astype(np.int64) for v in values),
            shapes.offsets,
            shapes.world,
            shapes.rank,
        )
        group, first = _group_worlds(self.world_start.size, [self.templates.group], [part])
        return _WorldGroups(group, first, self.templates.group[first])

    def world_shapes(self, world: int) -> np.ndarray:
        """The collidable shapes of ``world``, ascending."""
        offsets = self.shapes.offsets
        return self.local_shapes[offsets[world] : offsets[world + 1]]

    def filters_of(self, worlds: np.ndarray, shape_world: np.ndarray) -> np.ndarray:
        """The exclusions (packed codes) that belong to ``worlds``: their templates', shifted to their shapes."""
        found = [self.global_filters[:0]]
        template = self.templates.group[worlds]
        for t in np.unique(template).tolist():
            members = worlds[template == t]
            codes = self.template_filters[t]
            shift = (self.world_start[members] - self.world_start[self.templates.first[t]])[:, None]
            first, second = codes >> 32, codes & 0xFFFFFFFF
            first = first + shift * (shape_world[first] >= 0)
            second = second + shift * (shape_world[second] >= 0)
            found.append(((first << 32) | second).reshape(-1))
        return np.concatenate(found)

    def shift(self, groups: _WorldGroups) -> tuple[np.ndarray, np.ndarray]:
        """Each world's shape and body offset from its group's representative world."""
        representative = groups.first[groups.group]
        return (
            self.world_start - self.world_start[representative],
            self.body_start - self.body_start[representative],
        )


class _ShapeContactPairs:
    """Finalized collision topology for counting and enumerating collision pairs."""

    def __init__(
        self,
        shape_body: ArrayLike,
        shape_world: ArrayLike,
        shape_group: ArrayLike,
        shape_flags: ArrayLike,
        filter_pairs: ArrayLike,
        world_count: int,
    ) -> None:
        self.shape_body = np.maximum(np.asarray(shape_body, dtype=np.int32), -1)
        self.shape_world = np.array(shape_world, dtype=np.int32, copy=True)
        self.shape_group = np.array(shape_group, dtype=np.int32, copy=True)
        self.shape_flags = np.array(shape_flags, dtype=np.int32, copy=True)
        self.filter_pairs = np.asarray(filter_pairs, dtype=np.int64)
        self.world_count = world_count
        self.pair_templates: ShapePairTemplates | None = None
        """The per-world templates of the last full :meth:`build_pairs`."""
        for array in (self.shape_body, self.shape_world, self.shape_group, self.shape_flags, self.filter_pairs):
            array.setflags(write=False)

    def _active(self, shape_mask: np.ndarray | None = None) -> np.ndarray:
        active = ((self.shape_flags & int(ShapeFlags.COLLIDE_SHAPES)) != 0) & (self.shape_group != 0)
        if shape_mask is not None:
            active &= shape_mask
        return active

    @cached_property
    def worlds(self) -> _WorldTemplates:
        """The worlds' templates, found once."""
        return _WorldTemplates.of(self)

    def _groups(self, shape_mask: np.ndarray | None, categories: np.ndarray | None = None) -> _WorldGroups:
        """The templates, split where ``shape_mask`` or ``categories`` differ between their worlds."""
        values = [v for v in (shape_mask, categories) if v is not None]
        return self.worlds.refine(*values)

    def _sample(self, groups: _WorldGroups, shape_mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        """The collidable shapes (within ``shape_mask``) of the global set and of each group's first world, and the
        exclusions that belong to them (packed codes)."""
        worlds = self.worlds
        selected = np.zeros(self.shape_body.size, dtype=bool)
        selected[worlds.globals_] = True
        sampled = np.zeros(self.world_count, dtype=bool)
        sampled[groups.first] = True
        selected[worlds.local_shapes[sampled[worlds.shapes.world]]] = True
        filters = np.concatenate((worlds.global_filters, worlds.filters_of(groups.first, self.shape_world)))
        return selected & self._active(shape_mask), filters

    def _active_filters(self, codes: np.ndarray, active: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The exclusions ``codes`` that remove a pair: both shapes ``active``, of different bodies, compatible
        worlds and compatible groups."""
        first = (codes >> 32).astype(np.int32)
        second = (codes & 0xFFFFFFFF).astype(np.int32)
        world_a, world_b = self.shape_world[first], self.shape_world[second]
        group_a, group_b = self.shape_group[first], self.shape_group[second]
        keep = (
            active[first]
            & active[second]
            & (self.shape_body[first] != self.shape_body[second])
            & ((world_a == world_b) | (world_a == -1) | (world_b == -1))
            & np.where(group_a > 0, (group_a == group_b) | (group_b < 0), group_a != group_b)
        )
        return first[keep], second[keep]

    @cached_property
    def counts(self) -> np.ndarray:
        """Exact pair counts: globals in slot zero, local world w in slot w + 1."""
        return self.count_pairs()

    def count_pairs(
        self,
        shape_mask: np.ndarray | None = None,
        *,
        categories: np.ndarray | None = None,
        weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """Count pairs or sum symmetric category weights without enumerating pairs.

        Each group of worlds (:meth:`_groups`) is counted on its first world; the others take its count.
        """
        groups = self._groups(shape_mask, categories)
        active, filters = self._sample(groups, shape_mask)
        sampled = self._count_active(active, filters, categories, weights)
        counts = np.empty(self.world_count + 1, dtype=np.int64)
        counts[0] = sampled[0]
        counts[1:] = sampled[1 + groups.first[groups.group]]
        return counts

    def _count_active(
        self, active: np.ndarray, filters: np.ndarray, categories: np.ndarray | None, weights: np.ndarray | None
    ) -> np.ndarray:
        """Count the pairs among the ``active`` shapes, less the exclusions among ``filters``, by world slot."""
        indices = np.flatnonzero(active)
        counts = np.zeros(self.world_count + 1, dtype=np.int64)
        if not len(indices):
            return counts
        categories = np.zeros(len(self.shape_body), dtype=np.int32) if categories is None else categories
        weights = np.ones((1, 1), dtype=np.int64) if weights is None else weights

        bodies = self.shape_body[indices]
        body_values, body_sizes = np.unique(bodies, return_counts=True)
        repeated = np.isin(bodies, body_values[body_sizes > 1])
        # Owner zero counts all shapes; other owners subtract same-body pairs.
        indices = np.concatenate((indices, indices[repeated]))
        owners = np.concatenate((np.zeros(len(bodies), dtype=np.int64), bodies[repeated].astype(np.int64) + 2))
        slots = self.shape_world[indices].astype(np.int64) + 1
        kinds = categories[indices]
        groups, group_ids = np.unique(self.shape_group[indices], return_inverse=True)
        group_count, kind_count = len(groups), len(weights)
        if not np.any(repeated):
            context_codes, contexts = np.arange(len(counts), dtype=np.int64), slots
        else:
            context_codes, contexts = np.unique(owners * len(counts) + slots, return_inverse=True)
        context_slots = context_codes % len(counts)
        context_owners = context_codes // len(counts)
        # Bound dense group histograms by the number of shapes; sparse groups
        # must not allocate a world-by-group Cartesian product.
        if len(context_codes) * group_count <= 2 * len(indices):
            group_codes = np.arange(len(context_codes) * group_count, dtype=np.int64)
            group_rows = contexts * group_count + group_ids
        else:
            group_codes, group_rows = np.unique(contexts * group_count + group_ids, return_inverse=True)
        group_contexts, group_types = np.divmod(group_codes, group_count)
        kind_codes = contexts * kind_count + kinds
        total = np.bincount(kind_codes, minlength=len(context_codes) * kind_count).reshape(-1, kind_count)
        positive = groups[group_ids] > 0
        pos = np.bincount(kind_codes[positive], minlength=total.size).reshape(total.shape)
        grouped = np.bincount(group_rows * kind_count + kinds, minlength=len(group_codes) * kind_count).reshape(
            -1, kind_count
        )
        signs = np.where(groups[group_types] > 0, 1, -1)

        # Subtract positive-only pairs, then add matching positives and remove
        # matching negatives. Apply the same rules within worlds and to globals.
        rows = np.arange(len(context_codes), dtype=np.int64)
        values = np.zeros(len(context_codes), dtype=np.int64)
        contributions = (
            (total, rows, context_owners, 1),
            (pos, rows, context_owners, -1),
            (grouped, group_contexts, context_owners[group_contexts] * group_count + group_types, signs),
        )
        for hist, rows, keys, sign in contributions:
            coefficients = np.broadcast_to(sign, len(hist))
            weighted = hist @ weights
            within = (np.sum(weighted * hist, axis=1) - hist @ weights.diagonal()) // 2
            np.add.at(values, rows, coefficients * within)
            globals_ = context_slots[rows] == 0
            global_keys = keys[globals_]
            if len(global_keys):
                positions = np.searchsorted(global_keys, keys)
                clipped = np.minimum(positions, len(global_keys) - 1)
                valid = (positions < len(global_keys)) & (global_keys[clipped] == keys) & ~globals_
                cross = np.sum(weighted[valid] * hist[globals_][clipped[valid]], axis=1)
                np.add.at(values, rows[valid], coefficients[valid] * cross)
        np.add.at(counts, context_slots, np.where(context_owners == 0, values, -values))
        first, second = self._active_filters(filters, active)
        slot = np.maximum(self.shape_world[first], self.shape_world[second]).astype(np.int64) + 1
        # Each slot's excluded pairs by their category pair, weighed once per category pair (exact in int64).
        kinds = len(weights)
        excluded = np.bincount(
            (slot * kinds + categories[first]) * kinds + categories[second],
            minlength=len(counts) * kinds * kinds,
        ).reshape(len(counts), kinds * kinds)
        counts -= excluded @ np.asarray(weights, dtype=np.int64).reshape(-1)
        return counts

    def _store_pairs(self, output: np.ndarray, a: np.ndarray, b: np.ndarray) -> int:
        """Canonicalize candidate indices and remove authored exclusions."""
        shape_a, shape_b = np.minimum(a, b), np.maximum(a, b)
        if len(self.filter_pairs) and len(shape_a):
            codes = (shape_a.astype(np.int64) << 32) | shape_b
            positions = np.searchsorted(self.filter_pairs, codes)
            excluded = (positions < len(self.filter_pairs)) & (
                self.filter_pairs[np.minimum(positions, len(self.filter_pairs) - 1)] == codes
            )
            shape_a, shape_b = shape_a[~excluded], shape_b[~excluded]
        output[: len(shape_a), 0] = shape_a
        output[: len(shape_a), 1] = shape_b
        return len(shape_a)

    def _write_dense_pairs(self, output: np.ndarray, first: np.ndarray, second: np.ndarray, *, triangular: bool) -> int:
        """Fill compatible uniform groups with cache-sized rectangular blocks."""
        written = 0
        rows_per_chunk = max(1, _PAIR_CHUNK_SIZE // len(second))
        for start in range(0, len(first), rows_per_chunk):
            rows = first[start : start + rows_per_chunk]
            columns = second[start + 1 :] if triangular else second
            for column_start in range(0, len(columns), _PAIR_CHUNK_SIZE):
                cols = columns[column_start : column_start + _PAIR_CHUNK_SIZE]
                keep = self.shape_body[rows, None] != self.shape_body[cols]
                if triangular:
                    keep &= rows[:, None] < cols
                row, column = np.nonzero(keep)
                written += self._store_pairs(output[written:], rows[row], cols[column])
        return written

    def _write_pair_ranges(
        self, output: np.ndarray, first: np.ndarray, second: np.ndarray, starts: np.ndarray, ends: np.ndarray
    ) -> int:
        """Emit ragged rows of group-compatible candidates in bounded chunks."""
        sizes = ends - starts
        offsets = np.empty(len(first) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(sizes, out=offsets[1:])
        written = 0
        for begin in range(0, int(offsets[-1]), _PAIR_CHUNK_SIZE):
            end = min(begin + _PAIR_CHUNK_SIZE, int(offsets[-1]))
            first_row = int(np.searchsorted(offsets, begin, side="right")) - 1
            last_row = int(np.searchsorted(offsets, end - 1, side="right"))
            repeats = sizes[first_row:last_row].copy()
            repeats[0] -= begin - offsets[first_row]
            repeats[-1] -= offsets[last_row] - end
            rows = np.repeat(np.arange(first_row, last_row), repeats)
            columns = starts[rows] + np.arange(begin, end) - offsets[rows]
            a, b = first[rows], second[columns]
            keep = self.shape_body[a] != self.shape_body[b]
            written += self._store_pairs(output[written:], a[keep], b[keep])
        return written

    def _write_pairs(
        self,
        output: np.ndarray,
        first: np.ndarray,
        second: np.ndarray,
        *,
        triangular: bool = False,
    ) -> int:
        """Enumerate compatible groups while skipping dominant same-body blocks."""
        if not len(first) or not len(second):
            return 0

        # A body occupying most of both sets can otherwise cause quadratic work
        # for only a linear number of output pairs. Split away that diagonal.
        # Small world templates need no partitioning overhead.
        if len(first) * len(second) > _PAIR_CHUNK_SIZE:
            bodies, counts = np.unique(self.shape_body[first], return_counts=True)
            largest = int(np.argmax(counts))
            if counts[largest] * 2 > len(first):
                same_first = self.shape_body[first] == bodies[largest]
                same_second = same_first if triangular else self.shape_body[second] == bodies[largest]
                if np.count_nonzero(same_second) * 2 > len(second):
                    other_first, other_second = first[~same_first], second[~same_second]
                    count = self._write_pairs(output, other_first, other_second, triangular=triangular)
                    count += self._write_pairs(output[count:], first[same_first], other_second)
                    if not triangular:
                        count += self._write_pairs(output[count:], other_first, second[same_second])
                    return count

        groups = self.shape_group[first]
        second_groups = groups if triangular else self.shape_group[second]
        if np.all(groups == groups[0]) and (triangular or np.all(second_groups == second_groups[0])):
            group_a, group_b = int(groups[0]), int(second_groups[0])
            compatible = (group_a == group_b or group_b < 0) if group_a > 0 else group_a != group_b
            if not compatible:
                return 0
            return self._write_dense_pairs(output, first, second, triangular=triangular)

        # Sorting by (group, body) makes both a positive group and its excluded
        # same-body runs contiguous, even when no body dominates the whole set.
        codes = (second_groups.astype(np.int64) << 32) | (self.shape_body[second].astype(np.int64) + 1)
        order = np.argsort(codes, kind="stable")
        codes = codes[order]
        second = second[order]
        second_groups = second_groups[order]
        if triangular:
            # Negative groups collide with every later group; positive groups
            # collide only within themselves. Each row is one contiguous range.
            first = second
            group_ends = np.searchsorted(second_groups, second_groups, side="right")
            body_ends = np.searchsorted(codes, codes, side="right")
            starts = np.where(second_groups < 0, group_ends, body_ends)
            ends = np.where(second_groups < 0, len(second), group_ends)
            return self._write_pair_ranges(output, first, second, starts, ends)

        # Cross-world/global pairs need at most three ranges per row: negative
        # groups accept everything except their own group; positives accept all
        # negatives and their matching positive group, excluding the same body.
        left = np.searchsorted(second_groups, groups, side="left")
        right = np.searchsorted(second_groups, groups, side="right")
        first_codes = (groups.astype(np.int64) << 32) | (self.shape_body[first].astype(np.int64) + 1)
        body_left = np.searchsorted(codes, first_codes, side="left")
        body_right = np.searchsorted(codes, first_codes, side="right")
        negative_end = int(np.searchsorted(second_groups, 0))
        count = self._write_pair_ranges(
            output, first, second, np.zeros(len(first), dtype=np.int64), np.where(groups < 0, left, negative_end)
        )
        count += self._write_pair_ranges(
            output[count:],
            first,
            second,
            np.where(groups < 0, right, left),
            np.where(groups < 0, len(second), body_left),
        )
        count += self._write_pair_ranges(output[count:], first, second, np.where(groups < 0, right, body_right), right)
        return count

    def build_pairs(self, shape_mask: np.ndarray | None = None) -> np.ndarray:
        """Build only the requested pairs, reusing homogeneous world templates.

        Building every pair (no ``shape_mask``) also keeps the per-world templates the list replays in
        :attr:`pair_templates`, which :class:`~newton.CollisionPipeline` tests per world instead of
        streaming the list.
        """
        templates = self._full_templates.record if shape_mask is None else self._templates(shape_mask).record
        if shape_mask is None:
            self.pair_templates = templates
        return templates.replay()

    @cached_property
    def _full_templates(self) -> _Templates:
        return self._templates(None)

    def sample(self, *values: np.ndarray) -> _PairSample:
        """The pairs of the global shapes and of one world of each group of the templates split by the per-shape
        ``values`` (:meth:`_WorldTemplates.refine`): what every pair-derived quantity a world's ``values`` decide
        is computed on, a group's representative standing for its other worlds."""
        from ..geometry.broad_phase_nxn import replay_shape_pair_templates  # noqa: PLC0415

        full = self._full_templates
        record = full.record
        groups = self.worlds.refine(*values)
        template = full.world_template[groups.first]
        replayed = np.flatnonzero(template >= 0)
        pairs = replay_shape_pair_templates(
            record.global_pairs, record.templates, template[replayed], full.world_origin[groups.first[replayed]]
        )
        sizes = np.asarray([g.shape[0] + l.shape[0] for g, l in record.templates], dtype=np.int64)
        pair_group = np.concatenate(
            (np.full(record.global_pairs.shape[0], -1, dtype=np.int64), np.repeat(replayed, sizes[template[replayed]]))
        )
        shift, _ = self.worlds.shift(groups)
        return _PairSample(
            pairs,
            pair_group,
            groups.sizes,
            groups.first,
            groups.group,
            shift,
            self.shape_world,
            self.worlds.global_bodies,
            self,
        )

    def _templates(self, shape_mask: np.ndarray | None) -> _Templates:
        """The pairs (within ``shape_mask``) as the global pairs and per-world templates: each group's first world's
        pairs (:meth:`_groups`), which its other worlds replay shifted by their first shape."""
        from ..geometry.broad_phase_nxn import ShapePairTemplates  # noqa: PLC0415

        groups = self._groups(shape_mask)
        counts = self.counts if shape_mask is None else self.count_pairs(shape_mask)
        worlds = self.worlds
        active = self._active(shape_mask)
        globals_ = worlds.globals_[active[worlds.globals_]]
        global_pairs = np.empty((int(counts[0]), 2), dtype=np.int32)
        if len(global_pairs):
            self._write_pairs(global_pairs, globals_, globals_, triangular=True)
        stored = []
        origins = np.zeros(groups.first.size, dtype=np.int64)
        template_of_group = np.full(groups.first.size, -1, dtype=np.int64)
        for group, world in enumerate(groups.first.tolist()):
            count = int(counts[world + 1])
            if not count:
                continue
            local = worlds.world_shapes(world)
            local = local[active[local]]
            target = np.empty((count, 2), dtype=np.int32)
            global_count = self._write_pairs(target, globals_, local)
            local_count = self._write_pairs(target[global_count:], local, local, triangular=True)
            if global_count + local_count != count:
                raise RuntimeError("Shape contact-pair count disagrees with enumeration")
            # Global-local pairs as (absolute global id, world-local id), local pairs world-locally, relative to
            # the world's first collidable shape.
            origin = int(local[0])
            mixed = target[:global_count]
            is_global = self.shape_world[mixed] == -1
            global_local = np.column_stack(
                (
                    np.where(is_global[:, 0], mixed[:, 0], mixed[:, 1]),
                    np.where(is_global[:, 0], mixed[:, 1], mixed[:, 0]) - origin,
                )
            ).astype(np.int32)
            template_of_group[group] = len(stored)
            origins[group] = origin
            stored.append((global_local, (target[global_count:] - origin).astype(np.int32)))
        shift, _ = worlds.shift(groups)
        world_template = template_of_group[groups.group]
        world_origin = origins[groups.group] + shift
        slot = world_template >= 0
        record = ShapePairTemplates(
            global_pairs=global_pairs,
            templates=tuple(stored),
            slot_template=world_template[slot].astype(np.int32),
            slot_start=world_origin[slot].astype(np.int32),
        )
        return _Templates(record, world_template, world_origin)


@dataclass(frozen=True, eq=False)
class _Templates:
    """A pair list's templates and, per world, the template it replays (``-1`` for none) and its first shape."""

    record: ShapePairTemplates
    world_template: np.ndarray
    world_origin: np.ndarray


@dataclass(frozen=True, eq=False)
class _PairSample:
    """A pair list as the global pairs and one representative world's pairs per group of worlds.

    Every quantity that a world's pairs decide through the per-shape values the groups were split by
    (:meth:`refine`) is the same for the worlds of a group, so it is computed on the representative's pairs and
    expanded: a sum by :meth:`weights`, a per-world value by :meth:`per_world`, a per-shape one by :meth:`spread`.
    A supplied pair list samples every world (each world its own group).
    """

    pairs: np.ndarray
    """The sampled pairs, shape [n, 2], int32: the global pairs, then each representative's."""
    group: np.ndarray
    """Each sampled pair's group, ``-1`` for a pair of global shapes."""
    sizes: np.ndarray
    """The number of worlds of each group."""
    first: np.ndarray
    """Each group's representative world."""
    world_group: np.ndarray
    """Each world's group."""
    world_shift: np.ndarray
    """Each world's shape offset from its group's representative."""
    shape_world: np.ndarray
    """Each shape's world."""
    global_bodies: np.ndarray
    """The bodies that global shapes are on: the same in every world, unlike a world's own bodies."""
    topology: _ShapeContactPairs | None = None
    """The topology the sample's templates come from; ``None`` for a supplied list."""

    @classmethod
    def of_list(cls, pairs: np.ndarray, shape_world: np.ndarray, world_count: int) -> _PairSample:
        """Sample a supplied pair list whole: each world is its own group, an invalid or global pair its own."""
        pairs = np.asarray(pairs, dtype=np.int32).reshape((-1, 2))
        valid = np.all((pairs >= 0) & (pairs < shape_world.size), axis=1)
        world = np.full(len(pairs), -1, dtype=np.int64)
        world[valid] = np.max(shape_world[pairs[valid]], axis=1)
        world_count = max(world_count, int(world.max(initial=-1)) + 1)
        worlds = np.arange(world_count, dtype=np.int64)
        return cls(
            pairs,
            world,
            np.ones(world_count, dtype=np.int64),
            worlds,
            worlds,
            np.zeros(world_count, dtype=np.int64),
            shape_world,
            np.zeros(0, dtype=np.int64),
        )

    def refine(self, *values: np.ndarray) -> _PairSample:
        """The same list sampled with its groups split by the per-shape ``values``."""
        return self if self.topology is None else self.topology.sample(*values)

    def weights(self) -> np.ndarray:
        """The number of the list's pairs each sampled pair stands for."""
        weights = np.ones(self.group.size, dtype=np.int64)
        grouped = self.group >= 0
        weights[grouped] = self.sizes[self.group[grouped]]
        return weights

    def pair_list(self) -> np.ndarray:
        """The whole pair list on the host, shape [n, 2], int32."""
        return self.pairs if self.topology is None else self.topology.build_pairs()

    def count(self, selected: np.ndarray) -> int:
        """The number of the list's pairs whose sampled pair is ``selected``."""
        return int(self.weights()[selected].sum())

    def per_world(self, slots: np.ndarray) -> np.ndarray:
        """Per-slot values (globals in slot zero, world w in slot w + 1) given for the global slot and the
        representatives, for every world: each world takes its representative's."""
        out = np.empty(self.world_group.size + 1, dtype=slots.dtype)
        out[0] = slots[0]
        out[1:] = slots[1 + self.first[self.world_group]]
        return out

    def spread(self, values: np.ndarray) -> np.ndarray:
        """Per-shape ``values`` given on the global shapes and the representatives' shapes, for every shape: each
        world's shapes take their representative's."""
        source = np.arange(values.size)
        local = self.shape_world >= 0
        source[local] -= self.world_shift[self.shape_world[local]]
        return values[source]

    def shapes(self) -> np.ndarray:
        """Whether each shape is in a pair of the list."""
        listed = np.zeros(self.shape_world.size, dtype=bool)
        ids = self.pairs.reshape(-1)
        listed[ids[(ids >= 0) & (ids < listed.size)]] = True
        return self.spread(listed)

    def shared_bodies(self, bodies: np.ndarray) -> np.ndarray:
        """Whether a sampled pair's ``bodies`` (shape [n, 2]) are the same in every world: no body of its world's own."""
        own = (bodies >= 0) & ~np.isin(bodies, self.global_bodies)
        return ~own.any(axis=1)

    def sorted_codes(self, selected: np.ndarray, device) -> tuple[wp.array, wp.array]:
        """The list's pairs whose sampled pair is ``selected``, as ascending packed codes ``(a << 32) | b`` and each
        one's world (``world_count`` for a pair of global shapes), on ``device``.

        A representative's pairs are sorted once and its other worlds' are written shifted by a kernel. The codes
        ascend by first shape: those of the leading global shapes (with global shapes, then every world's shapes in
        world order, then trailing global shapes), then each world's own, then those of the trailing global shapes.
        """
        pairs = self.pairs[selected].astype(np.int64)
        group = self.group[selected]
        world_count = self.world_group.size
        local = self.shape_world >= 0
        front = int(np.argmax(local)) if local.any() else self.shape_world.size
        codes = (pairs[:, 0] << 32) | pairs[:, 1]
        leading = pairs[:, 0] < front
        # Leading global shapes' codes: by first shape, the global pairs before and after every world's.
        lead_global = leading & (group < 0)
        before = lead_global & (pairs[:, 1] < front)
        lead_codes = np.concatenate(
            (
                np.sort(codes[before]),
                self._expand(codes, group, leading & (group >= 0)),
                np.sort(codes[lead_global & ~before]),
            )
        )
        lead_codes = lead_codes[np.argsort(lead_codes >> 32, kind="stable")]
        own = ~leading & (group >= 0)
        trailing = np.sort(codes[~leading & (group < 0)])
        own_order = np.lexsort((codes[own], group[own]))
        rep_codes = codes[own][own_order]
        rep_group = group[own][own_order]
        counts = np.bincount(rep_group, minlength=self.first.size)
        starts = np.cumsum(counts) - counts
        world_counts = counts[self.world_group]
        world_offset = lead_codes.size + np.cumsum(world_counts) - world_counts
        total = lead_codes.size + int(world_counts.sum()) + trailing.size
        out = wp.empty(total, dtype=wp.uint64, device=device)
        worlds = wp.empty(total, dtype=wp.int32, device=device)
        if lead_codes.size:
            # A leading global shape's code belongs to its second shape's world (``world_count`` for a global one).
            head_world = self.shape_world[lead_codes & 0xFFFFFFFF]
            head_world = np.where(head_world < 0, world_count, head_world).astype(np.int32)
            wp.copy(out, wp.array(lead_codes.view(np.uint64), dtype=wp.uint64, device=device), count=lead_codes.size)
            wp.copy(worlds, wp.array(head_world, dtype=wp.int32, device=device), count=lead_codes.size)
        if trailing.size:
            offset = total - trailing.size
            wp.copy(out, wp.array(trailing.view(np.uint64), dtype=wp.uint64, device=device), dest_offset=offset)
            wp.copy(worlds, wp.full(trailing.size, world_count, dtype=wp.int32, device=device), dest_offset=offset)
        if world_counts.size and world_counts.max(initial=0) > 0:
            # A shift moves the first shape and a local second shape.
            step = (1 << 32) + (self.shape_world[rep_codes & 0xFFFFFFFF] >= 0)
            wp.launch(
                _expand_world_codes,
                dim=(world_count, int(world_counts.max())),
                inputs=[
                    wp.array(rep_codes.view(np.uint64), dtype=wp.uint64, device=device),
                    wp.array(step.astype(np.uint64), dtype=wp.uint64, device=device),
                    wp.array(starts[self.world_group], dtype=wp.int64, device=device),
                    wp.array(world_counts, dtype=wp.int64, device=device),
                    wp.array(world_offset, dtype=wp.int64, device=device),
                    wp.array(self.world_shift, dtype=wp.int64, device=device),
                    out,
                    worlds,
                ],
                device=device,
            )
        return out, worlds

    def _expand(self, codes: np.ndarray, group: np.ndarray, selected: np.ndarray) -> np.ndarray:
        """The codes of a leading global shape with a world's shape, ``selected`` among the sampled ones, for every
        world in world order, each world's in its representative's order (shifted in their second shape)."""
        rep_codes, rep_group = codes[selected], group[selected]
        if not rep_codes.size:
            return rep_codes
        counts = np.bincount(rep_group, minlength=self.first.size)
        rep_codes = rep_codes[np.lexsort((rep_codes, rep_group))]
        starts = np.cumsum(counts) - counts
        world_counts = counts[self.world_group]
        world, rank = _segments(world_counts)
        return rep_codes[starts[self.world_group[world]] + rank] + self.world_shift[world]


def _shape_contact_pair_counts(model: Model | ModelView, *, shape_mask: np.ndarray | None = None) -> np.ndarray:
    """Cache read-only counts for globals (slot zero) and each local world.

    Default pairs use finalized topology; supplied arrays use their full contents.
    A mask selects pairs whose two shapes are selected.
    """
    if shape_mask is not None:
        shape_mask = np.asarray(shape_mask)
        if shape_mask.dtype != np.bool_ or shape_mask.shape != (model.shape_count,):
            raise ValueError("shape_mask must be a boolean array of length shape_count")
    key = None if shape_mask is None else np.packbits(shape_mask).tobytes()
    supplied, pairs = _shape_contact_pair_override(model)
    # Supplied arrays can change without invalidating summaries.
    cache = None if supplied else model._shape_contact_pair_counts
    if cache is not None and key in cache:
        return cache[key]
    if supplied:
        counts = np.zeros(model.world_count + 1, dtype=np.int64)
        if pairs is not None:
            pairs = pairs.numpy().reshape((-1, 2))
            if shape_mask is not None:
                pairs = pairs[shape_mask[pairs].all(axis=1)]
            if len(pairs):
                worlds = model.shape_world.numpy()
                counts = np.bincount(np.max(worlds[pairs], axis=1) + 1, minlength=model.world_count + 1)
    else:
        counts = _shape_contact_pair_topology(model).count_pairs(shape_mask)
    counts.setflags(write=False)
    if cache is not None:
        cache[key] = counts
    return counts


def _shape_contact_pair_topology(model: Model | ModelView) -> _ShapeContactPairs:
    """Use the finalized Model topology inherited alongside the default pair list."""
    topology = model._shape_contact_pair_data
    assert topology is not None
    return topology


def _shape_contact_pair_override(model: Model | ModelView) -> tuple[bool, wp.array[wp.vec2i] | None]:
    """Distinguish authored pair lists from a cache of model-derived pairs."""
    overrides = vars(model).get("_overrides")
    if overrides is not None:
        if "shape_contact_pairs" in overrides:
            return True, model.shape_contact_pairs
        supplied, _ = _shape_contact_pair_override(vars(model)["_parent"])
        return supplied, model.shape_contact_pairs if supplied else None
    if model._shape_contact_pair_data is None:
        return True, model.shape_contact_pairs
    return False, None


def _shape_contact_pairs_for_mask(
    model: Model | ModelView, shape_mask: np.ndarray, *, shape_pairs: wp.array[wp.vec2i] | None = None
) -> np.ndarray:
    """Get a subset without materializing the model's full explicit pair cache."""
    supplied, pairs = _shape_contact_pair_override(model)
    if shape_pairs is None and not supplied:
        return _shape_contact_pair_topology(model).build_pairs(shape_mask)
    pairs = pairs if shape_pairs is None else shape_pairs
    if pairs is None:
        return np.empty((0, 2), dtype=np.int32)
    pairs = pairs.numpy().reshape((-1, 2))
    return pairs[shape_mask[pairs[:, 0]] & shape_mask[pairs[:, 1]]]


def _shape_contact_pairs_host(model: Model | ModelView) -> np.ndarray | None:
    """The model's pairs on the host without building its device pair array, or ``None`` when it has none."""
    supplied, pairs = _shape_contact_pair_override(model)
    if supplied:
        return None if pairs is None else pairs.numpy().reshape((-1, 2))
    return _shape_contact_pair_topology(model).build_pairs()


def _shape_contact_pair_count_for_mask(
    model: Model | ModelView, shape_mask: np.ndarray, *, shape_pairs: wp.array[wp.vec2i] | None = None
) -> int:
    """Count a subset without triggering lazy pair construction."""
    if shape_pairs is None:
        return int(_shape_contact_pair_counts(model, shape_mask=shape_mask).sum())
    pairs = shape_pairs.numpy().reshape((-1, 2))
    return int(np.count_nonzero(shape_mask[pairs[:, 0]] & shape_mask[pairs[:, 1]]))


def _shape_contact_pair_sample(model: Model | ModelView, *values: np.ndarray) -> _PairSample:
    """The model's pairs as a :class:`_PairSample` whose groups agree on the per-shape ``values``: the finalized
    topology's templates, or a supplied pair list world by world."""
    supplied, pairs = _shape_contact_pair_override(model)
    if not supplied:
        return _shape_contact_pair_topology(model).sample(*values)
    pairs = np.empty((0, 2), dtype=np.int32) if pairs is None else pairs.numpy().reshape((-1, 2))
    return _PairSample.of_list(pairs, model.shape_world.numpy(), model.world_count)


def _shape_pair_list_sample(model: Model | ModelView, pairs: wp.array[wp.vec2i] | np.ndarray) -> _PairSample:
    """The pair list ``pairs`` sampled: by the model's templates when it is the model's own built list, else whole."""
    supplied, _ = _shape_contact_pair_override(model)
    if not supplied and pairs is model._shape_contact_pairs:
        return _shape_contact_pair_topology(model).sample()
    pairs = pairs.numpy() if isinstance(pairs, wp.array) else np.asarray(pairs)
    return _PairSample.of_list(pairs, model.shape_world.numpy(), model.world_count)


@wp.kernel(enable_backward=False)
def _expand_world_codes(
    codes: wp.array[wp.uint64],
    step: wp.array[wp.uint64],
    world_start: wp.array[wp.int64],
    world_count: wp.array[wp.int64],
    world_offset: wp.array[wp.int64],
    world_shift: wp.array[wp.int64],
    out: wp.array[wp.uint64],
    out_world: wp.array[wp.int32],
):
    world, rank = wp.tid()
    if wp.int64(rank) >= world_count[world]:
        return
    source = world_start[world] + wp.int64(rank)
    target = world_offset[world] + wp.int64(rank)
    out[target] = codes[source] + wp.uint64(world_shift[world]) * step[source]
    out_world[target] = wp.int32(world)
