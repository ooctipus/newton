# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Host index toolkit: the ragged-array operations FeatherPGS's construction builds its tables from.

A ragged array is a flat list of elements cut into consecutive segments, given
by the segments' lengths (``counts``) or by their CSR bounds (``offsets``:
segment ``s`` spans ``[offsets[s], offsets[s + 1])``). Each operation has one
implementation here, vectorized with NumPy and exact in integers:

- :func:`offsets`: the CSR bounds of segments of given lengths;
- :func:`expand`: each element's segment and rank within it;
- :func:`segmented_rank`: an exclusive scan restarted at each segment;
- :func:`group_by`: the elements grouped by an integer key, as CSR;
- :func:`ordered_unique`: distinct values in the order they first occur;
- :func:`lookup`: the segment that holds each item.

Results are int64 unless a ``dtype`` is given.
"""

from __future__ import annotations

import numpy as np

__all__ = ["expand", "group_by", "lookup", "offsets", "ordered_unique", "segmented_rank"]


def offsets(counts, dtype=np.int64) -> np.ndarray:
    """Return the CSR bounds of consecutive segments of lengths ``counts``.

    Args:
        counts: Segment lengths, one per segment.
        dtype: Integer type in which the bounds accumulate.

    Returns:
        ``len(counts) + 1`` bounds: 0, then the running totals of ``counts``.
    """
    counts = np.asarray(counts)
    bounds = np.zeros(counts.size + 1, dtype=dtype)
    np.cumsum(counts, dtype=dtype, out=bounds[1:])
    return bounds


def expand(counts) -> tuple[np.ndarray, np.ndarray]:
    """Enumerate the elements of consecutive segments of non-negative lengths ``counts``.

    Element ``k`` of segment ``s`` is ``(s, k)``.

    Returns:
        ``(segment, rank)``, each of length ``counts.sum()``, in segment order.
    """
    counts = np.asarray(counts, dtype=np.int64)
    segment = np.repeat(np.arange(counts.size), counts)
    rank = np.arange(segment.size) - np.repeat(np.cumsum(counts) - counts, counts)
    return segment, rank


def segmented_rank(segment, weight=None) -> np.ndarray:
    """Return, for elements grouped by segment, the total ``weight`` of the earlier elements of their segment.

    Args:
        segment: Each element's segment; the elements of a segment are consecutive.
        weight: Integer weight per element; ``None`` counts elements (the rank within the segment).

    Returns:
        The exclusive scan of ``weight`` restarted at every segment, one value per element.
    """
    segment = np.asarray(segment)
    weight = np.ones(segment.size, dtype=np.int64) if weight is None else np.asarray(weight, dtype=np.int64)
    before = np.cumsum(weight) - weight
    if segment.size == 0:
        return before
    first = np.empty(segment.size, dtype=bool)
    first[0] = True
    np.not_equal(segment[1:], segment[:-1], out=first[1:])
    head = np.maximum.accumulate(np.where(first, np.arange(segment.size), 0))
    return before - before[head]


def group_by(keys, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Group elements by an integer key, as CSR.

    Args:
        keys: Each element's key, non-negative.
        count: The number of keys, at least; keys at or past it get groups of their own.

    Returns:
        ``(bounds, order)``: ``order`` lists the elements by key, each key's in their
        original order, and key ``k``'s elements are ``order[bounds[k]:bounds[k + 1]]``.
    """
    keys = np.asarray(keys)
    return offsets(np.bincount(keys, minlength=count)), np.argsort(keys, kind="stable")


def ordered_unique(values) -> np.ndarray:
    """Return the distinct ``values`` in the order they first occur."""
    values = np.asarray(values)
    _, first = np.unique(values, return_index=True)
    return values[np.sort(first)]


def lookup(bounds, items) -> np.ndarray:
    """Return the segment holding each item: ``s`` with ``bounds[s] <= item < bounds[s + 1]``, or -1 for none.

    Args:
        bounds: Non-decreasing CSR bounds of the segments.
        items: Positions to look up.
    """
    bounds = np.asarray(bounds)
    items = np.asarray(items)
    segment = np.searchsorted(bounds, items, side="right") - 1
    inside = (segment >= 0) & (segment < bounds.size - 1)
    held = np.where(inside, segment, 0)
    inside &= (bounds[held] <= items) & (items < bounds[held + 1])
    return np.where(inside, segment, -1)
