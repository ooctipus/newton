# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Distinct values and equal-row ids on the host, by sorting.

Model construction groups large integer and position arrays (welding a terrain mesh's vertices, deduplicating its
edges, classing worlds). :func:`numpy.unique` is slow on both kinds:

- without ``return_index``, ``return_inverse`` or ``return_counts``, NumPy 2.3 and later use a hash table, which
  collides on structured keys such as ``a * n + b`` (3.1 s against 0.09 s by sorting for 3.6 million edge keys);
- rows (``axis=0``, or a ``void`` view of each row) sort by a generic comparison per element pair.

Each function here returns exactly what the :func:`numpy.unique` call it names returns, from integer sorts.
"""

from __future__ import annotations

import numpy as np

__all__ = ["distinct", "row_ids"]


def distinct(values) -> np.ndarray:
    """The sorted distinct values of ``values`` (flattened): ``np.unique(values)``, by sorting.

    Args:
        values: An integer array.

    Returns:
        The distinct values, ascending.
    """
    ordered = np.sort(np.asarray(values).reshape(-1))
    if ordered.size == 0:
        return ordered
    keep = np.empty(ordered.size, dtype=bool)
    keep[0] = True
    np.not_equal(ordered[1:], ordered[:-1], out=keep[1:])
    return ordered[keep]


def row_ids(rows, *, by_bytes: bool = False, return_index: bool = False):
    """Each row's id among the distinct rows of ``rows``, numbered in the distinct rows' sorted order.

    By value (the default) the rows order as ``np.unique(rows, axis=0)`` orders them, lexicographically by element
    value: integer rows. ``by_bytes`` orders them as ``np.unique`` orders a ``void`` view of each row (its bytes,
    first to last), the order a bit-exact weld of float positions has used; elements of 4 or 8 bytes.

    Args:
        rows: A 2D array, one row an item.
        by_bytes: Order and compare the rows' bytes instead of their values.
        return_index: Also return each distinct row's first occurrence (``return_index=True``).

    Returns:
        ``ids`` (int64, one a row: the ``return_inverse`` of the matching ``np.unique`` call), and with
        ``return_index`` the first occurrences: ``(first, ids)``.
    """
    rows = np.ascontiguousarray(rows)
    if rows.ndim != 2:
        raise ValueError("row_ids takes a 2D array")
    count, width = rows.shape
    if by_bytes:
        if rows.itemsize not in (4, 8):
            raise ValueError("row_ids orders bytes of 4- or 8-byte elements")
        # Little-endian elements byteswapped: unsigned order is the bytes' order.
        columns = rows.view(np.uint32 if rows.itemsize == 4 else np.uint64).byteswap()
    else:
        if rows.dtype.kind not in "iub":
            raise ValueError("row_ids orders integer rows by value")
        columns = rows
    if width == 0 or count == 0:
        ids = np.zeros(count, dtype=np.int64)
        return (np.zeros(min(count, 1), dtype=np.int64), ids) if return_index else ids
    # Stable, the first key last: equal rows keep their order, so a group's first member is its first occurrence.
    order = np.lexsort(tuple(columns[:, k] for k in range(width - 1, -1, -1)))
    ordered = columns[order]
    new = np.empty(count, dtype=bool)
    new[0] = True
    np.any(ordered[1:] != ordered[:-1], axis=1, out=new[1:])
    ids = np.empty(count, dtype=np.int64)
    ids[order] = np.cumsum(new) - 1
    return (order[new], ids) if return_index else ids
