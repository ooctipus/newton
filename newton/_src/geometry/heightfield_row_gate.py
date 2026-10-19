# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Live triangle-stream lengths for the rows of the typed heightfield query table.

Every typed terrain row scans the whole compacted triangle stream and keeps the
triangles whose current shape pair it serves. The gate reads the current pair
list once per collision pass and hands each row the stream length when at least
one current pair can reach the row, and zero otherwise, so a row without pairs
skips its scan. It evaluates the rows' own predicates on the current shape types
and dimensions, so a type or dimension edit between passes moves the work to
the row that serves it, as before.
"""

import warp as wp

from .contact_reduction_global import heightfield_typed_primitive
from .types import GeoType

HEIGHTFIELD_TYPED_ROWS = (int(GeoType.SPHERE), int(GeoType.CYLINDER), int(GeoType.CAPSULE), int(GeoType.BOX))
"""Shape types of the typed terrain rows in launch order; the generic fallback row follows them."""

HEIGHTFIELD_ROW_COUNT = len(HEIGHTFIELD_TYPED_ROWS) + 1
"""Typed rows plus the generic fallback row."""


@wp.func
def heightfield_row(type_a: int, type_b: int, data_b: wp.vec4, exclude_convex: int) -> int:
    """Return the row that serves a triangle pair of these shapes, or -1 for the terrain convex-cells owner.

    Mirrors the filters of the typed and fallback kernels made by
    ``create_mesh_triangle_contacts_to_reducer_kernel``: a typed row takes its
    type on a heightfield (a cylinder only with a zero barrel radius); the
    fallback takes every other pair unless the convex-cells owner serves
    heightfield/convex-hull pairs.
    """
    if type_a == GeoType.HFIELD:
        # The row order is HEIGHTFIELD_TYPED_ROWS.
        if heightfield_typed_primitive(type_b, data_b):
            if type_b == GeoType.SPHERE:
                return 0
            if type_b == GeoType.CYLINDER:
                return 1
            if type_b == GeoType.CAPSULE:
                return 2
            return 3
        if exclude_convex != 0 and type_b == GeoType.CONVEX_MESH:
            return -1
    return wp.static(len(HEIGHTFIELD_TYPED_ROWS))


@wp.kernel(enable_backward=False)
def heightfield_row_gate_kernel(
    pairs: wp.array[wp.vec2i],
    pair_count: wp.array[int],
    shape_types: wp.array[int],
    shape_data: wp.array[wp.vec4],
    triangle_pairs_count: wp.array[int],
    exclude_convex: int,
    total_num_threads: int,
    row_counts: wp.array[int],
):
    """Store the triangle-stream length for every row that a current pair reaches.

    ``row_counts`` must be zero on entry; rows no pair reaches keep zero.
    Concurrent writers of one row store the same value.
    """
    count = triangle_pairs_count[0]
    for i in range(wp.tid(), wp.min(pair_count[0], pairs.shape[0]), total_num_threads):
        pair = pairs[i]
        row = heightfield_row(shape_types[pair[0]], shape_types[pair[1]], shape_data[pair[1]], exclude_convex)
        if row >= 0:
            if row_counts[row] == 0:
                row_counts[row] = count
