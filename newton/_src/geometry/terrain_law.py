# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared float32 primitives of the per-pair terrain contact law: the row layout and the per-pair finish.

The law emits the support set of one (heightfield, convex shape) pair directly: one row per local minimum of the
core-to-surface distance within the writer's reach, plus the corners of flat support regions, each owned by one
surface feature (a plane cluster, a crease edge or a vertex). This module holds what every per-type kernel shares.
Nothing routes contacts through the law yet.

Rows carry a window-local feature id in the 23-bit ``sort_sub_key`` of the contact contract (19 bits:
``r mod 64``, ``c mod 64``, kind, sub-feature, point slot), unique within any window under 64 cells across.

The per-pair finish (:func:`terrain_finish`) merges coincident rows, drops the rows the others of their plane
group imply (the lower convex hull of tangent position and distance, residual ``1e-6 m``), keeps at most four rows
per plane group (deepest, farthest, largest triangle, largest quadrilateral) and at most ``row_limit`` rows per
pair, each group's deepest row first and then its extremes; an overflow sets a sticky bit of the narrow phase's
buffer status word, which ``verify_narrow_phase_buffers`` keeps.
"""

from __future__ import annotations

import warp as wp

# ----------------------------------------------------------------------------------------------------------------
# tolerances and layout
# ----------------------------------------------------------------------------------------------------------------

TERRAIN_MERGE_SCALE = 4.0
"""Rows closer than this many predicate bands (with normals within :data:`TERRAIN_MERGE_NORMAL`) coincide."""
TERRAIN_MERGE_NORMAL = 1.0e-3
"""Largest normal difference of coincident rows."""
TERRAIN_GROUP_NORMAL = 1.0e-5
"""Largest normal difference of rows on one plane (one plane group)."""
TERRAIN_IMPLIED_RESIDUAL = 1.0e-6
"""A row no deeper than this below the lower hull of its plane group's other rows is implied [m]."""
TERRAIN_BARYCENTRIC_TOLERANCE = 1.0e-5
"""Relative tolerance of the tangent-position containment of implied rows."""
TERRAIN_GROUP_ROWS = 4
"""Rows kept per plane group."""
TERRAIN_ROW_CANDIDATES = 32
"""Candidate rows one pair's finish holds."""

TERRAIN_EDGE_X = wp.constant(0)
"""Edge ``(r, c)-(r, c + 1)``, between cell ``(r - 1, c)`` sub 1 and cell ``(r, c)`` sub 0."""
TERRAIN_EDGE_Y = wp.constant(1)
"""Edge ``(r, c)-(r + 1, c)``, between cell ``(r, c - 1)`` sub 0 and cell ``(r, c)`` sub 1."""
TERRAIN_EDGE_DIAGONAL = wp.constant(2)
"""Edge ``(r, c)-(r + 1, c + 1)``, between the two triangles of cell ``(r, c)``."""

TERRAIN_FEATURE_FACE = wp.constant(0)
TERRAIN_FEATURE_EDGE = wp.constant(1)
TERRAIN_FEATURE_VERTEX = wp.constant(2)
TERRAIN_FEATURE_RECOVERY = wp.constant(3)
TERRAIN_FEATURE_BITS = 19
"""Bits of a feature id: ``r mod 64`` (6), ``c mod 64`` (6), kind (2), sub-feature (2), point slot (3)."""

TERRAIN_FINISH_CANDIDATE_OVERFLOW = wp.constant(1)
"""Finish flag: the pair produced more candidates than :data:`TERRAIN_ROW_CANDIDATES`."""
TERRAIN_FINISH_ROW_OVERFLOW = wp.constant(2)
"""Finish flag: more rows survived than the pair's row limit."""
TERRAIN_STATUS_ROWS = wp.constant(1 << 13)
"""Sticky bit of the narrow phase's buffer status word: a terrain pair overflowed its candidates or rows."""
TERRAIN_STATUS_STALE = wp.constant(1 << 14)
"""Sticky bit of the narrow phase's buffer status word: terrain features older than the heights were read."""

TerrainRowsVec3 = wp.types.matrix(shape=(TERRAIN_ROW_CANDIDATES, 3), dtype=wp.float32)
TerrainRowsFloat = wp.types.vector(length=TERRAIN_ROW_CANDIDATES, dtype=wp.float32)
TerrainRowsInt = wp.types.vector(length=TERRAIN_ROW_CANDIDATES, dtype=wp.int32)

_CANDIDATES = wp.constant(TERRAIN_ROW_CANDIDATES)
_MERGE_SCALE = wp.constant(TERRAIN_MERGE_SCALE)
_MERGE_NORMAL = wp.constant(TERRAIN_MERGE_NORMAL)
_GROUP_NORMAL = wp.constant(TERRAIN_GROUP_NORMAL)
_IMPLIED_RESIDUAL = wp.constant(TERRAIN_IMPLIED_RESIDUAL)
_BARYCENTRIC_TOLERANCE = wp.constant(TERRAIN_BARYCENTRIC_TOLERANCE)
_GROUP_ROWS = wp.constant(TERRAIN_GROUP_ROWS)


# ----------------------------------------------------------------------------------------------------------------
# candidate rows
# ----------------------------------------------------------------------------------------------------------------


@wp.struct
class TerrainRows:
    """Candidate rows of one pair, in feature order (window frame)."""

    count: int
    """Rows held."""
    dropped: int
    """Rows refused because the candidates were full."""
    q: TerrainRowsVec3
    """Terrain points [m]."""
    k: TerrainRowsVec3
    """Core points [m]."""
    n: TerrainRowsVec3
    """Unit normals from the terrain to the shape."""
    d: TerrainRowsFloat
    """Core distances ``n . (k - q)`` [m]."""
    feature: TerrainRowsInt
    """Feature ids (:func:`terrain_feature_id`)."""


# ----------------------------------------------------------------------------------------------------------------
# coincident rows
# ----------------------------------------------------------------------------------------------------------------


@wp.func
def terrain_rows_coincide(q_a: wp.vec3, n_a: wp.vec3, q_b: wp.vec3, n_b: wp.vec3, eps: float) -> bool:
    """Return whether two rows are one constraint: terrain points within ``4 eps``, normals within ``1e-3``."""
    return wp.length(q_a - q_b) <= _MERGE_SCALE * eps and wp.length(n_a - n_b) <= _MERGE_NORMAL


# ----------------------------------------------------------------------------------------------------------------
# sticky status
# ----------------------------------------------------------------------------------------------------------------


@wp.func
def terrain_report(status: wp.array[wp.int32], overflow: wp.array[wp.int32], flags: int, current: bool):
    """Record a pair's finish flags and staleness in the narrow phase's sticky status word and counters.

    ``overflow[0]`` counts pairs whose rows overflowed, ``overflow[1]`` pairs whose candidates did.
    """
    if (flags & TERRAIN_FINISH_ROW_OVERFLOW) != 0:
        wp.atomic_or(status, 0, TERRAIN_STATUS_ROWS)
        wp.atomic_add(overflow, 0, 1)
    if (flags & TERRAIN_FINISH_CANDIDATE_OVERFLOW) != 0:
        wp.atomic_or(status, 0, TERRAIN_STATUS_ROWS)
        wp.atomic_add(overflow, 1, 1)
    if not current:
        wp.atomic_or(status, 0, TERRAIN_STATUS_STALE)


# ----------------------------------------------------------------------------------------------------------------
# feature ids
# ----------------------------------------------------------------------------------------------------------------


@wp.func
def terrain_feature_id(r: int, c: int, kind: int, sub: int, slot: int) -> int:
    """Return the 19-bit window-local feature id of a row (the contact's ``sort_sub_key``).

    Bits 18-13 hold ``r mod 64``, 12-7 ``c mod 64``, 6-5 the kind (face, edge, vertex, recovery), 4-3 the
    sub-feature (triangle 0/1, or edge x/y/diagonal) and 2-0 the point slot.
    """
    return ((((r & 63) * 64 + (c & 63)) * 4 + (kind & 3)) * 4 + (sub & 3)) * 8 + (slot & 7)


@wp.func
def terrain_feature_parts(feature: int) -> wp.vec4i:
    """Return ``(r mod 64, c mod 64, kind, sub-feature)`` of a feature id (the slot is ``feature & 7``)."""
    return wp.vec4i((feature >> 13) & 63, (feature >> 7) & 63, (feature >> 5) & 3, (feature >> 3) & 3)


# ----------------------------------------------------------------------------------------------------------------
# per-pair finish
# ----------------------------------------------------------------------------------------------------------------


@wp.func
def terrain_rows_push(rows: TerrainRows, q: wp.vec3, k: wp.vec3, n: wp.vec3, d: float, feature: int) -> TerrainRows:
    """Append a candidate row (counted as dropped when the candidates are full)."""
    i = rows.count
    if i < _CANDIDATES:
        for j in range(3):
            rows.q[i, j] = q[j]
            rows.k[i, j] = k[j]
            rows.n[i, j] = n[j]
        rows.d[i] = d
        rows.feature[i] = feature
        rows.count = i + 1
    else:
        rows.dropped = rows.dropped + 1
    return rows


@wp.func
def terrain_row_q(rows: TerrainRows, i: int) -> wp.vec3:
    """Return candidate ``i``'s terrain point."""
    return wp.vec3(rows.q[i, 0], rows.q[i, 1], rows.q[i, 2])


@wp.func
def terrain_row_k(rows: TerrainRows, i: int) -> wp.vec3:
    """Return candidate ``i``'s core point."""
    return wp.vec3(rows.k[i, 0], rows.k[i, 1], rows.k[i, 2])


@wp.func
def terrain_row_n(rows: TerrainRows, i: int) -> wp.vec3:
    """Return candidate ``i``'s normal."""
    return wp.vec3(rows.n[i, 0], rows.n[i, 1], rows.n[i, 2])


@wp.func
def _tangent_frame(n: wp.vec3) -> tuple[wp.vec3, wp.vec3]:
    reference = wp.vec3(1.0, 0.0, 0.0)
    if wp.abs(n[0]) >= 0.9:
        reference = wp.vec3(0.0, 1.0, 0.0)
    e1 = wp.normalize(reference - wp.dot(reference, n) * n)
    return e1, wp.cross(n, e1)


@wp.func
def _cross2(a: wp.vec2, b: wp.vec2) -> float:
    return a[0] * b[1] - a[1] * b[0]


@wp.func
def _lower_hull_value(
    u: TerrainRowsVec3, members: TerrainRowsInt, count: int, skip: int, point: wp.vec2, eps: float
) -> tuple[bool, float]:
    """Lowest convex interpolation of the members' distances (``u[i, 2]``) at a tangent position, ``skip`` excluded.

    ``u[i]`` holds a member's tangent coordinates and distance. Positions within ``4 eps`` coincide; a position
    on a segment or in a triangle of other members interpolates their distances.
    """
    found = bool(False)
    best = float(0.0)
    tolerance = _MERGE_SCALE * eps
    for ai in range(count):
        if ai == skip:
            continue
        a = members[ai]
        pa = wp.vec2(u[a, 0], u[a, 1])
        if wp.length(point - pa) <= tolerance:
            if not found or u[a, 2] < best:
                best = u[a, 2]
                found = True
        for bi in range(ai + 1, count):
            if bi == skip:
                continue
            b = members[bi]
            pb = wp.vec2(u[b, 0], u[b, 1])
            ab = pb - pa
            length2 = wp.dot(ab, ab)
            if length2 > 0.0:
                s = wp.dot(point - pa, ab) / length2
                perpendicular = wp.abs(_cross2(ab, point - pa)) / wp.sqrt(length2)
                if s >= -_BARYCENTRIC_TOLERANCE and s <= 1.0 + _BARYCENTRIC_TOLERANCE and perpendicular <= tolerance:
                    value = u[a, 2] + s * (u[b, 2] - u[a, 2])
                    if not found or value < best:
                        best = value
                        found = True
            for ci in range(bi + 1, count):
                if ci == skip:
                    continue
                cc = members[ci]
                pc = wp.vec2(u[cc, 0], u[cc, 1])
                det = _cross2(pb - pa, pc - pa)
                scale = wp.length(pb - pa) * wp.length(pc - pa)
                if wp.abs(det) > _BARYCENTRIC_TOLERANCE * scale and scale > 0.0:
                    beta = _cross2(point - pa, pc - pa) / det
                    gamma = _cross2(pb - pa, point - pa) / det
                    if (
                        beta >= -_BARYCENTRIC_TOLERANCE
                        and gamma >= -_BARYCENTRIC_TOLERANCE
                        and beta + gamma <= 1.0 + _BARYCENTRIC_TOLERANCE
                    ):
                        value = u[a, 2] + beta * (u[b, 2] - u[a, 2]) + gamma * (u[cc, 2] - u[a, 2])
                        if not found or value < best:
                            best = value
                            found = True
    return found, best


@wp.func
def _group_members(
    alive: TerrainRowsInt, group: TerrainRowsInt, count: int, representative: int
) -> tuple[TerrainRowsInt, int]:
    members = TerrainRowsInt()
    size = int(0)
    for i in range(count):
        if alive[i] != 0 and group[i] == representative:
            members[size] = i
            size += 1
    return members, size


@wp.func
def terrain_finish(rows: TerrainRows, eps: float, row_limit: int) -> tuple[TerrainRowsInt, int, int]:
    """Select one pair's rows: merge, implied-row removal, four rows per plane group, ``row_limit`` per pair.

    Returns ``(order, kept, flags)``: the kept candidates' indices in candidate order, their count, and
    :data:`TERRAIN_FINISH_CANDIDATE_OVERFLOW` / :data:`TERRAIN_FINISH_ROW_OVERFLOW` bits. Ties are broken by
    candidate (feature) order, never by memory order.
    """
    count = wp.min(rows.count, _CANDIDATES)
    flags = int(0)
    if rows.dropped > 0:
        flags = flags | TERRAIN_FINISH_CANDIDATE_OVERFLOW
    alive = TerrainRowsInt()
    group = TerrainRowsInt()
    rank = TerrainRowsInt()
    # 1. coincident rows: the first in feature order stays
    for i in range(count):
        alive[i] = 1
        qi = terrain_row_q(rows, i)
        ni = terrain_row_n(rows, i)
        for j in range(i):
            if alive[j] != 0 and terrain_rows_coincide(terrain_row_q(rows, j), terrain_row_n(rows, j), qi, ni, eps):
                alive[i] = 0
                break
    # 2. plane groups: the first row of a plane represents it
    for i in range(count):
        group[i] = -1
        if alive[i] == 0:
            continue
        qi = terrain_row_q(rows, i)
        ni = terrain_row_n(rows, i)
        group[i] = i
        for j in range(i):
            if alive[j] != 0 and group[j] == j:
                nj = terrain_row_n(rows, j)
                if wp.length(ni - nj) <= _GROUP_NORMAL and wp.abs(wp.dot(nj, qi - terrain_row_q(rows, j))) <= (
                    _MERGE_SCALE * eps
                ):
                    group[i] = j
                    break
    # 3. implied rows, the shallowest first, one at a time
    u = TerrainRowsVec3()
    for g in range(count):
        if alive[g] == 0 or group[g] != g:
            continue
        e1, e2 = _tangent_frame(terrain_row_n(rows, g))
        changed = bool(True)
        while changed:
            changed = False
            members, size = _group_members(alive, group, count, g)
            if size < 3:
                break
            for mi in range(size):
                i = members[mi]
                ki = terrain_row_k(rows, i)
                u[i, 0] = wp.dot(ki, e1)
                u[i, 1] = wp.dot(ki, e2)
                u[i, 2] = rows.d[i]
            tested = TerrainRowsInt()
            for _pass in range(size):
                pick = int(-1)
                for mi in range(size):
                    if tested[mi] == 0:
                        if pick < 0 or rows.d[members[mi]] > rows.d[members[pick]]:
                            pick = mi
                tested[pick] = 1
                j = members[pick]
                found, value = _lower_hull_value(u, members, size, pick, wp.vec2(u[j, 0], u[j, 1]), eps)
                if found and rows.d[j] >= value - _IMPLIED_RESIDUAL:
                    alive[j] = 0
                    changed = True
                    break
    # 4. at most four rows per group: deepest, farthest, largest triangle, largest quadrilateral
    for i in range(count):
        rank[i] = 4
    total = int(0)
    for g in range(count):
        # A group keeps its first row's id even when that row was implied.
        if group[g] != g:
            continue
        members, size = _group_members(alive, group, count, g)
        if size == 0:
            continue
        n = terrain_row_n(rows, g)
        i0 = int(-1)
        for mi in range(size):
            if i0 < 0 or rows.d[members[mi]] < rows.d[i0]:
                i0 = members[mi]
        p0 = terrain_row_k(rows, i0)
        p0 = p0 - wp.dot(p0, n) * n
        rank[i0] = 0
        total += 1
        i1 = int(-1)
        far = float(-1.0)
        for mi in range(size):
            j = members[mi]
            if rank[j] == 0:
                continue
            pj = terrain_row_k(rows, j)
            distance = wp.length(pj - wp.dot(pj, n) * n - p0)
            if distance > far:
                far = distance
                i1 = j
        if i1 < 0:
            continue
        rank[i1] = 1
        total += 1
        p1 = terrain_row_k(rows, i1)
        p1 = p1 - wp.dot(p1, n) * n
        i2 = int(-1)
        largest = float(-1.0)
        for mi in range(size):
            j = members[mi]
            if rank[j] <= 1:
                continue
            pj = terrain_row_k(rows, j)
            area = wp.abs(wp.dot(wp.cross(p1 - p0, pj - wp.dot(pj, n) * n - p0), n))
            if area > largest:
                largest = area
                i2 = j
        if i2 < 0:
            continue
        rank[i2] = 2
        total += 1
        p2 = terrain_row_k(rows, i2)
        p2 = p2 - wp.dot(p2, n) * n
        i3 = int(-1)
        gain_best = float(1.0e-15)
        for mi in range(size):
            j = members[mi]
            if rank[j] <= 2:
                continue
            pj = terrain_row_k(rows, j)
            pj = pj - wp.dot(pj, n) * n
            gain = float(0.0)
            for side in range(3):
                pa = p0
                pb = p1
                pc = p2
                if side == 1:
                    pa = p1
                    pb = p2
                    pc = p0
                elif side == 2:
                    pa = p2
                    pb = p0
                    pc = p1
                side_c = wp.dot(wp.cross(pb - pa, pc - pa), n)
                side_j = wp.dot(wp.cross(pb - pa, pj - pa), n)
                if side_c * side_j < 0.0:
                    gain = wp.max(gain, wp.abs(side_j))
            if gain > gain_best:
                gain_best = gain
                i3 = j
        if i3 >= 0:
            rank[i3] = 3
            total += 1
        if size <= _GROUP_ROWS:
            # A group of at most four rows keeps them all.
            for mi in range(size):
                j = members[mi]
                if rank[j] == _GROUP_ROWS:
                    rank[j] = _GROUP_ROWS - 1
                    total += 1
    # 5. at most row_limit rows: the groups' deepest rows first, then their extremes
    keep = TerrainRowsInt()
    if total <= row_limit:
        for i in range(count):
            if alive[i] != 0 and rank[i] < _GROUP_ROWS:
                keep[i] = 1
    else:
        flags = flags | TERRAIN_FINISH_ROW_OVERFLOW
        taken = int(0)
        for level in range(_GROUP_ROWS):
            while taken < row_limit:
                pick = int(-1)
                for i in range(count):
                    if alive[i] != 0 and rank[i] == level and keep[i] == 0:
                        if pick < 0 or rows.d[i] < rows.d[pick]:
                            pick = i
                if pick < 0:
                    break
                keep[pick] = 1
                taken += 1
    order = TerrainRowsInt()
    kept = int(0)
    for i in range(count):
        if keep[i] != 0:
            order[kept] = i
            kept += 1
    return order, kept, flags
