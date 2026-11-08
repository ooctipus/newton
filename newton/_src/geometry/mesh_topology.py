# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Welded surface topology of triangle meshes, for exact mesh contacts.

A mesh's contacts with a convex shape are the local minima of the distance
between the shape and the mesh surface. Each minimum lies on one feature of the
surface (a face, an edge or a vertex) and its direction lies in that feature's
normal cone: the triangle queried alone cannot tell, because the cone of a
boundary edge or vertex of one triangle depends on the triangles around it.
This module builds, once per mesh, the topology the rule needs:

- vertices welded by bit-equal positions (``-0.0`` and ``0.0`` weld);
- for each triangle edge, the triangle across it and that triangle's
  opposite vertex, or none for a boundary edge; an edge met by more than two
  triangles, or by two triangles with the same opposite vertex, is flagged
  non-manifold and treated as a boundary;
- for each welded vertex, its incident triangles in ascending order (its
  one-ring, and its owner: the first);
- each triangle's mesh-local unit normal, which of its edges join a coplanar
  triangle (the surface is one plane across them), and for each corner the
  lowest coplanar triangle there (the owner of a face contact at the corner).

Near-equal positions are not welded and T-junctions are not split: such seams
are boundaries, where every triangle keeps its own contact (the triangle soup's
behaviour), so they can duplicate a contact but never lose one.

The arrays live on the device in mesh-local vertex indices; a moving rigid mesh
keeps its topology. Edge ``k`` of a triangle is the edge opposite its corner
``k`` in the mesh's index order.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from .contact_data import ContactData
from .heightfield_features import (
    HeightfieldFeatureContext,
    faces_witness,
    seam_surface_witness,
)
from .support_function import (
    GenericShapeData,
    GeoTypeEx,
    closest_point_on_triangle,
    create_shape_support_function,
    create_triangle_prism_penetration_refiner,
)
from .types import GeoType

ROW = 13
"""Ints per triangle row: neighbour (3), opposite vertex (3), welded vertex (3), face-corner owner (3), flat edges."""

COPLANAR_COSINE = 1.0 - 1.0e-6
"""Cosine [-] above which two face normals are one plane: their contacts at a shared edge or vertex coincide."""


@wp.struct
class MeshTopologyData:
    """Device topology of every meshed shape, concatenated over meshes (see :class:`MeshTopology`)."""

    topology: wp.array[wp.int32]
    """Triangle rows of ``ROW`` ints, then each mesh's welded-vertex starts, then their incident triangles."""
    normal: wp.array[wp.vec3]
    """Per triangle row: the mesh-local unit normal (zero for a degenerate triangle)."""
    shape_base: wp.array[wp.vec2i]
    """Per shape: ``(triangle row base, index of its welded-vertex starts in topology)``, or ``(-1, -1)``."""
    shape_radius: wp.array[float]
    """The model's per-shape bounding radius about the shape origin [m] (the model's array itself)."""


def weld_mesh(vertices: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, int]:
    """Weld a vertex soup by bit-equal positions.

    Args:
        vertices: ``(n, 3)`` float32 mesh-local positions, as the device mesh stores them.
        indices: ``(m, 3)`` triangle corner indices.

    Returns:
        ``(welded, count)``: the welded id of each triangle corner and the number of welded vertices.
    """
    points = np.ascontiguousarray(np.asarray(vertices, dtype=np.float32).reshape(-1, 3) + np.float32(0.0))
    keys = points.view(np.dtype((np.void, 12))).ravel()
    _, inverse = np.unique(keys, return_inverse=True)
    inverse = inverse.reshape(-1).astype(np.int64)
    count = int(inverse.max()) + 1 if inverse.size else 0
    return inverse[np.asarray(indices, dtype=np.int64).reshape(-1, 3)], count


def _corner_owners(welded: np.ndarray, normals: np.ndarray, start: np.ndarray, incident: np.ndarray) -> np.ndarray:
    """For each triangle corner, the lowest incident triangle coplanar with the triangle (itself at least)."""
    m = welded.shape[0]
    owners = np.repeat(np.arange(m, dtype=np.int64)[:, None], 3, axis=1)
    valence = np.diff(start)
    for size in np.unique(valence):
        if size < 2:
            continue
        vertices = np.flatnonzero(valence == size)
        group = incident[start[vertices][:, None] + np.arange(size)[None, :]]
        dots = np.einsum("vik,vjk->vij", normals[group], normals[group])
        # The first coplanar member (ascending ids) owns each member's face contact at the vertex.
        first = np.argmax(dots >= COPLANAR_COSINE, axis=2)
        owner = np.take_along_axis(group, first, axis=1)
        tri = group.reshape(-1)
        corner_vertex = np.repeat(vertices, size)
        corner = np.argmax(welded[tri] == corner_vertex[:, None], axis=1)
        owners[tri, corner] = owner.reshape(-1)
    return owners


def build_mesh_topology(vertices: np.ndarray, indices: np.ndarray) -> dict[str, np.ndarray]:
    """Build one mesh's welded topology on the host, in O(m log m) for ``m`` triangles.

    Args:
        vertices: ``(n, 3)`` mesh-local vertex positions.
        indices: ``(m, 3)`` or flat triangle corner indices.

    Returns:
        A dict: ``neighbor``, ``opposite``, ``welded``, ``owner`` (int, ``(m, 3)``), ``flat`` (edge bits, ``(m,)``),
        ``normal`` (float32, ``(m, 3)``), ``vertex_start`` (local rows, ``(w + 1,)``), ``vertex_triangles``
        (``(3 m,)``), and the counts ``boundary_edges`` and ``nonmanifold_edges``.
    """
    tri = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
    points = np.asarray(vertices, dtype=np.float32).reshape(-1, 3).astype(np.float64)
    m = tri.shape[0]
    welded, count = weld_mesh(vertices, tri)
    neighbor = np.full((m, 3), -1, dtype=np.int64)
    opposite = np.full((m, 3), -1, dtype=np.int64)
    normal = np.cross(points[tri[:, 1]] - points[tri[:, 0]], points[tri[:, 2]] - points[tri[:, 0]])
    length = np.linalg.norm(normal, axis=1)
    normal = np.where(length[:, None] > 0.0, normal / np.maximum(length, 1.0e-300)[:, None], 0.0)
    boundary_edges = nonmanifold_edges = 0
    if m:
        # Half-edge k of triangle t joins corners k+1 and k+2 (it is opposite corner k).
        a = welded[:, [1, 2, 0]].reshape(-1)
        b = welded[:, [2, 0, 1]].reshape(-1)
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        # A collapsed edge (both ends welded) has no neighbour.
        valid = lo != hi
        key = lo * np.int64(count) + hi
        order = np.argsort(np.where(valid, key, -1), kind="stable")
        order = order[valid[order]]
        sorted_key = key[order]
        start = np.flatnonzero(np.r_[True, sorted_key[1:] != sorted_key[:-1]])
        size = np.diff(np.r_[start, sorted_key.size])
        pairs = start[size == 2]
        first = order[pairs]
        second = order[pairs + 1]
        t0, k0 = first // 3, first % 3
        t1, k1 = second // 3, second % 3
        # Two triangles with the same opposite vertex are coincident (a doubled or folded sheet).
        manifold = welded[t0, k0] != welded[t1, k1]
        t0, k0, t1, k1 = t0[manifold], k0[manifold], t1[manifold], k1[manifold]
        neighbor[t0, k0] = t1
        neighbor[t1, k1] = t0
        opposite[t0, k0] = tri[t1, k1]
        opposite[t1, k1] = tri[t0, k0]
        boundary_edges = int(np.count_nonzero(size == 1))
        nonmanifold_edges = int(np.count_nonzero(size > 2) + np.count_nonzero(~manifold))
    corner_vertex = welded.reshape(-1)
    order = np.argsort(corner_vertex, kind="stable")
    vertex_triangles = order // 3
    vertex_start = np.zeros(count + 1, dtype=np.int64)
    np.cumsum(np.bincount(corner_vertex, minlength=count), out=vertex_start[1:])
    flat = np.zeros(m, dtype=np.int64)
    for k in range(3):
        other = neighbor[:, k]
        has = other >= 0
        dots = np.einsum("ij,ij->i", normal, normal[np.maximum(other, 0)])
        flat |= np.where(has & (dots >= COPLANAR_COSINE), 1 << k, 0)
    owner = _corner_owners(welded, normal, vertex_start, vertex_triangles)
    return {
        "neighbor": neighbor.astype(np.int32),
        "opposite": opposite.astype(np.int32),
        "welded": welded.astype(np.int32),
        "owner": owner.astype(np.int32),
        "flat": flat.astype(np.int32),
        "normal": normal.astype(np.float32),
        "vertex_start": vertex_start.astype(np.int32),
        "vertex_triangles": vertex_triangles.astype(np.int32),
        "boundary_edges": boundary_edges,
        "nonmanifold_edges": nonmanifold_edges,
    }


class MeshTopology:
    """Welded topology of a model's meshed shapes on one device (see :class:`MeshTopologyData`)."""

    def __init__(self, shape_types, shape_sources, device, shape_radius=None):
        """Build the topology of every distinct mesh among the shapes.

        Args:
            shape_types: Host sequence of shape types.
            shape_sources: Host sequence of shape sources (a :class:`~newton.Mesh` for mesh shapes).
            device: Device for the arrays.
            shape_radius: The model's device array of per-shape bounding radii, read in place.
        """
        shape_count = len(shape_types)
        shape_base = np.full((max(shape_count, 1), 2), -1, dtype=np.int64)
        blocks: dict[int, int] = {}
        parts: list[dict[str, np.ndarray]] = []
        triangle_rows = 0
        for shape in range(shape_count):
            if int(shape_types[shape]) != int(GeoType.MESH):
                continue
            mesh = shape_sources[shape]
            if mesh is None or getattr(mesh, "indices", None) is None:
                continue
            key = id(mesh)
            if key not in blocks:
                blocks[key] = len(parts)
                part = build_mesh_topology(mesh.vertices, mesh.indices)
                part["row_base"] = triangle_rows
                triangle_rows += part["neighbor"].shape[0]
                parts.append(part)
            shape_base[shape, 0] = parts[blocks[key]]["row_base"]
            shape_base[shape, 1] = blocks[key]
        rows = [
            np.concatenate([p["neighbor"], p["opposite"], p["welded"], p["owner"], p["flat"][:, None]], 1)
            for p in parts
        ]
        cursor = triangle_rows * ROW
        starts = []
        for p in parts:
            starts.append(cursor)
            cursor += p["vertex_start"].shape[0]
        entries = cursor
        flat = [r.reshape(-1) for r in rows]
        for p in parts:
            flat.append(p["vertex_start"] + entries)
            entries += p["vertex_triangles"].shape[0]
        flat += [p["vertex_triangles"] for p in parts]
        for shape in range(shape_count):
            if shape_base[shape, 1] >= 0:
                shape_base[shape, 1] = starts[shape_base[shape, 1]]
        topology = np.concatenate(flat).astype(np.int32) if parts else np.zeros(1, dtype=np.int32)
        normals = np.concatenate([p["normal"] for p in parts]) if parts else np.zeros((1, 3), np.float32)
        self.mesh_count = len(parts)
        self.boundary_edges = sum(p["boundary_edges"] for p in parts)
        self.nonmanifold_edges = sum(p["nonmanifold_edges"] for p in parts)
        self.data = MeshTopologyData()
        self.data.topology = wp.array(topology, dtype=wp.int32, device=device)
        self.data.normal = wp.array(normals, dtype=wp.vec3, device=device)
        self.data.shape_base = wp.array(shape_base.astype(np.int32), dtype=wp.vec2i, device=device)
        if shape_radius is None:
            shape_radius = wp.zeros(1, dtype=float, device=device)
        self.data.shape_radius = shape_radius

    @staticmethod
    def from_model(model, device=None) -> MeshTopology:
        """Build the topology of a finalized model's meshed shapes."""
        return MeshTopology(
            model.shape_type.numpy(), model.shape_source, device or model.device, model.shape_collision_radius
        )


def empty_mesh_topology(device) -> MeshTopologyData:
    """Topology with no meshes: every mesh triangle queried alone, as a triangle soup."""
    return MeshTopology([], [], device).data


# ---------------------------------------------------------------------------
# Device side: the feature rule
# ---------------------------------------------------------------------------

FRONT_TOLERANCE = 1.0e-5
"""Direction cosine [-] by which a point must lie in front of a face for the face to be preferred as an owner.

A feature's owner must be a triangle the midphase admits; the midphase admits a triangle whose plane has the
partner's interior point in front of it, in its own frame and rounding. Owners are chosen among clearly
front-facing triangles first, so that rounding at the plane cannot hand a contact to a culled triangle.
"""

FACE_NORMAL_COSINE = 1.0 - 2.0e-6
"""Cosine [-] above which a GJK/MPR contact normal is its triangle's face normal (2 mrad, above their rounding)."""

SEGMENT_STRICT_COSINE = 1.0e-5
"""Direction cosine [-] by which a capsule core's interior minimum must clear its feature's faces.

A capsule core's interior point is a contact only where the surface distance has a strict minimum along the core:
over a convex edge or vertex. Over flat seams (and within rounding of them) the distance is constant across the
seam, and the core's ends carry the contact, as on a plane.
"""


@wp.struct
class SurfaceFeatureContext:
    """What the feature rule needs about the queried triangle and its surface."""

    heightfield: HeightfieldFeatureContext
    """The heightfield seam context (heightfield triangles only)."""
    topology: wp.array[wp.int32]
    normal: wp.array[wp.vec3]
    row: int
    """Index of this triangle's row in ``topology``, or -1 when the triangle has no topology."""
    triangle_base: int
    vertex_start: int
    triangle: int
    mesh_id: wp.uint64
    scale: wp.vec3
    transform: wp.transform
    mirror: int
    """1 when the mesh scale mirrors it, so that the triangle shape swaps corners 1 and 2."""
    p0: wp.vec3
    p1: wp.vec3
    p2: wp.vec3
    """World corners in the mesh's index order, computed as every triangle sharing them computes them."""
    partner_center: wp.vec3
    """World interior point of the convex partner, the point the midphase's back-face test uses."""
    partner_radius: float
    """The partner's bounding radius about its origin [m]."""
    unit: wp.vec3
    """The outward unit face normal (zero for a degenerate or untopologized triangle)."""
    edge_normal: wp.mat33
    """Row k: the outward in-plane unit normal of edge k (opposite corner k)."""
    edge_offset: wp.vec3
    """Component k: the plane offset of edge k along its normal, so that ``dot(p, n_k) - offset_k`` is p's distance
    beyond it."""
    edge_band: wp.vec3
    """Component k: the edge's own rounding band [m] (relative to its length)."""
    flat: int
    """Bit k: edge k joins a coplanar triangle (the row's flat-edge bits)."""


@wp.func
def surface_query_filter(
    triangle: GenericShapeData, point: wp.vec3, normal: wp.vec3, distance: float, context: SurfaceFeatureContext
) -> tuple[wp.vec3, wp.vec3, float]:
    """The heightfield seam witness of a separated query (mesh triangles keep theirs; their rule is per contact)."""
    return seam_surface_witness(triangle, point, normal, distance, context.heightfield)


@wp.func
def _mesh_point(context: SurfaceFeatureContext, vertex: int) -> wp.vec3:
    """World position of a mesh vertex, as :func:`get_triangle_shape_from_mesh` computes it."""
    mesh = wp.mesh_get(context.mesh_id)
    return wp.transform_point(context.transform, wp.cw_mul(mesh.points[vertex], context.scale))


@wp.func
def _world_normal(context: SurfaceFeatureContext, triangle: int) -> wp.vec3:
    """Outward world normal of a mesh triangle from its stored local normal (normals map by the inverse scale)."""
    local = context.normal[context.triangle_base + triangle]
    return wp.transform_vector(context.transform, wp.cw_div(local, context.scale))


@wp.func
def _facing(point: wp.vec3, reference: wp.vec3, normal: wp.vec3) -> int:
    """2 when ``point`` lies clearly in front of the plane, 1 in front or on it, 0 behind."""
    offset = point - reference
    side = wp.dot(offset, normal)
    if side > FRONT_TOLERANCE * wp.length(offset) * wp.length(normal):
        return 2
    if side >= 0.0:
        return 1
    return 0


@wp.func
def _edge_owner(context: SurfaceFeatureContext, reference: wp.vec3, first: int, second: int) -> int:
    """The triangle that reports a contact both triangles of an edge produce: the more front-facing, then the lower."""
    level_first = _facing(context.partner_center, reference, _world_normal(context, first))
    level_second = _facing(context.partner_center, reference, _world_normal(context, second))
    owner = wp.min(first, second)
    if level_first > level_second:
        owner = first
    elif level_second > level_first:
        owner = second
    return owner


@wp.func
def _edge_side(point: wp.vec3, low: wp.vec3, high: wp.vec3, opposite: wp.vec3) -> float:
    """How far ``point`` lies beyond the edge ``low--high``, away from ``opposite``, in their plane (scaled)."""
    u = high - low
    w = opposite - low
    outward = u * wp.dot(u, w) - w * wp.dot(u, u)
    return wp.dot(point - low, outward)


@wp.func
def _side_cosine(point: wp.vec3, low: wp.vec3, high: wp.vec3, opposite: wp.vec3) -> float:
    """:func:`_edge_side` as a direction cosine of ``point - low`` to the edge's outward in-plane direction."""
    u = high - low
    w = opposite - low
    outward = u * wp.dot(u, w) - w * wp.dot(u, u)
    offset = point - low
    scale = wp.length(offset) * wp.length(outward)
    if scale <= 0.0:
        return 0.0
    return wp.dot(offset, outward) / scale


@wp.func
def _beyond_vertex(point: wp.vec3, vertex: wp.vec3, other: wp.vec3, vertex_is_low: bool) -> bool:
    """Whether ``other`` lies in front of the plane through ``vertex`` with normal ``point - vertex``.

    Decided on the edge's parameter from its lower-id end, so that every triangle of the edge, and every
    triangle around either end, evaluates the same expression on the same operands.
    """
    if vertex_is_low:
        return wp.dot(point - vertex, other - vertex) > 0.0
    u = vertex - other
    return wp.dot(point - other, u) < wp.dot(u, u)


@wp.func
def _vertex_rule(
    context: SurfaceFeatureContext, point: wp.vec3, welded_vertex: int, vertex: wp.vec3, exact: bool
) -> bool:
    """Whether this triangle reports the contact at a vertex: the one-ring test, then the owner.

    With ``exact``, a one-ring vertex retires the contact when it lies in front of the plane through the vertex
    with normal ``point - vertex`` (the canonical edge parameter); otherwise by :func:`faces_witness`, with
    ``point - vertex`` the unit witness normal. The owner is the lowest incident triangle among the most
    front-facing ones.
    """
    start = context.topology[context.vertex_start + welded_vertex]
    end = context.topology[context.vertex_start + welded_vertex + 1]
    mesh = wp.mesh_get(context.mesh_id)
    direction = point - vertex
    ghost = bool(False)
    owner = context.triangle
    best = int(-1)
    for entry in range(start, end):
        other = context.topology[entry]
        row = (context.triangle_base + other) * ROW
        for j in range(3):
            welded = context.topology[row + 6 + j]
            if welded != welded_vertex:
                position = _mesh_point(context, mesh.indices[other * 3 + j])
                if exact:
                    if _beyond_vertex(point, vertex, position, welded_vertex < welded):
                        ghost = True
                elif faces_witness(position - vertex, direction):
                    ghost = True
        level = _facing(context.partner_center, vertex, _world_normal(context, other))
        if level > best:
            best = level
            owner = other
    return (not ghost) and owner == context.triangle


@wp.func
def _segment_vertex_rule(
    context: SurfaceFeatureContext, point: wp.vec3, axis: wp.vec3, welded_vertex: int, vertex: wp.vec3
) -> bool:
    """Whether a capsule core's interior point over a vertex is its strict minimum there, owned by this triangle.

    The vertex must be a minimum of the distance from ``point`` (no one-ring vertex in front, as for a point), and
    the surface must fall away strictly on both sides along the core ``axis``: some one-ring vertex lies strictly
    behind the plane ahead of the vertex, and some behind it. Over a flat vertex nothing falls away and the core's
    ends carry the contact.
    """
    start = context.topology[context.vertex_start + welded_vertex]
    end = context.topology[context.vertex_start + welded_vertex + 1]
    mesh = wp.mesh_get(context.mesh_id)
    direction = point - vertex
    direction_length = wp.length(direction)
    ghost = bool(False)
    ahead = bool(False)
    behind = bool(False)
    owner = context.triangle
    best = int(-1)
    for entry in range(start, end):
        other = context.topology[entry]
        row = (context.triangle_base + other) * ROW
        for j in range(3):
            welded = context.topology[row + 6 + j]
            if welded != welded_vertex:
                position = _mesh_point(context, mesh.indices[other * 3 + j])
                if _beyond_vertex(point, vertex, position, welded_vertex < welded):
                    ghost = True
                toward = position - vertex
                if wp.dot(direction, toward) < -SEGMENT_STRICT_COSINE * direction_length * wp.length(toward):
                    along = wp.dot(axis, toward)
                    if along > 0.0:
                        ahead = True
                    elif along < 0.0:
                        behind = True
        level = _facing(context.partner_center, vertex, _world_normal(context, other))
        if level > best:
            best = level
            owner = other
    return (not ghost) and ahead and behind and owner == context.triangle


@wp.func
def mesh_point_feature(
    context: SurfaceFeatureContext, point: wp.vec3, reach: float
) -> tuple[int, wp.vec3, wp.vec3, float]:
    """The contact of a mesh triangle with a point (a sphere's center): exact, one per local minimum.

    A point in front of the face gets the closest point of the triangle, classified as face, edge or vertex;
    an edge or vertex contact is reported only if it is a local minimum of the distance to the surface (no vertex
    adjacent to the feature in front of the plane through it with normal ``point - foot``) and this triangle owns
    the feature. A point behind the face gets the face's normal and plane distance if it projects onto the face
    (the triangle prism's one-sided semantics). A triangle with no topology is a lone triangle: all its edges are
    boundaries. Feet farther than ``reach`` are not reported, before any neighbour is read.

    Returns:
        ``(report, foot, normal, distance)``: whether to report, the foot on the triangle, the unit normal
        toward the point and the signed distance of the point from the foot along it.
    """
    p0 = context.p0
    p1 = context.p1
    p2 = context.p2
    normal = wp.cross(p1 - p0, p2 - p0)
    if context.mirror != 0:
        normal = -normal
    length = wp.length(normal)
    if length <= 1.0e-30:
        return 0, p0, wp.vec3(0.0, 0.0, 1.0), 0.0
    unit = normal / length
    plane = wp.dot(point - p0, unit)
    if plane > reach:
        return 0, p0, unit, plane
    has_topology = context.row >= 0
    welded = wp.vec3i(0, 1, 2)
    neighbor = wp.vec3i(-1, -1, -1)
    if has_topology:
        welded = wp.vec3i(
            context.topology[context.row + 6], context.topology[context.row + 7], context.topology[context.row + 8]
        )
        neighbor = wp.vec3i(
            context.topology[context.row + 0], context.topology[context.row + 1], context.topology[context.row + 2]
        )
    corners = wp.mat33(p0[0], p0[1], p0[2], p1[0], p1[1], p1[2], p2[0], p2[1], p2[2])
    sides = wp.vec3(0.0)
    for k in range(3):
        a = (k + 1) % 3
        b = (k + 2) % 3
        if welded[b] < welded[a]:
            swap = a
            a = b
            b = swap
        sides[k] = _edge_side(point, corners[a], corners[b], corners[k])
    if plane < 0.0:
        # Behind the face: its normal and plane distance, where the point projects onto it.
        report = int(1)
        for k in range(3):
            if sides[k] > 0.0:
                report = 0
            elif sides[k] == 0.0 and neighbor[k] >= 0 and neighbor[k] < context.triangle:
                report = 0
        return report, point - plane * unit, unit, plane
    if sides[0] < 0.0 and sides[1] < 0.0 and sides[2] < 0.0:
        return 1, point - plane * unit, unit, plane
    for k in range(3):
        a = (k + 1) % 3
        b = (k + 2) % 3
        if welded[b] < welded[a]:
            swap = a
            a = b
            b = swap
        low = corners[a]
        high = corners[b]
        if sides[k] >= 0.0 and _beyond_vertex(point, low, high, True) and _beyond_vertex(point, high, low, False):
            # The edge's interior: a minimum if the point also lies beyond it for the neighbour.
            u = high - low
            foot = low + (wp.dot(point - low, u) / wp.dot(u, u)) * u
            offset = point - foot
            distance = wp.length(offset)
            direction = unit
            if distance > 0.0:
                direction = offset / distance
            report = int(1)
            if distance > reach:
                report = 0
            elif neighbor[k] >= 0:
                other = _mesh_point(context, context.topology[context.row + 3 + k])
                if _edge_side(point, low, high, other) < 0.0:
                    report = 0
                elif _edge_owner(context, low, context.triangle, neighbor[k]) != context.triangle:
                    report = 0
            return report, foot, direction, distance
    # A vertex region: the corner none of whose triangle edges leaves its plane toward the point.
    corner = int(-1)
    for j in range(3):
        if corner < 0:
            a = (j + 1) % 3
            b = (j + 2) % 3
            if not _beyond_vertex(point, corners[j], corners[a], welded[j] < welded[a]) and not _beyond_vertex(
                point, corners[j], corners[b], welded[j] < welded[b]
            ):
                corner = j
    if corner < 0:
        corner = 0
        nearest = wp.length_sq(point - p0)
        if wp.length_sq(point - p1) < nearest:
            corner = 1
            nearest = wp.length_sq(point - p1)
        if wp.length_sq(point - p2) < nearest:
            corner = 2
    vertex = corners[corner]
    offset = point - vertex
    distance = wp.length(offset)
    direction = unit
    if distance > 0.0:
        direction = offset / distance
    report = int(1)
    if distance > reach:
        report = 0
    elif has_topology:
        if not _vertex_rule(context, point, welded[corner], vertex, True):
            report = 0
    return report, vertex, direction, distance


@wp.func
def mesh_contact_filter(contact: ContactData, context: SurfaceFeatureContext) -> bool:
    """The feature rule for one GJK/MPR contact (manifold point) of a mesh triangle.

    The contact's point on the triangle is classified as face interior, edge or vertex (within the world
    coordinates' rounding), and its normal as the face normal or tilted. A face-normal contact is the face's
    own: the triangle containing it reports it (on an edge or a corner shared with coplanar triangles, the
    lowest of them). A tilted contact on an edge or vertex is reported only if no vertex adjacent to the feature
    lies in front of the plane through it (:func:`faces_witness`) and this triangle owns the feature.
    Contacts beyond the writer's gap, of triangles without topology and of heightfields are left to the writer.
    """
    if context.row < 0:
        return True
    if contact.contact_distance - contact.margin_a - contact.margin_b > contact.gap_sum:
        return True
    unit = context.unit
    if wp.length_sq(unit) == 0.0:
        return True
    normal = contact.contact_normal_a_to_b
    face_contact = wp.dot(normal, unit) >= FACE_NORMAL_COSINE
    if face_contact and context.flat == 0:
        # A face contact of a triangle with no coplanar neighbour: the manifold was clipped to the triangle itself.
        return True
    point = contact.contact_point_center - 0.5 * contact.contact_distance * normal
    reach = 4.0e-6 * (1.0 + wp.max(wp.abs(point[0]), wp.max(wp.abs(point[1]), wp.abs(point[2]))))
    on_edge = wp.vec3i(0)
    edges = int(0)
    edge = int(-1)
    outside = bool(False)
    for k in range(3):
        beyond = wp.dot(point, context.edge_normal[k]) - context.edge_offset[k]
        band = reach + context.edge_band[k]
        if beyond >= -band:
            on_edge[k] = 1
            edges += 1
            edge = k
        if beyond > band:
            outside = True
    if face_contact and outside:
        # A face contact's manifold spans the coplanar facet (mesh_manifold_triangle): the triangle under it owns it.
        return False
    if edges == 0:
        return True
    return _contact_feature_rule(context, point, normal, face_contact, on_edge, edges, edge)


@wp.func
def _contact_feature_rule(
    context: SurfaceFeatureContext,
    point: wp.vec3,
    normal: wp.vec3,
    face_contact: bool,
    on_edge: wp.vec3i,
    edges: int,
    edge: int,
) -> bool:
    """:func:`mesh_contact_filter` for a point on an edge or a corner of its triangle."""
    t = context.triangle
    p0 = context.p0
    p1 = context.p1
    p2 = context.p2
    corners = wp.mat33(p0[0], p0[1], p0[2], p1[0], p1[1], p1[2], p2[0], p2[1], p2[2])
    row = context.row
    if edges == 1:
        other = context.topology[row + edge]
        if other < 0:
            return True
        if face_contact:
            if (context.topology[row + 12] >> edge) & 1 != 0:
                return t < other
            return True
        a = (edge + 1) % 3
        b = (edge + 2) % 3
        if context.topology[row + 6 + b] < context.topology[row + 6 + a]:
            swap = a
            a = b
            b = swap
        if faces_witness(_mesh_point(context, context.topology[row + 3 + edge]) - point, normal):
            return False
        if faces_witness(corners[edge] - point, normal):
            # The witness leaves this triangle's own cone: it is not a closest feature; keep it.
            return True
        return _edge_owner(context, corners[a], t, other) == t
    # Two or three edges: the vertex they share (the corner off neither).
    corner = int(0)
    for j in range(3):
        if on_edge[j] == 0:
            corner = j
    if edges == 3:
        corner = 0
        nearest = wp.length_sq(point - p0)
        if wp.length_sq(point - p1) < nearest:
            corner = 1
            nearest = wp.length_sq(point - p1)
        if wp.length_sq(point - p2) < nearest:
            corner = 2
    if face_contact:
        return context.topology[row + 9 + corner] == t
    vertex = corners[corner]
    return _vertex_rule(context, vertex + normal, context.topology[row + 6 + corner], vertex, False)


@wp.func
def mesh_manifold_triangle(
    triangle: GenericShapeData, point_a: wp.vec3, normal: wp.vec3, context: SurfaceFeatureContext
) -> tuple[GenericShapeData, wp.vec3, bool]:
    """The triangle a face contact's manifold clips against: this one grown across its coplanar edges.

    Across an edge shared with a coplanar triangle the surface is one plane, so the contact patch is not cut
    there: each coplanar edge moves outward past the partner's reach, and :func:`mesh_contact_filter` keeps the
    points this triangle contains (on a shared edge or corner, the lowest coplanar triangle). Every point of the
    patch is then reported once, by the triangle under it, with no seam points. Tilted contacts, triangles
    without topology and heightfields keep the triangle.

    Returns:
        ``(triangle, origin, moved)``: the manifold's triangle, its first corner in the triangle's frame, and
        whether it is grown.
    """
    if context.row < 0:
        return triangle, wp.vec3(0.0), False
    flat = context.topology[context.row + 12]
    if flat == 0:
        return triangle, wp.vec3(0.0), False
    if wp.dot(normal, context.unit) < FACE_NORMAL_COSINE:
        return triangle, wp.vec3(0.0), False
    reach = wp.length(context.partner_center - context.p0) + 2.0 * context.partner_radius + wp.length(point_a)
    corners = wp.mat33(
        0.0,
        0.0,
        0.0,
        triangle.scale[0],
        triangle.scale[1],
        triangle.scale[2],
        triangle.auxiliary[0],
        triangle.auxiliary[1],
        triangle.auxiliary[2],
    )
    # The triangle's frame is the world's turned by nothing (a mesh triangle's), so the context's world edge normals
    # serve; shape corners 1 and 2 are the mesh's corners 2 and 1 under a mirroring scale.
    offsets = wp.vec3(0.0)
    outward = wp.mat33(0.0)
    for s in range(3):
        k = s
        if context.mirror != 0 and s > 0:
            k = 3 - s
        outward[s] = context.edge_normal[k]
        if (flat >> k) & 1 != 0:
            offsets[s] = reach
    moved = wp.mat33(0.0)
    for s in range(3):
        q = (s + 1) % 3
        r = (s + 2) % 3
        c = wp.dot(outward[q], outward[r])
        denominator = 1.0 - c * c
        if denominator < 1.0e-4:
            # A sliver's corner would run far along its edges: keep the triangle.
            return triangle, wp.vec3(0.0), False
        alpha = (offsets[q] - c * offsets[r]) / denominator
        beta = (offsets[r] - c * offsets[q]) / denominator
        moved[s] = corners[s] + alpha * outward[q] + beta * outward[r]
    result = triangle
    result.scale = moved[1] - moved[0]
    result.auxiliary = moved[2] - moved[0]
    return result, moved[0], True


@wp.func
def mesh_triangle_corners(
    mesh_id: wp.uint64, scale: wp.vec3, transform: wp.transform, triangle: int
) -> tuple[wp.vec3, wp.vec3, wp.vec3]:
    """World corners of a mesh triangle in the mesh's index order (as :func:`get_triangle_shape_from_mesh`)."""
    mesh = wp.mesh_get(mesh_id)
    p0 = wp.transform_point(transform, wp.cw_mul(mesh.points[mesh.indices[triangle * 3 + 0]], scale))
    p1 = wp.transform_point(transform, wp.cw_mul(mesh.points[mesh.indices[triangle * 3 + 1]], scale))
    p2 = wp.transform_point(transform, wp.cw_mul(mesh.points[mesh.indices[triangle * 3 + 2]], scale))
    return p0, p1, p2


@wp.func
def triangle_shape_from_corners(p0: wp.vec3, p1: wp.vec3, p2: wp.vec3, scale: wp.vec3) -> GenericShapeData:
    """The triangle shape :func:`get_triangle_shape_from_mesh` builds, from the corners (mirroring swaps 1 and 2)."""
    shape = GenericShapeData()
    shape.shape_type = int(GeoTypeEx.TRIANGLE)
    if scale[0] * scale[1] * scale[2] < 0.0:
        shape.scale = p2 - p0
        shape.auxiliary = p1 - p0
    else:
        shape.scale = p1 - p0
        shape.auxiliary = p2 - p0
    shape.center = wp.vec3(0.0, 0.0, 0.0)
    return shape


@wp.func
def mesh_surface_context(
    topology: MeshTopologyData,
    shape: int,
    triangle: int,
    mesh_id: wp.uint64,
    scale: wp.vec3,
    transform: wp.transform,
    p0: wp.vec3,
    p1: wp.vec3,
    p2: wp.vec3,
    partner_center: wp.vec3,
    partner: int,
) -> SurfaceFeatureContext:
    """The feature context of a mesh shape's triangle (no topology: a lone triangle); ``p0..p2`` its world corners."""
    context = SurfaceFeatureContext()
    context.topology = topology.topology
    context.normal = topology.normal
    context.row = -1
    context.triangle_base = 0
    context.vertex_start = 0
    if shape < topology.shape_base.shape[0]:
        base = topology.shape_base[shape]
        if base[0] >= 0:
            context.triangle_base = base[0]
            context.vertex_start = base[1]
            context.row = (base[0] + triangle) * ROW
            context.flat = topology.topology[context.row + 12]
    context.triangle = triangle
    context.mesh_id = mesh_id
    context.scale = scale
    context.transform = transform
    context.mirror = int(scale[0] * scale[1] * scale[2] < 0.0)
    context.p0 = p0
    context.p1 = p1
    context.p2 = p2
    if context.row >= 0:
        # Per-query edge planes, so that each manifold point's feature costs three dot products.
        face = wp.cross(p1 - p0, p2 - p0)
        if context.mirror != 0:
            face = -face
        face_length = wp.length(face)
        if face_length > 1.0e-30:
            context.unit = face / face_length
            corners = wp.mat33(p0[0], p0[1], p0[2], p1[0], p1[1], p1[2], p2[0], p2[1], p2[2])
            normals = wp.mat33(0.0)
            offsets = wp.vec3(0.0)
            bands = wp.vec3(0.0)
            for k in range(3):
                a = corners[(k + 1) % 3]
                u = corners[(k + 2) % 3] - a
                w = corners[k] - a
                outward = u * wp.dot(u, w) - w * wp.dot(u, u)
                outward_length = wp.length(outward)
                if outward_length > 0.0:
                    outward = outward / outward_length
                    normals[k] = outward
                    offsets[k] = wp.dot(outward, a)
                    bands[k] = 1.0e-6 * wp.length(u)
                else:
                    # A degenerate edge never claims a point: it lies far behind it.
                    offsets[k] = 1.0e30
            context.edge_normal = normals
            context.edge_offset = offsets
            context.edge_band = bands
    context.partner_center = partner_center
    context.partner_radius = 0.0
    if partner < topology.shape_radius.shape[0]:
        context.partner_radius = topology.shape_radius[partner]
    return context


@wp.func
def triangle_plane_clears(
    triangle: GenericShapeData,
    origin: wp.vec3,
    support_point: wp.vec3,
    threshold: float,
) -> bool:
    """Whether the partner's deepest point toward a triangle's face lies beyond the contact threshold.

    ``support_point`` is the partner's world support point along minus the face normal. The plane distance
    bounds the distance to the triangle from below, so no contact of the triangle can exist; the margin is the
    reducer writer's world-rounding allowance.
    """
    face = wp.cross(triangle.scale, triangle.auxiliary)
    length = wp.length(face)
    if length <= 1.0e-30:
        return False
    gap = wp.dot(support_point - origin, face) / length
    allowance = 3.814697265625e-6 * (
        1.0 + wp.max(wp.abs(support_point[0]), wp.max(wp.abs(support_point[1]), wp.abs(support_point[2]))) + threshold
    )
    return gap > threshold + allowance


@wp.func
def _point_contact(foot: wp.vec3, normal: wp.vec3, distance: float, radius: float) -> ContactData:
    """A round shape's contact from its core point's foot, normal and core distance (behind: the face's)."""
    contact = ContactData()
    contact.contact_normal_a_to_b = normal
    contact.contact_distance = distance - radius
    if distance >= 0.0:
        contact.contact_point_center = foot + normal * (0.5 * contact.contact_distance)
        contact.radius_eff_b = radius
    else:
        deepest = foot + normal * (distance - radius)
        contact.contact_point_center = 0.5 * (foot + deepest)
    return contact


def create_mesh_sphere_contact(writer_func):
    """Create the closed-form sphere contact of a mesh triangle (:func:`mesh_point_feature`), written by ``writer_func``."""

    @wp.func
    def mesh_sphere_contact(
        context: SurfaceFeatureContext,
        center: wp.vec3,
        radius: float,
        threshold: float,
        margin_a: float,
        margin_b: float,
        shape_a: int,
        shape_b: int,
        gap_sum: float,
        sort_sub_key: int,
        writer_data: Any,
    ):
        """Write the sphere's contact with this triangle, if the triangle reports one within ``threshold``.

        In front of the face the contact is the sphere's (``radius_eff_b`` = radius, the point midway between
        the foot and the sphere's surface); behind it, the face's one-sided contact (the deepest sphere point's
        plane distance), as the triangle path gives.
        """
        report, foot, normal, distance = mesh_point_feature(context, center, threshold + radius)
        if report == 0:
            return
        contact = _point_contact(foot, normal, distance, radius)
        if contact.contact_distance < threshold:
            contact.margin_a = margin_a
            contact.margin_b = margin_b
            contact.shape_a = shape_a
            contact.shape_b = shape_b
            contact.gap_sum = gap_sum
            contact.sort_sub_key = sort_sub_key
            writer_func(contact, writer_data, -1)

    return mesh_sphere_contact


@wp.func
def _core_behind_face(
    context: SurfaceFeatureContext, start: wp.vec3, end: wp.vec3
) -> tuple[int, wp.vec3, wp.vec3, float]:
    """The deepest point of a core segment behind a triangle's face, over the face.

    The segment is clipped to the triangle's prism (its in-plane edge half-spaces); the deepest clipped point
    along the face normal is an end of the clipped interval. A point behind the plane is reported with the face's
    outward normal and its (negative) plane distance; its foot is its projection on the face.
    """
    p0 = context.p0
    p1 = context.p1
    p2 = context.p2
    face = wp.cross(p1 - p0, p2 - p0)
    if context.mirror != 0:
        face = -face
    length = wp.length(face)
    if length <= 1.0e-30:
        return 0, p0, wp.vec3(0.0, 0.0, 1.0), 0.0
    unit = face / length
    h0 = wp.dot(start - p0, unit)
    h1 = wp.dot(end - p0, unit)
    if h0 >= 0.0 and h1 >= 0.0:
        return 0, p0, unit, 0.0
    corners = wp.mat33(p0[0], p0[1], p0[2], p1[0], p1[1], p1[2], p2[0], p2[1], p2[2])
    d = end - start
    low = float(0.0)
    high = float(1.0)
    for k in range(3):
        a = corners[(k + 1) % 3]
        u = corners[(k + 2) % 3] - a
        w = corners[k] - a
        outward = u * wp.dot(u, w) - w * wp.dot(u, u)
        c0 = wp.dot(start - a, outward)
        c1 = wp.dot(d, outward)
        # Keep c0 + sigma c1 <= 0.
        if c1 > 0.0:
            high = wp.min(high, -c0 / c1)
        elif c1 < 0.0:
            low = wp.max(low, -c0 / c1)
        elif c0 > 0.0:
            low = 1.0
            high = 0.0
    if low > high:
        return 0, p0, unit, 0.0
    sigma = low
    if h0 + high * (h1 - h0) < h0 + low * (h1 - h0):
        sigma = high
    point = start + sigma * d
    depth = wp.dot(point - p0, unit)
    if depth >= 0.0:
        return 0, p0, unit, 0.0
    return 1, point - depth * unit, unit, depth


def create_mesh_capsule_contacts(writer_func):
    """Create the closed-form capsule contacts of a mesh triangle, written by ``writer_func``.

    The core segment's ends take the point rule (:func:`mesh_point_feature`): the ends of a capsule resting on
    a face carry it as on a plane. An interior point of the core is a contact where it is the core's strict
    local minimum over a convex edge (the closest points of the core and the edge, both interior, beyond the
    edge for both its faces) or a convex vertex (the vertex's projection on the core, every one-ring vertex
    behind), reported by the feature's owner.
    """

    @wp.func
    def write_capsule_contact(
        foot: wp.vec3,
        normal: wp.vec3,
        distance: float,
        radius: float,
        threshold: float,
        margin_a: float,
        margin_b: float,
        shape_a: int,
        shape_b: int,
        gap_sum: float,
        sort_sub_key: int,
        writer_data: Any,
    ):
        contact = _point_contact(foot, normal, distance, radius)
        if contact.contact_distance < threshold:
            contact.margin_a = margin_a
            contact.margin_b = margin_b
            contact.shape_a = shape_a
            contact.shape_b = shape_b
            contact.gap_sum = gap_sum
            contact.sort_sub_key = sort_sub_key
            writer_func(contact, writer_data, -1)

    @wp.func
    def mesh_capsule_contacts(
        context: SurfaceFeatureContext,
        start: wp.vec3,
        end: wp.vec3,
        radius: float,
        threshold: float,
        margin_a: float,
        margin_b: float,
        shape_a: int,
        shape_b: int,
        gap_sum: float,
        sort_sub_key: int,
        writer_data: Any,
    ):
        """Write a capsule's contacts with this triangle; ``start``/``end`` are its core's world ends."""
        reach = threshold + radius
        for e in range(2):
            point = start
            if e == 1:
                point = end
            report, foot, normal, distance = mesh_point_feature(context, point, reach)
            # An end behind the face is the clipped core's (below); the point rule keeps the ends in front.
            if report != 0 and distance >= 0.0:
                write_capsule_contact(
                    foot,
                    normal,
                    distance,
                    radius,
                    threshold,
                    margin_a,
                    margin_b,
                    shape_a,
                    shape_b,
                    gap_sum,
                    (sort_sub_key << 3) | e,
                    writer_data,
                )
        # The core behind the face: the one-sided contact of its deepest point over the face (the core clipped to
        # the triangle's prism), the face's normal and that point's plane distance, as the triangle prism gives.
        behind_report, behind_foot, behind_normal, behind_distance = _core_behind_face(context, start, end)
        if behind_report != 0:
            write_capsule_contact(
                behind_foot,
                behind_normal,
                behind_distance,
                radius,
                threshold,
                margin_a,
                margin_b,
                shape_a,
                shape_b,
                gap_sum,
                (sort_sub_key << 3) | 1,
                writer_data,
            )
        t = context.triangle
        row = context.row
        has_topology = row >= 0
        welded = wp.vec3i(0, 1, 2)
        if has_topology:
            welded = wp.vec3i(context.topology[row + 6], context.topology[row + 7], context.topology[row + 8])
        p0 = context.p0
        p1 = context.p1
        p2 = context.p2
        corners = wp.mat33(p0[0], p0[1], p0[2], p1[0], p1[1], p1[2], p2[0], p2[1], p2[2])
        face = wp.cross(p1 - p0, p2 - p0)
        if context.mirror != 0:
            face = -face
        d = end - start
        dd = wp.dot(d, d)
        if dd <= 0.0:
            return
        for k in range(3):
            a = (k + 1) % 3
            b = (k + 2) % 3
            if welded[b] < welded[a]:
                swap = a
                a = b
                b = swap
            low = corners[a]
            high = corners[b]
            # A convex or boundary edge: the core's closest point to it, interior to both, strictly beyond
            # its faces.
            u = high - low
            uu = wp.dot(u, u)
            if uu > 0.0:
                du = wp.dot(d, u)
                denominator = dd * uu - du * du
                if denominator > 1.0e-6 * dd * uu:
                    w0 = start - low
                    sigma = (du * wp.dot(u, w0) - uu * wp.dot(d, w0)) / denominator
                    if sigma > 0.0 and sigma < 1.0:
                        point = start + sigma * d
                        foot = low + (wp.dot(point - low, u) / uu) * u
                        offset = point - foot
                        distance = wp.length(offset)
                        if (
                            distance > 0.0
                            and distance <= reach
                            and _beyond_vertex(point, low, high, True)
                            and _beyond_vertex(point, high, low, False)
                        ):
                            minimum = (
                                _side_cosine(point, low, high, corners[k]) > SEGMENT_STRICT_COSINE
                                and wp.dot(point - low, face) > 0.0
                            )
                            other = int(-1)
                            if has_topology:
                                other = context.topology[row + k]
                            if minimum and other >= 0:
                                opposite = _mesh_point(context, context.topology[row + 3 + k])
                                minimum = (
                                    _side_cosine(point, low, high, opposite) > SEGMENT_STRICT_COSINE
                                    and _edge_owner(context, low, t, other) == t
                                )
                            if minimum:
                                write_capsule_contact(
                                    foot,
                                    offset / distance,
                                    distance,
                                    radius,
                                    threshold,
                                    margin_a,
                                    margin_b,
                                    shape_a,
                                    shape_b,
                                    gap_sum,
                                    (sort_sub_key << 3) | (2 + k),
                                    writer_data,
                                )
        for j in range(3):
            vertex = corners[j]
            sigma = wp.dot(vertex - start, d) / dd
            if sigma > 0.0 and sigma < 1.0:
                point = start + sigma * d
                offset = point - vertex
                distance = wp.length(offset)
                if distance > 0.0 and distance <= reach and wp.dot(offset, face) > 0.0:
                    keep = False
                    if has_topology:
                        keep = _segment_vertex_rule(context, point, d, welded[j], vertex)
                    else:
                        # A lone triangle: its own two other corners are the vertex's one-ring.
                        keep = True
                        ahead = bool(False)
                        behind = bool(False)
                        for q in range(1, 3):
                            position = corners[(j + q) % 3]
                            if _beyond_vertex(point, vertex, position, j < (j + q) % 3):
                                keep = False
                            toward = position - vertex
                            if wp.dot(offset, toward) < -SEGMENT_STRICT_COSINE * distance * wp.length(toward):
                                if wp.dot(d, toward) > 0.0:
                                    ahead = True
                                elif wp.dot(d, toward) < 0.0:
                                    behind = True
                        keep = keep and ahead and behind
                    if keep:
                        write_capsule_contact(
                            vertex,
                            offset / distance,
                            distance,
                            radius,
                            threshold,
                            margin_a,
                            margin_b,
                            shape_a,
                            shape_b,
                            gap_sum,
                            (sort_sub_key << 3) | (5 + j),
                            writer_data,
                        )

    return mesh_capsule_contacts


def create_surface_penetration_refiner(support_func: Any):
    """Create the one-sided refinement of penetrating MPR results on surface triangles.

    A lone triangle is a convex body to MPR, so a penetration's direction may tilt off the face toward an edge,
    where the partner would leave the lone triangle sideways; the surface goes on there. As the heightfield's
    triangle prism refiner does (``create_triangle_prism_penetration_refiner``, which this calls for prisms),
    a penetrating mesh-triangle result is the face's: its outward normal, the partner's support depth along it,
    and the partner's deepest point when it lies over the face (else MPR's witness on the partner). The feature
    rule's ghost test is then for separated witnesses only, as on heightfields.
    """
    prism_refiner = create_triangle_prism_penetration_refiner(support_func)
    shape_support = create_shape_support_function(support_func, center_ties=True)

    @wp.func
    def refine_surface_penetration(
        geom_a: Any,
        geom_b: Any,
        orientation_b: wp.quat,
        position_b: wp.vec3,
        extend: float,
        data_provider: Any,
        point_a: wp.vec3,
        point_b: wp.vec3,
        normal: wp.vec3,
        penetration: float,
    ) -> tuple[wp.vec3, wp.vec3, wp.vec3, float]:
        if geom_a.shape_type == int(GeoTypeEx.TRIANGLE_PRISM):
            return prism_refiner(
                geom_a, geom_b, orientation_b, position_b, extend, data_provider, point_a, point_b, normal, penetration
            )
        if geom_a.shape_type != int(GeoTypeEx.TRIANGLE):
            return point_a, point_b, normal, penetration
        face = wp.cross(geom_a.scale, geom_a.auxiliary)
        length_sq = wp.length_sq(face)
        if length_sq < 1.0e-24:
            return point_a, point_b, normal, penetration
        unit = face / wp.sqrt(length_sq)
        surface_point_b = (
            wp.quat_rotate(
                orientation_b, shape_support(geom_b, wp.quat_rotate_inv(orientation_b, -unit), data_provider)
            )
            + position_b
        )
        surface_point_a = wp.vec3(0.0)
        if extend != 0.0:
            offset = unit * extend * 0.5
            surface_point_a += offset
            surface_point_b -= offset
        projected_b = surface_point_b - wp.dot(surface_point_b, unit) * unit
        closest_b = closest_point_on_triangle(projected_b, wp.vec3(0.0), geom_a.scale, geom_a.auxiliary)
        if wp.length_sq(projected_b - closest_b) >= 1.0e-10:
            # The deepest point lies off this face (a neighbour, or no surface, carries it): this face's
            # overlap is measured at MPR's own witness on the partner.
            surface_point_b = point_b
        surface_penetration = wp.dot(surface_point_a - surface_point_b, unit)
        return surface_point_b + surface_penetration * unit, surface_point_b, unit, surface_penetration

    return refine_surface_penetration
