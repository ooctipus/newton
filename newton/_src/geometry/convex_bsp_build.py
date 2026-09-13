# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bounded exact static normal-fan construction; no runtime fixture dependencies."""

import itertools
import math
import time
from collections import Counter

import numpy as np


def adjacency(vertices, faces):
    """Validate a closed convex all-extreme mesh using exact integer predicates."""
    ratios = [[float(x).as_integer_ratio() for x in row] for row in vertices]
    denominator = max((d for row in ratios for _, d in row))
    points = [tuple((n * (denominator // d) for n, d in row)) for row in ratios]

    def sub(a, b):
        return tuple((x - y for x, y in zip(a, b, strict=True)))

    def cross(a, b):
        return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])

    def dot(a, b):
        return sum((x * y for x, y in zip(a, b, strict=True)))

    if not len(set(points)) == len(points):
        raise ValueError("duplicate vertices require full scan")
    if not len({tuple(sorted(face)) for face in faces}) == len(faces):
        raise ValueError("Invalid exact convex hull representation")
    edges, incident = (Counter(), [[] for _ in points])
    neighbors = [set() for _ in points]
    for face in faces:
        ia, ib, ic = (int(value) for value in face)
        if not (0 <= min(ia, ib, ic) and max(ia, ib, ic) < len(points) and (len({ia, ib, ic}) == 3)):
            raise ValueError("Invalid exact convex hull representation")
        a, b, c = (points[ia], points[ib], points[ic])
        normal = cross(sub(b, a), sub(c, a))
        if not any(normal):
            raise ValueError("degenerate triangle requires full scan")
        signs = [dot(normal, sub(p, a)) for p in points]
        if not (min(signs) >= 0 or max(signs) <= 0):
            raise ValueError("nonconvex mesh requires full scan")
        for i in (ia, ib, ic):
            incident[i].append(normal)
        for i, j in ((ia, ib), (ib, ic), (ic, ia)):
            neighbors[i].add(j)
            neighbors[j].add(i)
            edges[tuple(sorted((i, j)))] += 1
    if not (set(edges.values()) == {2} and len(points) - len(edges) + len(faces) == 2):
        raise ValueError("Invalid exact convex hull representation")
    seen, todo = (set(), [0])
    while todo:
        i = todo.pop()
        if i not in seen:
            seen.add(i)
            todo.extend(neighbors[i] - seen)
    if not len(seen) == len(points):
        raise ValueError("Invalid exact convex hull representation")
    for normals in incident:
        if not any((dot(a, cross(b, c)) != 0 for a, b, c in itertools.combinations(normals, 3))):
            raise ValueError("non-extreme stored vertex requires full scan")
    return [sorted(row) for row in neighbors]


def edge_graph(vertices, faces):
    """Validate the hull and remove only exactly coplanar triangle diagonals."""
    adjacency(vertices, faces)
    ratios = [[float(x).as_integer_ratio() for x in row] for row in vertices]
    den = max((d for row in ratios for _, d in row))
    points = [tuple((n * (den // d) for n, d in row)) for row in ratios]
    planes, edges = ([], {})
    for face in faces:
        a, b, c = (points[int(i)] for i in face)
        u, v = ([b[i] - a[i] for i in range(3)], [c[i] - a[i] for i in range(3)])
        normal = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
        plane = (*normal, -sum(normal[i] * a[i] for i in range(3)))
        divisor = math.gcd(*plane)
        plane = tuple(x // divisor for x in plane)
        if next(x for x in plane if x) < 0:
            plane = tuple(-x for x in plane)
        planes.append(plane)
        for i, j in zip(face, np.roll(face, -1), strict=True):
            edges.setdefault(tuple(sorted((int(i), int(j)))), []).append(len(planes) - 1)
    graph = [set() for _ in vertices]
    for (i, j), adjacent in edges.items():
        if not len(adjacent) == 2:
            raise ValueError("Invalid exact convex hull representation")
        if planes[adjacent[0]] != planes[adjacent[1]]:
            graph[i].add(j)
            graph[j].add(i)
    if not all(len(row) >= 3 for row in graph):
        raise ValueError("Invalid exact convex hull representation")
    seen, todo = (set(), [0])
    while todo:
        i = todo.pop()
        if i not in seen:
            seen.add(i)
            todo.extend(graph[i] - seen)
    if not len(seen) == len(vertices):
        raise ValueError("Invalid exact convex hull representation")
    return [sorted(row) for row in graph]


def dot(a, b):
    """Evaluate an integer homogeneous predicate exactly."""
    return sum((x * y for x, y in zip(a, b, strict=True)))


def cross(a, b):
    """Compute an exact homogeneous cross product."""
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def reduced(a, *, unoriented=False):
    """Remove positive common factors, optionally canonicalizing plane orientation."""
    divisor = math.gcd(*a)
    if not divisor:
        return tuple(a)
    result = tuple(x // divisor for x in a)
    if unoriented and next(x for x in result if x) < 0:
        result = tuple(-x for x in result)
    return result


def integer_points(vertices):
    """Represent all finite binary input coordinates in a common integer scale."""
    ratios = [[float(x).as_integer_ratio() for x in row] for row in vertices]
    denominator = max((d for row in ratios for _, d in row))
    return [tuple((n * (denominator // d) for n, d in row)) for row in ratios]


def pack_plane(plane):
    """Pack an at-most-106-bit homogeneous plane exactly into two doubles per coefficient."""
    exponent = max(abs(x).bit_length() for x in plane)
    if not 0 < exponent <= 106:
        raise ValueError("Invalid exact convex hull representation")
    high = np.array([math.ldexp(float(x), -exponent) for x in plane])
    residual = [x - int(math.ldexp(float(h), exponent)) for x, h in zip(plane, high, strict=True)]
    if not all(abs(x).bit_length() <= 53 for x in residual):
        raise ValueError("Invalid exact convex hull representation")
    low = np.array([math.ldexp(float(x), -exponent) for x in residual])
    for original, a, b in zip(plane, high, low, strict=True):
        if not int(math.ldexp(float(a), exponent)) + int(math.ldexp(float(b), exponent)) == original:
            raise ValueError("Invalid exact convex hull representation")
    return (high, low, exponent)


def normal_fan(vertices, faces):
    """Build ordered normal-cone polygons from actual hull faces and true edges."""
    graph = edge_graph(vertices, faces)
    points = integer_points(vertices)
    facets, triangle_facet = ({}, [])
    for face in faces:
        a, b, c = (points[int(i)] for i in face)
        normal = cross(
            tuple((y - x for x, y in zip(a, b, strict=True))), tuple((y - x for x, y in zip(a, c, strict=True)))
        )
        normal = reduced(normal)
        values = [dot(normal, tuple((y - x for x, y in zip(a, p, strict=True)))) for p in points]
        if max(values) > 0:
            normal = tuple(-x for x in normal)
            values = [-x for x in values]
        if not (max(values) == 0 and min(values) < 0):
            raise ValueError("Invalid exact convex hull representation")
        plane = (*normal, -dot(normal, a))
        if plane not in facets:
            facets[plane] = len(facets)
        triangle_facet.append(facets[plane])
    normals = [p[:3] for p in facets]
    edge_faces = {}
    incident = [set() for _ in points]
    for face, facet in zip(faces, triangle_facet, strict=True):
        for vertex in face:
            incident[int(vertex)].add(facet)
        for i, j in zip(face, np.roll(face, -1), strict=True):
            edge_faces.setdefault(tuple(sorted((int(i), int(j)))), set()).add(facet)
    cones = []
    for vertex, neighbors in enumerate(graph):
        links = {f: set() for f in incident[vertex]}
        for neighbor in neighbors:
            a, b = edge_faces[tuple(sorted((vertex, neighbor)))]
            links[a].add(b)
            links[b].add(a)
        if not all(len(row) == 2 for row in links.values()):
            raise ValueError("Invalid exact convex hull representation")
        ordered, previous, current = ([], None, min(links))
        while current not in ordered:
            ordered.append(current)
            following = min(links[current] - ({previous} if previous is not None else set()))
            previous, current = (current, following)
        if not (current == ordered[0] and len(ordered) == len(links)):
            raise ValueError("Invalid exact convex hull representation")
        cones.append([normals[i] for i in ordered])
    planes = set()
    for polygon in cones:
        for a, b in zip(polygon, polygon[1:] + polygon[:1], strict=True):
            planes.add(reduced(cross(a, b), unoriented=True))

        def bisect(poly, lo, hi):
            if hi - lo <= 2:
                return
            middle = (lo + hi) // 2
            planes.add(reduced(cross(poly[lo], poly[middle]), unoriented=True))
            bisect(poly, lo, middle)
            bisect(poly, middle, hi)

        bisect(polygon, 0, len(polygon))
    planes.discard((0, 0, 0))
    return (points, cones, sorted(planes))


def clip(polygon, plane, positive):
    """Clip a convex spherical polygon using exact homogeneous intersections."""
    sign = 1 if positive else -1
    result = []
    for a, b in zip(polygon, polygon[1:] + polygon[:1], strict=True):
        da, db = (sign * dot(plane, a), sign * dot(plane, b))
        if da >= 0:
            result.append(a)
        if da * db < 0:
            ray = tuple((abs(db) * x + abs(da) * y for x, y in zip(a, b, strict=True)))
            result.append(reduced(ray))
    unique = []
    for ray in result:
        if ray not in unique:
            unique.append(ray)
    if len(unique) < 3:
        return None
    if not any(dot(unique[0], cross(unique[i], unique[i + 1])) for i in range(1, len(unique) - 1)):
        return None
    return unique


def build_tree(vertices, faces, *, node_limit=10000, depth_limit=40):
    """Construct a bounded exact normal-fan BSP with measured duplicated leaves."""
    begin = time.monotonic()
    points, cones, planes = normal_fan(vertices, faces)
    nodes, leaves, used_planes = ([], [], set())

    def visit(regions, available, depth):
        if not (depth <= depth_limit and len(nodes) < node_limit):
            raise ValueError("BSP construction limit")
        if len(regions) == 1:
            vertex = regions[0][0]
            leaves.append((vertex, depth))
            return -vertex - 1
        best = None
        for p in available:
            plane = planes[p]
            positive = negative = 0
            for _, polygon in regions:
                values = [dot(plane, ray) for ray in polygon]
                positive += max(values) > 0
                negative += min(values) < 0
            if positive == 0 or negative == 0:
                continue
            key = (max(positive, negative), positive + negative, abs(positive - negative), p)
            if best is None or key < best:
                best = key
        if not best is not None:
            raise ValueError("No available normal-fan separating plane")
        p = best[-1]
        positive, negative = ([], [])
        for vertex, polygon in regions:
            for side, output in ((True, positive), (False, negative)):
                cut = clip(polygon, planes[p], side)
                if cut is not None:
                    output.append((vertex, cut))
        if not (positive and negative):
            raise ValueError("Invalid exact convex hull representation")
        index = len(nodes)
        nodes.append(None)
        used_planes.add(p)
        remaining = [x for x in available if x != p]
        a = visit(positive, remaining, depth + 1)
        b = visit(negative, remaining, depth + 1)
        nodes[index] = (p, a, b)
        return index

    root = visit(list(enumerate(cones)), list(range(len(planes))), 0)
    rounded, rounded64, plane_low, plane_exponents = ([], [], [], [])
    for plane in planes:
        high, low, exponent = pack_plane(plane)
        rounded.append(high.astype(np.float32))
        rounded64.append(high)
        plane_low.append(low)
        plane_exponents.append(exponent)
    return {
        "vertices": np.asarray(vertices),
        "points": points,
        "planes": planes,
        "rounded": rounded,
        "rounded64": rounded64,
        "plane_low": plane_low,
        "plane_exponents": plane_exponents,
        "nodes": nodes,
        "root": root,
        "leaves": leaves,
        "used_planes": len(used_planes),
        "build_seconds": time.monotonic() - begin,
    }
