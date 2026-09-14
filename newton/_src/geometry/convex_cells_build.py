# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Validate immutable cooked hulls and build conservative direction-cell masks."""

import itertools
import math
from collections import Counter
from fractions import Fraction as F

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
        raise ValueError("Invalid closed convex extreme hull")
    edges, incident = (Counter(), [[] for _ in points])
    neighbors = [set() for _ in points]
    for face in faces:
        ia, ib, ic = (int(i) for i in face)
        if not (0 <= min(ia, ib, ic) and max(ia, ib, ic) < len(points) and (len({ia, ib, ic}) == 3)):
            raise ValueError("Invalid closed convex extreme hull")
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
        raise ValueError("Invalid closed convex extreme hull")
    seen, todo = (set(), [0])
    while todo:
        i = todo.pop()
        if i not in seen:
            seen.add(i)
            todo.extend(neighbors[i] - seen)
    if not len(seen) == len(points):
        raise ValueError("Invalid closed convex extreme hull")
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
            raise ValueError("Invalid closed convex extreme hull")
        if planes[adjacent[0]] != planes[adjacent[1]]:
            graph[i].add(j)
            graph[j].add(i)
    if not all(len(row) >= 3 for row in graph):
        raise ValueError("Invalid closed convex extreme hull")
    seen, todo = (set(), [0])
    while todo:
        i = todo.pop()
        if i not in seen:
            seen.add(i)
            todo.extend(graph[i] - seen)
    if not len(seen) == len(vertices):
        raise ValueError("Invalid closed convex extreme hull")
    return [sorted(row) for row in graph]


EPS, TINY, LOW = F(1, 2**23), F(1, 2**126), F(1, 2**60)
DELTA = 8 * EPS


def _clip(poly, a, b, c):
    if not poly:
        return []
    out = []
    prev = poly[-1]
    fp = a * prev[0] + b * prev[1] + c
    for cur in poly:
        fc = a * cur[0] + b * cur[1] + c
        if (fp >= 0) != (fc >= 0):
            t = fp / (fp - fc)
            out.append((prev[0] + t * (cur[0] - prev[0]), prev[1] + t * (cur[1] - prev[1])))
        if fc >= 0:
            out.append(cur)
        prev, fp = cur, fc
    return out


def build_masks(v, faces):
    """Construct conservative fixed16 masks over entire exact relaxed normal cones."""
    graph = edge_graph(v, faces)
    pts = [[F(float(x)) for x in row] for row in v]
    bound = sum(max(abs(row[k]) for row in pts) for k in range(3))
    if bound > 1:
        raise ValueError("Direction-cell coordinates require summed axis bounds <= 1")
    sigma = 16 * EPS * bound + 32 * TINY / LOW
    masks = np.zeros((6, 16, 16), dtype=np.uint64)
    polygons = 0
    for face in range(6):
        axis, sign = face // 2, (1 if face % 2 == 0 else -1)
        other = [k for k in range(3) if k != axis]
        for i in range(len(v)):
            lo, hi = -1 - DELTA, 1 + DELTA
            poly = [(lo, lo), (hi, lo), (hi, hi), (lo, hi)]
            for j in graph[i]:
                d = [pts[i][k] - pts[j][k] for k in range(3)]
                poly = _clip(poly, d[other[0]], d[other[1]], sign * d[axis] + sigma)
                if not poly:
                    break
            if not poly:
                continue
            polygons += 1
            xmin, xmax = min(x for x, _ in poly), max(x for x, _ in poly)
            ymin, ymax = min(y for _, y in poly), max(y for _, y in poly)
            # Floating bounds only select a superset; one extra whole cell protects conversion.
            ix0 = max(0, math.floor(float((xmin - DELTA + 1) * 8)) - 1)
            ix1 = min(15, math.floor(float((xmax + DELTA + 1) * 8)) + 1)
            iy0 = max(0, math.floor(float((ymin - DELTA + 1) * 8)) - 1)
            iy1 = min(15, math.floor(float((ymax + DELTA + 1) * 8)) + 1)
            for x in range(ix0, ix1 + 1):
                for y in range(iy0, iy1 + 1):
                    x0, x1 = F(x, 8) - 1 - DELTA, F(x + 1, 8) - 1 + DELTA
                    y0, y1 = F(y, 8) - 1 - DELTA, F(y + 1, 8) - 1 + DELTA
                    p = _clip(poly, 1, 0, -x0)
                    p = _clip(p, -1, 0, x1)
                    p = _clip(p, 0, 1, -y0)
                    p = _clip(p, 0, -1, y1)
                    if p:
                        masks[face, x, y] |= np.uint64(1 << i)
    if not np.all(masks != 0):
        raise ValueError("Direction-cell mask coverage must be nonempty")
    return masks, {"nonempty_vertex_face_cones": polygons, "sigma": float(sigma), "coordinate_l1_bound": float(bound)}
