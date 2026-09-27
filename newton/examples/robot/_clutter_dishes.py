# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Original, metre-scale parametric CAD for a compact dish-drying rack.

Collision and display use identical meshes: revolved ceramic shells, profiled
cutlery and swept rack wires. No downloaded assets or convex hulls are used.
The runner cooks ``sdf_meshes`` once on its selected GPU before finalization.
"""

import math
import random
from collections import Counter
from functools import lru_cache
from itertools import pairwise

import numpy as np
import warp as wp

import newton
from newton.examples.robot._clutter_container import handle_mesh


def _positive_mesh(vertices, faces):
    """Orient a closed surface outward and retain modest CAD tessellation."""
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    triangle = vertices[faces]
    volume = np.einsum("ij,ij->i", triangle[:, 0], np.cross(triangle[:, 1], triangle[:, 2])).sum()
    if volume < 0:
        faces = faces[:, ::-1].copy()
    return vertices, faces


def _ceramic(radius, depth, thickness, *, bowl=False):
    """Revolve an open shell with a rounded lip and low ceramic foot ring."""
    r = np.linspace(0.0, radius, 17)
    u = r / radius
    z = depth * (u**2.5 if bowl else np.maximum(0.0, (u - 0.52) / 0.48) ** 2)
    foot = 0.0015 * np.exp(-(((u - 0.47) / 0.07) ** 2))
    profile = list(zip(r, z, strict=True))
    for angle in np.linspace(0.0, math.pi, 7)[1:]:
        profile.append((radius + thickness / 2 * math.sin(angle), depth - thickness / 2 * (1 - math.cos(angle))))
    profile.extend(zip(r[-2::-1], (z - thickness - foot)[-2::-1], strict=True))
    vertices, rings, faces = [], [], []
    for radius_i, height in profile:
        ring = []
        for angle in np.linspace(0.0, 2 * math.pi, 64 if radius_i > 1.0e-10 else 1, endpoint=False):
            ring.append(len(vertices))
            vertices.append((radius_i * math.cos(angle), radius_i * math.sin(angle), height))
        rings.append(ring)
    for before, after in pairwise(rings):
        for j in range(max(len(before), len(after))):
            a, b = before[j % len(before)], before[(j + 1) % len(before)]
            c, d = after[j % len(after)], after[(j + 1) % len(after)]
            if a != b:
                faces.append((a, b, c))
            if c != d:
                faces.append((b, d, c))
    vertices, faces = _positive_mesh(vertices, faces)
    vertices[:, 2] -= (vertices[:, 2].max() + vertices[:, 2].min()) / 2
    return vertices, faces


def _triangulate(outline):
    """Ear-clip a simple counterclockwise CAD outline, including tine gaps."""
    points = np.asarray(outline, dtype=np.float64)
    active, triangles = list(range(len(points))), []

    def cross(a, b):
        return a[0] * b[1] - a[1] * b[0]

    while len(active) > 3:
        for j, current in enumerate(active):
            before, after = active[j - 1], active[(j + 1) % len(active)]
            a, b, c = points[[before, current, after]]
            if cross(b - a, c - b) <= 1.0e-14:
                continue
            p = points[[i for i in active if i not in (before, current, after)]]
            inside = (b[0] - a[0]) * (p[:, 1] - a[1]) - (b[1] - a[1]) * (p[:, 0] - a[0]) >= -1.0e-14
            inside &= (c[0] - b[0]) * (p[:, 1] - b[1]) - (c[1] - b[1]) * (p[:, 0] - b[0]) >= -1.0e-14
            inside &= (a[0] - c[0]) * (p[:, 1] - c[1]) - (a[1] - c[1]) * (p[:, 0] - c[0]) >= -1.0e-14
            if not np.any(inside):
                triangles.append((before, current, after))
                active.pop(j)
                break
        else:
            raise ValueError("CAD outline is degenerate or self-intersecting")
    triangles.append(tuple(active))
    return points, triangles


def _profiled_solid(outline, height, thickness, *, refine=True):
    """Thicken a shaped outline with controlled curved-surface chord error."""
    xy, faces = _triangulate(outline)
    if refine:
        xy, faces = _refine_planar(xy, faces)
    edges = [(a, b) for face in faces for a, b in zip(face, (*face[1:], face[0]), strict=True)]
    counts = Counter(tuple(sorted(edge)) for edge in edges)
    boundary = [(a, b) for a, b in edges if counts[tuple(sorted((a, b)))] == 1]
    z = np.asarray(height(xy[:, 0], xy[:, 1])) + np.zeros(len(xy))
    size = len(xy)
    vertices = np.concatenate((np.column_stack((xy, z + thickness / 2)), np.column_stack((xy, z - thickness / 2))))
    surface = list(faces) + [(c + size, b + size, a + size) for a, b, c in faces]
    for a, b in boundary:
        surface.extend(((a, a + size, b), (b, a + size, b + size)))
    return _positive_mesh(vertices, surface)


def _refine_planar(points, faces):
    """Clip CAD facets to a 10 x 3 mm grid without filling concave outlines."""
    vertices, indices, lookup = [], [], {}
    pitch = np.array((0.010, 0.003))

    def clip(polygon, axis, bound, side):
        result = []
        for a, b in zip(polygon, polygon[1:] + polygon[:1], strict=True):
            da, db = side * (a[axis] - bound), side * (b[axis] - bound)
            if da >= -1.0e-12:
                result.append(a)
            if (da < -1.0e-12) != (db < -1.0e-12):
                result.append(a + da / (da - db) * (b - a))
        return result

    for face in faces:
        triangle = points[list(face)]
        lower = np.floor(triangle.min(axis=0) / pitch).astype(int)
        upper = np.floor(triangle.max(axis=0) / pitch).astype(int)
        for ix in range(lower[0], upper[0] + 1):
            for iy in range(lower[1], upper[1] + 1):
                polygon = list(triangle)
                for axis, cell in enumerate((ix, iy)):
                    polygon = clip(polygon, axis, cell * pitch[axis], 1)
                    polygon = clip(polygon, axis, (cell + 1) * pitch[axis], -1)
                for j in range(1, len(polygon) - 1):
                    facet = np.asarray((polygon[0], polygon[j], polygon[j + 1]))
                    a, b = facet[1] - facet[0], facet[2] - facet[0]
                    if abs(a[0] * b[1] - a[1] * b[0]) < 1.0e-14:
                        continue
                    face_indices = []
                    for point in facet:
                        # Weld sub-micrometre clipping slivers, far below SDF pitch.
                        key = tuple(np.round(point, 6))
                        if key not in lookup:
                            lookup[key] = len(vertices)
                            vertices.append(point)
                        face_indices.append(lookup[key])
                    if len(set(face_indices)) == 3:
                        indices.append(tuple(face_indices))
    return np.asarray(vertices), indices


def _round_outline(points):
    """Cut each corner once, retaining tine slots rather than filling them."""
    p = np.asarray(points)
    q = np.roll(p, -1, axis=0)
    return np.stack((0.8 * p + 0.2 * q, 0.2 * p + 0.8 * q), axis=1).reshape(-1, 2)


def _cutlery(kind):
    """Make stamped cutlery with a real concave spoon and four curved tines."""
    lower = [(-0.078, -0.002), (-0.074, -0.0048), (-0.058, -0.0045), (-0.022, -0.0028), (0.014, -0.0028)]
    upper = [(x, -y) for x, y in lower[::-1]]
    if kind == "fork":
        head = [(0.025, -0.0074)]
        for j in range(4):
            y = (j - 1.5) * 0.0037
            head.extend(
                (
                    (0.040, y - 0.001),
                    (0.073, y - 0.001),
                    (0.079, y - 0.0006),
                    (0.079, y + 0.0006),
                    (0.073, y + 0.001),
                    (0.040, y + 0.001),
                )
            )
        head.append((0.025, 0.0074))
        return _profiled_solid(
            _round_outline(lower + head + upper), lambda x, y: 0.0035 * np.clip((x - 0.022) / 0.056, 0, 1) ** 2, 0.0017
        )
    if kind == "spoon":
        angle = np.linspace(-2.72, 2.72, 30)
        head = list(zip(0.052 + 0.026 * np.cos(angle), 0.0095 * np.sin(angle), strict=True))
        return _profiled_solid(
            _round_outline(lower + head + upper),
            lambda x, y: -0.0045 * np.maximum(0.0, 1 - ((x - 0.052) / 0.026) ** 2 - (y / 0.0095) ** 2),
            0.0015,
        )
    if kind == "knife":
        outline = [
            *lower,
            (0.025, -0.0048),
            (0.075, -0.0048),
            (0.079, -0.002),
            (0.075, 0.007),
            (0.068, 0.008),
            (0.030, 0.007),
            (0.016, 0.003),
            *upper,
        ]
        return _profiled_solid(_round_outline(outline), lambda x, y: 0.0 * x, 0.0019)
    raise ValueError(f"Unknown cutlery: {kind}")


def _tube(points, radius, *, closed=False, sides=8):
    """Sweep a circular wire around a centreline with flat end caps."""
    points = np.asarray(points, dtype=np.float64)
    vertices, faces = [], []
    for i, center in enumerate(points):
        before = points[(i - 1) % len(points)] if closed or i else points[0]
        after = points[(i + 1) % len(points)] if closed or i + 1 < len(points) else points[-1]
        tangent = after - before
        tangent /= np.linalg.norm(tangent)
        normal = np.cross(tangent, (0, 0, 1) if abs(tangent[2]) < 0.9 else (0, 1, 0))
        normal /= np.linalg.norm(normal)
        binormal = np.cross(tangent, normal)
        r = radius if np.isscalar(radius) else radius[i]
        for angle in np.linspace(0, 2 * math.pi, sides, endpoint=False):
            vertices.append(center + r * (math.cos(angle) * normal + math.sin(angle) * binormal))
    for i in range(len(points) if closed else len(points) - 1):
        for j in range(sides):
            a, b = i * sides + j, i * sides + (j + 1) % sides
            c, d = ((i + 1) % len(points)) * sides + j, ((i + 1) % len(points)) * sides + (j + 1) % sides
            faces.extend(((a, b, c), (b, d, c)))
    if not closed:
        for i in (0, len(points) - 1):
            center = len(vertices)
            vertices.append(points[i])
            for j in range(sides):
                triangle = (center, i * sides + (j + 1) % sides, i * sides + j)
                faces.append(triangle if i == 0 else triangle[::-1])
    return _positive_mesh(vertices, faces)


def _rounded_rectangle(hx, hy, radius, z=0.0):
    points = []
    for x, y, offset in (
        (hx - radius, hy - radius, 0),
        (-hx + radius, hy - radius, 90),
        (-hx + radius, -hy + radius, 180),
        (hx - radius, -hy + radius, 270),
    ):
        for angle in np.radians(np.linspace(offset, offset + 90, 7, endpoint=False)):
            points.append((x + radius * math.cos(angle), y + radius * math.sin(angle), z))
    return np.asarray(points)


def _combine(parts):
    vertices, faces, offset = [], [], 0
    for v, f in parts:
        vertices.append(v)
        faces.append(f + offset)
        offset += len(v)
    return np.concatenate(vertices), np.concatenate(faces)


@lru_cache(maxsize=8)
def dish_cad_geometry(kind: str):
    """Return original low-polygon CAD vertices [m] and triangle indices."""
    if kind == "plate":
        return _ceramic(0.071, 0.006, 0.0026)
    if kind == "bowl":
        return _ceramic(0.0615, 0.012, 0.0023, bowl=True)
    if kind == "chopstick":
        return _tube(
            [(-0.076, 0, 0), (-0.048, 0, 0), (0.035, 0, 0), (0.076, 0, 0)], (0.0024, 0.00235, 0.0018, 0.00125), sides=10
        )
    return _cutlery(kind)


def build_dishes(
    builder: newton.ModelBuilder,
    *,
    center: tuple[float, float, float] = (0.55, 0.0, 0.0),
    count: int = 40,
    seed: int = 42,
) -> dict:
    """Add loose CAD crockery/cutlery in a free 0.35 x 0.28 m rack [m].

    These original parametric models are not dimensionally certified product
    scans. Small side plates suit the compact Franka workspace. Shared SDF grids
    resolve ceramic walls with >=4 cells and 1.7 mm fork gaps with >=8 cells.
    The extra container body is excluded from ``count`` and moves through a
    physical handle pinch, not a prescribed pose or a grasp constraint.
    """
    if count < 1:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    cx, cy, cz = center
    cfgs = {
        "ceramic": newton.ModelBuilder.ShapeConfig(density=2200.0, mu=0.55, gap=0.0003),
        "steel": newton.ModelBuilder.ShapeConfig(density=7800.0, mu=0.4, gap=0.0003),
        "wood": newton.ModelBuilder.ShapeConfig(density=650.0, mu=0.6, gap=0.0003),
        "rack_steel": newton.ModelBuilder.ShapeConfig(density=7800.0, mu=0.08, gap=0.0003),
    }
    rack_cfg = newton.ModelBuilder.ShapeConfig(density=1100.0, mu=0.08, gap=0.0003)
    silver = (0.63, 0.68, 0.72)
    sdf_meshes, rack_shapes = [], []
    rack_z = cz + 0.0006
    container = builder.add_body(xform=wp.transform(p=(cx, cy, rack_z)), label="dish_rack")

    def mesh_shape(body, geometry, material, color, *, position=(0, 0, 0), voxel_size=0.00045):
        vertices, faces = geometry
        mesh = newton.Mesh(
            vertices,
            faces.ravel(),
            roughness=0.23 if material == "ceramic" else 0.28,
            metallic=0.85 if material in ("steel", "rack_steel") else 0.0,
        )
        shape = builder.add_shape_mesh(
            body, mesh=mesh, xform=wp.transform(p=position), cfg=cfgs.get(material, rack_cfg), color=color
        )
        sdf_meshes.append((mesh, voxel_size))
        return shape, mesh

    dish_count = min(12, max(1, count // 3))
    pitch = 0.026
    wires = [_tube(_rounded_rectangle(0.168, 0.131, 0.016, z), 0.0025, closed=True) for z in (0.012, 0.042)]
    for y in (-0.106, -0.074, -0.042, -0.010, 0.022, 0.054, 0.086, 0.114):
        wires.append(_tube([(-0.153, y, 0.009), (0.153, y, 0.009)], 0.002))
    for x in (-0.151, 0.151):
        wires.append(_tube([(x, -0.118, 0.010), (x, -0.118, 0.040), (x, 0.118, 0.040), (x, 0.118, 0.010)], 0.0025))
    for i in range(dish_count + 1):
        x = (i - dish_count / 2) * pitch
        wires.append(
            _tube(
                [
                    (x, 0.005, 0.100),
                    (x, 0.005, 0.024),
                    (x, 0.010, 0.014),
                    (x, 0.081, 0.014),
                    (x, 0.086, 0.024),
                    (x, 0.086, 0.100),
                ],
                0.0018,
            )
        )
    rack_shapes.append(mesh_shape(container, _combine(wires), "rack_steel", silver, voxel_size=0.0006)[0])
    tray = _profiled_solid(_rounded_rectangle(0.175, 0.140, 0.015)[:, :2], lambda x, y: 0.0015, 0.003, refine=False)
    rack_shapes.append(mesh_shape(container, tray, "rack", (0.14, 0.21, 0.25), voxel_size=0.0006)[0])
    handle = handle_mesh()
    handle_local = (0.0, -0.205, 0.037)
    rack_shapes.append(
        builder.add_shape_mesh(
            container,
            mesh=handle,
            xform=wp.transform(p=handle_local),
            cfg=newton.ModelBuilder.ShapeConfig(density=1100.0, mu=1.0, gap=0.0003),
            color=(0.14, 0.21, 0.25),
            label="dish_rack_handle",
        )
    )
    sdf_meshes.append((handle, 0.0008))

    meshes = {}
    for kind in ("plate", "bowl", "fork", "spoon", "knife", "chopstick"):
        vertices, faces = dish_cad_geometry(kind)
        meshes[kind] = newton.Mesh(
            vertices,
            faces.ravel(),
            roughness=0.23 if kind in ("plate", "bowl") else 0.3,
            metallic=0.85 if kind in ("fork", "spoon", "knife") else 0.0,
        )
        sdf_meshes.append((meshes[kind], 0.00045 if kind in ("plate", "bowl") else 0.0002))

    bodies, kinds, positions = [], [], []
    for i in range(count):
        if i < dish_count:
            kind = "bowl" if i % 3 == 2 else "plate"
            radius = 0.0723 if kind == "plate" else 0.06265
            position = (cx + (i - (dish_count - 1) / 2) * pitch, cy + 0.043, rack_z + 0.013 + radius)
            rotation = wp.quat_from_axis_angle(wp.vec3(0, 1, 0), math.pi / 2)
            color = (0.88, 0.89, 0.83) if kind == "bowl" else ((0.70, 0.82, 0.85) if i % 2 else (0.87, 0.79, 0.64))
            material = "ceramic"
        else:
            layer, slot = divmod(i - dish_count, 8)
            kind = ("fork", "spoon", "knife", "chopstick")[(i - dish_count) % 4]
            bottom = float(dish_cad_geometry(kind)[0][:, 2].min())
            position = (
                cx + (-0.082 if slot % 2 == 0 else 0.082),
                cy - 0.116 + 0.023 * (slot // 2),
                rack_z + 0.013 - bottom + 0.011 * layer,
            )
            rotation = wp.quat_from_axis_angle(wp.vec3(0, 0, 1), rng.uniform(-0.012, 0.012))
            material, color = ("wood", (0.47, 0.27, 0.11)) if kind == "chopstick" else ("steel", silver)
        body = builder.add_body(xform=wp.transform(p=position, q=rotation), label=f"{kind}_{i:03d}")
        builder.add_shape_mesh(body, mesh=meshes[kind], cfg=cfgs[material], color=color)
        bodies.append(body)
        kinds.append(kind)
        positions.append(position)

    handle_target = (cx, cy + handle_local[1], rack_z + handle_local[2])
    return {
        "object_bodies": bodies,
        "target": handle_target,
        "motion": "container_slide",
        "container_body": container,
        "container_initial_position": (cx, cy, rack_z),
        "handle_target": handle_target,
        "handle_center": handle_target,
        "handle_width": 0.024,
        "handle_half_width": 0.012,
        "slide_axis": (1.0, 0.0, 0.0),
        "slide_distance": 0.05,
        "positions": positions,
        "object_kinds": kinds,
        "rack_shapes": rack_shapes,
        "dish_count": dish_count,
        "bounds": ((cx - 0.175, cy - 0.217, cz), (cx + 0.175, cy + 0.14, cz + 0.17)),
        "sdf_meshes": sdf_meshes,
        "asset_provenance": "original parametric CAD (Apache-2.0), metres",
    }
