# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""A filled movable bin of low-poly CAD hardware, cooked once per mesh to SDF.

Threaded nuts and bolts come from NVIDIA Factory (BSD-3-Clause), the asset
family used by Isaac Lab's NIST tasks. Washers and involute gears are authored
here; they are not attributed to NIST. Dimensions and clearances are in meters.
"""

import functools
import itertools
import math
import random

import numpy as np
import trimesh
import warp as wp

import newton
import newton.examples
from newton.examples.robot._clutter_container import container_mesh, handle_mesh

_FACTORY_REPO = "https://github.com/isaac-sim/IsaacGymEnvs.git"
_FACTORY_REF = "aeed298638a1f7b5421b38f5f3cc2d1079b6d9c3"
_FACTORY_FOLDER = "assets/factory/mesh/factory_nut_bolt"


def _stitch_seams(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Split CAD export T-junctions without filling the nut's physical bore."""
    edges, counts = np.unique(mesh.edges_sorted, axis=0, return_counts=True)
    boundary = edges[counts == 1]
    vertices = np.unique(boundary)
    points = mesh.vertices[vertices]
    splits = {}
    for a, b in boundary:
        ab = mesh.vertices[b] - mesh.vertices[a]
        t = (points - mesh.vertices[a]) @ ab / (ab @ ab)
        error = np.linalg.norm(points - (mesh.vertices[a] + t[:, None] * ab), axis=1)
        inside = (t > 1.0e-5) & (t < 1.0 - 1.0e-5) & (error < 1.0e-6)
        if np.any(inside):
            splits[(a, b)] = vertices[inside][np.argsort(t[inside])].tolist()
    faces = []
    for triangle in mesh.faces:
        for k in range(3):
            a, b, c = triangle[k], triangle[(k + 1) % 3], triangle[(k + 2) % 3]
            key = tuple(sorted((a, b)))
            if key in splits:
                chain = [a] + (splits[key] if a < b else splits[key][::-1]) + [b]
                faces.extend((p, q, c) for p, q in itertools.pairwise(chain))
                break
        else:
            faces.append(triangle)
    return trimesh.Trimesh(mesh.vertices, faces, process=False)


@functools.lru_cache(maxsize=4)
def _factory_mesh(kind: str, size: int) -> newton.Mesh:
    """Repair export seams, then reduce CAD triangles before SDF construction."""
    folder = newton.examples.download_external_git_folder(_FACTORY_REPO, _FACTORY_FOLDER, ref=_FACTORY_REF)
    suffix = "_subdiv_3x" if kind == "nut" else ""
    source = trimesh.load(folder / f"factory_{kind}_m{size}_loose{suffix}.obj", force="mesh", process=False)
    source.merge_vertices(merge_tex=True, merge_norm=True, digits_vertex=7)
    source.update_faces(source.unique_faces())
    source.update_faces(source.nondegenerate_faces())
    source.remove_unreferenced_vertices()
    source = _stitch_seams(source)
    if not source.is_watertight or not source.is_winding_consistent:
        raise ValueError(f"Factory {kind} M{size} has an invalid surface after seam repair.")
    # Millimeter coordinates keep the QEM's absolute tolerances below thread detail.
    source.apply_translation(-source.bounds.mean(axis=0))
    source.apply_scale(1000.0)
    mesh = source.simplify_quadric_decimation(face_count=3200, aggression=5)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    if not mesh.is_watertight or not mesh.is_winding_consistent:
        raise ValueError(f"Factory {kind} M{size} decimation damaged the closed surface.")
    if abs(mesh.volume / source.volume - 1.0) > 0.01 or mesh.euler_number != source.euler_number:
        raise ValueError(f"Factory {kind} M{size} decimation changed volume or bore topology.")
    mesh.apply_scale(0.001)
    return newton.Mesh(mesh.vertices, mesh.faces.flatten(), roughness=0.28, metallic=0.8)


def _annular_surface(angles: np.ndarray, loops: list[tuple[np.ndarray, float]]) -> newton.Mesh:
    """Join closed radial loops into one outward-oriented annular CAD surface."""
    n = len(angles)
    vertices = [(r[i] * math.cos(a), r[i] * math.sin(a), z) for r, z in loops for i, a in enumerate(angles)]
    faces = []
    for j in range(len(loops)):
        k = (j + 1) % len(loops)
        for i in range(n):
            a, b, c, d = j * n + i, j * n + (i + 1) % n, k * n + (i + 1) % n, k * n + i
            faces.extend((a, b, c, a, c, d))
    mesh = trimesh.Trimesh(vertices, np.asarray(faces).reshape(-1, 3), process=False)
    # Radial tooth-root edges share an angle; weld their coincident bore points.
    mesh.merge_vertices(digits_vertex=10)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    mesh = _stitch_seams(mesh)
    return newton.Mesh(mesh.vertices, mesh.faces.flatten(), roughness=0.3, metallic=0.8)


@functools.lru_cache(maxsize=4)
def _washer_mesh(size: int) -> newton.Mesh:
    """Revolve an eight-point beveled washer section, preserving the through bore."""
    outer, inner, half = (0.012, 0.0065, 0.0008) if size == 12 else (0.015, 0.0085, 0.001)
    angles = np.linspace(0.0, math.tau, 64, endpoint=False)
    bevel = 0.0002
    section = (
        (outer - bevel, -half),
        (outer, -half + bevel),
        (outer, half - bevel),
        (outer - bevel, half),
        (inner + bevel, half),
        (inner, half - bevel),
        (inner, -half + bevel),
        (inner + bevel, -half),
    )
    return _annular_surface(angles, [(np.full(len(angles), radius), z) for radius, z in section])


@functools.lru_cache(maxsize=4)
def _gear_mesh(size: int) -> newton.Mesh:
    """Extrude a 20-degree involute tooth profile with a beveled rim and open bore."""
    teeth = 20
    module = 0.001 if size == 12 else 0.00125
    pitch, pressure = module * teeth / 2.0, math.radians(20.0)
    root, base, tip = pitch - 1.25 * module, pitch * math.cos(pressure), pitch + module
    half_angle = math.pi / (2 * teeth)

    def flank_angle(radius):
        alpha = math.acos(min(base / radius, 1.0))
        return half_angle + math.tan(pressure) - pressure - (math.tan(alpha) - alpha)

    angles, radii = [], []
    for tooth in range(teeth):
        center = tooth * math.tau / teeth
        flank = np.linspace(base, tip, 4)
        profile = [(root, -flank_angle(base))]
        profile.extend((r, -flank_angle(r)) for r in flank)
        profile.append((tip, 0.0))
        profile.extend((r, flank_angle(r)) for r in flank[::-1])
        profile.append((root, flank_angle(base)))
        for radius, angle in profile:
            radii.append(radius)
            angles.append(center + angle)
    radius = np.asarray(radii)
    bore, half, bevel = module * 3.2, module * 3.0, module * 0.25
    loops = [
        (radius - bevel, -half),
        (radius, -half + bevel),
        (radius, half - bevel),
        (radius - bevel, half),
        (np.full(len(radius), bore + bevel), half),
        (np.full(len(radius), bore), half - bevel),
        (np.full(len(radius), bore), -half + bevel),
        (np.full(len(radius), bore + bevel), -half),
    ]
    return _annular_surface(np.asarray(angles), loops)


def _hardware_meshes() -> dict[tuple[str, int], newton.Mesh]:
    """Share eight low-poly CAD meshes across all hardware instances."""
    return {
        (kind, size): (
            _factory_mesh(kind, size)
            if kind in ("bolt", "nut")
            else _washer_mesh(size)
            if kind == "washer"
            else _gear_mesh(size)
        )
        for kind in ("bolt", "nut", "washer", "gear")
        for size in (12, 16)
    }


def build_hardware(
    builder: newton.ModelBuilder,
    *,
    center: tuple[float, float, float] = (0.55, 0.0, 0.0),
    count: int = 256,
    seed: int = 42,
) -> dict:
    """Add a movable 240 x 160 x 55 mm bin filled with loose CAD hardware.

    The bin rests on the table and is moved only through handle contact. A
    nonintersecting lattice starts above the rim and settles under gravity; no
    settled pose or object attachment is prescribed. ``sdf_meshes`` requests
    one persistent SDF cook per unique mesh, before world replication.
    """
    if count < 0:
        raise ValueError("Hardware count must be nonnegative.")
    cx, cy, cz = center
    cz += 0.0006  # Start just clear of the supporting tabletop.
    gap = 0.0003
    steel = newton.ModelBuilder.ShapeConfig(density=7800.0, mu=0.55, gap=gap)
    plastic = newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.08, gap=gap)
    grip = newton.ModelBuilder.ShapeConfig(density=1000.0, mu=1.0, gap=gap)
    bin_mesh = container_mesh(half_extents=(0.12, 0.08), height=0.055, wall=0.004, floor=0.004, radius=0.012)
    handle = handle_mesh()
    container = builder.add_body(xform=wp.transform((cx, cy, cz), wp.quat_identity()), label="hardware_bin")
    builder.add_shape_mesh(container, mesh=bin_mesh, cfg=plastic, color=(0.10, 0.23, 0.34), label="hardware_bin_shell")
    handle_offset = (0.0, -0.145, 0.049)
    builder.add_shape_mesh(
        container,
        mesh=handle,
        xform=wp.transform(handle_offset, wp.quat_identity()),
        cfg=grip,
        color=(0.10, 0.23, 0.34),
        label="hardware_bin_handle",
    )
    meshes = _hardware_meshes() if count else {}
    kinds = ("bolt", "nut", "washer", "gear")
    colors = ((0.58, 0.62, 0.67), (0.72, 0.74, 0.76), (0.80, 0.82, 0.85), (0.57, 0.42, 0.18))
    rng = random.Random(seed)
    bodies, object_kinds, initial_bounds, initial_aabbs = [], [], [], []
    # Pack long bolts on their sides, and flat pieces into shallow layers. This
    # starts a compact pile above the bin, rather than a half-meter shower.
    placement = {}
    height = cz + 0.0055
    for kind, columns, rows, pitch_x, pitch_y, half_height in (
        ("bolt", 5, 5, 0.046, 0.027, 0.0121),
        ("nut", 7, 4, 0.030, 0.030, 0.0066),
        ("washer", 7, 4, 0.032, 0.032, 0.0011),
        ("gear", 7, 4, 0.030, 0.030, 0.0038),
    ):
        members = list(range(kinds.index(kind), count, 4))
        for start in range(0, len(members), columns * rows):
            for slot, index in enumerate(members[start : start + columns * rows]):
                row, column = divmod(slot, columns)
                placement[index] = (
                    cx + (column - (columns - 1) / 2) * pitch_x,
                    cy + (row - (rows - 1) / 2) * pitch_y,
                    height + half_height,
                )
            height += 2 * half_height + 0.001
    volume = 0.0
    for i in range(count):
        kind = kinds[i % 4]
        size = 12 if (i // 4) % 2 == 0 else 16
        mesh = meshes[(kind, size)]
        position = placement[i]
        yaw = rng.choice((0.0, math.pi)) if kind == "bolt" else rng.uniform(-math.pi, math.pi)
        rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw)
        if kind == "bolt":
            rotation *= wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), math.pi / 2)
        label = f"hardware_{kind}_m{size}_{i:03d}"
        body = builder.add_body(xform=wp.transform(position, rotation), label=label)
        builder.add_shape_mesh(body, mesh=mesh, cfg=steel, color=colors[i % 4], label=label)
        vertices = np.asarray(mesh.vertices)
        rotated = vertices @ np.asarray(wp.quat_to_matrix(rotation)).reshape(3, 3).T
        radius = float(np.linalg.norm(rotated[:, :2], axis=1).max())
        half_height = float(np.abs(rotated[:, 2]).max())
        bodies.append(body)
        object_kinds.append(kind)
        initial_bounds.append((position, radius, half_height))
        initial_aabbs.append((tuple(rotated.min(axis=0) + position), tuple(rotated.max(axis=0) + position)))
        volume += mesh.mass
    target = (cx + handle_offset[0], cy + handle_offset[1], cz + handle_offset[2])
    return {
        "object_bodies": bodies,
        "object_kinds": object_kinds,
        "initial_bounds": initial_bounds,
        "initial_aabbs": initial_aabbs,
        "target": target,
        "container_body": container,
        "handle_target": target,
        "handle_width": 0.024,
        "motion": "container_slide",
        "settle_time": 3.0,
        "bin_bounds": ((cx - 0.116, cy - 0.076, cz + 0.004), (cx + 0.116, cy + 0.076, cz + 0.055)),
        "solid_volume_m3": volume,
        "interior_volume_m3": 0.232 * 0.152 * 0.051,
        "sdf_meshes": [(bin_mesh, 0.0008), (handle, 0.0006)] + [(mesh, 0.0001) for mesh in meshes.values()],
        "asset_provenance": f"Factory nuts/bolts @ {_FACTORY_REF}; authored involute gears/beveled washers",
    }
