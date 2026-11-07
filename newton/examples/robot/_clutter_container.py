# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Low-polygon, watertight CAD profiles for the two movable containers."""

import math

import numpy as np

import newton


def _rounded_rectangle(hx: float, hy: float, radius: float, z: float) -> np.ndarray:
    """Sample a counterclockwise rounded rectangle, with six edges per corner."""
    points = []
    for quadrant in range(4):
        angle = quadrant * math.pi / 2
        cx = (hx - radius) * (1 if quadrant in (0, 3) else -1)
        cy = (hy - radius) * (1 if quadrant in (0, 1) else -1)
        for step in range(7):
            theta = angle + step * math.pi / 12
            points.append((cx + radius * math.cos(theta), cy + radius * math.sin(theta), z))
    return np.asarray(points, dtype=np.float32)


def _profile_mesh(profiles: list[tuple[float, float, float, float]], *, closed: bool) -> newton.Mesh:
    """Connect rounded loops; close either a shell's caps or an annular profile."""
    loops = [_rounded_rectangle(*profile) for profile in profiles]
    vertices = np.concatenate(loops)
    width = len(loops[0])
    triangles = []
    for layer in range(len(loops) if closed else len(loops) - 1):
        next_layer = (layer + 1) % len(loops)
        for index in range(width):
            following = (index + 1) % width
            a, b = layer * width + index, layer * width + following
            c, d = next_layer * width + index, next_layer * width + following
            triangles.extend(((a, b, d), (a, d, c)))
    if not closed:
        # Bottom faces down; the inside floor faces up. Both caps are convex.
        for index in range(1, width - 1):
            triangles.append((0, index + 1, index))
            offset = (len(loops) - 1) * width
            triangles.append((offset, offset + index, offset + index + 1))
    return newton.Mesh(vertices, np.asarray(triangles, dtype=np.int32).flatten(), roughness=0.35, metallic=0.65)


def container_mesh(
    *,
    half_extents: tuple[float, float] = (0.20, 0.15),
    height: float = 0.026,
    wall: float = 0.004,
    floor: float = 0.005,
    radius: float = 0.018,
) -> newton.Mesh:
    """Make an open rounded container with a solid floor and bevelled rim [m].

    Eight parallel profiles express the entire watertight shape. Geometry is
    created at its bottom plane, not at its center of mass; the body origin may
    therefore sit directly above a tabletop. The inside is genuinely hollow.
    """
    hx, hy = half_extents
    if not (0 < wall < radius < min(hx, hy) and 0 < floor < height):
        raise ValueError("Container dimensions must leave a positive hollow interior")
    bevel = min(wall / 4, floor / 4, (height - floor) / 4)
    profiles = [
        (hx - bevel, hy - bevel, radius - bevel, 0.0),
        (hx, hy, radius, bevel),
        (hx, hy, radius, height - bevel),
        (hx - bevel, hy - bevel, radius - bevel, height),
        (hx - wall + bevel, hy - wall + bevel, radius - wall + bevel, height),
        (hx - wall, hy - wall, radius - wall, height - bevel),
        (hx - wall, hy - wall, radius - wall, floor + bevel),
        (hx - wall - bevel, hy - wall - bevel, radius - wall - bevel, floor),
    ]
    return _profile_mesh(profiles, closed=False)


def handle_mesh() -> newton.Mesh:
    """Make a rounded loop handle whose front grip bar is centered at the origin.

    The 150 mm-wide loop has a 24 x 16 mm grip section. Its rear mounting bar
    is 65 mm along +Y, leaving 41 mm of finger clearance inside the handle.
    A downward Franka gripper closes along Y; use 30 mm open / 7 mm closed
    finger-joint targets. The handle is one watertight hollow mesh, not a weld
    or a grasp constraint. Its mounting bar overlaps the container rim slightly.
    """
    profiles = [
        (0.074, 0.0435, 0.013, -0.008),
        (0.075, 0.0445, 0.014, -0.007),
        (0.075, 0.0445, 0.014, 0.007),
        (0.074, 0.0435, 0.013, 0.008),
        (0.058, 0.0215, 0.009, 0.008),
        (0.057, 0.0205, 0.008, 0.007),
        (0.057, 0.0205, 0.008, -0.007),
        (0.058, 0.0215, 0.009, -0.008),
    ]
    mesh = _profile_mesh(profiles, closed=True)
    # The front bar, rather than the loop center, defines the grasp frame.
    return mesh.copy(vertices=mesh.vertices + np.array((0.0, 0.0325, 0.0)), recompute_inertia=True)
