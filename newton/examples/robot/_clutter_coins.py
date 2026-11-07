# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""A coin tower on a free tray that Franka grasps and slides by its handle."""

import math
import random

import warp as wp

import newton
from newton.examples.robot._clutter_container import container_mesh, handle_mesh


def build_coins(
    builder: newton.ModelBuilder,
    *,
    center: tuple[float, float, float] = (0.55, 0.0, 0.0),
    count: int = 520,
    seed: int = 42,
) -> dict:
    """Add ``count`` loose coins and a movable 0.40 x 0.30 m tray [m].

    The extra tray body is not included in ``count``. Its negative-Y handle
    accepts a downward Franka pinch; motion comes only from finger contact.
    Coins are 25 mm wide and 2.8 mm thick, with 0.3 mm initial clearance.
    The default is a 64-coin bed plus 38 complete, staggered 12-coin layers;
    increasing count makes the tower taller without widening its footprint.
    The returned mesh SDF requests are cooked by the shared example runner.
    """
    if count < 1:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    cx, cy, cz = center
    cfg = newton.ModelBuilder.ShapeConfig(density=7800.0, mu=0.4, gap=0.0003)
    tray_cfg = newton.ModelBuilder.ShapeConfig(density=1100.0, mu=0.08, gap=0.0003)
    handle_cfg = newton.ModelBuilder.ShapeConfig(density=1100.0, mu=1.0, gap=0.0003)
    tray_color = (0.18, 0.23, 0.28)
    tray_z = cz + 0.0006
    tray_body = builder.add_body(xform=wp.transform(p=(cx, cy, tray_z)), label="coin_tray")
    tray = container_mesh()
    handle = handle_mesh()
    tray.metallic = handle.metallic = 0.15
    handle_local = (0.0, -0.215, 0.024)
    tray_shapes = [
        builder.add_shape_mesh(tray_body, mesh=tray, cfg=tray_cfg, color=tray_color, label="coin_tray_shell"),
        builder.add_shape_mesh(
            tray_body,
            mesh=handle,
            xform=wp.transform(p=handle_local),
            cfg=handle_cfg,
            color=tray_color,
            label="coin_tray_handle",
        ),
    ]

    radius, thickness, clearance = 0.0125, 0.0028, 0.0003
    pitch, ring_radius, ring_size = 0.027, 0.052, 12
    bed_count = min(count, 64)
    phase = rng.uniform(0.0, 2.0 * math.pi)
    bodies = []
    positions = []
    for i in range(count):
        if i < bed_count:
            row, column = divmod(i, 8)
            x = (column - 3.5 + 0.5 * (row % 2) - 0.25) * pitch
            y = (row - 3.5) * pitch * math.sqrt(3.0) / 2.0
            layer = 0
        else:
            layer, slot = divmod(i - bed_count, ring_size)
            angle = phase + 2.0 * math.pi * (slot + 0.5 * (layer % 2)) / ring_size
            x, y = ring_radius * math.cos(angle), ring_radius * math.sin(angle)
            layer += 1
        position = (cx + x, cy + y, tray_z + 0.005 + clearance + thickness / 2 + layer * (thickness + clearance))
        body = builder.add_body(xform=wp.transform(p=position), label=f"coin_{i:03d}")
        shade = rng.uniform(0.82, 1.0)
        builder.add_shape_cylinder(
            body,
            radius=radius,
            half_height=thickness / 2,
            cfg=cfg,
            color=(0.80 * shade, 0.59 * shade, 0.24 * shade),
            label=f"coin_{i:03d}",
        )
        bodies.append(body)
        positions.append(position)

    tower_layers = math.ceil((count - bed_count) / ring_size)
    top = tray_z + 0.005 + (tower_layers + 1) * (thickness + clearance)
    handle_target = (cx, cy + handle_local[1], tray_z + handle_local[2])
    return {
        "object_bodies": bodies,
        "target": handle_target,
        "motion": "container_slide",
        "container_body": tray_body,
        "tray_body": tray_body,
        "container_initial_position": (cx, cy, tray_z),
        "handle_target": handle_target,
        "handle_center": handle_target,
        "handle_width": 0.024,
        "handle_half_width": 0.012,
        "slide_axis": (1.0, 0.0, 0.0),
        "slide_distance": 0.05,
        "sdf_meshes": [(tray, 0.001), (handle, 0.0008)],
        "positions": positions,
        "tray_shapes": tray_shapes,
        "bed_count": bed_count,
        "tower_layers": tower_layers,
        "coin_radius": radius,
        "coin_thickness": thickness,
        "bounds": ((cx - 0.20, cy - 0.227, cz), (cx + 0.20, cy + 0.15, max(top, tray_z + 0.032))),
    }
