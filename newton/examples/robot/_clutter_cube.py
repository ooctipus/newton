# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Build a tightly fitted, chamfered, contact-coupled 3x3 cube surrogate.

The joint graph follows MuJoCo's official cube: six center hinges and twenty
core-centered ball joints. Cubie contacts couple each turning layer; there are
no layer welds, target rotations, or runtime pose rewrites. The small clearance
and chamfered faces matter: widely separated boxes do not form a working cube.
This remains a surrogate, not the internal tracks of a commercial mechanism.

Reference (Apache-2.0): https://github.com/google-deepmind/mujoco/blob/
5ceb72b1bbe87612a2859304d2c22a15cd3c180d/model/cube/cube_3x3x3.xml
Meshes below are generated from compact CAD profiles rather than copied assets.
"""

from itertools import combinations, product

import numpy as np
import warp as wp

import newton


def _cubie_mesh(half: float, chamfer: float) -> newton.Mesh:
    """Construct a 24-vertex chamfered cube with six faces and bevelled edges."""
    inset = half - chamfer
    vertices, triangles, indices = [], [], {}

    def face(points, normal):
        points = np.asarray(points)
        if np.dot(np.cross(points[1] - points[0], points[2] - points[0]), normal) < 0:
            points = points[::-1]
        polygon = []
        for point in points:
            key = tuple(point)
            if key not in indices:
                indices[key] = len(vertices)
                vertices.append(point)
            polygon.append(indices[key])
        for index in range(1, len(polygon) - 1):
            triangles.append((polygon[0], polygon[index], polygon[index + 1]))

    for axis in range(3):
        u, v = [other for other in range(3) if other != axis]
        for sign in (-1, 1):
            points = []
            for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                point = np.zeros(3)
                point[axis], point[u], point[v] = sign * half, a * inset, b * inset
                points.append(point)
            face(points, np.eye(3)[axis] * sign)
    for first, second in combinations(range(3), 2):
        third = 3 - first - second
        for a, b in product((-1, 1), repeat=2):
            points = []
            for outer_axis, end in ((first, -1), (second, -1), (second, 1), (first, 1)):
                point = np.zeros(3)
                point[first], point[second], point[third] = a * inset, b * inset, end * inset
                point[outer_axis] = (a if outer_axis == first else b) * half
                points.append(point)
            face(points, np.eye(3)[first] * a + np.eye(3)[second] * b)
    for signs in product((-1, 1), repeat=3):
        points = []
        for axis in range(3):
            point = np.asarray(signs) * inset
            point[axis] = signs[axis] * half
            points.append(point)
        face(points, signs)
    return newton.Mesh(vertices, np.asarray(triangles).flatten(), roughness=0.28)


def _facelet_mesh() -> newton.Mesh:
    """Make one rounded, render-only colored inset in its local XY plane."""
    half, radius = 0.0075, 0.0012
    vertices = [(0.0, 0.0, 0.0)]
    for quadrant in range(4):
        cx = (half - radius) * (1 if quadrant in (0, 3) else -1)
        cy = (half - radius) * (1 if quadrant in (0, 1) else -1)
        for angle in np.linspace(quadrant * np.pi / 2, (quadrant + 1) * np.pi / 2, 7):
            vertices.append((cx + radius * np.cos(angle), cy + radius * np.sin(angle), 0.0))
    triangles = [(0, index + 1, (index + 1) % 28 + 1) for index in range(28)]
    return newton.Mesh(vertices, np.asarray(triangles).flatten(), compute_inertia=False, roughness=0.22)


def build_cube(
    builder: newton.ModelBuilder,
    *,
    center: tuple[float, float, float] = (0.55, 0.0, 0.08),
    count: int = 26,
    seed: int = 42,
    clearance: float = 0.00008,
    armature: float = 0.0001,
    damping: float = 0.0005,
    internal_friction: float = 0.0,
    collision: str = "sdf",
) -> dict:
    """Add a 57 mm contact-coupled cube and a static pedestal standing on z = 0.

    Args:
        builder: Model builder receiving the cube articulation and fixture.
        center: Fixed core position [m], at least 55 mm above the ground.
        count: Number of movable cubies; must be 26 for a complete 3x3 shell.
        seed: Shared scene-builder argument; the solved cube is deterministic.
        clearance: Distance between adjacent cubie faces [m].
        armature: Added joint-axis inertia [kg m²]; the reference stabilizer is
            not physical cubie inertia. Use zero for mechanical calibration.
        damping: Passive joint-axis viscous damping [N m s/rad]. It cannot
            substitute for a real cube's preload-dependent static friction.
        internal_friction: Cubie contact friction coefficient; there is no
            explicit screw preload or commercial core compliance in this model.
        collision: ``sdf`` cooks the CAD to a grid; ``convex`` uses the same
            chamfered hull directly, isolating grid-related contact effects.

    Returns:
        Cubie body indices, a top-layer grasp target [m], and mechanism metadata.
        Cubie body origins coincide with the core; use their transformed centers
        of mass when measuring piece positions.
    """
    if count != 26:
        raise ValueError("The cube requires exactly 26 cubies.")
    if collision not in ("sdf", "convex"):
        raise ValueError("Cube collision must be 'sdf' or 'convex'")
    if center[2] <= 0.055:
        raise ValueError("The cube center must be above 55 mm to clear its pedestal.")
    if not np.isfinite((clearance, armature, damping, internal_friction)).all():
        raise ValueError("Cube physical parameters must be finite")
    if not (0 <= clearance <= 0.002) or min(armature, damping, internal_friction) < 0:
        raise ValueError("Cube parameters require nonnegative values and clearance at most 2 mm")

    pitch, seam = 0.019, clearance
    half = (pitch - seam) / 2
    chamfer, mass = 0.00142, 0.00253704
    half_extent = pitch + half
    cube_xform = wp.transform(center, wp.quat_identity())
    # Frictionless internal sliding follows the reference's condim=1. Robot
    # fingers still obtain friction through the pair's mixed material value.
    plastic = newton.ModelBuilder.ShapeConfig(density=0.0, mu=internal_friction, gap=0.00003)
    sticker = newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False, has_particle_collision=False)
    cubie_mesh, facelet_mesh = _cubie_mesh(half, chamfer), _facelet_mesh()
    add_cubie_shape = builder.add_shape_mesh if collision == "sdf" else builder.add_shape_convex_hull
    inertia = np.asarray(cubie_mesh.inertia).reshape(3, 3) * (mass / cubie_mesh.mass)
    colors = (
        ((1.0, 0.28, 0.025), (0.85, 0.025, 0.025)),
        ((0.025, 0.20, 0.85), (0.025, 0.62, 0.14)),
        ((1.0, 0.83, 0.025), (0.95, 0.95, 0.95)),
    )

    # The low plate clears a turning face; the narrow post fits between them.
    base_top = center[2] - 0.055
    post_top = center[2] - half_extent - seam
    fixture_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.5)
    fixture_shapes = [
        builder.add_shape_box(
            -1,
            xform=wp.transform((center[0], center[1], base_top / 2.0), wp.quat_identity()),
            hx=0.044,
            hy=0.044,
            hz=base_top / 2.0,
            cfg=fixture_cfg,
            color=(0.19, 0.22, 0.25),
            label="cube/base",
        ),
        builder.add_shape_box(
            -1,
            xform=wp.transform((center[0], center[1], (base_top + post_top) / 2.0), wp.quat_identity()),
            hx=0.003,
            hy=0.003,
            hz=(post_top - base_top) / 2.0,
            cfg=fixture_cfg,
            color=(0.34, 0.37, 0.40),
            label="cube/post",
        ),
    ]

    core = builder.add_link(xform=cube_xform, label="cube/core")
    core_joint = builder.add_joint_fixed(-1, core, parent_xform=cube_xform, label="cube/mount")
    builder.add_shape_sphere(core, radius=0.009, cfg=sticker, color=(0.18, 0.18, 0.19), label="cube/core_hub")
    lattice = [p for p in product((-1, 0, 1), repeat=3) if any(p)]
    bodies, joints, collision_shapes, sticker_shapes = [], [], [], []
    for index, p in enumerate(lattice):
        offset = tuple(pitch * coordinate for coordinate in p)
        body = builder.add_link(
            xform=cube_xform,
            com=offset,
            mass=mass,
            inertia=inertia,
            lock_inertia=True,
            label=f"cube/cubie_{index}",
        )
        bodies.append(body)
        collision_shapes.append(
            add_cubie_shape(
                body,
                xform=wp.transform(offset, wp.quat_identity()),
                mesh=cubie_mesh,
                cfg=plastic,
                color=(0.035, 0.035, 0.035),
                label=f"cube/cubie_{index}/plastic",
            )
        )
        if sum(abs(coordinate) for coordinate in p) == 1:
            joint = builder.add_joint_revolute(
                core,
                body,
                axis=p,
                target_ke=0.0,
                target_kd=0.0,
                damping=damping,
                armature=armature,
                actuator_mode=newton.JointTargetMode.NONE,
                collision_filter_parent=False,
                label=f"cube/cubie_{index}/hinge",
            )
        else:
            joint = builder.add_joint_ball(
                core,
                body,
                damping=damping,
                armature=armature,
                actuator_mode=newton.JointTargetMode.NONE,
                collision_filter_parent=False,
                label=f"cube/cubie_{index}/ball",
            )
        joints.append(joint)

        for axis, sign in enumerate(p):
            if sign == 0:
                continue
            sticker_offset = list(offset)
            sticker_offset[axis] += sign * (half + 0.00003)
            normal = np.eye(3)[axis] * sign
            rotation = wp.quat_between_vectors(wp.vec3(0, 0, 1), wp.vec3(*normal))
            sticker_shapes.append(
                builder.add_shape_mesh(
                    body,
                    xform=wp.transform(sticker_offset, rotation),
                    mesh=facelet_mesh,
                    cfg=sticker,
                    color=colors[axis][int(sign > 0)],
                    label=f"cube/cubie_{index}/sticker_{axis}",
                )
            )

    builder.add_articulation([core_joint, *joints], label="cube")
    return {
        "object_bodies": bodies,
        # Grasp the upper part of the row: the 18.5 mm Franka pads otherwise
        # bridge into the middle row under even small tracking errors.
        "target": (center[0], center[1], center[2] + pitch + 0.005),
        "sdf_meshes": [(cubie_mesh, 0.00025)] if collision == "sdf" else [],
        "metadata": {
            "core": core,
            "core_joint": core_joint,
            "joints": joints,
            "lattice": lattice,
            "cube_center": tuple(center),
            "cube_size": 2.0 * half_extent,
            "half_extent": half_extent,
            "pitch": pitch,
            "cubie_half_extent": half,
            "cubie_mass": mass,
            "chamfer": chamfer,
            "armature_kg_m2_per_axis": armature,
            "damping_nm_s_per_rad_per_axis": damping,
            "internal_friction": internal_friction,
            "collision": collision,
            "top_layer_physical_izz_kg_m2": float(9 * inertia[2, 2] + 12 * mass * pitch**2),
            "top_layer_added_izz_kg_m2": 9 * armature,
            "seam": seam,
            "collision_shapes": collision_shapes,
            "sticker_shapes": sticker_shapes,
            "fixture_shapes": fixture_shapes,
            "mechanism": "MuJoCo-inspired contact-coupled surrogate: six center hinges and twenty core-centered ball joints",
        },
    }
