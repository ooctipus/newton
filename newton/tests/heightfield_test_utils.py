# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Portable synthetic scenes from the recorded finite-heightfield qualification.

All scene hulls are immutable. Pipeline reconstruction is required after a
same-pointer hull refit; this helper never claims support for that operation.
"""

import os
from dataclasses import dataclass
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.geometry.contact_data import ContactData
from newton.sensors import SensorContact


@wp.struct
class WriterData:
    count: wp.array[int]
    ids: wp.array[int]
    points: wp.array[wp.vec3]
    normals: wp.array[wp.vec3]
    depths: wp.array[float]


@wp.func
def record_writer(contact: ContactData, writer: WriterData, output_index: int):
    index = wp.atomic_add(writer.count, 0, 1)
    writer.ids[index] = contact.sort_sub_key
    writer.points[index] = contact.contact_point_center
    writer.normals[index] = contact.contact_normal_a_to_b
    writer.depths[index] = contact.contact_distance


DT = 0.0025
MASS = 2.0
CHILD_MASS = 0.25
HALF = np.array([0.1, 0.08, 0.025], dtype=np.float32)
MARGIN = 0.0025


@dataclass(frozen=True)
class Case:
    """Authored physical control; solver sweeps and collision capacities are fixed."""

    name: str
    kind: str
    steps: int = 1
    restitution: float = 0.0
    mu: float = 0.0
    speed: float = 0.0
    gap: float = 0.0
    tilt: float = 0.0
    terrain_angle: float = 0.0
    x: float = 0.0
    hull: bool = False
    scalar: bool = False


CASES = (
    Case("scalar_inelastic", "impact", speed=3.0, scalar=True),
    Case(
        "scalar_rebound",
        "impact",
        speed=3.0,
        restitution=0.6,
        gap=0.003,
        hull=True,
        scalar=True,
    ),
    Case("inelastic", "impact", speed=3.0),
    Case("rebound", "impact", speed=3.0, restitution=0.6, gap=0.003, hull=True),
    Case("support", "support", steps=240, mu=0.4),
    Case("tilted_foot", "tilt", steps=320, mu=0.4, tilt=0.08, hull=True),
    Case("rotated_terrain", "support", steps=240, mu=0.5, terrain_angle=0.15, hull=True),
    Case("sliding", "friction", steps=360, mu=0.4, speed=2.0),
    Case("finite_border", "border", steps=200, speed=1.0, x=1.05, hull=True),
)


def quat_matrix(quat):
    """Read a world rotation independently through ordinary quaternion vectors."""
    return np.column_stack([np.asarray(wp.quat_rotate(quat, wp.vec3(*axis))) for axis in np.eye(3)])


def cube_mesh():
    """Use exactly eight immutable cuboid corners, as in admitted G1 foot hulls."""
    vertices = np.array(
        [(x, y, z) for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)],
        dtype=np.float32,
    )
    # The actual convex support owner uses the hull; complete triangle faces
    # make this a valid Newton source rather than a fabricated device pointer.
    faces = np.array(
        [
            (0, 1, 3),
            (0, 3, 2),
            (4, 6, 7),
            (4, 7, 5),
            (0, 4, 5),
            (0, 5, 1),
            (2, 3, 7),
            (2, 7, 6),
            (0, 2, 6),
            (0, 6, 4),
            (1, 5, 7),
            (1, 7, 3),
        ],
        dtype=np.int32,
    )
    return newton.Mesh(vertices=vertices, indices=faces.ravel())


def build_scene(case, enabled, device, *, reduce=True, make_solver=True, hull_scale=1.0, terrain_resolution=21):
    """Bind one complete current model/state/collision/sensor owner, never saved rows."""
    terrain_q = wp.quat_from_axis_angle(wp.vec3(0, 1, 0), case.terrain_angle)
    body_q = terrain_q * wp.quat_from_axis_angle(wp.vec3(0, 1, 0), case.tilt)
    terrain_rotation = quat_matrix(terrain_q)
    relative_rotation = terrain_rotation.T @ quat_matrix(body_q)
    normal = terrain_rotation[:, 2]
    # Gravity stays vertical on a ramp: static friction must provide its actual
    # tangential support. Impact controls have no gravitational energy input.
    gravity = np.zeros(3) if case.kind == "impact" else np.array([0.0, 0.0, -9.81])
    builder = newton.ModelBuilder(gravity=tuple(gravity))
    cfg = builder.ShapeConfig(density=0.0, mu=case.mu, restitution=case.restitution, margin=MARGIN, gap=0.02)
    terrain = newton.Heightfield(
        data=np.zeros((terrain_resolution, terrain_resolution), dtype=np.float32),
        nrow=terrain_resolution,
        ncol=terrain_resolution,
        hx=1.2,
        hy=1.2,
    )
    builder.add_shape_heightfield(
        heightfield=terrain,
        xform=wp.transform((0, 0, 0), terrain_q),
        cfg=cfg,
        label="terrain",
    )
    support_height = float(np.abs(relative_rotation[2]) @ HALF)
    authored_local = np.array([case.x, 0.0, support_height + 2 * MARGIN + case.gap])
    authored_world = terrain_rotation @ authored_local
    xform = wp.transform(wp.vec3(*authored_world), body_q)
    inertia = (
        MASS
        / 3.0
        * np.array(
            [
                HALF[1] ** 2 + HALF[2] ** 2,
                HALF[0] ** 2 + HALF[2] ** 2,
                HALF[0] ** 2 + HALF[1] ** 2,
            ]
        )
    )
    foot = builder.add_link(
        xform=xform,
        mass=MASS,
        inertia=wp.mat33(*np.diag(inertia).ravel()),
        lock_inertia=True,
        label="foot",
    )
    ballast = builder.add_link(
        xform=xform,
        mass=CHILD_MASS,
        inertia=wp.mat33(0.002, 0, 0, 0, 0.002, 0, 0, 0, 0.002),
        lock_inertia=True,
        label="ballast",
    )
    if case.scalar:
        free = builder.add_joint_prismatic(parent=-1, child=foot, parent_xform=xform, axis=wp.vec3(0, 0, 1))
    else:
        free = builder.add_joint_free(child=foot)
    fixed = builder.add_joint_fixed(parent=foot, child=ballast)
    builder.add_articulation([free, fixed])
    if case.hull:
        builder.add_shape_convex_hull(
            body=foot,
            mesh=cube_mesh(),
            scale=tuple(HALF * hull_scale),
            cfg=cfg,
            label="foot_contact",
        )
    else:
        builder.add_shape_box(
            body=foot,
            hx=float(HALF[0]),
            hy=float(HALF[1]),
            hz=float(HALF[2]),
            cfg=cfg,
            label="foot_contact",
        )
    model = builder.finalize(device=device)
    sensor = SensorContact(model, sensing_bodies=["foot"])
    with patch.dict(
        os.environ,
        {
            "NEWTON_HEIGHTFIELD_CELL_REJECT": "1",
            "NEWTON_HEIGHTFIELD_FINITE_QUERY": str(int(enabled)),
            # The finite owner is the terrain route's: keep the route for every heightfield pair.
            "NEWTON_TERRAIN_LAW": "none",
        },
    ):
        pipeline = newton.CollisionPipeline(
            model,
            broad_phase="nxn",
            reduce_contacts=reduce,
            max_triangle_pairs=128,
        )
    contacts = pipeline.contacts()
    states = (model.state(), model.state())
    velocity = -case.speed * normal if case.kind == "impact" else case.speed * terrain_rotation[:, 0]
    qd = np.zeros(model.joint_dof_count, dtype=np.float32)
    if case.scalar:
        qd[0] = -case.speed
    else:
        qd[:3] = velocity
    for state in states:
        wp.copy(state.joint_q, model.joint_q)
        state.joint_qd.assign(qd)
        state.clear_forces()
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    solver = None
    if make_solver:
        # Same current G1 finite-sweep law: no changed omega/CFM or warm-start
        # recipe. Only small-scene capacities.
        solver = newton.solvers.SolverFeatherPGS(
            model,
            pgs_iterations=8,
            dense_max_constraints=100,
            mf_max_constraints=32,
            pgs_warmstart=False,
        )
        if solver.options.pgs_iterations != 8:
            raise AssertionError("Wrong actual iteration budget")
    return NS(
        case=case,
        enabled=enabled,
        model=model,
        sensor=sensor,
        pipeline=pipeline,
        contacts=contacts,
        states=states,
        solver=solver,
        control=model.control(),
        normal=normal,
        terrain_rotation=terrain_rotation,
        authored=authored_world,
        gravity=gravity,
        foot=foot,
        ballast=ballast,
        mass=MASS + CHILD_MASS,
        half=HALF.copy(),
        distance=wp.empty(contacts.rigid_contact_max, dtype=float, device=device),
        point0=wp.empty(contacts.rigid_contact_max, dtype=wp.vec3, device=device),
        point1=wp.empty(contacts.rigid_contact_max, dtype=wp.vec3, device=device),
    )


def geometry(scene, state):
    """Read live logical ownership and public finite geometry at test-only boundaries."""
    pipeline, contacts = scene.pipeline, scene.contacts
    pipeline.narrow_phase.check_buffer_capacity()
    count = int(contacts.rigid_contact_count.numpy()[0])
    logical = int(pipeline.narrow_phase.triangle_pairs_count.numpy()[0])
    if not 0 <= count <= 100 or not 0 <= logical <= 128:
        raise AssertionError("Collision capacity overflow")
    triples = pipeline.narrow_phase.triangle_pairs.numpy()[:logical]
    marked = int(np.count_nonzero(triples[:, 2] < 0))
    if bool(pipeline.narrow_phase._heightfield_finite_query) != scene.enabled or (not scene.enabled and marked):
        raise AssertionError("Wrong actual finite owner")
    newton.eval_rigid_contact_kinematics(
        scene.model,
        state,
        contacts,
        out_distance=scene.distance,
        out_point0_world=scene.point0,
        out_point1_world=scene.point1,
    )
    distances = scene.distance.numpy()[:count].astype(float)
    normal = contacts.rigid_contact_normal.numpy()[:count].astype(float)
    points = scene.point0.numpy()[:count].astype(float)
    if not all(np.isfinite(x).all() for x in (distances, normal, points)):
        raise AssertionError("Nonfinite public contact geometry")
    if count and (np.max(np.abs(np.linalg.norm(normal, axis=1) - 1.0)) > 2e-5 or np.min(normal @ scene.normal) < -2e-5):
        raise AssertionError("Invalid surface normal")
    return {
        "contacts": count,
        "logical": logical,
        "marked": marked,
        "minimum_separation": float(distances.min()) if count else None,
    }
