# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Physics tests of the scene features FeatherPGS takes from Newton's model and contacts.

Conveyor surfaces: a mesh with surface velocities (``Mesh(enable_surface_velocity=True)``) moves its material past its
body. A box set on a belt slides until it rides at the belt's speed, accelerating at Coulomb's rate ``mu g`` on the way,
then rides with it without creep: its friction rows' target is the surfaces' relative velocity and its friction anchor
travels with the belt.

Tracking volumes (``tracking_volumes``): shapes whose contacts are made and reported but take no rows. A box falls
through a scanner volume as through air, and a sensor shape on a sliding box neither lifts it nor takes its friction.

Kinematic bodies prescribed by their body state (``body_q``/``body_qd``, as maximal-coordinate scenes write them): a
kinematic pusher written that way pushes a box as one written through its joint state does, and a kinematic root carries
its jointed children.

Jointed bodies without a root joint or an articulation (``add_link`` and ``add_joint_*`` alone, as maximal-coordinate
scenes build ragdolls and bridges): a figure falls freely, lands and rests with every joint closed; a bridge of hinged
planks hangs between its two world hinges, a closed loop. A D6 joint whose three angular axes are free is a ball joint,
held as one: a pendulum on it swings through the Euler chart's singular configuration as on a ball joint, and a pile of
D6 ragdolls follows the pile of ball-jointed ones.

Joint reaction wrenches (Newton's ``BODY_PARENT_F`` observable): a hinge carries a hanging bob's weight, a released
level rod's pivot a quarter of its weight, and a fixed platform its own weight and that of a box resting on it.
"""

import itertools
import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices

_G = 9.81
_MU = 0.5
_HALF = 0.1
_DT = 1.0 / 200.0


def _belt_mesh(half_extents, top: float):
    """A static box mesh whose top face sits at height ``top``, its surface free to move."""
    box = newton.Mesh.create_box(
        *half_extents, duplicate_vertices=False, compute_normals=False, compute_uvs=False, compute_inertia=False
    )
    vertices = np.asarray(box.vertices, dtype=np.float32) + np.array([0.0, 0.0, top - half_extents[2]], np.float32)
    return newton.Mesh(vertices, np.asarray(box.indices), compute_inertia=False, enable_surface_velocity=True)


def _belt_scene(device, belt_speed: float, articulated: bool):
    """A box resting on a static belt moving at ``belt_speed`` along x: a free body, or the free root of a two-link
    articulation (a small pendulum bob on a stiff hinge) so its rows take the articulated route."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -_G))
    cfg = newton.ModelBuilder.ShapeConfig(mu=_MU, restitution=0.0)
    belt = _belt_mesh((4.0, 1.0, 0.05), 0.0)
    builder.add_shape_mesh(-1, mesh=belt, cfg=cfg)
    pose = wp.transform(wp.vec3(-1.0, 0.0, _HALF), wp.quat_identity())
    if articulated:
        box = builder.add_link(xform=pose)
        root = builder.add_joint_free(child=box)
        bob = builder.add_link(xform=wp.transform(wp.vec3(-1.0, 0.0, 2.5 * _HALF), wp.quat_identity()))
        builder.add_shape_sphere(bob, radius=0.02, cfg=newton.ModelBuilder.ShapeConfig(density=500.0, mu=_MU))
        hinge = builder.add_joint_revolute(
            box,
            bob,
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, 1.5 * _HALF), wp.quat_identity()),
            axis=newton.Axis.Y,
            target_ke=1.0e3,
            target_kd=10.0,
        )
        builder.add_articulation([root, hinge])
    else:
        box = builder.add_body(xform=pose)
    builder.add_shape_box(box, hx=_HALF, hy=_HALF, hz=_HALF, cfg=cfg)
    model = builder.finalize(device=device)
    belt.mesh.velocities.fill_(wp.vec3(belt_speed, 0.0, 0.0))
    return model, box


def _ride(model, box: int, steps: int):
    """Step the scene, colliding every step; return the box's COM velocity and position at every step."""
    solver = newton.solvers.SolverFeatherPGS(model)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_in, state_out, control = model.state(), model.state(), model.control()
    velocity, position = [], []
    for _ in range(steps):
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, control, contacts, _DT)
        state_in, state_out = state_out, state_in
        velocity.append(state_in.body_qd.numpy()[box][:3].copy())
        position.append(state_in.body_q.numpy()[box][:3].copy())
    return np.array(velocity), np.array(position), contacts


def test_box_reaches_belt_speed_at_coulomb_rate(test, device):
    """A box set at rest on a belt moving at 1 m/s slides at Coulomb's rate ``mu g`` until it rides at the belt's speed
    (t = u / (mu g) = 0.204 s), then rides with it without creep, on the free bodies' route and the articulated one."""
    speed = 1.0
    for articulated in (False, True):
        with test.subTest(articulated=articulated):
            model, box = _belt_scene(device, speed, articulated)
            test.assertIsNotNone(
                newton.CollisionPipeline(model).contacts().rigid_contact_surface_velocity,
                "the pipeline samples the belt's surface velocity",
            )
            velocity, position, _contacts = _ride(model, box, 160)
            time = _DT * np.arange(1, 161)
            sliding = time < 0.18
            # Coulomb's rate while sliding: v = mu g t.
            np.testing.assert_allclose(velocity[sliding, 0], _MU * _G * time[sliding], atol=0.03)
            # Riding: the belt's speed, the box's travel at it, no sideways or vertical motion.
            riding = time > 0.4
            np.testing.assert_allclose(velocity[riding, 0], speed, atol=2.0e-3)
            travel = (position[-1, 0] - position[np.argmax(riding), 0]) / (time[-1] - time[np.argmax(riding)])
            test.assertAlmostEqual(travel, speed, delta=2.0e-3)
            test.assertLess(np.abs(velocity[:, 1]).max(), 1.0e-3)
            test.assertLess(np.abs(velocity[riding, 2]).max(), 1.0e-2)


def test_still_belt_holds_the_box(test, device):
    """A belt whose surface does not move holds a resting box as a static floor does."""
    model, box = _belt_scene(device, 0.0, articulated=False)
    velocity, position, _contacts = _ride(model, box, 80)
    test.assertLess(np.abs(velocity[-20:, :2]).max(), 1.0e-4)
    test.assertLess(np.abs(position[-1, :2] - np.array([-1.0, 0.0])).max(), 1.0e-4)


def _fall(model, steps: int, body: int, **options):
    """Step with the contact forces observed; return the body's poses and velocities, and per step the count and the
    largest solved force of the contacts on each tracked shape (``options['tracking_volumes']``)."""
    solver = newton.solvers.SolverFeatherPGS(model, **options)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    observables = solver.observables(contacts=contacts, kinds={solver.ObservableKind.CONTACT_F})
    tracked = np.asarray(options.get("tracking_volumes", ()), dtype=np.int64)
    state_in, state_out, control = model.state(), model.state(), model.control()
    poses, speeds, touching, force = [], [], [], []
    for _ in range(steps):
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, control, contacts, _DT, observables=observables)
        state_in, state_out = state_out, state_in
        poses.append(state_in.body_q.numpy()[body][:3].copy())
        speeds.append(state_in.body_qd.numpy()[body][:3].copy())
        count = int(contacts.rigid_contact_count.numpy()[0])
        on = np.isin(contacts.rigid_contact_shape0.numpy()[:count], tracked) | np.isin(
            contacts.rigid_contact_shape1.numpy()[:count], tracked
        )
        touching.append(int(on.sum()))
        force.append(float(np.abs(observables.contact_f.numpy()[:count][on]).max(initial=0.0)))
    return np.array(poses), np.array(speeds), np.array(touching), np.array(force)


def test_box_falls_through_a_tracking_volume(test, device):
    """A box dropped through a static scanner volume falls as through air, its contacts with the volume made and
    reported with zero force, then lands on the floor below."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -_G))
    builder.add_ground_plane()
    volume = builder.add_shape_box(
        -1, xform=wp.transform(wp.vec3(0.0, 0.0, 0.6), wp.quat_identity()), hx=0.3, hy=0.3, hz=0.2
    )
    box = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()))
    builder.add_shape_box(box, hx=_HALF, hy=_HALF, hz=_HALF)
    model = builder.finalize(device=device)
    steps = 120
    poses, speeds, touching, force = _fall(model, steps, box, tracking_volumes=[volume])
    # Free fall by the integrator's own law until the floor: z_n = z_0 - g dt^2 n (n + 1) / 2.
    n = np.arange(1, steps + 1)
    falling = 1.0 - _G * _DT**2 * n * (n + 1) / 2 > _HALF + 0.02
    np.testing.assert_allclose(poses[falling, 2], (1.0 - _G * _DT**2 * n * (n + 1) / 2)[falling], atol=1.0e-5)
    test.assertGreater(touching[falling].max(), 0, "the volume's contacts are made and reported")
    test.assertEqual(force.max(), 0.0, "the volume exerts no force")
    test.assertAlmostEqual(poses[-1, 2], _HALF, delta=2.0e-3, msg="the box lands on the floor")
    test.assertLess(np.abs(speeds[-1]).max(), 1.0e-2)


def test_sensor_shape_neither_lifts_nor_holds_its_body(test, device):
    """A massless sensor sphere hanging below a box into the floor is a tracking volume: the box rests on its own
    face and, pushed along the floor, slides to rest at Coulomb's rate as a box without the sensor does."""
    results = []
    for sensor in (False, True):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -_G))
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=_MU))
        box = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, _HALF), wp.quat_identity()))
        builder.add_shape_box(box, hx=_HALF, hy=_HALF, hz=_HALF, cfg=newton.ModelBuilder.ShapeConfig(mu=_MU))
        volumes = []
        if sensor:
            volumes.append(
                builder.add_shape_sphere(
                    box,
                    xform=wp.transform(wp.vec3(0.03, 0.0, -_HALF), wp.quat_identity()),
                    radius=0.05,
                    cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu=_MU),
                )
            )
        model = builder.finalize(device=device)
        qd = model.joint_qd.numpy()
        qd[0] = 1.0
        model.joint_qd.assign(qd)
        results.append(_fall(model, 80, box, tracking_volumes=volumes))
    (plain, plain_speed, _, _), (sensed, sensed_speed, touching, force) = results
    test.assertGreater(touching.min(), 0, "the sensor touches the floor at every step")
    test.assertEqual(force.max(), 0.0)
    # Coulomb's rate: v = 1 - mu g t until rest at t = 0.204 s.
    time = _DT * np.arange(1, 81)
    sliding = time < 0.18
    np.testing.assert_allclose(sensed_speed[sliding, 0], 1.0 - _MU * _G * time[sliding], atol=0.03)
    np.testing.assert_allclose(sensed, plain, atol=1.0e-5)
    np.testing.assert_allclose(sensed_speed, plain_speed, atol=1.0e-4)
    test.assertAlmostEqual(sensed[-1, 2], _HALF, delta=1.0e-3)


class TestFeatherPGSTrackingVolumes(unittest.TestCase):
    pass


_PUSH = 0.5


def _pusher_scene(device):
    """A kinematic pusher box behind a dynamic box on the floor, 0.3 m apart."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -_G))
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=_MU))
    cfg = newton.ModelBuilder.ShapeConfig(mu=_MU)
    pusher = builder.add_body(xform=wp.transform(wp.vec3(-0.5, 0.0, _HALF), wp.quat_identity()), is_kinematic=True)
    builder.add_shape_box(pusher, hx=_HALF, hy=_HALF, hz=_HALF, cfg=cfg)
    box = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, _HALF), wp.quat_identity()))
    builder.add_shape_box(box, hx=_HALF, hy=_HALF, hz=_HALF, cfg=cfg)
    return builder.finalize(device=device), pusher, box


def _push(device, through_body_state: bool, steps: int = 240):
    """Drive the pusher along +x at 0.5 m/s, writing its body state (pose and twist) before each step, or its joint
    state with notify_state_changed; return the pusher's published poses and the box's poses and velocities."""
    model, pusher, box = _pusher_scene(device)
    solver = newton.solvers.SolverFeatherPGS(model)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_in, state_out, control = model.state(), model.state(), model.control()
    q_start = int(model.joint_q_start.numpy()[model.joint_child.numpy().tolist().index(pusher)])
    articulation = wp.array([int(model.joint_articulation.numpy()[q_start // 7])], dtype=wp.int32, device=device)
    pushed, poses, speeds = [], [], []
    for n in range(steps):
        x = -0.5 + _PUSH * _DT * (n + 1)
        if through_body_state:
            body_q, body_qd = state_in.body_q.numpy(), state_in.body_qd.numpy()
            body_q[pusher][:3] = (x, 0.0, _HALF)
            body_qd[pusher] = (_PUSH, 0.0, 0.0, 0.0, 0.0, 0.0)
            state_in.body_q.assign(body_q)
            state_in.body_qd.assign(body_qd)
        else:
            joint_q, joint_qd = state_in.joint_q.numpy(), state_in.joint_qd.numpy()
            joint_q[q_start : q_start + 3] = (x, 0.0, _HALF)
            joint_qd[q_start - q_start // 7 : q_start - q_start // 7 + 6] = (_PUSH, 0.0, 0.0, 0.0, 0.0, 0.0)
            state_in.joint_q.assign(joint_q)
            state_in.joint_qd.assign(joint_qd)
            newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
            solver.notify_state_changed(articulation)
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, control, contacts, _DT)
        state_in, state_out = state_out, state_in
        pushed.append(state_in.body_q.numpy()[pusher][:3].copy())
        poses.append(state_in.body_q.numpy()[box][:3].copy())
        speeds.append(state_in.body_qd.numpy()[box][:3].copy())
    return np.array(pushed), np.array(poses), np.array(speeds)


def test_body_state_drives_a_kinematic_pusher(test, device):
    """A kinematic pusher written through its body state holds each written pose and pushes the box ahead of it at
    its own speed, as the same pusher written through its joint state does."""
    pushed, poses, speeds = _push(device, through_body_state=True)
    written = -0.5 + _PUSH * _DT * np.arange(1, len(pushed) + 1)
    np.testing.assert_allclose(pushed[:, 0], written, atol=1.0e-6)
    test.assertLess(np.abs(poses[:100, 0]).max(), 1.0e-4, "the box rests until the pusher reaches it")
    np.testing.assert_allclose(speeds[-20:, 0], _PUSH, atol=1.0e-2)
    test.assertGreater(poses[-1, 0], 0.2, "the box was pushed")
    _, joint_poses, joint_speeds = _push(device, through_body_state=False)
    np.testing.assert_allclose(poses, joint_poses, atol=1.0e-4)
    np.testing.assert_allclose(speeds, joint_speeds, atol=1.0e-3)


def test_body_state_drives_a_kinematic_root_and_its_children(test, device):
    """A kinematic free root carrying a hinged pendulum, driven along +x through its body state: the root holds each
    written pose, the hinge stays closed and the pendulum travels with it."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -_G))
    root = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()), is_kinematic=True)
    builder.add_shape_box(root, hx=0.05, hy=0.05, hz=0.05)
    bob = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.7), wp.quat_identity()))
    builder.add_shape_sphere(bob, radius=0.05)
    hinge_in_root = wp.vec3(0.0, 0.0, -0.05)
    hinge_in_bob = wp.vec3(0.0, 0.0, 0.25)
    joints = [
        builder.add_joint_free(child=root),
        builder.add_joint_revolute(
            root,
            bob,
            parent_xform=wp.transform(hinge_in_root, wp.quat_identity()),
            child_xform=wp.transform(hinge_in_bob, wp.quat_identity()),
            axis=newton.Axis.Y,
        ),
    ]
    builder.add_articulation(joints)
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverFeatherPGS(model)
    state_in, state_out, control = model.state(), model.state(), model.control()
    speed, closure, roots, bobs = 0.3, [], [], []
    for n in range(200):
        body_q, body_qd = state_in.body_q.numpy(), state_in.body_qd.numpy()
        body_q[root][:3] = (speed * _DT * (n + 1), 0.0, 1.0)
        body_qd[root] = (speed, 0.0, 0.0, 0.0, 0.0, 0.0)
        state_in.body_q.assign(body_q)
        state_in.body_qd.assign(body_qd)
        solver.step(state_in, state_out, control, None, _DT)
        state_in, state_out = state_out, state_in
        q = state_in.body_q.numpy()
        anchor_root = wp.transform_point(wp.transform(wp.vec3(*q[root][:3]), wp.quat(*q[root][3:])), hinge_in_root)
        anchor_bob = wp.transform_point(wp.transform(wp.vec3(*q[bob][:3]), wp.quat(*q[bob][3:])), hinge_in_bob)
        closure.append(np.linalg.norm(np.array(anchor_root) - np.array(anchor_bob)))
        roots.append(q[root][0])
        bobs.append(q[bob][0])
    np.testing.assert_allclose(roots, speed * _DT * np.arange(1, 201), atol=1.0e-6)
    test.assertLess(max(closure), 1.0e-4, "the hinge stays closed")
    test.assertAlmostEqual((bobs[-1] - bobs[99]) / (100 * _DT), speed, delta=0.05)


class TestFeatherPGSKinematicBodyState(unittest.TestCase):
    pass


def _anchor(q, frame):
    """The world point of a body frame ``frame`` ([x, y, z]) under the body pose ``q`` (7 floats)."""
    return np.asarray(
        wp.transform_point(wp.transform(wp.vec3(*q[:3]), wp.quat(*q[3:])), wp.vec3(*frame)), dtype=np.float64
    )


def _joint_gaps(model, body_q):
    """Each joint's gap between its two frames' origins [m] (a joint on the world anchors at its parent frame)."""
    parents, children = model.joint_parent.numpy(), model.joint_child.numpy()
    frames_p, frames_c = model.joint_X_p.numpy(), model.joint_X_c.numpy()
    gaps = []
    for j in range(model.joint_count):
        at_parent = frames_p[j][:3] if parents[j] < 0 else _anchor(body_q[parents[j]], frames_p[j][:3])
        gaps.append(np.linalg.norm(at_parent - _anchor(body_q[children[j]], frames_c[j][:3])))
    return np.array(gaps)


# A figure of six boxes: (name, centre [m], half extents [m]); its limbs hang on ball joints at the anchors.
_FIGURE = (
    ("torso", (0.0, 0.0, 1.0), (0.15, 0.1, 0.25)),
    ("head", (0.0, 0.0, 1.35), (0.08, 0.08, 0.08)),
    ("arm_l", (-0.3, 0.0, 1.1), (0.12, 0.05, 0.05)),
    ("arm_r", (0.3, 0.0, 1.1), (0.12, 0.05, 0.05)),
    ("leg_l", (-0.08, 0.0, 0.45), (0.06, 0.06, 0.2)),
    ("leg_r", (0.08, 0.0, 0.45), (0.06, 0.06, 0.2)),
)
_PIVOTS = (
    (1, (0.0, 0.0, 1.27)),
    (2, (-0.17, 0.0, 1.1)),
    (3, (0.17, 0.0, 1.1)),
    (4, (-0.08, 0.0, 0.7)),
    (5, (0.08, 0.0, 0.7)),
)


def test_loose_figure_falls_and_its_joints_hold(test, device):
    """Six boxes joined by ball joints without an articulation or a root joint (as a maximal-coordinate ragdoll is
    built) fall freely together and land standing on their legs, every joint closed throughout (its frictionless arms
    swing on), on a floor held by a kinematic link without a joint, which stays where it stands; the caller's model
    keeps its layout."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -_G))
    ground = builder.add_link(is_kinematic=True)
    builder.add_shape_plane(
        body=ground, width=0.0, length=0.0, cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu=_MU)
    )
    cfg = newton.ModelBuilder.ShapeConfig(mu=_MU, collision_group=-1)
    bodies = []
    for _name, centre, half in _FIGURE:
        bodies.append(builder.add_link(xform=wp.transform(wp.vec3(*centre), wp.quat_identity())))
        builder.add_shape_box(bodies[-1], hx=half[0], hy=half[1], hz=half[2], cfg=cfg)
    for limb, pivot in _PIVOTS:
        builder.add_joint_ball(
            bodies[0],
            bodies[limb],
            parent_xform=wp.transform(wp.vec3(*np.subtract(pivot, _FIGURE[0][1])), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(*np.subtract(pivot, _FIGURE[limb][1])), wp.quat_identity()),
        )
    model = builder.finalize(device=device, skip_validation_joints=True)
    articulation = model.joint_articulation.numpy().copy()
    solver = newton.solvers.SolverFeatherPGS(model, dense_max_constraints=64)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_in, state_out, control = model.state(), model.state(), model.control()
    torso, gaps, lowest = [], [], []
    steps = 200
    for _ in range(steps):
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, control, contacts, _DT)
        state_in, state_out = state_out, state_in
        body_q = state_in.body_q.numpy()
        torso.append(body_q[bodies[0]][2])
        gaps.append(_joint_gaps(model, body_q).max())
        lowest.append(body_q[bodies, 2].min())
        np.testing.assert_array_equal(
            body_q[ground], np.array(wp.transform_identity()), "the floor's link stands still"
        )
    np.testing.assert_array_equal(model.joint_articulation.numpy(), articulation)
    test.assertEqual(state_in.joint_q.shape[0], model.joint_coord_count)
    n = np.arange(1, steps + 1)
    falling = n * _DT < 0.2
    np.testing.assert_allclose(np.array(torso)[falling], (1.0 - _G * _DT**2 * n * (n + 1) / 2)[falling], atol=1.0e-4)
    test.assertLess(max(gaps), 1.0e-4, "every joint stays closed")
    test.assertGreater(min(lowest), 0.04, "no box sinks into the floor")
    # Landed on its legs (their feet at the floor puts the torso's centre at 0.75 m), and standing.
    np.testing.assert_allclose(np.array(torso)[n * _DT > 0.4], 0.75, atol=0.01)
    test.assertTrue(np.isfinite(state_in.joint_q.numpy()).all())


def test_loose_bridge_hangs_on_its_end_hinges(test, device):
    """Four planks hinged in a row between two hinges on the world, without an articulation: a closed loop the
    solver roots at its first world hinge and closes at the other. It spans its gap under gravity, every hinge
    closed."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -_G))
    planks = [
        builder.add_link(xform=wp.transform(wp.vec3(-0.75 + 0.5 * k, 0.0, 1.0), wp.quat_identity())) for k in range(4)
    ]
    for plank in planks:
        builder.add_shape_box(plank, hx=0.25, hy=0.2, hz=0.02)
    edge = wp.transform(wp.vec3(-0.25, 0.0, 0.0), wp.quat_identity())
    far_edge = wp.transform(wp.vec3(0.25, 0.0, 0.0), wp.quat_identity())
    builder.add_joint_revolute(
        -1, planks[0], parent_xform=wp.transform(wp.vec3(-1.0, 0.0, 1.0), wp.quat_identity()), child_xform=edge,
        axis=newton.Axis.Y,
    )  # fmt: skip
    for left, right in itertools.pairwise(planks):
        builder.add_joint_revolute(left, right, parent_xform=far_edge, child_xform=edge, axis=newton.Axis.Y)
    builder.add_joint_revolute(
        -1, planks[3], parent_xform=wp.transform(wp.vec3(1.0, 0.0, 1.0), wp.quat_identity()), child_xform=far_edge,
        axis=newton.Axis.Y,
    )  # fmt: skip
    model = builder.finalize(device=device, skip_validation_joints=True)
    solver = newton.solvers.SolverFeatherPGS(model, pgs_iterations=32)
    state_in, state_out, control = model.state(), model.state(), model.control()
    gaps, heights = [], []
    for _ in range(200):
        solver.step(state_in, state_out, control, None, _DT)
        state_in, state_out = state_out, state_in
        body_q = state_in.body_q.numpy()
        gaps.append(_joint_gaps(model, body_q).max())
        heights.append(body_q[planks, 2].min())
    test.assertLess(max(gaps), 1.0e-4, "every hinge stays closed, the loop's too")
    test.assertGreater(min(heights), 1.0 - 5.0e-3, "the taut bridge holds its span")


def _sphere_pendulum(device, kind: str):
    """A pendulum released level on a D6 joint with three free angular axes (``"d6"``) or on a ball joint
    (``"ball"``) at its pivot: its swing about the y axis reaches 90 degrees, the D6's middle Euler angle, at the
    bottom."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -_G))
    bob = builder.add_link(xform=wp.transform(wp.vec3(0.5, 0.0, 1.0), wp.quat_identity()))
    builder.add_shape_sphere(bob, radius=0.05)
    frames = {
        "parent_xform": wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
        "child_xform": wp.transform(wp.vec3(-0.5, 0.0, 0.0), wp.quat_identity()),
    }
    if kind == "ball":
        joint = builder.add_joint_ball(-1, bob, **frames)
    else:
        axes = [newton.ModelBuilder.JointDofConfig.create_unlimited(axis) for axis in newton.Axis]
        joint = builder.add_joint_d6(-1, bob, angular_axes=axes, **frames)
    builder.add_articulation([joint])
    return builder.finalize(device=device), bob


def test_free_d6_swings_through_its_singular_angles(test, device):
    """A pendulum on a D6 with three free angular axes swings through the bottom (its middle Euler angle at 90
    degrees, where the Euler chart is singular) exactly as on a ball joint, its length kept; its published Euler angles
    give the pendulum's pose by Newton's D6 forward kinematics."""
    poses = {}
    for kind in ("d6", "ball"):
        model, bob = _sphere_pendulum(device, kind)
        solver = newton.solvers.SolverFeatherPGS(model)
        state_in, state_out, control = model.state(), model.state(), model.control()
        poses[kind] = []
        for _ in range(300):
            solver.step(state_in, state_out, control, None, _DT)
            state_in, state_out = state_out, state_in
            poses[kind].append(state_in.body_q.numpy()[bob].copy())
        if kind == "d6":
            published = model.state()
            newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, published)
            np.testing.assert_allclose(published.body_q.numpy()[bob], state_in.body_q.numpy()[bob], atol=1.0e-5)
    d6, ball = np.array(poses["d6"]), np.array(poses["ball"])
    test.assertTrue(np.isfinite(d6).all())
    test.assertLess(d6[:, 2].min(), 0.5 + 1.0e-3, "the bob swings through the bottom")
    np.testing.assert_allclose(np.linalg.norm(d6[:, :3] - [0.0, 0.0, 1.0], axis=1), 0.5, atol=1.0e-5)
    np.testing.assert_allclose(d6, ball, atol=1.0e-6)


def _four_figure_pile(device, kind: str, turn: float = 0.0):
    """Two layers of four figures whose limbs hang on D6 joints with three free angular axes (``"d6"``) or on ball
    joints (``"ball"``), built without an articulation (as maximal-coordinate ragdoll piles build them), the whole pile
    turned by ``turn`` radians about the vertical through the origin; its body poses by step [m] over 240 steps, turned
    back, and its largest joint gap."""
    angular = [newton.ModelBuilder.JointDofConfig.create_unlimited(axis) for axis in newton.Axis]
    c, s = np.cos(turn), np.sin(turn)
    spin = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    heading = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(turn))
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -_G))
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=_MU))
    for figure in range(4):
        offset = np.array([0.6 * (figure % 2), 0.0, 1.6 * (figure // 2)])
        cfg = newton.ModelBuilder.ShapeConfig(mu=_MU, collision_group=-(figure + 1))
        bodies = []
        for _name, centre, half in _FIGURE:
            bodies.append(builder.add_link(xform=wp.transform(wp.vec3(*(spin @ np.add(centre, offset))), heading)))
            builder.add_shape_box(bodies[-1], hx=half[0], hy=half[1], hz=half[2], cfg=cfg)
        for limb, pivot in _PIVOTS:
            frames = {
                "parent_xform": wp.transform(wp.vec3(*np.subtract(pivot, _FIGURE[0][1])), wp.quat_identity()),
                "child_xform": wp.transform(wp.vec3(*np.subtract(pivot, _FIGURE[limb][1])), wp.quat_identity()),
            }
            if kind == "d6":
                builder.add_joint_d6(bodies[0], bodies[limb], angular_axes=angular, **frames)
            else:
                builder.add_joint_ball(bodies[0], bodies[limb], **frames)
    model = builder.finalize(device=device, skip_validation_joints=True)
    solver = newton.solvers.SolverFeatherPGS(model, dense_max_constraints=1024)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_in, state_out, control = model.state(), model.state(), model.control()
    poses, gaps = [], []
    for _ in range(240):
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, control, contacts, _DT)
        state_in, state_out = state_out, state_in
        body_q = state_in.body_q.numpy()
        poses.append(body_q[:, :3].astype(np.float64) @ spin)
        gaps.append(_joint_gaps(model, body_q).max())
    return np.array(poses), max(gaps)


def test_pile_of_d6_figures_follows_the_ball_jointed_pile(test, device):
    """Figures whose limbs hang on D6 joints with three free angular axes land on each other and on the floor as the
    same figures on ball joints do: the same motion until the pile's contacts amplify the two joints' rounding, then
    finite, every joint closed, nothing sunk into the floor.

    A pile is chaotic: any rounding difference grows from about 1e-7 m to past 1e-4 m in about half a second. The D6
    pile is held to what a rounding twin of the ball pile does, the same ball pile turned 30 degrees about the vertical
    (the same physics; every axis, frame and quaternion rounds differently). The D6 pile parts from the ball pile by
    1e-4 m no sooner than the turned pile does, less half the turned pile's climb from 1e-6 to 1e-4 m: a D6 may round
    up to ten times further from a ball joint a step than the turned copy does (two renderings of the same joint), which
    the chaos's growth turns into half that climb. Measured on the CPU, the first steps past 1e-6 and 1e-4 m: before the
    turns' energy-keeping step D6 46, 114 and turned 49, 114 (bound 81.5); with it D6 45, 108 and turned 46, 108
    (bound 77)."""
    piles, closure = {}, {}
    for kind, turn in (("d6", 0.0), ("ball", 0.0), ("turned", np.radians(30.0))):
        piles[kind], closure[kind] = _four_figure_pile(device, "d6" if kind == "d6" else "ball", turn)
    test.assertTrue(np.isfinite(piles["d6"]).all())
    test.assertGreater(piles["d6"][:, :, 2].min(), 0.04, "no box sinks into the floor")
    test.assertLess(closure["d6"], 1.0e-4, "every joint stays closed")

    def parting(kind, gap):
        apart = np.abs(piles[kind] - piles["ball"]).max(axis=(1, 2)) > gap
        return int(np.argmax(apart)) if apart.any() else len(apart)

    twin_start, twin_end = parting("turned", 1.0e-6), parting("turned", 1.0e-4)
    test.assertGreater(twin_end, twin_start, "the turned pile parts from the ball pile by rounding, then by chaos")
    test.assertGreaterEqual(
        parting("d6", 1.0e-4),
        (twin_start + twin_end) / 2,
        f"the D6 pile parts from the ball pile no sooner than a rounding twin does (turned: {twin_start}..{twin_end})",
    )


def _figure_pile(device, **options):
    """Forty-eight figures on ball joints (a tree each) dropped in two layers onto the floor, and its solver."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -_G))
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=_MU))
    for figure in range(48):
        # A metre apart in x (a figure's arms span 0.84 m) and 0.5 m in y (its depth 0.2 m): no figure starts inside
        # another; the upper layer lands on the lower.
        offset = np.array([1.0 * (figure % 6), 0.5 * (figure // 6 % 4), 1.6 * (figure // 24)])
        cfg = newton.ModelBuilder.ShapeConfig(mu=_MU, collision_group=-(figure + 1))
        bodies = []
        for _name, centre, half in _FIGURE:
            bodies.append(builder.add_link(xform=wp.transform(wp.vec3(*(np.add(centre, offset))), wp.quat_identity())))
            builder.add_shape_box(bodies[-1], hx=half[0], hy=half[1], hz=half[2], cfg=cfg)
        for limb, pivot in _PIVOTS:
            builder.add_joint_ball(
                bodies[0],
                bodies[limb],
                parent_xform=wp.transform(wp.vec3(*np.subtract(pivot, _FIGURE[0][1])), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(*np.subtract(pivot, _FIGURE[limb][1])), wp.quat_identity()),
            )
    model = builder.finalize(device=device, skip_validation_joints=True)
    return model, newton.solvers.SolverFeatherPGS(model, dense_max_constraints=8192, **options)


def _drop_pile(model, solver, steps: int):
    """Step the pile ``steps`` times; return its last poses and velocities, the steps whose reservation took the colors
    of groups, each step's largest joint gap and the most posts of the world's job words in a step (its sweeps and its
    end, sweep_routes.group_rows)."""
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_in, state_out, control = model.state(), model.state(), model.control()
    grouped, gaps, posts = 0, [], 0
    for _ in range(steps):
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, control, contacts, _DT)
        state_in, state_out = state_out, state_in
        grouped += int(solver.placement.row_groups.colors.numpy()[0] > 0)
        gaps.append(_joint_gaps(model, state_in.body_q.numpy()).max())
        if solver.sweep.wide_jobs is not None:
            posts = max(posts, int(solver.sweep.wide_jobs.numpy()[0]))
    return state_in.body_q.numpy(), state_in.body_qd.numpy(), grouped, gaps, posts


def test_pile_of_figures_sweeps_by_colors_of_groups(test, device):
    """Forty-eight figures on ball joints (a tree each) dropped in two layers onto the floor: a world of many trees
    sweeps its dense line by colors of groups (row_slots.RowGroups: up to four body pairs between the same trees,
    first-fit colors, a warp a group in the world's block), which its reservation takes once the pile holds enough
    groups; the pile lands finite, every joint closed, nothing sunk into the floor. Its trees outnumber the block's
    warps, so its sweeps take the device's other blocks beside the world's, each group run once its trees' groups
    before it ran (sweep_routes.group_rows): each tree's groups in the colors' order, the same visits, so the same
    bits as the world's block alone by colors."""
    model, solver = _figure_pile(device)
    test.assertIsNotNone(solver.sweep.mf_grouped_kernel, "a world of many trees builds the grouped sweep")
    test.assertGreater(solver.sweep.grouped_blocks, 1, "a world's colors may outgrow its block")
    poses, velocities, grouped, gaps, posts = _drop_pile(model, solver, 300)
    test.assertGreater(grouped, 0, "the pile's reservation takes the colors of groups")
    test.assertTrue(np.isfinite(poses).all())
    test.assertGreater(poses[:, 2].min(), 0.04, "no box sinks into the floor")
    test.assertLess(max(gaps), 1.0e-4, "every joint stays closed")
    test.assertGreater(posts, 1, "the world's sweeps take the group")
    model, alone = _figure_pile(device, pins=newton.solvers.SolverFeatherPGS.Pins(world_group=False))
    test.assertEqual(alone.sweep.grouped_blocks, 1)
    poses_alone, velocities_alone, _, _, _ = _drop_pile(model, alone, 300)
    np.testing.assert_array_equal(poses, poses_alone)
    np.testing.assert_array_equal(velocities, velocities_alone)


class TestFeatherPGSLooseBodies(unittest.TestCase):
    pass


def _wrenches(model, steps: int, dt: float, contacts_on: bool = False, **options):
    """Step ``model`` from rest, reading the BODY_PARENT_F observable at every step; return the wrenches [N, N m]."""
    solver = newton.solvers.SolverFeatherPGS(model, **options)
    test_kinds = {solver.ObservableKind.BODY_PARENT_F}
    pipeline = newton.CollisionPipeline(model) if contacts_on else None
    contacts = pipeline.contacts() if contacts_on else None
    if contacts_on:
        test_kinds.add(solver.ObservableKind.CONTACT_F)
    observables = solver.observables(contacts=contacts, kinds=test_kinds)
    state_in, state_out, control = model.state(), model.state(), model.control()
    wrenches = []
    for _ in range(steps):
        if contacts_on:
            pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, control, contacts, dt, observables=observables)
        state_in, state_out = state_out, state_in
        wrenches.append(observables.body_parent_f.numpy().copy())
    return np.array(wrenches)


def test_hinge_carries_a_hanging_load(test, device):
    """A bob hanging at rest from a hinge on the world: the hinge's wrench is the bob's weight, upward, its moment
    about the bob's centre zero (the hinge lies above it)."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -_G))
    bob = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()))
    builder.add_shape_sphere(bob, radius=0.05)
    hinge = builder.add_joint_revolute(
        -1, bob, parent_xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()), axis=newton.Axis.Y,
    )  # fmt: skip
    builder.add_articulation([hinge])
    model = builder.finalize(device=device)
    weight = float(model.body_mass.numpy()[bob]) * _G
    wrenches = _wrenches(model, 20, _DT)
    np.testing.assert_allclose(wrenches[:, bob, :3], np.tile([0.0, 0.0, weight], (20, 1)), atol=1.0e-4 * weight)
    np.testing.assert_allclose(wrenches[:, bob, 3:], 0.0, atol=1.0e-5 * weight)


def test_hinge_wrench_of_a_released_rod(test, device):
    """A uniform rod pivoted at its end on the world, released level from rest: its first instant's pivot force is a
    quarter of its weight upward (the rod's centre falls at three quarters of g), with no force along it."""
    length, dt = 1.0, 1.0e-3
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -_G))
    rod = builder.add_link(xform=wp.transform(wp.vec3(0.5 * length, 0.0, 1.0), wp.quat_identity()))
    builder.add_shape_box(rod, hx=0.5 * length, hy=0.01, hz=0.01)
    pivot = builder.add_joint_revolute(
        -1, rod, parent_xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.5 * length, 0.0, 0.0), wp.quat_identity()), axis=newton.Axis.Y,
    )  # fmt: skip
    builder.add_articulation([pivot])
    model = builder.finalize(device=device)
    weight = float(model.body_mass.numpy()[rod]) * _G
    first = _wrenches(model, 1, dt)[0, rod]
    test.assertAlmostEqual(first[2], 0.25 * weight, delta=0.01 * weight)
    test.assertAlmostEqual(first[0], 0.0, delta=0.01 * weight)
    # Its moment about the rod's centre: the pivot force's lever, half the rod back.
    test.assertAlmostEqual(first[4], 0.5 * length * 0.25 * weight, delta=0.01 * weight)


def test_fixed_platform_carries_what_rests_on_it(test, device):
    """A platform fixed to the world carries its own weight and that of a box resting on it, through its contacts."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -_G))
    platform = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()))
    builder.add_shape_box(platform, hx=0.5, hy=0.5, hz=0.02)
    builder.add_articulation(
        [builder.add_joint_fixed(-1, platform, parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()))]
    )
    box = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.52 + _HALF), wp.quat_identity()))
    builder.add_shape_box(box, hx=_HALF, hy=_HALF, hz=_HALF)
    model = builder.finalize(device=device)
    mass = model.body_mass.numpy()
    wrenches = _wrenches(model, 100, _DT, contacts_on=True)
    carried = wrenches[-20:, platform]
    np.testing.assert_allclose(carried[:, 2], (mass[platform] + mass[box]) * _G, rtol=1.0e-3)
    np.testing.assert_allclose(carried[:, :2], 0.0, atol=1.0e-3 * mass[box] * _G)
    np.testing.assert_allclose(carried[:, 3:], 0.0, atol=1.0e-3 * mass[box] * _G)
    # The box's own free joint carries nothing: the contact holds it.
    np.testing.assert_allclose(wrenches[-20:, box], 0.0, atol=1.0e-3 * mass[box] * _G)


class TestFeatherPGSJointWrenches(unittest.TestCase):
    pass


class TestFeatherPGSConveyorSurface(unittest.TestCase):
    pass


devices = get_test_devices()
for _fn in (test_box_reaches_belt_speed_at_coulomb_rate, test_still_belt_holds_the_box):
    add_function_test(TestFeatherPGSConveyorSurface, _fn.__name__, _fn, devices=devices)
for _fn in (test_box_falls_through_a_tracking_volume, test_sensor_shape_neither_lifts_nor_holds_its_body):
    add_function_test(TestFeatherPGSTrackingVolumes, _fn.__name__, _fn, devices=devices)
for _fn in (test_body_state_drives_a_kinematic_pusher, test_body_state_drives_a_kinematic_root_and_its_children):
    add_function_test(TestFeatherPGSKinematicBodyState, _fn.__name__, _fn, devices=devices)
for _fn in (
    test_loose_figure_falls_and_its_joints_hold,
    test_loose_bridge_hangs_on_its_end_hinges,
    test_free_d6_swings_through_its_singular_angles,
    test_pile_of_d6_figures_follows_the_ball_jointed_pile,
):
    add_function_test(TestFeatherPGSLooseBodies, _fn.__name__, _fn, devices=devices)
add_function_test(
    TestFeatherPGSLooseBodies,
    test_pile_of_figures_sweeps_by_colors_of_groups.__name__,
    test_pile_of_figures_sweeps_by_colors_of_groups,
    devices=[device for device in devices if wp.get_device(device).is_cuda],
)
for _fn in (
    test_hinge_carries_a_hanging_load,
    test_hinge_wrench_of_a_released_rod,
    test_fixed_platform_carries_what_rests_on_it,
):
    add_function_test(TestFeatherPGSJointWrenches, _fn.__name__, _fn, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
