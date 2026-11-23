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

Joint reaction wrenches (Newton's ``BODY_PARENT_F`` observable): a hinge carries a hanging bob's weight, a released
level rod's pivot a quarter of its weight, and a fixed platform its own weight and that of a box resting on it.
"""

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
    test_hinge_carries_a_hanging_load,
    test_hinge_wrench_of_a_released_rod,
    test_fixed_platform_carries_what_rests_on_it,
):
    add_function_test(TestFeatherPGSJointWrenches, _fn.__name__, _fn, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
