# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Physics tests of the scene features FeatherPGS takes from Newton's model and contacts.

Conveyor surfaces: a mesh with surface velocities (``Mesh(enable_surface_velocity=True)``) moves its material past its
body. A box set on a belt slides until it rides at the belt's speed, accelerating at Coulomb's rate ``mu g`` on the way,
then rides with it without creep: its friction rows' target is the surfaces' relative velocity and its friction anchor
travels with the belt.

Tracking volumes (``tracking_volumes``): shapes whose contacts are made and reported but take no rows. A box falls
through a scanner volume as through air, and a sensor shape on a sliding box neither lifts it nor takes its friction.
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


class TestFeatherPGSConveyorSurface(unittest.TestCase):
    pass


devices = get_test_devices()
for _fn in (test_box_reaches_belt_speed_at_coulomb_rate, test_still_belt_holds_the_box):
    add_function_test(TestFeatherPGSConveyorSurface, _fn.__name__, _fn, devices=devices)
for _fn in (test_box_falls_through_a_tracking_volume, test_sensor_shape_neither_lifts_nor_holds_its_body):
    add_function_test(TestFeatherPGSTrackingVolumes, _fn.__name__, _fn, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
