# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for a held contact on a rolling body: where the collision declares a contact touches
(``Contacts.rigid_contact_point_share``), a cylinder's or cone's curved side held as its axis point plus a radius, and
FeatherPGS rolling bodies that keep their energy while a collision pass's contacts are held over its substeps."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS

_DT, _SUBSTEPS = 0.01, 8
_R, _HALF, _MASS = 0.008, 0.02, 0.019


def _lying(builder, mass=_MASS, z=_R, x=0.0, inertia=None):
    """A body with its local z axis along the world y axis (a lying cylinder or capsule), its centre at height z."""
    q = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * np.pi)
    if inertia is None:
        inertia = np.diag([mass * (3 * _R * _R + 4 * _HALF * _HALF) / 12] * 2 + [0.5 * mass * _R * _R])
    return builder.add_body(
        xform=wp.transform(wp.vec3(x, 0.0, z), q), mass=mass, inertia=wp.mat33(*np.ravel(inertia)), lock_inertia=True
    )


def _contacts(model, device):
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state = model.state()
    pipeline.collide(state, contacts)
    n = int(contacts.rigid_contact_count.numpy()[0])
    return {
        name: getattr(contacts, f"rigid_contact_{name}").numpy()[:n]
        for name in ("shape0", "shape1", "point0", "point1", "normal", "margin0", "margin1", "point_share")
    }


def _roll(model, body, radius, v, steps, inertia_axis, device):
    """Roll ``body`` at ``v`` along +x (spinning about +y, rolling without slip) under a collision pass a 10 ms step and
    8 substeps; return its kinetic energy [J] at the start and after each step."""
    state_0, state_1, control = model.state(), model.state(), model.control()
    qd = np.zeros((model.body_count, 6), dtype=np.float32)
    qd[body, 0] = v
    qd[body, 4] = v / radius
    state_0.body_qd.assign(qd)
    joint_qd = np.zeros(model.joint_dof_count, dtype=np.float32)
    joint_qd[0:6] = qd[body]
    state_0.joint_qd.assign(joint_qd)
    solver = SolverFeatherPGS(model, pgs_iterations=16)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    mass = float(model.body_mass.numpy()[body])

    def energy(state):
        w = state.body_qd.numpy()[body]
        return 0.5 * mass * float(w[:3] @ w[:3]) + 0.5 * inertia_axis * float(w[3:] @ w[3:])

    energies = [energy(state_0)]
    for _ in range(steps):
        pipeline.collide(state_0, contacts)
        for _ in range(_SUBSTEPS):
            solver.step(state_0, state_1, control, contacts, _DT / _SUBSTEPS)
            state_0, state_1 = state_1, state_0
        energies.append(energy(state_0))
    return np.array(energies)


class TestFeatherPGSRolling(unittest.TestCase):
    def _declared(self, device):
        # A sphere and a lying cylinder on the ground, a sphere on a box, two touching spheres, a box on a box, a
        # standing cylinder: the share and the held points the collision writes.
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        sphere = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, _R), wp.quat_identity()), mass=_MASS)
        builder.add_shape_sphere(sphere, radius=_R)
        cylinder = _lying(builder, x=1.0)
        builder.add_shape_cylinder(cylinder, radius=_R, half_height=_HALF)
        standing = builder.add_body(xform=wp.transform(wp.vec3(2.0, 0.0, _HALF), wp.quat_identity()), mass=_MASS)
        builder.add_shape_cylinder(standing, radius=_R, half_height=_HALF)
        model = builder.finalize(device=device)
        c = _contacts(model, device)
        shape_body = model.shape_body.numpy()
        for i in range(len(c["shape0"])):
            if int(shape_body[c["shape0"][i]]) >= 0 and int(shape_body[c["shape1"][i]]) >= 0:
                continue  # only the bodies' contacts with the ground
            side = 1 if int(shape_body[c["shape0"][i]]) < 0 else 0  # the body's side; the ground is the other
            body = int(shape_body[c["shape1" if side else "shape0"][i]])
            margin = float(c["margin1" if side else "margin0"][i])
            point = c["point1" if side else "point0"][i]
            share = float(c["point_share"][i])
            if body == sphere:
                # A sphere touches at its own point, held at its centre with its radius.
                self.assertEqual(share, float(side))
                np.testing.assert_allclose(point, 0.0, atol=1e-6)
                self.assertAlmostEqual(margin, _R, delta=1e-6)
            elif body == cylinder and margin > 0.0:
                # A lying cylinder's side: held at its axis point with its radius, touching at its own point.
                self.assertEqual(share, float(side))
                self.assertAlmostEqual(float(np.hypot(point[0], point[1])), 0.0, delta=1e-6)
                self.assertAlmostEqual(margin, _R, delta=1e-6)
            elif body == standing:
                # A standing cylinder's cap face: its rim points keep no radius and the midpoint.
                self.assertEqual(margin, 0.0)
                self.assertEqual(share, 0.5)

    def test_declared_share_and_curved_sides_cpu(self):
        self._declared(wp.get_device("cpu"))

    def test_shares_between_two_sides_cpu(self):
        # Two touching spheres of radii r and 3r meet at r / 4r of the way; a sphere on a box touches at the sphere's
        # point; a box on a box keeps the midpoint.
        builder = newton.ModelBuilder()
        small = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()), mass=1.0)
        builder.add_shape_sphere(small, radius=0.01)
        big = builder.add_body(xform=wp.transform(wp.vec3(0.0399, 0.0, 0.0), wp.quat_identity()), mass=1.0)
        builder.add_shape_sphere(big, radius=0.03)
        ball = builder.add_body(xform=wp.transform(wp.vec3(1.0, 0.0, 0.0499), wp.quat_identity()), mass=1.0)
        builder.add_shape_sphere(ball, radius=0.01)
        slab = builder.add_body(xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()), mass=1.0)
        builder.add_shape_box(slab, hx=0.1, hy=0.1, hz=0.04)
        top = builder.add_body(xform=wp.transform(wp.vec3(2.0, 0.0, 0.0399), wp.quat_identity()), mass=1.0)
        builder.add_shape_box(top, hx=0.02, hy=0.02, hz=0.02)
        bottom = builder.add_body(xform=wp.transform(wp.vec3(2.0, 0.0, 0.0), wp.quat_identity()), mass=1.0)
        builder.add_shape_box(bottom, hx=0.1, hy=0.1, hz=0.02)
        model = builder.finalize(device="cpu")
        c = _contacts(model, "cpu")
        shape_body = model.shape_body.numpy()
        seen = set()
        for i in range(len(c["shape0"])):
            pair = {int(shape_body[c["shape0"][i]]), int(shape_body[c["shape1"][i]])}
            share = float(c["point_share"][i])
            body0 = int(shape_body[c["shape0"][i]])
            if pair == {small, big}:
                expected = 0.25 if body0 == small else 0.75
                self.assertAlmostEqual(share, expected, delta=1e-6)
                seen.add("spheres")
            elif pair == {ball, slab}:
                self.assertEqual(share, 0.0 if body0 == ball else 1.0)
                seen.add("ball on box")
            elif pair == {top, bottom}:
                self.assertEqual(share, 0.5)
                seen.add("box on box")
        self.assertEqual(seen, {"spheres", "ball on box", "box on box"})

    def _keeps_energy(self, shape, device):
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        if shape == "sphere":
            inertia = 0.4 * _MASS * _R * _R
            body = builder.add_body(
                xform=wp.transform(wp.vec3(0.0, 0.0, _R), wp.quat_identity()),
                mass=_MASS,
                inertia=wp.mat33(*np.ravel(np.eye(3) * inertia)),
                lock_inertia=True,
            )
            builder.add_shape_sphere(body, radius=_R)
        else:
            inertia = 0.5 * _MASS * _R * _R
            body = _lying(builder)
            if shape == "capsule":
                builder.add_shape_capsule(body, radius=_R, half_height=_HALF)
            else:
                builder.add_shape_cylinder(body, radius=_R, half_height=_HALF)
        model = builder.finalize(device=device)
        energies = _roll(model, body, _R, 0.06, 50, inertia, device)
        # Rolling without slip on flat ground keeps its kinetic energy; main's midpoint row point pumped it 5-14x in
        # these 0.5 s.
        np.testing.assert_allclose(energies, energies[0], rtol=2e-3, err_msg=shape)

    def test_rolling_sphere_keeps_its_energy_cpu(self):
        self._keeps_energy("sphere", wp.get_device("cpu"))

    def test_rolling_capsule_keeps_its_energy_cpu(self):
        self._keeps_energy("capsule", wp.get_device("cpu"))

    def test_rolling_cylinder_keeps_its_energy_cpu(self):
        self._keeps_energy("cylinder", wp.get_device("cpu"))

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA")
    def test_rolling_cylinder_keeps_its_energy_cuda(self):
        self._keeps_energy("cylinder", wp.get_device("cuda:0"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
