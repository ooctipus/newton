# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Authored contact materials (:attr:`~newton.ShapeFlags.CONTACT_MATERIAL`) under FeatherPGS.

A contact of a shape that authors its stiffness and damping takes MuJoCo's soft contact at rest
(:func:`~newton._src.solvers.feather_pgs.row_laws.authored_contact_law`): the law is checked against MJWarp's own
constraint parameters and against the numbers MJWarp produced on the NIST assembly scene, and a resting sphere is
checked to sink to MuJoCo's static depth. A model without an authored shape keeps the rigid law and its routes.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.row_laws import authored_contact_law, contact_material_rhs
from newton.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_test_devices

SUBSTEP = 0.0125 / 10.0
"""NIST's substep, 1.25 ms, where MuJoCo's time-constant floor ``2 dt`` (2.5 ms) binds its stiff pairs."""

PART, GROUND, FINGER = (1.6e5, 800.0), (2.5e3, 100.0), (1.0e6, 2000.0)
"""NIST's authored materials (ke [N/m], kd [N s/m]): parts and sockets, the ground (Newton's default), the fingers."""


@wp.kernel
def _authored(
    materials: wp.array[wp.vec4],
    phi: wp.array[float],
    dt: float,
    kappa: wp.array[float],
    beta: wp.array[float],
    rho: wp.array[float],
):
    i = wp.tid()
    m = materials[i]
    k, b = authored_contact_law(m[0], m[1], m[2], m[3], phi[i], dt)
    _rhs, r = contact_material_rhs(phi[i], 0.0, k, b, dt)
    kappa[i] = k
    beta[i] = b
    rho[i] = r


def authored(pairs, phis, dt=SUBSTEP, device="cpu"):
    """The law's ``(kappa, beta, rho)`` for each ``((ke_a, kd_a), (ke_b, kd_b))`` pair at gap ``phi``."""
    materials = wp.array([wp.vec4(*a, *b) for a, b in pairs], dtype=wp.vec4, device=device)
    phi = wp.array(np.asarray(phis, dtype=np.float32), dtype=float, device=device)
    out = [wp.zeros(len(pairs), dtype=float, device=device) for _ in range(3)]
    wp.launch(_authored, dim=len(pairs), inputs=[materials, phi, dt], outputs=out, device=device)
    return tuple(a.numpy().astype(np.float64) for a in out)


def mujoco_reference(a, b, phi, dt):
    """MuJoCo's resting spring and dashpot per unit effective mass, written independently in float64: Newton's
    ``convert_solref(ke, kd, 1, 1)`` per geom, the mean of two geoms' solref, refsafe, ``_contact_kbimp`` on the
    default solimp, and the resting force ``(k imp pos + b v) / R`` with ``R = (1 - imp) / imp * w``."""

    def solref(ke, kd):
        return (2.0 / kd, kd / 2.0 / np.sqrt(ke)) if ke > 0.0 and kd > 0.0 else (0.02, 1.0)

    tc, dr = 0.5 * (np.asarray(solref(*a)) + np.asarray(solref(*b)))
    tc = max(tc, 2.0 * dt)
    dmin, dmax, width, mid = 0.9, 0.95, 0.001, 0.5
    k = 1.0 / (dmax * dmax * tc * tc * dr * dr)
    damp = 2.0 / (dmax * tc)
    x = min(abs(phi) / width, 1.0)
    y = 2.0 * x * x if x <= mid else 1.0 - 2.0 * (1.0 - x) ** 2
    imp = dmin + y * (dmax - dmin)
    return k * imp * imp / (1.0 - imp), damp * imp / (1.0 - imp)


class TestAuthoredContactLaw(unittest.TestCase):
    def test_law_matches_mujoco_reference(self):
        """The law is MuJoCo's pair rule and resting row, over NIST's materials, depths and the refsafe floor."""
        materials = (PART, GROUND, FINGER, (3.0e4, 50.0), (0.0, 100.0))
        pairs, phis, expected = [], [], []
        for i, a in enumerate(materials):
            for b in materials[i:]:
                for phi in (0.0, -2.0e-4, -5.0e-4, -8.0e-4, -3.0e-3, 4.0e-4):
                    pairs.append((a, b))
                    phis.append(phi)
                    expected.append(mujoco_reference(a, b, phi, SUBSTEP))
        kappa, beta, _rho = authored(pairs, phis)
        expected = np.asarray(expected)
        np.testing.assert_allclose(kappa, expected[:, 0], rtol=2e-5)
        np.testing.assert_allclose(beta, expected[:, 1], rtol=2e-5)

    def test_law_matches_mjwarp_on_nist(self):
        """The resting stiffness and damping MJWarp's NIST rows held (a stage-1 dump of the demo, 8 worlds, the trained
        policy, steps 60 to 1,000): ``K = kappa / (g w)`` [N/m] and ``C = beta / (g w)`` [N s/m] at MJWarp's own
        ``body_invweight0`` sum ``w`` and pyramid factor ``g = mu^2 (1 + mu^2) / 2``, the smallest at ``imp = 0.9``
        and the largest at ``imp = 0.95`` (deeper than 1 mm)."""
        # (pair, gap, w [1/kg], mu, K, C) from MJWarp's rows.
        rows = (
            ((PART, GROUND), 0.0, 200.00003, 1.0, 354.5752402416547, 8.42115151729455),
            ((PART, GROUND), -2.0e-3, 20.0, 1.0, 7901.233364361962, 177.77774304520327),
            ((PART, PART), 0.0, 200.00204, 0.75, 16338.608668265257, 86.23153852977404),
            ((PART, PART), -2.0e-3, 20.00017547607422, 0.75, 364085.62394359394, None),
        )
        kappa, beta, _rho = authored([row[0] for row in rows], [row[1] for row in rows])
        for (_pair, _phi, w, mu, stiffness, damping), k, b in zip(rows, kappa, beta, strict=True):
            g = mu * mu * (1.0 + mu * mu) / 2.0
            self.assertAlmostEqual(k / (g * w) / stiffness, 1.0, delta=2e-4)
            if damping is not None:
                self.assertAlmostEqual(b / (g * w) / damping, 1.0, delta=2e-4)

    def test_relative_compliance_on_nist(self):
        """The row's relative compliance ``rho = 1 / (h (h kappa + beta))`` at NIST's substep: 0.04 a seated part (the
        refsafe floor, deeper than 1 mm), 0.45 a part touching the ground."""
        _kappa, _beta, rho = authored([(PART, PART), (PART, GROUND)], [-2.0e-3, 0.0])
        self.assertAlmostEqual(rho[0], 0.04, delta=5e-4)
        self.assertAlmostEqual(rho[1], 0.451, delta=2e-3)


def rest_depth(material, dt=SUBSTEP):
    """MuJoCo's static depth of a body resting on one contact with the ground: ``kappa(phi) phi + g = 0``."""
    low, high = 0.0, 0.1
    for _ in range(200):
        depth = 0.5 * (low + high)
        kappa, _beta = mujoco_reference(material, GROUND, -depth, dt)
        if kappa * depth < 9.81:
            low = depth
        else:
            high = depth
    return 0.5 * (low + high)


def resting_sphere(device, *, articulated, authored_material, steps=800):
    """A 50 g sphere dropped onto the ground (default material) through FeatherPGS at NIST's substep: its final height
    over its radius (negative: depth), its speed, and the solver. A free sphere has a twin far beside it, so the free
    bodies keep their own run (regions on the CPU, the colored mapping on CUDA)."""
    radius = 0.02
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))
    xform = wp.transform(wp.vec3(0.0, 0.0, radius + 0.002), wp.quat_identity())
    if articulated:
        body = builder.add_link(xform=xform)
        joint = builder.add_joint_prismatic(parent=-1, child=body, axis=newton.Axis.Z, parent_xform=xform)
        builder.add_articulation([joint])
    else:
        body = builder.add_body(xform=xform)
    cfg = newton.ModelBuilder.ShapeConfig(density=0.05 / (4.0 / 3.0 * np.pi * radius**3), mu=0.5)
    if authored_material:
        cfg.ke, cfg.kd = PART
    builder.add_shape_sphere(body, radius=radius, cfg=cfg)
    if not articulated:
        twin = builder.add_body(xform=wp.transform(wp.vec3(0.5, 0.0, radius + 0.002), wp.quat_identity()))
        builder.add_shape_sphere(twin, radius=radius, cfg=cfg)
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=16)
    contacts = pipeline.contacts()
    solver = SolverFeatherPGS(model, pgs_iterations=16, dense_max_constraints=16, mf_max_constraints=64)
    state, next_state = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    control = model.control()
    for _ in range(steps):
        state.clear_forces()
        pipeline.collide(state, contacts)
        solver.step(state, next_state, control, contacts, SUBSTEP)
        state, next_state = next_state, state
    height = float(state.body_q.numpy()[body, 2]) - radius
    speed = float(np.linalg.norm(state.body_qd.numpy()[body]))
    return height, speed, solver


def test_routes_follow_authorship(test, device):
    """A model without an authored shape takes no compliance law (today's rows and sweeps); one with an authored
    shape takes it, keeps its worlds' acts word (a weighted step from zero at a positive residual stays at zero), and
    its contact's normal row stores ``-rho`` and the weight ``1 / (1 + rho)``."""
    for authored_material in (False, True):
        # Resting, so its world acts (a world whose rows all separate holds them unwritten).
        _height, _speed, solver = resting_sphere(device, articulated=True, authored_material=authored_material, steps=200)
        laws = solver.laws.set
        test.assertEqual(laws.authored, authored_material)
        test.assertEqual(laws.material, authored_material)
        test.assertEqual(laws.weighted, authored_material)
        test.assertTrue(laws.separable)
        if authored_material:
            count = int(solver.placement.constraint_count.numpy()[0])
            normal = solver.placement.row_type.numpy()[0, :count] == 0
            eps = solver.placement.row_eps.numpy()[0, :count][normal]
            weight = solver.placement.row_w.numpy()[0, :count][normal]
            test.assertGreater(len(eps), 0)
            test.assertTrue(np.all(eps < 0.0))
            np.testing.assert_allclose(weight, 1.0 / (1.0 - eps), rtol=1e-6)


def test_rest_depth_is_mujocos(test, device):
    """An authored sphere rests at MuJoCo's static depth (the mass cancels), as a dense row (articulated) and as a free
    body's region; the same sphere unauthored rests rigid."""
    expected = rest_depth(PART)
    for articulated in (True, False):
        with test.subTest(articulated=articulated):
            height, speed, solver = resting_sphere(device, articulated=articulated, authored_material=True)
            # The free body's contact is a region on the CPU: the region law reads the row's relative compliance.
            if not articulated and wp.get_device(device).is_cpu:
                test.assertIsNotNone(solver.sweep.regions)
            test.assertAlmostEqual(-height, expected, delta=0.02 * expected)
            test.assertLess(speed, 1e-3)
            rigid, rigid_speed, _ = resting_sphere(device, articulated=articulated, authored_material=False)
            test.assertLess(abs(rigid), 0.1 * expected)
            test.assertLess(rigid_speed, 1e-3)


class TestAuthoredContacts(unittest.TestCase):
    @unittest.skipUnless(wp.get_device().is_cuda, "The predictor publishes its body twists on CUDA")
    def test_separated_worlds_match_the_complete_path(self):
        """Arms over tossed free boxes, every shape authored: a world none of whose rows can act at the predictor (by
        the authored law's residual) takes no sweep, and every world steps as on the complete path."""
        from newton.tests.test_feather_pgs_separated_worlds import _rollout  # noqa: PLC0415
        from newton.tests.test_feather_pgs_mixed_tree_free_body import _build  # noqa: PLC0415

        model = _build(wp.get_device(), worlds=6)
        model.shape_flags.assign(model.shape_flags.numpy() | int(newton.ShapeFlags.CONTACT_MATERIAL))
        model.shape_material_ke.fill_(PART[0])
        model.shape_material_kd.fill_(PART[1])
        plain_solver, plain, plain_masks = _rollout(model, separated=False)
        self.assertTrue(plain_solver.laws.set.authored)
        self.assertEqual(plain_masks.shape[1], 0)
        _solver, poses, masks = _rollout(model)
        self.assertGreater(int(masks.sum()), 0)
        self.assertLess(int(masks.sum()), masks.size)
        np.testing.assert_array_equal(poses, plain)


for _name, _function in (
    ("test_routes_follow_authorship", test_routes_follow_authorship),
    ("test_rest_depth_is_mujocos", test_rest_depth_is_mujocos),
):
    add_function_test(TestAuthoredContacts, _name, _function, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main()
