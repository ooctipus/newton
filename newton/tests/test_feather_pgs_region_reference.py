# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regime tests of the region block law's float64 reference (:mod:`feather_pgs_region_reference`).

Each test states a problem whose answer is known in closed form, or by the
law's own identities, and checks that the reference returns it: rest and the
total force on an indeterminate support, the tipping threshold, Coulomb's
angle, the spin capacity, compliance, a speculative gap, the pass rule, a
region whose normals exceed rank three, a wedged body, two movable bodies.
"""

import unittest

import numpy as np

from newton.tests import feather_pgs_region_reference as ref

GRAVITY = 9.81
DT = 0.01
MASS = 1.5
HALF = 0.1
INERTIA = np.full(3, 2.0 * MASS * HALF * HALF / 3.0)
CORNERS = np.array([[x, y, -HALF] for x in (-HALF, HALF) for y in (-HALF, HALF)])
UP = np.array([0.0, 0.0, 1.0])
ANCHOR = np.array([0.0, 0.0, -HALF])
TANGENTS = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def point_speed(points, directions, linear, angular):
    """The speed of a body's ``points`` along ``directions`` at the velocity ``(linear, angular)``."""
    points, directions = np.atleast_2d(points), np.atleast_2d(directions)
    return np.einsum("ij,ij->i", directions, linear + np.cross(angular, points))


def box_on_ground(linear, angular, mu=0.5, mu_t=0.02, eps=0.0, bias=0.0, normals=None, spin=True):
    """A box on the ground as one region: four corner candidates and one friction unit at the base's centre, at the
    free velocity ``(linear, angular)``."""
    normals = np.tile(UP, (4, 1)) if normals is None else normals
    rows = ref.body_rows(MASS, INERTIA, CORNERS, normals)
    rhs = point_speed(CORNERS, normals, linear, angular) + bias
    unit_rows = ref.body_rows(MASS, INERTIA, np.tile(ANCHOR, (2, 1)), TANGENTS)
    unit_rhs = point_speed(np.tile(ANCHOR, (2, 1)), TANGENTS, linear, angular)
    coef = np.array([mu, mu])
    if spin:
        unit_rows = np.vstack([unit_rows, ref.spin_row(INERTIA, UP)])
        unit_rhs = np.append(unit_rhs, angular @ UP)
        coef = np.append(coef, mu_t)
    return ref.Region(rows, rhs, np.full(4, eps), ref.FrictionUnit(unit_rows, unit_rhs, coef))


def end_velocity(linear, angular, velocity):
    dv, dw = ref.unwhiten(MASS, INERTIA, velocity)
    return linear + dv, angular + dw


FALL = np.array([0.0, 0.0, -GRAVITY * DT])
STILL = np.zeros(3)


class TestRegionBlock(unittest.TestCase):
    def test_one_candidate_is_the_scalar_row(self):
        region = ref.Region(ref.body_rows(MASS, INERTIA, CORNERS[:1], UP), np.array([-0.3]), np.zeros(1))
        velocity, states, _ = ref.solve([region], 6, 1)
        scalar_velocity, scalar_states = np.zeros(6), ref.new_states([region])
        ref.scalar_sweep([region], scalar_states, scalar_velocity)
        np.testing.assert_allclose(states[0].lam, scalar_states[0].lam, rtol=1e-13)
        np.testing.assert_allclose(velocity, scalar_velocity, rtol=1e-13, atol=1e-16)

    def test_resting_box_settles_in_one_visit_and_is_passed(self):
        """Q13: four corners are an indeterminate support; the velocity and the total force are the answer."""
        region = box_on_ground(FALL, STILL)
        velocity, states, log = ref.solve([region], 6, 3)
        self.assertEqual([outcomes[0] for outcomes in log], ["stick", "passed", "passed"])
        linear, angular = end_velocity(FALL, STILL, velocity)
        np.testing.assert_allclose(linear, 0.0, atol=1e-14)
        np.testing.assert_allclose(angular, 0.0, atol=1e-13)
        self.assertAlmostEqual(states[0].lam.sum(), MASS * GRAVITY * DT, delta=1e-13)
        np.testing.assert_allclose(states[0].phi, 0.0, atol=1e-13)

    def test_tipping_threshold(self):
        """Q08: a box pushed at its mass centre tips about its leading edge past ``a = g`` (a cube), in one visit."""
        for push, tips in ((0.5 * GRAVITY, False), (1.5 * GRAVITY, True)):
            free = FALL + np.array([push * DT, 0.0, 0.0])
            region = box_on_ground(free, STILL, mu=2.0)
            velocity, states, log = ref.solve([region], 6, 2)
            self.assertEqual([outcomes[0] for outcomes in log], ["stick", "passed"])
            linear, angular = end_velocity(free, STILL, velocity)
            if not tips:
                np.testing.assert_allclose(np.concatenate([linear, angular]), 0.0, atol=1e-13)
                continue
            expected = DT * 3.0 * (push - GRAVITY) / (8.0 * HALF)
            np.testing.assert_allclose(angular, [0.0, expected, 0.0], atol=1e-12)
            leading = CORNERS[:, 0] > 0.0
            np.testing.assert_array_equal(states[0].lam > 0.0, leading)
            edge = linear + np.cross(angular, CORNERS[leading])
            np.testing.assert_allclose(edge, 0.0, atol=1e-12)

    def test_coulomb_angle(self):
        """Q05: below the friction angle the box sticks in one visit; above it slides at ``g (sin - mu cos)``."""
        mu = 0.5
        for degrees, sticks in ((20.0, True), (35.0, False)):
            theta = np.radians(degrees)
            free = GRAVITY * DT * np.array([np.sin(theta), 0.0, -np.cos(theta)])
            region = box_on_ground(free, STILL, mu=mu)
            velocity, _, log = ref.solve([region], 6, 60)
            linear, angular = end_velocity(free, STILL, velocity)
            if sticks:
                self.assertEqual(log[0], ["stick"])
                self.assertEqual(log[1], ["passed"])
                np.testing.assert_allclose(np.concatenate([linear, angular]), 0.0, atol=1e-13)
                continue
            self.assertEqual({outcomes[0] for outcomes in log}, {"slide"})
            expected = GRAVITY * DT * (np.sin(theta) - mu * np.cos(theta))
            np.testing.assert_allclose(linear, [expected, 0.0, 0.0], atol=1e-10)
            np.testing.assert_allclose(angular, 0.0, atol=1e-9)
            # Eight sweeps of the two steps stand within the float32 floor of the region's fixed point.
            velocity8, _, _ = ref.solve([region], 6, 8)
            self.assertLess(abs(end_velocity(free, STILL, velocity8)[0][0] - expected), 1e-6 * expected)

    def test_spin_capacity(self):
        """Q06: the unit's spin row holds a spin within ``mu_t`` times the load and takes that much from a larger."""
        mu_t = 0.02
        capacity = mu_t * MASS * GRAVITY * DT / INERTIA[2]
        for spin, holds in ((0.5 * capacity, True), (3.0 * capacity, False)):
            free_spin = np.array([0.0, 0.0, spin])
            region = box_on_ground(FALL, free_spin, mu_t=mu_t)
            velocity, _, log = ref.solve([region], 6, 4)
            linear, angular = end_velocity(FALL, free_spin, velocity)
            np.testing.assert_allclose(linear, 0.0, atol=1e-13)
            self.assertEqual(log[0], ["stick" if holds else "slide"])
            np.testing.assert_allclose(angular, [0.0, 0.0, 0.0 if holds else spin - capacity], atol=1e-12)

    def test_compliant_candidates_carry_their_own_impulse(self):
        """A compliant candidate's impulse is its spring's: every closing candidate is active (four here, not at most
        three), the impulses are unique and they are today's rows' fixed point."""
        eps = 0.8
        region = box_on_ground(FALL, np.array([0.3, -0.2, 0.0]), eps=eps, mu=5.0)
        velocity, states, log = ref.solve([region], 6, 2)
        self.assertEqual([outcomes[0] for outcomes in log], ["stick", "passed"])
        lam = states[0].lam
        self.assertTrue((lam > 0.0).all())
        residual = region.rows @ velocity + region.rhs + eps * lam
        np.testing.assert_allclose(residual, 0.0, atol=1e-13)
        scalar_velocity, scalar_states = np.zeros(6), ref.new_states([region])
        for _ in range(4000):
            ref.scalar_sweep([region], scalar_states, scalar_velocity)
        np.testing.assert_allclose(scalar_states[0].lam, lam, rtol=1e-9)
        np.testing.assert_allclose(scalar_velocity, velocity, atol=1e-11)
        # The centred fall has the closed form lam = g dt / (eps + 4 / m) a corner.
        centred = box_on_ground(FALL, STILL, eps=eps)
        _, states, _ = ref.solve([centred], 6, 1)
        np.testing.assert_allclose(states[0].lam, GRAVITY * DT / (eps + 4.0 / MASS), rtol=1e-12)

    def test_rigid_and_compliant_candidates_in_one_region(self):
        region = box_on_ground(FALL, np.array([0.1, 0.25, 0.0]), mu=5.0)
        region.eps[:2] = 0.5
        velocity, states, _ = ref.solve([region], 6, 1)
        residual = region.rows @ velocity + region.rhs + region.eps * states[0].lam
        self.assertGreaterEqual(residual.min(), -1e-13)
        np.testing.assert_allclose(states[0].lam * residual, 0.0, atol=1e-14)
        scalar_velocity, scalar_states = np.zeros(6), ref.new_states([region])
        for _ in range(20000):
            ref.scalar_sweep([region], scalar_states, scalar_velocity)
        np.testing.assert_allclose(scalar_velocity, velocity, atol=1e-9)

    def test_coplanar_candidates_are_solved_exactly(self):
        """Thirteen candidates on one plane have rank three: the stick solve holds at most three of them and leaves
        none closing, whatever the load's tilt (a solve that lets a dependent candidate wait leaves one closing)."""
        rng = np.random.default_rng(11)
        points = np.column_stack([rng.uniform(-HALF, HALF, size=(13, 2)), np.full(13, -HALF)])
        normals = np.tile(UP, (13, 1))
        rows = ref.body_rows(MASS, INERTIA, points, normals)
        unit = box_on_ground(FALL, STILL, mu=5.0, mu_t=5.0).unit
        sticks = 0
        for _ in range(40):
            free_linear = FALL * rng.uniform(0.2, 3.0) + np.append(rng.normal(size=2) * 0.01, 0.0)
            free_angular = rng.normal(size=3) * 0.3
            rhs = point_speed(points, normals, free_linear, free_angular) + rng.uniform(0.0, 0.02, size=13)
            unit.rhs[:] = np.append(point_speed(np.tile(ANCHOR, (2, 1)), TANGENTS, free_linear, free_angular), free_angular @ UP)
            region = ref.Region(rows, rhs, np.zeros(13), unit)
            velocity, states, log = ref.solve([region], 6, 1)
            if log[0] != ["stick"]:
                continue
            sticks += 1
            residual = rows @ velocity + rhs
            self.assertGreaterEqual(residual.min(), -1e-12 * np.abs(rhs).max())
            self.assertLessEqual((states[0].lam > 0.0).sum(), 3)
            np.testing.assert_allclose(states[0].lam * residual, 0.0, atol=1e-15)
            np.testing.assert_allclose(unit.rows @ velocity + unit.rhs, 0.0, atol=1e-14)
        self.assertGreater(sticks, 20)

    def test_speculative_candidate(self):
        """Q15: a candidate across an open gap takes impulse only when the approach would close the gap in the step."""
        gap = 0.004
        row = ref.body_rows(MASS, INERTIA, np.zeros((1, 3)), UP)
        for approach, closes in ((0.3, False), (1.0, True)):
            region = ref.Region(row, np.array([-approach + gap / DT]), np.zeros(1))
            velocity, states, log = ref.solve([region], 6, 2)
            self.assertEqual(log[0], ["stick" if closes else "open"])
            self.assertEqual(log[1], ["passed"])
            after = -approach + ref.unwhiten(MASS, INERTIA, velocity)[0][2]
            self.assertAlmostEqual(after, -gap / DT if closes else -approach, delta=1e-13)
            self.assertEqual(states[0].lam[0] > 0.0, closes)


class TestPassRule(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.regions = []
        for k in range(5):
            tilt = rng.normal(size=3) * 0.4 + UP
            normal = tilt / np.linalg.norm(tilt)
            points = rng.uniform(-HALF, HALF, size=(4, 3))
            tangent = np.cross(normal, [1.0, 0.0, 0.0])
            tangent /= np.linalg.norm(tangent)
            tangents = np.array([tangent, np.cross(normal, tangent)])
            anchor = points.mean(axis=0)
            free_linear = np.array([0.02, -0.01, -0.1])
            free_angular = np.array([0.2, 0.1, -0.3])
            rows = ref.body_rows(MASS, INERTIA, points, np.tile(normal, (4, 1)))
            rhs = point_speed(points, np.tile(normal, (4, 1)), free_linear, free_angular) + 0.02 * k
            unit = ref.FrictionUnit(
                np.vstack(
                    [ref.body_rows(MASS, INERTIA, np.tile(anchor, (2, 1)), tangents), ref.spin_row(INERTIA, normal)]
                ),
                np.append(
                    point_speed(np.tile(anchor, (2, 1)), tangents, free_linear, free_angular), free_angular @ normal
                ),
                np.array([0.6, 0.6, 0.01]),
            )
            self.regions.append(ref.Region(rows, rhs, np.zeros(4), unit))

    def test_passing_is_the_visit(self):
        """A settled region's visit at the velocity it left returns that velocity, so passing it changes nothing
        beyond the visit's own rounding: on regions that keep moving their body, and on a body at rest beside two
        regions that hold nothing."""
        velocity, _, _ = ref.solve(self.regions, 6, 8, passing=True)
        visited, _, visited_log = ref.solve(self.regions, 6, 8, passing=False)
        self.assertEqual(sum(outcomes.count("passed") for outcomes in visited_log), 0)
        np.testing.assert_allclose(velocity, visited, atol=1e-13 * np.abs(visited).max())
        away = [
            ref.Region(region.rows, np.abs(region.rhs) + 1.0, region.eps, region.unit) for region in self.regions[:2]
        ]
        resting = [away[0], box_on_ground(FALL, STILL), away[1]]
        velocity, _, log = ref.solve(resting, 6, 8, passing=True)
        visited, _, _ = ref.solve(resting, 6, 8, passing=False)
        # The first region was seen before the box moved the body: it is visited once more, then every region is
        # passed (the cycle back to the last visit that moved the body).
        self.assertEqual(log[0], ["open", "stick", "open"])
        self.assertEqual(log[1], ["open", "passed", "passed"])
        self.assertEqual(log[2:], [["passed"] * 3] * 6)
        np.testing.assert_allclose(velocity, visited, atol=1e-13 * np.abs(visited).max())

    def test_a_settled_visit_is_idempotent(self):
        region = box_on_ground(FALL + np.array([0.004, 0.0, 0.0]), np.array([0.05, 0.0, 0.01]))
        velocity, states = np.zeros(6), ref.new_states([region])
        self.assertEqual(ref.visit(region, states[0], velocity), "stick")
        first, lam, phi = velocity.copy(), states[0].lam.copy(), states[0].phi.copy()
        self.assertEqual(ref.visit(region, states[0], velocity), "stick")
        np.testing.assert_allclose(velocity, first, atol=1e-15)
        self.assertAlmostEqual(states[0].lam.sum(), lam.sum(), delta=1e-15)
        np.testing.assert_allclose(states[0].phi, phi, atol=1e-15)

    def test_anything_that_moves_the_body_ends_the_pass(self):
        """A dense row or a speed row that changes the body's velocity between two sweeps is seen by its bits."""
        region = box_on_ground(FALL, STILL)
        velocity, states = np.zeros(6), ref.new_states([region])
        self.assertEqual(ref.sweep([region], states, velocity), ["stick"])
        self.assertEqual(ref.sweep([region], states, velocity), ["passed"])
        velocity[2] -= 0.01
        self.assertEqual(ref.sweep([region], states, velocity), ["stick"])
        np.testing.assert_allclose(end_velocity(FALL, STILL, velocity)[0], 0.0, atol=1e-14)

    def test_a_sliding_region_is_not_passed(self):
        theta = np.radians(35.0)
        free = GRAVITY * DT * np.array([np.sin(theta), 0.0, -np.cos(theta)])
        _, _, log = ref.solve([box_on_ground(free, STILL)], 6, 4)
        self.assertEqual(log, [["slide"]] * 4)


class TestRegionShapes(unittest.TestCase):
    def test_normals_past_rank_three(self):
        """A region whose normals lean two degrees apart: the block over the candidates' own covectors holds
        independent candidates only (at most the rows' rank, four here), and the region flattened onto one normal
        (at most three) differs from it by the lean."""
        lean = np.radians(2.0)
        toward = 1.3 * np.arange(4) + 0.4
        normals = np.stack(
            [np.sin(lean) * np.cos(toward), np.sin(lean) * np.sin(toward), np.full(4, np.cos(lean))], axis=1
        )
        for free_spin in (STILL, np.array([0.15, -0.05, 0.0])):
            own = box_on_ground(FALL, free_spin, normals=normals, mu=5.0)
            flat = box_on_ground(FALL, free_spin, mu=5.0)
            self.assertEqual(np.linalg.matrix_rank(own.rows), 4)
            self.assertEqual(np.linalg.matrix_rank(flat.rows), 3)
            velocity_own, states_own, _ = ref.solve([own], 6, 1)
            velocity_flat, states_flat, _ = ref.solve([flat], 6, 1)
            active = states_own[0].lam > 0.0
            self.assertEqual(active.sum(), np.linalg.matrix_rank(own.rows[active]))
            self.assertLessEqual((states_flat[0].lam > 0.0).sum(), 3)
            residual = own.rows @ velocity_own + own.rhs
            self.assertGreaterEqual(residual.min(), -1e-13)
            np.testing.assert_allclose(states_own[0].lam * residual, 0.0, atol=1e-14)
            load = states_flat[0].lam.sum()
            self.assertAlmostEqual(states_own[0].lam.sum(), load, delta=2.0 * lean * load)
            difference = np.abs(own.rows @ (velocity_own - velocity_flat)).max()
            self.assertLess(difference, 2.0 * np.sin(lean) * GRAVITY * DT)

    def test_wedged_body_keeps_a_bounded_compromise(self):
        """No velocity satisfies a floor and a ceiling that both push: each region alone has one, and the truncated
        sweeps return a velocity bounded by the demands at every sweep count while the impulses grow."""
        floor_push, ceiling_push = 0.3, 0.2
        top = CORNERS * np.array([1.0, 1.0, -1.0])
        floor = ref.Region(
            ref.body_rows(MASS, INERTIA, CORNERS, np.tile(UP, (4, 1))), np.full(4, -floor_push), np.zeros(4)
        )
        ceiling = ref.Region(
            ref.body_rows(MASS, INERTIA, top, np.tile(-UP, (4, 1))), np.full(4, -ceiling_push), np.zeros(4)
        )
        loads = []
        for sweeps in (8, 32):
            velocity, states, _ = ref.solve([floor, ceiling], 6, sweeps)
            linear, angular = ref.unwhiten(MASS, INERTIA, velocity)
            self.assertAlmostEqual(linear[2], -ceiling_push, delta=1e-12)
            np.testing.assert_allclose(angular, 0.0, atol=1e-12)
            violation, _ = ref.kkt([floor, ceiling], states, velocity)
            self.assertAlmostEqual(violation, floor_push + ceiling_push, delta=1e-12)
            loads.append(states[0].lam.sum())
            scalar_velocity, scalar_states = np.zeros(6), ref.new_states([floor, ceiling])
            for _ in range(sweeps):
                ref.scalar_sweep([floor, ceiling], scalar_states, scalar_velocity)
            self.assertLessEqual(violation, ref.kkt([floor, ceiling], scalar_states, scalar_velocity)[0] + 1e-12)
        self.assertAlmostEqual(loads[1] / loads[0], (32 - 0.5) / (8 - 0.5), delta=0.2)

    def test_two_movable_bodies(self):
        """A region between two movable bodies is the same block over the pair's twelve coordinates. Two stacked boxes
        reach rest and their loads through the regions' sweeps (not in one: the bodies are not coupled in a block),
        and at once when the two regions' candidates are one block."""
        zero = np.zeros((4, 6))
        ground_rows = np.hstack([ref.body_rows(MASS, INERTIA, CORNERS, np.tile(UP, (4, 1))), zero])
        top = CORNERS * np.array([1.0, 1.0, -1.0])
        between_rows = np.hstack(
            [
                -ref.body_rows(MASS, INERTIA, top, np.tile(UP, (4, 1))),
                ref.body_rows(MASS, INERTIA, CORNERS, np.tile(UP, (4, 1))),
            ]
        )
        fall = np.full(4, -GRAVITY * DT)
        ground = ref.Region(ground_rows, fall, np.zeros(4))
        between = ref.Region(between_rows, np.zeros(4), np.zeros(4))
        weight = MASS * GRAVITY * DT
        errors = []
        for sweeps in (1, 8, 60):
            velocity, states, _ = ref.solve([ground, between], 12, sweeps)
            lower = FALL + ref.unwhiten(MASS, INERTIA, velocity[:6])[0]
            upper = FALL + ref.unwhiten(MASS, INERTIA, velocity[6:])[0]
            errors.append(max(np.abs(lower).max(), np.abs(upper).max()))
        self.assertGreater(errors[0], 0.1 * GRAVITY * DT)
        self.assertLess(errors[1], errors[0])
        self.assertLess(errors[2], 1e-12)
        self.assertAlmostEqual(states[0].lam.sum(), 2.0 * weight, delta=1e-10)
        self.assertAlmostEqual(states[1].lam.sum(), weight, delta=1e-10)
        coupled = ref.Region(np.vstack([ground_rows, between_rows]), np.concatenate([fall, np.zeros(4)]), np.zeros(8))
        velocity, states, _ = ref.solve([coupled], 12, 1)
        np.testing.assert_allclose(FALL + ref.unwhiten(MASS, INERTIA, velocity[:6])[0], 0.0, atol=1e-13)
        np.testing.assert_allclose(FALL + ref.unwhiten(MASS, INERTIA, velocity[6:])[0], 0.0, atol=1e-13)
        self.assertAlmostEqual(states[0].lam[:4].sum(), 2.0 * weight, delta=1e-12)

    def test_regions_and_scalar_rows_reach_one_velocity(self):
        """On a frictionless determinate support the block law and today's rows have the same fixed point."""
        rng = np.random.default_rng(3)
        regions = []
        for _ in range(3):
            tilt = rng.normal(size=3) * 0.3 + UP
            normal = np.tile(tilt / np.linalg.norm(tilt), (5, 1))
            points = rng.uniform(-HALF, HALF, size=(5, 3))
            regions.append(
                ref.Region(
                    ref.body_rows(MASS, INERTIA, points, normal),
                    point_speed(points, normal, FALL * 10.0, np.array([0.4, -0.3, 0.2])),
                    np.zeros(5),
                )
            )
        velocity, states, _ = ref.solve(regions, 6, 200)
        scalar_velocity, scalar_states = np.zeros(6), ref.new_states(regions)
        for _ in range(20000):
            ref.scalar_sweep(regions, scalar_states, scalar_velocity)
        np.testing.assert_allclose(velocity, scalar_velocity, atol=1e-9)
        self.assertLess(ref.kkt(regions, states, velocity)[0], 1e-12)
        two, two_states, _ = ref.solve(regions, 6, 2)
        eight, eight_states = np.zeros(6), ref.new_states(regions)
        for _ in range(8):
            ref.scalar_sweep(regions, eight_states, eight)
        self.assertLessEqual(ref.kkt(regions, two_states, two)[0], ref.kkt(regions, eight_states, eight)[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
