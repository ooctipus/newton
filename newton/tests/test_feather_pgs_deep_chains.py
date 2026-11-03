# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Deep chains and thin axes without armature (Z4.A, Z4.L): the solver and its float32 walk against float64.

Main added armature where its float32 dense Cholesky would fail: every coordinate near the head of a path of 32 joints
or more (the deep-chain rule), and a floor on thin axes of every tree (the assembly floor). The walk, the leaf-first
factor in local frames (:class:`.tree_factor.TreeWalk`), keeps its float32 pivots positive without either, so the
solver adds neither (Z4.L). These tests hold both sides to float64 references that share nothing with the solver:

- **the deep-chain regime:** a tip wrench on ``long_chain(N)`` without armature, N = 32, 106, 300 and 1,000, its
  velocity change after one step of the solver against ``H^-1 J^T`` in float64. Main reads 2.6e-3 to 6.6e-3 (F2) and
  is a known failure of Z4.L; the float32 walk reads about 6e-7;
- **families F2 never ran:** a chain whose masses alternate by 1e3, a branched cable (sprawl's fingers 40 links long)
  and loops cut at a body (the four-bar, the Stewart platform). The float32 walk's pivots stay positive and a leaf
  contact's response within 1e-5 of float64 without armature, on the host and on each device (FL3.1's device walk);
- **thin links**, a chain whose links turn with 1e-6 of their own inertia: ``cond(H)`` 9e11 at 106 links, past float32.
  The float32 walk without armature reads 7e-4 there, and main's factor with its armature failed (non-finite), 0.1 off
  at 32 links where the walk reads 5e-5: the walk is the closer at every length, and 1e-5 is out of float32's reach
  (a documented limit, not a defect of either);
- **the assembly floor's former cases**: the thin ball loop at every rocker inertia around the rounding, the thin ball
  2 m out on a pure tree and the thin hinges, on the walk's local pivots with no floor, from the float32 forward
  kinematics' poses (the floor's case: a thin body far from the origin), and the solver stepping the loop and a bare
  100-link chain without either rule (:class:`TestStepsWithoutArmature`).

``python -m newton.tests.test_feather_pgs_deep_chains report`` prints the measured tables.
"""

import sys
import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.body_cut import BodyCut
from newton._src.solvers.feather_pgs.tree_factor import TreeWalk, local_frames
from newton.solvers import SolverFeatherPGS
from newton.tests import feather_pgs_scenes as scenes
from newton.tests.feather_pgs_law_harness import dense_factor32
from newton.tests.feather_pgs_mass_reference import MassReference
from newton.tests.test_feather_pgs_connect import _LOOP_FRAMES, _four_bar_bars, _loop_anchor_gap
from newton.tests.test_feather_pgs_regimes import LIMIT, known_failure
from newton.tests.test_feather_pgs_walk import (
    contact_wrenches,
    jacobian,
    newton_state_body_q,
    relative,
    transform_point,
)
from newton.tests.test_feather_pgs_walk_device import DeviceWalk, devices

DEEP = (32, 106, 300, 1000)
"""The deep-chain regime's lengths (Z4.A): the deep-chain rule's threshold, the bench's chain, the ZB chains."""
REGIME_TOLERANCE = 1.0e-4
"""The regime's bar on a tip impulse's response, relative: main reads 2.6e-3 to 6.6e-3, the walk about 6e-7."""
WALK_TOLERANCE = 1.0e-5
"""The families' bar on the float32 walk's response, relative to float64 without armature."""


# ---------------------------------------------------------------------------------------------------------------------
# The deep-chain regime: the solver's step against float64
# ---------------------------------------------------------------------------------------------------------------------


def tip_impulse(links: int) -> dict:
    """One step of ``long_chain(links)`` without armature, gravity or contact, at rest, under a wrench on the tip
    body's centre of mass: the joint velocity change over the step's ``h`` against ``H^-1 J^T f`` in float64 (the
    float64 walk of the model with no armature, at the step's start; its own spread against the independent mass matrix
    is below 1e-10, F2's truth floor), and the float32 walk's answer beside it."""
    scene = scenes.long_chain(links=links, armature=0.0)
    model = scene.build("cpu")
    model.set_gravity((0.0, 0.0, 0.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver = SolverFeatherPGS(model, dense_max_constraints=8, angular_damping=0.0)
    state, out, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    joint_q = state.joint_q.numpy().astype(np.float64)
    n = int(model.joint_dof_count)
    tip = int(model.joint_child.numpy()[links])
    force, torque = np.array([0.3, -0.2, 0.5]), np.array([0.0, 0.0, 0.01])
    wrench = np.zeros((model.body_count, 6), dtype=np.float32)
    wrench[tip] = np.r_[force, torque]
    state.body_f.assign(wrench)
    h = scene.dt
    solver.step(state, out, control, None, h)
    dv = (out.joint_qd.numpy().astype(np.float64) - state.joint_qd.numpy()) / h
    # The wrench at the tip body's walk frame (its joint's anchor, world axes): the torque taken about it.
    poses = newton_state_body_q(model, joint_q)
    centre = transform_point(poses[tip], model.body_com.numpy()[tip].astype(np.float64))
    anchor = transform_point(poses[int(model.joint_parent.numpy()[links])], model.joint_X_p.numpy()[links, :3])
    W = np.r_[force, torque + np.cross(centre - anchor, force)]
    truth = TreeWalk(*local_frames(model, joint_q, 0), np.zeros(n))
    walk32 = TreeWalk(*local_frames(model, joint_q, 0), np.zeros(n), dtype=np.float32)
    dv64, _ = truth.down(truth.up(links, W)[0])
    dv32, _ = walk32.down(walk32.up(links, W.astype(np.float32))[0])
    return {
        "links": links,
        "solver": relative(dv, dv64),
        "walk32": relative(dv32, dv64),
        "finite": bool(np.all(np.isfinite(dv))),
    }


# ---------------------------------------------------------------------------------------------------------------------
# The families: the float32 walk without armature, host and device
# ---------------------------------------------------------------------------------------------------------------------


def _scaled_chain(links: int, mass: np.ndarray, inertia: np.ndarray):
    """``long_chain(links)`` without armature, its links' masses and inertias scaled per link (the head keeps its)."""
    model = scenes.long_chain(links=links, armature=0.0).build("cpu")
    body_mass, body_inertia = model.body_mass.numpy(), model.body_inertia.numpy()
    children = model.joint_child.numpy()[1:]
    body_mass[children] *= mass
    body_inertia[children] *= inertia[:, None, None]
    model.body_mass.assign(body_mass)
    model.body_inertia.assign(body_inertia)
    return model


def mass_ratio_chain(links: int = 106):
    """Masses alternating by 1e3 along the chain: a heavy link, then a light one."""
    ratio = np.where(np.arange(links) % 2 == 0, 1.0, 1.0e-3)
    return _scaled_chain(links, ratio, ratio)


def thin_chain(links: int = 106):
    """Thin links: each keeps its mass at its centre and turns with almost no inertia of its own (1e-6 of a box's)."""
    return _scaled_chain(links, np.ones(links), np.full(links, 1.0e-6))


def branched_cable(finger_links: int = 40):
    """Sprawl with fingers of ``finger_links`` links: ten branches of a cable off two wrists."""
    return scenes.sprawl(finger_links=finger_links).build("cpu")


def loops():
    """Loops cut at a body (:mod:`.body_cut`): the four-bar closed by a ball joint, and the Stewart platform."""
    return [BodyCut.of(scene.build("cpu")).model for scene in (scenes.four_bar(loop="ball"), scenes.stewart_platform())]


def _leaves(parent: np.ndarray) -> list:
    """Every body no other body hangs from."""
    hung = {int(p) for p in parent if p >= 0}
    return [b for b in range(parent.size) if b not in hung]


def family_response(model, devices_to_run=(), *, float32_poses: bool = False) -> dict:
    """Every tree of ``model`` at its ``joint_q`` without armature: a unit contact's three rows at each leaf body's frame
    point (offset 5 cm), up and down through the float32 walk on the host and the device walk on each device, against
    the float64 walk; the float32 walk's frames from the float64 composition of ``joint_q`` or, with
    ``float32_poses``, from Newton's float32 forward kinematics. Worst relative error of ``dv = H^-1 J^T`` and whether
    every float32 pivot is positive."""
    joint_q = model.joint_q.numpy().astype(np.float64)
    starts = model.articulation_start.numpy()
    qd_start = model.joint_qd_start.numpy()
    poses = newton_state_body_q(model, joint_q) if float32_poses else None
    worst = {"host": 0.0, **{str(device): 0.0 for device in devices_to_run}, "pivots_positive": True, "trees": 0}
    for art in range(int(model.articulation_count)):
        n = int(qd_start[starts[art + 1]] - qd_start[starts[art]])
        if n == 0:
            continue
        truth = TreeWalk(*local_frames(model, joint_q, art), np.zeros(n))
        walk32 = TreeWalk(*local_frames(model, joint_q, art, poses), np.zeros(n), dtype=np.float32)
        worst["trees"] += 1
        worst["pivots_positive"] &= bool(np.all(np.isfinite(walk32.ell)) and np.all(walk32.ell > 0.0))
        W = contact_wrenches(np.array([0.05, 0.0, -0.02]))
        bodies = _leaves(truth.parent)
        dv64 = []
        for body in bodies:
            z64, _ = truth.up(body, W)
            dv64.append(np.stack([truth.down(z64[:, r])[0] for r in range(3)], axis=1))
            z32, _ = walk32.up(body, W.astype(np.float32))
            dv32 = np.stack([walk32.down(z32[:, r])[0] for r in range(3)], axis=1)
            worst["host"] = max(worst["host"], relative(dv32, dv64[-1]))
        for device in devices_to_run:
            walk = DeviceWalk([walk32], device)
            count = len(bodies)
            tops = [int(truth.first[b + 1]) - 1 for b in bodies]
            z = walk.up([0] * count, bodies, tops, [0] * count, [W.T] * count, [np.zeros(3)] * count)
            for k in range(count):
                dv = walk.down([0, 0, 0], z[k])[0].T
                worst[str(device)] = max(worst[str(device)], relative(dv, dv64[k]))
    return worst


def main_on_chain(model) -> dict:
    """A chain's tip contact (three rows) through main's float32 dense factor (:func:`dense_factor32` of the float32 ``H``
    about the origin, no armature added), applied in float32, and through the float32 walk without armature, both
    against the float64 walk without armature (F2's comparison, :func:`.test_feather_pgs_walk.chain_precision`)."""
    from scipy.linalg import solve_triangular

    joint_q = model.joint_q.numpy().astype(np.float64)
    n = int(model.joint_dof_count)
    truth = TreeWalk(*local_frames(model, joint_q, 0), np.zeros(n))
    walk32 = TreeWalk(*local_frames(model, joint_q, 0), np.zeros(n), dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver = SolverFeatherPGS(model, dense_max_constraints=8)
    body = truth.parent.size - 1
    W = contact_wrenches(np.array([0.05, 0.0, -0.02]))
    z64, _ = truth.up(body, W)
    dv64 = np.stack([truth.down(z64[:, r])[0] for r in range(3)], axis=1)
    z32, _ = walk32.up(body, W.astype(np.float32))
    dv32 = np.stack([walk32.down(z32[:, r])[0] for r in range(3)], axis=1)
    H32 = MassReference(model, np.float32).mass_matrix(joint_q, 0).astype(np.float64)
    L32 = dense_factor32(H32 + np.diag(solver.factor.diagonal(0)))
    with np.errstate(all="ignore"):
        x32 = jacobian(truth, body, W).astype(np.float32)[::-1]
        z_main = solve_triangular(L32, x32, lower=True, check_finite=False)
        dv_main = solve_triangular(L32, z_main, lower=True, trans=1, check_finite=False)[::-1]
    main = relative(dv_main, dv64) if np.all(np.isfinite(dv_main)) else np.inf
    return {
        "walk32": relative(dv32, dv64),
        "main": main,
        "pivots_positive": bool(np.all(np.isfinite(walk32.ell)) and np.all(walk32.ell > 0.0)),
    }


def families() -> list:
    """``(name, model)`` of every family the 1e-5 bar holds on."""
    four_bar, stewart = loops()
    return [
        ("mass ratio 1e3, 106 links", mass_ratio_chain()),
        ("branched cable, 10 x 40 links", branched_cable()),
        ("four-bar cut at a body", four_bar),
        ("Stewart platform cut at a body", stewart),
    ]


# ---------------------------------------------------------------------------------------------------------------------
# The assembly floor's cases on the walk
# ---------------------------------------------------------------------------------------------------------------------


def thin_rocker(inertia: float):
    """The assembly floor's thin ball loop (:meth:`TestStepsWithoutArmature.rocker`), cut at the rocker."""
    return BodyCut.of(TestStepsWithoutArmature.rocker(inertia)).model


def thin_ball_far_out():
    """The floor's pure tree: a thin body on a ball joint 2 m out along a fixed root's bar."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    base = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()))
    builder.add_shape_box(base, hx=0.5, hy=0.05, hz=0.05)
    root = builder.add_joint_fixed(parent=-1, child=base, parent_xform=wp.transform(wp.vec3(0.0, 0.0, 1.0)))
    thin = builder.add_link(xform=wp.transform(wp.vec3(2.0, 0.0, 0.9), wp.quat_identity()), mass=1.0)
    builder.body_inertia[thin] = wp.mat33(1.0e-9 * np.eye(3))
    ball = builder.add_joint_ball(
        parent=base,
        child=thin,
        parent_xform=wp.transform(wp.vec3(2.0, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()),
    )
    builder.add_articulation([root, ball])
    return builder.finalize(device="cpu")


def thin_hinges():
    """The floor's two thin hinges about ``x``, 2 and 2.5 m out from a fixed root."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    base = builder.add_link(xform=wp.transform_identity(), mass=1.0)
    builder.body_inertia[base] = wp.mat33(np.eye(3))
    joints = [builder.add_joint_fixed(parent=-1, child=base)]
    parent = base
    for offset, x, mass in ((2.0, 2.0, 1.0), (0.5, 2.5, 0.5)):
        body = builder.add_link(xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()), mass=mass)
        builder.body_inertia[body] = wp.mat33(1.0e-9 * np.eye(3))
        frame = wp.transform(wp.vec3(offset, 0.0, 0.0), wp.quat_identity())
        joints.append(
            builder.add_joint_revolute(
                parent=parent, child=body, parent_xform=frame, axis=(1.0, 0.0, 0.0), armature=0.0
            )
        )
        parent = body
    builder.add_articulation(joints)
    return builder.finalize(device="cpu")


FLOOR_INERTIAS = (1.0e-7, 3.0e-8, 1.0e-8, 3.0e-9, 1.0e-9, 3.0e-10, 1.0e-10)
"""The thin ball loop's rocker inertias, around the rounding of its shadow's ball joint's diagonal."""


def floor_cases() -> list:
    """``(name, model)`` of every assembly-floor case."""
    return [
        *((f"thin ball loop, rocker {inertia:.0e}", thin_rocker(inertia)) for inertia in FLOOR_INERTIAS),
        ("thin ball 2 m out, pure tree", thin_ball_far_out()),
        ("thin hinges 2 and 2.5 m out", thin_hinges()),
    ]


# ---------------------------------------------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------------------------------------------


class TestDeepChainRegime(unittest.TestCase):
    """A tip impulse on ``long_chain(N)`` without armature, the solver's step against float64."""

    def test_a_tip_impulse_answers_within_1e_4(self):
        """Main's float32 dense factor, with its conditioning armature, answered 2.6e-3 to 6.6e-3 off float64 from 106
        links; the step, solved by the walk (Z4.2) with no armature added (Z4.L), holds 1e-4."""
        worst = max(tip_impulse(links)["solver"] for links in DEEP)
        self.assertLess(worst, REGIME_TOLERANCE)

    def test_the_float32_walk_answers_it_without_armature(self):
        """The regime's bar holds for the float32 walk at every length, with no armature: what Z4.L stands on."""
        for links in DEEP:
            with self.subTest(links=links):
                result = tip_impulse(links)
                self.assertTrue(result["finite"])
                self.assertLess(result["walk32"], WALK_TOLERANCE, f"{result}")


class TestWalkFamilies(unittest.TestCase):
    """The float32 walk without armature on the families F2 never ran, host and devices."""

    def test_every_family_without_armature(self):
        for name, model in families():
            with self.subTest(family=name):
                worst = family_response(model, devices(), float32_poses=True)
                self.assertGreater(worst["trees"], 0)
                self.assertTrue(worst["pivots_positive"], f"{worst}")
                for where in ("host", *devices()):
                    self.assertLess(worst[str(where)], WALK_TOLERANCE, f"{where}: {worst}")


class TestThinLinks(unittest.TestCase):
    """A chain of thin links without armature: past float32 (``cond(H)`` 2.5e9 at 32 links, 9.2e11 at 106)."""

    def test_the_walk_is_closer_than_main(self):
        """The float32 walk without armature is finite, keeps its pivots positive and answers a tip contact within
        1e-3 of float64, closer than main's float32 dense factor (a failed factor, NaN, is not closer)."""
        for links in (32, 106):
            with self.subTest(links=links):
                result = main_on_chain(thin_chain(links))
                self.assertTrue(result["pivots_positive"], f"{result}")
                self.assertLess(result["walk32"], 1.0e-3, f"{result}")
                self.assertFalse(result["main"] <= result["walk32"], f"{result}")

    @known_failure(LIMIT, "cond(H) 9.2e11 on 106 thin links: the float32 walk reads 7.2e-4, main's factor fails")
    def test_within_1e_5(self):
        self.assertLess(main_on_chain(thin_chain(106))["walk32"], WALK_TOLERANCE)


class TestAssemblyFloorOnTheWalk(unittest.TestCase):
    """The assembly floor's former cases on the walk's local pivots, no floor, from float32 poses."""

    def test_every_floor_case_without_the_floor(self):
        for name, model in floor_cases():
            with self.subTest(case=name):
                worst = family_response(model, devices(), float32_poses=True)
                self.assertGreater(worst["trees"], 0)
                self.assertTrue(worst["pivots_positive"], f"{worst}")
                for where in ("host", *devices()):
                    self.assertLess(worst[str(where)], WALK_TOLERANCE, f"{where}: {worst}")


class TestStepsWithoutArmature(unittest.TestCase):
    """The solver steps the assembly floor's and the deep-chain rule's cases with no armature added."""

    @staticmethod
    def rocker(inertia: float):
        """The four-bar closed by a ball joint at a thin rocker of axial ``inertia``."""
        bars, coupler, rocker = _four_bar_bars()
        bars.body_inertia[rocker] = wp.mat33(inertia * np.eye(3))
        bars.add_joint_ball(parent=coupler, child=rocker, **_LOOP_FRAMES)
        return bars.finalize(device="cpu")

    def test_a_thin_ball_loop_steps(self):
        """The rocker's axial inertia of 1e-9 kg m^2 is below the rounding of its shadow's ball joint's diagonal
        about the origin, about 3e-9: main's factor went NaN at the first step without the floor. The loop closes."""
        model = self.rocker(1.0e-9)
        solver = SolverFeatherPGS(model, pgs_iterations=8, pgs_beta=0.1)
        np.testing.assert_array_equal(solver.factor.armature, model.joint_armature.numpy())
        state_0, state_1, control = model.state(), model.state(), model.control()
        targets = model.joint_target_q.numpy().copy()
        targets[0] = 0.3
        control.joint_target_q.assign(targets)
        for _ in range(120):
            state_0.clear_forces()
            solver.step(state_0, state_1, control, None, 1.0 / 240.0)
            state_0, state_1 = state_1, state_0
        self.assertTrue(np.isfinite(state_0.body_q.numpy()).all())
        self.assertLess(_loop_anchor_gap(model, state_0), 1.0e-5)
        self.assertAlmostEqual(float(state_0.joint_q.numpy()[0]), 0.2879, delta=2.0e-3)

    def test_every_inertia_around_the_rounding_steps(self):
        """Rounding lands on either side of zero around the floor, so no inertia near it may fail."""
        for inertia in FLOOR_INERTIAS:
            with self.subTest(inertia=inertia):
                model = self.rocker(inertia)
                solver = SolverFeatherPGS(model, pgs_iterations=4, pgs_beta=0.1)
                state_0, state_1, control = model.state(), model.state(), model.control()
                for _ in range(8):
                    state_0.clear_forces()
                    solver.step(state_0, state_1, control, None, 1.0 / 240.0)
                    state_0, state_1 = state_1, state_0
                self.assertTrue(np.isfinite(state_0.body_q.numpy()).all())

    def test_a_bare_chain_stays_finite(self):
        """Step a bare 100-link chain on the ground without a non-finite factor or velocity."""
        scene = scenes.long_chain(armature=0.0)
        model = scene.build("cpu")
        solver = SolverFeatherPGS(model, dense_max_constraints=384)
        np.testing.assert_array_equal(solver.factor.armature, model.joint_armature.numpy())
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=scene.rigid_contact_max)
        contacts = pipeline.contacts()
        state_0, state_1 = model.state(), model.state()
        control = model.control()
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
        for _ in range(scene.steps):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, scene.dt)
            state_0, state_1 = state_1, state_0
        self.assertTrue(np.isfinite(state_0.joint_qd.numpy()).all())
        self.assertTrue(np.isfinite(solver.factor.walk_data.numpy()).all())


def report() -> None:
    """Print the measured tables."""
    print("deep-chain regime: tip impulse's dv after one step against float64 (no armature)")
    for links in DEEP:
        r = tip_impulse(links)
        print(
            f"  N {links:5d}: solver {r['solver']:8.1e}  float32 walk {r['walk32']:8.1e}",
            flush=True,
        )
    where = devices()
    for links in (32, 106):
        r = main_on_chain(thin_chain(links))
        print(
            f"thin links {links}: float32 walk {r['walk32']:8.1e}  main with its armature {r['main']:8.1e}", flush=True
        )
    print(f"families and floor cases: float32 walk without armature, worst dv relative to float64 ({', '.join(where)})")
    for name, model in (*families(), *floor_cases()):
        for poses in (False, True):
            w = family_response(model, where, float32_poses=poses)
            cells = "  ".join(f"{d} {w[str(d)]:8.1e}" for d in ("host", *where))
            print(
                f"  {name:36s} {'float32 poses' if poses else 'float64 poses'}: trees {w['trees']:2d} pivots positive "
                f"{w['pivots_positive']}  {cells}",
                flush=True,
            )


if __name__ == "__main__":
    if sys.argv[1:2] == ["report"]:
        wp.config.quiet = True
        report()
    else:
        unittest.main()
