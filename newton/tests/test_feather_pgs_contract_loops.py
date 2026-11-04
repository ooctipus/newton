# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Loops: a loop joint closes its mechanism, on the public API of SolverFeatherPGS.

A loop joint (a joint outside every articulation, or between two of them) holds its anchors together under drive,
carries a load on a moving kinematic carrier, publishes its coordinate, takes a drive or a limit like a tree joint, and
keeps a floating mechanism's momentum. A contract module: :mod:`newton.tests.test_feather_pgs_contract_lint` keeps it on
the public API.
"""

import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton.tests import feather_pgs_scenes as scenes


def _four_bar_bars():
    """The bars of a planar four-bar, not yet closed: two grounded revolute chains, the crank position-driven.

    Ground pivots at x=0 and x=0.4; crank and rocker hang down 0.2 m; the coupler reaches from the crank tip to the
    rocker tip.
    """
    b = newton.ModelBuilder(up_axis=newton.Axis.Z)
    z0 = 0.6

    crank = b.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, z0 - 0.1), wp.quat_identity()))
    b.add_shape_box(crank, hx=0.02, hy=0.02, hz=0.1)
    j_crank = b.add_joint_revolute(
        parent=-1,
        child=crank,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, z0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()),
    )
    coupler = b.add_link(xform=wp.transform(wp.vec3(0.2, 0.0, z0 - 0.2), wp.quat_identity()))
    b.add_shape_box(coupler, hx=0.2, hy=0.02, hz=0.02)
    j_coupler = b.add_joint_revolute(
        parent=crank,
        child=coupler,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, -0.1), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0), wp.quat_identity()),
    )
    rocker = b.add_link(xform=wp.transform(wp.vec3(0.4, 0.0, z0 - 0.1), wp.quat_identity()))
    b.add_shape_box(rocker, hx=0.02, hy=0.02, hz=0.1)
    j_rocker = b.add_joint_revolute(
        parent=-1,
        child=rocker,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(wp.vec3(0.4, 0.0, z0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()),
    )
    b.add_articulation([j_crank, j_coupler, j_rocker], label="four_bar")

    # Drive the crank only.
    b.joint_target_ke[0] = 50.0
    b.joint_target_kd[0] = 5.0
    b.joint_target_mode[0] = int(newton.JointTargetMode.POSITION)
    return b, coupler, rocker


_LOOP_FRAMES = {
    "parent_xform": wp.transform(wp.vec3(0.2, 0.0, 0.0), wp.quat_identity()),
    "child_xform": wp.transform(wp.vec3(0.0, 0.0, -0.1), wp.quat_identity()),
}
"""The coupler tip and the rocker tip, where a loop joint closes the bars."""


def _build_four_bar():
    """The planar four-bar closed by a BALL loop joint, trailing as the MJCF importer closes ``connect`` equalities. The
    rocker is undriven, so any coherent motion it does comes from the closed loop."""
    b, coupler, rocker = _four_bar_bars()
    b.add_joint_ball(parent=coupler, child=rocker, **_LOOP_FRAMES)
    return b


def _loop_anchor_gap(model, state) -> float:
    """World-space distance between the loop joint's parent and child anchors [m]."""
    bq = state.body_q.numpy().reshape(-1, 7).astype(np.float64)
    jt = model.joint_type.numpy()
    jp = model.joint_parent.numpy()
    jc = model.joint_child.numpy()
    Xp = model.joint_X_p.numpy()
    Xc = model.joint_X_c.numpy()
    j = int(np.nonzero(jt == int(newton.JointType.BALL))[0][-1])

    def anchor(body, X):
        t = wp.transform(wp.vec3(*bq[body, :3]), wp.quat(*bq[body, 3:]))
        a = wp.transform(wp.vec3(*X[:3]), wp.quat(*X[3:]))
        w = wp.transform_multiply(t, a)
        return np.array([w.p[0], w.p[1], w.p[2]])

    return float(np.linalg.norm(anchor(int(jp[j]), Xp[j]) - anchor(int(jc[j]), Xc[j])))


def _run(steps: int = 720, crank_target: float = 0.6, **solver_kwargs):
    builder = _build_four_bar()
    model = builder.finalize()
    solver = newton.solvers.SolverFeatherPGS(model, pgs_iterations=16, pgs_beta=0.1, **solver_kwargs)
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    targets = model.joint_target_q.numpy().copy()
    targets[0] = crank_target
    control.joint_target_q.assign(targets)
    dt = 1.0 / 240.0
    for _ in range(steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0
    return solver, model, state_0


@unittest.skipUnless(wp.get_device().is_cuda, "SolverFeatherPGS matrix-free mode requires CUDA")
class TestFeatherPGSConnect(unittest.TestCase):
    def test_loop_joint_presence_is_stable(self):
        """Verify a trailing BALL loop joint no longer NaNs the articulation solve."""
        _, _, state = _run(steps=120, crank_target=0.0)
        self.assertTrue(np.isfinite(state.body_q.numpy()).all())
        self.assertTrue(np.isfinite(state.joint_qd.numpy()).all())

    def test_loop_closure_is_enforced(self):
        """Verify the connect rows hold the four-bar's loop anchors together under drive."""
        _, model, state = _run(steps=720, crank_target=0.6)
        self.assertTrue(np.isfinite(state.body_q.numpy()).all())
        gap = _loop_anchor_gap(model, state)
        self.assertLess(gap, 2.0e-3, f"loop anchor gap {gap * 1e3:.2f} mm — four-bar not closed")

    def test_rocker_follows_crank(self):
        """Verify the undriven rocker moves coherently with the driven crank via the loop."""
        _, _model, state = _run(steps=720, crank_target=0.6)
        q = state.joint_q.numpy()
        # Parallel four-bar (equal crank/rocker lengths): rocker angle tracks crank angle.
        self.assertAlmostEqual(q[2], q[0], delta=0.08)
        self.assertGreater(abs(q[2]), 0.3, "rocker did not move — loop not transmitting")

    def test_standalone_world_root_is_not_loop_joint(self):
        """Do not treat an unrelated unowned world-root joint as a closure."""
        b = newton.ModelBuilder(up_axis=newton.Axis.Z)

        articulated = b.add_link()
        b.add_shape_box(articulated, hx=0.05, hy=0.05, hz=0.1)
        tree_joint = b.add_joint_revolute(parent=-1, child=articulated, axis=newton.Axis.Y)
        b.add_articulation([tree_joint], label="articulated")

        standalone = b.add_link()
        b.add_shape_box(standalone, hx=0.05, hy=0.05, hz=0.1)
        standalone_joint = b.add_joint_fixed(parent=-1, child=standalone)

        model = b.finalize()
        self.assertEqual(model.joint_articulation.numpy().tolist(), [0, -1])
        self.assertEqual(standalone_joint, 1)

        solver = newton.solvers.SolverFeatherPGS(model)
        self.assertIs(solver.model, model)

    def test_connect_survives_foreign_joint_between_tree_and_loop(self):
        """Attribute loop joints by body ownership when another articulation interleaves.

        ``add_usd`` of a subtree containing a closed-loop robot AND a floating rigid
        body creates the body's FREE joint during traversal but appends the equality
        loop joints at the very end of the parse — so the closures no longer trail
        their articulation's joint-index range (the trailing sentinel range of the LAST
        articulation swallows them instead). Index-range attribution then silently
        produced ZERO connect rows and the closed linkage fell open (measured on the
        Robotiq 2F-85 through the IsaacLab scene path: 107 mm anchor gap, fingers
        collapsed). The closure must be attributed via its child body's articulation.
        This test reconstructs that joint ordering directly: four-bar tree, then a
        free-jointed object articulation, THEN the closure ball joint.
        """
        b = newton.ModelBuilder(up_axis=newton.Axis.Z)
        z0 = 0.6
        crank = b.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, z0 - 0.1), wp.quat_identity()))
        b.add_shape_box(crank, hx=0.02, hy=0.02, hz=0.1)
        j_crank = b.add_joint_revolute(
            parent=-1,
            child=crank,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, z0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()),
        )
        coupler = b.add_link(xform=wp.transform(wp.vec3(0.2, 0.0, z0 - 0.2), wp.quat_identity()))
        b.add_shape_box(coupler, hx=0.2, hy=0.02, hz=0.02)
        j_coupler = b.add_joint_revolute(
            parent=crank,
            child=coupler,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, -0.1), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0), wp.quat_identity()),
        )
        rocker = b.add_link(xform=wp.transform(wp.vec3(0.4, 0.0, z0 - 0.1), wp.quat_identity()))
        b.add_shape_box(rocker, hx=0.02, hy=0.02, hz=0.1)
        j_rocker = b.add_joint_revolute(
            parent=-1,
            child=rocker,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(wp.vec3(0.4, 0.0, z0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()),
        )
        b.add_articulation([j_crank, j_coupler, j_rocker], label="four_bar")

        # a second articulation's joint lands BETWEEN the tree and the loop joint
        body = b.add_link(xform=wp.transform(wp.vec3(1.0, 0.0, 0.5), wp.quat_identity()))
        b.add_shape_sphere(body, radius=0.05)
        j_free = b.add_joint_free(child=body)
        b.add_articulation([j_free], label="free_object")

        # the loop closure, appended last (the add_usd equality-parse ordering)
        b.add_joint_ball(
            parent=coupler,
            child=rocker,
            parent_xform=wp.transform(wp.vec3(0.2, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, -0.1), wp.quat_identity()),
        )
        b.joint_target_ke[0] = 50.0
        b.joint_target_kd[0] = 5.0
        b.joint_target_mode[0] = int(newton.JointTargetMode.POSITION)
        model = b.finalize()

        solver = newton.solvers.SolverFeatherPGS(model, pgs_iterations=16, pgs_beta=0.1)

        state_0, state_1 = model.state(), model.state()
        control = model.control()
        targets = model.joint_target_q.numpy().copy()
        targets[0] = 0.6
        control.joint_target_q.assign(targets)
        gap_max = 0.0
        for _ in range(360):
            state_0.clear_forces()
            solver.step(state_0, state_1, control, None, 1.0 / 240.0)
            state_0, state_1 = state_1, state_0
            gap_max = max(gap_max, _loop_anchor_gap(model, state_0))
        self.assertTrue(np.isfinite(state_0.body_q.numpy()).all())
        self.assertLess(gap_max, 2.0e-3, f"loop closure not enforced (anchor gap {gap_max:.4f} m)")


# ---------------------------------------------------------------------------
# Closures with a prescribed (kinematic or world) parent
# ---------------------------------------------------------------------------

_CHILD_ANCHORS = ((0.0, 0.0, 0.0), (0.05, 0.0, 0.0), (0.0, 0.05, 0.0))
_REL_P = np.array([0.0, 0.0, -0.2])


def _build_carried_load(*, enabled: bool = True, world_parent: bool = False):
    """A kinematic carrier holding a dynamic box through three BALL loop joints.

    Three non-collinear point closures pin all six relative degrees of freedom, so the
    load must follow the carrier as if welded at ``_REL_P`` below it. With
    ``world_parent`` the carrier is the world and the anchors are world points.
    """
    b = newton.ModelBuilder(up_axis=newton.Axis.Z)
    carrier_p = np.array([0.0, 0.0, 1.0])
    carrier = -1
    if not world_parent:
        carrier = b.add_body(xform=wp.transform(wp.vec3(*carrier_p), wp.quat_identity()), is_kinematic=True)
        b.add_shape_box(carrier, hx=0.03, hy=0.03, hz=0.03)
    load = b.add_body(xform=wp.transform(wp.vec3(*(carrier_p + _REL_P)), wp.quat_identity()))
    b.add_shape_box(load, hx=0.04, hy=0.04, hz=0.04, cfg=newton.ModelBuilder.ShapeConfig(density=500.0))
    joints = []
    for c in _CHILD_ANCHORS:
        p = (_REL_P if not world_parent else carrier_p + _REL_P) + np.array(c)
        joints.append(
            b.add_joint_ball(
                parent=carrier,
                child=load,
                parent_xform=wp.transform(wp.vec3(*p), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(*c), wp.quat_identity()),
                enabled=enabled,
            )
        )
    return b, carrier, load, joints


def _anchor_gaps(model, state, joints, solver):
    """World-space parent/child anchor distances [m] using the solver's live anchors."""
    bq = state.body_q.numpy().astype(np.float64)
    jp = model.joint_parent.numpy()
    jc = model.joint_child.numpy()

    def anchor(body, local):
        if body < 0:
            return np.asarray(local, dtype=np.float64)
        t = wp.transform(wp.vec3(*bq[body, :3]), wp.quat(*bq[body, 3:]))
        w = wp.transform_point(t, wp.vec3(*local))
        return np.array([w[0], w[1], w[2]])

    gaps = []
    for j in joints:
        anchor_p, anchor_c = solver.loop_joint_anchors(j)
        gaps.append(float(np.linalg.norm(anchor(int(jp[j]), anchor_p) - anchor(int(jc[j]), anchor_c))))
    return gaps


def _articulation_of(model, body):
    """The articulation of ``body``'s joint, as the index array :func:`newton.eval_fk` takes."""
    joint = int(np.flatnonzero(model.joint_child.numpy() == body)[0])
    return wp.array([int(model.joint_articulation.numpy()[joint])], dtype=wp.int32, device=model.device)


def _prescribe_carrier(model, solver, state, articulation, t, v, omega):
    """Write the carrier's free-joint pose/velocity for time ``t`` and refresh its FK."""
    q = state.joint_q.numpy()
    qd = state.joint_qd.numpy()
    angle = float(np.linalg.norm(omega)) * t
    axis = np.asarray(omega, dtype=np.float64) / max(float(np.linalg.norm(omega)), 1.0e-12)
    rot = wp.quat_from_axis_angle(wp.vec3(*axis), angle)
    q[0:3] = np.array([0.0, 0.0, 1.0]) + np.asarray(v) * t
    q[3:7] = [rot[0], rot[1], rot[2], rot[3]]
    qd[0:3] = v
    qd[3:6] = omega
    state.joint_q.assign(q)
    state.joint_qd.assign(qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state, indices=articulation)
    solver.notify_state_changed(articulation)


class TestFeatherPGSPrescribedParentConnect(unittest.TestCase):
    def test_kinematic_parent_closure_carries_load(self):
        """A load pinned to a moving kinematic carrier must ride along with it.

        Without prescribed-parent support the closure is ignored (it crosses
        articulations) and the load free-falls; with it, the three point closures hold
        the load at its relative pose while the carrier translates and spins.
        """
        b, carrier, load, joints = _build_carried_load()
        model = b.finalize()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            solver = newton.solvers.SolverFeatherPGS(model, pgs_iterations=16, pgs_beta=0.2)
        self.assertFalse([w for w in caught if "loop joint" in str(w.message)], [str(w.message) for w in caught])
        articulation = _articulation_of(model, carrier)
        state_0, state_1 = model.state(), model.state()
        control = model.control()
        dt = 1.0 / 240.0
        v = np.array([0.3, 0.0, 0.1])
        omega = np.array([0.0, 0.0, 1.5])
        for i in range(240):
            _prescribe_carrier(model, solver, state_0, articulation, i * dt, v, omega)
            state_0.clear_forces()
            solver.step(state_0, state_1, control, None, dt)
            state_0, state_1 = state_1, state_0
        _prescribe_carrier(model, solver, state_0, articulation, 240 * dt, v, omega)

        bq = state_0.body_q.numpy().astype(np.float64)
        self.assertTrue(np.isfinite(bq).all())
        carrier_t = wp.transform(wp.vec3(*bq[carrier, :3]), wp.quat(*bq[carrier, 3:]))
        expected = wp.transform_point(carrier_t, wp.vec3(*_REL_P))
        error = np.linalg.norm(bq[load, :3] - np.array([expected[0], expected[1], expected[2]]))
        self.assertLess(error, 3.0e-3, f"load lagged the carrier by {error:.4f} m")
        for gap in _anchor_gaps(model, state_0, joints, solver):
            self.assertLess(gap, 3.0e-3)
        # The carrier actually moved, so the test exercised a non-trivial target velocity.
        self.assertGreater(np.linalg.norm(bq[carrier, :3] - np.array([0.0, 0.0, 1.0])), 0.25)

    def test_world_parent_closure_holds_hanging_load(self):
        b, _, load, joints = _build_carried_load(world_parent=True)
        model = b.finalize()
        solver = newton.solvers.SolverFeatherPGS(model, pgs_iterations=16, pgs_beta=0.2)
        state_0, state_1 = model.state(), model.state()
        control = model.control()
        for _ in range(240):
            state_0.clear_forces()
            solver.step(state_0, state_1, control, None, 1.0 / 240.0)
            state_0, state_1 = state_1, state_0
        bq = state_0.body_q.numpy().astype(np.float64)
        self.assertTrue(np.isfinite(bq).all())
        self.assertLess(np.linalg.norm(bq[load, :3] - (np.array([0.0, 0.0, 1.0]) + _REL_P)), 3.0e-3)
        for gap in _anchor_gaps(model, state_0, joints, solver):
            self.assertLess(gap, 3.0e-3)

    def test_runtime_enable_and_anchor_update(self):
        """Closures can be released, re-targeted at the measured pose and re-engaged."""
        b, carrier, load, joints = _build_carried_load()
        model = b.finalize()
        solver = newton.solvers.SolverFeatherPGS(model, pgs_iterations=16, pgs_beta=0.2)
        state_0, state_1 = model.state(), model.state()
        control = model.control()
        dt = 1.0 / 240.0

        def run(steps):
            nonlocal state_0, state_1
            for _ in range(steps):
                state_0.clear_forces()
                solver.step(state_0, state_1, control, None, dt)
                state_0, state_1 = state_1, state_0

        for j in joints:
            solver.set_loop_joint_enabled(j, False)
        z0 = float(state_0.body_q.numpy()[load, 2])
        run(48)
        bq = state_0.body_q.numpy().astype(np.float64)
        dropped = z0 - float(bq[load, 2])
        self.assertGreater(dropped, 0.15, "released load should free-fall")

        # Re-target the closures at the measured relative pose, then re-engage them.
        carrier_t = wp.transform(wp.vec3(*bq[carrier, :3]), wp.quat(*bq[carrier, 3:]))
        load_t = wp.transform(wp.vec3(*bq[load, :3]), wp.quat(*bq[load, 3:]))
        rel = wp.transform_multiply(wp.transform_inverse(carrier_t), load_t)
        for j, c in zip(joints, _CHILD_ANCHORS, strict=True):
            p = wp.transform_point(rel, wp.vec3(*c))
            solver.set_loop_joint_anchors(j, (p[0], p[1], p[2]), c)
            solver.set_loop_joint_enabled(j, True)
        held_z = float(bq[load, 2])
        run(120)
        bq = state_0.body_q.numpy().astype(np.float64)
        self.assertTrue(np.isfinite(bq).all())
        self.assertLess(abs(float(bq[load, 2]) - held_z), 3.0e-3, "re-engaged closure should hold the load")
        for gap in _anchor_gaps(model, state_0, joints, solver):
            self.assertLess(gap, 3.0e-3)

        for j in joints:
            solver.set_loop_joint_enabled(j, False)
        run(48)
        self.assertGreater(held_z - float(state_0.body_q.numpy()[load, 2]), 0.1, "released again: must fall")

    def test_unknown_joint_is_rejected(self):
        b, _, _, joints = _build_carried_load()
        model = b.finalize()
        solver = newton.solvers.SolverFeatherPGS(model)
        with self.assertRaises(ValueError):
            solver.set_loop_joint_enabled(joints[0] + 100, False)


# ---------------------------------------------------------------------------
# The loop joint is a real tree joint, of any type
# ---------------------------------------------------------------------------


def _frames(model, state, joint):
    """The world frames of a loop joint's parent and child anchors."""
    body_q = state.body_q.numpy().astype(np.float64)

    def frame(body, local):
        pose = wp.transform(*local)
        return pose if body < 0 else wp.transform_multiply(wp.transform(*body_q[body]), pose)

    return (
        frame(int(model.joint_parent.numpy()[joint]), model.joint_X_p.numpy()[joint]),
        frame(int(model.joint_child.numpy()[joint]), model.joint_X_c.numpy()[joint]),
    )


def _gap(model, state, joint):
    """How far a loop joint's anchors stand apart [m]."""
    parent, child = _frames(model, state, joint)
    return float(wp.length(wp.transform_get_translation(parent) - wp.transform_get_translation(child)))


def _simulate(model, steps=240, *, targets=None, **options):
    """Step ``model`` from its joint state with the crank's ``targets``; return the solver and the final state."""
    solver = newton.solvers.SolverFeatherPGS(model, pgs_iterations=16, pgs_beta=0.1, **options)
    state_0, state_1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    if targets is not None:
        values = model.joint_target_q.numpy().copy()
        for index, value in targets.items():
            values[index] = value
        control.joint_target_q.assign(values)
    for _ in range(steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, 1.0 / 240.0)
        state_0, state_1 = state_1, state_0
    return solver, state_0


def _closed_bars(add_loop):
    """The four-bar's bars closed by ``add_loop(builder, coupler, rocker)``, and that joint."""
    b, coupler, rocker = _four_bar_bars()
    return b, add_loop(b, coupler, rocker)


class TestLoopJointTypes(unittest.TestCase):
    def test_a_revolute_loop_joint_closes(self):
        """The catalogue's four-bar on a revolute loop joint: the loop holds, and the rocker follows."""
        scene = scenes.four_bar(loop="revolute")
        model = scene.build(wp.get_device())
        joint = int(np.flatnonzero(model.joint_articulation.numpy() < 0)[0])
        _, state = _simulate(model, targets={0: 0.6})
        self.assertLess(_gap(model, state, joint), 1.0e-5)
        q = state.joint_q.numpy()
        q_start = model.joint_q_start.numpy()
        self.assertAlmostEqual(float(q[q_start[3]]), float(q[q_start[1]]), delta=0.02)
        self.assertGreater(abs(float(q[q_start[3]])), 0.3)

    def test_a_loop_joints_coordinate_is_published(self):
        """A revolute loop joint's coordinate is the angle between its bars, on the caller's state."""
        scene = scenes.four_bar(loop="revolute")
        model = scene.build(wp.get_device())
        joint = int(np.flatnonzero(model.joint_articulation.numpy() < 0)[0])
        _, state = _simulate(model, targets={0: 0.6})
        q = state.joint_q.numpy()
        q_start = model.joint_q_start.numpy()
        # A parallelogram: the coupler stays level, so the loop joint turns as the rocker does.
        self.assertAlmostEqual(float(q[q_start[joint]]), float(q[q_start[3]]), delta=1.0e-3)

    def test_a_fixed_loop_joint_locks_the_mechanism(self):
        """A weld takes the four-bar's one freedom: the driven crank cannot move."""
        bars, joint = _closed_bars(lambda b, coupler, rocker: b.add_joint_fixed(coupler, rocker, **_LOOP_FRAMES))
        model = bars.finalize()
        _, state = _simulate(model, targets={0: 0.6})
        self.assertLess(abs(float(state.joint_q.numpy()[0])), 1.0e-3)
        self.assertLess(_gap(model, state, joint), 1.0e-5)

    def test_a_prismatic_loop_joint_slides_along_its_axis_alone(self):
        """A slider along the coupler frees the anchors along its axis and nowhere else."""
        bars, joint = _closed_bars(
            lambda b, coupler, rocker: b.add_joint_prismatic(
                coupler, rocker, axis=wp.vec3(1.0, 0.0, 0.0), **_LOOP_FRAMES
            )
        )
        model = bars.finalize()
        _, state = _simulate(model, targets={0: 0.4})
        parent, child = _frames(model, state, joint)
        offset = wp.quat_rotate_inv(
            wp.transform_get_rotation(parent),
            wp.transform_get_translation(child) - wp.transform_get_translation(parent),
        )
        self.assertLess(abs(float(offset[1])) + abs(float(offset[2])), 1.0e-5)
        self.assertTrue(np.isfinite(state.body_q.numpy()).all())

    def test_a_dynamic_parent_in_another_articulation_closes(self):
        """A loop joint between two articulations, ignored before the body cut, closes them."""
        b = newton.ModelBuilder(up_axis=newton.Axis.Z)
        bodies = []
        for x in (0.0, 0.4):
            body = b.add_link(xform=wp.transform(wp.vec3(x, 0.0, 0.5), wp.quat_identity()))
            b.add_shape_box(body, hx=0.02, hy=0.02, hz=0.1)
            b.add_articulation(
                [
                    b.add_joint_revolute(
                        parent=-1,
                        child=body,
                        axis=wp.vec3(0.0, 1.0, 0.0),
                        parent_xform=wp.transform(wp.vec3(x, 0.0, 0.6), wp.quat_identity()),
                        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()),
                    )
                ]
            )
            bodies.append(body)
        # A rod's worth of offset: the two pendulums swing as one.
        joint = b.add_joint_fixed(
            bodies[0],
            bodies[1],
            parent_xform=wp.transform(wp.vec3(0.4, 0.0, 0.0), wp.quat_identity()),
        )
        model = b.finalize()
        _, state = _simulate(model)
        self.assertLess(_gap(model, state, joint), 1.0e-5)

    def test_the_loop_joints_drive_moves_the_mechanism(self):
        """The crank is free; only the revolute loop joint is driven, and the mechanism takes its target."""

        def add(b, coupler, rocker):
            return b.add_joint_revolute(
                coupler, rocker, axis=wp.vec3(0.0, 1.0, 0.0), target_ke=200.0, target_kd=20.0, **_LOOP_FRAMES
            )

        bars, joint = _closed_bars(add)
        bars.joint_target_ke[0] = 0.0
        bars.joint_target_kd[0] = 0.0
        bars.joint_target_mode[3] = int(newton.JointTargetMode.POSITION)
        model = bars.finalize()
        _, state = _simulate(model, steps=480, targets={int(model.joint_qd_start.numpy()[joint]): 0.3})
        q = state.joint_q.numpy()
        self.assertAlmostEqual(float(q[model.joint_q_start.numpy()[joint]]), 0.3, delta=0.05)
        self.assertLess(_gap(model, state, joint), 1.0e-4)

    def test_a_limit_on_a_loop_joint_holds(self):
        """A limit on the loop joint stops the driven mechanism at the joint's bound."""

        def add(b, coupler, rocker):
            return b.add_joint_revolute(
                coupler, rocker, axis=wp.vec3(0.0, 1.0, 0.0), limit_lower=-0.2, limit_upper=0.2, **_LOOP_FRAMES
            )

        bars, joint = _closed_bars(add)
        model = bars.finalize()
        _, state = _simulate(model, steps=480, targets={0: 0.6}, enable_joint_limits=True)
        q = state.joint_q.numpy()
        self.assertLess(abs(float(q[model.joint_q_start.numpy()[joint]])), 0.21)
        self.assertLess(_gap(model, state, joint), 1.0e-4)

    def test_a_coordinate_stays_continuous_past_a_turn(self):
        """A loop joint coaxial with its bars closes nothing, so it spins freely: its coordinate integrates past pi."""
        b = newton.ModelBuilder(up_axis=newton.Axis.Z)
        joints, bodies = [], []
        for _ in range(2):
            body = b.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()))
            b.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
            joints.append(
                b.add_joint_revolute(
                    parent=-1,
                    child=body,
                    axis=wp.vec3(0.0, 1.0, 0.0),
                    parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
                )
            )
            bodies.append(body)
        b.add_articulation(joints)
        loop = b.add_joint_revolute(bodies[0], bodies[1], axis=wp.vec3(0.0, 1.0, 0.0))
        b.joint_qd[0] = 8.0
        b.gravity = 0.0
        model = b.finalize()
        model.set_gravity((0.0, 0.0, 0.0))
        _, state = _simulate(model, steps=240)
        q = state.joint_q.numpy()
        self.assertGreater(abs(float(q[model.joint_q_start.numpy()[loop]])), 3.5)


class TestLoopsInScenes(unittest.TestCase):
    def test_every_world_closes_its_own_loops(self):
        """Replicated worlds, joints permuted per world: each four-bar closes, and each takes its own target."""
        template = _build_four_bar()
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        worlds = 4
        builder.replicate(template, worlds)
        model = builder.finalize()
        solver = newton.solvers.SolverFeatherPGS(model, pgs_iterations=16, pgs_beta=0.1)
        state_0, state_1, control = model.state(), model.state(), model.control()
        targets = model.joint_target_q.numpy().copy()
        cranks = np.arange(worlds) * (model.joint_count // worlds)
        goals = 0.1 + 0.1 * np.arange(worlds)
        targets[model.joint_target_q_start.numpy()[cranks]] = goals
        control.joint_target_q.assign(targets)
        for _ in range(480):
            state_0.clear_forces()
            solver.step(state_0, state_1, control, None, 1.0 / 240.0)
            state_0, state_1 = state_1, state_0
        q = state_0.joint_q.numpy()
        coords = model.joint_coord_count // worlds
        for world, joint in enumerate(np.flatnonzero(model.joint_articulation.numpy() < 0)):
            self.assertLess(_gap(model, state_0, int(joint)), 1.0e-5)
            # The crank and the rocker of a parallelogram turn together, and the worlds differ.
            self.assertAlmostEqual(float(q[world * coords]), float(q[world * coords + 2]), delta=1.0e-3)
        self.assertGreater(float(q[(worlds - 1) * coords]), float(q[0]) + 0.1)

    def test_a_free_rooted_loop_conserves_momentum(self):
        """A floating closed mechanism, each tree on its own free root's origin, keeps its loop and its momentum."""
        b = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)
        base = b.add_link(xform=wp.transform(wp.vec3(0.2, 0.0, 1.0), wp.quat_identity()))
        b.add_shape_box(base, hx=0.25, hy=0.05, hz=0.05)
        root = b.add_joint_free(child=base)
        joints, tips = [root], []
        for x in (-0.2, 0.2):
            link = b.add_link(xform=wp.transform(wp.vec3(0.2 + x, 0.0, 0.8), wp.quat_identity()))
            b.add_shape_box(link, hx=0.02, hy=0.02, hz=0.15)
            joints.append(
                b.add_joint_revolute(
                    parent=base,
                    child=link,
                    axis=wp.vec3(0.0, 1.0, 0.0),
                    parent_xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()),
                    child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.2), wp.quat_identity()),
                )
            )
            tips.append(link)
        bar = b.add_link(xform=wp.transform(wp.vec3(0.2, 0.0, 0.6), wp.quat_identity()))
        b.add_shape_box(bar, hx=0.2, hy=0.02, hz=0.02)
        joints.append(
            b.add_joint_revolute(
                parent=tips[0],
                child=bar,
                axis=wp.vec3(0.0, 1.0, 0.0),
                parent_xform=wp.transform(wp.vec3(0.0, 0.0, -0.2), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0), wp.quat_identity()),
            )
        )
        b.add_articulation(joints)
        loop = b.add_joint_revolute(
            parent=tips[1],
            child=bar,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, -0.2), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.2, 0.0, 0.0), wp.quat_identity()),
        )
        # A parallelogram's motion: both links turn alike and the bar counters them, so the state keeps the closure.
        b.joint_qd[0:3] = [0.3, -0.1, 0.2]
        b.joint_qd[6:9] = [2.0, 2.0, -2.0]
        model = b.finalize()
        solver = newton.solvers.SolverFeatherPGS(model, pgs_iterations=16, pgs_beta=0.1)
        state_0, state_1, control = model.state(), model.state(), model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

        def momentum(state):
            return (state.body_qd.numpy()[:, :3].astype(np.float64) * model.body_mass.numpy()[:, None]).sum(axis=0)

        first = momentum(state_0)
        for _ in range(240):
            state_0.clear_forces()
            solver.step(state_0, state_1, control, None, 1.0 / 240.0)
            state_0, state_1 = state_1, state_0
        self.assertLess(_gap(model, state_0, loop), 1.0e-4)
        # The integrator's own drift, the same with the joint cut (3e-3, 5e-3 kg m/s).
        np.testing.assert_allclose(momentum(state_0), first, atol=1.0e-2)

    def _floating_mechanism(self, *velocities):
        """A free-rooted parallelogram closed by a revolute loop joint, with the coordinates' ``velocities``."""
        b = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)
        base = b.add_link(xform=wp.transform(wp.vec3(0.2, 0.0, 1.0), wp.quat_identity()))
        b.add_shape_box(base, hx=0.25, hy=0.05, hz=0.05)
        joints, tips = [b.add_joint_free(child=base)], []
        for x in (-0.2, 0.2):
            link = b.add_link(xform=wp.transform(wp.vec3(0.2 + x, 0.0, 0.8), wp.quat_identity()))
            b.add_shape_box(link, hx=0.02, hy=0.02, hz=0.15)
            joints.append(
                b.add_joint_revolute(
                    parent=base,
                    child=link,
                    axis=wp.vec3(0.0, 1.0, 0.0),
                    parent_xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()),
                    child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.2), wp.quat_identity()),
                )
            )
            tips.append(link)
        bar = b.add_link(xform=wp.transform(wp.vec3(0.2, 0.0, 0.6), wp.quat_identity()))
        b.add_shape_box(bar, hx=0.2, hy=0.02, hz=0.02)
        joints.append(
            b.add_joint_revolute(
                parent=tips[0],
                child=bar,
                axis=wp.vec3(0.0, 1.0, 0.0),
                parent_xform=wp.transform(wp.vec3(0.0, 0.0, -0.2), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0), wp.quat_identity()),
            )
        )
        b.add_articulation(joints)
        loop = b.add_joint_revolute(
            parent=tips[1],
            child=bar,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, -0.2), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.2, 0.0, 0.0), wp.quat_identity()),
        )
        b.joint_qd[0:3] = [0.3, -0.1, 0.2]
        b.joint_qd[6 : 6 + len(velocities)] = velocities
        return b.finalize(), loop, bar

    def test_a_state_that_breaks_the_closure_lands_on_it(self):
        """A reset's velocities can violate the closure, and then have no unique continuation: the first step lands on
        the closure, and the momentum holds from then on."""
        model, loop, _bar = self._floating_mechanism(2.0)
        solver = newton.solvers.SolverFeatherPGS(model, pgs_iterations=16, pgs_beta=0.1)
        state_0, state_1, control = model.state(), model.state(), model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
        mass = model.body_mass.numpy().astype(np.float64)

        def momentum(state):
            return (state.body_qd.numpy()[:, :3].astype(np.float64) * mass[:, None]).sum(axis=0)

        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, 1.0 / 240.0)
        state_0, state_1 = state_1, state_0
        after = momentum(state_0)
        for _ in range(239):
            state_0.clear_forces()
            solver.step(state_0, state_1, control, None, 1.0 / 240.0)
            state_0, state_1 = state_1, state_0
        self.assertTrue(np.isfinite(state_0.body_q.numpy()).all())
        self.assertTrue(np.isfinite(state_0.joint_qd.numpy()).all())
        self.assertLess(_gap(model, state_0, loop), 1.0e-4)
        np.testing.assert_allclose(momentum(state_0), after, atol=1.0e-2)

    def test_a_free_body_lands_on_a_loop(self):
        """A free body's contact with a linked tree is a dense row the block corrects, wide past its two sides."""
        b, coupler, rocker = _four_bar_bars()
        b.add_joint_ball(parent=coupler, child=rocker, **_LOOP_FRAMES)
        ball = b.add_body(xform=wp.transform(wp.vec3(0.2, 0.0, 0.46), wp.quat_identity()))
        b.add_shape_sphere(ball, radius=0.04)
        model = b.finalize()
        solver = newton.solvers.SolverFeatherPGS(model, pgs_iterations=16, pgs_beta=0.1, dense_max_constraints=64)
        pipeline = newton.CollisionPipeline(model, contact_matching="latest")
        contacts = pipeline.contacts()
        state_0, state_1, control = model.state(), model.state(), model.control()
        worst = 0.0
        joint = int(np.flatnonzero(model.joint_articulation.numpy() < 0)[0])
        for _ in range(240):
            pipeline.collide(state_0, contacts)
            state_0.clear_forces()
            solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
            state_0, state_1 = state_1, state_0
            worst = max(worst, _gap(model, state_0, joint))
        self.assertTrue(np.isfinite(state_0.body_q.numpy()).all())
        self.assertLess(worst, 1.0e-4)
        # The ball rests on the coupler, or rolled off it: it never fell through.
        z = float(state_0.body_q.numpy()[ball, 2])
        coupler_z = float(state_0.body_q.numpy()[coupler, 2])
        self.assertGreater(z, coupler_z - 0.5)


class TestSplitBody(unittest.TestCase):
    def test_the_state_reports_the_primary(self):
        """``state.body_q`` holds the model's bodies, the primary pieces, where the tree's coordinates put them."""
        model = scenes.four_bar().build(wp.get_device())
        _, state = _simulate(model, targets={0: 0.6})
        self.assertEqual(state.body_q.shape[0], model.body_count)
        forward = model.state()
        newton.eval_fk(model, state.joint_q, state.joint_qd, forward)
        np.testing.assert_allclose(forward.body_q.numpy(), state.body_q.numpy(), atol=2.0e-6)

    def test_a_scene_keeps_its_contact_budget(self):
        """A link's rows come on top of the option's capacity: the Stewart's thirty leave the contacts their rows."""
        scene = scenes.stewart_platform()
        model = scene.build(wp.get_device())
        solver = newton.solvers.SolverFeatherPGS(model, dense_max_constraints=32)
        pipeline = newton.CollisionPipeline(model, contact_matching="latest")
        contacts = pipeline.contacts()
        state_0, state_1, control = model.state(), model.state(), model.control()
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
        for step in range(40):
            scene.control(model, control, state_0, step)
            pipeline.collide(state_0, contacts)
            state_0.clear_forces()
            solver.step(state_0, state_1, control, contacts, scene.dt)
            state_0, state_1 = state_1, state_0
        # Each pressed contact takes its rows, which the thirty link rows would crowd out of 32.
        self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 1)
        solver.check_constraint_capacity()


if __name__ == "__main__":
    unittest.main()
