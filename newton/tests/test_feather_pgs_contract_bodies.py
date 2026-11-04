# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The laws of trees and free bodies under SolverFeatherPGS, on its public API.

A free base's momentum against the exported forces, and the gyroscopic update of free bodies. A contract module:
:mod:`newton.tests.test_feather_pgs_contract_lint` keeps it on the public API.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS, SolverFeatherstone
from newton.tests.unittest_utils import add_function_test, get_test_devices

# ---------------------------------------------------------------------------------------------------------------------
# free base momentum
#
# Momentum conservation for a rotating floating-base articulation.
#
# FeatherPGS solves the free base in a world-aligned frame centred on a material point of the root
# body. Three things have to agree about that point, and they did not:
#
# * the root link's linear inertial wrench was zeroed, discarding the term that carries the
#   reference point as the body rotates;
# * the free-base linear coordinate was integrated without its ``omega x v`` transport term;
# * the public-to-internal shift moved the coordinate to the root body's *origin* while the frame
#   sits on the root body's *centre of mass*.
#
# Each partially compensated the others, so any one of them in isolation looked defensible. Together
# they made every rotating multi-link articulation create linear momentum from nothing — and also
# every *tumbling* single free body whose centre of mass is offset from the body origin, because the
# shift's step-to-step cancellation only survives while the angular velocity is constant.
#
# With no gravity, no contacts and no external wrench, the centre-of-mass velocity of a free
# articulation is exactly constant, so any measured drift is spurious. :class:`SolverFeatherstone`
# solves the same dynamics without those three deviations and is used here as the reference.
# ---------------------------------------------------------------------------------------------------------------------

DT = 1.0 / 200.0
STEPS = 60
# Spin about x, transverse to the stack axis, so the composite COM offset is swept through the
# rotation. A spin parallel to the offset cannot excite the term at all.
OMEGA = (20.0, 0.0, 0.0)
LINK_MASS = 1.0
LINK_HALF = 0.10  # half the stack spacing; the composite COM sits this far above the root COM


def _link_inertia(mass, radius=0.05, height=0.20):
    """Solid-cylinder inertia tensor about the link's own centre of mass."""
    i_xy = mass * (3.0 * radius**2 + height**2) / 12.0
    i_z = 0.5 * mass * radius**2
    return wp.mat33(i_xy, 0.0, 0.0, 0.0, i_xy, 0.0, 0.0, 0.0, i_z)


def _build_welded_pair(device, root_com_z=0.0):
    """Two links welded by a fixed joint: composite COM sits ``LINK_HALF`` above the root COM.

    ``root_com_z`` offsets the root link's own centre of mass from its origin, which is the case
    every real robot link has and no bare primitive does.
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    root = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        mass=LINK_MASS,
        com=wp.vec3(0.0, 0.0, root_com_z),
        inertia=_link_inertia(LINK_MASS),
    )
    upper = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 2.0 * LINK_HALF), wp.quat_identity()),
        mass=LINK_MASS,
        com=wp.vec3(0.0, 0.0, 0.0),
        inertia=_link_inertia(LINK_MASS),
    )
    free = builder.add_joint_free(parent=-1, child=root)
    weld = builder.add_joint_fixed(
        parent=root,
        child=upper,
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 2.0 * LINK_HALF), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
    )
    builder.add_articulation([free, weld])
    return builder.finalize(device=device)


def _build_equivalent_single(device):
    """One link carrying the welded pair's composite mass properties.

    Same total mass, same centre-of-mass offset from the body origin and same inertia about that
    centre of mass as :func:`_build_welded_pair`, so the two models are the same physical object.
    """
    total = 2.0 * LINK_MASS
    single = _link_inertia(LINK_MASS)
    i_xy = 2.0 * (single[0, 0] + LINK_MASS * LINK_HALF**2)
    i_z = 2.0 * single[2, 2]
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        mass=total,
        com=wp.vec3(0.0, 0.0, LINK_HALF),
        inertia=wp.mat33(i_xy, 0.0, 0.0, 0.0, i_xy, 0.0, 0.0, 0.0, i_z),
    )
    joint = builder.add_joint_free(parent=-1, child=body)
    builder.add_articulation([joint])
    return builder.finalize(device=device)


def _build_serial_chain(device, depth=2):
    """Free root followed by ``depth`` revolute links in series, axes transverse to the spin."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    prev = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        mass=LINK_MASS,
        com=wp.vec3(0.0, 0.0, 0.0),
        inertia=_link_inertia(LINK_MASS),
    )
    joints = [builder.add_joint_free(parent=-1, child=prev)]
    for level in range(depth):
        link = builder.add_link(
            xform=wp.transform(wp.vec3(0.0, 0.0, 2.0 * LINK_HALF * (level + 1)), wp.quat_identity()),
            mass=LINK_MASS,
            com=wp.vec3(0.0, 0.0, 0.0),
            inertia=_link_inertia(LINK_MASS),
        )
        joints.append(
            builder.add_joint_revolute(
                parent=prev,
                child=link,
                axis=wp.vec3(0.0, 1.0, 0.0),
                parent_xform=wp.transform(wp.vec3(0.0, 0.0, 2.0 * LINK_HALF), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            )
        )
        prev = link
    builder.add_articulation(joints)
    return builder.finalize(device=device)


def _build_single_offset_com_tumbling(device):
    """One free link with a triaxial inertia and a centre of mass offset from its origin.

    Distinct principal moments let a non-principal spin tumble, so the angular velocity changes
    within every step. The public-to-internal shift's residual is ``(w_out - w_in) x com_offset``
    per step: it needs both the tumble and the offset to be nonzero, which is why bodies spun about
    a principal axis (and every zero-offset primitive) cannot see it. The offset magnitude matches
    a humanoid pelvis.
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        mass=2.0 * LINK_MASS,
        com=wp.vec3(0.0, 0.0, -0.076),
        inertia=wp.mat33(0.04, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.06),
    )
    joint = builder.add_joint_free(parent=-1, child=body)
    builder.add_articulation([joint])
    return builder.finalize(device=device)


def _com_velocity(model, state):
    """Centre-of-mass velocity of the whole articulation [m/s]."""
    mass = model.body_mass.numpy().astype(np.float64)
    body_qd = state.body_qd.numpy().astype(np.float64).reshape(-1, 6)
    return (mass[:, None] * body_qd[:, :3]).sum(axis=0) / mass.sum()


def _spin_free_articulation(model, omega=OMEGA):
    """Spin the articulation with its composite centre of mass at rest, then step.

    Returns the largest centre-of-mass velocity drift [m/s] seen over the run.
    """
    state_0, state_1 = model.state(), model.state()

    # The free joint's linear coordinate is the ROOT body's COM velocity, which for a multi-link
    # articulation is not the composite COM. Give it the orbital velocity that leaves the composite
    # COM at rest, so both models start from the same physical state.
    body_q = state_0.body_q.numpy().astype(np.float64).reshape(-1, 7)
    mass = model.body_mass.numpy().astype(np.float64)
    com_local = model.body_com.numpy().astype(np.float64).reshape(-1, 3)
    com_world = body_q[:, :3] + com_local  # identity orientation at build time
    composite = (mass[:, None] * com_world).sum(axis=0) / mass.sum()
    root_com = com_world[0]

    joint_qd = state_0.joint_qd.numpy()
    joint_qd[0:3] = np.cross(np.array(omega), root_com - composite)
    joint_qd[3:6] = omega
    state_0.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

    solver = SolverFeatherPGS(model, angular_damping=0.0)
    control = model.control()
    reference = _com_velocity(model, state_0)
    worst = 0.0
    for _ in range(STEPS):
        solver.step(state_0, state_1, control, None, DT)
        state_0, state_1 = state_1, state_0
        worst = max(worst, float(np.linalg.norm(_com_velocity(model, state_0) - reference)))
    return worst


class TestFeatherPgsFreeBaseMomentum(unittest.TestCase):
    """Free articulations must not create linear momentum while they rotate."""

    def test_single_link_conserves_com_velocity(self):
        """A lone rotating free body in free fall holds its centre-of-mass velocity.

        With no gravity the public-to-internal shift cancels step to step, so this case passes even
        on the defective formulation; it pins the baseline, not the fix. The gravity variant below
        is the discriminating single-body case.
        """
        model = _build_equivalent_single(wp.get_device())
        self.assertLess(_spin_free_articulation(model), 1e-6)

    def test_single_link_offset_com_tumbling(self):
        """A lone tumbling free body with an offset centre of mass conserves its COM velocity.

        Tumbling changes the angular velocity within each step, so the public-to-internal shift's
        reference-point error stops cancelling step to step: on the defective formulation this body
        drifts at ~0.5 m/s within the run, versus ~1e-7 here, on every route. This is the
        single-body case the fix corrects — every
        real robot root link has an offset centre of mass; it fails without the fix.
        """
        model = _build_single_offset_com_tumbling(wp.get_device())
        self.assertLess(_spin_free_articulation(model, omega=(20.0, 0.5, 0.0)), 1e-3)

    def test_welded_pair_conserves_com_velocity(self):
        """A welded two-link articulation holds it to the integrator's own accuracy.

        The pair is the same physical object as the single link above -- same mass, centre of mass
        and inertia -- but its composite centre of mass sits away from the frame origin, so it
        exercises the reference-point bias. Drift over this run is ~0.20 m/s with the per-link
        frame correction and ~1.64 m/s without it; the bound sits between the two.
        """
        model = _build_welded_pair(wp.get_device())
        self.assertLess(_spin_free_articulation(model), 0.5)

    def test_welded_pair_drift_converges_with_timestep(self):
        """The residual drift is first-order in dt, not a fixed formulation error.

        Halving the step must roughly halve the drift. A formulation error would sit flat instead,
        which is what distinguished this defect from ordinary truncation error.
        """
        global DT  # noqa: PLW0603 - the step size is the quantity under test
        original = DT
        try:
            DT = 1.0 / 200.0
            coarse = _spin_free_articulation(_build_welded_pair(wp.get_device()))
            DT = 1.0 / 400.0
            fine = _spin_free_articulation(_build_welded_pair(wp.get_device()))
        finally:
            DT = original
        self.assertGreater(coarse, 0.0)
        self.assertLess(fine, 0.65 * coarse)

    def test_offset_root_com_conserves_com_velocity(self):
        """A root link whose own COM is offset from its origin conserves momentum too.

        This exercises the public-to-internal free-base shift, which is an identity only when the
        frame origin and the coordinate refer to the same point. It is a no-op for a bare primitive
        and the dominant error term for a real robot link.
        """
        model = _build_welded_pair(wp.get_device(), root_com_z=-0.076)
        self.assertLess(_spin_free_articulation(model), 0.5)

    def test_matches_featherstone_reference(self):
        """FeatherPGS tracks SolverFeatherstone on chains deep enough to expose the free base.

        Featherstone solves the same dynamics without FeatherPGS's frame re-centring machinery, so
        it is an independent reference. A depth-2 serial chain is the shallowest case that
        distinguishes a correct free-base bias from one that merely looks right on a welded pair.
        """
        model = _build_serial_chain(wp.get_device(), depth=2)
        trajectories = []
        for solver_cls in (SolverFeatherstone, SolverFeatherPGS):
            state_0, state_1 = model.state(), model.state()
            joint_qd = state_0.joint_qd.numpy()
            joint_qd[3:6] = OMEGA
            state_0.joint_qd.assign(joint_qd)
            newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
            solver = solver_cls(model, angular_damping=0.0)
            control = model.control()
            for _ in range(20):
                solver.step(state_0, state_1, control, None, DT)
                state_0, state_1 = state_1, state_0
            trajectories.append(state_0.body_qd.numpy().astype(np.float64).copy())
        self.assertLess(float(np.abs(trajectories[0] - trajectories[1]).max()), 1e-4)


# ---------------------------------------------------------------------------------------------------------------------
# gyroscopic
#
# Free rigid bodies must not gain energy from their gyroscopic bias.
# ---------------------------------------------------------------------------------------------------------------------


def _body_omega(state, body=0):
    q = wp.quat(*state.body_q.numpy()[body, 3:])
    omega = wp.vec3(*state.body_qd.numpy()[body, 3:])
    return np.asarray(wp.quat_rotate_inv(q, omega), dtype=np.float64)


def _model(device, inertia, omega, *, armature=0.0, mass=1.0):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_body(mass=mass, inertia=wp.mat33(inertia), com=wp.vec3(0.03, -0.02, 0.01))
    model = builder.finalize(device=device)
    model.joint_armature.fill_(armature)
    state, output = model.state(), model.state()
    state.joint_qd.assign(np.array([0.1, 0.2, 0.3, *omega], dtype=np.float32))
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return model, state, output


def test_free_spin_energy(test, device):
    inertia = np.diag([0.001, 0.02, 0.0205])
    model, state, output = _model(device, inertia, (100.0, 20.0, 50.0))
    solver = newton.solvers.SolverFeatherPGS(model, angular_damping=0.0)
    control = model.control()
    omega = _body_omega(state)
    energy = 0.5 * omega @ inertia @ omega
    momentum = np.linalg.norm(inertia @ omega)
    for step in range(1000):
        solver.step(state, output, control, None, 1.0 / 240.0)
        state, output = output, state
        omega = _body_omega(state)
        observed = 0.5 * omega @ inertia @ omega
        test.assertTrue(np.isfinite(observed), f"nonfinite energy at step {step}")
        test.assertAlmostEqual(observed / energy, 1.0, delta=1.0e-3, msg=f"step {step}")
    test.assertAlmostEqual(np.linalg.norm(inertia @ omega) / momentum, 1.0, delta=1.0e-3)
    np.testing.assert_allclose(state.body_qd.numpy()[0, :3], [0.1, 0.2, 0.3], atol=2.0e-4)


def test_principal_axis_torque(test, device):
    inertia = np.diag([0.04, 0.02, 0.05])
    for armature in (0.0, 0.03):
        model, state, output = _model(device, inertia, (0.0, 0.0, 2.0), armature=armature)
        solver = newton.solvers.SolverFeatherPGS(model, angular_damping=0.0)
        control = model.control()
        torque = 0.1
        dt = 1.0 / 240.0
        for _ in range(120):
            state.body_f.assign(np.array([[0.0, 0.0, 0.0, 0.0, 0.0, torque]], dtype=np.float32))
            solver.step(state, output, control, None, dt)
            state, output = output, state
        expected = 2.0 + 120 * dt * torque / (inertia[2, 2] + armature)
        test.assertAlmostEqual(float(state.body_qd.numpy()[0, 5]), expected, delta=1.0e-4)


def test_rotated_inertia_and_armature(test, device):
    rotation = wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 2.0, 3.0)), 0.7)
    basis = np.asarray(wp.quat_to_matrix(rotation), dtype=np.float64).reshape(3, 3)
    local_basis = np.asarray(wp.quat_to_matrix(wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.4))).reshape(3, 3)
    for scale, armature in ((1.0e-5, 0.0), (1.0, 0.003), (1.0e5, 0.0)):
        inertia = scale * local_basis @ np.diag([0.001, 0.02, 0.0205]) @ local_basis.T
        omega = basis @ local_basis @ np.array([20.0, 4.0, 10.0])
        model, state, output = _model(device, inertia, omega, mass=scale)
        rotor = armature * np.array([1.0, 2.0, 3.0])
        model.joint_armature.assign(np.array([0.0, 0.0, 0.0, *rotor], dtype=np.float32))
        q = state.joint_q.numpy()
        q[3:7] = np.asarray(rotation)
        state.joint_q.assign(q)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        solver = newton.solvers.SolverFeatherPGS(model, angular_damping=0.0)
        control = model.control()
        w = _body_omega(state)
        initial = w @ inertia @ w + np.dot(rotor, omega * omega)
        for _ in range(200):
            solver.step(state, output, control, None, 1.0 / 240.0)
            state, output = output, state
        w = _body_omega(state)
        omega = state.body_qd.numpy()[0, 3:].astype(np.float64)
        final = w @ inertia @ w + np.dot(rotor, omega * omega)
        test.assertAlmostEqual(final / initial, 1.0, delta=1.0e-3, msg=f"scale={scale}, armature={armature}")


def test_gyro_precession_converges(test, device):
    inertia = np.diag([0.001, 0.02, 0.0205])
    inverse = np.linalg.inv(inertia)
    initial = np.array([20.0, 4.0, 10.0])
    duration = 0.2

    def derivative(omega):
        return inverse @ np.cross(inertia @ omega, omega)

    # Independent double-precision Euler-equation reference in the body frame.
    reference = initial.copy()
    h = duration / 4096
    for _ in range(4096):
        k1 = derivative(reference)
        k2 = derivative(reference + 0.5 * h * k1)
        k3 = derivative(reference + 0.5 * h * k2)
        k4 = derivative(reference + h * k3)
        reference += h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    errors = []
    for steps in (48, 96):
        model, state, output = _model(device, inertia, initial)
        solver = newton.solvers.SolverFeatherPGS(model, angular_damping=0.0)
        control = model.control()
        for _ in range(steps):
            solver.step(state, output, control, None, duration / steps)
            state, output = output, state
        errors.append(np.linalg.norm(_body_omega(state) - reference))
    test.assertLess(errors[0], 0.1)
    test.assertLess(errors[1], 0.35 * errors[0])


def _mixed_scene(device, *, arm, totes):
    """Build two worlds with free bodies on both sides of a 19-DOF articulation."""
    template = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    if totes:
        template.add_body(
            xform=wp.transform(wp.vec3(-0.5, 0.0, 0.0), wp.quat_identity()),
            mass=1.0,
            inertia=wp.mat33(np.diag([0.001, 0.02, 0.0205])),
            com=wp.vec3(0.03, -0.02, 0.01),
        )
    if arm:
        joints = []
        parent = -1
        for _ in range(19):
            body = template.add_link(mass=0.5, inertia=wp.mat33(np.eye(3) * 0.02))
            joints.append(
                template.add_joint_revolute(
                    parent,
                    body,
                    parent_xform=wp.transform(wp.vec3(0.1, 0.0, 0.0), wp.quat_identity()),
                    axis=newton.Axis.Z,
                    target_ke=40.0,
                    target_kd=4.0,
                    armature=0.01,
                )
            )
            parent = body
        template.add_articulation(joints)
    if totes:
        template.add_body(
            xform=wp.transform(wp.vec3(0.5, 0.0, 0.0), wp.quat_identity()),
            mass=2.0,
            inertia=wp.mat33(np.diag([0.04, 0.002, 0.041])),
            com=wp.vec3(-0.02, 0.01, 0.04),
        )
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.replicate(template, 2, spacing=(3.0, 0.0, 0.0))
    model = builder.finalize(device=device)
    state, output = model.state(), model.state()
    velocity = state.joint_qd.numpy()
    free_joints = np.flatnonzero(model.joint_type.numpy() == newton.JointType.FREE)
    for i, joint in enumerate(free_joints):
        start = int(model.joint_qd_start.numpy()[joint])
        velocity[start + 3 : start + 6] = (100.0, 20.0, 50.0) if i % 2 == 0 else (20.0, 100.0, 50.0)
    state.joint_qd.assign(velocity)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    control = model.control()
    control.joint_target_q.fill_(0.1)
    # None of these scenes has a constraint row.
    solver = newton.solvers.SolverFeatherPGS(model, angular_damping=0.0)
    return model, solver, control, [state, output]


def _route(solver):
    """Return the route-deciding entries of a solver's path selection."""
    selection = solver.get_path_selection()
    sweep = selection["routes"]["sweep"]
    return selection["cached_prediction"], selection["sparse_mass_matrix"], sweep["route"], sweep["tiers"]


def test_mixed_arm_and_spinning_bodies(test, device):
    """Preserve isolated arm and rotor dynamics with mixed response sizes and graph replay."""
    executions = ("eager", "graph") if wp.get_device(device).is_cuda else ("eager",)
    for execution in executions:
        with test.subTest(execution=execution):
            scenes = [
                _mixed_scene(device, arm=True, totes=True),
                _mixed_scene(device, arm=True, totes=False),
                _mixed_scene(device, arm=False, totes=True),
            ]
            # The comparisons below hold within one route. The spin is not chaotic, but world
            # angular velocity w = R I^-1 R^T L with I_min = 0.001 kg*m^2 turns a one-ulp
            # orientation difference into 2e-4 rad/s within seven steps.
            routes = [_route(scene[1]) for scene in scenes]
            test.assertEqual(routes, [routes[0]] * len(routes), "scenes compared across routes")
            test.assertEqual(routes[0][0], "tree")
            test.assertIsNone(routes[0][3])
            model, _, _, states = scenes[0]
            free_bodies = (0, 20, 21, 41)
            inertias = model.body_inertia.numpy().astype(np.float64)

            def energies(state, free_bodies=free_bodies, inertias=inertias):
                """Measure each free body's rotational energy in its inertia frame."""
                values = []
                for body in free_bodies:
                    omega = _body_omega(state, body)
                    values.append(omega @ inertias[body] @ omega)
                return np.asarray(values)

            initial_energy = energies(states[0])

            def step(scenes=scenes):
                """Advance the mixed scene and its independent controls together."""
                for _, scene_solver, control, pair in scenes:
                    scene_solver.step(pair[0], pair[1], control, None, 1.0 / 240.0)
                    pair[0], pair[1] = pair[1], pair[0]

            # Compile and initialize the same layouts before capture.
            step()
            graph = None
            if execution == "graph":
                with wp.ScopedCapture(device=device) as capture:
                    step()
                    step()
                graph = capture.graph
            for _ in range(100):
                if graph is None:
                    step()
                    step()
                else:
                    wp.capture_launch(graph)
                current = scenes[0][3][0]
                np.testing.assert_allclose(energies(current) / initial_energy, 1.0, atol=1.0e-3, rtol=0.0)
                for field in ("body_q", "body_qd"):
                    width = 7 if field == "body_q" else 6
                    mixed = getattr(current, field).numpy().reshape(2, 21, width)
                    arm = getattr(scenes[1][3][0], field).numpy().reshape(2, 19, width)
                    totes = getattr(scenes[2][3][0], field).numpy().reshape(2, 2, width)
                    np.testing.assert_allclose(mixed[:, 1:20], arm, atol=3.0e-5, rtol=1.0e-6)
                    np.testing.assert_allclose(mixed[:, [0, 20]], totes, atol=3.0e-5, rtol=1.0e-6)
            test.assertGreater(float(np.abs(scenes[1][3][0].joint_q.numpy()).max()), 0.01)


class TestFeatherPGSGyroscopic(unittest.TestCase):
    """The free roots' gyroscopic update, on every test device."""


for _device in get_test_devices():
    add_function_test(
        TestFeatherPGSGyroscopic,
        "test_mixed_arm_and_spinning_bodies",
        test_mixed_arm_and_spinning_bodies,
        devices=[_device],
    )
    add_function_test(TestFeatherPGSGyroscopic, "test_free_spin_energy", test_free_spin_energy, devices=[_device])
    add_function_test(
        TestFeatherPGSGyroscopic, "test_principal_axis_torque", test_principal_axis_torque, devices=[_device]
    )
    add_function_test(
        TestFeatherPGSGyroscopic,
        "test_rotated_inertia_and_armature",
        test_rotated_inertia_and_armature,
        devices=[_device],
    )
    add_function_test(
        TestFeatherPGSGyroscopic, "test_gyro_precession_converges", test_gyro_precession_converges, devices=[_device]
    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
