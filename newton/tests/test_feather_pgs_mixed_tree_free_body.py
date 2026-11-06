# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The generic mixed-tree cache owner covers fixed-base arms together with free rigid bodies."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.core import memory
from newton._src.solvers.feather_pgs import tree_kinetic_state
from newton._src.solvers.feather_pgs.held_tree import HeldTree
from newton._src.solvers.feather_pgs.solver_feather_pgs import SolverFeatherPGS
from newton.tests.feather_pgs_law_harness import recomputing
from newton.tests.feather_pgs_patch_parity import assert_history_close

DT = 1.0 / 240.0
# PGS iterations of the parity subtests.
# At 8 iterations a resting box's patch anchors switch diagonals on steps that float32 rounding of its
# world position decides, each switch kicking the box by about 5e-3 rad/s, so the ordinary route fails
# these parity checks against its own replicated worlds (TestFeatherPGSMixedTreePatchFixture).
PATCH_PARITY_ITERATIONS = 32


def _add_quadruped(scene, cfg):
    """A free-root trunk with four three-joint legs: 13 bodies, 18 coordinates (a quadruped's class)."""
    stance = wp.transform(wp.vec3(-0.8, 0.0, 0.42), wp.quat_identity())
    trunk = scene.add_link(xform=stance, mass=6.0, inertia=wp.mat33(np.eye(3) * 0.2), lock_inertia=True)
    scene.add_shape_box(trunk, hx=0.25, hy=0.12, hz=0.05, cfg=cfg)
    joints = [scene.add_joint_free(child=trunk, parent_xform=stance)]
    for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        parent = trunk
        for link in range(3):
            body = scene.add_link(
                mass=0.6 - 0.1 * link, inertia=wp.mat33(np.eye(3) * (0.01 - 0.002 * link)), lock_inertia=True
            )
            if link == 0:
                origin, axis = wp.vec3(0.2 * sx, 0.12 * sy, 0.0), newton.Axis.X
            else:
                origin, axis = wp.vec3(0.0, 0.0, -0.18), newton.Axis.Y
            joints.append(
                scene.add_joint_revolute(
                    parent=parent,
                    child=body,
                    axis=axis,
                    parent_xform=wp.transform(origin, wp.quat_identity()),
                    target_ke=40.0,
                    target_kd=3.0,
                    limit_lower=-1.5,
                    limit_upper=1.5,
                )
            )
            if link:
                scene.add_shape_capsule(parent, radius=0.02, half_height=0.08, cfg=cfg)
            parent = body
        scene.add_shape_sphere(parent, radius=0.03, cfg=cfg)
    scene.add_articulation(joints)


def _add_cabinet(scene, cfg):
    """A fixed cabinet with two prismatic drawers on its base: 3 bodies, 2 independent coordinates."""
    base = scene.add_link(mass=5.0, inertia=wp.mat33(np.eye(3) * 0.3), lock_inertia=True)
    joints = [
        scene.add_joint_fixed(
            parent=-1, child=base, parent_xform=wp.transform(wp.vec3(0.6, 0.0, 0.25), wp.quat_identity())
        )
    ]
    scene.add_shape_box(base, hx=0.15, hy=0.2, hz=0.25, cfg=cfg)
    for level in (-0.1, 0.1):
        drawer = scene.add_link(mass=0.4, inertia=wp.mat33(np.eye(3) * 0.004), lock_inertia=True)
        joints.append(
            scene.add_joint_prismatic(
                parent=base,
                child=drawer,
                axis=newton.Axis.X,
                parent_xform=wp.transform(wp.vec3(-0.15, 0.0, level), wp.quat_identity()),
                target_ke=20.0,
                target_kd=2.0,
                limit_lower=-0.3,
                limit_upper=0.0,
            )
        )
        scene.add_shape_box(drawer, hx=0.12, hy=0.15, hz=0.04, cfg=cfg)
    scene.add_articulation(joints)


def _add_pads(scene, body, cfg, angle, links):
    """Four overlapping sphere pads in a row along y that touch the box's -x face at ``angle``.

    The pads sit on the arm's last link, whose origin hangs 0.2 m per later joint below the first
    joint. They reach the face at the box's starting mid height when the first joint is at ``angle``
    rad and the other joints are at zero.
    """
    radius = 0.012
    pivot = 0.6 + max(0, links - 3) * 0.2
    drop = 0.2 * (links - 1)
    # Pad centers relative to the first joint, in the world, and then in the last link's frame.
    x, z = 0.1 - radius, 0.06 - pivot
    cos, sin = np.cos(angle), np.sin(angle)
    center_x, center_z = cos * x - sin * z, sin * x + cos * z + drop
    for y in (0.02, 0.04, 0.06, 0.08):
        center = wp.transform(wp.vec3(center_x, y, center_z), wp.quat_identity())
        scene.add_shape_sphere(body, xform=center, radius=radius, cfg=cfg)


def _build(
    device,
    worlds=4,
    mimic=False,
    prescribed=False,
    shoulder=False,
    quadruped=False,
    cabinet=False,
    speed_limit=None,
    links=3,
    pads=None,
):
    """Per world: a fixed-base three-joint arm over a free box resting on the ground.

    With ``mimic`` the third arm joint follows the second through a mimic row,
    as a gripper finger follows its leader. With ``prescribed`` a kinematic
    free body (a moving tray) sits beside the box, as a goal marker or
    conveyor does in manipulation scenes. With ``quadruped`` a free-root
    18-coordinate walker stands beside them. With ``cabinet`` a fixed
    two-drawer cabinet (independent prismatic coordinates: the solver's
    diagonal-mass response class) replaces the box. ``speed_limit`` bounds
    every arm joint's velocity (the hand recipes' joint velocity limits).
    ``links`` is the arm's joint count (17 or more with the free box selects
    the solver's paired response, the Kuka scenes' class). ``pads`` replaces
    the arm's capsule with a row of sphere pads that touch the box's side when
    the first joint is at ``pads`` rad: one contact region, whose end pads own
    its two friction anchors and whose inner pads are normal-only units.
    """
    scene = newton.ModelBuilder()
    cfg = scene.default_shape_cfg.copy()
    cfg.mu = 0.6
    if quadruped:
        _add_quadruped(scene, cfg)
    # Cached trees require a fixed (or free) root: a base link welded to the world.
    parent = scene.add_link(mass=2.0, inertia=wp.mat33(np.eye(3) * 0.1), lock_inertia=True)
    joints = [
        scene.add_joint_fixed(
            parent=-1,
            child=parent,
            # A longer arm hangs from a higher base so that it clears the ground.
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.3 + max(0, links - 3) * 0.2), wp.quat_identity()),
        )
    ]
    for link in range(links):
        body = scene.add_link(
            mass=1.0 + 0.2 * link,
            inertia=wp.mat33(np.eye(3) * (0.04 + 0.01 * link)),
            com=wp.vec3(0.0, 0.0, -0.05),
            lock_inertia=True,
        )
        parent_xform = wp.transform(wp.vec3(0.0, 0.0, 0.3 if link == 0 else -0.2), wp.quat_identity())
        if shoulder and link == 1:
            # A two-coordinate D6 joint (a shoulder or hip): the descendant
            # joint class that humanoid-type robots carry.
            dof = newton.ModelBuilder.JointDofConfig
            joints.append(
                scene.add_joint_d6(
                    parent=parent,
                    child=body,
                    angular_axes=[
                        dof(axis=newton.Axis.X, target_ke=30.0, target_kd=2.0, limit_lower=-1.5, limit_upper=1.5),
                        dof(axis=newton.Axis.Y, target_ke=30.0, target_kd=2.0, limit_lower=-1.5, limit_upper=1.5),
                    ],
                    parent_xform=parent_xform,
                )
            )
        else:
            joints.append(
                scene.add_joint_revolute(
                    parent=parent,
                    child=body,
                    axis=newton.Axis.Y if link != 1 else newton.Axis.X,
                    parent_xform=parent_xform,
                    target_ke=30.0,
                    target_kd=2.0,
                    limit_lower=-1.5,
                    limit_upper=1.5,
                    velocity_limit=speed_limit,
                )
            )
        parent = body
    if pads is None:
        scene.add_shape_capsule(parent, radius=0.03, half_height=0.1, cfg=cfg)
    else:
        _add_pads(scene, parent, cfg, pads, links)
    scene.add_articulation(joints)
    if mimic:
        scene.add_constraint_mimic(joint0=joints[3], joint1=joints[2], coef0=0.0, coef1=1.0)
    if cabinet:
        _add_cabinet(scene, cfg)
    else:
        box = scene.add_body(xform=wp.transform(wp.vec3(0.15, 0.05, 0.06), wp.quat_identity()), mass=0.5)
        scene.add_shape_box(box, hx=0.05, hy=0.05, hz=0.05, cfg=cfg)
    if prescribed:
        tray = scene.add_body(xform=wp.transform(wp.vec3(-0.4, 0.0, 0.03), wp.quat_identity()), is_kinematic=True)
        scene.add_shape_box(tray, hx=0.08, hy=0.08, hz=0.03, cfg=cfg)
    builder = newton.ModelBuilder()
    builder.replicate(scene, worlds, spacing=(3.0, 3.0, 0.0))
    builder.add_ground_plane()
    model = builder.finalize(device=device)
    # The walker's thirteen shapes multiply the candidate pairs per world.
    model.rigid_contact_max = 2048 if quadruped else 256
    return model


def _fixed_root_joints(model):
    """Root joints of articulations welded to the world (FIXED joints without a parent)."""
    roots = model.articulation_start.numpy()[:-1]
    fixed = (model.joint_type.numpy()[roots] == int(newton.JointType.FIXED)) & (model.joint_parent.numpy()[roots] == -1)
    return roots[fixed]


def _repose_fixed_roots(model, solver, lift):
    """Lift every fixed root through its parent frame and notify, as Isaac Lab's fixed-base root writes do."""
    roots = _fixed_root_joints(model)
    frames = model.joint_X_p.numpy()
    frames[roots, 2] += lift
    model.joint_X_p.assign(frames)
    solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)
    return roots


TRAY_VELOCITY = 0.25


def _kinematic_free_joints(model):
    """Coordinate and velocity starts of free joints whose body is kinematic, and their articulations."""
    types, flags = model.joint_type.numpy(), model.body_flags.numpy()
    child = model.joint_child.numpy()
    joints = np.flatnonzero(
        (types == int(newton.JointType.FREE)) & (flags[child] & int(newton.BodyFlags.KINEMATIC) != 0)
    )
    articulations = wp.array(model.joint_articulation.numpy()[joints], dtype=wp.int32, device=model.device)
    return model.joint_q_start.numpy()[joints], model.joint_qd_start.numpy()[joints], articulations


def _rollout(
    model,
    *,
    owner,
    steps=120,
    swing_joint=0,
    rows=None,
    attrs=None,
    swing_stride=3,
    pgs_iterations=None,
    observe=None,
    scalar=None,
):
    """Roll out a scene and return the solver, its owner, the body pose trajectory and the final twists.

    ``pgs_iterations`` defaults to :data:`PATCH_PARITY_ITERATIONS`. ``observe`` is called after every step with the
    solver and its contacts. ``scalar=False`` pins the star to the held tree (``star='held'``); otherwise the cabinet's drawers run as scalar branches of
    the held-tree owner.
    """
    if pgs_iterations is None:
        pgs_iterations = PATCH_PARITY_ITERATIONS
    pins = {"storage": "dense"}
    # ``_rollout.forest``: False/True pins the group form; None takes the owner's default rule.
    forest = getattr(_rollout, "forest", False)
    if forest is not None:
        pins["grouping"] = "forest" if forest else "per_size"
    if scalar is False:
        pins["star"] = "held"
    solver = SolverFeatherPGS(
        model,
        pins=SolverFeatherPGS.Pins(**pins),
        pgs_iterations=pgs_iterations,
        enable_joint_limits=True,
        update_mass_matrix_interval=2,
        **({} if rows is None else {"dense_max_constraints": rows}),
        **(attrs or {}),
    )
    state_owner = solver.factor.held
    if not owner:
        recomputing(solver)
    # Sorted contacts give both arms one contact and row order whatever the thread scheduling of the atomic pair,
    # contact and row allocations. The arms are then compared, not two schedules: resting boxes and feet are
    # statically indeterminate supports, on which the order alone moves an 8-iteration rollout by up to 2.4e-4
    # (the quadruped scene). Patch rollouts also compare anchor history by contact index (assert_history_close).
    pipeline = newton.CollisionPipeline(model, deterministic=True)
    contacts = pipeline.contacts()
    states = [model.state() for _ in range(2)]
    newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
    control = model.control()
    targets = model.joint_target_q.numpy().copy()
    # Swing the arms so the capsules sweep over the boxes.
    targets[: len(targets)] = 0.0
    control.joint_target_q.assign(targets)
    # Swing the first arm joint of every world.
    qd = np.zeros(model.joint_dof_count, dtype=np.float32)
    types = model.joint_type.numpy()
    qd_start = model.joint_qd_start.numpy()
    for joint in np.flatnonzero(types == int(newton.JointType.REVOLUTE))[swing_joint::swing_stride]:
        qd[qd_start[joint]] = 1.5
    # Drawers (the cabinet scene) start sliding out.
    for joint in np.flatnonzero(types == int(newton.JointType.PRISMATIC)):
        qd[qd_start[joint]] = -0.4
    # Prescribed trays glide along +x at a constant speed.
    # The solver holds a kinematic joint's coordinates, so the tray pose is
    # advanced externally each step (as Isaac Lab writes kinematic targets)
    # and its velocity is given. Like Isaac Lab, the rollout notifies the
    # solver of each write (notify_state_changed).
    tray_q, tray_qd, tray_articulations = _kinematic_free_joints(model)
    for start in tray_qd:
        qd[start] = TRAY_VELOCITY
    states[0].joint_qd.assign(qd)
    trajectory = []
    joint_trajectory = []
    for _ in range(steps):
        if len(tray_q):
            q = states[0].joint_q.numpy()
            q[tray_q] += TRAY_VELOCITY * DT
            states[0].joint_q.assign(q)
            solver.notify_state_changed(tray_articulations)
        pipeline.collide(states[0], contacts)
        states[0].clear_forces()
        solver.step(states[0], states[1], control, contacts, DT)
        states.reverse()
        if observe is not None:
            observe(solver, contacts)
        trajectory.append(states[0].body_q.numpy().copy())
        joint_trajectory.append(states[0].joint_q.numpy().copy())
    wp.synchronize_device(model.device)
    solver.joint_q_trajectory = np.array(joint_trajectory)
    return solver, state_owner, np.array(trajectory), states[0].body_qd.numpy().copy()


@unittest.skipUnless(wp.is_cuda_available(), "Cached tree owners are a CUDA path")
class TestFeatherPGSMixedTreeFreeBody(unittest.TestCase):
    def _assert_patch_history(self, plain_solver, owner_solver, *, loaded=True):
        """Under patch friction, end both routes with the same anchor history.

        ``loaded``: resting boxes hold sticking anchors at the end, so the
        compared history is populated. The cabinet scene has no patch contact.
        The history compares by contact index, one more reason rollouts sort
        their contacts. Rounding of the box's corner positions picks the
        diagonal that carries its anchors: rounding-level differences keep it
        (TestFeatherPGSMixedTreePatchFixture), but a box yaw of 1e-6 rad moves
        it on CPU, changing validity bits while the twists still agree.
        """
        if loaded:
            self.assertTrue(
                plain_solver.laws.friction_patches.previous.valid.numpy().any(), "Resting boxes hold anchors"
            )
        assert_history_close(plain_solver, owner_solver)

    def test_free_body_group_matches_the_ordinary_route(self):
        """A fixed-base arm plus a free box is owned by the generic mixed cache and tracks the ordinary route."""
        model = _build(wp.get_device("cuda:0"))
        plain_solver, _plain_owner, plain, plain_qd = _rollout(model, owner=False)
        self.assertTrue(plain_solver.forest.has_free_rigid_bodies)
        owner_solver, owner, cached, cached_qd = _rollout(model, owner=True)
        self.assertIsInstance(owner, HeldTree)
        self.assertEqual(sorted(group.layout.dofs for group in owner.groups), [3, 6])
        self.assertEqual(sorted(group.layout.bodies for group in owner.groups), [1, 4])
        self.assertTrue(any(group.layout.root_dofs == 6 for group in owner.groups))
        self.assertTrue(np.isfinite(cached).all() and np.isfinite(cached_qd).all())
        # Same physics through both routes: poses within a tenth of a millimetre after 120 steps.
        np.testing.assert_allclose(cached[:, :, :3], plain[:, :, :3], atol=1.0e-4, rtol=0.0)
        np.testing.assert_allclose(cached[:, :, 3:], plain[:, :, 3:], atol=1.0e-4, rtol=0.0)
        np.testing.assert_allclose(cached_qd, plain_qd, atol=2.0e-3, rtol=0.0)
        # The boxes stay on the ground and the arms stay up.
        final = cached[-1].reshape(model.world_count, -1, 7)
        self.assertTrue(np.all(final[:, 4, 2] > 0.04))
        self.assertTrue(np.all(final[:, 1, 2] > 0.2))
        self.assertEqual(int(owner.groups[0].status.numpy().sum()), 0)
        self._assert_patch_history(plain_solver, owner_solver)

    def test_multi_coordinate_descendant_joints_are_owned(self):
        """An arm with a two-coordinate D6 shoulder (Humanoid's joint class) is owned and tracks the ordinary route."""
        model = _build(wp.get_device("cuda:0"), shoulder=True)
        _plain_solver, _plain_owner, plain, plain_qd = _rollout(model, owner=False)
        _owner_solver, owner, cached, cached_qd = _rollout(model, owner=True)
        self.assertIsInstance(owner, HeldTree)
        self.assertEqual(sorted(group.layout.dofs for group in owner.groups), [4, 6])
        self.assertTrue(np.isfinite(cached).all() and np.isfinite(cached_qd).all())
        np.testing.assert_allclose(cached[:, :, :3], plain[:, :, :3], atol=1.0e-4, rtol=0.0)
        np.testing.assert_allclose(cached[:, :, 3:], plain[:, :, 3:], atol=1.0e-4, rtol=0.0)
        np.testing.assert_allclose(cached_qd, plain_qd, atol=2.0e-3, rtol=0.0)

    def test_forest_group_owns_the_whole_world(self):
        """Pinned to the forest grouping, one group owns the arm, the box and the tray of each world."""
        model = _build(wp.get_device("cuda:0"), prescribed=True)
        _plain_solver, _plain_owner, plain, plain_qd = _rollout(model, owner=False)
        _rollout.forest = True
        try:
            _owner_solver, owner, cached, cached_qd = _rollout(model, owner=True)
        finally:
            _rollout.forest = False
        self.assertIsInstance(owner, HeldTree)
        self.assertEqual(len(owner.groups), 1)
        layout = owner.groups[0].layout
        self.assertEqual(layout.roots, (0, 4, 5))
        self.assertEqual(layout.root_coordinates, (0, 6, 6))
        self.assertEqual(layout.root_families, (3, 6, 0))
        self.assertTrue(np.isfinite(cached).all() and np.isfinite(cached_qd).all())
        np.testing.assert_allclose(cached[:, :, :3], plain[:, :, :3], atol=1.0e-4, rtol=0.0)
        np.testing.assert_allclose(cached[:, :, 3:], plain[:, :, 3:], atol=1.0e-4, rtol=0.0)
        np.testing.assert_allclose(cached_qd, plain_qd, atol=2.0e-3, rtol=0.0)

    def test_free_root_trees_above_twelve_coordinates_are_owned(self):
        """An 18-coordinate free-root walker (a quadruped's class) gets a held family and tracks the ordinary route."""
        model = _build(wp.get_device("cuda:0"), quadruped=True)
        # The walker's contact and limit rows need the row budget the quadruped recipes set.
        plain_solver, _plain_owner, plain, plain_qd = _rollout(model, owner=False, rows=256)
        self.assertIsNone(plain_solver.placement.sparse_mass_matrix_size)
        self.assertIn(18, plain_solver.forest.size_groups)
        self.assertFalse(plain_solver.constraint_capacity_status()["dense"])
        _owner_solver, owner, cached, cached_qd = _rollout(model, owner=True, rows=256)
        self.assertIsInstance(owner, HeldTree)
        self.assertEqual(sorted(group.layout.dofs for group in owner.groups), [3, 6, 18])
        self.assertTrue(np.isfinite(cached).all() and np.isfinite(cached_qd).all())
        # Same physics through both routes; the walker's twenty-odd contact rows
        # per world accumulate a little more factor-order noise than the arm scenes.
        np.testing.assert_allclose(cached[:, :, :3], plain[:, :, :3], atol=3.0e-4, rtol=0.0)
        np.testing.assert_allclose(cached[:, :, 3:], plain[:, :, 3:], atol=3.0e-4, rtol=0.0)
        np.testing.assert_allclose(cached_qd, plain_qd, atol=3.0e-3, rtol=0.0)
        # The walker stands: its trunk stays above its resting height after the rollout.
        final = cached[-1].reshape(model.world_count, -1, 7)
        self.assertTrue(np.all(final[:, 0, 2] > 0.2))
        self.assertEqual(int(sum(int(group.status.numpy().sum()) for group in owner.groups)), 0)

    def test_quadruped_scenes_are_owned(self):
        """Quadruped scenes (the legged recipes' class) hand their dynamics to the owner."""
        model = _build(wp.get_device("cuda:0"), quadruped=True)
        _plain_solver, _plain_owner, plain, plain_qd = _rollout(model, owner=False, rows=256)
        _owner_solver, owner, cached, cached_qd = _rollout(model, owner=True, rows=256)
        self.assertIsInstance(owner, HeldTree)
        self.assertEqual(sorted(group.layout.dofs for group in owner.groups), [3, 6, 18])
        self.assertTrue(np.isfinite(cached).all() and np.isfinite(cached_qd).all())
        np.testing.assert_allclose(cached[:, :, :3], plain[:, :, :3], atol=3.0e-4, rtol=0.0)
        np.testing.assert_allclose(cached[:, :, 3:], plain[:, :, 3:], atol=3.0e-4, rtol=0.0)
        np.testing.assert_allclose(cached_qd, plain_qd, atol=3.0e-3, rtol=0.0)
        final = cached[-1].reshape(model.world_count, -1, 7)
        self.assertTrue(np.all(final[:, 0, 2] > 0.2))

    def test_forest_group_carries_a_diagonal_family(self):
        """A world of an arm and a two-drawer cabinet (Franka cabinet's class) is one forest group with a diagonal family.

        The drawers run as scalar branches by default (see
        :meth:`test_scalar_branches_own_the_cabinet_drawers`), which no forest spans, so the forest
        takes them with the star pinned to the held tree.
        """
        model = _build(wp.get_device("cuda:0"), cabinet=True)
        plain_solver, _plain_owner, plain, plain_qd = _rollout(model, owner=False)
        _rollout.forest = True
        try:
            owner_solver, owner, cached, cached_qd = _rollout(model, owner=True, scalar=False)
        finally:
            _rollout.forest = False
        self.assertIsInstance(owner, HeldTree)
        self.assertEqual(len(owner.groups), 1)
        layout = owner.groups[0].layout
        self.assertEqual(layout.roots, (0, 4))
        self.assertEqual(layout.root_families, (3, -2))
        # The diagonal family's refresh reads H_ii from its root's offset in the layout.
        self.assertEqual([family[:2] for family in owner.groups[0].sparse.families], [(2, layout.root_offsets[1])])
        self.assertTrue(np.isfinite(cached).all() and np.isfinite(cached_qd).all())
        np.testing.assert_allclose(cached[:, :, :3], plain[:, :, :3], atol=1.0e-4, rtol=0.0)
        np.testing.assert_allclose(cached[:, :, 3:], plain[:, :, 3:], atol=1.0e-4, rtol=0.0)
        np.testing.assert_allclose(cached_qd, plain_qd, atol=2.0e-3, rtol=0.0)
        # The drawers slid out before their drives pulled them back to the closed limit.
        q = owner_solver.joint_q_trajectory.reshape(len(cached), model.world_count, -1)
        self.assertLess(float(q[:, :, 3:5].min()), -0.02)
        self.assertLessEqual(float(q[-1, :, 3:5].max()), 1.0e-3)
        self._assert_patch_history(plain_solver, owner_solver, loaded=False)

    def test_scalar_branches_track_the_cabinet_drawers(self):
        """Run the cabinet's drawers as scalar branches beside the arm's tree group; track the ordinary route.

        The drawers hang from a world-fixed root on one-coordinate prismatic joints, the
        diagonal-mass class the scalar rule admits. Poses and twists stay within the per-size and
        forest owners' bounds of ordinary dynamics; the route probe records the owner's groups.
        """
        model = _build(wp.get_device("cuda:0"), cabinet=True)
        plain_solver, _plain_owner, plain, plain_qd = _rollout(model, owner=False, scalar=False)
        owner_solver, owner, cached, cached_qd = _rollout(model, owner=True, scalar=True)
        self.assertIsNotNone(owner.scalar)
        self.assertTrue(np.isfinite(cached).all() and np.isfinite(cached_qd).all())
        np.testing.assert_allclose(cached[:, :, :3], plain[:, :, :3], atol=1.0e-4, rtol=0.0)
        np.testing.assert_allclose(cached[:, :, 3:], plain[:, :, 3:], atol=1.0e-4, rtol=0.0)
        np.testing.assert_allclose(cached_qd, plain_qd, atol=2.0e-3, rtol=0.0)
        # The drawers slid out before their drives pulled them back to the closed limit.
        q = owner_solver.joint_q_trajectory.reshape(len(cached), model.world_count, -1)
        self.assertLess(float(q[:, :, 3:5].min()), -0.02)
        self.assertLessEqual(float(q[-1, :, 3:5].max()), 1.0e-3)
        self.assertEqual(int(owner.groups[0].status.numpy().sum()), 0)
        self._assert_patch_history(plain_solver, owner_solver, loaded=False)

    def test_joint_velocity_limits_are_owned(self):
        """Joint velocity limits (the hand recipes) are ordinary rows over the owner's motion velocities."""
        model = _build(wp.get_device("cuda:0"), speed_limit=0.8)
        recipe = {"enable_joint_velocity_limits": True}
        plain_solver, _plain_owner, plain, plain_qd = _rollout(model, owner=False, attrs=recipe)
        self.assertTrue(plain_solver.options.enable_joint_velocity_limits)
        owner_solver, owner, cached, cached_qd = _rollout(model, owner=True, attrs=recipe)
        self.assertIsInstance(owner, HeldTree)
        self.assertFalse(any(group.bank_compare for group in owner.groups))
        self.assertTrue(np.isfinite(cached).all() and np.isfinite(cached_qd).all())
        np.testing.assert_allclose(cached[:, :, :3], plain[:, :, :3], atol=1.0e-4, rtol=0.0)
        np.testing.assert_allclose(cached[:, :, 3:], plain[:, :, 3:], atol=1.0e-4, rtol=0.0)
        np.testing.assert_allclose(cached_qd, plain_qd, atol=2.0e-3, rtol=0.0)
        # The swung joint started at 1.5 rad/s and was held under its 0.8 rad/s limit.
        q = owner_solver.joint_q_trajectory.reshape(len(cached), model.world_count, -1)
        self.assertLess(float(np.abs(np.diff(q[:, :, 0], axis=0)).max()) / DT, 1.0)
        self._assert_patch_history(plain_solver, owner_solver)

    def test_paired_response_scenes_are_owned(self):
        """A 17-joint arm with a free box per world (the Kuka scenes' class) selects the paired response and is owned."""
        model = _build(wp.get_device("cuda:0"), links=17)
        plain_solver, _plain_owner, plain, plain_qd = _rollout(model, owner=False, rows=192, swing_stride=17)
        self.assertEqual(plain_solver.sweep.paired_response_primary_size, 17)
        self.assertTrue(plain_solver.sweep.paired_factor_coordinates)
        owner_solver, owner, cached, cached_qd = _rollout(model, owner=True, rows=192, swing_stride=17)
        self.assertIsInstance(owner, HeldTree)
        self.assertEqual(owner_solver.sweep.paired_response_primary_size, 17)
        self.assertEqual(sorted(group.layout.dofs for group in owner.groups), [6, 17])
        self.assertTrue(np.isfinite(cached).all() and np.isfinite(cached_qd).all())
        np.testing.assert_allclose(cached[:, :, :3], plain[:, :, :3], atol=3.0e-4, rtol=0.0)
        np.testing.assert_allclose(cached[:, :, 3:], plain[:, :, 3:], atol=3.0e-4, rtol=0.0)
        np.testing.assert_allclose(cached_qd, plain_qd, atol=3.0e-3, rtol=0.0)
        final = cached[-1].reshape(model.world_count, -1, 7)
        self.assertTrue(np.all(final[:, -1, 2] > 0.04))

    def test_mimic_rows_compose_with_the_mixed_owner(self):
        """Mimic rows are ordinary matrix-free rows over the held factors: the owner admits them and tracks the plain route."""
        model = _build(wp.get_device("cuda:0"), mimic=True)
        plain_solver, _plain_owner, plain, plain_qd = _rollout(model, owner=False, swing_joint=1)
        self.assertEqual(int(plain_solver.laws.mimic_count), model.world_count)
        owner_solver, owner, cached, cached_qd = _rollout(model, owner=True, swing_joint=1)
        self.assertIsInstance(owner, HeldTree)
        self.assertTrue(np.isfinite(cached).all() and np.isfinite(cached_qd).all())
        np.testing.assert_allclose(cached[:, :, :3], plain[:, :, :3], atol=1.0e-4, rtol=0.0)
        np.testing.assert_allclose(cached[:, :, 3:], plain[:, :, 3:], atol=1.0e-4, rtol=0.0)
        np.testing.assert_allclose(cached_qd, plain_qd, atol=2.0e-3, rtol=0.0)
        # The follower joint tracks its leader under the cached route.
        q = owner_solver.joint_q_trajectory.reshape(len(cached), model.world_count, -1)
        self.assertGreater(float(np.abs(q[:, :, 1]).max()), 0.02)
        self.assertLess(float(np.abs(q[:, :, 2] - q[:, :, 1]).max()), 0.02)
        self._assert_patch_history(plain_solver, owner_solver)

    def test_prescribed_free_roots_are_owned(self):
        """A kinematic free body per world is owned as a zero-response root and glides at its prescribed speed."""
        model = _build(wp.get_device("cuda:0"), prescribed=True)
        plain_solver, _plain_owner, plain, plain_qd = _rollout(model, owner=False)
        self.assertTrue(plain_solver.forest.has_prescribed_response)
        owner_solver, owner, cached, cached_qd = _rollout(model, owner=True)
        self.assertIsInstance(owner, HeldTree)
        self.assertEqual(sorted(group.layout.dofs for group in owner.groups), [3, 6, 6])
        self.assertTrue(any(getattr(group.sparse, "prescribed", False) for group in owner.groups))
        # Lane widths follow from each group's tree (tree_kinetic_state.held_lanes); every block is whole.
        for group in owner.groups:
            self.assertEqual(
                group.layout.lanes,
                tree_kinetic_state.held_lanes(group.tree, group.group_count, memory.device(model.device)),
            )
            self.assertEqual(group.launch_count * group.layout.pack, group.group_count)
        self.assertTrue(np.isfinite(cached).all() and np.isfinite(cached_qd).all())
        np.testing.assert_allclose(cached[:, :, :3], plain[:, :, :3], atol=1.0e-4, rtol=0.0)
        np.testing.assert_allclose(cached[:, :, 3:], plain[:, :, 3:], atol=1.0e-4, rtol=0.0)
        np.testing.assert_allclose(cached_qd, plain_qd, atol=2.0e-3, rtol=0.0)
        # The tray is the last body of every world: the published pose follows
        # the externally advanced coordinates (v*t over the rollout) and stays level.
        first = cached[0].reshape(model.world_count, -1, 7)
        final = cached[-1].reshape(model.world_count, -1, 7)
        self.assertEqual(final.shape[1], 6)
        np.testing.assert_allclose(
            final[:, 5, 0] - first[:, 5, 0], TRAY_VELOCITY * (len(cached) - 1) * DT, atol=2.0e-3, rtol=0.0
        )
        np.testing.assert_allclose(final[:, 5, 2], 0.03, atol=1.0e-4, rtol=0.0)
        tray_qd = cached_qd.reshape(model.world_count, -1, 6)[:, 5]
        self.assertGreater(float(np.abs(tray_qd).max()), 0.9 * TRAY_VELOCITY)
        self.assertEqual(int(sum(int(group.status.numpy().sum()) for group in owner.groups)), 0)
        self._assert_patch_history(plain_solver, owner_solver)

    def test_fixed_root_repose_keeps_the_owner(self):
        """Re-pose fixed roots mid-rollout through joint_X_p and JOINT_PROPERTIES and keep tracking the ordinary route.

        Isaac Lab writes a fixed-base articulation's root pose into
        ``joint_X_p`` and notifies ``JOINT_PROPERTIES`` on every reset. The held
        kernels read joint frames live: a write that moves no frame keeps every
        cache (the next step only finishes), and a moved root invalidates every
        cache (the next step repairs every tree), so the owner accepts the new
        frames instead of refusing them.
        """
        lift, rewrite_after, repose_after = 0.03, 10, 20
        rollouts, generations = [], []
        for owner in (False, True):
            model = _build(wp.get_device("cuda:0"))
            steps = []

            def repose(solver, contacts, model=model, steps=steps, owner=owner):
                steps.append(len(steps))
                if owner:
                    # Every finish and every recomputing repair adds one to its tree's generation.
                    held = solver.factor.held
                    generations.append([int(group.data.generation.numpy().sum()) for group in held.groups])
                if len(steps) == rewrite_after:
                    model.joint_X_p.assign(model.joint_X_p.numpy())
                    solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)
                if len(steps) == repose_after:
                    _repose_fixed_roots(model, solver, lift)

            rollouts.append((model, *_rollout(model, owner=owner, steps=60, observe=repose)))
        (model, _plain_solver, _plain_owner, plain, plain_qd) = rollouts[0]
        (_, owner_solver, owner, cached, cached_qd) = rollouts[1]
        self.assertIsInstance(owner, HeldTree)
        self.assertIs(owner_solver.factor.held, owner)
        self.assertTrue(np.isfinite(cached).all() and np.isfinite(cached_qd).all())
        np.testing.assert_allclose(cached[:, :, :3], plain[:, :, :3], atol=1.0e-4, rtol=0.0)
        np.testing.assert_allclose(cached[:, :, 3:], plain[:, :, 3:], atol=1.0e-4, rtol=0.0)
        np.testing.assert_allclose(cached_qd, plain_qd, atol=2.0e-3, rtol=0.0)
        # The welded bases moved by the lift and hold the new pose.
        bases = model.joint_child.numpy()[_fixed_root_joints(model)]
        np.testing.assert_allclose(cached[-1, bases, 2] - cached[repose_after - 2, bases, 2], lift, atol=1.0e-6)
        self.assertEqual(int(sum(int(group.status.numpy().sum()) for group in owner.groups)), 0)
        # Generations added by each step; entry k is the step after the k-th observation.
        added = np.diff(np.asarray(generations), axis=0)
        trees = np.asarray([group.group_count for group in owner.groups])
        np.testing.assert_array_equal(added[rewrite_after - 1], trees, err_msg="unmoved frames keep every cache")
        np.testing.assert_array_equal(added[repose_after - 1], 2 * trees, err_msg="a moved root repairs every tree")


def _box_anchor_corners(model, solver, contacts, box=4):
    """Per world: the box-frame corner signs (x, y) of the box contacts that carry patch anchors."""
    count = int(contacts.rigid_contact_count.numpy()[0])
    bodies_per_world = model.body_count // model.world_count
    shape_body = model.shape_body.numpy()
    carriers = np.flatnonzero(solver.laws.friction_patches.view.weight.numpy()[:count] > 0.0)
    corners = [set() for _ in range(model.world_count)]
    for shapes, points in (
        (contacts.rigid_contact_shape0.numpy(), contacts.rigid_contact_point0.numpy()),
        (contacts.rigid_contact_shape1.numpy(), contacts.rigid_contact_point1.numpy()),
    ):
        for c in carriers:
            body = int(shape_body[shapes[c]])
            if body >= 0 and body % bodies_per_world == box:
                corners[body // bodies_per_world].add((int(np.sign(points[c][0])), int(np.sign(points[c][1]))))
    return corners


class TestFeatherPGSMixedTreePatchFixture(unittest.TestCase):
    """Check on CPU that the patch parity subtests compare rounding-stable outcomes."""

    def test_patch_parity_fixture_is_roundoff_stable(self):
        """Hold the replicated worlds and a rounding-perturbed rerun of the patch fixture on one anchor history.

        The four worlds differ only in the float32 rounding of their positions, and a box mass of
        x(1+1e-6) changes the rounding again, so a cross-route comparison of the box is well posed only
        if all of them rest on the same anchors with the same velocity. Measured on CPU (ordinary split
        route): at 8 iterations the final box velocity spreads by 7.4e-3 across the worlds; at 24 the
        spread is 2.0e-5, but the anchor corners and validity bits still switch on rounding-chosen
        steps; at :data:`PATCH_PARITY_ITERATIONS` nothing switches and the spread is 2.7e-6.
        """
        histories = []
        for mass_scale in (1.0, 1.0 + 1.0e-6):
            model = _build(wp.get_device("cpu"))
            if mass_scale != 1.0:
                boxes = np.arange(model.world_count) * (model.body_count // model.world_count) + 4
                mass, inverse = model.body_mass.numpy(), model.body_inv_mass.numpy()
                mass[boxes] *= np.float32(mass_scale)
                inverse[boxes] = 1.0 / mass[boxes]
                model.body_mass.assign(mass)
                model.body_inv_mass.assign(inverse)
            corners, valid = [], []

            def observe(solver, contacts, model=model, corners=corners, valid=valid):
                corners.append(_box_anchor_corners(model, solver, contacts))
                valid.append(solver.laws.friction_patches.previous.valid.numpy().copy())

            solver, _owner, _trajectory, final_qd = _rollout(model, owner=False, observe=observe)
            self.assertEqual(solver.options.pgs_iterations, PATCH_PARITY_ITERATIONS)
            histories.append(valid)
            if mass_scale != 1.0:
                continue
            box = final_qd.reshape(model.world_count, -1, 6)[:, 4]
            # A tenth of the parity subtests' 2e-3 twist tolerance.
            self.assertLess(float(np.ptp(box, axis=0).max()), 2.0e-4)
            for step, per_world in enumerate(corners):
                self.assertEqual(per_world, [per_world[0]] * len(per_world), f"box anchor corners at step {step}")
            self.assertEqual(len(corners[-1][0]), 2, "The resting box holds two anchors")
        for step, (first, second) in enumerate(zip(*histories, strict=True)):
            np.testing.assert_array_equal(first, second, err_msg=f"previous.valid at step {step}")
        self.assertTrue(histories[0][-1].any(), "The resting boxes hold anchors")


class TestFeatherPGSMixedTreeFrames(unittest.TestCase):
    """Check on CPU which model changes the held-tree groups accept."""

    def test_joint_frame_notifications_keep_the_held_groups(self):
        """Report moved joint frames under JOINT_PROPERTIES without raising, prove unmoved ones, refuse a changed axis.

        The owner is built directly on CPU, where the solver keeps ordinary
        dynamics, and validated as ``notify_model_changed`` validates it: a
        proof that finds every frame and axis unchanged (``True``) keeps the
        caches, a moved frame (``False``) invalidates them, and the moved frames
        are the proof's reference from then on.
        """
        joint = newton.ModelFlags.JOINT_PROPERTIES
        model = _build(wp.get_device("cpu"), worlds=2)
        solver = SolverFeatherPGS(model)
        owner = HeldTree(solver)
        self.assertTrue(owner.groups)
        roots = _fixed_root_joints(model)
        self.assertEqual(len(roots), model.world_count)
        self.assertTrue(owner.validate_model(joint))
        frames = model.joint_X_p.numpy()
        frames[roots, :3] += (0.1, -0.05, 0.03)
        model.joint_X_p.assign(frames)
        self.assertFalse(owner.validate_model(joint))
        self.assertTrue(owner.validate_model(joint))
        child_frames = model.joint_X_c.numpy()
        child_frames[roots + 1, 2] += 0.01
        model.joint_X_c.assign(child_frames)
        self.assertFalse(owner.validate_model(joint))
        self.assertTrue(owner.validate_model(newton.ModelFlags.ALL))
        axes = model.joint_axis.numpy()
        axes[model.joint_qd_start.numpy()[roots + 1]] = (1.0, 0.0, 0.0)
        model.joint_axis.assign(axes)
        with self.assertRaisesRegex(RuntimeError, "joint_axis changed"):
            owner.validate_model(joint)


if __name__ == "__main__":
    unittest.main()
