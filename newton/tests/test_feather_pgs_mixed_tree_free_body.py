# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The generic mixed-tree cache owner covers fixed-base arms together with free rigid bodies."""

import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.mixed_tree_kinetic import MixedTreeKineticState
from newton._src.solvers.feather_pgs.solver_feather_pgs import SolverFeatherPGS

DT = 1.0 / 240.0


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


def _build(device, worlds=4, mimic=False, prescribed=False, shoulder=False, quadruped=False):
    """Per world: a fixed-base three-joint arm over a free box resting on the ground.

    With ``mimic`` the third arm joint follows the second through a mimic row,
    as a gripper finger follows its leader. With ``prescribed`` a kinematic
    free body (a moving tray) sits beside the box, as a goal marker or
    conveyor does in manipulation scenes. With ``quadruped`` a free-root
    18-coordinate walker stands beside them.
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
            parent=-1, child=parent, parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.3), wp.quat_identity())
        )
    ]
    for link in range(3):
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
                )
            )
        parent = body
    scene.add_shape_capsule(parent, radius=0.03, half_height=0.1, cfg=cfg)
    scene.add_articulation(joints)
    if mimic:
        scene.add_constraint_mimic(joint0=joints[3], joint1=joints[2], coef0=0.0, coef1=1.0)
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


TRAY_VELOCITY = 0.25


def _kinematic_free_joints(model):
    """Coordinate and velocity starts of free joints whose body is kinematic."""
    types, flags = model.joint_type.numpy(), model.body_flags.numpy()
    child = model.joint_child.numpy()
    joints = np.flatnonzero(
        (types == int(newton.JointType.FREE)) & (flags[child] & int(newton.BodyFlags.KINEMATIC) != 0)
    )
    return model.joint_q_start.numpy()[joints], model.joint_qd_start.numpy()[joints]


def _rollout(model, *, owner, steps=120, swing_joint=0, pgs_mode="matrix_free", rows=None):
    env = {
        "FEATHER_PGS_TREE_FOREST": str(int(getattr(_rollout, "forest", False))),
        "FEATHER_PGS_MIXED_TREE_KINETIC_STATE": str(int(owner)),
        "FEATHER_PGS_FRANKA_KINETIC_STATE": "0",
        "FEATHER_PGS_TREE_KINETIC_STATE": "0",
        "FEATHER_PGS_WORLD_SCAN_PUBLICATION": "0",
        "FEATHER_PGS_LOCAL_ROW_PACKETS": "0",
    }
    with patch.dict(os.environ, env):
        solver = SolverFeatherPGS(
            model,
            pgs_mode=pgs_mode,
            pgs_iterations=8,
            enable_joint_limits=True,
            update_mass_matrix_interval=2,
            friction_anchor_beta=0.0,
            **({} if rows is None else {"dense_max_constraints": rows}),
        )
    state_owner = getattr(solver, "_tree_kinetic_state", None)
    pipeline = newton.CollisionPipeline(model)
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
    for joint in np.flatnonzero(types == int(newton.JointType.REVOLUTE))[swing_joint::3]:
        qd[qd_start[joint]] = 1.5
    # Prescribed trays glide along +x at a constant speed.
    # The solver holds a kinematic joint's coordinates, so the tray pose is
    # advanced externally each step (as Isaac Lab writes kinematic targets)
    # and its velocity is given.
    tray_q, tray_qd = _kinematic_free_joints(model)
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
        pipeline.collide(states[0], contacts)
        states[0].clear_forces()
        solver.step(states[0], states[1], control, contacts, DT)
        states.reverse()
        trajectory.append(states[0].body_q.numpy().copy())
        joint_trajectory.append(states[0].joint_q.numpy().copy())
    wp.synchronize_device(model.device)
    solver.joint_q_trajectory = np.array(joint_trajectory)
    return solver, state_owner, np.array(trajectory), states[0].body_qd.numpy().copy()


@unittest.skipUnless(wp.is_cuda_available(), "Cached tree owners are a CUDA path")
class TestFeatherPGSMixedTreeFreeBody(unittest.TestCase):
    def test_free_body_group_matches_the_ordinary_route(self):
        """A fixed-base arm plus a free box is owned by the generic mixed cache and tracks the ordinary route."""
        model = _build(wp.get_device("cuda:0"))
        plain_solver, plain_owner, plain, plain_qd = _rollout(model, owner=False)
        self.assertIsNone(plain_owner)
        self.assertTrue(plain_solver._has_free_rigid_bodies)
        _owner_solver, owner, cached, cached_qd = _rollout(model, owner=True)
        self.assertIsInstance(owner, MixedTreeKineticState)
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

    def test_split_mode_is_owned_and_matches_the_ordinary_route(self):
        """Split-mode scenes (Cartpole's class) get the same owner: it refreshes the held factors the split rows use."""
        model = _build(wp.get_device("cuda:0"))
        plain_solver, plain_owner, plain, plain_qd = _rollout(model, owner=False, pgs_mode="split")
        self.assertIsNone(plain_owner)
        self.assertEqual(plain_solver.pgs_mode, "split")
        _owner_solver, owner, cached, cached_qd = _rollout(model, owner=True, pgs_mode="split")
        self.assertIsInstance(owner, MixedTreeKineticState)
        self.assertTrue(np.isfinite(cached).all() and np.isfinite(cached_qd).all())
        np.testing.assert_allclose(cached[:, :, :3], plain[:, :, :3], atol=1.0e-4, rtol=0.0)
        np.testing.assert_allclose(cached[:, :, 3:], plain[:, :, 3:], atol=1.0e-4, rtol=0.0)
        np.testing.assert_allclose(cached_qd, plain_qd, atol=2.0e-3, rtol=0.0)
        final = cached[-1].reshape(model.world_count, -1, 7)
        self.assertTrue(np.all(final[:, 4, 2] > 0.04))

    def test_multi_coordinate_descendant_joints_are_owned(self):
        """An arm with a two-coordinate D6 shoulder (Humanoid's joint class) is owned and tracks the ordinary route."""
        model = _build(wp.get_device("cuda:0"), shoulder=True)
        _plain_solver, plain_owner, plain, plain_qd = _rollout(model, owner=False)
        self.assertIsNone(plain_owner)
        _owner_solver, owner, cached, cached_qd = _rollout(model, owner=True)
        self.assertIsInstance(owner, MixedTreeKineticState)
        self.assertEqual(sorted(group.layout.dofs for group in owner.groups), [4, 6])
        self.assertTrue(np.isfinite(cached).all() and np.isfinite(cached_qd).all())
        np.testing.assert_allclose(cached[:, :, :3], plain[:, :, :3], atol=1.0e-4, rtol=0.0)
        np.testing.assert_allclose(cached[:, :, 3:], plain[:, :, 3:], atol=1.0e-4, rtol=0.0)
        np.testing.assert_allclose(cached_qd, plain_qd, atol=2.0e-3, rtol=0.0)

    def test_forest_group_owns_the_whole_world(self):
        """With FEATHER_PGS_TREE_FOREST=1 one group owns the arm, the box and the tray of each world."""
        model = _build(wp.get_device("cuda:0"), prescribed=True)
        _plain_solver, plain_owner, plain, plain_qd = _rollout(model, owner=False)
        self.assertIsNone(plain_owner)
        _rollout.forest = True
        try:
            _owner_solver, owner, cached, cached_qd = _rollout(model, owner=True)
        finally:
            _rollout.forest = False
        self.assertIsInstance(owner, MixedTreeKineticState)
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
        plain_solver, plain_owner, plain, plain_qd = _rollout(model, owner=False, rows=256)
        self.assertIsNone(plain_owner)
        self.assertIsNone(plain_solver._sparse_mass_matrix_size)
        self.assertIn(18, plain_solver.size_groups)
        self.assertFalse(plain_solver.constraint_capacity_status()["dense"])
        _owner_solver, owner, cached, cached_qd = _rollout(model, owner=True, rows=256)
        self.assertIsInstance(owner, MixedTreeKineticState)
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

    def test_mimic_rows_compose_with_the_mixed_owner(self):
        """Mimic rows are ordinary matrix-free rows over the held factors: the owner admits them and tracks the plain route."""
        model = _build(wp.get_device("cuda:0"), mimic=True)
        plain_solver, plain_owner, plain, plain_qd = _rollout(model, owner=False, swing_joint=1)
        self.assertIsNone(plain_owner)
        self.assertEqual(int(plain_solver._mimic_count), model.world_count)
        owner_solver, owner, cached, cached_qd = _rollout(model, owner=True, swing_joint=1)
        self.assertIsInstance(owner, MixedTreeKineticState)
        self.assertTrue(np.isfinite(cached).all() and np.isfinite(cached_qd).all())
        np.testing.assert_allclose(cached[:, :, :3], plain[:, :, :3], atol=1.0e-4, rtol=0.0)
        np.testing.assert_allclose(cached[:, :, 3:], plain[:, :, 3:], atol=1.0e-4, rtol=0.0)
        np.testing.assert_allclose(cached_qd, plain_qd, atol=2.0e-3, rtol=0.0)
        # The follower joint tracks its leader under the cached route.
        q = owner_solver.joint_q_trajectory.reshape(len(cached), model.world_count, -1)
        self.assertGreater(float(np.abs(q[:, :, 1]).max()), 0.02)
        self.assertLess(float(np.abs(q[:, :, 2] - q[:, :, 1]).max()), 0.02)

    def test_prescribed_free_roots_are_owned(self):
        """A kinematic free body per world is owned as a zero-response root and glides at its prescribed speed."""
        model = _build(wp.get_device("cuda:0"), prescribed=True)
        plain_solver, plain_owner, plain, plain_qd = _rollout(model, owner=False)
        self.assertIsNone(plain_owner)
        self.assertTrue(plain_solver._has_prescribed_response)
        _owner_solver, owner, cached, cached_qd = _rollout(model, owner=True)
        self.assertIsInstance(owner, MixedTreeKineticState)
        self.assertEqual(sorted(group.layout.dofs for group in owner.groups), [3, 6, 6])
        self.assertTrue(any(getattr(group.sparse, "prescribed", False) for group in owner.groups))
        # Lane widths were timed at construction and the fastest candidate kept.
        for group in owner.groups:
            self.assertGreaterEqual(len(group.lane_timing), 2)
            self.assertEqual(group.layout.lanes, min(group.lane_timing, key=group.lane_timing.get))
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


if __name__ == "__main__":
    unittest.main()
