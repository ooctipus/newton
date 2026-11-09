# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The separated worlds: a world none of whose rows can act at the predictor keeps zero impulse, its exact solution.

The allocators mark each world's acts word (:meth:`.placement.Placement.hold_acts`); the claims, the one writer and
the patch finish leave a world without it at zero, as the complete path's sweeps would. Each case here steps the
switch against the complete path (the test pin ``separated_worlds=False``).
"""

import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.sim.enums import BodyFlags
from newton._src.solvers.feather_pgs import field_group
from newton._src.solvers.feather_pgs.contact_input import ContactInput
from newton._src.solvers.feather_pgs.friction_patches import patch_routes
from newton._src.solvers.feather_pgs.row_slots import allocate_world_contact_slots
from newton._src.solvers.feather_pgs.solver_feather_pgs import SolverFeatherPGS
from newton.tests.feather_pgs_law_harness import anchored_patches, row_lines
from newton.tests.feather_pgs_patch_parity import assert_history_close
from newton.tests.test_feather_pgs_friction_patches import _patch_fixture
from newton.tests.test_feather_pgs_mixed_tree_free_body import DT, _add_quadruped, _build


def _contact_acts(
    speed_a, speed_b=0.0, *, gap=0.01, prescribed_b=False, kinematic_b=False, restitution=0.0, twists=True
):
    """Allocate one contact of body 0 (tree 0) over body 1 (tree 1, at rest unless ``speed_b``), its normal +z from B to
    A at gap ``gap``, each body's predicted twist moving along +z, and return the world's acts word and the contact's
    slot (dt 0.01, beta 0.2, the speculative scale 1)."""
    device = "cpu"

    def ints(values):
        return wp.array(values, dtype=wp.int32, device=device)

    def floats(values):
        return wp.array(values, dtype=wp.float32, device=device)

    twist = [wp.spatial_vector(0.0, 0.0, speed_a, 0.0, 0.0, 0.0), wp.spatial_vector(0.0, 0.0, speed_b, 0.0, 0.0, 0.0)]
    data = field_group.upload(
        ContactInput,
        device,
        count=ints([1]),
        shape0=ints([0]),
        shape1=ints([1]),
        point0=wp.zeros(1, dtype=wp.vec3, device=device),
        point1=wp.zeros(1, dtype=wp.vec3, device=device),
        normal=wp.array([wp.vec3(0.0, 0.0, -1.0)], dtype=wp.vec3, device=device),
        margin0=floats([0.0]),
        margin1=floats([0.0]),
        body_q=wp.array(
            [wp.transform((0.0, 0.0, gap), wp.quat_identity()), wp.transform_identity()],
            dtype=wp.transform,
            device=device,
        ),
        body_v=wp.array(twist, dtype=wp.spatial_vector, device=device),
        shape_body=ints([0, 1]),
        response_dofs=ints([6, 0 if prescribed_b else 6]),
        dof_start=ints([0, 6]),
        origin=wp.zeros(2, dtype=wp.vec3, device=device),
        prescribed=ints([0, int(prescribed_b)]),
        incident=floats(np.zeros(12)),
        material_mu=floats([0.5, 0.5]),
        material_restitution=floats([restitution, restitution]),
        max_depenetration_velocity=floats([np.inf, np.inf]),
        beta=0.2,
        dt=0.01,
        speculative_scale=1.0,
        restitution_threshold=0.1,
        world=ints([-9]),
        slot=ints([-9]),
        art0=ints([-9]),
        art1=ints([-9]),
        path=ints([-9]),
        slots_needed=ints([-9]),
        first=row_lines(device),
    )
    acts, counter = ints([0]), ints([0])
    flags = [0, int(BodyFlags.KINEMATIC) if kinematic_b else 0]
    published = wp.array(twist if twists else [], dtype=wp.spatial_vector, device=device)
    # The allocator's arguments in order: one thread; the bodies' trees and world, flags, coordinates and free flags;
    # no free bodies' run, no gap gates, capacities of eight rows, friction, its gap, no telemetry, the patches, twists.
    inputs = [data, 1, ints([0, 1]), ints([0, 0]), ints(flags), ints([1, 1]), ints([0, 0]), 0, 0.0, 0.0, 0.0, 8, 8, 1]
    outputs = [acts, counter, ints([0]), ints([0]), ints([0]), ints([2**31 - 1]), ints([2**31 - 1])]
    wp.launch(
        allocate_world_contact_slots,
        dim=1,
        inputs=[*inputs, float("inf"), 0, anchored_patches(1, device), published],
        outputs=[*outputs, wp.zeros((0, 0), dtype=wp.int32, device=device)],
        device=device,
    )
    return int(acts.numpy()[0]), int(data.slot.numpy()[0])


class TestSeparatedWorlds(unittest.TestCase):
    def test_contact_marks_its_world_where_its_residual_may_not_separate(self):
        """The contact allocator's residual at the predictor, ``n . (v_a - v_b)`` plus the speculative gap's rate:
        above the margin the contact leaves the world's acts word, at it or below (or without twists, or with an
        endpoint whose velocity the publication does not hold) it marks the word; its rows are reserved either way."""
        for case, expected in (
            ({"speed_a": 1.0}, 0),
            ({"speed_a": -2.0}, 1),
            ({"speed_a": -1.0}, 1),
            ({"speed_a": -1.0 + 2.0e-5}, 0),
            ({"speed_a": 0.0, "speed_b": 3.0}, 1),
            ({"speed_a": 0.0, "speed_b": 3.0, "prescribed_b": True}, 1),
            ({"speed_a": 0.0, "speed_b": -3.0, "prescribed_b": True}, 0),
            ({"speed_a": 1.0, "kinematic_b": True}, 1),
            ({"speed_a": 1.0, "twists": False}, 1),
            ({"speed_a": -2.0, "gap": -0.001, "restitution": 0.5}, 1),
        ):
            with self.subTest(**case):
                acts, slot = _contact_acts(**case)
                self.assertEqual(acts, expected)
                self.assertGreaterEqual(slot, 0)

    def test_patch_finish_releases_the_contacts_of_a_world_that_does_not_act(self):
        """The finish releases the anchor of a contact on its route in a world that does not act (its rows' zero
        load), keeps a capacity-rejected one's and finishes the others; without acts words every world acts."""
        model, state, _, patches = _patch_fixture([[-0.2, 0, 0], [0.0, 0, 0], [0.2, 0, 0], [0.4, 0, 0]])
        count = 4
        state.body_qd = wp.zeros(model.body_count, dtype=wp.spatial_vector, device="cpu")
        com = wp.zeros(model.body_count, dtype=wp.vec3, device="cpu")
        # 0 routed in world 0, which does not act; 1 capacity-rejected; 2 routed with load; 3 routed without load.
        world = wp.array([0, 0, 1, 1], dtype=int, device="cpu")
        slot = wp.array([0, -1, 0, 3], dtype=int, device="cpu")
        path = wp.array([0, -1, 0, 0], dtype=int, device="cpu")
        needed = wp.array([3, 0, 3, 3], dtype=int, device="cpu")
        parents = wp.array([[-1, 0, 0, -1, 3, 3, -1, -1]] * 2, dtype=int, device="cpu")
        mu = wp.full((2, 8), 0.5, dtype=float, device="cpu")
        impulses = wp.array([[1.0] + [0.0] * 7] * 2, dtype=float, device="cpu")
        for acts, expected in (([0, 1], [0, 1, 1, 0]), ([], [1, 1, 1, 0])):
            with self.subTest(acts=acts):
                routes = patch_routes(((0, parents, mu, impulses, row_lines(parents.device)),))
                routes.acts = wp.array(acts, dtype=int, device="cpu")
                patches.current.valid.assign([1] * count + [0] * (patches.capacity - count))
                patches.finish(state, state, com, world, slot, path, needed, routes, DT)
                np.testing.assert_array_equal(patches.previous.valid.numpy()[:count], expected)


def _rollout(model, *, separated=True, steps=90, history=None, solver_kwargs=None, toss_every=2, pins=None):
    """Swing the arms and toss every ``toss_every``-th free body; return the solver, poses and the masks of the worlds
    none of whose rows can act at the predictor (``separated``: the switch on, :meth:`.placement.Placement.hold_acts`;
    off, every world with rows takes its sweeps and the masks are empty).

    ``history`` collects one record per step: the stored anchor validity (``valid``) and the anchors the patch finish
    released in a world that does not act (``released``).
    """
    # Every step the patch law, on the general sweep unless ``pins`` says otherwise.
    pins = {"storage": "dense", "tiers": 0, "separated_worlds": separated, **(pins or {})}
    solver = SolverFeatherPGS(
        model,
        pins=SolverFeatherPGS.Pins(**pins),
        pgs_iterations=8,
        enable_joint_limits=True,
        update_mass_matrix_interval=2,
        **(solver_kwargs or {}),
    )
    pipeline = newton.CollisionPipeline(model, deterministic=True, growable_contacts=True)
    contacts = pipeline.contacts()
    states = [model.state() for _ in range(2)]
    newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
    control = model.control()
    control.joint_target_q.assign(np.zeros_like(model.joint_target_q.numpy()))
    qd = np.zeros(model.joint_dof_count, dtype=np.float32)
    types, qd_start = model.joint_type.numpy(), model.joint_qd_start.numpy()
    for joint in np.flatnonzero(types == int(newton.JointType.REVOLUTE))[::3]:
        qd[qd_start[joint]] = 1.5
    # A tossed body starts moving up: its ground contact separates at the predictor.
    for joint in np.flatnonzero(types == int(newton.JointType.FREE))[::toss_every]:
        qd[qd_start[joint] + 2] = 1.0
    states[0].joint_qd.assign(qd)
    held = []
    if history is not None:
        patches = solver.laws.friction_patches
        finish = patches.finish

        def finish_and_record(*args, **kwargs):
            # The anchors the contacts hold before the finish keeps or releases them.
            held.append(patches.current.valid.numpy().copy())
            finish(*args, **kwargs)

        patches.finish = finish_and_record
    poses, masks = [], []
    for _ in range(steps):
        held.clear()
        pipeline.collide(states[0], contacts)
        states[0].clear_forces()
        solver.step(states[0], states[1], control, contacts, DT)
        states.reverse()
        poses.append(states[0].body_q.numpy().copy())
        acts = solver.placement.world_acts.numpy()
        masks.append(acts == 0)
        if history is not None:
            count = int(contacts.rigid_contact_count.numpy()[0])
            (anchors,) = held
            skipped = np.zeros(count, dtype=bool)
            if acts.size:
                world = solver.placement.contact_world.numpy()[:count]
                skipped = (solver.placement.contact_slot.numpy()[:count] >= 0) & (acts[np.maximum(world, 0)] == 0)
            record = {
                "valid": patches.previous.valid.numpy().copy(),
                "released": int(np.sum(skipped & (anchors[:count] != 0))),
            }
            history.append(record)
    wp.synchronize_device(model.device)
    return solver, np.array(poses), np.array(masks)


def _tossed_spheres(device, worlds=4):
    """Per world, a two-joint arm hanging clear of the ground and a free sphere 1 cm above it.

    The sphere's one contact holds a patch anchor, whose three rows a matrix-free capacity of two
    rows rejects on the complete path whatever the slot order, so the anchor keeps its history.
    """
    scene = newton.ModelBuilder()
    cfg = scene.default_shape_cfg.copy()
    cfg.mu = 0.6
    parent, joints = -1, []
    for link in range(2):
        body = scene.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * 0.04), lock_inertia=True)
        joints.append(
            scene.add_joint_revolute(
                parent=parent,
                child=body,
                axis=newton.Axis.Y,
                parent_xform=wp.transform(wp.vec3(0.0, 0.0, 1.0 if link == 0 else -0.2), wp.quat_identity()),
                target_ke=30.0,
                target_kd=2.0,
                limit_lower=-1.5,
                limit_upper=1.5,
            )
        )
        parent = body
    scene.add_shape_capsule(parent, radius=0.03, half_height=0.1, cfg=cfg)
    scene.add_articulation(joints)
    sphere = scene.add_body(xform=wp.transform(wp.vec3(0.6, 0.0, 0.06), wp.quat_identity()), mass=0.5)
    scene.add_shape_sphere(sphere, radius=0.05, cfg=cfg)
    builder = newton.ModelBuilder()
    builder.replicate(scene, worlds, spacing=(3.0, 3.0, 0.0))
    builder.add_ground_plane()
    model = builder.finalize(device=device)
    model.rigid_contact_max = 256
    return model


@unittest.skipUnless(wp.get_device().is_cuda, "The predictor publishes its body twists on CUDA")
class TestSeparatedWorldsDevice(unittest.TestCase):
    """The switch against the complete path on the default device, where the predictor publishes its twists."""

    def test_separated_worlds_match_the_complete_path(self):
        """Fixed-base arms over free boxes: a world none of whose rows can act at the predictor (a tossed box's) takes
        no sweep and releases its anchors, and every world steps as on the complete path: arms of three links on the
        general sweep, of seventeen as the rules place them (the tiers where they take the worlds)."""
        for links, pins, kwargs in ((3, {}, {}), (17, {"tiers": None}, {"dense_max_constraints": 128})):
            with self.subTest(links=links):
                model = _build(wp.get_device(), worlds=6, links=links)
                options = {"pins": pins, "solver_kwargs": kwargs}
                complete_history, history = [], []
                plain_solver, plain, plain_masks = _rollout(model, separated=False, history=complete_history, **options)
                self.assertEqual(plain_masks.shape[1], 0)
                solver, poses, masks = _rollout(model, history=history, **options)
                self.assertTrue(solver.get_path_selection()["routes"]["separated_worlds"]["contacts"])
                # The tossed boxes' worlds while they rise; once a box rests its contact acts.
                self.assertGreater(int(masks.sum()), 0)
                self.assertLess(int(masks.sum()), masks.size)
                self.assertTrue(masks[:5, ::2].all())
                self.assertGreater(
                    sum(record["released"] for record in history), 0, "no skipped contact held an anchor"
                )
                for step, (record, expected) in enumerate(zip(history, complete_history, strict=True)):
                    np.testing.assert_array_equal(record["valid"], expected["valid"], err_msg=f"anchors at {step}")
                assert_history_close(solver, plain_solver)
                np.testing.assert_array_equal(poses, plain)

    def test_row_watermark_counts_the_skipped_worlds_rows(self):
        """The row watermark (a diagnostic) skips the same worlds and steps the same state as the telemetry off; the
        rows a skipped world holds count as any world's."""
        model = _build(wp.get_device(), worlds=6)
        _solver, poses, masks = _rollout(model, steps=30)
        diagnostics = {"diagnostics": SolverFeatherPGS.Diagnostics(row_watermark=True)}
        solver, watched_poses, watched_masks = _rollout(model, steps=30, solver_kwargs=diagnostics)
        np.testing.assert_array_equal(watched_masks, masks)
        np.testing.assert_array_equal(watched_poses, poses)
        self.assertGreater(int(masks.sum()), 0)
        self.assertGreater(solver.constraint_row_watermarks()["dense_high_water"], 0)

    def test_capacity_rejections_keep_their_anchors(self):
        """A separating world whose contact rows overflow a route: the allocator rejects them as on the complete path,
        and the rejected contact keeps its anchor while the world takes no sweep."""
        model = _tossed_spheres(wp.get_device())
        # Thirty steps: each sphere rises about 5 cm and starts falling, inside the 10 cm contact gap.
        # A free body's contacts are rows of the line: two rows a world cannot hold a contact's three.
        options = {"steps": 30, "toss_every": 1, "solver_kwargs": {"dense_max_constraints": 1, "mf_max_constraints": 1}}
        complete_history, history = [], []
        plain_solver, plain, _ = _rollout(model, separated=False, history=complete_history, **options)
        solver, poses, _ = _rollout(model, history=history, **options)
        self.assertGreater(min(int(np.sum(record["valid"])) for record in complete_history), 0, "no anchor kept")
        for step, (record, expected) in enumerate(zip(history, complete_history, strict=True)):
            np.testing.assert_array_equal(record["valid"], expected["valid"], err_msg=f"anchors at {step}")
        assert_history_close(solver, plain_solver)
        np.testing.assert_array_equal(poses, plain)


def _walkers(device, worlds=4):
    """One free-root quadruped per world on the ground: a class the general sparse route owns."""
    scene = newton.ModelBuilder()
    cfg = scene.default_shape_cfg.copy()
    cfg.mu = 0.6
    _add_quadruped(scene, cfg)
    builder = newton.ModelBuilder()
    builder.replicate(scene, worlds, spacing=(3.0, 3.0, 0.0))
    builder.add_ground_plane()
    model = builder.finalize(device=device)
    model.rigid_contact_max = 2048
    return model


@unittest.skipUnless(wp.is_cuda_available(), "The sparse route is a CUDA path")
class TestSeparatedWorldsSparseRoute(unittest.TestCase):
    def test_sparse_route_sweeps_every_world(self):
        """Removed packed-route flags leave the general sparse route, which no claim stage serves: no acts word."""
        model = _walkers(wp.get_device("cuda:0"))
        # A recipe that still sets the removed packed sparse-factor selectors: unknown names.
        env = {
            "FEATHER_PGS_SPARSE_FACTOR": "1",
            "FEATHER_PGS_SPARSE_LEVEL_UPDATE": "1",
            "FEATHER_PGS_SPARSE_PARALLEL_LIMITS": "1",
        }
        with patch.dict(os.environ, env):
            solver = SolverFeatherPGS(
                model,
                pgs_iterations=8,
                enable_joint_limits=True,
                dense_max_constraints=56,
            )
        self.assertLessEqual(set(env), set(solver.get_path_selection()["environment"]["unknown"]))
        self.assertIsNotNone(solver.spec.classes.packed_size)
        routes = solver.get_path_selection()["routes"]
        self.assertNotIn("research_sparse_factor", routes)
        self.assertEqual(routes["sweep"]["route"], "sparse_pgs")
        self.assertFalse(routes["separated_worlds"]["enabled"], routes["separated_worlds"]["reason"])
        self.assertEqual(solver.placement.world_acts.shape, (0,))
        pipeline = newton.CollisionPipeline(model, growable_contacts=True)
        contacts = pipeline.contacts()
        states = [model.state() for _ in range(2)]
        newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
        control = model.control()
        for _ in range(30):
            pipeline.collide(states[0], contacts)
            states[0].clear_forces()
            solver.step(states[0], states[1], control, contacts, DT)
            states.reverse()
        wp.synchronize_device(model.device)
        poses = states[0].body_q.numpy().reshape(model.world_count, -1, 7)
        self.assertTrue(np.isfinite(poses).all())
        self.assertTrue(np.all(poses[:, 0, 2] > 0.2))
        self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)


if __name__ == "__main__":
    unittest.main()
