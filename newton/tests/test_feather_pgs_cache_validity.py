# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Keep held-tree caches valid across host calls and state copies that change none of their inputs."""

import contextlib
import functools
import itertools
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import cached_state_binding, capture_joins, tree_kinetic_owner
from newton._src.solvers.feather_pgs.mixed_tree_kinetic import MixedTreeKineticState
from newton._src.solvers.feather_pgs.solver_feather_pgs import SolverFeatherPGS
from newton.tests.test_feather_pgs_mixed_tree_free_body import _build
from newton.tests.test_feather_pgs_sparse_solver import STATE_FIELDS, _build_model, _make_solver

DT = 1.0 / 240.0
JOINT = newton.ModelFlags.JOINT_PROPERTIES
# One tree per block: by default each held-tree group times its candidate lane widths at
# construction, so two solver instances could run different layouts (and reduction orders).
ONE_TREE_PER_BLOCK = {"FEATHER_PGS_TREE_LANE_PACK": "0"}


def _groups(owner):
    """The owner's cache groups: a mixed owner composes several, a sparse tree owner is one."""
    return list(owner.groups) if hasattr(owner, "groups") else [owner]


def _generations(owner):
    """Per-group cache generations: every finish and every recomputing repair adds one."""
    return [group.data.generation.numpy().copy() for group in _groups(owner)]


def _group_worlds(solver, owner):
    """The world of every cache group, aligned with :func:`_generations`."""
    art_to_world = solver.art_to_world.numpy()
    return [art_to_world[group.sparse.plan.group_to_art.numpy()] for group in _groups(owner)]


def _group_articulations(owner):
    """Every cache group's articulations, one row per group (a forest group owns several), aligned with :func:`_generations`."""
    rows = []
    for group in _groups(owner):
        forest = getattr(group.sparse, "forest_arts", None)
        rows.append(np.asarray(forest) if forest is not None else group.sparse.plan.group_to_art.numpy()[:, None])
    return rows


def _fixed_root(model, world):
    """The fixed root joint of ``world``'s arm."""
    roots = (model.joint_parent.numpy() == -1) & (model.joint_type.numpy() == int(newton.JointType.FIXED))
    return int(np.flatnonzero(roots & (model.joint_world.numpy() == world))[0])


def _mixed_solver(model, *, forest=False, owner=True, lanes=None, **options):
    """A dense-route solver with the mixed held-tree owner, or with ordinary dynamics when ``owner`` is False.

    ``lanes`` fixes the held-tree lane width (several small trees per block); by default each tree has a block.
    """
    environment = {
        "FEATHER_PGS_MIXED_TREE_KINETIC_STATE": str(int(owner)),
        "FEATHER_PGS_TREE_KINETIC_STATE": "0",
        "FEATHER_PGS_TREE_FOREST": str(int(forest)),
        **ONE_TREE_PER_BLOCK,
    }
    if lanes is not None:
        environment["FEATHER_PGS_TREE_LANE_PACK"] = str(lanes)
    settings = {
        "pgs_mode": "matrix_free",
        "pgs_iterations": 8,
        "enable_joint_limits": True,
        "update_mass_matrix_interval": 2,
        "friction_anchor_beta": 0.0,
        **options,
    }
    with patch.dict(os.environ, environment):
        return SolverFeatherPGS(model, **settings)


def _owned_case(name, device, **options):
    """A model and a solver with the named held-tree owner kind: per-size groups, the forest or the sparse tree.

    ``options`` override the solver's settings. A captured graph bakes the mass refresh flags of the steps it
    records, so a test that compares replays with eager steps uses an ``update_mass_matrix_interval`` that
    divides the steps per graph (the mixed cases refresh every 2 steps, the sparse case every 4 by default).
    """
    if name == "mixed":
        model = _build(device, worlds=4)
        return model, _mixed_solver(model, **options)
    if name == "forest":
        model = _build(device, worlds=4, prescribed=True)
        return model, _mixed_solver(model, forest=True, **options)
    model = _build_model()
    with patch.dict(os.environ, ONE_TREE_PER_BLOCK):
        return model, _make_solver(model, True, **{"cached_dynamics": "on", "friction_anchor_beta": 0.0, **options})


class _Rollout:
    """Ping-pong one solver's states over sorted contacts, as a captured Isaac Lab physics step does.

    With ``copy_back`` the input bank stays the same object and each step copies the output bank
    back into it, as Isaac Lab does after an odd substep count. Rollouts of separate solver
    instances compare within 2e-6 (rtol 0), the bound the repository's other cross-instance tests
    use. Their held-tree lane width is pinned (``ONE_TREE_PER_BLOCK``); what can remain between
    instances is the order of float atomic accumulations.
    """

    def __init__(self, model, solver, *, copy_back=False):
        self.model, self.solver, self.copy_back = model, solver, copy_back
        self.pipeline = newton.CollisionPipeline(model, deterministic=True)
        self.contacts = self.pipeline.contacts()
        self.control = model.control()
        self.states = [model.state(), model.state()]
        newton.eval_fk(model, self.states[0].joint_q, self.states[0].joint_qd, self.states[0])
        qd = np.zeros(model.joint_dof_count, dtype=np.float32)
        types, starts = model.joint_type.numpy(), model.joint_qd_start.numpy()
        # Swing every third revolute joint so that the caches carry motion.
        for joint in np.flatnonzero(types == int(newton.JointType.REVOLUTE))[::3]:
            qd[starts[joint]] = 1.5
        self.states[0].joint_qd.assign(qd)
        self.initial_q, self.initial_qd = self.states[0].joint_q.numpy().copy(), qd

    def step(self):
        self.pipeline.collide(self.states[0], self.contacts)
        self.states[0].clear_forces()
        self.solver.step(self.states[0], self.states[1], self.control, self.contacts, DT)
        if self.copy_back:
            self.states[0].assign(self.states[1])
        else:
            self.states.reverse()

    def block(self):
        """Two steps: the states return to their objects, as in an even-substep captured graph."""
        self.step()
        self.step()

    def write_root_poses(self):
        """Rewrite the joint frames with their own values and notify, as Isaac Lab's root pose write does."""
        self.model.joint_X_p.assign(self.model.joint_X_p.numpy())
        self.solver.notify_model_changed(JOINT)

    def reset_worlds(self, worlds):
        """Write ``worlds``' initial coordinates back and reset them, as an episode reset does."""
        model, state = self.model, self.states[0]
        joint_world = model.joint_world.numpy()
        coordinates = np.isin(np.repeat(joint_world, np.diff(model.joint_q_start.numpy())), worlds)
        velocities = np.isin(np.repeat(joint_world, np.diff(model.joint_qd_start.numpy())), worlds)
        q, qd = state.joint_q.numpy(), state.joint_qd.numpy()
        q[coordinates] = self.initial_q[coordinates]
        qd[velocities] = self.initial_qd[velocities]
        state.joint_q.assign(q)
        state.joint_qd.assign(qd)
        self.solver.reset(
            state, wp.array(np.isin(np.arange(model.world_count), worlds), dtype=wp.bool, device=model.device)
        )


class TestFeatherPGSJointNotification(unittest.TestCase):
    def test_proven_joint_notification_keeps_held_caches(self):
        """Keep held-tree validity on a joint-property notification the owner proves unchanged.

        Any other cache input invalidates as before, a moved joint frame still raises before any
        change, and without an owner nothing proves the frames, so the FK/ID cache is invalidated.
        """
        model = _build(wp.get_device("cpu"), worlds=2)
        solver = SolverFeatherPGS(model, pgs_mode="split", friction_anchor_beta=0.0)
        owner = MixedTreeKineticState(solver)
        # Install the owner as cached_dynamics does on CUDA; CPU admission keeps ordinary dynamics.
        solver._tree_kinetic_state = owner
        solver._fk_id_cache_enabled = True
        validity_arrays = [solver._fk_id_cache_valid, *(group.geometry_valid for group in owner.groups)]
        invalidators = [cached_state_binding._invalidate_kernel(group._cache_to_art.shape[1]) for group in owner.groups]

        def mark_valid():
            for array in validity_arrays:
                array.fill_(1)

        def validity():
            return [int(array.numpy().min()) for array in validity_arrays]

        def invalidating_launches(flags):
            launched, launch = [], wp.launch

            def record(kernel, *args, **kwargs):
                launched.append(kernel)
                return launch(kernel, *args, **kwargs)

            with patch.object(wp, "launch", record):
                solver.notify_model_changed(flags)
            return [kernel for kernel in launched if any(kernel is other for other in invalidators)]

        valid, invalid = [1] * len(validity_arrays), [0] * len(validity_arrays)
        self.assertTrue(owner.validate_model(JOINT))
        self.assertFalse(owner.validate_model(newton.ModelFlags.JOINT_DOF_PROPERTIES))
        for flags in (JOINT, JOINT | newton.ModelFlags.SHAPE_PROPERTIES, JOINT | newton.ModelFlags.ACTUATOR_PROPERTIES):
            with self.subTest(kept=int(flags)):
                mark_valid()
                self.assertEqual(invalidating_launches(flags), [])
                self.assertEqual(validity(), valid)
        for flags in (
            JOINT | newton.ModelFlags.BODY_INERTIAL_PROPERTIES,
            JOINT | newton.ModelFlags.MODEL_PROPERTIES,
            newton.ModelFlags.JOINT_DOF_PROPERTIES,
            newton.ModelFlags.ALL,
        ):
            with self.subTest(invalidated=int(flags)):
                mark_valid()
                self.assertTrue(invalidating_launches(flags))
                self.assertEqual(validity(), invalid)
        # A moved fixed root raises before any cache changes; restoring it is accepted again.
        mark_valid()
        frames = model.joint_X_p.numpy().copy()
        moved = frames.copy()
        moved[_fixed_root(model, 1), 2] += 0.25
        model.joint_X_p.assign(moved)
        with self.assertRaisesRegex(RuntimeError, "joint_X_p"):
            solver.notify_model_changed(JOINT)
        self.assertEqual(validity(), valid)
        model.joint_X_p.assign(frames)
        self.assertEqual(invalidating_launches(JOINT), [])
        self.assertEqual(validity(), valid)
        ordinary = SolverFeatherPGS(model, pgs_mode="split", friction_anchor_beta=0.0)
        ordinary._fk_id_cache_enabled = True
        ordinary._fk_id_cache_valid.fill_(1)
        ordinary.notify_model_changed(JOINT)
        np.testing.assert_array_equal(ordinary._fk_id_cache_valid.numpy(), 0)

    @unittest.skipUnless(wp.is_cuda_available(), "Held-tree owners are a CUDA path")
    def test_cuda_joint_notification_every_step_repairs_nothing(self):
        """Repair no group when a proven joint notification precedes every step, for every owner kind.

        A finish adds one to each group's generation and a recomputing repair adds another, so a
        steady ping-pong rollout adds exactly one per step. A third rollout invalidates every group
        before each step, as each joint notification did before, and adds two. The notified
        trajectory matches both: the cache a finish publishes is what a repair recomputes from the
        same coordinates.
        """
        device = wp.get_device("cuda:0")
        steps = 6
        for name in ("mixed", "forest", "sparse"):
            with self.subTest(owner=name):
                finals = {}
                for mode, per_step in (("plain", 1), ("notified", 1), ("invalidated", 2)):
                    model, solver = _owned_case(name, device)
                    owner = solver._tree_kinetic_state
                    self.assertIsNotNone(owner, "The fixture must admit a held-tree owner")
                    if name == "forest":
                        self.assertEqual(len(owner.groups), 1, owner.forest_reason)
                        self.assertTrue(owner.groups[0].layout.forest, owner.forest_reason)
                    rollout = _Rollout(model, solver)
                    rollout.step()  # the first repair builds every cache
                    before = _generations(owner)
                    for _ in range(steps):
                        if mode == "notified":
                            solver.notify_model_changed(JOINT)
                        elif mode == "invalidated":
                            owner.invalidate()
                        rollout.step()
                    for start, end in zip(before, _generations(owner), strict=True):
                        np.testing.assert_array_equal(end - start, per_step * steps, err_msg=mode)
                    finals[mode] = {field: getattr(rollout.states[0], field).numpy() for field in STATE_FIELDS}
                for reference in ("plain", "invalidated"):
                    for field in STATE_FIELDS:
                        np.testing.assert_allclose(
                            finals["notified"][field],
                            finals[reference][field],
                            rtol=0.0,
                            atol=2e-6,
                            err_msg=f"{field} against the {reference} rollout",
                        )

    @unittest.skipUnless(wp.is_cuda_available(), "Held-tree owners are a CUDA path")
    def test_cuda_reset_worlds_alone_repair_around_pose_writes(self):
        """Repair only the reset worlds when root pose writes and masked resets precede each block.

        Isaac Lab rewrites the fixed roots, notifies joint properties, writes the reset worlds'
        coordinates and resets those worlds. A captured two-step block replayed after these host
        calls matches eager stepping, and only the reset worlds' groups repair.
        """
        device = wp.get_device("cuda:0")
        reset = (3, 7)
        finals = []
        for captured in (False, True):
            model = _build(device, worlds=8)
            solver = _mixed_solver(model)
            owner = solver._tree_kinetic_state
            self.assertIsNotNone(owner)
            rollout = _Rollout(model, solver)
            rollout.block()
            graph = None
            if captured:
                with wp.ScopedCapture(device=device) as capture:
                    solver.seed_double_buffer_events()
                    rollout.block()
                graph = capture.graph
            worlds = _group_worlds(solver, owner)
            for _ in range(4):
                rollout.write_root_poses()
                rollout.reset_worlds(reset)
                before = _generations(owner)
                if graph is None:
                    rollout.block()
                else:
                    wp.capture_launch(graph)
                for start, end, world in zip(before, _generations(owner), worlds, strict=True):
                    # Two finishes per block, and one repair in each group of a reset world.
                    np.testing.assert_array_equal(end - start, 2 + np.isin(world, reset), err_msg=f"{captured=}")
            finals.append({field: getattr(rollout.states[0], field).numpy().copy() for field in STATE_FIELDS})
        for field in STATE_FIELDS:
            np.testing.assert_allclose(finals[1][field], finals[0][field], rtol=0.0, atol=2e-6, err_msg=field)

    @unittest.skipUnless(wp.is_cuda_available(), "Held-tree owners are a CUDA path")
    def test_cuda_moved_frames_and_other_inputs_keep_their_contract(self):
        """Raise on a moved root and invalidate every group on joint plus inertial notifications."""
        model = _build(wp.get_device("cuda:0"), worlds=4)
        solver = _mixed_solver(model)
        owner = solver._tree_kinetic_state
        self.assertIsNotNone(owner)
        rollout = _Rollout(model, solver)
        rollout.block()
        np.testing.assert_array_equal(solver._fk_id_cache_valid.numpy(), 1)
        frames = model.joint_X_p.numpy().copy()
        moved = frames.copy()
        moved[_fixed_root(model, 2), 0] += 0.1
        model.joint_X_p.assign(moved)
        with self.assertRaisesRegex(RuntimeError, "joint_X_p"):
            solver.notify_model_changed(JOINT)
        np.testing.assert_array_equal(solver._fk_id_cache_valid.numpy(), 1)
        model.joint_X_p.assign(frames)
        solver.notify_model_changed(JOINT)
        np.testing.assert_array_equal(solver._fk_id_cache_valid.numpy(), 1)
        solver.notify_model_changed(JOINT | newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
        np.testing.assert_array_equal(solver._fk_id_cache_valid.numpy(), 0)
        for group in owner.groups:
            np.testing.assert_array_equal(group.geometry_valid.numpy(), 0)


def _recorded_launches(kinds=("launch_tiled",)):
    """Patch the given Warp launch functions to record their kernels and inputs without running them."""
    launched = []

    def record(kernel, *args, inputs=(), **kwargs):
        launched.append((kernel, list(inputs), kwargs))

    stack = contextlib.ExitStack()
    for kind in kinds:
        stack.enter_context(patch.object(wp, kind, record))
    return stack, launched


def _cpu_owner(*, motion_velocity=False):
    """A CPU solver with a mixed held-tree owner installed, as cached_dynamics installs one on CUDA.

    ``motion_velocity`` builds the owner of a velocity-limit solver (whose matrix-free mode CPU
    lacks): its groups read prescaled velocities.
    """
    model = _build(wp.get_device("cpu"), worlds=2)
    solver = SolverFeatherPGS(model, pgs_mode="split", friction_anchor_beta=0.0)
    solver.enable_joint_velocity_limits = motion_velocity
    owner = MixedTreeKineticState(solver)
    solver._tree_kinetic_state = owner
    solver._fk_id_cache_enabled = True
    states = [model.state(), model.state()]
    solver._last_step_dt = DT
    return model, solver, owner, states, solver._prepare_augmented_state(states[0], states[1], model.control())


class TestFeatherPGSInputBank(unittest.TestCase):
    def test_bank_comparing_owner_keeps_validity_on_the_host(self):
        """Skip the host's input-identity invalidation for an owner that compares input banks.

        The step and an episode reset leave current validity to the owner's device comparison
        and the step hands the owner its output bank. An owner that validates by identity alone
        (a motion-velocity layout) is still invalidated when the input is not the last output.
        """
        model, solver, _, states, augmented = _cpu_owner()
        calls = []
        for bank_compare in (True, False):
            with self.subTest(bank_compare=bank_compare):
                solver._tree_kinetic_state = SimpleNamespace(
                    bank_compare=bank_compare,
                    begin=lambda *args, **kwargs: calls.append(kwargs),
                    invalidate=lambda world_mask=None: None,
                    forget_continuity=lambda: None,
                )
                expected = int(bank_compare)
                solver._fk_id_cache_valid.fill_(1)
                solver._fk_id_cache_source_state = states[1]
                solver._stage1_fk_id(states[0], augmented, states[1])
                np.testing.assert_array_equal(solver._fk_id_cache_valid.numpy(), expected)
                self.assertIs(calls[-1].get("state_out"), states[1])
                solver._fk_id_cache_valid.fill_(1)
                solver._fk_id_cache_source_state = states[1]
                none = wp.zeros(model.world_count, dtype=wp.bool, device=model.device)
                solver.reset(states[0], none)
                np.testing.assert_array_equal(solver._fk_id_cache_valid.numpy(), expected)

    def test_comparison_runs_when_the_stamps_name_the_output_bank(self):
        """Compare the banks only when the last finish published into the step's output bank.

        That is the copy-back pattern: the finish wrote ``s1`` and the next step reads ``s0``, a copy
        of it. An unclassified repair compares inside a second one-pass kernel; a ping-pong step
        reads the bank the finish wrote and launches the ordinary repair, and an owner of a
        motion-velocity layout never compares. (A classified repair compares in a pass of its own:
        see :class:`TestFeatherPGSClassifiedRepair`.)
        """
        _, _, owner, (s0, s1), augmented = _cpu_owner()
        self.assertTrue(owner.bank_compare)
        for group in owner.groups:
            self.assertFalse(group.classify_first)
            self.assertIsNone(group.bank_kernel)
            self.assertIsNotNone(group.bank_repair_kernel)
        kinds = {}
        for group in owner.groups:
            kinds.update({group.repair_kernel: "repair", group.bank_repair_kernel: "compare"})
        self.assertEqual(len(kinds), 2 * len(owner.groups))

        def launches(call):
            stack, records = _recorded_launches()
            with stack:
                call()
            return [kinds[kernel] for kernel, _, _ in records if kernel in kinds]

        def begin(state_in, state_out=None):
            return launches(lambda: owner.begin(state_in, augmented, state_in.joint_qd, DT, False, state_out=state_out))

        def finish(state_in, state_out):
            launches(lambda: owner.finish(state_in, augmented, state_out, DT, False))

        repair, compared = ["repair"] * len(owner.groups), ["compare"] * len(owner.groups)
        self.assertEqual(begin(s0, s1), repair)  # nothing published yet
        finish(s0, s1)
        self.assertEqual(begin(s0, s1), compared)  # copy-back: the finish published s1, the step reads s0
        self.assertEqual(begin(s0, s1), repair)  # the repair stamped s0
        finish(s0, s1)
        self.assertEqual(begin(s1, s0), repair)  # ping-pong: the step reads the bank the finish wrote
        finish(s0, s1)
        self.assertEqual(begin(s0), repair)  # without the output bank, identity alone decides
        motion = _cpu_owner(motion_velocity=True)[2]
        self.assertFalse(motion.bank_compare)
        self.assertTrue(all(group.bank_kernel is group.bank_repair_kernel is None for group in motion.groups))

    @unittest.skipUnless(wp.is_cuda_available(), "Held-tree owners are a CUDA path")
    def test_cuda_copied_input_bank_repairs_nothing(self):
        """Keep every group's cache when each step copies its output bank back into its input bank.

        After the first step a steady copy-back rollout adds one generation per step (the finish),
        for every owner kind, whether the repair compares inside its blocks (unclassified, the
        fixtures' size) or in a pass before a classified repair: the comparison finds the copied
        words and the repair exits. A second rollout invalidates every group before each step and
        repairs twice as often. The trajectories agree with each other and with a ping-pong rollout.
        """
        device = wp.get_device("cuda:0")
        steps = 6
        for name, classified in itertools.product(("mixed", "forest", "sparse"), (False, True)):
            with self.subTest(owner=name, classified=classified):
                finals = {}
                for mode, per_step in (("copy_back", 1), ("forced", 2), ("ping_pong", 1)):
                    model, solver, owner = self._owned(name, device)
                    if classified:
                        _classify_every_launch(owner)
                    rollout = _Rollout(model, solver, copy_back=mode != "ping_pong")
                    rollout.step()
                    before = _generations(owner)
                    for _ in range(steps):
                        if mode == "forced":
                            solver._fk_id_cache_valid.zero_()
                        rollout.step()
                    for start, end in zip(before, _generations(owner), strict=True):
                        np.testing.assert_array_equal(end - start, per_step * steps, err_msg=mode)
                    finals[mode] = {field: getattr(rollout.states[0], field).numpy() for field in STATE_FIELDS}
                for reference in ("forced", "ping_pong"):
                    for field in STATE_FIELDS:
                        np.testing.assert_allclose(
                            finals["copy_back"][field],
                            finals[reference][field],
                            rtol=0.0,
                            atol=2e-6,
                            err_msg=f"{field} against the {reference} rollout",
                        )

    @unittest.skipUnless(wp.is_cuda_available(), "Held-tree owners are a CUDA path")
    def test_cuda_written_copy_repairs_its_world(self):
        """Repair exactly the world whose copied input coordinates were written in place without a notification.

        The comparison (inside the one-pass repair or before a classified one) finds the changed
        words, so that world recomputes from the written state, as a forced repair of every world
        does; the other worlds keep their caches.
        """
        device = wp.get_device("cuda:0")
        for name, classified in itertools.product(("mixed", "sparse"), (False, True)):
            with self.subTest(owner=name, classified=classified):
                finals = []
                for forced in (False, True):
                    model, solver, owner = self._owned(name, device)
                    if classified:
                        _classify_every_launch(owner)
                    rollout = _Rollout(model, solver, copy_back=True)
                    rollout.block()
                    # The first revolute joint of world 1: one group owns its articulation.
                    revolute = (model.joint_type.numpy() == int(newton.JointType.REVOLUTE)) & (
                        model.joint_world.numpy() == 1
                    )
                    joint = int(np.flatnonzero(revolute)[0])
                    q = rollout.states[0].joint_q.numpy()
                    q[model.joint_q_start.numpy()[joint]] += np.float32(0.05)
                    rollout.states[0].joint_q.assign(q)
                    if forced:
                        solver._fk_id_cache_valid.zero_()
                    before = _generations(owner)
                    rollout.step()
                    written = model.joint_articulation.numpy()[joint]
                    for start, end, arts in zip(before, _generations(owner), _group_articulations(owner), strict=True):
                        repaired = np.ones(len(arts), dtype=bool) if forced else np.any(arts == written, axis=1)
                        # The finish adds one; a repair adds one more.
                        np.testing.assert_array_equal(end - start, 1 + repaired, err_msg=f"{forced=}")
                    rollout.block()
                    finals.append({field: getattr(rollout.states[0], field).numpy() for field in STATE_FIELDS})
                for field in STATE_FIELDS:
                    np.testing.assert_allclose(finals[0][field], finals[1][field], rtol=0.0, atol=2e-6, err_msg=field)

    @unittest.skipUnless(wp.is_cuda_available(), "Held-tree owners are a CUDA path")
    def test_cuda_captured_copy_back_matches_eager(self):
        """Replay a captured copy-back step (collide, step, copy) and match eager stepping, repairing nothing.

        The comparison is part of the graph, so each replay compares the copied bank again, inside
        the one-pass repair or in a pass before a classified repair.
        """
        device = wp.get_device("cuda:0")
        for classified in (False, True):
            with self.subTest(classified=classified):
                finals = []
                for captured in (False, True):
                    # One step per graph: refresh every step, as every replay does.
                    model, solver, owner = self._owned("sparse", device, update_mass_matrix_interval=1)
                    if classified:
                        _classify_every_launch(owner)
                    rollout = _Rollout(model, solver, copy_back=True)
                    rollout.step()
                    advance = rollout.step
                    if captured:
                        # Capture records the step without running it.
                        with wp.ScopedCapture(device=device) as capture:
                            rollout.step()
                        advance = functools.partial(wp.capture_launch, capture.graph)
                    before = _generations(owner)
                    for _ in range(5):
                        advance()
                    for start, end in zip(before, _generations(owner), strict=True):
                        np.testing.assert_array_equal(end - start, 5, err_msg=f"{captured=}")
                    finals.append({field: getattr(rollout.states[0], field).numpy() for field in STATE_FIELDS})
                for field in STATE_FIELDS:
                    np.testing.assert_allclose(finals[1][field], finals[0][field], rtol=0.0, atol=2e-6, err_msg=field)

    @unittest.skipUnless(wp.is_cuda_available(), "Held-tree owners are a CUDA path")
    def test_cuda_velocity_limit_layouts_validate_by_identity(self):
        """Repair every group after a copy-back under joint velocity limits, whose repair reads prescaled velocities."""
        model = _build(wp.get_device("cuda:0"), worlds=4, speed_limit=0.8)
        solver = _mixed_solver(model, enable_joint_velocity_limits=True)
        owner = solver._tree_kinetic_state
        self.assertIsNotNone(owner)
        self.assertFalse(owner.bank_compare)
        rollout = _Rollout(model, solver, copy_back=True)
        rollout.step()
        before = _generations(owner)
        for _ in range(3):
            rollout.step()
        for start, end in zip(before, _generations(owner), strict=True):
            np.testing.assert_array_equal(end - start, 6)

    @staticmethod
    def _owned(name, device, **options):
        model, solver = _owned_case(name, device, **options)
        owner = solver._tree_kinetic_state
        if owner is None:
            raise AssertionError(f"The {name} fixture must admit a held-tree owner")
        if name == "forest" and not (len(owner.groups) == 1 and owner.groups[0].layout.forest):
            raise AssertionError(owner.forest_reason)
        return model, solver, owner


@contextlib.contextmanager
def _counted_launches(kernels):
    """Run every tiled launch and count, in the yielded list, those of ``kernels``."""
    counted, launch = [0], wp.launch_tiled

    def count(kernel, *args, **kwargs):
        counted[0] += kernel in kernels
        return launch(kernel, *args, **kwargs)

    with patch.object(wp, "launch_tiled", count):
        yield counted


def _repair_kernels(owner):
    """Every kernel that repairs a group's cache: one-pass (with or without a bank comparison) or over a block list."""
    kernels = {group.repair_kernel for group in _groups(owner)}
    for group in _groups(owner):
        kernels |= {group.bank_repair_kernel, group.listed_repair_kernel}
    return kernels - {None}


def _classify_every_launch(owner):
    """Classify every repair launch of a small fixture on a one-block grid.

    A fixture's launches have fewer blocks than the listed grid, which leaves them unclassified; a
    one-block grid classifies them and makes that block stride over every listed block.
    """
    for group in _groups(owner):
        group._repair_grid = 1
        group._bind_kernels()
        if not group.classify_first:
            raise AssertionError("The fixture's repair must be classified on CUDA")


class TestFeatherPGSCapturedRepair(unittest.TestCase):
    def test_repair_after_a_finish_of_the_same_capture_launches_nothing(self):
        """Launch no repair after a finish of the same capture that published the step's input bank.

        The next repair launches when the capture changes or ends, when the step reads another
        bank, when it needs geometry the finish did not compute, after any host call that changes
        validity, requests or the step duration, under ``FEATHER_PGS_TREE_REPAIR_SKIP=0`` and for a
        motion-velocity layout. CPU has no graph capture: launches are recorded, not run, and the
        capture id is stubbed.
        """
        case = _cpu_owner()
        model, solver, owner, _, _ = case
        self.assertTrue(owner.skip_captured_repairs)
        everyone = len(owner.groups)

        def repairs(target, call, capture):
            """The repair launches of ``call`` inside capture ``capture`` (0 outside any capture)."""
            kernels = _repair_kernels(target)
            stack, records = _recorded_launches()
            with stack, patch.object(capture_joins, "capture_id", return_value=capture):
                call()
            return sum(kernel in kernels for kernel, _, _ in records)

        def step(case=case, *, input_bank=1, next_refresh=False, refresh=False, finish_capture=7, capture=7):
            """Finish from bank 0 into bank 1 in ``finish_capture``, then count the repairs of a step from ``input_bank``."""
            _, _, target, banks, augmented = case
            finish = functools.partial(target.finish, banks[0], augmented, banks[1], DT, next_refresh)
            repairs(target, finish, finish_capture)
            return begin(case, input_bank=input_bank, refresh=refresh, capture=capture)

        def begin(case=case, *, input_bank=1, refresh=False, capture=7):
            _, _, target, banks, augmented = case
            state_in, state_out = banks[input_bank], banks[1 - input_bank]
            call = functools.partial(
                target.begin, state_in, augmented, state_in.joint_qd, DT, refresh, state_out=state_out
            )
            return repairs(target, call, capture)

        self.assertEqual(step(), 0)  # the next step of the same capture
        self.assertEqual(begin(), everyone)  # a begin consumes the record
        self.assertEqual(step(refresh=True), everyone)  # the step needs geometry the finish did not compute
        self.assertEqual(step(next_refresh=True, refresh=True), 0)  # the finish computed it
        self.assertEqual(step(capture=8), everyone)  # another capture
        self.assertEqual(step(capture=0), everyone)  # the capture ended
        self.assertEqual(step(finish_capture=0, capture=0), everyone)  # eager steps
        self.assertEqual(step(input_bank=0), everyone)  # another bank (a copy-back compares, then repairs)
        no_worlds = wp.zeros(model.world_count, dtype=wp.bool, device=model.device)
        banks = case[3]
        for name, call in (
            ("reset", lambda: solver.reset(banks[1], no_worlds)),
            ("notify_state_changed", solver.notify_state_changed),
            ("notify_model_changed", lambda: solver.notify_model_changed(JOINT)),
            ("step duration", lambda: solver._track_step_dt(2.0 * DT)),
        ):
            with self.subTest(host_call=name):
                solver._last_step_dt = DT
                self.assertEqual(step(), 0)
                repairs(owner, functools.partial(owner.finish, banks[0], case[4], banks[1], DT, False), 7)
                call()
                self.assertEqual(begin(), everyone)
        with patch.dict(os.environ, {"FEATHER_PGS_TREE_REPAIR_SKIP": "0"}):
            switched = _cpu_owner()
        self.assertFalse(switched[2].skip_captured_repairs)
        self.assertEqual(step(switched), len(switched[2].groups))
        motion = _cpu_owner(motion_velocity=True)
        self.assertFalse(motion[2].skip_captured_repairs)
        self.assertEqual(step(motion), len(motion[2].groups))

    @unittest.skipUnless(wp.is_cuda_available(), "Held-tree owners are a CUDA path")
    def test_cuda_captured_steps_repair_once(self):
        """Launch one repair per group in a captured block of 2 or 8 steps and replay it as eager stepping does.

        The later steps of a block follow a finish of the same capture on the bank it published,
        so their repairs launch nothing. Worlds reset between replays repair through the block's
        first repair, one-pass or classified; every cache generation, geometry generation and
        state matches eager stepping, including the interval-2 geometry refresh cadence.
        """
        device = wp.get_device("cuda:0")
        for name, steps, classified in itertools.product(("mixed", "forest", "sparse"), (2, 8), (False, True)):
            if classified and steps == 8:
                continue
            with self.subTest(owner=name, steps=steps, classified=classified):
                finals = []
                for captured in (False, True):
                    model, solver, owner = TestFeatherPGSInputBank._owned(name, device, update_mass_matrix_interval=2)
                    if classified:
                        _classify_every_launch(owner)
                    rollout = _Rollout(model, solver)
                    rollout.block()

                    def block(rollout=rollout, steps=steps):
                        for _ in range(steps):
                            rollout.step()

                    advance = block
                    if captured:
                        with (
                            _counted_launches(_repair_kernels(owner)) as counted,
                            wp.ScopedCapture(device=device) as capture,
                        ):
                            solver.seed_double_buffer_events()
                            block()
                        self.assertEqual(counted[0], len(_groups(owner)))
                        advance = functools.partial(wp.capture_launch, capture.graph)
                    for _ in range(3):
                        rollout.reset_worlds((1,))
                        advance()
                    finals.append(
                        {
                            **{field: getattr(rollout.states[0], field).numpy() for field in STATE_FIELDS},
                            "generation": _generations(owner),
                            "geometry_generation": [group.data.geometry_generation.numpy() for group in _groups(owner)],
                            "geometric": [group.geometric.numpy() for group in _groups(owner)],
                        }
                    )
                for key in ("generation", "geometry_generation"):
                    for eager, replayed in zip(finals[0][key], finals[1][key], strict=True):
                        np.testing.assert_array_equal(replayed, eager, err_msg=key)
                for eager, replayed in zip(finals[0]["geometric"], finals[1]["geometric"], strict=True):
                    np.testing.assert_allclose(replayed, eager, rtol=0.0, atol=2e-6, err_msg="geometric")
                for field in STATE_FIELDS:
                    np.testing.assert_allclose(finals[1][field], finals[0][field], rtol=0.0, atol=2e-6, err_msg=field)

    @unittest.skipUnless(wp.is_cuda_available(), "Held-tree owners are a CUDA path")
    def test_cuda_host_calls_between_captured_steps_keep_the_repair(self):
        """Launch the second step's repair when a reset, a state notification or a new step duration falls between."""
        device = wp.get_device("cuda:0")
        for hook in ("reset", "notify_state_changed", "step duration"):
            with self.subTest(host_call=hook):
                finals = []
                for captured in (False, True):
                    model, solver, owner = TestFeatherPGSInputBank._owned("mixed", device)
                    rollout = _Rollout(model, solver)
                    rollout.block()
                    no_worlds = wp.zeros(model.world_count, dtype=wp.bool, device=device)

                    def block(rollout=rollout, solver=solver, hook=hook, no_worlds=no_worlds):
                        rollout.step()
                        if hook == "reset":
                            solver.reset(rollout.states[0], no_worlds)
                        elif hook == "notify_state_changed":
                            solver.notify_state_changed()
                        else:
                            solver._track_step_dt(1.5 * DT)
                        rollout.step()

                    advance = block
                    if captured:
                        with (
                            _counted_launches(_repair_kernels(owner)) as counted,
                            wp.ScopedCapture(device=device) as capture,
                        ):
                            solver.seed_double_buffer_events()
                            block()
                        self.assertEqual(counted[0], 2 * len(_groups(owner)))
                        advance = functools.partial(wp.capture_launch, capture.graph)
                    for _ in range(3):
                        advance()
                    finals.append({field: getattr(rollout.states[0], field).numpy() for field in STATE_FIELDS})
                for field in STATE_FIELDS:
                    np.testing.assert_allclose(finals[1][field], finals[0][field], rtol=0.0, atol=2e-6, err_msg=field)

    @unittest.skipUnless(wp.is_cuda_available(), "Held-tree owners are a CUDA path")
    def test_cuda_motion_layouts_and_separate_captures_repair_every_time(self):
        """Keep every captured repair of a velocity-limit layout and the first repair of each separate capture.

        Two single-step graphs captured one after the other, replayed with an eager reset between
        them, each repair, and match eager stepping.
        """
        device = wp.get_device("cuda:0")
        model = _build(device, worlds=4, speed_limit=0.8)
        solver = _mixed_solver(model, enable_joint_velocity_limits=True)
        owner = solver._tree_kinetic_state
        self.assertFalse(owner.skip_captured_repairs)
        rollout = _Rollout(model, solver)
        rollout.block()
        with _counted_launches(_repair_kernels(owner)) as counted, wp.ScopedCapture(device=device):
            solver.seed_double_buffer_events()
            rollout.block()
        self.assertEqual(counted[0], 2 * len(owner.groups))

        def replay(rollout, graph):
            wp.capture_launch(graph)
            rollout.states.reverse()  # as the captured step did

        finals = []
        for captured in (False, True):
            # Two single-step graphs alternate: refresh every other step, as their replays do.
            model, solver, owner = TestFeatherPGSInputBank._owned("sparse", device, update_mass_matrix_interval=2)
            rollout = _Rollout(model, solver)
            rollout.block()
            steps = [rollout.step, rollout.step]
            if captured:
                steps = []
                for _ in range(2):
                    with (
                        _counted_launches(_repair_kernels(owner)) as counted,
                        wp.ScopedCapture(device=device) as capture,
                    ):
                        rollout.step()
                    self.assertEqual(counted[0], len(_groups(owner)))
                    steps.append(functools.partial(replay, rollout, capture.graph))
            for _ in range(3):
                for world, advance in enumerate(steps):
                    rollout.reset_worlds((world,))
                    advance()
            finals.append({field: getattr(rollout.states[0], field).numpy() for field in STATE_FIELDS})
        for field in STATE_FIELDS:
            np.testing.assert_allclose(finals[1][field], finals[0][field], rtol=0.0, atol=2e-6, err_msg=field)


def _repair_arrays(group, data):
    """Every array a group's repair reads or writes: its publication operands and its cache, by name."""
    arrays = {}
    for prefix, struct in (("data", data), ("cache", group.data)):
        for name in struct._cls.vars:
            value = getattr(struct, name)
            if isinstance(value, wp.array) and name != "stage_clocks":
                arrays[f"{prefix}.{name}"] = value
    return arrays


class TestFeatherPGSClassifiedRepair(unittest.TestCase):
    def test_repair_lists_its_blocks_when_the_groups_should_be_current(self):
        """Classify and repair over the block list when the stamps should name the input bank; else launch every block.

        The list is expected after a finish that published the input bank and after a bank
        comparison, which then runs as a pass of its own. The first step, and a step from a bank no
        finish published without a comparison, launch the one-pass repair. Only a launch of more
        blocks than the listed grid is classified, and CPU keeps the one-pass repair: the test
        switches the CPU groups on with a one-block grid. Launches are recorded, not run.
        """
        _, _, owner, (s0, s1), augmented = _cpu_owner()
        for group in owner.groups:
            self.assertFalse(group.classify_first)
            self.assertIsNone(group.classify_kernel)
            group._classify_switch = True
            group._repair_grid = group.launch_count
            group._bind_kernels()
            self.assertFalse(group.classify_first)  # not more blocks than the grid
            group._repair_grid = 1
            group._bind_kernels()
            self.assertTrue(group.classify_first)
            self.assertIsNone(group.bank_repair_kernel)
        kinds = {}
        for group in owner.groups:
            kinds.update(
                {
                    group.repair_kernel: "repair",
                    group.bank_kernel: "bank",
                    group.classify_kernel: "classify",
                    group.listed_repair_kernel: "listed",
                }
            )

        def launches(call):
            stack, records = _recorded_launches(("launch", "launch_tiled"))
            with stack:
                call()
            listed = [kwargs.get("max_blocks") for kernel, _, kwargs in records if kinds.get(kernel) == "listed"]
            self.assertTrue(all(blocks == owner.groups[0]._repair_grid for blocks in listed))
            return [kinds[kernel] for kernel, _, _ in records if kernel in kinds]

        def begin(state_in, state_out=None):
            return launches(lambda: owner.begin(state_in, augmented, state_in.joint_qd, DT, False, state_out=state_out))

        def finish():
            launches(lambda: owner.finish(s0, augmented, s1, DT, False))

        groups = len(owner.groups)
        self.assertEqual(begin(s0, s1), ["repair"] * groups)  # nothing published yet
        finish()
        self.assertEqual(begin(s1, s0), ["classify", "listed"] * groups)  # the finish published the input bank
        finish()
        self.assertEqual(begin(s0, s1), ["bank", "classify", "listed"] * groups)  # a copy, compared first
        finish()
        self.assertEqual(begin(s0), ["repair"] * groups)  # another bank, no comparison
        with patch.dict(os.environ, {"FEATHER_PGS_TREE_REPAIR_CLASSIFY": "0"}):
            self.assertEqual(tree_kinetic_owner.launch_switches(), (True, False))
        self.assertEqual(tree_kinetic_owner.launch_switches(), (True, True))

    @unittest.skipUnless(wp.is_cuda_available(), "Held-tree owners are a CUDA path")
    def test_cuda_classified_repair_matches_the_one_pass_repair(self):
        """Compute bitwise the same caches and publication with the classified repair as with one block per group.

        From one device state with random invalid, requested, geometrically stale, foreign-stamped
        and (velocity-limit layouts) motion-changed groups, the one-pass repair and the classified
        repair write identical words to every array the repair reaches, for packed per-size groups,
        a packed forest, chain-scan sparse trees and a velocity-limit layout. The blocks the
        classification lists are exactly those with a group the one-pass repair recomputed.
        """
        device = wp.get_device("cuda:0")
        rng = np.random.default_rng(20261019)
        # Eight lanes per tree: four trees share each block of the per-size groups and of the forest.
        cases = {
            "mixed": lambda: _mixed_solver(_build(device, worlds=8), lanes=8),
            "forest": lambda: _mixed_solver(_build(device, worlds=8, prescribed=True), forest=True, lanes=8),
            "sparse": lambda: _owned_case("sparse", device)[1],
            "motion": lambda: _mixed_solver(
                _build(device, worlds=8, speed_limit=0.8), lanes=8, enable_joint_velocity_limits=True
            ),
        }
        for name, build in cases.items():
            with self.subTest(owner=name):
                solver = build()
                owner = solver._tree_kinetic_state
                self.assertIsNotNone(owner)
                # On a one-block grid, which also makes that block stride over every listed block.
                _classify_every_launch(owner)
                packs = [group.layout.pack for group in _groups(owner)]
                self.assertTrue(all(pack == 4 for pack in packs) if name != "sparse" else packs == [1], packs)
                self.assertEqual(name == "sparse", _groups(owner)[0].chain_scan)
                rollout = _Rollout(solver.model, solver)
                rollout.block()
                state = rollout.states[0]
                for seed in range(4):
                    motion = wp.clone(state.joint_qd)
                    data = _groups(owner)[0]._publication(state, solver, state, DT)
                    arrays = {}
                    for index, group in enumerate(_groups(owner)):
                        arrays.update({f"{index}.{key}": value for key, value in _repair_arrays(group, data).items()})
                    arrays["requests"] = solver._mass_update_requested
                    self._scramble(rng, owner, solver, state, motion, seed)
                    start = {key: value.numpy().copy() for key, value in arrays.items()}
                    results = []
                    for classify in (False, True):
                        for key, value in arrays.items():
                            value.assign(start[key])
                        for group in _groups(owner):
                            group.classify_first = classify
                            group._published = (state.joint_q.ptr, state.joint_qd.ptr)
                            group._continuity = None
                        velocity = motion if _groups(owner)[0].motion_velocity else state.joint_qd
                        owner.begin(state, solver, velocity, DT, bool(seed % 2))
                        results.append({key: value.numpy().copy() for key, value in arrays.items()})
                        if classify:
                            listed = [
                                np.sort(group._repair_blocks.numpy()[: int(group._repair_count.numpy()[0])])
                                for group in _groups(owner)
                            ]
                    for key in arrays:
                        np.testing.assert_array_equal(results[1][key], results[0][key], err_msg=f"{name} {seed} {key}")
                    for index, (group, blocks) in enumerate(zip(_groups(owner), listed, strict=True)):
                        generation = f"{index}.cache.generation"
                        repaired = results[0][generation] != start[generation]
                        expected = np.flatnonzero(repaired.reshape(-1, group.layout.pack).any(axis=1))
                        np.testing.assert_array_equal(blocks, expected, err_msg=f"{name} {seed} group {index}")
                    for group in _groups(owner):
                        group.classify_first = True

    @staticmethod
    def _scramble(rng, owner, solver, state, motion, seed):
        """Invalidate, request, stale, foreign-stamp and (motion layouts) motion-change random groups."""
        arts = solver.model.articulation_count
        valid = solver._fk_id_cache_valid.numpy()
        valid[rng.random(arts) < 0.3] = 0
        solver._fk_id_cache_valid.assign(valid)
        solver._mass_update_requested.assign((rng.random(arts) < 0.2).astype(np.int32))
        for group in _groups(owner):
            count = group.group_count
            geometry = group.geometry_valid.numpy()
            geometry[rng.random(count) < 0.2] = 0
            group.geometry_valid.assign(geometry)
            stamps = group.data.source_q.numpy()
            stamps[rng.random(count) < 0.2] = 12345
            group.data.source_q.assign(stamps)
        if _groups(owner)[0].motion_velocity:
            qd = motion.numpy()
            changed = rng.random(qd.size) < 0.1
            qd[changed] = qd[changed] * 0.5 + 0.01 * (seed + 1)
            motion.assign(qd)


def _mass_matrices(solver):
    """Both double-buffer slots' dense mass matrices, by slot and DOF size."""
    return [(slot, size, array) for slot, buffers in enumerate(solver._H_bufs) for size, array in buffers.items()]


class TestFeatherPGSHeldMassMatrix(unittest.TestCase):
    def test_unread_sizes_follow_the_factor_classifier(self):
        """Report exactly the sizes stage 2 factors without reading a dense mass matrix, under an owner.

        ``_stage2_factor`` routes a compact size and the sizes with a fused CRBA-Cholesky kernel to no
        factorization, and a diagonal size and a tiled or loop size without its fused kernel to one
        that reads H. ``_unread_mass_matrix_sizes`` lists the first kind.
        """
        sizes = (2, 4, 6, 18, 24, 25)
        fused_tiled = {24}
        fused_warp = {6, 25}

        def opaque():
            return {size: object() for size in sizes}

        solver = SimpleNamespace(
            _tree_kinetic_state=object(),
            size_groups=list(sizes),
            _compact_diagonal_mass_size=2,
            _sparse_mass_matrix_size=None,
            _execution_plan=SimpleNamespace(
                use_diagonal_mass=lambda size: size == 4, use_tiled_cholesky=lambda size: size >= 20
            ),
            _crba_cholesky_kernels_by_size={size: object() if size in fused_tiled else None for size in sizes},
            _crba_cholesky_warp_kernels_by_size={size: object() if size in fused_warp else None for size in sizes},
            _cholesky_kernels_by_size=opaque(),
            _paired_response_primary_size=None,
            _paired_factor_coordinates=False,
            model=SimpleNamespace(device="cpu"),
            n_arts_by_size=dict.fromkeys(sizes, 3),
            tile_threads=64,
            mass_update_mask=object(),
            H_by_size=opaque(),
            R_by_size=opaque(),
            L_by_size=opaque(),
            group_to_art=opaque(),
        )
        for name in ("_stage2_factor", "_stage2_cholesky_diagonal", "_stage2_cholesky_tiled", "_stage2_cholesky_loop"):
            setattr(solver, name, functools.partial(getattr(SolverFeatherPGS, name), solver))
        factors = {size: SolverFeatherPGS._stage2_factor(solver, size) for size in sizes}
        self.assertEqual(
            {size: None if factor is None else factor.func.__name__ for size, factor in factors.items()},
            {
                2: None,
                4: "_stage2_cholesky_diagonal",
                6: None,
                18: "_stage2_cholesky_loop",
                24: None,
                25: "_stage2_cholesky_tiled",
            },
        )
        self.assertEqual(SolverFeatherPGS._unread_mass_matrix_sizes(solver), (2, 6, 24))
        launched = []

        def record(kernel, *args, inputs=(), **kwargs):
            launched.extend(inputs)

        with patch.object(wp, "launch", record), patch.object(wp, "launch_tiled", record):
            for size, factor in factors.items():
                if factor is not None:
                    launched.clear()
                    factor(size)
                    self.assertTrue(any(item is solver.H_by_size[size] for item in launched), f"size {size}")
        solver._tree_kinetic_state = None
        self.assertEqual(SolverFeatherPGS._unread_mass_matrix_sizes(solver), ())

    def test_step_factors_every_size_through_the_classifier(self):
        """Dispatch each size's stage-2 factorization in ``step`` through ``_stage2_factor``.

        ``_unread_mass_matrix_sizes`` reads the same classifier, so the sizes whose clear it skips
        cannot drift from the factorizations a step runs.
        """
        model = _build(wp.get_device("cpu"), worlds=2)
        solver = SolverFeatherPGS(model, pgs_mode="split", friction_anchor_beta=0.0)
        state_in, state_out = model.state(), model.state()
        newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
        classify = SolverFeatherPGS._stage2_factor
        with patch.object(SolverFeatherPGS, "_stage2_factor", autospec=True, side_effect=classify) as recorded:
            solver.step(state_in, state_out, model.control(), None, DT)
        self.assertEqual([call.args[1] for call in recorded.call_args_list], list(solver.size_groups))
        self.assertTrue(all(call.args[0] is solver for call in recorded.call_args_list))

    def test_owner_stage1_writes_no_mass_matrix(self):
        """Leave every dense mass matrix untouched in stage 1 while a held-tree owner refreshes the factors.

        No other stage writes H, so under an owner H keeps its allocation zeros and the end-of-step
        clear its sizes skip could not change a value even with a reader. Without the owner the
        same stage clears and fills H.
        """
        model = _build(wp.get_device("cpu"), worlds=2)
        solver = SolverFeatherPGS(model, pgs_mode="split", friction_anchor_beta=0.0)
        state_in = model.state()
        newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
        state_aug = solver._prepare_augmented_state(state_in, model.state(), model.control())
        matrices = list(solver.H_by_size.values())
        self.assertTrue(matrices)

        def touched_matrices():
            """Run stage 1 with launches and clears recorded, not executed; return the H arrays they touch."""
            touched = []

            def launch(kernel, *args, inputs=(), outputs=(), **kwargs):
                touched.extend([*inputs, *outputs])

            def zero(array):
                touched.append(array)

            with (
                patch.object(wp, "launch", launch),
                patch.object(wp, "launch_tiled", launch),
                patch.object(wp.array, "zero_", zero),
            ):
                solver._stage1_crba(state_aug, None, None)
            return [array for array in matrices if any(item is array for item in touched)]

        refreshed = []
        solver._tree_kinetic_state = SimpleNamespace(sparse=SimpleNamespace(refresh=refreshed.append))
        self.assertEqual(touched_matrices(), [])
        self.assertEqual(refreshed, [state_aug])
        solver._tree_kinetic_state = None
        self.assertEqual(len(touched_matrices()), len(matrices))

    def test_clear_record_names_the_double_buffer(self):
        """Record no double-buffer mass matrix clear on a route without a double buffer.

        Stage 1 of the ordinary route clears H there, so an empty list must not read as "no clears".
        """
        model = _build(wp.get_device("cpu"), worlds=2)
        solver = SolverFeatherPGS(model, pgs_mode="split", friction_anchor_beta=0.0)
        self.assertIsNone(solver._memset_stream)
        dynamics = solver.get_path_selection()["routes"]["dynamics"]
        self.assertEqual(dynamics["double_buffer_mass_matrix_clear_dofs"], [])
        self.assertNotIn("mass_matrix_clear_dofs", dynamics)

    @unittest.skipUnless(wp.is_cuda_available(), "The double-buffered mass matrix is a CUDA path")
    def test_cuda_held_owner_clears_no_mass_matrix(self):
        """Issue no end-of-step mass matrix clear under a held-tree owner, and every clear without one."""
        device = wp.get_device("cuda:0")
        for owner in (True, False):
            with self.subTest(owner=owner):
                model = _build(device, worlds=4)
                solver = _mixed_solver(model, owner=owner)
                self.assertEqual(solver._tree_kinetic_state is not None, owner)
                self.assertIsNotNone(solver._memset_stream)
                record = solver.get_path_selection()["routes"]["dynamics"]["double_buffer_mass_matrix_clear_dofs"]
                self.assertEqual(record, [] if owner else sorted(int(size) for size in solver.size_groups))
                matrices = _mass_matrices(solver)
                rollout = _Rollout(model, solver)
                zeroed, zero = [], wp.array.zero_

                def recording(array, zeroed=zeroed, zero=zero):
                    zeroed.append(array)
                    return zero(array)

                with patch.object(wp.array, "zero_", recording):
                    for _ in range(6):
                        rollout.step()
                cleared = {(slot, size) for slot, size, array in matrices if any(array is z for z in zeroed)}
                # With update_mass_matrix_interval=2 the refresh steps use slot 0 alone.
                expected = set() if owner else {(0, int(size)) for size in solver.size_groups}
                self.assertEqual(cleared, expected)

    @unittest.skipUnless(wp.is_cuda_available(), "The double-buffered mass matrix is a CUDA path")
    def test_cuda_poisoned_mass_matrix_is_never_read(self):
        """Step identically with both mass matrix slots poisoned by NaN under a held-tree owner.

        The poison is never cleared and never reaches the state, so no stage reads H.
        """
        device = wp.get_device("cuda:0")
        finals = []
        for poisoned in (False, True):
            model = _build(device, worlds=4)
            solver = _mixed_solver(model)
            self.assertIsNotNone(solver._tree_kinetic_state)
            if poisoned:
                for _, _, array in _mass_matrices(solver):
                    array.fill_(float("nan"))
            rollout = _Rollout(model, solver)
            for _ in range(20):
                rollout.step()
            if poisoned:
                for slot, size, array in _mass_matrices(solver):
                    self.assertTrue(np.isnan(array.numpy()).all(), f"slot {slot}, size {size}")
            finals.append({field: getattr(rollout.states[0], field).numpy() for field in STATE_FIELDS})
        for field in STATE_FIELDS:
            self.assertTrue(np.isfinite(finals[1][field]).all(), field)
            np.testing.assert_allclose(finals[1][field], finals[0][field], rtol=0.0, atol=2e-6, err_msg=field)


if __name__ == "__main__":
    unittest.main()
