# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Keep held-tree caches valid across host calls and state copies that change none of their inputs."""

import collections
import contextlib
import dataclasses
import functools
import itertools
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import (
    cached_state_binding,
    tree_kinetic_owner,
    tree_kinetic_state,
)
from newton._src.solvers.feather_pgs.factor.kernels import Clock
from newton._src.solvers.feather_pgs.held_tree import HeldTree
from newton._src.solvers.feather_pgs.solver_feather_pgs import SolverFeatherPGS
from newton.tests.test_feather_pgs_mixed_tree_free_body import _build
from newton.tests.test_feather_pgs_sparse_solver import STATE_FIELDS, _build_model, _make_solver

DT = 1.0 / 240.0
JOINT = newton.ModelFlags.JOINT_PROPERTIES


def fixed_lanes(lanes):
    """Pin every held-tree group's lane width (``tree_kinetic_state.held_lanes``), for packed layouts on trees
    whose data would choose another width."""
    return patch.object(tree_kinetic_state, "held_lanes", lambda tree, groups, device: lanes)


def _one_tree_per_block(build, *args, **kwargs):
    """``build(*args, **kwargs)`` with every held-tree group at 32 lanes: one tree per block, chain scans."""
    with fixed_lanes(32):
        return build(*args, **kwargs)


def _set_clock(factor, refresh, ahead=0):
    """Set the factor's device clock so the step ``ahead`` steps on (0: a repair's, 1: a finish's next) refreshes
    the mass, or not, at an unchanged step duration (:class:`.factor.kernels.Clock`)."""
    clock = Clock()
    clock.step, clock.dt, clock.prev_dt, clock.interval, clock.cadence = 1 + ahead, DT, DT, 2, int(refresh)
    factor.clock.assign([clock])


def _groups(owner):
    """The owner's cache groups: a mixed owner composes several, a sparse tree owner is one."""
    return list(owner.groups) if hasattr(owner, "groups") else [owner]


def _generations(owner):
    """Per-group cache generations: every finish and every recomputing repair adds one."""
    return [group.data.generation.numpy().copy() for group in _groups(owner)]


def _group_worlds(solver, owner):
    """The world of every cache group, aligned with :func:`_generations`."""
    art_to_world = solver.forest.art_to_world.numpy()
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


def _mixed_solver(model, *, forest=False, lanes=None, **options):
    """A dense-route solver with the mixed held-tree owner.

    ``lanes`` fixes the held-tree lane width (several small trees per block); by default the trees' data chooses.
    """
    pins = {"storage": "dense", "grouping": "forest" if forest else "per_size"}
    settings = {
        "pgs_iterations": 8,
        "enable_joint_limits": True,
        "update_mass_matrix_interval": 2,
        "friction_anchor_beta": 0.0,
        **options,
    }
    with (
        patch.object(SolverFeatherPGS, "_route_pins", pins),
        fixed_lanes(lanes) if lanes is not None else contextlib.nullcontext(),
    ):
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
    return model, _make_solver(model, True, **{"friction_anchor_beta": 0.0, **options})


class _Rollout:
    """Ping-pong one solver's states over sorted contacts, as a captured Isaac Lab physics step does.

    With ``copy_back`` the input bank stays the same object and each step copies the output bank
    back into it, as Isaac Lab does after an odd substep count. Rollouts of separate solver
    instances compare within 2e-6 (rtol 0), the bound the repository's other cross-instance tests
    use. Their held-tree layouts follow from the trees' data alone; what can remain between
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

        Any other cache input invalidates as before. A moved joint frame invalidates every cache
        without raising, and the moved frame is the proof's reference from then on: a later
        notification that moves nothing keeps the caches, and restoring the frame moves it again.
        A changed joint axis raises before any change, and without an owner nothing proves the
        frames, so the FK/ID cache is invalidated.
        """
        model = _build(wp.get_device("cpu"), worlds=2)
        solver = SolverFeatherPGS(model, friction_anchor_beta=0.0)
        owner = solver.factor.held
        self.assertIsInstance(owner, HeldTree)
        validity_arrays = [solver.factor.fk_id_cache_valid, *(group.geometry_valid for group in owner.groups)]
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
        # A moved fixed root invalidates every cache; the next notification that moves nothing keeps
        # them, and restoring the root moves it again.
        frames = model.joint_X_p.numpy().copy()
        moved = frames.copy()
        moved[_fixed_root(model, 1), 2] += 0.25
        for written, kept in ((moved, False), (moved, True), (frames, False), (frames, True)):
            mark_valid()
            model.joint_X_p.assign(written)
            launched = invalidating_launches(JOINT)
            self.assertEqual(bool(launched), not kept)
            self.assertEqual(validity(), valid if kept else invalid)
        # A move announced under another non-numeric flag invalidates as well: the proof that
        # stores the moved frame must not leave caches built from the old one.
        actuator = newton.ModelFlags.ACTUATOR_PROPERTIES
        for written, kept in ((moved, False), (moved, True), (frames, False)):
            mark_valid()
            model.joint_X_p.assign(written)
            self.assertEqual(bool(invalidating_launches(actuator)), not kept)
            self.assertEqual(validity(), valid if kept else invalid)
        # A changed joint axis raises before any cache changes.
        mark_valid()
        axes = model.joint_axis.numpy().copy()
        turned = axes.copy()
        turned[model.joint_qd_start.numpy()[_fixed_root(model, 1) + 1]] = (1.0, 0.0, 0.0)
        model.joint_axis.assign(turned)
        with self.assertRaisesRegex(RuntimeError, "joint_axis"):
            solver.notify_model_changed(JOINT)
        self.assertEqual(validity(), valid)
        model.joint_axis.assign(axes)
        self.assertEqual(invalidating_launches(JOINT), [])

    def test_refused_notification_stores_no_moved_frame(self):
        """Store no moved frame in the held-tree proof when a row owner refuses the notification.

        The proof stores a moved frame as its new reference, so it runs after every check that can
        still refuse the notification. A refusal changes no cache, and the next notification still
        finds the frame moved and invalidates every cache instead of keeping caches built from the
        old frame.
        """
        model = _build(wp.get_device("cpu"), worlds=2)
        solver = SolverFeatherPGS(model, friction_anchor_beta=0.0)
        owner = solver.factor.held
        self.assertIsInstance(owner, HeldTree)
        validity_arrays = [solver.factor.fk_id_cache_valid, *(group.geometry_valid for group in owner.groups)]

        def refuse(flags):
            raise RuntimeError("Row topology changed; reconstruct the solver")

        refusal = SimpleNamespace(validate_notification=refuse)
        frames = model.joint_X_p.numpy().copy()
        moved = frames.copy()
        moved[_fixed_root(model, 1), 2] += 0.25
        # Move the root: a move the refused notification must not store.
        for array in validity_arrays:
            array.fill_(1)
        model.joint_X_p.assign(moved)
        with (
            patch.object(solver.placement, "keyed_rows", refusal),
            self.assertRaisesRegex(RuntimeError, "Row topology"),
        ):
            solver.notify_model_changed(JOINT)
        self.assertEqual([int(array.numpy().min()) for array in validity_arrays], [1] * len(validity_arrays))
        solver.notify_model_changed(JOINT)
        self.assertEqual([int(array.numpy().max()) for array in validity_arrays], [0] * len(validity_arrays))

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
                    owner = solver.factor.held
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
            owner = solver.factor.held
            self.assertIsNotNone(owner)
            rollout = _Rollout(model, solver)
            rollout.block()
            graph = None
            if captured:
                with wp.ScopedCapture(device=device) as capture:
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
        """Invalidate every group on a moved root and on joint plus inertial notifications; keep unmoved frames.

        The moved root's welded base takes its new pose on the next step, and a notification that
        moves nothing afterwards keeps every cache.
        """
        model = _build(wp.get_device("cuda:0"), worlds=4)
        solver = _mixed_solver(model)
        owner = solver.factor.held
        self.assertIsNotNone(owner)
        rollout = _Rollout(model, solver)
        rollout.block()
        np.testing.assert_array_equal(solver.factor.fk_id_cache_valid.numpy(), 1)
        root = _fixed_root(model, 2)
        base = int(model.joint_child.numpy()[root])
        before = rollout.states[0].body_q.numpy()[base].copy()
        moved = model.joint_X_p.numpy().copy()
        moved[root, 0] += 0.1
        model.joint_X_p.assign(moved)
        solver.notify_model_changed(JOINT)
        np.testing.assert_array_equal(solver.factor.fk_id_cache_valid.numpy(), 0)
        for group in owner.groups:
            np.testing.assert_array_equal(group.geometry_valid.numpy(), 0)
        rollout.block()
        np.testing.assert_array_equal(solver.factor.fk_id_cache_valid.numpy(), 1)
        shift = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        np.testing.assert_allclose(rollout.states[0].body_q.numpy()[base], before + shift, atol=1.0e-6)
        solver.notify_model_changed(JOINT)
        np.testing.assert_array_equal(solver.factor.fk_id_cache_valid.numpy(), 1)
        solver.notify_model_changed(JOINT | newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
        np.testing.assert_array_equal(solver.factor.fk_id_cache_valid.numpy(), 0)
        for group in owner.groups:
            np.testing.assert_array_equal(group.geometry_valid.numpy(), 0)


def _input_mismatches(kernel, inputs):
    """Name each parameter of ``kernel`` whose launch input has another type, or both counts when they differ.

    Warp checks an array input's dtype and dimensions at launch but packs whatever it is given for a
    struct parameter, so a struct, or an array, in another struct's place would launch unchecked.
    """
    parameters = kernel.adj.args
    if len(inputs) != len(parameters):
        return [f"{len(inputs)} inputs for {len(parameters)} parameters"]
    mismatched = []
    for parameter, value in zip(parameters, inputs, strict=True):
        if wp.types.is_array(parameter.type):
            matches = (
                isinstance(value, wp.array)
                and wp.types.types_equal(value.dtype, parameter.type.dtype)
                and value.ndim == parameter.type.ndim
            )
        elif wp.types.type_is_struct(parameter.type):
            matches = getattr(value, "_cls", None) is parameter.type
        elif wp.types.type_is_int(parameter.type):
            # Warp converts a Python integer to the parameter's integer type.
            matches = isinstance(value, (int, parameter.type))
        elif wp.types.type_is_float(parameter.type):
            matches = isinstance(value, (float, parameter.type))
        else:
            matches = isinstance(value, parameter.type)
        if not matches:
            mismatched.append(parameter.label)
    return mismatched


def _cpu_owner(*, motion_velocity=False, worlds=2, pins=None):
    """A CPU solver with a mixed held-tree owner installed, as cached_dynamics installs one on CUDA.

    ``motion_velocity`` builds the owner of a velocity-limit solver (whose matrix-free mode CPU
    lacks): its groups read prescaled velocities. Each group holds one tree per world.
    """
    model = _build(wp.get_device("cpu"), worlds=worlds)
    solver = SolverFeatherPGS(model, friction_anchor_beta=0.0)
    solver.options = solver.laws.options = solver.factor.options = solver.placement.options = solver.sweep.options = (
        dataclasses.replace(solver.options, enable_joint_velocity_limits=motion_velocity)
    )
    solver._route_pins.update(pins or {})
    owner = HeldTree(solver)
    solver.factor.held = owner
    states = [model.state(), model.state()]
    return model, solver, owner, states, solver.factor


class TestFeatherPGSInputBank(unittest.TestCase):
    def test_bank_comparing_owner_keeps_validity_on_the_host(self):
        """Skip the host's input-identity invalidation for an owner that compares input banks.

        The step and an episode reset leave current validity to the owner's device comparison
        and the step hands the owner its output bank. An owner that validates by identity alone
        (a motion-velocity layout) is still invalidated when the input is not the last output.
        """
        model, solver, _, states, _augmented = _cpu_owner()
        calls = []
        for bank_compare in (True, False):
            with self.subTest(bank_compare=bank_compare):
                solver.factor.held = SimpleNamespace(
                    bank_compare=bank_compare,
                    begin=lambda *args, **kwargs: calls.append(kwargs),
                    invalidate=lambda world_mask=None: None,
                )
                expected = int(bank_compare)
                solver.factor.fk_id_cache_valid.fill_(1)
                solver.factor.fk_id_cache_source_state = states[1]
                solver.factor.fk_id(states[0], states[1], states[0].joint_qd, DT)
                np.testing.assert_array_equal(solver.factor.fk_id_cache_valid.numpy(), expected)
                self.assertIs(calls[-1].get("state_out"), states[1])
                solver.factor.fk_id_cache_valid.fill_(1)
                solver.factor.fk_id_cache_source_state = states[1]
                none = wp.zeros(model.world_count, dtype=wp.bool, device=model.device)
                solver.reset(states[0], none)
                np.testing.assert_array_equal(solver.factor.fk_id_cache_valid.numpy(), expected)

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
                            solver.factor.fk_id_cache_valid.zero_()
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
                        solver.factor.fk_id_cache_valid.zero_()
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
        owner = solver.factor.held
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
        owner = solver.factor.held
        if owner is None:
            raise AssertionError(f"The {name} fixture must admit a held-tree owner")
        if name == "forest" and not (len(owner.groups) == 1 and owner.groups[0].layout.forest):
            raise AssertionError(owner.forest_reason)
        return model, solver, owner


@contextlib.contextmanager
def _cleared_arrays():
    """Run every ``wp.array.zero_`` and record, in the yielded list, the arrays it cleared."""
    cleared, zero = [], wp.array.zero_

    def record(array):
        cleared.append(array)
        return zero(array)

    with patch.object(wp.array, "zero_", record):
        yield cleared


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
    """Every group's one repair: one-pass (with or without a bank comparison) or over a block list."""
    return {group.repair_kernel for group in _groups(owner)}


def _repair_kind(group):
    """The group's repair as :func:`_launch_kinds` counts it: ``listed`` (classified), ``bank_repair`` (one-pass,
    comparing inside its blocks) or ``repair``."""
    if group.classify_first:
        return "listed"
    return "bank_repair" if "_banked" in group.repair_kernel.key else "repair"


@contextlib.contextmanager
def _launch_kinds(owner):
    """Run every launch and count, in the yielded counter, the owner's repair-family launches by kind."""
    kinds = {}
    for group in _groups(owner):
        kinds.update(
            {group.repair_kernel: _repair_kind(group), group.bank_kernel: "bank", group.classify_kernel: "classify"}
        )
    kinds.pop(None, None)
    counted = collections.Counter()

    def counting(launch):
        def count(kernel, *args, **kwargs):
            if kernel in kinds:
                counted[kinds[kernel]] += 1
            return launch(kernel, *args, **kwargs)

        return count

    with patch.multiple(wp, launch=counting(wp.launch), launch_tiled=counting(wp.launch_tiled)):
        yield counted


def _force_classification(group, classify=True):
    """Classify (or not) a group's repair launches on a one-block grid, whatever their size and device.

    A fixture's launches have fewer blocks than the classification threshold, and CPU groups keep
    the one-pass repair; a one-block grid also makes that block stride over every listed block.
    """
    group._classify_switch = True
    group._repair_grid = 1
    group._classify_above = 0 if classify else group.launch_count
    group._bind_kernels()


def _classify_every_launch(owner):
    """Classify every repair launch of a small fixture on a one-block grid."""
    for group in _groups(owner):
        _force_classification(group)
        if not group.classify_first:
            raise AssertionError("The fixture's repair must be classified on CUDA")


class TestFeatherPGSCapturedRepair(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "Held-tree owners are a CUDA path")
    def test_cuda_captured_steps_repair_on_the_device(self):
        """Launch each group's repair on the first step of a captured block of 2 or 8 steps and replay it as eager
        stepping does.

        The device decides every group of the first step, which follows whatever ran between replays; the later
        steps continue a finish of the same capture on the bank it published, whose groups would all take the early
        exit, so they launch no repair. Worlds reset between replays repair, one-pass or classified; every cache
        generation, geometry generation and state matches eager stepping, including the interval-2 geometry refresh
        cadence.
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
                            "bias": [group.bias.numpy() for group in _groups(owner)],
                        }
                    )
                for key in ("generation", "geometry_generation"):
                    for eager, replayed in zip(finals[0][key], finals[1][key], strict=True):
                        np.testing.assert_array_equal(replayed, eager, err_msg=key)
                for eager, replayed in zip(finals[0]["bias"], finals[1]["bias"], strict=True):
                    np.testing.assert_allclose(replayed, eager, rtol=0.0, atol=2e-6, err_msg="bias")
                for field in STATE_FIELDS:
                    np.testing.assert_allclose(finals[1][field], finals[0][field], rtol=0.0, atol=2e-6, err_msg=field)

    @unittest.skipUnless(wp.is_cuda_available(), "Held-tree owners are a CUDA path")
    def test_cuda_host_calls_between_captured_steps_keep_the_repair(self):
        """Repair through the second step's repair when a reset or a state notification falls between."""
        device = wp.get_device("cuda:0")
        for hook in ("reset", "notify_state_changed"):
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
                        else:
                            solver.notify_state_changed()
                        rollout.step()

                    advance = block
                    if captured:
                        with (
                            _counted_launches(_repair_kernels(owner)) as counted,
                            wp.ScopedCapture(device=device) as capture,
                        ):
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
        """Keep every captured repair of a velocity-limit layout and of each separate capture.

        Two single-step graphs captured one after the other, replayed with an eager reset between
        them, each repair, and match eager stepping.
        """
        device = wp.get_device("cuda:0")
        model = _build(device, worlds=4, speed_limit=0.8)
        solver = _mixed_solver(model, enable_joint_velocity_limits=True)
        owner = solver.factor.held
        rollout = _Rollout(model, solver)
        rollout.block()
        with _counted_launches(_repair_kernels(owner)) as counted, wp.ScopedCapture(device=device):
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


class TestFeatherPGSColdCapture(unittest.TestCase):
    """Capture a solver's first step before any eager step, as Isaac Lab records its physics graph at start-up."""

    @unittest.skipUnless(wp.is_cuda_available(), "Held-tree owners are a CUDA path")
    def test_cuda_cold_captured_copy_back_repairs_nothing(self):
        """Keep every cache across replays of a copy-back step captured before the solver stepped.

        Isaac Lab closes an odd substep count with ``state_0.assign(state_1)`` and records the
        graph before any eager step, so the host cannot know at capture time which bank the
        replays' stamps will name. The captured repair compares on the device: after the first
        replay, which builds every cache, each replay adds one generation per group (its finish),
        one-pass or classified, for the per-size groups (one tree per block, and four), the forest
        and the free-root sparse tree. The same graph replayed with every group invalidated first
        repairs each time, and the two trajectories agree within 2e-6.
        """
        device = wp.get_device("cuda:0")
        replays = 6

        def build(name):
            # One step per graph: refresh every step, as every replay does.
            if name != "packed":
                return TestFeatherPGSInputBank._owned(name, device, update_mass_matrix_interval=1)
            model = _build(device, worlds=8)
            solver = _mixed_solver(model, lanes=8, update_mass_matrix_interval=1)
            self.assertEqual([group.layout.pack for group in _groups(solver.factor.held)], [4, 4])
            return model, solver, solver.factor.held

        for name, classified in itertools.product(("mixed", "packed", "forest", "sparse"), (False, True)):
            with self.subTest(owner=name, classified=classified):
                finals = []
                for forced in (False, True):
                    model, solver, owner = build(name)
                    if classified:
                        _classify_every_launch(owner)
                    rollout = _Rollout(model, solver, copy_back=True)
                    with _launch_kinds(owner) as kinds, wp.ScopedCapture(device=device) as capture:
                        rollout.step()
                    groups = len(_groups(owner))
                    if classified:
                        self.assertEqual(kinds, {"classify": groups, "bank": groups, "listed": groups})
                    else:
                        self.assertEqual(kinds, {"bank_repair": groups})
                    wp.capture_launch(capture.graph)  # builds every cache
                    before = _generations(owner)
                    for _ in range(replays - 1):
                        if forced:
                            solver.factor.fk_id_cache_valid.zero_()
                        wp.capture_launch(capture.graph)
                    for start, end in zip(before, _generations(owner), strict=True):
                        np.testing.assert_array_equal(end - start, (1 + forced) * (replays - 1), err_msg=f"{forced=}")
                    finals.append({field: getattr(rollout.states[0], field).numpy() for field in STATE_FIELDS})
                for field in STATE_FIELDS:
                    np.testing.assert_allclose(finals[0][field], finals[1][field], rtol=0.0, atol=2e-6, err_msg=field)

    @unittest.skipUnless(wp.is_cuda_available(), "Held-tree owners are a CUDA path")
    def test_cuda_cold_captured_block_classifies_its_repairs(self):
        """Classify the first repair of a two-step block captured before the solver stepped.

        In a ping-pong block (an even substep count) the first repair's replays follow resets the host
        does not see at capture time, so it lists its blocks (and those to compare) on the device; the
        second step continues the first's finish and launches none. A velocity-limit layout keeps the
        one-pass repair on both steps, also when forced to classify. Replays with a
        world reset before each match the same block captured with one-pass repairs: states within
        2e-6 and, outside the velocity-limit layout (whose repairs follow clamped velocities that
        separate instances may round apart), equal cache and geometry generations.
        """
        device = wp.get_device("cuda:0")

        def velocity_limits():
            model = _build(device, worlds=4, speed_limit=0.8)
            return model, _mixed_solver(model, enable_joint_velocity_limits=True)

        cases = {
            "mixed": lambda: TestFeatherPGSInputBank._owned("mixed", device)[:2],
            "forest": lambda: TestFeatherPGSInputBank._owned("forest", device)[:2],
            "sparse": lambda: TestFeatherPGSInputBank._owned("sparse", device, update_mass_matrix_interval=2)[:2],
            "motion": velocity_limits,
        }
        for name, build in cases.items():
            with self.subTest(owner=name):
                finals = []
                for classified in (False, True):
                    model, solver = build()
                    owner = solver.factor.held
                    self.assertIsNotNone(owner)
                    groups = len(_groups(owner))
                    motion = name == "motion"
                    if classified and motion:
                        for group in _groups(owner):
                            _force_classification(group)
                            self.assertFalse(group.classify_first)
                    elif classified:
                        _classify_every_launch(owner)
                    rollout = _Rollout(model, solver)
                    with _launch_kinds(owner) as kinds, wp.ScopedCapture(device=device) as capture:
                        rollout.block()
                    if motion:
                        expected = {"repair": 2 * groups}
                    elif classified:
                        expected = {"classify": groups, "bank": groups, "listed": groups}
                    else:
                        expected = {"bank_repair": groups}
                    self.assertEqual(kinds, expected)
                    for _ in range(4):
                        rollout.reset_worlds((1,))
                        wp.capture_launch(capture.graph)
                    finals.append(
                        {
                            **{field: getattr(rollout.states[0], field).numpy() for field in STATE_FIELDS},
                            "generation": _generations(owner),
                            "geometry_generation": [group.data.geometry_generation.numpy() for group in _groups(owner)],
                        }
                    )
                for key in ("generation", "geometry_generation") if name != "motion" else ():
                    for one_pass, listed in zip(finals[0][key], finals[1][key], strict=True):
                        np.testing.assert_array_equal(listed, one_pass, err_msg=key)
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
    def test_only_launches_above_two_grids_are_classified(self):
        """Classify a repair only when its launch has more blocks than two listed grids.

        Up to two grids the one-pass launch's early exits cost about what the classified repair's
        launches do, and a copied bank is compared inside the one-pass repair. CPU has no listed
        grid and keeps the one-pass repair: the owners are built with a one-block grid and switched
        on as on CUDA, so two-world groups (two blocks) keep the one-pass repair, with the in-kernel
        comparison, and three-world groups classify.
        """
        with patch.object(tree_kinetic_owner, "listed_repair_blocks", return_value=1):
            small, large = _cpu_owner(worlds=2)[2], _cpu_owner(worlds=3)[2]
        for owner, blocks, classified in ((small, 2, False), (large, 3, True)):
            for group in owner.groups:
                with self.subTest(blocks=blocks, bodies=group.layout.bodies):
                    group._classify_switch = True
                    group._bind_kernels()
                    self.assertEqual((group._repair_grid, group.launch_count), (1, blocks))
                    self.assertEqual(group.classify_first, classified)
                    self.assertEqual(group.classify_kernel is not None, classified)
                    self.assertEqual(_repair_kind(group), "listed" if classified else "bank_repair")

    @unittest.skipUnless(wp.is_cuda_available(), "Held-tree owners are a CUDA path")
    def test_cuda_captured_motion_repair_above_two_grids_is_one_pass(self):
        """Capture a velocity-limit repair above two listed grids as the one-pass launch, without a classification.

        Built with a one-block listed grid and one tree per block (32 lanes), the owners' four-world
        groups lie above the two-block threshold, as a 16384-world scene's groups do on the device's
        own grid, and the owner decides on CUDA unforced. In a captured block of two steps each velocity-limit group
        launches its one-pass repair twice, launches no classification and clears no block list; an
        ordinary-motion owner of the same size classifies the first, and its second step, which continues the
        first's finish, launches none. Replays with a world reset before each match eager
        stepping of the same configuration within 2e-6.
        """
        device = wp.get_device("cuda:0")

        def velocity_limits():
            model = _build(device, worlds=4, speed_limit=0.8)
            return model, _mixed_solver(model, enable_joint_velocity_limits=True, lanes=32)

        for name, build in (("motion", velocity_limits), ("mixed", lambda: _owned_case("mixed", device, lanes=32))):
            with self.subTest(owner=name):
                motion = name == "motion"
                finals = []
                for captured in (False, True):
                    with patch.object(tree_kinetic_owner, "listed_repair_blocks", return_value=1):
                        model, solver = build()
                    owner = solver.factor.held
                    self.assertIsNotNone(owner)
                    groups = _groups(owner)
                    for group in groups:
                        self.assertEqual(group.motion_velocity, motion)
                        self.assertEqual((group._repair_grid, group._classify_above), (1, 2))
                        self.assertGreater(group.launch_count, group._classify_above)
                        self.assertEqual(group.classify_first, not motion)
                    rollout = _Rollout(model, solver)
                    advance = rollout.block
                    if captured:
                        with (
                            _launch_kinds(owner) as kinds,
                            _cleared_arrays() as cleared,
                            wp.ScopedCapture(device=device) as capture,
                        ):
                            rollout.block()
                        cleared_lists = [any(array is group._list_counts for array in cleared) for group in groups]
                        count = len(groups)
                        if motion:
                            self.assertEqual(kinds, {"repair": 2 * count})
                        else:
                            self.assertEqual(kinds, {"classify": count, "bank": count, "listed": count})
                        self.assertEqual(cleared_lists, [not motion] * len(groups))
                        advance = functools.partial(wp.capture_launch, capture.graph)
                    for _ in range(4):
                        rollout.reset_worlds((1,))
                        advance()
                    finals.append({field: getattr(rollout.states[0], field).numpy() for field in STATE_FIELDS})
                for field in STATE_FIELDS:
                    np.testing.assert_allclose(finals[1][field], finals[0][field], rtol=0.0, atol=2e-6, err_msg=field)

    def check_owner_launches(self, device):
        """Run every launch form of ordinary and motion-velocity owners; each must match its kernel's parameters.

        Three owners of two-world scenes take two steps and a copy-back begin: an ordinary owner
        forced to classify on a one-block grid (classification and listed repair, then
        classification, comparison pass and listed repair), an ordinary owner left unforced (the
        one-pass repair that compares banks) and a motion-velocity owner forced alike (its one-pass
        motion repair every time). Every held-tree
        launch, those at construction included, is checked before it runs, and every kernel each
        group bound must have run in a launch of that group's cache. These groups are per-size trees
        on kinetic plans.
        """
        device = wp.get_device(device)
        launched = []

        def checked(launch):
            def launch_checked(kernel, *args, **kwargs):
                if kernel.key.startswith("tree_kinetic_"):
                    inputs = kwargs.get("inputs", ())
                    # Checked before the launch: Warp would run a struct passed in another struct's place.
                    self.assertEqual(_input_mismatches(kernel, inputs), [], kernel.key)
                    launched.append((kernel, inputs))
                return launch(kernel, *args, **kwargs)

            return launch_checked

        # Each owner's (alternate motion, forced classification) and the kernels it binds besides its
        # repair, finish and predictor.
        owners = {
            "classified": (False, True, {"classify", "bank"}),
            "one_pass": (False, False, set()),
            "motion": (True, True, set()),
        }
        with patch.multiple(wp, launch=checked(wp.launch), launch_tiled=checked(wp.launch_tiled)):
            for name, (motion, forced, bound) in owners.items():
                with self.subTest(owner=name):
                    launched.clear()
                    model = _build(device, worlds=2, speed_limit=0.8 if motion else None)
                    solver = SolverFeatherPGS(model, friction_anchor_beta=0.0)
                    solver.options = solver.laws.options = solver.factor.options = solver.placement.options = (
                        solver.sweep.options
                    ) = dataclasses.replace(solver.options, enable_joint_velocity_limits=motion)
                    owner = HeldTree(solver)
                    groups = []
                    for group in owner.groups:
                        if forced:
                            _force_classification(group)
                        self.assertEqual(group.classify_first, forced and not motion)
                        kernels = {
                            group.repair_kernel: "repair",
                            group.finish_kernel: "finish",
                            group.predictor_kernel: "predict",
                            group.classify_kernel: "classify",
                            group.bank_kernel: "bank",
                        }
                        kernels.pop(None, None)
                        self.assertEqual(sorted(kernels.values()), sorted({"repair", "finish", "predict", *bound}))
                        groups.append((group, kernels))
                    states, control = [model.state(), model.state()], model.control()
                    # A motion-velocity owner reads a prescaled velocity bank of its own.
                    velocities = [wp.clone(state.joint_qd) if motion else state.joint_qd for state in states]
                    for source, refresh in ((0, True), (1, False)):
                        state, output, velocity = states[source], states[1 - source], velocities[source]
                        _set_clock(solver.factor, refresh)
                        owner.begin(state, solver.factor, velocity, DT)
                        solver.factor.mass_update_mask.fill_(1)
                        owner.refresh(solver.factor)
                        owner.predict(state, solver.factor, control, velocity, DT)
                        solver.v_out.assign(solver.v_hat)
                        _set_clock(solver.factor, refresh, ahead=1)
                        owner.finish(state, solver.factor, output, DT)
                    # A copy of the bank the last finish published, one coordinate changed.
                    states[1].assign(states[0])
                    coordinates = states[1].joint_q.numpy()
                    coordinates[0] += np.float32(0.01)
                    states[1].joint_q.assign(coordinates)
                    _set_clock(solver.factor, False)
                    owner.begin(states[1], solver.factor, velocities[1], DT, state_out=states[0])
                    # Checked per group, by the cache each held-tree launch passes: the owner holds two groups (a
                    # free body and an arm), and every kind would still run with one group's classified forms skipped.
                    for group, kernels in groups:
                        ran = {kernel for kernel, inputs in launched if any(value is group.data for value in inputs)}
                        unlaunched = sorted(kind for kernel, kind in kernels.items() if kernel not in ran)
                        self.assertEqual(unlaunched, [], f"group of {group.layout.bodies} bodies")
                    if motion:
                        self.assertTrue(all("_motion" in group.repair_kernel.key for group in owner.groups))

    def test_owner_launches_match_their_kernels(self):
        """Launch every held-tree kernel an owner binds with one input of each parameter's type, on CPU.

        The classification takes the six common arguments, not the motion input of the launch ABI
        that every finish and repair shares; the classified forms are ordinary motion's. CPU owners
        keep the one-pass repair, so classification is forced; see :meth:`check_owner_launches`.
        """
        self.check_owner_launches("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "Held-tree owners are a CUDA path")
    def test_cuda_owner_launches_match_their_kernels(self):
        """Launch every held-tree kernel a CUDA owner binds, construction's lane timing included, with matching inputs."""
        self.check_owner_launches("cuda:0")

    @unittest.skipUnless(wp.is_cuda_available(), "Held-tree owners are a CUDA path")
    def test_cuda_classified_repair_matches_the_one_pass_repair(self):
        """Compute bitwise the same caches and publication with the classified repair as with one block per group.

        From one device state with random invalid, requested, geometrically stale and
        foreign-stamped groups, the one-pass repair and the classified repair write identical words
        to every array the repair reaches, for packed per-size groups, a packed forest and sparse trees
        of one tree per block. The blocks the classification lists are exactly those with a group the
        one-pass repair recomputed. (Velocity-limit layouts keep the one-pass repair.)
        """
        device = wp.get_device("cuda:0")
        rng = np.random.default_rng(20261019)
        # Eight lanes per tree: four trees share each block of the per-size groups and of the forest.
        cases = {
            "mixed": lambda: _mixed_solver(_build(device, worlds=8), lanes=8),
            "forest": lambda: _mixed_solver(_build(device, worlds=8, prescribed=True), forest=True, lanes=8),
            "sparse": lambda: _one_tree_per_block(_owned_case, "sparse", device)[1],
        }
        for name, build in cases.items():
            with self.subTest(owner=name):
                solver = build()
                owner = solver.factor.held
                self.assertIsNotNone(owner)
                # On a one-block grid, which also makes that block stride over every listed block.
                _classify_every_launch(owner)
                packs = [group.layout.pack for group in _groups(owner)]
                self.assertTrue(all(pack == 4 for pack in packs) if name != "sparse" else packs == [1], packs)
                self.assertEqual(name == "sparse", _groups(owner)[0].chains)
                rollout = _Rollout(solver.model, solver)
                rollout.block()
                state = rollout.states[0]
                for seed in range(4):
                    data = _groups(owner)[0]._publication(state, solver.factor, state, DT)
                    arrays = {}
                    for index, group in enumerate(_groups(owner)):
                        arrays.update({f"{index}.{key}": value for key, value in _repair_arrays(group, data).items()})
                    arrays["requests"] = solver.factor.mass_update_requested
                    self._scramble(rng, owner, solver)
                    start = {key: value.numpy().copy() for key, value in arrays.items()}
                    results = []
                    for classify in (False, True):
                        for key, value in arrays.items():
                            value.assign(start[key])
                        for group in _groups(owner):
                            _force_classification(group, classify)
                        _set_clock(solver.factor, bool(seed % 2))
                        # Each group's repair: the scramble wrote behind the solver, which the owner's record of the
                        # last finish (HeldTree.begin) does not see.
                        for group in _groups(owner):
                            group.begin(state, solver.factor, state.joint_qd, DT)
                        results.append({key: value.numpy().copy() for key, value in arrays.items()})
                        if classify:
                            listed = [
                                np.sort(group._repair_blocks.numpy()[: int(group._list_counts.numpy()[0])])
                                for group in _groups(owner)
                            ]
                    for key in arrays:
                        np.testing.assert_array_equal(results[1][key], results[0][key], err_msg=f"{name} {seed} {key}")
                    for index, (group, blocks) in enumerate(zip(_groups(owner), listed, strict=True)):
                        generation = f"{index}.cache.generation"
                        repaired = results[0][generation] != start[generation]
                        expected = np.flatnonzero(repaired.reshape(-1, group.layout.pack).any(axis=1))
                        np.testing.assert_array_equal(blocks, expected, err_msg=f"{name} {seed} group {index}")

    @unittest.skipUnless(wp.is_cuda_available(), "Held-tree owners are a CUDA path")
    def test_cuda_classified_comparison_matches_the_one_pass_comparison(self):
        """Compare copied banks bitwise alike in the classified repair and in the one-pass repair.

        After copy-back steps every group is stamped with the output bank and the input bank holds
        its words. From random invalid, requested, geometrically stale and foreign-stamped groups
        and random words written into the input bank, the classified repair (the classification
        lists the copied groups' blocks, a pass compares them and lists the blocks that still need
        a repair) and the one-pass repair that compares inside its blocks write identical words to
        every array the repair reaches, for packed per-size groups and a packed forest (four trees
        per block, so a block can hold equal, unequal and repairing groups at once) and chain-scan
        sparse trees. The repair list holds exactly the blocks with a recomputed group, once each.
        """
        device = wp.get_device("cuda:0")
        rng = np.random.default_rng(20261020)
        cases = {
            "mixed": lambda: _mixed_solver(_build(device, worlds=8), lanes=8),
            "forest": lambda: _mixed_solver(_build(device, worlds=8, prescribed=True), forest=True, lanes=8),
            "sparse": lambda: _one_tree_per_block(_owned_case, "sparse", device)[1],
        }
        for name, build in cases.items():
            with self.subTest(owner=name):
                solver = build()
                owner = solver.factor.held
                self.assertIsNotNone(owner)
                self.assertTrue(owner.bank_compare)
                packs = [group.layout.pack for group in _groups(owner)]
                self.assertTrue(all(pack == 4 for pack in packs) if name != "sparse" else packs == [1], packs)
                rollout = _Rollout(solver.model, solver, copy_back=True)
                rollout.block()
                state, output = rollout.states
                data = _groups(owner)[0]._publication(state, solver.factor, state, DT)
                arrays = {}
                for index, group in enumerate(_groups(owner)):
                    arrays.update({f"{index}.{key}": value for key, value in _repair_arrays(group, data).items()})
                arrays["requests"] = solver.factor.mass_update_requested
                for group in _groups(owner):
                    np.testing.assert_array_equal(group.data.source_q.numpy(), output.joint_q.ptr)
                for seed in range(4):
                    self._scramble(rng, owner, solver)
                    q = state.joint_q.numpy()
                    q[rng.random(q.size) < 0.05] += np.float32(0.01)
                    state.joint_q.assign(q)
                    start = {key: value.numpy().copy() for key, value in arrays.items()}
                    results = []
                    for classify in (False, True):
                        for key, value in arrays.items():
                            value.assign(start[key])
                        for group in _groups(owner):
                            _force_classification(group, classify)
                            self.assertEqual(group.classify_first, classify)
                        _set_clock(solver.factor, bool(seed % 2))
                        owner.begin(state, solver.factor, state.joint_qd, DT, state_out=output)
                        results.append({key: value.numpy().copy() for key, value in arrays.items()})
                        if classify:
                            listed = [
                                group._repair_blocks.numpy()[: int(group._list_counts.numpy()[0])]
                                for group in _groups(owner)
                            ]
                    for key in arrays:
                        np.testing.assert_array_equal(results[1][key], results[0][key], err_msg=f"{name} {seed} {key}")
                    for index, (group, blocks) in enumerate(zip(_groups(owner), listed, strict=True)):
                        generation = f"{index}.cache.generation"
                        repaired = results[0][generation] != start[generation]
                        expected = np.flatnonzero(repaired.reshape(-1, group.layout.pack).any(axis=1))
                        np.testing.assert_array_equal(np.sort(blocks), expected, err_msg=f"{name} {seed} group {index}")

    @staticmethod
    def _scramble(rng, owner, solver):
        """Invalidate, request, stale and foreign-stamp random groups."""
        arts = solver.model.articulation_count
        valid = solver.factor.fk_id_cache_valid.numpy()
        valid[rng.random(arts) < 0.3] = 0
        solver.factor.fk_id_cache_valid.assign(valid)
        solver.factor.mass_update_requested.assign((rng.random(arts) < 0.2).astype(np.int32))
        for group in _groups(owner):
            count = group.group_count
            geometry = group.geometry_valid.numpy()
            geometry[rng.random(count) < 0.2] = 0
            group.geometry_valid.assign(geometry)
            stamps = group.data.source_q.numpy()
            stamps[rng.random(count) < 0.2] = 12345
            group.data.source_q.assign(stamps)


if __name__ == "__main__":
    unittest.main()
