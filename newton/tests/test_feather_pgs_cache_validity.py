# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Keep held-tree caches valid across host calls that change none of their inputs."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import cached_state_binding
from newton._src.solvers.feather_pgs.mixed_tree_kinetic import MixedTreeKineticState
from newton._src.solvers.feather_pgs.solver_feather_pgs import SolverFeatherPGS
from newton.tests.test_feather_pgs_mixed_tree_free_body import _build
from newton.tests.test_feather_pgs_sparse_solver import STATE_FIELDS, _build_model, _make_solver

DT = 1.0 / 240.0
JOINT = newton.ModelFlags.JOINT_PROPERTIES


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


def _fixed_root(model, world):
    """The fixed root joint of ``world``'s arm."""
    roots = (model.joint_parent.numpy() == -1) & (model.joint_type.numpy() == int(newton.JointType.FIXED))
    return int(np.flatnonzero(roots & (model.joint_world.numpy() == world))[0])


def _mixed_solver(model, *, forest=False, owner=True):
    """A dense-route solver with the mixed held-tree owner, or with ordinary dynamics when ``owner`` is False."""
    environment = {
        "FEATHER_PGS_MIXED_TREE_KINETIC_STATE": str(int(owner)),
        "FEATHER_PGS_TREE_KINETIC_STATE": "0",
        "FEATHER_PGS_TREE_FOREST": str(int(forest)),
    }
    with patch.dict(os.environ, environment):
        return SolverFeatherPGS(
            model,
            pgs_mode="matrix_free",
            pgs_iterations=8,
            enable_joint_limits=True,
            update_mass_matrix_interval=2,
            friction_anchor_beta=0.0,
        )


def _owned_case(name, device):
    """A model and a solver with the named held-tree owner kind: per-size groups, the forest or the sparse tree."""
    if name == "mixed":
        model = _build(device, worlds=4)
        return model, _mixed_solver(model)
    if name == "forest":
        model = _build(device, worlds=4, prescribed=True)
        return model, _mixed_solver(model, forest=True)
    model = _build_model()
    return model, _make_solver(model, True, cached_dynamics="on", friction_anchor_beta=0.0)


class _Rollout:
    """Ping-pong one solver's states over sorted contacts, as a captured Isaac Lab physics step does."""

    def __init__(self, model, solver):
        self.model, self.solver = model, solver
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
        steady ping-pong rollout adds exactly one per step. The trajectory equals the rollout
        without notifications.
        """
        device = wp.get_device("cuda:0")
        steps = 6
        for name in ("mixed", "forest", "sparse"):
            with self.subTest(owner=name):
                finals = []
                for notified in (False, True):
                    model, solver = _owned_case(name, device)
                    owner = solver._tree_kinetic_state
                    self.assertIsNotNone(owner, "The fixture must admit a held-tree owner")
                    rollout = _Rollout(model, solver)
                    rollout.step()  # the first repair builds every cache
                    before = _generations(owner)
                    for _ in range(steps):
                        if notified:
                            solver.notify_model_changed(JOINT)
                        rollout.step()
                    for start, end in zip(before, _generations(owner), strict=True):
                        np.testing.assert_array_equal(end - start, steps, err_msg=f"{notified=}")
                    finals.append(rollout.states[0])
                for field in STATE_FIELDS:
                    np.testing.assert_allclose(
                        getattr(finals[1], field).numpy(),
                        getattr(finals[0], field).numpy(),
                        rtol=1e-5,
                        atol=2e-6,
                        err_msg=field,
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
            np.testing.assert_allclose(finals[1][field], finals[0][field], rtol=1e-5, atol=2e-6, err_msg=field)

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


def _mass_matrices(solver):
    """Both double-buffer slots' dense mass matrices, by slot and DOF size."""
    return [(slot, size, array) for slot, buffers in enumerate(solver._H_bufs) for size, array in buffers.items()]


class TestFeatherPGSHeldMassMatrix(unittest.TestCase):
    def test_unread_sizes_mirror_the_factor_dispatch(self):
        """Report exactly the sizes whose stage-2 factorization reads no dense mass matrix under an owner.

        A compact size passes; a diagonal size, and a tiled or loop size without its fused
        CRBA-Cholesky kernel, factor H; the fused kernels' sizes return before reading it.
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
        unread = SolverFeatherPGS._unread_mass_matrix_sizes(solver)
        self.assertEqual(unread, (2, 6, 24))
        launched = []

        def record(kernel, *args, inputs=(), **kwargs):
            launched.extend(inputs)

        with patch.object(wp, "launch", record), patch.object(wp, "launch_tiled", record):
            for size in sizes:
                launched.clear()
                if size == solver._compact_diagonal_mass_size:
                    pass  # stage 2 skips compact sizes
                elif solver._execution_plan.use_diagonal_mass(size):
                    SolverFeatherPGS._stage2_cholesky_diagonal(solver, size)
                elif solver._execution_plan.use_tiled_cholesky(size):
                    SolverFeatherPGS._stage2_cholesky_tiled(solver, size)
                else:
                    SolverFeatherPGS._stage2_cholesky_loop(solver, size)
                reads = any(item is solver.H_by_size[size] for item in launched)
                self.assertEqual(reads, size not in unread, f"size {size}")
        solver._tree_kinetic_state = None
        self.assertEqual(SolverFeatherPGS._unread_mass_matrix_sizes(solver), ())

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
                record = solver.get_path_selection()["routes"]["dynamics"]["mass_matrix_clear_dofs"]
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
            np.testing.assert_allclose(finals[1][field], finals[0][field], rtol=1e-6, atol=1e-6, err_msg=field)


if __name__ == "__main__":
    unittest.main()
