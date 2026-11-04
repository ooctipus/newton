# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.factor.kernels import Clock, advance_clock
from newton._src.solvers.feather_pgs.tree_factor import TreeFactor
from newton.solvers import SolverFeatherPGS

DT = 1.0 / 60.0
# FK/ID reuse belongs to the dense size groups: keep sparse factors off under either friction law.
DENSE_PINS = {"storage": "dense"}


def _dense_solver(model, **options):
    with mock.patch.dict(SolverFeatherPGS._route_pins, DENSE_PINS):
        return SolverFeatherPGS(model, **options)


# Per-articulation initial pose/velocity: [base, left branch, right branch].
INITIAL_JOINT_Q = (0.7, 0.3, -0.4)
INITIAL_JOINT_QD = (0.5, -0.2, 0.3)


def _build_model(device, num_worlds=2, ground=True):
    """Two-branch pendulum: base revolute joint plus two sibling branch links.

    The sibling branches are not ancestor-related, so H has structural zeros
    between their DOFs; with a ground plane the branch tips swing into contact.
    """
    env = newton.ModelBuilder()
    env.default_shape_cfg.density = 1000.0

    base = env.add_link()
    env.add_shape_box(base, hx=0.08, hy=0.08, hz=0.08)
    left = env.add_link()
    env.add_shape_box(left, hx=0.25, hy=0.05, hz=0.05)
    right = env.add_link()
    env.add_shape_box(right, hx=0.25, hy=0.05, hz=0.05)

    j_base = env.add_joint_revolute(
        -1,
        base,
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.55), wp.quat_identity()),
        axis=newton.Axis.Y,
        target_ke=15.0,
        target_kd=1.5,
    )
    j_left = env.add_joint_revolute(
        base,
        left,
        parent_xform=wp.transform(wp.vec3(0.08, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.25, 0.0, 0.0), wp.quat_identity()),
        axis=newton.Axis.Y,
        target_ke=15.0,
        target_kd=1.5,
    )
    j_right = env.add_joint_revolute(
        base,
        right,
        parent_xform=wp.transform(wp.vec3(-0.08, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.25, 0.0, 0.0), wp.quat_identity()),
        axis=newton.Axis.Y,
        target_ke=15.0,
        target_kd=1.5,
    )
    env.add_articulation([j_base, j_left, j_right])

    builder = newton.ModelBuilder()
    builder.replicate(env, num_worlds)
    if ground:
        builder.add_ground_plane()
    return builder.finalize(device=device)


def _build_floating_model(device, num_worlds=2):
    """Floating two-link articulation with nonzero root and joint motion."""
    env = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    root = env.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
        mass=2.0,
        com=wp.vec3(0.07, -0.03, 0.02),
        inertia=wp.mat33(0.03, 0.0, 0.0, 0.0, 0.04, 0.0, 0.0, 0.0, 0.05),
    )
    child = env.add_link(
        xform=wp.transform(wp.vec3(0.3, 0.0, 1.0), wp.quat_identity()),
        mass=1.0,
        com=wp.vec3(0.04, 0.01, -0.02),
        inertia=wp.mat33(0.01, 0.0, 0.0, 0.0, 0.012, 0.0, 0.0, 0.0, 0.014),
    )
    free = env.add_joint_free(root)
    hinge = env.add_joint_revolute(
        root,
        child,
        parent_xform=wp.transform(wp.vec3(0.2, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.1, 0.0, 0.0), wp.quat_identity()),
        axis=newton.Axis.Y,
    )
    env.add_articulation([free, hinge])
    builder = newton.ModelBuilder()
    builder.replicate(env, num_worlds)
    return builder.finalize(device=device)


def _make_floating_state(model):
    state = model.state()
    qd = state.joint_qd.numpy()
    for articulation in range(model.articulation_count):
        start = articulation * 7
        qd[start : start + 7] = (0.4, -0.2, 0.1, 0.5, -0.7, 0.3, 0.8)
    state.joint_qd.assign(qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return state


def _run_floating_trajectory(model, solver, num_steps, *, invalidate_fk_each_step):
    state_0 = _make_floating_state(model)
    state_1 = model.state()
    contacts = model.contacts()
    control = model.control()
    history = {name: [] for name in ("joint_q", "joint_qd", "body_q", "body_qd")}
    mass_masks = []
    for _ in range(num_steps):
        if invalidate_fk_each_step:
            # A reset also refreshes mass factors; isolate FK/ID reuse instead.
            solver.factor.fk_id_cache_valid.zero_()
        solver.step(state_0, state_1, control, contacts, DT)
        mass_masks.append(solver.factor.mass_update_mask.numpy().copy())
        state_0, state_1 = state_1, state_0
        for name, values in history.items():
            values.append(getattr(state_0, name).numpy().copy())
    return {name: np.stack(values) for name, values in history.items()}, np.stack(mass_masks)


def _make_initial_state(model):
    state = model.state()
    num_arts = model.articulation_count
    state.joint_q.assign(np.tile(np.asarray(INITIAL_JOINT_Q, dtype=np.float32), num_arts))
    state.joint_qd.assign(np.tile(np.asarray(INITIAL_JOINT_QD, dtype=np.float32), num_arts))
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return state


def _run_trajectory(model, solver, num_steps):
    state_0 = _make_initial_state(model)
    state_1 = model.state()
    contacts = model.contacts()
    control = model.control()
    joint_q_history = []
    for _ in range(num_steps):
        model.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, DT)
        state_0, state_1 = state_1, state_0
        joint_q_history.append(state_0.joint_q.numpy().copy())
    return np.stack(joint_q_history)


def _run_state_trajectory(model, solver, num_steps, *, invalidate_fk_each_step):
    state_0 = _make_initial_state(model)
    state_1 = model.state()
    contacts = model.contacts()
    control = model.control()
    history = {name: [] for name in ("joint_q", "joint_qd", "body_q", "body_qd")}
    mass_masks = []
    for _ in range(num_steps):
        if invalidate_fk_each_step:
            # A reset also refreshes mass factors; isolate FK/ID reuse instead.
            solver.factor.fk_id_cache_valid.zero_()
        model.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, DT)
        mass_masks.append(solver.factor.mass_update_mask.numpy().copy())
        state_0, state_1 = state_1, state_0
        for name, values in history.items():
            values.append(getattr(state_0, name).numpy().copy())
    return {name: np.stack(values) for name, values in history.items()}, np.stack(mass_masks)


class TestFeatherPGSMassUpdateInterval(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "FK/ID reuse requires CUDA")
    def test_split_fk_id_cache_does_not_alias_active_step_state(self):
        """Validate split-mode cached dynamics by input content, with no private snapshot of the solve.

        A split-mode solver keeps no copy of its FK/ID cache (the snapshot is an opt-in bisection
        aid). Its held-tree owner, here over a fixed-base arm and a free box per world, publishes
        the caches in place: a step whose input bank is a copy of the bank the last step published
        compares the two and keeps every cache, and a copy with one written arm repairs that arm's
        group alone. Both match a rollout that recomputes every step.
        """
        from newton.tests.test_feather_pgs_mixed_tree_free_body import _build  # noqa: PLC0415

        trajectories = []
        for forced in (False, True):
            model = _build("cuda:0", worlds=3)
            with mock.patch.object(SolverFeatherPGS, "_route_pins", {"storage": "dense"}):
                solver = SolverFeatherPGS(model, update_mass_matrix_interval=2, friction_anchor_beta=0.0)
            owner = solver.factor.held
            self.assertIsNotNone(owner)
            self.assertTrue(owner.bank_compare)
            state_in, state_out = model.state(), model.state()
            newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
            control, contacts = model.control(), model.contacts()
            revolute = (model.joint_type.numpy() == int(newton.JointType.REVOLUTE)) & (model.joint_world.numpy() == 1)
            joint = int(np.flatnonzero(revolute)[0])
            history = []
            for step in range(8):
                if step == 4:
                    # Change world 1's first arm joint in the copied input bank, without a notification.
                    qd = state_in.joint_qd.numpy()
                    qd[model.joint_qd_start.numpy()[joint]] += 0.25
                    state_in.joint_qd.assign(qd)
                if forced:
                    solver.factor.fk_id_cache_valid.zero_()
                before = [group.data.generation.numpy().copy() for group in owner.groups]
                solver.step(state_in, state_out, control, contacts, DT)
                state_in.assign(state_out)
                if step and not forced:
                    for group, start in zip(owner.groups, before, strict=True):
                        written = group.sparse.plan.group_to_art.numpy() == model.joint_articulation.numpy()[joint]
                        # One finish per step; only the written arm's group repairs as well.
                        np.testing.assert_array_equal(
                            group.data.generation.numpy() - start, 1 + (written & (step == 4))
                        )
                history.append({name: getattr(state_in, name).numpy().copy() for name in ("joint_q", "joint_qd")})
            trajectories.append(history)
        for cached, recomputed in zip(*trajectories, strict=True):
            for name in cached:
                np.testing.assert_allclose(cached[name], recomputed[name], rtol=0.0, atol=2.0e-6, err_msg=name)

    @unittest.skipUnless(wp.is_cuda_available(), "FK/ID reuse requires CUDA")
    def test_cached_fk_id_matches_forced_recomputation(self):
        """Match forced recomputation across cached fixed-base steps."""
        trajectories = []
        for invalidate_fk_each_step in (False, True):
            model = _build_model("cuda:0", ground=False)
            solver = _dense_solver(model, update_mass_matrix_interval=2)
            trajectory, mass_masks = _run_state_trajectory(
                model, solver, num_steps=20, invalidate_fk_each_step=invalidate_fk_each_step
            )
            trajectories.append(trajectory)
            expected = np.broadcast_to((np.arange(20) % 2 == 0)[:, None], mass_masks.shape)
            np.testing.assert_array_equal(mass_masks, expected)

        for name in trajectories[0]:
            np.testing.assert_allclose(
                trajectories[0][name], trajectories[1][name], rtol=0.0, atol=2.0e-6, err_msg=name
            )

    @unittest.skipUnless(wp.is_cuda_available(), "FK/ID reuse requires CUDA")
    def test_cached_fk_id_matches_forced_recomputation_for_floating_base(self):
        """Match forced recomputation across cached floating-base steps."""
        trajectories = []
        for invalidate_fk_each_step in (False, True):
            model = _build_floating_model("cuda:0")
            solver = _dense_solver(model, update_mass_matrix_interval=2)
            trajectory, mass_masks = _run_floating_trajectory(
                model, solver, num_steps=20, invalidate_fk_each_step=invalidate_fk_each_step
            )
            trajectories.append(trajectory)
            expected = np.broadcast_to((np.arange(20) % 2 == 0)[:, None], mass_masks.shape)
            np.testing.assert_array_equal(mass_masks, expected)

        for name in trajectories[0]:
            np.testing.assert_allclose(
                trajectories[0][name], trajectories[1][name], rtol=0.0, atol=2.0e-6, err_msg=name
            )

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA graph capture requires CUDA")
    def test_captured_parallel_dynamics_matches_fused_trajectory(self):
        """Match captured parallel dynamics against eager fused execution."""
        graph_model = _build_model("cuda:0", ground=False)
        eager_model = _build_model("cuda:0", ground=False)
        graph_solver = _dense_solver(graph_model, update_mass_matrix_interval=1)
        eager_solver = _dense_solver(eager_model, update_mass_matrix_interval=1)
        graph_state, graph_out = _make_initial_state(graph_model), graph_model.state()
        eager_state, eager_out = _make_initial_state(eager_model), eager_model.state()
        graph_control, eager_control = graph_model.control(), eager_model.control()
        graph_pipeline = newton.CollisionPipeline(graph_model)
        eager_pipeline = newton.CollisionPipeline(eager_model)
        graph_contacts, eager_contacts = graph_pipeline.contacts(), eager_pipeline.contacts()

        def one_step(solver, state_in, state_out, control, contacts):
            solver.step(state_in, state_out, control, contacts, DT)
            wp.copy(state_in.body_q, state_out.body_q)
            wp.copy(state_in.body_qd, state_out.body_qd)
            wp.copy(state_in.joint_q, state_out.joint_q)
            wp.copy(state_in.joint_qd, state_out.joint_qd)

        one_step(graph_solver, graph_state, graph_out, graph_control, graph_contacts)
        one_step(eager_solver, eager_state, eager_out, eager_control, eager_contacts)
        with wp.ScopedCapture("cuda:0") as capture:
            one_step(graph_solver, graph_state, graph_out, graph_control, graph_contacts)

        for _ in range(20):
            wp.capture_launch(capture.graph)
            one_step(eager_solver, eager_state, eager_out, eager_control, eager_contacts)

        for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
            np.testing.assert_allclose(
                getattr(graph_state, name).numpy(),
                getattr(eager_state, name).numpy(),
                rtol=0.0,
                atol=2.0e-6,
                err_msg=name,
            )

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA graph capture requires CUDA")
    def test_captured_two_substeps_match_eager(self):
        """Replay two captured substeps whose solver launches first inside the capture, as eager steps do.

        A zero-initialised allocation on the first step, inside a CUDA graph capture, records a memset that replays
        before every frame and clears whatever the solver carries across steps; the solver allocates at construction.
        """
        warm_model = _build_model("cuda:0", ground=False)
        warm_solver = SolverFeatherPGS(warm_model, update_mass_matrix_interval=2)
        warm_solver.step(_make_initial_state(warm_model), warm_model.state(), warm_model.control(), None, DT)
        wp.synchronize_device("cuda:0")

        graph_model = _build_model("cuda:0", ground=False)
        eager_model = _build_model("cuda:0", ground=False)
        graph_solver = SolverFeatherPGS(graph_model, update_mass_matrix_interval=2)
        eager_solver = SolverFeatherPGS(eager_model, update_mass_matrix_interval=2)
        graph_a, graph_b = _make_initial_state(graph_model), graph_model.state()
        eager_a, eager_b = _make_initial_state(eager_model), eager_model.state()
        graph_control, eager_control = graph_model.control(), eager_model.control()

        def two_substeps(solver, state_in, state_out, control):
            # Keep one input state object across substeps and frames.
            for _ in range(2):
                solver.step(state_in, state_out, control, None, DT)
                for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
                    wp.copy(getattr(state_in, name), getattr(state_out, name))

        # The very first launches of graph_solver happen inside the capture.
        with wp.ScopedCapture("cuda:0") as capture:
            two_substeps(graph_solver, graph_a, graph_b, graph_control)
        for _ in range(64):
            wp.capture_launch(capture.graph)
            two_substeps(eager_solver, eager_a, eager_b, eager_control)
        wp.synchronize_device("cuda:0")

        for name in ("joint_q", "joint_qd"):
            captured = getattr(graph_a, name).numpy()
            self.assertTrue(np.isfinite(captured).all(), f"{name} became non-finite under graph replay")
            np.testing.assert_allclose(captured, getattr(eager_a, name).numpy(), rtol=0.0, atol=2.0e-6, err_msg=name)

    def test_interval_two_contact_trajectory_stays_close_to_reference(self):
        device = wp.get_device()
        history = {}
        for interval in (1, 2):
            model = _build_model(device)
            solver = SolverFeatherPGS(model, update_mass_matrix_interval=interval)
            history[interval] = _run_trajectory(model, solver, num_steps=60)

        self.assertTrue(np.isfinite(history[2]).all(), "interval=2 trajectory diverged to non-finite values")
        # The trajectories are intentionally NOT bit-identical (stale H between
        # mass updates is the optimization); they must stay close on a short
        # contact-rich horizon.
        drift = np.abs(history[2] - history[1]).max()
        self.assertLess(drift, 0.05, f"interval=2 drifted {drift} rad from the interval=1 reference")
        # Sanity: the scene actually moved, so the comparison is not vacuous.
        moved = np.abs(history[1][-1] - np.tile(np.asarray(INITIAL_JOINT_Q, dtype=np.float32), 2)).max()
        self.assertGreater(moved, 1.0e-3)

    def test_global_flag_cadence_bakes_one_zero_pattern(self):
        device = wp.get_device()
        model = _build_model(device)
        solver = SolverFeatherPGS(model, update_mass_matrix_interval=2)
        state_0 = _make_initial_state(model)
        state_1 = model.state()
        contacts = model.contacts()
        control = model.control()

        expected_masks = ([1, 1], [0, 0], [1, 1], [0, 0])
        for step_index, expected in enumerate(expected_masks):
            model.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
            self.assertEqual(
                solver.factor.mass_update_mask.numpy().tolist(),
                expected,
                f"unexpected mass_update_mask after step {step_index}",
            )

    def test_model_change_request_refreshes_a_reuse_step(self):
        model = _build_model(wp.get_device(), ground=False)
        solver = SolverFeatherPGS(model, update_mass_matrix_interval=2)
        state_0 = _make_initial_state(model)
        state_1 = model.state()
        contacts = model.contacts()
        control = model.control()

        model.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, DT)
        state_0, state_1 = state_1, state_0
        solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
        model.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, DT)

        self.assertEqual(solver.factor.mass_update_mask.numpy().tolist(), [1, 1])
        self.assertEqual(solver.factor.mass_update_requested.numpy().tolist(), [0, 0])

    def test_held_tree_route_consumes_mass_requests(self):
        """Refresh the held tree once and consume every request, at the clock's cadence or its changed duration."""
        for changed, step, requested, expected in (
            (True, 1, [0, 0, 0], [1, 1, 1]),
            (False, 0, [0, 0, 0], [1, 1, 1]),
            (False, 1, [1, 1, 1], [1, 1, 1]),
            (False, 1, [0, 0, 0], [0, 0, 0]),
            (False, 1, [0, 1, 0], [0, 1, 0]),
        ):
            with self.subTest(changed=changed, step=step, requested=requested):
                calls = []
                clock = Clock()
                clock.step, clock.dt, clock.interval = step - 1, 2.0 * DT if changed else DT, 2
                factor = SimpleNamespace(
                    model=SimpleNamespace(articulation_count=3, device=wp.get_device("cpu")),
                    clock=wp.array([clock], dtype=Clock, device="cpu"),
                    held=SimpleNamespace(refresh=calls.append),
                    mass_update_requested=wp.array(requested, dtype=int, device="cpu"),
                    mass_update_mask=wp.zeros(3, dtype=int, device="cpu"),
                )
                wp.launch(advance_clock, dim=1, inputs=[factor.clock, DT], device="cpu")
                TreeFactor.refresh(factor, None)
                self.assertEqual(calls, [factor])
                np.testing.assert_array_equal(factor.mass_update_mask.numpy(), expected)
                np.testing.assert_array_equal(factor.mass_update_requested.numpy(), [0, 0, 0])

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA graph capture requires CUDA")
    def test_captured_cadence_follows_the_device_clock(self):
        """Refresh every third step across replays of a two-step graph: the clock counts the replayed steps, where a
        host step index baked into the capture repeated its own two-step pattern (refresh, reuse)."""
        model = _build_model("cuda:0", ground=False)
        solver = SolverFeatherPGS(model, update_mass_matrix_interval=3)
        state_in, state_out, control = _make_initial_state(model), model.state(), model.control()

        def two_steps():
            for _ in range(2):
                solver.step(state_in, state_out, control, None, DT)
                for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
                    wp.copy(getattr(state_in, name), getattr(state_out, name))

        with wp.ScopedCapture("cuda:0") as capture:
            two_steps()
        masks = []
        for _ in range(3):
            wp.capture_launch(capture.graph)
            masks.append(solver.factor.mass_update_mask.numpy().tolist())
        # The second step of each replay: steps 1, 3 and 5; only step 3 refreshes.
        self.assertEqual(masks, [[0, 0], [1, 1], [0, 0]])

    def test_teardown_drains_device_without_accessing_finalized_streams(self):
        """Drain all owned streams without dereferencing finalized handles."""
        solver = object.__new__(SolverFeatherPGS)
        local = object()
        residual = object()
        dynamics = object()
        solver.sweep = SimpleNamespace(
            local_internal_stream=local, local_residual_stream=residual, local_pair_stream=local
        )
        solver.factor = SimpleNamespace(dynamics_stream=dynamics)
        solver.model = mock.Mock(device=object())
        with (
            mock.patch.object(wp, "synchronize_stream", side_effect=AssertionError("finalized handle")),
            mock.patch.object(wp, "synchronize_device") as synchronize_device,
        ):
            solver.__del__()
            synchronize_device.assert_called_once_with(solver.model.device)
        with mock.patch.object(wp, "synchronize_device", side_effect=RuntimeError("CUDA is shutting down")):
            solver.__del__()

        # Prevent a second meaningful synchronization when Python collects the
        # deliberately uninitialized instance after this regression returns.
        solver.sweep = None
        solver.factor.dynamics_stream = None
        solver.model = None

    def test_mass_refresh_has_no_obsolete_limit_count_state(self):
        solver = SolverFeatherPGS(_build_model(wp.get_device(), ground=False))
        for attribute in ("aug_limit_counts", "aug_prev_limit_counts", "limit_change_mask"):
            self.assertFalse(hasattr(solver, attribute), f"obsolete mass-refresh state {attribute!r} was restored")


if __name__ == "__main__":
    unittest.main(verbosity=2)
