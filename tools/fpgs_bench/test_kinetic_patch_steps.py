# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Complete CUDA steps composing native held dynamics with shared patch friction."""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.solver_feather_pgs import SolverFeatherPGS
from tools.fpgs_bench.test_kinetic_live_bindings import live_model

DT = 1.0 / 240.0


def make_case(enabled, *, beta=0.2, sparse="off", cached=None, iterations=8, capacity=192):
    """Exercise dense, coupled, and MF contacts through the real CUDA constructor."""
    model = live_model(3, device="cuda:0", shapes=True)
    solver = SolverFeatherPGS(
        model,
        pgs_mode="matrix_free",
        pgs_iterations=iterations,
        dense_max_constraints=capacity,
        mf_max_constraints=64,
        enable_joint_limits=True,
        joint_limit_activation_gap=0.0,
        fuse_joint_velocity_limits=False,
        mf_gs_incremental_rows=0,
        update_mass_matrix_interval=2,
        use_parallel_streams=True,
        friction_anchor_beta=beta,
        cached_dynamics=cached if cached is not None else "on" if enabled else "off",
        sparse_mass_matrix=sparse,
    )
    state, output, control = model.state(), model.state(), model.control()
    state.joint_qd.assign(np.linspace(-0.0008, 0.0012, model.joint_dof_count, dtype=np.float32))
    state.body_f.assign(np.random.default_rng(101).normal(0, 0.05, (model.body_count, 6)).astype(np.float32))
    output.body_f.assign(state.body_f)
    control.joint_f.assign(np.random.default_rng(102).normal(0, 0.07, model.joint_dof_count).astype(np.float32))
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    contacts = newton.Contacts(rigid_contact_max=256, soft_contact_max=0, device=model.device)
    first, second = np.full(256, -1, np.int32), np.full(256, -1, np.int32)
    point0, point1, normal = (np.zeros((256, 3), np.float32) for _ in range(3))
    poses = state.body_q.numpy()
    # Four contacts in each region force normal-only rows alongside two anchors.
    for region, (world, a, b) in enumerate(((0, 13, 17), (1, 13, 30), (2, 30, 31), (1, 30, 31))):
        body0, body1 = world * 32 + a, world * 32 + b
        for i, yz in enumerate(((-0.01, -0.005), (-0.01, 0.005), (0.01, -0.005), (0.01, 0.005))):
            c = region * 4 + i
            first[c], second[c] = body0, body1
            normal[c] = [1, 0, 0]
            point0[c] = [0, *yz]
            world_point = wp.transform_point(wp.transform(*poses[body0]), wp.vec3(*point0[c]))
            point1[c] = np.asarray(
                wp.transform_point(
                    wp.transform_inverse(wp.transform(*poses[body1])), world_point - wp.vec3(0.001, 0, 0)
                )
            )
    for name, values in (
        ("shape0", first),
        ("shape1", second),
        ("point0", point0),
        ("point1", point1),
        ("normal", normal),
    ):
        getattr(contacts, "rigid_contact_" + name).assign(values)
    contacts.rigid_contact_count.fill_(16)
    return model, solver, state, output, control, contacts


def step(case):
    model, solver, state, output, control, contacts = case
    solver.step(state, output, control, contacts, DT)
    return model, solver, output, state, control, contacts


@unittest.skipUnless(wp.is_cuda_available(), "CUDA unavailable")
class TestKineticPatchSteps(unittest.TestCase):
    def test_explicit_cache_precedes_sparse_storage_for_compatible_patch_recipe(self):
        """Honor explicit native caching while retaining the faster automatic route."""
        # Regression: the old allocation order admitted sparse factors first,
        # preventing the otherwise-compatible native cache from ever being tried.
        from newton._src.solvers.feather_pgs import cached_dynamics  # noqa: PLC0415

        with patch.object(cached_dynamics, "reserve_native_patch_factors", return_value=False):
            with self.assertRaisesRegex(ValueError, "no admitted cache"):
                make_case(True, sparse="auto", cached="on")
        automatic = make_case(True, sparse="auto", cached="auto")
        self.assertEqual(automatic[1]._sparse_mass_matrix_size, 23)
        self.assertIsNone(automatic[1]._kinetic_world)
        self.assertIsNone(make_case(True, sparse="off", cached="auto")[1]._kinetic_world)
        case = make_case(True, sparse="auto", cached="on")
        solver = case[1]
        self.assertIsNone(solver._sparse_mass_matrix_size)
        self.assertIsNotNone(solver._kinetic_world)
        self.assertIs(solver._kinetic_world.host_plan, solver._kinetic_world_plan)
        self.assertIn("native patch-dynamics", solver.get_path_selection()["reasons"]["sparse_mass_matrix"])
        cases = [automatic, case]
        for _ in range(4):
            cases = [step(value) for value in cases]
            self.assert_pair(cases)
        for options in (
            {"sparse": "on", "cached": "auto"},
            {"sparse": "auto", "cached": "off"},
            {"sparse": "auto", "cached": "auto", "iterations": 4},
            {"sparse": "auto", "cached": "auto", "capacity": 256},
        ):
            with self.subTest(options=options):
                other = make_case(False, **options)[1]
                self.assertEqual(other._sparse_mass_matrix_size, 23)
                self.assertIsNone(other._kinetic_world)

    def assert_pair(self, cases, *, cached=True):
        cone_errors = []
        for case in cases:
            errors = np.zeros((2, 3))
            solver = case[1]
            solver.check_constraint_capacity()
            for prefix in ("", "mf_"):
                counts = getattr(solver, prefix + "constraint_count").numpy()
                impulses = getattr(solver, prefix + "impulses").numpy()
                kinds = getattr(solver, prefix + "row_type").numpy()
                parents = getattr(solver, prefix + "row_parent").numpy()
                mu = getattr(solver, prefix + "row_mu").numpy()
                for world, count in enumerate(counts):
                    self.assertTrue(np.isfinite(impulses[world, :count]).all())
                    for row in range(count):
                        if kinds[world, row] in (0, 3):
                            self.assertGreaterEqual(float(impulses[world, row]), -1e-7)
                        if kinds[world, row] != 2 or row != parents[world, row] + 1:
                            continue
                        normal = parents[world, row]
                        load = float(impulses[world, normal])
                        linked = parents[world, normal]
                        visited = {normal}
                        while linked >= 0 and linked != normal:
                            self.assertNotIn(int(linked), visited)
                            visited.add(int(linked))
                            load += float(impulses[world, linked])
                            linked = parents[world, linked]
                        radius = max(0, mu[world, row] * load)
                        tangent = np.linalg.norm(impulses[world, row : row + 2])
                        # A later normal update can alter this anchor's budget
                        # in the fixed eight-sweep law. Compare its residual
                        # with the ordinary solver, without adding iterations.
                        family = int(bool(prefix))
                        errors[family, world] = max(errors[family, world], (tangent - radius) / (1 + radius))
            cone_errors.append(errors)
        self.assertTrue(np.all(cone_errors[1] <= cone_errors[0] + 2e-5), cone_errors)
        self.assertEqual(cases[1][1]._kinetic_world.last_private, cached)
        for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
            a, b = (getattr(case[2], name).numpy().reshape(3, -1) for case in cases)
            error = np.linalg.norm(a - b, axis=1) / (1 + np.linalg.norm(a, axis=1))
            self.assertLess(float(error.max()), 7e-4, name)

    def test_complete_loaded_step(self):
        """Preserve loaded patch impulses, coupling and persistent anchor history."""
        cases = [make_case(enabled) for enabled in (False, True)]
        self.assertIsNotNone(cases[1][1]._kinetic_world)
        cases = [step(case) for case in cases]
        self.assert_pair(cases)
        for case in cases:
            solver = case[1]
            patches = solver._friction_patches
            np.testing.assert_array_equal(solver.constraint_count.numpy(), [8, 8, 0])
            np.testing.assert_array_equal(solver.mf_constraint_count.numpy(), [0, 8, 8])
            self.assertEqual(np.count_nonzero(patches.view.weight.numpy()), 8)
            self.assertGreater(np.count_nonzero(patches.previous.valid.numpy()), 0)
            self.assertGreater(float(np.linalg.norm(patches.previous.tangent_impulse.numpy())), 1e-6)
            self.assertGreater(float(solver.impulses.numpy()[1].sum()), 0)
            self.assertGreater(float(np.linalg.norm(solver.mf_impulses.numpy())), 0)
        self.assertEqual(cases[1][1]._kinetic_world.bindings.hybrid_eligible.numpy()[1], 1)
        consumed_history = False
        for _ in range(3):
            cases = [step(case) for case in cases]
            self.assert_pair(cases)
            consumed_history |= bool(np.any(cases[1][1]._friction_patches.current.source.numpy() >= 0))
        self.assertTrue(consumed_history)

    def test_held_factors_resets_and_model_changes(self):
        """Reuse held factors and repair them after resets and inertial changes."""
        cases = [make_case(enabled) for enabled in (False, True)]
        held = None
        for index in range(24):
            for arm, case in enumerate(cases):
                model, solver, state, *_ = case
                if index in (5, 13, 19):
                    mask = wp.array([index == 5, index == 13, index == 19], dtype=wp.bool, device=model.device)
                    solver.reset(state, world_mask=mask)
                if index == 9:
                    model.body_mass.assign(model.body_mass.numpy() * np.float32(1.01))
                    model.body_inertia.assign(model.body_inertia.numpy() * np.float32(1.02))
                    solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
                cases[arm] = step(case)
                if arm and index == 0:
                    held = solver._kinetic_world.held.T.numpy().copy()
                if arm and index == 1:
                    np.testing.assert_array_equal(solver._kinetic_world.held.T.numpy(), held)
            self.assert_pair(cases)

    def test_graph_replay_with_dynamic_reset_masks(self):
        """Replay captured steps with selective and complete world resets."""
        cases = [make_case(enabled) for enabled in (False, True)]
        graphs, masks = [], []
        for arm in range(len(cases)):
            case = cases[arm]
            for _ in range(4):
                case = step(case)
            mask = wp.array([False] * 3, dtype=wp.bool, device=case[0].device)
            wp.synchronize()
            with wp.ScopedCapture(device=case[0].device) as capture:
                case[1].seed_double_buffer_events()
                case[1].reset(case[2], world_mask=mask)
                for _ in range(4):
                    case = step(case)
            cases[arm] = case
            graphs.append(capture.graph)
            masks.append(mask)
        for selected in ([True, False, False], [False, True, False], [False, False, True], [False] * 3, [True] * 3):
            for mask, graph in zip(masks, graphs, strict=True):
                mask.assign(np.array(selected))
                wp.capture_launch(graph)
            self.assert_pair(cases)

    def test_point_contact_control(self):
        """Preserve the existing native point-contact result."""
        cases = [make_case(enabled, beta=0.0) for enabled in (False, True)]
        for _ in range(4):
            cases = [step(case) for case in cases]
            self.assert_pair(cases)

    def test_no_contact_call_demotes_and_reenters(self):
        """Clear patch history and reenter cached dynamics after contact-free calls."""
        cases = [make_case(enabled) for enabled in (False, True)]
        for _ in range(4):
            cases = [step(case) for case in cases]
        for arm, (model, solver, state, output, control, contacts) in enumerate(cases):
            solver.step(state, output, control, None, DT)
            cases[arm] = model, solver, output, state, control, contacts
            self.assertFalse(np.any(solver._friction_patches.previous.valid.numpy()))
        self.assert_pair(cases, cached=False)
        # The old held factors resume only on the next scheduled refresh.
        cases = [step(case) for case in cases]
        self.assert_pair(cases, cached=False)
        cases = [step(case) for case in cases]
        self.assert_pair(cases)


if __name__ == "__main__":
    unittest.main()
