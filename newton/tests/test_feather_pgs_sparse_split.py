# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Preserve the ordered split solve when changing its response representation."""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.kernels import PGS_CONSTRAINT_TYPE_CONTACT as CONTACT
from newton._src.solvers.feather_pgs.kernels import PGS_CONSTRAINT_TYPE_FRICTION as FRICTION
from newton._src.solvers.feather_pgs.kernels import PGS_CONSTRAINT_TYPE_JOINT_LIMIT as LIMIT
from newton._src.solvers.feather_pgs.solver_feather_pgs import _get_pgs_solve_tiled_row_kernel
from newton._src.solvers.feather_pgs.sparse_pgs import _get_pgs_solve_sparse_kernel
from newton.solvers import SolverFeatherPGS
from newton.tests import test_feather_pgs_sparse_contact_triplets as triplet_tests
from newton.tests import test_feather_pgs_sparse_solver as sparse_tests
from newton.tests.test_feather_pgs_sparse_pgs import _sweep
from newton.tests.test_feather_pgs_sparse_research_admission import _eligible_solver, _TopologyReached

DT = 1.0 / 400.0
STATE_FIELDS = ("joint_q", "joint_qd", "body_q", "body_qd")


def _probe():
    solver = _eligible_solver()
    solver.pgs_mode = "split"
    solver.pgs_kernel = "tiled_row"
    return solver


def _model(*, floating=True, branches=3, depth=1, device="cuda:0"):
    """Build branched floating or fixed trees with a single contacting foot."""
    scene = newton.ModelBuilder()
    root = scene.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.16), wp.quat_identity()),
        mass=2.0,
        inertia=wp.mat33(np.diag([0.025, 0.03, 0.04])),
    )
    joints = [
        scene.add_joint_free(root)
        if floating
        else scene.add_joint_fixed(-1, root, parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.16), wp.quat_identity()))
    ]
    for branch in range(branches):
        parent = root
        for level in range(depth):
            child = scene.add_link(mass=0.5, inertia=wp.mat33(np.diag([0.002, 0.003, 0.004])))
            joints.append(
                scene.add_joint_revolute(
                    parent,
                    child,
                    axis=newton.Axis.Y,
                    parent_xform=wp.transform(
                        wp.vec3(0.1 if level == 0 else 0.0, 0.12 * branch if level == 0 else 0.0, -0.11 / depth),
                        wp.quat_identity(),
                    ),
                    target_pos=0.03,
                    target_ke=15.0,
                    target_kd=1.0,
                    armature=0.05,
                    limit_lower=-0.2,
                    limit_upper=0.2,
                    effort_limit=10.0,
                )
            )
            if branch == 0 and level == depth - 1:
                cfg = scene.default_shape_cfg.copy()
                cfg.density, cfg.mu, cfg.restitution = 0.0, 0.6, 0.4
                scene.add_shape_sphere(
                    child,
                    xform=wp.transform(wp.vec3(0.02, 0.0, 0.0), wp.quat_identity()),
                    radius=0.055,
                    cfg=cfg,
                )
            parent = child
    scene.add_articulation(joints)
    scene.joint_q[7 if floating else 0] = 0.23
    scene.joint_qd[:] = [0.7] * len(scene.joint_qd)
    if floating:
        scene.joint_qd[:6] = [0.12, -0.08, -0.7, 0.03, -0.04, 0.02]
    builder = newton.ModelBuilder()
    builder.replicate(scene, 2)
    ground = builder.default_shape_cfg.copy()
    ground.restitution = 0.4
    builder.add_ground_plane(cfg=ground)
    return builder.finalize(device=device)


def _solver(model, enabled, **overrides):
    """Change only representation admission between the paired solvers."""
    options = {
        "pgs_mode": "split",
        "pgs_kernel": "tiled_row",
        "pgs_schedule": "interleaved",
        "articulated_contact_response": "immediate",
        "drive_mode": "augmented",
        "friction_mode": "current",
        "friction_anchor_beta": 0.0,
        "pgs_iterations": 8,
        "pgs_velocity_iterations": 0,
        "pgs_warmstart": False,
        "mf_warmstart": False,
        "enable_joint_limits": True,
        "enable_joint_velocity_limits": False,
        "enable_restitution": True,
        "restitution_velocity_threshold": 0.01,
        "dense_max_constraints": 32,
        "mf_max_constraints": 32,
        "use_parallel_streams": True,
        "double_buffer": False,
        "update_mass_matrix_interval": 2,
    }
    options.update(overrides)
    with mock.patch.dict(os.environ, {"FEATHER_PGS_SPARSE_SPLIT": str(int(enabled))}):
        return SolverFeatherPGS(model, **options)


def _case(enabled, **shape):
    model = _model(**shape)
    solver = _solver(model, enabled)
    state, out, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    control.joint_f.assign(np.linspace(-0.08, 0.11, model.joint_dof_count, dtype=np.float32))
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=32)
    return model, solver, state, out, control, pipeline, pipeline.contacts()


@wp.kernel
def _external_forces(force: wp.array[wp.spatial_vector]):
    body = wp.tid()
    force[body] = wp.spatial_vector(0.03, -0.02, -0.07, 0.002, -0.003, 0.001)


def _step(case):
    model, solver, state, out, control, pipeline, contacts = case
    state.clear_forces()
    wp.launch(_external_forces, dim=model.body_count, inputs=[state.body_f], device=model.device)
    pipeline.collide(state, contacts)
    solver.step(state, out, control, contacts, DT)
    for field in STATE_FIELDS:
        wp.copy(getattr(state, field), getattr(out, field))


class TestSparseSplitAdmission(unittest.TestCase):
    def test_opt_in_reaches_topology_without_changing_mode(self):
        """Admit the split recipe before choosing its topology-derived representation."""
        solver = _probe()
        model = SimpleNamespace(device=SimpleNamespace(is_cuda=True), requires_grad=False)
        with (
            mock.patch.dict(os.environ, {"FEATHER_PGS_SPARSE_SPLIT": "1"}, clear=True),
            self.assertRaises(_TopologyReached),
        ):
            SolverFeatherPGS._setup_sparse_mass_matrix(solver, model)
        self.assertEqual(solver.pgs_mode, "split")

    def test_default_and_unsupported_recipes_keep_dense_storage(self):
        """Preserve seeded, alternate-law and mixed-response owners before allocation."""
        model = SimpleNamespace(device=SimpleNamespace(is_cuda=True), requires_grad=False)
        cases = (
            ("default", {}, "0"),
            ("warmstart", {"pgs_warmstart": True}, "1"),
            ("patches", {"_friction_anchors_enabled": True}, "1"),
            ("velocity_limits", {"enable_joint_velocity_limits": True}, "1"),
            ("free_response", {"_has_free_rigid_bodies": True}, "1"),
            ("alternate_kernel", {"pgs_kernel": "loop"}, "1"),
            ("velocity_pass", {"pgs_velocity_iterations": 1}, "1"),
            ("force_drives", {"drive_mode": "force"}, "1"),
            ("compliance", {"contact_compliance": True}, "1"),
        )
        for name, changes, flag in cases:
            with self.subTest(name=name), mock.patch.dict(os.environ, {"FEATHER_PGS_SPARSE_SPLIT": flag}, clear=True):
                solver = _probe()
                for key, value in changes.items():
                    setattr(solver, key, value)
                SolverFeatherPGS._setup_sparse_mass_matrix(solver, model)
                self.assertFalse(solver._sparse_split)
                self.assertIsNone(solver._sparse_mass_matrix_size)
                self.assertEqual(solver.pgs_mode, "split")


@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
class TestSparseSplitPhysics(unittest.TestCase):
    def _assert_state(self, dense, sparse):
        """Use the existing integrated sparse-state physical error allowances."""
        sparse_tests.TestFeatherPGSSparseSolver.assert_states_close(self, dense[2], sparse[2])
        for case in (dense, sparse):
            for field in STATE_FIELDS:
                self.assertTrue(np.isfinite(getattr(case[2], field).numpy()).all(), field)
            self.assertFalse(case[1].constraint_overflow.numpy().any())
        # Check the observable whole-system momentum response to identical external loads.
        momentum = [
            (case[0].body_mass.numpy()[:, None] * case[2].body_qd.numpy()[:, :3]).sum(axis=0)
            for case in (dense, sparse)
        ]
        np.testing.assert_allclose(momentum[1], momentum[0], atol=2e-3, rtol=2e-4)

    def _assert_rows(self, dense, sparse):
        """Compare the physical operator and residuals in the same ordered live rows."""
        old, new = dense[1], sparse[1]
        counts = new.constraint_count.numpy()
        np.testing.assert_array_equal(counts, old.constraint_count.numpy())
        np.testing.assert_array_equal(new._sparse_mass_matrix_status.numpy(), 0)
        size = new._sparse_mass_matrix_size
        old_j = old.J_by_size[size].numpy()
        groups = old.group_to_art[size].numpy()
        worlds = old.art_to_world.numpy()[groups]
        for world, live_count in enumerate(counts):
            count = int(live_count)
            z, physical_j = sparse_tests.TestFeatherPGSSparseSolver._sparse_rows_physical(self, new, world, count)
            old_group = int(np.flatnonzero(worlds == world)[0])
            np.testing.assert_allclose(physical_j, old_j[old_group, :count], atol=3e-5, rtol=3e-4)
            for name in ("row_type", "row_parent"):
                np.testing.assert_array_equal(
                    getattr(new, name).numpy()[world, :count], getattr(old, name).numpy()[world, :count]
                )
            np.testing.assert_allclose(
                new.diag.numpy()[world, :count], old.diag.numpy()[world, :count], atol=3e-5, rtol=3e-4
            )
            np.testing.assert_allclose(z @ z.T, old.C.numpy()[world, :count, :count], atol=3e-5, rtol=3e-4)
            rhs = new.rhs.numpy()[world, :count]
            incident = new._sparse_row_incident.numpy()[world, :count]
            np.testing.assert_allclose(rhs + incident, old.rhs.numpy()[world, :count], atol=3e-5, rtol=3e-4)
            dofs = new.world_dof_indices.numpy()[world]
            result = new.v_out.numpy()[dofs]
            residual = physical_j @ result + rhs
            expected_residual = (
                old_j[old_group, :count] @ old.v_out.numpy()[dofs]
                + old.rhs.numpy()[world, :count]
                - old_j[old_group, :count] @ old.v_hat.numpy()[dofs]
            )
            np.testing.assert_allclose(residual, expected_residual, atol=5e-5, rtol=5e-4)
            impulse = new.impulses.numpy()[world, :count]
            kinds, parents, mu = (
                getattr(new, name).numpy()[world, :count] for name in ("row_type", "row_parent", "row_mu")
            )
            self.assertTrue(np.all(impulse[(kinds == CONTACT) | (kinds == LIMIT)] >= -1e-7))
            for row in np.flatnonzero(kinds == FRICTION):
                parent = int(parents[row])
                if row == parent + 1:
                    self.assertLessEqual(np.linalg.norm(impulse[row : row + 2]), mu[row] * impulse[parent] + 2e-5)

    def test_branched_floating_and_fixed_physical_step(self):
        """Preserve eight ordered sweeps, restitution, drives and external forces."""
        for shape, dofs in (
            ({"floating": True, "branches": 3, "depth": 1}, 9),
            ({"floating": False, "branches": 2, "depth": 2}, 4),
        ):
            with self.subTest(shape=shape):
                dense, sparse = _case(False, **shape), _case(True, **shape)
                self.assertFalse(dense[1]._sparse_split)
                self.assertTrue(sparse[1]._sparse_split)
                self.assertEqual(sparse[1]._sparse_mass_matrix_size, dofs)
                self.assertIsNone(sparse[1].C)
                self.assertEqual(sparse[1].pgs_mode, "split")
                self.assertEqual(sparse[1].pgs_iterations, 8)
                for _ in range(4):
                    _step(dense)
                    _step(sparse)
                    self._assert_state(dense, sparse)
                self._assert_rows(dense, sparse)
                count = int(sparse[1].constraint_count.numpy()[0])
                kinds = sparse[1].row_type.numpy()[0, :count]
                self.assertIn(CONTACT, kinds)
                self.assertIn(LIMIT, kinds)
                self.assertTrue(np.any(sparse[1].row_restitution.numpy()[0, :count] > 0.0))

    def test_held_factor_masked_reset_and_inertial_change(self):
        """Refresh only reset worlds and preserve the same held-factor epoch."""
        dense, sparse = _case(False), _case(True)
        self.assertTrue(sparse[1]._sparse_split)
        for case in (dense, sparse):
            _step(case)
        held = sparse[1]._sparse_Linv.numpy().copy()
        for model, solver, state, *_ in (dense, sparse):
            q = state.joint_q.numpy().copy()
            q[2] += 0.002
            state.joint_q.assign(q)
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
            solver.reset(state, wp.array([True, False], dtype=bool, device=model.device))
        for case in (dense, sparse):
            _step(case)
        np.testing.assert_array_equal(sparse[1].mass_update_mask.numpy(), [1, 0])
        np.testing.assert_array_equal(sparse[1]._sparse_Linv.numpy()[1], held[1])
        self._assert_state(dense, sparse)
        self._assert_rows(dense, sparse)
        for model, solver, *_ in (dense, sparse):
            mass = model.body_mass.numpy().copy()
            mass[0] *= 1.1
            model.body_mass.assign(mass)
            solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
        for case in (dense, sparse):
            _step(case)
        np.testing.assert_array_equal(sparse[1].mass_update_mask.numpy(), [1, 1])
        self._assert_state(dense, sparse)

    def test_graph_refresh_and_held_epoch(self):
        """Replay refresh/held factors as contacting shapes leave and reenter the scene."""
        eager, captured = _case(True), _case(True)
        self.assertTrue(captured[1]._sparse_split)
        for case in (eager, captured):
            _step(case)
        with wp.ScopedCapture("cuda:0") as capture:
            _step(captured)
            _step(captured)
        initial_q = [case[2].joint_q.numpy().copy() for case in (eager, captured)]
        for height in (0.0, 2.0, 0.0):
            for case, initial in zip((eager, captured), initial_q, strict=True):
                model, solver, state, *_ = case
                q = initial.copy()
                q.reshape(2, -1)[:, 2] += height
                state.joint_q.assign(q)
                newton.eval_fk(model, state.joint_q, state.joint_qd, state)
                solver.reset(state)
            _step(eager)
            _step(eager)
            wp.capture_launch(capture.graph)
            self._assert_state(eager, captured)
            contact_count = int(captured[-1].rigid_contact_count.numpy()[0])
            if height:
                self.assertEqual(contact_count, 0)
            else:
                self.assertGreater(contact_count, 0)

    def test_prescribed_dof_notification_requires_reconstruction(self):
        """Reject a newly prescribed response before publishing incompatible cached masks."""
        case = _case(True)
        model, solver, *_ = case
        self.assertTrue(solver._sparse_split)
        _step(case)
        old_mask = solver._kinematic_dof_mask_host.copy()
        flags = model.body_flags.numpy().copy()
        flags[0] |= int(newton.BodyFlags.KINEMATIC)
        model.body_flags.assign(flags)
        with self.assertRaisesRegex(RuntimeError, "Sparse response gained prescribed DOFs"):
            solver.notify_model_changed(newton.ModelFlags.BODY_PROPERTIES)
        np.testing.assert_array_equal(solver._kinematic_dof_mask_host, old_mask)

    def test_unsupported_actual_models_and_seeded_modes_fall_back(self):
        """Keep dense allocation for mixed, prescribed and carried-impulse recipes."""
        model = _model()
        for overrides in ({"pgs_warmstart": True}, {"friction_anchor_beta": 0.2}):
            with self.subTest(overrides=overrides):
                solver = _solver(model, True, **overrides)
                self.assertFalse(solver._sparse_split)
                self.assertIsNone(solver._sparse_mass_matrix_size)
                self.assertIsNotNone(solver.C)
                self.assertEqual(solver.pgs_mode, "split")
        with self.assertRaises(NotImplementedError):
            _solver(model, True, enable_joint_velocity_limits=True)
        for kwargs in ({"free_count": 1}, {"robot_count": 2}, {"prescribed": True}):
            with self.subTest(model=kwargs):
                mixed = sparse_tests._build_model(**kwargs)
                solver = _solver(mixed, True)
                self.assertFalse(solver._sparse_split)
                self.assertIsNone(solver._sparse_mass_matrix_size)
                self.assertIsNotNone(solver.C)

    def test_full_order_dense_and_physical_oracle(self):
        """Visit newly violated rows in the current sweep and retain the paired friction law."""
        for activity, iterations, delay in (
            ("deferred_activation", 1, 0),
            ("deferred_activation", 8, 0),
            (None, 8, 0),
            (None, 8, 2),
        ):
            with self.subTest(activity=activity, iterations=iterations, delay=delay):
                data, physical = triplet_tests.fixture(activity=activity, delay=delay)
                # Keep the physical fixture; bound this direct solve to ordinary small storage.
                for name, array in data.items():
                    if array.ndim > 1 and array.shape[1] == 192:
                        data[name] = array[:, :32].copy()
                data["iterations"] = np.array(iterations)

                def full_order(capacity, dofs, width, **_kwargs):
                    return _get_pgs_solve_sparse_kernel(
                        capacity, dofs, width, active_rows=False, use_support_table=True
                    )

                with mock.patch.object(triplet_tests, "factory", full_order):
                    solve = triplet_tests.DeviceSolve(data, candidate=False)
                solve.launch()
                delta, impulses = solve.result()
                count = int(data["world_constraint_count"][0])
                j, response = physical["jacobian"], physical["response"]
                expected_velocity, expected_impulse = _sweep(
                    j,
                    response,
                    physical["initial"],
                    data["rhs_bias"][0, :count],
                    data["diagonal"][0, :count],
                    data["row_type"][0, :count],
                    data["row_parent"][0, :count],
                    data["row_mu"][0, :count],
                    physical["initial_impulse"],
                    iterations=iterations,
                    friction_start=delay,
                    active_rows=False,
                )
                dense_c = np.zeros((3, 32, 32), np.float32)
                dense_c[:, :count, :count] = j @ response.T
                dense_impulses = wp.array(data["initial_impulses"], device="cuda:0")
                kernel = _get_pgs_solve_tiled_row_kernel(32, str(wp.get_device("cuda:0").arch))
                wp.launch_tiled(
                    kernel,
                    dim=[3],
                    block_dim=32,
                    device="cuda:0",
                    inputs=[
                        solve.inputs[1],
                        solve.inputs[3],
                        wp.array(dense_c, device="cuda:0"),
                        wp.array(data["rhs_bias"] + data["row_incident"], device="cuda:0"),
                        dense_impulses,
                        iterations,
                        1.0,
                        solve.inputs[9],
                        solve.inputs[10],
                        solve.inputs[11],
                        delay,
                        0,
                    ],
                )
                np.testing.assert_allclose(impulses, dense_impulses.numpy(), atol=3e-5, rtol=3e-4)
                velocity = physical["initial"] + np.linalg.solve(physical["factor"].T, delta[0])
                np.testing.assert_allclose(velocity, expected_velocity, atol=3e-5, rtol=3e-4)
                np.testing.assert_allclose(impulses[0, :count], expected_impulse, atol=3e-5, rtol=3e-4)
                if activity == "deferred_activation" and iterations == 1:
                    self.assertGreater(impulses[0, 3], 0.0)


if __name__ == "__main__":
    unittest.main()
