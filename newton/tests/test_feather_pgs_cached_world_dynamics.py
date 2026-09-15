# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare cached and original K1 with the same two-substep publication cadence."""

import inspect
import os
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import solver_feather_pgs as implementation
from newton._src.solvers.feather_pgs.fused_dynamics import get_fused_dynamics_kernel
from newton.solvers import SolverFeatherPGS

DT = 0.0025
WORLDS = 5


def _build_model(device):
    """Extend the existing floating-COM fixture pattern to four three-hinge legs."""
    template = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    root = template.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
        mass=3.0,
        com=wp.vec3(0.07, -0.03, 0.02),
        inertia=wp.mat33(0.3, 0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.5),
    )
    joints = [template.add_joint_free(root)]
    for leg in range(4):
        parent = root
        for depth in range(3):
            child = template.add_link(
                mass=0.4,
                com=wp.vec3(0.02, -0.01, -0.05),
                inertia=wp.mat33(0.02, 0.0, 0.0, 0.0, 0.025, 0.0, 0.0, 0.0, 0.03),
            )
            anchor = wp.vec3(0.18 if leg < 2 else -0.18, 0.14 if leg % 2 else -0.14, -0.06)
            joint = template.add_joint_revolute(
                parent,
                child,
                parent_xform=wp.transform(anchor if depth == 0 else wp.vec3(0.0, 0.0, -0.15), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(0.02, 0.0, 0.06), wp.quat_identity()),
                axis=newton.Axis.X if depth == 0 else newton.Axis.Y,
                armature=0.06,
                target_ke=15.0,
                target_kd=1.5,
            )
            dof = template.joint_qd_start[joint]
            template.joint_spring_stiffness[dof] = 0.4
            template.joint_spring_ref[dof] = 0.12
            template.joint_damping[dof] = 0.08
            joints.append(joint)
            parent = child
    template.add_articulation(joints)
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    builder.replicate(template, WORLDS)
    return builder.finalize(device=device)


@wp.kernel
def _reset_selected(
    initial_q: wp.array[float],
    initial_qd: wp.array[float],
    mask: wp.array[bool],
    q: wp.array[float],
    qd: wp.array[float],
):
    index = wp.tid()
    world = index // 19
    local = index % 19
    if mask[world]:
        q[index] = initial_q[index]
        if local < 18:
            qd[world * 18 + local] = initial_qd[world * 18 + local]


def _make_case(device, cached):
    model = _build_model(device)
    with mock.patch.object(implementation, "_FUSED_K1_CACHE_ON", cached):
        solver = SolverFeatherPGS(
            model,
            pgs_mode="matrix_free",
            grouped_dynamics=True,
            lazy_kinematics=True,
            use_parallel_streams=True,
            double_buffer=False,
            update_mass_matrix_interval=2,
            pgs_iterations=8,
            dense_max_constraints=24,
            mf_max_constraints=8,
        )
    state, out = model.state(), model.state()
    q = state.joint_q.numpy().reshape(WORLDS, 19)
    q[:, 7:] = 0.1 * np.sin(np.arange(12))
    qd = state.joint_qd.numpy().reshape(WORLDS, 18)
    qd[:] = 0.03 * np.cos(np.arange(18))
    qd[:, :6] = (0.2, -0.1, 0.05, 0.3, -0.2, 0.1)
    state.joint_q.assign(q.reshape(-1))
    state.joint_qd.assign(qd.reshape(-1))
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return SimpleNamespace(
        model=model,
        solver=solver,
        state=state,
        out=out,
        control=model.control(),
        initial_q=wp.clone(state.joint_q),
        initial_qd=wp.clone(state.joint_qd),
        before=wp.empty_like(solver._fk_id_cache_valid),
        after=wp.empty_like(solver._fk_id_cache_valid),
        first_L=wp.empty_like(solver.L_by_size[18]),
    )


def _author_forces(case, phase):
    factor = 1.0 if phase % 2 == 0 else -0.7
    force = np.zeros((case.model.body_count, 6), dtype=np.float32)
    force[:, 0] = factor * 0.3
    force[:, 2] = 0.15 + phase * 0.01
    force[:, 4] = factor * 0.04
    for state in (case.state, case.out):
        state.body_f.assign(force)
    joint_f = np.zeros((WORLDS, 18), dtype=np.float32)
    joint_f[:, :6] = factor * np.asarray((0.1, -0.2, 0.3, 0.02, -0.03, 0.04))
    joint_f[:, 6:] = factor * 0.2 * np.sin(np.arange(12) + 0.4)
    case.control.joint_f.assign(joint_f.reshape(-1))
    target_q = case.control.joint_target_q.numpy()
    width = target_q.size // WORLDS
    target_q.reshape(WORLDS, width)[:, -12:] = factor * 0.15
    case.control.joint_target_q.assign(target_q)
    target_qd = np.zeros((WORLDS, 18), dtype=np.float32)
    target_qd[:, 6:] = factor * 0.07
    case.control.joint_target_qd.assign(target_qd.reshape(-1))


def _tick(case):
    """Use exactly two solver calls and one explicit lazy publication."""
    solver = case.solver
    wp.copy(case.before, solver._fk_id_cache_valid)
    solver.step(case.state, case.out, case.control, None, DT)
    wp.copy(case.after, solver._fk_id_cache_valid)
    wp.copy(case.first_L, solver.L_by_size[18])
    solver.step(case.out, case.state, case.control, None, DT)
    solver.publish_kinematics(case.state)


class TestFeatherPGSCachedWorldDynamics(unittest.TestCase):
    def _cases(self):
        if not wp.is_cuda_available():
            self.skipTest("Actual fused K1 requires CUDA")
        device = wp.get_device(os.environ.get("FPGS_TEST_DEVICE", "cuda:0"))
        self.enterContext(
            mock.patch.multiple(
                implementation,
                _FUSED_K1_OFF=False,
                _FUSED_K1_MASS_ON=False,
                _GROUPED_CHECK=False,
                _K1_REAL_CHECK=False,
                _GROUPED_FK_OFF=False,
                _GROUPED_TAU_MASS_ON=False,
                _GROUPED_MASS_ON=False,
                _FK_ID_CACHE_OFF=False,
            )
        )
        cases = [_make_case(device, cached) for cached in (False, True)]
        for cached, case in zip((False, True), cases, strict=True):
            solver = case.solver
            self.assertTrue(solver._fused_k1)
            self.assertEqual(solver._fused_k1_cached, cached)
            self.assertTrue(solver._fk_id_cache_enabled)
            self.assertFalse(solver._fk_id_cache_uses_snapshot)
            self.assertTrue(solver.lazy_kinematics)
            self.assertEqual(tuple(solver.size_groups), (18,))
        return cases

    def _compare(self, cases):
        baseline, candidate = cases
        for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
            np.testing.assert_allclose(
                getattr(candidate.state, name).numpy(),
                getattr(baseline.state, name).numpy(),
                rtol=3e-6,
                atol=3e-6,
                err_msg=name,
            )
        for name in ("joint_tau", "body_f_s", "joint_S_s", "articulation_origin", "v_hat"):
            np.testing.assert_allclose(
                getattr(candidate.solver, name).numpy(),
                getattr(baseline.solver, name).numpy(),
                rtol=3e-5,
                atol=5e-6,
                err_msg=name,
            )
        np.testing.assert_allclose(
            candidate.solver.L_by_size[18].numpy(), baseline.solver.L_by_size[18].numpy(), rtol=3e-5, atol=5e-6
        )
        np.testing.assert_array_equal(candidate.after.numpy(), 0)
        for case in cases:
            np.testing.assert_array_equal(case.solver._fk_id_cache_valid.numpy(), 1)
            self.assertIs(case.solver._fk_id_cache_source_state, case.state)
            case.solver.check_constraint_capacity()

    def test_cached_k1_matches_live_forces(self):
        """Match cold and warm dynamics with current body, joint, passive and PD forces."""
        self.assertIn("reuse_cached", inspect.signature(get_fused_dynamics_kernel).parameters)
        cases = self._cases()
        initial_qd = cases[0].state.joint_qd.numpy().copy()
        for phase in range(3):
            for case in cases:
                _author_forces(case, phase)
                with mock.patch.object(case.solver, "_launch_fused_k1", wraps=case.solver._launch_fused_k1) as calls:
                    _tick(case)
                self.assertEqual(calls.call_count, 2)
                np.testing.assert_array_equal(case.first_L.numpy(), case.solver.L_by_size[18].numpy())
            np.testing.assert_array_equal(cases[1].before.numpy(), int(phase > 0))
            self._compare(cases)
        self.assertGreater(np.max(np.abs(cases[1].state.joint_qd.numpy() - initial_qd)), 1e-3)
        # Pointer identity cannot invalidate the intermediate lazy state when
        # both integration outputs alias their inputs.
        for case in cases:
            case.out = case.state
            _author_forces(case, 4)
            _tick(case)
        self._compare(cases)

    def test_captured_mixed_reset(self):
        """Replay mixed same-warp resets without changing the two-substep lazy cadence."""
        cases = self._cases()
        graphs = []
        masks = []
        for case in cases:
            _author_forces(case, 0)
            _tick(case)
            mask = wp.zeros(WORLDS, dtype=wp.bool, device=case.model.device)

            def reset_and_tick(case, mask):
                wp.launch(
                    _reset_selected,
                    dim=WORLDS * 19,
                    inputs=[case.initial_q, case.initial_qd, mask, case.state.joint_q, case.state.joint_qd],
                    device=case.model.device,
                )
                case.solver.reset(case.state, mask)
                _tick(case)

            reset_and_tick(case, mask)
            with wp.ScopedCapture(device=case.model.device) as capture:
                reset_and_tick(case, mask)
            graphs.append(capture.graph)
            masks.append(mask)
        for phase, selected in enumerate(
            ((False,) * WORLDS, (True, False, True, False, False), (False, True, False, True, True))
        ):
            for case, mask, graph in zip(cases, masks, graphs, strict=True):
                _author_forces(case, phase + 3)
                mask.assign(np.asarray(selected, dtype=np.bool_))
                wp.capture_launch(graph)
            np.testing.assert_array_equal(cases[1].before.numpy(), np.logical_not(selected).astype(np.int32))
            self._compare(cases)

    def test_notified_mass_and_state_identity(self):
        """Refresh notified inertias on a held-mass substep and reject foreign state identity."""
        cases = self._cases()
        for case in cases:
            _author_forces(case, 0)
            _tick(case)
            case.solver.step(case.state, case.out, case.control, None, DT)
            wp.copy(case.after, case.solver._fk_id_cache_valid)
            mass = case.model.body_mass.numpy()
            mass[1::13] *= 1.2
            case.model.body_mass.assign(mass)
            inertia = case.model.body_inertia.numpy()
            inertia[1::13] *= 1.1
            case.model.body_inertia.assign(inertia)
            com = case.model.body_com.numpy()
            com[1::13] += np.asarray((0.02, -0.01, 0.015), dtype=np.float32)
            case.model.body_com.assign(com)
            case.solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
            np.testing.assert_array_equal(case.solver._fk_id_cache_valid.numpy(), 0)
            case.solver.step(case.out, case.state, case.control, None, DT)
            case.solver.publish_kinematics(case.state)
            np.testing.assert_array_equal(case.solver.mass_update_mask.numpy(), 1)
            np.testing.assert_array_equal(case.solver._mass_update_requested.numpy(), 0)
        self._compare(cases)
        for case in cases:
            foreign = case.model.state()
            wp.copy(foreign.joint_q, case.state.joint_q)
            qd = case.state.joint_qd.numpy()
            qd[::18] += 0.03
            foreign.joint_qd.assign(qd)
            self.assertIsNot(case.solver._fk_id_cache_source_state, foreign)
            case.state, case.out = foreign, case.model.state()
            _author_forces(case, 4)
            _tick(case)
        self._compare(cases)


if __name__ == "__main__":
    unittest.main()
