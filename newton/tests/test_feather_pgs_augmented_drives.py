# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.sim.enums import JointType
from newton._src.solvers.feather_pgs import field_group
from newton._src.solvers.feather_pgs.solver_fields import DriveRows
from newton._src.solvers.feather_pgs.tree_factor import prepare_augmented_drives


def _mixed_target_model(coord_layout, device):
    """Place scalar drives before and after free/ball joints in two worlds."""
    with patch.object(newton, "use_coord_layout_targets", coord_layout):
        scene = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        for joint_kind in ("revolute", "free", "ball", "revolute"):
            body = scene.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
            joint = getattr(scene, f"add_joint_{joint_kind}")(parent=-1, child=body)
            scene.add_articulation([joint])
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        builder.replicate(scene, 2)
        model = builder.finalize(device=device)
    joints = np.flatnonzero(model.joint_type.numpy() == int(JointType.REVOLUTE))
    dofs = model.joint_qd_start.numpy()[joints]
    coords = model.joint_q_start.numpy()[joints]
    targets = model.joint_target_q_start.numpy()[joints]
    state, control = model.state(), model.control()
    q, qd = state.joint_q.numpy().copy(), state.joint_qd.numpy().copy()
    q[coords], qd[dofs] = [0.04, 0.05, 0.06, 0.07], [0.06, -0.05, 0.04, -0.03]
    state.joint_q.assign(q)
    state.joint_qd.assign(qd)
    target_q, target_qd = control.joint_target_q.numpy().copy(), control.joint_target_qd.numpy().copy()
    target_q[targets], target_qd[dofs] = [0.2, -0.3, 0.4, -0.5], [0.12, -0.07, 0.03, -0.11]
    control.joint_target_q.assign(target_q)
    control.joint_target_qd.assign(target_qd)
    ke, kd = np.zeros(model.joint_dof_count), np.zeros(model.joint_dof_count)
    ke[dofs], kd[dofs] = 20.0, 3.0
    model.joint_target_ke.assign(ke)
    model.joint_target_kd.assign(kd)
    model.joint_effort_limit.fill_(100.0)
    return model, state, control, joints, dofs, coords, targets


def _drive_rows(row_K, articulations, dofs, device):
    """The arrays a prepare kernel writes around ``row_K``, with no last step and no drive at its limit yet."""
    return field_group.upload(
        DriveRows,
        device,
        row_K=row_K,
        row_dv=wp.zeros_like(row_K),
        drive=wp.zeros(dofs, dtype=wp.vec4, device=device),
        mass_update_requested=wp.zeros(articulations, dtype=int, device=device),
        v_out=wp.zeros(dofs, dtype=float, device=device),
        v_hat=wp.zeros(dofs, dtype=float, device=device),
    )


class TestFeatherPGSAugmentedDrives(unittest.TestCase):
    def test_mixed_joint_target_layout_trajectories(self):
        """Match public solver trajectories across both target layouts."""
        trajectories = []
        for coord_layout in (False, True):
            model, state, control, _joints, dofs, coords, targets = _mixed_target_model(coord_layout, wp.get_device())
            solver = newton.solvers.SolverFeatherPGS(
                model,
                enable_joint_limits=False,
                dense_max_constraints=16,
                mf_max_constraints=16,
                pgs_iterations=8,
            )
            initial_qd = state.joint_qd.numpy()[dofs].copy()
            position_error = control.joint_target_q.numpy()[targets] - state.joint_q.numpy()[coords]
            following = model.state()
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
            history = []
            for step in range(3):
                state.clear_forces()
                solver.step(state, following, control, None, 0.01)
                state, following = following, state
                q, qd = state.joint_q.numpy(), state.joint_qd.numpy()
                if step == 0:
                    self.assertTrue(np.all((qd[dofs] - initial_qd) * position_error > 0.0))
                history.append(np.concatenate((q, qd)))
            trajectories.append(history)
        np.testing.assert_allclose(trajectories[0], trajectories[1], rtol=0.0, atol=2e-6)

    def test_mixed_joint_position_target_layouts(self):
        """Keep distinct world targets aligned after free and ball coordinates."""
        device, dt = wp.get_device(), 0.01
        for coord_layout in (False, True):
            model, state, control, joints, dofs, coords, targets = _mixed_target_model(coord_layout, device)
            max_dofs = 6
            row_count = model.articulation_count * max_dofs
            expected = np.zeros(model.joint_dof_count)
            expected[dofs] = 20.0 * (
                control.joint_target_q.numpy()[targets]
                - state.joint_q.numpy()[coords]
                - dt * state.joint_qd.numpy()[dofs]
            ) + 3.0 * (control.joint_target_qd.numpy()[dofs] - state.joint_qd.numpy()[dofs])
            with self.subTest(coord_layout=coord_layout):
                row_K = wp.zeros(row_count, dtype=float, device=device)
                tau = wp.zeros(model.joint_dof_count, dtype=float, device=device)
                rows = np.full(model.joint_dof_count, -1, dtype=np.int32)
                q_indices = np.zeros(model.joint_dof_count, dtype=np.int32)
                target_indices = np.zeros(model.joint_dof_count, dtype=np.int32)
                rows[dofs], q_indices[dofs] = joints * max_dofs, coords
                target_indices[dofs] = targets
                wp.launch(
                    prepare_augmented_drives,
                    dim=model.joint_dof_count,
                    inputs=[
                        wp.array(rows, dtype=int, device=device),
                        wp.array(q_indices, dtype=int, device=device),
                        wp.array(target_indices, dtype=int, device=device),
                        state.joint_q,
                        state.joint_qd,
                        model.joint_target_ke,
                        model.joint_target_kd,
                        control.joint_target_q,
                        control.joint_target_qd,
                        model.joint_effort_limit,
                        wp.zeros(model.joint_dof_count, dtype=float, device=device),
                        max_dofs,
                        dt,
                    ],
                    outputs=[_drive_rows(row_K, model.articulation_count, model.joint_dof_count, device), tau],
                    device=device,
                )
                np.testing.assert_allclose(tau.numpy(), expected, rtol=0, atol=2e-6)

    def test_per_coordinate_drives(self):
        """Write each coordinate's K and clamped force from its row, and clear both without gains."""
        device = wp.get_device()
        joint_q = wp.array([0.25, -0.35], dtype=wp.float32, device=device)
        joint_qd = wp.array([0.1, -0.2], dtype=wp.float32, device=device)
        target_ke = wp.array([10.0, 20.0], dtype=wp.float32, device=device)
        target_kd = wp.array([1.0, 2.0], dtype=wp.float32, device=device)
        target_q = wp.array([0.1, -0.1], dtype=wp.float32, device=device)
        target_qd = wp.array([0.0, 0.05], dtype=wp.float32, device=device)
        effort_limit = wp.array([1.25, 2.5], dtype=wp.float32, device=device)
        damping = wp.array([0.2, 0.3], dtype=wp.float32, device=device)
        rows = wp.array([0, 1], dtype=wp.int32, device=device)

        def prepare_by_dof(ke, kd, stiffness, tau):
            wp.launch(
                prepare_augmented_drives,
                dim=2,
                inputs=[
                    rows,
                    rows,
                    rows,
                    joint_q,
                    joint_qd,
                    ke,
                    kd,
                    target_q,
                    target_qd,
                    effort_limit,
                    damping,
                    2,
                    0.1,
                ],
                outputs=[_drive_rows(stiffness, 1, 2, device), tau],
                device=device,
            )

        stiffness = wp.zeros(2, dtype=wp.float32, device=device)
        tau = wp.zeros(2, dtype=wp.float32, device=device)
        prepare_by_dof(target_ke, target_kd, stiffness, tau)
        # The drive's K, ke dt^2 + kd dt, with the passive damper's, c dt.
        np.testing.assert_allclose(stiffness.numpy(), [0.22, 0.43], rtol=0.0, atol=1.0e-7)
        # Both forces are past their effort limits: -(10 (0.25 - 0.1 + 0.01) + 0.1) and -(20 (-0.25 - 0.02) + 2 (-0.25)).
        np.testing.assert_allclose(tau.numpy(), [-1.25, 2.5], rtol=0.0, atol=1.0e-6)

        # Without gains the preparation keeps no drive force, and the passive damper keeps its K.
        zero_gain = wp.zeros_like(target_ke)
        passive_stiffness = wp.array([7.0, 7.0], dtype=wp.float32, device=device)
        passive_tau = wp.array([3.0, -2.0], dtype=wp.float32, device=device)
        prepare_by_dof(zero_gain, zero_gain, passive_stiffness, passive_tau)
        np.testing.assert_allclose(passive_stiffness.numpy(), [0.02, 0.03], rtol=0.0, atol=1.0e-7)
        np.testing.assert_array_equal(passive_tau.numpy(), [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
