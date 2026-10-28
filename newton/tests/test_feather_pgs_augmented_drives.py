# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.sim.enums import JointType
from newton._src.solvers.feather_pgs import field_group
from newton._src.solvers.feather_pgs.dense_rows import DenseRows
from newton._src.solvers.feather_pgs.joint_table import JointTable
from newton._src.solvers.feather_pgs.kernels import (
    eval_rigid_tau_and_augmented_drives,
    populate_physx_drive_J_for_size,
)
from newton._src.solvers.feather_pgs.solver_feather_pgs import _prepare_augmented_joint_drives_by_dof
from newton._src.solvers.feather_pgs.tree_passes import DynamicsData, inverse_dynamics_tau
from newton._src.solvers.feather_pgs.tree_plan import TreePlan


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


class TestFeatherPGSAugmentedDrives(unittest.TestCase):
    @unittest.skipUnless(wp.get_device().is_cuda, "Matrix-free solver dispatch requires CUDA")
    def test_mixed_joint_target_layout_trajectories(self):
        """Match public solver trajectories across both target layouts and drive dispatches."""
        for drive_mode, parallel_streams in (("augmented", False), ("augmented", True), ("physx_pgs", False)):
            with self.subTest(drive_mode=drive_mode, parallel_streams=parallel_streams):
                trajectories = []
                for coord_layout in (False, True):
                    model, state, control, _joints, dofs, coords, targets = _mixed_target_model(
                        coord_layout, wp.get_device()
                    )
                    solver = newton.solvers.SolverFeatherPGS(
                        model,
                        pgs_mode="matrix_free",
                        drive_mode=drive_mode,
                        use_parallel_streams=parallel_streams,
                        enable_joint_limits=False,
                        dense_max_constraints=16,
                        mf_max_constraints=16,
                        pgs_iterations=8,
                    )
                    # Ball roots keep the held tree out: ordinary dynamics evaluate the drives with the joint forces.
                    self.assertIsNone(solver._tree_kinetic_state)
                    self.assertFalse(solver._parallel_augmented_drive_topology)
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
            starts = model.joint_qd_start.numpy()
            counts = wp.array(np.diff(starts), dtype=int, device=device)
            max_dofs = 6
            row_count = model.articulation_count * max_dofs
            expected = np.zeros(model.joint_dof_count)
            expected[dofs] = 20.0 * (
                control.joint_target_q.numpy()[targets]
                - state.joint_q.numpy()[coords]
                - dt * state.joint_qd.numpy()[dofs]
            ) + 3.0 * (control.joint_target_qd.numpy()[dofs] - state.joint_qd.numpy()[dofs])
            for path in ("combined", "per_dof"):
                with self.subTest(coord_layout=coord_layout, path=path):
                    row_counts = wp.zeros(model.articulation_count, dtype=int, device=device)
                    row_indices = wp.zeros(row_count, dtype=int, device=device)
                    row_K = wp.zeros(row_count, dtype=float, device=device)
                    tau = wp.zeros(model.joint_dof_count, dtype=float, device=device)
                    gains = [
                        model.joint_target_ke,
                        model.joint_target_kd,
                        control.joint_target_q,
                        control.joint_target_qd,
                        model.joint_effort_limit,
                    ]
                    if path == "per_dof":
                        rows = np.full(model.joint_dof_count, -1, dtype=np.int32)
                        q_indices = np.zeros(model.joint_dof_count, dtype=np.int32)
                        target_indices = np.zeros(model.joint_dof_count, dtype=np.int32)
                        rows[dofs], q_indices[dofs] = joints * max_dofs, coords
                        target_indices[dofs] = targets
                        wp.launch(
                            _prepare_augmented_joint_drives_by_dof,
                            dim=model.joint_dof_count,
                            inputs=[
                                wp.array(rows, dtype=int, device=device),
                                wp.array(q_indices, dtype=int, device=device),
                                wp.array(target_indices, dtype=int, device=device),
                                state.joint_q,
                                state.joint_qd,
                                *gains,
                                dt,
                            ],
                            outputs=[row_K, tau],
                            device=device,
                        )
                    else:
                        zeros = wp.zeros(model.joint_dof_count, dtype=float, device=device)
                        body_forces = wp.zeros(model.body_count, dtype=wp.spatial_vector, device=device)
                        dynamics = field_group.upload(
                            DynamicsData,
                            device,
                            articulation_start=model.articulation_start,
                            articulation_joint_end=wp.array(
                                model.articulation_start.numpy()[1:], dtype=int, device=device
                            ),
                            joint_type=model.joint_type,
                            joint_parent=model.joint_parent,
                            joint_child=model.joint_child,
                            joint_articulation=model.joint_articulation,
                            joint_qd_start=model.joint_qd_start,
                            joint_q_start=model.joint_q_start,
                            joint_dof_dim=model.joint_dof_dim,
                            joint_f=control.joint_f,
                            joint_q=state.joint_q,
                            joint_qd=state.joint_qd,
                            spring_stiffness=zeros,
                            spring_reference=zeros,
                            damping=zeros,
                            joint_S_s=wp.zeros(model.joint_dof_count, dtype=wp.spatial_vector, device=device),
                            body_f_s=body_forces,
                            body_f=body_forces,
                            body_flags=model.body_flags,
                            body_q=state.body_q,
                            body_com=model.body_com,
                            articulation_origin=wp.zeros(model.articulation_count, dtype=wp.vec3, device=device),
                            body_ft_s=wp.zeros_like(body_forces),
                            tau=tau,
                            row_K=row_K,
                            up_flags=wp.array(
                                TreePlan.build(model.joint_ancestor.numpy()).up_flags, dtype=int, device=device
                            ),
                        )
                        wp.launch(
                            eval_rigid_tau_and_augmented_drives,
                            dim=model.articulation_count,
                            inputs=[dynamics, counts, model.joint_target_q_start, *gains, max_dofs, dt],
                            outputs=[row_counts, row_indices],
                            device=device,
                        )
                    np.testing.assert_allclose(tau.numpy(), expected, rtol=0, atol=2e-6)

    def test_physx_drive_position_target_layouts(self):
        """Read position coordinates and DOF velocities independently in drive rows."""
        device = wp.get_device()
        for coord_layout in (False, True):
            with self.subTest(coord_layout=coord_layout):
                model, state, control, joints, dofs, coords, targets = _mixed_target_model(coord_layout, device)
                slots = np.full(model.joint_dof_count, -1, dtype=np.int32)
                slots[dofs] = [0, 1, 0, 1]
                fields = [wp.zeros((2, 2), dtype=int if i < 2 else float, device=device) for i in range(12)]
                wp.launch(
                    populate_physx_drive_J_for_size,
                    dim=len(joints),
                    inputs=[
                        field_group.upload(
                            JointTable,
                            device,
                            model,
                            articulation_dof_start=model.joint_qd_start.numpy()[:-1],
                            art_to_world=np.repeat([0, 1], 4),
                        ),
                        state.joint_q,
                        control.joint_target_q,
                        control.joint_target_qd,
                        0,
                        wp.array(slots, dtype=int, device=device),
                        wp.array(joints, dtype=int, device=device),
                    ],
                    outputs=[
                        wp.zeros((4, 2, 1), dtype=float, device=device),
                        field_group.upload(
                            DenseRows,
                            device,
                            **dict(
                                zip(
                                    ("row_type", "row_parent", "row_mu", "row_beta", "row_cfm", "phi", "target"),
                                    fields[:7],
                                    strict=True,
                                )
                            ),
                        ),
                        *fields[7:],
                    ],
                    device=device,
                )
                np.testing.assert_allclose(
                    fields[9].numpy().ravel(),
                    control.joint_target_q.numpy()[targets] - state.joint_q.numpy()[coords],
                    atol=1e-7,
                )
                np.testing.assert_array_equal(fields[6].numpy().ravel(), control.joint_target_qd.numpy()[dofs])

    def test_per_coordinate_drives_match_combined_kernel(self):
        """Match the held tree's per-coordinate drives plus the joint forces to the combined kernel; clear drives
        without gains."""
        device = wp.get_device()
        articulation_start = wp.array([0, 2], dtype=wp.int32, device=device)
        articulation_joint_end = wp.array([2], dtype=wp.int32, device=device)
        articulation_dof_count = wp.array([2], dtype=wp.int32, device=device)
        joint_type = wp.array([JointType.REVOLUTE, JointType.REVOLUTE], dtype=wp.int32, device=device)
        joint_parent = wp.array([-1, 0], dtype=wp.int32, device=device)
        joint_child = wp.array([0, 1], dtype=wp.int32, device=device)
        joint_articulation = wp.zeros(2, dtype=wp.int32, device=device)
        up_flags = wp.array(TreePlan.build([-1, 0]).up_flags, dtype=int, device=device)
        joint_start = wp.array([0, 1], dtype=wp.int32, device=device)
        joint_dof_dim = wp.array([[0, 1], [0, 1]], dtype=wp.int32, device=device)
        joint_f = wp.array([0.4, -0.2], dtype=wp.float32, device=device)
        joint_q = wp.array([0.25, -0.35], dtype=wp.float32, device=device)
        joint_qd = wp.array([0.1, -0.2], dtype=wp.float32, device=device)
        spring_stiffness = wp.array([1.5, 0.75], dtype=wp.float32, device=device)
        spring_ref = wp.array([0.0, 0.1], dtype=wp.float32, device=device)
        damping = wp.array([0.2, 0.3], dtype=wp.float32, device=device)
        joint_S_s = wp.array(
            [wp.spatial_vector(0.0, 0.0, 1.0, 0.0, 0.0, 0.0)] * 2,
            dtype=wp.spatial_vector,
            device=device,
        )
        body_fb_s = wp.array(
            [
                wp.spatial_vector(0.1, 0.2, 0.3, 1.0, 1.5, -0.5),
                wp.spatial_vector(-0.2, 0.4, -0.1, -0.5, 0.25, 0.75),
            ],
            dtype=wp.spatial_vector,
            device=device,
        )
        body_f_ext = wp.array(
            [
                wp.spatial_vector(0.05, 0.0, -0.1, 0.0, 0.25, 0.0),
                wp.spatial_vector(0.0, -0.05, 0.1, 0.15, 0.0, -0.2),
            ],
            dtype=wp.spatial_vector,
            device=device,
        )
        body_flags = wp.zeros(2, dtype=wp.int32, device=device)
        body_q = wp.array([wp.transform_identity()] * 2, dtype=wp.transform, device=device)
        body_com = wp.array([[0.1, 0.0, 0.0], [0.0, 0.2, 0.0]], dtype=wp.vec3, device=device)
        articulation_origin = wp.array([[0.0, 0.0, 0.0]], dtype=wp.vec3, device=device)
        target_ke = wp.array([10.0, 20.0], dtype=wp.float32, device=device)
        target_kd = wp.array([1.0, 2.0], dtype=wp.float32, device=device)
        target_q = wp.array([0.1, -0.1], dtype=wp.float32, device=device)
        target_qd = wp.array([0.0, 0.05], dtype=wp.float32, device=device)
        effort_limit = wp.array([1.25, 2.5], dtype=wp.float32, device=device)

        def dynamics(body_ft_s, tau, row_K=None):
            return field_group.upload(
                DynamicsData,
                device,
                articulation_start=articulation_start,
                articulation_joint_end=articulation_joint_end,
                joint_type=joint_type,
                joint_parent=joint_parent,
                joint_child=joint_child,
                joint_articulation=joint_articulation,
                joint_qd_start=joint_start,
                joint_q_start=joint_start,
                joint_dof_dim=joint_dof_dim,
                joint_f=joint_f,
                joint_q=joint_q,
                joint_qd=joint_qd,
                spring_stiffness=spring_stiffness,
                spring_reference=spring_ref,
                damping=damping,
                joint_S_s=joint_S_s,
                body_f_s=body_fb_s,
                body_f=body_f_ext,
                body_flags=body_flags,
                body_q=body_q,
                body_com=body_com,
                articulation_origin=articulation_origin,
                body_ft_s=body_ft_s,
                tau=tau,
                row_K=row_K if row_K is not None else wp.zeros(1, dtype=wp.float32, device=device),
                up_flags=up_flags,
            )

        sequential_body_ft = wp.zeros(2, dtype=wp.spatial_vector, device=device)
        sequential_tau = wp.zeros(2, dtype=wp.float32, device=device)
        wp.launch(inverse_dynamics_tau, dim=1, inputs=[dynamics(sequential_body_ft, sequential_tau)], device=device)

        combined_body_ft = wp.zeros_like(sequential_body_ft)
        combined_tau = wp.zeros_like(sequential_tau)
        combined_counts = wp.zeros(1, dtype=wp.int32, device=device)
        combined_indices = wp.zeros(2, dtype=wp.int32, device=device)
        combined_stiffness = wp.zeros(2, dtype=wp.float32, device=device)
        wp.launch(
            eval_rigid_tau_and_augmented_drives,
            dim=1,
            inputs=[
                dynamics(combined_body_ft, combined_tau, row_K=combined_stiffness),
                articulation_dof_count,
                joint_start,
                target_ke,
                target_kd,
                target_q,
                target_qd,
                effort_limit,
                2,
                0.1,
            ],
            outputs=[combined_counts, combined_indices],
            device=device,
        )
        np.testing.assert_array_equal(combined_body_ft.numpy(), sequential_body_ft.numpy())
        expected_tau = sequential_tau.numpy() + np.array([-1.25, 2.5], dtype=np.float32)
        np.testing.assert_allclose(combined_tau.numpy(), expected_tau, rtol=0.0, atol=1.0e-6)
        np.testing.assert_array_equal(combined_counts.numpy(), [2])
        np.testing.assert_array_equal(combined_indices.numpy(), [0, 1])
        np.testing.assert_allclose(combined_stiffness.numpy(), [0.2, 0.4], rtol=0.0, atol=1.0e-7)

        topology_stiffness = wp.zeros_like(combined_stiffness)
        topology_tau = wp.zeros_like(combined_tau)
        rows = wp.array([0, 1], dtype=wp.int32, device=device)

        def prepare_by_dof(ke, kd, stiffness, tau):
            wp.launch(
                _prepare_augmented_joint_drives_by_dof,
                dim=2,
                inputs=[rows, rows, rows, joint_q, joint_qd, ke, kd, target_q, target_qd, effort_limit, 0.1],
                outputs=[stiffness, tau],
                device=device,
            )

        prepare_by_dof(target_ke, target_kd, topology_stiffness, topology_tau)
        np.testing.assert_array_equal(topology_stiffness.numpy(), combined_stiffness.numpy())
        np.testing.assert_allclose(topology_tau.numpy(), [-1.25, 2.5], rtol=0.0, atol=1.0e-6)

        # The combined kernel's force is one fp32 sum of the joint force and the drive force.
        np.testing.assert_array_equal(sequential_tau.numpy() + topology_tau.numpy(), combined_tau.numpy())

        # Without gains neither preparation keeps a drive row or a drive force.
        zero_target_ke = wp.zeros_like(target_ke)
        zero_target_kd = wp.zeros_like(target_kd)
        passive_stiffness = wp.array([7.0, 7.0], dtype=wp.float32, device=device)
        passive_tau = wp.array([3.0, -2.0], dtype=wp.float32, device=device)
        prepare_by_dof(zero_target_ke, zero_target_kd, passive_stiffness, passive_tau)
        np.testing.assert_array_equal(passive_stiffness.numpy(), [0.0, 0.0])
        np.testing.assert_array_equal(passive_tau.numpy(), [0.0, 0.0])

        passive_counts = wp.array([7], dtype=wp.int32, device=device)
        passive_combined_tau = wp.zeros_like(combined_tau)
        wp.launch(
            eval_rigid_tau_and_augmented_drives,
            dim=1,
            inputs=[
                dynamics(wp.zeros_like(combined_body_ft), passive_combined_tau, row_K=wp.zeros_like(target_ke)),
                articulation_dof_count,
                joint_start,
                zero_target_ke,
                zero_target_kd,
                target_q,
                target_qd,
                effort_limit,
                2,
                0.1,
            ],
            outputs=[passive_counts, wp.zeros_like(combined_indices)],
            device=device,
        )
        np.testing.assert_array_equal(passive_counts.numpy(), [0])
        np.testing.assert_array_equal(passive_combined_tau.numpy(), sequential_tau.numpy())


if __name__ == "__main__":
    unittest.main()
