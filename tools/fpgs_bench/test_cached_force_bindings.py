# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check shared force arithmetic inside the retained block-factor predictor."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs.kinetic_predictor import get_predictor_kernel, unpack_T
from tools.fpgs_bench.test_kinetic_live_bindings import bound_call


def drive_force(inputs, dofs):
    """Evaluate the declared capped augmented drive law in physical coordinates."""
    q_index = inputs.drive_q_index_by_dof.numpy()[dofs]
    q = inputs.joint_q.numpy()[q_index].astype(float)
    qd = inputs.joint_qd.numpy()[dofs].astype(float)
    ke, kd = inputs.joint_target_ke.numpy()[dofs], inputs.joint_target_kd.numpy()[dofs]
    target, velocity = inputs.joint_target_pos.numpy()[dofs], inputs.joint_target_vel.numpy()[dofs]
    value = -(ke * (q - target + inputs.dt * qd) + kd * (qd - velocity))
    cap = inputs.joint_effort_limit.numpy()[dofs]
    value = np.where(cap > 0, np.clip(value, -cap, cap), value)
    return np.where((inputs.drive_row_by_dof.numpy()[dofs] >= 0) & (ke * inputs.dt + kd > 0), value, 0)


def copy_arguments(arguments, device):
    """Copy current/held test operands while preserving storage aliases."""
    arrays = {}

    def copy(value):
        if isinstance(value, wp.array):
            key = (value.ptr, value.dtype, value.size)
            if key not in arrays:
                arrays[key] = wp.array(value.numpy(), dtype=value.dtype, device=device)
            return arrays[key].reshape(value.shape)
        if hasattr(value, "_cls"):
            out = value._cls()
            for name in value._cls.vars:
                setattr(out, name, copy(getattr(value, name)))
            return out
        return value

    return [copy(value) for value in arguments]


class TestCachedForceBindings(unittest.TestCase):
    def check_cancellation_and_drives(self, device):
        solver, owner, state, _, _, _, call = bound_call()
        state.joint_qd.assign(np.linspace(-0.1, 0.2, state.joint_qd.size, dtype=np.float32))
        call.construct_launch()
        call.refresh_launch()
        call.free.launch()
        call.predict_launch()
        before = call.output.joint_qdd.numpy().astype(float)
        held = call.slots.held.T.numpy().copy()
        dofs = owner.host_plan.host["dof_ids"][0, :23]
        old_drive = drive_force(call.inputs, dofs)
        axes = call.slots.current.axes.numpy()[0].astype(float)
        offsets = call.slots.current.com_offset.numpy()[0].astype(float)
        body_forces = state.body_f.numpy()
        delta = np.zeros(23)
        wrench = np.array([0.3, -0.2, 0.1, 0.07, 0.02, -0.09])
        for local, sign in ((13, 1), (18, -1)):
            body = int(owner.host_plan.host["body_ids"][0, local])
            body_forces[body, :3] = sign * wrench[:3]
            body_forces[body, 3:] = sign * (wrench[3:] - np.cross(offsets[local], wrench[:3]))
            support = int(solver.body_response_dof_mask.numpy()[body])
            for dof in range(23):
                if support & (1 << dof):
                    delta[dof] += axes[dof] @ (sign * wrench)
        np.testing.assert_allclose(delta[:7], 0, atol=1e-14)
        self.assertGreater(float(np.linalg.norm(delta[7:])), 0.01)
        state.body_f.assign(body_forces)
        call.inputs.joint_target_pos.assign(np.linspace(-2, 3, call.inputs.joint_target_pos.size, dtype=np.float32))
        call.inputs.joint_effort_limit.fill_(0.025)
        new_drive = drive_force(call.inputs, dofs)
        self.assertTrue(np.any(np.isclose(np.abs(new_drive), 0.025)))
        delta += new_drive - old_drive
        for name, value in (("joint_spring_stiffness", 0.17), ("joint_spring_ref", 0.13), ("joint_damping", 0.11)):
            getattr(call.inputs, name).fill_(value)
        q = state.joint_q.numpy()[owner.host_plan.host["q_index"][0]].astype(float)
        qd = state.joint_qd.numpy()[dofs].astype(float)
        delta += 0.17 * (0.13 - q) - 0.11 * qd
        t = unpack_T(held[0].astype(float))
        expected = before[dofs] + t.T @ (t @ delta)
        scale = 1 + np.abs(t.T) @ (np.abs(t) @ np.abs(delta))
        arguments = copy_arguments(call.kernel_arguments["predict"], device)
        _plan, _current, response, inputs, output = arguments
        predictor = np.linspace(0.21, -0.31, inputs.predictor_qd.size, dtype=np.float32)
        inputs.predictor_qd = wp.array(predictor, dtype=float, device=device)
        wp.launch_tiled(
            get_predictor_kernel(str(wp.get_device(device).arch)),
            dim=[2],
            inputs=arguments,
            block_dim=32,
            device=device,
        )
        np.testing.assert_array_equal(output.status.numpy(), 0)
        np.testing.assert_array_equal(response.T.numpy(), held)
        self.assertLess(float(np.max(np.abs(output.joint_qdd.numpy()[dofs] - expected) / scale)), 2**-17)
        np.testing.assert_allclose(
            output.v_hat.numpy()[dofs], predictor[dofs] + inputs.dt * expected, atol=2e-6, rtol=2e-5
        )

    def test_branch_cancellation_and_capped_drives_cpu(self):
        """Preserve distinct finger forces despite a cancelling shared ancestor wrench."""
        self.check_cancellation_and_drives("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA unavailable")
    def test_branch_cancellation_and_capped_drives_cuda(self):
        """Check the full-warp force mapping with CPU-prepared current and held operands."""
        self.check_cancellation_and_drives("cuda:0")


if __name__ == "__main__":
    unittest.main()
