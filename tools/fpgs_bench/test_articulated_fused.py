# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Same-input controls for joint factor/current-force/predictor ownership."""

import unittest

import articulated_factor_check as check
import numpy as np
import warp as wp
from test_articulated_factor import fixture

from newton._src.sim.enums import BodyFlags, JointType
from newton._src.solvers.feather_pgs import articulated_factor as module
from newton._src.solvers.feather_pgs.kernels import eval_rigid_tau_add


def make_case():
    """Build two branched trees and distinct current force/control inputs."""
    data = fixture()
    rng = np.random.default_rng(417)
    arts, bodies = data["parents"].shape
    dofs = data["diagonal"].shape[1]
    count = arts * bodies
    data["body_mass"] = rng.uniform(0.4, 2.0, (arts, bodies)).astype(np.float32)
    offsets = rng.normal(0, 0.15, (arts, bodies, 3)).astype(np.float32)
    cross = np.zeros((arts, bodies, 3, 3))
    cross[..., 0, 1], cross[..., 0, 2] = -offsets[..., 2], offsets[..., 1]
    cross[..., 1, 0], cross[..., 1, 2] = offsets[..., 2], -offsets[..., 0]
    cross[..., 2, 0], cross[..., 2, 1] = -offsets[..., 1], offsets[..., 0]
    inertia_origin = np.eye(3) * 0.3 - data["body_mass"][..., None, None] * (cross @ cross)
    data["compact_terms"] = np.concatenate((offsets, inertia_origin.reshape(arts, bodies, 9)), axis=-1).astype(
        np.float32
    )
    data["inertia"] = check.spatial_from_compact(data["body_mass"], data["compact_terms"]).astype(np.float32)
    data["R"] = data["diagonal"] * 0.5
    data["K"] = data["diagonal"] * 0.5
    current = module.ArticulatedCurrentData()
    current.articulation_start = wp.array(np.arange(arts) * bodies, dtype=int, device="cpu")
    slots = data["slots"]
    globals_ = np.where(slots >= 0, slots + np.arange(arts)[:, None] * dofs, -1)
    current.joint_q_start = wp.array(np.maximum(globals_, 0).ravel(), dtype=int, device="cpu")
    for name in ("joint_q", "joint_qd", "joint_f", "spring_ref"):
        setattr(current, name, wp.array(rng.normal(0, 0.4, arts * dofs).astype(np.float32), device="cpu"))
    for name in ("spring_stiffness", "damping"):
        setattr(current, name, wp.array(rng.uniform(0.1, 0.7, arts * dofs).astype(np.float32), device="cpu"))
    current.joint_S = check.factor_inputs(data, "cpu")[1]
    current.body_fb_s = wp.array(rng.normal(0, 0.4, (count, 6)), dtype=wp.spatial_vector, device="cpu")
    current.body_f_ext = wp.array(rng.normal(0, 0.3, (count, 6)), dtype=wp.spatial_vector, device="cpu")
    flags = np.zeros(count, np.int32)
    flags[4] = int(BodyFlags.KINEMATIC)
    current.body_flags = wp.array(flags, dtype=int, device="cpu")
    poses = np.zeros((count, 7), np.float32)
    poses[:, :3] = rng.normal(0, 0.2, (count, 3))
    poses[:, 6] = 1
    current.body_q = wp.array(poses, dtype=wp.transform, device="cpu")
    current.body_com = wp.array(rng.normal(0, 0.1, (count, 3)), dtype=wp.vec3, device="cpu")
    current.articulation_origin = wp.array([[0.5, 0.2, -0.1], [-0.4, 0.3, 0.2]], dtype=wp.vec3, device="cpu")
    current.joint_tau = wp.array(rng.normal(0, 0.5, arts * dofs).astype(np.float32), device="cpu")
    current.joint_qdd = wp.full(arts * dofs + 3, 9876.0, dtype=float, device="cpu")
    return data, current


def original_tau(data, current):
    """Execute the actual original inverse-dynamics law with private outputs."""
    arts, bodies = data["parents"].shape
    device = current.joint_tau.device
    slots = data["slots"]
    parent = np.where(data["parents"] >= 0, data["parents"] + np.arange(arts)[:, None] * bodies, -1)
    dimensions = np.zeros((arts * bodies, 2), np.int32)
    dimensions[:, 1] = (slots >= 0).ravel()
    types = np.where(slots >= 0, int(JointType.D6), int(JointType.FIXED))
    output = wp.clone(current.joint_tau)
    scratch = wp.zeros(arts * bodies, dtype=wp.spatial_vector, device=device)
    wp.launch(
        eval_rigid_tau_add,
        dim=arts,
        inputs=[
            current.articulation_start,
            wp.array((np.arange(arts) + 1) * bodies, dtype=int, device=device),
            wp.array(types.ravel(), dtype=int, device=device),
            wp.array(parent.ravel(), dtype=int, device=device),
            wp.array(np.arange(arts * bodies), dtype=int, device=device),
            wp.array(np.repeat(np.arange(arts), bodies), dtype=int, device=device),
            current.joint_q_start,
            current.joint_q_start,
            wp.array(dimensions, dtype=int, device=device),
            current.joint_f,
            current.joint_q,
            current.joint_qd,
            current.spring_stiffness,
            current.spring_ref,
            current.damping,
            current.joint_S,
            current.body_fb_s,
            current.body_f_ext,
            current.body_flags,
            current.body_q,
            current.body_com,
            current.articulation_origin,
        ],
        outputs=[scratch, output],
        device=device,
    )
    return output.numpy().reshape(arts, -1)


def launch(data, current, factor=None, *, compact=False, poison_unused=False):
    """Execute one actual fused owner, preserving its in-place drive bucket."""
    device = current.joint_tau.device
    factor = check.allocate_factor(module, data, device) if factor is None else factor
    arts, bodies = data["parents"].shape
    dofs = data["diagonal"].shape[1]
    inertia = data["inertia"]
    terms = data["compact_terms"]
    if poison_unused:
        if compact:
            inertia = np.full_like(inertia, np.nan)
        else:
            terms = np.full_like(terms, np.nan)
    wp.launch_tiled(
        module.get_fused_dynamics_kernel(bodies, dofs),
        dim=[arts],
        inputs=[
            factor,
            current,
            wp.array(inertia.reshape(-1, 6, 6), dtype=wp.spatial_matrix, device=device),
            wp.array(data["R"], dtype=float, device=device),
            wp.array(np.arange(arts * dofs), dtype=int, device=device),
            wp.array(data["K"].ravel(), dtype=float, device=device),
            wp.array(data["refresh"], dtype=int, device=device),
            wp.array(terms.reshape(-1, 12), dtype=float, device=device),
            wp.array(data["body_mass"].ravel(), dtype=float, device=device),
            int(compact),
        ],
        block_dim=32,
        device=device,
    )
    return factor


def assert_physics(test, data, current, expected_tau, h):
    """Check current torque and the held physical mass equation, not bit identity."""
    arts, dofs = expected_tau.shape
    actual_tau = current.joint_tau.numpy().reshape(arts, dofs)
    np.testing.assert_allclose(actual_tau, expected_tau, rtol=3e-6, atol=2e-6)
    actual_qdd = current.joint_qdd.numpy()
    check.action_error(h, actual_tau[:, None, :], actual_qdd[: arts * dofs].reshape(arts, 1, dofs))
    np.testing.assert_array_equal(actual_qdd[arts * dofs :], 9876)
    test.assertTrue(np.isfinite(actual_tau).all())


class TestArticulatedFused(unittest.TestCase):
    @unittest.skipUnless(wp.get_cuda_devices(), "CUDA native reduction control")
    def test_actual_cuda_current_tau_and_held_predictor(self):
        """Exercise actual cooperative reductions against original same-device RNEA."""
        data, host = make_case()
        current = module.ArticulatedCurrentData()
        for name in module.ArticulatedCurrentData.vars:
            value = getattr(host, name)
            setattr(current, name, wp.array(value.numpy(), dtype=value.dtype, device="cuda:0"))
        h = check.dense_mass(
            data["parents"], data["slots"], data["inertia"], data["motion"], data["diagonal"], data["counts"]
        )
        expected = original_tau(data, current)
        factor = launch(data, current, compact=True, poison_unused=True)
        assert_physics(self, data, current, expected, h)
        retained = [value.numpy() for value in (factor.S, factor.U, factor.invD)]
        data["refresh"][:] = 0
        current.joint_S.assign(current.joint_S.numpy() * 0.8 + 0.15)
        current.articulation_origin.assign([[0.8, -0.2, 0.5], [-0.2, 0.5, -0.3]])
        current.joint_tau.fill_(0.4)
        expected = original_tau(data, current)
        launch(data, current, factor, compact=False, poison_unused=True)
        assert_physics(self, data, current, expected, h)
        for before, after in zip(retained, (factor.S, factor.U, factor.invD), strict=True):
            np.testing.assert_array_equal(before, after.numpy())

    def test_refresh_current_tau_and_compact_full_mass(self):
        """Match original RNEA and independent mass using either real inertia owner."""
        for compact in (False, True):
            with self.subTest(compact=compact):
                data, current = make_case()
                expected = original_tau(data, current)
                factor = launch(data, current, compact=compact, poison_unused=True)
                h = check.dense_mass(
                    data["parents"], data["slots"], data["inertia"], data["motion"], data["diagonal"], data["counts"]
                )
                np.testing.assert_array_equal(factor.valid.numpy(), 1)
                assert_physics(self, data, current, expected, h)

    def test_reuse_changed_current_axes_origin_and_forces(self):
        """Preserve held factors while current axes, forces, origin and u0 change."""
        data, current = make_case()
        factor = launch(data, current, compact=True)
        h = check.dense_mass(
            data["parents"], data["slots"], data["inertia"], data["motion"], data["diagonal"], data["counts"]
        )
        held = [array.numpy() for array in (factor.S, factor.U, factor.invD)]
        data["refresh"][:] = 0
        for name in ("inertia", "compact_terms", "body_mass", "R", "K"):
            data[name] = np.full_like(data[name], np.nan)
        current.joint_S.assign(current.joint_S.numpy() * 0.73 + 0.11)
        current.articulation_origin.assign([[1.1, -0.2, 0.3], [-0.8, 0.9, 0.6]])
        current.body_fb_s.assign(current.body_fb_s.numpy() * 1.2 - 0.2)
        current.body_f_ext.assign(current.body_f_ext.numpy() * -1.5)
        current.joint_tau.assign(np.linspace(-0.5, 0.8, 8, dtype=np.float32))
        expected = original_tau(data, current)
        launch(data, current, factor, compact=True)
        for before, after in zip(held, (factor.S, factor.U, factor.invD), strict=True):
            np.testing.assert_array_equal(before, after.numpy())
        assert_physics(self, data, current, expected, h)

    def test_mixed_refresh_and_current_drive_coefficients(self):
        """Refresh only the requested tree while preserving the other held epoch."""
        data, current = make_case()
        factor = launch(data, current)
        h = check.dense_mass(
            data["parents"], data["slots"], data["inertia"], data["motion"], data["diagonal"], data["counts"]
        )
        held_s = factor.S.numpy()
        data["refresh"][:] = [1, 0]
        data["inertia"][0] *= 1.3
        data["R"][0] += 0.2
        data["K"][0] += 0.4
        data["diagonal"] = data["R"] + data["K"]
        data["motion"][0] *= 0.8
        current.joint_S = check.factor_inputs(data, "cpu")[1]
        current.joint_tau.assign(np.linspace(0.1, 0.7, 8, dtype=np.float32))
        expected = original_tau(data, current)
        fresh_h = check.dense_mass(
            data["parents"], data["slots"], data["inertia"], data["motion"], data["diagonal"], data["counts"]
        )
        h[0] = fresh_h[0]
        launch(data, current, factor)
        np.testing.assert_array_equal(held_s[1], factor.S.numpy()[1])
        assert_physics(self, data, current, expected, h)

    def test_invalid_mass_and_wrong_held_axis_negative_controls(self):
        """Reject an invalid native mass and detect replacing held by current axes."""
        data, current = make_case()
        factor = launch(data, current)
        h = check.dense_mass(
            data["parents"], data["slots"], data["inertia"], data["motion"], data["diagonal"], data["counts"]
        )
        factor.S.assign(factor.S.numpy() * 1.3)
        data["refresh"][:] = 0
        current.joint_tau.fill_(0.2)
        expected = original_tau(data, current)
        launch(data, current, factor)
        with self.assertRaises(AssertionError):
            assert_physics(self, data, current, expected, h)
        data["refresh"][:] = 1
        data["inertia"][:] = 0
        data["R"][:] = 0
        data["K"][:] = 0
        current.joint_tau.fill_(0.2)
        launch(data, current, factor)
        np.testing.assert_array_equal(factor.valid.numpy(), 0)
        self.assertTrue(np.isnan(current.joint_qdd.numpy()[:8]).all())


if __name__ == "__main__":
    unittest.main()
