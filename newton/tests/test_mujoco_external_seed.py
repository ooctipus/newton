# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Keep native CPU collision demand out of external-contact seed allocation."""

import unittest
import warnings
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverMuJoCo


def _model():
    """Build three penetrating contacts and two non-contact rows per world."""
    template = newton.ModelBuilder()
    for i in range(3):
        body = template.add_link(xform=wp.transform((2.0 * i, 0.0, 0.05), wp.quat_identity()))
        template.add_shape_sphere(body, radius=0.1)
        template.add_articulation([template.add_joint_free(body)])
    body = template.add_link(xform=wp.transform((8.0, 0.0, 1.0), wp.quat_identity()))
    template.add_shape_sphere(body, radius=0.1)
    joint = template.add_joint_prismatic(
        -1,
        body,
        parent_xform=wp.transform((8.0, 0.0, 1.0), wp.quat_identity()),
        limit_lower=0.0,
        limit_upper=0.1,
        friction=2.0,
    )
    template.add_articulation([joint])
    template.joint_q[-1] = 0.2
    template.joint_qd[-1] = 0.3
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    for i in range(2):
        builder.add_world(template, xform=wp.transform((20.0 * i, 0.0, 0.0), wp.quat_identity()))
    return builder.finalize(device="cpu")


class TestMuJoCoExternalSeed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mujoco, cls.mjw = SolverMuJoCo.import_mujoco()

    def test_external_seed_retains_small_requested_pool_and_noncontact_rows(self):
        """Allocate the requested pool from a fully initialized contact-free seed."""
        with warnings.catch_warnings(record=True) as emitted:
            solver = SolverMuJoCo(_model(), use_mujoco_contacts=False, nconmax=1, njmax=2)
        self.assertEqual(solver.mj_data.ncon, 0)
        self.assertEqual(solver.mj_data.nefc, 2)
        self.assertEqual(solver.mjw_data.naconmax, 2)
        self.assertEqual(solver.mjw_data.njmax, 2)
        self.assertEqual(int(solver.mjw_data.nacon.numpy()[0]), 0)
        self.assertFalse(solver.mjw_model.opt.run_collision_detection)
        self.assertFalse(any("Value for nconmax" in str(item.message) for item in emitted))
        types = set(map(int, solver.mj_data.efc_type))
        self.assertEqual(
            types,
            {int(self.mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF), int(self.mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT)},
        )
        np.testing.assert_array_equal(solver.mjw_data.qacc_warmstart.numpy(), 0.0)
        np.testing.assert_array_equal(solver.mjw_data.nefc.numpy(), [2, 2])
        np.testing.assert_array_equal(solver.mjw_data.efc.type.numpy()[:, :2], np.tile(solver.mj_data.efc_type, (2, 1)))
        self.assertTrue(np.isfinite(solver.mjw_data.qLD.numpy()).all())

    def test_native_and_cpu_backends_keep_contact_seed_and_capacity_floor(self):
        """Preserve the existing seed forward for native contacts and the CPU backend."""
        for use_cpu, use_contacts in ((False, True), (True, False), (True, True)):
            with self.subTest(use_cpu=use_cpu, use_contacts=use_contacts), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                original_forward = self.mujoco.mj_forward
                observed = []

                def forward(model, data, original_forward=original_forward, observed=observed):
                    self.assertFalse(int(model.opt.disableflags) & int(self.mujoco.mjtDisableBit.mjDSBL_CONTACT))
                    reference = self.mujoco.MjData(model)
                    for name in ("qpos", "qvel", "ctrl", "act", "qfrc_applied", "xfrc_applied", "qacc_warmstart"):
                        getattr(reference, name)[:] = getattr(data, name)
                    original_forward(model, data)
                    original_forward(model, reference)
                    for name in ("qacc", "qacc_warmstart", "efc_force", "efc_type", "M"):
                        np.testing.assert_allclose(
                            getattr(data, name), getattr(reference, name), rtol=1e-12, atol=1e-12
                        )
                    observed.append(True)

                with mock.patch.object(self.mujoco, "mj_forward", side_effect=forward):
                    solver = SolverMuJoCo(
                        _model(),
                        separate_worlds=True,
                        use_mujoco_cpu=use_cpu,
                        use_mujoco_contacts=use_contacts,
                        nconmax=1,
                        njmax=2,
                    )
                self.assertEqual(observed, [True])
                self.assertEqual(solver.mj_data.ncon, 3)
                self.assertGreater(solver.mj_data.nefc, 2)
                self.assertEqual(solver.mjw_data.naconmax, 6)
                self.assertEqual(solver.mjw_data.njmax, solver.mj_data.nefc)

    def test_flags_restored_before_upload_with_consistent_noncontact_state(self):
        """Restore all model flags before either upload while retaining the complete forward state."""
        contact = int(self.mujoco.mjtDisableBit.mjDSBL_CONTACT)
        sensor = int(self.mujoco.mjtDisableBit.mjDSBL_SENSOR)
        original_model, original_data = self.mjw.put_model, self.mjw.put_data
        observed = []

        def put_model(model, *args, **kwargs):
            self.assertEqual(int(model.opt.disableflags) & (contact | sensor), sensor)
            observed.append("model")
            return original_model(model, *args, **kwargs)

        def put_data(model, data, *args, **kwargs):
            self.assertEqual(int(model.opt.disableflags) & (contact | sensor), sensor)
            self.assertEqual((data.ncon, data.nefc), (0, 2))
            reference = self.mujoco.MjData(model)
            reference.qpos[:] = data.qpos
            reference.qvel[:] = data.qvel
            old_flags = int(model.opt.disableflags)
            model.opt.disableflags = old_flags | contact
            try:
                self.mujoco.mj_forward(model, reference)
            finally:
                model.opt.disableflags = old_flags
            for name in (
                "qpos",
                "qvel",
                "xpos",
                "M",
                "qfrc_smooth",
                "qfrc_bias",
                "qacc",
                "qacc_warmstart",
                "efc_J",
                "efc_force",
            ):
                np.testing.assert_allclose(getattr(data, name), getattr(reference, name), rtol=1e-12, atol=1e-12)
            observed.append("data")
            return original_data(model, data, *args, **kwargs)

        with (
            mock.patch.object(self.mjw, "put_model", side_effect=put_model),
            mock.patch.object(self.mjw, "put_data", side_effect=put_data),
        ):
            SolverMuJoCo(_model(), use_mujoco_contacts=False, disable_sensors=True, nconmax=1, njmax=2)
        self.assertEqual(observed, ["model", "data"])

    def test_forward_exception_restores_exact_original_flags(self):
        """Restore the original bit mask even when seed initialization raises."""
        seen = []
        contact = int(self.mujoco.mjtDisableBit.mjDSBL_CONTACT)
        sensor = int(self.mujoco.mjtDisableBit.mjDSBL_SENSOR)

        def fail(model, data):
            seen.append((model, int(model.opt.disableflags)))
            raise RuntimeError("seed forward failed")

        with (
            mock.patch.object(self.mujoco, "mj_forward", side_effect=fail),
            self.assertRaisesRegex(RuntimeError, "seed forward failed"),
        ):
            SolverMuJoCo(_model(), use_mujoco_contacts=False, disable_sensors=True, nconmax=1, njmax=2)
        self.assertEqual(len(seen), 1)
        model, during = seen[0]
        self.assertTrue(during & contact)
        self.assertTrue(during & sensor)
        self.assertEqual(int(model.opt.disableflags), during & ~contact)

    def test_explicit_contact_disable_remains_disabled(self):
        """Preserve an explicitly disabled contact flag and zero contact allocation."""
        solver = SolverMuJoCo(_model(), use_mujoco_contacts=False, disable_contacts=True, nconmax=1, njmax=2)
        contact = int(self.mujoco.mjtDisableBit.mjDSBL_CONTACT)
        self.assertTrue(int(solver.mj_model.opt.disableflags) & contact)
        self.assertTrue(int(solver.mjw_model.opt.disableflags) & contact)
        self.assertEqual(solver.mjw_data.naconmax, 0)
        self.assertEqual(solver.mj_data.nefc, 2)

    def test_noncontact_capacity_floor_and_reset_remain_intact(self):
        """Retain the real non-contact row floor and clear warmstart on reset."""
        with self.assertWarnsRegex(UserWarning, "Value for njmax is changed from 1 to 2"):
            solver = SolverMuJoCo(_model(), use_mujoco_contacts=False, nconmax=1, njmax=1)
        self.assertEqual(solver.mjw_data.njmax, 2)
        self.assertEqual(solver.mjw_data.naconmax, 2)
        solver.mjw_data.qacc_warmstart.fill_(7.0)
        solver.reset(solver.model.state())
        np.testing.assert_array_equal(solver.mjw_data.qacc_warmstart.numpy(), 0.0)

    def test_native_forward_flag_edits_are_not_restored(self):
        """Retain native forward callback flag edits outside the external seed scope."""
        original = self.mujoco.mj_forward
        sensor = int(self.mujoco.mjtDisableBit.mjDSBL_SENSOR)

        def forward(model, data):
            original(model, data)
            model.opt.disableflags |= sensor

        for use_cpu in (False, True):
            with self.subTest(use_cpu=use_cpu), mock.patch.object(self.mujoco, "mj_forward", side_effect=forward):
                solver = SolverMuJoCo(
                    _model(),
                    separate_worlds=True,
                    use_mujoco_cpu=use_cpu,
                    use_mujoco_contacts=True,
                    nconmax=4,
                    njmax=16,
                )
            self.assertTrue(int(solver.mj_model.opt.disableflags) & sensor)
            self.assertTrue(int(solver.mjw_model.opt.disableflags) & sensor)


if __name__ == "__main__":
    unittest.main()
