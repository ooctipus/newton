# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Preserve model refreshes without re-reading topology for numeric changes."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import warp as wp

from newton import ModelFlags
from newton._src.solvers.feather_pgs import franka_row_packets as packets
from newton._src.solvers.feather_pgs import solver_feather_pgs as solver_module


def fixture():
    """Bind the real notification method to observable existing refresh owners."""
    solver = SimpleNamespace(
        _fk_id_cache_enabled=True,
        _fk_id_cache_valid=Mock(),
        _update_kinematic_state=Mock(),
        _scatter_armature_to_groups=Mock(),
        _mass_update_requested=Mock(),
        model=SimpleNamespace(
            body_count=1,
            body_inertia=object(),
            body_mass=object(),
            body_com=object(),
            device="cpu",
            gravity=np.array([[0.0, 0.0, -9.81]]),
        ),
        body_I_m=object(),
        body_X_com=object(),
    )
    owner = object.__new__(packets.LocalRowPackets)
    owner.solver, owner._topology_signature = solver, ("unchanged",)
    solver._row_packets = owner
    return solver


class TestPacketNotifications(unittest.TestCase):
    def test_numeric_subsets_have_no_topology_readbacks(self):
        """Skip topology reads for every nonempty subset of four numeric flags."""
        flags = (
            ModelFlags.JOINT_DOF_PROPERTIES,
            ModelFlags.BODY_INERTIAL_PROPERTIES,
            ModelFlags.SHAPE_PROPERTIES,
            ModelFlags.MODEL_PROPERTIES,
        )
        for subset in range(1, 1 << len(flags)):
            selected = sum(int(flag) for bit, flag in enumerate(flags) if subset & (1 << bit))
            solver = fixture()
            with (
                self.subTest(flags=selected),
                patch.object(packets, "topology_signature", side_effect=AssertionError("topology readback")),
                patch.object(packets, "topology_supported", side_effect=AssertionError("topology scan")),
                patch.object(solver_module.wp, "launch") as launch,
            ):
                solver_module.SolverFeatherPGS.notify_model_changed(solver, selected)
                self.assertEqual(
                    solver._fk_id_cache_valid.zero_.call_count,
                    int(bool(selected & ~int(ModelFlags.SHAPE_PROPERTIES))),
                )
                dof = bool(selected & ModelFlags.JOINT_DOF_PROPERTIES)
                inertial = bool(selected & ModelFlags.BODY_INERTIAL_PROPERTIES)
                self.assertEqual(solver._update_kinematic_state.call_count, int(dof))
                self.assertEqual(solver._scatter_armature_to_groups.call_count, int(dof))
                self.assertEqual(solver._mass_update_requested.fill_.call_count, int(dof) + int(inertial))
                self.assertEqual(launch.call_count, 2 * int(inertial))
                if inertial:
                    self.assertIs(launch.call_args_list[0].args[0], solver_module.compute_spatial_inertia)
                    self.assertIs(launch.call_args_list[1].args[0], solver_module.compute_com_transforms)

    def test_gravity_remains_live_and_invalidates_fk(self):
        """Keep the reset-authored gravity and invalidate the original FK cache."""
        solver = fixture()
        solver.model.gravity[:] = (1.0, -2.0, -3.0)
        with (
            patch.object(packets, "topology_signature", side_effect=AssertionError("gravity topology readback")),
            patch.object(packets, "topology_supported", side_effect=AssertionError("gravity topology scan")),
        ):
            solver_module.SolverFeatherPGS.notify_model_changed(solver, ModelFlags.MODEL_PROPERTIES)
        np.testing.assert_array_equal(solver.model.gravity, [[1.0, -2.0, -3.0]])
        solver._fk_id_cache_valid.zero_.assert_called_once_with()
        solver._mass_update_requested.fill_.assert_not_called()

    def test_structural_unknown_and_mixed_notifications_still_validate(self):
        """Retain both complete checks outside the explicit numeric allowlist."""
        flags = (
            0,
            ModelFlags.BODY_PROPERTIES,
            ModelFlags.JOINT_PROPERTIES,
            ModelFlags.CONSTRAINT_PROPERTIES,
            ModelFlags.TENDON_PROPERTIES,
            ModelFlags.ACTUATOR_PROPERTIES,
            ModelFlags.ALL,
            1 << 17,
            ModelFlags.MODEL_PROPERTIES | ModelFlags.BODY_PROPERTIES,
            ModelFlags.JOINT_DOF_PROPERTIES | (1 << 17),
        )
        for selected in flags:
            solver = fixture()
            with (
                self.subTest(flags=selected),
                patch.object(packets, "topology_signature", return_value=("unchanged",)) as signature,
                patch.object(packets, "topology_supported", return_value=True) as supported,
                patch.object(solver_module.wp, "launch"),
            ):
                solver_module.SolverFeatherPGS.notify_model_changed(solver, selected)
                signature.assert_called_once_with(solver)
                supported.assert_called_once_with(solver)

    def test_changed_body_membership_requires_reconstruction(self):
        """Reject changed body flags and stale admission before normal refreshes."""
        for changed_signature, supported in ((True, True), (False, False)):
            solver = fixture()
            with (
                self.subTest(signature_changed=changed_signature),
                patch.object(
                    packets,
                    "topology_signature",
                    return_value=("changed-body-flags",) if changed_signature else ("unchanged",),
                ),
                patch.object(packets, "topology_supported", return_value=supported),
                self.assertRaisesRegex(RuntimeError, "reconstruct"),
            ):
                solver_module.SolverFeatherPGS.notify_model_changed(solver, ModelFlags.BODY_PROPERTIES)
            solver._update_kinematic_state.assert_not_called()
            solver._fk_id_cache_valid.zero_.assert_not_called()

    def test_body_flag_array_mutation_is_rejected(self):
        """Reject an actual changed body-flag array under BODY and full flags."""
        names = (
            "constraint_mimic_joint0",
            "constraint_mimic_joint1",
            "constraint_mimic_world",
            "joint_type",
            "joint_parent",
            "joint_child",
            "joint_q_start",
            "joint_qd_start",
            "body_flags",
        )
        solver = fixture()
        for name in names:
            setattr(solver.model, name, wp.array([0, 1], dtype=int, device="cpu"))
        solver._row_packets._topology_signature = packets.topology_signature(solver)
        solver.model.body_flags.fill_(2)
        for selected in (ModelFlags.BODY_PROPERTIES, ModelFlags.ALL):
            with self.subTest(flags=selected), self.assertRaisesRegex(RuntimeError, "reconstruct"):
                solver_module.SolverFeatherPGS.notify_model_changed(solver, selected)
        solver._update_kinematic_state.assert_not_called()
        solver._fk_id_cache_valid.zero_.assert_not_called()


if __name__ == "__main__":
    unittest.main()
