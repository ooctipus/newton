# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check callback-local topology reads without changing device physics."""

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from newton import ModelFlags
from newton._src.solvers.feather_pgs import (
    kinetic_live_bindings,
    kinetic_live_owner,
    kuka_joint_owner,
    world_scan_owner,
)
from newton._src.solvers.feather_pgs.solver_feather_pgs import SolverFeatherPGS


class Readback:
    """Count actual reads while returning independent host snapshots."""

    def __init__(self, values):
        self.values = np.asarray(values)
        self.shape = self.values.shape[:-1] if self.values.ndim == 2 else self.values.shape
        self.is_contiguous = True
        self.calls = 0

    def numpy(self):
        """Model Warp's separate CPU snapshot, without using a GPU."""
        self.calls += 1
        return self.values.copy()


def notification_solver(*, kinetic=True):
    """Bind the three real validator methods to a small current model."""
    model = SimpleNamespace(**{name: Readback([1, 2, 3]) for name in world_scan_owner.PLAN_FIELDS})
    model.joint_X_p = Readback([0.0, 0.0, 0.0])
    model.gravity = Readback([[0.0, 0.0, -9.81], [0.0, 0.0, -8.0], [0.0, 0.0, -9.81]])
    model.body_count = 0
    publications = []
    solver = SimpleNamespace(
        model=model,
        _row_packets=None,
        _allegro_kinetic_rows=None,
        _kinetic_world=None,
        _fk_id_cache_enabled=False,
        _update_kinematic_state=lambda: publications.append("kinematic"),
        _scatter_armature_to_groups=lambda: publications.append("armature"),
        _mass_update_requested=SimpleNamespace(fill_=lambda value: publications.append(("mass", value))),
    )

    def expected():
        """Keep each owner's own admission values, not a shared truth shortcut."""
        return {name: getattr(model, name).values.copy() for name in world_scan_owner.PLAN_FIELDS}

    joint = object.__new__(kuka_joint_owner.JointWorldOwner)
    joint.solver, joint.model_plan_values = solver, expected()
    joint.join_raw = lambda: None
    publication = object.__new__(world_scan_owner.WorldScanOwner)
    publication.solver, publication.model_plan_values = solver, expected()
    solver._joint_world, solver._world_scan_publication = joint, publication
    if kinetic:
        owner = object.__new__(kinetic_live_owner.KineticWorldOwner)
        bindings = object.__new__(kinetic_live_bindings.LiveBindings)
        bindings.solver, bindings.worlds, bindings.model_plan_values = solver, 2, expected()
        owner.bindings, owner.last_call = bindings, None
        owner.reset = lambda state: publications.append("private_invalidate")
        solver._kinetic_world = owner
    return solver, publications


class TestModelNotificationSnapshot(unittest.TestCase):
    """Preserve rejection semantics while deleting repeated D2H snapshots."""

    def test_three_owners_share_one_current_snapshot(self):
        """Read each immutable field once with all current owners enabled."""
        solver, publications = notification_solver()
        solver.model.joint_X_p.values[0] = 4.0
        SolverFeatherPGS.notify_model_changed(solver, ModelFlags.JOINT_PROPERTIES)
        self.assertEqual(publications, ["private_invalidate"])
        self.assertEqual([getattr(solver.model, name).calls for name in world_scan_owner.PLAN_FIELDS], [1] * 8)

    def test_baseline_two_owners_also_share_snapshot(self):
        """Delete the existing duplicate even when kinetic ownership is disabled."""
        solver, publications = notification_solver(kinetic=False)
        SolverFeatherPGS.notify_model_changed(solver, ModelFlags.JOINT_PROPERTIES)
        self.assertEqual(publications, [])
        self.assertEqual([getattr(solver.model, name).calls for name in world_scan_owner.PLAN_FIELDS], [1] * 8)

    def test_mutated_topology_rejects_before_publication(self):
        """Retain conservative topology rejection for joint-only and ALL flags."""
        for flags in (ModelFlags.JOINT_PROPERTIES, ModelFlags.ALL):
            with self.subTest(flags=flags):
                solver, publications = notification_solver()
                solver.model.joint_parent.values[0] += 1
                with self.assertRaisesRegex(ValueError, "topology changed"):
                    SolverFeatherPGS.notify_model_changed(solver, flags)
                self.assertEqual(publications, [])

    def test_each_owner_retains_its_own_comparison(self):
        """A later owner's incompatible proof fails before private invalidation."""
        solver, publications = notification_solver()
        solver._world_scan_publication.model_plan_values["joint_type"][0] += 1
        with self.assertRaisesRegex(RuntimeError, "World publication static ownership changed"):
            SolverFeatherPGS.notify_model_changed(solver, ModelFlags.JOINT_PROPERTIES)
        self.assertEqual(publications, [])

    def test_snapshot_does_not_survive_the_notification(self):
        """A later write cannot reuse a previous call's successful snapshot."""
        solver, publications = notification_solver()
        SolverFeatherPGS.notify_model_changed(solver, ModelFlags.JOINT_PROPERTIES)
        publications.clear()
        before = solver.model.joint_parent.calls
        solver.model.joint_parent.values[0] += 1
        with self.assertRaisesRegex(ValueError, "topology changed"):
            SolverFeatherPGS.notify_model_changed(solver, ModelFlags.JOINT_PROPERTIES)
        self.assertEqual(publications, [])
        self.assertEqual(solver.model.joint_parent.calls, before + 1)

    def test_numeric_notifications_do_not_read_topology(self):
        """Retain the numeric fast path and the current world-gravity check."""
        solver, publications = notification_solver()
        for flags in (ModelFlags.SHAPE_PROPERTIES, ModelFlags.MODEL_PROPERTIES):
            SolverFeatherPGS.notify_model_changed(solver, flags)
        self.assertEqual([getattr(solver.model, name).calls for name in world_scan_owner.PLAN_FIELDS], [0] * 8)
        self.assertEqual(solver.model.gravity.calls, 1)
        self.assertEqual(publications, ["private_invalidate"])
        publications.clear()
        solver.model.gravity.values[1, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite owned-world gravity"):
            SolverFeatherPGS.notify_model_changed(solver, ModelFlags.MODEL_PROPERTIES)
        self.assertEqual(publications, [])

    def test_no_owner_needs_no_shared_snapshot(self):
        """Keep ordinary solvers free of optional owner readback work."""
        solver, publications = notification_solver(kinetic=False)
        solver._joint_world = solver._world_scan_publication = None
        with mock.patch.object(solver.model.joint_parent, "numpy", side_effect=AssertionError("unexpected read")):
            SolverFeatherPGS.notify_model_changed(solver, ModelFlags.JOINT_PROPERTIES)
        self.assertEqual(publications, [])

    def test_numeric_flags_allocate_no_snapshot(self):
        """Create callback-local dictionaries only for structural admission work."""
        for flags in (
            ModelFlags.JOINT_DOF_PROPERTIES,
            ModelFlags.BODY_INERTIAL_PROPERTIES,
            ModelFlags.SHAPE_PROPERTIES,
            ModelFlags.MODEL_PROPERTIES,
        ):
            self.assertIsNone(kuka_joint_owner.notification_plan_snapshot(flags))
        first = kuka_joint_owner.notification_plan_snapshot(ModelFlags.JOINT_PROPERTIES)
        second = kuka_joint_owner.notification_plan_snapshot(ModelFlags.JOINT_PROPERTIES)
        self.assertEqual(first, {})
        self.assertIsNot(first, second)

    def test_sparse_validation_precedes_kinetic_invalidation(self):
        """Preserve the accepted owner's fail-before-mutation notification gate."""
        solver, publications = notification_solver()
        validator = mock.Mock(side_effect=ValueError("sparse ownership changed"))
        solver._sparse_factor = SimpleNamespace(validate_notification=validator)
        with self.assertRaisesRegex(ValueError, "sparse ownership changed"):
            SolverFeatherPGS.notify_model_changed(solver, ModelFlags.JOINT_PROPERTIES)
        validator.assert_called_once_with(ModelFlags.JOINT_PROPERTIES)
        self.assertEqual(publications, [])
        self.assertEqual([getattr(solver.model, name).calls for name in world_scan_owner.PLAN_FIELDS], [0] * 8)


if __name__ == "__main__":
    unittest.main()
