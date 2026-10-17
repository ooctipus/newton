# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The masked contact-row builder writes the persistent patch row law as the per-contact builder does."""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import solver_feather_pgs as solver_module
from newton.solvers import SolverFeatherPGS
from newton.tests.feather_pgs_patch_parity import (
    assert_contact_impulses_close,
    assert_history_close,
    assert_rows_close,
    contact_rows,
    standing_model,
)
from newton.tests.test_feather_pgs_response_diagonal import _build_mixed_response_model

_DT = 1.0 / 240.0

# The dense route's row arrays and the contact routing that indexes them.
_ROW_ARRAYS = (
    "constraint_count",
    "contact_slot",
    "contact_path",
    "contact_slots_needed",
    "row_type",
    "row_parent",
    "row_mu",
    "row_beta",
    "row_cfm",
    "phi",
    "target_velocity",
    "row_restitution",
    "rhs",
    "impulses",
)


def _arrays(solver):
    """Host copies of the whole row arrays and Jacobian blocks, the slots past each world's row count included."""
    arrays = {name: getattr(solver, name).numpy().copy() for name in _ROW_ARRAYS}
    arrays.update({f"J_by_size[{size}]": solver.J_by_size[size].numpy().copy() for size in solver.size_groups})
    return arrays


def _rows(solver, contacts):
    """Each routed contact's rows by contact index, with the Jacobian rows of every articulation it touches.

    Matrix-free mode reads the world rows. Split mode keeps each articulation's rows in its size group,
    so a contact's Jacobian concatenates its two articulations' group rows.
    """
    if solver.pgs_mode == "matrix_free":
        return contact_rows(solver, contacts, jacobian=True)
    records = contact_rows(solver, contacts)
    count = len(solver.contact_slot.numpy())
    slot, needed = solver.contact_slot.numpy()[:count], solver.contact_slots_needed.numpy()[:count]
    arts = (solver.contact_art_a.numpy()[:count], solver.contact_art_b.numpy()[:count])
    dofs, groups = solver.articulation_response_dof_count.numpy(), solver.art_group_idx.numpy()
    jacobians = {size: solver.J_by_size[size].numpy() for size in solver.size_groups}
    for c, record in records.items():
        rows = slice(int(slot[c]), int(slot[c]) + int(needed[c]))
        record["jacobian"] = np.concatenate(
            [jacobians[int(dofs[art[c]])][groups[art[c]], rows] for art in arts if art[c] >= 0 and dofs[art[c]] > 0],
            axis=1,
        )
    return records


def _rollout(model, *, masked, steps, environment=None, **solver_kwargs):
    """Step the scene with the masked builder on or off; return the solver and each step's rows by contact.

    Each step records its rows by contact, the body poses and, on CPU, the whole row arrays.
    """
    with mock.patch.dict(os.environ, environment or {}), mock.patch.object(solver_module, "_ROWS_MASKED", masked):
        solver = SolverFeatherPGS(model, **solver_kwargs)
        pipeline = newton.CollisionPipeline(model, broad_phase="nxn", reduce_contacts=False)
        contacts = pipeline.contacts()
        states = [model.state(), model.state()]
        control = model.control()
        newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
        records = []
        for _ in range(steps):
            states[0].clear_forces()
            pipeline.collide(states[0], contacts)
            solver.step(states[0], states[1], control, contacts, _DT)
            states.reverse()
            arrays = None if model.device.is_cuda else _arrays(solver)
            records.append((_rows(solver, contacts), states[0].body_q.numpy().copy(), arrays))
        # The record mirrors the row builders under the same import-time switches.
        methods = solver.get_path_selection()["routes"]["contact_rows"]["general_method_by_dof"]
    return solver, records, set(methods.values())


def _mixed_model(device, dof_count):
    """An arm and a free box sliding down onto it: two masked size groups, the box's last."""
    model = _build_mixed_response_model(device, dof_count=dof_count, friction=0.7)
    qd = model.joint_qd.numpy()
    qd[-6:] = (0.3, 0.0, -1.0, 0.0, 0.0, 0.0)
    model.joint_qd.assign(qd)
    return model


def _anchor_count(records):
    return sum(len(record["types"]) == 3 for rows, _, _ in records for record in rows.values())


class TestMaskedPatchRowAdmission(unittest.TestCase):
    def _admit(self, **overrides):
        solver = SimpleNamespace(
            _point_contact_paths=False,
            _patch_dense_paths=True,
            _world_rows_owned_le=lambda: 0,
        )
        flags = {"_ROWS_NATIVE": False, "_ROWS_FUSED": False, "_CHECK_ROWS": False, "_CHECK_ROWS_FUSED": False}
        for name, value in overrides.items():
            if name in flags:
                flags[name] = value
            else:
                setattr(solver, name, value)
        with mock.patch.multiple(solver_module, **flags):
            return SolverFeatherPGS._masked_contact_rows(solver)

    def test_patch_takes_masked_rows_unless_an_experiment_or_the_flag_needs_point_rows(self):
        """Patch friction takes masked rows by policy; point-only row experiments and deferred launches keep it off."""
        self.assertTrue(self._admit())
        self.assertTrue(self._admit(_point_contact_paths=True, _patch_dense_paths=False))
        self.assertTrue(self._admit(_point_contact_paths=True, _ROWS_NATIVE=True, _world_rows_owned_le=lambda: 64))
        self.assertFalse(self._admit(_patch_dense_paths=False))
        for experiment in ("_ROWS_NATIVE", "_ROWS_FUSED", "_CHECK_ROWS", "_CHECK_ROWS_FUSED"):
            with self.subTest(experiment=experiment):
                self.assertFalse(self._admit(**{experiment: True}))
        # In-kernel world rows defer masked launches past the patch link.
        self.assertFalse(self._admit(_world_rows_owned_le=lambda: 64))


class TestMaskedPatchRows(unittest.TestCase):
    def _assert_parity(self, model, steps, **solver_kwargs):
        masked_solver, masked, masked_methods = _rollout(model, masked=True, steps=steps, **solver_kwargs)
        general_solver, general, general_methods = _rollout(model, masked=False, steps=steps, **solver_kwargs)
        self.assertEqual(masked_methods, {"masked_combined"})
        self.assertEqual(general_methods, {"per_contact"})
        self.assertGreater(_anchor_count(masked), 0, "no contact owned an anchor")
        self.assertGreater(
            sum(len(rows) for rows, _, _ in masked), _anchor_count(masked), "every contact owned an anchor"
        )
        # One CPU thread reserves rows in contact order, so both rollouts give each contact the same
        # slots and the builders agree bit for bit. The whole row arrays then also compare, past each
        # world's row count and on rows other contacts own, where a builder that writes rows its contact
        # did not reserve shows even when a later contact overwrites them. CUDA reserves slots in atomic
        # order: rows compare by contact.
        exact = not model.device.is_cuda
        for step, (record, expected) in enumerate(zip(masked, general, strict=True)):
            (rows, poses, arrays), (expected_rows, expected_poses, expected_arrays) = record, expected
            with self.subTest(step=step):
                if exact:
                    for name, values in arrays.items():
                        np.testing.assert_array_equal(values, expected_arrays[name], err_msg=name)
                    assert_rows_close(rows, expected_rows, atol=0.0, rtol=0.0)
                    assert_contact_impulses_close(rows, expected_rows, atol=0.0, rtol=0.0)
                    np.testing.assert_array_equal(poses, expected_poses)
                else:
                    assert_rows_close(rows, expected_rows, atol=1.0e-6, rtol=1.0e-6)
                    assert_contact_impulses_close(rows, expected_rows)
                    np.testing.assert_allclose(poses, expected_poses, atol=2.0e-5, rtol=0.0)
        if exact:
            assert_history_close(masked_solver, general_solver, atol=0.0, rtol=0.0)
        else:
            assert_history_close(masked_solver, general_solver)
        return masked_solver

    def test_cpu_masked_patch_rows_match_the_per_contact_builder(self):
        """Patch rows, ring links, Jacobians and steps equal the per-contact builder's bit for bit (split mode)."""
        for name, model, steps in (
            ("standing", standing_model("cpu"), 4),
            ("arm_and_box", _mixed_model("cpu", 13), 8),
        ):
            with self.subTest(scene=name):
                self._assert_parity(model, steps, pgs_mode="split", pgs_iterations=8)

    def test_cpu_flag_off_restores_the_per_contact_builder(self):
        """FEATHER_PGS_PATCH_DENSE_PATHS=0 keeps patch friction on the per-contact builder."""
        model = standing_model("cpu")
        solver, _, methods = _rollout(
            model, masked=True, steps=1, environment={"FEATHER_PGS_PATCH_DENSE_PATHS": "0"}, pgs_mode="split"
        )
        self.assertEqual(methods, {"per_contact"})
        self.assertFalse(hasattr(solver, "_rows_prelude"))

    @unittest.skipUnless(wp.is_cuda_available(), "the matrix-free dense route requires CUDA")
    def test_cuda_masked_patch_rows_match_the_per_contact_builder(self):
        """Matrix-free dense route: masked patch rows step in parity with the per-contact builder.

        Contact Jacobians of articulations wider than the compact path's 10 DOFs come from the
        general builders; the 23-DOF arm also runs the paired factor sweep over patch units.
        """
        device = wp.get_device("cuda:0")
        for dof_count in (13, 23):
            with self.subTest(dof_count=dof_count):
                solver = self._assert_parity(
                    _mixed_model(device, dof_count),
                    6,
                    pgs_mode="matrix_free",
                    pgs_iterations=8,
                    enable_joint_limits=True,
                    dense_max_constraints=96,
                    mf_max_constraints=32,
                )
                self.assertFalse(solver._compact_contact_jacobian)


if __name__ == "__main__":
    unittest.main()
