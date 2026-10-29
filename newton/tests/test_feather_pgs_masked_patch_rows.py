# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The masked contact-row builder writes the persistent patch row law as the per-contact builder does."""

import contextlib
import dataclasses
import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import path_selection
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
# On CUDA the masked and per-contact kernels contract FMAs differently. Over six steps of the 13-DOF,
# 23-DOF and three-size scenes the rows differed by at most 6.0e-8 in a Jacobian entry (6.0e-7
# relative), 5.1e-8 in a right-hand side and 1.1e-9 in phi; coefficients, targets, impulses, poses
# and anchor history were equal.
_CUDA_ROW_TOLERANCE = 1.0e-6

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
    # The matrix-free sweeps' producers write the packed rows (a placeholder otherwise).
    arrays["_packed_J"] = solver._packed_J.numpy().copy()
    return arrays


def _rows(solver, contacts):
    """Each routed contact's rows by contact index, with the Jacobian rows of every articulation it touches.

    Every row's phi and target are recorded, the tangent rows' included. Matrix-free mode reads the
    world rows. Split mode keeps each articulation's rows in its size group, so a contact's Jacobian
    concatenates its two articulations' group rows.
    """
    if solver.pgs_mode == "matrix_free":
        return contact_rows(solver, contacts, jacobian=True, every_row=True)
    records = contact_rows(solver, contacts, every_row=True)
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


_PLAN = solver_module.contact_row_plan


def _per_contact_plan(solver):
    """The solver's contact-row plan with the per-contact builder in place of every masked method."""
    plan = _PLAN(solver)
    return dataclasses.replace(plan, methods=dict.fromkeys(plan.methods, "per_contact"), combined=())


@contextlib.contextmanager
def _builder(masked):
    """Keep the plan's masked builder, or substitute the per-contact builder in the row build and the record."""
    if masked:
        yield
        return
    with (
        mock.patch.object(solver_module, "contact_row_plan", _per_contact_plan),
        mock.patch.object(path_selection, "contact_row_plan", _per_contact_plan),
    ):
        yield


def _rollout(model, *, masked, steps, pins=None, **solver_kwargs):
    """Step the scene with the masked or the per-contact builder; return the solver and each step's rows by contact.

    The masked builder writes every size of at most 32 DOFs; ``masked=False`` substitutes the
    per-contact builder for it. Each step records its rows by contact, the body poses and, on CPU,
    the whole row arrays.
    """
    with _builder(masked), mock.patch.dict(SolverFeatherPGS._route_pins, pins or {}):
        solver = SolverFeatherPGS(model, **solver_kwargs)
        pipeline = newton.CollisionPipeline(model, broad_phase="nxn", reduce_contacts=False, deterministic=True)
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
        # The record mirrors the row builders of the same plan.
        methods = solver.get_path_selection()["routes"]["contact_rows"]["general_method_by_dof"]
    return solver, records, set(methods.values())


def _mixed_model(device, dof_count):
    """An arm and a free box sliding down onto it: two masked size groups, the box's last."""
    model = _build_mixed_response_model(device, dof_count=dof_count, friction=0.7)
    qd = model.joint_qd.numpy()
    qd[-6:] = (0.3, 0.0, -1.0, 0.0, 0.0, 0.0)
    model.joint_qd.assign(qd)
    return model


def _add_chain(builder, cfg, parent, links, half_width):
    """Append revolute links given as (axis, offset) and a box foot under the last; return the joints."""
    joints = []
    for axis, offset in links:
        body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * 0.05), lock_inertia=True)
        joints.append(
            builder.add_joint_revolute(
                parent=parent,
                child=body,
                axis=axis,
                parent_xform=wp.transform(wp.vec3(*offset), wp.quat_identity()),
                target_ke=400.0,
                target_kd=20.0,
            )
        )
        parent = body
    foot = wp.transform(wp.vec3(0.0, 0.0, -0.2), wp.quat_identity())
    builder.add_shape_box(parent, xform=foot, hx=0.06, hy=half_width, hz=0.03, cfg=cfg)
    return joints


def _add_robot(builder, cfg):
    """Stand a floating torso on two legs of three joints (12 DOFs), each box foot on its four corners at z = 0."""
    torso = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.48), wp.quat_identity()),
        mass=2.0,
        inertia=wp.mat33(np.eye(3) * 0.1),
        lock_inertia=True,
    )
    joints = [builder.add_joint_free(child=torso)]
    for side in (-1.0, 1.0):
        # Hip roll and pitch, then the knee.
        hip, knee = (0.0, 0.15 * side, 0.0), (0.0, 0.0, -0.25)
        legs = ((newton.Axis.X, hip), (newton.Axis.Y, (0.0, 0.0, 0.0)), (newton.Axis.Y, knee))
        joints += _add_chain(builder, cfg, torso, legs, 0.03)
    builder.add_articulation(joints)


def _three_size_model(device, worlds=2):
    """Per world, a standing robot (12 DOFs), a fixed arm resting its tip on the ground (2) and a free box (6).

    Three size groups of at most 32 DOFs take the masked builder size by size instead of in one
    combined launch. The robot's 12 DOFs keep CUDA off the compact contact producer (up to 10 DOFs).
    """
    builder = newton.ModelBuilder()
    cfg = builder.default_shape_cfg.copy()
    cfg.density, cfg.mu = 0.0, 0.8
    for _ in range(worlds):
        builder.begin_world()
        _add_robot(builder, cfg)
        # A two-joint arm on a fixed shoulder, its tip box resting on the ground.
        arm = ((newton.Axis.Y, (0.6, 0.0, 0.33)), (newton.Axis.Y, (0.0, 0.0, -0.1)))
        builder.add_articulation(_add_chain(builder, cfg, -1, arm, 0.06))
        box = builder.add_link(
            xform=wp.transform(wp.vec3(-0.6, 0.0, 0.05), wp.quat_identity()),
            mass=0.5,
            inertia=wp.mat33(np.eye(3) * 0.01),
        )
        builder.add_shape_box(box, hx=0.05, hy=0.05, hz=0.05, cfg=cfg)
        builder.add_articulation([builder.add_joint_free(parent=-1, child=box)])
        builder.end_world()
    builder.add_ground_plane()
    return builder.finalize(device=device)


# The conveyor plate's twist: linear (rising at 2 cm/s), then angular velocity [m/s, rad/s].
_CONVEYOR_TWIST = (0.2, 0.05, 0.02, 0.0, 0.0, 0.5)


def _conveyor_model(device, worlds=2):
    """Per world, the 12-DOF standing robot on a kinematic plate that carries a prescribed twist.

    Every contact row on the plate gets a target velocity. The feet's tangent rows take theirs at the
    patch friction location (the midpoint of a foot's short edge), which the spin moves differently
    from the contact's own corner.
    """
    builder = newton.ModelBuilder()
    cfg = builder.default_shape_cfg.copy()
    cfg.density, cfg.mu = 0.0, 0.8
    for _ in range(worlds):
        builder.begin_world()
        plate = builder.add_body(xform=wp.transform(wp.vec3(0.1, 0.0, -0.02), wp.quat_identity()), is_kinematic=True)
        builder.add_shape_box(plate, hx=0.6, hy=0.6, hz=0.02)
        _add_robot(builder, cfg)
        builder.end_world()
    model = builder.finalize(device=device)
    types, flags, child = model.joint_type.numpy(), model.body_flags.numpy(), model.joint_child.numpy()
    plates = np.flatnonzero(
        (types == int(newton.JointType.FREE)) & (flags[child] & int(newton.BodyFlags.KINEMATIC) != 0)
    )
    qd = model.joint_qd.numpy()
    for start in model.joint_qd_start.numpy()[plates]:
        qd[start : start + 6] = _CONVEYOR_TWIST
    model.joint_qd.assign(qd)
    return model


def _anchor_count(records):
    return sum(len(record["types"]) == 3 for rows, _, _ in records for record in rows.values())


class TestMaskedPatchRows(unittest.TestCase):
    def _assert_parity(self, model, steps, method="masked_combined", **solver_kwargs):
        """Step both builders; return the masked solver.

        ``method`` is the masked route: one combined launch, or one per size.
        """
        masked_solver, masked, masked_methods = _rollout(model, masked=True, steps=steps, **solver_kwargs)
        general_solver, general, general_methods = _rollout(model, masked=False, steps=steps, **solver_kwargs)
        self.assertEqual(masked_methods, {method})
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
                    assert_rows_close(rows, expected_rows, atol=_CUDA_ROW_TOLERANCE, rtol=_CUDA_ROW_TOLERANCE)
                    assert_contact_impulses_close(rows, expected_rows)
                    np.testing.assert_allclose(poses, expected_poses, atol=2.0e-5, rtol=0.0)
        if exact:
            assert_history_close(masked_solver, general_solver, atol=0.0, rtol=0.0)
        else:
            assert_history_close(masked_solver, general_solver)
        return masked_solver

    def test_cpu_masked_patch_rows_match_the_per_contact_builder(self):
        """Patch rows, ring links, Jacobians and steps equal the per-contact builder's bit for bit (split mode)."""
        for name, model, steps, method in (
            ("standing", standing_model("cpu"), 4, "masked_combined"),
            ("arm_and_box", _mixed_model("cpu", 13), 8, "masked_combined"),
            ("three_sizes", _three_size_model("cpu"), 6, "masked"),
        ):
            with self.subTest(scene=name):
                self._assert_parity(model, steps, method, pgs_mode="split", pgs_iterations=8)

    @unittest.skipUnless(wp.is_cuda_available(), "the matrix-free dense route requires CUDA")
    def test_cuda_masked_patch_rows_match_the_per_contact_builder(self):
        """Matrix-free dense route: masked patch rows step in parity with the per-contact builder.

        Contact Jacobians of articulations wider than the compact path's 10 DOFs come from the
        general builders; the 23-DOF arm also runs the paired factor sweep over patch units, and
        three size groups take the masked builder size by size.
        """
        device = wp.get_device("cuda:0")
        for name, model, method in (
            ("arm_and_box_13", _mixed_model(device, 13), "masked_combined"),
            ("arm_and_box_23", _mixed_model(device, 23), "masked_combined"),
            ("three_sizes", _three_size_model(device), "masked"),
        ):
            with self.subTest(scene=name):
                solver = self._assert_parity(
                    model,
                    6,
                    method,
                    pgs_mode="matrix_free",
                    pgs_iterations=8,
                    enable_joint_limits=True,
                    dense_max_constraints=96,
                    mf_max_constraints=32,
                )
                self.assertFalse(solver._compact_contact_jacobian)

    @unittest.skipUnless(wp.is_cuda_available(), "prescribed contact targets need the CUDA matrix-free route")
    def test_cuda_masked_patch_rows_carry_prescribed_targets(self):
        """A robot on a conveyor: each patch row's target matches the per-contact builder's.

        Split mode gives a kinematic body response coordinates instead of prescribing its motion, so
        only the matrix-free route writes targets. The rows compare at the first step, which both
        builders build from the same state. Later states depend on the order of the atomic slot
        reservations, which changes how the sweep splits a foot's load among its four corners (over
        six steps, impulses differed by up to 4e-3 and poses by 7e-8), and a sliding foot then
        releases different anchors. Sparse factors stay off, as they would own a free root's rows.
        """
        model = _conveyor_model(wp.get_device("cuda:0"))
        options = {"pgs_mode": "matrix_free", "pgs_iterations": 8, "dense_max_constraints": 96}
        dense = {"storage": "dense"}
        solver, masked, methods = _rollout(model, masked=True, steps=1, pins=dense, **options)
        _, general, general_methods = _rollout(model, masked=False, steps=1, pins=dense, **options)
        self.assertEqual(methods, {"masked_combined"})
        self.assertEqual(general_methods, {"per_contact"})
        self.assertTrue(solver._has_prescribed_response)
        self.assertFalse(solver._compact_contact_jacobian)
        (rows, _, _), (expected, _, _) = masked[0], general[0]
        assert_rows_close(rows, expected, atol=_CUDA_ROW_TOLERANCE, rtol=_CUDA_ROW_TOLERANCE)
        anchors = [record["target"] for record in rows.values() if len(record["types"]) == 3]
        self.assertGreater(len(anchors), 0, "no contact owned an anchor")
        # The plate's surface moves at about 0.2 m/s along the ground and rises at 0.02 m/s.
        self.assertGreater(float(np.abs(np.concatenate([target[1:] for target in anchors])).max()), 0.1)
        self.assertGreater(float(np.abs([target[0] for target in anchors]).min()), 0.01)


if __name__ == "__main__":
    unittest.main()
