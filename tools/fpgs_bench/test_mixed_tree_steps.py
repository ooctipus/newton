# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise complete coupled solves around heterogeneous cached tree groups."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
)
from newton.tests.feather_pgs_patch_parity import (
    assert_contact_impulses_close,
    assert_history_close,
    contact_rows,
    make_loaded,
)
from tools.fpgs_bench.public_state_oracle import physical_mass, public_state
from tools.fpgs_bench.test_mixed_tree_cache import DT, make_solver, model_fixture


def step(case):
    """Advance one step and exchange the state banks."""
    model, solver, state, output, control, contacts = case
    solver.step(state, output, control, contacts, DT)
    return model, solver, output, state, control, contacts


def make_case(enabled, reordered, *, friction_anchor_beta=0.0):
    """Couple the two factor families with a loaded contact in every world."""
    model = model_fixture("cuda:0", reordered=reordered)
    solver = make_solver(model, enabled=enabled, permuted=True, friction_anchor_beta=friction_anchor_beta)
    state, output, control = model.state(), model.state(), model.control()
    state.joint_qd.assign(np.linspace(-0.12, 0.16, model.joint_dof_count, dtype=np.float32))
    state.body_f.assign(np.random.default_rng(111).normal(0, 0.05, (model.body_count, 6)).astype(np.float32))
    output.body_f.assign(state.body_f)
    control.joint_f.assign(np.random.default_rng(112).normal(0, 0.07, model.joint_dof_count).astype(np.float32))
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    contacts = newton.Contacts(rigid_contact_max=32, soft_contact_max=0, device=model.device)
    first, second = np.full(32, -1, np.int32), np.full(32, -1, np.int32)
    point0, point1, normal = (np.zeros((32, 3), np.float32) for _ in range(3))
    shapes, worlds, poses = model.shape_body.numpy(), model.shape_world.numpy(), state.body_q.numpy()
    for world in range(model.world_count):
        first[world], second[world] = np.flatnonzero(worlds == world)
        a, b = shapes[[first[world], second[world]]]
        normal[world] = [0, 0, 1]
        point0[world] = [0.01, -0.01, 0]
        position = wp.transform_point(wp.transform(*poses[a]), wp.vec3(*point0[world]))
        point1[world] = np.asarray(
            wp.transform_point(wp.transform_inverse(wp.transform(*poses[b])), position - wp.vec3(0, 0, 0.001))
        )
    for name, values in (
        ("shape0", first),
        ("shape1", second),
        ("point0", point0),
        ("point1", point1),
        ("normal", normal),
    ):
        getattr(contacts, "rigid_contact_" + name).assign(values)
    contacts.rigid_contact_count.fill_(model.world_count)
    return model, solver, state, output, control, contacts


@unittest.skipUnless(wp.is_cuda_available(), "CUDA unavailable")
class TestMixedTreeSteps(unittest.TestCase):
    friction_anchor_beta = 0.0

    def make_cases(self, reordered):
        """Build the ordinary and the held-tree arm of one fixture."""
        return [
            make_case(enabled, reordered, friction_anchor_beta=self.friction_anchor_beta) for enabled in (False, True)
        ]

    def assert_pair(self, cases):
        """Check public state, finite impulses and unchanged unilateral row laws."""
        for model, solver, state, *_ in cases:
            solver.check_constraint_capacity()
            public_state(self, model, state)
            total = 0
            for prefix in ("", "mf_"):
                counts = getattr(solver, prefix + "constraint_count").numpy()
                impulses = getattr(solver, prefix + "impulses").numpy()
                kinds = getattr(solver, prefix + "row_type").numpy()
                total += int(counts.sum())
                for world, count in enumerate(counts):
                    active = impulses[world, :count]
                    self.assertTrue(np.isfinite(active).all())
                    unilateral = np.isin(
                        kinds[world, :count], [PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_JOINT_LIMIT]
                    )
                    self.assertTrue(np.all(active[unilateral] >= -1e-7))
            self.assertGreater(total, 0)
        for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
            a, b = (getattr(case[2], name).numpy().reshape(case[0].world_count, -1).astype(float) for case in cases)
            error = np.linalg.norm(a - b, axis=1) / (1 + np.linalg.norm(a, axis=1))
            self.assertLess(float(error.max()), 7e-4, name)
        for group in cases[1][1].factor.held.groups:
            np.testing.assert_array_equal(group.status.numpy(), 0)
            np.testing.assert_array_equal(group.data.current_valid.numpy(), 1)

    def assert_factor(self, case, state):
        """Check original held factors against public-twist mass and drive diagonals."""
        model, solver, *_ = case
        physical = physical_mass(model, state)
        starts = solver.forest.articulation_dof_start.numpy()
        rows, coefficients = solver.factor.drive_row_by_dof.numpy(), solver.factor.aug_row_K.numpy()
        for group in solver.factor.held.groups:
            operator, size = group.sparse, group.layout.dofs
            for index, art in enumerate(operator.plan.group_to_art.numpy()):
                ids = np.arange(starts[art], starts[art] + size)
                regularizer = solver.factor.R_by_size[size].numpy()[index].astype(float)
                regularizer += [max(0.0, float(coefficients[rows[dof]])) if rows[dof] >= 0 else 0 for dof in ids]
                expected = physical[np.ix_(ids, ids)] + np.diag(regularizer)
                if operator.diagonal:
                    actual = np.diag(1.0 / solver.factor.diagonal_inverse_mass.numpy()[ids])
                else:
                    lower = np.tril(solver.factor.L_by_size[size].numpy()[index].astype(float))
                    actual = lower @ lower.T
                np.testing.assert_allclose(actual, expected, atol=5e-6, rtol=5e-5)

    def test_loaded_steps_and_model_changes(self):
        """Preserve coupled steps, held factors, selective resets and numeric updates."""
        for reordered in (False, True):
            cases = self.make_cases(reordered)
            self.assertIsNone(cases[0][1].factor.held)
            self.assertIsNotNone(cases[1][1].factor.held)
            held = None
            for index in range(24):
                for arm, case in enumerate(cases):
                    model, solver, state, *_ = case
                    if index in (5, 13, 19):
                        mask = wp.array([index == 5, index == 13, index == 19], dtype=wp.bool, device=model.device)
                        solver.reset(state, world_mask=mask)
                    if index == 9:
                        model.body_mass.assign(model.body_mass.numpy() * np.float32(1.01))
                        model.body_inertia.assign(model.body_inertia.numpy() * np.float32(1.02))
                        solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
                    cases[arm] = step(case)
                    if arm and index == 0:
                        self.assert_factor(cases[arm], state)
                        held = (
                            solver.factor.L_by_size[9].numpy().copy(),
                            solver.factor.diagonal_inverse_mass.numpy().copy(),
                        )
                    if arm and index == 1:
                        np.testing.assert_array_equal(solver.factor.L_by_size[9].numpy(), held[0])
                        np.testing.assert_array_equal(solver.factor.diagonal_inverse_mass.numpy(), held[1])
                self.assert_pair(cases)
            wp.synchronize()  # Finish auxiliary streams before replacing this fixture.

    def test_complete_graph_and_dynamic_masks(self):
        """Replay complete held-factor cycles under changing reset masks."""
        cases = self.make_cases(True)
        graphs, masks = [], []
        for arm in range(len(cases)):
            case = cases[arm]
            for _ in range(4):
                case = step(case)
            mask = wp.array([False] * 3, dtype=wp.bool, device=case[0].device)
            # Finish asynchronous setup allocations before capturing parallel streams.
            wp.synchronize()
            with wp.ScopedCapture(device=case[0].device) as capture:
                case[1].seed_double_buffer_events()
                case[1].reset(case[2], world_mask=mask)
                for _ in range(4):
                    case = step(case)
            cases[arm] = case
            graphs.append(capture.graph)
            masks.append(mask)
        for selected in ([True, False, False], [False, True, False], [False, False, True], [False] * 3, [True] * 3):
            for mask, graph in zip(masks, graphs, strict=True):
                mask.assign(np.array(selected))
                wp.capture_launch(graph)
            self.assert_pair(cases)
        wp.synchronize()  # Keep fixture arrays alive until all captured streams finish.


class TestPatchMixedTreeSteps(TestMixedTreeSteps):
    """Run the held-tree owner around the ordinary persistent-friction row pipeline."""

    friction_anchor_beta = 0.2

    def assert_pair(self, cases):
        """Add close contact impulses and the anchor lifecycle to the inherited physical checks."""
        super().assert_pair(cases)
        for case in cases:
            self.assertTrue(case[1].laws.friction_anchors_enabled)
            self.assertFalse(case[1]._point_contact_paths)
        # Per contact, within 5e-3 of its normal impulse: in the reordered fixture world 2's contact
        # slides off an impact at steps 9-13, where 1e-6 rounding moves the ordinary route's own
        # saturated rows by up to 2.1e-3 of the falling load (max ratio 0.56 to this bound on CPU).
        rows = [contact_rows(case[1], case[-1]) for case in cases]
        assert_contact_impulses_close(rows[0], rows[1], atol=2e-5, rtol=0.0, load_rtol=5e-3)
        # The other rows by slot, at the patch history contract.
        reference, candidate = (case[1] for case in cases)
        for world, count in enumerate(reference.constraint_count.numpy()):
            kinds = reference.row_type.numpy()[world, :count]
            other = (kinds != PGS_CONSTRAINT_TYPE_CONTACT) & (kinds != PGS_CONSTRAINT_TYPE_FRICTION)
            np.testing.assert_array_equal(candidate.row_type.numpy()[world, :count][other], kinds[other])
            np.testing.assert_allclose(
                candidate.impulses.numpy()[world, :count][other],
                reference.impulses.numpy()[world, :count][other],
                atol=2e-5,
                rtol=7e-4,
            )
        assert_history_close(reference, candidate)

    def test_patch_rows_are_loaded(self):
        """Carry sticking anchors and their friction impulse into the next frame."""
        cases = self.make_cases(False)
        for model, _solver, state, _output, _control, contacts in cases:
            make_loaded(model, state, contacts)
        cases = [step(case) for case in cases]
        self.assert_pair(cases)
        for case in cases:
            patches = case[1].laws.friction_patches
            self.assertGreater(np.count_nonzero(patches.view.weight.numpy()), 0)
            self.assertGreater(np.count_nonzero(patches.previous.valid.numpy()), 0)
            self.assertGreater(float(np.linalg.norm(patches.previous.tangent_impulse.numpy())), 0.0)
        cases = [step(case) for case in cases]
        self.assert_pair(cases)
        for case in cases:
            self.assertTrue(np.any(case[1].laws.friction_patches.current.source.numpy() >= 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
