# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check explicit motion velocity without conflating raw-state force operands."""

import dataclasses
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import tree_scan as ts
from newton._src.solvers.feather_pgs.tree_kinetic_layout import TreeLayout
from newton._src.solvers.feather_pgs.tree_kinetic_owner import TreeKineticState
from newton._src.solvers.feather_pgs.tree_kinetic_state import HeldSchedules, get_classify_kernel, get_state_kernel
from newton.tests import feather_pgs_sparse_scene as split
from newton.tests import test_feather_pgs_tree_stages as stages
from tools.fpgs_bench.public_state_oracle import physical_mass
from tools.fpgs_bench.test_tree_kinetic_state import fixture as tree_fixture


def fixture(floating, device):
    """Bind ordinary and cached dynamics to one declared prescale law."""
    model, solver, reference, operator, _, mass = tree_fixture(floating, device)
    model.joint_velocity_limit.fill_(0.25)
    for item in (solver, reference):
        # Exercise the motion/predictor kernels directly on CPU too; complete
        # velocity-limit solving remains CUDA/matrix-free and is tested below.
        item.options = item.laws.options = dataclasses.replace(item.options, enable_joint_velocity_limits=True)
        item._fk_id_cache_enabled = False
        item._passive_spring_stiffness.fill_(0.27)
        item._passive_spring_ref.fill_(-0.1)
        item._passive_joint_damping.fill_(0.19)
    owner = TreeKineticState(operator)
    operator.kinetic_state = owner
    return model, solver, reference, operator, owner, mass


# The reference forms' schedules: the factories' contracts hold at every schedule.
_SCHEDULES = HeldSchedules(ts.Schedule.JUMP_SMEM, ts.Schedule.JUMP_SMEM, ts.Schedule.LEVEL, ts.Schedule.LEVEL)


class TestPrescaledTreeContract(unittest.TestCase):
    def test_motion_input_is_an_explicit_factory_contract(self):
        """Select the alternate motion by factory argument; every repair and finish takes the motion input."""
        layout = TreeLayout(4, 3, 6, 3, 2, 0, (2, 1, 0))
        ordinary = get_state_kernel(False, _SCHEDULES, layout=layout)
        alternate = get_state_kernel(False, _SCHEDULES, layout=layout, motion_velocity=True)
        finish = get_state_kernel(True, _SCHEDULES, layout=layout)
        self.assertIs(ordinary, get_state_kernel(False, _SCHEDULES, layout=layout))
        self.assertIsNot(ordinary, alternate)
        # One launch ABI keeps every stage one call below the kernel; only the alternate repair reads the input,
        # which follows the six common arguments.
        for kernel in (ordinary, alternate, finish):
            self.assertEqual([arg.label for arg in kernel.adj.args][6], "motion_qd")

    def test_classified_repair_reads_no_motion_input(self):
        """Build the classification and the listed repair for ordinary motion only.

        Owners keep the one-pass repair of a motion-velocity layout, which repairs on every step, so
        no owner launches a classified repair of an alternate motion: the classification takes no
        motion velocity (its arguments are the repair's operands, the comparison bank, the two block
        lists and their counts), and the listed repair refuses one. The listed repair keeps the
        launch ABI every finish and repair shares, motion input included, and its generated code
        reads that input nowhere.
        """
        layout = TreeLayout(4, 3, 6, 3, 2, 0, (2, 1, 0))
        classify = get_classify_kernel(layout=layout)
        self.assertEqual(
            [arg.label for arg in classify.adj.args],
            [
                "p",
                "plan",
                "data",
                "cache",
                "requests",
                "global_refresh",
                "bank",
                "repair_blocks",
                "compare_blocks",
                "counts",
            ],
        )
        listed = get_state_kernel(False, _SCHEDULES, layout=layout, listed=True)
        labels = [arg.label for arg in listed.adj.args]
        self.assertEqual(labels[6:], ["motion_qd", "repair_blocks", "repair_count", "bank"])
        wp.init()
        self.assertEqual(stages._input_uses(listed), [])
        with self.assertRaisesRegex(ValueError, "ordinary motion"):
            get_state_kernel(False, _SCHEDULES, layout=layout, motion_velocity=True, listed=True)

    def check_current_motion(self, device):
        for floating in (False, True):
            with self.subTest(device=device, floating=floating):
                model, solver, reference, operator, owner, mass = fixture(floating, device)
                self.assertNotEqual(tuple(mass.permutation), tuple(range(mass.dof_count - 1, -1, -1)))
                state, output, control = model.state(), model.state(), model.control()
                count, dt = mass.dof_count, split.DT
                rng = np.random.default_rng(413)
                raw = rng.uniform(-0.8, 0.8, model.joint_dof_count).astype(np.float32)
                raw[count:] *= 0.03
                state.joint_qd.assign(raw)
                state.body_f.assign(rng.normal(0, 0.07, (model.body_count, 6)).astype(np.float32))
                control.joint_f.assign(rng.normal(0, 0.13, model.joint_dof_count).astype(np.float32))
                newton.eval_fk(model, state.joint_q, state.joint_qd, state)
                reference._last_step_dt = solver._last_step_dt = dt
                augmented = solver._prepare_augmented_state(state, output, control)
                physical = physical_mass(model, state)
                matrices = [physical[w * count : (w + 1) * count, w * count : (w + 1) * count] for w in range(2)]
                previous_generation = None
                held = None
                captured = None
                for epoch, limit in enumerate((0.25, 0.1, 10.0, 10.0, 10.0)):
                    model.joint_velocity_limit.fill_(limit)
                    if reference._fk_id_cache_enabled:
                        reference._fk_id_cache_valid.zero_()
                    ref_aug = reference._prepare_augmented_state(state, model.state(), control)
                    motion = reference._stage1_fk_id(state, ref_aug, model.state())
                    reference._stage1_drives(state, ref_aug, control, dt)
                    rhs = ref_aug.joint_tau.numpy().reshape(2, count).astype(float)
                    actual_motion = motion.numpy()
                    self.assertEqual(motion.ptr, reference.qd_work.ptr)
                    if epoch < 2:
                        self.assertGreater(np.linalg.norm(actual_motion[:count] - raw[:count]), 0.1)
                        np.testing.assert_array_equal(actual_motion[count:], raw[count:])
                        if floating:
                            np.testing.assert_array_equal(actual_motion[:6], raw[:6])
                    else:
                        np.testing.assert_array_equal(actual_motion, raw)
                    augmented.joint_tau.zero_()
                    if captured is not None and epoch != 4:
                        wp.capture_launch(captured)
                    else:
                        owner.begin(state, augmented, motion, dt, epoch in (0, 4))
                        if epoch == 0:
                            solver.mass_update_mask.fill_(1)
                            operator.refresh(augmented)
                            held = solver._sparse_Linv.numpy().copy()
                        owner.predict(state, augmented, control, motion, dt)
                    generation = owner.data.generation.numpy().copy()
                    if previous_generation is not None:
                        np.testing.assert_array_equal(generation - previous_generation, [int(epoch != 3), 0])
                    if epoch == 4:
                        np.testing.assert_array_equal(owner.data.geometry_valid.numpy(), 1)
                        np.testing.assert_array_equal(owner.data.geometry_generation.numpy(), generation)
                    previous_generation = generation
                    expected = np.empty_like(raw, dtype=float)
                    for world, matrix in enumerate(matrices):
                        np.testing.assert_allclose(
                            solver.R_by_size[count].numpy()[world], reference.R_by_size[count].numpy()[world]
                        )
                        expected[world * count : (world + 1) * count] = np.linalg.solve(
                            matrix + np.diag(solver.R_by_size[count].numpy()[world]), rhs[world]
                        )
                    np.testing.assert_allclose(augmented.joint_qdd.numpy(), expected, atol=3e-5, rtol=3e-4)
                    velocity = actual_motion.astype(float) + dt * expected
                    if floating:
                        for world in range(2):
                            start = world * count
                            velocity[start : start + 3] += dt * np.cross(
                                actual_motion[start + 3 : start + 6], actual_motion[start : start + 3]
                            )
                    np.testing.assert_allclose(solver.v_hat.numpy(), velocity, atol=3e-5, rtol=3e-4)
                    np.testing.assert_array_equal(solver._sparse_Linv.numpy(), held)
                    np.testing.assert_array_equal(state.joint_qd.numpy(), raw)
                    np.testing.assert_array_equal(owner.status.numpy(), 0)
                    if epoch == 0 and device.startswith("cuda"):
                        with wp.ScopedCapture(device=device) as capture:
                            owner.begin(state, augmented, motion, dt, False)
                            owner.predict(state, augmented, control, motion, dt)
                        captured = capture.graph
                current = owner.data.current_valid.numpy().copy()
                generation = owner.data.generation.numpy().copy()
                invalid = [
                    wp.zeros(raw.size - 1, dtype=float, device=device),
                    wp.zeros(raw.size, dtype=wp.float64, device=device),
                    wp.zeros(2 * raw.size, dtype=float, device=device)[::2],
                    solver.v_hat,
                ]
                for value in invalid:
                    with self.assertRaises(RuntimeError):
                        owner.begin(state, augmented, value, dt, False)
                    np.testing.assert_array_equal(owner.data.current_valid.numpy(), current)
                    np.testing.assert_array_equal(owner.data.generation.numpy(), generation)
                other = wp.array(raw, dtype=float, device=device)
                with self.assertRaisesRegex(RuntimeError, "differs from the current-cache binding"):
                    owner.predict(state, augmented, control, other, dt)

    def test_current_motion_cpu(self):
        """Match raw passive forces and prescaled bias across mutable working epochs."""
        self.check_current_motion("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA unavailable")
    def test_current_motion_cuda(self):
        """Keep per-world repair and return-to-raw transitions correct under replay."""
        self.check_current_motion("cuda:0")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA unavailable")
    def test_ordinary_dynamics_keep_no_cache(self):
        """Ordinary dynamics (here by the evaluator pin) keep no FK/ID cache of their own."""
        model = split.branched_model(floating=False)
        solver = split.make_solver(model, True, pins={"evaluator": "ordinary"}, enable_joint_velocity_limits=True)
        self.assertIsNone(solver._tree_kinetic_state)
        self.assertFalse(solver._fk_id_cache_enabled)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA unavailable")
    def test_loaded_steps_and_masked_reset(self):
        """Preserve contact, held factors and reset behavior with prescaled motion."""
        for floating in (False, True):
            for mass_interval in (1, 3):
                with self.subTest(floating=floating, mass_interval=mass_interval):
                    model = split.branched_model(floating=floating, branches=2, depth=2)
                    model.joint_velocity_limit.fill_(0.2)
                    solvers = [
                        split.make_solver(
                            model,
                            True,
                            pins=None if enabled else {"evaluator": "ordinary"},
                            enable_joint_velocity_limits=True,
                            update_mass_matrix_interval=mass_interval,
                        )
                        for enabled in (False, True)
                    ]
                    self.assertIsNone(solvers[0]._tree_kinetic_state)
                    self.assertIsNotNone(solvers[1]._tree_kinetic_state)
                    self.assertFalse(solvers[0]._fk_id_cache_enabled)
                    self.assertTrue(solvers[1]._fk_id_cache_enabled)
                    self.assertTrue(solvers[1]._tree_kinetic_state.groups[0].motion_velocity)
                    pairs = [[model.state(), model.state()] for _ in solvers]
                    controls = [model.control() for _ in solvers]
                    pipelines = [newton.CollisionPipeline(model, rigid_contact_max=32) for _ in solvers]
                    contacts = [pipeline.contacts() for pipeline in pipelines]
                    mask = wp.array([True, False], dtype=wp.bool, device=model.device)
                    observed_contact = False
                    for pair in pairs:
                        newton.eval_fk(model, pair[0].joint_q, pair[0].joint_qd, pair[0])
                    for step in range(10):
                        if step == 6:
                            model.joint_velocity_limit.fill_(0.1)
                        for solver, pair, control, pipeline, contact in zip(
                            solvers, pairs, controls, pipelines, contacts, strict=True
                        ):
                            if step == 4:
                                velocity = pair[0].joint_qd.numpy()
                                velocity[: model.joint_dof_count // 2] *= 2.3
                                pair[0].joint_qd.assign(velocity)
                                newton.eval_fk(model, pair[0].joint_q, pair[0].joint_qd, pair[0])
                                solver.reset(pair[0], world_mask=mask)
                            if step == 6:
                                solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
                            if step == 8:
                                solver.notify_state_changed()
                            case = (model, solver, *pair, control, pipeline, contact)
                            split.step(case)
                            solver.check_constraint_capacity()
                            observed_contact |= bool(contact.rigid_contact_count.numpy()[0])
                        for name in split.STATE_FIELDS:
                            np.testing.assert_allclose(
                                getattr(pairs[1][1], name).numpy(),
                                getattr(pairs[0][1], name).numpy(),
                                atol=3e-4,
                                rtol=3e-4,
                                err_msg=name,
                            )
                        for pair in pairs:
                            pair.reverse()
                    self.assertTrue(observed_contact)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA unavailable")
    def test_loaded_graph_replay_and_reset(self):
        """Replay prescale, prediction, contact and publication across masked resets."""
        for floating in (False, True):
            with self.subTest(floating=floating):
                model = split.branched_model(floating=floating, branches=2, depth=2)
                model.joint_velocity_limit.fill_(0.2)
                cases, graphs = [], []
                for enabled in (False, True):
                    solver = split.make_solver(
                        model,
                        True,
                        pins=None if enabled else {"evaluator": "ordinary"},
                        enable_joint_velocity_limits=True,
                        update_mass_matrix_interval=2,
                    )
                    self.assertEqual(solver._tree_kinetic_state is not None, enabled)
                    first, second, control = model.state(), model.state(), model.control()
                    newton.eval_fk(model, first.joint_q, first.joint_qd, first)
                    pipeline = newton.CollisionPipeline(model, rigid_contact_max=32)
                    cases.append((solver, first, second, control, pipeline, pipeline.contacts()))

                def two_steps(case):
                    solver, first, second, control, pipeline, contacts = case
                    model = solver.model
                    for state, output in ((first, second), (second, first)):
                        state.clear_forces()
                        wp.launch(
                            split._external_forces, dim=model.body_count, inputs=[state.body_f], device=model.device
                        )
                        pipeline.collide(state, contacts)
                        solver.step(state, output, control, contacts, split.DT)

                for case in cases:
                    two_steps(case)
                    with wp.ScopedCapture(device=model.device) as capture:
                        two_steps(case)
                    graphs.append(capture.graph)
                mask = wp.array([True, False], dtype=wp.bool, device=model.device)
                contact_seen = False
                for replay in range(6):
                    if replay == 2:
                        model.joint_velocity_limit.fill_(0.12)
                        for solver, state, *_ in cases:
                            coordinates = state.joint_q.numpy()
                            coordinates[7 if floating else 0] += 0.025
                            state.joint_q.assign(coordinates)
                            velocity = state.joint_qd.numpy()
                            velocity[: model.joint_dof_count // 2] *= 1.8
                            state.joint_qd.assign(velocity)
                            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
                            solver.reset(state, world_mask=mask)
                    for graph, case in zip(graphs, cases, strict=True):
                        wp.capture_launch(graph)
                        case[0].check_constraint_capacity()
                        contact_seen |= bool(case[-1].rigid_contact_count.numpy()[0])
                    for name in split.STATE_FIELDS:
                        np.testing.assert_allclose(
                            getattr(cases[1][1], name).numpy(),
                            getattr(cases[0][1], name).numpy(),
                            atol=3e-4,
                            rtol=3e-4,
                            err_msg=name,
                        )
                    np.testing.assert_array_equal(cases[1][0]._tree_kinetic_state.groups[0].status.numpy(), 0)
                self.assertTrue(contact_seen)


if __name__ == "__main__":
    unittest.main()
