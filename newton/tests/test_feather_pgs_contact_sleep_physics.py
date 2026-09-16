# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Physical collision, row and force ownership for loaded sleeping components."""

import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
import newton._src.solvers.feather_pgs.solver_feather_pgs as solver_module
from newton.tests.test_feather_pgs_sleeping import DT, _compare, _physical_pair, _step


def _fixture(test, device, leaves, *, impact=False):
    with (
        mock.patch.object(solver_module, "_CONTACT_ISLANDS", True),
        mock.patch.object(solver_module, "_AWAKE_PIPELINE", True),
    ):
        model, cases, joints, bodies, dofs, targets = _physical_pair(test, device, leaves=leaves, floor=True)
    test.assertIsNone(cases[0].solver._contact_sleep)
    test.assertIsNotNone(cases[1].solver._contact_sleep)
    # Keep a positive collision envelope: exact zero-gap tangency can disappear
    # under roundoff while its physical ground load is unchanged.
    model.shape_gap.fill_(0.001)
    loaded = np.array((0, 1, leaves, leaves + 1))
    pairs = [(0, 4), (1, 4), (5, 9), (6, 9)]
    if impact:
        pairs.extend(((0, 1), (5, 6)))
    for case in cases:
        target = case.control.joint_target_q.numpy()
        target[targets[loaded]] = 0.0  # Gravity, not the drive, must be balanced by ground force.
        if impact:
            raised = np.array((0, leaves))
            q = case.state.joint_q.numpy()
            q[dofs[raised]] = 0.04
            case.state.joint_q.assign(q)
            target[targets[raised]] = (model.body_mass.numpy()[bodies[raised]] + 13.0 * 0.04) / 12.0
            newton.eval_fk(model, case.state.joint_q, case.state.joint_qd, case.state)
        case.control.joint_target_q.assign(target)
        case.pipeline = newton.CollisionPipeline(
            model,
            broad_phase="explicit",
            shape_pairs_filtered=wp.array(pairs, dtype=wp.vec2i, device=device),
            rigid_contact_max=8,
        )
        case.contacts = case.pipeline.contacts()
    return model, cases, joints, bodies, dofs, targets, loaded


def _advance(cases, *, inplace=False):
    for case in cases:
        case.pipeline.collide(case.state, case.contacts)
        _step(case, inplace=inplace)
        _step(case, inplace=inplace)
        case.solver.publish_kinematics(case.state)
        case.solver.update_contacts(case.contacts)


def _capture_tick(case, device, *, stream=None):
    with wp.ScopedCapture(device=device, stream=stream) as capture:
        case.solver.seed_double_buffer_events()
        case.pipeline.collide(case.state, case.contacts)
        case.solver.step(case.state, case.out, case.control, case.contacts, DT)
        case.solver.step(case.out, case.state, case.control, case.contacts, DT)
        case.solver.publish_kinematics(case.state)
        case.solver.update_contacts(case.contacts)
    return capture.graph


def _records(contacts):
    count = int(contacts.rigid_contact_count.numpy()[0])
    shape0 = contacts.rigid_contact_shape0.numpy()[:count]
    shape1 = contacts.rigid_contact_shape1.numpy()[:count]
    order = np.lexsort((shape1, shape0))
    return count, order


def _compare_contacts(test, cases):
    counts_orders = [_records(case.contacts) for case in cases]
    test.assertEqual(counts_orders[0][0], counts_orders[1][0])
    for suffix in (
        "shape0",
        "shape1",
        "point0",
        "point1",
        "offset0",
        "offset1",
        "normal",
        "margin0",
        "margin1",
        "force",
    ):
        first, second = (
            getattr(case.contacts, "rigid_contact_" + suffix).numpy()[:count][order]
            for case, (count, order) in zip(cases, counts_orders, strict=True)
        )
        if suffix.startswith("shape"):
            np.testing.assert_array_equal(second, first, err_msg=suffix)
        else:
            np.testing.assert_allclose(second, first, rtol=3e-5, atol=5e-6, err_msg="contact " + suffix)


def _settle_loaded(test, model, cases, bodies, loaded):
    for tick in range(64):
        _advance(cases)
        if tick % 4 == 3:
            _compare(test, model, cases)
            _compare_contacts(test, cases)
    candidate = cases[1]
    controller = candidate.solver._sleeping
    np.testing.assert_array_equal(controller.body_awake.numpy()[bodies[loaded]], 0)
    count = int(candidate.contacts.rigid_contact_count.numpy()[0])
    test.assertGreater(count, 0)
    owner = candidate.solver._contact_sleep
    np.testing.assert_array_equal(owner.contact_asleep.numpy()[:count], 1)
    np.testing.assert_array_equal(candidate.solver.contact_slot.numpy()[:count], -1)
    np.testing.assert_array_equal(candidate.solver.contact_slots_needed.numpy()[:count], 0)
    test.assertGreater(int(np.sum(cases[0].solver.contact_slots_needed.numpy()[:count])), 0)
    test.assertLess(
        int(np.sum(candidate.solver.constraint_count.numpy())), int(np.sum(cases[0].solver.constraint_count.numpy()))
    )
    cache = owner.cache
    np.testing.assert_array_equal(cache.held_valid.numpy()[:count], 1)
    test.assertTrue(np.all(cache.source_index.numpy()[:count] >= 0))
    np.testing.assert_allclose(
        candidate.contacts.rigid_contact_force.numpy()[:count], cache.held_force.numpy()[:count], rtol=3e-5, atol=5e-6
    )
    test.assertEqual(int(cache.status.numpy()[0]), 0)


class TestFeatherPGSContactSleepPhysics(unittest.TestCase):
    def test_cold_first_step_capture_acquires_loaded_lease(self):
        """Prepare contact ownership explicitly before the first captured step."""
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("Cold captured contact ownership requires CUDA")
        for device in devices:
            # Compile through disposable solvers without warming the actual
            # captured instances. Explicit preparation does not execute a step.
            _model, warm_cases, *_rest = _fixture(self, device, 37)
            _advance(warm_cases)
            model, cases, _joints, bodies, _dofs, _targets, loaded = _fixture(self, device, 37)
            self.assertEqual(cases[1].solver._step, 0)
            for case in cases:
                case.solver.prepare_contact_sleep(case.contacts)
            graphs = [_capture_tick(case, device) for case in cases]
            for _ in range(64):
                for graph in graphs:
                    wp.capture_launch(graph)
            wp.synchronize_device(device)
            for case in cases:
                case.solver.seed_double_buffer_events()
            _compare(self, model, cases)
            _compare_contacts(self, cases)
            candidate = cases[1]
            owner = candidate.solver._contact_sleep
            np.testing.assert_array_equal(candidate.solver._sleeping.body_awake.numpy()[bodies[loaded]], 0)
            count = int(candidate.contacts.rigid_contact_count.numpy()[0])
            self.assertEqual(count, 4)
            np.testing.assert_array_equal(owner.contact_asleep.numpy()[:count], 1)
            np.testing.assert_array_equal(candidate.solver.contact_slot.numpy()[:count], -1)
            self.assertTrue(np.all(owner.cache.source_index.numpy()[:count] >= 0))
            self.assertEqual(int(owner.cache.status.numpy()[0]), 0)

    def test_synchronized_eager_to_new_capture_stream(self):
        """Permit the synchronized stream handoff used by relaxed Lab capture."""
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("Native synchronized stream migration requires CUDA")
        for device in devices:
            model, cases, _joints, bodies, _dofs, _targets, loaded = _fixture(self, device, 37)
            _settle_loaded(self, model, cases, bodies, loaded)
            # Lab finishes eager warmup before capturing on a fresh nonblocking
            # stream. No collision or solver call overlaps the old stream here.
            wp.synchronize_stream(wp.get_stream(device))
            stream = wp.Stream(device)
            with wp.ScopedStream(stream, sync_enter=False):
                graphs = [_capture_tick(case, device, stream=stream) for case in cases]
                for _ in range(4):
                    for graph in graphs:
                        wp.capture_launch(graph, stream=stream)
            wp.synchronize_stream(stream)
            _compare(self, model, cases)
            _compare_contacts(self, cases)
            candidate = cases[1]
            np.testing.assert_array_equal(candidate.solver._sleeping.body_awake.numpy()[bodies[loaded]], 0)
            count = int(candidate.contacts.rigid_contact_count.numpy()[0])
            self.assertEqual(count, 4)
            self.assertTrue(np.all(candidate.solver._contact_sleep.cache.source_index.numpy()[:count] >= 0))

    def test_loaded_static_contacts_preserve_force_and_restore_awake_rows(self):
        """Retire real loaded rows at both branch counts and wake after collision."""
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("Loaded direct-diagonal owner requires CUDA")
        for device in devices:
            for leaves in (37, 108):
                with self.subTest(device=str(device), leaves=leaves):
                    model, cases, _joints, bodies, dofs, targets, loaded = _fixture(self, device, leaves)
                    _settle_loaded(self, model, cases, bodies, loaded)
                    for case in cases:
                        count, order = _records(case.contacts)
                        self.assertEqual(count, 4)
                        force = case.contacts.rigid_contact_force.numpy()[:count][order]
                        self.assertGreater(float(np.min(np.linalg.norm(force, axis=1))), 0.05)
                        np.testing.assert_allclose(
                            np.abs(force[:, 2]), model.body_mass.numpy()[bodies[loaded]], rtol=3e-5, atol=5e-6
                        )
                        case.pipeline.collide(case.state, case.contacts)
                        # This write deliberately follows collision/cache filtering.
                        target = case.control.joint_target_q.numpy()
                        target[targets[0]] += 0.05
                        case.control.joint_target_q.assign(target)
                        _step(case)
                        case.solver.publish_kinematics(case.state)
                        case.solver.update_contacts(case.contacts)
                    _compare(self, model, cases)
                    _compare_contacts(self, cases)
                    candidate = cases[1]
                    self.assertEqual(int(candidate.solver._sleeping.body_awake.numpy()[bodies[0]]), 1)
                    self.assertEqual(int(candidate.solver._sleeping.body_awake.numpy()[bodies[leaves]]), 0)
                    self.assertGreater(float(candidate.state.joint_qd.numpy()[dofs[0]]), 1e-5)
                    count = int(candidate.contacts.rigid_contact_count.numpy()[0])
                    a = candidate.contacts.rigid_contact_shape0.numpy()[:count]
                    b = candidate.contacts.rigid_contact_shape1.numpy()[:count]
                    slot = int(np.flatnonzero((a == 0) | (b == 0))[0])
                    self.assertEqual(int(candidate.solver._contact_sleep.contact_asleep.numpy()[slot]), 0)
                    self.assertGreaterEqual(int(candidate.solver.contact_slot.numpy()[slot]), 0)

    def test_new_collision_wakes_before_filter(self):
        """Wake a loaded sleeping neighbor from an actual newly arriving sphere."""
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("Actual broad-pair wake and narrow filtering require CUDA")
        for device in devices:
            model, cases, _joints, bodies, dofs, _targets, loaded = _fixture(self, device, 37, impact=True)
            grounded = np.array((1, 38))
            _settle_loaded(self, model, cases, bodies, grounded)
            for case in cases:
                count, _ = _records(case.contacts)
                self.assertEqual(count, 2)
                q, qd = case.state.joint_q.numpy(), case.state.joint_qd.numpy()
                q[dofs[0]], qd[dofs[0]] = 0.012, -0.1
                case.state.joint_q.assign(q)
                case.state.joint_qd.assign(qd)
                newton.eval_fk(model, case.state.joint_q, case.state.joint_qd, case.state)
                case.pipeline.collide(case.state, case.contacts)
                count, _ = _records(case.contacts)
                a = case.contacts.rigid_contact_shape0.numpy()[:count]
                b = case.contacts.rigid_contact_shape1.numpy()[:count]
                self.assertTrue(np.any(((a == 0) & (b == 1)) | ((a == 1) & (b == 0))))
                _step(case)
                case.solver.publish_kinematics(case.state)
                case.solver.update_contacts(case.contacts)
            _compare(self, model, cases)
            _compare_contacts(self, cases)
            awake = cases[1].solver._sleeping.body_awake.numpy()
            np.testing.assert_array_equal(awake[bodies[loaded[:2]]], 1)
            self.assertEqual(int(awake[bodies[38]]), 0)

    def test_captured_reset_and_fresh_output_preserve_cached_contacts(self):
        """Preserve loaded leases across replay, reset, fresh output and in-place steps."""
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("Captured loaded-contact lifecycle requires CUDA")
        for device in devices:
            model, cases, _joints, bodies, _dofs, _targets, loaded = _fixture(self, device, 37)
            _settle_loaded(self, model, cases, bodies, loaded)
            mask = wp.array((True, False), dtype=wp.bool, device=device)
            graphs = []
            for case in cases:
                with wp.ScopedCapture(device=device) as capture:
                    case.solver.seed_double_buffer_events()
                    case.solver.reset(case.state, mask)
                    case.pipeline.collide(case.state, case.contacts)
                    case.solver.step(case.state, case.out, case.control, case.contacts, DT)
                    case.solver.step(case.out, case.state, case.control, case.contacts, DT)
                    case.solver.publish_kinematics(case.state)
                    case.solver.update_contacts(case.contacts)
                graphs.append(capture.graph)
            for world in (0, 1):
                mask.assign(np.array((world == 0, world == 1), dtype=np.bool_))
                for graph in graphs:
                    wp.capture_launch(graph)
                _compare(self, model, cases)
                _compare_contacts(self, cases)
                np.testing.assert_array_equal(
                    cases[1].solver._sleeping.body_awake.numpy()[bodies[world * 37 : (world + 1) * 37]], 1
                )
            wp.synchronize_device(device)
            for case in cases:
                case.solver.seed_double_buffer_events()
            _settle_loaded(self, model, cases, bodies, loaded)
            for case in cases:
                case.out = model.state()
            _advance(cases)
            _compare(self, model, cases)
            _compare_contacts(self, cases)
            _advance(cases, inplace=True)
            _compare(self, model, cases)
            _compare_contacts(self, cases)


if __name__ == "__main__":
    unittest.main()
