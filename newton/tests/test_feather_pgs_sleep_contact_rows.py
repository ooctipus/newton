# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Allocation and force controls for dormant raw contacts."""

import inspect
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import kernels
from newton._src.solvers.feather_pgs import sleep_contact_rows as sleeping


def _array(values, device, dtype=wp.int32):
    return wp.array(values, dtype=dtype, device=device)


def _allocation(device):
    """Build dense, free-MF and untouched-tail allocation inputs."""
    integers = {
        "contact_count": [3],
        "contact_shape0": [0, 1, 2, -99],
        "contact_shape1": [3, 3, 3, -99],
        "shape_body": [0, 1, 2, -1],
        "body_to_articulation": [0, 1, 2],
        "art_to_world": [0, 0, 1],
        "articulation_response_dof_count": [1, 1, 6],
        "body_flags": [1, 1, 1],
        "body_has_response_dofs": [1, 1, 1],
        "is_free_rigid": [0, 0, 1],
        "resolved_worlds": [],
        "world_slot_counter": [2, 0],
        "mf_slot_counter": [0, 0],
        "propagation_slot_counter": [0, 0],
        "dense_contact_world_flag": [0, 0],
        "dense_dropped_contact_rows": [0, 0],
        "mf_dropped_contact_rows": [0, 0],
        "propagation_dropped_contact_rows": [0, 0],
        "capacity_status": [0, 0, 0, 0],
    }
    values = {name: _array(data, device) for name, data in integers.items()}
    for name in (
        "contact_world",
        "contact_slot",
        "contact_art_a",
        "contact_art_b",
        "contact_path",
        "contact_slots_needed",
    ):
        values[name] = wp.full(4, -99, dtype=wp.int32, device=device)
    for name in ("contact_point0", "contact_point1"):
        values[name] = wp.zeros(4, dtype=wp.vec3, device=device)
    values["contact_normal"] = _array([[0, 0, 1]] * 4, device, wp.vec3)
    for name in ("contact_thickness0", "contact_thickness1"):
        values[name] = wp.zeros(4, dtype=float, device=device)
    for name, count in (("body_q", 3), ("shape_transform", 4)):
        values[name] = _array([[0, 0, 0, 0, 0, 0, 1]] * count, device, wp.transform)
    values.update(
        total_num_threads=1,
        has_free_rigid=1,
        propagation_articulated_contacts=0,
        propagation_same_articulation=0,
        propagation_free_free=0,
        contact_gap_gate=0.0,
        same_articulation_contact_gap_gate=0.0,
        articulation_pair_contact_gap_gate=0.0,
        max_constraints=16,
        mf_max_constraints=16,
        propagation_max_constraints=16,
        enable_friction=1,
        contact_friction_gap_threshold=1.0,
        contact_friction_anchor_limit=0,
        contact_friction_articulation_pairs_only=0,
        row_capacity_telemetry=1,
    )
    return values


def _launch_allocate(kernel, values, device):
    wp.launch(kernel, dim=1, inputs=[values[name] for name in inspect.signature(kernel.func).parameters], device=device)


def _check_allocation(test, device):
    names = list(inspect.signature(kernels.allocate_world_contact_slots.func).parameters)
    test.assertEqual(
        list(inspect.signature(sleeping.allocate_awake_world_contact_slots.func).parameters), ["dormant", *names]
    )
    for changes in (
        {},
        {"propagation_articulated_contacts": 1},
        {"propagation_free_free": 1},
        {"max_constraints": 4, "mf_max_constraints": 2},
        {"resolved_worlds": [1, 0]},
        {"contact_count": [5]},
    ):
        original, candidate = _allocation(device), _allocation(device)
        for values in (original, candidate):
            for name, value in changes.items():
                values[name] = _array(value, device) if isinstance(value, list) else value
        candidate["dormant"] = _array([0] * 4, device)
        _launch_allocate(kernels.allocate_world_contact_slots, original, device)
        _launch_allocate(sleeping.allocate_awake_world_contact_slots, candidate, device)
        for name in names[names.index("contact_world") :]:
            np.testing.assert_array_equal(candidate[name].numpy(), original[name].numpy(), err_msg=name)
    values = _allocation(device)
    values["dormant"] = _array([0, 1, 0, 0], device)
    _launch_allocate(sleeping.allocate_awake_world_contact_slots, values, device)
    np.testing.assert_array_equal(values["world_slot_counter"].numpy(), [5, 0])
    np.testing.assert_array_equal(values["mf_slot_counter"].numpy(), [0, 3])
    np.testing.assert_array_equal(values["contact_slot"].numpy(), [2, -1, 0, -99])
    np.testing.assert_array_equal(values["contact_path"].numpy(), [0, -1, 1, -99])
    np.testing.assert_array_equal(values["contact_slots_needed"].numpy(), [3, 0, 3, -99])
    np.testing.assert_array_equal(values["contact_world"].numpy(), [0, 0, 1, -99])
    np.testing.assert_array_equal(values["contact_art_a"].numpy(), [0, 1, 2, -99])
    np.testing.assert_array_equal(values["contact_art_b"].numpy(), [-1, -1, -1, -99])


def _check_forces(test, device):
    count = _array([5], device)
    dormant = _array([1, 1, 1, 1, 0], device)
    normal = _array([[0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]], device, wp.vec3)
    world = _array([0] * 5, device)
    slot = _array([0, 3, -1, -1, 0], device)
    path = _array([0, 0, -1, -1, 0], device)
    impulse = _array([[2, 0.3, -0.4, 1, 8, 9]], device, float)
    row_count = _array([6], device)
    row_type = _array([[0, 2, 2, 0, 2, 2]], device)
    row_parent = _array([[-1, 0, 0, -1, 3, 0]], device)  # Row5 is not row3's sibling.
    force = _array([[7, 8, 9]] * 5, device, wp.vec3)
    valid = _array([1, 0, 1, 0, 0], device)
    status = _array([0], device)
    args = [
        count,
        dormant,
        normal,
        world,
        slot,
        path,
        impulse,
        row_count,
        row_type,
        row_parent,
        1,
        240.0,
        force,
        valid,
        status,
    ]
    # Use the actual original exporter as the dense-force reference.
    reference = wp.zeros(5, dtype=wp.vec3, device=device)
    wp.launch(
        kernels.compute_contact_linear_force_from_impulses,
        dim=5,
        inputs=[
            count,
            normal,
            world,
            slot,
            path,
            impulse,
            impulse,
            impulse,
            row_count,
            row_count,
            row_count,
            row_type,
            row_parent,
            row_type,
            row_parent,
            row_type,
            row_parent,
            1,
            240.0,
            reference,
        ],
        device=device,
    )
    wp.launch(sleeping.capture_dormant_contact_forces, dim=5, inputs=args, device=device)
    test.assertEqual(int(status.numpy()[0]), 0)
    np.testing.assert_allclose(force.numpy()[:2], reference.numpy()[:2], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(force.numpy()[0], [96, -72, -480], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(force.numpy()[1], [-240, 0, 0], rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(force.numpy()[2], [7, 8, 9])
    np.testing.assert_array_equal(force.numpy()[3], [0, 0, 0])
    np.testing.assert_array_equal(force.numpy()[4], [7, 8, 9])
    np.testing.assert_array_equal(valid.numpy(), [1, 1, 1, 1, 0])
    public = wp.full(5, wp.vec3(-10), dtype=wp.vec3, device=device)
    wp.launch(
        sleeping.publish_dormant_contact_forces, dim=5, inputs=[count, dormant, force, valid, public], device=device
    )
    np.testing.assert_array_equal(public.numpy()[:4], force.numpy()[:4])
    np.testing.assert_array_equal(public.numpy()[4], [-10] * 3)
    if device.is_cuda:
        with wp.ScopedCapture(device=device) as capture:
            wp.launch(sleeping.capture_dormant_contact_forces, dim=5, inputs=args, device=device)
            wp.launch(
                sleeping.publish_dormant_contact_forces,
                dim=5,
                inputs=[count, dormant, force, valid, public],
                device=device,
            )
        impulse.assign(np.array([[3, 0.3, -0.4, 1, 8, 9]], np.float32))
        wp.capture_launch(capture.graph)
        test.assertAlmostEqual(float(force.numpy()[0, 2]), -720.0)
    # Unsupported paths, bad world/slot, and executed nonfinite force cannot
    # grant a lease. The unrelated valid skipped contact remains untouched.
    for values, index, expected in (([1, 0, -1, -1, 0], 5, 1), ([9, 0, 0, 0, 0], 3, 2), ([8, 3, -1, -1, 0], 4, 2)):
        original = args[index]
        args[index] = _array(values, device)
        status.zero_()
        wp.launch(sleeping.capture_dormant_contact_forces, dim=5, inputs=args, device=device)
        test.assertEqual(int(status.numpy()[0]), expected)
        test.assertEqual(int(valid.numpy()[0]), 0)
        args[index] = original
    impulse.assign(np.array([[np.nan, 0, 0, 1, 8, 9]], np.float32))
    status.zero_()
    wp.launch(sleeping.capture_dormant_contact_forces, dim=5, inputs=args, device=device)
    test.assertEqual(int(status.numpy()[0]), 4)
    test.assertEqual(int(valid.numpy()[0]), 0)
    count.assign(np.array([6], np.int32))
    status.zero_()
    wp.launch(sleeping.capture_dormant_contact_forces, dim=5, inputs=args, device=device)
    test.assertEqual(int(status.numpy()[0]), 8)
    np.testing.assert_array_equal(valid.numpy(), [0] * 5)


class TestSleepContactRowsCPU(unittest.TestCase):
    def test_factory_api(self):
        """Require the separate dormant-aware allocation kernel."""
        self.assertIsNotNone(sleeping.allocate_awake_world_contact_slots)

    def test_original_allocation_and_dormant_routes(self):
        """Preserve awake allocation and overflow while excluding dormant rows."""
        _check_allocation(self, wp.get_device("cpu"))

    def test_dense_force_capture_and_publication(self):
        """Preserve dense force conventions and reject invalid force leases."""
        _check_forces(self, wp.get_device("cpu"))


@unittest.skipUnless(wp.is_cuda_available(), "CUDA is required")
class TestSleepContactRowsCUDA(unittest.TestCase):
    def test_allocation_force_and_replay(self):
        """Exercise actual allocation and force helpers under native replay."""
        device = wp.get_device("cuda:0")
        _check_allocation(self, device)
        _check_forces(self, device)


if __name__ == "__main__":
    unittest.main()
