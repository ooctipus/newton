# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Current-contact ownership regressions using the existing live Kuka fixture."""

import importlib
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import kernels, kinetic_rows
from newton._src.solvers.feather_pgs import kinetic_current_contact as module
from newton._src.solvers.feather_pgs.kinetic_rows_types import RawRowInput, RowState
from newton.tests import test_feather_pgs_simple_world as simple_tests
from tools.fpgs_bench import test_kinetic_live_bindings as live_tests
from tools.fpgs_bench.test_kinetic_live_bindings import bound_call


class TestCurrentContactZeroLaw(simple_tests.TestSimpleWorld):
    """Run the existing numerical-law cases through the actual new raw owner."""

    def setUp(self):
        original = simple_tests.evaluate

        def compare(data, raw, scratch, *, enabled=1):
            expected = original(data, raw, scratch, enabled=enabled)
            capacity = raw.shape0.size
            cache = module.allocate(1, capacity, "cpu")
            ready = int(enabled != 0 and scratch.limit_ok.numpy()[0] != 0 and np.all(scratch.body_valid.numpy()))
            cache.ready.assign([ready])
            cache.rejected.assign([1 - ready])
            state = RowState()
            state.endpoint_twists = scratch.body_twist
            state.v_hat = data.v_hat
            state.raw_invalid = wp.zeros(1, dtype=int, device="cpu")
            current_raw = RawRowInput()
            for name in simple_tests.sw._SimpleRawContacts.vars:
                setattr(current_raw, name, getattr(raw, name))
            for name in ("body_to_articulation", "art_to_world", "body_q"):
                setattr(current_raw, name, getattr(data, name))
            for name in ("world", "art_a", "art_b"):
                setattr(current_raw, name, wp.empty(capacity, dtype=int, device="cpu"))
            wp.launch(module.get_raw_kernel("cpu"), dim=1, inputs=[1, current_raw, data, state, cache], device="cpu")
            actual = int(cache.rejected.numpy()[0] == 0 and state.raw_invalid.numpy()[0] == 0)
            self.assertEqual(actual, expected)
            return actual

        replacement = patch.object(simple_tests, "evaluate", compare)
        replacement.start()
        self.addCleanup(replacement.stop)


def contact_call(enabled, *, friction_limit=0, shared_anchor=False):
    """Give the unchanged live topology actual shape/material descriptors."""
    original = newton.ModelBuilder.add_link
    original_solver = live_tests.live_solver

    def add_link(builder, *args, **kwargs):
        body = original(builder, *args, **kwargs)
        builder.add_shape_sphere(body=body, radius=0.02)
        return body

    def solver(worlds):
        result = original_solver(worlds)
        # Original CPU ABI shim omits demand-owned dense J. Give the test its
        # real production shape before binding to exercise mixed dense+MF.
        result.J_world = wp.zeros((worlds, 192, 29), dtype=float, device="cpu")
        result.contact_friction_anchor_limit = friction_limit
        result.contact_shared_anchor = shared_anchor
        result.contact_friction_shared_anchor = shared_anchor
        return result

    with (
        patch.object(newton.ModelBuilder, "add_link", add_link),
        patch.object(live_tests, "live_solver", solver),
        patch.dict("os.environ", {"FEATHER_PGS_KUKA_CURRENT_CONTACT": str(int(enabled))}),
    ):
        result = bound_call(worlds=4)
    call = result[-1]
    raw = call.rows.raw
    shape0 = np.full(raw.shape0.shape, -999, np.int32)
    shape1 = shape0.copy()
    # Self dense, free MF, arm/free coupled, and a distant feasible world.
    shape0[:8] = [13, 13, 13, 32 + 30, 64 + 13, 64 + 13, 96 + 13, 64 + 30]
    shape1[:8] = [17, 17, 17, -1, 64 + 30, 64 + 31, -1, -1]
    raw.shape0.assign(shape0)
    raw.shape1.assign(shape1)
    raw.normal.assign(np.tile([0, 0, 1], (raw.normal.size, 1)))
    raw.point0.zero_()
    raw.point1.zero_()
    raw.margin0.fill_(0.2)
    raw.margin1.zero_()
    points = raw.point1.numpy()
    points[6, 2] = 100.0
    raw.point1.assign(points)
    raw.count.assign([8])
    q = call.inputs.joint_q.numpy()
    q[0], q[1] = 0.3, 1.01
    call.inputs.joint_q.assign(q)
    call.construct_launch()
    call.refresh_launch()
    call.free.launch()
    call.predict_launch()
    return result


def device_contact_call(source, device):
    """Move this existing generated current fixture, preserving array aliases.

    This tests the new boundary only: CPU-generated current/held operands are
    inputs, not a claim that dynamics or the full simulation ran on this device.
    """
    device = wp.get_device(device)
    memo = {}

    def copy(value):
        if isinstance(value, wp.array):
            key = (value.ptr, value.dtype, value.size)
            if key not in memo:
                memo[key] = wp.array(value.numpy(), dtype=value.dtype, device=device)
            result = memo[key]
            return result if result.shape == value.shape else result.reshape(value.shape)
        if hasattr(value, "_cls"):
            result = value._cls()
            for name in value._cls.vars:
                setattr(result, name, copy(getattr(value, name)))
            return result
        if isinstance(value, (list, tuple)):
            return [copy(item) for item in value]
        return value

    args = copy(source.current_contact.arguments)
    row_args = copy(source.rows.arguments)
    limit_args = copy(source.solve.allocation.limit_args)
    allocation = copy(source.solve.allocation.data)
    mf_end = copy(source.solve.allocation.mf.contact_end)
    state, raw, out = args[0][4], args[1][1], row_args[0][-1]
    arch = str(device.arch)
    entries = [
        module.get_prepare_kernel(arch),
        module.get_raw_kernel(arch),
        module.get_finalize_kernel(arch),
        module.get_allocate_kernel(arch),
        module.get_contact_kernel(arch),
    ]
    row_kernels = [
        kinetic_rows.get_arm_kernel(arch),
        kinetic_rows.get_prefix_kernel(arch),
        entries[4],
        kinetic_rows.get_validate_kernel(arch),
    ]
    worlds, workers = source.rows.worlds, source.solve.allocation.workers

    def launch():
        wp.launch_tiled(entries[0], dim=[worlds], inputs=args[0], block_dim=32, device=device)
        wp.launch(entries[1], dim=workers, inputs=args[1], device=device)
        wp.launch(entries[2], dim=worlds, inputs=args[2], device=device)
        wp.launch(entries[3], dim=workers, inputs=args[3], device=device)
        wp.copy(mf_end, allocation.mf_counter)
        wp.launch(
            kernels.allocate_rigid_velocity_limit_slots,
            dim=source.solve.allocation.mf.free_bodies.size,
            inputs=limit_args,
            device=device,
        )
        for counter, cap, family, count in (
            (allocation.dense_counter, 192, 0, state.dense_count),
            (allocation.mf_counter, 64, 1, state.mf_count),
            (allocation.propagation_counter, 192, 2, allocation.propagation_count),
        ):
            wp.launch(
                kernels.finalize_constraint_counts_with_status,
                dim=worlds,
                inputs=[counter, cap, family, count, state.capacity_status],
                device=device,
            )
        out.global_status.zero_()
        for kernel, arguments, dim in zip(
            row_kernels, row_args, (worlds, worlds, source.rows.settings.workers, worlds), strict=True
        ):
            wp.launch_tiled(kernel, dim=[dim], inputs=arguments, block_dim=32, device=device)

    return SimpleNamespace(launch=launch, raw=raw, state=state, out=out, device=device)


class TestKineticCurrentContact(unittest.TestCase):
    def compare_current_rows(self, **recipe):
        first = contact_call(False, **recipe)[-1]
        second = contact_call(True, **recipe)[-1]
        for call in (first, second):
            call.zero.launch()
            call.solve.setup()
            call.solve.allocate()
            call.solve.build_rows()
            np.testing.assert_array_equal(call.rows.state.raw_invalid.numpy(), 0)
            np.testing.assert_array_equal(call.rows.state.capacity_status.numpy(), 0)
            np.testing.assert_array_equal(call.rows.out.status.numpy(), 0)
            np.testing.assert_array_equal(call.rows.out.global_status.numpy(), 0)
        np.testing.assert_array_equal(first.rows.state.resolved.numpy(), second.rows.state.resolved.numpy())
        self.assertEqual(int(second.rows.state.resolved.numpy()[3]), 1)
        self.assertEqual(int(second.rows.raw.slot.numpy()[6]), -1)
        self.assertGreater(int(second.rows.state.mf_count.numpy()[1]), 0)
        self.assertGreater(int(second.rows.state.dense_count.numpy()[2]), 0)
        for name in ("world", "slot", "path", "art_a", "art_b", "slots_needed"):
            np.testing.assert_array_equal(
                getattr(first.rows.raw, name).numpy()[:8], getattr(second.rows.raw, name).numpy()[:8]
            )
        self.assertGreater(int(second.rows.state.mf_count.numpy()[2]), 0)
        self.assertGreater(int(second.solve.allocation.data.prefix_count.numpy()[0]), 0)
        for world, count in enumerate(first.rows.state.dense_count.numpy()):
            if count == 0:
                continue
            for name in (
                "response",
                "physical_J",
                "rhs",
                "diag",
                "row_type",
                "row_parent",
                "row_mu",
                "row_beta",
                "row_cfm",
                "phi",
                "target_velocity",
                "row_restitution",
                "row_w",
                "valid",
            ):
                if name == "physical_J" and first.rows.state.mf_count.numpy()[world] == 0:
                    continue
                if name == "row_w" and first.rows.out.row_w.shape[0] == 1:
                    continue
                a = getattr(first.rows.out, name).numpy()[world, :count]
                b = getattr(second.rows.out, name).numpy()[world, :count]
                np.testing.assert_array_equal(a, b, err_msg=name)
        # Same collision generation, changed current poses: no old packet may
        # survive a new substep. The original ZERO/rows remain the oracle.
        for call in (first, second):
            poses = call.rows.raw.body_q.numpy()
            poses[13, 2] += 0.125
            call.rows.raw.body_q.assign(poses)
            call.zero.launch()
            call.solve.allocate()
            call.solve.build_rows()
        np.testing.assert_array_equal(first.rows.out.phi.numpy()[0, :9], second.rows.out.phi.numpy()[0, :9])

    def test_current_rows_match_original_and_zero_never_builds_response(self):
        self.compare_current_rows()

    def test_raw_neighbor_friction_and_shared_anchor(self):
        self.compare_current_rows(friction_limit=1, shared_anchor=True)

    def test_complete_invalid_prefix_and_grow_repair(self):
        """Complete incidence failures may not leak stale ZERO or row slots."""
        _solver, owner, _first, _second, _control, _contacts, call = contact_call(True)
        raw, state = call.rows.raw, call.rows.state
        first, second = raw.shape0.numpy().copy(), raw.shape1.numpy().copy()
        for pair in ((-1, -1), (0, 32), (-2, 0), (999, 0)):
            a, b = first.copy(), second.copy()
            a[0], b[0] = pair
            raw.shape0.assign(a)
            raw.shape1.assign(b)
            call.zero.launch()
            self.assertNotEqual(int(state.raw_invalid.numpy()[0]), 0)
            np.testing.assert_array_equal(state.resolved.numpy(), 0)
            call.solve.allocate()
            np.testing.assert_array_equal(raw.slot.numpy(), -1)
            np.testing.assert_array_equal(raw.path.numpy(), -1)
        raw.shape0.assign(first)
        raw.shape1.assign(second)
        for count in (-1, raw.shape0.size + 1):
            raw.count.assign([count])
            call.zero.launch()
            call.solve.allocate()
            self.assertNotEqual(int(state.capacity_status.numpy()[3]), 0)
        for count in (0, 8, 1, 8):
            raw.count.assign([count])
            call.zero.launch()
            call.solve.allocate()
            call.solve.build_rows()
            np.testing.assert_array_equal(state.raw_invalid.numpy(), 0)
            np.testing.assert_array_equal(state.capacity_status.numpy(), 0)
            np.testing.assert_array_equal(call.rows.out.global_status.numpy(), 0)
        # Invalid current provenance withdraws every world even with no raw
        # contact; count/proof are independent and refreshed on repair.
        raw.count.assign([0])
        generation = call.rows.current.generation.numpy().copy()
        call.rows.current.generation.assign(generation + 1)
        call.zero.launch()
        np.testing.assert_array_equal(state.resolved.numpy(), 0)
        call.rows.current.generation.assign(generation)
        call.zero.launch()
        self.assertEqual(int(state.resolved.numpy()[3]), 1)
        self.assertEqual(owner.bindings.current_contact.values.shape, (256, 20))

    def check_device_boundary(self, device):
        """Actual new entries, raw-ID matched rows and original ZERO/slot law."""
        reference = contact_call(False, friction_limit=1, shared_anchor=True)[-1]
        candidate = device_contact_call(contact_call(True, friction_limit=1, shared_anchor=True)[-1], device)

        def check():
            reference.zero.launch()
            reference.solve.allocate()
            reference.solve.build_rows()
            for name in ("raw_invalid", "capacity_status"):
                np.testing.assert_array_equal(getattr(candidate.state, name).numpy(), 0)
            for name in ("resolved", "dense_count", "mf_count"):
                np.testing.assert_array_equal(
                    getattr(candidate.state, name).numpy(), getattr(reference.rows.state, name).numpy()
                )
            np.testing.assert_array_equal(candidate.out.global_status.numpy(), 0)
            np.testing.assert_array_equal(candidate.out.status.numpy(), 0)
            np.testing.assert_array_equal(candidate.out.phase_bounds.numpy(), reference.rows.out.phase_bounds.numpy())
            count = int(reference.rows.raw.count.numpy()[0])
            for c in range(count):
                path = int(reference.rows.raw.path.numpy()[c])
                self.assertEqual(int(candidate.raw.path.numpy()[c]), path)
                if path != 0:
                    continue
                world = int(reference.rows.raw.world.numpy()[c])
                a, b = int(reference.rows.raw.slot.numpy()[c]), int(candidate.raw.slot.numpy()[c])
                nr = int(reference.rows.raw.slots_needed.numpy()[c])
                self.assertEqual(int(candidate.raw.slots_needed.numpy()[c]), nr)
                names = ("response", "rhs", "r0", "diag", "row_mu", "phi", "target_velocity", "row_restitution")
                if int(reference.rows.state.mf_count.numpy()[world]) > 0:
                    names += ("physical_J",)
                for name in names:
                    expected = getattr(reference.rows.out, name).numpy()[world, a : a + nr]
                    actual = getattr(candidate.out, name).numpy()[world, b : b + nr]
                    scale = 1.0 + np.max(np.abs(expected))
                    self.assertLessEqual(float(np.max(np.abs(actual - expected))) / scale, 2.0**-17, name)
                np.testing.assert_array_equal(
                    candidate.out.row_type.numpy()[world, b : b + nr],
                    reference.rows.out.row_type.numpy()[world, a : a + nr],
                )
                np.testing.assert_array_equal(candidate.out.valid.numpy()[world, b : b + nr], 1)
                self.assertEqual(int(candidate.out.row_parent.numpy()[world, b]), -1)
                if nr == 3:
                    np.testing.assert_array_equal(candidate.out.row_parent.numpy()[world, b + 1 : b + 3], b)

        candidate.launch()
        check()
        if candidate.device.is_cuda:
            with wp.ScopedCapture(device=candidate.device) as first:
                candidate.launch()
            with wp.ScopedCapture(device=candidate.device) as second:
                candidate.launch()
            for count, graph in ((0, first.graph), (8, second.graph), (1, first.graph), (8, second.graph)):
                reference.rows.raw.count.assign([count])
                candidate.raw.count.assign([count])
                wp.capture_launch(graph)
                check()

    def test_rebound_boundary_cpu_control(self):
        self.check_device_boundary("cpu")

    @unittest.skipUnless(os.environ.get("FPGS_TEST_DEVICE", "cpu").startswith("cuda"), "Explicit root-owned CUDA lease")
    def test_cuda_current_contact_boundary(self):
        self.check_device_boundary(os.environ["FPGS_TEST_DEVICE"])

    def test_default_off_and_empty_current_rebuild(self):
        module = importlib.import_module("newton._src.solvers.feather_pgs.kinetic_current_contact")
        self.assertTrue(callable(module.install))
        with patch.dict("os.environ", {"FEATHER_PGS_KUKA_CURRENT_CONTACT": "0"}):
            original = bound_call()
        self.assertIsNone(original[1].bindings.current_contact)
        with patch.dict("os.environ", {"FEATHER_PGS_KUKA_CURRENT_CONTACT": "1"}):
            _solver, owner, _first, _second, _control, _contacts, call = bound_call()
        self.assertIsNotNone(owner.bindings.current_contact)
        call.construct_launch()
        call.refresh_launch()
        call.free.launch()
        call.predict_launch()
        with patch.object(call.zero.buckets, "build", side_effect=AssertionError("retired CSR producer")):
            call.zero.launch()
            np.testing.assert_array_equal(call.rows.state.resolved.numpy(), 1)
            call.solve.setup()
            call.solve.allocate()
            call.solve.build_rows()
        np.testing.assert_array_equal(call.rows.state.dense_count.numpy(), 0)
        np.testing.assert_array_equal(call.rows.out.global_status.numpy(), 0)
        # An empty subsequent substep must overwrite old proof/counter state.
        owner.bindings.current_contact.rejected.fill_(1)
        call.zero.launch()
        np.testing.assert_array_equal(call.rows.state.resolved.numpy(), 1)


if __name__ == "__main__":
    unittest.main()
