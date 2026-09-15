# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise the early MF0 owner through the existing CPU live-array fixture.

The native original dense-eight CPU arm is the numerical oracle for MF0.
Positive-MF checks cover retained row ownership and routing, not an original
CUDA general solve running on CPU through the diagnostic constructor shim.
"""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import kernels, kinetic_live_owner, kinetic_rows, kinetic_solve
from newton._src.solvers.feather_pgs import kinetic_current_contact as current_contact
from newton._src.solvers.feather_pgs import kinetic_private_response as private_response
from newton._src.solvers.feather_pgs.solver_feather_pgs import SolverFeatherPGS
from tools.fpgs_bench.test_kinetic_current_contact import contact_call
from tools.fpgs_bench.test_kinetic_live_bindings import live_model


def private_call(enabled, *, current=True):
    """Bind the actual opt-in owner without changing the fixture's topology."""
    with patch.dict("os.environ", {"FEATHER_PGS_KUKA_PRIVATE_RESPONSE": str(int(enabled))}):
        call = contact_call(current)[-1]
    # Oblique contacts exercise articulated response as well as the active
    # prefix limit in world 0; the original fixture's vertical normal can be
    # orthogonal to every revolute response axis.
    call.rows.raw.normal.assign(np.tile([0.6, 0.0, 0.8], (call.rows.raw.normal.size, 1)))
    return call


def solve_tail(call):
    """Execute the unchanged post-allocation production closures."""
    call.solve.build_rows()
    call.solve.prepare_mf()
    call.solve.qualify()
    call.solve.materialize()
    call.solve.solve()


def device_boundary(source, device, *, private):
    """Copy the existing boundary descriptors and run actual native entries.

    As in device_contact_call, current geometry and held operators are generated
    on CPU. On CUDA both the original and new row/eight-sweep arms run native
    CUDA arithmetic; this is not a full CUDA dynamics or MF-general fixture.
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
    private_args = copy(source.private_response.arguments)
    mapped_args = copy(source.private_response.allocator_arguments)
    solve_args = copy(source.solve.arguments["offset_eight"])
    plan, held, state, out, solve = solve_args
    raw = private_args[3]
    worlds, workers, arch = source.rows.worlds, source.solve.allocation.workers, str(device.arch)
    prepare = current_contact.get_prepare_kernel(arch)
    geometry = current_contact.get_raw_kernel(arch)
    finalize = current_contact.get_finalize_kernel(arch)
    allocator = private_response.get_allocate_kernel(arch) if private else current_contact.get_allocate_kernel(arch)
    allocator_args = mapped_args if private else args[3]
    response_eight = private_response.get_kernel(arch)
    offset_eight = kinetic_solve.get_solve_kernel(arch)
    # These are the actual installed complement kernels. Their native snippets
    # select the CUDA branch when compiled on CUDA, even from a CPU-bound call.
    row_kernels = (
        source.private_response.complementary_rows
        if private
        else (
            kinetic_rows.get_arm_kernel(arch),
            kinetic_rows.get_prefix_kernel(arch),
            current_contact.get_contact_kernel(arch),
            kinetic_rows.get_validate_kernel(arch),
        )
    )

    def launch():
        out.global_status.zero_()
        solve.status.zero_()
        wp.copy(solve.v_out, state.v_hat)
        out.response.fill_(12345.25 if private else 0.0)
        out.physical_J.fill_(12345.25 if private else 0.0)
        wp.launch_tiled(prepare, dim=[worlds], inputs=args[0], block_dim=32, device=device)
        wp.launch(geometry, dim=workers, inputs=args[1], device=device)
        wp.launch(finalize, dim=worlds, inputs=args[2], device=device)
        wp.launch(allocator, dim=workers, inputs=allocator_args, device=device)
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
        # This boundary fixture deliberately does not qualify/solve MF worlds.
        # Their positive finalized count tells the original offset owner to skip.
        wp.copy(solve.selector, state.mf_count)
        if private:
            wp.launch_tiled(response_eight, dim=[worlds], inputs=private_args, block_dim=32, device=device)
        for kernel, arguments, dim in zip(
            row_kernels, row_args, (worlds, worlds, source.rows.settings.workers, worlds), strict=True
        ):
            wp.launch_tiled(kernel, dim=[dim], inputs=arguments, block_dim=32, device=device)
        if not private:
            wp.launch_tiled(offset_eight, dim=[(worlds + 1) // 2], inputs=solve_args, block_dim=64, device=device)

    return SimpleNamespace(
        launch=launch,
        device=device,
        plan=plan,
        held=held,
        state=state,
        raw=raw,
        out=out,
        solve=solve,
        slot_raw=private_args[-1],
        max_linear=copy(source.services.max_linear),
        limit_slot=copy(source.services.limit_slot),
    )


class TestKineticPrivateResponse(unittest.TestCase):
    """Check actual early execution, retired panels and finalized admission."""

    def assert_scaled_close(self, actual, expected):
        """Retain the existing physical velocity bound, not bit identity."""
        actual = np.asarray(actual, dtype=np.float64)
        expected = np.asarray(expected, dtype=np.float64)
        self.assertTrue(np.isfinite(actual).all())
        self.assertTrue(np.isfinite(expected).all())
        if actual.size:
            self.assertLessEqual(float(np.max(np.abs(actual - expected) / (1.0 + np.abs(expected)))), 3.0e-5)

    def compare_step(self, original, candidate):
        """Compare the real early owner with original rows plus dense eight."""
        self.assertIsNotNone(getattr(candidate, "private_response", None))
        original.zero.launch()
        original.solve.setup()
        original.solve.allocate()
        solve_tail(original)

        state, out = original.rows.state, original.rows.out
        selected = (state.resolved.numpy() == 0) & (state.mf_count.numpy() == 0)
        untouched = state.mf_count.numpy() == 0
        self.assertTrue(selected[0])
        candidate.zero.launch()
        candidate.solve.setup()
        poison = {}
        for name in ("response", "physical_J"):
            values = getattr(candidate.rows.out, name).numpy()
            values[untouched] = 12345.25
            getattr(candidate.rows.out, name).assign(values)
            poison[name] = values[untouched].copy()
        candidate.solve.allocate()
        # CPU execution is synchronous: the complete selected solve must have
        # happened at allocation, before the old row and MF producer chain.
        ids = original.plan.dof_ids.numpy()[:, :29]
        for world in np.flatnonzero(selected):
            self.assert_scaled_close(
                candidate.solve.solve_descriptor.v_out.numpy()[ids[world]],
                original.solve.solve_descriptor.v_out.numpy()[ids[world]],
            )
        solve_tail(candidate)

        for call in (original, candidate):
            np.testing.assert_array_equal(call.rows.state.raw_invalid.numpy(), 0)
            np.testing.assert_array_equal(call.rows.state.capacity_status.numpy(), 0)
            np.testing.assert_array_equal(call.rows.out.status.numpy(), 0)
            np.testing.assert_array_equal(call.rows.out.global_status.numpy(), 0)
            np.testing.assert_array_equal(call.solve.solve_descriptor.status.numpy(), 0)
            np.testing.assert_array_equal(call.solve.guard.frame_status.numpy(), 0)
        for name in ("resolved", "dense_count", "mf_count"):
            np.testing.assert_array_equal(getattr(candidate.rows.state, name).numpy(), getattr(state, name).numpy())
        for name in ("world", "slot", "path", "slots_needed"):
            np.testing.assert_array_equal(
                getattr(candidate.rows.raw, name).numpy(), getattr(original.rows.raw, name).numpy()
            )
        for name, expected in poison.items():
            np.testing.assert_array_equal(getattr(candidate.rows.out, name).numpy()[untouched], expected)
        for world, count in enumerate(state.dense_count.numpy()):
            for name in ("row_type", "row_parent", "valid"):
                np.testing.assert_array_equal(
                    getattr(candidate.rows.out, name).numpy()[world, :count],
                    getattr(out, name).numpy()[world, :count],
                )
            for name in ("rhs", "r0", "diag", "row_mu", "phi", "target_velocity", "impulses"):
                self.assert_scaled_close(
                    getattr(candidate.rows.out, name).numpy()[world, :count],
                    getattr(out, name).numpy()[world, :count],
                )
            if not untouched[world]:
                for name in ("response", "physical_J"):
                    self.assert_scaled_close(
                        getattr(candidate.rows.out, name).numpy()[world, :count],
                        getattr(out, name).numpy()[world, :count],
                    )
        np.testing.assert_array_equal(candidate.rows.out.phase_bounds.numpy(), out.phase_bounds.numpy())
        for world in np.flatnonzero(selected):
            self.assert_scaled_close(
                candidate.solve.solve_descriptor.v_out.numpy()[ids[world]],
                original.solve.solve_descriptor.v_out.numpy()[ids[world]],
            )

    def test_default_off_and_current_contact_dependency(self):
        """Install an actual owner only when both experimental flags are set."""
        for enabled, current in ((False, False), (False, True), (True, False)):
            with self.subTest(private=enabled, current=current):
                call = private_call(enabled, current=current)
                self.assertIsNone(getattr(call, "private_response", None))
        call = private_call(True)
        owner = getattr(call, "private_response", None)
        self.assertIsNotNone(owner)
        self.assertTrue(callable(owner.kernel.func))
        self.assertTrue(owner.arguments)

    def test_early_eight_retires_selected_panels_and_preserves_fallback(self):
        """Solve MF0 before old rows while preserving positive-MF and ZERO."""
        original, candidate = private_call(False), private_call(True)
        self.compare_step(original, candidate)
        self.assertGreater(int(candidate.rows.state.dense_count.numpy()[0]), 0)
        np.testing.assert_array_equal(candidate.rows.state.mf_count.numpy()[[1, 2]] > 0, True)
        self.assertEqual(int(candidate.rows.state.resolved.numpy()[3]), 1)
        count = int(candidate.rows.state.dense_count.numpy()[0])
        self.assertGreater(float(np.max(np.abs(candidate.rows.out.impulses.numpy()[0, :count]))), 1.0e-3)

    def test_current_geometry_and_count_reuse_preserve_held_operator(self):
        """Rebuild current slot maps through shrink, empty and grow transitions."""
        original, candidate = private_call(False), private_call(True)
        held = candidate.rows.held.T.numpy().copy()
        inverse = candidate.rows.held.inverse6.numpy().copy()
        for count in (8, 1, 0, 8):
            with self.subTest(raw_count=count):
                for call in (original, candidate):
                    call.rows.raw.count.assign([count])
                    poses = call.rows.raw.body_q.numpy()
                    poses[13, 0] += 0.125
                    call.rows.raw.body_q.assign(poses)
                self.compare_step(original, candidate)
                np.testing.assert_array_equal(candidate.rows.held.T.numpy(), held)
                np.testing.assert_array_equal(candidate.rows.held.inverse6.numpy(), inverse)
        # The original allocator withdraws a capacity-dropping call. A stale
        # private map must neither authorize public finish nor survive repair.
        for call in (original, candidate):
            call.rows.raw.shape0.fill_(13)
            call.rows.raw.shape1.fill_(17)
            call.rows.raw.count.assign([call.rows.raw.shape0.size])
            call.zero.launch()
            call.solve.setup()
            call.solve.allocate()
            solve_tail(call)
            self.assertNotEqual(int(call.rows.state.capacity_status.numpy()[0]), 0)
            self.assertNotEqual(int(call.solve.guard.frame_status.numpy()[0]), 0)
            before = call.next_publication.joint_q.numpy().copy()
            call.finish_launch()
            np.testing.assert_array_equal(call.next_publication.joint_q.numpy(), before)
        np.testing.assert_array_equal(candidate.rows.raw.slot.numpy(), original.rows.raw.slot.numpy())
        np.testing.assert_array_equal(candidate.rows.state.dense_count.numpy(), original.rows.state.dense_count.numpy())
        for call in (original, candidate):
            call.rows.raw.count.assign([8])
        self.compare_step(original, candidate)
        np.testing.assert_array_equal(candidate.rows.held.T.numpy(), held)
        np.testing.assert_array_equal(candidate.rows.held.inverse6.numpy(), inverse)

    def test_finalized_rigid_limit_withdraws_private_world(self):
        """Retain positive-MF fallback when the final allocator adds type 4."""
        original, candidate = private_call(False), private_call(True)
        self.assertIsNotNone(getattr(candidate, "private_response", None))
        for call in (original, candidate):
            limits = call.services.max_linear.numpy()
            limits[30] = 0.01
            call.services.max_linear.assign(limits)
            velocity = call.rows.state.v_hat.numpy()
            ids = call.plan.dof_ids.numpy()[0, 23:29]
            velocity[ids[0]] = 1.0
            call.rows.state.v_hat.assign(velocity)
            call.zero.launch()
            call.solve.setup()
            call.rows.out.response.fill_(12345.25)
            call.solve.allocate()
            self.assertGreater(int(call.rows.state.mf_count.numpy()[0]), 0)
            # No private solve may run from a contact-only MF0 decision.
            np.testing.assert_array_equal(
                call.solve.solve_descriptor.v_out.numpy()[ids], call.rows.state.v_hat.numpy()[ids]
            )
            solve_tail(call)
            np.testing.assert_array_equal(call.solve.guard.frame_status.numpy(), 0)
            count = int(call.rows.state.mf_count.numpy()[0])
            self.assertTrue(np.any(call.services.row_type.numpy()[0, :count] == 4))
            dense = int(call.rows.state.dense_count.numpy()[0])
            self.assertFalse(np.all(call.rows.out.response.numpy()[0, :dense] == 12345.25))
        count = int(original.rows.state.dense_count.numpy()[0])
        for name in ("response", "physical_J", "rhs", "diag"):
            self.assert_scaled_close(
                getattr(candidate.rows.out, name).numpy()[0, :count],
                getattr(original.rows.out, name).numpy()[0, :count],
            )

    def check_device_boundary(self, device):
        """Compare original and private native response/eight on one device."""
        source = private_call(True)
        original = device_boundary(source, device, private=False)
        candidate = device_boundary(source, device, private=True)
        ids = source.plan.dof_ids.numpy()[:, :29]

        def check():
            for name in ("resolved", "dense_count", "mf_count"):
                np.testing.assert_array_equal(
                    getattr(candidate.state, name).numpy(), getattr(original.state, name).numpy()
                )
            for arm in (original, candidate):
                np.testing.assert_array_equal(arm.state.raw_invalid.numpy(), 0)
                np.testing.assert_array_equal(arm.state.capacity_status.numpy(), 0)
                np.testing.assert_array_equal(arm.out.status.numpy(), 0)
                np.testing.assert_array_equal(arm.out.global_status.numpy(), 0)
                np.testing.assert_array_equal(arm.solve.status.numpy(), 0)
            mf = candidate.state.mf_count.numpy()
            untouched = mf == 0
            for name in ("response", "physical_J"):
                np.testing.assert_array_equal(getattr(candidate.out, name).numpy()[untouched], 12345.25)
            for world in np.flatnonzero((candidate.state.resolved.numpy() == 0) & untouched):
                count = int(candidate.state.dense_count.numpy()[world])
                self.assert_scaled_close(
                    candidate.solve.v_out.numpy()[ids[world]], original.solve.v_out.numpy()[ids[world]]
                )
                for name in ("rhs", "r0", "diag", "row_mu", "phi", "target_velocity", "impulses"):
                    self.assert_scaled_close(
                        getattr(candidate.out, name).numpy()[world, :count],
                        getattr(original.out, name).numpy()[world, :count],
                    )
                for name in ("row_type", "row_parent", "valid"):
                    np.testing.assert_array_equal(
                        getattr(candidate.out, name).numpy()[world, :count],
                        getattr(original.out, name).numpy()[world, :count],
                    )
            np.testing.assert_array_equal(candidate.out.phase_bounds.numpy(), original.out.phase_bounds.numpy())
            # Atomic reservations need not assign the same slot across arms.
            # The selected world's three duplicate contacts have identical row
            # coefficients; check each new map entry against its own reservation.
            for raw_id in range(int(candidate.raw.count.numpy()[0])):
                if int(candidate.raw.path.numpy()[raw_id]) == 0:
                    world = int(candidate.raw.world.numpy()[raw_id])
                    slot = int(candidate.raw.slot.numpy()[raw_id])
                    self.assertEqual(int(candidate.slot_raw.numpy()[world, slot]), raw_id)

        original.launch()
        candidate.launch()
        check()
        self.assertEqual(int(candidate.state.mf_count.numpy()[0]), 0)
        self.assertGreater(float(np.max(np.abs(candidate.out.impulses.numpy()[0, :10]))), 1.0e-3)
        if candidate.device.is_cuda:
            graphs = []
            for _ in range(2):
                with wp.ScopedCapture(device=candidate.device) as capture:
                    original.launch()
                    candidate.launch()
                graphs.append(capture.graph)
            for count, graph in zip((8, 1, 0, 8), graphs * 2, strict=True):
                for arm in (original, candidate):
                    arm.raw.count.assign([count])
                wp.capture_launch(graph)
                check()
        for arm in (original, candidate):
            limits = arm.max_linear.numpy()
            limits[30] = 0.01
            arm.max_linear.assign(limits)
            velocity = arm.state.v_hat.numpy()
            velocity[ids[0, 23]] = 1.0
            arm.state.v_hat.assign(velocity)
            arm.launch()
            self.assertEqual(int(arm.state.mf_count.numpy()[0]), 6)
            self.assertTrue(np.any(arm.limit_slot.numpy() >= 0))
            np.testing.assert_array_equal(arm.solve.v_out.numpy()[ids[0]], arm.state.v_hat.numpy()[ids[0]])
        check()
        count = int(candidate.state.dense_count.numpy()[0])
        self.assertFalse(np.all(candidate.out.response.numpy()[0, :count] == 12345.25))

    @unittest.skipUnless(os.environ.get("FPGS_TEST_DEVICE", "cpu").startswith("cuda"), "Explicit root-owned CUDA lease")
    def test_cuda_private_response_boundary(self):
        """Run actual CUDA prefix/triplet/eight, graphs and finalized-MF skip."""
        self.check_device_boundary(os.environ["FPGS_TEST_DEVICE"])

    @unittest.skipUnless(os.environ.get("FPGS_TEST_DEVICE", "cpu").startswith("cuda"), "Explicit root-owned CUDA lease")
    def test_cuda_two_bank_stream_lifecycle(self):
        """Run real CUDA step/fork/join/finish across MF and graph transitions."""
        for suffix in (
            "SIMPLE_WORLD_ZERO",
            "INDEPENDENT_COMPONENTS",
            "PAIRED_GENERAL_OVERLAP",
            "LOCAL_ROW_PACKETS",
            "KUKA_JOINT_WORLD",
            "WORLD_SCAN_PUBLICATION",
            "KUKA_KINETIC_WORLD",
            "KUKA_CURRENT_CONTACT",
            "KUKA_PRIVATE_RESPONSE",
        ):
            flag = "FEATHER_PGS_" + suffix
            self.assertEqual(os.environ.get(flag), "1", flag + " must be set before Python imports")
        device = wp.get_device(os.environ["FPGS_TEST_DEVICE"])
        self.assertTrue(device.is_cuda)
        add_link = newton.ModelBuilder.add_link

        def shaped_link(builder, *args, **kwargs):
            body = add_link(builder, *args, **kwargs)
            builder.add_shape_sphere(body=body, radius=0.02)
            return body

        with patch.object(newton.ModelBuilder, "add_link", shaped_link):
            model = live_model(2, device=device)

        def make_arm(private):
            with patch.dict("os.environ", {"FEATHER_PGS_KUKA_PRIVATE_RESPONSE": str(int(private))}):
                solver = SolverFeatherPGS(
                    model,
                    pgs_mode="matrix_free",
                    pgs_iterations=8,
                    dense_max_constraints=192,
                    mf_max_constraints=64,
                    enable_joint_limits=True,
                    joint_limit_activation_gap=0.0,
                    fuse_joint_velocity_limits=False,
                    mf_gs_incremental_rows=0,
                    update_mass_matrix_interval=2,
                    use_parallel_streams=True,
                )
            owner = solver._kinetic_world
            self.assertIsNotNone(owner, "Actual CUDA constructor did not admit the fixed live recipe")
            self.assertTrue(kinetic_live_owner.supported(solver))
            self.assertIsNotNone(owner.bindings.stream)
            self.assertIsNot(owner.bindings.stream, wp.get_stream(device))
            self.assertEqual(owner.bindings.private_response_slots is not None, private)
            first, second = model.state(), model.state()
            q = first.joint_q.numpy()
            starts = model.joint_q_start.numpy()
            joints = owner.host_plan.host["joint_ids"]
            for world in range(2):
                q[starts[joints[world, 1]]] = 0.3
                q[starts[joints[world, 2]]] = 1.01
            first.joint_q.assign(q)
            contacts = newton.Contacts(rigid_contact_max=256, soft_contact_max=0, device=device)
            a = np.full(256, -999, np.int32)
            b = a.copy()
            # The final raw contact alone controls world 0's positive-MF arm.
            # World 1 retains MF work during world 0's early private solve.
            a[:3], b[:3] = [13, 62, 30], [17, -1, -1]
            contacts.rigid_contact_shape0.assign(a)
            contacts.rigid_contact_shape1.assign(b)
            contacts.rigid_contact_normal.assign(np.tile([0.6, 0.0, 0.8], (256, 1)))
            contacts.rigid_contact_margin0.fill_(0.005)
            contacts.rigid_contact_margin1.zero_()
            contacts.rigid_contact_point0.zero_()
            contacts.rigid_contact_point1.zero_()
            margin = contacts.rigid_contact_margin0.numpy()
            margin[0] = 0.02
            contacts.rigid_contact_margin0.assign(margin)
            ground = contacts.rigid_contact_point1.numpy()
            ground[1] = model.body_q.numpy()[62, :3]
            contacts.rigid_contact_point1.assign(ground)
            contacts.rigid_contact_count.assign([3])
            return SimpleNamespace(
                solver=solver, owner=owner, states=(first, second), control=model.control(), contacts=contacts, calls=[]
            )

        original, candidate = make_arm(False), make_arm(True)
        arms = (original, candidate)

        def step(direction):
            for arm in arms:
                arm.solver.step(
                    arm.states[direction], arm.states[1 - direction], arm.control, arm.contacts, 1.0 / 240.0
                )
                self.assertTrue(arm.owner.last_private, "Step silently fell back from the actual live owner")
                self.assertFalse(arm.owner.bindings.pending, "Private writes remain pending after production join")

        def check(direction, count):
            for arm in arms:
                call = arm.owner.last_call if not arm.calls else arm.calls[direction]
                np.testing.assert_array_equal(call.solve.guard.frame_status.numpy(), 0)
                np.testing.assert_array_equal(call.slots.next_schedule.status.numpy(), 0)
                np.testing.assert_array_equal(call.slots.next_current.valid.numpy(), 1)
                np.testing.assert_array_equal(
                    call.slots.next_current.generation.numpy(), call.slots.next_generation.numpy()
                )
                np.testing.assert_array_equal(
                    arm.owner.canonical_generation.numpy(), call.slots.next_generation.numpy()
                )
                self.assertEqual(int(call.rows.state.mf_count.numpy()[0]) > 0, count == 3)
                self.assertGreater(int(call.rows.state.mf_count.numpy()[1]), 0)
                if count == 2:
                    self.assertEqual(int(call.rows.state.resolved.numpy()[0]), 0, "Fixture lost its loaded MF0 cohort")
                    self.assertGreater(int(call.rows.state.dense_count.numpy()[0]), 0)
            for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
                self.assert_scaled_close(
                    getattr(candidate.states[1 - direction], name).numpy(),
                    getattr(original.states[1 - direction], name).numpy(),
                )

        # Warm both directed calls and both response owners before capture.
        for direction, count in ((0, 3), (1, 2)):
            for arm in arms:
                arm.contacts.rigid_contact_count.assign([count])
            step(direction)
            check(direction, count)
        graphs = []
        for direction in (0, 1):
            with wp.ScopedCapture(device=device) as capture:
                step(direction)
            graphs.append(capture.graph)
            for arm in arms:
                arm.calls.append(arm.owner.last_call)
        for arm in arms:
            forward, backward = arm.calls
            self.assertIs(forward.slots.next_current, backward.slots.current)
            self.assertIs(backward.slots.next_current, forward.slots.current)
            self.assertIsNot(forward.slots.refresh_status, backward.slots.refresh_status)
            self.assertIs(forward.solve.stream, backward.solve.stream)
        for direction, count in ((0, 3), (1, 2), (0, 3)):
            held = [arm.owner.held.T.numpy().copy() for arm in arms]
            before = [arm.owner.clock.numpy().copy() for arm in arms]
            for arm in arms:
                arm.contacts.rigid_contact_count.assign([count])
            wp.capture_launch(graphs[direction])
            check(direction, count)
            for arm, old_clock, old_held in zip(arms, before, held, strict=True):
                self.assertTrue(np.all(arm.owner.clock.numpy() > old_clock))
                if direction == 1:
                    np.testing.assert_array_equal(arm.owner.held.T.numpy(), old_held)
        # A malformed next held call must join both branches but publish no
        # generalized or body output, and must withdraw the destination bank.
        snapshots = [
            {name: getattr(arm.states[0], name).numpy().copy() for name in ("joint_q", "joint_qd", "body_q", "body_qd")}
            for arm in arms
        ]
        for arm in arms:
            arm.contacts.rigid_contact_count.assign([257])
        wp.capture_launch(graphs[1])
        for arm, before in zip(arms, snapshots, strict=True):
            call = arm.calls[1]
            self.assertNotEqual(int(call.solve.guard.frame_status.numpy()[0]), 0)
            np.testing.assert_array_equal(call.slots.next_current.valid.numpy(), 0)
            np.testing.assert_array_equal(arm.owner.canonical_generation.numpy(), -1)
            for name, expected in before.items():
                np.testing.assert_array_equal(getattr(arm.states[0], name).numpy(), expected)


if __name__ == "__main__":
    unittest.main()
