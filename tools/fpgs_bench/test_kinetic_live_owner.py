# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: PLC0415

"""Exercise device lifecycle decisions without captured state or CUDA work."""

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import warp as wp


class TestKineticLiveOwner(unittest.TestCase):
    """Check state generations, repair, and held-only canonical conversion."""

    def test_owner_api_exists(self):
        """Require the actual opt-in owner before testing its lifecycle."""
        from newton._src.solvers.feather_pgs import kinetic_live_owner as owner

        self.assertTrue(callable(owner.create_owner))

    def test_device_schedule_refresh_reuse_and_bank_repair(self):
        """Advance real state generations and preserve held reuse epochs."""
        from newton._src.solvers.feather_pgs import kinetic_live_owner as owner

        device = "cpu"
        worlds = 3

        def ints(values, dtype=int):
            return wp.array(values, dtype=dtype, device=device)

        clock = ints([5, 5, 5], wp.int64)
        incoming = ints([5, 5, 5], wp.int64)
        outgoing = ints([-1, -1, -1], wp.int64)
        current_generation = ints([5, 5, 5], wp.int64)
        current_valid = ints([1, 0, 1])
        geometry_generation = ints([4, 5, 4], wp.int64)
        geometry_valid = ints([1, 1, 1])
        held_generation = ints([4, 4, 4], wp.int64)
        canonical = ints([5, 5, 4], wp.int64)
        requested = ints([0, 0, 0])
        repair = ints([0, 0, 0])
        expected_held = ints([-1, -1, -1], wp.int64)
        next_geometry = ints([0, 0, 0])
        construct_status = ints([99, 99, 99])
        finish_status = ints([99, 99, 99])
        mass_request = ints([0])
        mass_mask = ints([99] * 9)
        args = [
            clock,
            incoming,
            outgoing,
            current_generation,
            current_valid,
            geometry_generation,
            geometry_valid,
            held_generation,
            canonical,
            mass_request,
            mass_mask,
            0,
            1,
            requested,
            repair,
            expected_held,
            next_geometry,
            construct_status,
            finish_status,
        ]
        wp.launch(owner._prepare_epochs, worlds, inputs=args, device=device)
        np.testing.assert_array_equal(outgoing.numpy(), [6, 7, 6])
        np.testing.assert_array_equal(requested.numpy(), 0)
        np.testing.assert_array_equal(mass_mask.numpy(), 0)
        np.testing.assert_array_equal(repair.numpy(), [0, 1, 1])
        np.testing.assert_array_equal(expected_held.numpy(), 4)
        np.testing.assert_array_equal(next_geometry.numpy(), 1)
        np.testing.assert_array_equal(held_generation.numpy(), 4)
        current_valid.fill_(1)
        wp.copy(current_generation, incoming)
        mass_request.fill_(1)
        wp.launch(owner._prepare_epochs, worlds, inputs=args, device=device)
        np.testing.assert_array_equal(requested.numpy(), 1)
        np.testing.assert_array_equal(mass_mask.numpy(), 1)
        np.testing.assert_array_equal(expected_held.numpy(), [5, 6, 5])
        np.testing.assert_array_equal(repair.numpy(), 1)
        np.testing.assert_array_equal(held_generation.numpy(), 4)

    def test_subset_reset_does_not_refresh_held(self):
        """Invalidate only authored worlds without changing their held mass."""
        from newton._src.solvers.feather_pgs import kinetic_live_owner as owner

        current = wp.array([1, 1, 1], dtype=int, device="cpu")
        geometric = wp.array([1, 1, 1], dtype=int, device="cpu")
        canonical = wp.array([6, 6, 6], dtype=wp.int64, device="cpu")
        mask = wp.array([False, True, False], dtype=wp.bool, device="cpu")
        wp.launch(owner._invalidate_current, 3, inputs=[mask, 0, current, geometric, canonical], device="cpu")
        np.testing.assert_array_equal(current.numpy(), [1, 0, 1])
        np.testing.assert_array_equal(geometric.numpy(), [1, 0, 1])
        np.testing.assert_array_equal(canonical.numpy(), [6, -1, 6])

    def test_demotion_uses_held_augmented_and_group_map(self):
        """Recover canonical factors from held blocks without current geometry."""
        from newton._src.solvers.feather_pgs import kinetic_live_owner as owner

        rng = np.random.default_rng(42)
        dense = np.zeros((2, 23, 23), dtype=np.float32)
        packed = np.zeros((2, 180), dtype=np.float32)
        for world in range(2):
            dense[world] = np.eye(23, dtype=np.float32) * 5.0
            for finger in range(4):
                cross = rng.normal(0.0, 0.05, (7, 4)).astype(np.float32)
                dense[world, :7, 7 + finger * 4 : 11 + finger * 4] = cross
                dense[world, 7 + finger * 4 : 11 + finger * 4, :7] = cross.T
                for row in range(4):
                    for col in range(row + 1):
                        packed[world, finger * 10 + row * (row + 1) // 2 + col] = dense[
                            world, 7 + finger * 4 + row, 7 + finger * 4 + col
                        ]
                packed[world, 68 + finger * 28 : 96 + finger * 28] = cross.ravel()
            for row in range(7):
                for col in range(row + 1):
                    packed[world, 40 + row * (row + 1) // 2 + col] = dense[world, row, col]
        arrays = [
            wp.array(packed, dtype=float, device="cpu"),
            wp.array([8, 10], dtype=wp.int64, device="cpu"),
            wp.array([1, 1], dtype=int, device="cpu"),
            wp.array([1, 0], dtype=int, device="cpu"),
            wp.zeros((2, 23, 23), dtype=float, device="cpu"),
            wp.zeros((2, 23, 23), dtype=float, device="cpu"),
            wp.array([-1, -1], dtype=wp.int64, device="cpu"),
            wp.zeros(2, dtype=int, device="cpu"),
        ]
        wp.launch_tiled(owner.get_demotion_kernel(), dim=[2], block_dim=32, inputs=arrays, device="cpu")
        lower, inverse = arrays[4].numpy(), arrays[5].numpy()
        for world, group in enumerate([1, 0]):
            np.testing.assert_allclose(lower[group] @ lower[group].T, dense[world], atol=2e-6, rtol=2e-6)
            np.testing.assert_allclose(inverse[group] @ lower[group], np.eye(23), atol=2e-6, rtol=2e-6)
        np.testing.assert_array_equal(arrays[6].numpy(), [8, 10])
        np.testing.assert_array_equal(arrays[7].numpy(), 0)
        arrays[0].fill_(np.nan)
        wp.launch_tiled(owner.get_demotion_kernel(), dim=[2], block_dim=32, inputs=arrays, device="cpu")
        np.testing.assert_array_equal(arrays[4].numpy(), lower)
        np.testing.assert_array_equal(arrays[5].numpy(), inverse)
        np.testing.assert_array_equal(arrays[7].numpy(), 0)
        arrays[1].assign(np.array([9, 11], dtype=np.int64))
        wp.launch_tiled(owner.get_demotion_kernel(), dim=[2], block_dim=32, inputs=arrays, device="cpu")
        self.assertTrue(np.all(arrays[7].numpy() != 0))
        np.testing.assert_array_equal(arrays[6].numpy(), [8, 10])

    def test_construct_failure_survives_reuse_and_blocks_bank_tag(self):
        """Keep repair errors visible after a no-refresh return and publication."""
        from newton._src.solvers.feather_pgs import kinetic_live_owner as owner

        status = wp.array([2, 0], dtype=int, device="cpu")
        refreshed = wp.zeros(2, dtype=int, device="cpu")
        generation = wp.array([7, 7], dtype=wp.int64, device="cpu")
        valid = wp.array([0, 1], dtype=int, device="cpu")
        bank = wp.array([6, 6], dtype=wp.int64, device="cpu")
        wp.launch(owner._merge_construct_status, 2, inputs=[status, refreshed], device="cpu")
        wp.launch(owner._publish_bank_generation, 2, inputs=[generation, valid, status, bank], device="cpu")
        np.testing.assert_array_equal(refreshed.numpy(), [256, 0])
        np.testing.assert_array_equal(bank.numpy(), [-1, 7])

    def test_repair_entry_skips_unreadable_descriptors(self):
        """Skip cold construction before any plan or numeric descriptor read."""
        from newton._src.solvers.feather_pgs import kinetic_live_owner as owner
        from newton._src.solvers.feather_pgs import kinetic_state, kinetic_types

        kernel = owner.get_repair_kernel("None")
        self.assertEqual(kernel.repair_original_ast, kernel.repair_recovered_ast)
        wp.launch_tiled(
            kernel,
            dim=[3],
            block_dim=32,
            inputs=[
                kinetic_types.KineticPlan(),
                kinetic_state.PublicationData(),
                kinetic_types.KineticSchedule(),
                kinetic_types.CurrentKineticCache(),
                kinetic_types.GeometricCache(),
                wp.zeros(3, dtype=int, device="cpu"),
            ],
            device="cpu",
        )

    def test_public_force_uses_actual_api_cadence(self):
        """Export only the last solved Contacts through the public API hook."""
        from newton._src.solvers.feather_pgs import kinetic_live_owner as module

        owner = object.__new__(module.KineticWorldOwner)
        contacts = object()
        events = []
        owner.last_private = False
        owner.last_call = SimpleNamespace(
            contacts=contacts,
            solve=SimpleNamespace(join=lambda: events.append("join")),
            force=SimpleNamespace(launch=lambda: events.append("force")),
        )
        self.assertFalse(owner.update_contacts(contacts))
        self.assertEqual(events, [])
        owner.last_private = True
        self.assertTrue(owner.update_contacts(contacts))
        self.assertEqual(events, ["join", "force"])
        with self.assertRaises(ValueError):
            owner.update_contacts(object())

    def test_cold_unsupported_does_not_demote_or_write(self):
        """Leave before-ever-admitted unsupported calls on the original path."""
        from newton._src.solvers.feather_pgs import kinetic_live_owner as module

        owner = object.__new__(module.KineticWorldOwner)
        owner.solver = object()
        owner.last_private = False
        with (
            mock.patch.object(module, "supported", return_value=False),
            mock.patch.object(owner, "_demote", side_effect=AssertionError("unexpected demotion")),
        ):
            self.assertFalse(owner.try_step(None, None, None, None, None, 0.01))

    def test_new_unsupported_capture_rejects_before_conversion(self):
        """Require recapture before demoting a newly unsupported graph call."""
        from newton._src.solvers.feather_pgs import kinetic_live_owner as module

        owner = object.__new__(module.KineticWorldOwner)
        owner.device = SimpleNamespace(is_capturing=True)
        owner.last_call = None
        with mock.patch.object(wp, "launch_tiled", side_effect=AssertionError("unexpected write")):
            with self.assertRaisesRegex(RuntimeError, "recapture"):
                owner._demote()

    def test_solver_hook_precedes_old_producers(self):
        """Keep the complete owner ahead of old begin and preserve force cadence."""
        import inspect

        from newton._src.solvers.feather_pgs.solver_feather_pgs import SolverFeatherPGS

        source = inspect.getsource(SolverFeatherPGS.step)
        self.assertLess(source.index("self._kinetic_world.try_step"), source.index("self._joint_world.begin"))
        self.assertLess(source.index("self._kinetic_world.try_step"), source.index("self._stage1_fk_id"))
        self.assertNotIn("self._kinetic_world.update_contacts", source)

    def test_admission_excludes_postpredict_radial_clamp(self):
        """Keep endpoint twists consistent with the actual predictor velocity."""
        from newton._src.solvers.feather_pgs import kinetic_live_owner as owner
        from newton._src.solvers.feather_pgs import simple_world, solver_feather_pgs

        solver = SimpleNamespace(
            model=SimpleNamespace(particle_count=0),
            _fk_id_cache_enabled=True,
            update_mass_matrix_interval=2,
            _parallel_augmented_drive_topology=True,
            _has_root_free=True,
            _has_rigid_body_velocity_limits=True,
            rigid_velocity_limit_slot=None,
        )
        with (
            mock.patch.object(simple_world, "_supported", return_value=True),
            mock.patch.object(solver_feather_pgs, "_FPGS_CAPTURE", False),
        ):
            self.assertFalse(owner.supported(solver))
            solver.rigid_velocity_limit_slot = object()
            self.assertTrue(owner.supported(solver))


if __name__ == "__main__":
    unittest.main()
