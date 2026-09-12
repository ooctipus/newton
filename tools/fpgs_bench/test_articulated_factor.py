# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent numerical controls for the experimental held articulated factor."""

import importlib
import unittest

import articulated_factor_check as check
import numpy as np
import warp as wp


def fixture():
    """Build two branched fixed-base trees with welds and nonuniform R plus drive K."""
    rng = np.random.default_rng(91)
    parents = np.tile(np.array([-1, 0, 0, 1, 3, 2], np.int32), (2, 1))
    slots = np.tile(np.array([-1, 0, 1, -1, 2, 3], np.int32), (2, 1))
    raw = rng.normal(size=(2, 6, 6, 6))
    inertia = raw @ raw.swapaxes(-1, -2) + np.eye(6) * 0.2
    motion = rng.normal(size=(2, 6, 6))
    motion[slots < 0] = 0
    return {
        "parents": parents,
        "slots": slots,
        "counts": np.full(2, 6, np.int32),
        "inertia": inertia.astype(np.float32),
        "motion": motion.astype(np.float32),
        "diagonal": np.array([[0.01, 0.15, 0.2, 0.05], [0.1, 0.3, 0.07, 0.09]], np.float32),
        "refresh": np.ones(2, np.int32),
    }


def mass(data):
    """Form an independent current primitive mass matrix in generalized coordinates."""
    return check.dense_mass(
        data["parents"], data["slots"], data["inertia"], data["motion"], data["diagonal"], data["counts"]
    )


class TestArticulatedFactor(unittest.TestCase):
    def test_module_exists(self):
        """Require the new factor owner rather than silently using the old factor."""
        module = importlib.import_module("newton._src.solvers.feather_pgs.articulated_factor")
        self.assertIsNotNone(module)

    def test_branched_welds_basis_mimic_and_coupled_row_actions(self):
        """Match FP64 mass actions for full basis, mimic, normal and friction RHS."""
        module = importlib.import_module("newton._src.solvers.feather_pgs.articulated_factor")
        data = fixture()
        rhs = (
            np.broadcast_to(
                np.vstack((np.eye(4), [1, -0.5, 0, 0], [0.4, -0.2, 0.1, -0.3], [0.1, 0.2, -0.6, 0.7])), (2, 7, 4)
            )
            .astype(np.float32)
            .copy()
        )
        factor = check.run_factor(module, data, "cpu")
        np.testing.assert_array_equal(factor.valid.numpy(), [1, 1])
        actual = check.run_action(module, factor, data, rhs, "cpu")
        check.action_error(mass(data), rhs, actual)
        self.assertGreater(np.linalg.norm(actual[:, 4, 1:]), 0)

    def test_held_epoch_with_current_rhs_and_mixed_refresh(self):
        """Keep held S and mass on reuse while applying new forces and contact rows."""
        module = importlib.import_module("newton._src.solvers.feather_pgs.articulated_factor")
        data = fixture()
        factor = check.run_factor(module, data, "cpu")
        held = mass(data)
        old_s, old_u, old_d = factor.S.numpy(), factor.U.numpy(), factor.invD.numpy()
        data["inertia"] *= 1.7
        data["motion"] *= 0.8
        data["diagonal"] += 0.4
        data["refresh"][:] = 0
        check.run_factor(module, data, "cpu", factor)
        np.testing.assert_array_equal(factor.S.numpy(), old_s)
        np.testing.assert_array_equal(factor.U.numpy(), old_u)
        np.testing.assert_array_equal(factor.invD.numpy(), old_d)
        rhs = np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 7 - 1
        actual = check.run_action(module, factor, data, rhs, "cpu")
        check.action_error(held, rhs, actual)
        with self.assertRaises(AssertionError):
            check.action_error(mass(data), rhs, actual)
        data["refresh"][0] = 1
        check.run_factor(module, data, "cpu", factor)
        mixed = mass(data)
        mixed[1] = held[1]
        check.action_error(mixed, rhs, check.run_action(module, factor, data, rhs, "cpu"))

    def test_row_counts_leave_unowned_output_untouched(self):
        """Respect current per-articulation RHS counts without writing inactive tails."""
        module = importlib.import_module("newton._src.solvers.feather_pgs.articulated_factor")
        data = fixture()
        factor = check.run_factor(module, data, "cpu")
        rhs = np.ones((2, 4, 4), np.float32)
        actual = check.run_action(module, factor, data, rhs, "cpu", np.array([0, 2], np.int32))
        np.testing.assert_array_equal(actual[0], 12345)
        np.testing.assert_array_equal(actual[1, 2:], 12345)
        check.action_error(mass(data)[1:], rhs[1:, :2], actual[1:, :2])

    def test_nonfinite_and_wrong_mass_negative_controls(self):
        """Reject poisoned action output and a wrong operator despite finite data."""
        h = mass(fixture())
        rhs = np.ones((2, 2, 4))
        exact = np.linalg.solve(h, rhs.swapaxes(1, 2)).swapaxes(1, 2)
        check.action_error(h, rhs, exact)
        for candidate in (exact * 1.1, np.full_like(exact, np.nan)):
            with self.assertRaises(AssertionError):
                check.action_error(h, rhs, candidate)

    def test_native_invalid_pivot_is_not_silently_clamped(self):
        """Reject a singular native factor and publish nonfinite owned actions."""
        module = importlib.import_module("newton._src.solvers.feather_pgs.articulated_factor")
        data = fixture()
        data["inertia"][:] = 0
        data["diagonal"][:] = 0
        factor = check.run_factor(module, data, "cpu")
        np.testing.assert_array_equal(factor.valid.numpy(), 0)
        actual = check.run_action(module, factor, data, np.ones((2, 1, 4), np.float32), "cpu")
        self.assertFalse(np.isfinite(actual).all())

    def test_saved_actual_kuka_refresh_reuse_epoch(self):
        """Check four actual Kuka hands against the original held post3 factor."""
        module = importlib.import_module("newton._src.solvers.feather_pgs.articulated_factor")
        root = "/tmp/fpgs-kuka-live-512-20260911-02/gpu0/"
        pins = (
            "5e79694fee997003ef299b33eed223d861244200bd57ac9ee6ef69c8d3b74c34",
            "85084c49880a217e3ab52cbc3340bd5f174349fab8baa77cf0e099e1a9c6d815",
        )
        factor = None
        for index, pin in enumerate(pins):
            data = check.load_kuka(root + f"kuka_step{1600 + index}_capture0{index}.npz", pin, limit=4)
            self.assertEqual(data["dt"], 1 / 240)
            self.assertGreater(data["drive_nonzero"], 0)
            np.testing.assert_array_equal(data["refresh"], 1 - index)
            factor = check.run_factor(module, data, "cpu", factor)
            actual = check.run_action(module, factor, data, data["rhs"], "cpu")
            check.action_error(data["reference_h"], data["rhs"], actual)

    def test_actual_live_compact_and_full_sources_ignore_poisoned_opposite(self):
        """Read only the selected current inertia source and retain it on reuse."""
        module = importlib.import_module("newton._src.solvers.feather_pgs.articulated_factor")
        data = check.load_kuka(
            "/tmp/fpgs-kuka-live-512-20260911-02/gpu0/kuka_step1600_capture00.npz",
            "5e79694fee997003ef299b33eed223d861244200bd57ac9ee6ef69c8d3b74c34",
            limit=4,
        )
        for compact in (False, True):
            with self.subTest(compact=compact):
                data["refresh"][:] = 1
                factor = check.run_live_factor(module, data, "cpu", compact_source=compact, poison_unused=True)
                check.action_error(
                    data["reference_h"], data["rhs"], check.run_action(module, factor, data, data["rhs"], "cpu")
                )
                retained = (factor.S.numpy(), factor.U.numpy(), factor.invD.numpy())
                held = dict(data)
                held["refresh"] = np.zeros(4, np.int32)
                held["inertia"] = np.full_like(data["inertia"], np.nan)
                held["compact_terms"] = np.full_like(data["compact_terms"], np.nan)
                held["motion"] = np.full_like(data["motion"], np.nan)
                check.run_live_factor(module, held, "cpu", factor, compact_source=compact)
                for before, after in zip(
                    retained, (factor.S.numpy(), factor.U.numpy(), factor.invD.numpy()), strict=True
                ):
                    np.testing.assert_array_equal(before, after)

    def test_current_response_publication_permuted_groups_worlds_offsets(self):
        """Publish current J/Y only in owned rows/coordinates under permuted maps."""
        module = importlib.import_module("newton._src.solvers.feather_pgs.articulated_factor")
        response = importlib.import_module("newton._src.solvers.feather_pgs.articulated_response")
        data = fixture()
        factor = check.run_factor(module, data, "cpu")
        rows = np.arange(40, dtype=np.float32).reshape(2, 5, 4) / 13 - 1
        current_j = wp.array(rows, dtype=float, device="cpu")
        groups = wp.array([3, 1], dtype=int, device="cpu")
        worlds = wp.array([1, 0, 1, 2], dtype=int, device="cpu")
        offsets = wp.array([0, 2, 0, 7], dtype=int, device="cpu")
        counts = wp.array([2, 0, 4], dtype=int, device="cpu")
        for publish in (True, False):
            with self.subTest(publish=publish):
                y = wp.full((2, 5, 4), 12345.0, device="cpu")
                world_j = wp.full((3, 5, 13), 12345.0, device="cpu")
                world_y = wp.full((3, 5, 13), 12345.0, device="cpu")
                wp.launch_tiled(
                    response.get_group_response_kernel(6, 4, write_world=publish),
                    dim=[2],
                    inputs=[factor, current_j, groups, worlds, offsets, counts, y, world_j, world_y],
                    block_dim=32,
                    device="cpu",
                )
                expected_j = np.full((3, 5, 13), 12345.0, np.float32)
                expected_y = expected_j.copy()
                for group, world, offset, count in ((0, 2, 7, 4), (1, 0, 2, 2)):
                    actual = y.numpy()[group : group + 1, :count]
                    check.action_error(mass(data)[group : group + 1], rows[group : group + 1, :count], actual)
                    np.testing.assert_array_equal(y.numpy()[group, count:], 12345)
                    if publish:
                        expected_j[world, :count, offset : offset + 4] = rows[group, :count]
                        expected_y[world, :count, offset : offset + 4] = actual[0]
                np.testing.assert_array_equal(world_j.numpy(), expected_j)
                np.testing.assert_array_equal(world_y.numpy(), expected_y)
                np.testing.assert_array_equal(current_j.numpy(), rows)


if __name__ == "__main__":
    unittest.main()
