# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check exact dense-row support tiles without sparsity thresholds."""

import inspect
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs.mf_dense_support import DenseRowSupport
from newton._src.solvers.feather_pgs.solver_feather_pgs import _get_pgs_solve_mf_gs_kernel
from newton.tests.test_feather_pgs_independent_components import solve_fixture


class TestFeatherPGSMFDenseSupport(unittest.TestCase):
    """Verify support preparation on CPU without launching GPU work."""

    def test_exact_tiles_include_j_y_remote_dofs_and_last_row(self):
        """Include both operators across mask words while ignoring padding and inactive rows."""
        support = DenseRowSupport(2, 4, 2053, "cpu", stage_velocities=True)
        jacobian = np.zeros((2, 4, 2080), np.float32)
        response = np.zeros_like(jacobian)
        jacobian[0, 0, [0, 31, 1023, 2052]] = 1.0
        response[0, 0, [32, 1024]] = 2.0
        jacobian[0, 1, 63] = -0.5
        response[0, 2, 1601] = 0.25
        jacobian[0, 3, 64] = 1.0
        response[0, 3, 96] = -1.0
        response[1, 0, 1022] = 1.0
        jacobian[1, 0, 2053:] = 9.0
        jacobian[1, 1, 2048] = 1.0
        jacobian[1, 2:] = np.nan
        response[1, 2:] = 1.0
        support.prepare(
            wp.array([4, 2], dtype=int, device="cpu"),
            wp.array(jacobian, dtype=float, device="cpu"),
            wp.array(response, dtype=float, device="cpu"),
        )
        expected = np.zeros((2, 4, 3), np.uint32)
        expected[0, 0] = [(1 << 0) | (1 << 1) | (1 << 31), 1, 1]
        expected[0, 1, 0] = 1 << 1
        expected[0, 2, 1] = 1 << 18
        expected[0, 3, 0] = (1 << 2) | (1 << 3)
        expected[1, 0, 0] = 1 << 31
        expected[1, 1, 2] = 1
        self.assertEqual(support.masks.dtype, wp.uint32)
        np.testing.assert_array_equal(support.masks.numpy(), expected)
        np.testing.assert_array_equal(support.world_masks.numpy(), np.bitwise_or.reduce(expected, axis=1))
        overflow_counts = wp.array([7, 2], dtype=int, device="cpu")
        support.prepare(
            overflow_counts,
            wp.array(jacobian, dtype=float, device="cpu"),
            wp.array(response, dtype=float, device="cpu"),
        )
        np.testing.assert_array_equal(support.masks.numpy(), expected)
        np.testing.assert_array_equal(support.world_masks.numpy(), np.bitwise_or.reduce(expected, axis=1))
        np.testing.assert_array_equal(overflow_counts.numpy(), [7, 2])

    def test_nonfinite_and_tiny_values_are_conservative(self):
        """Retain NaNs and arbitrarily small nonzero entries but not signed zero."""
        support = DenseRowSupport(1, 3, 64, "cpu")
        jacobian = np.zeros((1, 3, 64), np.float32)
        response = np.zeros_like(jacobian)
        jacobian[0, 0, 0] = 1.0e-30
        response[0, 0, 32] = -1.0e-30
        jacobian[0, 1, 1] = np.nan
        response[0, 1, 63] = np.inf
        jacobian[0, 2] = -0.0
        support.prepare(
            wp.array([3], dtype=int, device="cpu"),
            wp.array(jacobian, dtype=float, device="cpu"),
            wp.array(response, dtype=float, device="cpu"),
        )
        np.testing.assert_array_equal(support.masks.numpy(), np.array([[[3], [3], [0]]], np.uint32))

    def test_reset_clears_inactive_rows_and_previous_world_support(self):
        """Rebuild masks exactly when counts and operator contents change."""
        support = DenseRowSupport(2, 2, 65, "cpu", stage_velocities=True)
        counts = wp.array([2, 2], dtype=int, device="cpu")
        jacobian = wp.ones((2, 2, 65), dtype=float, device="cpu")
        response = wp.zeros_like(jacobian)
        support.prepare(counts, jacobian, response)
        np.testing.assert_array_equal(support.masks.numpy(), np.full((2, 2, 1), 7, np.uint32))
        np.testing.assert_array_equal(support.world_masks.numpy(), [[7], [7]])
        counts.assign(np.array([0, 1], np.int32))
        jacobian.zero_()
        changed = np.zeros((2, 2, 65), np.float32)
        changed[0] = 4.0
        changed[1, 0, 64] = 2.0
        changed[1, 1] = np.nan
        response.assign(changed)
        support.prepare(counts, jacobian, response)
        expected = np.zeros((2, 2, 1), np.uint32)
        expected[1, 0, 0] = 4
        np.testing.assert_array_equal(support.masks.numpy(), expected)
        np.testing.assert_array_equal(support.world_masks.numpy(), [[0], [4]])
        counts.zero_()
        support.prepare(counts, jacobian, response)
        np.testing.assert_array_equal(support.masks.numpy(), 0)
        np.testing.assert_array_equal(support.world_masks.numpy(), 0)

    def test_union_preparation_is_explicitly_opt_in(self):
        """Do no hidden union work when staging is disabled, including after toggling it."""
        support = DenseRowSupport(1, 2, 65, "cpu")
        counts = wp.array([2], dtype=int, device="cpu")
        jacobian = wp.zeros((1, 2, 65), dtype=float, device="cpu")
        response = wp.ones_like(jacobian)
        sentinel = np.array([[0xA5A5A5A5]], np.uint32)
        support.world_masks.assign(sentinel)
        support.prepare(counts, jacobian, response)
        np.testing.assert_array_equal(support.masks.numpy(), [[[7], [7]]])
        np.testing.assert_array_equal(support.world_masks.numpy(), sentinel)
        support.stage_velocities = True
        support.prepare(counts, jacobian, response)
        np.testing.assert_array_equal(support.world_masks.numpy(), [[7]])
        support.stage_velocities = False
        counts.zero_()
        support.prepare(counts, jacobian, response)
        np.testing.assert_array_equal(support.masks.numpy(), 0)
        np.testing.assert_array_equal(support.world_masks.numpy(), [[7]])
        support.stage_velocities = True
        support.prepare(counts, jacobian, response)
        np.testing.assert_array_equal(support.world_masks.numpy(), 0)

    def test_opt_in_factory_preserves_legacy_argument_order(self):
        """Add only trailing masks without modifying the default kernel ABI."""
        options = {"has_drive_rows": True, "has_dense_velocity_limit_rows": True}
        legacy = _get_pgs_solve_mf_gs_kernel(8, 6, 2053, "cpu", **options)
        explicit = _get_pgs_solve_mf_gs_kernel(8, 6, 2053, "cpu", dense_support_tiles=False, **options)
        candidate = _get_pgs_solve_mf_gs_kernel(8, 6, 2053, "cpu", dense_support_tiles=True, **options)
        old_args = list(inspect.signature(legacy.func).parameters)
        new_args = list(inspect.signature(candidate.func).parameters)
        self.assertEqual(old_args[-1], "v_out")
        self.assertEqual(new_args, [*old_args, "dense_support_masks", "dense_support_world_masks"])
        self.assertEqual(legacy.key, explicit.key)
        self.assertNotEqual(legacy.key, candidate.key)
        option = inspect.signature(_get_pgs_solve_mf_gs_kernel).parameters["dense_support_tiles"]
        self.assertEqual(option.kind, inspect.Parameter.KEYWORD_ONLY)
        staged = _get_pgs_solve_mf_gs_kernel(
            8, 6, 2053, "cpu", dense_support_tiles=True, stage_velocities=True, **options
        )
        self.assertEqual(list(inspect.signature(staged.func).parameters), new_args)
        self.assertNotEqual(staged.key, candidate.key)
        with self.assertRaises(ValueError):
            _get_pgs_solve_mf_gs_kernel(8, 6, 2053, "cpu", stage_velocities=True, **options)

    def test_paired_dense_impulses_have_one_owner(self):
        """Keep the paired tangent law and fenced publication in all dense projectors."""
        snippets = []
        projectors = []
        native = wp.func_native

        def record(snippet, *args, **kwargs):
            snippets.append(snippet)
            return native(snippet, *args, **kwargs)

        _get_pgs_solve_mf_gs_kernel.cache_clear()
        with patch.object(wp, "func_native", record):
            for tiled, staged, friction in (
                (False, False, "current"),
                (True, False, "current"),
                (True, True, "current"),
                (False, False, "coulomb_newton"),
            ):
                _get_pgs_solve_mf_gs_kernel(
                    256,
                    15984,
                    3135,
                    "cpu",
                    friction_mode=friction,
                    has_drive_rows=False,
                    has_dense_velocity_limit_rows=False,
                    stream_mf_impulses=friction == "current",
                    dense_support_tiles=tiled,
                    stage_velocities=staged,
                )
                projectors.append(snippets[-1])
        self.assertEqual(len(projectors), 4)
        for snippet in projectors:
            self.assertIn("__shfl_sync(MASK, lane == 0 ? s_lam_dense[i] : 0.0f, 0)", snippet)
            self.assertIn("if (lane == 0) s_lam_dense[i] = 0.0f;", snippet)
            self.assertIn("if (lane == 0) s_lam_dense[i] = new_impulse;", snippet)
            start = snippet.index("const float sibling_impulse = s_lam_dense[sib];")
            end = snippet.index("if (sib_delta != 0.0f) iteration_changed = 1;", start)
            owner = snippet[start:end]
            self.assertIn("friction_pair_candidate(denom, cross, s_diag_dense[sib]", owner)
            self.assertIn("float sib_delta = b * scale - sibling_impulse;", owner)
            self.assertIn(
                "__syncwarp();\n                    if (lane == 0) s_lam_dense[sib] = b * scale;\n                    __syncwarp();",
                owner,
            )
        for snippet in projectors[1:3]:
            start = snippet.index("float sibling_residual = 0.0f;")
            end = snippet.index("float2 pair = friction_pair_candidate(denom", start)
            self.assertIn("dense_support_masks.data[(world * 256 + (sib))", snippet[start:end])
            self.assertNotIn("for (int d = lane; d < 3135; d += 32)", snippet[start:end])


def sparse_solve_fixture(device, *, drives=True):
    """Extend the established mixed-row fixture across three exact support-mask words."""
    _, values = solve_fixture(device)
    worlds, rows, dofs = 5, 8, 2053

    def bind(value):
        value = np.asarray(value)
        return wp.array(value, dtype=int if value.dtype == np.int32 else float, device=device)

    jacobian = np.zeros((worlds, rows, dofs), np.float32)
    response = np.zeros_like(jacobian)
    jacobian[:, 0, 0] = response[:, 0, 0] = 1.0  # Excluded final velocity-limit row.
    jacobian[:, 1, 32] = response[:, 1, 32] = 1.0
    jacobian[:, 2, 3] = 1.0
    response[:, 2, 3] = 0.2
    response[:, 2, 1023] = 0.7  # Sibling response in bit 31, not the current row's mask.
    jacobian[:, 3, 64] = 1.0
    response[:, 3, 64] = 0.2
    response[:, 3, 2052] = -0.6  # A sibling response in the final partial tile.
    jacobian[:, 4, 1024] = response[:, 4, 1024] = 1.0
    response[:, 6, 2048] = 0.4  # Y-only support must still update velocity.
    jacobian[:, 7, 2052] = -0.3  # J-only support must still contribute to residual.
    mapping = np.arange(worlds * dofs, dtype=np.int32).reshape(worlds, dofs)[::-1].copy()
    velocity = np.linspace(-0.125, 0.25, worlds * dofs, dtype=np.float32)
    velocity[mapping[:, 3]] = 2.0
    velocity[mapping[:, 64]] = -1.5
    velocity[mapping[:, 2052]] = 0.7
    kinds = np.tile(np.array([4, 0, 2, 2, 1 if drives else 5, 0, 3, 3], np.int32), (worlds, 1))
    parents = np.tile(np.array([-1, -1, 1, 1, -1, -1, -1, -1], np.int32), (worlds, 1))
    replacements = {
        "J_world": jacobian,
        "Y_world": response,
        "world_dof_indices": mapping,
        "world_deferred_dof_mask": np.zeros((worlds, dofs), np.int32),
        "v_out": velocity,
        "world_constraint_count": np.array([8, 8, 8, 8, 0], np.int32),
        "dense_phase_bounds": np.array([[0, 1]] * 4 + [[0, 0]], np.int32),
        "world_diag": np.ones((worlds, rows), np.float32),
        "world_row_type": kinds,
        "world_row_parent": parents,
        "rhs_bias": np.tile(np.array([-10.0, -0.8, 0.0, 0.0, -0.2, -0.5, -0.3, -0.2], np.float32), (worlds, 1)),
        "world_row_mu": np.full((worlds, rows), 0.4, np.float32),
        "world_row_w": np.full((worlds, rows), 0.8, np.float32),
    }
    for name, value in replacements.items():
        values[name] = bind(value)
    for field, scalar in (
        ("target_vel_bias", 0.4),
        ("vel_multiplier", -0.2),
        ("impulse_multiplier", 0.8),
        ("max_impulse", 0.6),
        ("vel_limit", 0.7),
    ):
        values[f"world_drive_{field}"] = bind(np.full((worlds, rows), scalar, np.float32))
    values.update(
        general_world_grid_stride=worlds,
        use_general_world_queue=0,
        iterations=8,
        omega=1.0,
        regularize=0,
        row_phase=6,
        friction_start_iteration=0,
        iteration_offset=0,
        freeze_drive_rows=0,
        defer_dense_response=0,
    )
    support = DenseRowSupport(worlds, rows, dofs, device, stage_velocities=True)
    values["dense_support_masks"] = support.masks
    values["dense_support_world_masks"] = support.world_masks
    initial = {name: values[name].numpy().copy() for name in ("v_out", "world_impulses", "mf_impulses")}
    return values, support, initial


def coins_solve_fixture(device):
    """Use Coins' capacity/dispatch contract with larger mixed dense contact sets."""
    values, _, _ = sparse_solve_fixture(device, drives=False)
    worlds, rows, dofs, mf_rows = 5, 256, 3135, 15984
    counts = np.array([95, 47, 128, 256, 0], np.int32)
    jacobian = np.zeros((worlds, rows, dofs), np.float32)
    response = np.zeros_like(jacobian)
    kinds = np.full((worlds, rows), 5, np.int32)
    parents = np.full((worlds, rows), -1, np.int32)
    rhs = np.full((worlds, rows), -0.025, np.float32)
    for world, count in enumerate(counts):
        for row in range(count):
            dof = (97 * row + 53 * world) % dofs
            jacobian[world, row, dof] = 1.0
            response[world, row, dof] = 0.6
        if count:
            kinds[world, 1] = 3
        for normal in range(2, count - 2, 3):
            kinds[world, normal] = 0
            for tangent in range(normal + 1, min(normal + 3, count)):
                kinds[world, tangent] = 2
                parents[world, tangent] = normal
                rhs[world, tangent] = 0.0
                # Distinct sibling responses cross bit31 and the final partial tile.
                response[world, tangent, 1023 if tangent == normal + 1 else 3134] += 0.05
    mapping = np.roll(np.arange(worlds * dofs, dtype=np.int32).reshape(worlds, dofs)[::-1], 17, axis=1).copy()
    replacements = {
        "world_constraint_count": counts,
        "dense_phase_bounds": np.zeros((worlds, 2), np.int32),
        "world_dof_indices": mapping,
        "world_deferred_dof_mask": np.zeros((worlds, dofs), np.int32),
        "J_world": jacobian,
        "Y_world": response,
        "world_row_type": kinds,
        "world_row_parent": parents,
        "rhs_bias": rhs,
        "world_diag": np.sum(jacobian * response, axis=2) + np.float32(0.01),
        "world_row_w": np.full((worlds, rows), 0.65, np.float32),
        "world_row_mu": np.full((worlds, rows), 0.4, np.float32),
        "world_impulses": np.full((worlds, rows), 0.01, np.float32),
        "mf_impulses": np.full((worlds, mf_rows), 0.02, np.float32),
        "v_out": np.linspace(-0.3, 0.5, worlds * dofs, dtype=np.float32),
    }
    for name, value in replacements.items():
        values[name] = wp.array(value, dtype=int if value.dtype == np.int32 else float, device=device)
    values["iterations"] = 24
    support = DenseRowSupport(worlds, rows, dofs, device, stage_velocities=True)
    values["dense_support_masks"], values["dense_support_world_masks"] = support.masks, support.world_masks
    initial = {name: values[name].numpy().copy() for name in ("v_out", "world_impulses", "mf_impulses")}
    return values, support, initial


@unittest.skipUnless(wp.is_cuda_available(), "Dense-support native solve requires CUDA")
class TestFeatherPGSMFDenseSupportCUDA(unittest.TestCase):
    """Compare the opt-in phase-six consumer with the complete original projector."""

    device = "cuda:0"

    def launch(self, kernel, values):
        """Use the established native kernel ABI and one warp per world."""
        wp.launch_tiled(
            kernel,
            dim=[5],
            block_dim=32,
            device=self.device,
            inputs=[values[name] for name in inspect.signature(kernel.func).parameters],
        )

    def assert_unstaged_velocities_unchanged(self, values, support, output, initial):
        """Preserve nonzero free-body velocities outside the active union exactly."""
        mapping = values["world_dof_indices"].numpy()
        tiles = np.arange(mapping.shape[1]) // 32
        masks = support.world_masks.numpy()
        touched = ((masks[:, tiles // 32] >> (tiles % 32)) & 1).astype(bool)
        outside = mapping[~touched]
        np.testing.assert_array_equal(output[outside], initial[outside])

    def test_sparse_support_matches_dense_projection_and_sibling_updates(self):
        """Preserve drives, regularization and delayed friction without mutating MF impulses."""
        for drives in (False, True):
            for shared_metadata in (False, True):
                values, support, initial = sparse_solve_fixture(self.device, drives=drives)
                kernels = [
                    _get_pgs_solve_mf_gs_kernel(
                        8,
                        6,
                        2053,
                        str(wp.get_device(self.device).arch),
                        has_drive_rows=drives,
                        has_dense_velocity_limit_rows=True,
                        shared_metadata=shared_metadata,
                        dense_support_tiles=enabled,
                        stage_velocities=staged,
                    )
                    for enabled, staged in ((False, False), (True, False), (True, True))
                ]
                support.prepare(values["world_constraint_count"], values["J_world"], values["Y_world"])
                np.testing.assert_array_equal(support.masks.numpy()[:, 5], 0)
                for regularize, delay in ((0, 0), (1, 3), (0, 8)):
                    with self.subTest(
                        drives=drives, shared_metadata=shared_metadata, regularize=regularize, delay=delay
                    ):
                        values.update(regularize=regularize, friction_start_iteration=delay)
                        outputs = []
                        for kernel in kernels:
                            for name, data in initial.items():
                                values[name].assign(data)
                            self.launch(kernel, values)
                            outputs.append({name: values[name].numpy() for name in initial})
                        for output in outputs[1:]:
                            for name in initial:
                                np.testing.assert_allclose(
                                    output[name], outputs[0][name], rtol=3e-5, atol=5e-6, err_msg=name
                                )
                            np.testing.assert_array_equal(output["mf_impulses"], initial["mf_impulses"])
                            np.testing.assert_array_equal(
                                output["world_impulses"][:, 0], initial["world_impulses"][:, 0]
                            )
                            self.assertTrue(np.all(output["world_impulses"][:4, 5] > initial["world_impulses"][:4, 5]))
                            self.assert_unstaged_velocities_unchanged(
                                values, support, output["v_out"], initial["v_out"]
                            )
                            if delay == 8:
                                np.testing.assert_array_equal(output["world_impulses"][:4, 2:4], 0.0)

    def test_empty_union_does_not_skip_nonzero_rhs_projection(self):
        """Project a normal impulse even when no world velocity component is staged."""
        values, support, initial = sparse_solve_fixture(self.device)
        values["J_world"].zero_()
        values["Y_world"].zero_()
        values["world_constraint_count"].assign(np.array([1, 0, 0, 0, 0], np.int32))
        values["dense_phase_bounds"].zero_()
        values["world_row_type"].zero_()
        support.prepare(values["world_constraint_count"], values["J_world"], values["Y_world"])
        np.testing.assert_array_equal(support.world_masks.numpy(), 0)
        outputs = []
        for tiled, staged in ((False, False), (True, False), (True, True)):
            kernel = _get_pgs_solve_mf_gs_kernel(
                8,
                6,
                2053,
                str(wp.get_device(self.device).arch),
                has_drive_rows=True,
                has_dense_velocity_limit_rows=True,
                dense_support_tiles=tiled,
                stage_velocities=staged,
            )
            for name, data in initial.items():
                values[name].assign(data)
            self.launch(kernel, values)
            outputs.append(values["world_impulses"].numpy())
            np.testing.assert_array_equal(values["v_out"].numpy(), initial["v_out"])
            np.testing.assert_array_equal(values["mf_impulses"].numpy(), initial["mf_impulses"])
        self.assertGreater(outputs[0][0, 0], initial["world_impulses"][0, 0])
        for output in outputs[1:]:
            np.testing.assert_allclose(output, outputs[0], rtol=3e-5, atol=5e-6)

    def test_graph_regenerates_support_after_rows_and_operators_shrink(self):
        """Rebuild masks for changed operators and empty worlds inside the captured graph."""
        values, support, initial = sparse_solve_fixture(self.device)
        kernels = [
            _get_pgs_solve_mf_gs_kernel(
                8,
                6,
                2053,
                str(wp.get_device(self.device).arch),
                has_drive_rows=True,
                has_dense_velocity_limit_rows=True,
                dense_support_tiles=enabled,
                stage_velocities=staged,
            )
            for enabled, staged in ((False, False), (True, False), (True, True))
        ]

        def candidate(index):
            support.stage_velocities = index == 2
            support.prepare(values["world_constraint_count"], values["J_world"], values["Y_world"])
            self.launch(kernels[index], values)

        original_j, original_y = values["J_world"].numpy(), values["Y_world"].numpy()
        graphs = []
        for index in (1, 2):
            candidate(index)
            with wp.ScopedCapture(self.device) as capture:
                candidate(index)
            graphs.append(capture.graph)
        for mode in ("loaded", "shrunk", "empty", "loaded"):
            jacobian, response = original_j.copy(), original_y.copy()
            counts = np.array([8, 8, 8, 8, 0], np.int32)
            if mode != "loaded":
                jacobian.fill(0.0)
                response.fill(0.0)
                counts[:] = [2, 0, 1, 0, 0] if mode == "shrunk" else 0
                if mode == "shrunk":
                    jacobian[0, 1, 1601], response[0, 1, 1601] = 1.0, 0.5
            values["world_constraint_count"].assign(counts)
            values["J_world"].assign(jacobian)
            values["Y_world"].assign(response)
            outputs = []
            for graph in (None, *graphs):
                for name, data in initial.items():
                    values[name].assign(data)
                wp.capture_launch(graph) if graph is not None else self.launch(kernels[0], values)
                outputs.append({name: values[name].numpy() for name in initial})
            for output in outputs[1:]:
                for name in initial:
                    np.testing.assert_allclose(
                        output[name], outputs[0][name], rtol=3e-5, atol=5e-6, err_msg=f"{mode}: {name}"
                    )
                self.assert_unstaged_velocities_unchanged(values, support, output["v_out"], initial["v_out"])
            expected = np.zeros((5, 8, 3), np.uint32)
            for world, row, dof in np.argwhere((jacobian != 0) | (response != 0)):
                if row < counts[world]:
                    tile = dof // 32
                    expected[world, row, tile // 32] |= np.uint32(1) << np.uint32(tile % 32)
            np.testing.assert_array_equal(support.masks.numpy(), expected)
            np.testing.assert_array_equal(support.world_masks.numpy(), np.bitwise_or.reduce(expected, axis=1))

    def test_coins_dispatch_dimensions_and_loaded_graph_transitions(self):
        """Check Coins-sized staging and the shared current/coulomb dense projector."""
        values, support, initial = coins_solve_fixture(self.device)
        kernels = [
            _get_pgs_solve_mf_gs_kernel(
                256,
                15984,
                3135,
                str(wp.get_device(self.device).arch),
                has_drive_rows=False,
                has_dense_velocity_limit_rows=False,
                stream_mf_impulses=True,
                dense_support_tiles=tiled,
                stage_velocities=staged,
            )
            for tiled, staged in ((False, False), (True, False), (True, True))
        ]
        kernels.append(
            _get_pgs_solve_mf_gs_kernel(
                256,
                15984,
                3135,
                str(wp.get_device(self.device).arch),
                friction_mode="coulomb_newton",
                has_drive_rows=False,
                has_dense_velocity_limit_rows=False,
                stream_mf_impulses=True,
            )
        )

        def candidate(index):
            support.stage_velocities = index == 2
            support.prepare(values["world_constraint_count"], values["J_world"], values["Y_world"])
            self.launch(kernels[index], values)

        graphs = []
        for index in (1, 2):
            candidate(index)
            with wp.ScopedCapture(self.device) as capture:
                candidate(index)
            graphs.append(capture.graph)
        original_j, original_y = values["J_world"].numpy(), values["Y_world"].numpy()
        original_counts = values["world_constraint_count"].numpy()
        for mode in ("loaded", "shrunk", "empty", "loaded"):
            counts, jacobian, response = original_counts.copy(), original_j.copy(), original_y.copy()
            if mode != "loaded":
                counts[:] = [5, 0, 3, 1, 0] if mode == "shrunk" else 0
                jacobian.fill(0)
                response.fill(0)
                if mode == "shrunk":
                    jacobian[0, 2, 2048], response[0, 2, 2048] = 1.0, 0.6
                    response[0, 3, 1023], response[0, 4, 3134] = 0.05, -0.05
            values["world_constraint_count"].assign(counts)
            values["J_world"].assign(jacobian)
            values["Y_world"].assign(response)
            for regularize, delay in ((0, 0), (1, 3)):
                # Captures fix scalar parameters, so use their original values here.
                values.update(regularize=regularize, friction_start_iteration=delay)
                outputs = []
                for index in range(len(kernels)):
                    for name, data in initial.items():
                        values[name].assign(data)
                    if index in (1, 2) and not regularize:
                        wp.capture_launch(graphs[index - 1])
                    elif index in (1, 2):
                        candidate(index)
                    else:
                        self.launch(kernels[index], values)
                    outputs.append({name: values[name].numpy() for name in initial})
                for output in outputs[1:]:
                    for name in initial:
                        np.testing.assert_allclose(
                            output[name],
                            outputs[0][name],
                            rtol=3e-5,
                            atol=5e-6,
                            err_msg=f"Coins {mode} regularize={regularize}: {name}",
                        )
                    np.testing.assert_array_equal(output["mf_impulses"], initial["mf_impulses"])
                    self.assert_unstaged_velocities_unchanged(values, support, output["v_out"], initial["v_out"])
            np.testing.assert_array_equal(
                support.world_masks.numpy(), np.bitwise_or.reduce(support.masks.numpy(), axis=1)
            )


if __name__ == "__main__":
    unittest.main()
