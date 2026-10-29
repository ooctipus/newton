# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise active-storage budgets while retaining the dense solve's full capacity."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs.solver_feather_pgs import (
    SolverFeatherPGS,
)
from newton._src.solvers.feather_pgs.sweep_routes_diag import tiled_row_kernel, tiled_row_source, tiled_row_staging


class TestFeatherPGSDenseRowBudgetFactory(unittest.TestCase):
    def test_create_distinct_budgets(self):
        """Create three disjoint storage budgets without changing the launch signature."""
        for capacity in (64, 128):
            with self.subTest(capacity=capacity):
                baseline = tiled_row_kernel(capacity, 120)
                kernels = [
                    tiled_row_kernel(capacity, 120, row_budget=budget, min_rows=minimum)
                    for budget, minimum in ((16, 0), (32, 16), (capacity, 32))
                ]
                self.assertEqual(len({kernel.key for kernel in [baseline, *kernels]}), 4)
                labels = [argument.label for argument in baseline.adj.args]
                for kernel in kernels:
                    self.assertEqual([argument.label for argument in kernel.adj.args], labels)

    def test_reject_unsupported_budget_shapes(self):
        """Reject unsupported capacity and budget combinations before compilation."""
        for capacity, budget, minimum in (
            (32, 16, 0),
            (64, 8, 0),
            (64, 32, 32),
            (64, 16, -1),
            (128, 64, 0),
            (128, 32, 32),
        ):
            with self.subTest(capacity=capacity, budget=budget, minimum=minimum):
                with self.assertRaises(ValueError):
                    tiled_row_kernel(capacity, 120, row_budget=budget, min_rows=minimum)

    def test_default_capacity_names(self):
        """Retain the default kernel identity for previously supported capacities."""
        for capacity in (32, 64, 96, 128):
            self.assertEqual(tiled_row_kernel(capacity, 120).key, f"sweep_tiled_row_{capacity}")

    def test_staging_follows_device_shared_memory(self):
        """Stage the Delassus triangle in shared memory only while it fits a block's static budget on the device.

        The budgets are the RTX PRO 6000's (sm_120) and the GB300's static shared memory per block; ptxas reports the
        shared-staging blocks of 224 and 384 rows at 0x19f40 and 0x4a780 bytes.
        """
        self.assertEqual(tiled_row_source(224, staging="shared")[1].bytes, 0x19F40)
        self.assertEqual(tiled_row_source(384, staging="shared")[1].bytes, 0x4A780)
        rtx_pro_6000, gb300 = 100352, 231424
        for capacity, limit, staging in (
            (192, rtx_pro_6000, "shared"),
            (224, rtx_pro_6000, "global"),
            (256, gb300, "shared"),
            (384, gb300, "global"),
        ):
            with self.subTest(capacity=capacity, limit=limit):
                selected, reason = tiled_row_staging(capacity, limit)
                self.assertEqual(selected, staging)
                self.assertIn(str(limit), reason)
        with self.assertRaisesRegex(ValueError, "fewer dense rows"):
            tiled_row_staging(8192, rtx_pro_6000)
        self.assertEqual(tiled_row_kernel(384, 120, staging="global").key, "sweep_tiled_row_384_global")
        with self.assertRaises(ValueError):
            tiled_row_kernel(64, 120, row_budget=32, staging="global")

    def test_dispatch_preserves_arguments(self):
        """Keep one default launch or two partition launches with the original arguments."""
        state = SimpleNamespace(
            _pgs_solve_tiled_row_kernel=object(),
            _pgs_solve_tiled_row_budgets=(),
            world_count=17,
            model=SimpleNamespace(device="mock-device"),
            pgs_omega=0.8,
            _regularization_enabled=True,
            _solve_readout=SimpleNamespace(operands=lambda kernel: []),
        )
        for name in ("constraint_count", "diag", "C", "rhs", "impulses", "row_type", "row_parent", "row_mu", "row_w"):
            setattr(state, name, object())
        expected = [
            state.constraint_count,
            state.diag,
            state.C,
            state.rhs,
            state.impulses,
            8,
            0.8,
            state.row_type,
            state.row_parent,
            state.row_mu,
            state.row_w,
            1,
            2,
            3,
        ]
        for kernels in ((), (object(), object())):
            state._pgs_solve_tiled_row_budgets = kernels
            with patch("newton._src.solvers.feather_pgs.solver_feather_pgs.wp.launch_tiled") as launch:
                SolverFeatherPGS._stage5_pgs_solve_world_tiled_row(state, 8, 2, 3)
            self.assertEqual(launch.call_count, len(kernels) or 1)
            for call, kernel in zip(
                launch.call_args_list, kernels or (state._pgs_solve_tiled_row_kernel,), strict=True
            ):
                self.assertIs(call.args[0], kernel)
                self.assertEqual(
                    call.kwargs, {"dim": [17], "inputs": expected, "block_dim": 32, "device": "mock-device"}
                )


def _dense_fixture(capacity=64):
    """Cover every active size, cross-budget triples and readable padding fallback."""
    rng = np.random.default_rng(278)
    counts = np.array([*range(capacity + 1), 16, 32, 3], dtype=np.int32)
    worlds = len(counts)
    factors = rng.normal(size=(worlds, capacity, 8)).astype(np.float32) * 0.06
    matrix = factors @ factors.transpose(0, 2, 1) + np.eye(capacity, dtype=np.float32)[None]
    diag = np.diagonal(matrix, axis1=1, axis2=2).copy() + np.float32(0.03)
    rhs = rng.uniform(-0.7, 0.7, size=(worlds, capacity)).astype(np.float32)
    impulses = rng.uniform(0.05, 0.4, size=(worlds, capacity)).astype(np.float32)
    types = np.tile(np.array([0, 1, 3, 4, 5, 6, 1, 0], dtype=np.int32), (worlds, capacity // 8))
    parents = np.full((worlds, capacity), -1, dtype=np.int32)
    mu = np.full((worlds, capacity), 0.35, dtype=np.float32)
    weights = rng.uniform(0.25, 1.0, size=(worlds, capacity)).astype(np.float32)
    for world, count in enumerate(counts):
        for normal in (0, 7, 14, 23, 30, 39, 46, 55, 61, 70, 78, 87, 94, 103, 110, 119, 125):
            if normal + 2 < count:
                types[world, normal] = 0
                types[world, normal + 1 : normal + 3] = 2
                parents[world, normal + 1 : normal + 3] = normal
                rhs[world, normal] = -0.5
        if count and world % 3 == 0:
            diag[world, count // 2] = 0.0 if world % 2 else -1.0
    # Original metadata can legally reference capacity padding. A reduced
    # storage budget must return this whole world to the full-capacity kernel.
    for world, row, parent in ((capacity + 1, 15, 14), (capacity + 2, 31, 30), (capacity + 3, 1, 10)):
        types[world, row] = 2
        parents[world, row] = parent
        diag[world, row] = 1.0
        impulses[world, parent] = 0.1
        impulses[world, parent + 1 : parent + 3] = 0.7
    return counts, diag, matrix, rhs, impulses, types, parents, mu, weights


@unittest.skipUnless(wp.is_cuda_available(), "Native dense row budgets require CUDA")
class TestFeatherPGSDenseRowBudget(unittest.TestCase):
    def _compare(
        self,
        *,
        iterations,
        omega,
        friction_start,
        iteration_offset,
        capacity=64,
        graph=False,
        nan_diag=False,
        change_counts=False,
        regularize=0,
    ):
        """Compare both budget partitions against the default factory byte for byte."""
        device = wp.get_device("cuda:0")
        arrays = list(_dense_fixture(capacity))
        # Keep the shared C64 fixture unchanged for its existing consumers, but
        # do not numerically compare rows whose tangent sibling lies beyond m:
        # the reference only initializes the active matrix triangle. Admission
        # for those metadata cases is checked independently below.
        for world, active in ((capacity + 1, 16), (capacity + 2, 32)):
            arrays[5][world] = arrays[5][active]
            arrays[6][world] = arrays[6][active]
        if nan_diag:
            # NaN does not take the baseline's denom <= 0 early continue. The
            # malformed but in-capacity parent must still use full-budget storage.
            arrays[1][capacity + 3, 1] = np.nan
        values = [wp.array(value, device=device) for value in arrays]
        snapshots = [value.numpy() for value in values]
        counts, diag, matrix, rhs, initial, types, parents, mu, weights = values
        original = tiled_row_kernel(capacity, device.arch)
        reference = wp.clone(initial)

        def launch(kernel, impulses):
            wp.launch_tiled(
                kernel,
                dim=[len(arrays[0])],
                inputs=[
                    counts,
                    diag,
                    matrix,
                    rhs,
                    impulses,
                    iterations,
                    omega,
                    types,
                    parents,
                    mu,
                    weights,
                    regularize,
                    friction_start,
                    iteration_offset,
                ],
                block_dim=32,
                device=device,
            )

        launch(original, reference)
        expected = reference.numpy()
        for partition in (((32, 0), (capacity, 32)), ((16, 0), (32, 16), (capacity, 32))):
            with self.subTest(partition=partition, graph=graph):
                result = wp.clone(initial)
                kernels = [
                    tiled_row_kernel(capacity, device.arch, row_budget=budget, min_rows=minimum)
                    for budget, minimum in partition
                ]
                # Compile before capture, then restore the original warm impulses.
                for kernel in kernels:
                    launch(kernel, result)
                if graph:
                    wp.copy(result, initial)
                    with wp.ScopedCapture(device=device) as capture:
                        for kernel in kernels:
                            launch(kernel, result)
                    wp.capture_launch(capture.graph)
                np.testing.assert_array_equal(result.numpy().view(np.uint32), expected.view(np.uint32))
                if change_counts:
                    # Replay the same graph as worlds enter/leave each owner,
                    # including the rows above the failed 96-row capacity.
                    for changed in (np.full_like(arrays[0], capacity), np.zeros_like(arrays[0]), arrays[0]):
                        counts.assign(changed)
                        wp.copy(reference, initial)
                        wp.copy(result, initial)
                        launch(original, reference)
                        wp.capture_launch(capture.graph)
                        np.testing.assert_array_equal(result.numpy().view(np.uint32), reference.numpy().view(np.uint32))
                for before, value in zip(snapshots, values, strict=True):
                    self.assertEqual(before.tobytes(), value.numpy().tobytes())
        # The zero-row world and normal capacity tails remain exactly unchanged.
        self.assertEqual(expected[0].tobytes(), arrays[4][0].tobytes())
        for world, count in enumerate(arrays[0][: capacity + 1]):
            self.assertEqual(expected[world, count:].tobytes(), arrays[4][world, count:].tobytes())

    def test_global_staging_matches_shared_staging(self):
        """Solve a symmetric Delassus problem bit for bit alike from shared and from global staging.

        The shared staging reads the operator's lower triangle and the global staging its rows, so the two read
        the same values when the operator is symmetric bit for bit; every lane then sums the same terms in the
        same order.
        """
        device = wp.get_device("cuda:0")
        for capacity in (64, 128):
            with self.subTest(capacity=capacity):
                arrays = list(_dense_fixture(capacity))
                # As in _compare: keep the tangent siblings of these worlds inside their active rows.
                for world, active in ((capacity + 1, 16), (capacity + 2, 32)):
                    arrays[5][world] = arrays[5][active]
                    arrays[6][world] = arrays[6][active]
                lower = np.tril(arrays[2])
                arrays[2] = lower + np.tril(lower, -1).transpose(0, 2, 1)
                counts, diag, matrix, rhs, initial, types, parents, mu, weights = (
                    wp.array(value, device=device) for value in arrays
                )
                results = []
                for staging in ("shared", "global"):
                    impulses = wp.clone(initial)
                    wp.launch_tiled(
                        tiled_row_kernel(capacity, device.arch, staging=staging),
                        dim=[len(arrays[0])],
                        inputs=[counts, diag, matrix, rhs, impulses, 8, 1.0, types, parents, mu, weights, 0, 2, 0],
                        block_dim=32,
                        device=device,
                    )
                    results.append(impulses.numpy())
                self.assertFalse(np.array_equal(results[0], arrays[4]))
                np.testing.assert_array_equal(results[1].view(np.uint32), results[0].view(np.uint32))

    def test_all_counts_and_projection_phases(self):
        """Preserve every count, row projector, contact regularization, warm state and delayed-friction phase."""
        for capacity in (64, 128):
            for iterations, omega, delay, offset, regularize in (
                (0, 1.0, 0, 0, 0),
                (1, 0.8, 0, 0, 0),
                (8, 1.0, 2, 0, 0),
                (3, 1.2, 2, 3, 0),
                (8, 1.0, 2, 0, 1),
            ):
                with self.subTest(
                    capacity=capacity,
                    iterations=iterations,
                    omega=omega,
                    delay=delay,
                    offset=offset,
                    regularize=regularize,
                ):
                    self._compare(
                        capacity=capacity,
                        iterations=iterations,
                        omega=omega,
                        friction_start=delay,
                        iteration_offset=offset,
                        regularize=regularize,
                    )

    def test_graph_preserves_full_capacity(self):
        """Preserve all capacity padding when the disjoint kernels run in a graph."""
        for capacity in (64, 128):
            with self.subTest(capacity=capacity):
                self._compare(
                    capacity=capacity,
                    iterations=8,
                    omega=1.0,
                    friction_start=2,
                    iteration_offset=0,
                    graph=True,
                    change_counts=True,
                )

    def test_nan_diagonal_preserves_fallback(self):
        """Mirror the baseline NaN denominator branch before reducing row storage."""
        for capacity in (64, 128):
            with self.subTest(capacity=capacity):
                self._compare(
                    capacity=capacity, iterations=2, omega=1.0, friction_start=0, iteration_offset=0, nan_diag=True
                )

    def test_padding_dependencies_decline_small_storage(self):
        """Keep padding-dependent friction and normal chains on full storage."""
        device = wp.get_device("cuda:0")
        for capacity in (64, 128):
            for denominator in (1.0, 0.0, -1.0, np.nan):
                with self.subTest(capacity=capacity, denominator=denominator):
                    arrays = list(_dense_fixture(capacity))
                    # The first tangent references a sibling outside the active
                    # prefix; it still needs full storage for any denominator.
                    for world, row in ((capacity + 1, 15), (capacity + 2, 31)):
                        arrays[1][world, row] = denominator
                    # A valid active tangent pair can depend on a normal impulse
                    # stored in padding through its patch-normal chain.
                    world = capacity + 3
                    arrays[5][world, :3] = (0, 2, 2)
                    arrays[6][world, :3] = (10, 0, 0)
                    arrays[6][world, 10] = -1
                    values = [wp.array(value, device=device) for value in arrays]
                    counts, diag, matrix, rhs, initial, types, parents, mu, weights = values
                    for budget, minimum in ((16, 0), (32, 0), (32, 16)):
                        result = wp.clone(initial)
                        kernel = tiled_row_kernel(capacity, device.arch, row_budget=budget, min_rows=minimum)
                        wp.launch_tiled(
                            kernel,
                            dim=[len(arrays[0])],
                            inputs=[counts, diag, matrix, rhs, result, 2, 1.0, types, parents, mu, weights, 0, 0, 0],
                            block_dim=32,
                            device=device,
                        )
                        np.testing.assert_array_equal(result.numpy()[capacity + 1 :], arrays[4][capacity + 1 :])

                    # Unlike an out-of-prefix tangent sibling, this chain reads
                    # defined matrix entries, so compare its full-owner output.
                    expected = wp.clone(initial)
                    result = wp.clone(initial)
                    for target, kernel in (
                        (expected, tiled_row_kernel(capacity, device.arch)),
                        (
                            result,
                            tiled_row_kernel(capacity, device.arch, row_budget=capacity, min_rows=32),
                        ),
                    ):
                        wp.launch_tiled(
                            kernel,
                            dim=[len(arrays[0])],
                            inputs=[counts, diag, matrix, rhs, target, 2, 1.0, types, parents, mu, weights, 0, 0, 0],
                            block_dim=32,
                            device=device,
                        )
                    np.testing.assert_allclose(result.numpy()[world], expected.numpy()[world], rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
