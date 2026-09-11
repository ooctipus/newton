# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise exact active-storage budgets for the existing dense C64 PGS solve."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs.solver_feather_pgs import SolverFeatherPGS, _get_pgs_solve_tiled_row_kernel


class TestFeatherPGSDenseRowBudgetFactory(unittest.TestCase):
    def test_create_distinct_budgets(self):
        """Create three disjoint storage budgets without changing the launch signature."""
        baseline = _get_pgs_solve_tiled_row_kernel(64, 120)
        kernels = [
            _get_pgs_solve_tiled_row_kernel(64, 120, row_budget=budget, min_rows=minimum)
            for budget, minimum in ((16, 0), (32, 16), (64, 32))
        ]
        self.assertEqual(len({kernel.key for kernel in [baseline, *kernels]}), 4)
        labels = [argument.label for argument in baseline.adj.args]
        for kernel in kernels:
            self.assertEqual([argument.label for argument in kernel.adj.args], labels)

    def test_reject_unsupported_budget_shapes(self):
        """Reject unsupported capacity and budget combinations before compilation."""
        for capacity, budget, minimum in ((32, 16, 0), (64, 8, 0), (64, 32, 32), (64, 16, -1)):
            with self.subTest(capacity=capacity, budget=budget, minimum=minimum):
                with self.assertRaises(ValueError):
                    _get_pgs_solve_tiled_row_kernel(capacity, 120, row_budget=budget, min_rows=minimum)

    def test_default_capacity_names(self):
        """Retain the default kernel identity for previously supported capacities."""
        for capacity in (32, 64, 96):
            self.assertEqual(_get_pgs_solve_tiled_row_kernel(capacity, 120).key, f"pgs_solve_tiled_row_{capacity}")

    def test_dispatch_preserves_arguments(self):
        """Keep one default launch or two partition launches with the original arguments."""
        state = SimpleNamespace(
            _pgs_solve_tiled_row_kernel=object(),
            _pgs_solve_tiled_row_budgets=(),
            world_count=17,
            model=SimpleNamespace(device="mock-device"),
            pgs_iterations=8,
            pgs_omega=0.8,
            _pgs_friction_start_iteration=2,
            _pgs_iteration_offset=3,
        )
        for name in ("constraint_count", "diag", "C", "rhs", "impulses", "row_type", "row_parent", "row_mu"):
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
            2,
            3,
        ]
        for kernels in ((), (object(), object())):
            state._pgs_solve_tiled_row_budgets = kernels
            with patch("newton._src.solvers.feather_pgs.solver_feather_pgs.wp.launch_tiled") as launch:
                SolverFeatherPGS._stage5_pgs_solve_world_tiled_row(state)
            self.assertEqual(launch.call_count, len(kernels) or 1)
            for call, kernel in zip(
                launch.call_args_list, kernels or (state._pgs_solve_tiled_row_kernel,), strict=True
            ):
                self.assertIs(call.args[0], kernel)
                self.assertEqual(
                    call.kwargs, {"dim": [17], "inputs": expected, "block_dim": 32, "device": "mock-device"}
                )


def _dense_fixture():
    """Cover every active size, cross-budget triples and readable padding fallback."""
    rng = np.random.default_rng(278)
    counts = np.array([*range(65), 16, 32, 3], dtype=np.int32)
    worlds = len(counts)
    factors = rng.normal(size=(worlds, 64, 8)).astype(np.float32) * 0.06
    matrix = factors @ factors.transpose(0, 2, 1) + np.eye(64, dtype=np.float32)[None]
    diag = np.diagonal(matrix, axis1=1, axis2=2).copy() + np.float32(0.03)
    rhs = rng.uniform(-0.7, 0.7, size=(worlds, 64)).astype(np.float32)
    impulses = rng.uniform(0.05, 0.4, size=(worlds, 64)).astype(np.float32)
    types = np.tile(np.array([0, 1, 3, 4, 5, 6, 1, 0], dtype=np.int32), (worlds, 8))
    parents = np.full((worlds, 64), -1, dtype=np.int32)
    mu = np.full((worlds, 64), 0.35, dtype=np.float32)
    for world, count in enumerate(counts):
        for normal in (0, 7, 14, 23, 30, 39, 46, 55, 61):
            if normal + 2 < count:
                types[world, normal] = 0
                types[world, normal + 1 : normal + 3] = 2
                parents[world, normal + 1 : normal + 3] = normal
                rhs[world, normal] = -0.5
        if count and world % 3 == 0:
            diag[world, count // 2] = 0.0 if world % 2 else -1.0
    # Original C64 metadata can legally reference capacity padding. A reduced
    # storage budget must return this whole world to the full-capacity kernel.
    for world, row, parent in ((65, 15, 14), (66, 31, 30), (67, 1, 10)):
        types[world, row] = 2
        parents[world, row] = parent
        diag[world, row] = 1.0
        impulses[world, parent] = 0.1
        impulses[world, parent + 1 : parent + 3] = 0.7
    return counts, diag, matrix, rhs, impulses, types, parents, mu


@unittest.skipUnless(wp.is_cuda_available(), "Native dense row budgets require CUDA")
class TestFeatherPGSDenseRowBudget(unittest.TestCase):
    def _compare(self, *, iterations, omega, friction_start, iteration_offset, graph=False, nan_diag=False):
        """Compare both budget partitions against the default factory byte for byte."""
        device = wp.get_device("cuda:0")
        arrays = list(_dense_fixture())
        if nan_diag:
            # NaN does not take the baseline's denom <= 0 early continue. The
            # malformed but in-capacity parent must still use full-budget storage.
            arrays[1][67, 1] = np.nan
        values = [wp.array(value, device=device) for value in arrays]
        snapshots = [value.numpy() for value in values]
        counts, diag, matrix, rhs, initial, types, parents, mu = values
        original = _get_pgs_solve_tiled_row_kernel(64, device.arch)
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
                    friction_start,
                    iteration_offset,
                ],
                block_dim=32,
                device=device,
            )

        launch(original, reference)
        expected = reference.numpy()
        for partition in (((32, 0), (64, 32)), ((16, 0), (32, 16), (64, 32))):
            with self.subTest(partition=partition, graph=graph):
                result = wp.clone(initial)
                kernels = [
                    _get_pgs_solve_tiled_row_kernel(64, device.arch, row_budget=budget, min_rows=minimum)
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
                for before, value in zip(snapshots, values, strict=True):
                    self.assertEqual(before.tobytes(), value.numpy().tobytes())
        # The zero-row world and normal capacity tails remain exactly unchanged.
        self.assertEqual(expected[0].tobytes(), arrays[4][0].tobytes())
        for world, count in enumerate(arrays[0][:65]):
            self.assertEqual(expected[world, count:].tobytes(), arrays[4][world, count:].tobytes())

    def test_all_counts_and_projection_phases(self):
        """Preserve every count, row projector, warm state and delayed-friction phase."""
        for iterations, omega, delay, offset in ((0, 1.0, 0, 0), (1, 0.8, 0, 0), (8, 1.0, 2, 0), (3, 1.2, 2, 3)):
            with self.subTest(iterations=iterations, omega=omega, delay=delay, offset=offset):
                self._compare(iterations=iterations, omega=omega, friction_start=delay, iteration_offset=offset)

    def test_graph_preserves_full_capacity(self):
        """Preserve all capacity padding when the disjoint kernels run in a graph."""
        self._compare(iterations=8, omega=1.0, friction_start=2, iteration_offset=0, graph=True)

    def test_nan_diagonal_preserves_fallback(self):
        """Mirror the baseline NaN denominator branch before reducing row storage."""
        self._compare(iterations=2, omega=1.0, friction_start=0, iteration_offset=0, nan_diag=True)


if __name__ == "__main__":
    unittest.main()
