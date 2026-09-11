# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exact C64 dense GS with lane-owned state and original matrix reads."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs.dense_row_state import get_dense_row_state_kernel
from newton._src.solvers.feather_pgs.solver_feather_pgs import SolverFeatherPGS, _get_pgs_solve_tiled_row_kernel
from newton.tests.test_feather_pgs_dense_row_budget import _dense_fixture


class TestFeatherPGSDenseRowStateFactory(unittest.TestCase):
    def test_signature_and_invalid_packing(self):
        """Retain the dense launch arguments and reject unsupported CTA packing."""
        reference = _get_pgs_solve_tiled_row_kernel(64, 120)
        labels = [arg.label for arg in reference.adj.args]
        for worlds in (1, 2, 4):
            for budget in (32, 64):
                kernel = get_dense_row_state_kernel(120, worlds, row_budget=budget)
                self.assertEqual([arg.label for arg in kernel.adj.args], labels)
                self.assertIs(kernel, get_dense_row_state_kernel(120, worlds, row_budget=budget))
        for worlds in (0, 3, 8, -1):
            with self.subTest(worlds=worlds), self.assertRaises(ValueError):
                get_dense_row_state_kernel(120, worlds)
        for budget in (0, 16, 48, 96):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                get_dense_row_state_kernel(120, row_budget=budget)

    def test_odd_world_dispatch_and_fallback(self):
        """Pack both register budgets without changing row ownership or arguments."""
        register, fallback = object(), object()
        state = SimpleNamespace(
            _pgs_solve_tiled_row_kernel=object(),
            _pgs_solve_tiled_row_budgets=(register, fallback),
            _pgs_solve_tiled_row_register_kernels=(register, fallback),
            world_count=69,
            model=SimpleNamespace(device="mock-device"),
            pgs_iterations=8,
            pgs_omega=0.8,
            _pgs_friction_start_iteration=2,
            _pgs_iteration_offset=3,
        )
        for name in ("constraint_count", "diag", "C", "rhs", "impulses", "row_type", "row_parent", "row_mu"):
            setattr(state, name, object())
        with patch("newton._src.solvers.feather_pgs.solver_feather_pgs.wp.launch_tiled") as launch:
            SolverFeatherPGS._stage5_pgs_solve_world_tiled_row(state)
        self.assertEqual(launch.call_count, 2)
        small, full = launch.call_args_list
        self.assertIs(small.args[0], register)
        self.assertIs(full.args[0], fallback)
        self.assertEqual(small.kwargs["dim"], [35])
        self.assertEqual(small.kwargs["block_dim"], 64)
        self.assertEqual(full.kwargs["dim"], [35])
        self.assertEqual(full.kwargs["block_dim"], 64)
        self.assertEqual(small.kwargs["inputs"], full.kwargs["inputs"])


@unittest.skipUnless(wp.is_cuda_available(), "Native dense register state requires CUDA")
class TestFeatherPGSDenseRowState(unittest.TestCase):
    def test_raw_owner_projection_phases_and_stream_graphs(self):
        """Preserve all typed bytes and alias-owner padding in eager/graph solves."""
        device = wp.get_device("cuda:0")
        arrays = [np.concatenate((value, value[31:32]), axis=0) for value in _dense_fixture()]
        arrays[1].view(np.uint32)[67, 1] = np.uint32(0x7FC12345)
        arrays[1][6, 1] = np.float32(-0.0)
        arrays[5][6, 1] = 2
        arrays[6][6, 1] = np.iinfo(np.int32).min
        arrays[7][7, 1:3] = (-0.0, -1.0)
        arrays[4][7, :3] = (0.0, -0.0, 0.7)
        # Full-budget worlds may legally read/project capacity padding, even
        # with a row-parent relation outside the usual in-prefix triplet.
        arrays[5][34, 1] = 2
        arrays[6][34, 1] = 60
        arrays[1][34, 1] = 1.0
        arrays[4][34, 60:63] = (0.1, 0.7, 0.8)
        arrays[5][40, 1] = 2
        arrays[6][40, 1] = 1
        arrays[1][40, 1] = 1.0
        arrays[5][41:43, 1] = 2
        arrays[6][41:43, 1] = (62, 63)
        arrays[1][41:43, 1] = 1.0
        arrays[7][42, 1] = -1.0  # Radius <= 0 must not read the out-of-capacity sibling.
        arrays[4][42, 63] = 0.5
        upper = np.triu_indices(64, 1)
        arrays[2].view(np.uint32)[:, upper[0], upper[1]] = np.uint32(0xFFC54321)
        for world, count in enumerate(arrays[0]):
            arrays[2].view(np.uint32)[world, count:, :] = np.uint32(0x7FC12345)
        arrays[2].view(np.uint32)[35, 2, 1] = np.uint32(0x7FC98765)
        arrays[2].view(np.uint32)[36, 32, 1] = np.uint32(0x80000001)
        arrays[3].view(np.uint32)[36, 32] = np.uint32(0x80000000)
        permutation = np.random.default_rng(891).permutation(69)
        arrays = [np.ascontiguousarray(value[permutation]) for value in arrays]

        offsets, cursor = [], 256
        for value in arrays:
            offsets.append(cursor)
            cursor = ((cursor + value.nbytes + 255) // 256) * 256 + 256
        raw = np.full(cursor + 256, 0xA7, dtype=np.uint8)
        for offset, value in zip(offsets, arrays, strict=True):
            raw[offset : offset + value.nbytes] = value.view(np.uint8).ravel()
        changed_raw = raw.copy()
        changed_matrix = (
            changed_raw[offsets[2] : offsets[2] + arrays[2].nbytes].view(np.float32).reshape(arrays[2].shape)
        )
        changed_world = int(np.flatnonzero(arrays[0] == 64)[0])
        changed_matrix[changed_world, 0, 0] *= np.float32(1.5)

        def pack():
            owner = wp.array(raw, dtype=wp.uint8, device=device)
            values = []
            for offset, host in zip(offsets, arrays, strict=True):
                view = wp.array(
                    ptr=owner.ptr + offset,
                    shape=host.shape,
                    dtype=wp.int32 if host.dtype == np.int32 else wp.float32,
                    device=device,
                    copy=False,
                )
                view._ref = owner
                values.append(view)
            return owner, values

        def arguments(values, config):
            counts, diag, matrix, rhs, impulses, types, parents, mu = values
            iterations, omega, delay, offset = config
            return [counts, diag, matrix, rhs, impulses, iterations, omega, types, parents, mu, delay, offset]

        stream = wp.Stream(device)
        with wp.ScopedStream(stream):
            reference_owner, reference_values = pack()
            trial_owner, trial_values = pack()
            initial = wp.array(raw, dtype=wp.uint8, device=device)
            changed_initial = wp.array(changed_raw, dtype=wp.uint8, device=device)
            original = _get_pgs_solve_tiled_row_kernel(64, device.arch)
            for config in ((0, 1.0, 0, 0), (1, 0.8, 0, 0), (8, 1.0, 2, 0), (3, 1.2, 2, 3), (1, -0.0, 8, 0)):
                wp.copy(reference_owner, initial)
                wp.launch_tiled(original, dim=[69], inputs=arguments(reference_values, config), block_dim=32)
                expected = reference_owner.numpy()
                for worlds in (1, 2, 4):
                    with self.subTest(config=config, worlds=worlds):
                        kernel = get_dense_row_state_kernel(device.arch, worlds)
                        fallback = get_dense_row_state_kernel(device.arch, worlds, row_budget=64)

                        def launch(config=config, kernel=kernel, worlds=worlds, fallback=fallback):
                            args = arguments(trial_values, config)
                            wp.launch_tiled(
                                kernel, dim=[(69 + worlds - 1) // worlds], inputs=args, block_dim=32 * worlds
                            )
                            wp.launch_tiled(
                                fallback, dim=[(69 + worlds - 1) // worlds], inputs=args, block_dim=32 * worlds
                            )

                        wp.copy(trial_owner, initial)
                        launch()
                        self.assertEqual(trial_owner.numpy().tobytes(), expected.tobytes())
                        with wp.ScopedCapture(device=device, stream=stream) as captured:
                            launch()
                        for seed in (initial, changed_initial):
                            # Refresh C at the same address between graph replays.
                            wp.copy(reference_owner, seed)
                            wp.launch_tiled(
                                original, dim=[69], inputs=arguments(reference_values, config), block_dim=32
                            )
                            graph_expected = reference_owner.numpy()
                            if seed is changed_initial and config[0] > 0 and config[1] != 0.0:
                                changed_start = offsets[4] + changed_world * 64 * np.dtype(np.float32).itemsize
                                changed_end = changed_start + 64 * np.dtype(np.float32).itemsize
                                self.assertNotEqual(
                                    graph_expected[changed_start:changed_end].tobytes(),
                                    expected[changed_start:changed_end].tobytes(),
                                    "The refreshed matrix must affect this world's solved impulses",
                                )
                            wp.copy(trial_owner, seed)
                            wp.capture_launch(captured.graph, stream=stream)
                            self.assertEqual(trial_owner.numpy().tobytes(), graph_expected.tobytes())
                start, end = offsets[4], offsets[4] + arrays[4].nbytes
                self.assertEqual(expected[:start].tobytes(), raw[:start].tobytes())
                self.assertEqual(expected[end:].tobytes(), raw[end:].tobytes())


if __name__ == "__main__":
    unittest.main()
