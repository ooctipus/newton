# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU factory contracts for the opt-in register-whitening layout."""

import hashlib
import unittest
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import solver_feather_pgs as solver


class TestRegisterWhitening(unittest.TestCase):
    """Keep selection shape-based, fail closed, and preserve opt-out native code."""

    def setUp(self):
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        self.patches.enter_context(
            patch.object(wp, "init", side_effect=AssertionError("CPU factory test initialized Warp"))
        )
        self.patches.enter_context(
            patch("warp._src.context.init", side_effect=AssertionError("CPU factory test initialized Warp"))
        )
        for name in ("_INK_CHECK", "_WR_CHECK", "_WR_WARM"):
            self.patches.enter_context(patch.object(solver, name, False))
        self.patches.enter_context(patch.dict("os.environ", {"FEATHER_PGS_WR_STOP": "0"}))

    def _mode(self, **changes):
        args = {
            "enabled": True,
            "device_arch": 120,
            "rows": 32,
            "max_world_dofs": 18,
            "matrix_free": True,
            "inkernel_response": (18, 0, 0, 0),
            "world_rows": True,
            "exact_row_sums": True,
        }
        args.update(changes)
        return solver._parallel_register_whitening_mode(**args)

    def _source(self, rows=32, arch=120, enabled=False, **changes):
        kwargs = {
            "rows": rows,
            "min_rows": 0 if rows == 32 else 32,
            "sweeps": 24,
            "matrix_free": True,
            "inkernel_response": (18, 0, 0, 0),
            "world_rows": True,
            "exact_row_sums": True,
        }
        kwargs.update(changes)
        snippets = []
        native = wp.func_native

        def capture(snippet, *args, **kwargs):
            snippets.append(snippet)
            return native(snippet, *args, **kwargs)

        with patch.object(solver, "_REGISTER_WHITENING", enabled), patch.object(wp, "func_native", capture):
            kernel = solver._get_pgs_solve_parallel_kernel(192, 32, 18, arch, **kwargs)
        self.assertEqual(len(snippets), 1)
        return kernel, snippets[0]

    def test_mode_selection(self):
        """Only the measured row-capacity/architecture pairs select new layouts."""
        for arch, rows, expected in ((120, 32, 1), (103, 32, 2), (120, 48, 1), (103, 48, 1)):
            with self.subTest(arch=arch, rows=rows):
                self.assertEqual(self._mode(device_arch=arch, rows=rows), expected)
                self.assertEqual(self._mode(device_arch=str(arch), rows=rows), expected)

    def test_unsupported_selection(self):
        """Opt-out, unknown architectures and unmeasured modes retain the old path."""
        changes = (
            {"enabled": False},
            {"device_arch": 90},
            {"device_arch": 999},
            {"device_arch": "sm_120"},
            {"device_arch": None},
            {"device_arch": 120.5},
            {"rows": 16},
            {"rows": 64},
            {"max_world_dofs": 22},
            {"inkernel_response": None},
            {"inkernel_response": (12, 6, 0, 12)},
            {"matrix_free": False},
            {"world_rows": False},
            {"exact_row_sums": False},
            {"has_drive_rows": True},
            {"has_dense_velocity_limit_rows": True},
            {"nesterov": False},
            {"sweeps": 8},
            {"warm_start": True},
            {"debug": True},
        )
        for changed in changes:
            with self.subTest(changed=changed):
                self.assertEqual(self._mode(**changed), 0)

    def test_default_native_identity(self):
        """Opt-out and unknown architecture emit the exact published native text."""
        # Fingerprints captured from published a2ca01b14 before this feature.
        expected = {
            32: "f20baadbfba9e45387b58e643080c676769dae10e5d7ce3637ea498b94f07ffe",
            48: "c1a55b6860c8b72ca5e6c18921a8b0e1e8f88c21e17ed0915676dfe3bb4680fd",
        }
        for rows, digest in expected.items():
            reference, source = self._source(rows=rows)
            self.assertEqual(hashlib.sha256(source.encode()).hexdigest(), digest)
            fallback, fallback_source = self._source(rows=rows, arch=999, enabled=True)
            self.assertEqual(fallback_source, source)
            self.assertEqual(fallback.key, reference.key)

    def test_selected_native_order_and_ownership(self):
        """Selected kernels keep ascending recurrences and protect overlay phases."""
        for arch, rows, mode in ((120, 32, 1), (103, 32, 2), (120, 48, 1), (103, 48, 1)):
            with self.subTest(arch=arch, rows=rows):
                kernel, source = self._source(rows=rows, arch=arch, enabled=True)
                self.assertTrue(kernel.key.endswith(f"_rw{mode}"))
                self.assertIn("const volatile float* Lg = s_L;", source)
                self.assertEqual(sum(source.count(f"z{a} -= Lg[") for a in range(18)), 306)
                position = source.index("const volatile float* Lg = s_L;")
                for a in range(18):
                    for k in range(a):
                        term = f"z{a} -= Lg[{a * 18 + k}] * Jr[{k}];"
                        position = source.index(term, position) + len(term)
                publication = source.index("s_Yt[d * YS + i] = Jr[d];")
                self.assertLess(position, publication)
                if mode == 2:
                    self.assertIn("float* s_L = s_Yt;", source)
                    self.assertIn("SYNC();\n    if (lane < n_rows)", source[publication - 150 : publication])
                    reduction = source.index("// u = Z^T")
                    reload_factor = source.index("s_L[e] = ink_L_a.data", reduction)
                    reconstruct = source.index("for (int a = 18 - 1", reload_factor)
                    self.assertIn("SYNC();", source[reload_factor:reconstruct])

    def test_unsupported_native_identity(self):
        """Turning the flag on does not alter unsupported valid factory variants."""
        for changes in (
            {"rows": 64},
            {"exact_row_sums": False},
            {"world_rows": False},
            {"has_drive_rows": True},
            {"has_dense_velocity_limit_rows": True},
            {"nesterov": False},
            {"sweeps": 8},
            {"lean_sweep": True},
            {"bound_finger": True},
        ):
            with self.subTest(changes=changes):
                reference, source = self._source(**changes)
                candidate, candidate_source = self._source(enabled=True, **changes)
                self.assertEqual(candidate_source, source)
                self.assertEqual(candidate.key, reference.key)


class TestRegisterWhiteningNative(unittest.TestCase):
    """Exercise whitening, projection and phase boundaries with identical inputs."""

    def test_phase_and_row_boundaries(self):
        """Compare full native buffers for warm impulses, cross-warp friction and graph replay."""
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("native parallel projection requires CUDA")
        device = devices[0]
        if device.arch not in (103, 120):
            self.skipTest("register whitening is not selected for this architecture")
        counts = np.array([0, 1, 2, 3, 31, 32, 33, 34, 47, 48, 49], dtype=np.int32)
        worlds, capacity, dofs = len(counts), 192, 18
        rng = np.random.default_rng(1941)
        factor = np.tril(rng.normal(0.0, 0.02, (worlds, dofs, dofs))).astype(np.float32)
        factor[:, np.arange(dofs), np.arange(dofs)] += 2.0
        row_type = np.full((worlds, capacity), 3, dtype=np.int32)
        parent = np.full((worlds, capacity), -1, dtype=np.int32)
        for world, count in enumerate(counts):
            if count >= 3:
                normal = 30 if count >= 33 else min(count - 3, 29)
                row_type[world, normal] = 0
                row_type[world, normal + 1 : normal + 3] = 2
                parent[world, normal + 1 : normal + 3] = normal
        metadata = np.zeros((worlds, 4), dtype=np.int32)
        metadata[:, 0] = np.arange(worlds)
        metadata[:, 2] = -1
        host = {
            "general_world_count": np.array([worlds], dtype=np.int32),
            "general_worlds": np.arange(worlds - 1, -1, -1, dtype=np.int32),
            "world_constraint_count": counts,
            "mf_constraint_count": np.zeros(worlds, dtype=np.int32),
            "dense_phase_bounds": np.column_stack((np.minimum(counts, 3), np.minimum(counts, 6))).astype(np.int32),
            "world_dof_indices": np.arange(worlds * dofs, dtype=np.int32).reshape(worlds, dofs),
            "world_impulses": rng.uniform(0.0, 0.1, (worlds, capacity)).astype(np.float32),
            "world_row_type": row_type,
            "world_row_parent": parent,
            "world_row_mu": np.full((worlds, capacity), 0.5, dtype=np.float32),
            "world_row_cfm": np.full((worlds, capacity), 0.001, dtype=np.float32),
            "world_contact_counts": np.zeros(worlds, dtype=np.int32),
            "rhs_bias": rng.uniform(-1.0, 0.2, (worlds, capacity)).astype(np.float32),
            "ink_meta": metadata.flatten(),
            "ink_L_a": factor,
            "ink_J_a": rng.normal(0.0, 0.1, (worlds, capacity, dofs)).astype(np.float32),
            "v_out": rng.normal(0.0, 0.1, worlds * dofs).astype(np.float32),
        }
        host["v_out"][0] = -0.0
        stream = wp.Stream(device)
        with ExitStack() as patches, wp.ScopedStream(stream):
            for name in ("_INK_CHECK", "_WR_CHECK", "_WR_WARM"):
                patches.enter_context(patch.object(solver, name, False))
            patches.enter_context(patch.dict("os.environ", {"FEATHER_PGS_WR_STOP": "0"}))
            for rows in (32, 48):
                kwargs = {
                    "rows": rows,
                    "min_rows": 0 if rows == 32 else 32,
                    "sweeps": 24,
                    "matrix_free": True,
                    "inkernel_response": (18, 0, 0, 0),
                    "world_rows": True,
                    "exact_row_sums": True,
                }
                kernels = []
                for enabled in (False, True):
                    with patch.object(solver, "_REGISTER_WHITENING", enabled):
                        kernels.append(solver._get_pgs_solve_parallel_kernel(capacity, 32, dofs, device.arch, **kwargs))
                self.assertIn("_rw", kernels[1].key)
                initial = {name: wp.array(value, device=device) for name, value in host.items()}
                initial.update({"use_general_world_queue": 1, "general_world_grid_stride": 3, "omega": 0.8})
                for argument in kernels[0].adj.args:
                    if argument.label not in initial:
                        initial[argument.label] = (
                            wp.zeros((1,) * argument.type.ndim, dtype=argument.type.dtype, device=device)
                            if hasattr(argument.type, "ndim")
                            else 0.0
                            if argument.type is wp.float32
                            else 0
                        )
                copies = [
                    {name: wp.clone(value) if isinstance(value, wp.array) else value for name, value in initial.items()}
                    for _ in kernels
                ]
                for phase in range(6):
                    for iterations, friction_start, offset in ((0, 0, 0), (1, 9, 7), (24, 3, 0)):
                        for captured in (False, True):
                            with self.subTest(rows=rows, phase=phase, iterations=iterations, captured=captured):
                                for kernel, values in zip(kernels, copies, strict=True):
                                    for name, value in initial.items():
                                        if isinstance(value, wp.array):
                                            wp.copy(values[name], value, stream=stream)
                                    values.update(
                                        {
                                            "row_phase": phase,
                                            "iterations": iterations,
                                            "friction_start_iteration": friction_start,
                                            "iteration_offset": offset,
                                        }
                                    )
                                    arguments = [values[arg.label] for arg in kernel.adj.args]
                                    launch = {
                                        "dim": [3],
                                        "inputs": arguments[:-1],
                                        "outputs": arguments[-1:],
                                        "block_dim": kernel._fpgs_block_dim,
                                        "device": device,
                                        "stream": stream,
                                    }
                                    if captured:
                                        with wp.ScopedCapture(device=device, stream=stream) as capture:
                                            wp.launch_tiled(kernel, **launch)
                                        wp.capture_launch(capture.graph, stream=stream)
                                    else:
                                        wp.launch_tiled(kernel, **launch)
                                # All launches use this non-default stream; complete it before host comparisons.
                                wp.synchronize_stream(stream)
                                for name, value in initial.items():
                                    if isinstance(value, wp.array):
                                        self.assertEqual(
                                            copies[0][name].numpy().tobytes(), copies[1][name].numpy().tobytes(), name
                                        )
                                self.assertTrue(np.isfinite(copies[1]["v_out"].numpy()).all())
                                self.assertTrue(np.isfinite(copies[1]["world_impulses"].numpy()).all())


if __name__ == "__main__":
    unittest.main()
