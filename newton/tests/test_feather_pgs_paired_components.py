# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise the paired articulation/free-body component split and its qualification."""

import inspect
import os
import sys
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp
from warp._src.context import pack_arg

from newton._src.solvers.feather_pgs import paired_components, sweep_routes, sweep_routes_local
from newton.tests.feather_pgs_paired_fixture import (
    bind_components,
    component_fixture,
    launch_components,
    launch_solve_pair,
    pack_endpoints,
    solve_fixture,
    solve_kernels,
    solve_pair_launches,
)


def _compile_offline(kernel):
    """Compile ``kernel`` for sm_120 at 64 threads without a GPU (the repository's ``tools/fpgs_bench``: NVRTC and
    ptxas) and return its resources, or ``None`` where the tools are not at hand."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if not os.path.isdir(os.path.join(root, "tools", "fpgs_bench")):
        return None
    sys.path.insert(0, root)
    try:
        from tools.fpgs_bench import kernel_sass  # noqa: PLC0415
    finally:
        sys.path.remove(root)
    return kernel_sass.compile_kernel(kernel, arch=120, block_dim=64)


class TestPairedComponents(unittest.TestCase):
    """Keep signed dispatch private and reject unsupported current operators."""

    def test_mapping_factor_action_and_worker_stride(self):
        """Prepare all 517 worlds with permuted groups and both physical block orders."""
        host = component_fixture(517)
        host["mf_counts"][1] = 0
        host["Y"][2, 2, host["secondary_offset"][2]] = np.float32(1e-30)
        data = bind_components(host)
        launch_components(data)
        expected_selector = np.full(517, -1, np.int32)
        expected_selector[1:3] = [0, 3]
        np.testing.assert_array_equal(data.selector.numpy(), expected_selector)
        # The rows are already whitened (packed at their world coordinates): the qualification reads them only.
        dense = host["mu"].shape[1]
        for name, before in host.items():
            if name in ("selector", "J", "L", "primary_group"):
                continue
            # The row store's records: the dense rows' from row 0, the free bodies' coefficients from the dense capacity.
            if name == "mf_mu":
                actual = data.mu.numpy()[:, dense:]
            elif name in ("rhs", "diag", "mu", "row_type", "row_parent"):
                actual = getattr(data, name).numpy()[:, :dense]
            else:
                actual = getattr(data, name).numpy()
            np.testing.assert_array_equal(actual, before, err_msg=name)

    def test_reject_current_coupling_metadata_and_nonfinite(self):
        """Retain the complete mixed sweep on each independent rejection reason."""
        cases = (
            "secondary_y",
            "nan_y",
            "nan_factor",
            "large_factor",
            "large_y",
            "dense_type",
            "dense_parent",
            "dense_diag",
            "dense_rhs",
            "dense_mu",
            "mf_endpoint",
            "mf_absent",
            "mf_parent",
            "mf_type",
            "mf_diag",
            "mf_rhs",
            "mf_mu",
            "mf_response",
            "offset",
            "group",
            "negative_count",
            "excess_count",
            "negative_mf",
            "excess_mf",
        )
        for case in cases:
            with self.subTest(case=case):
                host = component_fixture(1)
                if case == "secondary_y":
                    host["Y"][0, 2, 23] = np.float32(1e-30)
                elif case == "nan_y":
                    host["Y"][0, 1, 2] = np.nan
                elif case in ("nan_factor", "large_factor"):
                    host["L"][0, 1, 1] = np.nan if case.startswith("nan") else 1e20
                elif case == "large_y":
                    host["Y"][0, 1, 1] = 1e20
                elif case == "dense_type":
                    host["row_type"][0, 3] = 4
                elif case == "dense_parent":
                    host["row_parent"][0, 2] = 1
                elif case in ("dense_diag", "dense_rhs", "dense_mu"):
                    name = {"dense_diag": "diag", "dense_rhs": "rhs", "dense_mu": "mu"}[case]
                    host[name][0, 0] = np.nan if name == "rhs" else -1
                elif case in ("mf_endpoint", "mf_absent"):
                    host["mf_meta"][0, 0] = pack_endpoints(0 if case == "mf_endpoint" else -1, -1)
                elif case == "mf_parent":
                    host["mf_meta"][0, 11] = pack_endpoints(1, 2)
                elif case == "mf_type":
                    host["mf_meta"][0, 3] = pack_endpoints(-1, 3)
                elif case in ("mf_diag", "mf_rhs"):
                    host["mf_meta"][0, 1 if case == "mf_diag" else 2] = np.float32(np.nan).view(np.int32)
                elif case == "mf_mu":
                    host["mf_mu"][0, 1] = -1
                elif case == "mf_response":
                    host["mf_Z"][0, 2, 11] = np.inf
                elif case == "offset":
                    host["primary_offset"][0] = 1
                elif case == "group":
                    host["primary_group"][0] = 1
                else:
                    field = "mf_counts" if case.endswith("mf") else "counts"
                    host[field][0] = -1 if case.startswith("negative") else 9
                data = bind_components(host)
                launch_components(data)
                self.assertGreater(int(data.selector.numpy()[0]), 0)
                np.testing.assert_array_equal(data.Y.numpy(), host["Y"])
                np.testing.assert_array_equal(data.counts.numpy(), host["counts"])
                np.testing.assert_array_equal(data.mf_counts.numpy(), host["mf_counts"])

    def test_current_empty_growing_shrinking_and_poisoned_tail(self):
        """Reset dispatch each call and read only each newly authoritative active prefix."""
        host = component_fixture(4)
        data = bind_components(host)
        for counts, mf_counts in (([0, 4, 4, 4], [3, 0, 3, 3]), ([8, 4, 4, 0], [6, 3, 6, 0]), ([0] * 4, [0] * 4)):
            current = host["Y"].copy()
            for world, count in enumerate(counts):
                current[world, count:] = np.nan
            data.Y.assign(current)
            data.counts.assign(np.asarray(counts, np.int32))
            data.mf_counts.assign(np.asarray(mf_counts, np.int32))
            launch_components(data)
            np.testing.assert_array_equal(data.selector.numpy(), np.where(np.asarray(mf_counts) > 0, -1, 0))
            result = data.Y.numpy()
            for world, count in enumerate(counts):
                np.testing.assert_array_equal(result[world, count:], current[world, count:])
                self.assertTrue(np.all(np.isfinite(result[world, :count])))

    def test_sizes_come_from_the_pair(self):
        """Any articulation of at most 32 coordinates with one free body; nothing else."""
        for primary in (7, 9, 17, 23, 32):
            kernel = paired_components.paired_components_kernel(primary)
            native = inspect.getclosurevars(kernel.func)
            source = (
                native.nonlocals.get("paired_components_native") or native.globals["paired_components_native"]
            ).native_snippet
            self.assertIn(f"r * {primary + 6} + d", source)
        for primary, secondary in ((0, 6), (33, 6), (23, 3), (23, 7)):
            with self.subTest(primary=primary, secondary=secondary), self.assertRaises(ValueError):
                paired_components.paired_components_kernel(primary, secondary)

    def test_split_sources_keep_the_complete_sweep(self):
        """The component variants add only the selector gate around each unchanged sweep."""
        sources = []
        native = wp.func_native

        def capture(snippet, *args, **kwargs):
            sources.append(snippet)
            return native(snippet, *args, **kwargs)

        for factory in (sweep_routes_local.paired_factor_kernel, sweep_routes.mf_gs_kernel):
            factory.cache_clear()
        with patch.object(wp, "func_native", capture):
            baseline = solve_kernels(False)
            candidate = solve_kernels(True)
        self.assertEqual(len(sources), 4)
        for index in range(2):
            self.assertNotEqual(baseline[index].key, candidate[index].key)
            self.assertNotIn("split_component", sources[index])
            self.assertIn("split_component", sources[index + 2])
            begin = "for (int iter = 0; iter < iterations;"
            end = "if (global_dof >= 0) walk.du.data[global_dof] +="
            original_sweep = sources[index][sources[index].index(begin) : sources[index].index(end)]
            changed_sweep = sources[index + 2][sources[index + 2].index(begin) :]
            self.assertTrue(changed_sweep.startswith(original_sweep))

    def test_native_fixture_binds_actual_kernel_signatures(self):
        """Bind the fixture's arrays to the plain and split sweeps' signatures as a launch does, and compile the
        sweeps offline for sm_120, without a GPU: their bodies are CUDA-only (a CPU build fails)."""
        device = wp.get_device("cpu")
        for enabled in (False, True):
            data, arrays = solve_fixture(device)
            if enabled:
                launch_components(data, device)
                np.testing.assert_array_equal(data.selector.numpy(), [-1, -1, 3, 0, -1])
            for kernel, _, _, arguments in solve_pair_launches(data, arrays, enabled=enabled, arch="120"):
                with self.subTest(enabled=enabled, kernel=kernel.key):
                    self.assertEqual(len(arguments), len(kernel.adj.args))
                    for arg, value in zip(kernel.adj.args, arguments, strict=True):
                        pack_arg(kernel, arg.type, arg.label, value, device)
                    compiled = _compile_offline(kernel)
                    if compiled is not None:
                        self.assertEqual(compiled["local"], 0)


class TestPairedComponentsCUDA(unittest.TestCase):
    """Run the actual multiwarp classifier and its captured state transitions."""

    def test_current_graph_transition_and_native_cpu_equivalence(self):
        """Compare 517-world CUDA publication with CPU through grow/shrink graph replay."""
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("CUDA graph control requires a CUDA device")
        device = devices[0]
        host = component_fixture(517)
        cpu, gpu = bind_components(host), bind_components(host, device)
        launch_components(cpu)
        launch_components(gpu, device)
        np.testing.assert_array_equal(gpu.selector.numpy(), cpu.selector.numpy())
        np.testing.assert_allclose(gpu.Y.numpy(), cpu.Y.numpy(), rtol=2e-6, atol=2e-7)
        with wp.ScopedCapture(device=device) as capture:
            launch_components(gpu, device)
        for epoch in range(3):
            counts, mf_counts = host["counts"].copy(), host["mf_counts"].copy()
            counts[epoch::3] = 0
            mf_counts[(epoch + 1) :: 3] = 0
            current = host["Y"].copy()
            current[2, 2, host["secondary_offset"][2]] = 1 if epoch == 1 else 0
            for data in (cpu, gpu):
                data.counts.assign(counts)
                data.mf_counts.assign(mf_counts)
                data.Y.assign(current)
            launch_components(cpu)
            wp.capture_launch(capture.graph)
            np.testing.assert_array_equal(gpu.selector.numpy(), cpu.selector.numpy())
            np.testing.assert_allclose(gpu.Y.numpy(), cpu.Y.numpy(), rtol=2e-6, atol=2e-7)
            np.testing.assert_array_equal(gpu.counts.numpy(), counts)
            np.testing.assert_array_equal(gpu.mf_counts.numpy(), mf_counts)

    def test_same_input_mixed_sweeps_and_component_publication(self):
        """Compare the mixed fallback against the split owners."""
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("the sweep control requires CUDA")
        device = devices[0]
        for iterations in (1, 8):
            with self.subTest(iterations=iterations):
                baseline, old = solve_fixture(device)
                candidate, new = solve_fixture(device)
                immutable = {name: getattr(candidate, name).numpy() for name in ("counts", "mf_counts", "Y", "mf_meta")}
                launch_solve_pair(baseline, old, enabled=False, device=device, iterations=iterations)
                launch_solve_pair(candidate, new, enabled=True, device=device, iterations=iterations)
                np.testing.assert_array_equal(candidate.selector.numpy(), [-1, -1, 3, 0, -1])
                # The velocity change both sweeps leave in the walk's coordinates, and the impulses.
                for name in ("du", "world_impulses"):
                    np.testing.assert_allclose(new[name].numpy(), old[name].numpy(), rtol=2e-5, atol=3e-6, err_msg=name)
                for name, before in immutable.items():
                    np.testing.assert_array_equal(getattr(candidate, name).numpy(), before)


if __name__ == "__main__":
    unittest.main()
