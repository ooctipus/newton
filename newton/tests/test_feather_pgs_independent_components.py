# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise the current-row independent-component ownership boundary."""

import hashlib
import inspect
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import independent_components as components
from newton._src.solvers.feather_pgs import solver_feather_pgs as solver


def pack_endpoints(a, b):
    """Encode the original signed int16 endpoint pair without overflow casts."""
    return np.array(((int(a) & 65535) << 16) | (int(b) & 65535), dtype=np.uint32).view(np.int32).item()


def component_fixture(worlds=4, capacity=8, mf_capacity=6):
    """Build nonidentity held factors and complete dense/MF contact triples."""
    rng = np.random.default_rng(713)

    def floats(shape):
        return np.zeros(shape, np.float32)

    def integers(shape):
        return np.zeros(shape, np.int32)

    host = {
        "counts": np.full(worlds, 4, np.int32),
        "mf_counts": np.full(worlds, 3, np.int32),
        "primary_offset": np.where(np.arange(worlds) % 2, 6, 0).astype(np.int32),
        "secondary_offset": np.where(np.arange(worlds) % 2, 0, 23).astype(np.int32),
        "primary_group": np.arange(worlds - 1, -1, -1, dtype=np.int32),
        "selector": np.full(worlds, 91, np.int32),
        "J": floats((worlds, capacity, 29)),
        "Y": floats((worlds, capacity, 29)),
        "L": np.tril(rng.normal(0, 0.025, (worlds, 23, 23))).astype(np.float32),
        "rhs": floats((worlds, capacity)),
        "diag": np.ones((worlds, capacity), np.float32),
        "mu": np.full((worlds, capacity), 0.6, np.float32),
        "row_type": np.full((worlds, capacity), 3, np.int32),
        "row_parent": np.full((worlds, capacity), -1, np.int32),
        "mf_meta": integers((worlds, mf_capacity * 4)),
        "mf_mu": np.full((worlds, mf_capacity), 0.4, np.float32),
    }
    host["L"][:, np.arange(23), np.arange(23)] += 1.7
    for name in ("mf_J_a", "mf_J_b", "mf_MiJt_a", "mf_MiJt_b"):
        host[name] = floats((worlds, mf_capacity, 6))
    for world in range(worlds):
        offset, secondary = host["primary_offset"][world], host["secondary_offset"][world]
        factor = host["L"][host["primary_group"][world]].astype(np.float64)
        jacobian = rng.normal(0, 0.2, (capacity, 23)).astype(np.float32)
        host["J"][world, :, offset : offset + 23] = jacobian
        host["Y"][world, :, offset : offset + 23] = np.linalg.solve(factor @ factor.T, jacobian.T).T
        host["rhs"][world] = rng.normal(0, 0.2, capacity)
        host["row_type"][world, :3] = [0, 2, 2]
        host["row_parent"][world, :3] = [-1, 0, 0]
        for row in range(mf_capacity):
            endpoint_a, endpoint_b = (secondary, -1) if world % 2 == 0 else (-1, secondary)
            normal = row // 3 * 3
            kind, parent = (0, -1) if row % 3 == 0 else (2, normal)
            host["mf_meta"][world, row * 4 : row * 4 + 4] = [
                pack_endpoints(endpoint_a, endpoint_b),
                np.float32(0.8).view(np.int32),
                np.float32(0.1 * (row + 1)).view(np.int32),
                pack_endpoints(parent, kind),
            ]
            side = "a" if endpoint_a >= 0 else "b"
            host[f"mf_J_{side}"][world, row] = rng.normal(0, 0.2, 6)
            host[f"mf_MiJt_{side}"][world, row] = host[f"mf_J_{side}"][world, row] * 0.7
    return host


def bind_components(host, device="cpu"):
    """Bind the actual packed-two-dimensional metadata ABI on either device."""
    data = components.IndependentComponentData()
    data.world_count = len(host["counts"])
    for name, value in host.items():
        setattr(data, name, wp.array(value, dtype=wp.int32 if value.dtype == np.int32 else wp.float32, device=device))
    return data


def launch_components(data, device="cpu"):
    """Use the production worker-stride launch and original native CPU branch."""
    workers = min(data.world_count, 512)
    wp.launch_tiled(
        components.get_prepare_kernel(), dim=[workers], inputs=[workers, data], block_dim=128, device=device
    )


def solve_kernels(enabled, arch="120", *, contact_triples=False):
    """Construct exactly the two production ownership variants with eight-sweep ABI."""
    paired = solver._get_pgs_solve_paired_factor_kernel(
        8, 29, 23, 6, arch, contact_triples=contact_triples, independent_components=enabled
    )
    mixed = solver._get_pgs_solve_mf_gs_kernel(
        8,
        6,
        29,
        arch,
        factor_coordinates=True,
        has_drive_rows=False,
        has_dense_velocity_limit_rows=False,
        independent_components=enabled,
    )
    return paired, mixed


def solve_fixture(device):
    """Build disjoint hand/object, genuinely coupled fallback and no-MF worlds."""
    host = component_fixture(5)
    host["mf_counts"][3] = 0
    host["mf_counts"][1] = 4
    host["mf_meta"][1, 15] = pack_endpoints(-1, 4)
    host["counts"][4] = 0
    host["J"][2, 0, 23] = 0.35
    host["Y"][2, 0, 23] = 0.245
    host["diag"] = np.sum(host["J"] * host["Y"], axis=2) + np.float32(0.03)
    # The real fast paired path owns a position-limit prefix then contact triples.
    for name in ("J", "Y", "rhs", "diag", "mu", "row_type", "row_parent"):
        host[name][:, :4] = host[name][:, [3, 0, 1, 2]]
    host["row_parent"][:, :4] = [-1, -1, 1, 1]
    # Original paired row production already stores factor rows when MF is absent.
    for world in (3,):
        offset, group = host["primary_offset"][world], host["primary_group"][world]
        host["Y"][world, :, offset : offset + 23] = host["Y"][world, :, offset : offset + 23] @ host["L"][group]
    data = bind_components(host, device)

    def array(value):
        value = np.asarray(value)
        return wp.array(value, dtype=wp.int32 if value.dtype == np.int32 else wp.float32, device=device)

    rng = np.random.default_rng(784)
    arrays = {
        "general_world_count": array(np.array([5], np.int32)),
        "general_worlds": array(np.arange(5, dtype=np.int32)),
        "rb_class": array(np.zeros(5, np.int32)),
        "dense_phase_bounds": array(np.array([[1, 1]] * 4 + [[0, 0]], np.int32)),
        "local_solve_owner": array(np.zeros(5, np.int32)),
        "world_dof_indices": array(np.arange(145, dtype=np.int32).reshape(5, 29)),
        "world_deferred_dof_mask": array(np.zeros((5, 29), np.int32)),
        "world_impulses": array(np.full((5, 8), 0.01, np.float32)),
        "mf_impulses": array(np.full((5, 6), 0.02, np.float32)),
        "world_row_w": array(np.ones((5, 8), np.float32)),
        "mf_row_w": array(np.ones((5, 6), np.float32)),
        "mf_contact_rows_end": array(np.full(5, 3, np.int32)),
        "primary_L": data.L,
        "primary_Linv": array(np.linalg.inv(host["L"].astype(np.float64)).astype(np.float32)),
        "secondary_L": array(np.repeat((np.eye(6) / np.sqrt(0.7))[None], 5, axis=0).astype(np.float32)),
        "secondary_Linv": array(np.repeat((np.eye(6) * np.sqrt(0.7))[None], 5, axis=0).astype(np.float32)),
        "primary_group_by_world": data.primary_group,
        "secondary_group_by_world": array(np.arange(5, dtype=np.int32)),
        "primary_offset_by_world": data.primary_offset,
        "secondary_offset_by_world": data.secondary_offset,
        "v_out": array(rng.normal(0, 0.2, 145).astype(np.float32)),
    }
    for target, field in (
        ("world_constraint_count", "counts"),
        ("mf_constraint_count", "mf_counts"),
        ("world_mf_constraint_count", "mf_counts"),
        ("rhs_bias", "rhs"),
        ("world_diag", "diag"),
        ("world_row_type", "row_type"),
        ("world_row_parent", "row_parent"),
        ("world_row_mu", "mu"),
        ("J_world", "J"),
        ("Y_world", "Y"),
        ("factor_rows", "Y"),
        ("mf_row_mu", "mf_mu"),
        ("mf_meta", "mf_meta"),
        ("mf_J_a", "mf_J_a"),
        ("mf_J_b", "mf_J_b"),
        ("mf_MiJt_a", "mf_MiJt_a"),
        ("mf_MiJt_b", "mf_MiJt_b"),
    ):
        arrays[target] = getattr(data, field)
    for name in ("target_vel_bias", "vel_multiplier", "impulse_multiplier", "max_impulse", "vel_limit"):
        arrays[f"world_drive_{name}"] = array(np.zeros((1, 1), np.float32))
    return data, arrays


def launch_solve_pair(data, arrays, *, enabled, device, iterations=8, friction_start=0):
    """Use separate writable outputs but identical current data and original solve law."""
    if enabled:
        launch_components(data, device)
    values = dict(arrays)
    values.update(
        world_count=5,
        general_world_grid_stride=5,
        use_general_world_queue=0,
        iterations=iterations,
        omega=1.0,
        regularize=0,
        row_phase=0,
        friction_start_iteration=friction_start,
        iteration_offset=0,
        freeze_drive_rows=0,
        defer_dense_response=0,
    )
    if enabled:
        values["local_solve_owner"] = values["world_mf_constraint_count"] = data.selector
    for kernel, grid, block in zip(
        solve_kernels(enabled, str(device.arch), contact_triples=True), (3, 5), (64, 32), strict=True
    ):
        arguments = [values[name] for name in inspect.signature(kernel.func).parameters]
        wp.launch_tiled(kernel, dim=[grid], inputs=arguments, block_dim=block, device=device)


class TestIndependentComponents(unittest.TestCase):
    """Keep signed dispatch private and reject unsupported current operators."""

    def test_mapping_factor_action_and_worker_stride(self):
        """Prepare all 517 worlds with permuted groups and both physical block orders."""
        host = component_fixture(517)
        host["mf_counts"][1] = 0
        host["J"][2, 2, host["secondary_offset"][2]] = np.float32(1e-30)
        data = bind_components(host)
        launch_components(data)
        expected_selector = np.full(517, -1, np.int32)
        expected_selector[1:3] = [0, 3]
        np.testing.assert_array_equal(data.selector.numpy(), expected_selector)
        expected = host["Y"].copy()
        for world in np.flatnonzero(expected_selector == -1):
            offset, count = host["primary_offset"][world], host["counts"][world]
            factor = host["L"][host["primary_group"][world]].astype(np.float64)
            expected[world, :count, offset : offset + 23] = (
                host["Y"][world, :count, offset : offset + 23].astype(np.float64) @ factor
            )
        np.testing.assert_allclose(data.Y.numpy(), expected, rtol=2e-6, atol=2e-7)
        for name, before in host.items():
            if name not in ("Y", "selector"):
                np.testing.assert_array_equal(getattr(data, name).numpy(), before, err_msg=name)

    def test_reject_current_coupling_metadata_and_nonfinite(self):
        """Retain the complete original operator on each independent rejection reason."""
        cases = (
            "secondary_j",
            "secondary_y",
            "nan_j",
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
                if case in ("secondary_j", "secondary_y"):
                    host["J" if case.endswith("j") else "Y"][0, 2, 23] = np.float32(1e-30)
                elif case in ("nan_j", "nan_y"):
                    host[case[-1].upper()][0, 1, 2] = np.nan
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
                    host["mf_MiJt_b"][0, 2, 5] = np.inf
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

    def test_unsupported_primary_size(self):
        """Reject factory widths outside the explicit 23-plus-6 contract."""
        for size in (0, 6, 9, 22, 24):
            with self.subTest(size=size), self.assertRaises(ValueError):
                components.get_prepare_kernel(size)

    def test_default_source_and_unchanged_sweep_law(self):
        """Preserve accepted42 default source and each complete numerical sweep unchanged."""
        sources = []
        native = wp.func_native

        def capture(snippet, *args, **kwargs):
            sources.append(snippet)
            return native(snippet, *args, **kwargs)

        for factory in (solver._get_pgs_solve_paired_factor_kernel, solver._get_pgs_solve_mf_gs_kernel):
            factory.cache_clear()
        with patch.object(wp, "func_native", capture):
            baseline = solve_kernels(False)
            candidate = solve_kernels(True)
        self.assertEqual(len(sources), 4)
        # Captured independently from clean accepted42f1492, before this experiment.
        expected = (
            "7ce07adf4cf5273c158df70a64e59e865c92d6d1ba7fc0ed51e98fe4f7c58c93",
            "a7babcbe7a58a6e0dcbc693f7cb4d53ee1567ccf3ec62bb6119d6dfc54cc73a8",
        )
        for index in range(2):
            self.assertEqual(hashlib.sha256(sources[index].encode()).hexdigest(), expected[index])
            self.assertNotEqual(baseline[index].key, candidate[index].key)
            self.assertIn("split_component", sources[index + 2])
            begin = "for (int iter = 0; iter < iterations;"
            end = "if (global_dof >= 0) v_out.data[global_dof] ="
            original_sweep = sources[index][sources[index].index(begin) : sources[index].index(end)]
            changed_sweep = sources[index + 2][sources[index + 2].index(begin) :]
            self.assertTrue(changed_sweep.startswith(original_sweep))

    def test_native_fixture_binds_actual_kernel_signatures(self):
        """Compile both original and selected CUDA-test ABIs on CPU before device gating."""
        device = wp.get_device("cpu")
        for enabled in (False, True):
            data, arrays = solve_fixture(device)
            launch_solve_pair(data, arrays, enabled=enabled, device=device)
            if enabled:
                np.testing.assert_array_equal(data.selector.numpy(), [-1, -1, 3, 0, -1])


class TestIndependentComponentsCUDA(unittest.TestCase):
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

    def test_same_input_original_eight_sweeps_and_component_publication(self):
        """Compare native mixed fallback against the split owners, including delayed friction."""
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("original GS numerical control requires CUDA")
        device = devices[0]
        for iterations, friction_start in ((1, 0), (8, 0), (8, 2)):
            with self.subTest(iterations=iterations, friction_start=friction_start):
                baseline, old = solve_fixture(device)
                candidate, new = solve_fixture(device)
                immutable = {name: getattr(candidate, name).numpy() for name in ("counts", "mf_counts", "J", "mf_meta")}
                launch_solve_pair(
                    baseline, old, enabled=False, device=device, iterations=iterations, friction_start=friction_start
                )
                launch_solve_pair(
                    candidate, new, enabled=True, device=device, iterations=iterations, friction_start=friction_start
                )
                np.testing.assert_array_equal(candidate.selector.numpy(), [-1, -1, 3, 0, -1])
                for name in ("v_out", "world_impulses", "mf_impulses"):
                    np.testing.assert_allclose(new[name].numpy(), old[name].numpy(), rtol=2e-5, atol=3e-6, err_msg=name)
                for name, before in immutable.items():
                    np.testing.assert_array_equal(getattr(candidate, name).numpy(), before)


if __name__ == "__main__":
    unittest.main()
