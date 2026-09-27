# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check capacity-safe storage selection without changing the MF-GS schedule."""

import inspect
import re
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import solver_feather_pgs as fpgs
from newton.tests.test_feather_pgs_independent_components import solve_fixture


class TestFeatherPGSMFStorage(unittest.TestCase):
    def test_hardware_capacity_streams_only_when_required(self):
        """Preserve all hardware rows when resident impulses exceed the device limit."""
        options = {"has_drive_rows": False, "fuse_vel_limits": False}
        self.assertEqual(fpgs._select_mfgs_storage(256, 37248, 1551, 101376, **options), (False, True))
        self.assertEqual(fpgs._select_mfgs_storage(256, 37248, 1551, 232448, **options), (True, False))
        self.assertEqual(fpgs._select_mfgs_storage(256, 15984, 3135, 101376, **options), (True, False))

    def test_storage_boundary_and_remaining_state(self):
        """Include tile storage and reject an unrepresentable velocity working set."""
        options = {"has_drive_rows": False, "fuse_vel_limits": False}
        # Four streamed metadata arrays leave velocities and two impulse arrays.
        required = 128 + 4 * (29 + 1024 + 4096)
        self.assertEqual(fpgs._select_mfgs_storage(1024, 4096, 29, required, **options), (False, False))
        self.assertEqual(fpgs._select_mfgs_storage(1024, 4096, 29, required - 1, **options), (False, True))
        with self.assertRaisesRegex(ValueError, "MF-GS.*shared memory"):
            fpgs._select_mfgs_storage(256, 37248, 26000, 101376, **options)

    def test_streaming_factory_owns_global_impulses_on_one_lane(self):
        """Stream the uniform paired projection; reject single-lane nonlinear paths."""
        for friction in ("current", "bisection", "bisection_desaxce", "coulomb_newton"):
            for shared_metadata in (False, True):
                with self.subTest(friction=friction, shared_metadata=shared_metadata):
                    if friction != "current":
                        with self.assertRaisesRegex(ValueError, "warp-uniform current-friction"):
                            fpgs._get_pgs_solve_mf_gs_kernel(
                                16,
                                64,
                                12,
                                "cpu",
                                friction_mode=friction,
                                shared_metadata=shared_metadata,
                                stream_mf_impulses=True,
                            )
                        continue
                    kernel = fpgs._get_pgs_solve_mf_gs_kernel(
                        16,
                        64,
                        12,
                        "cpu",
                        friction_mode=friction,
                        shared_metadata=shared_metadata,
                        stream_mf_impulses=True,
                    )
                    native = inspect.getclosurevars(kernel.func).nonlocals["pgs_solve_mf_gs_native"]
                    source = native.native_snippet
                    self.assertIn("_glambda", kernel.key)
                    self.assertNotIn("__shared__ float s_lam_mf", source)
                    self.assertNotIn("mf_impulses.data[off_mf + i] =", source)
                    self.assertIn("__shfl_sync(MASK, lane == 0 ? g_lam_mf[i] : 0.0f, 0)", source)
                    for line in source.splitlines():
                        if re.search(r"g_lam_mf\[[^]]+\]\s*=", line):
                            self.assertIn("if (lane == 0)", line)

    def test_limit_only_phase_does_not_stage_contact_prefix(self):
        """Keep contacts untouched and retain work for empty-segment fused clamps."""
        for fused in (False, True):
            kernel = fpgs._get_pgs_solve_mf_gs_kernel(16, 4096, 128, "cpu", fuse_vel_limits=fused)
            native = inspect.getclosurevars(kernel.func).nonlocals["pgs_solve_mf_gs_native"]
            source = native.native_snippet
            self.assertIn("int mf_load_lo = (row_phase == 5) ? mf_contact_end : 0;", source)
            self.assertEqual(source.count("for (int i = mf_load_lo + lane; i < m_mf; i += 32)"), 2)
            guard = "if (row_phase == 5 && dense_lo == dense_hi && mf_contact_end == m_mf) return;"
            if fused:
                self.assertNotIn(guard, source)
            else:
                self.assertIn(guard, source)
                self.assertLess(source.index(guard), source.index("// LOAD PHASE"))


@unittest.skipUnless(wp.is_cuda_available(), "MF-GS native solve requires CUDA")
class TestFeatherPGSMFStorageCUDA(unittest.TestCase):
    def test_streamed_impulses_match_resident_with_friction_and_limits(self):
        """Match mixed contact triples, warm impulses, limits and graph replay exactly."""
        device = wp.get_device()
        if not device.is_cuda:
            self.skipTest("Select a CUDA device for native MF-GS execution")
        _, arrays = solve_fixture(device)
        values = dict(arrays)
        values.update(
            general_world_grid_stride=5,
            use_general_world_queue=0,
            iterations=8,
            omega=1.0,
            regularize=0,
            row_phase=0,
            friction_start_iteration=2,
            iteration_offset=0,
            freeze_drive_rows=0,
            defer_dense_response=0,
        )
        outputs = ("v_out", "world_impulses", "mf_impulses")
        for friction in ("current",):
            for shared_metadata in (False, True):
                with self.subTest(friction=friction, shared_metadata=shared_metadata):
                    variants = []
                    for stream in (False, True):
                        kernel = fpgs._get_pgs_solve_mf_gs_kernel(
                            8,
                            6,
                            29,
                            str(device.arch),
                            friction_mode=friction,
                            has_drive_rows=False,
                            has_dense_velocity_limit_rows=False,
                            shared_metadata=shared_metadata,
                            stream_mf_impulses=stream,
                        )
                        bound = dict(values)
                        bound.update({name: wp.clone(values[name]) for name in outputs})
                        variants.append((kernel, bound))

                    def run(variants=variants):
                        for kernel, bound in variants:
                            for name in outputs:
                                wp.copy(bound[name], values[name])
                            wp.launch_tiled(
                                kernel,
                                dim=[5],
                                block_dim=32,
                                device=device,
                                inputs=[bound[name] for name in inspect.signature(kernel.func).parameters],
                            )

                    for phase in (0, 1, 2, 3, 4, 5):
                        for _, bound in variants:
                            bound["row_phase"] = phase
                        run()
                        for name in outputs:
                            np.testing.assert_array_equal(variants[0][1][name].numpy(), variants[1][1][name].numpy())
                    for _, bound in variants:
                        bound["row_phase"] = 0
                    with wp.ScopedCapture(device=device) as capture:
                        run()
                    wp.capture_launch(capture.graph)
                    for name in outputs:
                        np.testing.assert_array_equal(variants[0][1][name].numpy(), variants[1][1][name].numpy())


if __name__ == "__main__":
    unittest.main()
