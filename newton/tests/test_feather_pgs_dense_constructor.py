# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise actual dense-kernel selection through a small real solver constructor."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.staging import arch_budget
from newton._src.solvers.feather_pgs.sweep_routes_diag import tiled_row_source


def _small_worlds(device, world_count=3):
    """Create independent one-joint worlds with ordinary finite body inertia."""
    template = newton.ModelBuilder(up_axis=newton.Axis.Z)
    body = template.add_link()
    template.add_shape_box(body, hx=0.1, hy=0.05, hz=0.05)
    joint = template.add_joint_revolute(parent=-1, child=body, axis=wp.vec3(0.0, 0.0, 1.0))
    template.add_articulation([joint])
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.replicate(template, world_count=world_count)
    return builder.finalize(device=device)


class TestFeatherPGSDenseConstructor(unittest.TestCase):
    """Keep the dense row budgets behind their shape rule: 128 rows and 1024 or more worlds."""

    def _check_device(self, device, world_count=3, cases=None):
        model = _small_worlds(device, world_count)
        self.assertEqual(model.articulation_count, world_count)
        cases = cases or (
            # capacity, dense kernel
            (64, "tiled_row"),
            (32, "tiled_row"),
            (96, "tiled_row"),
            (128, "tiled_row"),
            (64, "loop"),
        )
        for capacity, mode in cases:
            with self.subTest(device=str(device), capacity=capacity, mode=mode):
                if model.device.is_cpu and mode != "loop":
                    with self.assertRaisesRegex(ValueError, "pgs_kernel.*requires CUDA"):
                        newton.solvers.SolverFeatherPGS(model, pgs_kernel=mode)
                    continue
                solver = newton.solvers.SolverFeatherPGS(
                    model,
                    pgs_mode="split",
                    pgs_kernel=mode,
                    dense_max_constraints=capacity,
                    mf_max_constraints=8,
                    use_parallel_streams=False,
                    double_buffer=False,
                )
                # No fabricated device, mocked factory, patched constructor, or
                # synthetic solver shell participates in these assertions.
                self.assertEqual(solver.world_count, world_count)
                self.assertEqual(solver.dense_max_constraints, capacity)
                self.assertEqual(solver.C.shape, (world_count, capacity, capacity))
                selected = solver._pgs_solve_tiled_row_budgets
                eligible = model.device.is_cuda and mode == "tiled_row" and capacity == 128 and world_count >= 1024
                resolved_mode = mode if model.device.is_cuda else "loop"
                self.assertEqual(solver.pgs_kernel, resolved_mode)
                if resolved_mode == "tiled_row":
                    self.assertEqual(
                        solver._pgs_solve_tiled_row_kernel.key,
                        f"sweep_tiled_row_{capacity}",
                    )
                else:
                    self.assertIsNone(solver._pgs_solve_tiled_row_kernel)
                if not eligible:
                    self.assertEqual(selected, ())
                else:
                    self.assertEqual(
                        [kernel.key for kernel in selected],
                        [f"sweep_tiled_row_{capacity}_b32_m0", f"sweep_tiled_row_{capacity}_b{capacity}_m32"],
                    )
                if selected:
                    original_labels = [argument.label for argument in solver._pgs_solve_tiled_row_kernel.adj.args]
                    for kernel in selected:
                        self.assertEqual(
                            [argument.label for argument in kernel.adj.args],
                            original_labels,
                        )
                # Destroy the owning solver before its model/device scope exits.
                # No streams are borrowed from another solver or shadow object.
                del solver

    def test_cpu_constructor_rejects_forced_cuda_routes(self):
        """Reject forced CUDA routes and retain the explicitly selected scalar route."""
        self._check_device(wp.get_device("cpu"))

    def test_cuda_constructor_selects_only_eligible_layouts(self):
        """Select explicit CUDA budgets only for 64- and 128-row tiled_row capacities."""
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("CUDA constructor selection requires a CUDA device")
        self._check_device(devices[0])

    def test_cuda_constructor_stages_rows_past_shared_memory(self):
        """Read Delassus rows from global memory where their packed triangle exceeds a block's shared memory.

        The capacity is the smallest multiple of 32 rows whose triangle exceeds the device's shared memory per
        block; a step loads and runs the kernel.
        """
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("CUDA constructor selection requires a CUDA device")
        device = devices[0]
        limit = arch_budget(device.arch)
        capacity = 64
        while tiled_row_source(capacity, staging="shared")[1].bytes <= limit:
            capacity += 32
        model = _small_worlds(device)
        solver = newton.solvers.SolverFeatherPGS(
            model,
            pgs_mode="split",
            pgs_kernel="tiled_row",
            dense_max_constraints=capacity,
            mf_max_constraints=8,
            use_parallel_streams=False,
            double_buffer=False,
        )
        dense = solver.get_path_selection()["routes"]["sweep"]["dense"]
        self.assertEqual(dense["kernels"], [f"sweep_tiled_row_{capacity}_global"])
        self.assertEqual(dense["staging"], "global")
        self.assertIn(str(limit), dense["staging_reason"])
        state_0, state_1 = model.state(), model.state()
        solver.step(state_0, state_1, model.control(), None, 1.0 / 60.0)
        self.assertTrue(np.isfinite(state_1.joint_qd.numpy()).all())
        del solver

    def test_cuda_constructor_large_capacity_default(self):
        """Enable the row budgets for C128 batches of 1024 worlds or more."""
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("CUDA constructor selection requires a CUDA device")
        for world_count in (1023, 1024):
            self._check_device(
                devices[0],
                world_count=world_count,
                cases=((128, "tiled_row"), (64, "tiled_row")),
            )


if __name__ == "__main__":
    unittest.main()
