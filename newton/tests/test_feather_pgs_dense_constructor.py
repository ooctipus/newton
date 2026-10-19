# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise actual dense-kernel selection through a small real solver constructor."""

import os
import unittest
from unittest.mock import patch

import warp as wp

import newton


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
    """Keep the dense row budgets behind their actual shape and flag guards."""

    def _check_device(self, device, world_count=3, cases=None):
        model = _small_worlds(device, world_count)
        self.assertEqual(model.articulation_count, world_count)
        cases = cases or (
            # budget flag, capacity, dense kernel
            (None, 64, "tiled_row"),
            ("0", 64, "tiled_row"),
            ("1", 64, "tiled_row"),
            ("1", 32, "tiled_row"),
            ("1", 96, "tiled_row"),
            (None, 128, "tiled_row"),
            ("0", 128, "tiled_row"),
            ("1", 128, "tiled_row"),
            ("1", 64, "loop"),
            ("1", 64, "tiled_contact"),
            ("1", 64, "streaming"),
        )
        for budgets, capacity, mode in cases:
            with self.subTest(device=str(device), budgets=budgets, capacity=capacity, mode=mode):
                with patch.dict(os.environ):
                    if budgets is None:
                        os.environ.pop("FEATHER_PGS_DENSE_ROW_BUDGETS", None)
                    else:
                        os.environ["FEATHER_PGS_DENSE_ROW_BUDGETS"] = budgets
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
                        pgs_chunk_size=16,
                        use_parallel_streams=False,
                        double_buffer=False,
                    )
                # No fabricated device, mocked factory, patched constructor, or
                # synthetic solver shell participates in these assertions.
                self.assertEqual(solver.world_count, world_count)
                self.assertEqual(solver.dense_max_constraints, capacity)
                self.assertEqual(solver.C.shape, (world_count, capacity, capacity))
                selected = solver._pgs_solve_tiled_row_budgets
                eligible = (
                    model.device.is_cuda
                    and mode == "tiled_row"
                    and capacity in (64, 128)
                    and (budgets == "1" or (budgets is None and capacity == 128 and world_count >= 1024))
                )
                resolved_mode = mode if model.device.is_cuda else "loop"
                self.assertEqual(solver.pgs_kernel, resolved_mode)
                if resolved_mode == "tiled_row":
                    self.assertEqual(
                        solver._pgs_solve_tiled_row_kernel.key,
                        f"pgs_solve_tiled_row_{capacity}",
                    )
                else:
                    self.assertIsNone(solver._pgs_solve_tiled_row_kernel)
                if not eligible:
                    self.assertEqual(selected, ())
                else:
                    self.assertEqual(
                        [kernel.key for kernel in selected],
                        [f"pgs_solve_tiled_row_{capacity}_b32_m0", f"pgs_solve_tiled_row_{capacity}_b{capacity}_m32"],
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

    def test_cuda_constructor_large_capacity_default(self):
        """Enable large C128 batches by default while preserving explicit opt-out."""
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("CUDA constructor selection requires a CUDA device")
        for world_count in (1023, 1024):
            self._check_device(
                devices[0],
                world_count=world_count,
                cases=(
                    (None, 128, "tiled_row"),
                    ("0", 128, "tiled_row"),
                    ("1", 128, "tiled_row"),
                    (None, 64, "tiled_row"),
                ),
            )


if __name__ == "__main__":
    unittest.main()
