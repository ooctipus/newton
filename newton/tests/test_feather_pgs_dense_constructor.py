# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise actual dense-kernel selection through a small real solver constructor."""

import os
import unittest
from unittest.mock import patch

import warp as wp

import newton


def _small_worlds(device):
    """Create three independent one-joint worlds with ordinary finite body inertia."""
    template = newton.ModelBuilder(up_axis=newton.Axis.Z)
    body = template.add_link()
    template.add_shape_box(body, hx=0.1, hy=0.05, hz=0.05)
    joint = template.add_joint_revolute(parent=-1, child=body, axis=wp.vec3(0.0, 0.0, 1.0))
    template.add_articulation([joint])
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.replicate(template, world_count=3)
    return builder.finalize(device=device)


class TestFeatherPGSDenseConstructor(unittest.TestCase):
    """Keep the register-state feature behind its actual shape and flag guards."""

    def _check_device(self, device):
        model = _small_worlds(device)
        self.assertEqual(model.articulation_count, 3)
        cases = (
            # budget flag, register flag, capacity, dense kernel
            (None, None, 64, "tiled_row"),
            ("0", "0", 64, "tiled_row"),
            ("0", "1", 64, "tiled_row"),
            ("1", "0", 64, "tiled_row"),
            ("1", "1", 64, "tiled_row"),
            ("1", "1", 32, "tiled_row"),
            ("1", "1", 96, "tiled_row"),
            ("1", "1", 64, "loop"),
            ("1", "1", 64, "tiled_contact"),
            ("1", "1", 64, "streaming"),
        )
        for budgets, registers, capacity, mode in cases:
            with self.subTest(
                device=str(device),
                budgets=budgets,
                registers=registers,
                capacity=capacity,
                mode=mode,
            ):
                with patch.dict(os.environ):
                    for name, value in (
                        ("FEATHER_PGS_DENSE_ROW_BUDGETS", budgets),
                        ("FEATHER_PGS_DENSE_ROW_REGISTERS", registers),
                    ):
                        if value is None:
                            os.environ.pop(name, None)
                        else:
                            os.environ[name] = value
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
                self.assertEqual(solver.world_count, 3)
                selected = solver._pgs_solve_tiled_row_budgets
                registered = getattr(solver, "_pgs_solve_tiled_row_register_kernels", ())
                eligible = model.device.is_cuda and mode == "tiled_row" and capacity == 64 and budgets == "1"
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
                    self.assertEqual(registered, ())
                elif registers == "1":
                    self.assertEqual(selected, registered)
                    self.assertEqual(
                        [kernel.key for kernel in selected],
                        [
                            "pgs_solve_dense_row_state_b32_w2",
                            "pgs_solve_dense_row_state_b64_w2",
                        ],
                    )
                else:
                    self.assertEqual(registered, ())
                    self.assertEqual(len(selected), 2)
                    self.assertNotEqual(selected[0].key, selected[1].key)
                    self.assertTrue(all("dense_row_state" not in kernel.key for kernel in selected))
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

    def test_cpu_constructor_retains_legacy_routes(self):
        """Keep every CPU flag combination on the real constructor's legacy route."""
        self._check_device(wp.get_device("cpu"))

    def test_cuda_constructor_selects_only_eligible_layouts(self):
        """Select both register budgets only for CUDA tiled-row C64 with both flags."""
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("CUDA constructor selection requires a CUDA device")
        self._check_device(devices[0])


if __name__ == "__main__":
    unittest.main()
