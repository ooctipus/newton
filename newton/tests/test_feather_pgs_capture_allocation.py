# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Allocate the inputs a captured step reads before the capture."""

import contextlib
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import solver_feather_pgs as source
from newton._src.solvers.feather_pgs.solver_feather_pgs import SolverFeatherPGS
from newton.tests.feather_pgs_patch_parity import standing_model
from newton.tests.test_feather_pgs_parallel_patch import _matrix_free_solver

DT = 1.0 / 240.0
STATE_FIELDS = ("joint_q", "joint_qd", "body_q", "body_qd")

# Module switches and environment of each tier configuration: the default point tiers, and the
# in-kernel response with in-sweep world rows (keyed rows declined so the general tier builds them).
_CONFIGURATIONS = {
    "default": ({}, {}),
    "in_kernel_rows": ({"_INK_ON": True, "_WR_ON": True}, {"FEATHER_PGS_KEYED_ROWS": "0"}),
}


def _tier_solver(model, name):
    switches, environment = _CONFIGURATIONS[name]
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, environment))
        for switch, value in switches.items():
            stack.enter_context(patch.object(source, switch, value))
        return _matrix_free_solver(model, beta=0.0, parallel_rows=32)


class _Scene:
    def __init__(self, model, solver):
        self.model, self.solver = model, solver
        self.state, self.out, self.control = model.state(), model.state(), model.control()
        newton.eval_fk(model, self.state.joint_q, self.state.joint_qd, self.state)
        self.pipeline = newton.CollisionPipeline(model, rigid_contact_max=64, deterministic=True)
        self.contacts = self.pipeline.contacts()

    def step(self):
        self.state.clear_forces()
        self.pipeline.collide(self.state, self.contacts)
        self.solver.step(self.state, self.out, self.control, self.contacts, DT)


class TestFeatherPGSTierPlaceholders(unittest.TestCase):
    def test_tier_inputs_read_placeholders_allocated_up_front(self):
        """Read the tiers' placeholder inputs without allocating, and leave only the warm-start table to first use.

        ``_allocate_tier_placeholders`` runs with the tiers' construction. Afterwards the input
        helpers allocate nothing, except the per-world contact table of the in-kernel warm start
        (``FEATHER_PGS_WR_WARM`` with world rows), which is tier state rather than a placeholder.
        """
        device = wp.get_device("cpu")
        for warm, world_rows in ((False, False), (False, True), (True, False), (True, True)):
            with self.subTest(warm=warm, world_rows=world_rows):
                solver = SimpleNamespace(
                    model=SimpleNamespace(device=device),
                    world_count=3,
                    pgs_warmstart_decay=0.5,
                    row_cfm=wp.zeros((3, 4), dtype=float, device=device),
                    _ink_sizes=None,
                    _wr_world_contacts=wp.zeros((3, 4), dtype=int, device=device) if world_rows else None,
                )
                solver._warm_table_inputs = lambda solver=solver: SolverFeatherPGS._warm_table_inputs(solver)
                with patch.object(source, "_WR_WARM", warm):
                    SolverFeatherPGS._allocate_tier_placeholders(solver)
                    allocated, zeros = [], wp.zeros

                    def allocate(*args, allocated=allocated, zeros=zeros, **kwargs):
                        array = zeros(*args, **kwargs)
                        allocated.append(array.shape)
                        return array

                    with patch.object(wp, "zeros", allocate):
                        ink = SolverFeatherPGS._inkernel_response_inputs(solver)
                        rows = SolverFeatherPGS._world_rows_inputs(solver)
                        again = SolverFeatherPGS._world_rows_inputs(solver)
                state = warm and world_rows
                # The warm-start table: count, two shape tables, points and impulses.
                self.assertEqual(allocated, [(3,), (3, 64), (3, 64), (3, 64), (3, 64)] if state else [])
                self.assertEqual(ink[0].shape, (4,))
                self.assertTrue(all(array is solver._ink_dummy3 for array in ink[1:5]))
                table = rows[-6:-1]
                self.assertTrue(all(a is b for a, b in zip(table, again[-6:-1], strict=True)))
                self.assertEqual([array.shape[0] for array in table], [3, 3, 3, 3, 3] if state else [1, 1, 1, 1, 1])
                self.assertEqual(rows[-1], 0.5)
                for value in rows[:-6]:
                    if isinstance(value, wp.array):
                        self.assertTrue(any(value is placeholder for placeholder in solver._wr_dummies.values()))


@unittest.skipUnless(wp.is_cuda_available(), "Parallel matrix-free tiers require CUDA")
class TestFeatherPGSCaptureAllocation(unittest.TestCase):
    def test_tier_placeholders_precede_a_captured_first_step(self):
        """Give a first step captured right after construction tier inputs that existed before the capture.

        The tiers take placeholder arrays for the features they were built without. Allocated at
        their first launch, inside the capture, each ``wp.zeros`` would be a memset node that every
        replay runs again. The captured step then matches the same first step taken eagerly.
        """
        device = wp.get_device("cuda:0")
        for name in _CONFIGURATIONS:
            with self.subTest(configuration=name):
                model = standing_model()
                captured = _Scene(model, _tier_solver(model, name))
                tiers = captured.solver._pgs_solve_mf_gs_incremental_kernels
                self.assertTrue(tiers)
                self.assertTrue(all(getattr(kernel, "_fpgs_extra_inputs", False) for kernel in tiers))
                allocated, tier_inputs = [], []
                zeros, launch_tiled = wp.zeros, wp.launch_tiled

                def allocate(*args, allocated=allocated, zeros=zeros, **kwargs):
                    array = zeros(*args, **kwargs)
                    allocated.append(array)
                    return array

                def launch(kernel, *args, tier_inputs=tier_inputs, launch_tiled=launch_tiled, **kwargs):
                    if getattr(kernel, "_fpgs_extra_inputs", False):
                        tier_inputs.extend(kwargs.get("inputs", ()))
                    return launch_tiled(kernel, *args, **kwargs)

                with patch.object(wp, "zeros", allocate), patch.object(wp, "launch_tiled", launch):
                    with wp.ScopedCapture(device=device) as capture:
                        captured.step()
                self.assertTrue(tier_inputs, "The captured step must launch the tiers")
                late = [array for array in allocated if any(array is value for value in tier_inputs)]
                self.assertEqual(late, [], "Tier inputs were allocated inside the capture")
                wp.capture_launch(capture.graph)
                eager = _Scene(model, _tier_solver(model, name))
                eager.step()
                for field in STATE_FIELDS:
                    expected, actual = getattr(eager.out, field).numpy(), getattr(captured.out, field).numpy()
                    self.assertTrue(np.isfinite(actual).all(), field)
                    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4, err_msg=field)


if __name__ == "__main__":
    unittest.main()
