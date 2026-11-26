# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Allocate the inputs a captured step reads before the capture."""

import copy
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton.tests.feather_pgs_patch_parity import standing_model
from newton.tests.test_feather_pgs_contract_loops import _build_four_bar
from newton.tests.test_feather_pgs_parallel_patch import _matrix_free_solver

DT = 1.0 / 240.0
STATE_FIELDS = ("joint_q", "joint_qd", "body_q", "body_qd")


def _tier_solver(model):
    """The point tiers of 32 rows (a test pin), forming the response in-kernel, which the standing model fits."""
    return _matrix_free_solver(model, beta=0.2, parallel_rows=32)


class _Scene:
    def __init__(self, model, solver):
        self.model, self.solver = model, solver
        self.state, self.out, self.control = model.state(), model.state(), model.control()
        newton.eval_fk(model, self.state.joint_q, self.state.joint_qd, self.state)
        self.pipeline = newton.CollisionPipeline(model, rigid_contact_max=64, growable_contacts=True)
        self.contacts = self.pipeline.contacts()

    def step(self):
        self.state.clear_forces()
        self.pipeline.collide(self.state, self.contacts)
        self.solver.step(self.state, self.out, self.control, self.contacts, DT)


@unittest.skipUnless(wp.is_cuda_available(), "Parallel matrix-free tiers require CUDA")
class TestFeatherPGSCaptureAllocation(unittest.TestCase):
    def test_tier_inputs_precede_a_captured_first_step(self):
        """Give a first step captured right after construction tier inputs that existed before the capture.

        The tiers' in-kernel response inputs are allocated with their construction. Allocated at
        their first launch, inside the capture, each ``wp.zeros`` would be a memset node that every
        replay runs again. The captured step then matches the same first step taken eagerly.
        """
        device = wp.get_device("cuda:0")
        model = standing_model()
        captured = _Scene(model, _tier_solver(model))
        tiers = captured.solver.sweep.tiers
        self.assertTrue(tiers)
        tier_kernels = [tier.kernel for tier in tiers]
        self.assertIsNotNone(captured.solver.spec.classes.tiers.sizes)
        allocated, tier_inputs = [], []
        zeros, launch_tiled = wp.zeros, wp.launch_tiled

        def allocate(*args, allocated=allocated, zeros=zeros, **kwargs):
            array = zeros(*args, **kwargs)
            allocated.append(array)
            return array

        def launch(
            kernel,
            *args,
            tier_inputs=tier_inputs,
            tier_kernels=tier_kernels,
            launch_tiled=launch_tiled,
            **kwargs,
        ):
            if any(kernel is tier_kernel for tier_kernel in tier_kernels):
                tier_inputs.extend(kwargs.get("inputs", ()))
            return launch_tiled(kernel, *args, **kwargs)

        with patch.object(wp, "zeros", allocate), patch.object(wp, "launch_tiled", launch):
            with wp.ScopedCapture(device=device) as capture:
                captured.step()
        self.assertTrue(tier_inputs, "The captured step must launch the tiers")
        late = [array for array in allocated if any(array is value for value in tier_inputs)]
        self.assertEqual(late, [], "Tier inputs were allocated inside the capture")
        wp.capture_launch(capture.graph)
        eager = _Scene(model, _tier_solver(model))
        eager.step()
        for field in STATE_FIELDS:
            expected, actual = getattr(eager.out, field).numpy(), getattr(captured.out, field).numpy()
            self.assertTrue(np.isfinite(actual).all(), field)
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-6, err_msg=field)

    def test_a_cut_forest_steps_eagerly_after_a_captured_first_step(self):
        """Step a cut forest (a four-bar closed by a loop joint) eagerly right after a capture that took its first step.

        The solver steps mirrors of the caller's states and control in the forest's layout. Allocated at the first step
        that sees them, inside the capture, they would belong to the graph, unmapped until its first replay: the eager
        step would write through their addresses (CUDA 700). A state made before the solver and one made after it both
        hold their mirrors before the capture, and the eager step matches a first step taken without any capture.
        """
        device = wp.get_device("cuda:0")
        model = _build_four_bar().finalize(device=device)

        def run(capture: bool):
            state = model.state()
            solver = newton.solvers.SolverFeatherPGS(model, pgs_iterations=16, pgs_beta=0.1)
            out, control = model.state(), model.control()
            if capture:
                with wp.ScopedCapture(device=device) as captured:
                    solver.step(state, out, control, None, DT)
            solver.step(state, out, control, None, DT)
            wp.synchronize_device(device)
            if capture:
                wp.capture_launch(captured.graph)
                wp.synchronize_device(device)
            return out

        after_capture, eager = run(capture=True), run(capture=False)
        for field in STATE_FIELDS:
            actual = getattr(after_capture, field).numpy()
            self.assertTrue(np.isfinite(actual).all(), field)
            np.testing.assert_array_equal(actual, getattr(eager, field).numpy(), err_msg=field)

    def test_a_capture_does_not_take_a_state_the_model_did_not_make(self):
        """A state the model did not make takes its mirror at its first step outside capture; a capture that would be
        the first to step it raises instead of allocating the mirror inside the graph."""
        device = wp.get_device("cuda:0")
        model = _build_four_bar().finalize(device=device)
        solver = newton.solvers.SolverFeatherPGS(model, pgs_iterations=16, pgs_beta=0.1)
        state, control = model.state(), model.control()
        unknown = copy.copy(model.state())
        with self.assertRaisesRegex(RuntimeError, "Model.state"):
            with wp.ScopedCapture(device=device):
                solver.step(state, unknown, control, None, DT)
        solver.step(state, unknown, control, None, DT)
        wp.synchronize_device(device)
        self.assertTrue(np.isfinite(unknown.body_q.numpy()).all())


if __name__ == "__main__":
    unittest.main()
