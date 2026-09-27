# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Focused CPU tests for the ordinary FP32 pre-allocation selector."""

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import simple_world as sw


def make_case(*, free=False):
    """Build one responsive body and one prescribed endpoint on CPU."""
    device = "cpu"
    width = 6 if free else 1

    def a(value, dtype=wp.float32):
        return wp.array(value, dtype=dtype, device=device)

    data = sw._SimpleWorldInput()
    data.world_dof_indices = a([list(range(width))], wp.int32)
    data.world_dof_count = a([width], wp.int32)
    data.limit_q_index = a([-1] * width if free else [0], wp.int32)
    data.lower, data.upper = a([-np.inf] * width), a([np.inf] * width)
    data.q, data.v_hat = a([0.0]), a([0.0] * width)
    data.body_to_articulation = a([0, 1], wp.int32)
    data.art_to_world = a([0, 0], wp.int32)
    data.articulation_dof_start = a([0, width], wp.int32)
    data.articulation_response_dof_count = a([width, 0], wp.int32)
    data.prescribed_articulation = a([0, 1], wp.int32)
    data.is_free_rigid = a([int(free), 0], wp.int32)
    data.body_flags = a([1, 2], wp.int32)
    data.body_has_response_dofs = a([1, 0], wp.int32)
    data.body_response_dof_mask = a([(1 << width) - 1, 0], wp.uint32)
    data.body_q = a([[0, 0, 0, 0, 0, 0, 1]] * 2, wp.transform)
    data.body_v_s = a([[999, 999, 999, 0, 0, 0], [0, 0, 0, 0, 0, 0]], wp.spatial_vector)
    data.joint_S_s = a(np.eye(6, dtype=np.float32)[:width], wp.spatial_vector)
    data.articulation_origin = a([[0, 0, 0], [0, 0, 0]], wp.vec3)
    data.max_linear_velocity = a([np.inf, np.inf])
    data.max_angular_velocity = a([np.inf, np.inf])
    data.max_depenetration_velocity = a([np.inf, np.inf])
    data.dt, data.beta, data.speculative_scale = 0.1, 0.2, 1.0
    data.activation_gap, data.restitution_threshold = 0.0, 0.5
    data.shared_anchor = 0
    data.absolute_margin, data.relative_margin = 0.0, 0.0
    raw = sw._SimpleRawContacts()
    raw.count = a([1], wp.int32)
    raw.shape0, raw.shape1 = a([0], wp.int32), a([1], wp.int32)
    raw.point0, raw.point1 = a([[0.1, 0, 0]], wp.vec3), a([[0, 0, 0]], wp.vec3)
    raw.normal = a([[-1, 0, 0]], wp.vec3)
    raw.margin0, raw.margin1 = a([0]), a([0])
    raw.shape_body = a([0, 1], wp.int32)
    raw.shape_mu, raw.shape_restitution = a([0.5, 0.5]), a([0, 0])
    scratch = sw._SimpleWorldScratch()
    for name in ("limit_ok", "rejected", "resolved", "checked_limits", "checked_normals", "global_invalid"):
        setattr(scratch, name, a([0], wp.int32))
    scratch.minimum_residual = a([0])
    scratch.body_twist = a(np.zeros((2, 6)), wp.spatial_vector)
    scratch.body_valid = a([0, 0], wp.int32)
    return data, raw, scratch


def evaluate(data, raw, scratch, *, enabled=1):
    """Execute the same four kernels on CPU, without initializing CUDA."""
    wp.launch(sw.prepare_simple_worlds, dim=1, inputs=[enabled, data, scratch], device="cpu")
    wp.launch(sw.build_predicted_body_twists, dim=2, inputs=[data, scratch], device="cpu")
    wp.launch(sw.check_raw_contact_normals, dim=1, inputs=[1, raw, data, scratch], device="cpu")
    wp.launch(sw.finalize_simple_worlds, dim=1, inputs=[scratch], device="cpu")
    return int(scratch.resolved.numpy()[0])


class TestSimpleWorld(unittest.TestCase):
    def test_current_predictor_not_incoming_body_twist(self):
        """Reject a closing predictor despite a separating stale body twist."""
        data, raw, scratch = make_case()
        self.assertEqual(evaluate(data, raw, scratch), 1)
        data.v_hat.assign([-2.0])
        self.assertEqual(evaluate(data, raw, scratch), 0)
        np.testing.assert_array_equal(scratch.body_twist.numpy()[0], [-2, 0, 0, 0, 0, 0])
        self.assertEqual(int(scratch.checked_normals.numpy()[0]), 1)

    def test_position_limits_both_sides_and_speculative_bias(self):
        """Check activated lower and upper rows without constructing J."""
        data, raw, scratch = make_case()
        data.lower.assign([0.0])
        data.q.assign([-0.01])
        self.assertEqual(evaluate(data, raw, scratch), 0)
        data.v_hat.assign([0.03])
        self.assertEqual(evaluate(data, raw, scratch), 1)
        data.q.assign([0.01])
        data.activation_gap = 0.02
        data.v_hat.assign([-0.2])
        self.assertEqual(evaluate(data, raw, scratch), 0)
        data.lower.assign([-np.inf])
        data.upper.assign([0.0])
        data.q.assign([0.01])
        data.v_hat.assign([-0.03])
        self.assertEqual(evaluate(data, raw, scratch), 1)
        self.assertEqual(int(scratch.checked_limits.numpy()[0]), 1)

    def test_prescribed_endpoint_and_restitution(self):
        """Retain known endpoint motion and frozen incident rebound semantics."""
        data, raw, scratch = make_case()
        raw.point0.assign([[0, 0, 0]])
        raw.shape_restitution.assign([1.0, 1.0])
        data.body_v_s.assign([[999, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]])
        self.assertEqual(evaluate(data, raw, scratch), 0)
        self.assertAlmostEqual(float(scratch.minimum_residual.numpy()[0]), -2.0, places=6)
        data.body_v_s.assign([[999, 0, 0, 0, 0, 0], [-1, 0, 0, 0, 0, 0]])
        self.assertEqual(evaluate(data, raw, scratch), 1)

    def test_zero_friction_radius_allows_slip(self):
        """Accept zero normal impulse without requiring zero tangential speed."""
        data, raw, scratch = make_case(free=True)
        raw.point0.assign([[0, 0, 0]])
        raw.shape_mu.assign([1000, 1000])
        data.v_hat.assign([0, 100, 0, 0, 0, 0])
        self.assertEqual(evaluate(data, raw, scratch), 1)

    def test_rigid_caps_and_uncapped_mf_depenetration(self):
        """Preserve finite speed caps and positive-infinity uncapped bias."""
        data, raw, scratch = make_case(free=True)
        data.v_hat.assign([0, 0, 0, 3, 0, 0])
        data.max_angular_velocity.assign([2, np.inf])
        self.assertEqual(evaluate(data, raw, scratch), 0)
        data.max_angular_velocity.assign([np.inf, np.inf])
        raw.point0.assign([[-1, 0, 0]])
        data.v_hat.assign([1, 0, 0, 0, 0, 0])
        self.assertEqual(evaluate(data, raw, scratch), 0)
        data.max_depenetration_velocity.assign([0.5, np.inf])
        self.assertEqual(evaluate(data, raw, scratch), 1)
        self.assertAlmostEqual(float(scratch.minimum_residual.numpy()[0]), 0.5, places=6)

    def test_shared_anchor_and_large_world_translation(self):
        """Use shared normal anchors and stable endpoint-relative geometry."""
        data, raw, scratch = make_case()
        data.joint_S_s.assign([[0, 0, 0, 0, 0, 1]])
        data.v_hat.assign([1])
        raw.point0.assign([[0, 1, 0]])
        raw.point1.assign([[0, -1, 0]])
        self.assertEqual(evaluate(data, raw, scratch), 0)
        data.shared_anchor = 1
        self.assertEqual(evaluate(data, raw, scratch), 1)
        data.body_q.assign([[1e6, 0, 0, 0, 0, 0, 1]] * 2)
        data.articulation_origin.assign([[1e6, 0, 0]] * 2)
        self.assertEqual(evaluate(data, raw, scratch), 1)
        self.assertEqual(float(scratch.minimum_residual.numpy()[0]), 0.0)

    def test_empty_disabled_and_transient_decisions(self):
        """Clear prior acceptance on disabled and newly violating calls."""
        data, raw, scratch = make_case()
        self.assertEqual(evaluate(data, raw, scratch), 1)
        self.assertEqual(evaluate(data, raw, scratch, enabled=0), 0)
        self.assertEqual(evaluate(data, raw, scratch), 1)
        raw.count.assign([0])
        self.assertEqual(evaluate(data, raw, scratch), 1)
        self.assertEqual(int(scratch.checked_normals.numpy()[0]), 0)

    def test_malformed_prefix_ids_and_nonfinite_input(self):
        """Reject incomplete or invalid raw constraints instead of dropping them."""
        data, raw, scratch = make_case()
        raw.count.assign([2])
        self.assertEqual(evaluate(data, raw, scratch), 0)
        raw.count.assign([1])
        raw.shape0.assign([999])
        self.assertEqual(evaluate(data, raw, scratch), 0)
        raw.shape0.assign([0])
        raw.normal.assign([[0, 0, 0]])
        self.assertEqual(evaluate(data, raw, scratch), 0)
        raw.normal.assign([[np.nan, 0, 0]])
        self.assertEqual(evaluate(data, raw, scratch), 0)
        raw.normal.assign([[-1, 0, 0]])
        data.v_hat.assign([np.inf])
        self.assertEqual(evaluate(data, raw, scratch), 0)

    def test_rejection_margin_is_not_violation_allowance(self):
        """Send numerically ambiguous nonnegative residuals to fallback."""
        data, raw, scratch = make_case()
        raw.point0.assign([[0, 0, 0]])
        self.assertEqual(evaluate(data, raw, scratch), 1)
        data.absolute_margin = 1.0e-5
        self.assertEqual(evaluate(data, raw, scratch), 0)
        data.v_hat.assign([2.0e-5])
        self.assertEqual(evaluate(data, raw, scratch), 1)

    def test_unsupported_constructor_has_empty_mask(self):
        """Keep unsupported original solvers disabled without raising."""
        solver = SimpleNamespace(model=SimpleNamespace(device=wp.get_device("cpu")))
        classifier = sw.SimpleWorldClassifier(solver)
        self.assertFalse(classifier.enabled)
        self.assertEqual(classifier.resolved.shape, (0,))
        classifier.classify(None, None, None, 0.0)


if __name__ == "__main__":
    unittest.main()
