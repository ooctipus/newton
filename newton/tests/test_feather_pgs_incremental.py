# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check directed responses in the incremental-residual contact solve."""

import functools
import inspect
import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import sweep_routes
from newton._src.solvers.feather_pgs.kernels import PGS_CONSTRAINT_TYPE_CONTACT as CONTACT
from newton._src.solvers.feather_pgs.kernels import PGS_CONSTRAINT_TYPE_FRICTION as FRICTION
from newton.solvers import SolverFeatherPGS
from newton.tests import test_feather_pgs_sparse_solver as sparse_tests
from newton.tests import test_feather_pgs_sparse_split as split_tests
from newton.tests.feather_pgs_law_harness import articulation_factor
from newton.tests.feather_pgs_patch_parity import (
    SLIDING_FEET,
    SLIDING_ITERATIONS,
    SLIDING_REGION_BOUNDS,
    STANDING_REGION_BOUNDS,
    STANDING_VELOCITY_ATOL,
    anchor_sites,
    assert_anchor_sites_close,
    assert_history_matches_rows,
    assert_loaded_anchors,
    assert_patch_impulses_close,
    assert_patch_rows_close,
    assert_region_impulses_close,
    assert_ring_coulomb,
    assert_states_close,
    patch_rows,
    ring_budget_ratios,
    standing_model,
)


@unittest.skipUnless(wp.is_cuda_available(), "Incremental native solve requires CUDA")
class TestFeatherPGSIncremental(unittest.TestCase):
    def test_preeliminated_response_does_not_reintroduce_bilateral_velocity(self):
        """Preserve an eliminated bilateral row when only contact responses are projected."""
        device = wp.get_device()
        if not device.is_cuda:
            self.skipTest("Select a CUDA device for native incremental execution")

        def array(value, dtype=wp.float32):
            return wp.array(value, dtype=dtype, device=device)

        zero_int = array([0], wp.int32)
        zero_float = array([[0.0]])
        zero_spatial = wp.zeros((1, 1, 6), dtype=wp.float32, device=device)
        # H=I (one articulation of two coordinates, L=I), the bilateral row's Z_B=(1,0) and the contact row
        # projected off it, Z_C=(0,1): applying the contact impulse cannot change the bilateral velocity.
        values = {
            "general_world_count": array([1], wp.int32),
            "general_worlds": zero_int,
            "general_world_grid_stride": 1,
            "use_general_world_queue": 0,
            "world_constraint_count": array([2], wp.int32),
            "dense_phase_bounds": array([[0, 0]], wp.int32),
            "local_solve_owner": zero_int,
            "world_dof_indices": array([[0, 1]], wp.int32),
            "rhs_bias": array([[0.0, -1.0]]),
            "world_diag": array([[1.0, 1.0]]),
            "world_row_w": array([[1.0, 1.0]]),
            "world_impulses": array([[0.0, 0.0]]),
            "response_dofs": array([2], wp.int32),
            "articulation_world_dof_offset": array([0], wp.int32),
            "packed_dense": 1,
            "packed_Z": array([[[1.0, 0.0], [0.0, 1.0]]]),
            "row_sides": wp.array([[[0, -1], [0, -1]]], dtype=wp.vec2i, device=device),
            "world_row_type": array([[5, 0]], wp.int32),
            "world_row_parent": array([[-1, -1]], wp.int32),
            "world_row_mu": array([[0.0, 0.0]]),
            "world_torsion_group": array([[-1]], wp.int32),
            "contact_torsion_radius": 0.0,
            "mf_constraint_count": zero_int,
            "mf_contact_rows_end": zero_int,
            "mf_meta": array([[0]], wp.int32),
            "mf_impulses": zero_float,
            "mf_Z_a": zero_spatial,
            "mf_Z_b": zero_spatial,
            "factors": articulation_factor(np.eye(2), device)[0],
            "mf_row_mu": zero_float,
            "mf_row_w": zero_float,
            "regularize": 1,
            "row_phase": 0,
            "friction_start_iteration": 0,
            "iteration_offset": 0,
            "freeze_drive_rows": 0,
            "v_out": array([0.0, 0.0]),
        }
        for name in ("target_vel_bias", "vel_multiplier", "impulse_multiplier", "max_impulse", "vel_limit"):
            values[f"world_drive_{name}"] = zero_float
        kernel = sweep_routes.incremental_kernel(
            2, 1, 2, str(device.arch), incremental_rows=2, has_drive_rows=False, has_dense_velocity_limit_rows=False
        )
        for omega in (1.0, 0.65):
            for weight in (1.0, 0.8):
                for iterations in (1, 2, 8):
                    with self.subTest(omega=omega, weight=weight, iterations=iterations):
                        values["v_out"].zero_()
                        values["world_impulses"].zero_()
                        values["world_row_w"].assign(np.array([[1.0, weight]], dtype=np.float32))
                        values.update(omega=omega, iterations=iterations)
                        wp.launch_tiled(
                            kernel,
                            dim=[1],
                            inputs=[values[name] for name in inspect.signature(kernel.func).parameters],
                            block_dim=32,
                            device=device,
                        )
                        # The scalar contact recurrence is lambda'=lambda+omega*(weight-lambda).
                        expected = [0.0, weight * (1.0 - (1.0 - omega) ** iterations)]
                        np.testing.assert_allclose(values["v_out"].numpy(), expected, rtol=1.0e-6, atol=1.0e-7)
                        np.testing.assert_allclose(
                            values["world_impulses"].numpy()[0], expected, rtol=1.0e-6, atol=1.0e-7
                        )


def _patch_case(model, incremental_rows, *, iterations=8):
    """Solve dense matrix-free rows under patch friction, optionally through the incremental kernel."""
    # Sorted contacts: both arms see the same contact order, which the by-contact patch checks key on.
    # The pipeline sets model.rigid_contact_max, which sizes the solver's contact and patch buffers, so it is
    # built first: both arms then hold the same capacity on the shared model.
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=32, deterministic=True)
    with mock.patch.dict(SolverFeatherPGS._route_pins, {"storage": "dense", "evaluator": "ordinary"}):
        solver = SolverFeatherPGS(
            model,
            pgs_mode="matrix_free",
            pgs_iterations=iterations,
            enable_joint_limits=True,
            friction_anchor_beta=0.2,
            mf_gs_incremental_rows=incremental_rows,
            dense_max_constraints=32,
            update_mass_matrix_interval=2,
        )
    state, out, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return model, solver, state, out, control, pipeline, pipeline.contacts()


@unittest.skipUnless(wp.is_cuda_available(), "Incremental native solve requires CUDA")
class TestFeatherPGSIncrementalPatch(unittest.TestCase):
    def test_patch_rows_match_the_ordinary_sweep(self):
        """Solve linked patch rows through the incremental kernel as the ordinary matrix-free sweep does.

        The standing fixture keeps loaded, ring-linked anchors on its box feet at every step, so the
        anchor history comparison covers carried anchors; the tripod covers limit rows beside contacts;
        the sliding feet hold every anchor at its pooled ring budget, which the incremental kernel's
        projection must pool as the ordinary sweep does. Rows compare by contact for normal rows and by
        friction location for anchors (feather_pgs_patch_parity): rounding picks which corner of a
        rectangular foot's support edge carries the edge's friction rows.
        """
        for name, build, iterations, bounds in (
            ("standing", standing_model, 8, STANDING_REGION_BOUNDS),
            ("tripod", sparse_tests._build_model, 8, None),
            ("sliding", functools.partial(standing_model, **SLIDING_FEET), SLIDING_ITERATIONS, SLIDING_REGION_BOUNDS),
        ):
            sliding = name == "sliding"
            with self.subTest(fixture=name):
                model = build()
                reference = _patch_case(model, 0, iterations=iterations)
                candidate = _patch_case(model, 32, iterations=iterations)
                self.assertEqual(reference[1]._friction_patches.capacity, candidate[1]._friction_patches.capacity)
                self.assertEqual(reference[1]._incremental_rows, 0)
                self.assertEqual(candidate[1]._incremental_rows, 32)
                self.assertTrue(candidate[1]._mf_gs_tiers)
                # The route record the GPU gates grade: the tier, and the budget it serves.
                for case, rows in ((reference, 0), (candidate, 32)):
                    selected = case[1].get_path_selection()
                    self.assertEqual(selected["routes"]["friction"]["law"], "patch")
                    self.assertEqual(selected["options"]["mf_gs_incremental_rows"]["applied"], rows)
                    tiers = selected["routes"]["sweep"]["tiers"]
                    self.assertEqual(None if tiers is None else tiers["family"], "incremental" if rows else None)
                linked = carried = False
                ratios = []
                for index in range(12):
                    for case in (reference, candidate):
                        split_tests._step(case)
                    # Only float order differs; it compounds through the rollout like other route changes.
                    velocity_atol = 2e-3 if bounds is None else STANDING_VELOCITY_ATOL
                    assert_states_close(reference[2], candidate[2], velocity_atol=velocity_atol)
                    rows = patch_rows(reference[1], reference[-1], split_tests.DT)
                    candidate_rows = patch_rows(candidate[1], candidate[-1], split_tests.DT)
                    assert_patch_rows_close(rows, candidate_rows, lifecycle=not sliding)
                    if bounds is None:
                        assert_patch_impulses_close(rows, candidate_rows)
                    else:
                        assert_region_impulses_close(rows, candidate_rows, **bounds)
                    # The other rows keep their slots.
                    for world, count in enumerate(reference[1].constraint_count.numpy()):
                        kinds = reference[1].row_type.numpy()[world, :count]
                        other = (kinds != CONTACT) & (kinds != FRICTION)
                        np.testing.assert_array_equal(candidate[1].row_type.numpy()[world, :count][other], kinds[other])
                        np.testing.assert_allclose(
                            candidate[1].impulses.numpy()[world, :count][other],
                            reference[1].impulses.numpy()[world, :count][other],
                            atol=2e-5,
                            rtol=7e-4,
                        )
                    for case in (reference, candidate):
                        assert_ring_coulomb(case[1])
                        assert_history_matches_rows(case[1], case[-1])
                        if sliding:
                            ratios.extend(ring_budget_ratios(case[1]))
                    if not sliding:
                        assert_anchor_sites_close(
                            anchor_sites(reference[1], reference[-1]),
                            anchor_sites(candidate[1], candidate[-1]),
                            tangent=bounds is None,
                        )
                    if name == "standing":
                        for case in (reference, candidate):
                            assert_loaded_anchors(case[1], case[-1], carried=index > 0)
                    linked |= any(record["next"] not in (-1, c) for c, record in candidate_rows["normals"].items())
                    carried |= bool(candidate[1]._friction_patches.previous.valid.numpy().any())
                if sliding:
                    self.assertGreater(
                        float(np.mean(np.asarray(ratios) > 0.99)), 0.9, "Sliding anchors must load their budget"
                    )
                else:
                    self.assertTrue(carried, "Anchors must carry history between steps")
                if name != "tripod":
                    self.assertTrue(linked, "A box foot must link several normals in one ring")


if __name__ == "__main__":
    unittest.main()
