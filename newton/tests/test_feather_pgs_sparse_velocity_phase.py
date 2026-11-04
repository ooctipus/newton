# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Preserve the final scalar velocity phase across padded support tiles."""

import unittest
from unittest import mock

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs.row_kernels import PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT as VELOCITY_LIMIT
from newton._src.solvers.feather_pgs.sweep_routes_sparse import sparse_kernel
from newton.tests import test_feather_pgs_sparse_contact_triplets as triplet_tests
from newton.tests import test_feather_pgs_sparse_velocity_limits as velocity_tests


def _padded_problem(sign, iterations):
    """Embed the coupled physical oracle across two support tiles and retain holes."""
    data, physical, factor = velocity_tests._solve_data(sign, contact=True, iterations=iterations, omega=0.4)
    # Velocity-limit rows accumulate onto carried totals, also where contacts act.
    physical[-1][:2] = [3.0, 2.0]
    data["initial_impulses"][:, :2] = physical[-1][:2]
    expected_velocity, expected_impulses = velocity_tests._physical_sweep(*physical, iterations=iterations, omega=0.4)
    worlds, capacity, _ = data["row_factor"].shape
    dofs, width = 48, 40
    coordinates = np.array([1, 21, 46])
    entries = np.array([0, 16, 36])
    indices = np.full(width, -1, np.int32)
    indices[entries] = coordinates
    factors = np.full((worlds, capacity, width), 17.0, np.float32)
    factors[:, :, entries] = data["row_factor"]
    data["row_factor"] = factors
    data["support_class_dof"] = indices[None]
    data["reference_factor_delta"] = np.zeros((worlds, dofs), np.float32)

    # Disabled denominator rows leave their published impulses untouched.
    # They remain in the final phase and must never contribute a correction.
    count = len(physical[3])
    data["world_constraint_count"] = np.array([count + 2, 0, count + 2], np.int32)
    data["row_type"][:, count : count + 2] = VELOCITY_LIMIT
    data["diagonal"][:, count : count + 2] = [0.0, -1.0]
    data["rhs_bias"][:, count : count + 2] = [-3.0, -4.0]
    data["initial_impulses"][:, count : count + 2] = [0.7, 0.9]
    expected_impulses = np.concatenate((expected_impulses, [0.7, 0.9]))
    physical_factor = np.eye(dofs)
    physical_factor[np.ix_(coordinates, coordinates)] = factor
    initial = np.zeros(dofs)
    initial[coordinates] = physical[2]
    expected = initial.copy()
    expected[coordinates] = expected_velocity
    return data, physical_factor, initial, expected, expected_impulses


class TestSparseVelocityPhase(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_padded_velocity_phase_matches_physical_response(self):
        """Keep signed final totals with two tiles and either index layout."""
        for sign in (-1.0, 1.0):
            for iterations in (1, 8):
                for table in (False, True):
                    with self.subTest(sign=sign, iterations=iterations, table=table):
                        data, factor, initial, expected_velocity, expected_impulses = _padded_problem(sign, iterations)

                        def factory(capacity, dofs, width, _table=table, **_kwargs):
                            return sparse_kernel(capacity, dofs, width, use_support_table=_table, velocity_limits=True)

                        with mock.patch.object(triplet_tests, "factory", factory):
                            solve = triplet_tests.DeviceSolve(data, candidate=True)
                        if not table:
                            solve.support.enabled = 0
                            row_indices = np.broadcast_to(data["support_class_dof"][0], data["row_factor"].shape).copy()
                            solve.inputs[6] = wp.array(row_indices, dtype=int, device=solve.device)
                        solve.launch()
                        delta, impulses = solve.result()
                        for world in (0, 2):
                            velocity = initial + np.linalg.solve(factor.T, delta[world])
                            np.testing.assert_allclose(velocity, expected_velocity, atol=3e-5, rtol=3e-4)
                            np.testing.assert_allclose(
                                impulses[world, : len(expected_impulses)],
                                expected_impulses,
                                atol=3e-5,
                                rtol=3e-4,
                            )
                        np.testing.assert_array_equal(delta[1], 0.0)


if __name__ == "__main__":
    unittest.main()
