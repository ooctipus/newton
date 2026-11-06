# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check packed independent worlds against the physical velocity/contact law."""

import unittest
from itertools import product
from unittest import mock

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs.sweep_routes_sparse import sparse_kernel
from newton.tests import test_feather_pgs_sparse_contact_triplets as fixtures
from newton.tests import test_feather_pgs_sparse_velocity_limits as velocity


class TestSparseSmallWorlds(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_packed_worlds_preserve_physical_sweeps_and_replay(self):
        """Keep signed clamps and contacts isolated through empty/regrowing graph replay."""
        for lanes, local_factors in ((8, False), (16, False), (8, True), (16, True)):
            for table in (False, True):
                with self.subTest(lanes=lanes, table=table, local_factors=local_factors):
                    data, physical, factor = velocity._solve_data(-1.0, contact=True, iterations=8, omega=0.4)
                    expected, impulses = velocity._physical_sweep(*physical, iterations=8, omega=0.4)
                    count = len(impulses)
                    worlds = 17  # Multiple blocks and a partial last subgroup.
                    for name, values in list(data.items()):
                        if values.ndim and values.shape[0] == 3 and name != "support_class_dof":
                            data[name] = np.repeat(values[:1], worlds, axis=0)
                    data["world_constraint_count"][1::3] = 0
                    # Put the last physical coordinate in the upper half of a
                    # 16-lane group, so reducing over only eight lanes fails.
                    coordinates = np.array([0, lanes // 2, lanes - 1])
                    indices = np.full(lanes, -1, np.int32)
                    indices[coordinates] = coordinates
                    factors = np.full((*data["row_factor"].shape[:2], lanes), 17.0, np.float32)
                    factors[:, :, coordinates] = data["row_factor"]
                    data["row_factor"] = factors
                    data["support_class_dof"] = indices[None]
                    data["reference_factor_delta"] = np.zeros((worlds, lanes), np.float32)

                    def factory(capacity, dofs, width, *, _table=table, _lanes=lanes, _local=local_factors, **_kwargs):
                        return sparse_kernel(
                            capacity,
                            dofs,
                            width,
                            use_support_table=_table,
                            velocity_limits=True,
                            lanes_per_world=_lanes,
                            local_factors=_local,
                        )

                    with mock.patch.object(fixtures, "factory", factory):
                        solve = fixtures.DeviceSolve(data, candidate=True)
                    if not table:
                        solve.support.enabled = 0
                        row_indices = np.broadcast_to(indices, data["row_factor"].shape).copy()
                        solve.inputs[6] = wp.array(row_indices, dtype=int, device=solve.device)

                    def launch(solve=solve, lanes=lanes, worlds=worlds):
                        groups = 64 // lanes
                        wp.launch_tiled(
                            solve.kernel,
                            dim=[(worlds + groups - 1) // groups],
                            inputs=solve.inputs,
                            outputs=[solve.delta, solve.support],
                            block_dim=64,
                            device=solve.device,
                        )

                    launch()
                    with wp.ScopedCapture(device=solve.device) as capture:
                        launch()
                    for phase in range(3):
                        counts = np.full(worlds, count, np.int32)
                        counts[phase::3] = 0
                        solve.inputs[1].assign(counts)
                        solve.reset_impulses()
                        wp.capture_launch(capture.graph)
                        delta, actual_impulses = solve.result()
                        for world in range(worlds):
                            if counts[world] == 0:
                                np.testing.assert_array_equal(delta[world], 0.0)
                            else:
                                actual = physical[2] + np.linalg.solve(factor.T, delta[world, coordinates])
                                np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=3e-4)
                                np.testing.assert_allclose(
                                    actual_impulses[world, :count],
                                    impulses,
                                    atol=3e-5,
                                    rtol=3e-4,
                                )

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_lane_groups_compose_the_active_scan_units_and_local_factors(self):
        """A group of 8 or 16 lanes runs the active scan and contact units on worlds wider than the group, and local
        factors serve worlds wider than 16 coordinates (the 192-row fixture's 18): each as a warp's sweep does, the
        physical active sweep's answer within rounding, and the warp's. The 29-row offset puts the units and the
        scan's selections across 32-row words, which a group fills a part at a time."""
        settings = ({"seeded": True}, {"offset": 29, "seeded": True}, {"omega": 0.7, "seeded": True})
        for (lanes, local), options in product(((8, False), (16, False), (32, True)), settings):
            with self.subTest(lanes=lanes, local_factors=local, **options):
                data, physical = fixtures.fixture(**options)
                warp = fixtures.DeviceSolve(data, candidate=True)

                def factory(capacity, dofs, width, *, _lanes=lanes, _local=local, **_kwargs):
                    return sparse_kernel(
                        capacity,
                        dofs,
                        width,
                        active_rows=True,
                        use_support_table=True,
                        contact_triplets=True,
                        lanes_per_world=_lanes,
                        local_factors=_local,
                    )

                with mock.patch.object(fixtures, "factory", factory):
                    group = fixtures.DeviceSolve(data, candidate=True)
                self.assertTrue(group.units)
                self.assertIn("_active_support_table_contact_triplets", group.kernel.key)
                warp.launch()
                group.launch()
                fixtures.assert_output(self, data, *group.result(), physical=physical, reference=warp.result())

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_velocity_limits_compose_the_active_scan_and_units(self):
        """The last-word velocity-limit pass after an active sweep with contact units, in a warp or a smaller group:
        the physical sweep's answer (contacts first, then the velocity-limit totals)."""
        for lanes, sign in product((32, 16, 8), (-1.0, 1.0)):
            with self.subTest(lanes=lanes, sign=sign):
                data, physical, factor = velocity._solve_data(sign, contact=True, iterations=8, omega=0.4)
                data["initial_impulses"][:] = 0.0
                physical = (*physical[:-1], np.zeros_like(physical[-1]))

                def factory(capacity, dofs, width, *, _lanes=lanes, **_kwargs):
                    return sparse_kernel(
                        capacity,
                        dofs,
                        width,
                        active_rows=True,
                        use_support_table=True,
                        contact_triplets=True,
                        velocity_limits=True,
                        lanes_per_world=_lanes,
                    )

                with mock.patch.object(fixtures, "factory", factory):
                    solve = fixtures.DeviceSolve(data, candidate=True)
                self.assertTrue(solve.units)
                self.assertIn("_active_support_table_contact_triplets_velocity_limits", solve.kernel.key)
                solve.launch()
                delta, impulses = solve.result()
                expected_velocity, expected_impulses = velocity._physical_sweep(*physical, iterations=8, omega=0.4)
                for world in (0, 2):
                    actual = physical[2] + np.linalg.solve(factor.T, delta[world])
                    np.testing.assert_allclose(actual, expected_velocity, atol=3e-5, rtol=3e-4)
                    np.testing.assert_allclose(
                        impulses[world, : len(expected_impulses)], expected_impulses, atol=3e-5, rtol=3e-4
                    )
                np.testing.assert_array_equal(delta[1], 0.0)


if __name__ == "__main__":
    unittest.main()
