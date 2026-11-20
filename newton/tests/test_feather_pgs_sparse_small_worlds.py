# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check packed independent worlds against the physical contact law."""

import unittest
from itertools import product
from unittest import mock

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs.row_laws import ROW_CONTACT, ROW_FRICTION
from newton._src.solvers.feather_pgs.sweep import sweep_kernel
from newton._src.solvers.feather_pgs.sweep_routes_sparse import SparseSweep
from newton.tests import test_feather_pgs_sparse_contact_triplets as fixtures
from newton.tests.test_feather_pgs_sparse_pgs import _sweep


def _contact_problem(*, iterations, omega):
    """A coupled contact and its friction pair on a three-coordinate articulation, in three worlds (the middle one
    empty): the sweep's inputs and the physical sweep's answer (velocity and impulses)."""
    mass = np.array([[2.0, 0.3, 0.1], [0.3, 1.5, 0.2], [0.1, 0.2, 0.8]])
    factor = np.linalg.cholesky(mass)
    jacobian = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    kinds = np.array([ROW_CONTACT, ROW_FRICTION, ROW_FRICTION], np.int32)
    parents = np.array([-1, 0, 0], np.int32)
    bias = np.array([-1.0, 0.0, 0.0])
    mu = np.array([0.0, 0.3, 0.3])
    initial = np.array([-0.1, 0.4, -0.2])
    seeds = np.zeros(3)
    response = np.linalg.solve(mass, jacobian.T).T
    z = np.linalg.solve(factor, jacobian.T).T
    diagonal = np.einsum("ij,ij->i", jacobian, response)
    count, worlds, capacity = len(kinds), 3, 8

    def padded(value, dtype=np.float32):
        result = np.zeros((worlds, capacity, *value.shape[1:]), dtype=dtype)
        result[:, : len(value)] = value
        return result

    data = {
        "world_constraint_count": np.array([count, 0, count], np.int32),
        "rhs_bias": padded(bias),
        "diagonal": padded(diagonal),
        "initial_impulses": padded(seeds),
        "row_factor": padded(z),
        "row_incident": padded(jacobian @ initial),
        "row_type": padded(kinds, np.int32),
        "row_parent": padded(parents, np.int32),
        "row_mu": padded(mu),
        "support_row_class": np.zeros((worlds, capacity), np.int32),
        "support_class_dof": np.arange(3, dtype=np.int32)[None],
        "iterations": np.array(iterations),
        "omega": np.array(omega),
        "reference_factor_delta": np.zeros((worlds, 3), np.float32),
    }
    expected, impulses = _sweep(
        jacobian, response, initial, bias, diagonal, kinds, parents, mu, seeds, iterations=iterations, omega=omega
    )
    return data, initial, factor, expected, impulses


class TestSparseSmallWorlds(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_packed_worlds_preserve_physical_sweeps_and_replay(self):
        """Keep contacts isolated through empty/regrowing graph replay."""
        for lanes, local_factors in ((8, False), (16, False), (8, True), (16, True)):
            for table in (False, True):
                with self.subTest(lanes=lanes, table=table, local_factors=local_factors):
                    data, initial, factor, expected, impulses = _contact_problem(iterations=8, omega=0.4)
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
                        return sweep_kernel(
                            SparseSweep(
                                capacity,
                                dofs,
                                width,
                                use_support_table=_table,
                                lanes_per_world=_lanes,
                                local_factors=_local,
                            )
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
                                actual = initial + np.linalg.solve(factor.T, delta[world, coordinates])
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
                    return sweep_kernel(
                        SparseSweep(
                            capacity,
                            dofs,
                            width,
                            active_rows=True,
                            use_support_table=True,
                            contact_triplets=True,
                            lanes_per_world=_lanes,
                            local_factors=_local,
                        )
                    )

                with mock.patch.object(fixtures, "factory", factory):
                    group = fixtures.DeviceSolve(data, candidate=True)
                self.assertTrue(group.units)
                self.assertIn("_active_support_table_contact_triplets", group.kernel.key)
                warp.launch()
                group.launch()
                fixtures.assert_output(self, data, *group.result(), physical=physical, reference=warp.result())


if __name__ == "__main__":
    unittest.main()
