# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Construction of SolverFeatherPGS on heterogeneous multi-world models.

Heterogeneous multi-world models (worlds whose per-world DOF counts differ)
construct on the matrix-free and split paths, as homogeneous models do.
"""

from __future__ import annotations

import unittest

import warp as wp

import newton
from newton.solvers import SolverFeatherPGS


def _make_chain_world(n_links: int) -> newton.ModelBuilder:
    """A fixed-base serial chain of ``n_links`` revolute links (n_links DOFs)."""
    builder = newton.ModelBuilder()
    joints = []
    prev = -1
    for i in range(n_links):
        link = builder.add_link()
        builder.add_shape_box(link, hx=0.15, hy=0.03, hz=0.03)
        if prev == -1:
            parent_xform = wp.transform(p=wp.vec3(0.0, 0.0, 1.0), q=wp.quat_identity())
        else:
            parent_xform = wp.transform(p=wp.vec3(0.15, 0.0, 0.0), q=wp.quat_identity())
        joints.append(
            builder.add_joint_revolute(
                parent=prev,
                child=link,
                axis=wp.vec3(0.0, 1.0, 0.0),
                parent_xform=parent_xform,
                child_xform=wp.transform(p=wp.vec3(-0.15, 0.0, 0.0), q=wp.quat_identity()),
                label=f"chain_joint_{i}",
            )
        )
        prev = link
    builder.add_articulation(joints, label="chain")
    return builder


def _build_model(world_link_counts: list[int], device: str) -> newton.Model:
    """One chain world per entry in ``world_link_counts``, plus a ground plane."""
    scene = newton.ModelBuilder()
    for n_links in world_link_counts:
        scene.add_world(_make_chain_world(n_links))
    scene.add_ground_plane()
    return scene.finalize(device=device)


class TestFeatherPGSHeteroGuard(unittest.TestCase):
    """Heterogeneous and homogeneous models construct."""

    @classmethod
    def setUpClass(cls):
        if wp.get_cuda_device_count() == 0:
            raise unittest.SkipTest("SolverFeatherPGS construction tests require cuda:0")
        cls.device = "cuda:0"
        # Worlds alternate a 1-link pendulum and a 3-link chain: per-world DOF
        # counts [1, 3, 1, 3] -> heterogeneous.
        cls.hetero_model = _build_model([1, 3, 1, 3], cls.device)
        # All 3-link chains: per-world DOF counts uniform -> homogeneous.
        cls.homogeneous_model = _build_model([3, 3, 3, 3], cls.device)

    def test_hetero_constructs(self):
        solver = SolverFeatherPGS(self.hetero_model)
        self.assertEqual(solver._matrix_free, self.hetero_model.device.is_cuda)

    def test_homogeneous_constructs(self):
        solver = SolverFeatherPGS(self.homogeneous_model)
        self.assertEqual(solver._matrix_free, self.homogeneous_model.device.is_cuda)


if __name__ == "__main__":
    unittest.main()
