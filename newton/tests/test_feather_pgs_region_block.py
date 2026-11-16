# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""A compliant group of regions visited as one block (``region_sweep.form_blocks``).

A plate rests on a grid of small immovable spheres, a block sits on the plate, every shape authors NIST's part
material. With more spheres under the plate than its six coordinates, the plate's regions against them close at the
predictor in excess of its coordinates and the listing makes them one block; with fewer, they stay regions. Either
way the plate comes to rest on its springs.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS

PART = (1.6e5, 800.0)
"""NIST's authored part material (ke [N/m], kd [N s/m])."""


def resting_plate(side: int, steps: int = 400):
    """The plate on a ``side`` x ``side`` grid of spheres, stepped on the CPU: the plate's heights and speeds a step,
    and the solver."""
    device = wp.get_device("cpu")
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.75, ke=PART[0], kd=PART[1])
    radius, pitch = 0.004, 0.02
    for i in range(side):
        for j in range(side):
            offset = (np.array([i, j]) - 0.5 * (side - 1)) * pitch
            builder.add_shape_sphere(
                -1,
                xform=wp.transform(wp.vec3(offset[0], offset[1], radius), wp.quat_identity()),
                radius=radius,
                cfg=cfg,
            )
    half = 0.5 * pitch * side
    plate = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 2.0 * radius + 0.004), wp.quat_identity()))
    builder.add_shape_box(
        plate, hx=half, hy=half, hz=0.004, cfg=newton.ModelBuilder.ShapeConfig(density=500.0, **_material())
    )
    top = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 2.0 * radius + 0.0125), wp.quat_identity()))
    builder.add_shape_box(
        top, hx=0.01, hy=0.01, hz=0.004, cfg=newton.ModelBuilder.ShapeConfig(density=500.0, **_material())
    )
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=64)
    contacts = pipeline.contacts()
    solver = SolverFeatherPGS(model, pgs_iterations=16, dense_max_constraints=16, mf_max_constraints=128)
    state, next_state, control = model.state(), model.state(), model.control()
    heights, speeds = [], []
    for _ in range(steps):
        state.clear_forces()
        pipeline.collide(state, contacts)
        solver.step(state, next_state, control, contacts, 1.0 / 800.0)
        state, next_state = next_state, state
        heights.append(float(state.body_q.numpy()[plate, 2]))
        speeds.append(float(np.abs(state.body_qd.numpy()[plate]).max()))
    return np.array(heights), np.array(speeds), solver


def _material():
    return {"mu": 0.75, "ke": PART[0], "kd": PART[1]}


class TestRegionBlock(unittest.TestCase):
    def test_an_over_determined_group_is_one_block_and_rests(self):
        """Nine spheres under the plate (more closing regions than its six coordinates): one block, a quiet rest."""
        heights, speeds, solver = resting_plate(3)
        regions = solver.sweep.regions
        self.assertIsNotNone(regions)
        pairs = regions.blk_pair.numpy()
        self.assertTrue(((pairs[:, 0] >= 0) & (pairs[:, 1] < 0)).any(), "the plate's group against the spheres")
        self.assertLess(speeds[-50:].max(), 1.0e-3)
        self.assertLess(np.ptp(heights[-50:]), 1.0e-5)

    def test_a_group_within_its_coordinates_keeps_its_regions(self):
        """Four spheres under the plate: its regions stay regions; the same rest."""
        heights, speeds, solver = resting_plate(2)
        pairs = solver.sweep.regions.blk_pair.numpy()
        self.assertFalse((pairs[:, 0] >= 0).any())
        self.assertLess(speeds[-50:].max(), 1.0e-3)
        self.assertLess(np.ptp(heights[-50:]), 1.0e-5)


if __name__ == "__main__":
    unittest.main()
