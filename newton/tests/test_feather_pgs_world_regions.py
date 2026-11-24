# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""A world's regions are its own (:class:`.region_sweep.Regions`): every region in a world's span is between that
world's free bodies, also where a step's listing does not fit the stream.

Worlds never interact, and a region in another world's span is swept by that world's lanes beside its own world's, a
race. The places of a listing are taken as worlds arrive, so a place a world takes held another world's region in an
earlier listing. A listing that does not fit (:func:`.region_sweep.place_bodies`) must therefore list no region at
all: the worlds that took their places before the stream ran out would otherwise hold an older listing's regions.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.solver_feather_pgs import SolverFeatherPGS

HALF = 0.05
DT = 1.0 / 240.0


def _pairs(device, worlds: int):
    """Per world a pair of boxes: on the ground, one on the other, in the odd worlds; one beside the other 1 m up in
    the even worlds, which :func:`_set_down` stacks on the ground."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    for world in range(worlds):
        builder.begin_world()
        for k in range(2):
            if world % 2:
                p = (0.0, 0.0, HALF + k * (2.0 * HALF + 0.0005))
            else:
                p = (k * 0.5, 0.0, 1.0)
            body = builder.add_body(xform=wp.transform(p, wp.quat_identity()))
            builder.add_shape_box(body, hx=HALF, hy=HALF, hz=HALF)
        builder.end_world()
    return builder.finalize(device=device)


def _set_down(model, state):
    """The even worlds' pairs stacked on the ground, as the odd worlds' are."""
    q = state.joint_q.numpy()
    starts = model.joint_q_start.numpy()
    children = model.joint_child.numpy()
    for joint, body in enumerate(children):
        world, k = divmod(int(body), 2)
        if world % 2 == 0:
            q[starts[joint] : starts[joint] + 7] = (0.0, 0.0, HALF + k * (2.0 * HALF + 0.0005), 0.0, 0.0, 0.0, 1.0)
    state.joint_q.assign(q)
    state.joint_qd.zero_()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)


def _foreign(regions, free_world) -> int:
    """The regions in a world's span (``[world_regions.x, world_order_end)``) with a body of another world, or whose
    head takes another world."""
    span = regions.world_regions.numpy()
    end = regions.world_order_end.numpy()
    order = regions.reg_order.numpy()
    body = regions.reg_body.numpy()
    head = regions.reg_head.numpy()
    foreign = 0
    for world in range(span.shape[0]):
        for g in order[span[world][0] : end[world]]:
            sides = [int(f) for f in body[g] if f >= 0]
            if any(free_world[f] != world for f in sides) or not sides or int(head[g][1]) != world:
                foreign += 1
    return foreign


class TestFeatherPGSWorldRegions(unittest.TestCase):
    def _check(self, device, worlds: int):
        model = _pairs(device, worlds)
        solver = SolverFeatherPGS(model, pgs_iterations=8)
        regions = solver.sweep.regions
        self.assertIsNotNone(regions)
        starts = regions.world_free_start.numpy()
        order = regions.world_free.numpy()
        free_world = np.empty(order.size, dtype=np.int64)
        for world in range(worlds):
            free_world[order[starts[world] : starts[world + 1]]] = world
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        state, out, control = model.state(), model.state(), model.control()

        def step():
            nonlocal state, out
            pipeline.collide(state, contacts)
            solver.step(state, out, control, contacts, DT)
            state, out = out, state

        # The odd worlds' regions take the stream's places.
        step()
        listed = int(regions.cursor.numpy()[0])
        self.assertGreater(listed, 0)
        self.assertEqual(_foreign(regions, free_world), 0)
        # Every world's pair on the ground, and a stream that holds about half the step's regions' candidates.
        _set_down(model, state)
        full = regions.ready
        regions.ready = wp.full(1, int(contacts.rigid_contact_count.numpy()[0]) // 2, dtype=wp.int32, device=device)
        step()
        cursor = regions.cursor.numpy()
        self.assertEqual(int(cursor[3]), 1, "the listing must not fit")
        self.assertEqual(_foreign(regions, free_world), 0)
        # A listing that does not fit lists no region.
        np.testing.assert_array_equal(regions.world_regions.numpy(), 0)
        self.assertEqual(int(cursor[2]) + int(cursor[4]), 0)
        # The stream's whole capacity again: every world lists its own regions.
        regions.ready = full
        step()
        self.assertEqual(int(regions.cursor.numpy()[3]), 0)
        self.assertGreater(int((regions.world_order_end.numpy() - regions.world_regions.numpy()[:, 0]).sum()), 0)
        self.assertEqual(_foreign(regions, free_world), 0)

    def test_short_stream_lists_no_region_cpu(self):
        self._check("cpu", worlds=4)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA")
    def test_short_stream_lists_no_region_cuda(self):
        # From 1,024 worlds the free bodies' contacts are regions on the GPU (no world takes the colored mapping).
        self._check("cuda:0", worlds=1024)


if __name__ == "__main__":
    unittest.main()
