# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The free bodies' regions that cannot act wait for the sweeps' end (:func:`.region_sweep.region_left`).

A region none of whose candidates closes at the predictor, that carried no load and is no block's, is passed by the
sweeps without a test; where they would stop, the lane (a closed body's) or the world's lanes (the general sweep's region
phase) test it once at the twist they leave, and one that closes joins the sweeps, which go on. The scene lifts boxes
by their own depenetration into partners that hover above them, so regions left out at the predictor close at the
solution: the test holds the published motion to the sweep of every region from the start (the test pin
``left_regions=False``) and to no interpenetration.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import region_sweep
from newton._src.solvers.feather_pgs.solve_policy import QUIET
from newton._src.solvers.feather_pgs.solver_feather_pgs import SolverFeatherPGS
from newton.tests.test_feather_pgs_mixed_tree_free_body import DT

GENERAL = {"storage": "dense", "tiers": 0}
HALF = 0.05
GAP = 0.0003
SINK = 0.003


def _lifted_pairs(device, worlds: int = 2):
    """Per world: a box sunk ``SINK`` into the ground under a box hovering ``GAP`` above it, beside a second such pair
    resting, a lone box at rest, and a closed box sunk ``SINK`` under a static slab ``GAP`` above it. At the predictor
    the upper boxes separate (both boxes of a pair fall alike), so their regions are left out; the lower boxes'
    depenetration then lifts them into the upper ones, and the sunk box into the slab, which the sweeps' end must
    catch. Each world also holds an arm, so its rows are swept with the regions (a world of free bodies alone is the
    same law). Returns the model and each world's (lower, upper) bodies of the sunk pair and its sunk box."""
    builder = newton.ModelBuilder()
    cfg = builder.default_shape_cfg.copy()
    cfg.mu = 0.6
    builder.add_ground_plane(cfg=cfg)
    pairs, sunk_boxes = [], []
    for _ in range(worlds):
        builder.begin_world()
        base = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * 0.01), lock_inertia=True)
        builder.add_shape_box(base, hx=0.04, hy=0.04, hz=0.04, cfg=cfg)
        joints = [builder.add_joint_fixed(-1, base, parent_xform=wp.transform((-1.0, 0.0, 0.3), wp.quat_identity()))]
        link = builder.add_link(mass=0.5, inertia=wp.mat33(np.eye(3) * 0.004), lock_inertia=True)
        builder.add_shape_capsule(link, radius=0.03, half_height=0.1, cfg=cfg)
        joints.append(
            builder.add_joint_revolute(
                base,
                link,
                axis=(1.0, 0.0, 0.0),
                parent_xform=wp.transform((0.0, 0.0, -0.08), wp.quat_identity()),
                child_xform=wp.transform((0.0, 0.0, 0.1), wp.quat_identity()),
            )
        )
        builder.add_articulation(joints)
        for x, sink in ((0.0, SINK), (1.0, 0.0)):
            lower = builder.add_body(xform=wp.transform((x, 0.0, HALF - sink), wp.quat_identity()))
            builder.add_shape_box(lower, hx=HALF, hy=HALF, hz=HALF, cfg=cfg)
            upper = builder.add_body(xform=wp.transform((x, 0.0, 3.0 * HALF + GAP - sink), wp.quat_identity()))
            builder.add_shape_box(upper, hx=HALF, hy=HALF, hz=HALF, cfg=cfg)
            if sink > 0.0:
                pairs.append((lower, upper))
        lone = builder.add_body(xform=wp.transform((2.0, 0.0, HALF), wp.quat_identity()))
        builder.add_shape_box(lone, hx=HALF, hy=HALF, hz=HALF, cfg=cfg)
        sunk = builder.add_body(xform=wp.transform((3.0, 0.0, HALF - SINK), wp.quat_identity()))
        builder.add_shape_box(sunk, hx=HALF, hy=HALF, hz=HALF, cfg=cfg)
        slab = wp.transform((3.0, 0.0, 2.0 * HALF - SINK + GAP + 0.01), wp.quat_identity())
        builder.add_shape_box(-1, xform=slab, hx=0.2, hy=0.2, hz=0.01, cfg=cfg)
        sunk_boxes.append(sunk)
        builder.end_world()
    return builder.finalize(device=device), pairs, sunk_boxes


def _left_counts(solver) -> tuple[int, int]:
    """The step's regions left out at the fill (:func:`.region_sweep.region_left` on the formed candidates) and those
    of them still left out at the sweeps' end (the others joined)."""
    regions = solver.sweep.regions
    count = int(regions.cursor.numpy()[0])
    spans = regions.reg_cand.numpy()[:count]
    kinds = regions.reg_kind.numpy()[:count]
    states = regions.reg_state.numpy()[:count]
    cand = regions.cand.numpy()
    floor = regions.cand_floor.numpy()
    contact = regions.cand_contact.numpy()
    frame = solver.laws.friction_patches.current
    anchored, source = frame.valid.numpy(), frame.source.numpy()
    blocks = int(region_sweep.KIND_BLOCK) | int(region_sweep.KIND_MEMBER)
    predicted = 0
    for g in range(count):
        c0, c1 = spans[g]
        members = contact[c0:c1]
        carried = np.any((anchored[members] != 0) & (source[members] >= 0))
        if (
            (kinds[g] & blocks) == 0
            and not carried
            and np.all(cand[c0:c1, 2] >= -(QUIET * np.abs(cand[c0:c1, 2]) + floor[c0:c1]))
        ):
            predicted += 1
    return predicted, int(np.count_nonzero(states == int(region_sweep.REGION_LEFT)))


def _settle(model, *, left_regions: bool, steps: int, capture: bool = False):
    """Step the model at rest (captured: the first step eagerly, then a graph each way); return the solver, each step's
    poses and the first step's left-out counts (:func:`_left_counts`)."""
    solver = SolverFeatherPGS(
        model,
        pins=SolverFeatherPGS.Pins(**GENERAL, left_regions=left_regions),
        pgs_iterations=8,
        dense_max_constraints=64,
    )
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    states = [model.state() for _ in range(2)]
    newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
    control = model.control()
    poses = []
    counts = None

    def step(src, dst):
        pipeline.collide(src, contacts)
        src.clear_forces()
        solver.step(src, dst, control, contacts, DT)

    graphs = None
    for index in range(steps):
        if graphs is None:
            step(states[0], states[1])
            if counts is None:
                counts = _left_counts(solver)
            if capture:
                graphs = []
                for src, dst in ((states[1], states[0]), (states[0], states[1])):
                    with wp.ScopedCapture(device=model.device) as captured:
                        step(src, dst)
                    graphs.append(captured.graph)
        else:
            wp.capture_launch(graphs[(index - 1) % 2])
        states.reverse()
        poses.append(states[0].body_q.numpy().copy())
    return solver, np.array(poses), counts


RADIUS = 0.1


def _falling_spheres(device, worlds: int, spheres_a_world: int, drop: float):
    """Worlds of free spheres dropped ``drop`` onto the ground far from the origin beside an arm (the walk's tree): a
    contact and a region a sphere, so a sphere count off whole warps leaves the fill's last lane group partial."""
    builder = newton.ModelBuilder()
    cfg = builder.default_shape_cfg.copy()
    cfg.mu = 0.6
    builder.add_ground_plane(cfg=cfg)
    spheres = []
    for _ in range(worlds):
        builder.begin_world()
        base = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * 0.01), lock_inertia=True)
        builder.add_shape_box(base, hx=0.04, hy=0.04, hz=0.04, cfg=cfg)
        joints = [builder.add_joint_fixed(-1, base, parent_xform=wp.transform((-1.0, 0.0, 1.0), wp.quat_identity()))]
        link = builder.add_link(mass=0.5, inertia=wp.mat33(np.eye(3) * 0.004), lock_inertia=True)
        builder.add_shape_capsule(link, radius=0.03, half_height=0.1, cfg=cfg)
        joints.append(
            builder.add_joint_revolute(
                base,
                link,
                axis=(1.0, 0.0, 0.0),
                parent_xform=wp.transform((0.0, 0.0, -0.08), wp.quat_identity()),
                child_xform=wp.transform((0.0, 0.0, 0.1), wp.quat_identity()),
            )
        )
        builder.add_articulation(joints)
        for k in range(spheres_a_world):
            sphere = builder.add_body(
                xform=wp.transform((5.0 + 0.5 * k, 5.0, RADIUS - 0.002 + drop), wp.quat_identity())
            )
            builder.add_shape_sphere(sphere, radius=RADIUS, cfg=cfg)
            spheres.append(sphere)
        builder.end_world()
    return builder.finalize(device=device), np.array(spheres)


class TestFeatherPGSLeftRegions(unittest.TestCase):
    def _check(self, device, worlds: int, capture: bool = False):
        model, pairs, sunk_boxes = _lifted_pairs(device, worlds)
        solver, poses, (predicted, left) = _settle(model, left_regions=True, steps=12, capture=capture)
        full, poses_full, (_predicted_full, left_full) = _settle(model, left_regions=False, steps=12, capture=capture)
        self.assertIsNotNone(solver.sweep.regions)
        self.assertEqual(left_full, 0)
        # Regions were left out at the fill, some joined at the sweeps' end (the lifted pairs and the sunk boxes), and
        # some stayed out (the resting pairs' hovering boxes).
        self.assertGreater(left, 0)
        self.assertGreater(predicted - left, 0)
        # The published motion is the full sweep's to its rounding: every region left out to the end holds nothing
        # at a motion where none of its candidates closes.
        np.testing.assert_allclose(poses, poses_full, rtol=0.0, atol=2.0e-6)
        # No interpenetration: the upper box of each lifted pair stays on the lower one, the sunk box under the slab.
        for lower, upper in pairs:
            gap = poses[:, upper, 2] - poses[:, lower, 2] - 2.0 * HALF
            self.assertGreater(float(gap.min()), -1.0e-4)
        slab_bottom = 2.0 * HALF - SINK + GAP
        for sunk in sunk_boxes:
            self.assertLess(float((poses[:, sunk, 2] + HALF - slab_bottom).max()), 1.0e-4)
        del full

    def test_left_out_regions_join_at_the_sweeps_end_cpu(self):
        self._check("cpu", worlds=2)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA")
    def test_left_out_regions_join_at_the_sweeps_end_cuda(self):
        # From 1,024 worlds the free bodies' contacts are regions on the GPU (no world takes the colored mapping).
        self._check("cuda:0", worlds=1024)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA")
    def test_left_out_regions_join_at_the_sweeps_end_captured(self):
        self._check("cuda:0", worlds=1024, capture=True)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA")
    def test_captured_fill_takes_whole_lane_groups(self):
        # 5,200 one-contact regions once landed (from 1,024 worlds the free bodies' contacts are regions on the GPU):
        # the fill's last lane group of the live contacts holds 16. A replay fills them as the first step did (a group
        # cut short by the live count read lanes that do not run: every candidate's lever was its distance from the
        # world origin, 7-9 m here, and landing spheres spun up and sank 7 mm).
        model, spheres = _falling_spheres("cuda:0", worlds=1040, spheres_a_world=5, drop=0.3)
        solver = SolverFeatherPGS(
            model, pins=SolverFeatherPGS.Pins(**GENERAL), pgs_iterations=8, dense_max_constraints=64
        )
        self.assertIsNotNone(solver.sweep.regions)
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        states = [model.state(), model.state()]
        newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
        control = model.control()

        def step(src, dst):
            pipeline.collide(src, contacts)
            src.clear_forces()
            solver.step(src, dst, control, contacts, 1.0 / 240.0)

        step(states[0], states[1])
        states.reverse()
        graphs = []
        for src, dst in ((states[0], states[1]), (states[1], states[0])):
            with wp.ScopedCapture(device=model.device) as capture:
                step(src, dst)
            graphs.append(capture.graph)
        spin, low = 0.0, np.inf
        for i in range(120):
            wp.capture_launch(graphs[i % 2])
            state = states[(i + 1) % 2]
            spin = max(spin, float(np.linalg.norm(state.body_qd.numpy()[spheres, 3:6], axis=1).max()))
            low = min(low, float(state.body_q.numpy()[spheres, 2].min()))
        self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 5200)
        self.assertLess(spin, 1.0e-3)
        self.assertGreater(low, RADIUS - 1.0e-3)


if __name__ == "__main__":
    unittest.main()
