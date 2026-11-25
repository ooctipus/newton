# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for FeatherPGS contact row allocation under capacity overflow and in the frame's order.

Each world's contact rows are reserved in the frame's order (its pair and contact keys), so
the rows, and the Gauss-Seidel solve that visits them, do not depend on the order in which
the collision's threads wrote the contacts. When a reservation does not fit, the row count
must still bound every accepted contact: a row that is built but lies at or past the count
is never solved, yet friction patch links can chain into it, and the patch-chain walk in
the solve kernels then never terminates.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import field_group
from newton._src.solvers.feather_pgs.contact_input import ContactInput
from newton.solvers import SolverFeatherPGS
from newton.tests.feather_pgs_law_harness import (
    allocate_contacts,
    anchored_patches,
    finalize_counts,
    row_lines,
    row_order,
)
from newton.tests.test_feather_pgs_sparse_solver import _build_model

_UNBOUNDED = 2**31 - 1

# The matrix-free route of test_feather_pgs_solve_readout (split mode on CPU): per world two
# contacting tripods beside two free spheres, one sphere deep in a foot, whose dense and
# matrix-free contact rows an 8-sweep Gauss-Seidel solve visits in slot order.
_ROUTE_OPTIONS = {
    "enable_joint_limits": True,
    "dense_max_constraints": 96,
    "mf_max_constraints": 64,
    "pgs_iterations": 8,
}
_ROW_PATHS = (0, 1)  # dense, matrix-free


def _allocate_overflowing_contacts(device, contact_count, row_capacity):
    """Route ``contact_count`` contacts into one world with ``row_capacity`` rows.

    Contacts alternate between penetrating (three rows: normal plus two friction) and
    separated within the gap (one speculative normal row). Mixed reservation sizes are
    what let a rolled-back counter re-admit a contact behind a rejected one.
    """
    contact_shape0 = wp.zeros((contact_count,), dtype=wp.int32, device=device)
    contact_shape1 = wp.full((contact_count,), -1, dtype=wp.int32, device=device)
    points = np.zeros((contact_count, 3), dtype=np.float32)
    points[:, 2] = np.where(np.arange(contact_count) % 2 == 0, -0.01, 0.01)
    contact_point0 = wp.array(points, dtype=wp.vec3, device=device)
    contact_point1 = wp.zeros((contact_count,), dtype=wp.vec3, device=device)
    contact_normal = wp.array(
        np.tile([0.0, 0.0, -1.0], (contact_count, 1)).astype(np.float32), dtype=wp.vec3, device=device
    )
    contact_slot = wp.full((contact_count,), -1, dtype=wp.int32, device=device)
    contact_path = wp.full((contact_count,), -1, dtype=wp.int32, device=device)
    contact_world = wp.zeros((contact_count,), dtype=wp.int32, device=device)
    slots_needed = wp.zeros((contact_count,), dtype=wp.int32, device=device)
    counter = wp.zeros((1,), dtype=wp.int32, device=device)
    dropped = wp.zeros((1,), dtype=wp.int32, device=device)
    first_rejected = wp.full((1,), _UNBOUNDED, dtype=wp.int32, device=device)
    count = wp.zeros((1,), dtype=wp.int32, device=device)
    allocate_contacts(
        [
            field_group.upload(
                ContactInput,
                device,
                count=wp.array([contact_count], dtype=wp.int32, device=device),
                shape0=contact_shape0,
                shape1=contact_shape1,
                point0=contact_point0,
                point1=contact_point1,
                normal=contact_normal,
                margin0=wp.zeros((contact_count,), dtype=wp.float32, device=device),
                margin1=wp.zeros((contact_count,), dtype=wp.float32, device=device),
                body_q=wp.array([wp.transform_identity()], dtype=wp.transform, device=device),
                shape_body=wp.zeros((1,), dtype=wp.int32, device=device),
                response_dofs=wp.array([6], dtype=wp.int32, device=device),
                world=contact_world,
                slot=contact_slot,
                art0=wp.full((contact_count,), -1, dtype=wp.int32, device=device),
                art1=wp.full((contact_count,), -1, dtype=wp.int32, device=device),
                path=contact_path,
                slots_needed=slots_needed,
                first=row_lines(device),
            ),
            wp.zeros((1,), dtype=wp.int32, device=device),
            wp.zeros((1,), dtype=wp.int32, device=device),
            wp.zeros((1,), dtype=wp.int32, device=device),
            wp.ones((1,), dtype=wp.int32, device=device),
            wp.zeros((1,), dtype=wp.int32, device=device),
            0,
            0.0,
            0.0,
            0.0,
            row_capacity,
            row_capacity,
            1,
            0.0,
            1,
            anchored_patches(contact_count, device),
            wp.empty(0, dtype=wp.spatial_vector, device=device),
            wp.empty(0, dtype=wp.int32, device=device),
            counter,
            wp.zeros((1,), dtype=wp.int32, device=device),
            dropped,
            wp.zeros((1,), dtype=wp.int32, device=device),
            first_rejected,
            wp.full((1,), _UNBOUNDED, dtype=wp.int32, device=device),
            wp.zeros((0, 0), dtype=wp.int32, device=device),
        ],
        device,
    )
    # The dense family's finalizer, as the solver launches it for these rows.
    wp.launch(
        finalize_counts,
        dim=1,
        inputs=[counter, row_capacity, 0, first_rejected],
        outputs=[count, wp.zeros(2, dtype=wp.int32, device=device)],
        device=device,
    )
    return (
        contact_slot.numpy(),
        slots_needed.numpy(),
        int(count.numpy()[0]),
        int(counter.numpy()[0]),
        int(dropped.numpy()[0]),
    )


def _step_tripods(device, *, steps=4, sorted_contacts=False, substeps=1, **options):
    """Step the tripod worlds, the collision pipeline matching its contacts (``sorted_contacts``) or not: either
    leaves them in its one order. Each collision pass serves ``substeps`` solver steps, whose later builds keep its
    contact frame.

    ``options`` override the route's solver options. Returns, per solver step, the live contacts' world,
    path and slot at their frame positions, whether the build kept the last build's frame, the worlds' dense and
    matrix-free row counts, contact rows' start and dense overflow, and the output joint state.
    """
    model = _build_model(free_count=2, robot_count=2, device=device)
    solver = SolverFeatherPGS(model, pins=SolverFeatherPGS.Pins(storage="dense"), **{**_ROUTE_OPTIONS, **options})
    state_in, state_out = model.state(), model.state()
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        reduce_contacts=False,
        contact_matching="latest" if sorted_contacts else "disabled",
        growable_contacts=True,
    )
    contacts = pipeline.contacts()
    control = model.control()
    records = []
    for step in range(steps * substeps):
        if step % substeps == 0:
            pipeline.collide(state_in, contacts)
        state_in.clear_forces()
        solver.step(state_in, state_out, control, contacts, 1.0 / 240.0)
        live = int(contacts.rigid_contact_count.numpy()[0])
        # On CPU numpy() views the solver's buffers, which the next step overwrites.
        record = {
            name: getattr(solver.placement, f"contact_{name}").numpy()[:live].copy()
            for name in ("world", "path", "slot")
        }
        record["pair_keys"] = solver.laws.friction_patches.current.keys.numpy()[:live].copy()
        record["kept"] = int(solver.laws.friction_patches.kept_frame().numpy()[0])
        # Whether the step kept the collision pass's contact set, and which worlds kept their contacts' slots.
        record["pass_kept"] = int(solver.placement.pass_words.numpy()[0])
        record["held"] = solver.placement.world_held.numpy().copy()
        # Each contact's trees with response coordinates (row_records.row_sides), which join its world's islands.
        responds = solver.forest.articulation_response_dof_count.numpy() > 0
        arts = np.stack(
            [solver.placement.contact_art_a.numpy()[:live], solver.placement.contact_art_b.numpy()[:live]], 1
        )
        trees = np.where((arts >= 0) & responds[np.maximum(arts, 0)], arts, -1)
        trees = np.where((trees[:, :1] == trees[:, 1:]), np.stack([trees[:, 0], np.full(live, -1)], 1), trees)
        record["sides"] = np.where(
            (trees >= 0).all(1, keepdims=True), np.sort(trees, 1), np.sort(trees, 1)[:, ::-1]
        ).copy()
        # Whether each contact has an immovable side (a shape without a body): those take a world's rows first.
        shape_body = model.shape_body.numpy()
        sides = [getattr(contacts, f"rigid_contact_{name}").numpy()[:live] for name in ("shape0", "shape1")]
        record["immovable"] = np.logical_or.reduce(
            [(side < 0) | (shape_body[np.maximum(side, 0)] < 0) for side in sides]
        )
        # Each family's contact rows in the world's line: the free bodies' from the line's body base.
        base = solver.placement.body_base.numpy()[solver.placement.dense_first.numpy()]
        record.update(
            counts=(
                solver.placement.constraint_count.numpy().copy(),
                base + solver.placement.mf_constraint_count.numpy(),
            ),
            dense_overflow=solver.placement.dense_first_rejected_slot.numpy() < solver.placement.slot_counter.numpy(),
            starts=(
                solver.placement.topology_rows.numpy().copy(),
                np.full(model.world_count, base, dtype=np.int32),
            ),
            joint_q=state_out.joint_q.numpy().copy(),
            joint_qd=state_out.joint_qd.numpy().copy(),
        )
        records.append(record)
        state_in, state_out = state_out, state_in
    return records


def _family_contacts(record, path, world):
    """Frame positions of the contacts that hold rows of ``path`` in ``world`` in their rows' order: those with an
    immovable side first, then the others, each in the world's row order over the body pairs of its contacts, admitted
    or not, by their islands (:func:`.feather_pgs_law_harness.row_order`)."""
    run = np.flatnonzero(record["world"] == world)
    order = row_order(run, record["pair_keys"][run], record["sides"][run])
    rank = np.empty(len(record["world"]), dtype=np.int64)
    rank[order] = np.arange(len(order))
    contacts = np.flatnonzero((record["path"] == path) & (record["world"] == world) & (record["slot"] >= 0))
    contacts = contacts[np.argsort(rank[contacts], kind="stable")]
    return contacts[np.argsort(~record["immovable"][contacts], kind="stable")]


class TestFeatherPGSContactRowOrder(unittest.TestCase):
    """Contact rows follow the frame's order, whatever order the collision wrote the contacts in."""

    def _check_collision_order_changes_nothing(self, device):
        """Step the tripods with contacts sorted by key and in the writers' order; require the same rows and states.

        The two pipelines write the same contacts in different orders. Both runs must place every world's dense and
        matrix-free contact rows in the frame's order and step bitwise alike.
        """
        sorted_runs = _step_tripods(device, sorted_contacts=True)
        written_runs = _step_tripods(device)
        for step, (expected, actual) in enumerate(zip(sorted_runs, written_runs, strict=True)):
            for name in ("world", "path", "slot"):
                np.testing.assert_array_equal(actual[name], expected[name], err_msg=f"contact {name} at step {step}")
            for path in _ROW_PATHS:
                for world in np.unique(expected["world"][expected["path"] == path]):
                    slots = expected["slot"][_family_contacts(expected, path, world)]
                    self.assertTrue(
                        np.all(np.diff(slots) > 0),
                        f"step {step}: rows of path {path} in world {world} leave the reservation order: {slots}",
                    )
            for name in ("joint_q", "joint_qd"):
                np.testing.assert_array_equal(actual[name], expected[name], err_msg=f"{name} at step {step}")

    def test_collision_order_changes_nothing_cpu(self):
        """Order the contact rows by the frame on CPU, where the two pipelines write their contacts differently."""
        self._check_collision_order_changes_nothing(wp.get_device("cpu"))

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA collision order")
    def test_collision_order_changes_nothing_cuda(self):
        """Order the contact rows by the frame whatever order the CUDA collision's threads write the contacts in."""
        self._check_collision_order_changes_nothing(wp.get_device("cuda:0"))

    def test_overflowing_family_keeps_the_frame_order_cpu(self):
        """Keep both row families in the frame's order when the dense family overflows.

        With 40 dense rows the tripods' worlds overflow their dense rows while their matrix-free
        rows fit. Each family's accepted rows tile the world's rows up to the finalized count, in
        the frame's order (one step: the next fails closed on the dropped rows).
        """
        records = _step_tripods(wp.get_device("cpu"), steps=1, dense_max_constraints=40)
        for step, record in enumerate(records):
            for path in _ROW_PATHS:
                counts, starts = record["counts"][path], record["starts"][path]
                for world in np.unique(record["world"][record["path"] == path]):
                    slots = record["slot"][_family_contacts(record, path, world)]
                    needed = 3  # Every routed contact of the scene takes a normal and two friction rows.
                    rows = np.sort(np.concatenate([np.arange(slot, slot + needed) for slot in slots]))
                    np.testing.assert_array_equal(
                        rows, np.arange(starts[world], counts[world]), err_msg=f"step {step} path {path}"
                    )
                    self.assertTrue(np.all(np.diff(slots) > 0), f"step {step}: {slots}")
            self.assertTrue(np.all(record["dense_overflow"]), f"step {step}: dense rows did not overflow")
            self.assertTrue(np.isfinite(record["joint_qd"]).all())


_HALF = 0.05
_SINK = 0.003
_GAP = 0.0003


def _pinch_scene(device, worlds: int, resting: bool, cover: bool):
    """Per world: a fixed-base arm, a stacked pair of boxes, and a box sunk ``_SINK`` into the ground under a static
    slab ``_GAP`` above it, so its depenetration and the slab ask more than the box can give (a squeeze none of their
    orders satisfies); after them, in the same world and touching none of them, a box resting on the ground
    (``resting``) and a static slab above it (``cover``). Returns the model and each world's first bodies (those of
    every variant) with its sunk box."""
    builder = newton.ModelBuilder()
    cfg = builder.default_shape_cfg.copy()
    cfg.mu = 0.6
    builder.add_ground_plane(cfg=cfg)
    kept, sunk_boxes = [], []
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
        first = base
        for z in (_HALF, 3.0 * _HALF):
            box = builder.add_body(xform=wp.transform((1.0, 0.0, z), wp.quat_identity()))
            builder.add_shape_box(box, hx=_HALF, hy=_HALF, hz=_HALF, cfg=cfg)
        sunk = builder.add_body(xform=wp.transform((3.0, 0.0, _HALF - _SINK), wp.quat_identity()))
        builder.add_shape_box(sunk, hx=_HALF, hy=_HALF, hz=_HALF, cfg=cfg)
        slab = wp.transform((3.0, 0.0, 2.0 * _HALF - _SINK + _GAP + 0.01), wp.quat_identity())
        builder.add_shape_box(-1, xform=slab, hx=0.2, hy=0.2, hz=0.01, cfg=cfg)
        kept.extend(range(first, sunk + 1))
        sunk_boxes.append(sunk)
        if resting:
            box = builder.add_body(xform=wp.transform((4.0, 0.0, _HALF), wp.quat_identity()))
            builder.add_shape_box(box, hx=_HALF, hy=_HALF, hz=_HALF, cfg=cfg)
        if cover:
            lid = wp.transform((4.0, 0.0, 2.0 * _HALF + 0.001 + 0.01), wp.quat_identity())
            builder.add_shape_box(-1, xform=lid, hx=0.2, hy=0.2, hz=0.01, cfg=cfg)
        builder.end_world()
    return builder.finalize(device=device), np.array(kept), sunk_boxes


def _settle_pinch(model, steps: int = 12) -> np.ndarray:
    """Step the scene from rest on the free bodies' regions (the general sweep's dense storage); each step's poses."""
    solver = SolverFeatherPGS(
        model, pins=SolverFeatherPGS.Pins(storage="dense", tiers=0), pgs_iterations=8, dense_max_constraints=64
    )
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_in, state_out = model.state(), model.state()
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
    control = model.control()
    poses = []
    for _ in range(steps):
        pipeline.collide(state_in, contacts)
        state_in.clear_forces()
        solver.step(state_in, state_out, control, contacts, 1.0 / 60.0)
        state_in, state_out = state_out, state_in
        poses.append(state_in.body_q.numpy().copy())
    return np.array(poses)


class TestFeatherPGSRowIslands(unittest.TestCase):
    """A world's row order is its islands': bodies that never touch keep each other's motion bit for bit."""

    def _check(self, device, worlds: int):
        """Step the pinch scene with and without a resting box and a slab over it, in every combination; the first
        bodies' motion must be the same bits in each. The order of a body's rows was once a stride over every pair of
        its world, so the resting box and its slab, a pair more, reordered the sunk box's rows: its squeeze resolved
        the other way and it rose 2.5 mm into its slab."""
        runs = {}
        for resting in (False, True):
            for cover in (False, True):
                model, kept, sunk_boxes = _pinch_scene(device, worlds, resting, cover)
                runs[resting, cover] = _settle_pinch(model)[:, kept]
        base = runs[False, False]
        for variant, poses in runs.items():
            np.testing.assert_array_equal(poses, base, err_msg=f"resting box, cover: {variant}")
        # The squeeze resolves at the slab, frame order (the ground's pair first) visiting the slab's last.
        model, kept, sunk_boxes = _pinch_scene(device, worlds, False, False)
        index = {body: k for k, body in enumerate(kept)}
        slab_bottom = 2.0 * _HALF - _SINK + _GAP
        for sunk in sunk_boxes:
            self.assertLess(float((base[:, index[sunk], 2] + _HALF - slab_bottom).max()), 1.0e-6)

    def _check_kept_frame(self, device):
        """Two solver steps a collision pass: the second build keeps the frame, and its reservation reads the islands'
        order the first left (the order is the frame's pairs' alone); every step's rows follow the reference order."""
        records = _step_tripods(device, steps=3, substeps=2)
        self.assertEqual([record["kept"] for record in records], [0, 1] * 3)
        for step, record in enumerate(records):
            for path in _ROW_PATHS:
                for world in np.unique(record["world"][record["path"] == path]):
                    slots = record["slot"][_family_contacts(record, path, world)]
                    self.assertTrue(np.all(np.diff(slots) > 0), f"step {step}, path {path}, world {world}: {slots}")

    def _check_pass_keeps_its_rows(self, device):
        """Three solver steps a collision pass: the contact set's work (admission, reservation, units) is the pass's.
        The later steps keep the frame, and each world whose topology rows did not move keeps every contact's path and
        slot and its rows; the rows then follow the reference order as on the pass's first step."""
        substeps = 3
        records = _step_tripods(device, steps=3, substeps=substeps)
        self.assertEqual([record["pass_kept"] for record in records], [0, 1, 1] * 3)
        held = 0
        for step, record in enumerate(records):
            if not record["pass_kept"]:
                self.assertFalse(record["held"].any(), f"step {step}")
                continue
            first = records[step - step % substeps]
            for world in np.flatnonzero(record["held"]):
                held += 1
                mine = first["world"] == world
                for name in ("path", "slot"):
                    np.testing.assert_array_equal(record[name][mine], first[name][mine], f"step {step}, {name}")
                for family in range(2):
                    self.assertEqual(record["counts"][family][world], first["counts"][family][world])
                    self.assertEqual(record["starts"][family][world], first["starts"][family][world])
        self.assertGreater(held, 0)

    def test_a_collision_pass_keeps_its_rows_cpu(self):
        self._check_pass_keeps_its_rows(wp.get_device("cpu"))

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA")
    def test_a_collision_pass_keeps_its_rows_cuda(self):
        self._check_pass_keeps_its_rows(wp.get_device("cuda:0"))

    def test_a_kept_frame_keeps_its_islands_order_cpu(self):
        self._check_kept_frame(wp.get_device("cpu"))

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA")
    def test_a_kept_frame_keeps_its_islands_order_cuda(self):
        self._check_kept_frame(wp.get_device("cuda:0"))

    def test_bodies_that_never_touch_keep_their_motion_cpu(self):
        self._check(wp.get_device("cpu"), worlds=1)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA")
    def test_bodies_that_never_touch_keep_their_motion_cuda(self):
        # From 1,024 worlds the free bodies' contacts are regions on the GPU (no world takes the colored mapping).
        self._check(wp.get_device("cuda:0"), worlds=1024)


class TestFeatherPGSRowAllocationCPU(unittest.TestCase):
    def test_overflowing_allocation_launches_and_bounds_rows_cpu(self):
        """Launch the overflowing allocation on CPU, so its operand list is checked without a GPU.

        The CPU's one lane reserves the world's contacts one at a time; the same invariants hold.
        """
        slot, needed, count, counter, dropped = _allocate_overflowing_contacts(wp.get_device("cpu"), 300, 48)
        accepted = slot >= 0
        self.assertTrue(accepted.any())
        self.assertGreater(dropped, 0)
        self.assertLessEqual(count, 48)
        self.assertEqual(counter, int(needed[accepted].sum()) + dropped)
        self.assertEqual(int(np.sum(needed[accepted])), count)
        self.assertLessEqual(int(np.max(slot[accepted] + needed[accepted])), count)


@unittest.skipUnless(wp.is_cuda_available(), "a warp's lanes reserve together on CUDA")
class TestFeatherPGSRowAllocation(unittest.TestCase):
    def test_overflow_keeps_every_accepted_contact_below_the_count(self):
        """No accepted contact may hold rows at or past the finalized count.

        A counter rolled back for a rejected reservation once left a contact accepted at slot
        s past the final count, its rows built but never solved while patch links still chained
        into them. With 3000 mixed one- and three-row contacts into 48 rows nearly every
        reservation is rejected.
        """
        device = wp.get_device()
        worst_excess = 0
        for _ in range(40):
            slot, needed, count, counter, dropped = _allocate_overflowing_contacts(device, 3000, 48)
            accepted = slot >= 0
            self.assertLessEqual(count, 48)
            self.assertEqual(
                counter,
                int(needed[accepted].sum()) + dropped,
                "counter must equal accepted rows plus rejected reservations",
            )
            excess = int(np.max(slot[accepted] + needed[accepted])) - count if accepted.any() else 0
            worst_excess = max(worst_excess, excess)
            self.assertEqual(int(np.sum(needed[accepted])), count, "accepted rows must fill exactly [0, count)")
        self.assertLessEqual(worst_excess, 0, f"accepted rows reach {worst_excess} past the row count")


if __name__ == "__main__":
    unittest.main()
