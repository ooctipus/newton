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
                position=wp.array(np.arange(contact_count, dtype=np.int32), device=device),
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


def _step_tripods(device, *, steps=4, deterministic=True, **options):
    """Step the tripod worlds, the collision pipeline sorting its contacts (``deterministic``) or not.

    ``options`` override the route's solver options. Returns, per step, the live contacts' world,
    path and slot at their frame positions, the worlds' dense and matrix-free row counts, contact
    rows' start and dense overflow, and the output joint state.
    """
    model = _build_model(free_count=2, robot_count=2, device=device)
    solver = SolverFeatherPGS(model, pins=SolverFeatherPGS.Pins(storage="dense"), **{**_ROUTE_OPTIONS, **options})
    state_in, state_out = model.state(), model.state()
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
    pipeline = newton.CollisionPipeline(
        model, broad_phase="nxn", reduce_contacts=False, deterministic=deterministic, growable_contacts=True
    )
    contacts = pipeline.contacts()
    control = model.control()
    records = []
    for _ in range(steps):
        pipeline.collide(state_in, contacts)
        state_in.clear_forces()
        solver.step(state_in, state_out, control, contacts, 1.0 / 240.0)
        live = int(contacts.rigid_contact_count.numpy()[0])
        # On CPU numpy() views the solver's buffers, which the next step overwrites.
        record = {
            name: getattr(solver.placement, f"contact_{name}").numpy()[:live].copy()
            for name in ("world", "path", "slot")
        }
        # Whether each contact was admitted: the admitted contacts counted before each position, and in all.
        before = solver.placement.contact_admitted_before.numpy()[:live]
        total = int(solver.placement.reserve_totals.numpy()[4, 0])
        record["admitted"] = np.diff(np.append(before, total)) > 0
        record["pair_keys"] = solver.laws.friction_patches.current.keys.numpy()[:live].copy()
        # Whether each contact has an immovable side (a shape without a body): those take a world's rows first.
        shape_body = model.shape_body.numpy()
        sides = [getattr(solver.laws.friction_patches.records, name).numpy()[:live] for name in ("shape0", "shape1")]
        record["immovable"] = np.logical_or.reduce(
            [(side < 0) | (shape_body[np.maximum(side, 0)] < 0) for side in sides]
        )
        # Each family's contact rows in the world's line: the free bodies' from the store's body base.
        base = solver.placement.body_base
        record.update(
            counts=(
                solver.placement.constraint_count.numpy().copy(),
                base + solver.placement.mf_constraint_count.numpy(),
            ),
            dense_overflow=solver.placement.dense_first_rejected_slot.numpy() < solver.placement.slot_counter.numpy(),
            starts=(
                solver.placement.dense_phase_bounds.numpy()[:, 1].copy(),
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
    immovable side first, then the others, each in the world's row order over its admitted contacts' body pairs
    (:func:`.feather_pgs_law_harness.row_order`)."""
    admitted = np.flatnonzero((record["world"] == world) & record["admitted"])
    order = row_order(admitted, record["pair_keys"][admitted])
    rank = np.empty(len(record["world"]), dtype=np.int64)
    rank[order] = np.arange(len(order))
    contacts = np.flatnonzero((record["path"] == path) & (record["world"] == world) & (record["slot"] >= 0))
    contacts = contacts[np.argsort(rank[contacts], kind="stable")]
    return contacts[np.argsort(~record["immovable"][contacts], kind="stable")]


class TestFeatherPGSContactRowOrder(unittest.TestCase):
    """Contact rows follow the frame's order, whatever order the collision wrote the contacts in."""

    def _check_collision_order_changes_nothing(self, device):
        """Step the tripods with a sorting and a non-sorting collision pipeline; require the same rows and states.

        The two pipelines write the same contacts in different orders. Both runs must place every world's dense and
        matrix-free contact rows in the frame's order and step bitwise alike.
        """
        sorted_runs = _step_tripods(device, deterministic=True)
        written_runs = _step_tripods(device, deterministic=False)
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
        the frame's order.
        """
        records = _step_tripods(wp.get_device("cpu"), steps=2, dense_max_constraints=40)
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
