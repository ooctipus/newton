# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for FeatherPGS contact row allocation under capacity overflow and in any reservation order.

Slot counters are shared per world and reserved with atomics. When a reservation does
not fit, the row count must still bound every accepted contact: a row that is built but
lies at or past the count is never solved, yet friction patch links can chain into it,
and the patch-chain walk in the solve kernels then never terminates.

On CUDA the warps of the allocator reach a world's counter in a run-varying order. Under
a deterministic collision pipeline the solver places each world's contact rows in the
sorted contacts' order instead, so two solvers stepping the same contacts sweep the same
rows; the default pipeline keeps the allocator's order and launches nothing more.
"""

import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import field_group, kernels, solver_feather_pgs
from newton._src.solvers.feather_pgs.contact_input import ContactInput
from newton._src.solvers.feather_pgs.friction_patches import FrictionPatches
from newton._src.solvers.feather_pgs.kernels import (
    allocate_world_contact_slots,
    finalize_constraint_counts_with_status,
)
from newton.solvers import SolverFeatherPGS
from newton.tests.test_feather_pgs_sparse_solver import _build_model

_UNBOUNDED = 2**31 - 1

# The matrix-free route of test_feather_pgs_solve_readout (split mode on CPU): per world two
# contacting tripods beside two free spheres, one sphere deep in a foot, whose dense and
# matrix-free contact rows an 8-sweep Gauss-Seidel solve visits in slot order.
_ROUTE_OPTIONS = {
    "friction_anchor_beta": 0.0,
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
    wp.launch(
        allocate_world_contact_slots,
        dim=contact_count,
        inputs=[
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
            ),
            contact_count,
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
            FrictionPatches(),
            wp.empty(0, dtype=wp.int32, device=device),
        ],
        outputs=[
            counter,
            wp.zeros((1,), dtype=wp.int32, device=device),
            dropped,
            wp.zeros((1,), dtype=wp.int32, device=device),
            first_rejected,
            wp.full((1,), _UNBOUNDED, dtype=wp.int32, device=device),
        ],
        device=device,
    )
    # The dense family's finalizer, as the solver launches it for these rows.
    wp.launch(
        finalize_constraint_counts_with_status,
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


def _step_tripods(device, *, threads=None, steps=4, deterministic=True, launched=None, **options):
    """Step the tripod worlds with the contact allocator on ``threads`` threads (``None``: the default).

    ``deterministic`` selects the collision pipeline's mode, ``options`` override the route's
    solver options, and ``launched``, a set, receives every kernel the steps launch. Returns,
    per step, the live contacts' world, path and slot, the worlds' dense and matrix-free row
    counts, contact rows' start and dense overflow, and the output joint state.
    """
    model = _build_model(free_count=2, robot_count=2, device=device)
    with mock.patch.dict(SolverFeatherPGS._route_pins, {"storage": "dense"}):
        solver = SolverFeatherPGS(model, **{**_ROUTE_OPTIONS, **options})
    state_in, state_out = model.state(), model.state()
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", reduce_contacts=False, deterministic=deterministic)
    contacts = pipeline.contacts()
    control = model.control()
    cap = solver_feather_pgs._CONTACT_BUILD_THREAD_CAP if threads is None else threads
    launch = wp.launch

    def record_launch(kernel, *args, **kwargs):
        if launched is not None:
            launched.add(kernel)
        return launch(kernel, *args, **kwargs)

    records = []
    with (
        mock.patch.object(solver_feather_pgs, "_CONTACT_BUILD_THREAD_CAP", cap),
        mock.patch.object(wp, "launch", record_launch),
    ):
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
            record.update(
                counts=(
                    solver.placement.constraint_count.numpy().copy(),
                    solver.placement.mf_constraint_count.numpy().copy(),
                ),
                dense_overflow=solver.placement.dense_first_rejected_slot.numpy()
                < solver.placement.slot_counter.numpy(),
                starts=(
                    solver.placement.dense_phase_bounds.numpy()[:, 1].copy(),
                    np.zeros(model.world_count, dtype=np.int32),
                ),
                joint_q=state_out.joint_q.numpy().copy(),
                joint_qd=state_out.joint_qd.numpy().copy(),
            )
            records.append(record)
            state_in, state_out = state_out, state_in
    return records


def _family_contacts(record, path, world):
    """Indices of the contacts that hold rows of ``path`` in ``world``, ascending."""
    return np.flatnonzero((record["path"] == path) & (record["world"] == world) & (record["slot"] >= 0))


def _thread_order_differs(contacts, threads):
    """Whether ``threads`` threads reserving the ascending ``contacts`` thread by thread take them out of order."""
    return not np.array_equal(sorted(contacts, key=lambda c: (c % threads, c)), contacts)


class TestFeatherPGSContactRowOrder(unittest.TestCase):
    """Sorted contacts' rows follow the contacts' order, whatever order the allocator reserves them in."""

    def _check_reservation_order_changes_nothing(self, device):
        """Step the tripods with the default allocator grid and with four threads; require the same rows and states.

        Four threads reserve a world's contacts thread by thread (on CPU contacts 0, 4, 8, ...
        before 1, 5, 9, ...), out of contact order, as CUDA warps that reach a world's counter
        late do. Both runs must place every world's dense and matrix-free contact rows in
        contact order and step bitwise alike.
        """
        threads = 4
        reference = _step_tripods(device)
        reordered = _step_tripods(device, threads=threads)
        interleaved = False
        for step, (expected, actual) in enumerate(zip(reference, reordered, strict=True)):
            for name in ("world", "path"):
                np.testing.assert_array_equal(actual[name], expected[name], err_msg=f"contact {name} at step {step}")
            for path in _ROW_PATHS:
                for world in np.unique(expected["world"][expected["path"] == path]):
                    contacts = _family_contacts(expected, path, world)
                    # The premise: four threads reserve some world's rows out of contact order.
                    interleaved |= _thread_order_differs(contacts, threads)
                    for record in (expected, actual):
                        slots = record["slot"][contacts]
                        self.assertTrue(
                            np.all(np.diff(slots) > 0),
                            f"step {step}: rows of path {path} in world {world} leave contact order: {slots}",
                        )
            np.testing.assert_array_equal(actual["slot"], expected["slot"], err_msg=f"contact slots at step {step}")
            for name in ("joint_q", "joint_qd"):
                np.testing.assert_array_equal(actual[name], expected[name], err_msg=f"{name} at step {step}")
        self.assertTrue(interleaved, "no world holds contacts that four threads reserve out of order")

    def test_reservation_order_changes_nothing_cpu(self):
        """Order the contact rows of a serial launch that reserves them thread by thread."""
        self._check_reservation_order_changes_nothing(wp.get_device("cpu"))

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA reservations")
    def test_reservation_order_changes_nothing_cuda(self):
        """Order the contact rows whatever order the CUDA allocator's warps reserve them in."""
        self._check_reservation_order_changes_nothing(wp.get_device("cuda:0"))

    def test_default_pipeline_keeps_the_allocator_order_cpu(self):
        """Keep the allocator's rows and launch no run kernel for contacts of the default pipeline.

        Newton promises a reproducible contact order only under ``deterministic=True``; for the
        default pipeline the solver reserves rows as before, so four threads leave some world's
        rows in thread order, out of contact order. The sorted pipeline's steps launch both run
        kernels.
        """
        threads = 4
        launched = set()
        records = _step_tripods(wp.get_device("cpu"), threads=threads, deterministic=False, launched=launched)
        self.assertFalse({kernels.record_world_contact_runs, kernels.place_world_contact_runs} & launched)
        out_of_order = False
        for record in records:
            for path in _ROW_PATHS:
                for world in np.unique(record["world"][record["path"] == path]):
                    contacts = _family_contacts(record, path, world)
                    if _thread_order_differs(contacts, threads):
                        out_of_order |= not np.all(np.diff(record["slot"][contacts]) > 0)
        self.assertTrue(out_of_order, "four threads left every world's rows in contact order")
        launched.clear()
        _step_tripods(wp.get_device("cpu"), steps=1, launched=launched)
        self.assertLessEqual({kernels.record_world_contact_runs, kernels.place_world_contact_runs}, launched)

    def test_overflowing_family_keeps_the_allocator_layout_cpu(self):
        """Keep the allocator's slots for an overflowing row family and place the other one in contact order.

        With 40 dense rows the tripods' worlds overflow their dense rows while their matrix-free
        rows fit. The dense family keeps the allocator's accepted rows, which tile the world's
        rows up to the finalized count; the matrix-free family follows the contacts' order.
        """
        records = _step_tripods(wp.get_device("cpu"), threads=4, steps=2, dense_max_constraints=40)
        for step, record in enumerate(records):
            for path in _ROW_PATHS:
                counts, starts = record["counts"][path], record["starts"][path]
                for world in np.unique(record["world"][record["path"] == path]):
                    contacts = _family_contacts(record, path, world)
                    slots = record["slot"][contacts]
                    needed = 3  # Every routed contact of the scene takes a normal and two friction rows.
                    rows = np.sort(np.concatenate([np.arange(slot, slot + needed) for slot in slots]))
                    np.testing.assert_array_equal(
                        rows, np.arange(starts[world], counts[world]), err_msg=f"step {step} path {path}"
                    )
                    if path == 1:
                        self.assertTrue(np.all(np.diff(slots) > 0), f"step {step}: {slots}")
            self.assertTrue(np.all(record["dense_overflow"]), f"step {step}: dense rows did not overflow")
            self.assertTrue(np.isfinite(record["joint_qd"]).all())


class TestFeatherPGSRowAllocationCPU(unittest.TestCase):
    def test_overflowing_allocation_launches_and_bounds_rows_cpu(self):
        """Launch the overflowing allocation on CPU, so its operand list is checked without a GPU.

        CPU reservations run one at a time, so the race the CUDA test provokes cannot occur;
        the same invariants hold.
        """
        slot, needed, count, counter, dropped = _allocate_overflowing_contacts(wp.get_device("cpu"), 300, 48)
        accepted = slot >= 0
        self.assertTrue(accepted.any())
        self.assertGreater(dropped, 0)
        self.assertLessEqual(count, 48)
        self.assertEqual(counter, int(needed[accepted].sum()) + dropped)
        self.assertEqual(int(np.sum(needed[accepted])), count)
        self.assertLessEqual(int(np.max(slot[accepted] + needed[accepted])), count)


@unittest.skipUnless(wp.is_cuda_available(), "the allocation race needs concurrent GPU reservations")
class TestFeatherPGSRowAllocation(unittest.TestCase):
    def test_overflow_keeps_every_accepted_contact_below_the_count(self):
        """No accepted contact may hold rows at or past the finalized count.

        Rolling the shared counter back for a rejected reservation raced with concurrent
        reservations: a contact accepted at slot s could see the counter end below s, so
        its rows were built but never solved while patch links still chained into them.
        With 3000 mixed one- and three-row contacts into 48 rows nearly every reservation
        is rejected, which makes the race easy to hit.
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
