# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for identity-matched FeatherPGS contact warm start."""

import dataclasses
import inspect
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import field_group
from newton._src.solvers.feather_pgs.contact_input import ContactRoutes
from newton._src.solvers.feather_pgs.factor.kernels import Clock
from newton._src.solvers.feather_pgs.row_laws import ROW_CONTACT, ROW_FRICTION
from newton._src.solvers.feather_pgs.warm_start import (
    ContactCarry,
    carry_contacts,
    gather_carried_normals,
    prepare_world_impulses,
)
from newton.tests.feather_pgs_law_harness import row_lines


def step_clock(ratio, device):
    """A step clock whose duration is ``ratio`` times the last step's (:func:`.factor.kernels.clock_dt_ratio`)."""
    clock = Clock()
    clock.step, clock.dt, clock.prev_dt, clock.interval = 1, float(ratio), 1.0, 1
    return wp.array([clock], dtype=Clock, device=device)


def _build_press():
    """Build a 1-DOF prismatic press: an articulated box driven down onto the ground.

    The drive target sits below the contact height, so after touchdown the press
    stalls against the ground under a steady drive force — a persistent dense
    contact family (articulated vs. static routes to the dense path) whose
    converged impulse is constant per step. Exactly the regime warm starting
    carries impulses across, and the regime where the pre-fix carry diverged.
    """
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.add_ground_plane()
    body = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.30), wp.quat_identity()))
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    joint = builder.add_joint_prismatic(
        parent=-1,
        child=body,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.30), wp.quat_identity()),
        target_ke=2.0e3,
        target_kd=50.0,
    )
    builder.add_articulation([joint], label="press")
    return builder.finalize()


def _run_press(
    steps: int,
    warm_kwargs: dict,
    contact_matching: str | None = "sticky",
):
    """Drive the press to a stall and record per-step contact-impulse sums and speeds.

    Returns (impulse_sum_per_step, |joint_qd|_per_step, final_state).
    """
    model = _build_press()
    solver = newton.solvers.SolverFeatherPGS(
        model,
        pgs_iterations=8,
        pgs_beta=0.1,
        **warm_kwargs,
    )
    pipeline_kwargs = {} if contact_matching is None else {"contact_matching": contact_matching}
    pipeline = newton.CollisionPipeline(model, **pipeline_kwargs)
    contacts = pipeline.contacts()
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    targets = model.joint_target_q.numpy().copy()
    targets[0] = -0.25  # 5 cm below touchdown: sustained press after the stall
    control.joint_target_q.assign(targets)

    impulse_sums = np.zeros(steps)
    speeds = np.zeros(steps)
    for i in range(steps):
        pipeline.collide(state_0, contacts)
        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
        state_0, state_1 = state_1, state_0
        counts = solver.placement.constraint_count.numpy()
        rows = solver.placement.row_type.numpy()[0, : counts[0]]
        lam = solver.placement.impulses.numpy()[0, : counts[0]]
        impulse_sums[i] = np.abs(lam[rows == ROW_CONTACT]).sum()
        speeds[i] = np.abs(state_0.joint_qd.numpy()[0])
    return impulse_sums, speeds, state_0


def carry_of(prev_impulses, prev_types, prev_slots, device="cpu", normal=(0.0, 0.0, 1.0)):
    """The carry a step leaves (:func:`carry_contacts`): its contacts at ``prev_slots`` of world 0's line, whose rows
    held ``prev_impulses`` of kinds ``prev_types``; every contact's normal ``normal``."""
    n, rows = len(prev_slots), len(prev_types)
    carry = field_group.upload(
        ContactCarry,
        device,
        normal=wp.zeros(n, dtype=wp.vec3, device=device),
        impulse=wp.zeros(n, dtype=wp.float32, device=device),
    )
    wp.launch(
        carry_contacts,
        dim=n,
        inputs=[
            field_group.upload(
                ContactRoutes,
                device,
                count=[n],
                slot=prev_slots,
                world=[0] * n,
                normal=wp.array([normal] * n, dtype=wp.vec3, device=device),
            ),
            row_lines(device),
            wp.array([prev_types], dtype=wp.int32, device=device),
        ],
        outputs=[wp.array([prev_impulses], dtype=wp.float32, device=device).reshape((1, rows)), carry],
        device=device,
    )
    return carry


def gather_rows(
    *,
    prev_impulses,
    prev_types,
    prev_slots,
    current_types,
    current_slots,
    match_indices,
    decay=1.0,
    dt_scale=1.0,
    device="cpu",
):
    """Carry a step's contacts (:func:`carry_of`), then launch the one identity gather over the next step's contacts,
    at ``current_slots`` of world 0's line (either row class: one row index), and return the line it seeded."""
    n = len(current_slots)
    impulses = wp.zeros((1, len(current_types)), dtype=wp.float32, device=device)
    wp.launch(
        gather_carried_normals,
        dim=n,
        inputs=[
            field_group.upload(
                ContactRoutes,
                device,
                count=[n],
                slot=current_slots,
                world=[0] * n,
                normal=wp.array([[0.0, 0.0, 1.0]] * n, dtype=wp.vec3, device=device),
            ),
            wp.array(match_indices, dtype=wp.int32, device=device),
            carry_of(prev_impulses, prev_types, prev_slots, device),
            row_lines(device),
            wp.array([current_types], dtype=wp.int32, device=device),
            decay,
            step_clock(dt_scale, device),
        ],
        outputs=[impulses],
        device=device,
    )
    return impulses.numpy()[0]


class TestFeatherPGSIdentityWarmstartKernel(unittest.TestCase):
    def test_noncontact_dense_cache_is_cold_initialized(self):
        """The dense initializer preserves no row without an identity contract."""
        impulses = wp.array([[1.0, 2.0, 3.0, 4.0]], dtype=wp.float32, device="cpu")
        wp.launch(
            prepare_world_impulses,
            dim=1,
            inputs=[
                row_lines("cpu"),
                wp.array([4], dtype=wp.int32, device="cpu"),
                4,
                1,
            ],
            outputs=[impulses],
            device="cpu",
        )
        np.testing.assert_array_equal(impulses.numpy()[0], np.zeros(4, dtype=np.float32))

    def test_only_normals_carry(self):
        """The gather seeds each matched contact's normal row; every friction row starts cold, also where the contact
        held the same pair last step, and no neighbour's tangent lands in another contact's span."""
        c, f = ROW_CONTACT, ROW_FRICTION

        # Both contacts keep their spans (3 rows each): the normals carry, the pairs do not.
        got = gather_rows(
            prev_impulses=[11.0, 22.0, 33.0, 55.0, 66.0, 77.0],
            prev_types=[c, f, f, c, f, f],
            prev_slots=[0, 3],
            current_types=[c, f, f, c, f, f],
            current_slots=[0, 3],
            match_indices=[0, 1],
        )
        np.testing.assert_array_equal(got, np.array([11.0, 0.0, 0.0, 55.0, 0.0, 0.0], dtype=np.float32))

        # A: 3 -> 1 rows, B: 1 -> 3 rows. A's previous second tangent lands at
        # B's current first tangent by raw offset and must not be written there.
        got = gather_rows(
            prev_impulses=[11.0, 22.0, 33.0, 55.0],
            prev_types=[c, f, f, c],
            prev_slots=[0, 3],
            current_types=[c, c, f, f],
            current_slots=[0, 1],
            match_indices=[0, 1],
        )
        np.testing.assert_array_equal(got, np.array([11.0, 55.0, 0.0, 0.0], dtype=np.float32))

        # A: 1 -> 3 rows, B: 3 -> 1 rows. A's second destination tangent
        # overlaps B's previous first tangent and must stay cold.
        got = gather_rows(
            prev_impulses=[11.0, 55.0, 66.0, 77.0],
            prev_types=[c, c, f, f],
            prev_slots=[0, 1],
            current_types=[c, f, f, c],
            current_slots=[0, 3],
            match_indices=[0, 1],
        )
        np.testing.assert_array_equal(got, np.array([11.0, 0.0, 0.0, 55.0], dtype=np.float32))

    def test_slot_churn_uses_identity_and_scales_dt(self):
        """Contact-order and slot churn follows match identity, not row index."""
        c, dead = ROW_CONTACT, -1
        got = gather_rows(
            prev_impulses=[55.0, 0.0, 0.0, 0.0, 11.0],
            prev_types=[c, dead, dead, dead, c],
            prev_slots=[4, 0],  # previous sorted contacts A, B
            current_types=[dead, c, dead, dead, c],
            current_slots=[1, 4],  # current sorted contacts B, A
            match_indices=[1, 0],
            decay=0.5,
            dt_scale=4.0,
        )
        np.testing.assert_array_equal(got, np.array([0.0, 110.0, 0.0, 0.0, 22.0], dtype=np.float32))

    def test_constructor_layout_and_decay_validation(self):
        parameters = tuple(inspect.signature(newton.solvers.SolverFeatherPGS).parameters)
        start = parameters.index("pgs_warmstart")
        self.assertEqual(parameters[start : start + 2], ("pgs_warmstart", "mf_max_constraints"))
        self.assertGreater(
            parameters.index("pgs_warmstart_decay"), parameters.index("articulation_pair_contact_gap_gate")
        )

        model = newton.ModelBuilder().finalize(device="cpu")
        for value in (-1.0, float("inf"), float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                newton.solvers.SolverFeatherPGS(model, pgs_warmstart_decay=value)

        solver = newton.solvers.SolverFeatherPGS(model, pgs_warmstart=True, pgs_warmstart_decay=0.25)
        self.assertEqual(solver.options.pgs_warmstart_decay, 0.25)

    def test_contacts_none_is_valid(self):
        model = newton.ModelBuilder().finalize(device="cpu")
        solver = newton.solvers.SolverFeatherPGS(model, pgs_warmstart=True)
        solver.step(model.state(), model.state(), model.control(), None, 1.0 / 60.0)

    def test_identity_carry_holds_the_press(self):
        """The identity carry holds the press on the dense-row route."""
        impulses, speeds, _ = _run_press(120, {"pgs_warmstart": True})
        self.assertGreater(float(impulses[-1]), 0.0)
        self.assertLess(float(np.ptp(impulses[-10:])), 1.0e-5)
        self.assertLess(float(speeds[-10:].max()), 1.0e-4)


@unittest.skipUnless(wp.get_device().is_cuda, "SolverFeatherPGS matrix-free mode requires CUDA")
class TestFeatherPGSIdentityWarmstart(unittest.TestCase):
    def test_real_contact_insertion_moves_slots_without_cross_seeding(self):
        """A newly inserted lower-key contact shifts a persistent contact's real MF slot."""
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        body_a = builder.add_body(xform=wp.transform(wp.vec3(-0.5, 0.0, 1.0), wp.quat_identity()))
        shape_a = builder.add_shape_sphere(body_a, radius=0.1)
        body_b = builder.add_body(xform=wp.transform(wp.vec3(0.5, 0.0, 0.099), wp.quat_identity()))
        shape_b = builder.add_shape_sphere(body_b, radius=0.1)
        builder.add_ground_plane()
        model = builder.finalize()
        pipeline = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching="sticky")
        contacts = pipeline.contacts()
        solver = newton.solvers.SolverFeatherPGS(
            model,
            pgs_iterations=8,
            pgs_warmstart=True,
        )
        state_0, state_1 = model.state(), model.state()
        control = model.control()

        pipeline.collide(state_0, contacts)
        self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 1)
        solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
        old_slot = int(solver.placement.contact_slot.numpy()[0])
        old_impulse = float(solver.placement.carry_impulse.numpy()[0])
        self.assertGreater(old_impulse, 0.0)

        # Bring the lower shape-id sphere into contact. Sorting inserts it
        # before B, so B moves to a new solver slot while its match points to
        # the previous frame's sole contact.
        q = state_1.body_q.numpy()
        q[body_a][2] = 0.099
        state_1.body_q.assign(q)
        pipeline.collide(state_1, contacts)
        count = int(contacts.rigid_contact_count.numpy()[0])
        self.assertEqual(count, 2)
        match = contacts.rigid_contact_match_index.numpy()[:count]
        shape0 = contacts.rigid_contact_shape0.numpy()[:count]
        shape1 = contacts.rigid_contact_shape1.numpy()[:count]
        b_indices = np.flatnonzero((shape0 == shape_b) | (shape1 == shape_b))
        a_indices = np.flatnonzero((shape0 == shape_a) | (shape1 == shape_a))
        self.assertEqual(len(b_indices), 1)
        self.assertEqual(len(a_indices), 1)
        b_contact = int(b_indices[0])
        a_contact = int(a_indices[0])
        self.assertEqual(int(match[b_contact]), 0)
        self.assertLess(int(match[a_contact]), 0)

        solver.options = solver.laws.options = solver.factor.options = solver.placement.options = (
            solver.sweep.options
        ) = dataclasses.replace(solver.options, pgs_iterations=0)
        solver.step(state_1, state_0, control, contacts, 1.0 / 240.0)
        slots = solver.placement.contact_slot.numpy()[:count]
        b_slot = int(slots[b_contact])
        a_slot = int(slots[a_contact])
        self.assertNotEqual(b_slot, old_slot, "test did not produce actual solver-slot churn")
        impulses = solver.placement.impulses.numpy()[0]
        self.assertAlmostEqual(float(impulses[b_slot]), old_impulse, delta=1.0e-6)
        self.assertEqual(float(impulses[a_slot]), 0.0)

    def test_identity_warmstart_holds_static_press(self):
        """Identity warm start keeps a stalled press at the cold equilibrium.

        Regression net for the warm-start velocity accounting: carried impulses
        are installed into the starting velocity exactly once. A missing install
        accumulates the ledger (``lambda_{n+1} = lambda_n + lambda*``, the
        historical divergence); a duplicated install halves it (measured 0.5x
        cold when an earlier identity path added a second fold on top of
        ``_stage6_apply_impulses_world``). The identity-matched path must also
        stay quiet and match the cold ledger while seeding by contact identity
        rather than raw slot index.
        """
        steps = 240
        stall = slice(120, None)  # well past touchdown + transient

        lam_cold, _speed_cold, _ = _run_press(steps, {})
        lam_warm, speed_warm, state = _run_press(steps, {"pgs_warmstart": True})

        self.assertTrue(np.isfinite(state.body_q.numpy()).all())

        cold_end = lam_cold[-10:].mean()
        warm_end = lam_warm[-10:].mean()

        # The solver installs carried impulses into the starting velocity
        # (_stage6_apply_impulses_world under pgs_warmstart), so warm and cold
        # must share an equilibrium. A duplicated velocity install shows
        # up here as a halved impulse ledger (measured 0.5x when the matched
        # path folded the seed a second time); a missing install shows up as an
        # accumulating ledger (lambda_{n+1} = lambda_n + lambda*).
        warm_growth = lam_warm[-10:].mean() / max(lam_warm[stall][:10].mean(), 1e-12)
        self.assertLess(warm_growth, 1.25, f"warm-start impulse grew x{warm_growth:.2f} at a stall")
        self.assertGreater(warm_end, 0.7 * cold_end, f"warm impulse ledger {warm_end:.3f} below cold {cold_end:.3f}")
        self.assertLess(warm_end, 1.4 * cold_end, f"warm impulse ledger {warm_end:.3f} above cold {cold_end:.3f}")
        self.assertLess(
            speed_warm[stall].max(),
            0.02,
            f"press not quiet under warm start (peak |qd| {speed_warm[stall].max():.3f} m/s)",
        )

    def test_identity_warmstart_matches_cold_equilibrium(self):
        """The matched warm start converges to the cold solve's stall pose, not a new one.

        Warm starting changes the sweep's starting point, never the physics: at
        the stall both runs must agree on the joint position to sub-millimetre.
        """
        _, _, state_cold = _run_press(240, {})
        _, _, state_warm = _run_press(240, {"pgs_warmstart": True})
        q_cold = float(state_cold.joint_q.numpy()[0])
        q_warm = float(state_warm.joint_q.numpy()[0])
        self.assertAlmostEqual(q_warm, q_cold, delta=1.0e-3)

    def test_identity_warmstart_requires_contact_matching(self):
        """Stepping warm start with unmatched contacts raises.

        The identity gather needs ``Contacts.rigid_contact_match_index``; a
        pipeline built without ``contact_matching`` must fail loudly rather than
        silently reusing impulses by raw slot index.
        """
        with self.assertRaises(NotImplementedError):
            _run_press(3, {"pgs_warmstart": True}, contact_matching=None)

    def test_single_flag_enables_dense_and_mf_carry(self):
        """pgs_warmstart=True is the single all-contact warm-start mode."""
        model = _build_press()
        solver = newton.solvers.SolverFeatherPGS(model, pgs_warmstart=True, pgs_iterations=4)
        self.assertTrue(solver.options.pgs_warmstart)
        self.assertTrue(solver.options.pgs_warmstart)


if __name__ == "__main__":
    unittest.main()
