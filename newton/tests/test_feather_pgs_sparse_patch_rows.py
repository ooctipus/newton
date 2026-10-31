# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Select persistent patch-friction rows at sparse sweep entry, as point friction does."""

import functools
import os
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import sweep_routes_sparse
from newton._src.solvers.feather_pgs.staging import arch_budget
from newton.solvers import SolverFeatherPGS
from newton.tests.feather_pgs_patch_parity import standing_model
from newton.tests.test_feather_pgs_sparse_research_admission import _branched_model
from newton.tests.test_feather_pgs_sparse_solver import _build_model, _make_solver

DT = 1.0 / 400.0
# The ordinary (unfused) contact updates, which these comparisons of the sweep laws hold fixed.
_UNFUSED = {"contact_units": None}


def _admission_probe(**changes):
    """Provide the host fields the sparse sweep policy reads, for an admitted 18-DOF, 256-row route."""
    probe = SimpleNamespace(
        _route_pins={},
        placement=SimpleNamespace(dense_max_constraints=256),
        forest=SimpleNamespace(max_world_dofs=18),
        laws=SimpleNamespace(friction_anchors_enabled=False),
    )
    for name, value in changes.items():
        setattr(probe, name, value)
    return probe


def _admit(probe, *, velocity=False, lanes=0, fits=True):
    return SolverFeatherPGS._admit_sparse_active_rows(
        probe, velocity_candidate=velocity, small_world_lanes=lanes, fits=lambda: fits
    )


class TestFeatherPGSSparsePatchRows(unittest.TestCase):
    def test_point_admission(self):
        """Admit point rows at sweep entry unless velocity limits, a pin or the scan's size keep the full sweep."""
        for pins, velocity, lanes, fits in (
            ({}, False, 0, True),
            ({}, False, 16, True),
            ({"active_rows": False}, False, 0, True),
            ({}, True, 0, True),
            ({}, False, 0, False),
        ):
            expected = not pins and not velocity and fits
            with self.subTest(pins=pins, velocity=velocity, fits=fits):
                probe = _admission_probe(_route_pins=pins)
                active, reason = _admit(probe, velocity=velocity, lanes=lanes, fits=fits)
                self.assertEqual(active, expected)
                if active:
                    self.assertEqual(reason, "Point-contact normal and limit rows are selected at sweep entry.")
                elif not pins and not velocity:
                    self.assertIn("bitsets exceed shared memory", reason)

    def test_patch_admission_follows_small_worlds(self):
        """Select patch rows at sweep entry unless small worlds, a pin or the scan's working set keep the full sweep."""
        for pins, lanes, fits, expected, phrase in (
            ({}, 0, True, True, "ring"),
            ({"active_rows": False}, 0, True, False, "active_rows=False"),
            ({}, 16, True, False, "16-lane"),
            ({}, 8, True, False, "8-lane"),
            ({}, 0, False, False, "bitsets exceed shared memory"),
        ):
            with self.subTest(pins=pins, lanes=lanes, fits=fits):
                probe = _admission_probe(laws=SimpleNamespace(friction_anchors_enabled=True), _route_pins=pins)
                active, reason = _admit(probe, lanes=lanes, fits=fits)
                self.assertEqual(active, expected)
                self.assertIn(phrase, reason)

    def test_small_world_lanes_follow_the_full_sweep(self):
        """Pack small complete sweeps into 8 or 16 lanes where the packed sweep's working set fits the budget."""
        for dofs, rows, environment, slack, expected in (
            (8, 64, {}, 0, 8),
            (9, 32, {}, 0, 16),
            (16, 64, {}, 0, 16),
            (16, 64, {}, -1, 0),
            (17, 32, {}, 0, 0),
            (9, 65, {}, 0, 0),
        ):
            with (
                self.subTest(dofs=dofs, rows=rows, environment=environment, slack=slack),
                mock.patch.dict(os.environ, environment, clear=True),
            ):
                probe = _admission_probe(
                    forest=SimpleNamespace(max_world_dofs=dofs),
                    placement=SimpleNamespace(dense_max_constraints=rows),
                )
                staging = functools.partial(sweep_routes_sparse.sparse_staging, rows, dofs, dofs)
                # A world past 16 DOFs or 64 rows never packs; its budget is moot.
                packable = dofs <= 16 and rows <= 64
                packed = staging(lanes_per_world=8 if dofs <= 8 else 16).bytes if packable else 1 << 20
                lanes = SolverFeatherPGS._sparse_small_world_lanes(probe, staging, packed + slack)
                self.assertEqual(lanes, expected)


def _rollout(solver, model, *, steps, slide_world=None, speed=0.0):
    """Step a standing fixture through the collision pipeline.

    Returns per-step anchor counts, contact counts and (body_q, body_qd, row impulses).
    """
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=64)
    contacts, control = pipeline.contacts(), model.control()
    state, output = model.state(), model.state()
    if slide_world is not None:
        # Slide one world's torso; its feet slide, decelerate and stick.
        qd = state.joint_qd.numpy()
        qd[model.joint_qd_start.numpy()[model.articulation_start.numpy()[slide_world]]] = speed
        state.joint_qd.assign(qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    patches = solver.laws.friction_patches
    anchors, contact_counts, states = [], [], []
    for _ in range(steps):
        pipeline.collide(state, contacts)
        solver.step(state, output, control, contacts, DT)
        state.assign(output)
        stored = int(patches.stored.numpy()[0])
        valid = patches.previous.valid.numpy()[:stored] != 0
        worlds = patches.previous_world.numpy()[:stored]
        anchors.append(np.bincount(worlds[valid], minlength=model.world_count))
        contact_counts.append(int(contacts.rigid_contact_count.numpy()[0]))
        states.append(
            (state.body_q.numpy().copy(), state.body_qd.numpy().copy(), solver.placement.impulses.numpy().copy())
        )
    return np.array(anchors), np.array(contact_counts), states


@unittest.skipUnless(wp.is_cuda_available(), "Sparse admission requires CUDA")
class TestFeatherPGSSparsePatchRowsCuda(unittest.TestCase):
    def test_route_record_reports_linked_active_rows(self):
        """Run patch friction on the linked active-row kernel, and the full sweep when the policy is off."""
        model = _build_model()
        with mock.patch.dict(SolverFeatherPGS._route_pins, _UNFUSED):
            point = _make_solver(model, True, friction_anchor_beta=0.0, dense_max_constraints=96)
            patch_friction = _make_solver(model, True, dense_max_constraints=96)
        with mock.patch.dict(SolverFeatherPGS._route_pins, {**_UNFUSED, "active_rows": False}):
            switched_off = _make_solver(model, True, dense_max_constraints=96)
        records = {
            name: solver.get_path_selection()["routes"]["sparse_mass_matrix"]
            for name, solver in (("point", point), ("patch", patch_friction), ("off", switched_off))
        }
        for name, active, linked in (("point", True, False), ("patch", True, True), ("off", False, False)):
            with self.subTest(name=name):
                record = records[name]
                self.assertEqual(record["size"], 9)
                self.assertEqual(record["solve_lanes"], 32)
                self.assertEqual(record["active_rows"], active)
                self.assertEqual("_active" in record["sweep_kernel"], active)
                self.assertEqual("_linked" in record["sweep_kernel"], linked)
        self.assertIn("ring", records["patch"]["active_rows_reason"])
        self.assertIn("active_rows=False", records["off"]["active_rows_reason"])
        # The pin restores the full sweep's kernel and launch.
        self.assertEqual(
            records["off"]["sweep_kernel"],
            records["patch"]["sweep_kernel"].replace("_active", "", 1).replace("_linked", ""),
        )

    def test_sparse_route_keeps_the_full_sweep_when_bitsets_exceed_shared_memory(self):
        """Keep the full sweep, not ordinary factors, under either friction law when only the row bitsets overflow.

        At the largest row capacity whose 18-DOF full sweep fits the device's static budget, the active
        scan's bitsets do not: the solver admits the route by the sweep it launches, the full ordered one.
        """
        model = _branched_model(4, 3, "cuda:0")
        budget = arch_budget(wp.get_device("cuda:0").arch)
        staging = functools.partial(sweep_routes_sparse.sparse_staging, max_world_dofs=18, max_row_dofs=18)
        low, high = 3, 3 * 4096
        while high - low > 1:
            middle = (low + high) // 2
            low, high = (middle, high) if staging(middle).fits(budget) else (low, middle)
        active = staging(low, active_rows=True)
        self.assertFalse(active.fits(budget))
        options = {
            "enable_joint_velocity_limits": False,
            "dense_max_constraints": low,
        }
        full = staging(low)
        for name, law in (("patch", {}), ("point", {"friction_anchor_beta": 0.0})):
            with self.subTest(law=name):
                solver = SolverFeatherPGS(model, **law, **options)
                self.assertEqual(solver.placement.sparse_mass_matrix_size, 18)
                self.assertFalse(solver._sparse_active_rows)
                self.assertIn("bitsets exceed shared memory", solver._sparse_active_rows_reason)
                self.assertNotIn("_active", solver._pgs_solve_sparse_kernel.key)
                self.assertIn(f"{full.bytes}/{budget}", solver.routes.row.reasons["K1"])

    def test_small_worlds_keep_their_packed_patch_sweep(self):
        """Keep the packed 16-lane full sweep of a 9-DOF, 32-row patch world, bitwise, with active rows pinned off."""
        model = _build_model()
        solvers = {"on": _make_solver(model, True)}
        with mock.patch.dict(SolverFeatherPGS._route_pins, {"active_rows": False}):
            solvers["off"] = _make_solver(model, True)
        for solver in solvers.values():
            self.assertEqual(solver._sparse_solve_lanes, 16)
            self.assertFalse(solver._sparse_active_rows)
            self.assertIn("_lanes16", solver._pgs_solve_sparse_kernel.key)
        self.assertIs(solvers["on"]._pgs_solve_sparse_kernel, solvers["off"]._pgs_solve_sparse_kernel)
        self.assertIn("16-lane", solvers["on"]._sparse_active_rows_reason)
        runs = {name: _rollout(solver, model, steps=24) for name, solver in solvers.items()}
        np.testing.assert_array_equal(runs["on"][0], runs["off"][0])
        self.assertGreater(runs["on"][1].min(), 0)
        for on, off in zip(runs["on"][2], runs["off"][2], strict=True):
            for value_on, value_off in zip(on, off, strict=True):
                np.testing.assert_array_equal(value_on, value_off)

    def test_switched_off_patch_sweep_with_fused_triplets_follows_it(self):
        """Step the switched-off full patch sweep with fused anchor units as it steps without them, to rounding.

        Pinned off the active-row scan, the full ordered sweep solves its anchors as contact units on their Gram
        blocks (in exact arithmetic the ordinary updates); pinned unfused it runs the ordinary updates. A resting and
        a sliding biped world follow each other within the bounds of
        :meth:`test_patch_history_follows_the_full_sweep`, whose two sweeps also differ by their arithmetic order
        only.
        """
        model = standing_model("cuda:0", worlds=2)
        options = {"dense_max_constraints": 96, "pgs_iterations": 8, "enable_joint_limits": False}
        solvers = {}
        for triplets in ("1", "0"):
            pins = {"active_rows": False}
            if triplets == "0":
                pins["contact_units"] = None
            with mock.patch.dict(SolverFeatherPGS._route_pins, pins):
                solvers[triplets] = _make_solver(model, True, **options)
        # The full ordered sweep: no "_active" between the support width and the table.
        self.assertRegex(solvers["1"]._pgs_solve_sparse_kernel.key, r"_\d+_support_table_contact_triplets_linked$")
        self.assertRegex(solvers["0"]._pgs_solve_sparse_kernel.key, r"_\d+_support_table$")
        self.assertFalse(solvers["1"]._sparse_active_rows)
        runs = {name: _rollout(solver, model, steps=120, slide_world=1, speed=0.6) for name, solver in solvers.items()}
        anchors, ordinary_anchors = runs["1"][0], runs["0"][0]
        self.assertTrue(np.any(ordinary_anchors[:, 1] < 4), "The sliding world must release anchors")
        np.testing.assert_array_equal(anchors[:, 0], ordinary_anchors[:, 0])
        mismatched = anchors[:, 1] != ordinary_anchors[:, 1]
        self.assertLessEqual(int(mismatched.sum()), 18, f"steps with different anchors: {np.flatnonzero(mismatched)}")
        self.assertLessEqual(np.abs(anchors[:, 1] - ordinary_anchors[:, 1]).max(), 2)
        states = list(zip(runs["1"][2], runs["0"][2], strict=True))
        # The units must act on this fixture, or the comparison would prove nothing.
        self.assertFalse(all(np.array_equal(unit[1], ordinary[1]) for unit, ordinary in states))
        bodies = model.body_count // model.world_count
        for world, pose_bound, velocity_bound in ((0, 5.0e-5, 1.0e-2), (1, 5.0e-5, 3.0e-2)):
            with self.subTest(world=world):
                body = slice(world * bodies, (world + 1) * bodies)
                pose_gap = max(np.abs(unit[0][body, :3] - ordinary[0][body, :3]).max() for unit, ordinary in states)
                velocity_gap = max(np.abs(unit[1][body] - ordinary[1][body]).max() for unit, ordinary in states)
                self.assertLess(pose_gap, pose_bound)
                self.assertLess(velocity_gap, velocity_bound)

    def test_patch_history_follows_the_full_sweep(self):
        """Keep the anchor-count trajectory of a resting and a sliding world near the full patch sweep's.

        The deferral law changes iterates, not the solved problem: anchors, contacts
        and states follow the full sweep through the slide, the stop and the rest.
        The resting world settles alike: measured on the RTX, equal anchor counts on
        all 240 steps, poses within 4.8e-6 m and velocities within 3.4e-3.

        The sliding world is chaotic: scaling its initial speed by 1 + 1e-7 changes
        the full sweep's own anchor counts on 13 of the 240 steps and its poses by
        5e-4 m. Its bounds therefore come from the law against the full sweep at ten
        such speeds (0.6 m/s scaled by 1 + s, |s| from 1e-7 to 1e-2): anchor counts
        differ on at most 6 steps, by 1, poses by 1.0e-5 m and velocities by 1.14e-2.
        Releases happen on 3 to 28 of the steps, so only their presence is asserted.
        The pair selection by a pair's own normal alone (a law error) differs on 25
        steps, by 3, and moves velocities by 0.79.
        """
        model = standing_model("cuda:0", worlds=2)
        options = {"dense_max_constraints": 96, "pgs_iterations": 8, "enable_joint_limits": False}
        with mock.patch.dict(SolverFeatherPGS._route_pins, _UNFUSED):
            solvers = {"active": _make_solver(model, True, **options)}
        with mock.patch.dict(SolverFeatherPGS._route_pins, {**_UNFUSED, "active_rows": False}):
            solvers["full"] = _make_solver(model, True, **options)
        self.assertTrue(solvers["active"]._sparse_active_rows)
        self.assertFalse(solvers["full"]._sparse_active_rows)
        runs = {name: _rollout(solver, model, steps=240, slide_world=1, speed=0.6) for name, solver in solvers.items()}
        anchors, full_anchors = runs["active"][0], runs["full"][0]
        np.testing.assert_array_equal(runs["active"][1], runs["full"][1])
        for history in (anchors, full_anchors):
            # Two box feet, four ring-linked corners each, hold two anchors per foot at rest.
            np.testing.assert_array_equal(history[-40:], 4)
            self.assertTrue(np.any(history[:, 1] < 4), "The sliding world must release anchors")
        np.testing.assert_array_equal(anchors[:, 0], full_anchors[:, 0])
        mismatched = anchors[:, 1] != full_anchors[:, 1]
        self.assertLessEqual(
            int(mismatched.sum()), 18, f"steps with different anchor counts: {np.flatnonzero(mismatched)}"
        )
        self.assertLessEqual(np.abs(anchors[:, 1] - full_anchors[:, 1]).max(), 2)
        states = list(zip(runs["active"][2], runs["full"][2], strict=True))
        # The law must act on this fixture, or the comparison would prove nothing.
        self.assertFalse(all(np.array_equal(active[1], full[1]) for active, full in states))
        bodies = model.body_count // model.world_count
        for world, pose_bound, velocity_bound in ((0, 5.0e-5, 1.0e-2), (1, 5.0e-5, 3.0e-2)):
            with self.subTest(world=world):
                body = slice(world * bodies, (world + 1) * bodies)
                pose_gap = max(np.abs(active[0][body, :3] - full[0][body, :3]).max() for active, full in states)
                velocity_gap = max(np.abs(active[1][body] - full[1][body]).max() for active, full in states)
                self.assertLess(pose_gap, pose_bound)
                self.assertLess(velocity_gap, velocity_bound)


if __name__ == "__main__":
    unittest.main()
