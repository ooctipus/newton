# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the slot-order and carrier independent patch-friction comparisons on CPU."""

import contextlib
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import sweep_routes_split
from newton._src.solvers.feather_pgs.friction_patches import link_patch_routes, seed_patch_routes
from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
)
from newton.solvers import SolverFeatherPGS
from newton.tests import test_feather_pgs_sparse_split as split
from newton.tests.feather_pgs_patch_parity import (
    SLIDING_FEET,
    SLIDING_ITERATIONS,
    SLIDING_REGION_BOUNDS,
    STANDING_REGION_BOUNDS,
    STANDING_VELOCITY_ATOL,
    anchor_sites,
    assert_anchor_sites_close,
    assert_contact_impulses_close,
    assert_factor_rows_consistent,
    assert_history_close,
    assert_history_matches_rows,
    assert_loaded_anchors,
    assert_patch_rows_close,
    assert_patch_seeds,
    assert_region_impulses_close,
    assert_ring_coulomb,
    assert_rows_close,
    assert_states_close,
    contact_rows,
    patch_rows,
    recorded_patch_seeds,
    ring_budget_ratios,
    ring_load,
    standing_model,
)
from newton.tests.feather_pgs_patch_reference import store_frame
from newton.tests.test_feather_pgs_friction_patches import _ground_box, _patch_fixture

# The dense split reference of the patch parity tests, on CPU.
_DENSE_SPLIT = {"friction_anchor_beta": 0.2}


def _standing_cases(count, *, twin=False, iterations=8, **shape):
    """Build dense split standing cases on CPU; with ``twin`` the last one's torsos start one float32 ulp higher."""
    options = {**_DENSE_SPLIT, "pgs_iterations": iterations}
    cases = [
        split._case(False, options=dict(options), deterministic=True, standing=True, device="cpu", **shape)
        for _ in range(count)
    ]
    if twin:
        model, _solver, state, *_ = cases[-1]
        q = state.joint_q.numpy().reshape(model.world_count, -1)
        q[:, 2] = np.nextafter(q[:, 2], np.float32(1.0))
        state.joint_q.assign(q.reshape(-1))
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return cases


@contextlib.contextmanager
def _ring_pool_dropped(solver):
    """Cut the normal rows out of their patch rings after the link, so each anchor's budget pools its own normal only."""
    launch = wp.launch

    def cut(kernel, *args, **kwargs):
        result = launch(kernel, *args, **kwargs)
        if kernel is link_patch_routes:
            kinds, parents = solver.row_type.numpy(), solver.row_parent.numpy()
            parents[kinds == PGS_CONSTRAINT_TYPE_CONTACT] = -1
            solver.row_parent.assign(parents)
        return result

    with mock.patch.object(wp, "launch", cut):
        yield


_MF_ROWS = (
    "mf_row_type",
    "mf_row_parent",
    "mf_row_mu",
    "mf_rhs",
    "mf_impulses",
    "mf_phi",
    "mf_Z_a",
    "mf_Z_b",
    "mf_body_a",
    "mf_body_b",
)


def _permuted_slots(solver, contacts, shift):
    """Reserve the same matrix-free row blocks in rotated order, as other atomic outcomes would."""
    count = int(contacts.rigid_contact_count.numpy()[0])
    slot, path = solver.contact_slot.numpy().copy(), solver.contact_path.numpy()
    world, needed = solver.contact_world.numpy(), solver.contact_slots_needed.numpy()
    routed = [c for c in range(count) if path[c] == 1 and slot[c] >= 0 and needed[c] > 0]
    rows = {name: getattr(solver, name).numpy() for name in _MF_ROWS}
    permuted = {name: values.copy() for name, values in rows.items()}
    moved = {}
    for w in np.unique(world[routed]):
        blocks = [c for c in routed if world[c] == w]
        # This fixture's matrix-free rows are contacts only, so the blocks tile the world's rows.
        assert sorted(int(slot[c]) + k for c in blocks for k in range(int(needed[c]))) == list(
            range(int(solver.mf_constraint_count.numpy()[w]))
        )
        start = 0
        for c in blocks[shift:] + blocks[:shift]:
            for k in range(int(needed[c])):
                moved[(w, int(slot[c]) + k)] = start + k
            slot[c] = start
            start += int(needed[c])
    for (w, old), new in moved.items():
        for name in _MF_ROWS:
            permuted[name][w, new] = rows[name][w, old]
        parent = int(rows["mf_row_parent"][w, old])
        permuted["mf_row_parent"][w, new] = moved.get((w, parent), parent)
    stand_in = SimpleNamespace(
        **{name: wp.array(values, device="cpu") for name, values in permuted.items()},
        mf_constraint_count=solver.mf_constraint_count,
        contact_slot=wp.array(slot, device="cpu"),
        contact_path=solver.contact_path,
        contact_world=solver.contact_world,
        contact_slots_needed=solver.contact_slots_needed,
        _friction_patches=solver._friction_patches,
        _sparse_mass_matrix_size=None,
        pgs_mode=solver.pgs_mode,
        # The bodies' factors, which whitened the rows.
        L_by_size=solver.L_by_size,
        body_to_articulation=solver.body_to_articulation,
        art_group_idx=solver.art_group_idx,
    )
    return stand_in


class TestFeatherPGSPatchParity(unittest.TestCase):
    def test_contact_rows_ignore_slot_order(self):
        """Key rows and patch rings by contact, so other slot reservations compare equal."""
        with wp.ScopedDevice("cpu"):
            _model, solver, contacts, step = _ground_box("cpu")
            for _ in range(20):
                step()
            rows = contact_rows(solver, contacts, route=1, jacobian=True)
        normals = [c for c, record in rows.items() if record["types"][0] == PGS_CONSTRAINT_TYPE_CONTACT]
        anchors = [c for c, record in rows.items() if len(record["types"]) == 3]
        # A resting box face: four ring-linked normals, friction rows only on its two anchors.
        self.assertEqual(len(normals), 4)
        self.assertEqual(len(anchors), 2)
        self.assertTrue(all(rows[c]["parents"][0] not in (-1, c) for c in normals))
        for c in anchors:
            np.testing.assert_array_equal(rows[c]["types"][1:], [PGS_CONSTRAINT_TYPE_FRICTION] * 2)
            np.testing.assert_array_equal(rows[c]["parents"][1:], [c, c])
            self.assertGreater(rows[c]["patch"][0], 0.0)
        self.assertEqual(assert_ring_coulomb(solver, route=1), 2)
        for shift in (1, 2, 3):
            with self.subTest(shift=shift):
                stand_in = _permuted_slots(solver, contacts, shift)
                self.assertFalse(np.array_equal(stand_in.contact_slot.numpy(), solver.contact_slot.numpy()))
                permuted = contact_rows(stand_in, contacts, route=1, jacobian=True)
                assert_rows_close(rows, permuted, atol=0.0, rtol=0.0)
                assert_contact_impulses_close(rows, permuted, atol=0.0, rtol=0.0)
        # A resting box carries its weight: an impulse changed on one contact is reported on that contact.
        self.assertGreater(sum(float(rows[c]["impulses"][0]) for c in normals), 0.0)
        changed = {c: dict(record, impulses=record["impulses"].copy()) for c, record in rows.items()}
        changed[anchors[0]]["impulses"][1] += 1.0e-3
        with self.assertRaisesRegex(AssertionError, f"contact {anchors[0]} impulses"):
            assert_contact_impulses_close(rows, changed)
        # A load-scaled bound admits a change within its share of that contact's normal impulse only.
        load = float(rows[anchors[0]]["impulses"][0])
        self.assertGreater(load, 0.0)
        changed[anchors[0]]["impulses"][1] = rows[anchors[0]]["impulses"][1] + 0.9 * 5.0e-3 * load
        assert_contact_impulses_close(rows, changed, atol=0.0, rtol=0.0, load_rtol=5.0e-3)
        changed[anchors[0]]["impulses"][1] = rows[anchors[0]]["impulses"][1] + 1.1 * 5.0e-3 * load
        with self.assertRaisesRegex(AssertionError, f"contact {anchors[0]} impulses"):
            assert_contact_impulses_close(rows, changed, atol=0.0, rtol=0.0, load_rtol=5.0e-3)

    def test_ring_coulomb_accepts_the_pooled_budget(self):
        """Accept friction bounded by the ring's pooled normal load and reject an exceeded budget."""
        capacity = 32
        count = wp.array([7], dtype=int, device="cpu")
        rows = wp.array([[0, 2, 2, 0, 0, 2, 2] + [-1] * 25], dtype=int, device="cpu")
        parents = wp.array([[3, 0, 0, 4, 0, 4, 4] + [-1] * 25], dtype=int, device="cpu")
        mu = wp.array([[0.5, 0.25, 0.25, 0.5, 0.5, 0.25, 0.25] + [0] * 25], dtype=float, device="cpu")
        rhs = wp.array([[0, -100, 0, -4, 0, -100, 0] + [0] * 25], dtype=float, device="cpu")
        impulses = wp.zeros((1, capacity), dtype=float, device="cpu")
        diag = wp.ones((1, capacity), dtype=float, device="cpu")
        matrix = wp.array(np.eye(capacity, dtype=np.float32)[None], dtype=float, device="cpu")
        inputs = [count, diag, matrix, rhs, impulses, 2, 1.0, rows, parents, mu, mu, 0, 0, 0]
        wp.launch(sweep_routes_split.loop_kernel(), dim=1, inputs=inputs, device="cpu")
        solved = impulses.numpy()
        np.testing.assert_allclose(solved[0, :7], [0, 1, 0, 4, 0, 1, 0], atol=1.0e-6)
        stand_in = SimpleNamespace(
            constraint_count=count,
            row_type=rows,
            row_parent=parents,
            row_mu=mu,
            rhs=rhs,
            impulses=impulses,
            phi=wp.zeros_like(rhs),
            target_velocity=wp.zeros_like(rhs),
        )
        # Each anchor's own normal carries no load: only the pooled ring budget admits the solution.
        self.assertGreater(solved[0, 1], solved[0, 0] * 0.25)
        self.assertEqual(assert_ring_coulomb(stand_in), 2)
        solved[0, 5] = 1.5
        impulses.assign(solved)
        with self.assertRaisesRegex(AssertionError, "pooled budget"):
            assert_ring_coulomb(stand_in)

    def test_ring_walk_rejects_malformed_rings(self):
        """Pool a closed ring and fail, instead of looping forever, on one that never returns to its start."""
        impulses = np.array([1.0, 2.0, 4.0, 8.0])
        self.assertEqual(ring_load(np.array([1, 2, 0, -1]), impulses, 0), 7.0)
        self.assertEqual(ring_load(np.array([-1, 2, 0, -1]), impulses, 0), 1.0)
        with self.assertRaisesRegex(AssertionError, "does not close"):
            ring_load(np.array([1, 2, 1, -1]), impulses, 0)
        with self.assertRaisesRegex(AssertionError, "past the 4 live rows"):
            ring_load(np.array([1, 7, 0, -1]), impulses, 0)

    def test_standing_feet_keep_loaded_ring_anchors(self):
        """Carry ring-linked anchors from step to step on the standing fixture, and restart a reset world's."""
        with wp.ScopedDevice("cpu"):
            model = standing_model("cpu")
            solver = SolverFeatherPGS(
                model, pgs_mode="split", friction_anchor_beta=0.2, pgs_iterations=8, dense_max_constraints=32
            )
            pipeline = newton.CollisionPipeline(model, rigid_contact_max=32)
            contacts, control = pipeline.contacts(), model.control()
            state, output = model.state(), model.state()
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
            for step in range(12):
                reset = step == 6
                if reset:
                    solver.reset(state, world_mask=wp.array([True, False], dtype=wp.bool, device="cpu"))
                pipeline.collide(state, contacts)
                solver.step(state, output, control, contacts, 1.0 / 400.0)
                state.assign(output)
                # Two feet per world, each a ring of four normals with two anchors.
                count = assert_loaded_anchors(solver, contacts, carried=step > 0, reset_worlds=(0,) if reset else ())
                self.assertEqual(count, 8)
            with self.assertRaisesRegex(AssertionError, "carried history"):
                assert_loaded_anchors(solver, contacts, carried=False)
            # The feet hold the torso at its standing height.
            np.testing.assert_allclose(state.body_q.numpy()[:: model.body_count // 2, 2], 0.48, atol=1.0e-3)
            self.assertLess(float(np.abs(state.body_qd.numpy()).max()), 1.0e-2)

    def test_history_comparison_detects_lifecycle_differences(self):
        """Match equal anchor histories and report a changed validity bit or carried displacement."""
        points = np.array([[x, y, 0.0] for x in (-0.05, 0.05) for y in (-0.05, 0.05)], dtype=np.float32)
        histories = []
        for _ in range(2):
            _model, state, _contacts, patches = _patch_fixture(points)
            store_frame(patches, state)
            histories.append(patches)
        valid = np.flatnonzero(histories[0].previous.valid.numpy())
        self.assertGreater(valid.size, 0)
        assert_history_close(*histories)
        changed = histories[1].previous.valid.numpy()
        changed[valid[0]] = 0
        histories[1].previous.valid.assign(changed)
        with self.assertRaisesRegex(AssertionError, "previous.valid"):
            assert_history_close(*histories)
        changed[valid[0]] = 1
        histories[1].previous.valid.assign(changed)
        displacement = histories[1].previous.displacement.numpy()
        displacement[valid[0], 0] += 1.0e-3
        histories[1].previous.displacement.assign(displacement)
        with self.assertRaisesRegex(AssertionError, "previous.displacement"):
            assert_history_close(*histories)

    def test_patch_rows_ignore_which_edge_member_carries_an_anchor(self):
        """Match the standing fixture and a one-ulp twin by anchor location, with rounding unable to move anchor rows.

        On the rectangular feet the two corners of each short support edge tie.
        The builder keeps an edge whole and carries its friction rows on the
        edge's last contact, so float32 rounding no longer picks the carrier.
        The twin's torsos start one ulp higher. The states (twists within
        STANDING_VELOCITY_ATOL), the carrier-agnostic rows, the region bounds, the
        anchor sites and each arm's stored history hold at every step of the
        masked-reset schedule, and both arms keep the same carriers throughout.
        """
        cases = _standing_cases(2, twin=True)
        moved = 0
        for index in range(24):
            if index in (5, 13, 19):
                selected = wp.array([index != 13, index != 5], dtype=wp.bool, device="cpu")
                for case in cases:
                    case[1].reset(case[2], world_mask=selected)
            for case in cases:
                split._step(case)
            assert_states_close(cases[0][2], cases[1][2], velocity_atol=STANDING_VELOCITY_ATOL)
            rows = [patch_rows(case[1], case[-1], split.DT) for case in cases]
            assert_patch_rows_close(*rows)
            assert_region_impulses_close(*rows, **STANDING_REGION_BOUNDS)
            assert_anchor_sites_close(*(anchor_sites(case[1], case[-1]) for case in cases), tangent=False)
            for case in cases:
                self.assertEqual(assert_history_matches_rows(case[1], case[-1]), 8)
            carriers = [
                {anchor["contact"] for region in record["regions"].values() for anchor in region["anchors"]}
                for record in rows
            ]
            moved += carriers[0] != carriers[1]
        self.assertEqual(moved, 0, "Rounding must not move an anchor's rows to the other corner of its edge")

    def test_sliding_feet_expose_a_dropped_ring_pool(self):
        """Keep sliding anchors at their pooled budget and reject a budget that pools only its own normal.

        Both feet slide (SLIDING_FEET) with every anchor at its pooled ring
        budget, so kinetic friction fixes each region's friction. A one-ulp twin
        stays within SLIDING_REGION_BOUNDS; an arm whose normal rows leave their
        rings after the patch link, so each anchor's divided budget pools only its
        own normal, does not.
        """
        reference, twin = _standing_cases(2, twin=True, iterations=SLIDING_ITERATIONS, **SLIDING_FEET)
        (unpooled,) = _standing_cases(1, iterations=SLIDING_ITERATIONS, **SLIDING_FEET)
        ratios, rejected = [], 0
        for _ in range(24):
            for case in (reference, twin):
                split._step(case)
            with _ring_pool_dropped(unpooled[1]):
                split._step(unpooled)
            assert_states_close(reference[2], twin[2], velocity_atol=STANDING_VELOCITY_ATOL)
            rows = [patch_rows(case[1], case[-1], split.DT) for case in (reference, twin, unpooled)]
            assert_patch_rows_close(rows[0], rows[1], lifecycle=False)
            assert_region_impulses_close(rows[0], rows[1], **SLIDING_REGION_BOUNDS)
            for case in (reference, twin):
                assert_ring_coulomb(case[1])
                ratios.extend(ring_budget_ratios(case[1]))
            try:
                assert_region_impulses_close(rows[0], rows[2], **SLIDING_REGION_BOUNDS)
            except AssertionError:
                rejected += 1
        self.assertGreater(float(np.mean(np.asarray(ratios) > 0.99)), 0.9, "Sliding anchors must load their budget")
        self.assertEqual(rejected, 24, "A budget without the ring's load must change region friction every step")

    def test_history_comparison_requires_equal_capacity(self):
        """Name both contact capacities instead of comparing differently sized histories slot by slot."""
        square = [[x, y, 0.0] for x in (-0.05, 0.05) for y in (-0.05, 0.05)]
        _model, state, _contacts, small = _patch_fixture(np.array(square, dtype=np.float32))
        _model, _state, _contacts, large = _patch_fixture(np.array(square * 4, dtype=np.float32))
        self.assertEqual((small.capacity, large.capacity), (8, 16))
        store_frame(small, state)
        with self.assertRaisesRegex(AssertionError, "hold 8 and 16 contact slots"):
            assert_history_close(small, large)

    def test_patch_seeds_follow_the_seeding_law(self):
        """Record each step's patch seed and reject a skipped, rescaled, altered or overwritten one.

        The warm-started dense split stance carries every loaded anchor's
        friction into the next step's seed. A step whose seed launch is
        skipped, runs at half the decay, writes other values than the seeding
        law, or whose seeded rows are overwritten before the sweep fails.
        """
        case = split._case(
            False,
            options={**_DENSE_SPLIT, "pgs_warmstart": True},
            contact_matching="latest",
            deterministic=True,
            standing=True,
            device="cpu",
        )
        solver, patches = case[1], case[1]._friction_patches
        decay = solver.pgs_warmstart_decay

        def seeded_step():
            with recorded_patch_seeds() as seeds:
                split._step(case)
            return seeds

        def scale_carried_friction(factor):
            count = int(case[-1].rigid_contact_count.numpy()[0])
            source = patches.current.source.numpy()[:count]
            slot, world = solver.contact_slot.numpy()[:count], solver.contact_world.numpy()[:count]
            needed, impulses = solver.contact_slots_needed.numpy()[:count], solver.impulses.numpy()
            for c in np.flatnonzero((source >= 0) & (slot >= 0) & (needed == 3)):
                impulses[world[c], slot[c] + 1 : slot[c] + 3] *= factor
            solver.impulses.assign(impulses)

        seed, dispatch, launch = patches.seed, solver._dispatch_dense_pgs_solve, wp.launch

        def rescaled(state, world, slot, path, slots_needed, routes):
            routes.scale = wp.vec3(0.5 * routes.scale[0], routes.scale[1], routes.scale[2])
            seed(state, world, slot, path, slots_needed, routes)

        def altered(kernel, *args, **kwargs):
            # The seed launch itself writes half the law's values.
            result = launch(kernel, *args, **kwargs)
            if kernel is seed_patch_routes:
                scale_carried_friction(0.5)
            return result

        def overwritten(*args, **kwargs):
            scale_carried_friction(0.0)
            return dispatch(*args, **kwargs)

        self.assertEqual(assert_patch_seeds(seeded_step(), scale=decay, carried=False), 0.0)
        largest = max(assert_patch_seeds(seeded_step(), scale=decay) for _ in range(3))
        self.assertGreater(largest, 1.0e-4, "Carried anchors must seed friction")
        defects = {
            "not 0 times": mock.patch.object(patches, "seed", lambda *args: None),
            "warm-start scale": mock.patch.object(patches, "seed", rescaled),
            "seeded friction rows": mock.patch.object(wp, "launch", altered),
            "the sweep starts from": mock.patch.object(solver, "_dispatch_dense_pgs_solve", overwritten),
        }
        for message, defect in defects.items():
            with self.subTest(defect=message), defect, self.assertRaisesRegex(AssertionError, message):
                assert_patch_seeds(seeded_step(), scale=decay)

    def test_sparse_rows_must_carry_their_impulses(self):
        """Reject a factor delta without a seeded row's response, a stale incident and an undecoded velocity.

        Synthetic rows stand in for a native sparse split step, which needs
        CUDA: physical rows J, factor rows z = J F for an invertible factor
        transform F, the factor delta z^T lambda and v_out = v_hat + F delta.
        """
        rng = np.random.default_rng(3)
        jacobian = rng.normal(size=(6, 4))
        transform = np.tril(rng.normal(size=(4, 4))) + 3.0 * np.eye(4)
        z = jacobian @ transform
        impulses = np.abs(rng.normal(size=6)) * 1.0e-2
        v_hat = rng.normal(size=4) * 0.1
        delta = z.T @ impulses
        rows = {
            "z": z,
            "jacobian": jacobian,
            "impulses": impulses,
            "incident": jacobian @ v_hat,
            "delta": delta,
            "v_hat": v_hat,
            "v_out": v_hat + transform @ delta,
        }
        assert_factor_rows_consistent(**rows)
        # A sweep that leaves a seeded row out of its starting delta still decodes v_out from that delta.
        unseeded = delta - z[4] * impulses[4]
        with self.assertRaisesRegex(AssertionError, "factor delta"):
            assert_factor_rows_consistent(**{**rows, "delta": unseeded, "v_out": v_hat + transform @ unseeded})
        with self.assertRaisesRegex(AssertionError, "row incident"):
            assert_factor_rows_consistent(**{**rows, "incident": jacobian @ (1.01 * v_hat)})
        with self.assertRaisesRegex(AssertionError, "decoded velocity"):
            assert_factor_rows_consistent(**{**rows, "v_out": v_hat})


if __name__ == "__main__":
    unittest.main()
