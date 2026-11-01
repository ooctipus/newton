# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Complete physical keyed rows on floating trees, pairs and fallback queues."""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.sim.contacts import contact_capacity_blocks
from newton._src.solvers.feather_pgs import keyed_rows as rows
from newton._src.solvers.feather_pgs.solver_feather_pgs import SolverFeatherPGS
from newton.tests.feather_pgs_patch_parity import (
    assert_contact_impulses_close,
    assert_history_close,
    assert_ring_coulomb,
    contact_rows,
    standing_model,
)
from newton.tests.test_feather_pgs_sparse_solver import STATE_FIELDS, _build_model, _make_solver, _step


def make_case(
    enabled,
    capacity,
    free_count=0,
    *,
    model_factory=None,
    beta=0.0,
    deterministic=False,
    rigid_contact_max=None,
):
    model = _build_model(free_count=free_count) if model_factory is None else model_factory()
    if rigid_contact_max is not None:
        # The solver sizes its contact scratch from the model's capacity at construction.
        model.rigid_contact_max = rigid_contact_max
    # Tiers of ``capacity`` rows on any class whose response they form; keyed rows feed them unless ``enabled`` is
    # False, which pins the general rows (``None`` leaves the producer to the rules, which choose keyed rows).
    pins = {"tiers": capacity, "tier_sweeps": 24, **({"producer": "general"} if enabled is False else {})}
    with patch.dict(SolverFeatherPGS._route_pins, pins):
        solver = _make_solver(
            model,
            False,
            pgs_iterations=8,
            dense_max_constraints=64,
            mf_max_constraints=64,
            friction_anchor_beta=beta,
        )
    state, out = model.state(), model.state()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    contact_capacity = 128 if rigid_contact_max is None else rigid_contact_max
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=contact_capacity, deterministic=deterministic)
    return model, solver, state, out, model.control(), pipeline, pipeline.contacts()


@unittest.skipUnless(wp.is_cuda_available(), "native parallel solving requires CUDA")
class TestKeyedRowIntegration(unittest.TestCase):
    def compare(self, ordinary, keyed):
        for field in STATE_FIELDS:
            np.testing.assert_allclose(
                getattr(keyed[2], field).numpy(),
                getattr(ordinary[2], field).numpy(),
                rtol=2e-4,
                atol=2e-4,
                err_msg=field,
            )
        for case in (ordinary, keyed):
            self.assertFalse(case[1].placement.constraint_overflow.numpy().any())
            for world, count in enumerate(case[1].placement.constraint_count.numpy()):
                impulse = case[1].placement.impulses.numpy()[world, :count]
                kind = case[1].placement.row_type.numpy()[world, :count]
                self.assertTrue(np.isfinite(impulse).all())
                self.assertTrue((impulse[np.isin(kind, (0, 3, 4))] >= -1e-7).all())

    def test_floating_tree_and_pair_current_inputs_and_fallback(self):
        """Compare current keyed inputs and ordered fallback against ordinary rows."""
        for free_count in (0, 1):
            for capacity in (8, 32):
                with self.subTest(free_count=free_count, capacity=capacity):
                    ordinary, keyed = make_case(False, capacity, free_count), make_case(True, capacity, free_count)
                    self.assertIsNone(ordinary[1].placement.keyed_rows)
                    self.assertIsNotNone(keyed[1].placement.keyed_rows)
                    self.assertEqual(keyed[1].placement.keyed_rows.layout.primary, 9)
                    seen = False
                    for _ in range(8):
                        _step(ordinary)
                        _step(keyed)
                        seen |= bool(keyed[-1].rigid_contact_count.numpy()[0])
                        self.compare(ordinary, keyed)
                    self.assertTrue(seen)

    def test_held_factor_graph_and_masked_reset(self):
        """Preserve held factors and masked reset inside CUDA graphs."""
        ordinary, keyed = make_case(False, 32), make_case(True, 32)
        self.assertIsNotNone(keyed[1].placement.keyed_rows)
        for case in (ordinary, keyed):
            _step(case)
        with wp.ScopedCapture("cuda:0") as capture:
            for _ in range(4):
                _step(keyed)
        for _ in range(4):
            wp.capture_launch(capture.graph)
            for _ in range(4):
                _step(ordinary)
        self.compare(ordinary, keyed)
        held = keyed[1].factor.L_by_size[9].numpy().copy()
        for model, solver, state, *_ in (ordinary, keyed):
            q = state.joint_q.numpy()
            q[2] += 0.01
            state.joint_q.assign(q)
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
            solver.reset(state, wp.array([True, False], dtype=bool, device=model.device))
        for case in (ordinary, keyed):
            _step(case)
        np.testing.assert_array_equal(keyed[1].factor.L_by_size[9].numpy()[1], held[1])
        self.compare(ordinary, keyed)

    def test_default_request_selects_keyed_inputs(self):
        """Adopt admitted physical keys without naming a task or adding a caller flag."""
        case = make_case(None, 32)
        self.assertIsNotNone(case[1].placement.keyed_rows)
        self.assertIsNotNone(case[1].placement.keyed_rows.layout)

    def test_patch_rows_match_ordinary_patch_rows(self):
        """Solve patch friction through keyed rows as through ordinary rows in the same linked tiers.

        Both arms run the ring-pooled parallel law, so they agree up to float
        order: the same contacts, row types, ring links and divided coefficients,
        impulses by contact, states within the point twin's tolerance, pooled cones
        and anchor history. Sorted contacts let the by-contact checks pair the arms'
        contacts. Keyed rows decline the resolved-world classifier, which would
        resolve the tripod's worlds after their first step and leave the ordinary
        arm without rows, so the test turns the classifier off
        (the route pin ``claims=()``) to compare rows on every step.
        test_patch_resolved_worlds_match_the_complete_path in
        test_feather_pgs_resolved_worlds covers the classifier's exactness under
        patch friction.
        """
        for name, factory, free_count in (("tripod", None, 1), ("standing", standing_model, 0)):
            with self.subTest(fixture=name):
                # The solver reads the switch only at construction; the classifier owner built there
                # selects the allocator variant and the predictor's twist publication on every step.
                with patch.dict(SolverFeatherPGS._route_pins, {"claims": ()}):
                    ordinary, keyed = (
                        make_case(enabled, 32, free_count, model_factory=factory, beta=0.2, deterministic=True)
                        for enabled in (False, True)
                    )
                for case in (ordinary, keyed):
                    classifier = case[1].get_path_selection()["routes"]["resolved_worlds"]
                    self.assertFalse(classifier["enabled"], classifier["reason"])
                self.assertIsNone(ordinary[1].placement.keyed_rows)
                self.assertTrue(keyed[1].placement.keyed_rows.patch)
                for case in (ordinary, keyed):
                    tiers = case[1].sweep.tiers
                    self.assertTrue(tiers)
                    self.assertTrue(all(tier.linked_rows for tier in tiers))
                linked = False
                for _ in range(8):
                    _step(ordinary)
                    _step(keyed)
                    self.compare(ordinary, keyed)
                    expected, actual = contact_rows(ordinary[1], ordinary[-1]), contact_rows(keyed[1], keyed[-1])
                    self.assertEqual(sorted(actual), sorted(expected))
                    assert_contact_impulses_close(expected, actual)
                    for c, record in expected.items():
                        np.testing.assert_array_equal(actual[c]["types"], record["types"])
                        np.testing.assert_array_equal(actual[c]["parents"], record["parents"])
                        np.testing.assert_allclose(actual[c]["mu"], record["mu"], rtol=1.0e-6)
                        linked |= record["parents"][0] not in (-1, c)
                    for case in (ordinary, keyed):
                        assert_ring_coulomb(case[1])
                    assert_history_close(ordinary[1], keyed[1])
                if name == "standing":
                    self.assertTrue(linked, "A box foot must link several normals in one ring")

    def test_key_scatters_take_the_recorded_blocks(self):
        """Launch the key scatters over the route record's block count, solving as on the capped default grid.

        Both laws scatter their keys once the counts are final, before the patch link. A solver
        forced to one block strides that block over the whole 1,024-slot
        buffer, four iterations whose later ones hold only empty slots, and solves the same rows
        as the default grid. This tripod has far fewer than 256 live contacts, so every live one
        falls in the first iteration; test_key_scatters_match_on_a_capped_grid in
        test_feather_pgs_capacity_launches covers live contacts in later iterations.
        """
        device = wp.get_device("cuda:0")
        scatter = rows.scatter_contact_keys
        for beta in (0.0, 0.2):
            with self.subTest(friction_anchor_beta=beta):
                with patch.dict(SolverFeatherPGS._route_pins, {"claims": ()}):
                    wave, single = (
                        make_case(True, 32, 1, beta=beta, deterministic=True, rigid_contact_max=1024) for _ in range(2)
                    )
                self.assertEqual(wave[1].placement.keyed_rows.patch, beta > 0.0)
                self.assertEqual(wave[-1].rigid_contact_max, 1024)
                self.assertEqual(wave[1].placement.max_contacts_alloc, 1024)
                record = wave[1].get_path_selection()["routes"]["contact_consumers"]
                self.assertEqual(record["max_blocks"], contact_capacity_blocks(device))
                single[1].placement.contact_capacity_blocks = 1
                blocks, launch = [], wp.launch

                def recording(kernel, *args, scatter=scatter, blocks=blocks, launch=launch, **kwargs):
                    if kernel is scatter:
                        blocks.append(kwargs.get("max_blocks", 0))
                    return launch(kernel, *args, **kwargs)

                for _ in range(4):
                    _step(wave)
                    with patch.object(wp, "launch", recording):
                        _step(single)
                    self.assertGreater(int(single[-1].rigid_contact_count.numpy()[0]), 0)
                    self.compare(wave, single)
                self.assertEqual(blocks, [1] * 4)


if __name__ == "__main__":
    unittest.main()
