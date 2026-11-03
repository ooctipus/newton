# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS


def _build_two_world_free_model(device):
    template = newton.ModelBuilder(gravity=0.0)
    body = template.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    joint = template.add_joint_free(parent=-1, child=body)
    template.add_articulation([joint])

    builder = newton.ModelBuilder(gravity=0.0)
    builder.replicate(template, 2)
    return builder.finalize(device=device)


def _make_solver(model, *, warmstart: bool):
    return SolverFeatherPGS(
        model,
        pgs_warmstart=warmstart,
        dense_max_constraints=4,
        mf_max_constraints=4,
    )


def _history_specs(solver):
    """The row store's impulses and its warm-start carry, both classes' rows."""
    specs = []
    if solver.options.pgs_warmstart:
        specs.append(("impulses", solver.placement.impulses, 7.0, 0.0))
    if solver.placement.ws_prev_dense_impulses is not None:
        specs.extend(
            (
                ("carried impulses", solver.placement.ws_prev_dense_impulses, 11.0, 0.0),
                ("carried row types", solver.placement.ws_prev_dense_row_type, 13, -1),
            )
        )
    return specs


def _poison(specs):
    for _name, array, poison, _cleared in specs:
        array.fill_(poison)


def _assert_worlds(test: unittest.TestCase, specs, reset_worlds: tuple[bool, bool]):
    for name, array, poison, cleared in specs:
        expected = np.full(array.shape, poison, dtype=array.numpy().dtype)
        for world, reset in enumerate(reset_worlds):
            if reset:
                expected[world] = cleared
        np.testing.assert_array_equal(array.numpy(), expected, err_msg=name)


class TestFeatherPGSReset(unittest.TestCase):
    def test_warmstart_clear_does_not_change_fk_cache_source(self):
        """Keep cache-source bookkeeping in reset, independent of impulse clearing."""
        model = _build_two_world_free_model("cpu")
        solver = _make_solver(model, warmstart=True)
        # A held tree that validates banks by identity keeps the cache's source (the others compare the banks).
        solver.factor.held.bank_compare = False
        state = model.state()
        solver.factor.fk_id_cache_source_state = state
        solver.factor.fk_id_cache_valid = wp.ones(model.articulation_count, dtype=wp.int32, device=model.device)
        specs = _history_specs(solver)
        _poison(specs)
        solver.sweep.clear_history(None)
        _assert_worlds(self, specs, (True, True))
        self.assertIs(solver.factor.fk_id_cache_source_state, state)
        np.testing.assert_array_equal(solver.factor.fk_id_cache_valid.numpy(), [1, 1])

        mask = wp.array((True, False), dtype=wp.bool, device=model.device)
        solver.reset(state, mask)
        np.testing.assert_array_equal(solver.factor.fk_id_cache_valid.numpy(), [0, 1])
        other_state = model.state()
        solver.reset(other_state, mask)
        self.assertIs(solver.factor.fk_id_cache_source_state, other_state)
        np.testing.assert_array_equal(solver.factor.fk_id_cache_valid.numpy(), [0, 0])

    def test_model_changes_invalidate_impulses_without_clearing_overflow(self):
        """Invalidate changed-model impulses while retaining invalid-transition status."""
        model = _build_two_world_free_model("cpu")
        solver = _make_solver(model, warmstart=True)
        for flags in (
            newton.ModelFlags.BODY_INERTIAL_PROPERTIES,
            newton.ModelFlags.JOINT_DOF_PROPERTIES,
            newton.ModelFlags.SHAPE_PROPERTIES,
        ):
            with self.subTest(flags=flags):
                specs = _history_specs(solver)
                _poison(specs)
                solver.placement.constraint_overflow.fill_(True)
                solver.notify_model_changed(flags)
                _assert_worlds(self, specs, (True, True))
                self.assertTrue(solver.placement.constraint_overflow.numpy().all())

    def test_reset_clears_enabled_histories_by_scope(self):
        cases = (
            ("selected", (True, False)),
            ("all", (True, True)),
        )
        for name, reset_worlds in cases:
            with self.subTest(name=name):
                model = _build_two_world_free_model("cpu")
                solver = _make_solver(model, warmstart=True)
                specs = _history_specs(solver)
                _poison(specs)
                mask = None
                if reset_worlds != (True, True):
                    mask = wp.array(reset_worlds, dtype=wp.bool, device=model.device)
                solver.reset(model.state(), mask)
                _assert_worlds(self, specs, reset_worlds)

    def test_reset_validates_mask_before_disabled_noop(self):
        model = _build_two_world_free_model("cpu")
        solver = _make_solver(model, warmstart=False)
        solver.placement.impulses.fill_(17.0)
        before = solver.placement.impulses.numpy().copy()

        invalid_mask = wp.ones(3, dtype=wp.bool, device=model.device)
        with self.assertRaisesRegex(ValueError, "expected 2"):
            solver.reset(model.state(), invalid_mask)

        valid_mask = wp.array((True, False), dtype=wp.bool, device=model.device)
        solver.reset(model.state(), valid_mask)
        np.testing.assert_array_equal(solver.placement.impulses.numpy(), before)
        self.assertIsNone(solver.placement.ws_prev_dense_impulses)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA graph reset test requires CUDA")
    def test_reset_is_cuda_graph_capturable(self):
        device = wp.get_device("cuda:0")
        model = _build_two_world_free_model(device)
        solver = _make_solver(model, warmstart=True)
        state = model.state()
        mask = wp.array((True, False), dtype=wp.bool, device=device)
        specs = _history_specs(solver)

        _poison(specs)
        solver.reset(state, mask)
        wp.synchronize_device(device)
        with wp.ScopedCapture(device=device) as capture:
            solver.reset(state, mask)

        _poison(specs)
        wp.capture_launch(capture.graph)
        _assert_worlds(self, specs, (True, False))


if __name__ == "__main__":
    unittest.main()
