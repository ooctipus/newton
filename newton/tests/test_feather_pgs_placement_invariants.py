# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""BACKBONE's placement invariant, ``|J v - Z^T u|`` at rounding level, at the predictor: a row's incident velocity,
which its writer forms from ``J``, is the velocity term of its whitened row over ``u_hat = L^T P v_hat``."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS
from newton.tests import feather_pgs_scenes as scenes
from newton.tests.feather_pgs_law_harness import free_body_incident_gaps, incident_gaps
from newton.tests.unittest_utils import add_function_test, get_test_devices

ROUNDING = 32 * 2.0**-24
"""The sum's rounding, in units of its terms' magnitude: ``n u`` over the rows' terms, on a row of the scenes' widths."""

FLOOR = 1.0e-9
"""Below this a gap [m/s] is the rounding of zero."""


def _gaps(scene, device, steps):
    """Step ``scene`` and yield each step's dense and free-body gaps and terms."""
    model = scene.build(device)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    solver = SolverFeatherPGS(model, **scene.solver_options)
    state, out, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    for _ in range(steps):
        state.clear_forces()
        pipeline.collide(state, contacts)
        solver.step(state, out, control, contacts, scene.dt)
        yield incident_gaps(solver), free_body_incident_gaps(solver)
        state, out = out, state


def test_incident_is_the_whitened_rows_term(test, device):
    """Hold the invariant on dense rows (a hand and its object) and free-body rows (a stack)."""
    for scene in (scenes.hand(), scenes.box_stack()):
        rows = 0
        for step, pairs in enumerate(_gaps(scene, device, 6)):
            for name, (gaps, terms) in zip(("dense", "free body"), pairs, strict=True):
                held = gaps <= ROUNDING * terms + FLOOR
                rows += int(np.count_nonzero(terms))
                test.assertTrue(held.all(), f"{scene.name} step {step}, {name}: worst gap {gaps.max():.3e}")
        test.assertGreater(rows, 0, f"{scene.name} wrote no row")


class TestPlacementInvariants(unittest.TestCase):
    pass


add_function_test(
    TestPlacementInvariants,
    "test_incident_is_the_whitened_rows_term",
    test_incident_is_the_whitened_rows_term,
    devices=get_test_devices(),
)

if __name__ == "__main__":
    wp.init()
    unittest.main()
