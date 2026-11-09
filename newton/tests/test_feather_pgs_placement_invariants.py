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


def _gaps(scene, device, steps, pins=None, solvers=None):
    """Step ``scene`` (with the test ``pins``) and yield each step's dense and free-body gaps and terms; ``solvers``
    collects the solver."""
    model = scene.build(device)
    pipeline = newton.CollisionPipeline(model, growable_contacts=True)
    contacts = pipeline.contacts()
    # The invariant reads the stored rows: the tiers' worlds take the one writer's STORE sink.
    pins = {"sink": "store", **(pins or {})}
    solver = SolverFeatherPGS(model, pins=SolverFeatherPGS.Pins(**pins), **scene.solver_options)
    if solvers is not None:
        solvers.append(solver)
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


def test_support_classes_hold_the_invariant(test, device):
    """Rows stored by their support classes (the general placement of a sprawl of five four-link limbs, a limb's rows
    on its root path): the invariant holds, each row narrower than its sides."""
    solvers, rows = [], 0
    for step, pairs in enumerate(_gaps(scenes.sprawl(worlds=16), device, 6, {"placement": "general"}, solvers)):
        gaps, terms = pairs[0]
        rows += int(np.count_nonzero(terms))
        test.assertTrue((gaps <= ROUNDING * terms + FLOOR).all(), f"step {step}: worst gap {gaps.max():.3e}")
    placement = solvers[-1].placement
    test.assertTrue(placement.classed, placement.row_classes_reason)
    test.assertLess(placement.packed_support, solvers[-1].forest.max_world_dofs)
    test.assertGreater(rows, 0, "the sprawl wrote no row")


def _run(scene, device, pins, steps):
    model = scene.build(device)
    pipeline = newton.CollisionPipeline(model, growable_contacts=True)
    contacts = pipeline.contacts()
    # Every key's limit rows fit a world's rows.
    options = {**scene.solver_options, "dense_max_constraints": 256}
    solver = SolverFeatherPGS(model, pins=SolverFeatherPGS.Pins(**pins), **options)
    state, out, control = model.state(), model.state(), model.control()
    for step in range(steps):
        state.clear_forces()
        if scene.control is not None:
            scene.control(model, control, state, step)
        pipeline.collide(state, contacts)
        solver.step(state, out, control, contacts, scene.dt)
        state, out = out, state
    return solver, state.body_qd.numpy()


def test_support_classes_step_as_sides(test, device):
    """The rows' support classes reorder no sum on the CPU: a world's entries keep their coordinates' order and drop
    zeros, so the classes step bit for bit as the rows' sides (the test pin ``row_support=False``); on CUDA the
    sweep's lanes sum other entries, so a step agrees to rounding (contacts grow it along a trajectory). Scenes whose
    classes hold fewer entries a lane: keyboard keys (a key's rows one entry wide, their limits), a sprawl of limbs."""
    cpu = wp.get_device(device).is_cpu
    for scene in (
        scenes.keyboard(worlds=32, keys=32),
        scenes.sprawl(worlds=16),
    ):
        steps = scene.steps if cpu else 1
        classes, qd = _run(scene, device, {"placement": "general"}, steps)
        sides, qd_sides = _run(scene, device, {"placement": "general", "row_support": False}, steps)
        test.assertTrue(classes.placement.classed, f"{scene.name}: {classes.placement.row_classes_reason}")
        test.assertFalse(sides.placement.classed)
        test.assertLess(classes.placement.packed_support, sides.placement.packed_support)
        if cpu:
            np.testing.assert_array_equal(qd, qd_sides, err_msg=scene.name)
        else:
            np.testing.assert_allclose(qd, qd_sides, rtol=0.0, atol=1e-4 * np.abs(qd_sides).max(), err_msg=scene.name)


class TestPlacementInvariants(unittest.TestCase):
    pass


add_function_test(
    TestPlacementInvariants,
    "test_support_classes_hold_the_invariant",
    test_support_classes_hold_the_invariant,
    devices=get_test_devices(),
)
add_function_test(
    TestPlacementInvariants,
    "test_support_classes_step_as_sides",
    test_support_classes_step_as_sides,
    devices=get_test_devices(),
)


add_function_test(
    TestPlacementInvariants,
    "test_incident_is_the_whitened_rows_term",
    test_incident_is_the_whitened_rows_term,
    devices=get_test_devices(),
)

if __name__ == "__main__":
    wp.init()
    unittest.main()
