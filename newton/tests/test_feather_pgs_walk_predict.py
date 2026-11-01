# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The predictor on the walk (Z4.0): the walk's response beside the held factor's, which no route selects yet.

The test pin ``walk_predictor`` solves each tree's trial by the walk (:func:`.tree_kinetic_state.walk_solve`): the
leaf-first pass of the force over the whole tree with its joins (``WALK_UP_TREE``), then the leave (``WALK_DOWN``), on
the all-``K`` walk the factor refreshes beside ``L`` (:class:`.tree_factor.WalkTrees`). The decided step then starts
from the trial's force plus the decision's change of it. These tests step the same states through both predictors and
hold:

- **the response:** ``v_hat`` by the walk within ``BOUND n u kappa`` of the held predictor's, relative to the step's
  velocity change, on every CPU catalogue problem and route and on the CUDA catalogue's held layouts;
- **the decision:** on trees whose drives saturate, the drives' decided ``K`` (the trial's decided set) equal at every
  step, CPU and CUDA, on each grouping.
"""

import unittest
import warnings
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS
from newton.tests import test_feather_pgs_conformance as conformance
from newton.tests import test_feather_pgs_law_repairs as repairs
from newton.tests.test_feather_pgs_walk import BOUND, ROUNDING, catalogue, pipeline_options
from newton.tests.test_feather_pgs_walk_device import devices

STEPS = 4
"""Steps of each catalogue problem, each from the reference predictor's state."""
PIN = {"walk_predictor": True}


def _kappa(solver, joint_q: np.ndarray) -> tuple[int, float]:
    """The largest coordinate count and condition number ``cond(H + R + K)`` over the model's articulations, from the
    factor's float64 walk."""
    model = solver.model
    starts, qd = model.articulation_start.numpy(), model.joint_qd_start.numpy()
    worst_n, worst_kappa = 1, 1.0
    for art in range(int(model.articulation_count)):
        n = int(qd[starts[art + 1]] - qd[starts[art]])
        if n == 0:
            continue
        walk = solver.factor.walk(joint_q, art)
        inverse, _ = walk.up(covector=np.eye(n))
        worst_n = max(worst_n, n)
        if not np.all(np.isfinite(inverse)):
            # A massless shadow's coordinate (a loop cut at a body): no rounding bound.
            worst_kappa = np.inf
            continue
        singular = np.linalg.svd(inverse, compute_uv=False)
        worst_kappa = max(worst_kappa, float((singular[0] / singular[-1]) ** 2))
    return worst_n, worst_kappa


def walked_coordinates(solver) -> np.ndarray:
    """The coordinates whose trees the walk holds: every articulation's but a free rigid body's, which keeps its 6x6
    factor until the free bodies convert (FL3.3 stage two C walks them)."""
    model, view = solver.model, solver.factor.walk_view()
    starts, qd = model.articulation_start.numpy(), model.joint_qd_start.numpy()
    free = view.free.numpy()
    mask = np.zeros(int(model.joint_dof_count), dtype=bool)
    for art in range(int(model.articulation_count)):
        if not free[art]:
            mask[qd[starts[art]] : qd[starts[art + 1]]] = True
    return mask


def _copy_state(source, target) -> None:
    for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
        array = getattr(source, name, None)
        if array is not None:
            getattr(target, name).assign(array)


def predict_pair(physics, route, device="cpu", steps=STEPS) -> dict:
    """Step ``physics`` through ``route`` with the held predictor and, from each of its input states and the same
    contacts, with the walk's: the worst ``|v_hat_walk - v_hat| / |v_hat - v|`` over the steps against the rounding
    bound ``n u kappa``, and whether every step decided the same drive stiffness."""
    scene = physics.scene
    options = {**conformance.BASE_OPTIONS, **scene.solver_options, **physics.options(physics.iterations)}
    options.update(route.options)
    options.pop("contact_compliance", None)
    solvers, models = [], []
    for pins in ({}, PIN):
        with conformance.route_context(route), mock.patch.dict(SolverFeatherPGS._route_pins, pins):
            model = scene.build(device)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                solvers.append(SolverFeatherPGS(model, **options))
            models.append(model)
    base, walk = solvers
    pipeline = newton.CollisionPipeline(models[0], **pipeline_options(scene))
    contacts = pipeline.contacts()
    states = [(m.state(), m.state(), m.control()) for m in models]
    newton.eval_fk(models[0], states[0][0].joint_q, states[0][0].joint_qd, states[0][0])
    worst, same, n, kappa = 0.0, True, 1, 1.0
    trees = walked_coordinates(walk)
    for step in range(steps):
        (state, out, control), (walk_in, walk_out, walk_control) = states
        state.clear_forces()
        if scene.control is not None:
            scene.control(models[0], control, state, step)
            scene.control(models[1], walk_control, state, step)
        _copy_state(state, walk_in)
        walk_in.clear_forces()
        if state.body_f is not None:
            walk_in.body_f.assign(state.body_f)
        pipeline.collide(state, contacts)
        v = base.cut.mirror(state).joint_qd.numpy().astype(np.float64)
        if step == 0:
            n, kappa = _kappa(base, base.cut.mirror(state).joint_q.numpy().astype(np.float64))
        base.step(state, out, control, contacts, scene.dt)
        walk.step(walk_in, walk_out, walk_control, contacts, scene.dt)
        a, b = base.v_hat.numpy().astype(np.float64)[trees], walk.v_hat.numpy().astype(np.float64)[trees]
        change = float(np.abs(a - v[trees]).max()) if a.size else 0.0
        if change > 0.0:
            worst = max(worst, float(np.abs(b - a).max()) / change)
        same &= bool(np.array_equal(base.factor.aug_row_Kd.numpy(), walk.factor.aug_row_Kd.numpy()))
        states[0] = (out, state, control)
        states[1] = (walk_out, walk_in, walk_control)
    return {
        "worst": worst,
        "bound": n * ROUNDING * kappa,
        "same_decisions": same,
        "n": n,
        "kappa": kappa,
        "coordinates": int(np.count_nonzero(trees)),
    }


def saturating_pair(device, scene, pins, steps=60) -> dict:
    """Trees whose drives saturate (:mod:`.test_feather_pgs_law_repairs`), stepped through both predictors from the
    held predictor's states: the decided ``K`` at every step and the worst ``v_hat`` difference over its change."""
    arms = []
    for walk_pin in ({}, PIN):
        with conformance.route_context(conformance.Route("drives", device.split(":")[0], pins={**pins, **walk_pin})):
            model, control = scene(device)
            arms.append((SolverFeatherPGS(model, angular_damping=0.0, pgs_iterations=4), model, control))
    (base, model, control), (walk, walk_model, walk_control) = arms
    state, out = model.state(), model.state()
    walk_in, walk_out = walk_model.state(), walk_model.state()
    targets = np.random.default_rng(1).uniform(-1.0, 1.0, (steps, model.joint_dof_count)).astype(np.float32)
    worst, same, decided = 0.0, True, 0
    for target in targets:
        control.joint_target_q.assign(target)
        walk_control.joint_target_q.assign(target)
        _copy_state(state, walk_in)
        v = state.joint_qd.numpy().astype(np.float64)
        base.step(state, out, control, None, 1.0 / 60.0)
        walk.step(walk_in, walk_out, walk_control, None, 1.0 / 60.0)
        a, b = base.v_hat.numpy().astype(np.float64), walk.v_hat.numpy().astype(np.float64)
        worst = max(worst, float(np.abs(b - a).max()) / max(float(np.abs(a - v).max()), 1.0e-30))
        Kd, K = base.factor.aug_row_Kd.numpy(), base.factor.aug_row_K.numpy()
        decided += int(np.count_nonzero(Kd != K))
        same &= bool(np.array_equal(Kd, walk.factor.aug_row_Kd.numpy()))
        state, out = out, state
    owner = walk.get_path_selection()["routes"]["held_tree"]["owner"]
    return {"worst": worst, "same_decisions": same, "saturated": decided, "owner": owner}


class TestWalkPredictor(unittest.TestCase):
    """The walk's response against the held predictor's, CPU and CUDA."""

    def test_cpu_catalogue(self):
        """Every CPU catalogue problem and route: ``v_hat`` within ``BOUND n u kappa`` and the same decisions."""
        for physics, route in catalogue():
            with self.subTest(problem=physics.name, route=route.name):
                result = predict_pair(physics, route)
                self.assertLessEqual(result["worst"], BOUND * result["bound"], f"{result}")
                self.assertTrue(result["same_decisions"], f"{result}")

    @unittest.skipUnless(wp.is_cuda_available(), "the CUDA catalogue's held layouts")
    def test_cuda_catalogue(self):
        """The CUDA catalogue's held layouts, each problem on its default route."""
        for group in conformance.cuda_groups():
            with self.subTest(problem=group.physics.name):
                result = predict_pair(group.physics, group.reference, device="cuda:0")
                self.assertLessEqual(result["worst"], BOUND * result["bound"], f"{result}")
                self.assertTrue(result["same_decisions"], f"{result}")

    def test_saturating_trees_decide_alike(self):
        """Trees whose drives saturate: the trial's decided set equal at every step, ``v_hat`` within rounding, on each
        device and grouping (the walk solves every root of a forest)."""
        for device in devices():
            for scene in (repairs._driven_forest, repairs._driven_chain):
                groupings = ({}, {"grouping": "forest"}) if scene is repairs._driven_forest else ({},)
                for pins in groupings:
                    with self.subTest(device=device, scene=scene.__name__, pins=pins):
                        result = saturating_pair(device, scene, pins)
                        self.assertGreater(result["saturated"], 0)
                        self.assertTrue(result["same_decisions"], f"{result}")
                        self.assertLess(result["worst"], 1.0e-4, f"{result}")


if __name__ == "__main__":
    unittest.main()
