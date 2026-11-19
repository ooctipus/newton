# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Every route solves the same law: one state through the region route and the row route, converged.

A route is a mapping chosen by the data's shape (the free bodies' contacts as regions, a block each, or as rows
in the colored, the general and the line mappings), never a different physics. The test steps a scene, then takes
its state once through each route from a fresh solver with a long sweep budget (both converged: their own results
at a quarter of the budget agree), and requires the same motion, the same normal loads and the same friction to
rounding. The test pin ``Pins(regions=False)`` keeps the free bodies' contacts rows on the CPU, where the rule takes
regions.

Cases: a sticking and a sliding box with the rows' regularization raised (``pgs_cfm``, which only a law that held a
sticking friction unit off zero residual would feel), a box sliding on one region whose contacts' normals differ
(:func:`.feather_pgs_scenes.curved_patch`: the region law's one normal on every route), and the same with authored
contact materials (each route's spring from the effective mass along that normal).
"""

import unittest

import numpy as np

import newton
from newton._src.solvers.feather_pgs.route_table import Pins
from newton._src.solvers.feather_pgs.row_laws import ROW_CONTACT, ROW_FRICTION
from newton.solvers import SolverFeatherPGS
from newton.tests import feather_pgs_scenes as scenes

SWEEPS = 400
"""The budget of the compared solves; a quarter of it is the convergence check."""


def _settled(scene):
    """``scene`` stepped its steps on the default route: its model and state."""
    model = scene.build("cpu")
    solver = SolverFeatherPGS(model, **scene.solver_options)
    state, out, control = model.state(), model.state(), model.control()
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    for _ in range(scene.steps):
        state.clear_forces()
        pipeline.collide(state, contacts)
        solver.step(state, out, control, contacts, scene.dt)
        state, out = out, state
    return model, state


def _solve(scene, model, start, regions: bool, sweeps: int, options: dict):
    """One step from ``start`` with a fresh solver on the region route (``regions``) or the row route: the bodies'
    velocities and each world's normal load and friction (the sums of its contact and friction rows' impulses)."""
    solver = SolverFeatherPGS(
        model, pins=Pins(regions=regions), **{**scene.solver_options, **options, "pgs_iterations": sweeps}
    )
    if (solver.sweep.regions is not None) != regions:
        raise AssertionError(f"{scene.name}: the region route is {'not ' if regions else ''}taken")
    state, out, control = model.state(), model.state(), model.control()
    for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
        getattr(state, name).assign(getattr(start, name))
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state.clear_forces()
    pipeline.collide(state, contacts)
    solver.step(state, out, control, contacts, scene.dt)
    impulses = solver.placement.impulses.numpy().astype(np.float64)
    kind = solver.placement.row_type.numpy()
    loads = np.array(
        [[impulses[w][kind[w] == ROW_CONTACT].sum(), np.abs(impulses[w][kind[w] == ROW_FRICTION]).sum()] for w in
         range(impulses.shape[0])]
    )  # fmt: skip
    return out.body_qd.numpy().astype(np.float64), loads


class TestRoutesSolveOneLaw(unittest.TestCase):
    """The region route and the row route reach one solution of one state."""

    def _check(self, scene, options=None):
        options = options or {}
        model, start = _settled(scene)
        speed = scenes.GRAVITY * scene.dt
        results = {}
        for regions in (True, False):
            for sweeps in (SWEEPS // 4, SWEEPS):
                results[regions, sweeps] = _solve(scene, model, start, regions, sweeps, options)
            converged = np.abs(results[regions, SWEEPS][0] - results[regions, SWEEPS // 4][0]).max()
            self.assertLess(converged, 1.0e-6 * speed, f"{scene.name}: route regions={regions} has not converged")
        (v_regions, l_regions), (v_rows, l_rows) = results[True, SWEEPS], results[False, SWEEPS]
        # The motion: every body's twist, against the speed a step of gravity gives.
        self.assertLess(np.abs(v_regions - v_rows).max(), 1.0e-5 * speed, f"{scene.name}: the routes' motions differ")
        # The loads: each world's normal impulse and friction impulse.
        scale = max(np.abs(l_rows).max(), 1.0e-12)
        np.testing.assert_allclose(l_regions, l_rows, rtol=0.0, atol=1.0e-4 * scale, err_msg=f"{scene.name}: loads")

    def test_sticking_and_sliding_under_regularization(self):
        """A held friction unit sits at zero residual on every route, whatever the rows' regularization: a box that
        sticks stays, one that slides slides alike (``pgs_cfm`` regularizes steps, never the law)."""
        self._check(scenes.incline_pair(), {"pgs_cfm": 0.1})

    def test_curved_patch(self):
        """A region's contacts act along its one normal on every route: a box sliding on a patch whose contacts'
        normals differ by up to three degrees moves alike."""
        self._check(scenes.curved_patch())

    def test_authored_contacts(self):
        """An authored contact's spring (a stiffness per unit effective mass, or a declared N/m) is one law on every
        route: each takes the effective mass of the one direction the contact acts along."""
        self._check(scenes.curved_patch(material=(1.6e5, 800.0)))


if __name__ == "__main__":
    unittest.main()
