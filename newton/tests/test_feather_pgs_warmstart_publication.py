# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The free-body warm start's publication: the carried impulses' velocity on every free rigid body.

A warm-started free-body sweep starts from ``(lambda0, u0 = u_hat + Z lambda0)``: the impulses carried by contact
identity and the velocity they imply, installed before the first sweep. Velocity and impulses then share one state,
so each free body's momentum change over the step equals ``dt`` times gravity and the contact forces the step
publishes. A body that starts with its carried impulses but without their velocity is counted twice, and its momentum
balance fails by the carried impulse, growing step by step.

Regression: the publication once held at most 64 bodies per world (a fixed body map), so every free body past the
64th in a world's row order started double counted. These worlds hold 80.
"""

from __future__ import annotations

import contextlib
import math
import unittest
from unittest import mock

import warp as wp

from newton._src.solvers.feather_pgs import sweep_units
from newton.tests import feather_pgs_scenes as scenes
from newton.tests import test_feather_pgs_conformance as conf

FREE_BODIES = 80
"""Free rigid bodies in the world: past the 64 the publication once held."""


def resting_boxes(count: int = FREE_BODIES, *, mu: float = 0.6) -> scenes.Scene:
    """``count`` 0.1 m, 0.2 kg boxes in a grid 0.5 mm above the ground, one world: each comes to rest on four
    contacts within a few steps and then carries its weight, so every body's carried impulses are nonzero."""
    side = math.ceil(math.sqrt(count))

    def build(device):
        builder = scenes._builder()
        builder.add_ground_plane(cfg=scenes._cfg(mu=mu))
        for k in range(count):
            i, j = divmod(k, side)
            scenes._free_box(builder, (0.15 * i, 0.15 * j, 0.0505), half=0.05, mass=0.2, mu=mu)
        return scenes._finalize(builder, device)

    return scenes.Scene(
        f"resting_boxes_{count}",
        build,
        dt=1.0 / 240.0,
        steps=24,
        params={"mu": [mu]},
        # Room for every box's four contacts and their friction rows: no row is dropped.
        solver_options={"mf_max_constraints": 16 * count},
        passive=True,
        contact_matching="latest",
    )


@contextlib.contextmanager
def _colored():
    """Every world takes the colored mapping (a world of one contact unit qualifies)."""
    with mock.patch.object(sweep_units, "COLORED_RULE", sweep_units.ColoredRule(min_units=1, parallelism=0.0)):
        yield


class TestWarmStartPublication(unittest.TestCase):
    """Each free body's momentum change balances the forces a warm-started step publishes, past 64 bodies a world."""

    def _balance(self, route: conf.Route, device, *, warm: bool, context=contextlib.nullcontext) -> float:
        scene = resting_boxes()
        options = {"pgs_iterations": 16, "friction_anchor_beta": 0.0}
        if warm:
            options["pgs_warmstart"] = True
        with context():
            run = conf.simulate(scene, route, device, options=options)
        data, _, _ = conf.model_data(scene)
        # The last step's contacts hold every box: the world's rows reach past the 64 bodies.
        last = run.contacts[-1]
        bodies = {
            int(data.shape_body[shape]) for shape in (*last["rigid_contact_shape0"], *last["rigid_contact_shape1"])
        }
        self.assertGreater(len(bodies - {-1}), 64)
        return conf.momentum_balance(run, data, torque=False)["free"]

    def _check(self, route: conf.Route, device, context=contextlib.nullcontext):
        bound = conf.MOMENTUM_BOUND["free"]
        warm = self._balance(route, device, warm=True, context=context)
        self.assertLessEqual(warm, bound, f"warm-started momentum balance {warm:.3e} over {bound:.0e} of m g dt")
        cold = self._balance(route, device, warm=False, context=context)
        self.assertLessEqual(cold, bound, f"cold momentum balance {cold:.3e} over {bound:.0e} of m g dt")

    def test_cpu(self):
        """The CPU's matrix-free sweep, one thread per world."""
        self._check(conf.CPU, wp.get_device("cpu"))

    def test_cuda_warp_mapping(self):
        """The matrix-free sweep's one-warp mapping."""
        if not wp.is_cuda_available():
            self.skipTest("No CUDA device")
        self._check(conf.REFERENCE_CUDA, wp.get_device("cuda:0"))

    def test_cuda_colored_mapping(self):
        """The matrix-free sweep's colored mapping."""
        if not wp.is_cuda_available():
            self.skipTest("No CUDA device")
        self._check(conf.REFERENCE_CUDA, wp.get_device("cuda:0"), context=_colored)


if __name__ == "__main__":
    unittest.main(verbosity=2)
