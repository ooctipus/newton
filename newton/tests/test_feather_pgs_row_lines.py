# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""A world's rows sit where its row tables say.

The tables hold the identity on every model, which would hide a read that ignores them. Here every table is reversed, so
each world's rows sit on another world's line: a read that takes its line from the world instead shows as a state that
differs from the identity's, on every route of the conformance catalogue.
"""

import dataclasses
import unittest
from unittest import mock

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs.placement import Placement
from newton.tests import test_feather_pgs_conformance as conformance

_construct = Placement.__init__


def _reversed_lines(self, *args, **kwargs):
    _construct(self, *args, **kwargs)
    lines = self.dense_first.shape[0]
    self.dense_first.assign(np.arange(lines - 1, -1, -1, dtype=np.int32))


def _states(group, route, device):
    physics = group.physics
    # Contacts in a fixed order: otherwise two runs of one route differ, tables or not.
    scene = dataclasses.replace(physics.scene, deterministic_contacts=True)
    run = conformance.simulate(scene, route, device, options=physics.options(physics.iterations), steps=30)
    return {name: getattr(run, name) for name in conformance.STATE_FIELDS}


class TestFeatherPGSRowLines(unittest.TestCase):
    def _check(self, groups, device):
        for group in groups:
            for route in group.routes:
                with self.subTest(group=group.name, route=route.name):
                    identity = _states(group, route, device)
                    with mock.patch.object(Placement, "__init__", _reversed_lines):
                        moved = _states(group, route, device)
                    for name, value in identity.items():
                        np.testing.assert_array_equal(value, moved[name], err_msg=name)

    def test_worlds_keep_their_physics_on_other_lines_cpu(self):
        """Run the CPU's routes with every row table reversed, bitwise as with the identity."""
        self._check(conformance.cpu_groups(), wp.get_device("cpu"))

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA only")
    def test_worlds_keep_their_physics_on_other_lines_cuda(self):
        """Run the CUDA matrix's routes with every row table reversed, bitwise as with the identity."""
        self._check(conformance.cuda_groups(), wp.get_device("cuda"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
