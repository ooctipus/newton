# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""A world's rows sit where its row tables say, at whatever length the store's lines have.

The tables hold the identity on every model, which would hide a read that ignores them. Here every table is reversed, so
each world's rows sit on another world's line: a read that takes its line from the world instead shows as a state that
differs from the identity's, on every route of the conformance catalogue.

The store's lines are as long as the capacities on every model, which would hide a read that takes a line's length from
the capacities instead of the store. Here the lines are as short as the rows each scene holds, so a line's rows run
into the next line's capacity: such a read shows as a state that differs from the full lines'.
"""

import math
import unittest
from unittest import mock

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs.placement import Placement
from newton.tests import test_feather_pgs_conformance as conformance

_construct = Placement.__init__
_close = Placement.close_rows

_LINE_QUANTUM = 4
"""Short lines' rows: a 16-byte line of floats. A world's rows stay on its own line, so its friction records (a slot per
three rows) do too."""


def _reversed_lines(self, *args, **kwargs):
    _construct(self, *args, **kwargs)
    lines = self.dense_first.shape[0]
    self.dense_first.assign(np.arange(lines - 1, -1, -1, dtype=np.int32))


def _states(group, route, device):
    physics = group.physics
    # The pipeline sorts its contacts, so two runs of one route agree, tables or not.
    run = conformance.simulate(physics.scene, route, device, options=physics.options(physics.iterations), steps=30)
    return {name: getattr(run, name) for name in conformance.STATE_FIELDS}


def _extent(group, route, device) -> tuple[dict, int]:
    """The route's states on full lines and the most rows any world's line held (its dense rows, or its free rows'
    end past its body base), read after each row pass."""
    held = [0]

    def close(self, *args, **kwargs):
        _close(self, *args, **kwargs)
        dense = self.constraint_count.numpy()
        free = self.mf_constraint_count.numpy()
        bases = self.body_base.numpy()[self.dense_first.numpy()]
        ends = np.where(free > 0, bases + free, np.minimum(dense, self.dense_max_constraints))
        held[0] = max(held[0], int(ends.max(initial=0)))

    with mock.patch.object(Placement, "close_rows", close):
        states = _states(group, route, device)
    return states, held[0]


def _short_lines(rows: int):
    def construct(self, *args, **kwargs):
        _construct(self, *args, **kwargs)
        self.line_rows = min(self.row_pitch, rows)
        self.arrays.line_strides.update(row=self.line_rows, contact=-(-self.line_rows // 3))

    return construct


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

    def _check_short(self, groups, device):
        for group in groups:
            for route in group.routes:
                with self.subTest(group=group.name, route=route.name):
                    full, held = _extent(group, route, device)
                    rows = max(math.ceil(held / _LINE_QUANTUM), 1) * _LINE_QUANTUM
                    with mock.patch.object(Placement, "__init__", _short_lines(rows)):
                        short = _states(group, route, device)
                    for name, value in full.items():
                        np.testing.assert_array_equal(value, short[name], err_msg=name)

    def test_worlds_keep_their_physics_on_other_lines_cpu(self):
        """Run the CPU's routes with every row table reversed, bitwise as with the identity."""
        self._check(conformance.cpu_groups(), wp.get_device("cpu"))

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA only")
    def test_worlds_keep_their_physics_on_other_lines_cuda(self):
        """Run the CUDA matrix's routes with every row table reversed, bitwise as with the identity."""
        self._check(conformance.cuda_groups(), wp.get_device("cuda"))

    def test_worlds_keep_their_physics_on_short_lines_cpu(self):
        """Run the CPU's routes on lines as short as their rows, bitwise as on the capacities' lines."""
        self._check_short(conformance.cpu_groups(), wp.get_device("cpu"))

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA only")
    def test_worlds_keep_their_physics_on_short_lines_cuda(self):
        """Run the CUDA matrix's routes on lines as short as their rows, bitwise as on the capacities' lines."""
        self._check_short(conformance.cuda_groups(), wp.get_device("cuda"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
