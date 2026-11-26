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

from newton._src.solvers.feather_pgs.placement import MOVE_LANES, Placement, _move_topology
from newton.tests import test_feather_pgs_conformance as conformance

_construct = Placement.__init__
_close = Placement.close_rows

_LINE_QUANTUM = 4
"""Short lines' rows: a 16-byte line of floats. A world's rows stay on its own line, so its friction records (a slot per
three rows) do too."""


def _reversed_lines(self, *args, **kwargs):
    _construct(self, *args, **kwargs)
    lines = self.home_first.shape[0]
    self.home_first.assign(np.arange(lines - 1, -1, -1, dtype=np.int32))
    self.dense_first.assign(self.home_first)


def _states(group, route, device, collide_every: int = 1):
    physics = group.physics
    # The pipeline sorts its contacts, so two runs of one route agree, tables or not.
    options = physics.options(physics.iterations)
    run = conformance.simulate(physics.scene, route, device, options=options, steps=30, collide_every=collide_every)
    return {name: getattr(run, name) for name in conformance.STATE_FIELDS}


def _extent(group, route, device, collide_every: int = 1) -> tuple[dict, int, int, int]:
    """The route's states on full lines, the most rows any world's line held (its dense rows, or its free rows' end
    past its body base) and its topology rows, read after each row pass, and its worlds."""
    held = [0, 0, 0]

    def close(self, *args, **kwargs):
        _close(self, *args, **kwargs)
        dense = self.constraint_count.numpy()
        free = self.mf_constraint_count.numpy()
        bases = self.body_base.numpy()[self.dense_first.numpy()]
        ends = np.where(free > 0, bases + free, np.minimum(dense, self.dense_max_constraints))
        held[0] = max(held[0], int(ends.max(initial=0)))
        held[1] = max(held[1], int(np.minimum(self.topology_rows.numpy(), dense).max(initial=0)))
        held[2] = self.forest.world_count

    with mock.patch.object(Placement, "close_rows", close):
        states = _states(group, route, device, collide_every)
    return states, held[0], held[1], held[2]


def _moved_topology(device):
    """Two worlds' topology rows moved home -> run (:func:`_move_topology`) in four fields that count a line in their
    own items, on lines of six rows: rows (a float a row), records (eight words a row), a friction metric (a slot of
    four floats per three rows) and units (an int a row, four rows a line: a field that holds fewer rows a world than
    the store's line). Returns the fields before and after, the worlds' runs and their topology rows."""
    rows, lines = 6, 6
    fields = [
        np.arange(lines * rows, dtype=np.float32).reshape(lines, rows) + 1.0,
        np.arange(lines * rows * 8, dtype=np.int32).reshape(lines, rows * 8) + 1,
        np.arange(lines * 2 * 4, dtype=np.float32).reshape(lines, 2, 4) + 1.0,
        np.arange(lines * 4, dtype=np.int32).reshape(lines, 4) + 1,
    ]
    arrays = [wp.array(field, device=device) for field in fields]
    runs, topology = [2, 4], [4, 1]
    wp.launch(
        _move_topology,
        dim=(2, len(arrays), MOVE_LANES),
        inputs=[
            wp.array([0, 1], dtype=wp.int32, device=device),
            wp.array([wp.vec2i(run, 1) for run in runs], dtype=wp.vec2i, device=device),
            wp.array(topology, dtype=wp.int32, device=device),
            wp.array([int(a.ptr) for a in arrays], dtype=wp.uint64, device=device),
            wp.array([int(a.strides[1]) for a in arrays], dtype=wp.int64, device=device),
            wp.array([int(a.strides[0]) for a in arrays], dtype=wp.int64, device=device),
            wp.array([rows, rows, rows, 4], dtype=wp.int64, device=device),
        ],
        device=device,
    )
    return fields, [a.numpy() for a in arrays], runs, topology


def _laid_out(rows: int, arena: int):
    def construct(self, *args, **kwargs):
        _construct(self, *args, **kwargs)
        self.lay_out_lines(min(self.row_pitch, rows), arena)

    return construct


def _quantum(rows: int) -> int:
    return max(math.ceil(rows / _LINE_QUANTUM), 1) * _LINE_QUANTUM


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

    def _check_short(self, groups, device, *, runs: bool = False, collide_every: int = 1):
        """Each route on lines as short as its rows, or (``runs``) as short as its topology rows, the rest of its rows
        in runs of an arena of a line a world for each line its rows hold; a collision pass every ``collide_every``
        steps (the steps between keep its frame: a world holds its slots and its run)."""
        for group in groups:
            for route in group.routes:
                with self.subTest(group=group.name, route=route.name):
                    full, held, topology, worlds = _extent(group, route, device, collide_every)
                    rows, arena = _quantum(held), 0
                    if runs:
                        rows = _quantum(topology)
                        arena = worlds * math.ceil(held / rows)
                    with mock.patch.object(Placement, "__init__", _laid_out(rows, arena)):
                        short = _states(group, route, device, collide_every)
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

    def _check_moved_topology(self, device):
        before, after, runs, topology = _moved_topology(device)
        expected = [field.copy() for field in before]
        for home, (run, count) in enumerate(zip(runs, topology, strict=True)):
            expected[0][run, :count] = before[0][home, :count]
            expected[1][run, : count * 8] = before[1][home, : count * 8]
            slots = math.ceil(count / 3)
            expected[2][run, :slots] = before[2][home, :slots]
            expected[3][run, :count] = before[3][home, :count]
        for name, want, got in zip(("rows", "records", "friction metric", "units"), expected, after, strict=True):
            np.testing.assert_array_equal(got, want, err_msg=name)

    def test_a_run_takes_its_topology_rows_in_every_field_cpu(self):
        """Move two worlds' topology rows to their runs in fields that count a line in rows, in records' words and in
        friction slots of three rows: each takes exactly its rows' items, and no line past its run changes."""
        self._check_moved_topology(wp.get_device("cpu"))

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA only")
    def test_a_run_takes_its_topology_rows_in_every_field_cuda(self):
        """The topology rows' move on the device, as on the CPU."""
        self._check_moved_topology(wp.get_device("cuda"))

    def test_a_full_arena_drops_rows_with_status_cpu(self):
        """Run a stack on lines that hold only its topology rows and an arena of one line, too short for its run: its
        contact rows have no run, so they are rejected past its line, the arena records the refused claim and the step
        fails closed on the dropped rows instead of writing past its line."""
        device = wp.get_device("cpu")
        group = next(group for group in conformance.cpu_groups() if group.name == "box_stack_patch")
        route = group.routes[0]
        _, held, topology, _worlds = _extent(group, route, device)
        rows = _quantum(topology)
        self.assertGreater(held, 2 * rows, "the stack's contact rows pass two of its topology rows' lines")
        placements = []

        def construct(self, *args, **kwargs):
            _laid_out(rows, 1)(self, *args, **kwargs)
            placements.append(self)

        with mock.patch.object(Placement, "__init__", construct):
            with self.assertRaisesRegex(RuntimeError, "dropped"):
                _states(group, route, device)
        self.assertGreater(int(placements[-1].arena_record_host.numpy()[1]), 0)

    def test_worlds_keep_their_physics_in_arena_runs_cpu(self):
        """Run the CPU's routes on lines that hold only their topology rows, every world's contact rows in a run of the
        arena's lines, bitwise as on the capacities' lines."""
        self._check_short(conformance.cpu_groups(), wp.get_device("cpu"), runs=True)

    def test_worlds_keep_their_runs_through_kept_frames_cpu(self):
        """Run the CPU's routes with arena runs and a collision pass every other step, bitwise as on the capacities'
        lines: a step that keeps its frame keeps each world's slots and run, its topology rows moved to it anew."""
        self._check_short(conformance.cpu_groups(), wp.get_device("cpu"), runs=True, collide_every=2)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA only")
    def test_worlds_keep_their_physics_in_arena_runs_cuda(self):
        """Run the CUDA matrix's routes on lines that hold only their topology rows, every world's contact rows in a run
        of the arena's lines, bitwise as on the capacities' lines."""
        self._check_short(conformance.cuda_groups(), wp.get_device("cuda"), runs=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
