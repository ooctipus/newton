# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Hold the sparse-diagonal contact schedule to its definition over the unit heads.

Every row of a world's contact segment other than a unit's member (a tangent row, or a spin row) heads a unit. A
marked head on a coordinate no coupled unit holds is a scalar contact; the builder levels the scalar contacts over
their coordinates (a coordinate's ``k``-th contact in row order is level ``k``) and lists each level's contacts in row
order, and lists the remaining heads serially. The tests compare every output of the builder, bitwise, with a
transliteration of that definition, on synthetic worlds (unit layouts and layouts no producer writes) and on the rows
the solver produces, which form units.
"""

import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.row_kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_COUNT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
    PGS_CONSTRAINT_TYPE_TORSION,
)
from newton._src.solvers.feather_pgs.sweep_levels import _get_sparse_contact_levels_kernel
from newton.tests.feather_pgs_law_harness import row_lines
from newton.tests.test_feather_pgs_star_rows import build_solver_fixture, install_patch_contacts

_FRICTION = PGS_CONSTRAINT_TYPE_FRICTION
_TORSION = PGS_CONSTRAINT_TYPE_TORSION
# Initial content of every output, so that entries a builder must not write are checked too.
_UNWRITTEN = -7
_OUTPUTS = ("sparse_row_dof", "level_count", "level_offsets", "level_units", "serial_count", "serial")
# (capacity, coordinates) of the synthetic twins: the pinned shape, rows across several chunks of
# 32 with a capacity off the chunk and triple grids, a single coordinate, and 114 coordinates.
_UNIT_SHAPES = ((64, 10), (97, 33), (40, 1), (130, 114))
# Ways the synthetic unit worlds break the unit layout, in the order ``_unit_worlds`` draws them.
_FLAWS = (
    "a tangent row starts the segment",
    "a first tangent row names another row",
    "a second tangent row names the first",
    "a third tangent row follows a unit",
    "the segment ends after a unit's first tangent row",
)


def _schedule(rows, *, serial_contacts):
    """Return every output of the schedule over the unit heads.

    Reserve the coordinates of coupled heads, level each marked head on an unreserved coordinate by its place among its
    coordinate's marked heads (a marked head on a reserved coordinate loses its mark, ``-1``), list each level's heads
    in row order and, when listed, the unmarked heads in row order. Entries the builder does not define keep their
    initial content; the level offsets and units are defined up to the level count and the leveled heads.
    """
    out = {name: rows[name].copy() for name in _OUTPUTS}
    marks = out["sparse_row_dof"]
    for w in range(rows["row_count"].shape[0]):
        start, end = int(rows["bounds"][w, 1]), int(rows["row_count"][w])
        # A unit's members name its head: its tangents and, where it has one, its spin row.
        heads = [row for row in range(start, end) if int(rows["row_type"][w, row]) not in (_FRICTION, _TORSION)]
        scalar = [row for row in heads if np.any(marks[w, row] < -1)]
        reserved = {int(c) for row in heads if row not in scalar for c in marks[w, row] if c >= 0}
        levels, placed = [], {}
        for row in scalar:
            coord_0, coord_1 = (int(value) for value in marks[w, row])
            coordinate = coord_0 if coord_0 >= 0 else coord_1
            if coordinate in reserved:
                marks[w, row, 0 if coord_0 < -1 else 1] = -1
                continue
            level = placed.get(coordinate, 0)
            placed[coordinate] = level + 1
            if level == len(levels):
                levels.append([])
            levels[level].append(row)
        units = [row for level in levels for row in level]
        out["level_count"][w] = len(levels)
        out["level_offsets"][w, : len(levels) + 1] = np.cumsum([0, *(len(level) for level in levels)])
        out["level_units"][w, : len(units)] = units
        if serial_contacts:
            serial = [row for row in heads if not np.any(marks[w, row] < -1)]
            out["serial"][w, : len(serial)] = serial
            out["serial_count"][w] = len(serial)
    return out


def _rows_form_units(rows, world):
    """Whether a world's contact rows form units, each head followed by its two tangent rows or none."""
    start, end = int(rows["bounds"][world, 1]), int(rows["row_count"][world])
    row_type, parent = rows["row_type"][world], rows["row_parent"][world]
    row = start
    while row < end:
        if row_type[row] == _FRICTION:
            return False
        if row + 1 < end and row_type[row + 1] == _FRICTION:
            if row + 2 >= end or row_type[row + 2] != _FRICTION or parent[row + 1] != row or parent[row + 2] != row:
                return False
            row += 3
        else:
            row += 1
    return True


def _empty_rows(worlds, capacity, serial_capacity):
    """Rows of empty contact segments, with every output at its unwritten value."""
    return {
        "row_count": np.zeros(worlds, np.int32),
        "bounds": np.zeros((worlds, 2), np.int32),
        # Rows outside the segments hold a type no row family uses.
        "row_type": np.full((worlds, capacity), PGS_CONSTRAINT_TYPE_COUNT, np.int32),
        "row_parent": np.full((worlds, capacity), -1, np.int32),
        "sparse_row_dof": np.full((worlds, capacity, 2), -1, np.int32),
        "level_count": np.full(worlds, _UNWRITTEN, np.int32),
        "level_offsets": np.full((worlds, capacity + 1), _UNWRITTEN, np.int32),
        "level_units": np.full((worlds, capacity), _UNWRITTEN, np.int32),
        "serial_count": np.full(worlds, _UNWRITTEN, np.int32),
        "serial": np.full((worlds, serial_capacity), _UNWRITTEN, np.int32),
    }


def _mark(rng, rows, world, row, coordinate, dofs, *, scalar):
    """Write a contact row's sparse coordinates as the marker leaves them.

    A scalar contact names one coordinate and marks the other slot ``-2``; otherwise the row is coupled
    to a second coordinate or to the dense articulation, or has no sparse coordinate.
    """
    draw = rng.random()
    if scalar or draw < 0.65:
        slot = int(rng.integers(0, 2))
        rows["sparse_row_dof"][world, row, slot] = coordinate
        rows["sparse_row_dof"][world, row, 1 - slot] = -2
    elif draw < 0.85:
        rows["sparse_row_dof"][world, row] = (coordinate, int(rng.integers(-1, dofs)))


def _unit_worlds(seed, *, worlds, capacity, dofs):
    """Rows of patch contact units, marked as the unit marker leaves them, in layouts that cover the builder.

    Worlds draw one layout each: random units on few coordinates (long chains, reserved coordinates and
    rows without a sparse coordinate), one-row units of one coordinate up to the capacity, units spread
    over every coordinate, three-row units filling the capacity, empty and inverted segments, and rows
    that break the unit layout, which the builder schedules with the serial walks. Broken worlds take
    the flaws of :data:`_FLAWS` in turn, and ``rows["flaw"]`` records each world's flaw (``-1`` where
    the rows keep the layout).
    """
    rng = np.random.default_rng(seed)
    rows = _empty_rows(worlds, capacity, capacity)
    rows["flaw"] = np.full(worlds, -1, np.int32)
    layouts = ("random", "random", "chain", "spread", "full", "empty", "inverted", "broken", "broken")
    broken = 0
    for w in range(worlds):
        layout = layouts[w % len(layouts)]
        start = 0 if layout in ("chain", "spread", "full") else int(rng.integers(1, 4))
        rows["row_type"][w, :start] = PGS_CONSTRAINT_TYPE_JOINT_LIMIT
        hot = rng.integers(0, dofs, size=3)
        row = start
        while layout not in ("empty", "inverted") and row < capacity:
            if layout == "chain":
                count = 1
            elif layout == "full":
                count = 3 if row + 3 <= capacity else 1
            else:
                count = 3 if row + 3 <= capacity and rng.random() < 0.5 else 1
                if rng.random() < 0.08:
                    break
            rows["row_type"][w, row] = PGS_CONSTRAINT_TYPE_CONTACT
            rows["row_parent"][w, row] = row
            rows["row_type"][w, row + 1 : row + count] = _FRICTION
            rows["row_parent"][w, row + 1 : row + count] = row
            coordinate = {"chain": 0, "spread": (7 * row) % dofs}.get(layout, int(rng.choice(hot)))
            _mark(rng, rows, w, row, coordinate, dofs, scalar=layout in ("chain", "spread", "full"))
            rows["sparse_row_dof"][w, row + 1 : row + count, 0] = coordinate
            row += count
        end = start - 1 if layout == "inverted" else row
        if layout == "broken":
            owners = [
                h
                for h in range(start, end - 3)
                if rows["row_type"][w, h] != _FRICTION and rows["row_type"][w, h + 1] == _FRICTION
            ]
            # Flaws past the first need a unit with tangent rows that another row follows.
            flaw = broken % len(_FLAWS) if owners else 0
            broken += 1
            rows["flaw"][w] = flaw
            owner = int(rng.choice(owners)) if owners else start
            if flaw == 0:
                # The segment starts at a tangent row.
                rows["row_type"][w, start] = _FRICTION
                end = max(end, start + 1)
            elif flaw == 1:
                # A first tangent row names another row.
                rows["row_parent"][w, owner + 1] = owner + 2
            elif flaw == 2:
                # A second tangent row names the first, which the reverse walk then visits; a mark there
                # makes the visit change the schedule.
                rows["row_parent"][w, owner + 2] = owner + 1
                rows["sparse_row_dof"][w, owner + 1] = (rows["sparse_row_dof"][w, owner + 1, 0], -2)
            elif flaw == 3:
                # A third tangent row follows a unit.
                rows["row_type"][w, owner + 3] = _FRICTION
                rows["row_parent"][w, owner + 3] = owner
            else:
                # The segment ends after a unit's first tangent row.
                end = owner + 2
        rows["bounds"][w, 1] = start
        rows["row_count"][w] = end
    return rows


def _unit_twin_rows():
    """Yield the synthetic unit worlds of the CUDA twins, one set per shape of :data:`_UNIT_SHAPES`."""
    for seed, (capacity, dofs) in enumerate(_UNIT_SHAPES):
        yield (capacity, dofs), _unit_worlds(seed, worlds=180, capacity=capacity, dofs=dofs)


def _builder_arrays(rows, device):
    """Return the builder's inputs and outputs, in launch order, as device copies of the rows."""
    inputs = ("row_count", "bounds", "row_type")
    arrays = [wp.array(rows[name], dtype=wp.int32, device=device) for name in inputs]
    arrays.insert(1, row_lines(device))
    return (
        arrays,
        {name: wp.array(rows[name], dtype=wp.int32, device=device) for name in _OUTPUTS},
    )


def _launch_builder(rows, device, *, serial_contacts, dofs):
    """Run the schedule builder over the rows; return every output."""
    worlds, capacity = rows["row_type"].shape
    kernel = _get_sparse_contact_levels_kernel(capacity, dofs, str(device.arch), build_serial_contacts=serial_contacts)
    inputs, outputs = _builder_arrays(rows, device)
    wp.launch_tiled(kernel, dim=[worlds], inputs=inputs, outputs=list(outputs.values()), block_dim=32, device=device)
    return {name: array.numpy() for name, array in outputs.items()}


def _assert_outputs_equal(actual, expected, label):
    """Every defined output equal: the marks, counts and serial lists whole; the level offsets up to each world's level
    count and the level units up to its leveled heads."""
    for name in ("sparse_row_dof", "level_count", "serial_count", "serial"):
        np.testing.assert_array_equal(actual[name], expected[name], err_msg=f"{label}: {name}")
    for w, levels in enumerate(expected["level_count"]):
        offsets = expected["level_offsets"][w, : levels + 1]
        np.testing.assert_array_equal(actual["level_offsets"][w, : levels + 1], offsets, err_msg=f"{label}: offsets")
        units = offsets[-1] if levels > 0 else 0
        np.testing.assert_array_equal(
            actual["level_units"][w, :units], expected["level_units"][w, :units], err_msg=f"{label}: units"
        )


def _record_schedules(model, device, *, steps=6):
    """Step a solver over the patch fixture's contacts; return each schedule build's rows before and outputs after."""
    solver = newton.solvers.SolverFeatherPGS(
        model,
        pgs_iterations=8,
        enable_joint_limits=True,
        dense_max_constraints=64,
        mf_max_constraints=64,
    )
    names = ("row_count", "first", "bounds", "row_type", *_OUTPUTS)
    records = []
    launch = wp.launch_tiled

    def recording_launch(kernel, *args, **kwargs):
        if kernel is not solver.sweep.sparse_contact_schedule:
            return launch(kernel, *args, **kwargs)
        arrays = dict(zip(names, (*kwargs["inputs"], *kwargs["outputs"]), strict=True))
        before = {name: array.numpy().copy() for name, array in arrays.items()}
        before["row_parent"] = solver.placement.row_parent.numpy().copy()
        result = launch(kernel, *args, **kwargs)
        records.append((before, {name: arrays[name].numpy().copy() for name in _OUTPUTS}))
        return result

    states = [model.state(), model.state()]
    control = model.control()
    contacts = newton.Contacts(24, 0, device=device)
    newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
    install_patch_contacts(model, states[0], contacts)
    with mock.patch.object(wp, "launch_tiled", recording_launch):
        for _ in range(steps):
            solver.step(states[0], states[1], control, contacts, 1 / 240)
            states.reverse()
    return solver, records


class TestFeatherPGSContactSchedule(unittest.TestCase):
    def test_synthetic_worlds_cover_the_builder(self):
        """Draw unit layouts and broken layouts; the layout predicate rejects exactly the broken worlds, and the
        schedule holds scalar contacts in several levels across batches of 32 and serial heads."""
        longest = deepest = 0
        for (_capacity, _dofs), rows in _unit_twin_rows():
            worlds = rows["row_count"].shape[0]
            forms = np.array([_rows_form_units(rows, world) for world in range(worlds)])
            np.testing.assert_array_equal(forms, rows["flaw"] < 0)
            self.assertGreater(int(np.sum(forms)), worlds // 2)
            schedule = _schedule(rows, serial_contacts=True)
            self.assertGreater(int(np.sum(schedule["level_count"] > 0)), worlds // 3)
            self.assertGreater(int(np.sum(np.maximum(schedule["serial_count"], 0))), worlds)
            longest = max(longest, int(np.max(rows["row_count"] - rows["bounds"][:, 1])))
            deepest = max(deepest, int(np.max(schedule["level_count"])))
        # Segments and a coordinate's contacts cross batches of 32.
        self.assertGreater(longest, 64)
        self.assertGreater(deepest, 32)

    @unittest.skipUnless(wp.is_cuda_available(), "the contact schedule builder requires CUDA")
    def test_builder_matches_the_schedule(self):
        """Build the schedule bitwise equal to its definition over the heads, in every layout and at capacity
        edges."""
        device = wp.get_device("cuda:0")
        for (capacity, dofs), rows in _unit_twin_rows():
            for serial_contacts in (False, True):
                with self.subTest(capacity=capacity, dofs=dofs, serial_contacts=serial_contacts):
                    expected = _schedule(rows, serial_contacts=serial_contacts)
                    actual = _launch_builder(rows, device, serial_contacts=serial_contacts, dofs=dofs)
                    _assert_outputs_equal(actual, expected, f"capacity {capacity}")

    @unittest.skipUnless(wp.is_cuda_available(), "the contact schedule builder requires CUDA")
    def test_builder_replays_in_graphs(self):
        """Rebuild the same schedule from a captured graph."""
        device = wp.get_device("cuda:0")
        rows = _unit_worlds(5, worlds=90, capacity=97, dofs=33)
        expected = _schedule(rows, serial_contacts=True)
        worlds, capacity = rows["row_type"].shape
        kernel = _get_sparse_contact_levels_kernel(capacity, 33, str(device.arch), build_serial_contacts=True)
        inputs, outputs = _builder_arrays(rows, device)
        marked = wp.array(rows["sparse_row_dof"], dtype=wp.int32, device=device)

        def build(kernel=kernel, inputs=inputs, outputs=outputs, marked=marked, worlds=worlds):
            wp.copy(outputs["sparse_row_dof"], marked)
            wp.launch_tiled(
                kernel, dim=[worlds], inputs=inputs, outputs=list(outputs.values()), block_dim=32, device=device
            )

        build()
        with wp.ScopedCapture(device=device) as capture:
            build()
        for replay in range(2):
            for name in ("level_count", "level_offsets", "level_units", "serial_count", "serial"):
                outputs[name].fill_(_UNWRITTEN)
            wp.capture_launch(capture.graph)
            actual = {name: array.numpy() for name, array in outputs.items()}
            _assert_outputs_equal(actual, expected, f"replay {replay}")

    @unittest.skipUnless(wp.is_cuda_available(), "the sparse-diagonal contact owner requires CUDA")
    def test_builder_matches_the_schedule_on_solver_rows(self):
        """Schedule the patch units the solver produces as their definition does.

        Every builder launch of six solver steps is recorded: its rows and marks before the launch and its outputs
        after. The rows form units (the row writer's layout), and both the levels and the serial list are populated.
        """
        device = wp.get_device("cuda:0")
        model = build_solver_fixture(device)
        solver, records = _record_schedules(model, device)
        self.assertTrue(solver.spec.classes.diagonal.units)
        self.assertTrue(solver.spec.classes.diagonal.speculative_batches)
        self.assertEqual(len(records), 6)
        for step, (before, after) in enumerate(records):
            for world in range(before["row_count"].shape[0]):
                self.assertTrue(_rows_form_units(before, world), f"step {step} world {world}")
            _assert_outputs_equal(after, _schedule(before, serial_contacts=True), f"step {step}")
        self.assertGreater(sum(int(np.sum(after["level_count"])) for _, after in records), 0)
        self.assertGreater(sum(int(np.sum(after["serial_count"])) for _, after in records), 0)


if __name__ == "__main__":
    unittest.main()
