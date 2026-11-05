# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Hold the sparse-diagonal contact schedules to the serial walks that define them.

The builders link each coordinate's scalar contacts (patch units) in row order and list the
remaining contacts serially. Two serial walks over a world's rows define these outputs; lane 0 ran them before the warp
built the schedules in parallel. The tests compare every output of the
builders, bitwise, with a transliteration of those walks, on synthetic worlds
and on the rows the solver produces, and check that the unit builder takes its
warp-parallel path exactly where the rows form units.
"""

import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import sweep_levels
from newton._src.solvers.feather_pgs.row_kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_COUNT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
    PGS_CONSTRAINT_TYPE_TORSION,
)
from newton._src.solvers.feather_pgs.sweep_levels import (
    _get_build_independent_sparse_contact_groups_kernel,
)
from newton.tests.feather_pgs_law_harness import row_lines
from newton.tests.test_feather_pgs_compact_contact import (
    build_solver_fixture,
    compact_boundary,
    install_patch_contacts,
)
from newton.tests.test_feather_pgs_response_diagonal import _mark_patch_units, _patch_unit_cases, _unit_schedule_oracle

_FRICTION = PGS_CONSTRAINT_TYPE_FRICTION
_TORSION = PGS_CONSTRAINT_TYPE_TORSION
# Initial content of every output, so that entries a builder must not write are checked too.
_UNWRITTEN = -7
_OUTPUTS = ("sparse_row_dof", "group_count", "group_heads", "serial_count", "serial")
# (capacity, coordinates) of the synthetic twins: the pinned shape, rows across several chunks of
# 32 with a capacity off the chunk and triple grids, a single coordinate, and 114 coordinates (seven
# coordinate bits, four chunks of coordinates to compact).
_UNIT_SHAPES = ((64, 10), (97, 33), (40, 1), (130, 114))
_TRIPLE_SHAPES = ((64, 10), (98, 33), (40, 1), (300, 114))
# Ways the synthetic unit worlds break the unit layout, in the order ``_unit_worlds`` draws them.
_FLAWS = (
    "a tangent row starts the segment",
    "a first tangent row names another row",
    "a second tangent row names the first",
    "a third tangent row follows a unit",
    "the segment ends after a unit's first tangent row",
)
# Flaws that can make the reverse walk visit a row the parallel path skips, so that a mark on that row
# changes the links. The others change only the serial list.
_LINK_FLAWS = (0, 2, 3)


def _reserve(links, world, heads_of_contacts, dofs):
    """Return the coordinates' chain heads after reserving those of coupled contacts (``-2``)."""
    heads = np.full(dofs, -1)
    for head in heads_of_contacts:
        coords = links[world, head]
        if not np.any(coords < -1):
            heads[coords[coords >= 0]] = -2
    return heads


def _prepend(links, heads, world, head):
    """Prepend a scalar contact to its coordinate's chain unless a coupled contact reserved it."""
    coord_0, coord_1 = (int(value) for value in links[world, head])
    if coord_0 < -1 or coord_1 < -1:
        coordinate = coord_0 if coord_0 >= 0 else coord_1
        if heads[coordinate] != -2:
            following = int(heads[coordinate])
            links[world, head, 0 if coord_0 < 0 else 1] = -(following + 3) if following >= 0 else -2
            heads[coordinate] = head
        else:
            links[world, head, 0 if coord_0 < -1 else 1] = -1


def _finish(out, world, heads, serial):
    """Store the chain heads compacted by coordinate and, when listed, the serial contacts."""
    chained = heads[heads >= 0]
    out["group_heads"][world, : len(chained)] = chained
    out["group_heads"][world, len(chained) :] = heads[len(chained) :]
    out["group_count"][world] = len(chained)
    if serial is not None:
        out["serial"][world, : len(serial)] = serial
        out["serial_count"][world] = len(serial)


def _serial_unit_walks(rows, *, serial_contacts):
    """Return every output of the serial walks over patch contact units.

    A transliteration of the code lane 0 ran: reserve the coordinates of coupled
    units, walk back from the segment end prepending each scalar unit to its
    coordinate's chain (a tangent row steps to its parent when the parent lies
    before it in the segment), list the remaining units walking forward from the
    segment start (a head that owns two tangent rows steps over them), and compact
    the chain heads by coordinate. Entries the builder does not write keep their
    initial content.
    """
    out = {name: rows[name].copy() for name in _OUTPUTS}
    links = out["sparse_row_dof"]
    for w in range(rows["row_count"].shape[0]):
        start, end = int(rows["bounds"][w, 1]), int(rows["row_count"][w])
        row_type, parent = rows["row_type"][w], rows["row_parent"][w]
        # A unit's members name its head: its tangents and, where it has one, its spin row.
        member = [int(kind) in (_FRICTION, _TORSION) for kind in row_type]
        heads = _reserve(links, w, [row for row in range(start, end) if not member[row]], len(out["group_heads"][w]))
        row = end - 1
        while row >= start:
            head = row
            if member[row] and start <= parent[row] < row:
                head = int(parent[row])
            _prepend(links, heads, w, head)
            row = head - 1
        serial = None
        if serial_contacts:
            serial, head = [], start
            while head < end:
                if not np.any(links[w, head] < -1):
                    serial.append(head)
                owns_tangents = head + 2 < end and row_type[head + 1] == _FRICTION and parent[head + 1] == head
                owns_spin = (
                    owns_tangents and head + 3 < end and row_type[head + 3] == _TORSION and parent[head + 3] == head
                )
                head += (4 if owns_spin else 3) if owns_tangents else 1
        _finish(out, w, heads, serial)
    return out


def _schedule_unit_heads(rows, *, serial_contacts):
    """Return every output of scheduling exactly the non-friction rows as unit heads.

    These are the outputs of the builder's warp-parallel path. They equal the serial walks' outputs when
    the rows form units (:func:`_rows_form_units`), because both walks then visit exactly these rows.
    """
    out = {name: rows[name].copy() for name in _OUTPUTS}
    links = out["sparse_row_dof"]
    for w in range(rows["row_count"].shape[0]):
        start, end = int(rows["bounds"][w, 1]), int(rows["row_count"][w])
        unit_heads = [row for row in range(start, end) if rows["row_type"][w, row] != _FRICTION]
        heads = _reserve(links, w, unit_heads, len(out["group_heads"][w]))
        for head in reversed(unit_heads):
            _prepend(links, heads, w, head)
        serial = [head for head in unit_heads if not np.any(links[w, head] < -1)] if serial_contacts else None
        _finish(out, w, heads, serial)
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


def _empty_rows(worlds, capacity, dofs, serial_capacity):
    """Rows of empty contact segments, with every output at its unwritten value."""
    return {
        "row_count": np.zeros(worlds, np.int32),
        "bounds": np.zeros((worlds, 2), np.int32),
        # Rows outside the segments hold a type no row family uses.
        "row_type": np.full((worlds, capacity), PGS_CONSTRAINT_TYPE_COUNT, np.int32),
        "row_parent": np.full((worlds, capacity), -1, np.int32),
        "sparse_row_dof": np.full((worlds, capacity, 2), -1, np.int32),
        "group_count": np.full(worlds, _UNWRITTEN, np.int32),
        "group_heads": np.full((worlds, dofs), _UNWRITTEN, np.int32),
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
    rows = _empty_rows(worlds, capacity, dofs, capacity)
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
    inputs = ("row_count", "bounds", "row_type", "row_parent")
    arrays = [wp.array(rows[name], dtype=wp.int32, device=device) for name in inputs]
    arrays.insert(1, row_lines(device))
    return (
        arrays,
        {name: wp.array(rows[name], dtype=wp.int32, device=device) for name in _OUTPUTS},
    )


def _launch(kernel, rows, device):
    """Run a schedule builder kernel over the rows; return every output."""
    inputs, outputs = _builder_arrays(rows, device)
    wp.launch(kernel, dim=rows["row_type"].shape[0] * 32, inputs=inputs, outputs=list(outputs.values()), device=device)
    return {name: array.numpy() for name, array in outputs.items()}


def _launch_builder(rows, device, *, serial_contacts):
    """Run a schedule builder over the rows; return every output."""
    capacity, dofs = rows["row_type"].shape[1], rows["group_heads"].shape[1]
    kernel = _get_build_independent_sparse_contact_groups_kernel(
        capacity, dofs, str(device.arch), build_serial_contacts=serial_contacts
    )
    return _launch(kernel, rows, device)


def _unit_builder_path_marks(rows, device):
    """Return, per world, ``1`` where the unit builder takes its warp-parallel path, else :data:`_UNWRITTEN`.

    The builder is rebuilt without serial lists, so that nothing else writes the serial count, and with a
    lane-0 store of ``1`` there after its parallel links, which only the parallel path runs.
    """
    link_source = sweep_levels._link_scalar_contacts_source

    def marked_link_source(**kwargs):
        return f"{link_source(**kwargs)}\n    if (lane == 0) serial_count_out.data[world] = 1;\n"

    capacity, dofs = rows["row_type"].shape[1], rows["group_heads"].shape[1]
    with mock.patch.object(sweep_levels, "_link_scalar_contacts_source", marked_link_source):
        kernel = sweep_levels._build_independent_sparse_contact_units_kernel(capacity, dofs, False)
    unmarked = {**rows, "serial_count": np.full(rows["row_type"].shape[0], _UNWRITTEN, np.int32)}
    return _launch(kernel, unmarked, device)["serial_count"]


def _assert_outputs_equal(actual, expected, label):
    for name in _OUTPUTS:
        np.testing.assert_array_equal(actual[name], expected[name], err_msg=f"{label}: {name}")


def _differing_worlds(first, second):
    """Return, per world, whether any output of two schedules differs."""
    worlds = len(first["group_count"])
    return np.any([np.any((first[name] != second[name]).reshape(worlds, -1), axis=1) for name in _OUTPUTS], axis=0)


def _record_schedules(model, device, *, compact, steps=6):
    """Step a solver over the patch fixture's contacts; return each schedule build's rows before and outputs after."""
    with (
        compact_boundary(compact),
    ):
        solver = newton.solvers.SolverFeatherPGS(
            model,
            pgs_iterations=8,
            enable_joint_limits=True,
            dense_max_constraints=64,
            mf_max_constraints=64,
        )
    names = ("row_count", "first", "bounds", "row_type", "row_parent", *_OUTPUTS)
    records = []
    launch = wp.launch

    def recording_launch(kernel, *args, **kwargs):
        if kernel is not solver.sweep.sparse_contact_schedule:
            return launch(kernel, *args, **kwargs)
        arrays = dict(zip(names, (*kwargs["inputs"], *kwargs["outputs"]), strict=True))
        before = {name: array.numpy().copy() for name, array in arrays.items()}
        result = launch(kernel, *args, **kwargs)
        records.append((before, {name: arrays[name].numpy().copy() for name in _OUTPUTS}))
        return result

    states = [model.state(), model.state()]
    control = model.control()
    contacts = newton.Contacts(24, 0, device=device)
    newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
    install_patch_contacts(model, states[0], contacts)
    with mock.patch.object(wp, "launch", recording_launch):
        for _ in range(steps):
            solver.step(states[0], states[1], control, contacts, 1 / 240)
            states.reverse()
    return solver, records


class TestFeatherPGSContactSchedule(unittest.TestCase):
    def test_serial_unit_walks_match_the_unit_oracle(self):
        """Reproduce the unit schedule oracle with the serial walks and with the schedule over the heads alone."""
        device = wp.get_device("cpu")
        for (dense, seed, capacity), rows in _patch_unit_cases():
            with self.subTest(dense=dense, seed=seed, capacity=capacity):
                marked = _mark_patch_units(rows, device)
                expected = _unit_schedule_oracle(rows, marked)
                worlds, dofs = rows["worlds"], rows["dofs"]
                inputs = {
                    **_empty_rows(worlds, capacity, dofs, capacity),
                    "row_count": rows["row_count"],
                    "bounds": rows["bounds"],
                    "row_type": rows["row_type"],
                    "row_parent": rows["row_parent"],
                    "sparse_row_dof": marked,
                }
                walks = _serial_unit_walks(inputs, serial_contacts=True)
                _assert_outputs_equal(_schedule_unit_heads(inputs, serial_contacts=True), walks, "heads alone")
                np.testing.assert_array_equal(walks["sparse_row_dof"], expected["sparse_row_dof"])
                np.testing.assert_array_equal(walks["group_count"], expected["group_count"])
                for world in range(worlds):
                    self.assertTrue(_rows_form_units(inputs, world))
                    count = int(expected["group_count"][world])
                    np.testing.assert_array_equal(
                        walks["group_heads"][world, :count], expected["group_heads"][world, :count]
                    )
                    heads = expected["serial"][world]
                    self.assertEqual(int(walks["serial_count"][world]), len(heads))
                    np.testing.assert_array_equal(walks["serial"][world, : len(heads)], heads)

    def test_synthetic_worlds_cover_both_builder_paths(self):
        """Draw unit layouts the builder takes in parallel and broken layouts it must leave to the serial walks.

        The layout check rejects exactly the broken worlds, and wherever the rows form units the schedule
        over the heads alone, which the parallel path builds, equals the serial walks'. In every shape, each
        flaw breaks a world where the two differ with serial lists built, and each flaw of
        :data:`_LINK_FLAWS` one where they differ without, so a layout check that admitted the flaw would
        fail the CUDA twin.
        """
        longest = 0
        for (capacity, dofs), rows in _unit_twin_rows():
            worlds = rows["row_count"].shape[0]
            forms = np.array([_rows_form_units(rows, world) for world in range(worlds)])
            np.testing.assert_array_equal(forms, rows["flaw"] < 0)
            self.assertGreater(int(np.sum(forms)), worlds // 2)
            for serial_contacts in (False, True):
                walks = _serial_unit_walks(rows, serial_contacts=serial_contacts)
                differ = _differing_worlds(walks, _schedule_unit_heads(rows, serial_contacts=serial_contacts))
                self.assertFalse(np.any(differ & forms), f"capacity {capacity}: the heads schedule left the walks")
                for flaw in range(len(_FLAWS)) if serial_contacts else _LINK_FLAWS:
                    with self.subTest(capacity=capacity, dofs=dofs, serial_contacts=serial_contacts, flaw=_FLAWS[flaw]):
                        self.assertTrue(np.any(differ & (rows["flaw"] == flaw)))
                if serial_contacts:
                    # Both schedules are populated.
                    self.assertGreater(int(np.sum(walks["group_count"])), worlds // 3)
                    self.assertGreater(int(np.sum(np.maximum(walks["serial_count"], 0))), worlds)
            longest = max(longest, int(np.max(rows["row_count"] - rows["bounds"][:, 1])))
        # Chains cross chunks of 32 rows.
        self.assertGreater(longest, 64)

    @unittest.skipUnless(wp.is_cuda_available(), "the contact schedule builders require CUDA")
    def test_unit_builder_matches_the_serial_walks(self):
        """Build unit schedules bitwise equal to the serial walks, in every layout and at capacity edges."""
        device = wp.get_device("cuda:0")
        for (capacity, dofs), rows in _unit_twin_rows():
            for serial_contacts in (False, True):
                with self.subTest(capacity=capacity, dofs=dofs, serial_contacts=serial_contacts):
                    expected = _serial_unit_walks(rows, serial_contacts=serial_contacts)
                    actual = _launch_builder(rows, device, serial_contacts=serial_contacts)
                    _assert_outputs_equal(actual, expected, f"capacity {capacity}")

    @unittest.skipUnless(wp.is_cuda_available(), "the contact schedule builders require CUDA")
    def test_unit_builder_takes_the_parallel_path_where_rows_form_units(self):
        """Schedule exactly the synthetic worlds whose rows form units on the unit builder's warp-parallel path.

        The serial walks give the same outputs on either path, so only a builder that marks its path tells
        them apart.
        """
        device = wp.get_device("cuda:0")
        for (capacity, dofs), rows in _unit_twin_rows():
            with self.subTest(capacity=capacity, dofs=dofs):
                forms = [_rows_form_units(rows, world) for world in range(rows["row_count"].shape[0])]
                marks = _unit_builder_path_marks(rows, device)
                np.testing.assert_array_equal(marks, np.where(forms, 1, _UNWRITTEN))

    @unittest.skipUnless(wp.is_cuda_available(), "the contact schedule builders require CUDA")
    def test_builders_replay_in_graphs(self):
        """Rebuild the same unit schedules from captured graphs as the serial walks give."""
        device = wp.get_device("cuda:0")
        rows = _unit_worlds(5, worlds=90, capacity=97, dofs=33)
        expected = _serial_unit_walks(rows, serial_contacts=True)
        worlds, capacity = rows["row_type"].shape
        kernel = _get_build_independent_sparse_contact_groups_kernel(
            capacity, 33, str(device.arch), build_serial_contacts=True
        )
        inputs, outputs = _builder_arrays(rows, device)
        marked = wp.array(rows["sparse_row_dof"], dtype=wp.int32, device=device)

        def build(kernel=kernel, inputs=inputs, outputs=outputs, marked=marked, worlds=worlds):
            wp.copy(outputs["sparse_row_dof"], marked)
            wp.launch(kernel, dim=worlds * 32, inputs=inputs, outputs=list(outputs.values()), device=device)

        build()
        with wp.ScopedCapture(device=device) as capture:
            build()
        for replay in range(2):
            for name in ("group_count", "group_heads", "serial_count", "serial"):
                outputs[name].fill_(_UNWRITTEN)
            wp.capture_launch(capture.graph)
            actual = {name: array.numpy() for name, array in outputs.items()}
            _assert_outputs_equal(actual, expected, f"replay {replay}")

    @unittest.skipUnless(wp.is_cuda_available(), "the sparse-diagonal contact owner requires CUDA")
    def test_unit_builder_matches_the_serial_walks_on_solver_rows(self):
        """Schedule the patch units the solver produces, through both producers, as the serial walks do.

        Every builder launch of six solver steps is recorded: its rows and marks before the launch and
        its outputs after. The rows satisfy the layout predicate, a builder that marks its path takes
        the warp-parallel path in every world, and both the independent and the serial schedule are
        populated.
        """
        device = wp.get_device("cuda:0")
        model = build_solver_fixture(device)
        for compact in (False, True):
            with self.subTest(compact=compact):
                solver, records = _record_schedules(model, device, compact=compact)
                self.assertTrue(solver.placement.sparse_diagonal_contact_units)
                self.assertTrue(solver.placement.sparse_diagonal_speculative_contact_batches)
                self.assertEqual(solver.placement.compact_contact_boundary, compact)
                self.assertEqual(len(records), 6)
                for step, (before, after) in enumerate(records):
                    for world in range(before["row_count"].shape[0]):
                        self.assertTrue(_rows_form_units(before, world), f"step {step} world {world}")
                    marks = _unit_builder_path_marks(before, device)
                    np.testing.assert_array_equal(marks, 1, err_msg=f"step {step}: worlds left the parallel path")
                    _assert_outputs_equal(after, _serial_unit_walks(before, serial_contacts=True), f"step {step}")
                self.assertGreater(sum(int(np.sum(after["group_count"])) for _, after in records), 0)
                self.assertGreater(sum(int(np.sum(after["serial_count"])) for _, after in records), 0)


if __name__ == "__main__":
    unittest.main()
