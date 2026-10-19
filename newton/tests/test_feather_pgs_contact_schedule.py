# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Hold the sparse-diagonal contact schedules to the serial walks that define them.

The builders link each coordinate's scalar contact units in row order and list
the remaining units serially. Two serial walks over a world's rows define these
outputs; lane 0 ran them before the warp built the schedules in parallel. The
tests compare every output of the builders, bitwise, with a transliteration of
those walks, on synthetic worlds and on the rows the solver produces.
"""

import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import solver_feather_pgs as solver_module
from newton._src.solvers.feather_pgs.kernels import PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_FRICTION
from newton._src.solvers.feather_pgs.solver_feather_pgs import _get_build_independent_sparse_contact_groups_kernel
from newton.tests.test_feather_pgs_compact_contact import build_solver_fixture, install_patch_contacts
from newton.tests.test_feather_pgs_response_diagonal import _mark_patch_units, _patch_unit_cases, _unit_schedule_oracle

_FRICTION = PGS_CONSTRAINT_TYPE_FRICTION
# Initial content of every output, so that entries a builder must not write are checked too.
_UNWRITTEN = -7
_OUTPUTS = ("sparse_row_dof", "group_count", "group_heads", "serial_count", "serial")


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
    dofs = out["group_heads"].shape[1]
    for w in range(rows["row_count"].shape[0]):
        start, end = int(rows["bounds"][w, 1]), int(rows["row_count"][w])
        row_type, parent = rows["row_type"][w], rows["row_parent"][w]
        heads = np.full(dofs, -1)
        for row in range(start, end):
            coords = links[w, row]
            if row_type[row] != _FRICTION and not np.any(coords < -1):
                heads[coords[coords >= 0]] = -2
        row = end - 1
        while row >= start:
            head = row
            if row_type[row] == _FRICTION and start <= parent[row] < row:
                head = int(parent[row])
            coord_0, coord_1 = (int(value) for value in links[w, head])
            if coord_0 < -1 or coord_1 < -1:
                coordinate = coord_0 if coord_0 >= 0 else coord_1
                if heads[coordinate] != -2:
                    following = int(heads[coordinate])
                    links[w, head, 0 if coord_0 < 0 else 1] = -(following + 3) if following >= 0 else -2
                    heads[coordinate] = head
                else:
                    links[w, head, 0 if coord_0 < -1 else 1] = -1
            row = head - 1
        if serial_contacts:
            count, head = 0, start
            while head < end:
                if not np.any(links[w, head] < -1):
                    out["serial"][w, count] = head
                    count += 1
                owns_tangents = head + 2 < end and row_type[head + 1] == _FRICTION and parent[head + 1] == head
                head += 3 if owns_tangents else 1
            out["serial_count"][w] = count
        chained = heads[heads >= 0]
        out["group_heads"][w, : len(chained)] = chained
        out["group_heads"][w, len(chained) :] = heads[len(chained) :]
        out["group_count"][w] = len(chained)
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


def _unit_worlds(seed, *, worlds, capacity, dofs):
    """Rows of patch contact units, marked as the unit marker leaves them, in layouts that cover the builder.

    Worlds draw one layout each: random units on few coordinates (long chains, reserved coordinates and
    rows without a sparse coordinate), one-row units of one coordinate up to the capacity, units spread
    over every coordinate, three-row units filling the capacity, empty and inverted segments, and rows
    that break the unit layout, which the builder schedules with the serial walks: a tangent row starts
    the segment, a first tangent row names another row, a second tangent row names the first, a third
    tangent row follows a unit, or the segment ends after a unit's first tangent row.
    """
    rng = np.random.default_rng(seed)
    rows = {
        "row_count": np.zeros(worlds, np.int32),
        "bounds": np.zeros((worlds, 2), np.int32),
        "row_type": np.full((worlds, capacity), 7, np.int32),
        "row_parent": np.full((worlds, capacity), -1, np.int32),
        "sparse_row_dof": np.full((worlds, capacity, 2), -1, np.int32),
        "group_count": np.full(worlds, _UNWRITTEN, np.int32),
        "group_heads": np.full((worlds, dofs), _UNWRITTEN, np.int32),
        "serial_count": np.full(worlds, _UNWRITTEN, np.int32),
        "serial": np.full((worlds, capacity), _UNWRITTEN, np.int32),
    }
    layouts = ("random", "random", "chain", "spread", "full", "empty", "inverted", "broken", "broken")
    for w in range(worlds):
        layout = layouts[w % len(layouts)]
        start = 0 if layout in ("chain", "spread", "full") else int(rng.integers(1, 4))
        rows["row_type"][w, :start] = 3
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
            draw = rng.random()
            if draw < 0.65 or layout in ("chain", "spread", "full"):
                slot = int(rng.integers(0, 2))
                rows["sparse_row_dof"][w, row, slot] = coordinate
                rows["sparse_row_dof"][w, row, 1 - slot] = -2
            elif draw < 0.85:
                rows["sparse_row_dof"][w, row] = (coordinate, int(rng.integers(-1, dofs)))
            for tangent in range(row + 1, row + count):
                rows["sparse_row_dof"][w, tangent, 0] = coordinate
            row += count
        end = start - 1 if layout == "inverted" else row
        if layout == "broken":
            owners = [
                h
                for h in range(start, end - 3)
                if rows["row_type"][w, h] != _FRICTION and rows["row_type"][w, h + 1] == _FRICTION
            ]
            flaw = int(rng.integers(0, 5)) if owners else 0
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


def _launch_unit_builder(rows, device, *, serial_contacts):
    """Run the unit builder over the rows; return every output."""
    worlds, capacity = rows["row_type"].shape
    dofs = rows["group_heads"].shape[1]
    kernel = _get_build_independent_sparse_contact_groups_kernel(
        capacity, dofs, str(device.arch), build_serial_contacts=serial_contacts, units=True
    )
    arrays = {name: wp.array(rows[name], dtype=wp.int32, device=device) for name in rows}
    wp.launch(
        kernel,
        dim=worlds * 32,
        inputs=[arrays["row_count"], arrays["bounds"], arrays["row_type"], arrays["row_parent"]],
        outputs=[arrays[name] for name in _OUTPUTS],
        device=device,
    )
    return {name: arrays[name].numpy() for name in _OUTPUTS}


def _assert_outputs_equal(actual, expected, label):
    for name in _OUTPUTS:
        np.testing.assert_array_equal(actual[name], expected[name], err_msg=f"{label}: {name}")


class TestFeatherPGSContactSchedule(unittest.TestCase):
    def test_serial_unit_walks_match_the_unit_oracle(self):
        """Reproduce the unit schedule oracle with the serial walks the builder must equal."""
        device = wp.get_device("cpu")
        for (dense, seed, capacity), rows in _patch_unit_cases():
            with self.subTest(dense=dense, seed=seed, capacity=capacity):
                marked = _mark_patch_units(rows, device)
                expected = _unit_schedule_oracle(rows, marked)
                worlds, dofs = rows["worlds"], rows["dofs"]
                inputs = {
                    "row_count": rows["row_count"],
                    "bounds": rows["bounds"],
                    "row_type": rows["row_type"],
                    "row_parent": rows["row_parent"],
                    "sparse_row_dof": marked,
                    "group_count": np.zeros(worlds, np.int32),
                    "group_heads": np.full((worlds, dofs), _UNWRITTEN, np.int32),
                    "serial_count": np.zeros(worlds, np.int32),
                    "serial": np.full((worlds, capacity), _UNWRITTEN, np.int32),
                }
                walks = _serial_unit_walks(inputs, serial_contacts=True)
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

    def test_unit_worlds_cover_both_builder_paths(self):
        """Draw unit layouts the builder takes in parallel and layouts it leaves to the serial walks."""
        rows = _unit_worlds(0, worlds=96, capacity=97, dofs=33)
        forms = [_rows_form_units(rows, world) for world in range(96)]
        walks = _serial_unit_walks(rows, serial_contacts=True)
        self.assertGreater(sum(forms), 48)
        self.assertGreater(forms.count(False), 4)
        # Chains cross chunks of 32 rows, and both schedules are populated.
        self.assertGreater(int(np.max(rows["row_count"] - rows["bounds"][:, 1])), 64)
        self.assertGreater(int(np.sum(walks["group_count"])), 96)
        self.assertGreater(int(np.sum(np.maximum(walks["serial_count"], 0))), 96)

    @unittest.skipUnless(wp.is_cuda_available(), "the contact schedule builders require CUDA")
    def test_unit_builder_matches_the_serial_walks(self):
        """Build unit schedules bitwise equal to the serial walks, in every layout and at capacity edges."""
        device = wp.get_device("cuda:0")
        for seed, (capacity, dofs) in enumerate(((64, 10), (97, 33), (40, 1), (130, 114))):
            rows = _unit_worlds(seed, worlds=180, capacity=capacity, dofs=dofs)
            for serial_contacts in (False, True):
                with self.subTest(capacity=capacity, dofs=dofs, serial_contacts=serial_contacts):
                    expected = _serial_unit_walks(rows, serial_contacts=serial_contacts)
                    actual = _launch_unit_builder(rows, device, serial_contacts=serial_contacts)
                    _assert_outputs_equal(actual, expected, f"capacity {capacity}")

    @unittest.skipUnless(wp.is_cuda_available(), "the contact schedule builders require CUDA")
    def test_unit_builder_replays_in_graphs(self):
        """Rebuild the same unit schedules from captured graphs as from eager launches."""
        device = wp.get_device("cuda:0")
        rows = _unit_worlds(5, worlds=90, capacity=97, dofs=33)
        expected = _serial_unit_walks(rows, serial_contacts=True)
        worlds, capacity = rows["row_type"].shape
        kernel = _get_build_independent_sparse_contact_groups_kernel(
            capacity, 33, str(device.arch), build_serial_contacts=True, units=True
        )
        arrays = {name: wp.array(rows[name], dtype=wp.int32, device=device) for name in rows}
        marked = wp.array(rows["sparse_row_dof"], dtype=wp.int32, device=device)

        def build():
            wp.copy(arrays["sparse_row_dof"], marked)
            wp.launch(
                kernel,
                dim=worlds * 32,
                inputs=[arrays["row_count"], arrays["bounds"], arrays["row_type"], arrays["row_parent"]],
                outputs=[arrays[name] for name in _OUTPUTS],
                device=device,
            )

        build()
        with wp.ScopedCapture(device=device) as capture:
            build()
        for replay in range(2):
            for name in ("group_count", "group_heads", "serial_count", "serial"):
                arrays[name].fill_(_UNWRITTEN)
            wp.capture_launch(capture.graph)
            actual = {name: arrays[name].numpy() for name in _OUTPUTS}
            _assert_outputs_equal(actual, expected, f"replay {replay}")

    @unittest.skipUnless(wp.is_cuda_available(), "the sparse-diagonal contact owner requires CUDA")
    def test_unit_builder_matches_the_serial_walks_on_solver_rows(self):
        """Schedule the patch units the solver produces, through both producers, as the serial walks do.

        Every builder launch of six solver steps is recorded: its rows and marks before the launch and
        its outputs after. The rows form units, so the builder takes its parallel path, and both the
        independent and the serial schedule are populated.
        """
        device = wp.get_device("cuda:0")
        model = build_solver_fixture(device)
        model.rigid_contact_max = 24
        launch = wp.launch
        for compact in (False, True):
            with (
                self.subTest(compact=compact),
                mock.patch.object(solver_module, "_COMPACT_CONTACT_BOUNDARY", compact),
                mock.patch.object(solver_module, "_SPARSE_CONTACT_DIRECT", True),
            ):
                solver = newton.solvers.SolverFeatherPGS(
                    model,
                    pgs_mode="matrix_free",
                    pgs_iterations=8,
                    use_parallel_streams=True,
                    enable_joint_limits=True,
                    dense_max_constraints=64,
                    mf_max_constraints=64,
                )
                self.assertTrue(solver._sparse_diagonal_contact_units)
                self.assertTrue(solver._sparse_diagonal_speculative_contact_batches)
                self.assertEqual(solver._compact_contact_boundary, compact)
                records = []

                def recording_launch(kernel, *args, records=records, solver=solver, **kwargs):
                    if kernel is not solver._build_independent_sparse_contact_groups_kernel:
                        return launch(kernel, *args, **kwargs)
                    names = ("row_count", "bounds", "row_type", "row_parent", *_OUTPUTS)
                    arrays = dict(zip(names, (*kwargs["inputs"], *kwargs["outputs"]), strict=True))
                    before = {name: array.numpy().copy() for name, array in arrays.items()}
                    result = launch(kernel, *args, **kwargs)
                    records.append((before, {name: arrays[name].numpy().copy() for name in _OUTPUTS}))
                    return result

                states = [model.state(), model.state()]
                control = model.control()
                contacts = newton.Contacts(model.rigid_contact_max, 0, device=device)
                newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
                install_patch_contacts(model, states[0], contacts)
                with mock.patch.object(wp, "launch", recording_launch):
                    for _ in range(6):
                        solver.step(states[0], states[1], control, contacts, 1 / 240)
                        states.reverse()
                self.assertEqual(len(records), 6)
                independent = serial = 0
                for step, (before, after) in enumerate(records):
                    for world in range(before["row_count"].shape[0]):
                        self.assertTrue(_rows_form_units(before, world), f"step {step} world {world}")
                    expected = _serial_unit_walks(before, serial_contacts=True)
                    _assert_outputs_equal(after, expected, f"step {step}")
                    independent += int(np.sum(after["group_count"]))
                    serial += int(np.sum(after["serial_count"]))
                self.assertGreater(independent, 0)
                self.assertGreater(serial, 0)


if __name__ == "__main__":
    unittest.main()
