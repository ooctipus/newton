# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the slot-order independent patch-friction comparisons on CPU."""

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    pgs_solve_loop,
)
from newton.tests.feather_pgs_patch_parity import (
    assert_history_close,
    assert_ring_coulomb,
    assert_rows_close,
    contact_rows,
)
from newton.tests.test_feather_pgs_friction_patches import _ground_box, _patch_fixture

_MF_ROWS = ("mf_row_type", "mf_row_parent", "mf_row_mu", "mf_rhs", "mf_impulses", "mf_phi", "mf_J_a", "mf_J_b")


def _permuted_slots(solver, contacts, shift):
    """Reserve the same matrix-free row blocks in rotated order, as other atomic outcomes would."""
    count = int(contacts.rigid_contact_count.numpy()[0])
    slot, path = solver.contact_slot.numpy().copy(), solver.contact_path.numpy()
    world, needed = solver.contact_world.numpy(), solver.contact_slots_needed.numpy()
    routed = [c for c in range(count) if path[c] == 1 and slot[c] >= 0 and needed[c] > 0]
    rows = {name: getattr(solver, name).numpy() for name in _MF_ROWS}
    permuted = {name: values.copy() for name, values in rows.items()}
    moved = {}
    for w in np.unique(world[routed]):
        blocks = [c for c in routed if world[c] == w]
        # This fixture's matrix-free rows are contacts only, so the blocks tile the world's rows.
        assert sorted(int(slot[c]) + k for c in blocks for k in range(int(needed[c]))) == list(
            range(int(solver.mf_constraint_count.numpy()[w]))
        )
        start = 0
        for c in blocks[shift:] + blocks[:shift]:
            for k in range(int(needed[c])):
                moved[(w, int(slot[c]) + k)] = start + k
            slot[c] = start
            start += int(needed[c])
    for (w, old), new in moved.items():
        for name in _MF_ROWS:
            permuted[name][w, new] = rows[name][w, old]
        parent = int(rows["mf_row_parent"][w, old])
        permuted["mf_row_parent"][w, new] = moved.get((w, parent), parent)
    stand_in = SimpleNamespace(
        **{name: wp.array(values, device="cpu") for name, values in permuted.items()},
        mf_constraint_count=solver.mf_constraint_count,
        contact_slot=wp.array(slot, device="cpu"),
        contact_path=solver.contact_path,
        contact_world=solver.contact_world,
        contact_slots_needed=solver.contact_slots_needed,
        _friction_patches=solver._friction_patches,
        _sparse_mass_matrix_size=None,
        pgs_mode=solver.pgs_mode,
    )
    return stand_in


class TestFeatherPGSPatchParity(unittest.TestCase):
    def test_contact_rows_ignore_slot_order(self):
        """Key rows and patch rings by contact, so other slot reservations compare equal."""
        with wp.ScopedDevice("cpu"):
            _model, solver, contacts, step = _ground_box("cpu")
            for _ in range(20):
                step()
            rows = contact_rows(solver, contacts, route=1, jacobian=True)
        normals = [c for c, record in rows.items() if record["types"][0] == PGS_CONSTRAINT_TYPE_CONTACT]
        anchors = [c for c, record in rows.items() if len(record["types"]) == 3]
        # A resting box face: four ring-linked normals, friction rows only on its two anchors.
        self.assertEqual(len(normals), 4)
        self.assertEqual(len(anchors), 2)
        self.assertTrue(all(rows[c]["parents"][0] not in (-1, c) for c in normals))
        for c in anchors:
            np.testing.assert_array_equal(rows[c]["types"][1:], [PGS_CONSTRAINT_TYPE_FRICTION] * 2)
            np.testing.assert_array_equal(rows[c]["parents"][1:], [c, c])
            self.assertGreater(rows[c]["patch"][0], 0.0)
        self.assertEqual(assert_ring_coulomb(solver, route=1), 2)
        for shift in (1, 2, 3):
            with self.subTest(shift=shift):
                stand_in = _permuted_slots(solver, contacts, shift)
                self.assertFalse(np.array_equal(stand_in.contact_slot.numpy(), solver.contact_slot.numpy()))
                assert_rows_close(rows, contact_rows(stand_in, contacts, route=1, jacobian=True), atol=0.0, rtol=0.0)

    def test_ring_coulomb_accepts_the_pooled_budget(self):
        """Accept friction bounded by the ring's pooled normal load and reject an exceeded budget."""
        capacity = 32
        count = wp.array([7], dtype=int, device="cpu")
        rows = wp.array([[0, 2, 2, 0, 0, 2, 2] + [-1] * 25], dtype=int, device="cpu")
        parents = wp.array([[3, 0, 0, 4, 0, 4, 4] + [-1] * 25], dtype=int, device="cpu")
        mu = wp.array([[0.5, 0.25, 0.25, 0.5, 0.5, 0.25, 0.25] + [0] * 25], dtype=float, device="cpu")
        rhs = wp.array([[0, -100, 0, -4, 0, -100, 0] + [0] * 25], dtype=float, device="cpu")
        impulses = wp.zeros((1, capacity), dtype=float, device="cpu")
        diag = wp.ones((1, capacity), dtype=float, device="cpu")
        matrix = wp.array(np.eye(capacity, dtype=np.float32)[None], dtype=float, device="cpu")
        inputs = [count, capacity, diag, matrix, rhs, impulses, 2, 1.0, rows, parents, mu, 0, 0]
        wp.launch(pgs_solve_loop, dim=1, inputs=inputs, device="cpu")
        solved = impulses.numpy()
        np.testing.assert_allclose(solved[0, :7], [0, 1, 0, 4, 0, 1, 0], atol=1.0e-6)
        stand_in = SimpleNamespace(
            constraint_count=count,
            row_type=rows,
            row_parent=parents,
            row_mu=mu,
            rhs=rhs,
            impulses=impulses,
            phi=wp.zeros_like(rhs),
            target_velocity=wp.zeros_like(rhs),
        )
        # Each anchor's own normal carries no load: only the pooled ring budget admits the solution.
        self.assertGreater(solved[0, 1], solved[0, 0] * 0.25)
        self.assertEqual(assert_ring_coulomb(stand_in), 2)
        solved[0, 5] = 1.5
        impulses.assign(solved)
        with self.assertRaisesRegex(AssertionError, "pooled budget"):
            assert_ring_coulomb(stand_in)

    def test_history_comparison_detects_lifecycle_differences(self):
        """Match equal anchor histories and report a changed validity bit or carried displacement."""
        points = np.array([[x, y, 0.0] for x in (-0.05, 0.05) for y in (-0.05, 0.05)], dtype=np.float32)
        histories = []
        for _ in range(2):
            _model, state, _contacts, patches = _patch_fixture(points)
            patches.store(state)
            histories.append(patches)
        valid = np.flatnonzero(histories[0].previous.valid.numpy())
        self.assertGreater(valid.size, 0)
        assert_history_close(*histories)
        changed = histories[1].previous.valid.numpy()
        changed[valid[0]] = 0
        histories[1].previous.valid.assign(changed)
        with self.assertRaisesRegex(AssertionError, "previous.valid"):
            assert_history_close(*histories)
        changed[valid[0]] = 1
        histories[1].previous.valid.assign(changed)
        displacement = histories[1].previous.displacement.numpy()
        displacement[valid[0], 0] += 1.0e-3
        histories[1].previous.displacement.assign(displacement)
        with self.assertRaisesRegex(AssertionError, "previous.displacement"):
            assert_history_close(*histories)


if __name__ == "__main__":
    unittest.main()
