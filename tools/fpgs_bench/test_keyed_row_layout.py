# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Topology-driven physical row input generation, independent of solve policy."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import keyed_rows as rows
from newton._src.solvers.feather_pgs.keyed_row_layout import KeyedLayout, prefix_kernel


class TestKeyedRowLayout(unittest.TestCase):
    def test_single_floating_tree_has_a_generated_reader(self):
        """Generate the walked row reader for one floating tree."""
        layout = KeyedLayout(18, 0, 0, 0, 48, False, 0)
        inputs = rows.get_direct_parallel_inputs(layout, 18, 0, 0, 0, 18, 72)
        self.assertIn("TIER_WALK(", inputs.read)

    def test_reversed_pair_offsets_are_valid(self):
        """Accept independent blocks in reversed world order."""
        layout = KeyedLayout(9, 6, 6, 0, 32, True, (1 << 15) - 1)
        inputs = rows.get_direct_parallel_inputs(layout, 9, 6, 6, 0, 15, 64)
        self.assertIn("TIER_WALK(", inputs.read)

    def test_reject_overlapping_layout(self):
        """Reject overlapping coordinate blocks."""
        with self.assertRaises(ValueError):
            KeyedLayout(9, 6, 0, 3, 32, False, 0)

    def test_prefix_laws_order_and_capacity(self):
        """Preserve unary laws, row order and bounded storage."""
        for na, nb, oa, ob, velocity, position_mask, velocity_mask in [
            (18, 0, 0, 0, False, (1 << 18) - 1, 0),
            (18, 0, 0, 0, False, ((1 << 12) - 1) << 6, 0),
            (9, 6, 6, 0, True, (1 << 15) - 1, (1 << 15) - 1),
            (3, 0, 0, 0, True, 7, 7),
            (9, 6, 6, 0, True, 0b101010101010100, 0b110011001100000),
            (18, 0, 0, 0, False, 0, 0),
        ]:
            with self.subTest(layout=(na, nb, oa, ob, velocity)):
                d, worlds, capacity = na + nb, 3, 17
                layout = KeyedLayout(na, nb, oa, ob, 16, velocity, velocity_mask, position_mask=position_mask)
                p, z = rows.PrefixInput(), rows.KineticRows()
                starts = np.array([[w * d, w * d + na] for w in range(worlds)], np.int32)
                q = np.linspace(-0.4, 0.4, worlds * d, dtype=np.float32)
                v = np.linspace(-3.0, 3.0, worlds * d, dtype=np.float32)
                p.starts = wp.array(starts, dtype=int, device="cpu")
                p.limit_q_index = wp.array(np.arange(worlds * d, dtype=np.int32), device="cpu")
                p.q, p.vhat = wp.array(q, device="cpu"), wp.array(v, device="cpu")
                p.lower = wp.full(worlds * d, -0.2, dtype=float, device="cpu")
                p.upper = wp.full(worlds * d, 0.2, dtype=float, device="cpu")
                p.velocity_limit = wp.full(worlds * d, 2.0, dtype=float, device="cpu")
                p.velocity_slots = wp.full(worlds * d * 2, -9, dtype=int, device="cpu")
                p.velocity_signs = wp.full(worlds * d * 2, -9.0, dtype=float, device="cpu")
                p.slot_counter = wp.empty(worlds, dtype=int, device="cpu")
                p.bounds = wp.empty((worlds, 2), dtype=int, device="cpu")
                for name in ("row_type", "row_parent"):
                    setattr(p, name, wp.full((worlds, capacity), -99, dtype=int, device="cpu"))
                for name in ("row_mu", "row_beta", "row_cfm", "phi", "target", "restitution", "rhs"):
                    setattr(p, name, wp.full((worlds, capacity), -99.0, dtype=float, device="cpu"))
                z.rowkeys = wp.full((worlds, capacity), 99, dtype=int, device="cpu")
                p.activation_gap, p.activation, p.beta, p.cfm, p.dt = 0.1, 0.7, 0.05, 1e-5, 0.01
                wp.launch_tiled(
                    prefix_kernel(layout, rows.PrefixInput, rows.KineticRows),
                    dim=[worlds],
                    inputs=[p, z],
                    block_dim=32,
                    device="cpu",
                )
                for world in range(worlds):
                    expected = []
                    for phase in (0, 1):
                        for coordinate in range(d):
                            primary = oa <= coordinate < oa + na
                            dof = starts[world, 0 if primary else 1] + coordinate - (oa if primary else ob)
                            for side in (0, 1):
                                sign = 1.0 if side == 0 else -1.0
                                phi = sign * q[dof] + 0.2 if phase == 0 else 0.0
                                active = (
                                    (bool(position_mask & (1 << coordinate)) and phi <= 0.1)
                                    if phase == 0
                                    else (velocity and bool(velocity_mask & (1 << coordinate)) and abs(v[dof]) >= 1.4)
                                )
                                if active:
                                    rhs = (0.05 if phi < 0 else 1.0) * phi / 0.01 if phase == 0 else 2.0
                                    expected.append((~(2 * coordinate + side), 3 + phase, phi, rhs))
                        self.assertEqual(p.bounds.numpy()[world, phase], len(expected))
                    self.assertEqual(p.slot_counter.numpy()[world], len(expected))
                    if velocity:
                        slots, signs = p.velocity_slots.numpy(), p.velocity_signs.numpy()
                        for coordinate in range(d):
                            primary = oa <= coordinate < oa + na
                            dof = starts[world, 0 if primary else 1] + coordinate - (oa if primary else ob)
                            for side in (0, 1):
                                matching = [
                                    i
                                    for i, row in enumerate(expected[:capacity])
                                    if row[0] == ~(2 * coordinate + side) and row[1] == 4
                                ]
                                slot = matching[0] if matching else -1
                                self.assertEqual(slots[dof * 2 + side], slot)
                                self.assertEqual(
                                    signs[dof * 2 + side], (1.0 if side == 0 else -1.0) if slot >= 0 else 0.0
                                )
                    for row, (key, kind, phi, rhs) in enumerate(expected[:capacity]):
                        self.assertEqual(z.rowkeys.numpy()[world, row], key)
                        self.assertEqual(p.row_type.numpy()[world, row], kind)
                        self.assertAlmostEqual(p.phi.numpy()[world, row], phi, places=6)
                        self.assertAlmostEqual(p.rhs.numpy()[world, row], rhs, places=5)


if __name__ == "__main__":
    unittest.main()
