# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The level builder (:mod:`.sweep_levels`): its levels are the row order's trace, and its work lists.

A world's units hold resources (the bodies, coordinates or trees their rows touch), and two units conflict when they
share one. The tests check, on the CPU:

- that a Gauss-Seidel sweep run by the builder's levels (each level's units in any order), by its chains (case K2: a
  lane a chain, the chains side by side, then the rest in row order) or by its components (case K5: a world's two
  trees apart) gives the row order's iterate bit for bit, and that an order putting one conflicting pair out of row
  order does not;
- that the level pass's rounds (:func:`.sweep_levels.level_pass`: a batch of 32 units, a unit committing once it owns
  both its resources) give the row-order levels (:func:`levels`, the rule the colored units' CUDA test compares the
  device with);
- that the Jacobi tiers' world lists bucket exactly each tier's worlds (:func:`.sweep_levels.list_worlds`, one
  :class:`~newton._src.core.execution.Queue`, a bucket a tier).
"""

import itertools
import unittest

import numpy as np
import warp as wp

from newton._src.core.execution import Queue
from newton._src.solvers.feather_pgs import sweep_levels


def _devices():
    return ("cpu", *(("cuda:0",) if wp.is_cuda_available() else ()))


def levels(resources) -> np.ndarray:
    """The level builder's rule on the host: each unit's level, one past the latest level of an earlier unit that shares
    one of its ``resources`` (each unit's tuple of resources, in row order), 0 for none."""
    last, out = {}, []
    for held in resources:
        level = max((last[r] + 1 for r in held if r in last), default=0)
        for r in held:
            last[r] = level
        out.append(level)
    return np.asarray(out, dtype=np.int64)


def rounds(resources) -> np.ndarray:
    """The level pass's rounds (:func:`.sweep_levels.level_pass`) on the host: a batch of 32 units at a time, each round
    every pending unit claims its resources (the lowest lane wins), and a unit that owns all of them commits its
    level from the resources' latest levels."""
    latest, out = {}, [-1] * len(resources)
    for base in range(0, len(resources), 32):
        pending = list(range(base, min(base + 32, len(resources))))
        while pending:
            owner = {}
            for unit in pending:
                for r in resources[unit]:
                    owner.setdefault(r, unit)
            ready = [unit for unit in pending if all(owner[r] == unit for r in resources[unit])]
            for unit in ready:
                out[unit] = max((latest.get(r, -1) + 1 for r in resources[unit]), default=0)
                for r in resources[unit]:
                    latest[r] = out[unit]
            pending = [unit for unit in pending if unit not in ready]
    return np.asarray(out, dtype=np.int64)


class World:
    """A world of free bodies (``dofs`` coordinates each) whose units are scalar unilateral rows or contacts (a normal
    and a friction pair on the box of ``mu`` times its impulse), each on one or two bodies; float32 throughout."""

    def __init__(self, rng, resources, dofs: int = 3):
        self.resources, self.dofs = resources, dofs
        bodies = 1 + max(r for held in resources for r in held)
        self.inverse_mass = rng.uniform(0.5, 2.0, bodies * dofs).astype(np.float32)
        self.v0 = rng.normal(0.0, 1.0, bodies * dofs).astype(np.float32)
        self.rows = []  # per unit: (coordinates, rows' entries, bias, inverse diagonals, mu)
        for held in resources:
            n = 3 if rng.random() < 0.6 else 1
            coordinates = np.concatenate([np.arange(r * dofs, (r + 1) * dofs) for r in held])
            entries = rng.normal(0.0, 1.0, (n, coordinates.size)).astype(np.float32)
            weights = self.inverse_mass[coordinates]
            diagonal = (entries * entries * weights).sum(axis=1, dtype=np.float32) + np.float32(1.0e-3)
            bias = rng.normal(-0.2, 0.5, n).astype(np.float32)
            self.rows.append((coordinates, entries, bias, (np.float32(1.0) / diagonal).astype(np.float32), 0.6))

    def sweep(self, order, sweeps: int = 4):
        """Gauss-Seidel over the units in ``order``, ``sweeps`` times: the velocities and impulses."""
        v = self.v0.copy()
        impulses = [np.zeros(len(rows[1]), dtype=np.float32) for rows in self.rows]
        for _ in range(sweeps):
            for unit in order:
                coordinates, entries, bias, inverse, mu = self.rows[unit]
                lam = impulses[unit]
                for row in range(len(entries)):
                    residual = np.float32(bias[row])
                    for k, c in enumerate(coordinates):
                        residual = np.float32(residual + entries[row, k] * v[c])
                    target = np.float32(lam[row] - residual * inverse[row])
                    if row == 0:
                        target = max(target, np.float32(0.0))
                    else:
                        bound = np.float32(mu * lam[0])
                        target = min(max(target, -bound), bound)
                    change = np.float32(target - lam[row])
                    lam[row] = target
                    for k, c in enumerate(coordinates):
                        v[c] = np.float32(v[c] + self.inverse_mass[c] * entries[row, k] * change)
        return v, np.concatenate(impulses)


def _trace(rng, units: int, bodies: int, pair: float):
    """Units of one body, or two (with probability ``pair``), drawn over ``bodies`` bodies."""
    out = []
    for _ in range(units):
        if rng.random() < pair:
            a, b = rng.choice(bodies, 2, replace=False)
            out.append((int(a), int(b)))
        else:
            out.append((int(rng.integers(bodies)),))
    return out


def _same(test, a, b):
    for x, y in zip(a, b, strict=True):
        test.assertTrue(np.array_equal(x.view(np.int32), y.view(np.int32)))


class TestLevelsAreTheRowOrder(unittest.TestCase):
    """A sweep run by the builder's schedules is the row order's sweep, bit for bit."""

    def test_the_rounds_give_the_row_order_levels(self):
        """The level pass's rounds give each unit the row-order rule's level, a batch boundary or not."""
        rng = np.random.default_rng(1)
        for units, bodies, pair in itertools.product((5, 31, 32, 33, 97), (3, 8, 40), (0.0, 0.5, 1.0)):
            resources = _trace(rng, units, bodies, pair)
            np.testing.assert_array_equal(rounds(resources), levels(resources))

    def test_a_level_s_units_share_no_resource(self):
        rng = np.random.default_rng(2)
        resources = _trace(rng, 200, 25, 0.5)
        level = levels(resources)
        for value in range(int(level.max()) + 1):
            held = [r for unit in np.flatnonzero(level == value) for r in resources[unit]]
            self.assertEqual(len(held), len(set(held)))

    def test_levels_sweep_the_row_order(self):
        """Levels in order, each level's units reversed or shuffled: the row order's velocities and impulses."""
        rng = np.random.default_rng(3)
        for units, bodies, pair in ((40, 12, 0.5), (60, 6, 0.8), (33, 30, 0.3)):
            world = World(rng, _trace(rng, units, bodies, pair))
            level = levels(world.resources)
            reference = world.sweep(range(units))
            for shuffle in (False, True):
                order = []
                for value in range(int(level.max()) + 1):
                    members = np.flatnonzero(level == value)[::-1]
                    order.extend(rng.permutation(members) if shuffle else members)
                _same(self, world.sweep(order), reference)

    def test_an_order_out_of_the_trace_differs(self):
        """Swapping one conflicting pair of consecutive units changes the iterate: the test sees order."""
        rng = np.random.default_rng(4)
        world = World(rng, _trace(rng, 40, 6, 0.6))
        reference = world.sweep(range(40))
        unit = next(u for u in range(1, 40) if set(world.resources[u]) & set(world.resources[u - 1]))
        order = list(range(40))
        order[unit - 1], order[unit] = order[unit], order[unit - 1]
        v, impulses = world.sweep(order)
        self.assertFalse(np.array_equal(v, reference[0]) and np.array_equal(impulses, reference[1]))

    def test_chains_sweep_the_row_order(self):
        """Case K2: one-resource units on a resource no two-resource unit holds form chains; a lane a chain, the
        chains side by side, then every other unit in row order."""
        rng = np.random.default_rng(5)
        for pair in (0.2, 0.5):
            # Coupled units on the first 8 bodies (a tree's coordinates), scalar ones on any of 20.
            resources = [held if len(held) > 1 else (int(rng.integers(20)),) for held in _trace(rng, 80, 8, pair)]
            world = World(rng, resources)
            coupled = {r for held in world.resources if len(held) > 1 for r in held}
            chains = {}
            for unit, held in enumerate(world.resources):
                if len(held) == 1 and held[0] not in coupled:
                    chains.setdefault(held[0], []).append(unit)
            self.assertGreater(len(chains), 1)
            # Lanes advance together: the chains' first units, then their second units, and so on.
            lanes = list(chains.values())
            order = [chain[k] for k in range(max(map(len, lanes))) for chain in lanes if k < len(chain)]
            chained = set(order)
            order += [unit for unit in range(len(world.resources)) if unit not in chained]
            _same(self, world.sweep(order), world.sweep(range(len(world.resources))))

    def test_components_sweep_the_row_order(self):
        """Case K5: a world whose rows each hold its articulation or its free body splits into the two."""
        rng = np.random.default_rng(6)
        resources = [(int(rng.integers(2)),) for _ in range(50)]
        world = World(rng, resources, dofs=6)
        body = [unit for unit, held in enumerate(resources) if held == (1,)]
        tree = [unit for unit, held in enumerate(resources) if held == (0,)]
        _same(self, world.sweep(body + tree), world.sweep(range(50)))


class TestWorldLists(unittest.TestCase):
    def test_each_tier_buckets_exactly_its_worlds(self):
        """A world of dense rows only and at most the bound lands in the bucket of its 32-row tier, once; others in none."""
        rng = np.random.default_rng(7)
        worlds, bound, width = 500, 100, 32
        rows = rng.integers(0, 140, worlds).astype(np.int32)
        free = (rng.random(worlds) < 0.2).astype(np.int32) * rng.integers(1, 9, worlds).astype(np.int32)
        tiers = -(-bound // width)
        for device in _devices():
            with self.subTest(device=device):
                lists = Queue(tiers * worlds, device, buckets=tiers)
                wp.launch(
                    sweep_levels.list_worlds,
                    dim=worlds,
                    inputs=[
                        wp.array(rows, dtype=int, device=device),
                        wp.array(free, dtype=int, device=device),
                        bound,
                        width,
                        lists.region,
                    ],
                    outputs=[lists.items, lists.count],
                    device=device,
                )
                for tier in range(tiers):
                    count, items = (array.numpy() for array in lists.bucket(tier))
                    lo, hi = tier * width, min((tier + 1) * width, bound)
                    want = np.flatnonzero((rows > lo) & (rows <= hi) & (free == 0))
                    self.assertEqual(int(count[0]), want.size)
                    np.testing.assert_array_equal(np.sort(items[: count[0]]), want)

    def test_a_bucket_is_a_list_of_its_own(self):
        """A bucket's views are its counter and its region of the slots."""
        lists = Queue(12, "cpu", buckets=3)
        self.assertEqual(lists.region, 4)
        count, items = lists.bucket(2)
        self.assertEqual((count.shape, items.shape), ((1,), (4,)))
        self.assertEqual(items.ptr, lists.items.ptr + 8 * 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
