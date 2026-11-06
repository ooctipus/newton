# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The level builder (:mod:`.sweep_levels`): its levels are the row order's trace, and its work lists.

A world's units hold resources (the bodies, coordinates or trees their rows touch), and two units conflict when they
share one. The tests check, on the CPU:

- that a Gauss-Seidel sweep run by the builder's levels (each level's units in any order), by its scalar contacts'
  levels (case K2: the levels of the one-coordinate contacts on coordinates no coupled contact holds, each level's
  contacts side by side, then the rest in row order) or by its components (case K5: a world's two trees apart) gives
  the row order's iterate bit for bit, and that an order putting one conflicting pair out of row order does not;
- that the level pass's rounds (:func:`.sweep_levels.level_pass`: a batch of 32 units, a unit committing once it owns
  both its resources) give the row-order levels (:func:`levels`, the rule the colored units' CUDA test compares the
  device with);
- that the claims write every world with rows into exactly its claim's bucket (:func:`.sweep_levels.claim_worlds`,
  one :class:`~newton._src.core.execution.Queue`, a bucket a Jacobi tier, then the general sweep's).
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

    def test_scalar_contact_levels_sweep_the_row_order(self):
        """Case K2: one-resource units on a resource no two-resource unit holds are leveled over their resources (a
        resource's chain, its ``k``-th unit level ``k``); each level's units side by side, in any order, then every
        other unit in row order."""
        rng = np.random.default_rng(5)
        for pair in (0.2, 0.5):
            # Coupled units on the first 8 bodies (a tree's coordinates), scalar ones on any of 20.
            resources = [held if len(held) > 1 else (int(rng.integers(20)),) for held in _trace(rng, 80, 8, pair)]
            world = World(rng, resources)
            coupled = {r for held in world.resources if len(held) > 1 for r in held}
            scalar = [unit for unit, held in enumerate(world.resources) if len(held) == 1 and held[0] not in coupled]
            level = levels([world.resources[unit] for unit in scalar])
            chains = {}
            for unit in scalar:
                chains.setdefault(world.resources[unit][0], []).append(unit)
            self.assertGreater(len(chains), 1)
            self.assertGreater(int(level.max()), 0)
            # A unit's level is its place in its resource's chain.
            for unit, value in zip(scalar, level, strict=True):
                self.assertEqual(chains[world.resources[unit][0]].index(unit), value)
            reference = world.sweep(range(len(world.resources)))
            for shuffle in (False, True):
                order = []
                for value in range(int(level.max()) + 1):
                    members = [unit for unit, held in zip(scalar, level, strict=True) if held == value][::-1]
                    order.extend(rng.permutation(members) if shuffle else members)
                leveled = set(order)
                order += [unit for unit in range(len(world.resources)) if unit not in leveled]
                _same(self, world.sweep(order), reference)

    def test_components_sweep_the_row_order(self):
        """Case K5: a world whose rows each hold its articulation or its free body splits into the two."""
        rng = np.random.default_rng(6)
        resources = [(int(rng.integers(2)),) for _ in range(50)]
        world = World(rng, resources, dofs=6)
        body = [unit for unit, held in enumerate(resources) if held == (1,)]
        tree = [unit for unit, held in enumerate(resources) if held == (0,)]
        _same(self, world.sweep(body + tree), world.sweep(range(50)))


class TestWorldLists(unittest.TestCase):
    def test_each_world_lands_in_its_claim_s_bucket(self):
        """A world of dense rows only (no speed row) and at most the bound lands in the bucket of its 32-row tier, once;
        any other world with rows in the colored candidates' bucket where it has a slot (and the claims have one), else
        in the general bucket; a world without rows in none."""
        rng = np.random.default_rng(7)
        worlds, bound, width = 500, 100, 32
        rows = rng.integers(0, 140, worlds).astype(np.int32)
        free = (rng.random(worlds) < 0.2).astype(np.int32) * rng.integers(1, 9, worlds).astype(np.int32)
        # Where the line's speed rows start: a world with any (rows past it) is the general sweep's too.
        speed_first = rows - (rng.random(worlds) < 0.2).astype(np.int32) * rng.integers(0, 3, worlds).astype(np.int32)
        slot = np.where(rng.random(worlds) < 0.3, np.arange(worlds), -1).astype(np.int32)
        tiers = -(-bound // width)
        for device, candidates in itertools.product(_devices(), (False, True)):
            with self.subTest(device=device, candidates=candidates):
                buckets = tiers + 1 + int(candidates)
                claims = Queue(buckets * worlds, device, buckets=buckets)
                wp.launch(
                    sweep_levels.claim_worlds,
                    dim=worlds,
                    inputs=[
                        wp.array(rows, dtype=int, device=device),
                        wp.array(free, dtype=int, device=device),
                        wp.array(speed_first, dtype=int, device=device),
                        wp.array(slot, dtype=wp.int32, device=device),
                        bound,
                        width,
                        tiers,
                        tiers + 1 if candidates else -1,
                        claims.region,
                    ],
                    outputs=[claims.items, claims.count],
                    device=device,
                )
                tiered = (rows > 0) & (rows <= bound) & (free == 0) & (rows <= speed_first)
                rest = ~tiered & ((rows > 0) | (free > 0))
                slotted = (slot >= 0) & candidates
                for bucket in range(buckets):
                    count, items = (array.numpy() for array in claims.bucket(bucket))
                    if bucket < tiers:
                        lo, hi = bucket * width, min((bucket + 1) * width, bound)
                        want = np.flatnonzero((rows > lo) & (rows <= hi) & tiered)
                    elif bucket == tiers:
                        want = np.flatnonzero(rest & ~slotted)
                    else:
                        want = np.flatnonzero(rest & slotted)
                    self.assertEqual(int(count[0]), want.size)
                    np.testing.assert_array_equal(np.sort(items[: count[0]]), want)

    def test_appends_past_a_region_are_dropped(self):
        """A bucket keeps its first ``region`` appends, each world once, and its counter the demand (the reservation's
        overflow contract), whichever warps share a bucket's atomic."""
        worlds, width, region = 1000, 32, 100
        rows = (np.arange(worlds) % 3 * width + 1).astype(np.int32)
        for device in _devices():
            with self.subTest(device=device):
                lists = Queue(4 * region, device, buckets=4)
                wp.launch(
                    sweep_levels.claim_worlds,
                    dim=worlds,
                    inputs=[
                        wp.array(rows, dtype=int, device=device),
                        wp.zeros(worlds, dtype=int, device=device),
                        wp.array(rows, dtype=int, device=device),
                        wp.zeros(1, dtype=wp.int32, device=device),
                        3 * width,
                        width,
                        3,
                        -1,
                        lists.region,
                    ],
                    outputs=[lists.items, lists.count],
                    device=device,
                )
                for tier in range(3):
                    count, items = (array.numpy() for array in lists.bucket(tier))
                    want = np.flatnonzero(rows == tier * width + 1)
                    self.assertEqual(int(count[0]), want.size)
                    self.assertEqual(np.unique(items).size, region)
                    self.assertTrue(np.isin(items, want).all())

    def test_a_bucket_is_a_list_of_its_own(self):
        """A bucket's views are its counter and its region of the slots."""
        lists = Queue(12, "cpu", buckets=3)
        self.assertEqual(lists.region, 4)
        count, items = lists.bucket(2)
        self.assertEqual((count.shape, items.shape), ((1,), (4,)))
        self.assertEqual(items.ptr, lists.items.ptr + 8 * 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
