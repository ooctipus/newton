# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The colored mapping of FeatherPGS's matrix-free sweep (:mod:`.sweep_units`, ``GeneralSweep(levels=LevelLaw(EXACT))``).

A world's contacts become units (a normal row and its friction pair) and each unit's color is its level in row
order (one past the latest color of an earlier unit on its bodies), so no two units of a color share a body and
units that share one keep their row order. The unit builder gives the colored mapping the worlds its rule picks
(``sweep_units.COLORED_RULE``); the tests relax the rule to one unit so every world with contacts takes it. They
check

- the unit builder against a host reference of the same rules (units, row-order levels, the slot records), that a
  color never shares a body or a patch, that the colors keep the row order, that a world past its slot's share keeps
  its units in the pool, and that a world past the pool's end keeps the one-warp mapping and sets the capacity status;
- that a world's result repeats bitwise and that the warp's worlds and the colored worlds split the batch;
- the physics: the colored mapping is the reference's row-order Gauss-Seidel sweep, so it meets the conformance
  gate's laws and its distance to the converged reference on a tray carrying a grid of boxes (a hub body) and on
  the gate's box stack.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import itertools
import unittest
from unittest import mock

import numpy as np
import warp as wp

from newton._src.core.memory import Place
from newton._src.solvers.feather_pgs import sweep_plan, sweep_routes, sweep_units
from newton._src.solvers.feather_pgs.row_laws import ROW_CONTACT, ROW_FRICTION, ROW_TORSION
from newton._src.solvers.feather_pgs.row_metadata import HEADER_WORDS, RECORD_WORDS, record_layout, record_views
from newton._src.solvers.feather_pgs.row_records import line_size
from newton.solvers import SolverFeatherPGS
from newton.tests import feather_pgs_scenes as scenes
from newton.tests import test_feather_pgs_conformance as conf
from newton.tests.feather_pgs_law_harness import row_lines
from newton.tests.test_feather_pgs_sweep_levels import levels


def tray(*, mu: float = 0.6, grid: int = 3, layers: int = 2) -> scenes.Scene:
    """A 2 kg tray on the ground under ``grid x grid`` columns of 0.1 m, 0.2 kg boxes, ``layers`` high.

    The tray carries every column, so its units outnumber any box's several times over: it takes a color per unit.
    """

    def build(device):
        builder = scenes._builder()
        builder.add_ground_plane(cfg=scenes._cfg(mu=mu))
        body = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0255), wp.quat_identity()),
            mass=2.0,
            inertia=scenes._box_inertia(2.0, 0.25, 0.25, 0.025),
        )
        builder.add_shape_box(body, hx=0.25, hy=0.25, hz=0.025, cfg=scenes._cfg(mu=mu, density=0.0))
        for layer in range(layers):
            for i in range(grid):
                for j in range(grid):
                    position = ((i - 1) * 0.15, (j - 1) * 0.15, 0.1015 + 0.101 * layer)
                    scenes._free_box(builder, position, half=0.05, mass=0.2, mu=mu)
        return scenes._finalize(builder, device)

    return scenes.Scene("tray", build, dt=1.0 / 240.0, steps=60, params={"mu": [mu]}, passive=True)


@contextlib.contextmanager
def colored(units: int = 1, solvers: list | None = None, parallelism: float = 0.0):
    """Every solver built inside gives the colored mapping the worlds of at least ``units`` contact units and
    ``parallelism`` units per color."""
    original = SolverFeatherPGS.__init__

    @functools.wraps(original)  # the solver checks its options against its constructor's signature
    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        if solvers is not None:
            solvers.append(self)

    rule = sweep_units.ColoredRule(min_units=units, parallelism=parallelism)
    with mock.patch.object(sweep_units, "COLORED_RULE", rule), mock.patch.object(SolverFeatherPGS, "__init__", init):
        yield


_ROUTE = conf.REFERENCE_CUDA.but("mf_colored")
"""The gate's reference route (the plain matrix-free sweep owns every world), its free-body rows then colored."""


def _signed(value: int, bits: int) -> int:
    value &= (1 << bits) - 1
    return value - (1 << bits) if value >> (bits - 1) else value


def reference_units(meta, count: int, end: int) -> dict:
    """The unit builder's rules on the host for one world: units, row-order levels and the slot records by color."""
    m = min(count, meta.shape[0])
    end = min(end, m)
    packed = [int(word) for word in meta[:, 3]]
    kind = [word & 0xFFFF for word in packed]
    parent = [_signed(word >> 16, 16) for word in packed]
    units = []
    for row in range(end):
        if kind[row] != ROW_CONTACT:
            continue
        rows = 1
        # Its friction pair, then its spin row where it has one.
        while row + rows < end and rows < 4:
            r = row + rows
            want = ROW_FRICTION if rows < 3 else ROW_TORSION
            if kind[r] != want or parent[r] != row:
                break
            rows += 1
        word = int(meta[row, 0])
        units.append((row, rows, _signed(word >> 16, 16), _signed(word, 16)))
    # A unit's color is its level over its bodies (the level builder's rule).
    colors = levels([tuple(dof for dof in {a, b} if dof >= 0) for _, _, a, b in units])
    count = int(colors.max(initial=-1)) + 1
    sizes = np.bincount(np.asarray(colors, dtype=np.int64), minlength=count)
    order = sorted(range(len(units)), key=lambda unit: (colors[unit], unit))
    # Each unit's turn: the colors of the latest earlier unit on its first and its second body (-1 for none).
    last, after = {}, []
    for (_, _, a, b), color in zip(units, colors, strict=True):
        after.append((last.get(a, -1) if a >= 0 else -1, last.get(b, -1) if b >= 0 and b != a else -1))
        for dof in (dof for dof in {a, b} if dof >= 0):
            last[dof] = int(color)
    return {
        "unit_count": len(units),
        "unit_color": np.asarray(colors, dtype=np.int64),
        "color_count": count,
        "color_offset": np.concatenate([[0], np.cumsum(sizes)]),
        "color_slot": np.asarray([units[unit] for unit in order], dtype=np.int64).reshape(-1, 4),
        "color_after": np.asarray([after[unit] for unit in order], dtype=np.int64).reshape(-1, 2),
    }


def _cuda():
    return wp.get_device("cuda:0") if wp.is_cuda_available() else None


class TestCapableWorlds(unittest.TestCase):
    """The worlds that hold colored storage, from the model at construction."""

    def test_a_world_needs_the_rule_s_free_bodies_and_rows(self):
        """Give storage to the worlds of at least the rule's units per color in free bodies, in world order."""
        rule = sweep_units.ColoredRule(min_units=64, parallelism=4.0)
        free = np.array([1, 4, 0, 520, 3])
        np.testing.assert_array_equal(sweep_units.capable_worlds(free, 512, rule), [1, 3])
        self.assertEqual(sweep_units.capable_worlds(free, 63, rule).size, 0)
        self.assertEqual(sweep_units.capable_worlds(np.ones(16384), 512, rule).size, 0)
        # A world of one free body runs one unit a color whatever the rule: it never takes the colored mapping.
        anything = sweep_units.ColoredRule(min_units=1, parallelism=0.0)
        np.testing.assert_array_equal(sweep_units.capable_worlds(free, 512, anything), [1, 3, 4])

    def test_waves_of_worlds_scale_the_units_per_color(self):
        """Colored worlds past the device's blocks at once run in waves: each wave multiplies the units per color a
        world needs, in the builder's condition and in the free bodies a capable world needs."""
        rule = sweep_units.ColoredRule(min_units=32, parallelism=1.75)
        self.assertEqual(rule.at_scale(64, 188), rule)
        self.assertEqual(rule.at_scale(188, 188).waves, 1)
        self.assertEqual(rule.at_scale(189, 188).units_per_color, 3.5)
        scaled = rule.at_scale(1024, 188)
        self.assertEqual((scaled.waves, scaled.units_per_color), (6, 10.5))
        self.assertIn("10.5f * (float)(c)", scaled.condition("(u)", "(c)"))
        self.assertNotEqual(scaled.name, rule.name)
        # A 10-box pyramid holds no storage at that scale; a 36-box one still does.
        np.testing.assert_array_equal(sweep_units.capable_worlds(np.array([10, 36]), 512, scaled), [1])


class TestStageSlots(unittest.TestCase):
    """The slot staging (:func:`.sweep_units.stage_slots`) reads only what its units' rows hold, on every device."""

    WORLDS, ROWS = 2, 8

    @staticmethod
    def units(torsion: bool) -> tuple:
        """Each world's units as (first row, rows): contacts with their friction pair (and spin row under contact
        ``torsion``) and contacts without one."""
        full = 4 if torsion else 3
        return (((0, full), (full, 1)), ((0, 1), (1, full), (1 + full, 1)))

    def _stage(self, device, weights, weighted: bool, torsion: bool) -> tuple:
        """Stage :meth:`units` with the row weights ``weights``: the staged units, their slot fields in the layout,
        and the rows' coefficients."""
        rng = np.random.default_rng(7)
        layout, units = sweep_units.slot_layout(torsion), self.units(torsion)
        # The free rows' records (row_metadata.pack_row_header): their kind and parent words.
        meta = np.zeros((self.WORLDS, self.ROWS, HEADER_WORDS), dtype=np.int64)
        records = np.zeros((self.WORLDS, max(len(world) for world in units), 4), dtype=np.int32)
        for world, world_units in enumerate(units):
            for p, (row, rows) in enumerate(world_units):
                records[world, p] = (row, rows, 2 * p, 2 * p + 1)
                # A normal row links no patch ring (0xFFFF); its friction pair's (and spin row's) parent is the normal.
                meta[world, row, 3] = ROW_CONTACT | (0xFFFF << 16)
                for r in range(row + 1, row + rows):
                    kind = ROW_TORSION if r == row + 3 else ROW_FRICTION
                    meta[world, r, 3] = kind | (row << 16)
        meta = meta.astype(np.uint32).view(np.int32).reshape(self.WORLDS, HEADER_WORDS * self.ROWS)
        mu = rng.uniform(0.1, 1.0, (self.WORLDS, self.ROWS)).astype(np.float32)
        # The pool: each world's entries (its units and one more) from its slot's first entry, a world's place after
        # the other's; its block holds its fields field-major over its entries; the launch's one thread a world
        # strides its units.
        need = np.array([len(world) + 1 for world in units])
        firsts = np.concatenate([[0], np.cumsum(need)[:-1]])
        pool = int(need.sum())
        slot_data = wp.zeros(layout.fields * pool, dtype=float, device=device)
        slot_words = wp.zeros(layout.words * pool, dtype=wp.int32, device=device)
        color_slots = np.zeros((pool, 4), dtype=np.int32)
        for world, first in enumerate(firsts):
            color_slots[first : first + len(units[world])] = records[world, : len(units[world])]
        wp.launch(
            sweep_units.stage_slots(weighted),
            dim=(self.WORLDS, 1),
            inputs=[
                row_lines(device),
                # Every world in the colored bucket, each in the slot of its index.
                wp.array([self.WORLDS], dtype=int, device=device),
                wp.array(np.arange(self.WORLDS), dtype=int, device=device),
                wp.array(np.arange(self.WORLDS), dtype=wp.int32, device=device),
                wp.array(firsts, dtype=wp.int32, device=device),
                wp.array(need, dtype=wp.int32, device=device),
                wp.array([len(world) for world in units], dtype=wp.int32, device=device),
                wp.array(color_slots, dtype=wp.int32, device=device),
                wp.array(meta, dtype=wp.int32, device=device),
                0,
                wp.array(rng.standard_normal((self.WORLDS, self.ROWS, 12)), dtype=float, device=device),
                wp.array(mu, dtype=float, device=device),
                weights,
                wp.zeros(self.WORLDS, dtype=wp.int32, device=device),
                layout.rows,
                layout.fields,
                layout.words,
                1,
            ],
            outputs=[slot_data, slot_words],
            device=device,
        )
        staged = [
            (world, p, row, rows)
            for world, world_units in enumerate(units)
            for p, (row, rows) in enumerate(world_units)
        ]
        # By world, field and unit, as the assertions read them.
        flat, width = slot_data.numpy(), records.shape[1]
        data = np.zeros((self.WORLDS, layout.fields, width), dtype=np.float32)
        for world, first in enumerate(firsts):
            block = flat[layout.fields * first : layout.fields * (first + need[world])].reshape(
                layout.fields, need[world]
            )
            data[world, :, : len(units[world])] = block[:, : len(units[world])]
        return staged, data, layout, mu

    def test_unweighted_rows_never_index_the_weights(self):
        """An unweighted row law binds a one-element stand-in for the row weights, as the solver does. It sits at the
        start of a buffer whose every other element is NaN, so a read of any other row's weight stages NaN; every
        unit stages the weight 1, its friction pair's coefficient and its spin row's, a unit without them 0. A weighted
        law stages each normal row's weight."""
        for device in ("cpu", *(("cuda:0",) if wp.is_cuda_available() else ())):
            for torsion in (False, True):
                with self.subTest(device=device, torsion=torsion):
                    guard = np.full((self.WORLDS, self.ROWS), np.nan, dtype=np.float32)
                    guard[0, 0] = 1.0
                    stand_in = wp.array(guard, dtype=float, device=device)[:1, :1]
                    staged, data, layout, row_mu = self._stage(device, stand_in, False, torsion)
                    for world, p, row, rows in staged:
                        self.assertEqual(
                            data[world, layout.weight, p], 1.0, f"world {world} unit {p}: stand-in indexed"
                        )
                        self.assertEqual(data[world, layout.mu, p], row_mu[world, row + 1] if rows >= 3 else 0.0)
                        if torsion:
                            self.assertEqual(
                                data[world, layout.spin_mu, p], row_mu[world, row + 3] if rows > 3 else 0.0
                            )
                    weights = np.random.default_rng(3).uniform(0.2, 0.9, (self.WORLDS, self.ROWS)).astype(np.float32)
                    staged, data, layout, _ = self._stage(
                        device, wp.array(weights, dtype=float, device=device), True, torsion
                    )
                    for world, p, row, _ in staged:
                        self.assertEqual(data[world, layout.weight, p], weights[world, row])


class TestColoredSweep(unittest.TestCase):
    device = _cuda()

    def setUp(self):
        if self.device is None:
            self.skipTest("No CUDA device")

    def _simulate(self, scene, law: str, solvers: list | None = None, units: int = 1, parallelism: float = 0.0):
        physics = conf._physics(scene, law, 32)
        with colored(units, solvers, parallelism):
            return conf.simulate(physics.scene, _ROUTE, self.device, options=physics.options(32))

    def _built(self, law: str = "patch") -> tuple:
        """The tray's solver after its last step, with its colored units and the packed rows they came from."""
        solvers = []
        self._simulate(tray(), law, solvers)
        solver = solvers[-1]
        self.assertIsNotNone(solver.sweep.colored_units, "the tray's worlds take the colored mapping")
        host = {name: array.numpy() for name, array in solver.sweep.colored_units.units.items()}
        # The free bodies' records (row_metadata.pack_row_header).
        placement = solver.placement
        free_first = placement.body_base.numpy() * placement.dense_record_words
        free = record_views(placement.row_record.numpy(), free_first)[1]
        # A world's records are its line's (a run's first line where its rows run in the arena).
        metas = np.stack([free[line][: placement.mf_max_constraints] for line in placement.dense_first.numpy()])
        return solver, host, metas

    def test_units_follow_the_host_rules(self):
        """Units, row-order levels and the slot records by color equal the host reference's."""
        solver, host, metas = self._built()
        counts, ends = solver.placement.mf_constraint_count.numpy(), solver.placement.mf_contact_rows_end.numpy()
        for world in range(solver.forest.world_count):
            want = reference_units(metas[world], int(counts[world]), int(ends[world]))
            first = int(host["slot_first"][int(host["world_slot"][world])])
            units = int(host["unit_count"][world])
            self.assertGreater(units, 0)
            self.assertEqual(units, want["unit_count"])
            # The world's entries of the pool: its units and its colors' offsets, one more.
            self.assertEqual(int(host["slot_need"][int(host["world_slot"][world])]), units + 1)
            np.testing.assert_array_equal(host["unit_color"][first : first + units], want["unit_color"])
            colors = int(host["color_count"][world])
            self.assertEqual(colors, want["color_count"])
            np.testing.assert_array_equal(host["color_offset"][first : first + colors + 1], want["color_offset"])
            np.testing.assert_array_equal(host["color_slot"][first : first + units], want["color_slot"])
            np.testing.assert_array_equal(host["color_after"][first : first + units], want["color_after"])
            # A unit's color is one past its bodies' latest earlier colors: its place's color.
            place_color = np.repeat(np.arange(colors), np.diff(want["color_offset"]))
            np.testing.assert_array_equal(host["color_after"][first : first + units].max(axis=1) + 1, place_color)
        # The slots take their entries in slot order, one after another.
        need = host["slot_need"]
        np.testing.assert_array_equal(host["slot_first"], np.concatenate([[0], np.cumsum(need)[:-1]]))
        stats = solver.sweep.colored_units.stats()
        self.assertEqual(len(stats), solver.forest.world_count)
        self.assertTrue(all(world["largest_color"] <= world["units"] for world in stats))
        self.assertFalse(solver.sweep.colored_units.capacity_status())

    def test_the_route_record_reports_the_storage(self):
        """The path selection records the colored kernel, its capable worlds and their storage's bytes."""
        solver, _, _ = self._built()
        record = solver.get_path_selection()["routes"]["sweep"]["mf_colored"]
        units = solver.sweep.colored_units
        self.assertEqual(record["capable_worlds"], solver.forest.world_count)
        self.assertEqual(record["waves"], 1, "the tray's worlds fit one wave on the device's SMs")
        arrays = (*units.units.values(), *units.slots.values(), units.status)
        self.assertEqual(record["storage_bytes"], sum(array.capacity for array in arrays))
        self.assertGreater(record["storage_bytes"], 0)
        # The pool holds every slot's share (a unit for every three rows of the row capacity) and its colors' one more,
        # and one world's every row past its share.
        rows = solver.placement.mf_max_constraints
        share = sweep_units.unit_capacity(rows)
        pool = units.slot_count * (share + 1) + rows - share
        self.assertEqual(sweep_units.pool_capacity(units.slot_count, rows), pool)
        self.assertEqual(units.slots["slot_data"].shape, (units.layout.fields, pool))

    def test_a_color_never_shares_a_body_or_a_patch(self):
        """No two units of a color touch the same body, and a patch's contacts never share a color."""
        solver, host, metas = self._built()
        parents = metas[:, :, 3] >> 16
        for world in range(solver.forest.world_count):
            first = int(host["slot_first"][int(host["world_slot"][world])])
            offsets = host["color_offset"][first:]
            for color in range(int(host["color_count"][world])):
                records = host["color_slot"][first + offsets[color] : first + offsets[color + 1]]
                bodies = [dof for record in records for dof in set(record[2:]) if dof >= 0]
                self.assertEqual(len(bodies), len(set(bodies)), f"color {color} shares a body")
                rows = {int(record[0]) for record in records}
                for row in rows:
                    ring, hop = int(parents[world, row]), 0
                    while ring >= 0 and ring != row and hop < 64:
                        self.assertNotIn(ring, rows, f"color {color} holds two contacts of one patch")
                        ring, hop = int(parents[world, ring]), hop + 1

    def test_colors_keep_the_row_order(self):
        """Units that share a body run in row order: the earlier one takes the lower color."""
        solver, host, _ = self._built()
        for world in range(solver.forest.world_count):
            first, last = int(host["slot_first"][int(host["world_slot"][world])]), {}
            for unit in range(int(host["unit_count"][world])):
                color = int(host["unit_color"][first + unit])
                dofs = int(host["unit_dofs"][first + unit])
                for dof in {dofs >> 16, _signed(dofs, 16)}:
                    if dof < 0:
                        continue
                    if dof in last:
                        self.assertGreater(color, last[dof], f"unit {unit} on body {dof}")
                    last[dof] = color

    def test_a_world_past_the_pool_keeps_the_warp(self):
        """A world of more units than the pool holds from its first entry keeps the one-warp mapping and sets the
        status: bitwise the world the rule refuses (both are the one-warp mapping beside a built colored one, today's
        rows). A model with no colored mapping solves its free bodies' contacts as regions, another law, until the
        colored route's unit is the region too."""
        # The tray's 76 units in a pool of one entry a slot.
        solvers = []
        with mock.patch.object(sweep_units, "pool_capacity", lambda slots, rows: slots):
            over = self._simulate(tray(), "patch", solvers)
        units = solvers[-1].sweep.colored_units
        self.assertIsNotNone(units)
        self.assertTrue(units.capacity_status())
        stats = units.stats()
        self.assertTrue(stats and not any(world["colored"] for world in stats), stats)
        self.assertTrue(all(world["units"] + 1 > units.slot_count for world in stats), stats)
        refused = []
        thin = self._simulate(tray(), "patch", refused, parallelism=5.0)
        self.assertIsNotNone(refused[-1].sweep.colored_units)
        for name in ("body_q", "body_qd"):
            np.testing.assert_array_equal(getattr(over, name), getattr(thin, name), err_msg=name)

    def test_a_world_past_its_share_keeps_its_units_in_the_pool(self):
        """A world of more units than its slot's share (a unit is a contact's normal row; under friction patches most
        contacts carry no friction pair) keeps its units where the pool has room: it takes the colored mapping and
        steps bitwise as with a share that holds it (the same units, colors and records, elsewhere in memory). The
        worlds take their entries in world order; one past the pool's end keeps the one-warp mapping."""
        # The tray's 76 units a world against a share of one unit: the pool holds one world's every row past it.
        default, solvers = [], []
        held = self._simulate(tray(), "patch", default)
        with mock.patch.object(sweep_units, "UNIT_ROWS", 1024):
            shared = self._simulate(tray(), "patch", solvers)
        units = solvers[-1].sweep.colored_units
        self.assertEqual(units.unit_capacity, max(solvers[-1].placement.mf_max_constraints // 1024, 1))
        stats = units.stats()
        colored_worlds = [world["world"] for world in stats if world["colored"]]
        self.assertTrue(colored_worlds, stats)
        self.assertTrue(all(world["units"] > units.unit_capacity for world in stats), stats)
        # Entries in world order: the colored worlds are the first ones, and a world past the end sets the status.
        self.assertEqual(colored_worlds, list(range(len(colored_worlds))))
        self.assertEqual(units.capacity_status(), len(colored_worlds) < len(stats))
        bodies = held.body_q.shape[0] // solvers[-1].forest.world_count
        for name in ("body_q", "body_qd"):
            for world in colored_worlds:
                span = slice(world * bodies, (world + 1) * bodies)
                np.testing.assert_array_equal(
                    getattr(shared, name)[span], getattr(held, name)[span], err_msg=f"{name} world {world}"
                )

    def test_repeats_bitwise_and_splits_the_worlds(self):
        """A repeat gives the same trajectory bitwise; a rule past the row capacity builds no colored mapping."""
        first, second = (self._simulate(tray(), "patch") for _ in range(2))
        for name in ("body_q", "body_qd"):
            np.testing.assert_array_equal(getattr(first, name), getattr(second, name), err_msg=name)
        solvers = []
        self._simulate(tray(), "patch", solvers, units=4096)
        self.assertIsNone(solvers[-1].sweep.colored_units, "no world's row capacity holds 4096 contact units")

    def test_every_sweep_reads_the_records_the_placement_wrote(self):
        """Where the one-warp sweep streams its dense rows' records and the colored mapping stages them, each its own
        stage's decision (coins at 512 worlds on the GB300), the colored mapping reads its records where the placement
        wrote them: its worlds step bitwise as where both stage."""
        original = sweep_plan.Sweep._staged

        def streamed(plan, config, items):
            staged = original(plan, config, items)
            if config.colored:
                return staged
            return dataclasses.replace(staged, places=dataclasses.replace(staged.places, metadata=Place.STREAMED))

        physics = conf._physics(tray(), "patch", 32)
        options = physics.options(32)
        runs, solvers = [], []
        for stage in (original, streamed):
            with mock.patch.object(sweep_plan.Sweep, "_staged", stage), colored(1, solvers):
                runs.append(conf.simulate(physics.scene, _ROUTE, self.device, options=options))
        placement, sweep = solvers[-1].placement, solvers[-1].sweep
        self.assertEqual(placement.record_first, 0, "the streamed sweep's dense records lead the line")
        self.assertEqual(sweep.colored_sweep.places.metadata, Place.SHARED)
        words = record_layout(placement.record_first, placement.dense_max_constraints, placement.row_pitch)[1]
        # The sweep reads each line's records at the store's line, the free rows' after the dense rows' (from the
        # line's body base).
        self.assertEqual(placement.row_record.shape[1], words)
        source = sweep_routes.mf_gs_source(sweep.colored_sweep)[0]
        free = f"free_row0 * {RECORD_WORDS}"
        self.assertIn(f"const int off_rec_mf = line * {line_size('row_record')} + {free};", source)
        for name in ("body_q", "body_qd"):
            np.testing.assert_array_equal(getattr(runs[0], name), getattr(runs[1], name), err_msg=name)

    def test_the_rule_leaves_thin_worlds_to_the_warp(self):
        """A world under the rule's units per color keeps the one-warp mapping beside a built colored one; the
        same model with no colored mapping (a rule past the row capacity) solves its free bodies' contacts as regions
        (the known gap of the colored route, which keeps today's rows: the two are compared where both run rows,
        :meth:`test_a_world_past_the_pool_keeps_the_warp`)."""
        # Five units per color: the tray's 19 free bodies admit it, its 76 units in 44 colors do not take it.
        solvers = []
        thin = self._simulate(tray(), "patch", solvers, parallelism=5.0)
        built = solvers[-1].sweep.colored_units
        self.assertIsNotNone(built)
        stats = built.stats()
        self.assertTrue(stats and not any(world["colored"] for world in stats), stats)
        self.assertIsNone(solvers[-1].sweep.regions, "beside a colored mapping the free bodies keep their rows")
        # The plain one-warp sweep: a rule past the row capacity builds no colored mapping (the default rule takes
        # the tray's worlds of 1.75 units per color and more), and the free bodies' contacts are regions.
        plain = []
        self._simulate(tray(), "patch", plain, units=4096)
        self.assertIsNone(plain[-1].sweep.colored_units)
        self.assertIsNotNone(plain[-1].sweep.regions)
        self.assertTrue(np.isfinite(thin.body_q).all())


class TestSpilledImpulses(unittest.TestCase):
    """The free rows' impulses past the block's staged slots, in place (:attr:`.sweep_routes.RowPlaces.free_slots`)."""

    def test_spilled_impulses_hold_the_staged_values(self):
        """The tray's free impulses with staged slots, the form a block that cannot hold every row's takes, step bit
        for bit as with every impulse staged: in place where the world's rows outgrow four slots, staged where they
        fit 256, on the CPU's sweep, and on CUDA on the colored mapping and the warp's."""
        for slots in (4, 256):
            with self.subTest(slots=slots):
                self._spilled(slots)

    def test_two_heads_hold_the_staged_values(self):
        """With a wide head too (a launch a head, each world swept by the launch whose head holds its live rows), the
        tray steps bit for bit as with every impulse staged: its world in the wide launch where its rows outgrow the
        stage's four slots, in the stage's launch where they fit its 256 (on CUDA at a prefix of the kernel's dynamic
        stage, whose wide head passes the static stage), and in place where they outgrow both heads."""
        for slots, wide in ((4, 12288), (256, 12288), (4, 8)):
            with self.subTest(slots=slots, wide=wide):
                self._spilled(slots, wide)

    def _spilled(self, slots: int, wide: int = 0):
        original = sweep_plan.Sweep._staged

        def spilled(plan, config, items):
            staged = original(plan, config, items)
            if not config.free_rows or staged.places.impulses is not Place.SHARED:
                return staged
            places = dataclasses.replace(staged.places, free_slots=slots, wide_slots=wide)
            return dataclasses.replace(staged, places=places)

        devices = ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])
        physics = conf._physics(tray(), "patch", 32)
        for device in devices:
            route = conf.REFERENCE_CPU if device == "cpu" else _ROUTE
            with self.subTest(device=device):
                runs, solvers = [], []
                for stage in (original, spilled):
                    with mock.patch.object(sweep_plan.Sweep, "_staged", stage), colored(1, solvers):
                        runs.append(conf.simulate(physics.scene, route, device, options=physics.options(32)))
                sweeps = [solvers[-1].sweep.general_sweep]
                if device != "cpu":
                    sweeps.append(solvers[-1].sweep.colored_sweep)
                for sweep in sweeps:
                    self.assertEqual(sweep.places.free_slots, slots, sweep.name)
                    self.assertEqual(sweep.places.wide_slots, wide, sweep.name)
                    self.assertIn(f"_slots{slots}" + (f"to{wide}" if wide else ""), sweep.name)
                launches = [launch for launch in solvers[-1].sweep.launches if launch.config in sweeps]
                self.assertEqual(len(launches), len(sweeps) * (2 if wide else 1))
                rows = int(solvers[-1].placement.mf_constraint_count.numpy().max())
                self.assertTrue(4 < rows <= 256, "the tray's rows outgrow four slots and fit 256")
                for name in ("body_q", "body_qd"):
                    np.testing.assert_array_equal(getattr(runs[0], name), getattr(runs[1], name), err_msg=name)


def _conformance_tests():
    """One conformance check per (scene, law, contact torsion): the gate's laws and the distance to its converged
    reference; and the compliant stack's material contacts, whose units step by the weights their slots stage."""
    variants = ({}, {"contact_torsion": True})
    problems = [
        conf._physics(scene, law, 32, "torsion" if variant else "", **variant)
        for law, variant, scene in itertools.product(("point", "patch"), variants, (tray(), conf.scenes.box_stack()))
    ]
    problems.append(conf._physics(conf.scenes.compliant_stack(), "point", 32, "compliance", contact_compliance=True))
    for physics in problems:
        group = conf.Group(physics, conf.REFERENCE_CUDA, ())

        def test(self, group=group):
            # The reference runs first, outside the colored mapping.
            conf.reference(group.physics, group.reference, self.device)
            with colored():
                failures = conf._failures(conf.run_checks(group, _ROUTE, self.device))
            self.assertFalse(failures, f"{group.name} through {_ROUTE.name}:\n{conf._message(failures)}")

        test.__doc__ = f"Check {group.name} through the colored mapping against the gate."
        setattr(TestColoredSweep, f"test_{group.name}_conforms", test)


_conformance_tests()


if __name__ == "__main__":
    unittest.main(verbosity=2)
