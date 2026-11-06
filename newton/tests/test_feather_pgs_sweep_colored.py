# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The colored mapping of FeatherPGS's matrix-free sweep (:mod:`.sweep_units`, ``mf_gs_kernel(mapping="colored")``).

A world's contacts become units (a normal row and its friction pair) and each unit's color is its level in row
order (one past the latest color of an earlier unit on its bodies), so no two units of a color share a body and
units that share one keep their row order. The unit builder gives the colored mapping the worlds its rule picks
(``sweep_units.COLORED_RULE``); the tests relax the rule to one unit so every world with contacts takes it. They
check

- the unit builder against a host reference of the same rules (units, row-order levels, the slot records), that a
  color never shares a body or a patch, that the colors keep the row order, and that a world past its slot's units
  keeps the one-warp mapping and sets the capacity status;
- that a world's result repeats bitwise and that the warp's worlds and the colored worlds split the batch;
- the physics: the colored mapping is the reference's row-order Gauss-Seidel sweep, so it meets the conformance
  gate's laws and its distance to the converged reference on a tray carrying a grid of boxes (a hub body) and on
  the gate's box stack.
"""

from __future__ import annotations

import contextlib
import dataclasses
import itertools
import unittest
from unittest import mock

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import sweep_units
from newton._src.solvers.feather_pgs.row_kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_TORSION,
)
from newton.solvers import SolverFeatherPGS
from newton.tests import feather_pgs_scenes as scenes
from newton.tests import test_feather_pgs_conformance as conf
from newton.tests.feather_pgs_law_harness import row_lines


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

    return scenes.Scene(
        "tray", build, dt=1.0 / 240.0, steps=60, params={"mu": [mu]}, passive=True, deterministic_contacts=True
    )


@contextlib.contextmanager
def colored(units: int = 1, solvers: list | None = None, parallelism: float = 0.0):
    """Every solver built inside gives the colored mapping the worlds of at least ``units`` contact units and
    ``parallelism`` units per color."""
    original = SolverFeatherPGS.__init__

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
        if kind[row] != PGS_CONSTRAINT_TYPE_CONTACT:
            continue
        rows = 1
        # Its friction pair, then its spin row where it has one.
        while row + rows < end and rows < 4:
            r = row + rows
            want = PGS_CONSTRAINT_TYPE_FRICTION if rows < 3 else PGS_CONSTRAINT_TYPE_TORSION
            if kind[r] != want or parent[r] != row:
                break
            rows += 1
        word = int(meta[row, 0])
        units.append((row, rows, _signed(word >> 16, 16), _signed(word, 16)))
    level, colors = {}, []
    for _, _, a, b in units:
        bodies = [dof for dof in {a, b} if dof >= 0]
        color = max([level[dof] + 1 for dof in bodies if dof in level], default=0)
        for dof in bodies:
            level[dof] = color
        colors.append(color)
    count = max(colors, default=-1) + 1
    sizes = np.bincount(np.asarray(colors, dtype=np.int64), minlength=count)
    order = sorted(range(len(units)), key=lambda unit: (colors[unit], unit))
    return {
        "unit_count": len(units),
        "unit_color": np.asarray(colors, dtype=np.int64),
        "color_count": count,
        "color_offset": np.concatenate([[0], np.cumsum(sizes)]),
        "color_slot": np.asarray([units[unit] for unit in order], dtype=np.int64).reshape(-1, 4),
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
        meta = np.zeros((self.WORLDS, self.ROWS, 4), dtype=np.int64)
        records = np.zeros((self.WORLDS, max(len(world) for world in units), 4), dtype=np.int32)
        for world, world_units in enumerate(units):
            for p, (row, rows) in enumerate(world_units):
                records[world, p] = (row, rows, 2 * p, 2 * p + 1)
                # A normal row links no patch ring (0xFFFF); its friction pair's (and spin row's) parent is the normal.
                meta[world, row, 3] = PGS_CONSTRAINT_TYPE_CONTACT | (0xFFFF << 16)
                for r in range(row + 1, row + rows):
                    kind = PGS_CONSTRAINT_TYPE_TORSION if r == row + 3 else PGS_CONSTRAINT_TYPE_FRICTION
                    meta[world, r, 3] = kind | (row << 16)
        meta = meta.astype(np.uint32).view(np.int32).reshape(self.WORLDS, 4 * self.ROWS)
        mu = rng.uniform(0.1, 1.0, (self.WORLDS, self.ROWS)).astype(np.float32)
        slot_data = wp.zeros((self.WORLDS, layout.fields, records.shape[1]), dtype=float, device=device)
        slot_words = wp.zeros((self.WORLDS, layout.words, records.shape[1]), dtype=wp.int32, device=device)
        wp.launch(
            sweep_units.stage_slots(weighted),
            dim=records.shape[:2],
            inputs=[
                row_lines(device),
                wp.array(np.arange(self.WORLDS), dtype=wp.int32, device=device),
                wp.ones(self.WORLDS, dtype=wp.int32, device=device),
                wp.array([len(world) for world in units], dtype=wp.int32, device=device),
                wp.array(records, dtype=wp.int32, device=device),
                wp.array(meta, dtype=wp.int32, device=device),
                wp.array(rng.standard_normal((self.WORLDS, self.ROWS, 12)), dtype=float, device=device),
                wp.array(mu, dtype=float, device=device),
                weights,
                0,
                layout.rows,
            ],
            outputs=[slot_data, slot_words],
            device=device,
        )
        staged = [
            (world, p, row, rows)
            for world, world_units in enumerate(units)
            for p, (row, rows) in enumerate(world_units)
        ]
        return staged, slot_data.numpy(), layout, mu

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
        metas = solver.placement.mf_meta_packed.numpy().reshape(
            solver.forest.world_count, solver.placement.mf_max_constraints, 4
        )
        return solver, host, metas

    def test_units_follow_the_host_rules(self):
        """Units, row-order levels and the slot records by color equal the host reference's."""
        solver, host, metas = self._built()
        counts, ends = solver.placement.mf_constraint_count.numpy(), solver.placement.mf_contact_rows_end.numpy()
        for world in range(solver.forest.world_count):
            want = reference_units(metas[world], int(counts[world]), int(ends[world]))
            slot = int(host["world_slot"][world])
            units = int(host["unit_count"][world])
            self.assertGreater(units, 0)
            self.assertEqual(units, want["unit_count"])
            np.testing.assert_array_equal(host["unit_color"][slot, :units], want["unit_color"])
            colors = int(host["color_count"][world])
            self.assertEqual(colors, want["color_count"])
            np.testing.assert_array_equal(host["color_offset"][slot, : colors + 1], want["color_offset"])
            np.testing.assert_array_equal(host["color_slot"][slot, :units], want["color_slot"])
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
        arrays = (*units.units.values(), *units.slots.values(), units.slot_world, units.status)
        self.assertEqual(record["storage_bytes"], sum(array.capacity for array in arrays))
        self.assertGreater(record["storage_bytes"], 0)
        # A slot holds a unit for every three rows of the row capacity.
        capacity = sweep_units.unit_capacity(solver.placement.mf_max_constraints)
        self.assertEqual(units.slots["slot_data"].shape, (units.slot_count, units.layout.fields, capacity))

    def test_a_color_never_shares_a_body_or_a_patch(self):
        """No two units of a color touch the same body, and a patch's contacts never share a color."""
        solver, host, metas = self._built()
        parents = metas[:, :, 3] >> 16
        for world in range(solver.forest.world_count):
            slot = int(host["world_slot"][world])
            offsets = host["color_offset"][slot]
            for color in range(int(host["color_count"][world])):
                records = host["color_slot"][slot, offsets[color] : offsets[color + 1]]
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
            slot, last = int(host["world_slot"][world]), {}
            for unit in range(int(host["unit_count"][world])):
                color = int(host["unit_color"][slot, unit])
                dofs = int(host["unit_dofs"][slot, unit])
                for dof in {dofs >> 16, _signed(dofs, 16)}:
                    if dof < 0:
                        continue
                    if dof in last:
                        self.assertGreater(color, last[dof], f"unit {unit} on body {dof}")
                    last[dof] = color

    def test_a_world_past_its_slot_keeps_the_warp(self):
        """A world of more units than its slot holds keeps the one-warp mapping, bitwise, and sets the status."""
        # The tray's 76 units in slots of one unit each.
        solvers = []
        with mock.patch.object(sweep_units, "UNIT_ROWS", 1024):
            over = self._simulate(tray(), "patch", solvers)
        units = solvers[-1].sweep.colored_units
        self.assertIsNotNone(units)
        self.assertTrue(units.capacity_status())
        stats = units.stats()
        self.assertTrue(stats and not any(world["colored"] for world in stats), stats)
        self.assertTrue(all(world["units"] > units.unit_capacity for world in stats), stats)
        plain = []
        warp = self._simulate(tray(), "patch", plain, units=4096)
        self.assertIsNone(plain[-1].sweep.colored_units)
        for name in ("body_q", "body_qd"):
            np.testing.assert_array_equal(getattr(over, name), getattr(warp, name), err_msg=name)

    def test_repeats_bitwise_and_splits_the_worlds(self):
        """A repeat gives the same trajectory bitwise; a rule past the row capacity builds no colored mapping."""
        first, second = (self._simulate(tray(), "patch") for _ in range(2))
        for name in ("body_q", "body_qd"):
            np.testing.assert_array_equal(getattr(first, name), getattr(second, name), err_msg=name)
        solvers = []
        self._simulate(tray(), "patch", solvers, units=4096)
        self.assertIsNone(solvers[-1].sweep.colored_units, "no world's row capacity holds 4096 contact units")

    def test_the_rule_leaves_thin_worlds_to_the_warp(self):
        """A world under the rule's units per color keeps the one-warp mapping beside a built colored one, bitwise."""
        # Five units per color: the tray's 19 free bodies admit it, its 76 units in 44 colors do not take it.
        solvers = []
        thin = self._simulate(tray(), "patch", solvers, parallelism=5.0)
        built = solvers[-1].sweep.colored_units
        self.assertIsNotNone(built)
        stats = built.stats()
        self.assertTrue(stats and not any(world["colored"] for world in stats), stats)
        # The plain one-warp sweep: a rule past the row capacity builds no colored mapping (the default rule takes
        # the tray's worlds of 1.75 units per color and more).
        plain = []
        warp = self._simulate(tray(), "patch", plain, units=4096)
        self.assertIsNone(plain[-1].sweep.colored_units)
        for name in ("body_q", "body_qd"):
            np.testing.assert_array_equal(getattr(thin, name), getattr(warp, name), err_msg=name)


def _conformance_tests():
    """One conformance check per (scene, law, contact torsion): the gate's laws and the distance to its converged
    reference; and the compliant stack's material contacts, whose units step by the weights their slots stage."""
    variants = ({}, {"contact_torsion": True})
    problems = [
        conf._physics(
            dataclasses.replace(scene, deterministic_contacts=True), law, 32, "torsion" if variant else "", **variant
        )
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
