# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The colored and mass-split sweeps of FeatherPGS's free-body rows (:mod:`.sweep_units`, :class:`.ColoredSweep`).

A world's contacts become units (a normal row and its friction pair), bodies with more units than the split budget
become copies, and each unit's color is its level in row order (one past the latest color of an earlier unit on its
body copies), so no two units of a color share a body copy and units that share one keep their row order. The tests
check

- the unit builder against a host reference of the same rules (units, copies per patch, row-order levels, the slot
  records the sweep reads), and that the colors keep the row order;
- that a world's result does not depend on the threads of its block (a color's units commute, the patch rings
  included), bitwise, and repeats bitwise;
- the physics of coloring (no split), the hybrid and mass splitting against the conformance gate's converged
  reference and contact laws, on a tray carrying a grid of boxes (a hub body) and on the gate's box stack. The
  measured gaps are expected failures (:data:`GAPS`).
"""

from __future__ import annotations

import contextlib
import dataclasses
import unittest
from unittest import mock

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs.kernels import PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_FRICTION
from newton._src.solvers.feather_pgs.sweep_routes import ColoredSweep
from newton._src.solvers.feather_pgs.sweep_units import AUTO_BUDGET
from newton.solvers import SolverFeatherPGS
from newton.tests import feather_pgs_scenes as scenes
from newton.tests import test_feather_pgs_conformance as conf


def tray(*, mu: float = 0.6, grid: int = 3, layers: int = 2) -> scenes.Scene:
    """A 2 kg tray on the ground under ``grid x grid`` columns of 0.1 m, 0.2 kg boxes, ``layers`` high.

    The tray carries every column, so its units outnumber any box's several times over: the hub a split budget
    splits first.
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
def colored(
    budget: int, threads: int, sweeps: list | None = None, *, extra: int = 512, colors: int = 64, cluster: int = 1
):
    """Every solver built inside runs the colored sweep at split budget ``budget`` with ``threads`` per world (in a
    cluster of ``cluster`` blocks)."""
    original = SolverFeatherPGS.__init__

    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        sweep = ColoredSweep(self, budget=budget, extra=extra, colors=colors, threads=threads, cluster=cluster)
        sweep.install()
        if sweeps is not None:
            sweeps.append(sweep)

    with mock.patch.object(SolverFeatherPGS, "__init__", init):
        yield


def _route(budget: int, threads: int) -> conf.Route:
    """The gate's reference route (the plain matrix-free sweep owns every world), its sweep then colored."""
    label = "auto" if budget == AUTO_BUDGET else budget
    return conf.REFERENCE_CUDA.but(f"mf_colored_b{label}_t{threads}")


def _signed(value: int, bits: int) -> int:
    value &= (1 << bits) - 1
    return value - (1 << bits) if value >> (bits - 1) else value


def reference_units(meta, count: int, end: int, budget: int, dofs: int, extra: int, max_colors: int) -> dict:
    """The unit builder's rules on the host for one world: units, splits, copies, row-order levels and slot records."""
    rows_capacity = meta.shape[0]
    m = min(count, rows_capacity)
    end = min(end, m)
    packed = [int(word) for word in meta[:, 3]]
    kind = [word & 0xFFFF for word in packed]
    parent = [_signed(word >> 16, 16) for word in packed]
    units = []
    for row in range(end):
        if kind[row] != PGS_CONSTRAINT_TYPE_CONTACT:
            continue
        rows = 1
        while row + rows < end and rows < 3:
            r = row + rows
            if kind[r] != PGS_CONSTRAINT_TYPE_FRICTION or parent[r] != row:
                break
            rows += 1
        word = int(meta[row, 0])
        units.append((row, rows, _signed(word >> 16, 16), _signed(word, 16)))
    used = sorted({d for _, _, a, b in units for d in (a, b) if d >= 0})
    node_of = {dof: n for n, dof in enumerate(used)}
    degree = dict.fromkeys(used, 0)
    for _, _, a, b in units:
        if a >= 0:
            degree[a] += 1
        if b >= 0 and b != a:
            degree[b] += 1
    limit = budget if budget > 0 else 1 << 31
    if budget == AUTO_BUDGET and used:
        limit = max(1, (2 * sum(degree.values()) + len(used) - 1) // len(used))
    bodies = dofs // 6 + 1
    # Candidates over the budget in coordinate order (at most the copy capacity's worth).
    candidates, short = [], False
    for dof in used:
        if degree[dof] > limit:
            if len(candidates) > extra:
                short = True
                continue
            candidates.append(dof)
    # Copies by patch, next fit in row order: a patch opens the next copy when the current one cannot hold it.
    fill, current, row_copy, unit_copy = dict.fromkeys(candidates, 0), dict.fromkeys(candidates, 0), {}, []
    for row, _rows, a, b in units:
        b_side = -1 if b == a else b
        sides = (a, b_side)
        copy = [0, 0]
        if any(dof in fill for dof in sides):
            head, size, r, hop = row, 1, parent[row], 0
            while r >= 0 and r != row and hop < rows_capacity:
                head, size = min(head, r), size + 1
                r, hop = parent[r], hop + 1
            for side, dof in enumerate(sides):
                if dof not in fill:
                    continue
                if head == row:
                    if fill[dof] > 0 and fill[dof] + size > limit:
                        current[dof], fill[dof] = current[dof] + 1, 0
                    fill[dof] += size
                    copy[side] = current[dof]
                else:
                    copy[side] = row_copy[head][side]
            if head == row:
                row_copy[row] = tuple(copy)
        unit_copy.append(copy)
    split, splits, taken_extra = {}, [], 0
    for dof in candidates:
        copies = current[dof] + 1
        if copies < 2 or taken_extra + copies - 1 > extra:
            short = short or copies >= 2
            continue
        split[dof] = (len(splits), copies, taken_extra)
        splits.append((dof, copies, taken_extra))
        taken_extra += copies - 1
    level_of = {}
    overflow, colors = False, 0
    placed, slots = [], []
    for (row, rows, a, b), copy in zip(units, unit_copy, strict=True):
        b_side = -1 if b == a else b
        nodes, offsets = [], []
        for side, dof in enumerate((a, b_side)):
            if dof < 0:
                nodes.append(-1)
                offsets.append(dof)
                continue
            if dof in split and copy[side] > 0:
                _, _, first = split[dof]
                nodes.append(bodies + first + copy[side] - 1)
                offsets.append(dofs + 6 * (first + copy[side] - 1))
                continue
            nodes.append(node_of[dof])
            offsets.append(dof)
        color = max([level_of[n] + 1 for n in nodes if n in level_of], default=0)
        if color >= max_colors:
            overflow, color = True, max_colors - 1
        for n in nodes:
            if n >= 0:
                level_of[n] = color
        colors = max(colors, color + 1)
        placed.append((row, rows, a, b, b_side, color, offsets))
    # A split body's units in a color scale their sides (each copy holds at most one); an overflow unit runs on the
    # body itself, alone.
    per_color = {}
    for _, _, a, _, b_side, color, _ in placed:
        for dof in (a, b_side):
            if dof in split:
                per_color[dof, color] = per_color.get((dof, color), 0) + 1
    split_units = np.zeros((len(splits), max_colors), dtype=np.int64)
    for (dof, color), count in per_color.items():
        split_units[split[dof][0], color] = 1 if overflow and color == max_colors - 1 else count
    for row, rows, a, b, b_side, color, placed_offsets in placed:
        serial = overflow and color == max_colors - 1
        scale, offsets = [1, 1], list(placed_offsets)
        for side, dof in enumerate((a, b_side)):
            if dof in split:
                scale[side] = 1 if serial else per_color[dof, color]
                if serial:
                    offsets[side] = dof
        if b == a:
            offsets[1], scale[1] = offsets[0], scale[0]
        slots.append((color, (row, rows | (scale[0] << 2) | (scale[1] << 17), offsets[0], offsets[1])))
    sizes = np.zeros(max_colors, dtype=np.int64)
    for color, _ in slots:
        sizes[color] += 1
    ordered = [record for _, record in sorted(enumerate(slots), key=lambda item: (item[1][0], item[0]))]
    return {
        "budget": 0 if limit == 1 << 31 else limit,
        "unit_count": len(units),
        "color_count": max_colors if overflow else colors,
        "color_offset": np.concatenate([[0], np.cumsum(sizes)]),
        "color_slot": np.array([record for _, record in ordered], dtype=np.int64).reshape(-1, 4),
        "splits": splits,
        "split_units": split_units,
        "extra_copies": taken_extra,
        "overflow": (1 if overflow else 0) | (2 if short else 0),
    }


def _cuda():
    return wp.get_device("cuda:0") if wp.is_cuda_available() else None


class TestColoredSweep(unittest.TestCase):
    device = _cuda()

    def setUp(self):
        if self.device is None:
            self.skipTest("No CUDA device")

    def _simulate(
        self, scene, law: str, budget: int, threads: int, sweeps: list | None = None, launches=None, cluster: int = 1
    ):
        physics = conf._physics(scene, law, 32)
        with colored(budget, threads, sweeps, cluster=cluster):
            if launches is None:
                return conf.simulate(physics.scene, _route(budget, threads), self.device, options=physics.options(32))
            original = ColoredSweep.launch

            def launch(sweep, **kwargs):
                launches.append((sweep, kwargs["mf_meta"]))
                original(sweep, **kwargs)

            with mock.patch.object(ColoredSweep, "launch", launch):
                return conf.simulate(physics.scene, _route(budget, threads), self.device, options=physics.options(32))

    def test_units_follow_the_host_rules(self):
        """Units, copies per patch, row-order levels and slot records equal the host reference's."""
        for budget in (0, 1, 3, 16, AUTO_BUDGET):
            with self.subTest(budget=budget):
                sweeps, launches = [], []
                self._simulate(tray(), "patch", budget, 64, sweeps, launches)
                sweep, meta = launches[-1]
                solver = sweep.solver
                host = {name: array.numpy() for name, array in sweep.units.items()}
                counts = solver.mf_constraint_count.numpy()
                ends = solver.mf_contact_rows_end.numpy()
                metas = meta.numpy().reshape(solver.world_count, solver.mf_max_constraints, 4)
                for world in range(solver.world_count):
                    want = reference_units(
                        metas[world],
                        int(counts[world]),
                        int(ends[world]),
                        budget,
                        solver.max_world_dofs,
                        sweep.extra,
                        sweep.colors,
                    )
                    units = int(host["unit_count"][world])
                    self.assertEqual(int(host["unit_budget"][world]), want["budget"])
                    self.assertEqual(units, want["unit_count"])
                    self.assertGreater(units, 0)
                    colors = int(host["color_count"][world])
                    self.assertEqual(colors, want["color_count"])
                    np.testing.assert_array_equal(
                        host["color_offset"][world, : colors + 1], want["color_offset"][: colors + 1]
                    )
                    np.testing.assert_array_equal(host["color_slot"][world, :units], want["color_slot"])
                    self.assertEqual(int(host["split_count"][world]), len(want["splits"]))
                    for s, (dof, copies, first) in enumerate(want["splits"]):
                        self.assertEqual(int(host["split_dof"][world, s]), dof)
                        self.assertEqual(int(host["split_copies"][world, s]), copies)
                        self.assertEqual(int(host["split_first"][world, s]), first)
                        np.testing.assert_array_equal(host["split_units"][world, s], want["split_units"][s])
                    self.assertEqual(int(host["extra_copies"][world]), want["extra_copies"])
                    self.assertEqual(int(host["color_overflow"][world]), want["overflow"])
                if budget in (1, 3, AUTO_BUDGET):
                    self.assertGreater(int(host["split_count"].max()), 0, "the budget splits the tray")
                if budget == AUTO_BUDGET:
                    # Twice the mean degree splits the tray and no box.
                    self.assertEqual(int(host["split_count"].max()), 1)

    def test_a_color_never_shares_a_body_copy(self):
        """No two units of a color touch the same body copy, and a patch's units never share a color."""
        launches = []
        # Three units per copy: the boxes' four-contact patches straddle their copies.
        self._simulate(tray(), "patch", 3, 64, None, launches)
        sweep, meta = launches[-1]
        host = {name: array.numpy() for name, array in sweep.units.items()}
        parents = meta.numpy().reshape(sweep.solver.world_count, -1, 4)[:, :, 3] >> 16
        for world in range(sweep.solver.world_count):
            colors = int(host["color_count"][world])
            offsets = host["color_offset"][world]
            for color in range(colors):
                records = host["color_slot"][world, offsets[color] : offsets[color + 1]]
                copies = [offset for record in records for offset in record[2:] if offset >= 0]
                self.assertEqual(len(copies), len(set(copies)), f"color {color} shares a body copy")
                rows = {int(record[0]) for record in records}
                for row in rows:
                    ring, hop = int(parents[world, row]), 0
                    while ring >= 0 and ring != row and hop < 64:
                        self.assertNotIn(ring, rows, f"color {color} holds two contacts of one patch")
                        ring, hop = int(parents[world, ring]), hop + 1

    def test_colors_keep_the_row_order(self):
        """Units that share a body copy run in row order: the earlier one takes the lower color."""
        for budget in (0, 3):
            launches = []
            self._simulate(tray(), "patch", budget, 64, None, launches)
            sweep, _ = launches[-1]
            host = {name: array.numpy() for name, array in sweep.units.items()}
            for world in range(sweep.solver.world_count):
                units = int(host["unit_count"][world])
                self.assertGreater(units, 0)
                last = {}
                for unit in range(units):
                    color = int(host["unit_color"][world, unit])
                    for offset in {int(host["unit_offset_a"][world, unit]), int(host["unit_offset_b"][world, unit])}:
                        if offset < 0:
                            continue
                        if offset in last:
                            self.assertGreater(color, last[offset], f"B={budget} unit {unit} on copy {offset}")
                        last[offset] = color

    def test_threads_do_not_change_the_result(self):
        """A color's units commute: 32, 64 and 128 threads per world, and a cluster of two blocks per world, give the
        same trajectory bitwise, and so does a repeat."""
        for budget in (0, 3):
            runs = [self._simulate(tray(), "patch", budget, threads) for threads in (32, 64, 128)]
            runs.append(self._simulate(tray(), "patch", budget, 64))
            if self.device.arch >= 90:
                runs.append(self._simulate(tray(), "patch", budget, 32, cluster=2))
            for run in runs[1:]:
                for name in ("body_q", "body_qd"):
                    np.testing.assert_array_equal(
                        getattr(run, name), getattr(runs[0], name), err_msg=f"B={budget} {name}"
                    )

    def test_stats_count_solves_iterations_and_colors(self):
        sweeps = []
        self._simulate(tray(), "point", 4, 64, sweeps)
        stats = sweeps[-1].stats()
        self.assertEqual(stats["budget"], 4)
        self.assertGreater(stats["solves"], 0)
        self.assertGreaterEqual(stats["iterations"], stats["solves"])
        self.assertLessEqual(stats["iterations"], 32 * stats["solves"])
        world = stats["worlds"][0]
        self.assertGreater(world["splits"], 0)
        self.assertLessEqual(world["largest_color"], world["units"])


GAPS = {
    ("box_stack_patch", "coloring"): "rounding on an indeterminate support: end velocity",
    ("box_stack_patch", "data_budget"): "rounding on an indeterminate support: end velocity",
    ("box_stack_patch", "hybrid"): "split boxes: end velocity",
    ("box_stack_patch", "mass_splitting"): "split boxes: end velocity",
    ("box_stack_point", "hybrid"): "split boxes: rotation",
    ("box_stack_point", "mass_splitting"): "split boxes: normal law, penetration",
    ("tray_point", "data_budget"): "a split hub: normal law, penetration",
    ("tray_point", "hybrid"): "a split hub: normal law, penetration",
    ("tray_point", "mass_splitting"): "a split hub: normal law, penetration",
    ("tray_patch", "data_budget"): "a split hub: normal law",
    ("tray_patch", "hybrid"): "a split hub: normal law",
    ("tray_patch", "mass_splitting"): "a split hub: normal law",
}
"""Gate checks the colored sweep misses at the gate's 32 iterations (measured 2026-10-27), by (problem, mapping).

Without a split the colored sweep is the reference's row-order sweep and differs from it only in the rounding of a
unit's sums: the box stack's poses agree to 2e-10 m for ten steps. Its patches rest on a statically indeterminate
support that amplifies rounding into the end velocity: 7.6e-4 m/s from the converged reference against the gate's
5.2e-4, where the reference route itself, its time step one float32 ulp longer, ends 1.6e-3 away (2.5e-4 and 4.1e-4
for two other one-ulp steps). A split body converges like Jacobi over its copies: the tray split in two misses the
normal law by 25 to 41 percent of its bound, in three (the data budget) by 3.8 to 4.4 times; splitting the stack's
boxes (B = 4, 1) moves them off the reference.
"""


def _conformance_tests():
    """One conformance check per (scene, law, budget): the gate's laws and the distance to its converged reference."""
    groups = []
    for law in ("point", "patch"):
        for scene in (tray(), conf.scenes.box_stack()):
            physics = conf._physics(dataclasses.replace(scene, deterministic_contacts=True), law, 32)
            groups.append(conf.Group(physics, conf.REFERENCE_CUDA, ()))
    budgets = {"coloring": 0, "hybrid": 4, "mass_splitting": 1, "data_budget": AUTO_BUDGET}

    for group in groups:
        for label, budget in budgets.items():

            def test(self, group=group, budget=budget):
                route = _route(budget, 64)
                # The reference runs first, outside the colored sweep.
                conf.reference(group.physics, group.reference, self.device)
                with colored(budget, 64):
                    failures = conf._failures(conf.run_checks(group, route, self.device))
                self.assertFalse(failures, f"{group.name} through {route.name}:\n{conf._message(failures)}")

            test.__doc__ = f"Check {group.name} through the colored sweep ({label}, B={budget}) against the gate."
            gap = GAPS.get((group.name, label))
            if gap is not None:
                test.__doc__ += f" Known gap: {gap}."
                test = unittest.expectedFailure(test)
            setattr(TestColoredSweep, f"test_{group.name}_{label}_conforms", test)


_conformance_tests()


if __name__ == "__main__":
    unittest.main(verbosity=2)
