# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The colored mapping of FeatherPGS's matrix-free sweep (:mod:`.sweep_units`, ``mf_gs_kernel(mapping="colored")``).

A world's contacts become units (a normal row and its friction pair) and each unit's color is its level in row
order (one past the latest color of an earlier unit on its bodies), so no two units of a color share a body and
units that share one keep their row order. The solver gives the colored mapping the worlds of at least
``sweep_routes.COLORED_ROWS`` free-body contact rows; the tests lower that bound to one so every world takes it.
They check

- the unit builder against a host reference of the same rules (units, row-order levels, the slot records), that a
  color never shares a body or a patch, and that the colors keep the row order;
- that a world's result repeats bitwise and that the warp's worlds and the colored worlds split the batch;
- the physics: the colored mapping is the reference's row-order Gauss-Seidel sweep, so it meets the conformance
  gate's laws and its distance to the converged reference on a tray carrying a grid of boxes (a hub body) and on
  the gate's box stack.
"""

from __future__ import annotations

import contextlib
import dataclasses
import unittest
from unittest import mock

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import sweep_routes
from newton._src.solvers.feather_pgs.kernels import PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_FRICTION
from newton.solvers import SolverFeatherPGS
from newton.tests import feather_pgs_scenes as scenes
from newton.tests import test_feather_pgs_conformance as conf


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
def colored(rows: int = 1, solvers: list | None = None):
    """Every solver built inside gives the colored mapping the worlds of at least ``rows`` free-body contact rows."""
    original = SolverFeatherPGS.__init__

    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        if solvers is not None:
            solvers.append(self)

    with mock.patch.object(sweep_routes, "COLORED_ROWS", rows), mock.patch.object(SolverFeatherPGS, "__init__", init):
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
        while row + rows < end and rows < 3:
            r = row + rows
            if kind[r] != PGS_CONSTRAINT_TYPE_FRICTION or parent[r] != row:
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


class TestColoredSweep(unittest.TestCase):
    device = _cuda()

    def setUp(self):
        if self.device is None:
            self.skipTest("No CUDA device")

    def _simulate(self, scene, law: str, solvers: list | None = None, rows: int = 1):
        physics = conf._physics(scene, law, 32)
        with colored(rows, solvers):
            return conf.simulate(physics.scene, _ROUTE, self.device, options=physics.options(32))

    def _built(self, law: str = "patch") -> tuple:
        """The tray's solver after its last step, with its colored units and the packed rows they came from."""
        solvers = []
        self._simulate(tray(), law, solvers)
        solver = solvers[-1]
        self.assertIsNotNone(solver._colored_units, "the tray's worlds take the colored mapping")
        host = {name: array.numpy() for name, array in solver._colored_units.units.items()}
        metas = solver.mf_meta_packed.numpy().reshape(solver.world_count, solver.mf_max_constraints, 4)
        return solver, host, metas

    def test_units_follow_the_host_rules(self):
        """Units, row-order levels and the slot records by color equal the host reference's."""
        solver, host, metas = self._built()
        counts, ends = solver.mf_constraint_count.numpy(), solver.mf_contact_rows_end.numpy()
        for world in range(solver.world_count):
            want = reference_units(metas[world], int(counts[world]), int(ends[world]))
            units = int(host["unit_count"][world])
            self.assertGreater(units, 0)
            self.assertEqual(units, want["unit_count"])
            np.testing.assert_array_equal(host["unit_color"][world, :units], want["unit_color"])
            colors = int(host["color_count"][world])
            self.assertEqual(colors, want["color_count"])
            np.testing.assert_array_equal(host["color_offset"][world, : colors + 1], want["color_offset"])
            np.testing.assert_array_equal(host["color_slot"][world, :units], want["color_slot"])
        stats = solver._colored_units.stats()
        self.assertEqual(len(stats), solver.world_count)
        self.assertTrue(all(world["largest_color"] <= world["units"] for world in stats))

    def test_a_color_never_shares_a_body_or_a_patch(self):
        """No two units of a color touch the same body, and a patch's contacts never share a color."""
        solver, host, metas = self._built()
        parents = metas[:, :, 3] >> 16
        for world in range(solver.world_count):
            offsets = host["color_offset"][world]
            for color in range(int(host["color_count"][world])):
                records = host["color_slot"][world, offsets[color] : offsets[color + 1]]
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
        for world in range(solver.world_count):
            last = {}
            for unit in range(int(host["unit_count"][world])):
                color = int(host["unit_color"][world, unit])
                dofs = int(host["unit_dofs"][world, unit])
                for dof in {dofs >> 16, _signed(dofs, 16)}:
                    if dof < 0:
                        continue
                    if dof in last:
                        self.assertGreater(color, last[dof], f"unit {unit} on body {dof}")
                    last[dof] = color

    def test_repeats_bitwise_and_splits_the_worlds(self):
        """A repeat gives the same trajectory bitwise; above the tray's rows every world keeps the warp mapping."""
        first, second = (self._simulate(tray(), "patch") for _ in range(2))
        for name in ("body_q", "body_qd"):
            np.testing.assert_array_equal(getattr(first, name), getattr(second, name), err_msg=name)
        solvers = []
        self._simulate(tray(), "patch", solvers, rows=4096)
        self.assertIsNone(solvers[-1]._colored_units, "no world's row capacity reaches 4096 contact rows")


def _conformance_tests():
    """One conformance check per (scene, law): the gate's laws and the distance to its converged reference."""
    for law in ("point", "patch"):
        for scene in (tray(), conf.scenes.box_stack()):
            physics = conf._physics(dataclasses.replace(scene, deterministic_contacts=True), law, 32)
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
