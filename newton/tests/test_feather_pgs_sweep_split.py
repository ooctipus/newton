# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Split mode's loop sweeps as configurations of the sweep emitter (CPU checks).

Each configuration replaces a module-level Warp kernel of the split route: it takes the same parameters in the same
order and the same launch, one thread per world. The sources keep the arithmetic order of the sweeps they replace, and
on the CPU, where neither build fuses a multiply-add, the kernels reproduce the Warp kernels' results bit for bit on
random worlds: contacts, patch rings with anchors, limits and joint rows; free bodies with speed rows; ring and anchor
units, delayed friction, regularization and a continued solve. With ``ptxas`` on ``PATH`` the kernels also assemble
offline for CUDA.
"""

import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import sweep_routes_split as routes
from newton._src.solvers.feather_pgs.contact_unit import RING_BLOCK_SIZE
from newton._src.solvers.feather_pgs.kernels import BodyFactors
from newton._src.solvers.feather_pgs.response_operator import Assembled, FreeBodyZ, Lanes, Velocity
from newton._src.solvers.feather_pgs.sweep import FRICTION, RowContext, RowLaws, row_update

CONTACT, DRIVE, LIMIT, VLIMIT, MIMIC, CONNECT, TORSION = 0, 1, 3, 4, 5, 6, 7


def _signature(kernel):
    return [(argument.label, wp.types.type_repr(argument.type)) for argument in kernel.adj.args]


def _balanced(test, text):
    test.assertEqual(text.count("{"), text.count("}"))
    test.assertEqual(text.count("("), text.count(")"))
    test.assertNotIn("$", text)


class TestSweepRoutesSplitABI(unittest.TestCase):
    """The loop kernels' operands."""

    def test_unit_kernels_add_the_unit_operands(self):
        plain, unit = _signature(routes.loop_kernel()), _signature(routes.loop_unit_kernel())
        self.assertLess(len(plain), len(unit))
        self.assertEqual([name for name, _ in plain if name not in dict(unit)], [])
        mf, mf_unit = _signature(routes.mf_loop_kernel()), _signature(routes.mf_loop_unit_kernel())
        self.assertEqual([name for name, _ in mf if name not in dict(mf_unit)], [])


class TestSweepRoutesSplitSources(unittest.TestCase):
    """One thread sweeps a world in place, in the replaced sweeps' order."""

    def setUp(self):
        self.sources = {
            "loop": routes.loop_source(),
            "loop_unit": routes.loop_source(contact_units=True),
            "mf_loop": routes.mf_loop_source(),
            "mf_loop_unit": routes.mf_loop_source(contact_units=True),
        }

    def test_one_thread_per_world_in_place(self):
        for name, (text, staging) in self.sources.items():
            with self.subTest(sweep=name):
                _balanced(self, text)
                self.assertEqual(staging.bytes, 0)
                self.assertIn("const int world = tid;", text)
                for lane_group in ("SYNC", "SUM(", "READ(", "BALLOT(", "__shared__"):
                    self.assertNotIn(lane_group, text)

    def test_assembled_rows_sum_from_the_rhs_and_read_row_i(self):
        text = self.sources["loop"][0]
        start = text.index("float jv = world_rhs.data[off1 + i];")
        self.assertLess(start, text.index("const float c = world_C.data[jv_row + j];"))
        self.assertLess(start, text.index("jv += c * lam[j];"))
        self.assertIn("float residual = jv;", text)
        # The pair's coupling is row i's entry C(i, s).
        self.assertIn("float cross = world_C.data[off2 + i * capacity + sib];", text)

    def test_free_bodies_take_the_loop_order(self):
        text = self.sources["mf_loop"][0]
        self.assertIn(
            "for (int k = 0; k < 6; ++k) jv += mf_Z_a.data[mf6_base + i * 6 + k] * v_out.data[dof_a + k];", text
        )
        self.assertIn("float residual = jv + mf_rhs.data[c_off + i];", text)
        # The friction step sums the pair's first tangent again, per coordinate from its right-hand side.
        pair = text[text.index("if (radius > 0.0f) {") :]
        self.assertIn("float residual = mf_rhs.data[c_off + i];", pair)
        self.assertIn("ProjectionLaw::speed_row(", text)

    def test_units_bind_the_operator(self):
        dense, free = self.sources["loop_unit"][0], self.sources["mf_loop_unit"][0]
        for text in (dense, free):
            for call in ("ContactUnit::solve(", "ContactUnit::ring(", "ContactUnit::latch(", "SORT_RING(i);"):
                self.assertIn(call, text)
        self.assertIn("return fmaxf(world_diag.data[off1 + r], entry);", dense)
        self.assertNotIn("UNIT_APPLY", dense)
        self.assertIn("UNIT_APPLY(n, new_n - o_n);", free)
        self.assertIn("if (dof_a == column_b) {", free)


class TestSplitOperators(unittest.TestCase):
    """The operators' one-thread layouts and the pair's own residual."""

    def test_assembled_one_thread(self):
        rhs = lambda row: f"b[{row}]"  # noqa: E731
        serial = Assembled(0, "n", "global", Lanes(1), impulse="x", rhs=rhs)
        self.assertTrue(serial.residual_includes_rhs)
        self.assertFalse(Assembled(64, "64", "global").residual_includes_rhs)
        self.assertIn("float s = b[sib];", serial.sibling_residual("s", "sib", "", "b[sib]"))
        self.assertEqual(serial.coupling("c", "i", "s", ""), "float c = world_C.data[off2 + i * n + s];")
        shared = Assembled(0, "n", "shared", Lanes(1), rhs=rhs)
        self.assertIn("s <= i ?", shared.coupling("c", "i", "s", ""))
        with self.assertRaisesRegex(ValueError, "right-hand side"):
            Assembled(0, "n", "global", Lanes(1))

    def test_free_body_orders(self):
        loop = FreeBodyZ(Lanes(1), velocity=Velocity("v"), order="loop")
        split = FreeBodyZ(Lanes(1), velocity=Velocity("v"))
        self.assertIn("for (int k = 0; k < 6; ++k) jv +=", loop.residual("jv", ""))
        self.assertNotIn("for (int k", split.residual("jv", ""))
        own = loop.pair_terms("i", "sib", lambda row: f"b[{row}]", "", own=True)
        self.assertTrue(own.startswith("float residual = b[i];\nfor (int k = 0; k < 6; ++k) {"))
        self.assertNotIn("float residual", loop.pair_terms("i", "sib", str, ""))
        coupling = loop.shared_coupling("e", "r", "c", ("ca", "cb"), "")
        self.assertLess(coupling.index("if (dof_a == ca)"), coupling.index("if (dof_a == cb)"))
        self.assertLess(coupling.index("if (dof_a == cb)"), coupling.index("if (dof_b == ca)"))
        with self.assertRaisesRegex(ValueError, "order"):
            FreeBodyZ(Lanes(1), order="split_loop")

    def test_the_pair_residual_hook(self):
        op = FreeBodyZ(Lanes(1), velocity=Velocity("v"), inverse=lambda row: f"inv[{row}]", order="loop")

        def update(**options):
            context = RowContext(
                row="i",
                kind="k",
                operator=op,
                load=lambda row: f"x[{row}]",
                store=lambda row, value: f"x[{row}] = {value};",
                rhs=lambda row: f"b[{row}]",
                mu=lambda row: f"mu[{row}]",
                weight=lambda row: f"w[{row}]",
                parent="parent",
                ring_load=lambda parent: f"L({parent})",
                lanes=Lanes(1),
                **options,
            )
            return row_update(context, RowLaws(), residual="jv", pad="")

        own = update(pair_residual=True)
        self.assertNotIn("float residual = b[i];", update())
        self.assertIn("float residual = b[i];", own[own.index(f"k == {FRICTION}") :])


# ---------------------------------------------------------------------------------------------------------------------
# Equivalence on the CPU: random worlds through the replaced kernels and the configurations, bit for bit
# ---------------------------------------------------------------------------------------------------------------------


def _layout(rng, rows, capacity):
    """Whole blocks of rows: point contacts, frictionless normals, patch rings (shuffled, anchors with tangent
    pairs) and joint rows; each row's kind, parent and block."""
    kinds, parents, blocks = [], [], []
    while len(kinds) < rows:
        roll, block = rng.random(), []
        if roll < 0.3:
            block = [(CONTACT, -1), (FRICTION, 0), (FRICTION, 0)]
        elif roll < 0.4:
            block = [(CONTACT, -1)]
        elif roll < 0.7:
            normals = []
            for _ in range(int(rng.integers(2, 11))):
                normals.append(len(block))
                block.append((CONTACT, -1))
                if rng.random() < 0.45:
                    block += [(FRICTION, normals[-1])] * 2
            order = rng.permutation(normals)
            for j, n in enumerate(order):
                block[n] = (CONTACT, int(order[(j + 1) % len(order)]))
        else:
            block = [(int(rng.choice([DRIVE, LIMIT, VLIMIT, MIMIC, CONNECT, TORSION])), -1)]
        base = len(kinds)
        if base + len(block) > capacity:
            break
        kinds += [kind for kind, _ in block]
        parents += [base + parent if parent >= 0 else -1 for _, parent in block]
        blocks += [len(set(blocks))] * len(block)
    return kinds, parents, blocks


def _options(rng):
    return {
        "iterations": int(rng.integers(1, 13)),
        "omega": float(rng.choice([1.0, 0.8])),
        "regularize": int(rng.integers(0, 2)),
        "friction_start": int(rng.integers(0, 4)),
        "offset": int(rng.choice([0, 0, 2])),
        "ring_units": int(rng.integers(0, 2)),
    }


def _units(rng, worlds, capacity, warm):
    state = np.zeros((worlds, capacity, 3), np.float32)
    if warm:
        state[..., 0] = np.where(rng.random((worlds, capacity)) < 0.3, rng.uniform(0.0, 2.0, (worlds, capacity)), 0.0)
    return [
        wp.array(state, device="cpu"),
        wp.zeros((worlds, capacity), dtype=wp.int32, device="cpu"),
        wp.zeros(worlds * RING_BLOCK_SIZE, dtype=wp.float32, device="cpu"),
    ]


def _dense_world(rng, capacity):
    kinds, parents, _ = _layout(rng, int(rng.integers(1, capacity - 12)), capacity)
    m = len(kinds)
    J = rng.standard_normal((m, m // 2 + 3))
    C = (J / rng.uniform(0.5, 3.0, J.shape[1])) @ J.T
    C *= 1.0 + 1e-6 * rng.standard_normal(C.shape)  # assembled in floating point: not exactly symmetric
    diag = np.diag(C) * rng.uniform(0.98, 1.05, m)
    lam = np.where(rng.random(m) < 0.5, 0.0, rng.uniform(0.0, 0.8, m)) if rng.random() < 0.5 else np.zeros(m)
    return {
        "count": m,
        "kinds": kinds,
        "parents": parents,
        "C": C,
        "diag": diag,
        "rhs": rng.uniform(-1.0, 0.4, m),
        "mu": rng.uniform(0.1, 1.0, m),
        "w": rng.uniform(0.3, 1.0, m),
        "lam": lam,
    }


def _dense_case(rng, unit):
    worlds, capacity = int(rng.integers(1, 4)), int(rng.choice([24, 40]))
    rows = [_dense_world(rng, capacity) for _ in range(worlds)]
    data = {
        "count": np.zeros(worlds, np.int32),
        "kinds": np.zeros((worlds, capacity), np.int32),
        "parents": np.full((worlds, capacity), -1, np.int32),
        "C": np.zeros((worlds, capacity, capacity), np.float32),
    }
    for name in ("diag", "rhs", "mu", "w", "lam"):
        data[name] = np.zeros((worlds, capacity), np.float32)
    for world, row in enumerate(rows):
        m = row["count"]
        data["count"][world] = m
        data["C"][world, :m, :m] = row["C"]
        for name in ("kinds", "parents", "diag", "rhs", "mu", "w", "lam"):
            data[name][world, :m] = row[name]
    options = _options(rng)
    state = rng.bit_generator.state

    def launch(kernel):
        a = {name: wp.array(value, device="cpu") for name, value in data.items()}
        inputs = [a["count"], a["diag"], a["C"], a["rhs"], a["lam"], options["iterations"], options["omega"]]
        inputs += [a["kinds"], a["parents"], a["mu"], a["w"], options["regularize"], options["friction_start"]]
        inputs.append(options["offset"])
        storage = []
        if unit:
            rng.bit_generator.state = state
            storage = _units(rng, worlds, capacity, options["offset"] > 0)
            inputs += [options["ring_units"], *storage]
        wp.launch(kernel, dim=worlds, inputs=inputs, device="cpu")
        return [a["lam"].numpy(), *(array.numpy() for array in storage)]

    return launch


def _free_body_case(rng, unit):
    worlds, capacity, bodies = int(rng.integers(1, 4)), int(rng.choice([32, 48])), int(rng.integers(2, 5))
    total = worlds * bodies
    articulation = rng.permutation(total).astype(np.int32)
    dof_start = (6 * rng.permutation(total) + 3).astype(np.int32)
    minv = [np.eye(6) * rng.uniform(0.2, 2.0) + 0.05 * (R @ R.T) for R in rng.standard_normal((total, 6, 6))]
    # Each body's held factor, H = L L^T with H^-1 = minv.
    lower = np.array([np.linalg.cholesky(np.linalg.inv(m)) for m in minv], np.float32)
    dof_factor = np.full(6 * total + 9, -1, np.int32)
    for body in range(total):
        dof_factor[dof_start[articulation[body]]] = body
    shape = (worlds, capacity)
    data = {"count": np.zeros(worlds, np.int32), "kinds": np.zeros(shape, np.int32)}
    data["parents"] = np.full(shape, -1, np.int32)
    data["body_a"], data["body_b"] = np.full(shape, -1, np.int32), np.full(shape, -1, np.int32)
    for name in ("Z_a", "Z_b"):
        data[name] = np.zeros((*shape, 6), np.float32)
    for name in ("inv", "rhs", "mu", "w", "lam"):
        data[name] = np.zeros(shape, np.float32)
    for world in range(worlds):
        own = list(range(world * bodies, (world + 1) * bodies))
        kinds, parents, blocks = _layout(rng, int(rng.integers(1, capacity - 16)), capacity - bodies)
        kinds = [kind if kind in (CONTACT, FRICTION) else CONTACT for kind in kinds]
        pairs = {}
        for block in set(blocks):
            a = int(rng.choice(own))
            pairs[block] = (a, int(rng.choice([b for b in own if b != a]))) if rng.random() < 0.6 else (a, -1)
        pair = [pairs[block] for block in blocks]
        for row, kind in enumerate(kinds):  # a patch member may list its pair in the other order
            if kind == CONTACT and parents[row] >= 0 and pair[row][1] >= 0 and rng.random() < 0.5:
                pair[row] = pair[row][::-1]
        for row, kind in enumerate(kinds):
            if kind == FRICTION:
                pair[row] = pair[parents[row]]
        for body in own:  # speed rows at the tail
            if rng.random() < 0.4:
                kinds.append(VLIMIT)
                parents.append(-1)
                pair.append((body, -1))
        m = len(kinds)
        data["count"][world] = m
        data["kinds"][world, :m], data["parents"][world, :m] = kinds, parents
        for row in range(m):
            a, b = pair[row]
            J_a = np.zeros(6) if kinds[row] == VLIMIT else rng.standard_normal(6)
            if kinds[row] == VLIMIT:
                J_a[3 * int(rng.integers(0, 2))] = 1.0
            J_b = rng.standard_normal(6) if b >= 0 else np.zeros(6)
            data["body_a"][world, row], data["body_b"][world, row] = a, b
            # Whitened rows, Z = L^-1 J^T; a speed row keeps its axis.
            Z_a = np.linalg.solve(lower[a].astype(np.float64), J_a)
            data["Z_a"][world, row] = J_a if kinds[row] == VLIMIT else Z_a
            response = float(Z_a @ Z_a)
            if b >= 0:
                data["Z_b"][world, row] = np.linalg.solve(lower[b].astype(np.float64), J_b)
                response += float(data["Z_b"][world, row].astype(np.float64) @ data["Z_b"][world, row])
            data["inv"][world, row] = 1.0 / (response + 1e-4)
            data["rhs"][world, row] = rng.uniform(0.05, 1.0) if kinds[row] == VLIMIT else rng.uniform(-1.0, 0.4)
        data["mu"][world, :m] = rng.uniform(0.1, 1.0, m)
        data["w"][world, :m] = rng.uniform(0.3, 1.0, m)
        if rng.random() < 0.5:
            data["lam"][world, :m] = np.where(rng.random(m) < 0.5, 0.0, rng.uniform(0.0, 0.8, m))
    velocity = rng.standard_normal(6 * total + 9).astype(np.float32)
    options = _options(rng)
    state = rng.bit_generator.state

    def launch(kernel):
        a = {name: wp.array(value, device="cpu") for name, value in data.items()}
        v = wp.array(velocity, device="cpu")
        factors = BodyFactors()
        factors.dof_factor = wp.array(dof_factor, device="cpu")
        factors.L = wp.array(lower, dtype=float, device="cpu")
        inputs = [a["count"], a["body_a"], a["body_b"], a["Z_a"], a["Z_b"], factors, a["inv"]]
        inputs += [a["rhs"], a["kinds"], a["parents"], a["mu"], a["w"]]
        inputs += [wp.array(articulation, device="cpu"), wp.array(dof_start, device="cpu")]
        inputs += [options["iterations"], options["omega"], options["regularize"]]
        storage = []
        if unit:
            rng.bit_generator.state = state
            storage = _units(rng, worlds, capacity, options["offset"] > 0)
            inputs += [options["friction_start"], options["offset"], options["ring_units"], *storage]
        else:
            inputs += [options["friction_start"], options["offset"]]
        wp.launch(kernel, dim=worlds, inputs=inputs, outputs=[a["lam"], v], device="cpu")
        return [a["lam"].numpy(), v.numpy(), *(array.numpy() for array in storage)]

    return launch


_KERNELS = {
    "loop": routes.loop_kernel,
    "loop_unit": routes.loop_unit_kernel,
    "mf_loop": routes.mf_loop_kernel,
    "mf_loop_unit": routes.mf_loop_unit_kernel,
}


class TestSweepRoutesSplitRuns(unittest.TestCase):
    """On the CPU every configuration sweeps random problems deterministically to finite impulses and velocities."""

    CASES = 6

    def _check(self, name, case):
        rng = np.random.default_rng(7)
        for index in range(self.CASES):
            launch = case(rng, name.endswith("_unit"))
            with self.subTest(sweep=name, case=index):
                first, second = launch(_KERNELS[name]()), launch(_KERNELS[name]())
                for a, b in zip(first, second, strict=True):
                    self.assertEqual(a.tobytes(), b.tobytes())
                    self.assertTrue(np.isfinite(a).all())

    def test_assembled_loop(self):
        self._check("loop", _dense_case)

    def test_assembled_loop_with_units(self):
        self._check("loop_unit", _dense_case)

    def test_free_body_loop(self):
        self._check("mf_loop", _free_body_case)

    def test_free_body_loop_with_units(self):
        self._check("mf_loop_unit", _free_body_case)


def _assemble(kernel, arch: int = 120) -> dict:
    """Assemble a kernel's module offline (Warp's PTX, then ``ptxas``) and return its entry's resources."""
    with tempfile.TemporaryDirectory() as directory:
        options = kernel.module.resolve_options(wp.config, block_dim=256)
        kernel.module._compile(None, directory, output_arch=arch, use_ptx=True, options=options)
        ptx = sorted(Path(directory).rglob("*.ptx"))[0]
        source = Path(directory) / "module.ptx"
        source.write_bytes(ptx.read_bytes().rstrip(b"\0"))
        result = subprocess.run(
            ["ptxas", f"-arch=sm_{arch}", "-v", str(source), "-o", str(Path(directory) / "module.cubin")],
            check=True,
            capture_output=True,
            text=True,
        )
    entry = result.stderr[result.stderr.index("_cuda_kernel_forward") :]
    return {
        "registers": int(re.search(r"Used (\d+) registers", entry).group(1)),
        "spills": int(re.search(r"(\d+) bytes spill stores", entry).group(1)),
    }


@unittest.skipUnless(shutil.which("ptxas"), "ptxas is not on PATH")
class TestSweepRoutesSplitAssembly(unittest.TestCase):
    """The one source assembles for CUDA too."""

    def test_kernels_assemble(self):
        for name, factory in _KERNELS.items():
            with self.subTest(sweep=name):
                self.assertGreater(_assemble(factory())["registers"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
