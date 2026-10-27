# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The dense response builds against the legacy kernels they replace: parameters, and bitwise outputs on the CPU.

Every build takes the parameters of the legacy kernel or factory it replaces (:data:`.response_builds_dense.DROP_IN`).
Each build a CPU can run is launched beside its legacy kernel on the same random inputs, and every output must match
bit for bit (the CPU contracts no products, so equal operation orders give equal bits). The CUDA-only builds are
proven by their compiled SASS (offline, when ``ptxas`` is on ``PATH``) and by the conformance scenes on a GPU.
"""

import functools
import importlib
import shutil
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import kernels as legacy
from newton._src.solvers.feather_pgs import response_builds_dense as builds
from newton._src.solvers.feather_pgs import row_packets, small_cholesky
from newton._src.solvers.feather_pgs import solver_feather_pgs as solver
from newton._src.solvers.feather_pgs.compact_contact import solve_six as legacy_solve_six
from newton._src.solvers.feather_pgs.response_operator import Assembled, CholeskyFactor, CholeskySolve, DenseJY
from newton._src.solvers.feather_pgs.staging import STATIC_SHARED_BUDGETS, STATIC_SHARED_BYTES, static_budget

CPU = "cpu"


def _signature(kernel) -> list:
    """A kernel's parameters: names and Warp types, in order."""
    return [(arg.label, wp.types.type_repr(arg.type)) for arg in kernel.adj.args]


def _bits(array) -> np.ndarray:
    return np.ascontiguousarray(array.numpy()).view(np.uint32)


class Problem:
    """Random dense response data of one size group: ``arts`` articulations of ``n`` DOFs in ``worlds`` worlds of
    ``rows`` row capacity, some articulations sharing a world, some rows inactive, one articulation's mass unchanged."""

    def __init__(self, n=7, rows=6, arts=5, worlds=3, seed=0, zero_pivots=False):
        rng = np.random.default_rng(seed)
        self.rng, self.n, self.rows, self.arts, self.worlds = rng, n, rows, arts, worlds
        self.art_to_world = np.array([k % worlds for k in range(arts)], dtype=np.int32)
        self.group_to_art = rng.permutation(arts).astype(np.int32)
        self.count = rng.integers(0, rows + 1, worlds).astype(np.int32)
        self.count[0] = rows
        world_dofs = np.zeros(worlds, dtype=np.int32)
        self.offset = np.zeros(arts, dtype=np.int32)
        for art in range(arts):
            world = self.art_to_world[art]
            self.offset[art] = world_dofs[world]
            world_dofs[world] += n
        self.world_dofs = world_dofs
        self.max_world_dofs = int(world_dofs.max())
        A = rng.standard_normal((arts, n, n)).astype(np.float32)
        self.H = (A @ A.transpose(0, 2, 1) + n * np.eye(n, dtype=np.float32)).astype(np.float32)
        self.R = rng.uniform(0.0, 0.1, (arts, n)).astype(np.float32)
        L = np.tril(rng.standard_normal((arts, n, n)).astype(np.float32))
        for g in range(arts):
            L[g][np.diag_indices(n)] = rng.uniform(0.5, 2.0, n).astype(np.float32)
        if zero_pivots:
            L[0, n // 2, n // 2] = 0.0
            L[1, 0, 0] = 0.0
        self.L = L
        J = rng.standard_normal((arts, rows, n)).astype(np.float32)
        J[J > 1.5] = 0.0
        self.J = J
        self.mask = np.ones(arts, dtype=np.int32)
        self.mask[arts // 2] = 0

    def array(self, value):
        return wp.array(value, device=CPU)

    def noise(self, shape):
        """A distinctive initial content of an output, so that entries a kernel leaves alone compare too."""
        return wp.array(self.rng.standard_normal(shape).astype(np.float32), device=CPU)

    def world_groups(self):
        """Each world's articulations in order (CSR), and each articulation's group index."""
        arts_of = [np.flatnonzero(self.art_to_world == w) for w in range(self.worlds)]
        start = np.zeros(self.worlds + 1, dtype=np.int32)
        start[1:] = np.cumsum([len(a) for a in arts_of])
        order = np.concatenate(arts_of).astype(np.int32)
        return self.array(start), self.array(order), self.array(np.argsort(self.group_to_art).astype(np.int32))


def _launch_twice(case, legacy_kernel, kernel, dim, inputs, outputs, *, tiled=False, block_dim=None):
    """Launch both kernels on copies of the same inputs and outputs; compare every output (and every input array,
    which a kernel may scratch) bit for bit."""
    results = []
    for k in (legacy_kernel, kernel):
        ins = [wp.clone(a) if isinstance(a, wp.array) else a for a in inputs]
        outs = [wp.clone(a) if isinstance(a, wp.array) else a for a in outputs]
        if tiled:
            wp.launch_tiled(k, dim=dim, inputs=ins, outputs=outs, block_dim=block_dim, device=CPU)
        else:
            wp.launch(k, dim=dim, inputs=ins, outputs=outs, device=CPU)
        results.append([a for a in (*ins, *outs) if isinstance(a, wp.array)])
    for index, (a, b) in enumerate(zip(*results, strict=True)):
        case.assertTrue(np.array_equal(_bits(a), _bits(b)), f"argument {index} differs")


class TestDropInParameters(unittest.TestCase):
    """Every replacement takes the parameters of the legacy global it replaces."""

    def _legacy_global(self, entry):
        return getattr(importlib.import_module(f"newton._src.solvers.feather_pgs.{entry.module}"), entry.name)

    def test_table_names_every_site(self):
        sites = {site for entry in builds.DROP_IN for site in entry.site.split(",")}
        expected = {f"F{k}" for k in range(1, 8)} | {f"S{k}" for k in (*range(1, 10), 11, 12, 13, 15, 16)}
        expected |= {f"S{k}" for k in (18, 19, 20, 21, 22)} | {"A1"}
        self.assertEqual(sites, expected)
        for entry in builds.DROP_IN:
            with self.subTest(entry.name):
                self._legacy_global(entry)

    def test_kernels(self):
        for entry in builds.DROP_IN:
            if not entry.call:
                continue
            with self.subTest(entry.name):
                self.assertEqual(_signature(entry.resolve()), _signature(self._legacy_global(entry)))

    def test_factories(self):
        configurations = {
            "get_kernel": [((6,), {}), ((9,), {"geometric": True}), ((9,), {"warps_per_block": 2})],
            "_get_crba_cholesky_kernel": [((12, 0, 64), {})],
            "_get_cholesky_kernel": [((18, 0, 64), {})],
            "_get_cholesky_inverse_tiled_kernel": [((18, 0, 64), {})],
            "_get_inverse_cholesky_register_kernel": [((6, 0), {}), ((6, 0), {"lower_only": True})],
            "_get_hinv_jt_plain_kernel": [((18, 32, 0, 64), {"constraint_chunk_size": 16, "write_world": True})],
            "_get_hinv_jt_kernel": [((18, 32, 0, 64), {"constraint_chunk_size": 16, "compute_diag": True})],
            "_get_hinv_jt_fused_kernel": [((18, 32, 0, 64), {})],
            "get_grouped_hinv_jt_kernel": [
                ((9, 32), {"write_world": True, "write_group": False, "compute_diag": True})
            ],
            "_get_paired_hinv_jt_kernel": [
                ((17, 6, 32, 23, "cpu"), {}),
                ((17, 6, 32, 23, 120), {"factor_coordinates": True}),
            ],
            "_get_delassus_kernel": [((18, 32, 0), {})],
        }
        for entry in builds.DROP_IN:
            if entry.call or entry.name == "solve_six":
                continue
            legacy_factory = self._legacy_global(entry)
            for args, kwargs in configurations[entry.name]:
                with self.subTest(entry.name, args=args, kwargs=kwargs):
                    expected = _signature(legacy_factory(*args, **kwargs))
                    self.assertEqual(_signature(entry.resolve()(*args, **kwargs)), expected)

    def test_solve_six(self):
        def types(function):
            return [(name, wp.types.type_repr(t)) for name, t in function.input_types.items()]

        self.assertEqual(types(builds.solve_six), types(legacy_solve_six))


class TestBitwiseCPU(unittest.TestCase):
    """Each CPU-runnable build against its legacy kernel, on the same random inputs."""

    def test_factor_articulations(self):
        p = Problem()
        inputs = [p.array(p.H), p.array(p.R), p.array(p.group_to_art), p.array(p.mask), p.n]
        kernel = builds.factor_articulations_kernel()
        _launch_twice(self, legacy.cholesky_loop, kernel, p.arts, inputs, [p.noise(p.H.shape)])

    def test_factor_coordinates(self):
        p = Problem()
        inputs = [p.array(p.H), p.array(p.R), p.array(p.group_to_art), p.array(p.mask), p.n]
        kernel = builds.factor_coordinates_kernel()
        _launch_twice(self, legacy.factor_diagonal_mass, kernel, p.arts * p.n, inputs, [p.noise(p.H.shape)])

    def _rows_inputs(self, p, write_world, owner=None, restitution=None):
        inputs = [
            p.array(p.L),
            p.array(p.J),
            p.array(p.group_to_art),
            p.array(p.art_to_world),
            p.array(p.offset),
            p.array(p.count),
        ]
        if owner is not None:
            inputs += [p.array(owner), p.array(restitution)]
        inputs += [p.n, p.rows, p.arts, write_world]
        world = (p.worlds, p.rows, p.max_world_dofs)
        return inputs, [p.noise(p.J.shape), p.noise(world), p.noise(world)]

    def test_response_rows(self):
        for write_world in (0, 1):
            for zero_pivots in (False, True):
                with self.subTest(write_world=write_world, zero_pivots=zero_pivots):
                    p = Problem(zero_pivots=zero_pivots)
                    inputs, outputs = self._rows_inputs(p, write_world)
                    kernel = builds.response_rows_kernel()
                    _launch_twice(self, legacy.hinv_jt_par_row, kernel, p.arts * p.rows, inputs, outputs)

    def test_response_lanes(self):
        for write_world in (0, 1):
            with self.subTest(write_world=write_world):
                p = Problem(zero_pivots=True, rows=40)
                owner = np.array([0, 1, 0], dtype=np.int32)
                restitution = p.rng.uniform(-1.0, 1.0, (p.worlds, p.rows)).astype(np.float32)
                inputs, outputs = self._rows_inputs(p, write_world, owner, restitution)
                kernel = builds.response_lanes_kernel()
                _launch_twice(self, legacy.hinv_jt_par_row_contact_fallback, kernel, p.arts * 32, inputs, outputs)

    def test_response_diagonal(self):
        for flags in ((0, 0, 0), (1, 0, 1), (1, 1, 1), (0, 1, 0)):
            with self.subTest(flags=flags):
                p = Problem()
                L = np.zeros_like(p.L)
                for g in range(p.arts):
                    L[g][np.diag_indices(p.n)] = np.diagonal(p.L[g])
                inputs, outputs = self._rows_inputs(p, 0)
                inputs = [p.array(L), *inputs[1:9], *flags]
                outputs.append(p.noise((p.arts, p.rows)))
                kernel = builds.response_diagonal_kernel()
                _launch_twice(self, legacy.hinv_jt_diagonal, kernel, p.arts * p.rows, inputs, outputs)

    def test_gather(self):
        p = Problem()
        Y = p.rng.standard_normal(p.J.shape).astype(np.float32)
        inputs = [p.array(p.group_to_art), p.array(p.art_to_world), p.array(p.offset), p.array(p.count)]
        inputs += [p.array(p.J), p.array(Y), p.n, p.rows, p.arts]
        world = (p.worlds, p.rows, p.max_world_dofs)
        kernel = builds.gather_coordinates_kernel()
        outputs = [p.noise(world), p.noise(world)]
        _launch_twice(self, legacy.gather_JY_to_world, kernel, p.arts * p.rows * p.n, inputs, outputs)

    def _JY(self, p):
        Y = p.rng.standard_normal(p.J.shape).astype(np.float32)
        Y[p.J == 0.0] = 0.0
        return p.array(p.J), p.array(Y)

    def test_delassus_entries(self):
        p = Problem()
        J, Y = self._JY(p)
        inputs = [J, Y, p.array(p.group_to_art), p.array(p.art_to_world), p.array(p.count), p.n, p.rows, p.arts]
        outputs = [p.noise((p.worlds, p.rows, p.rows)), p.noise((p.worlds, p.rows))]
        kernel = builds.delassus_entries_kernel()
        _launch_twice(self, legacy.delassus_par_row_col, kernel, p.arts * p.rows * p.rows, inputs, outputs)

    def test_diagonal_rows(self):
        for skip in (0, 3):
            with self.subTest(skip_rows_le=skip):
                p = Problem()
                J, Y = self._JY(p)
                mf = p.array(np.array([0, 2, 0], dtype=np.int32))
                inputs = [J, Y, p.array(p.group_to_art), p.array(p.art_to_world), p.array(p.count)]
                inputs += [p.n, p.rows, p.arts, mf, skip]
                kernel = builds.diagonal_rows_kernel()
                outputs = [p.noise((p.worlds, p.rows))]
                _launch_twice(self, legacy.diag_from_JY_par_art, kernel, p.arts * p.rows, inputs, outputs)

    def test_diagonal_groups(self):
        for skip in (0, 4):
            with self.subTest(skip_rows_le=skip):
                p = Problem()
                start, order, group_of = p.world_groups()
                diag = p.rng.standard_normal((p.arts, p.rows)).astype(np.float32)
                diag[0, 1] = 0.0
                mf = p.array(np.array([0, 0, 3], dtype=np.int32))
                inputs = [p.array(diag), start, order, group_of, p.array(p.count), p.rows, mf, skip]
                kernel = builds.diagonal_groups_kernel()
                outputs = [p.noise((p.worlds, p.rows))]
                _launch_twice(self, legacy.accumulate_group_diag_worlds, kernel, p.worlds * p.rows, inputs, outputs)

    def test_diagonal_world_rows(self):
        p = Problem()
        world = (p.worlds, p.rows, p.max_world_dofs)
        Jw = p.array(p.rng.standard_normal(world).astype(np.float32))
        Yw = p.array(p.rng.standard_normal(world).astype(np.float32))
        owner = p.array(np.array([0, 2, 0], dtype=np.int32))
        inputs = [p.array(p.count), owner, p.array(p.world_dofs), Jw, Yw, p.rows]
        kernel = builds.diagonal_world_rows_kernel()
        outputs = [p.noise((p.worlds, p.rows))]
        _launch_twice(self, legacy.diag_from_JY_world, kernel, p.worlds * p.rows, inputs, outputs)

    def test_cfm_worlds(self):
        for skip in (0, 4):
            with self.subTest(skip_rows_le=skip):
                p = Problem()
                cfm = p.array(p.rng.uniform(0.0, 1e-3, (p.worlds, p.rows)).astype(np.float32))
                mf = p.array(np.array([0, 0, 1], dtype=np.int32))
                inputs = [p.array(p.count), cfm, mf, skip]
                outputs = [p.noise((p.worlds, p.rows))]
                kernel = builds.cfm_worlds_kernel()
                _launch_twice(self, legacy.finalize_world_diag_cfm, kernel, p.worlds, inputs, outputs)

    def test_apply_coordinates(self):
        p = Problem()
        Y = p.rng.standard_normal(p.J.shape).astype(np.float32)
        impulses = p.rng.standard_normal((p.worlds, p.rows)).astype(np.float32)
        impulses[0, 0] = 0.0
        dof_start = p.array((np.arange(p.arts) * p.n).astype(np.int32))
        v_hat = p.array(p.rng.standard_normal(p.arts * p.n).astype(np.float32))
        inputs = [p.array(p.group_to_art), p.array(p.art_to_world), dof_start, p.n, p.arts]
        inputs += [p.array(p.count), p.rows, p.array(Y), p.array(impulses), v_hat]
        kernel = builds.apply_coordinates_kernel()
        outputs = [p.noise(p.arts * p.n)]
        _launch_twice(self, legacy.apply_impulses_world_par_dof, kernel, p.arts * p.n, inputs, outputs)

    def _queue(self, p, general):
        queue = row_packets.QueueData()
        queue.general_count = p.array(np.array([len(general)], dtype=np.int32))
        queue.general = p.array(np.array(general, dtype=np.int32))
        queue.count = p.array(p.count)
        return queue

    def test_response_queue(self):
        p = Problem(zero_pivots=True)
        start, order, group_of = p.world_groups()
        queue = self._queue(p, [2, 0])
        world = (p.worlds, p.rows, p.max_world_dofs)
        for workers in (1, 2):
            with self.subTest(workers=workers):
                inputs = [workers, queue, start, order, group_of, p.array(p.offset), p.n, p.array(p.L), p.array(p.J)]
                outputs = [p.noise(p.J.shape), p.noise(world), p.noise(world)]
                kernel = builds.response_queue_kernel()
                legacy_kernel = row_packets.general_response
                _launch_twice(self, legacy_kernel, kernel, [workers], inputs, outputs, tiled=True, block_dim=32)

    def test_diagonal_queue(self):
        p = Problem()
        queue = self._queue(p, [1, 2])
        world = (p.worlds, p.rows, p.max_world_dofs)
        cfm = p.array(p.rng.uniform(0.0, 1e-3, (p.worlds, p.rows)).astype(np.float32))
        initial = p.noise((p.worlds, p.rows))
        Jw = p.array(p.rng.standard_normal(world).astype(np.float32))
        Yw = p.array(p.rng.standard_normal(world).astype(np.float32))
        diagonals = []
        for kernel in (row_packets.general_diag, builds.diagonal_queue_kernel()):
            prefix = row_packets.PrefixData()
            prefix.diag, prefix.row_cfm = wp.clone(initial), cfm
            inputs = [2, prefix, queue, p.array(p.world_dofs), Jw, Yw]
            wp.launch_tiled(kernel, dim=[2], inputs=inputs, block_dim=32, device=CPU)
            diagonals.append(prefix.diag)
        self.assertTrue(np.array_equal(_bits(diagonals[0]), _bits(diagonals[1])))

    def test_solve_six(self):
        vec6 = wp.types.vector(length=6, dtype=wp.float32)

        def respond(function):
            def kernel(factor: wp.array3d[float], jacobian: wp.array[vec6], out: wp.array[vec6]):
                row = wp.tid()
                out[row] = function(factor, row % factor.shape[0], jacobian[row])

            kernel.__name__ = kernel.__qualname__ = f"solve_six_probe_{function.key}"
            return wp.kernel(kernel, module="unique")

        p = Problem(n=6, zero_pivots=True)
        jacobian = p.rng.standard_normal((9, 6)).astype(np.float32)
        jacobian[3] = 0.0
        jacobian[4, :5] = 0.0
        responses = []
        for function in (legacy_solve_six, builds.solve_six):
            out = wp.zeros(9, dtype=vec6, device=CPU)
            inputs = [p.array(p.L), wp.array(jacobian, dtype=vec6, device=CPU), out]
            wp.launch(respond(function), 9, inputs=inputs, device=CPU)
            responses.append(out)
        self.assertTrue(np.array_equal(_bits(responses[0]), _bits(responses[1])))

    def test_response_pairs(self):
        P, S, M, D, pairs = 5, 6, 4, 11, 3
        for factor_coordinates in (False, True):
            with self.subTest(factor_coordinates=factor_coordinates):
                rng = np.random.default_rng(3)
                primary = Problem(n=P, rows=M, arts=pairs, worlds=pairs, seed=1)
                secondary = Problem(n=S, rows=M, arts=pairs, worlds=pairs, seed=2)
                array = functools.partial(wp.array, device=CPU)
                inputs = [
                    array(rng.standard_normal((pairs, P, P)).astype(np.float32)),
                    array(np.tril(rng.standard_normal((pairs, P, P)).astype(np.float32))),
                    array(primary.J),
                    array(np.arange(pairs, dtype=np.int32)),
                    array(rng.standard_normal((pairs, S, S)).astype(np.float32)),
                    array(np.tril(rng.standard_normal((pairs, S, S)).astype(np.float32))),
                    array(secondary.J),
                    array(np.arange(pairs, 2 * pairs, dtype=np.int32)),
                    array(np.array([2, 0, 1], dtype=np.int32)),
                    array(np.arange(2 * pairs, dtype=np.int32) % pairs),
                    array(np.array([0] * pairs + [P] * pairs, dtype=np.int32)),
                    array(np.array([M, 2, 0], dtype=np.int32)),
                    array(np.array([0, 3, 0], dtype=np.int32)),
                    array(rng.uniform(0.0, 1e-3, (pairs, M)).astype(np.float32)),
                ]
                outputs = [primary.noise((pairs, M, D)), primary.noise((pairs, M, D)), primary.noise((pairs, M))]
                args, options = (P, S, M, D, "cpu"), {"factor_coordinates": factor_coordinates}
                legacy_kernel = solver._get_paired_hinv_jt_kernel(*args, **options)
                kernel = builds.response_pairs_kernel(*args, **options)
                _launch_twice(self, legacy_kernel, kernel, [pairs], inputs, outputs, tiled=True, block_dim=128)

    def test_tiled_kernels(self):
        """The tiled builds call the legacy tile primitives with the legacy shapes: equal bits on the CPU too."""
        p = Problem(n=8, rows=8, arts=3, worlds=3)
        identity = p.array(np.eye(p.n, dtype=np.float32))
        H, R, g2a, mask = p.array(p.H), p.array(p.R), p.array(p.group_to_art), p.array(p.mask)
        tiled = {"tiled": True, "block_dim": 64}
        with self.subTest("factor"):
            legacy_kernel = solver._get_cholesky_kernel(p.n, 0, 64)
            kernel = builds.factor_tiles_kernel(p.n, 0, 64)
            _launch_twice(self, legacy_kernel, kernel, [p.arts], [H, R, g2a, mask], [p.noise(p.H.shape)], **tiled)
        with self.subTest("factor inverse"):
            legacy_kernel = solver._get_cholesky_inverse_tiled_kernel(p.n, 0, 64)
            kernel = builds.factor_tiles_kernel(p.n, 0, 64, inverse=True)
            outputs = [p.noise(p.H.shape), p.noise(p.H.shape)]
            _launch_twice(self, legacy_kernel, kernel, [p.arts], [H, R, identity, g2a, mask], outputs, **tiled)
        L = np.linalg.cholesky(p.H.astype(np.float64)).astype(np.float32)
        rows = [p.array(L), p.array(p.J), g2a, p.array(p.art_to_world), p.array(p.offset), p.array(p.count)]
        world = (p.worlds, p.rows, p.max_world_dofs)
        for chunk, compute_diag in ((4, False), (8, True), (3, True)):
            with self.subTest("response", chunk=chunk, compute_diag=compute_diag):
                options = {"constraint_chunk_size": chunk, "write_world": True}
                if compute_diag:
                    legacy_kernel = solver._get_hinv_jt_kernel(p.n, p.rows, 0, 64, compute_diag=True, **options)
                else:
                    legacy_kernel = solver._get_hinv_jt_plain_kernel(p.n, p.rows, 0, 64, **options)
                kernel = builds.response_tiles_kernel(p.n, p.rows, 0, 64, compute_diag=compute_diag, **options)
                outputs = [p.noise(p.J.shape), p.noise(world), p.noise(world)]
                if compute_diag:
                    outputs.append(p.noise((p.arts, p.rows)))
                chunks = -(-p.rows // chunk)
                _launch_twice(self, legacy_kernel, kernel, [p.arts, chunks], rows, outputs, **tiled)
        with self.subTest("fused"):
            cfm = p.array(p.rng.uniform(0.0, 1e-3, (p.worlds, p.rows)).astype(np.float32))
            fused = [p.array(L), p.array(p.J), g2a, p.array(p.art_to_world), p.array(p.count), cfm]
            outputs = [p.noise((p.worlds, p.rows, p.rows)), p.noise((p.worlds, p.rows)), p.noise(p.J.shape)]
            legacy_kernel = solver._get_hinv_jt_fused_kernel(p.n, p.rows, 0, 64)
            kernel = builds.delassus_tiles_kernel(p.n, p.rows, 0, 64)
            _launch_twice(self, legacy_kernel, kernel, [p.arts], fused, outputs, **tiled)


class TestKit(unittest.TestCase):
    """The small dense kit's text."""

    def test_factor_matches_numpy(self):
        """The kit's column Cholesky factors a positive definite matrix (float32, against float64)."""
        p = Problem(n=9)
        L = wp.zeros(p.H.shape, dtype=float, device=CPU)
        ones = p.array(np.ones(p.arts, dtype=np.int32))
        inputs = [p.array(p.H), p.array(p.R), p.array(p.group_to_art), ones, p.n]
        wp.launch(builds.factor_articulations_kernel(), p.arts, inputs=inputs, outputs=[L], device=CPU)
        A = p.H.astype(np.float64) + np.einsum("gi,ij->gij", p.R.astype(np.float64), np.eye(p.n))
        np.testing.assert_allclose(np.tril(L.numpy()), np.linalg.cholesky(A), rtol=1e-4, atol=1e-5)

    def test_factor_text(self):
        factor = CholeskyFactor(3, lambda r, c: f"f[{r} * 3 + {c}]", unroll=True)
        text = factor.factor()
        self.assertEqual(text.count("#pragma unroll"), 4)
        self.assertIn("diagonal = sqrtf(diagonal);", text)
        self.assertIn("f[row * 3 + col] = value * inverse_diagonal;", text)
        self.assertEqual(factor.pivot("d"), "f[d * 3 + d] = sqrtf(f[d * 3 + d]);")

    def test_solve_text_keeps_its_default(self):
        """A matrix operand reads the factor as the storage form does, and the defaults add no unroll directive."""

        def x(i):
            return f"x[{i}]"

        def y(i):
            return f"y[{i}]"

        storage = CholeskySolve(6, "L")
        accessor = CholeskySolve(6, lambda r, c: f"(L)[{r}*6+{c}]")
        self.assertEqual(storage.forward(x, y), accessor.forward(x, y))
        self.assertEqual(storage.transpose(y, y), accessor.transpose(y, y))
        self.assertNotIn("#pragma", storage.forward(x, y) + storage.transpose(y, y))
        unrolled = CholeskySolve(6, "L", unroll=True)
        self.assertEqual(unrolled.forward(x, y).count("#pragma unroll"), 2)
        self.assertEqual(unrolled.transpose(y, y).count("#pragma unroll"), 2)


class TestDelassusStaging(unittest.TestCase):
    """The streaming Delassus's chunk is its staging's: the largest that fits the device's static budget."""

    def test_chunk(self):
        for D, M, budget in ((18, 32, 49152), (176, 32, 49152), (97, 64, 49152), (97, 64, 101376), (300, 128, 101376)):
            with self.subTest(D=D, M=M, budget=budget):
                chunk = builds.delassus_stream_chunk(D, M, budget)
                self.assertLessEqual(chunk, M)
                self.assertTrue(builds.delassus_stream_staging(D, chunk).fits(budget))
                if chunk < M:
                    self.assertFalse(builds.delassus_stream_staging(D, chunk + 1).fits(budget))
        # Where the legacy rule kept every row (2 M D 4 <= 45000 bytes), so does the staging, offline too.
        for D, M in ((18, 32), (43, 64), (12, 128)):
            self.assertLessEqual(2 * M * D * 4, 45000)
            self.assertEqual(builds.delassus_stream_chunk(D, M, STATIC_SHARED_BYTES), M)
        with self.assertRaises(ValueError):
            builds.delassus_stream_chunk(7000, 8, STATIC_SHARED_BYTES)

    def test_budget(self):
        self.assertEqual(static_budget(12345), STATIC_SHARED_BYTES)  # no device of the architecture: offline
        STATIC_SHARED_BUDGETS[12345] = 81724
        try:
            self.assertEqual(static_budget(12345), 81724)
        finally:
            del STATIC_SHARED_BUDGETS[12345]
        # 176 DOFs and 32 rows: the 45,000-byte rule took 64 rows (90,112 bytes); the staging keeps the 32 rows.
        self.assertIn("chunk32", builds.delassus_stream_kernel(176, 32, 12345).key)


class TestRegistry(unittest.TestCase):
    """The operators' builds by stage and mapping."""

    def test_dense(self):
        build = DenseJY.build("factor", "articulations")
        self.assertEqual((build.stage, build.mapping, build.tiled), ("factor", "articulations", False))
        self.assertIs(build.kernel, builds.factor_articulations_kernel())
        tiles = DenseJY.build(
            "response", "tiles", n_dofs=18, max_constraints=32, device_arch=0, tile_threads=64, compute_diag=True
        )
        self.assertEqual((tiles.block_dim, tiles.tiled, tiles.grid), (64, True, ("articulations", "row chunks")))
        warps = DenseJY.build("factor", "warps", size=9, warps_per_block=2)
        self.assertEqual((warps.block_dim, warps.tiled), (64, False))
        for stage, _mapping in (*builds.DENSE_JY_BUILDS, *builds.ASSEMBLED_BUILDS):
            self.assertIn(stage, builds.STAGES)

    def test_assembled(self):
        build = Assembled.build("delassus", "entries")
        self.assertIs(build.kernel, builds.delassus_entries_kernel())
        stream = Assembled.build("delassus", "stream", n_dofs=18, max_constraints=32, device_arch=0)
        self.assertEqual(stream.block_dim, builds.DELASSUS_THREADS)
        with self.assertRaises(KeyError):
            Assembled.build("factor", "articulations")


@unittest.skipUnless(shutil.which("ptxas") and shutil.which("cuobjdump"), "needs ptxas and cuobjdump on PATH")
class TestOfflineCompile(unittest.TestCase):
    """CUDA-only builds compile to the SASS of the kernels they replace (sm_120, offline)."""

    def test_sass(self):
        try:
            from tools.fpgs_bench import kernel_sass  # noqa: PLC0415
        except ImportError:
            self.skipTest("needs the repository's tools on the path")
        cases = (
            (
                solver._get_inverse_cholesky_register_kernel(6, 0, lower_only=True),
                builds.inverse_lanes_kernel(6, 0, lower_only=True),
                32,
            ),
            (small_cholesky.get_kernel(6), builds.factor_warps_kernel(6), 128),
        )
        for legacy_kernel, kernel, block_dim in cases:
            with self.subTest(kernel.key):
                a = kernel_sass.compile_kernel(legacy_kernel, arch=120, block_dim=block_dim)
                b = kernel_sass.compile_kernel(kernel, arch=120, block_dim=block_dim)
                self.assertEqual(a["sass"], b["sass"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
