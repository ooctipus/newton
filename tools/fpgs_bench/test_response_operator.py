# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare composable response stages with independent physical mass solves."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import response_operator as response


def make_probe(kind, size, width):
    """Bind the same physical vector contract to distinct numerical methods."""
    coefficients = "factors.data+group*factors.shape[1]"
    rhs = "rhs.data[group*rhs.shape[1]+permutation.data[$index]+1]"
    intermediate = "x[$index]"
    output = "out.data[group*out.shape[1]+permutation.data[$index]+1]"
    if kind == "cholesky":
        operator = response.CholeskySolve(size, "factor")
        stages = f"""if(lane==0){{
            {operator.forward(rhs, intermediate)}
            {operator.transpose(intermediate, output)}
        }}"""
    else:
        support = (
            response.IndexedSupport(size, "index.data", "column_count.data", "column_nodes.data", size)
            if kind == "indexed"
            else response.CompressedSupport(
                "row_offsets.data", "columns.data", "column_offsets.data", "column_entries.data", "entry_rows.data"
            )
        )
        operator = response.SparseInverseFactor("factor", support)
        stages = f"""for(int row=lane;row<{size};row+=stride){{
            {operator.forward("row", rhs)}
            x[row]=value;
        }}
#if defined(__CUDA_ARCH__)
        {"__syncthreads();" if width == 64 else "__syncwarp(mask);"}
#endif
        for(int col=lane;col<{size};col+=stride){{
            {operator.transpose("col", intermediate)}
            {output.replace("$index", "col")}=value;
        }}"""
    block = 64 if width == 64 else 128
    source = f"""
        const int group=tid/{width};
#if defined(__CUDA_ARCH__)
        const int lane=threadIdx.x%{width},stride={width};
        const unsigned mask={("0xffffu<<(threadIdx.x&16)" if width == 16 else "0xffffffffu")};
        __shared__ float scratch[{block // width * size}];
        float* x=scratch+(threadIdx.x/{width})*{size};
#else
        if(tid%{width})return;
        const int lane=0,stride=1;
        float x[{size}];
#endif
        const float* factor={coefficients};
        {stages}
    """

    @wp.func_native(source)
    def evaluate(
        tid: int,
        factors: wp.array2d[float],
        rhs: wp.array2d[float],
        permutation: wp.array[int],
        index: wp.array[int],
        column_count: wp.array[int],
        column_nodes: wp.array[int],
        row_offsets: wp.array[int],
        columns: wp.array[int],
        column_offsets: wp.array[int],
        column_entries: wp.array[int],
        entry_rows: wp.array[int],
        out: wp.array2d[float],
    ): ...

    @wp.kernel(module="unique", enable_backward=False)
    def probe(
        factors: wp.array2d[float],
        rhs: wp.array2d[float],
        permutation: wp.array[int],
        index: wp.array[int],
        column_count: wp.array[int],
        column_nodes: wp.array[int],
        row_offsets: wp.array[int],
        columns: wp.array[int],
        column_offsets: wp.array[int],
        column_entries: wp.array[int],
        entry_rows: wp.array[int],
        out: wp.array2d[float],
    ):
        evaluate(
            wp.tid(),
            factors,
            rhs,
            permutation,
            index,
            column_count,
            column_nodes,
            row_offsets,
            columns,
            column_offsets,
            column_entries,
            entry_rows,
            out,
        )

    return probe, block


def make_case(kind, size):
    """Construct a well-conditioned mass independently of the emitted action."""
    rng = np.random.default_rng(731 + size)
    worlds = 8
    permutation = rng.permutation(size).astype(np.int32)
    factors = []
    masses = []
    lower_mask = np.tril(rng.uniform(size=(size, size)) < 0.3) | np.eye(size, dtype=bool)
    rows, columns = np.nonzero(lower_mask)
    row_offsets = np.r_[0, np.cumsum(np.bincount(rows, minlength=size))].astype(np.int32)
    column_entries = np.argsort(columns, kind="stable").astype(np.int32)
    column_offsets = np.r_[0, np.cumsum(np.bincount(columns, minlength=size))].astype(np.int32)
    index = np.full((size, size), -1, np.int32)
    index[rows, columns] = np.arange(len(rows), dtype=np.int32)
    nodes = np.zeros((size, size), np.int32)
    counts = np.diff(column_offsets)
    for col in range(size):
        nodes[col, : counts[col]] = np.flatnonzero(lower_mask[:, col])
    for _ in range(worlds):
        matrix = np.tril(rng.normal(scale=0.04, size=(size, size))) + np.eye(size) * 1.4
        if kind in ("indexed", "compressed"):
            matrix *= lower_mask
            packed = matrix[rows, columns]
            inverse = np.linalg.solve(matrix, np.eye(size))
            mass = inverse @ inverse.T
        else:
            packed = matrix.ravel()
            mass = matrix @ matrix.T
        physical_mass = np.empty_like(mass)
        physical_mass[np.ix_(permutation, permutation)] = mass
        factors.append(packed)
        masses.append(physical_mass)
    rhs = rng.normal(size=(worlds, size + 2)).astype(np.float32)
    arrays = [
        permutation,
        index.ravel(),
        counts,
        nodes.ravel(),
        row_offsets,
        columns,
        column_offsets,
        column_entries,
        rows,
    ]
    return np.asarray(factors, np.float32), rhs, arrays, np.asarray(masses)


class TestResponseOperators(unittest.TestCase):
    def check_methods(self, device):
        cases = [
            ("cholesky", 9, 16),
            ("cholesky", 6, 32),
            ("indexed", 43, 64),
            ("compressed", 43, 32),
            ("indexed", 11, 32),
            ("compressed", 64, 32),
        ]
        for kind, size, width in cases:
            with self.subTest(device=device, kind=kind, size=size):
                kernel, block = make_probe(kind, size, width)
                factors, rhs, arrays, masses = make_case(kind, size)
                inputs = [wp.array(factors, dtype=float, device=device), wp.array(rhs, dtype=float, device=device)]
                inputs += [wp.array(value, dtype=int, device=device) for value in arrays]
                out = wp.full(rhs.shape, -731.0, dtype=float, device=device)
                arguments = {
                    "kernel": kernel,
                    "dim": rhs.shape[0] * width,
                    "inputs": [*inputs, out],
                    "block_dim": block,
                    "device": device,
                }
                wp.launch(**arguments)
                if device.startswith("cuda"):
                    with wp.ScopedCapture(device=device) as capture:
                        wp.launch(**arguments)
                for scale in (1, -0.7, 0):
                    inputs[1].assign(rhs * scale)
                    if device.startswith("cuda"):
                        wp.capture_launch(capture.graph)
                    else:
                        wp.launch(**arguments)
                    expected = np.linalg.solve(masses, (rhs[:, 1:-1] * scale)[..., None])[..., 0]
                    actual = out.numpy()
                    np.testing.assert_allclose(actual[:, 1:-1], expected, rtol=3e-5, atol=3e-6)
                    np.testing.assert_array_equal(actual[:, (0, -1)], -731.0)
                    residual = np.einsum("wij,wj->wi", masses, actual[:, 1:-1]) - rhs[:, 1:-1] * scale
                    self.assertLess(np.linalg.norm(residual), 3e-5 * (1 + np.linalg.norm(rhs[:, 1:-1] * scale)))

    def test_physical_response_cpu(self):
        """Compose distinct response methods against permuted physical mass systems."""
        self.check_methods("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA unavailable")
    def test_physical_response_cuda(self):
        """Check lane isolation, dependency barriers and graph replay on GPU0."""
        self.check_methods("cuda:0")

    def test_retained_zero_diagonal_convention(self):
        """Keep a zero Cholesky diagonal's zero-output policy on each device."""
        for device in ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else []):
            with self.subTest(device=device):
                factors, rhs, arrays, _ = make_case("cholesky", 3)
                factors[:] = np.diag([1.0, 0.0, 2.0]).ravel()
                arrays[0] = np.arange(3, dtype=np.int32)
                kernel, block = make_probe("cholesky", 3, 16)
                inputs = [wp.array(factors, dtype=float, device=device), wp.array(rhs, dtype=float, device=device)]
                inputs += [wp.array(value, dtype=int, device=device) for value in arrays]
                out = wp.full(rhs.shape, -731.0, dtype=float, device=device)
                wp.launch(kernel, dim=8 * 16, inputs=[*inputs, out], block_dim=block, device=device)
                expected = rhs[:, 1:-1] * np.array([1, 0, 0.25])
                np.testing.assert_allclose(out.numpy()[:, 1:-1], expected, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
