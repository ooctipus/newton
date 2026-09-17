# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exact native linear fragment controls; CUDA execution requires root's lease."""

import ctypes
import functools
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import krylov_linear


@functools.cache
def probe_kernel():
    """Wrap the actual fragment in exactly two external CTA handoffs."""
    snippet = (
        "#if defined(__CUDA_ARCH__)\n"
        + krylov_linear.NATIVE
        + r"""
    const int lane = threadIdx.x;
    for (int e=lane;e<171;e+=blockDim.x) kc_linear.gram[e]=values.data[e];
    for (int e=lane;e<18;e+=blockDim.x) {
        kc_linear.rhs[e]=values.data[171+e];
        kc_linear.eta[e]=values.data[189+e];
        kc_linear.block_start[e]=indices.data[e];
        kc_linear.block_size[e]=indices.data[18+e];
    }
    for (int e=lane;e<54;e+=blockDim.x) {
        kc_linear.projection[e]=values.data[207+e];
        kc_linear.columns[e]=indices.data[36+e];
    }
    __syncthreads();
    if (lane<32) kc_solve_warp(kc_linear, count, target, lane);
    __syncthreads();
    if (lane<18) output.data[lane]=kc_linear.delta[lane];
    if (lane==0) {
        output.data[18]=kc_linear.true_error;
        status.data[0]=kc_linear.ok;
        status.data[1]=kc_linear.reason;
        status.data[2]=kc_linear.steps;
    }
#else
    status.data[0]=-99;
#endif
"""
    )

    @wp.func_native(snippet)
    def native(
        values: wp.array[float],
        indices: wp.array[int],
        count: int,
        target: float,
        output: wp.array[float],
        status: wp.array[int],
    ):
        """No CPU numerical substitute for CUDA warp collectives."""

    @wp.kernel(enable_backward=False)
    def probe(
        values: wp.array[float],
        indices: wp.array[int],
        count: int,
        target: float,
        output: wp.array[float],
        status: wp.array[int],
    ):
        native(values, indices, count, target, output, status)

    return probe


def pack_case(gram, eta, projection, rhs, sizes):
    """Only pack inputs; the reference independently assembles the dense J."""
    count = len(rhs)
    if sum(sizes) != count or any(size not in (1, 2, 3) for size in sizes):
        raise ValueError("Invalid disjoint compact block sizes")
    values = np.zeros(261, dtype=np.float32)
    indices = np.full(90, -1, dtype=np.int32)
    for row in range(count):
        for column in range(row + 1):
            values[row * (row + 1) // 2 + column] = gram[row, column]
    values[171 : 171 + count] = rhs
    values[189 : 189 + count] = eta
    start = 0
    for size in sizes:
        for row in range(start, start + size):
            indices[row] = start
            indices[18 + row] = size
            for slot in range(size):
                values[207 + row * 3 + slot] = projection[row, start + slot]
                indices[36 + row * 3 + slot] = start + slot
        start += size
    # Use represented FP32 inputs, but independent FP64 matrix algebra.
    g = np.asarray(gram, dtype=np.float32).astype(np.float64)
    e = values[189 : 189 + count].astype(np.float64)
    p = np.asarray(projection, dtype=np.float32).astype(np.float64)
    jacobian = (np.eye(count) - p @ (np.eye(count) - e[:, None] * g)) / e[:, None]
    return values, indices, jacobian


def coupled_case(sizes):
    count = sum(sizes)
    rng = np.random.default_rng(1609 + count)
    response = rng.normal(size=(count, count))
    gram = np.eye(count) + 0.015 * response @ response.T
    eta = np.linspace(0.3, 0.7, count)
    projection = np.eye(count)
    start = 0
    for number, size in enumerate(sizes):
        if size == 3 and number % 2 == 0:
            # Sliding disk derivative, including its moving normal radius.
            unit = np.array([0.6, 0.8])
            projection[start + 1 : start + 3, start] = 0.4 * unit
            projection[start + 1 : start + 3, start + 1 : start + 3] = 0.55 * (np.eye(2) - np.outer(unit, unit))
        start += size
    rhs = rng.normal(size=count)
    return pack_case(gram, eta, projection, rhs, sizes)


class TestKrylovLinear(unittest.TestCase):
    def test_exact_shared_layout_and_warp_only_source(self):
        float_lengths = (171, 18, 18, 54, 162, 144, 72, 8, 8, 9, 8, 54, 18, 18)
        int_lengths = (54, 18, 18, 18, 1, 1, 1)
        fields = [(f"f{i}", ctypes.c_float * size) for i, size in enumerate(float_lengths)]
        fields += [(f"i{i}", ctypes.c_int * size) for i, size in enumerate(int_lengths)]
        fields.append(("true_error", ctypes.c_float))
        layout = type("Layout", (ctypes.Structure,), {"_fields_": fields})
        self.assertEqual(ctypes.sizeof(layout), krylov_linear.SCRATCH_BYTES)
        self.assertEqual(krylov_linear.SCRATCH_BYTES, 3496)
        self.assertNotIn("__syncthreads", krylov_linear.NATIVE)
        self.assertNotIn("SYNC()", krylov_linear.NATIVE)
        self.assertIn("right_basis[144]", krylov_linear.NATIVE)
        self.assertIn("const float defect = kc_action(solution) - local_rhs", krylov_linear.NATIVE)
        self.assertIs(probe_kernel(), probe_kernel())

    def test_compact_partial_groups_and_independent_jacobian(self):
        values, indices, jacobian = coupled_case([1, 2, 3, 3, 3, 3, 3])
        self.assertEqual(jacobian.shape, (18, 18))
        start = 0
        for size in (1, 2, 3, 3, 3, 3, 3):
            np.testing.assert_array_equal(indices[start : start + size], start)
            np.testing.assert_array_equal(indices[18 + start : 18 + start + size], size)
            start += size
        self.assertLess(np.linalg.cond(jacobian), 20)
        self.assertTrue(np.all(np.isfinite(np.linalg.solve(jacobian, values[171:189]))))


@unittest.skipUnless(wp.is_cuda_available(), "Root owns the CUDA lease")
class TestKrylovLinearCUDA(unittest.TestCase):
    def run_case(self, packed, block, *, target=None, count=None):
        values, indices, jacobian = packed
        count = len(jacobian) if count is None else count
        if target is None:
            target = 2e-5 * max(1.0, np.linalg.norm(values[171 : 171 + count]))
        device = wp.get_cuda_devices()[0]
        output = wp.full(19, -17.0, dtype=float, device=device)
        status = wp.full(3, -17, dtype=int, device=device)
        wp.launch(
            probe_kernel(),
            dim=block,
            block_dim=block,
            inputs=[wp.array(values, device=device), wp.array(indices, device=device), count, float(target)],
            outputs=[output, status],
            device=device,
        )
        return output.numpy(), status.numpy(), np.float32(target)

    def test_active18_partial_sliding_and_sticking_both_handoffs(self):
        for block in (32, 64):
            for sizes in ([1], [2], [3], [3, 3], [1, 2, 3, 3, 3, 3, 3]):
                with self.subTest(block=block, sizes=sizes):
                    packed = coupled_case(sizes)
                    output, status, target = self.run_case(packed, block)
                    count = sum(sizes)
                    rhs = packed[0][171 : 171 + count].astype(np.float64)
                    self.assertEqual(tuple(status[:2]), (1, 0))
                    self.assertLessEqual(status[2], min(count, 8))
                    defect = np.linalg.norm(packed[2] @ output[:count] - rhs)
                    self.assertLessEqual(defect, target * 1.1 + 2e-6)
                    self.assertLessEqual(output[18], target)
                    np.testing.assert_allclose(output[:count], np.linalg.solve(packed[2], rhs), rtol=3e-4, atol=3e-5)
                    np.testing.assert_array_equal(output[count:18], 0)

    def test_nonsymmetric_local_pivot_and_zero_or_loose_rhs(self):
        # Generic linear ABI test deliberately forces a local row interchange.
        desired = np.array([[0.0, 1.0, 0.2], [2.0, 1.0, 0.0], [0.1, 0.0, 3.0]])
        packed = pack_case(np.eye(3) * 0.5, np.ones(3), 2 * (np.eye(3) - desired), [1, -2, 3], [3])
        for block in (32, 64):
            output, status, _ = self.run_case(packed, block)
            self.assertEqual(tuple(status[:2]), (1, 0))
            np.testing.assert_allclose(output[:3], np.linalg.solve(packed[2], [1, -2, 3]), atol=3e-6)
            output, status, _ = self.run_case(packed, block, target=4.0)
            self.assertEqual(tuple(status), (1, 0, 0))
            np.testing.assert_array_equal(output[:18], 0)
            empty = pack_case(np.empty((0, 0)), [], np.empty((0, 0)), [], [])
            output, status, _ = self.run_case(empty, block, target=0.0)
            self.assertEqual(tuple(status), (1, 0, 0))
            np.testing.assert_array_equal(output, 0)

    def test_invalid_rank_and_eight_step_forcing_guards(self):
        for block in (32, 64):
            for count in (-1, 19):
                _, status, _ = self.run_case(coupled_case([3]), block, count=count, target=1e-5)
                self.assertEqual(tuple(status[:2]), (0, 1))
            values, indices, matrix = coupled_case([3])
            for offset in (0, 189, 207):
                bad = values.copy()
                bad[offset] = np.nan
                _, status, _ = self.run_case((bad, indices, matrix), block)
                self.assertEqual(tuple(status[:2]), (0, 1))
            bad_indices = indices.copy()
            bad_indices[1] = 1
            _, status, _ = self.run_case((values, bad_indices, matrix), block)
            self.assertEqual(tuple(status[:2]), (0, 1))
            singular = pack_case(np.zeros((3, 3)), np.ones(3), np.eye(3), [1, 2, 3], [3])
            _, status, _ = self.run_case(singular, block)
            self.assertEqual(tuple(status[:2]), (0, 2))
            rng = np.random.default_rng(41)
            response = rng.normal(size=(18, 18))
            gram = response @ response.T + np.eye(18) * 0.01
            hard = pack_case(gram, np.ones(18), np.eye(18), rng.normal(size=18), [1] * 18)
            _, status, _ = self.run_case(hard, block, target=1e-7)
            self.assertEqual(tuple(status), (0, 5, 8))


if __name__ == "__main__":
    unittest.main()
