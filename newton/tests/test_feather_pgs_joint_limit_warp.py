# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton._src.sim.enums import JointType
from newton._src.solvers.feather_pgs import field_group
from newton._src.solvers.feather_pgs.dense_rows import DenseRows
from newton._src.solvers.feather_pgs.joint_table import JointTable
from newton._src.solvers.feather_pgs.kernels import build_joint_limit_rows_for_size
from newton._src.solvers.feather_pgs.prefix_rows import PrefixLaw
from newton._src.solvers.feather_pgs.rows import joint_limit_warp_kernel
from newton.tests.feather_pgs_law_harness import group_rows


def _outputs(articulation_count: int, max_constraints: int, size: int, device):
    return (
        wp.zeros(articulation_count, dtype=wp.int32, device=device),
        wp.zeros((articulation_count, max_constraints, size), dtype=wp.float32, device=device),
        wp.zeros((articulation_count, max_constraints), dtype=wp.int32, device=device),
        wp.zeros((articulation_count, max_constraints), dtype=wp.int32, device=device),
        wp.zeros((articulation_count, max_constraints), dtype=wp.float32, device=device),
        wp.zeros((articulation_count, max_constraints), dtype=wp.float32, device=device),
        wp.zeros((articulation_count, max_constraints), dtype=wp.float32, device=device),
        wp.zeros((articulation_count, max_constraints), dtype=wp.float32, device=device),
        wp.zeros((articulation_count, max_constraints), dtype=wp.float32, device=device),
    )


def _dense_rows(outputs, device):
    """The row records of ``outputs`` (after the counter and the Jacobian), as the kernels' DenseRows."""
    names = ("row_type", "row_parent", "row_mu", "row_beta", "row_cfm", "phi", "target")
    sides = wp.zeros(outputs[3].shape, dtype=wp.vec2i, device=device)
    return field_group.upload(DenseRows, device, **dict(zip(names, outputs[2:], strict=True)), sides=sides)


def _problem(device, articulation_count=257, size=6):
    """One D6 joint per articulation and world, with positions drawn around its [-1, 1] limits."""
    dof_count = articulation_count * size
    rng = np.random.default_rng(42)
    return SimpleNamespace(
        device=device,
        articulation_count=articulation_count,
        size=size,
        max_constraints=2 * size,
        dof_count=dof_count,
        articulation_start=wp.array(np.arange(articulation_count + 1), dtype=wp.int32, device=device),
        articulation_dof_start=wp.array(np.arange(articulation_count) * size, dtype=wp.int32, device=device),
        joint_type=wp.array(np.full(articulation_count, JointType.D6), dtype=wp.int32, device=device),
        joint_q_start=wp.array(np.arange(articulation_count) * size, dtype=wp.int32, device=device),
        joint_qd_start=wp.array(np.arange(articulation_count) * size, dtype=wp.int32, device=device),
        joint_dof_dim=wp.array(np.tile([3, 3], (articulation_count, 1)), dtype=wp.int32, device=device),
        lower=wp.array(np.full(dof_count, -1.0, dtype=np.float32), dtype=wp.float32, device=device),
        upper=wp.array(np.full(dof_count, 1.0, dtype=np.float32), dtype=wp.float32, device=device),
        joint_q=wp.array(rng.uniform(-1.2, 1.2, dof_count).astype(np.float32), dtype=wp.float32, device=device),
        art_to_world=wp.array(np.arange(articulation_count), dtype=wp.int32, device=device),
        group_to_art=wp.array(np.arange(articulation_count), dtype=wp.int32, device=device),
    )


def _resolved_mask(problem, resolved):
    """The per-world resolved mask operand; ``None`` passes the empty mask (no world resolved)."""
    if resolved is None:
        return wp.empty(0, dtype=wp.int32, device=problem.device)
    return wp.array(np.asarray(resolved, np.int32), dtype=wp.int32, device=problem.device)


def _scalar_rows(problem, resolved=None):
    """Build the joint-limit rows with the scalar per-articulation kernel."""
    outputs = _outputs(problem.articulation_count, problem.max_constraints, problem.size, problem.device)
    counter, *rows = outputs
    first_rejected = wp.full(problem.articulation_count, 2**31 - 1, dtype=wp.int32, device=problem.device)
    wp.launch(
        build_joint_limit_rows_for_size,
        dim=problem.articulation_count,
        inputs=[
            field_group.upload(
                JointTable,
                problem.device,
                problem,
                joint_limit_lower=problem.lower,
                joint_limit_upper=problem.upper,
            ),
            problem.joint_q,
            0.25,
            problem.group_to_art,
            problem.max_constraints,
            0.2,
            1.0e-6,
            _resolved_mask(problem, resolved),
            group_rows(problem.device),
        ],
        outputs=[counter, first_rejected, rows[0], _dense_rows(outputs, problem.device)],
        device=problem.device,
    )
    return outputs


def _warp_rows(problem, resolved=None, warps_per_block=4):
    """Build the joint-limit rows with the warp-parallel kernel (CUDA only)."""
    outputs = _outputs(problem.articulation_count, problem.max_constraints, problem.size, problem.device)
    names = ("row_type", "row_parent", "row_mu", "row_beta", "row_cfm", "phi", "target")
    law = field_group.upload(
        PrefixLaw,
        problem.device,
        limit_q_index=np.arange(problem.dof_count),
        lower=problem.lower,
        upper=problem.upper,
        q=problem.joint_q,
        activation_gap=0.25,
        beta=0.2,
        cfm=1.0e-6,
        slot_counter=outputs[0],
        **dict(zip(names, outputs[2:], strict=True)),
        sides=wp.zeros(outputs[2].shape, dtype=wp.vec2i, device=problem.device),
    )
    wp.launch_tiled(
        joint_limit_warp_kernel(problem.size, warps_per_block),
        dim=[(problem.articulation_count + warps_per_block - 1) // warps_per_block],
        inputs=[
            problem.articulation_count,
            problem.articulation_dof_start,
            problem.art_to_world,
            problem.group_to_art,
            _resolved_mask(problem, resolved),
            law,
            group_rows(problem.device),
        ],
        outputs=[outputs[1]],
        block_dim=32 * warps_per_block,
        device=problem.device,
    )
    return outputs


class TestFeatherPGSJointLimitWarp(unittest.TestCase):
    def test_scalar_builder_skips_resolved_worlds_cpu(self):
        """Launch the scalar builder on CPU, so its operand list is checked without a GPU.

        An all-zero mask builds the same rows as the empty mask, and a resolved world builds none.
        """
        problem = _problem(wp.get_device("cpu"))
        unmasked = _scalar_rows(problem)
        self.assertGreater(int(unmasked[0].numpy().sum()), 0)
        for actual, expected in zip(_scalar_rows(problem, np.zeros(problem.articulation_count)), unmasked, strict=True):
            np.testing.assert_array_equal(actual.numpy(), expected.numpy())
        resolved = np.arange(problem.articulation_count) % 3 == 0
        for actual, original in zip(_scalar_rows(problem, resolved), unmasked, strict=True):
            expected = original.numpy()
            expected[resolved] = 0
            np.testing.assert_array_equal(actual.numpy(), expected)

    @unittest.skipUnless(wp.is_cuda_available(), "warp-parallel joint-limit assembly requires CUDA")
    def test_warp_builder_matches_scalar_row_order_and_values(self):
        """Build the scalar kernel's rows with the warp-parallel kernel, with and without resolved worlds."""
        problem = _problem(wp.get_cuda_device())
        scalar = _scalar_rows(problem)
        parallel = _warp_rows(problem)
        for parallel_array, scalar_array in zip(parallel, scalar, strict=True):
            np.testing.assert_array_equal(parallel_array.numpy(), scalar_array.numpy())

        # Mixed decisions share a block: returning resolved warps must leave
        # neighboring fallback warps' ballots and row order unchanged.
        resolved = np.arange(problem.articulation_count) % 3 == 0
        mixed = _warp_rows(problem, resolved)
        for actual, original in zip(mixed, scalar, strict=True):
            expected = original.numpy()
            expected[resolved] = 0
            np.testing.assert_array_equal(actual.numpy(), expected)


if __name__ == "__main__":
    unittest.main()
