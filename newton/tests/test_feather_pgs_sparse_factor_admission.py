# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Preserve admission for the surviving packed sparse representation."""

import unittest
from types import SimpleNamespace

from newton._src.solvers.feather_pgs import sparse_factor


def supported_solver():
    """Describe the admitted G1 contract without constructing a simulator."""
    return SimpleNamespace(
        model=SimpleNamespace(device=SimpleNamespace(is_cuda=True), requires_grad=False, articulation_count=4),
        world_count=4,
        max_world_dofs=43,
        size_groups=(43,),
        n_arts_by_size={43: 4},
        tile_threads=128,
        _execution_plan=SimpleNamespace(
            use_tiled_hinv_jt=lambda size: True,
            use_diagonal_mass=lambda size: False,
            use_tiled_cholesky=lambda size: True,
        ),
        trisolve_kernel="tiled",
        small_dof_threshold=32,
        _mimic_count=0,
        _connect_count=0,
        _debug_buffers_enabled=False,
        _grouped_tau_mass=False,
        _grouped_mass=False,
        _wr_world_contacts=None,
        _ink_sizes=None,
        _is_one_solve_art_per_world=True,
        _has_free_rigid_bodies=False,
        _preelim_active=False,
        _regularization_enabled=False,
        _local_internal_fast_path=False,
        _paired_factor_coordinates=False,
        _sparse_diagonal_contact_solve=False,
        _mf_warmstart_enabled=False,
        _jy_world_aliased=True,
        _hinv_jt_writes_world=False,
        pgs_mode="matrix_free",
        pgs_schedule="interleaved",
        pgs_iterations=8,
        pgs_velocity_iterations=0,
        pgs_warmstart=False,
        pgs_debug=False,
        drive_mode="augmented",
        friction_mode="current",
        enable_joint_velocity_limits=False,
        fuse_joint_velocity_limits=False,
        articulated_contact_response="immediate",
        mf_gs_response_block_rows=0,
        mf_gs_incremental_rows=0,
        mf_gs_parallel_rows=0,
        _propagation_contacts_enabled=lambda: False,
    )


class TestSparseFactorAdmission(unittest.TestCase):
    """Keep unsupported response and solve owners on their original path."""

    def test_supported_single_articulation(self):
        """Admit the original G1 budget without adding an inverse or response panel."""
        self.assertTrue(sparse_factor.supported(supported_solver()))

    def test_reject_incompatible_calls(self):
        """Reject each contract change before replacing original response production."""
        for name, value in (
            ("pgs_warmstart", True),
            ("pgs_debug", True),
            ("_has_free_rigid_bodies", True),
            ("pgs_velocity_iterations", 1),
            ("pgs_iterations", 0),
            ("_preelim_active", True),
            ("_regularization_enabled", True),
            ("enable_joint_velocity_limits", True),
            ("mf_gs_parallel_rows", 48),
            ("mf_gs_incremental_rows", 32),
            ("mf_gs_response_block_rows", 64),
            ("pgs_schedule", "contact_then_internal"),
            ("friction_mode", "bisection"),
            ("drive_mode", "physx_pgs"),
            ("_local_internal_fast_path", True),
            ("_paired_factor_coordinates", True),
            ("_propagation_contacts_enabled", lambda: True),
            ("max_world_dofs", 29),
            ("tile_threads", 16),
            ("tile_threads", 33),
            ("_execution_plan", SimpleNamespace(use_tiled_hinv_jt=lambda size: False)),
            (
                "_execution_plan",
                SimpleNamespace(use_tiled_hinv_jt=lambda size: True, use_diagonal_mass=lambda size: True),
            ),
        ):
            with self.subTest(name=name):
                solver = supported_solver()
                setattr(solver, name, value)
                self.assertFalse(sparse_factor.supported(solver))


if __name__ == "__main__":
    unittest.main()
