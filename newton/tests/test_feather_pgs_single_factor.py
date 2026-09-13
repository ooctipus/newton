# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the forward-only single-articulation contact representation."""

import inspect
import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import single_factor
from newton._src.solvers.feather_pgs.solver_feather_pgs import _get_pgs_solve_mf_gs_kernel


def supported_solver():
    """Describe the admitted G1 contract without constructing a simulator."""
    return SimpleNamespace(
        model=SimpleNamespace(device=SimpleNamespace(is_cuda=True), requires_grad=False, articulation_count=4),
        world_count=4,
        max_world_dofs=43,
        size_groups=(43,),
        n_arts_by_size={43: 4},
        _execution_plan=SimpleNamespace(use_tiled_hinv_jt=lambda size: True, use_diagonal_mass=lambda size: False),
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


class TestSingleFactorAdmission(unittest.TestCase):
    """Keep unsupported response and solve owners on their original path."""

    def test_supported_single_articulation(self):
        """Admit the original G1 budget without adding an inverse or response panel."""
        self.assertTrue(single_factor.supported(supported_solver()))

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
            ("_execution_plan", SimpleNamespace(use_tiled_hinv_jt=lambda size: False)),
            (
                "_execution_plan",
                SimpleNamespace(use_tiled_hinv_jt=lambda size: True, use_diagonal_mass=lambda size: True),
            ),
        ):
            with self.subTest(name=name):
                solver = supported_solver()
                setattr(solver, name, value)
                self.assertFalse(single_factor.supported(solver))


@unittest.skipUnless(wp.is_cuda_available(), "Single-factor tiled response requires CUDA")
class TestSingleFactorCUDA(unittest.TestCase):
    """Compare the complete representation with physical response and original GS."""

    def setUp(self):
        """Create nonidentity held factors and permuted full-response worlds."""
        self.device = wp.get_device("cuda:0")
        self.worlds, self.dofs, self.capacity = 4, 43, 100
        rng = np.random.default_rng(481)
        self.group_to_art = np.array([3, 1, 0, 2], np.int32)
        self.art_to_world = np.array([2, 0, 3, 1], np.int32)
        self.counts = np.array([0, 1, 7, 100], np.int32)
        self.starts = np.arange(5, dtype=np.int32) * self.dofs
        raw = rng.normal(0, 0.1, (4, 43, 43))
        mass = raw @ raw.transpose(0, 2, 1) + np.diag(np.geomspace(0.01, 10.0, 43))[None]
        self.L = np.linalg.cholesky(mass).astype(np.float32)
        self.J = rng.normal(0, 0.2, (4, 100, 43)).astype(np.float32)
        self.world_J = np.zeros_like(self.J)
        self.reference_Y = np.zeros_like(self.J)
        self.reference_Z = np.zeros_like(self.J)
        self.maps = np.empty((4, 43), np.int32)
        for group, art in enumerate(self.group_to_art):
            world = self.art_to_world[art]
            self.world_J[world] = self.J[group]
            factor = self.L[group].astype(np.float64)
            z = np.linalg.solve(factor, self.J[group].astype(np.float64).T)
            self.reference_Z[world] = z.T
            self.reference_Y[world] = np.linalg.solve(factor.T, z).T
            self.maps[world] = self.starts[art] + np.arange(43)
        self.vhat = rng.normal(0, 0.2, 172).astype(np.float32)
        self.a = {}
        for name, value in (
            ("L", self.L),
            ("J", self.J),
            ("group_to_art", self.group_to_art),
            ("art_to_world", self.art_to_world),
            ("counts", self.counts),
            ("starts", self.starts),
            ("vhat", self.vhat),
            ("vout", self.vhat),
            ("world_J", np.zeros_like(self.J)),
            ("Z", np.zeros_like(self.J)),
            ("diag", np.zeros((4, 100), np.float32)),
        ):
            self.a[name] = self.array(value)
        self.response = single_factor.get_response_kernel(43, 100, 16)
        self.decode = single_factor.get_decode_kernel(43)

    def array(self, value):
        """Allocate the native integer/float ABI without converting index maps."""
        value = np.asarray(value)
        return wp.array(value, dtype=wp.int32 if value.dtype == np.int32 else wp.float32, device=self.device)

    def launch_response(self):
        """Use actual group-to-world routing and a partial final response tile."""
        wp.launch_tiled(
            self.response,
            dim=(4, 7),
            inputs=[self.a[k] for k in ("L", "J", "group_to_art", "art_to_world", "counts")],
            outputs=[self.a[k] for k in ("world_J", "Z", "diag")],
            block_dim=64,
            device=self.device,
        )

    def launch_decode(self):
        """Decode into original global articulation velocity indices."""
        wp.launch_tiled(
            self.decode,
            dim=(4,),
            inputs=[self.a[k] for k in ("L", "group_to_art", "art_to_world", "starts", "counts", "vhat")],
            outputs=[self.a["vout"]],
            block_dim=64,
            device=self.device,
        )

    def test_forward_diagonal_and_decode(self):
        """Check 43-DOF forward action, active diagonal and physical final velocity."""
        self.launch_response()
        z, diagonal, physical_j = (self.a[k].numpy() for k in ("Z", "diag", "world_J"))
        for world, count in enumerate(self.counts):
            np.testing.assert_allclose(z[world, :count], self.reference_Z[world, :count], rtol=2e-5, atol=2e-6)
            np.testing.assert_allclose(
                diagonal[world, :count],
                np.sum(self.reference_Z[world, :count] ** 2, axis=1),
                rtol=2e-5,
                atol=2e-6,
            )
            np.testing.assert_array_equal(physical_j[world, :count], self.world_J[world, :count])
        delta = np.random.default_rng(32).normal(0, 0.1, 172).astype(np.float32)
        expected = self.vhat.copy()
        for group, art in enumerate(self.group_to_art):
            world = self.art_to_world[art]
            indices = self.maps[world]
            if self.counts[world]:
                expected[indices] += np.linalg.solve(self.L[group].astype(np.float64).T, delta[indices])
            else:
                delta[indices] = self.vhat[indices]
        self.a["vout"].assign(delta)
        self.launch_decode()
        np.testing.assert_allclose(self.a["vout"].numpy(), expected, rtol=2e-5, atol=2e-6)

    def solve_values(self, *, kinetic):
        """Bind the original dense/MF signature with empty MF and current row law."""
        rng = np.random.default_rng(778)
        kinds = np.full((4, 100), 3, np.int32)
        parents = np.full((4, 100), -1, np.int32)
        for row in range(1, 100, 3):
            kinds[:, row : row + 3] = [0, 2, 2]
            parents[:, row + 1 : row + 3] = row
        diagonal = np.sum(self.reference_Z**2, axis=2).astype(np.float32) + np.float32(0.001)
        values = {
            "general_world_count": self.array(np.array([4], np.int32)),
            "general_worlds": self.array(np.arange(4, dtype=np.int32)),
            "general_world_grid_stride": 4,
            "use_general_world_queue": 0,
            "rb_class": self.array(np.zeros(4, np.int32)),
            "world_constraint_count": self.a["counts"],
            "dense_phase_bounds": self.array(np.ones((4, 2), np.int32)),
            "local_solve_owner": self.array(np.zeros(4, np.int32)),
            "world_dof_indices": self.array(self.maps),
            "world_deferred_dof_mask": self.array(np.zeros((4, 43), np.int32)),
            "rhs_bias": self.array(rng.normal(0, 0.1, (4, 100)).astype(np.float32)),
            "world_diag": self.array(diagonal),
            "world_row_w": self.array(np.ones((4, 100), np.float32)),
            "world_impulses": self.array(np.zeros((4, 100), np.float32)),
            "J_world": self.a["world_J"] if kinetic else self.array(self.world_J),
            "Y_world": self.a["Z"] if kinetic else self.array(self.reference_Y),
            "world_row_type": self.array(kinds),
            "world_row_parent": self.array(parents),
            "world_row_mu": self.array(np.full((4, 100), 0.6, np.float32)),
            "mf_constraint_count": self.array(np.zeros(4, np.int32)),
            "mf_contact_rows_end": self.array(np.zeros(4, np.int32)),
            "mf_meta": self.array(np.zeros((4, 4), np.int32)),
            "mf_impulses": self.array(np.zeros((4, 1), np.float32)),
            "mf_row_mu": self.array(np.zeros((4, 1), np.float32)),
            "mf_row_w": self.array(np.ones((4, 1), np.float32)),
            "iterations": 8,
            "omega": 1.0,
            "regularize": 0,
            "row_phase": 0,
            "friction_start_iteration": 2,
            "iteration_offset": 0,
            "freeze_drive_rows": 0,
            "defer_dense_response": 0,
            "v_out": self.a["vout"] if kinetic else self.array(self.vhat),
        }
        for name in ("target_vel_bias", "vel_multiplier", "impulse_multiplier", "max_impulse", "vel_limit"):
            values["world_drive_" + name] = self.array(np.zeros((1, 1), np.float32))
        for name in ("mf_J_a", "mf_J_b", "mf_MiJt_a", "mf_MiJt_b"):
            values[name] = self.array(np.zeros((4, 1, 6), np.float32))
        return values

    def test_original_eight_sweeps_and_graph(self):
        """Retain delayed friction, limits and physical output across graph replay."""
        self.launch_response()
        reference = self.solve_values(kinetic=False)
        candidate = self.solve_values(kinetic=True)
        controls = []
        for kinetic, values in ((False, reference), (True, candidate)):
            kernel = _get_pgs_solve_mf_gs_kernel(
                100,
                1,
                43,
                str(self.device.arch),
                has_drive_rows=False,
                has_dense_velocity_limit_rows=False,
                single_factor_coordinates=kinetic,
            )
            arguments = [values[name] for name in inspect.signature(kernel.func).parameters]
            controls.append((kernel, arguments, values))

        def run():
            self.launch_response()
            for kernel, arguments, values in controls:
                wp.copy(values["v_out"], self.a["vhat"])
                values["world_impulses"].zero_()
                wp.launch_tiled(kernel, dim=(4,), inputs=arguments, block_dim=32, device=self.device)
            self.launch_decode()

        run()
        with wp.ScopedCapture(device=self.device) as capture:
            run()
        for _ in range(3):
            wp.capture_launch(capture.graph)
        np.testing.assert_allclose(candidate["v_out"].numpy(), reference["v_out"].numpy(), rtol=1e-4, atol=2e-5)
        np.testing.assert_allclose(
            candidate["world_impulses"].numpy(),
            reference["world_impulses"].numpy(),
            rtol=1e-4,
            atol=2e-5,
        )


if __name__ == "__main__":
    unittest.main()
