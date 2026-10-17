# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Keep general sparse dynamics distinct from retained research representations."""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS


class _TopologyReached(Exception):
    pass


class _AdmissionProbe(SimpleNamespace):
    @property
    def _model_plan(self):
        # Stop after admission, before any factor planning or device allocation.
        raise _TopologyReached

    # The solver defines both contact-law predicates before route selection;
    # derive them from the probe's law fields as it does.
    @property
    def _law_agnostic_contact_paths(self):
        return not (self._contact_torsion_enabled or self.contact_compliance)

    @property
    def _point_contact_paths(self):
        return self._law_agnostic_contact_paths and not self._friction_anchors_enabled

    # Patch friction may run the parallel tiers only under the opt-in
    # FEATHER_PGS_PATCH_PARALLEL_ROWS=1, which the probe's _patch_parallel_rows
    # stands for: on in the probe, off in the solver by default.
    @property
    def _parallel_contact_paths(self):
        return self._point_contact_paths or (self._law_agnostic_contact_paths and self._patch_parallel_rows)

    @property
    def _patch_dense_paths(self):
        return (
            self._law_agnostic_contact_paths
            and self._friction_anchors_enabled
            and os.environ.get("FEATHER_PGS_PATCH_DENSE_PATHS", "1") != "0"
        )


def _eligible_solver():
    """Provide only the host fields needed before topology planning starts."""
    return _AdmissionProbe(
        _kernel_overrides={},
        pgs_mode="matrix_free",
        articulated_contact_response="immediate",
        pgs_schedule="interleaved",
        drive_mode="augmented",
        _parallel_augmented_drive_topology=True,
        friction_mode="current",
        pgs_iterations=8,
        pgs_velocity_iterations=0,
        enable_joint_velocity_limits=False,
        pgs_warmstart=False,
        _mf_warmstart_enabled=False,
        _debug_buffers_enabled=False,
        _regularization_enabled=False,
        _preelim_active=False,
        _contact_torsion_enabled=False,
        _friction_anchors_enabled=False,
        contact_compliance=False,
        _mimic_count=0,
        _connect_count=0,
        _has_loop_joints=False,
        _sparse_diagonal_contact_solve=False,
        grouped_dynamics=False,
        _has_free_rigid_bodies=False,
        mf_gs_parallel_rows=0,
        mf_gs_incremental_rows=32,
        mf_gs_response_block_rows=0,
        _patch_parallel_rows=True,
    )


def _branched_model(branches, depth, device):
    """Build equal-DOF trees with different widths, without loading task assets."""
    scene = newton.ModelBuilder()
    root = scene.add_link(mass=2.0, inertia=wp.mat33(np.eye(3)))
    joints = [scene.add_joint_free(root)]
    for branch in range(branches):
        parent = root
        for level in range(depth):
            child = scene.add_link(mass=0.5, inertia=wp.mat33(np.eye(3) * 0.1))
            joints.append(
                scene.add_joint_revolute(
                    parent,
                    child,
                    axis=newton.Axis.Y,
                    parent_xform=wp.transform(wp.vec3(0.1, 0.1 * branch, -0.1), wp.quat_identity()),
                    target_ke=20.0,
                    target_kd=2.0,
                    armature=0.1,
                )
            )
            if level == depth - 1:
                scene.add_shape_sphere(child, radius=0.04)
            parent = child
    scene.add_articulation(joints)
    builder = newton.ModelBuilder()
    builder.replicate(scene, 2)
    return builder.finalize(device=device)


class TestSparseResearchAdmission(unittest.TestCase):
    def test_general_sparse_admission_reaches_topology(self):
        """Keep ordinary eligible trees on the general sparse selection path."""
        solver = _eligible_solver()
        model = SimpleNamespace(device=SimpleNamespace(is_cuda=True), requires_grad=False)
        with mock.patch.dict(os.environ, {}, clear=True), self.assertRaises(_TopologyReached):
            SolverFeatherPGS._setup_sparse_mass_matrix(solver, model)

    def test_point_research_requests_keep_ownership(self):
        """Do not replace buffers needed by an explicitly requested point-contact owner."""
        model = SimpleNamespace(device=SimpleNamespace(is_cuda=True), requires_grad=False)
        with mock.patch.dict(os.environ, {"FEATHER_PGS_SINGLE_FACTOR": "1"}, clear=True):
            solver = _eligible_solver()
            SolverFeatherPGS._setup_sparse_mass_matrix(solver, model)
            self.assertIsNone(solver._sparse_mass_matrix_size)

    def test_retired_packed_route_flags_reach_general_sparse(self):
        """Admit the general sparse route under either friction law when a recipe sets the retired flags."""
        model = SimpleNamespace(device=SimpleNamespace(is_cuda=True), requires_grad=False)
        flags = dict.fromkeys(
            (
                "FEATHER_PGS_G1_CHAIN_SCAN",
                "FEATHER_PGS_SPARSE_FACTOR",
                "FEATHER_PGS_SPARSE_LEVEL_UPDATE",
                "FEATHER_PGS_SPARSE_LIMIT_JACOBI",
                "FEATHER_PGS_SPARSE_METRIC_TANGENTS",
                "FEATHER_PGS_SPARSE_PARALLEL_LIMITS",
                "FEATHER_PGS_SPARSE_SPECTRAL_TANGENTS",
            ),
            "1",
        )
        for anchors in (False, True):
            with (
                self.subTest(friction_anchors=anchors),
                mock.patch.dict(os.environ, flags, clear=True),
                self.assertRaises(_TopologyReached),
            ):
                solver = _eligible_solver()
                solver._friction_anchors_enabled = anchors
                SolverFeatherPGS._setup_sparse_mass_matrix(solver, model)

    def test_free_rigid_bodies_keep_the_dense_route(self):
        """A world pairing an articulation with free rigid bodies declines the sparse route, for both friction laws."""
        model = SimpleNamespace(device=SimpleNamespace(is_cuda=True), requires_grad=False)
        for anchors in (False, True):
            with self.subTest(friction_anchors=anchors), mock.patch.dict(os.environ, {}, clear=True):
                solver = _eligible_solver()
                solver._friction_anchors_enabled = anchors
                solver._has_free_rigid_bodies = True
                SolverFeatherPGS._setup_sparse_mass_matrix(solver, model)
                self.assertIsNone(solver._sparse_mass_matrix_size)
                self.assertIn("free rigid bodies", solver._sparse_selection_reason)
        # FEATHER_PGS_PATCH_DENSE_PATHS=0 leaves patch friction's free-body worlds to sparse admission.
        with (
            mock.patch.dict(os.environ, {"FEATHER_PGS_PATCH_DENSE_PATHS": "0"}, clear=True),
            self.assertRaises(_TopologyReached),
        ):
            solver = _eligible_solver()
            solver._friction_anchors_enabled = True
            solver._has_free_rigid_bodies = True
            SolverFeatherPGS._setup_sparse_mass_matrix(solver, model)
        # The decline is about free rigid bodies, not the friction law: without them, both laws reach admission.
        for anchors in (False, True):
            with (
                self.subTest(friction_anchors=anchors, free_rigid_bodies=False),
                mock.patch.dict(os.environ, {}, clear=True),
                self.assertRaises(_TopologyReached),
            ):
                solver = _eligible_solver()
                solver._friction_anchors_enabled = anchors
                SolverFeatherPGS._setup_sparse_mass_matrix(solver, model)

    def test_contact_writer_requests_do_not_block_sparse(self):
        """Allow general sparse producers to supersede optional canonical row writers."""
        model = SimpleNamespace(device=SimpleNamespace(is_cuda=True), requires_grad=False)
        for option in ("ROWS_MASKED", "ROWS_NATIVE", "ROWS_FUSED"):
            with (
                self.subTest(option=option),
                mock.patch.dict(os.environ, {f"FEATHER_PGS_{option}": "1"}, clear=True),
                self.assertRaises(_TopologyReached),
            ):
                SolverFeatherPGS._setup_sparse_mass_matrix(_eligible_solver(), model)

    def test_grouped_dynamics_auto_and_legacy_opt_out(self):
        """Compose grouped dynamics automatically while retaining its explicit opt-out."""
        solver = _eligible_solver()
        solver.grouped_dynamics = True
        model = SimpleNamespace(device=SimpleNamespace(is_cuda=True), requires_grad=False)
        with mock.patch.dict(os.environ, {}, clear=True), self.assertRaises(_TopologyReached):
            SolverFeatherPGS._setup_sparse_mass_matrix(solver, model)
        with mock.patch.dict(os.environ, {"FEATHER_PGS_SPARSE_GROUPED_DYNAMICS": "0"}, clear=True):
            SolverFeatherPGS._setup_sparse_mass_matrix(solver, model)
        self.assertIsNone(solver._sparse_mass_matrix_size)

    def test_multiblock_request_keeps_existing_owner(self):
        """Leave explicit multiblock selection and its feature validation on the existing path."""
        model = SimpleNamespace(device=SimpleNamespace(is_cuda=True), requires_grad=False)
        for patches in (False, True):
            with (
                self.subTest(patches=patches),
                mock.patch.dict(os.environ, {"FEATHER_PGS_MF_MULTIBLOCK": "1"}, clear=True),
            ):
                solver = _eligible_solver()
                solver._friction_anchors_enabled = patches
                SolverFeatherPGS._setup_sparse_mass_matrix(solver, model)
                self.assertIsNone(solver._sparse_mass_matrix_size)

    def test_implicit_gyro_only_reserves_worlds_with_free_bodies(self):
        """Keep the alternative gyro owner without rejecting an inactive no-free-body request."""
        model = SimpleNamespace(device=SimpleNamespace(is_cuda=True), requires_grad=False)
        with mock.patch.dict(os.environ, {"FEATHER_PGS_FREE_BODY_IMPLICIT_GYRO": "1"}, clear=True):
            solver = _eligible_solver()
            with self.assertRaises(_TopologyReached):
                SolverFeatherPGS._setup_sparse_mass_matrix(solver, model)
            solver._has_free_rigid_bodies = True
            SolverFeatherPGS._setup_sparse_mass_matrix(solver, model)
            self.assertIsNone(solver._sparse_mass_matrix_size)

    def test_explicit_schedules_keep_their_solve_budget(self):
        """Do not silently replace a requested parallel or response-block solve."""
        model = SimpleNamespace(device=SimpleNamespace(is_cuda=True), requires_grad=False)
        for name in ("mf_gs_parallel_rows", "mf_gs_response_block_rows"):
            with self.subTest(name=name), mock.patch.dict(os.environ, {}, clear=True):
                solver = _eligible_solver()
                setattr(solver, name, 32)
                SolverFeatherPGS._setup_sparse_mass_matrix(solver, model)
                self.assertIsNone(solver._sparse_mass_matrix_size)

    def test_patch_friction_yields_only_to_a_parallel_tier_it_runs(self):
        """Keep the ordinary buffers for a patch parallel tier, but not for split mode, response blocks or the default.

        Opted in, patch friction runs the recipe's parallel tier as point friction does,
        so the tier's request declines sparse factors in matrix-free mode. Split mode
        builds no tier, response blocks stay point-only, and by default, without
        FEATHER_PGS_PATCH_PARALLEL_ROWS=1, patch friction keeps its sparse route.
        """
        model = SimpleNamespace(device=SimpleNamespace(is_cuda=True), requires_grad=False)
        for name, changes, declined in (
            ("parallel", {"mf_gs_parallel_rows": 48}, True),
            ("switched_off", {"mf_gs_parallel_rows": 48, "_patch_parallel_rows": False}, False),
            ("split", {"mf_gs_parallel_rows": 48, "pgs_mode": "split", "pgs_kernel": "tiled_row"}, False),
            ("response_block", {"mf_gs_response_block_rows": 64}, False),
        ):
            with self.subTest(name=name), mock.patch.dict(os.environ, {}, clear=True):
                solver = _eligible_solver()
                solver._friction_anchors_enabled = True
                for attribute, value in changes.items():
                    setattr(solver, attribute, value)
                if declined:
                    SolverFeatherPGS._setup_sparse_mass_matrix(solver, model)
                    self.assertIsNone(solver._sparse_mass_matrix_size)
                    self.assertIn("competing", solver._sparse_selection_reason)
                else:
                    with self.assertRaises(_TopologyReached):
                        SolverFeatherPGS._setup_sparse_mass_matrix(solver, model)

    @unittest.skipUnless(wp.is_cuda_available(), "Sparse construction requires CUDA")
    def test_branched_18_dof_trees_select_general_sparse(self):
        """Admit both wide and deeper trees without activating a second representation."""
        for branches, depth, nonzeros, support in ((4, 3, 117, 12), (2, 6, 135, 18)):
            with self.subTest(branches=branches, depth=depth), mock.patch.dict(os.environ, {}, clear=True):
                model = _branched_model(branches, depth, "cuda:0")
                solver = SolverFeatherPGS(
                    model,
                    pgs_mode="matrix_free",
                    drive_mode="augmented",
                    articulated_contact_response="immediate",
                    friction_anchor_beta=0.0,
                    enable_joint_velocity_limits=False,
                    pgs_warmstart=False,
                    mf_warmstart=False,
                    dense_max_constraints=40,
                    grouped_dynamics=False,
                    mf_gs_parallel_rows=0,
                    mf_gs_response_block_rows=0,
                    use_parallel_streams=True,
                )
                self.assertEqual(solver._sparse_mass_matrix_size, 18)
                self.assertEqual(solver._sparse_mass_matrix_plan.nonzero_count, nonzeros)
                self.assertEqual(solver._sparse_row_factor.shape[2], support)
                self.assertEqual(solver.mf_gs_incremental_rows, 32)
                self.assertIsNone(solver._pgs_solve_mf_gs_kernel)
                self.assertEqual(solver._pgs_solve_mf_gs_incremental_kernels, [])
                self.assertIsNone(solver._ink_sizes)
                self.assertEqual(solver._ink_rows, 0)
                for name in ("_fused_k1", "_grouped_tau_mass", "_grouped_mass"):
                    self.assertFalse(getattr(solver, name), name)


if __name__ == "__main__":
    unittest.main()
