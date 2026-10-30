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
from newton._src.solvers.feather_pgs.route_table import Laws, RouteTable
from newton.solvers import SolverFeatherPGS


class _TopologyReached(Exception):
    pass


class _ForestProbe(SimpleNamespace):
    @property
    def response_dof_count(self):
        # Stop after admission, before any factor planning or device allocation.
        raise _TopologyReached


class _LawsProbe(SimpleNamespace):
    """The probe's row-law tables, with the laws axis read from them and the probe's options as ``Laws.build`` reads
    a solver's (a CUDA model without gradients)."""

    @property
    def set(self):
        options = self.options
        return Laws(
            contact="patch" if self.friction_anchors_enabled else "point",
            regularized=self.regularization_enabled,
            torsion=self.contact_torsion_enabled,
            compliance=options.contact_compliance,
            limits=True,
            velocity_limits=options.enable_joint_velocity_limits,
            warm_start=options.pgs_warmstart,
            debug_rows=options.pgs_debug,
            preelimination=False,
            mimic=bool(self.mimic_count),
            connect=bool(self.connect_count),
            loops=self.forest.has_loop_joints,
            iterations=options.pgs_iterations > 0,
            cuda=True,
        )


class _AdmissionProbe(SimpleNamespace):
    # Every contact law may run the parallel tiers.
    _parallel_contact_paths = True


def _eligible_solver():
    """Provide only the host fields needed before topology planning starts."""
    forest = _ForestProbe(has_loop_joints=False, has_free_rigid_bodies=False)
    options = SimpleNamespace(
        pgs_iterations=8,
        enable_joint_velocity_limits=False,
        pgs_warmstart=False,
        pgs_debug=False,
        contact_compliance=False,
    )
    return _AdmissionProbe(
        _route_pins={},
        factor=SimpleNamespace(parallel_drives=True),
        _preelim_active=False,
        routes=RouteTable({}),
        forest=forest,
        options=options,
        laws=_LawsProbe(
            options=options,
            forest=forest,
            regularization_enabled=False,
            contact_torsion_enabled=False,
            friction_anchors_enabled=False,
            mimic_count=0,
            connect_count=0,
        ),
    )


def _change(solver, name, value):
    """Set a probe's field: an option in its options record, its forest plan in both places that read it."""
    if name == "forest":
        solver.forest = solver.laws.forest = value
    elif hasattr(solver.options, name):
        setattr(solver.options, name, value)
    else:
        setattr(solver, name, value)


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
        model = SimpleNamespace(device=SimpleNamespace(is_cuda=True))
        with mock.patch.dict(os.environ, {}, clear=True), self.assertRaises(_TopologyReached):
            SolverFeatherPGS._setup_sparse_mass_matrix(solver, model)

    def test_removed_packed_route_flags_reach_general_sparse(self):
        """Admit the general sparse route under either friction law when a recipe sets the removed flags."""
        model = SimpleNamespace(device=SimpleNamespace(is_cuda=True))
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
                solver.laws.friction_anchors_enabled = anchors
                SolverFeatherPGS._setup_sparse_mass_matrix(solver, model)

    def test_free_rigid_bodies_keep_the_dense_route(self):
        """A world pairing an articulation with free rigid bodies declines the sparse route, for both friction laws."""
        model = SimpleNamespace(device=SimpleNamespace(is_cuda=True))
        for anchors in (False, True):
            with self.subTest(friction_anchors=anchors), mock.patch.dict(os.environ, {}, clear=True):
                solver = _eligible_solver()
                solver.laws.friction_anchors_enabled = anchors
                solver.forest.has_free_rigid_bodies = True
                SolverFeatherPGS._setup_sparse_mass_matrix(solver, model)
                self.assertIsNone(solver._sparse_mass_matrix_size)
                self.assertIn("free rigid bodies", solver.routes.reasons["K1"])
        # The decline is about free rigid bodies, not the friction law: without them, both laws reach admission.
        for anchors in (False, True):
            with (
                self.subTest(friction_anchors=anchors, free_rigid_bodies=False),
                mock.patch.dict(os.environ, {}, clear=True),
                self.assertRaises(_TopologyReached),
            ):
                solver = _eligible_solver()
                solver.laws.friction_anchors_enabled = anchors
                SolverFeatherPGS._setup_sparse_mass_matrix(solver, model)

    def test_a_tier_pin_keeps_dense_storage(self):
        """Keep the dense storage the Jacobi tiers need where a test pin builds them on a single tree."""
        model = SimpleNamespace(device=SimpleNamespace(is_cuda=True))
        with mock.patch.dict(os.environ, {}, clear=True):
            solver = _eligible_solver()
            solver._route_pins = {"tiers": 32}
            SolverFeatherPGS._setup_sparse_mass_matrix(solver, model)
            self.assertIsNone(solver._sparse_mass_matrix_size)
            self.assertIn("Jacobi tiers", solver.routes.reasons["K1"])

    def test_patch_free_body_worlds_keep_the_dense_route(self):
        """Keep a patch-friction world with free rigid bodies on the dense route, as the free-body rule says."""
        model = SimpleNamespace(device=SimpleNamespace(is_cuda=True))
        with mock.patch.dict(os.environ, {}, clear=True):
            solver = _eligible_solver()
            solver.laws.friction_anchors_enabled = True
            solver.forest.has_free_rigid_bodies = True
            SolverFeatherPGS._setup_sparse_mass_matrix(solver, model)
            self.assertIsNone(solver._sparse_mass_matrix_size)
            self.assertIn("free rigid bodies", solver.routes.reasons["K1"])

    @unittest.skipUnless(wp.is_cuda_available(), "Sparse construction requires CUDA")
    def test_branched_18_dof_trees_select_general_sparse(self):
        """Admit both wide and deeper trees without activating a second representation."""
        for branches, depth, nonzeros, support in ((4, 3, 117, 12), (2, 6, 135, 18)):
            with self.subTest(branches=branches, depth=depth), mock.patch.dict(os.environ, {}, clear=True):
                model = _branched_model(branches, depth, "cuda:0")
                solver = SolverFeatherPGS(
                    model,
                    friction_anchor_beta=0.0,
                    enable_joint_velocity_limits=False,
                    pgs_warmstart=False,
                    dense_max_constraints=40,
                )
                self.assertEqual(solver._sparse_mass_matrix_size, 18)
                self.assertEqual(solver._sparse_mass_matrix_plan.nonzero_count, nonzeros)
                self.assertEqual(solver._sparse_row_factor.shape[2], support)
                self.assertIsNone(solver._pgs_solve_mf_gs_kernel)
                self.assertEqual(solver._mf_gs_tiers, [])
                self.assertIsNone(solver._ink_sizes)
                self.assertEqual(solver._ink_rows, 0)


if __name__ == "__main__":
    unittest.main()
