# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Keep cached dynamics independent from the sparse constraint-row policy."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import held_tree
from newton.solvers import SolverFeatherPGS

# (law-agnostic, point) contact-path predicates of each contact law.
_CONTACT_LAWS = {
    "point": (True, True),
    "patch": (True, False),
    "torsion": (False, False),
    "compliance": (False, False),
}


def _held_tree_solver(law):
    """Describe a dense-route solver that every other held-tree admission term accepts."""
    law_agnostic, point = _CONTACT_LAWS[law]
    return SimpleNamespace(
        model=SimpleNamespace(device=SimpleNamespace(is_cuda=True), requires_grad=False, particle_count=0),
        size_groups=[9],
        _sparse_mass_matrix_size=None,
        _law_agnostic_contact_paths=law_agnostic,
        _point_contact_paths=point,
        pgs_mode="matrix_free",
        _paired_response_primary_size=None,
        drive_mode="augmented",
        pgs_velocity_iterations=0,
        _connect_count=0,
        _has_loop_joints=False,
        _fk_id_cache_available=True,
        _parallel_augmented_drive_topology=True,
        _articulation_dynamics_stream=object(),
        _debug_buffers_enabled=False,
        _compact_diagonal_mass_size=None,
    )


def _sparse_tree_solver(**changes):
    """Describe a native sparse route that every sparse tree-owner admission term accepts."""
    solver = SimpleNamespace(
        model=SimpleNamespace(device=SimpleNamespace(is_cuda=True), requires_grad=False, particle_count=0),
        _sparse_mass_matrix_size=18,
        _has_free_rigid_bodies=False,
        _mimic_count=0,
        _sparse_velocity_limits=False,
        _mimic_valid_np=None,
        _connect_count=0,
        _has_loop_joints=False,
        _has_prescribed_response=False,
        _fk_id_cache_available=True,
        pgs_mode="matrix_free",
        _sparse_split=False,
        _parallel_augmented_drive_topology=True,
        _articulation_dynamics_stream=object(),
        _debug_buffers_enabled=False,
        _double_buffer=False,
    )
    for name, value in changes.items():
        setattr(solver, name, value)
    return solver


class TestSparseTreeAdmission(unittest.TestCase):
    def test_sparse_route_is_admitted(self):
        """Admit the sparse factor's held group where every admission term holds."""
        self.assertIsNone(held_tree.decline(_sparse_tree_solver()))


class TestHeldTreeContactLaw(unittest.TestCase):
    def test_held_tree_owner_serves_every_contact_law(self):
        """Admit the dense held-tree owner under either friction law, torsion and compliance."""
        for law in _CONTACT_LAWS:
            with self.subTest(law=law):
                self.assertTrue(held_tree.admitted(_held_tree_solver(law)))

    def test_contact_law_predicates_precede_route_selection(self):
        """Define the contact-law predicates, the parallel-row one included, before sparse route selection."""
        builder = newton.ModelBuilder()
        body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_articulation([builder.add_joint_revolute(-1, body)])
        model = builder.finalize(device="cpu")
        setup = SolverFeatherPGS._setup_sparse_mass_matrix
        seen = []

        def recording(solver, source):
            seen.append(
                (solver._law_agnostic_contact_paths, solver._point_contact_paths, solver._parallel_contact_paths)
            )
            return setup(solver, source)

        with (
            patch.object(SolverFeatherPGS, "_setup_sparse_mass_matrix", recording),
        ):
            for beta, law in ((0.0, "point"), (0.2, "patch")):
                with self.subTest(law=law):
                    solver = SolverFeatherPGS(model, friction_anchor_beta=beta)
                    # Both laws may run the parallel tiers.
                    self.assertEqual(seen[-1], (*_CONTACT_LAWS[law], True))
                    self.assertEqual(
                        (solver._law_agnostic_contact_paths, solver._point_contact_paths), _CONTACT_LAWS[law]
                    )


if __name__ == "__main__":
    unittest.main()
