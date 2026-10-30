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
from newton.tests.feather_pgs_scenes import plain_laws

# The law changes of each contact law from the plain point law.
_CONTACT_LAWS = {
    "point": {},
    "patch": {"contact": "patch"},
    "torsion": {"torsion": True},
    "compliance": {"compliance": True},
}


def _held_tree_solver(law):
    """Describe a dense-route solver that every other held-tree admission term accepts."""
    return SimpleNamespace(
        model=SimpleNamespace(particle_count=0),
        _sparse_mass_matrix_size=None,
        _fk_id_cache_available=True,
        _parallel_augmented_drive_topology=True,
        _compact_diagonal_mass_size=None,
        forest=SimpleNamespace(size_groups=[9]),
        laws=SimpleNamespace(set=plain_laws(**_CONTACT_LAWS[law])),
    )


def _sparse_tree_solver(**changes):
    """Describe a native sparse route that every sparse tree-owner admission term accepts."""
    solver = SimpleNamespace(
        model=SimpleNamespace(particle_count=0),
        _sparse_mass_matrix_size=18,
        _sparse_velocity_limits=False,
        _fk_id_cache_available=True,
        _parallel_augmented_drive_topology=True,
        forest=SimpleNamespace(has_free_rigid_bodies=False, has_prescribed_response=False),
        laws=SimpleNamespace(set=plain_laws(), mimic_count=0, mimic_valid_np=None),
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
        """Read the laws, and the parallel-row admission, before sparse route selection."""
        builder = newton.ModelBuilder()
        body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_articulation([builder.add_joint_revolute(-1, body)])
        model = builder.finalize(device="cpu")
        setup = SolverFeatherPGS._setup_sparse_mass_matrix
        seen = []

        def recording(solver, source):
            seen.append((solver.laws.set.point, solver._parallel_contact_paths))
            return setup(solver, source)

        with (
            patch.object(SolverFeatherPGS, "_setup_sparse_mass_matrix", recording),
        ):
            for beta, law in ((0.0, "point"), (0.2, "patch")):
                with self.subTest(law=law):
                    solver = SolverFeatherPGS(model, friction_anchor_beta=beta)
                    # Both laws may run the parallel tiers.
                    self.assertEqual(seen[-1], (law == "point", True))
                    self.assertEqual(solver.laws.set.contact, law)


if __name__ == "__main__":
    unittest.main()
