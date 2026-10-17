# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Keep cached dynamics independent from the sparse constraint-row policy."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import mixed_tree_kinetic, packed_tree_owner, sparse_factor
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
        articulated_contact_response="immediate",
        drive_mode="augmented",
        pgs_velocity_iterations=0,
        _connect_count=0,
        _has_loop_joints=False,
        _fk_id_cache_available=True,
        _async_augmented_drives=True,
        _parallel_augmented_drive_topology=True,
        _articulation_dynamics_stream=object(),
        _debug_buffers_enabled=False,
        _sparse_factor=None,
        _prismatic_publication=None,
        _compact_diagonal_mass_size=None,
        _execution_plan=SimpleNamespace(use_diagonal_mass=lambda size: False),
        _crba_cholesky_warp_kernels_by_size={9: object()},
        _crba_cholesky_kernels_by_size={},
    )


class TestSparseCacheAdmission(unittest.TestCase):
    def fixture(self, metric, parallel):
        """Describe an admitted operator and its current-state cache lifecycle."""
        return SimpleNamespace(
            model=SimpleNamespace(device=SimpleNamespace(is_cuda=True), requires_grad=False, particle_count=0),
            _sparse_factor=SimpleNamespace(
                level_update=True,
                metric_tangents=metric,
                parallel_limit_prefix=parallel,
                packet_rows=False,
                block_contacts=False,
            ),
            _fk_id_cache_enabled=True,
            _fk_id_cache_uses_snapshot=False,
            _fk_id_cache=None,
            _async_augmented_drives=True,
            _parallel_augmented_drive_topology=True,
            _articulation_dynamics_stream=object(),
            lazy_kinematics=False,
            _fused_k1=False,
        )

    def test_constraint_mapping_does_not_gate_state_cache(self):
        """Admit all row policies once the same operator and state lifecycle are legal."""
        with patch.object(sparse_factor, "supported", return_value=True):
            for metric in (False, True):
                for parallel in (False, True):
                    with self.subTest(metric=metric, parallel=parallel):
                        self.assertTrue(packed_tree_owner.supported(self.fixture(metric, parallel)))

    def test_operator_and_lifecycle_still_gate_state_cache(self):
        """Retain operator, epoch, stream and publication requirements for every row policy."""
        with patch.object(sparse_factor, "supported", return_value=False):
            self.assertFalse(packed_tree_owner.supported(self.fixture(False, False)))
        with patch.object(sparse_factor, "supported", return_value=True):
            for field, value in (
                ("_fk_id_cache_enabled", False),
                ("_fk_id_cache_uses_snapshot", True),
                ("_fk_id_cache", object()),
                ("_async_augmented_drives", False),
                ("_parallel_augmented_drive_topology", False),
                ("_articulation_dynamics_stream", None),
                ("lazy_kinematics", True),
                ("_fused_k1", True),
            ):
                with self.subTest(field=field):
                    solver = self.fixture(False, False)
                    setattr(solver, field, value)
                    self.assertFalse(packed_tree_owner.supported(solver))


class TestHeldTreeContactLaw(unittest.TestCase):
    def test_held_tree_owner_serves_point_and_patch_friction(self):
        """Admit the dense held-tree owner under either friction law, but not under torsion or compliance."""
        for law, admitted in (("point", True), ("patch", True), ("torsion", False), ("compliance", False)):
            with self.subTest(law=law):
                self.assertEqual(mixed_tree_kinetic.supported(_held_tree_solver(law)), admitted)

    def test_contact_law_predicates_precede_route_selection(self):
        """Define both contact-law predicates before sparse route selection can read them."""
        builder = newton.ModelBuilder()
        body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_articulation([builder.add_joint_revolute(-1, body)])
        model = builder.finalize(device="cpu")
        setup = SolverFeatherPGS._setup_sparse_mass_matrix
        seen = []

        def recording(solver, source):
            seen.append((solver._law_agnostic_contact_paths, solver._point_contact_paths))
            return setup(solver, source)

        with patch.object(SolverFeatherPGS, "_setup_sparse_mass_matrix", recording):
            for beta, law in ((0.0, "point"), (0.2, "patch")):
                with self.subTest(law=law):
                    solver = SolverFeatherPGS(model, friction_anchor_beta=beta)
                    self.assertEqual(seen[-1], _CONTACT_LAWS[law])
                    self.assertEqual(
                        (solver._law_agnostic_contact_paths, solver._point_contact_paths), _CONTACT_LAWS[law]
                    )


if __name__ == "__main__":
    unittest.main()
