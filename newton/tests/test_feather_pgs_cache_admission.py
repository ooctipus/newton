# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Keep cached dynamics independent from the sparse constraint-row policy."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from newton._src.solvers.feather_pgs import packed_tree_owner, sparse_factor


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
                ("_world_scan_publication", object()),
            ):
                with self.subTest(field=field):
                    solver = self.fixture(False, False)
                    setattr(solver, field, value)
                    self.assertFalse(packed_tree_owner.supported(solver))


if __name__ == "__main__":
    unittest.main()
