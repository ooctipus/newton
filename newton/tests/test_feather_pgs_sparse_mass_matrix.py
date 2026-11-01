# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs.response_builds_rows import sparse_factor_kernel
from newton._src.solvers.feather_pgs.sparse_mass_matrix import factor_indices
from newton._src.solvers.feather_pgs.tree_plan import TreePlan


def _skew(v):
    return np.array(((0.0, -v[2], v[1]), (v[2], 0.0, -v[0]), (-v[1], v[0], 0.0)))


def _random_inertia(rng):
    """A random rigid body's inertia about the origin: its inertia record and its dense spatial matrix."""
    mass = rng.uniform(0.5, 2.0)
    com = rng.normal(scale=0.3, size=3)
    q = rng.normal(size=(3, 3))
    rotational = q @ q.T + 0.1 * np.eye(3) + mass * (com @ com * np.eye(3) - np.outer(com, com))
    moment = mass * com
    dense = np.block([[mass * np.eye(3), -_skew(moment)], [_skew(moment), rotational]])
    return np.concatenate(([mass], moment, rotational.reshape(-1))), dense


def _endpoint_support(plan, joint_a, joint_b=-1):
    """Sparse coordinates supporting either endpoint joint; -1 denotes the world."""
    mask = 0
    for joint in (joint_a, joint_b):
        if joint != -1:
            mask |= int(plan.joint_ancestor_mask[joint])
    return np.asarray([dof for dof in range(plan.dof_count) if mask & (1 << dof)], dtype=np.int32)


def _branched_humanoid_topology():
    """Return 43 DOFs with a free base, legs, spine, arms, and branching fingers."""
    parents, counts = [-1], [6]

    def chain(parent, length):
        for _ in range(length):
            parents.append(parent)
            counts.append(1)
            parent = len(parents) - 1
        return parent

    for _ in range(2):
        chain(0, 6)
    spine = chain(0, 3)
    for _ in range(2):
        hand = chain(spine, 7)
        for _ in range(4):
            chain(hand, 1)
    return parents, counts


class TestSparseMassMatrix(unittest.TestCase):
    def test_row_support_catalogue(self):
        """Preserve ordered endpoint unions and full multi-DOF limit support."""
        for parents, counts in (([-1, 0, 0, 1], [3, 2, 1, 0]), ([-1], [64])):
            tree = TreePlan.build(parents, counts)
            plan = tree.factor
            body_masks = np.concatenate((np.zeros(1, dtype=np.uint64), plan.joint_ancestor_mask))
            support = tree.support_classes(body_masks, plan.dof_count)
            self.assertIsNotNone(support)
            self.assertEqual(support.body_class[0], 0)
            for first, first_mask in enumerate(body_masks):
                for second, second_mask in enumerate(body_masks):
                    index = support.pair_class[support.body_class[first], support.body_class[second]]
                    mask = int(first_mask | second_mask)
                    expected = [dof for dof in range(plan.dof_count) if mask & (1 << dof)]
                    expected += [-1] * (plan.dof_count - len(expected))
                    np.testing.assert_array_equal(support.class_dof[index], expected)
            for dof, mask in enumerate(plan.ancestor_mask):
                expected = [i for i in range(plan.dof_count) if int(mask) & (1 << i)]
                expected += [-1] * (plan.dof_count - len(expected))
                np.testing.assert_array_equal(support.class_dof[support.limit_class[dof]], expected)
        tree = TreePlan.build([-1, 0, 0, 1], [3, 2, 1, 0])
        plan = tree.factor
        support = tree.support_classes(plan.joint_ancestor_mask, 6)
        np.testing.assert_array_equal(support.class_dof[support.limit_class[0]], [3, 4, 5, -1, -1, -1])
        # Structurally zero coefficients inside a joint block must retain their slots.
        np.testing.assert_array_equal(np.flatnonzero(plan.lookup[:, plan.inverse_permutation[0]] >= 0), [5])

    def test_row_support_rejects_oversized_union(self):
        """Decline a catalogue that would truncate a non-colliding body's union."""
        tree = TreePlan.build([-1, 0, 0], [6, 3, 3])
        self.assertIsNone(tree.support_classes(tree.factor.joint_ancestor_mask, 9))

    def test_factor_schedules_follow_dependencies(self):
        """Cover each packed factor entry once and schedule independent branch pivots together."""
        for parents, counts in [([2, -1, 1, 2, -1, 4], [2, 3, 0, 1, 1, 2]), _branched_humanoid_topology()]:
            plan = TreePlan.build(parents, counts).factor
            levels = np.empty(plan.dof_count, dtype=int)
            for level in range(plan.factor_level_count):
                begin, end = plan.factor_level_offsets[level : level + 2]
                levels[plan.factor_level_columns[begin:end]] = level
            np.testing.assert_array_equal(np.sort(plan.factor_level_columns), np.arange(plan.dof_count))
            np.testing.assert_array_equal(np.sort(plan.column_entries), np.arange(plan.nonzero_count))
            np.testing.assert_array_equal(
                np.sort(plan.factor_level_entries), np.flatnonzero(plan.entry_rows != plan.columns)
            )
            for row in range(plan.dof_count):
                begin, end = plan.row_offsets[row : row + 2]
                first = row - (end - begin) + 1
                np.testing.assert_array_equal(plan.columns[begin:end], np.arange(first, row + 1))
                self.assertTrue(np.all(levels[first:row] < levels[row]))
                begin, end = plan.column_offsets[row : row + 2]
                entries = plan.column_entries[begin:end]
                np.testing.assert_array_equal(plan.columns[entries], row)
                self.assertTrue(np.all(np.diff(plan.entry_rows[entries]) > 0))
            if plan.dof_count == 43:
                self.assertLess(plan.factor_level_count, plan.dof_count // 2)

    def test_forest_fixed_multidof_order(self):
        """Derive fill-free support for unordered forests and fixed joints."""
        plan = TreePlan.build([2, -1, 1, 2, -1, 4], [2, 3, 0, 1, 1, 2]).factor
        np.testing.assert_array_equal(np.sort(plan.permutation), np.arange(9))
        np.testing.assert_array_equal(plan.inverse_permutation[plan.permutation], np.arange(9))
        self.assertEqual(_endpoint_support(plan, 2).size, 3)
        self.assertEqual(_endpoint_support(plan, 0, 5).size, 8)
        rng = np.random.default_rng(54)
        lower = np.zeros((9, 9))
        lower[plan.entry_rows, plan.columns] = rng.normal(size=plan.nonzero_count)
        lower[np.diag_indices(9)] = 2.0
        inverse = np.linalg.inv(lower)
        np.testing.assert_allclose(inverse[plan.lookup < 0], 0.0, atol=1.0e-14)
        matrix = lower @ lower.T
        structural = (plan.lookup >= 0) | (plan.lookup.T >= 0)
        np.testing.assert_allclose(matrix[~structural], 0.0, atol=1.0e-14)

    def test_invalid_topology(self):
        """Reject cycles, invalid parents, counts, and unsupported mask widths."""
        for parents, counts in [
            ([1, 0], [1, 1]),
            ([2], [1]),
            ([-2], [1]),
            ([-1], [-1]),
            ([-1], [1.5]),
            ([-1], []),
            ([], []),
        ]:
            with self.subTest(parents=parents, counts=counts), self.assertRaises(ValueError):
                _ = TreePlan.build(parents, counts).factor

    def _factor(self, parents, counts, *, mask=None, armature=0.4, device="cpu"):
        """Factor two articulations' mass matrices from their packed entries, as the held tree publishes them, with
        the regularization and the drives' stiffness on the diagonal; returns the dense oracle with both."""
        plan = TreePlan.build(parents, counts).factor
        n, j = plan.dof_count, len(parents)
        rng = np.random.default_rng(742)
        motion = rng.normal(size=(2 * n, 6)).astype(np.float32)
        dense = np.zeros((2, n, n))
        for art in range(2):
            for joint in range(j):
                _, body_inertia = _random_inertia(rng)
                physical_support = plan.permutation[_endpoint_support(plan, joint)]
                jacobian = np.zeros((6, n))
                jacobian[:, physical_support] = motion[art * n + physical_support].T
                dense[art] += jacobian.T @ body_inertia @ jacobian
        drive_map = np.arange(2 * n, dtype=np.int32)
        drive_map[::3] = -1
        stiffness = rng.uniform(-0.1, 0.5, size=2 * n).astype(np.float32)
        regularizer = np.full((2, n), armature, dtype=np.float32)
        mapping = np.array([1, 0], dtype=np.int32)
        # The held geometry: each group's entries in the factor's packed order, without the diagonal terms.
        physical_rows, physical_cols = plan.permutation[plan.entry_rows], plan.permutation[plan.columns]
        geometric = np.stack([dense[art][physical_rows, physical_cols] for art in mapping]).astype(np.float32)
        for group, art in enumerate(mapping):
            diagonal = regularizer[group].astype(np.float64)
            diagonal += np.where(
                drive_map[art * n : (art + 1) * n] >= 0, np.maximum(stiffness[art * n : (art + 1) * n], 0.0), 0.0
            )
            dense[art] += np.diag(diagonal)

        def array(value, dtype):
            return wp.array(value, dtype=dtype, device=device)

        inverse = wp.full((2, plan.nonzero_count), -98.0, dtype=float, device=device)
        status = wp.full(2, 17, dtype=int, device=device)
        inputs = [
            array(mapping, int),
            array([1, 1] if mask is None else mask, int),
            array([0, n, 2 * n], int),
            array(geometric, float),
            array(regularizer, float),
            array(drive_map, int),
            array(stiffness, float),
            array(np.zeros(2 * n, dtype=np.float32), float),
            array(np.zeros(2 * n, dtype=np.float32), float),
            factor_indices(plan, device),
        ]
        wp.launch(
            sparse_factor_kernel(n, plan.nonzero_count),
            dim=64,
            inputs=inputs,
            outputs=[inverse, status],
            device=device,
            block_dim=128,
        )
        return plan, dense, inverse, status

    def _assert_whitens(self, plan, dense, inverse, tolerance):
        """Each group's ``L^-1`` whitens its articulation's dense oracle: ``L^-1 chol(P H P^T) = I``."""
        n = plan.dof_count
        for group, art in enumerate([1, 0]):
            whiten = np.zeros((n, n))
            whiten[plan.entry_rows, plan.columns] = inverse.numpy()[group]
            reordered = dense[art][np.ix_(plan.permutation, plan.permutation)]
            np.testing.assert_allclose(
                whiten @ np.linalg.cholesky(reordered), np.eye(n), rtol=tolerance, atol=tolerance
            )

    def test_factor_and_endpoint_response(self):
        """Match the dense oracle and its endpoint response for branched multi-DOF forests."""
        for parents, counts in [([-1, 0, 0, 1], [6, 1, 3, 0]), ([2, -1, 1, 2, -1, 4], [2, 3, 0, 1, 1, 2])]:
            with self.subTest(parents=parents):
                plan, dense, inverse, status = self._factor(parents, counts)
                np.testing.assert_array_equal(status.numpy(), [0, 0])
                self._assert_whitens(plan, dense, inverse, 2.0e-5)
                n = plan.dof_count
                rng = np.random.default_rng(66)
                for group, art in enumerate([1, 0]):
                    whiten = np.zeros((n, n))
                    whiten[plan.entry_rows, plan.columns] = inverse.numpy()[group]
                    support = _endpoint_support(plan, 0, len(parents) - 1)
                    row = np.zeros(n)
                    row[plan.permutation[support]] = rng.normal(size=len(support))
                    z = whiten @ row[plan.permutation]
                    np.testing.assert_allclose(np.delete(z, support), 0.0, atol=1.0e-12)
                    np.testing.assert_allclose(z @ z, row @ np.linalg.solve(dense[art], row), rtol=3.0e-5, atol=1.0e-6)

    def test_mask_bit_63_and_fixed_root(self):
        """Preserve the highest mask bit and fixed-root endpoint support."""
        plan = TreePlan.build([-1, 0], [0, 64]).factor
        self.assertEqual(int(plan.joint_ancestor_mask[0]), 0)
        self.assertEqual(int(plan.joint_ancestor_mask[1]), (1 << 64) - 1)
        self.assertEqual(int(plan.ancestor_mask[0]), (1 << 64) - 1)

    def _check_branched_factor(self, device):
        """Compare a broad articulated tree with the independent dense oracle."""
        plan, dense, inverse, status = self._factor(*_branched_humanoid_topology(), armature=2.0, device=device)
        np.testing.assert_array_equal(status.numpy(), [0, 0])
        self._assert_whitens(plan, dense, inverse, 1.0e-4)

    def test_branched_43_dof_factor(self):
        """Match the dense oracle for a broad 43-DOF articulated tree on CPU."""
        self._check_branched_factor("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "Sparse factor CUDA schedule requires CUDA")
    def test_branched_43_dof_factor_cuda(self):
        """Match the dense oracle with parallel factor dependencies and inverse columns."""
        self._check_branched_factor("cuda:0")

    def test_mask_preserves_factor(self):
        """Retain the inverse factor and status when an articulation is not refreshed."""
        _, _, inverse, status = self._factor([-1, 0, 0], [2, 1, 1], mask=[0, 1])
        np.testing.assert_array_equal(inverse.numpy()[1], -98.0)
        np.testing.assert_array_equal(status.numpy(), [0, 17])

    def test_invalid_pivot_is_not_regularized(self):
        """Report an invalid factor without adding a new pivot floor."""
        _, _, inverse, status = self._factor([-1], [1], armature=-1.0e6)
        np.testing.assert_array_equal(status.numpy(), [1, 1])
        self.assertTrue(np.isnan(inverse.numpy()).all())


if __name__ == "__main__":
    unittest.main()
