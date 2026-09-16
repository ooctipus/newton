# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check current body-basis publication into the retained sparse row ABI."""

import json
import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import body_basis_rows
from tools.fpgs_bench.test_sparse_contact_block import full_rows, saved_records
from tools.fpgs_bench.test_sparse_factor import fixture, physical_rows, unpack
from tools.fpgs_bench.test_sparse_metric_tangents import METRIC_ENV, check_native, physical_metrics

ENV = {**METRIC_ENV, "FEATHER_PGS_BODY_BASIS_ROWS": "0", "FEATHER_PGS_SPARSE_PARALLEL_LIMITS": "0"}
DT = 0.0025


def bind_factor(case):
    """Populate existing held-factor storage from the independent physical H."""
    owner = case["owner"]
    lower = np.linalg.cholesky(case["H"][::-1, ::-1])
    inverse = np.linalg.solve(lower, np.eye(43))
    owner.data.W.assign(inverse[owner.host["row"], owner.host["col"]][None].astype(np.float32))
    owner.data.valid.fill_(1)


def set_limits(case, number=2):
    """Use real active lower/upper limits before the contact suffix."""
    solver, model = case["solver"], case["model"]
    indices = solver._joint_limit_q_index.numpy()
    dofs = np.flatnonzero(indices >= 0)[:number]
    q = case["state"].joint_q.numpy()
    lower, upper = np.full(43, -np.inf, np.float32), np.full(43, np.inf, np.float32)
    lower[dofs], upper[dofs] = q[indices[dofs]] - 0.001, q[indices[dofs]] + 0.001
    model.joint_limit_lower.assign(lower)
    model.joint_limit_upper.assign(upper)


def build(case):
    """Execute real allocation, metadata and replacement publication."""
    solver, owner = case["solver"], case["owner"]
    owner.build_rows(case["state"], solver, case["contacts"], DT)
    solver.check_constraint_capacity()
    owner.check()
    solver._stage4_compute_rhs_world(DT)
    owner.restitution(DT)
    solver.impulses.zero_()


def check_rows(test, case, *, coefficients=True):
    """Check physical current J, held W, support identities and active row tails."""
    solver, owner, contacts = case["solver"], case["owner"], case["contacts"]
    count = int(solver.constraint_count.numpy()[0])
    J = physical_rows(case)
    z, support = full_rows(owner, count)
    expected = (unpack(owner) @ J[:, ::-1].T).T
    test.assertTrue(np.isfinite(z).all())
    packed = owner.data.Z.numpy()[0, :count]
    for row, template in enumerate(support):
        length = owner.host["support_count"][template]
        np.testing.assert_array_equal(packed[row, length:], 0)
    types = solver.row_type.numpy()[0, :count]
    for row in np.flatnonzero(types == 3):
        dof = int(np.flatnonzero(J[row])[0])
        test.assertEqual(support[row], owner.host["limit_support"][dof])
    shape_body = case["model"].shape_body.numpy()
    shapes = (contacts.rigid_contact_shape0.numpy(), contacts.rigid_contact_shape1.numpy())
    slots, paths, needed = (
        getattr(solver, name).numpy() for name in ("contact_slot", "contact_path", "contact_slots_needed")
    )
    for raw in range(int(contacts.rigid_contact_count.numpy()[0])):
        if paths[raw] != 0 or slots[raw] < 0:
            continue
        bodies = [int(shape_body[side[raw]]) for side in shapes]
        tags = [0 if body < 0 else int(owner.host["body_tag"][body]) for body in bodies]
        template = owner.host["pair_support"][tags[0], tags[1]]
        np.testing.assert_array_equal(support[slots[raw] : slots[raw] + needed[raw]], template)
    if coefficients:
        np.testing.assert_allclose(z, expected, rtol=3e-5, atol=3e-6)
        np.testing.assert_allclose(
            owner.data.incident.numpy()[0, :count], J @ solver.v_hat.numpy(), rtol=3e-5, atol=3e-6
        )
        np.testing.assert_allclose(
            solver.diag.numpy()[0, :count],
            np.sum(expected**2, axis=1) + solver.row_cfm.numpy()[0, :count],
            rtol=2e-5,
            atol=2e-6,
        )
    return J, z


def dynamic_pair(case):
    """Replace one static endpoint by an actual body at the same world point."""
    contacts = case["contacts"]
    shape1 = contacts.rigid_contact_shape1.numpy()
    shape1[0] = contacts.rigid_contact_shape0.numpy()[1]
    contacts.rigid_contact_shape1.assign(shape1)
    points = contacts.rigid_contact_point1.numpy()
    pose = wp.transform(*case["state"].body_q.numpy()[13])
    points[0] = np.asarray(wp.transform_point(wp.transform_inverse(pose), wp.vec3(*points[0])))
    contacts.rigid_contact_point1.assign(points)


def physical_solve(test, case):
    """Retain metric-law, cone and independently assembled momentum gates."""
    solver = case["solver"]
    count = int(solver.constraint_count.numpy()[0])
    J = physical_rows(case)
    diagonal, rhs, types, parents, mu = (
        getattr(solver, name).numpy()[0, :count] for name in ("diag", "rhs", "row_type", "row_parent", "row_mu")
    )
    actual, impulses, _ = check_native(test, case)
    vhat = solver.v_hat.numpy()
    metrics = physical_metrics(J, diagonal, rhs, types, parents, mu, vhat, actual, impulses)
    delta = actual.astype(float) - vhat.astype(float)
    force = J.T @ impulses.astype(float)
    defect = np.linalg.norm(case["H"] @ delta - force, np.inf) / (
        1 + np.linalg.norm(case["H"], np.inf) * np.linalg.norm(delta, np.inf) + np.linalg.norm(force, np.inf)
    )
    test.assertTrue(np.isfinite(list(metrics.values())).all())
    test.assertLess(metrics["cone"], 3e-5)
    test.assertLess(defect, 2e-6)
    return actual, impulses, metrics, float(defect)


class TestBodyBasisRows(unittest.TestCase):
    def test_factories(self):
        """Require the production replacement and its metadata owner."""
        self.assertTrue(callable(body_basis_rows.install))
        self.assertTrue(callable(body_basis_rows.get_contact_kernel))
        self.assertTrue(callable(body_basis_rows.get_prepare_kernel))

    def test_admission_keeps_original_solver_and_storage(self):
        """Install only the row owner, retaining original GS and existing arrays."""
        with patch.dict(os.environ, ENV):
            case = fixture()
            rejected = fixture(capacity=101)["owner"]
        owner = case["owner"]
        self.assertFalse(owner.body_basis_rows)
        owner.packet_rows = True
        self.assertFalse(body_basis_rows.install(owner))
        owner.packet_rows = False
        kernel = owner.kernels.solve
        arrays = [owner.data.W, owner.data.Z, owner.data.support, owner.data.incident]
        self.assertTrue(body_basis_rows.install(owner))
        self.assertTrue(owner.body_basis_rows)
        self.assertIs(owner.kernels.solve, kernel)
        for before, after in zip(
            arrays, (owner.data.W, owner.data.Z, owner.data.support, owner.data.incident), strict=True
        ):
            self.assertIs(before, after)
        old = rejected.kernels.contacts
        self.assertFalse(body_basis_rows.install(rejected))
        self.assertIs(rejected.kernels.contacts, old)

    def test_current_held_prefix_dynamic_pair_and_cache_fallback(self):
        """Publish actual endpoints, including prefix/tails and complete cache fallback."""
        with patch.dict(os.environ, ENV):
            case = fixture()
        solver, owner = case["solver"], case["owner"]
        bind_factor(case)
        set_limits(case)
        self.assertTrue(body_basis_rows.install(owner))
        held = owner.data.W.numpy().copy()
        solver.v_hat.assign(np.random.default_rng(12).normal(0, 0.02, 43).astype(np.float32))
        for variant in ("static", "current_screw", "dynamic_pair", "fallback_one", "fallback_zero", "normal_only"):
            with self.subTest(variant=variant):
                if variant == "current_screw":
                    # Probe current S independently of the held mass epoch.
                    current = solver.joint_S_s.numpy()
                    current[8, :3] += np.array([0.003, -0.002, 0.001], np.float32)
                    solver.joint_S_s.assign(current)
                    points = case["contacts"].rigid_contact_point0.numpy()
                    points[:3, 0] += 0.007
                    case["contacts"].rigid_contact_point0.assign(points)
                elif variant == "dynamic_pair":
                    dynamic_pair(case)
                elif variant.startswith("fallback_"):
                    owner.kernels.contacts = body_basis_rows.get_contact_kernel(int(variant == "fallback_one"))
                elif variant == "normal_only":
                    solver.enable_contact_friction = False
                owner.data.Z.fill_(np.nan)
                owner.data.incident.fill_(np.nan)
                solver.diag.fill_(np.nan)
                build(case)
                check_rows(self, case)
                self.assertEqual(np.count_nonzero(solver.row_type.numpy()[0, :4] == 3), 4)
                np.testing.assert_array_equal(owner.data.W.numpy(), held)


@unittest.skipUnless(wp.is_cuda_available(), "Native body-basis publication requires CUDA")
class TestBodyBasisRowsCUDA(unittest.TestCase):
    def test_native_current_held_fallback_and_graph(self):
        """Reuse captured rows across held/current, empty and regrown contact buffers."""
        with patch.dict(os.environ, ENV):
            case = fixture("cuda:0")
        solver, owner = case["solver"], case["owner"]
        owner.refresh(solver)
        held = owner.data.W.numpy().copy()
        self.assertTrue(body_basis_rows.install(owner))
        set_limits(case, 0)
        solver.v_hat.assign(np.random.default_rng(12).normal(0, 0.02, 43).astype(np.float32))
        build(case)
        check_rows(self, case)
        physical_solve(self, case)
        solver.mass_update_mask.zero_()
        solver.body_I_c.fill_(wp.spatial_matrix(np.nan))
        owner.refresh(solver)
        np.testing.assert_array_equal(owner.data.W.numpy(), held)
        points = case["contacts"].rigid_contact_point0.numpy()
        points[:3, 0] += 0.007
        case["contacts"].rigid_contact_point0.assign(points)
        for cache_capacity in (3, 1, 0):
            owner.kernels.contacts = body_basis_rows.get_contact_kernel(cache_capacity)
            build(case)
            check_rows(self, case)
            physical_solve(self, case)
        owner.kernels.contacts = body_basis_rows.get_contact_kernel()
        build(case)
        with wp.ScopedCapture(device="cuda:0") as capture:
            owner.build_rows(case["state"], solver, case["contacts"], DT)
            solver._stage4_compute_rhs_world(DT)
            owner.restitution(DT)
            solver.impulses.zero_()
            owner.solve(solver.rhs, 8, 1.0, 0)
        for count in (3, 0, 3):
            case["contacts"].rigid_contact_count.assign(np.array([count], np.int32))
            wp.capture_launch(capture.graph)
            solver.check_constraint_capacity()
            owner.check()
            actual = solver.v_out.numpy().copy()
            actual_impulses = solver.impulses.numpy().copy()
            check_rows(self, case)
            solver.impulses.zero_()
            expected, impulses, _, _ = physical_solve(self, case)
            np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=3e-5)
            np.testing.assert_allclose(actual_impulses[0, : len(impulses)], impulses, rtol=3e-4, atol=3e-5)
            if count == 0:
                self.assertEqual(int(solver.constraint_count.numpy()[0]), 0)
                np.testing.assert_array_equal(actual, solver.v_hat.numpy())
        dynamic_pair(case)
        build(case)
        check_rows(self, case)
        physical_solve(self, case)

    def test_native_saved_sixteen_current_held(self):
        """Check candidate-row recurrence and independent cone/momentum invariants."""
        replay, records = saved_records()
        tested = 0
        for gpu, record, data in records:
            for index, world in enumerate(data["worlds"]):
                with self.subTest(gpu=gpu, step=record["step"], world=int(world)), patch.dict(os.environ, ENV):
                    case = replay.bind_world(record, data, index, "cuda:0")
                    bind_factor(case)
                    self.assertTrue(body_basis_rows.install(case["owner"]))
                    build(case)
                    # Near-cancelled saved coefficients remain diagnostics;
                    # these are inherited physical gates, not bit identity.
                    J, z = check_rows(self, case, coefficients=False)
                    coefficient = replay.component_diagnostic(z, (unpack(case["owner"]) @ J[:, ::-1].T).T)
                    _, _, metrics, defect = physical_solve(self, case)
                    print(
                        "body_basis_saved_native "
                        + json.dumps(
                            {
                                "gpu": gpu,
                                "step": record["step"],
                                "world": int(world),
                                "coefficient": coefficient,
                                "physical": metrics,
                                "momentum": defect,
                            }
                        ),
                        flush=True,
                    )
                    tested += 1
        self.assertEqual(tested, 16)


if __name__ == "__main__":
    unittest.main()
