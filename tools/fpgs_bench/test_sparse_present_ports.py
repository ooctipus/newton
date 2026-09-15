# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check complete present-port ownership against the existing physical metric law."""

import json
import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from tools.fpgs_bench import test_g1_kinetic_state as kinetic
from tools.fpgs_bench.test_sparse_contact_block import full_rows, saved_records
from tools.fpgs_bench.test_sparse_factor import fixture, physical_rows, unpack
from tools.fpgs_bench.test_sparse_metric_tangents import METRIC_ENV, physical_metrics, reference
from tools.fpgs_bench.test_sparse_packet_rows import problem as physical_problem

PORT_ENV = {**METRIC_ENV, "FEATHER_PGS_SPARSE_PRESENT_PORTS": "1"}


def set_limits(case, distinct):
    """Present both signs for selected coordinates without duplicating their port."""
    solver, model = case["solver"], case["model"]
    qi = solver._joint_limit_q_index.numpy()
    dofs = np.flatnonzero(qi >= 0)[:distinct]
    lower, upper = np.full(43, -np.inf, np.float32), np.full(43, np.inf, np.float32)
    position = case["state"].joint_q.numpy()
    lower[dofs] = position[qi[dofs]] - np.float32(0.001)
    upper[dofs] = position[qi[dofs]] + np.float32(0.001)
    model.joint_limit_lower.assign(lower)
    model.joint_limit_upper.assign(upper)


def expected_mode(case, jacobian):
    """Count physically present endpoint blocks and unsigned limit coordinates."""
    solver, model, contacts = case["solver"], case["model"], case["contacts"]
    shape_body = model.shape_body.numpy()
    first, second = contacts.rigid_contact_shape0.numpy(), contacts.rigid_contact_shape1.numpy()
    slots, paths = solver.contact_slot.numpy(), solver.contact_path.numpy()
    bodies = set()
    for raw in range(int(contacts.rigid_contact_count.numpy()[0])):
        if slots[raw] < 0 or paths[raw] != 0:
            continue
        a, b = int(shape_body[first[raw]]), int(shape_body[second[raw]])
        if (a >= 0) == (b >= 0):
            return 0
        body = max(a, b)
        if body not in (6, 13, 14):
            return 0
        bodies.add(body)
    types = solver.row_type.numpy()[0, : len(jacobian)]
    limits = np.flatnonzero(np.any(jacobian[types == 3] != 0, axis=0))
    return int(6 * len(bodies) + len(limits) <= 32)


def prepare(case):
    """Build current metadata while poisoning response storage that fast worlds retire."""
    solver, owner = case["solver"], case["owner"]
    owner.data.Z.fill_(np.nan)
    owner.data.incident.fill_(np.nan)
    solver.diag.fill_(np.nan)
    owner.build_rows(case["state"], solver, case["contacts"], 0.0025)
    solver.check_constraint_capacity()
    solver._stage4_compute_rhs_world(0.0025)
    owner.restitution(0.0025)
    solver.impulses.zero_()


def check_physical(test, case, *, iterations=8, omega=1.0, friction_start=0, component_diagnostic=None):
    """Compare physical current rows and held response with the existing metric oracle."""
    solver, owner = case["solver"], case["owner"]
    count = int(solver.constraint_count.numpy()[0])
    jacobian, W = physical_rows(case), unpack(owner)
    rows = (W @ jacobian[:, ::-1].T).T
    vhat = solver.v_hat.numpy().copy()
    incoming = solver.impulses.numpy()[0, :count].copy()
    expected_rhs = physical_problem(case)[3]
    owner.solve(solver.rhs, iterations, omega, friction_start)
    owner.check()
    mode = int(owner.port_owner.data.mode.numpy()[0])
    test.assertEqual(mode, expected_mode(case, jacobian))
    if mode:
        test.assertTrue(np.isnan(owner.data.Z.numpy()).all(), "Fast worlds must not rebuild old row responses")
    else:
        fallback_rows, _ = full_rows(owner, count)
        np.testing.assert_allclose(fallback_rows, rows, rtol=3e-5, atol=3e-6)
    diagonal, rhs, types, parents, mu = (
        getattr(solver, name).numpy()[0, :count] for name in ("diag", "rhs", "row_type", "row_parent", "row_mu")
    )
    incident = owner.data.incident.numpy()[0, :count]
    np.testing.assert_allclose(rhs, expected_rhs, rtol=3e-5, atol=3e-6)
    expected_incident = jacobian @ vhat
    expected_diagonal = np.sum(rows * rows, axis=1) + solver.row_cfm.numpy()[0, :count]
    if component_diagnostic is None:
        np.testing.assert_allclose(incident, expected_incident, rtol=3e-5, atol=3e-6)
        np.testing.assert_allclose(diagonal, expected_diagonal, rtol=2e-5, atol=2e-6)
        incident_report = None
        diagonal_report = None
    else:
        # Saved-payload references inherit the original FP32 incident as their
        # same-input oracle. Accepted baseline fails this componentwise check
        # on the same seven near-cancelled rows; retain it as a diagnostic.
        incident_report = component_diagnostic(incident, expected_incident)
        # The original solver likewise fails the added diagonal comparison on
        # the same four hard saved inputs. Do not block their physical gates.
        diagonal_report = component_diagnostic(diagonal, expected_diagonal)
    du, expected_lambda, stats = reference(
        rows,
        rows,
        diagonal,
        rhs.astype(float) + incident,
        types,
        parents,
        mu,
        np.zeros(43),
        iterations=iterations,
        omega=omega,
        friction_start=friction_start,
        incoming=incoming,
    )
    actual, impulses = solver.v_out.numpy(), solver.impulses.numpy()[0, :count]
    expected = vhat + (W.T @ du)[::-1]
    np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=3e-5)
    np.testing.assert_allclose(impulses, expected_lambda, rtol=3e-4, atol=3e-5)
    committed = impulses.astype(float) - incoming
    if friction_start > 0 and iterations > 0:
        committed[types == 2] += incoming[types == 2]
    force = jacobian.T @ committed
    delta = actual.astype(float) - vhat
    mass = case["H"]
    momentum = np.linalg.norm(mass @ delta - force, np.inf) / (
        1 + np.linalg.norm(mass, np.inf) * np.linalg.norm(delta, np.inf) + np.linalg.norm(force, np.inf)
    )
    metrics = physical_metrics(jacobian, diagonal, rhs, types, parents, mu, vhat, actual, impulses)
    test.assertTrue(np.isfinite(actual).all() and np.isfinite(list(metrics.values())).all())
    test.assertLess(metrics["cone"], 3e-5)
    test.assertLess(momentum, 2e-6)
    return {
        "mode": mode,
        "incident_vs_physical": incident_report,
        "diagonal_vs_physical": diagonal_report,
        "momentum": float(momentum),
        "physical": metrics,
        "transactions": stats,
    }


class TestSparsePresentPortsCPU(unittest.TestCase):
    def test_explicit_default_off_owner(self):
        """Require explicit metric100 admission without changing the default owner."""
        with patch.dict(os.environ, {**METRIC_ENV, "FEATHER_PGS_SPARSE_PRESENT_PORTS": "0"}):
            ordinary = fixture("cpu")["owner"]
        self.assertIs(getattr(ordinary, "present_ports", None), False)
        self.assertIsNone(ordinary.port_owner)
        self.assertEqual(ordinary.kernels.solve.key, "sparse_metric_tangent43_s18_c100")
        with patch.dict(os.environ, PORT_ENV):
            candidate = fixture("cpu")["owner"]
        self.assertIs(candidate.present_ports, True)
        self.assertIsNotNone(candidate.port_owner)
        self.assertEqual(candidate.kernels.solve.key, "sparse_present_ports43_p32_c100")
        self.assertEqual(candidate.port_owner.data.geometry.shape, (1, 100, 6))
        self.assertEqual(candidate.port_owner.data.mode.shape, (1,))
        self.assertEqual(candidate.data.Z.shape, (1, 100, 18))
        self.assertEqual(candidate.data.incident.shape, (1, 100))
        for capacity, metric in ((101, "1"), (100, "0")):
            with self.subTest(capacity=capacity, metric=metric):
                with patch.dict(os.environ, {**PORT_ENV, "FEATHER_PGS_SPARSE_METRIC_TANGENTS": metric}):
                    with self.assertRaises(ValueError):
                        fixture("cpu", capacity=capacity)

    def test_current_geometry_and_metadata_only_prefix(self):
        """Build current signed anchors and stable limits without reading the held response."""
        with patch.dict(os.environ, PORT_ENV):
            case = fixture("cpu")
        solver, owner, contacts = case["solver"], case["owner"], case["contacts"]
        set_limits(case, 8)
        owner.data.W.fill_(np.nan)
        owner.data.Z.fill_(np.nan)
        owner.data.incident.fill_(np.nan)
        solver.diag.fill_(np.nan)
        original = {
            name: getattr(contacts, "rigid_contact_" + name).numpy().copy()
            for name in ("shape0", "shape1", "point0", "point1", "normal")
        }
        for shared, friction_shared, reverse in ((0, 0, False), (0, 1, False), (1, 0, True)):
            with self.subTest(shared=shared, friction_shared=friction_shared, reverse=reverse):
                for name, values in original.items():
                    getattr(contacts, "rigid_contact_" + name).assign(values)
                if reverse:
                    for first, second in (("shape0", "shape1"), ("point0", "point1")):
                        getattr(contacts, "rigid_contact_" + first).assign(original[second])
                        getattr(contacts, "rigid_contact_" + second).assign(original[first])
                    contacts.rigid_contact_normal.assign(-original["normal"])
                solver.contact_shared_anchor = bool(shared)
                solver.contact_friction_shared_anchor = bool(friction_shared)
                owner.port_owner.data.geometry.fill_(np.nan)
                owner.build_rows(case["state"], solver, contacts, 0.0025)
                solver.check_constraint_capacity()
                self.assertEqual(int(solver.constraint_count.numpy()[0]), 25)
                jacobian = physical_rows(case)
                np.testing.assert_array_equal(jacobian[:16:2], -jacobian[1:16:2])
                keys = owner.data.support.numpy()[0, :16]
                expected = -(np.arange(12, 28, dtype=np.int32) + 1)
                np.testing.assert_array_equal(keys, expected)
                self.assertTrue(np.isnan(owner.data.Z.numpy()).all())
                self.assertTrue(np.isnan(owner.data.incident.numpy()).all())
                self.assertTrue(np.isnan(solver.diag.numpy()).all())
                geometry = owner.port_owner.data.geometry.numpy()[0]
                slots = solver.contact_slot.numpy()
                for raw, body in enumerate((6, 13, 14)):
                    mask = int(owner.host["body_mask"][body])
                    basis = case["S"].T.copy()
                    for dof in range(43):
                        if not mask & (1 << dof):
                            basis[:, dof] = 0
                    slot = int(slots[raw])
                    np.testing.assert_allclose(
                        geometry[slot : slot + 3] @ basis,
                        jacobian[slot : slot + 3],
                        rtol=3e-5,
                        atol=3e-6,
                    )


@unittest.skipUnless(wp.is_cuda_available(), "Native present-port controls require CUDA")
class TestSparsePresentPortsCUDA(unittest.TestCase):
    def test_native_current_held_admission_and_metric_lifecycle(self):
        """Check endpoint signs, limit deduplication, guarded fallback and committed impulses."""
        with patch.dict(os.environ, PORT_ENV):
            case = fixture("cuda:0")
        solver, owner, contacts = case["solver"], case["owner"], case["contacts"]
        owner.refresh(solver)
        held = owner.data.W.numpy().copy()
        original_threshold = solver._effective_restitution_velocity_threshold
        original = {
            name: getattr(contacts, "rigid_contact_" + name).numpy().copy()
            for name in ("shape0", "shape1", "point0", "point1", "normal")
        }
        for kind in (
            "three_bodies",
            "normal_only",
            "static_reversal",
            "self",
            "dedup",
            "over32",
            "incoming",
            "delayed",
            "mu",
            "omega",
            "restitution",
        ):
            with self.subTest(kind=kind):
                for name, values in original.items():
                    getattr(contacts, "rigid_contact_" + name).assign(values)
                set_limits(case, 15 if kind == "over32" else 8 if kind == "dedup" else 0)
                solver.contact_gap_gate = float("inf")
                solver.enable_contact_friction = kind != "normal_only"
                solver._effective_restitution_velocity_threshold = 0.0 if kind == "restitution" else original_threshold
                solver.contact_shared_anchor = kind == "self"
                solver.contact_friction_shared_anchor = kind == "three_bodies"
                if kind == "static_reversal":
                    for first, second in (("shape0", "shape1"), ("point0", "point1")):
                        getattr(contacts, "rigid_contact_" + first).assign(original[second])
                        getattr(contacts, "rigid_contact_" + second).assign(original[first])
                    contacts.rigid_contact_normal.assign(-original["normal"])
                elif kind == "self":
                    other = original["shape1"].copy()
                    other[0] = original["shape0"][1]
                    contacts.rigid_contact_shape1.assign(other)
                    points = original["point1"].copy()
                    pose = wp.transform(*case["state"].body_q.numpy()[13])
                    points[0] = np.asarray(wp.transform_point(wp.transform_inverse(pose), wp.vec3(*points[0])))
                    contacts.rigid_contact_point1.assign(points)
                solver.mass_update_mask.zero_()
                solver.body_I_c.fill_(wp.spatial_matrix(np.nan))
                owner.refresh(solver)
                np.testing.assert_array_equal(owner.data.W.numpy(), held)
                points = contacts.rigid_contact_point0.numpy()
                points[:3, 0] += np.float32(0.007)
                contacts.rigid_contact_point0.assign(points)
                solver.v_hat.assign(np.random.default_rng(12).normal(0, 0.02, 43).astype(np.float32))
                prepare(case)
                count = int(solver.constraint_count.numpy()[0])
                if kind in ("dedup", "over32"):
                    self.assertEqual(count, (16 if kind == "dedup" else 30) + 9)
                    J = physical_rows(case)
                    np.testing.assert_array_equal(J[: count - 9 : 2], -J[1 : count - 9 : 2])
                if kind in ("incoming", "delayed"):
                    impulse = solver.impulses.numpy()
                    impulse[0, :3] = [0.3, 0.01, -0.01]
                    solver.impulses.assign(impulse)
                if kind == "mu":
                    mu = solver.row_mu.numpy()
                    mu[0, 2] *= np.float32(0.5)
                    solver.row_mu.assign(mu)
                if kind == "restitution":
                    direction = physical_rows(case)[0]
                    solver.v_hat.assign((-direction / (direction @ direction)).astype(np.float32))
                    phi = solver.phi.numpy()
                    phi[0, 0] = -0.0025 * float(direction @ solver.v_hat.numpy()) + 0.5e-6
                    solver.phi.assign(phi)
                    restitution = solver.row_restitution.numpy()
                    restitution[0, 0] = 0.7
                    solver.row_restitution.assign(restitution)
                report = check_physical(
                    self,
                    case,
                    friction_start=int(kind == "delayed"),
                    omega=1.2 if kind == "omega" else 1.0,
                )
                self.assertEqual(report["mode"], int(kind not in ("self", "over32")))
        set_limits(case, 0)
        contacts.rigid_contact_count.zero_()
        prepare(case)
        solver.v_out.fill_(np.nan)
        check_physical(self, case, iterations=0)
        np.testing.assert_array_equal(solver.v_out.numpy(), solver.v_hat.numpy())
        solver.constraint_count.fill_(101)
        solver.v_out.fill_(123.0)
        solver.impulses.fill_(456.0)
        owner.solve(solver.rhs, 8, 1.0, 0)
        self.assertEqual(int(owner.port_owner.data.mode.numpy()[0]), 0)
        np.testing.assert_array_equal(solver.v_out.numpy(), np.full(43, 123.0, np.float32))
        np.testing.assert_array_equal(solver.impulses.numpy(), np.full((1, 100), 456.0, np.float32))
        solver.constraint_count.zero_()
        owner.data.status.fill_(4)
        solver.v_out.fill_(123.0)
        solver.impulses.fill_(456.0)
        owner.solve(solver.rhs, 8, 1.0, 0)
        np.testing.assert_array_equal(solver.v_out.numpy(), np.full(43, 123.0, np.float32))
        np.testing.assert_array_equal(solver.impulses.numpy(), np.full((1, 100), 456.0, np.float32))
        with self.assertRaisesRegex(RuntimeError, "guard failed"):
            owner.check()

    def test_native_saved_sixteen_current_held_epochs(self):
        """Check all sixteen pinned physical rows without reading retired fast-world Z."""
        replay, records = saved_records()
        tested = 0
        for gpu, record, data in records:
            for index, world in enumerate(data["worlds"]):
                with self.subTest(gpu=gpu, step=record["step"], world=int(world)), patch.dict(os.environ, PORT_ENV):
                    case = replay.bind_world(record, data, index, "cuda:0")
                    owner = case["owner"]
                    lower = np.linalg.cholesky(case["H"][::-1, ::-1])
                    W = np.linalg.solve(lower, np.eye(43))
                    owner.data.W.assign(W[owner.host["row"], owner.host["col"]][None].astype(np.float32))
                    owner.data.valid.fill_(1)
                    prepare(case)
                    report = check_physical(self, case, component_diagnostic=replay.component_diagnostic)
                    print(
                        "present_ports_saved "
                        + json.dumps({"gpu_fixture": gpu, "step": record["step"], "world": int(world), **report}),
                        flush=True,
                    )
                    tested += 1
        self.assertEqual(tested, 16)

    def test_actual_five_world_mixed_reset_and_graph(self):
        """Reuse the complete kinetic fixture with fast, self-contact and over32-port worlds."""
        original_fixture, original_solver = kinetic.multiworld_fixture, kinetic.make_solver
        created = []

        def mixed_fixture(device):
            model, contacts = original_fixture(device)
            seed = model.state()
            kinetic.rotated_input(model, seed)
            shape_body = model.shape_body.numpy()
            other = contacts.rigid_contact_shape1.numpy()
            other[1] = int(np.flatnonzero(shape_body == 44 + 13)[0])
            contacts.rigid_contact_shape1.assign(other)
            points = contacts.rigid_contact_point1.numpy()
            pose = wp.transform(*seed.body_q.numpy()[44 + 13])
            points[1] = np.asarray(wp.transform_point(wp.transform_inverse(pose), wp.vec3(*points[1])))
            contacts.rigid_contact_point1.assign(points)
            first = contacts.rigid_contact_shape0.numpy()
            first[2] = int(np.flatnonzero(shape_body == 2 * 44 + 14)[0])
            contacts.rigid_contact_shape0.assign(first)
            points = contacts.rigid_contact_point1.numpy()
            local = contacts.rigid_contact_point0.numpy()[2]
            pose = wp.transform(*seed.body_q.numpy()[2 * 44 + 14])
            points[2] = np.asarray(wp.transform_point(pose, wp.vec3(*local)))
            points[2, 2] += np.float32(0.003)
            contacts.rigid_contact_point1.assign(points)
            lower, upper = model.joint_limit_lower.numpy(), model.joint_limit_upper.numpy()
            position = seed.joint_q.numpy().reshape(5, 44)
            lower[3 * 43 + 6 : 4 * 43] = position[3, 7:] - np.float32(0.001)
            upper[3 * 43 + 6 : 4 * 43] = position[3, 7:] + np.float32(0.001)
            model.joint_limit_lower.assign(lower)
            model.joint_limit_upper.assign(upper)
            return model, contacts

        def make_solver(model, enabled, **overrides):
            with patch.dict(os.environ, {"FEATHER_PGS_SPARSE_PRESENT_PORTS": str(int(enabled))}):
                result = original_solver(model, enabled, **overrides)
            created.append(result)
            return result

        self.device = "cuda:0"
        with (
            patch.object(kinetic, "multiworld_fixture", mixed_fixture),
            patch.object(kinetic, "make_solver", make_solver),
        ):
            kinetic.TestG1KineticStateCUDA.test_actual_steps_masked_refresh_reset_and_graph(self)
        self.assertIsNone(created[0]._sparse_factor.port_owner)
        np.testing.assert_array_equal(created[1]._sparse_factor.port_owner.data.mode.numpy(), [1, 0, 1, 0, 1])


if __name__ == "__main__":
    unittest.main()
