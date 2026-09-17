# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check row-owned formation against the retained physical row producers."""

import importlib
import json
import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import sparse_limit_jacobi as jacobi
from newton._src.solvers.feather_pgs import sparse_packet_rows as packet
from tools.fpgs_bench import test_sparse_register_residual as register_tests
from tools.fpgs_bench.test_sparse_contact_block import full_rows, saved_records
from tools.fpgs_bench.test_sparse_factor import fixture, physical_rows, unpack
from tools.fpgs_bench.test_sparse_metric_tangents import physical_metrics

ENV = {**register_tests.ENV, "FEATHER_PGS_SPARSE_REGISTER_PACKETS": "1"}
DT = 0.0025


def prepare(f):
    """Build current keys and bias without forming accepted global rows."""
    s, owner = f["solver"], f["owner"]
    owner.build_rows(f["state"], s, f["contacts"], DT)
    s.check_constraint_capacity()
    s._stage4_compute_rhs_world(DT)
    owner.restitution(DT)
    s.impulses.zero_()


def prepared(device):
    """Use the actual tree and an independently assembled held kinetic operator."""
    with patch.dict(os.environ, ENV):
        f = fixture(device)
    s, owner = f["solver"], f["owner"]
    W = np.linalg.solve(np.linalg.cholesky(f["H"][::-1, ::-1]), np.eye(43))
    owner.data.W.assign(W[owner.host["row"], owner.host["col"]][None].astype(np.float32))
    owner.data.valid.fill_(1)
    owner.data.status.zero_()
    s.v_hat.assign(np.linspace(-0.11, 0.09, 43, dtype=np.float32))
    prepare(f)
    return f


def materialize(f):
    """Invoke the retained original row services with every world on route zero."""
    s, owner = f["solver"], f["owner"]
    route = owner.register_residual_routing
    route.zero_()
    wp.launch(
        owner.kernels.materialize_prefix,
        dim=(s.world_count, 96),
        inputs=[owner.plan, owner.data, s.constraint_count, s.v_hat, s.row_cfm, s.diag, route],
        block_dim=32,
        device=s.model.device,
    )
    if owner._register_packet_contacts is not None:
        workers, args = owner._register_packet_contacts
        wp.launch_tiled(
            owner.kernels.materialize_contacts,
            dim=[workers],
            inputs=[*args, route],
            block_dim=32,
            device=s.model.device,
        )
    wp.launch(
        owner.kernels.materialize_restitution,
        dim=(s.world_count, s.dense_max_constraints),
        inputs=[
            owner.data,
            s.constraint_count,
            s.row_type,
            s.phi,
            s.target_velocity,
            s.row_restitution,
            DT,
            s._effective_restitution_velocity_threshold,
            s.rhs,
            route,
        ],
        device=s.model.device,
    )


def check_native(test, f, *, iterations=8, omega=1.0, friction_start=0, J=None, expected_route=None, diagnose=False):
    """Compare original formation/solve and independently reconstruct physical momentum."""
    s, owner = f["solver"], f["owner"]
    test.assertTrue(owner.register_packets and owner.register_residual and not owner.packet_rows)
    module = importlib.import_module("newton._src.solvers.feather_pgs.sparse_register_packets")
    test.assertIs(owner.kernels.solve, module.get_solve_kernel())
    count = int(s.constraint_count.numpy()[0])
    J = physical_rows(f) if J is None else np.asarray(J, float)
    W = unpack(owner)
    Z = (W @ J[:, ::-1].T).T
    incoming = s.impulses.numpy().copy()
    vhat = s.v_hat.numpy().copy()
    keys, bias = owner.data.support.numpy().copy(), s.rhs.numpy().copy()
    metadata_names = (
        "constraint_count",
        "slot_counter",
        "dense_phase_bounds",
        "row_type",
        "row_parent",
        "row_mu",
        "row_cfm",
        "phi",
        "target_velocity",
    )
    metadata = {name: getattr(s, name).numpy().copy() for name in metadata_names}
    materialize(f)
    original_z, canonical = full_rows(owner, count)
    original_diag = s.diag.numpy().copy()
    original_incident = owner.data.incident.numpy().copy()
    original_rhs = s.rhs.numpy().copy()
    np.testing.assert_allclose(original_z, Z, rtol=3e-4, atol=3e-5)
    expected_lam, expected_v = wp.clone(s.impulses), wp.empty_like(s.v_out)
    wp.launch_tiled(
        jacobi.get_solve_kernel(),
        dim=[s.world_count],
        block_dim=32,
        inputs=register_tests.solve_inputs(
            f, expected_lam, expected_v, iterations=iterations, omega=omega, friction_start=friction_start
        ),
        device=s.model.device,
    )
    expected_lam, expected_v = expected_lam.numpy(), expected_v.numpy()
    owner.data.support.assign(keys)
    s.rhs.assign(bias)
    s.impulses.assign(incoming)
    owner.data.Z.fill_(123.25)
    owner.data.incident.fill_(321.5)
    s.diag.fill_(456.75)
    s.v_out.fill_(np.nan)
    owner.register_residual_routing.fill_(7)
    owner.solve(s.rhs, iterations, omega, friction_start)
    owner.check()
    actual, lam = s.v_out.numpy(), s.impulses.numpy()[0, :count]
    route = int(owner.register_residual_routing.numpy()[0])
    if expected_route is not None:
        test.assertEqual(route, expected_route)
    np.testing.assert_allclose(actual, expected_v, rtol=3e-4, atol=3e-5)
    np.testing.assert_allclose(lam, expected_lam[0, :count], rtol=3e-4, atol=3e-5)
    np.testing.assert_allclose(s.diag.numpy()[0, :count], original_diag[0, :count], rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(
        owner.data.incident.numpy()[0, :count], original_incident[0, :count], rtol=3e-5, atol=3e-6
    )
    np.testing.assert_allclose(s.rhs.numpy()[0, :count], original_rhs[0, :count], rtol=3e-5, atol=3e-6)
    np.testing.assert_array_equal(owner.data.support.numpy()[0, :count], canonical)
    for name, before in metadata.items():
        np.testing.assert_array_equal(getattr(s, name).numpy(), before, err_msg=name)
    if route == 1:
        np.testing.assert_array_equal(owner.data.Z.numpy(), np.full(owner.data.Z.shape, 123.25, np.float32))
    else:
        np.testing.assert_allclose(full_rows(owner, count)[0], original_z, rtol=0, atol=0)
    test.assertTrue(np.isfinite(actual).all() and np.isfinite(lam).all())
    applied = lam.astype(float) - incoming[0, :count]
    types = metadata["row_type"][0, :count]
    if friction_start > 0 and iterations > 0:
        # The original delayed branch clears incoming tangents without applying
        # their removal. Only subsequent changes contribute to momentum.
        applied[types == 2] += incoming[0, :count][types == 2]
    if diagnose:
        diagnostics = {}
        for name, velocity, impulse in (
            ("candidate", actual, lam),
            ("original", expected_v, expected_lam[0, :count]),
        ):
            change = impulse.astype(float) - incoming[0, :count]
            if friction_start > 0 and iterations > 0:
                change[types == 2] += incoming[0, :count][types == 2]
            physical_expected = vhat + (W.T @ (Z.T @ change))[::-1]
            rounded_expected = vhat + (W.T @ (original_z.T @ change))[::-1]
            delta = velocity.astype(float) - vhat.astype(float)
            force = J.T @ change
            diagnostics[name] = {
                "physical_JW_maxabs": float(np.max(np.abs(velocity - physical_expected), initial=0)),
                "physical_JW_tolerance_ratio": float(
                    np.max(np.abs(velocity - physical_expected) / (3e-6 + 3e-5 * np.abs(physical_expected)), initial=0)
                ),
                "original_rounded_Z_tolerance_ratio": float(
                    np.max(np.abs(velocity - rounded_expected) / (3e-6 + 3e-5 * np.abs(rounded_expected)), initial=0)
                ),
                "physical_H_backward": float(
                    np.linalg.norm(f["H"] @ delta - force, np.inf)
                    / (
                        1
                        + np.linalg.norm(f["H"], np.inf) * np.linalg.norm(delta, np.inf)
                        + np.linalg.norm(force, np.inf)
                    )
                ),
            }
        print("register_packet_momentum_diagnostic " + json.dumps(diagnostics), flush=True)
    # Retain the inherited actual rounded-row momentum gate. original_z was
    # saved from the independent producer BEFORE poisoning global Z. Using
    # FP64 reconstructed J here instead rejected the original control and
    # candidate identically on both7410 epochs; physical J/H is checked below
    # with the inherited backward-error gate, not this componentwise gate.
    np.testing.assert_allclose(actual, vhat + (W.T @ (original_z.T @ applied))[::-1], rtol=3e-5, atol=3e-6)
    stats = {"routing": route}
    if count:
        parents, mu = metadata["row_parent"][0, :count], metadata["row_mu"][0, :count]
        metrics = physical_metrics(
            J, original_diag[0, :count], original_rhs[0, :count], types, parents, mu, vhat, actual, lam
        )
        baseline = physical_metrics(
            J,
            original_diag[0, :count],
            original_rhs[0, :count],
            types,
            parents,
            mu,
            vhat,
            expected_v,
            expected_lam[0, :count],
        )
        test.assertTrue(np.isfinite(list(metrics.values())).all())
        test.assertLess(metrics["cone"], 3e-5)
        test.assertGreaterEqual(float(np.min(lam[np.isin(types, (0, 3))], initial=0)), -1e-7)
        stats.update(candidate=metrics, original=baseline)
    return actual.copy(), lam.copy(), stats


def seed_rows(f, count, *, incoming=False, ascending=False):
    """Author valid stable limit/contact keys across the small/fallback capacity boundary."""
    s, owner, contacts = f["solver"], f["owner"], f["contacts"]
    contact_count = 0 if count <= 13 or ascending else (6 if count == 100 else 3)
    prefix = count - 3 * contact_count
    if not 0 <= prefix <= 86:
        raise ValueError("Synthetic prefix exceeds the retained key contract")
    s.enable_joint_limits = False
    contacts.rigid_contact_count.fill_(contact_count)
    for suffix in ("shape0", "shape1", "point0", "point1", "normal", "margin0", "margin1"):
        array = getattr(contacts, "rigid_contact_" + suffix)
        values = array.numpy().copy()
        values[3:6] = values[:3]
        array.assign(values)
    for name in ("contact_art_a", "contact_art_b"):
        array = getattr(s, name)
        values = array.numpy().copy()
        values[3:6] = values[:3]
        array.assign(values)
    s.contact_slot.assign(np.r_[prefix + 3 * np.arange(contact_count), np.full(8 - contact_count, -1)].astype(np.int32))
    s.contact_world.zero_()
    s.contact_path.fill_(0)
    s.contact_slots_needed.assign(np.r_[np.full(contact_count, 3), np.zeros(8 - contact_count)].astype(np.int32))
    s.constraint_count.fill_(count)
    keys = np.zeros((1, 100), np.int32)
    types = np.zeros((1, 100), np.int32)
    parents = np.full((1, 100), -1, np.int32)
    keys[0, :prefix] = -1 - 2 * (42 - np.arange(prefix) % 37) - (np.arange(prefix) % 2)
    types[0, :prefix] = 3
    for raw in range(contact_count):
        row = prefix + 3 * raw
        keys[0, row : row + 3] = 3 * raw + np.arange(3)
        types[0, row : row + 3] = [0, 2, 2]
        parents[0, row + 1 : row + 3] = row
    owner.data.support.assign(keys)
    s.row_type.assign(types)
    s.row_parent.assign(parents)
    s.row_mu.fill_(0.6)
    s.row_cfm.fill_(0.01)
    s.row_restitution.zero_()
    s.target_velocity.zero_()
    s.phi.zero_()
    s.rhs.assign(np.tile(np.array([-0.2, 0.7, -0.8], np.float32), 34)[:100][None])
    s.impulses.zero_()
    if incoming:
        values = np.zeros((1, 100), np.float32)
        values[0, :count] = 0.01
        values[0, prefix:count:3] = 0.1
        s.impulses.assign(values)
    J = physical_rows(f)
    J[:prefix] = 0
    for row in range(prefix):
        key = -int(keys[0, row]) - 1
        J[row, key // 2] = -1 if key & 1 else 1
    if ascending:
        W = np.eye(43)
        W[42, :3] = 2
        owner.data.W.assign(W[owner.host["row"], owner.host["col"]][None].astype(np.float32))
        keys[0, :3] = -1 - 2 * np.array([42, 41, 40])
        owner.data.support.assign(keys)
        J[:] = 0
        J[np.arange(3), [42, 41, 40]] = 1
        s.v_hat.zero_()
        s.rhs.fill_(-5)
        s.row_cfm.zero_()
    return J


class TestSparseRegisterPackets(unittest.TestCase):
    def test_fallback_factory(self):
        """Require separately filtered original materialization factories."""
        module = importlib.import_module("newton._src.solvers.feather_pgs.sparse_register_packet_fallback")
        for name in ("get_prefix_kernel", "get_contact_kernel", "get_restitution_kernel"):
            factory = getattr(module, name)
            self.assertIs(factory(), factory())

    def test_factory(self):
        """Require the separately cached row-owned native factory."""
        module = importlib.import_module("newton._src.solvers.feather_pgs.sparse_register_packets")
        kernel = module.get_solve_kernel()
        self.assertIs(kernel, module.get_solve_kernel())
        self.assertEqual(kernel.key, "sparse_register_packets43_s18_c100")

    def test_actual_owner_and_flags(self):
        """Select the real producer/consumer pair only under its complete feature contract."""
        with patch.dict(os.environ, ENV):
            owner = fixture("cpu")["owner"]
        self.assertTrue(owner.register_packets and owner.register_residual and not owner.packet_rows)
        self.assertEqual(owner.data.Z.shape, (1, 100, 18))
        self.assertIs(owner.kernels.prefix, packet.get_prefix_kernel())
        self.assertIs(owner.kernels.contacts, packet.packet_contacts)
        for flag in (
            "FEATHER_PGS_SPARSE_REGISTER_RESIDUAL",
            "FEATHER_PGS_SPARSE_LIMIT_JACOBI",
            "FEATHER_PGS_SPARSE_SPECTRAL_TANGENTS",
        ):
            with patch.dict(os.environ, {**ENV, flag: "0"}), self.assertRaises(ValueError):
                fixture("cpu")
        with (
            patch.dict(os.environ, {**ENV, "FEATHER_PGS_SPARSE_REGISTER_PACKETS": "bad"}),
            self.assertRaises(ValueError),
        ):
            fixture("cpu")
        with patch.dict(os.environ, {**ENV, "FEATHER_PGS_SPARSE_REGISTER_PACKETS": "0"}):
            self.assertFalse(fixture("cpu")["owner"].register_packets)

    def test_cpu_prefix_filter_and_key_transaction(self):
        """Materialize valid negative keys only on route zero without altering allocation."""
        f = prepared("cpu")
        s, owner = f["solver"], f["owner"]
        seed_rows(f, 13)
        keys = owner.data.support.numpy().copy()
        counts, phase = s.constraint_count.numpy().copy(), s.dense_phase_bounds.numpy().copy()
        owner.data.Z.fill_(123.25)
        owner.register_residual_routing.fill_(1)
        args = [owner.plan, owner.data, s.constraint_count, s.v_hat, s.row_cfm, s.diag, owner.register_residual_routing]
        wp.launch(owner.kernels.materialize_prefix, dim=(1, 96), inputs=args, block_dim=32, device="cpu")
        np.testing.assert_array_equal(owner.data.support.numpy(), keys)
        np.testing.assert_array_equal(owner.data.Z.numpy(), np.full(owner.data.Z.shape, 123.25, np.float32))
        owner.register_residual_routing.zero_()
        wp.launch(owner.kernels.materialize_prefix, dim=(1, 96), inputs=args, block_dim=32, device="cpu")
        z, _ = full_rows(owner, 13)
        J = np.zeros((13, 43))
        for row, key in enumerate(keys[0, :13]):
            candidate = -int(key) - 1
            J[row, candidate // 2] = -1 if candidate & 1 else 1
        np.testing.assert_allclose(z, (unpack(owner) @ J[:, ::-1].T).T, rtol=3e-6, atol=3e-7)
        np.testing.assert_array_equal(s.constraint_count.numpy(), counts)
        np.testing.assert_array_equal(s.dense_phase_bounds.numpy(), phase)


@unittest.skipUnless(wp.is_cuda_available(), "Root owns native execution")
class TestSparseRegisterPacketsCUDA(unittest.TestCase):
    def test_native_rejection_preserves_transaction(self):
        """Preserve raw keys and all outputs until numerical fallback materializes them."""
        f = prepared("cuda:0")
        s, owner = f["solver"], f["owner"]
        seed_rows(f, 1)
        W = np.eye(43)
        W[0, 0] = 1e20
        owner.data.W.assign(W[owner.host["row"], owner.host["col"]][None].astype(np.float32))
        owner.data.Z.fill_(12.25)
        owner.data.incident.fill_(13.5)
        s.diag.fill_(14.75)
        s.v_out.fill_(15.0)
        arrays = [owner.data.support, owner.data.Z, owner.data.incident, s.diag, s.rhs, s.impulses, s.v_out]
        before = [array.numpy().copy() for array in arrays]
        owner.register_residual_routing.fill_(7)
        wp.launch_tiled(
            owner.kernels.solve,
            dim=[1],
            block_dim=32,
            inputs=[
                *register_tests.solve_inputs(f, s.impulses, s.v_out, iterations=1),
                owner.register_residual_routing,
                owner.packet_input,
            ],
            device="cuda:0",
        )
        np.testing.assert_array_equal(owner.register_residual_routing.numpy(), [0])
        for array, initial in zip(arrays, before, strict=True):
            np.testing.assert_array_equal(array.numpy(), initial)
        # Finite W produces an overflowing FP32 norm. The original producer's
        # infinite denominator performs no update; this control tests routing,
        # not admission of that operator as a production physical case.
        owner.solve(s.rhs, 1, 1.0, 0)
        np.testing.assert_array_equal(owner.register_residual_routing.numpy(), [0])
        self.assertTrue(np.isinf(s.diag.numpy()[0, 0]))
        np.testing.assert_array_equal(s.impulses.numpy(), before[-2])
        np.testing.assert_array_equal(s.v_out.numpy(), s.v_hat.numpy())
        self.assertGreaterEqual(int(owner.data.support.numpy()[0, 0]), 0)

    def test_native_graph_key_regeneration(self):
        """Replay one captured solve across small/large/empty worlds with fresh packet keys."""
        f = prepared("cuda:0")
        s, owner = f["solver"], f["owner"]
        graph = None
        for count in (31, 33, 32, 100, 0, 31):
            with self.subTest(count=count):
                J = seed_rows(f, count, incoming=True)
                keys, bias, incoming = (array.numpy().copy() for array in (owner.data.support, s.rhs, s.impulses))
                expected, expected_lam, _ = check_native(self, f, iterations=2, J=J)
                owner.data.support.assign(keys)
                s.rhs.assign(bias)
                s.impulses.assign(incoming)
                if graph is None:
                    with wp.ScopedCapture(device="cuda:0") as capture:
                        owner.solve(s.rhs, 2, 1.0, 0)
                    graph = capture.graph
                owner.register_residual_routing.fill_(7)
                wp.capture_launch(graph)
                np.testing.assert_array_equal(owner.register_residual_routing.numpy(), [int(count <= 32)])
                np.testing.assert_array_equal(s.v_out.numpy(), expected)
                np.testing.assert_array_equal(s.impulses.numpy()[0, :count], expected_lam)

    def test_native_thresholds_and_real_prefix_rejection(self):
        """Check actual formation at0/1/13/31/32/33/100 rows and ascending-energy rollback."""
        for count in (0, 1, 13, 31, 32, 33, 100):
            with self.subTest(count=count):
                f = prepared("cuda:0")
                J = seed_rows(f, count, incoming=True)
                check_native(self, f, iterations=2, J=J, expected_route=int(count <= 32))
        f = prepared("cuda:0")
        J = seed_rows(f, 3, ascending=True)
        _, lam, _ = check_native(self, f, iterations=1, J=J, expected_route=1)
        np.testing.assert_allclose(lam, [1, 0.2, 0.04], rtol=3e-5, atol=3e-6)

    def test_native_current_held_and_restitution(self):
        """Preserve current geometry, shared anchors, held W and the original impact trigger."""
        f = prepared("cuda:0")
        s, owner = f["solver"], f["owner"]
        held = owner.data.W.numpy().copy()
        for mode in ("current", "held", "shared", "friction_anchor", "impact"):
            with self.subTest(mode=mode):
                if mode == "held":
                    s.mass_update_mask.zero_()
                    s.body_I_c.fill_(wp.spatial_matrix(np.nan))
                    owner.refresh(s)
                    points = f["contacts"].rigid_contact_point0.numpy()
                    points[:3, 0] += 0.007
                    f["contacts"].rigid_contact_point0.assign(points)
                if mode in ("shared", "friction_anchor"):
                    s.contact_shared_anchor = mode == "shared"
                    s.contact_friction_shared_anchor = mode == "friction_anchor"
                    shapes = f["model"].shape_body.numpy()
                    values = f["contacts"].rigid_contact_shape1.numpy().copy()
                    values[0] = int(np.flatnonzero(shapes == 13)[0])
                    f["contacts"].rigid_contact_shape1.assign(values)
                    s.contact_gap_gate = float("inf")
                prepare(f)
                J = physical_rows(f)
                if mode == "impact":
                    row = int(np.flatnonzero(s.row_type.numpy()[0, : len(J)] == 0)[0])
                    velocity = -J[row] / (J[row] @ J[row])
                    s.v_hat.assign(velocity.astype(np.float32))
                    phi = s.phi.numpy()
                    relative = J[row] @ s.v_hat.numpy() - s.target_velocity.numpy()[0, row]
                    phi[0, row] = -DT * relative + 0.5e-6
                    s.phi.assign(phi)
                    bounce = s.row_restitution.numpy()
                    bounce[0, row] = 0.7
                    s.row_restitution.assign(bounce)
                check_native(self, f, J=J)
                np.testing.assert_array_equal(owner.data.W.numpy(), held)

    def test_native_incoming_delay_and_zero_allowance(self):
        """Check delta-only momentum, delayed tangent clearing and nonunit omega."""
        for iterations, omega, delay in ((8, 1.0, 0), (3, 1.0, 2), (2, 1.2, 0), (0, 1.0, 0)):
            with self.subTest(iterations=iterations, omega=omega, delay=delay):
                f = prepared("cuda:0")
                J = seed_rows(f, 31, incoming=True)
                check_native(self, f, iterations=iterations, omega=omega, friction_start=delay, J=J, expected_route=1)

    def test_native_saved_sixteen(self):
        """Reuse all sixteen captured current/held physical J/H fixtures without stale-Z reads."""
        replay, records = saved_records()
        tested = 0
        for gpu, record, data in records:
            for index, world in enumerate(data["worlds"]):
                with self.subTest(gpu=gpu, step=record["step"], world=int(world)), patch.dict(os.environ, ENV):
                    f = replay.bind_world(record, data, index, "cuda:0")
                    s, owner = f["solver"], f["owner"]
                    W = np.linalg.solve(np.linalg.cholesky(f["H"][::-1, ::-1]), np.eye(43))
                    owner.data.W.assign(W[owner.host["row"], owner.host["col"]][None].astype(np.float32))
                    owner.data.valid.fill_(1)
                    prepare(f)
                    J = physical_rows(f)
                    actual, lam, stats = check_native(self, f, J=J, diagnose=True)
                    delta = actual.astype(float) - s.v_hat.numpy().astype(float)
                    force = J.T @ lam.astype(float)
                    momentum = float(
                        np.linalg.norm(f["H"] @ delta - force, np.inf)
                        / (
                            1
                            + np.linalg.norm(f["H"], np.inf) * np.linalg.norm(delta, np.inf)
                            + np.linalg.norm(force, np.inf)
                        )
                    )
                    self.assertLess(momentum, 2e-6)
                    print(
                        "register_packet_saved_native "
                        + json.dumps(
                            {
                                "gpu_fixture": gpu,
                                "step": record["step"],
                                "world": int(world),
                                "momentum_defect": momentum,
                                **stats,
                            }
                        ),
                        flush=True,
                    )
                    tested += 1
        self.assertEqual(tested, 16)


if __name__ == "__main__":
    unittest.main()
