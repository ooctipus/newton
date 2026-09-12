# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise private ownership using the original complete-step regression."""

import re
import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import compact_contact as compact
from newton._src.solvers.feather_pgs import compact_island_residual as residual
from newton._src.solvers.feather_pgs import fused_contact_solve as fused
from newton._src.solvers.feather_pgs import private_contact_islands as islands
from newton._src.solvers.feather_pgs import solver_feather_pgs
from newton.tests import test_feather_pgs_fused_contact_solve as original_controls
from newton.tests.test_feather_pgs_compact_contact import OUTPUTS, fixture, run_original


def partition_fixture(device="cpu", *, key_offsets=(6, 0)):
    """Use actual raw topology with 108 distinct keys and either dense-coordinate order."""
    data, _ = fixture(device)
    worlds = np.repeat(np.arange(2, dtype=np.int32), 128)
    for name, values in {
        "world": worlds,
        "slot": np.full(256, -1, np.int32),
        "path": np.full(256, -1, np.int32),
        "slots_needed": np.full(256, 3, np.int32),
        "shape0": np.full(256, -1, np.int32),
        "shape1": np.full(256, -1, np.int32),
        "art0": np.full(256, -1, np.int32),
        "art1": np.full(256, -1, np.int32),
        "shape_body": np.arange(218, dtype=np.int32),
        "body_single_dof": np.concatenate([np.r_[-1, world * 114 + np.arange(6, 114)] for world in range(2)]),
        "response_dofs": np.array([6, 108, 6, 108], np.int32),
        "dof_start": np.array([0, 6, 114, 120], np.int32),
    }.items():
        setattr(data, name, wp.array(values, dtype=int, device=device))
    data.dof_offset = wp.array(
        [coordinate for offset in key_offsets for coordinate in (0 if offset == 6 else 108, offset)],
        dtype=int,
        device=device,
    )
    data.point0 = wp.zeros(256, dtype=wp.vec3, device=device)
    data.count.assign(np.array([256], np.int32))
    routing = islands.allocate_routing(2, np.array(key_offsets)[:, None] + np.arange(108), device)
    routing.row_contact.fill_(-777)
    aux = SimpleNamespace(
        device=device,
        counts=wp.zeros(2, dtype=int, device=device),
        bounds=wp.zeros((2, 2), dtype=int, device=device),
        routing=routing,
    )
    return data, aux


def publish_topology(data, aux, pairs, *, prefix=(0, 0), reverse=False):
    """Endpoints use -1 for static, 108 for dense, and 0..107 for scalar keys."""
    values = {name: np.full(256, -1, np.int32) for name in ("slot", "path", "shape0", "shape1", "art0", "art1")}
    for world, contacts in enumerate(pairs):
        for ordinal, endpoints in enumerate(contacts):
            raw = 128 * world + (len(contacts) - ordinal - 1 if reverse else ordinal)
            values["slot"][raw] = prefix[world] + 3 * ordinal
            values["path"][raw] = 0
            for endpoint, key in enumerate(endpoints):
                if key >= 0:
                    values[f"shape{endpoint}"][raw] = world * 109 + (0 if key == 108 else key + 1)
                    values[f"art{endpoint}"][raw] = world * 2 + (0 if key == 108 else 1)
    for name, host in values.items():
        getattr(data, name).assign(host)
    data.slots_needed.fill_(3)
    data.count.assign(np.array([256], np.int32))
    aux.counts.assign(np.array([prefix[w] + 3 * len(pairs[w]) for w in range(2)], np.int32))
    aux.bounds.assign(np.array([[0, prefix[0]], [0, prefix[1]]], np.int32))


def partition(data, aux, *, invert=True):
    """Execute the actual native CPU branch, not a separate NumPy classifier."""
    if invert:
        wp.launch(fused.route_contacts, dim=256, inputs=[data, aux.routing.row_contact], device=aux.device)
    wp.launch_tiled(
        islands.get_partition_kernel("test"),
        dim=[2],
        inputs=[data, aux.counts, aux.bounds, aux.routing],
        block_dim=32,
        device=aux.device,
    )


def limit_solve_fixture(device):
    """Bind only the actual scalar-limit operands with nonuniform world coordinate maps."""
    solve = fused.SolveData()
    solve.world_dof_indices = wp.array(np.arange(228).reshape(2, 114), dtype=int, device=device)
    solve.v_out = wp.array(np.linspace(-0.7, 0.8, 228, dtype=np.float32), device=device)
    solve.diagonal_inverse_mass = wp.array(np.linspace(0.3, 0.9, 228, dtype=np.float32), device=device)
    solve.fused_limit_active = wp.array(np.tile(np.arange(114) % 4, (2, 1)).astype(np.int32), device=device)
    solve.fused_limit_lower_rhs = wp.full((2, 114), -0.2, dtype=float, device=device)
    solve.fused_limit_upper_rhs = wp.full((2, 114), 0.3, dtype=float, device=device)
    solve.fused_limit_lower_lambda = wp.full((2, 114), 17.0, dtype=float, device=device)
    solve.fused_limit_upper_lambda = wp.full((2, 114), 19.0, dtype=float, device=device)
    solve.fused_limit_cfm = 0.07
    solve.omega = 0.8
    solve.iterations = 8
    solve.friction_start_iteration = 2
    solve.iteration_offset = 0
    solve.world_constraint_count = wp.zeros(2, dtype=int, device=device)
    solve.dense_phase_bounds = wp.zeros((2, 2), dtype=int, device=device)
    solve.dense_offsets = wp.array([0, 108], dtype=int, device=device)
    solve.dense_groups = wp.array([1, 0], dtype=int, device=device)
    solve.world_impulses = wp.full((2, 704), -777.0, dtype=float, device=device)
    solve.sparse_contact_group_count = wp.zeros(2, dtype=int, device=device)
    solve.sparse_contact_group_heads = wp.full((2, 114), -1, dtype=int, device=device)
    solve.sparse_contact_serial_count = wp.zeros(2, dtype=int, device=device)
    solve.sparse_contact_serial_normals = wp.full((2, 235), -1, dtype=int, device=device)
    return solve


@wp.kernel
def evaluate_rhs(
    packets: wp.array[compact.ContactRow], solve: fused.SolveData, bias: fused.BiasData, out: wp.array[float]
):
    """Expose the actual shared bias function without a native-only solve dependency."""
    index = wp.tid()
    out[index] = islands.contact_rhs(solve, bias, 0, packets[index])


class TestPrivateContactIslands(unittest.TestCase):
    feature_switch = "_PRIVATE_CONTACT_ISLANDS"
    # Fifty-four dense contacts require at least 162 residual rows, beyond
    # the private 160-row panel but within the unchanged public 704-row cap.
    dense_fallback_contacts = 54

    def test_four_five_contacts_and_reserved_chain_closure(self):
        """A fifth independent contact rejects the whole world; coupled keys stay in the residual."""
        data, aux = partition_fixture()
        before = {name: getattr(data, name).numpy().copy() for name in OUTPUTS}
        publish_topology(data, aux, [[(7, -1)] * 4, [(7, -1)] * 5], reverse=True)
        partition(data, aux)
        np.testing.assert_array_equal(aux.routing.owner.numpy(), [1, 0])
        self.assertEqual(aux.routing.key_counts.numpy()[0, 7], 4)
        np.testing.assert_array_equal(aux.routing.key_contact_ids.numpy()[0, 7], [3, 2, 1, 0])
        np.testing.assert_array_equal(aux.routing.fallback_counts.numpy(), [0, 15])
        publish_topology(data, aux, [[(7, -1)] * 5 + [(108, 7)], [(7, 8), (7, -1), (8, -1)]], prefix=(2, 1))
        partition(data, aux)
        np.testing.assert_array_equal(aux.routing.owner.numpy(), [1, 1])
        np.testing.assert_array_equal(aux.routing.residual_key_count.numpy(), [1, 2])
        np.testing.assert_array_equal(aux.routing.residual_key_coordinates.numpy()[0, :1], [13])
        np.testing.assert_array_equal(aux.routing.residual_key_coordinates.numpy()[1, :2], [7, 8])
        np.testing.assert_array_equal(aux.routing.residual_row_count.numpy(), [20, 10])
        self.assertEqual(aux.routing.key_counts.numpy().sum(), 0)
        for name in OUTPUTS:
            np.testing.assert_array_equal(getattr(data, name).numpy(), before[name], err_msg=name)

    def test_sixteen_seventeen_reserved_and_160_161_residual_rows(self):
        """Independent capacity limits reject before publishing a partial selected owner."""
        data, aux = partition_fixture()
        publish_topology(data, aux, [[(108, key) for key in range(16)], [(108, key) for key in range(17)]])
        partition(data, aux)
        np.testing.assert_array_equal(aux.routing.owner.numpy(), [1, 0])
        np.testing.assert_array_equal(aux.routing.residual_key_coordinates.numpy()[0], np.arange(6, 22))
        np.testing.assert_array_equal(aux.routing.fallback_counts.numpy(), [0, 51])
        publish_topology(data, aux, [[(108, -1)] * 53] * 2, prefix=(1, 2), reverse=True)
        partition(data, aux)
        np.testing.assert_array_equal(aux.routing.owner.numpy(), [1, 0])
        np.testing.assert_array_equal(aux.routing.residual_row_ids.numpy()[0], np.arange(160))
        np.testing.assert_array_equal(aux.routing.residual_contact_ids.numpy()[0], np.arange(52, -1, -1))
        np.testing.assert_array_equal(aux.routing.fallback_counts.numpy(), [0, 161])
        np.testing.assert_array_equal(aux.routing.fallback_bounds.numpy()[1], [0, 2])

    def test_reuse_empty_prefix_and_nonuniform_current_raw_ownership(self):
        """Shrinking and growing partitions cannot reuse old key reservations or row ownership."""
        data, aux = partition_fixture()
        pointer = aux.routing.owner.ptr
        for pairs, prefix, residual_count in (
            ([[(108, 3), (3, -1), (4, -1)], [(108, 5)]], (2, 3), (8, 6)),
            ([[], []], (12, 0), (12, 0)),
            ([[(5, 5)], [(108, -1), (9, -1)]], (0, 1), (0, 4)),
            ([[], []], (0, 0), (0, 0)),
        ):
            publish_topology(data, aux, pairs, prefix=prefix, reverse=True)
            partition(data, aux)
            np.testing.assert_array_equal(aux.routing.owner.numpy(), [1, 1])
            np.testing.assert_array_equal(aux.routing.residual_row_count.numpy(), residual_count)
            np.testing.assert_array_equal(aux.routing.fallback_counts.numpy(), [0, 0])
            self.assertEqual(aux.routing.owner.ptr, pointer)
        np.testing.assert_array_equal(aux.routing.key_counts.numpy(), 0)
        np.testing.assert_array_equal(aux.routing.key_reserved.numpy(), 0)
        np.testing.assert_array_equal(aux.routing.key_contact_ids.numpy(), -1)

    def test_invalid_current_metadata_and_raw_overflow_keep_complete_fallback(self):
        """Stale slot/source topology and over-capacity counts never acquire selected ownership."""
        data, aux = partition_fixture()
        publish_topology(data, aux, [[(1, -1)], [(2, -1)]], prefix=(2, 3))
        partition(data, aux)
        for name, value in (
            ("slot", 5),
            ("path", -1),
            ("slots_needed", 2),
            ("world", 1),
            ("art0", 99),
            ("shape0", 999),
        ):
            original = getattr(data, name).numpy().copy()
            modified = original.copy()
            modified[0] = value
            getattr(data, name).assign(modified)
            partition(data, aux, invert=False)
            np.testing.assert_array_equal(aux.routing.owner.numpy(), [0, 1], err_msg=name)
            np.testing.assert_array_equal(aux.routing.fallback_counts.numpy(), [5, 0])
            getattr(data, name).assign(original)
        for counts in ([705, 706], [-1, 704]):
            aux.counts.assign(np.array(counts, np.int32))
            partition(data, aux, invert=False)
            np.testing.assert_array_equal(aux.routing.owner.numpy(), [0, 0])
            np.testing.assert_array_equal(aux.counts.numpy(), counts)
            np.testing.assert_array_equal(aux.routing.fallback_counts.numpy(), np.clip(counts, 0, 704))

    def test_static_coordinate_admission(self):
        """Only complete contiguous 108+6 local coordinate partitions are admitted."""
        for coordinates in (np.arange(107)[None], np.arange(108)[None] + 1, np.arange(108)[None, ::-1]):
            with self.assertRaises(ValueError):
                islands.allocate_routing(1, coordinates, "cpu")

    def test_private_packet_lane_addresses_and_shared_banks(self):
        """The fixed GPU packet layout separates live fields and equal-field banks across lanes."""
        source = islands._SCALAR_STORAGE.split("#else")[0]
        allocation = re.search(r"packet\[32 \* (\d+)\]", source)
        stride = re.search(r"threadIdx\.x \* (\d+)", source)
        self.assertIsNotNone(allocation)
        self.assertIsNotNone(stride)
        stride = int(stride.group(1))
        extent = 32 * int(allocation.group(1))
        addresses = np.arange(32)[:, None] * stride + np.arange(64)
        self.assertEqual(np.unique(addresses).size, 32 * 64)
        self.assertLess(addresses.max(), extent)
        for field in range(64):
            self.assertEqual(np.unique(addresses[:, field] % 32).size, 32)

    def test_original_bias_targets_limits_and_incident_restitution(self):
        """Prescribed targets and frozen incident velocity enter the unchanged original RHS law."""
        solve = fused.SolveData()
        solve.world_dof_indices = wp.array(np.arange(114)[None], dtype=int, device="cpu")
        solve.dense_offsets = wp.array([108], dtype=int, device="cpu")
        bias = fused.BiasData()
        bias.dt = 0.01
        bias.contact_speculative_scale = 0.75
        bias.joint_limit_speculative_scale = 0.5
        bias.restitution_threshold = 0.2
        incident = np.linspace(-1, 1, 114, dtype=np.float32)
        bias.incident = wp.array(incident, device="cpu")
        packets, expected = [], []
        for kind, phi, restitution in (
            (0, -0.02, 0),
            (0, 0.03, 0),
            (3, -0.02, 0),
            (3, 0.02, 0),
            (2, 0, 0),
            (0, 0, 0.6),
            (0, 1, 0.6),
        ):
            packet = compact.ContactRow()
            packet.kind, packet.phi, packet.restitution = kind, phi, restitution
            packet.target, packet.beta, packet.cfm = 0.4, 0.2, 123.0
            packet.dof0, packet.dof1 = 7, -1
            packet.j0, packet.j1 = 0.5, float("nan")
            packet.J = compact._Vec6(-1.0, 0, 0, 0, 0, 0)
            packets.append(packet)
            rhs = -0.4
            if kind == 0:
                rhs += (0.2 if phi <= 0 else 0.75) * phi / 0.01
            elif kind == 3:
                rhs += (0.2 if phi < 0 else 0.5) * phi / 0.01
            speed = -float(incident[108]) + 0.5 * float(incident[7]) - 0.4
            if kind == 0 and restitution and speed < -0.2 and (phi <= 1e-6 or phi + 0.01 * speed <= 1e-6):
                rhs = -0.4 + restitution * speed
            expected.append(rhs)
        actual = wp.empty(len(packets), dtype=float, device="cpu")
        wp.launch(
            evaluate_rhs,
            dim=len(packets),
            inputs=[wp.array(packets, dtype=compact.ContactRow, device="cpu"), solve, bias, actual],
            device="cpu",
        )
        np.testing.assert_allclose(actual.numpy(), expected, rtol=2e-6, atol=2e-6)

    def test_scalar_limit_only_current_lambda_and_disjoint_owners(self):
        """Execute eight original denominator-only CFM sweeps and never touch reserved/fallback keys."""
        data, aux = partition_fixture()
        publish_topology(data, aux, [[], []])
        partition(data, aux)
        aux.routing.owner.assign(np.array([1, 0], np.int32))
        reserved = np.zeros((2, 108), np.int32)
        reserved[0, 2] = 1
        aux.routing.key_reserved.assign(reserved)
        solve = limit_solve_fixture("cpu")
        initial = solve.v_out.numpy().copy()
        response = solve.diagonal_inverse_mass.numpy()
        active = solve.fused_limit_active.numpy()
        expected_v = initial.astype(np.float64)
        expected_lo = np.full((2, 114), 17.0)
        expected_hi = np.full((2, 114), 19.0)
        for key in range(108):
            if reserved[0, key]:
                continue
            coordinate = 6 + key
            lower, upper = 0.0, 0.0
            value = expected_v[coordinate]
            for _ in range(8):
                if active[0, coordinate] & 1:
                    updated = max(0.0, lower - 0.8 * (value - 0.2) / (response[coordinate] + 0.07))
                    value += response[coordinate] * (updated - lower)
                    lower = updated
                if active[0, coordinate] & 2:
                    updated = max(0.0, upper - 0.8 * (-value + 0.3) / (response[coordinate] + 0.07))
                    value -= response[coordinate] * (updated - upper)
                    upper = updated
            expected_v[coordinate] = value
            expected_lo[0, coordinate], expected_hi[0, coordinate] = lower, upper
        wp.launch(
            islands.get_scalar_kernel("test"),
            dim=(2, 108),
            inputs=[data, solve, fused.BiasData(), aux.routing],
            block_dim=32,
            device="cpu",
        )
        np.testing.assert_allclose(solve.v_out.numpy(), expected_v, rtol=3e-6, atol=3e-6)
        np.testing.assert_allclose(solve.fused_limit_lower_lambda.numpy(), expected_lo, rtol=3e-6, atol=3e-6)
        np.testing.assert_allclose(solve.fused_limit_upper_lambda.numpy(), expected_hi, rtol=3e-6, atol=3e-6)
        # An active-to-inactive transition must overwrite yesterday's reactions.
        solve.fused_limit_active.zero_()
        velocity = solve.v_out.numpy().copy()
        wp.launch(
            islands.get_scalar_kernel("test"),
            dim=(2, 108),
            inputs=[data, solve, fused.BiasData(), aux.routing],
            block_dim=32,
            device="cpu",
        )
        np.testing.assert_array_equal(solve.v_out.numpy(), velocity)
        selected = np.arange(6, 114)
        selected = selected[selected != 8]
        np.testing.assert_array_equal(solve.fused_limit_lower_lambda.numpy()[0, selected], 0)
        np.testing.assert_array_equal(solve.fused_limit_upper_lambda.numpy()[0, selected], 0)

    def test_scalar_four_contact_friction_response_without_canonical_coefficients(self):
        """Four loaded triples retain disk friction and action despite poisoned canonical coefficients."""
        data, aux = fixture(prefix=(0, 0), kept=(4, 0))
        for name, value in (("shape0", 2), ("shape1", -1), ("art0", 1), ("art1", -1)):
            values = getattr(data, name).numpy()
            values[:4] = value
            getattr(data, name).assign(values)
        data.material_restitution.zero_()
        point0, point1 = data.point0.numpy(), data.point1.numpy()
        point0[:4] = 0
        point1[:4] = data.body_q.numpy()[2, :3] - 0.05 * data.normal.numpy()[:4]
        data.point0.assign(point0)
        data.point1.assign(point1)
        run_original(data, aux)
        jy = data.sparse_jy.numpy()[0, :12].astype(np.float64).reshape(4, 3, 4)
        diagonal = data.diag.numpy()[0, :12].astype(np.float64).reshape(4, 3)
        mu = data.row_mu.numpy()[0, :12].reshape(4, 3)[:, 1].astype(np.float64)
        rhs = -data.target.numpy()[0, :12].astype(np.float64).reshape(4, 3)
        rhs[:, 0] += data.beta * data.phi.numpy()[0, :12:3] * 240
        routing = islands.allocate_routing(2, np.tile(np.arange(6, 114), (2, 1)), "cpu")
        wp.launch(fused.route_contacts, dim=16, inputs=[data, routing.row_contact], device="cpu")
        wp.launch_tiled(
            islands.get_partition_kernel("test"),
            dim=[2],
            inputs=[data, aux.counts, aux.bounds, routing],
            block_dim=32,
            device="cpu",
        )
        np.testing.assert_array_equal(routing.owner.numpy(), [1, 1])
        self.assertEqual(routing.key_counts.numpy()[0, 0], 4)
        solve = limit_solve_fixture("cpu")
        solve.fused_limit_active.zero_()
        solve.rhs_bias = wp.full((2, 704), float("nan"), dtype=float, device="cpu")
        solve.diagonal_inverse_mass = data.inverse_mass
        bias = fused.BiasData()
        bias.dt, bias.contact_speculative_scale = 1 / 240, 1.0
        velocity = float(solve.v_out.numpy()[6])
        impulses = np.zeros((4, 3))
        for iteration in range(8):
            for contact in range(4):
                for component in range(3):
                    if component and (
                        iteration < 2 or (impulses[contact, 0] <= 0 and not np.any(impulses[contact, 1:]))
                    ):
                        continue
                    old = impulses[contact, component]
                    proposed = (
                        old
                        - 0.8
                        * (jy[contact, component, 0] * velocity + rhs[contact, component])
                        / diagonal[contact, component]
                    )
                    if component == 0:
                        proposed = max(0.0, proposed)
                    else:
                        sibling = 3 - component
                        tangent = np.array([proposed, impulses[contact, sibling]])
                        radius = max(mu[contact] * impulses[contact, 0], 0.0)
                        magnitude = np.linalg.norm(tangent)
                        if magnitude > radius:
                            tangent *= radius / magnitude
                        proposed = tangent[0]
                        velocity += jy[contact, sibling, 1] * (tangent[1] - impulses[contact, sibling])
                        impulses[contact, sibling] = tangent[1]
                    velocity += jy[contact, component, 1] * (proposed - old)
                    impulses[contact, component] = proposed
        for name in ("J", "Y", "sparse_jy"):
            getattr(data, name).fill_(float("nan"))
        data.sparse_dof.fill_(-777)
        wp.launch(
            islands.get_scalar_kernel("test"),
            dim=(2, 108),
            inputs=[data, solve, bias, routing],
            block_dim=32,
            device="cpu",
        )
        self.assertGreater(np.max(impulses[:, 0]), 0.01)
        self.assertGreater(np.max(np.abs(impulses[:, 1:])), 0.001)
        np.testing.assert_allclose(solve.world_impulses.numpy()[0, :12].reshape(4, 3), impulses, rtol=2e-5, atol=5e-6)
        self.assertAlmostEqual(float(solve.v_out.numpy()[6]), velocity, delta=1e-5)
        np.testing.assert_array_equal(solve.world_impulses.numpy()[0, 12:], -777)
        np.testing.assert_array_equal(solve.world_impulses.numpy()[1], -777)
        self.assertTrue(np.isnan(data.J.numpy()).all())
        self.assertTrue(np.isnan(data.Y.numpy()).all())
        self.assertTrue(np.isnan(data.sparse_jy.numpy()).all())

    def test_cuda_partition_two_graphs_and_current_capacity_transitions(self):
        """Both captured graphs reread current topology and reject overflowing whole worlds."""
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("Captured routing requires CUDA")
        for device in devices:
            data, aux = partition_fixture(device)
            publish_topology(data, aux, [[(7, -1)] * 4, [(7, -1)] * 5])
            partition(data, aux)
            graphs = []
            for _ in range(2):
                with wp.ScopedCapture(device=device) as capture:
                    partition(data, aux)
                graphs.append(capture.graph)
            for pairs, prefix, expected in (
                ([[(108, key) for key in range(17)], [(108, key) for key in range(16)]], (0, 0), [0, 1]),
                ([[(108, -1)] * 53] * 2, (1, 2), [1, 0]),
                ([[], []], (12, 0), [1, 1]),
                ([[(7, -1)] * 5, [(7, -1)] * 4], (0, 3), [0, 1]),
            ):
                publish_topology(data, aux, pairs, prefix=prefix, reverse=True)
                for graph in graphs:
                    aux.routing.owner.fill_(-999)
                    wp.capture_launch(graph)
                    np.testing.assert_array_equal(aux.routing.owner.numpy(), expected)

    def test_cuda_residual_limits_match_original_and_preserve_isolated_coordinates(self):
        """The compact 22-coordinate map owns only dense/reserved limits for both dense offsets."""
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("The storage-remapped residual recurrence requires CUDA")
        for device in devices:
            data, aux = partition_fixture(device)
            publish_topology(data, aux, [[], []])
            partition(data, aux)
            coordinates = np.full((2, 16), -1, np.int32)
            coordinates[0, :2] = [8, 11]
            coordinates[1, :2] = [2, 7]
            aux.routing.residual_key_count.fill_(2)
            aux.routing.residual_key_coordinates.assign(coordinates)
            original, selected = limit_solve_fixture(device), limit_solve_fixture(device)
            before_v = selected.v_out.numpy().copy()
            _, original_kernel = fused.get_kernels(str(device.arch))
            wp.launch_tiled(
                original_kernel,
                dim=[2],
                inputs=[original, wp.zeros(2, dtype=int, device=device)],
                block_dim=32,
                device=device,
            )
            wp.launch_tiled(
                residual.get_kernel(str(device.arch)),
                dim=[2],
                inputs=[data, selected, fused.BiasData(), aux.routing],
                block_dim=32,
                device=device,
            )
            after_v = selected.v_out.numpy()
            for world, dense in enumerate((0, 108)):
                owned = np.r_[np.arange(dense, dense + 6), coordinates[world, :2]]
                excluded = np.setdiff1d(np.arange(114), owned)
                np.testing.assert_allclose(
                    after_v[world * 114 + owned], original.v_out.numpy()[world * 114 + owned], rtol=3e-6, atol=3e-6
                )
                np.testing.assert_array_equal(after_v[world * 114 + excluded], before_v[world * 114 + excluded])
                for name, sentinel in (("fused_limit_lower_lambda", 17), ("fused_limit_upper_lambda", 19)):
                    actual = getattr(selected, name).numpy()[world]
                    np.testing.assert_allclose(
                        actual[owned], getattr(original, name).numpy()[world, owned], rtol=3e-6, atol=3e-6
                    )
                    np.testing.assert_array_equal(actual[excluded], sentinel)
            np.testing.assert_array_equal(selected.world_impulses.numpy(), -777)

    def test_residual_storage_has_no_canonical_contact_coefficient_reader(self):
        """Unused compact coordinates stay guarded and contact operands remain privately mapped."""
        _, source = residual.sources("test")
        for reader in ("solve.dense_J.data[", "solve.dense_Y.data[", "solve.sparse_row_jy.data[", "row_w"):
            self.assertNotIn(reader, source)
        self.assertEqual(source.count("lane < 22 && s_physical[lane] >= 0"), 2)

    test_common_contact_rows_original_producers_and_float64_action = (
        original_controls.TestFusedContactSolve.test_masked_fallback_complete_original_law_and_float64_action
    )

    def test_experimental_owner_is_default_off(self):
        """The new owner must be explicitly selected, never silently replace E2."""
        self.assertFalse(solver_feather_pgs._PRIVATE_CONTACT_ISLANDS)

    test_cuda_actual_full_solve_mixed_fallback_reset_notify_and_graphs = (
        original_controls.TestFusedContactSolve.test_cuda_actual_full_solve_mixed_fallback_reset_notify_and_graphs
    )


if __name__ == "__main__":
    unittest.main()
