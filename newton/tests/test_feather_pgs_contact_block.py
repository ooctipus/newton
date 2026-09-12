# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Control current packed-contact ownership and sequential friction algebra."""

import unittest
from unittest import mock

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import compact_contact as compact
from newton._src.solvers.feather_pgs import contact_block_data as packed
from newton._src.solvers.feather_pgs import contact_block_owner, solver_feather_pgs
from newton._src.solvers.feather_pgs import contact_block_residual as residual
from newton._src.solvers.feather_pgs import fused_contact_solve as fused
from newton._src.solvers.feather_pgs import private_contact_islands as islands
from newton.tests import test_feather_pgs_fused_contact_solve as original_controls
from newton.tests import test_feather_pgs_private_contact_islands as private_controls
from newton.tests.test_feather_pgs_compact_contact import OUTPUTS, fixture, run_original


@wp.kernel
def evaluate_response_products(rows: wp.array[compact.ContactRow], output: wp.array[float]):
    """Execute the actual product function on controlled physical-coordinate operands."""
    index = wp.tid()
    output[index] = packed.response_product(rows[2 * index], rows[2 * index + 1])


def advance_contact(j, y, velocity, initial, rhs, diagonal, mu, *, local, omega=0.8, friction=True, wrong=False):
    """Compare direct sequential velocity updates with four explicit response products in FP64."""
    velocity, impulse = velocity.copy(), initial.copy()
    start = velocity.copy()
    residual = j @ velocity
    gram = j @ y.T
    if wrong:
        gram *= 1.25
    changes = []
    for component in range(3):
        if component and not friction:
            continue
        if diagonal[component] <= 0:
            continue
        old = impulse[component]
        speed = residual[component] if local else j[component] @ velocity
        proposed = old - omega * (speed + rhs[component]) / diagonal[component]
        if component == 0:
            proposed = max(proposed, 0.0)
        else:
            sibling = 3 - component
            radius = max(mu[component] * impulse[0], 0.0)
            if radius <= 0:
                proposed = 0.0
            else:
                magnitude = np.hypot(proposed, impulse[sibling])
                if magnitude > radius:
                    scale = radius / magnitude
                    proposed *= scale
                    sibling_new = impulse[sibling] * scale
                    delta = sibling_new - impulse[sibling]
                    impulse[sibling] = sibling_new
                    changes.append(delta)
                    if local:
                        residual[component + 1 :] += gram[component + 1 :, sibling] * delta
                    else:
                        velocity += y[sibling] * delta
        delta = proposed - old
        impulse[component] = proposed
        changes.append(delta)
        if local:
            residual[component + 1 :] += gram[component + 1 :, component] * delta
        else:
            velocity += y[component] * delta
    if local:
        velocity = start + (impulse - initial) @ y
    return velocity, impulse, changes


def partition_block(data, aux):
    """Run actual inversion and current partition before any tagged packet is produced."""
    wp.launch(
        packed.route_contacts,
        dim=data.slot.shape[0],
        inputs=[data, aux.routing.row_contact, aux.block],
        device=aux.device,
    )
    wp.launch_tiled(
        packed.get_partition_kernel("test"),
        dim=[2],
        inputs=[data, aux.counts, aux.bounds, aux.routing, aux.block],
        block_dim=32,
        device=aux.device,
    )


def bind_producer(data, aux):
    """Bind current original geometry metadata and the true 240-entry articulation map."""
    solve = private_controls.limit_solve_fixture(aux.device)
    solve.world_dof_indices = wp.array(
        np.r_[np.arange(114), np.arange(120, 234)].reshape(2, 114), dtype=int, device=aux.device
    )
    solve.dense_offsets = wp.zeros(2, dtype=int, device=aux.device)
    solve.dense_groups = data.dense_group
    solve.rhs_bias = wp.full(data.diag.shape, float("nan"), dtype=float, device=aux.device)
    bias = fused.BiasData()
    bias.dt = 0.005
    bias.contact_speculative_scale = 0.7
    bias.joint_limit_speculative_scale = 0.4
    bias.restitution_threshold = 0.2
    bias.incident = wp.array(np.linspace(-0.5, 0.5, 240, dtype=np.float32), device=aux.device)
    return solve, bias


def expanded_rows(data, world, slot):
    """Merge sparse coordinates by physical identity, not by endpoint-column coincidence."""
    j, y = np.zeros((3, 114)), np.zeros((3, 114))
    group = data.dense_group.numpy()[world]
    j[:, :6] = data.J.numpy()[group, slot : slot + 3]
    y[:, :6] = data.Y.numpy()[group, slot : slot + 3]
    dof, jy = data.sparse_dof.numpy(), data.sparse_jy.numpy()
    for component in range(3):
        for endpoint in range(2):
            coordinate = dof[world, slot + component, endpoint]
            if coordinate >= 0:
                j[component, coordinate] += jy[world, slot + component, 2 * endpoint]
                y[component, coordinate] += jy[world, slot + component, 2 * endpoint + 1]
    return j, y


def synthetic_solve(device, *, owner=(1, 1), wrong=False):
    """Bind loaded canonical packets with both dense offsets and nonidentity group order."""
    data = compact.ContactBoundaryData()
    solve = private_controls.limit_solve_fixture(device)
    solve.dense_offsets = wp.array([108, 0], dtype=int, device=device)
    active = np.zeros((2, 114), np.int32)
    active[0, [0, 108, 110]], active[1, [6, 0, 2]] = [3, 1, 2], [3, 1, 2]
    solve.fused_limit_active.assign(active)
    data.dense_group = solve.dense_groups
    solve.world_constraint_count.fill_(7)
    impulse_seed = solve.world_impulses.numpy()
    impulse_seed[:, :7] = 0.0
    solve.world_impulses.assign(impulse_seed)
    solve.dense_phase_bounds.assign(np.array([[0, 1], [0, 1]], np.int32))
    solve.sparse_contact_serial_count.fill_(2)
    serial = solve.sparse_contact_serial_normals.numpy()
    serial[:, :2] = [1, 4]
    solve.sparse_contact_serial_normals.assign(serial)
    shapes = {"J": (2, 704, 6), "Y": (2, 704, 6), "sparse_dof": (2, 704, 2), "sparse_jy": (2, 704, 4)}
    for name in OUTPUTS:
        dtype = int if name in ("row_type", "row_parent", "sparse_dof") else float
        setattr(
            data, name, wp.full(shapes.get(name, (2, 704)), -1 if dtype is int else 0.0, dtype=dtype, device=device)
        )
    mappings = {
        "world_diag": "diag",
        "world_row_type": "row_type",
        "world_row_parent": "row_parent",
        "world_row_mu": "row_mu",
        "dense_J": "J",
        "dense_Y": "Y",
        "sparse_row_dof": "sparse_dof",
        "sparse_row_jy": "sparse_jy",
    }
    for target, source in mappings.items():
        setattr(solve, target, getattr(data, source))
    solve.rhs_bias = wp.zeros((2, 704), dtype=float, device=device)
    rng = np.random.default_rng(20260912)
    j, y = data.J.numpy(), data.Y.numpy()
    dof, jy = data.sparse_dof.numpy(), data.sparse_jy.numpy()
    types, parents, mu = data.row_type.numpy(), data.row_parent.numpy(), data.row_mu.numpy()
    diagonal, rhs = data.diag.numpy(), solve.rhs_bias.numpy()
    gram = np.zeros((4, 4), np.float32)
    coordinates = np.full((2, 16), -1, np.int32)
    for world in range(2):
        group = (1, 0)[world]
        key = (0, 6)[world]
        coordinates[world, 0] = key
        types[world, :7] = [3, 0, 2, 2, 0, 2, 2]
        parents[world, :7] = [-1, -1, 1, 1, -1, 4, 4]
        mu[world, :7] = 0.55
        j[group, 0, 0], y[group, 0, 0] = 1.0, 0.7
        diagonal[world, 0], rhs[world, 0] = 0.77, -0.08
        for contact, slot in enumerate((1, 4)):
            jd = rng.normal(0, 0.5, (3, 6)).astype(np.float32)
            yd = (jd * np.linspace(0.4, 0.9, 6)).astype(np.float32)
            js = rng.normal(0, 0.4, 3).astype(np.float32)
            ys = js * np.float32(0.6)
            j[group, slot : slot + 3], y[group, slot : slot + 3] = jd, yd
            dof[world, slot : slot + 3, 0] = key
            jy[world, slot : slot + 3, :2] = np.stack((js, ys), axis=1)
            # Negative coordinates deliberately contain NaN coefficients.
            jy[world, slot : slot + 3, 2:] = np.nan
            diagonal[world, slot : slot + 3] = np.sum(jd * yd, axis=1) + js * ys + 0.07
            rhs[world, slot : slot + 3] = [-1.3, 1.7, -1.1]
            products = jd.astype(float) @ yd.astype(float).T + np.outer(js, ys)
            gram[2 * world + contact] = [products[1, 0], products[2, 0], products[2, 1], products[2, 2]]
    for name, host in (
        ("J", j),
        ("Y", y),
        ("sparse_dof", dof),
        ("sparse_jy", jy),
        ("row_type", types),
        ("row_parent", parents),
        ("row_mu", mu),
        ("diag", diagonal),
    ):
        getattr(data, name).assign(host)
    solve.rhs_bias.assign(rhs)
    target = data.target.numpy()
    target[:, 0] = 0.08
    data.target.assign(target)
    routing = islands.allocate_routing(2, np.array([0, 6])[:, None] + np.arange(108), device)
    routing.owner.assign(np.array(owner, np.int32))
    routing.residual_row_count.fill_(7)
    routing.residual_key_count.fill_(1)
    routing.residual_key_coordinates.assign(coordinates)
    ids = routing.residual_row_ids.numpy()
    ids[:, :7] = np.arange(7)
    routing.residual_row_ids.assign(ids)
    contacts = routing.residual_contact_ids.numpy()
    contacts[:, :2] = [[0, 1], [2, 3]]
    routing.residual_contact_ids.assign(contacts)
    block = packed.allocate_block_data(4, device)
    block.gram.assign(gram * (1.25 if wrong else 1.0))
    bias = fused.BiasData()
    bias.dt = 0.005
    bias.joint_limit_speculative_scale = 1.0
    return data, solve, bias, routing, block


class TestContactBlock(unittest.TestCase):
    feature_switch = "_CONTACT_BLOCK"
    dense_fallback_contacts = 54
    solve_dt = 0.005

    def test_default_off_selector_and_owner_module(self):
        """Require an explicit selector and a separately owned packed-contact implementation."""
        self.assertFalse(solver_feather_pgs._CONTACT_BLOCK)
        self.assertTrue(callable(contact_block_owner.ContactBlockIslands))

    def test_fp64_sequential_algebra_and_corrupted_products(self):
        """Retain sibling projection, denominator-only CFM and friction delay under reassociation."""
        rng = np.random.default_rng(72)
        rejected = 0
        for trial in range(120):
            j, y = rng.normal(size=(3, 8)), rng.normal(size=(3, 8))
            velocity, initial = rng.normal(size=8), rng.normal(size=3)
            initial[0] = abs(initial[0])
            rhs, diagonal = rng.normal(size=3), rng.uniform(0.3, 2, 3)
            diagonal += 0.17  # CFM is in the denominator, never in the residual.
            if trial % 7 == 0:
                diagonal[trial % 3] = 0
            if trial % 11 == 0:
                diagonal[trial % 3] = -1
            friction = trial % 5 != 0
            args = (j, y, velocity, initial, rhs, diagonal, np.array([0, 0.3, 0.7]))
            direct = advance_contact(*args, local=False, friction=friction)
            local = advance_contact(*args, local=True, friction=friction)
            wrong = advance_contact(*args, local=True, friction=friction, wrong=True)
            np.testing.assert_allclose(local[0], direct[0], rtol=2e-12, atol=2e-12)
            np.testing.assert_allclose(local[1], direct[1], rtol=2e-12, atol=2e-12)
            np.testing.assert_allclose(local[2], direct[2], rtol=2e-12, atol=2e-12)
            rejected += int(np.max(np.abs(wrong[0] - direct[0])) > 1e-5)
        self.assertGreater(rejected, 30)

    def test_intermediate_projection_change_survives_zero_net_impulse(self):
        """Keep stationary-exit evidence when the sibling projections cancel their final impulse changes."""
        initial = np.array([1.0, 0.6, 0.8])
        midpoint = np.sqrt(0.5)
        proposed_last = 0.8 * midpoint / 0.6
        rhs = np.array([0.0, -0.25, (midpoint - proposed_last) / 0.8 - (midpoint - 0.8)])
        args = (np.eye(3), np.eye(3), np.zeros(3), initial, rhs, np.ones(3), np.ones(3))
        direct = advance_contact(*args, local=False)
        local = advance_contact(*args, local=True)
        np.testing.assert_allclose(local[1], initial, rtol=0, atol=3e-16)
        np.testing.assert_allclose(local[0], direct[0], rtol=0, atol=3e-16)
        self.assertGreater(np.max(np.abs(local[2])), 0.1)

    def test_synthetic_fixture_uses_current_canonical_owners(self):
        """Compile the concrete descriptor fixture without depending on a native CPU no-op result."""
        data, solve, _, routing, block = synthetic_solve("cpu")
        self.assertEqual(solve.dense_J.ptr, data.J.ptr)
        self.assertEqual(solve.sparse_row_jy.ptr, data.sparse_jy.ptr)
        np.testing.assert_array_equal(solve.dense_offsets.numpy(), [108, 0])
        np.testing.assert_array_equal(solve.dense_groups.numpy(), [1, 0])
        np.testing.assert_array_equal(routing.residual_contact_ids.numpy()[:, :2], [[0, 1], [2, 3]])
        self.assertTrue(np.all(np.isfinite(block.gram.numpy())))
        self.assertTrue(np.isnan(solve.sparse_row_jy.numpy()[:, 1:7, 2:]).all())

    def test_response_products_merge_duplicate_and_permuted_sparse_coordinates(self):
        """Match physical coordinates across components, including duplicate slots and ignored NaNs."""
        rows, expected = [], []
        for left, right in (((8, 8), (8, 8)), ((8, 9), (9, 8)), ((8, -1), (-1, 8)), ((-1, -1), (8, 9))):
            jacobian, response = compact.ContactRow(), compact.ContactRow()
            jacobian.J = compact._Vec6(1, 2, 0, -1, 0.5, 0)
            response.Y = compact._Vec6(0.2, -0.1, 0, 0.3, 0.4, 0)
            jacobian.dof0, jacobian.dof1 = left
            response.dof0, response.dof1 = right
            jacobian.j0, jacobian.j1 = [
                value if coordinate >= 0 else float("nan") for value, coordinate in zip((0.4, 0.6), left, strict=True)
            ]
            response.y0, response.y1 = [
                value if coordinate >= 0 else float("nan") for value, coordinate in zip((2.0, 3.0), right, strict=True)
            ]
            j, y = np.zeros(114), np.zeros(114)
            j[:6], y[:6] = list(jacobian.J), list(response.Y)
            for coordinate, value in zip(left, (0.4, 0.6), strict=True):
                if coordinate >= 0:
                    j[coordinate] += value
            for coordinate, value in zip(right, (2.0, 3.0), strict=True):
                if coordinate >= 0:
                    y[coordinate] += value
            rows.extend((jacobian, response))
            expected.append(j @ y)
        actual = wp.zeros(len(expected), dtype=float, device="cpu")
        wp.launch(
            evaluate_response_products,
            dim=len(expected),
            inputs=[wp.array(rows, dtype=compact.ContactRow, device="cpu"), actual],
            device="cpu",
        )
        np.testing.assert_allclose(actual.numpy(), expected, rtol=2e-6, atol=2e-6)

    def test_zero_capacity_has_no_artificial_allocation_floor(self):
        """Keep both added allocations bounded exactly by the existing configured raw capacity."""
        block = packed.allocate_block_data(0, "cpu")
        self.assertEqual(block.raw_tag.shape, (0,))
        self.assertEqual(block.gram.shape, (0, 4))
        with self.assertRaises(ValueError):
            packed.allocate_block_data(-1, "cpu")

    def test_partition_changes_only_final_admitted_tag_publication(self):
        """Recover the original bounded partition and refuse a changed admission insertion seam."""
        self.assertEqual(packed.partition_source().replace(packed._TAG_PUBLICATION, ""), islands._PARTITION)
        with mock.patch.object(islands, "_PARTITION", islands._PARTITION.replace(packed._OWNER_PUBLICATION, "")):
            with self.assertRaises(RuntimeError):
                packed.partition_source()

    def test_current_tags_fallback_growing_shrinking_and_empty_prefix(self):
        """Tag only this call's admitted residual contacts, never an old raw-index lease."""
        data, aux = private_controls.partition_fixture()
        aux.block = packed.allocate_block_data(256, "cpu")
        for pairs, prefix, owners in (
            ([[(108, 2), (2, -1), (3, -1)], [(108, key) for key in range(17)]], (2, 0), [1, 0]),
            ([[], []], (12, 0), [1, 1]),
            ([[(7, -1)] * 5, [(108, -1)] * 53], (0, 1), [0, 1]),
            ([[(108, -1)] * 53, [(7, 7)]], (2, 0), [0, 1]),
        ):
            private_controls.publish_topology(data, aux, pairs, prefix=prefix, reverse=True)
            aux.block.raw_tag.fill_(777)
            partition_block(data, aux)
            np.testing.assert_array_equal(aux.routing.owner.numpy(), owners)
            expected = np.zeros(256, np.int32)
            rows = aux.routing.residual_row_count.numpy()
            raw = aux.routing.residual_contact_ids.numpy()
            for world in range(2):
                if owners[world]:
                    expected[raw[world, : (rows[world] - prefix[world]) // 3]] = 1
            np.testing.assert_array_equal(aux.block.raw_tag.numpy(), expected)
        data.count.zero_()
        aux.counts.zero_()
        aux.bounds.zero_()
        partition_block(data, aux)
        np.testing.assert_array_equal(aux.block.raw_tag.numpy(), 0)

    def test_packed_producer_original_geometry_and_four_products(self):
        """Compare all published coupled packets and merged-coordinate Gram products with original producers."""
        original, original_aux = fixture()
        candidate, aux = original_controls.wide_fixture()
        run_original(original, original_aux)
        aux.routing = islands.allocate_routing(2, np.tile(np.arange(6, 114), (2, 1)), "cpu")
        aux.block = packed.allocate_block_data(16, "cpu")
        aux.block.gram.fill_(-777)
        partition_block(candidate, aux)
        solve, bias = bind_producer(candidate, aux)
        before = {name: getattr(candidate, name).numpy().copy() for name in OUTPUTS}
        wp.launch(
            packed.produce_contacts, dim=16, inputs=[candidate, solve, bias, aux.routing, aux.block], device="cpu"
        )
        self.assertTrue(np.all(aux.routing.owner.numpy() == 1))
        tags = aux.block.raw_tag.numpy()
        self.assertGreater(np.count_nonzero(tags), 8)
        for raw in np.flatnonzero(tags):
            world, slot = int(candidate.world.numpy()[raw]), int(candidate.slot.numpy()[raw])
            for name in OUTPUTS:
                index = int(candidate.dense_group.numpy()[world]) if name in ("J", "Y") else world
                actual = getattr(candidate, name).numpy()[index, slot : slot + 3]
                expected = getattr(original, name).numpy()[index, slot : slot + 3]
                np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=3e-6, err_msg=f"{raw}/{name}")
            j, y = expanded_rows(candidate, world, slot)
            dof = candidate.sparse_dof.numpy()[world, slot : slot + 3]
            np.testing.assert_array_equal(dof, np.tile(dof[0], (3, 1)))
            self.assertTrue(dof[0, 0] < 0 or dof[0, 1] < 0 or dof[0, 0] != dof[0, 1])
            response = j @ y.T
            np.testing.assert_allclose(
                aux.block.gram.numpy()[raw], response[[1, 2, 2, 2], [0, 0, 1, 2]], rtol=3e-6, atol=3e-6
            )
            target = candidate.target.numpy()[world, slot : slot + 3].astype(float)
            rhs = -target
            phi = float(candidate.phi.numpy()[world, slot])
            rhs[0] += (candidate.beta if phi <= 0 else bias.contact_speculative_scale) * phi / bias.dt
            incident = bias.incident.numpy()[solve.world_dof_indices.numpy()[world]]
            speed = j[0] @ incident - target[0]
            restitution = float(candidate.restitution.numpy()[world, slot])
            if (
                restitution > 0
                and speed < -bias.restitution_threshold
                and (phi <= 1e-6 or phi + bias.dt * speed <= 1e-6)
            ):
                rhs[0] = -target[0] + restitution * speed
            np.testing.assert_allclose(solve.rhs_bias.numpy()[world, slot : slot + 3], rhs, rtol=3e-5, atol=3e-6)
        # A paired-key row has no six-coordinate response but still needs all
        # six explicit zeros in both canonical coefficient owners.
        slot = candidate.slot.numpy()[3]
        group = candidate.dense_group.numpy()[0]
        np.testing.assert_array_equal(candidate.J.numpy()[group, slot : slot + 3], 0)
        np.testing.assert_array_equal(candidate.Y.numpy()[group, slot : slot + 3], 0)
        for name in OUTPUTS:
            values = getattr(candidate, name).numpy()
            written = np.zeros(values.shape[:2], dtype=bool)
            for raw in np.flatnonzero(tags):
                world, slot = int(candidate.world.numpy()[raw]), int(candidate.slot.numpy()[raw])
                index = int(candidate.dense_group.numpy()[world]) if name in ("J", "Y") else world
                written[index, slot : slot + 3] = True
            np.testing.assert_array_equal(values[~written], before[name][~written], err_msg=name)
        np.testing.assert_array_equal(aux.block.gram.numpy()[tags == 0], -777)

    def test_cuda_native_sequential_control_and_wrong_gram_rejected(self):
        """Compare actual eight-sweep native recurrence with original GS and reject corrupted Gram data."""
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("The native contact-block recurrence requires CUDA")
        for device in devices:
            baseline, selected, wrong = [synthetic_solve(device, wrong=bad) for bad in (False, False, True)]
            _, original = fused.get_kernels(str(device.arch))
            wp.launch_tiled(
                original,
                dim=[2],
                inputs=[baseline[1], wp.zeros(2, dtype=int, device=device)],
                block_dim=32,
                device=device,
            )
            for inputs in (selected, wrong):
                wp.launch_tiled(
                    residual.get_kernel(str(device.arch)), dim=[2], inputs=list(inputs), block_dim=32, device=device
                )
            for field in ("v_out", "world_impulses"):
                np.testing.assert_allclose(
                    getattr(selected[1], field).numpy(), getattr(baseline[1], field).numpy(), rtol=3e-5, atol=5e-6
                )
            self.assertGreater(np.max(np.abs(wrong[1].v_out.numpy() - baseline[1].v_out.numpy())), 1e-4)

    def test_cuda_native_mixed_owners_prefix_empty_and_delayed_friction(self):
        """Keep unowned output bytes while matching original limits, empty rows and nonpositive denominators."""
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("The native contact-block recurrence requires CUDA")
        for device in devices:
            for count in (7, 1, 0):
                baseline = synthetic_solve(device)
                selected = synthetic_solve(device, owner=(0, 1))
                for inputs in (baseline, selected):
                    data, solve, _, routing, _ = inputs
                    solve.friction_start_iteration = 2
                    solve.world_constraint_count.fill_(count)
                    solve.dense_phase_bounds.fill_(0)
                    bounds = solve.dense_phase_bounds.numpy()
                    bounds[:, 1] = min(count, 1)
                    solve.dense_phase_bounds.assign(bounds)
                    solve.sparse_contact_serial_count.fill_(2 if count == 7 else 0)
                    routing.residual_row_count.fill_(count)
                    diagonal = data.diag.numpy()
                    diagonal[:, 2:4] = [0, -1]
                    data.diag.assign(diagonal)
                seed_v = selected[1].v_out.numpy().copy()
                seed_impulses = selected[1].world_impulses.numpy().copy()
                _, original = fused.get_kernels(str(device.arch))
                wp.launch_tiled(
                    original,
                    dim=[2],
                    inputs=[baseline[1], wp.zeros(2, dtype=int, device=device)],
                    block_dim=32,
                    device=device,
                )
                wp.launch_tiled(
                    residual.get_kernel(str(device.arch)), dim=[2], inputs=list(selected), block_dim=32, device=device
                )
                np.testing.assert_array_equal(selected[1].v_out.numpy()[:114], seed_v[:114])
                np.testing.assert_array_equal(selected[1].world_impulses.numpy()[0], seed_impulses[0])
                np.testing.assert_allclose(
                    selected[1].v_out.numpy()[114:], baseline[1].v_out.numpy()[114:], rtol=3e-5, atol=5e-6
                )
                np.testing.assert_allclose(
                    selected[1].world_impulses.numpy()[1], baseline[1].world_impulses.numpy()[1], rtol=3e-5, atol=5e-6
                )
                for name in ("fused_limit_lower_lambda", "fused_limit_upper_lambda"):
                    actual = getattr(selected[1], name).numpy()
                    expected = getattr(baseline[1], name).numpy()
                    np.testing.assert_allclose(actual[1, :7], expected[1, :7], rtol=3e-5, atol=5e-6)
                    np.testing.assert_array_equal(actual[1, 7:], 17 if "lower" in name else 19)

    def test_cuda_current_tags_two_graphs(self):
        """Replayed graphs rebuild current tags through opposed capacity and empty transitions."""
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("Captured tag lifecycle requires CUDA")
        for device in devices:
            data, aux = private_controls.partition_fixture(device)
            aux.block = packed.allocate_block_data(256, device)
            private_controls.publish_topology(data, aux, [[(108, 1)], [(108, 2)]])
            partition_block(data, aux)
            graphs = []
            for _ in range(2):
                with wp.ScopedCapture(device=device) as capture:
                    partition_block(data, aux)
                graphs.append(capture.graph)
            for pairs, prefix, owners in (
                ([[(108, key) for key in range(17)], [(108, key) for key in range(16)]], (0, 0), [0, 1]),
                ([[], []], (12, 0), [1, 1]),
                ([[(108, -1)] * 53] * 2, (1, 2), [1, 0]),
            ):
                private_controls.publish_topology(data, aux, pairs, prefix=prefix, reverse=True)
                for graph in graphs:
                    aux.block.raw_tag.fill_(777)
                    wp.capture_launch(graph)
                    np.testing.assert_array_equal(aux.routing.owner.numpy(), owners)
                    tags = aux.block.raw_tag.numpy()
                    for world in range(2):
                        expected = len(pairs[world]) if owners[world] else 0
                        self.assertEqual(np.count_nonzero(tags[world * 128 : (world + 1) * 128]), expected)
                    self.assertTrue(np.isin(tags, [0, 1]).all())

    test_cuda_actual_full_solve_mixed_fallback_reset_notify_and_graphs = (
        original_controls.TestFusedContactSolve.test_cuda_actual_full_solve_mixed_fallback_reset_notify_and_graphs
    )


if __name__ == "__main__":
    unittest.main()
