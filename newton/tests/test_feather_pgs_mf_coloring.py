# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check free-rigid contact coloring and the complete serial overflow path."""

import inspect
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs.mf_multiblock import MultiBlockContactSolver
from newton._src.solvers.feather_pgs.solver_feather_pgs import _get_pgs_solve_mf_gs_kernel
from newton.tests.feather_pgs_paired_fixture import pack_endpoints, solve_fixture


def fixture(world_contacts, device, *, max_colors=2, mode="colored", direct_static_pairs=False, parallel_prepare=False):
    """Bind mixed contact units with unequal masses and nontrivial world indexing."""
    worlds = len(world_contacts)
    bodies = 1 + max((max(a, b) for contacts in world_contacts for a, b, _ in contacts), default=0)
    rows = max(1, *(sum(3 if friction else 1 for _, _, friction in contacts) for contacts in world_contacts))
    dofs = 6 * bodies
    mapping = np.arange(worlds * dofs, dtype=np.int32).reshape(worlds, dofs)[::-1].copy()
    host = {
        "counts": np.zeros(worlds, np.int32),
        "meta": np.zeros((worlds, 4 * rows), np.int32),
        "J_a": np.zeros((worlds, rows, 6), np.float32),
        "J_b": np.zeros((worlds, rows, 6), np.float32),
        "MiJt_a": np.zeros((worlds, rows, 6), np.float32),
        "MiJt_b": np.zeros((worlds, rows, 6), np.float32),
        "mu": np.full((worlds, rows), 0.4, np.float32),
        "w": np.ones((worlds, rows), np.float32),
        "v": np.zeros(worlds * dofs, np.float32),
        "body_a": np.full((worlds, rows), -1, np.int32),
        "body_b": np.full((worlds, rows), -1, np.int32),
        "parent": np.full((worlds, rows), -1, np.int32),
        "kind": np.zeros((worlds, rows), np.int32),
        "rhs": np.zeros((worlds, rows), np.float32),
        "eff": np.zeros((worlds, rows), np.float32),
    }
    masses = 1.0 + 0.25 * np.arange(bodies)
    for world, contacts in enumerate(world_contacts):
        for body in range(bodies):
            host["v"][mapping[world, 6 * body : 6 * body + 3]] = [0.6 * (-1) ** body, -0.4, -1.0 + 0.4 * body]
        row = 0
        for a, b, friction in contacts:
            normal = row
            for axis in (2, 0, 1) if friction else (2,):
                kind, parent = (0, -1) if row == normal else (2, normal)
                eff = 0.0
                # Reversing endpoints also reverses the physical contact normal.
                orientation = -1.0 if a > b >= 0 and axis != 1 else 1.0
                for side, body, sign in (("a", a, 1.0), ("b", b, -1.0)):
                    if body >= 0:
                        host[f"body_{side}"][world, row] = world * bodies + body
                        host[f"J_{side}"][world, row, axis] = sign * orientation
                        host[f"MiJt_{side}"][world, row, axis] = sign * orientation / masses[body]
                        eff += 1.0 / masses[body]
                rhs = -0.2 if row == normal else 0.0
                host["eff"][world, row] = 1.0 / eff
                host["rhs"][world, row] = rhs
                host["kind"][world, row], host["parent"][world, row] = kind, parent
                host["meta"][world, 4 * row : 4 * row + 4] = [
                    pack_endpoints(6 * a if a >= 0 else -1, 6 * b if b >= 0 else -1),
                    np.float32(1.0 / eff).view(np.int32),
                    np.float32(rhs).view(np.int32),
                    pack_endpoints(parent, kind),
                ]
                row += 1
        host["counts"][world] = row
    arrays = {
        name: wp.array(value, dtype=wp.int32 if value.dtype == np.int32 else float, device=device)
        for name, value in host.items()
    }
    arrays["impulses"] = wp.zeros((worlds, rows), dtype=float, device=device)
    owner = MultiBlockContactSolver(
        worlds,
        rows,
        dofs,
        wp.array(mapping, dtype=int, device=device),
        device,
        max_colors=max_colors,
        mode=mode,
        direct_static_pairs=direct_static_pairs,
        parallel_prepare=parallel_prepare,
    )
    owner.prepare(arrays["counts"], arrays["counts"], arrays["meta"])
    return owner, host, arrays, mapping, masses


def ordered_rows(owner, host):
    """Expand the production color-plus-tail schedule without reordering a contact."""
    indices, offsets = owner.row_indices.numpy(), owner.pair_offsets.numpy()
    pairs, counts = owner.color_pairs.numpy(), owner.pair_count.numpy()
    orders = []
    for world, count in enumerate(counts):
        order = []
        for pair in pairs[world, :count]:
            for normal in indices[world, offsets[world, pair] : offsets[world, pair + 1]]:
                order.append(int(normal))
                for row in (int(normal) + 1, int(normal) + 2):
                    if (
                        row < host["counts"][world]
                        and host["kind"][world, row] == 2
                        and host["parent"][world, row] == normal
                    ):
                        order.append(row)
        orders.append(order)
    return orders


def solve(owner, arrays, *, iterations=8, friction_start=0, regularize=0):
    """Execute one full color schedule and its serial tail per unchanged PGS sweep."""
    arguments = [arrays[name] for name in ("meta", "impulses", "J_a", "J_b", "MiJt_a", "MiJt_b", "mu", "w", "v")]
    for iteration in range(iterations):
        options = [1.0, regularize, int(iteration >= friction_start)]
        for color in range(owner.max_colors):
            owner.solve_color(color, *arguments, *options)
        owner.solve_tail(*arguments, *options)


def serial(host, mapping, device, order, *, iterations=8, friction_start=0, regularize=0):
    """Independent float64 reference for the retained scalar/disk iteration.

    Main's serial kernel now minimizes a coupled tangent block. This
    experiment retains scalar updates plus disk projection, so compare its
    scheduling against its own stated recurrence, not main's changed sweep.
    """
    del device
    worlds, rows = host["kind"].shape
    bodies = mapping.shape[1] // 6
    impulses = np.zeros((worlds, rows), np.float64)
    velocity = host["v"].astype(np.float64)

    def endpoints(world, row):
        for side in ("a", "b"):
            body = int(host[f"body_{side}"][world, row])
            if body >= 0:
                first = 6 * (body % bodies)
                indices = mapping[world, first : first + 6]
                valid = indices >= 0
                yield side, indices[valid], valid

    def apply(world, row, delta):
        for side, indices, valid in endpoints(world, row):
            velocity[indices] += host[f"MiJt_{side}"][world, row, valid].astype(np.float64) * delta

    for iteration in range(iterations):
        for world, permutation in enumerate(order):
            for row in permutation:
                kind = host["kind"][world, row]
                if kind == 2 and iteration < friction_start:
                    impulses[world, row] = 0.0
                    continue
                inverse = float(host["eff"][world, row])
                if inverse <= 0:
                    continue
                residual = float(host["rhs"][world, row])
                for side, indices, valid in endpoints(world, row):
                    residual += host[f"J_{side}"][world, row, valid] @ velocity[indices]
                old = impulses[world, row]
                weight = float(host["w"][world, row]) if regularize and kind == 0 else 1.0
                updated = weight * (old - residual * inverse)
                if kind == 0:
                    updated = max(updated, 0.0)
                else:
                    parent = int(host["parent"][world, row])
                    sibling = parent + (2 if row == parent + 1 else 1)
                    radius = max(float(host["mu"][world, row]) * impulses[world, parent], 0.0)
                    other = impulses[world, sibling]
                    magnitude = np.hypot(updated, other)
                    if radius <= 0:
                        updated = 0.0
                    elif magnitude > radius:
                        updated *= radius / magnitude
                        projected = other * radius / magnitude
                        impulses[world, sibling] = projected
                        apply(world, sibling, projected - other)
                impulses[world, row] = updated
                apply(world, row, updated - old)
    return velocity, impulses


class TestFeatherPGSMFColoring(unittest.TestCase):
    """Run the real preparation and scalar projection on CPU and the leased GPU."""

    device = "cpu"

    def assert_schedule(self, owner, host):
        """Require complete contact units, unique pair ownership and disjoint write sets."""
        orders = ordered_rows(owner, host)
        pair_count, pair_a, pair_b = owner.pair_count.numpy(), owner.pair_a.numpy(), owner.pair_b.numpy()
        colors, pairs, offsets = owner.pair_color.numpy(), owner.color_pairs.numpy(), owner.color_offsets.numpy()
        for world, order in enumerate(orders):
            self.assertEqual(sorted(order), list(range(int(host["counts"][world]))))
            self.assertEqual(sorted(pairs[world, : pair_count[world]].tolist()), list(range(int(pair_count[world]))))
            self.assertEqual(int(offsets[world, -1]), int(pair_count[world]))
            for color in range(owner.max_colors):
                owned = set()
                for pair in pairs[world, offsets[world, color] : offsets[world, color + 1]]:
                    self.assertEqual(int(colors[world, pair]), color)
                    endpoints = {int(value) for value in (pair_a[world, pair], pair_b[world, pair]) if value >= 0}
                    self.assertFalse(owned & endpoints)
                    owned.update(endpoints)
        return orders

    def test_pair_grouping_static_endpoints_and_empty_world(self):
        """Keep reversed pairs together and ignore shared static or kinematic endpoints."""
        contacts = [[(0, 1, True), (2, 3, False), (1, 0, False), (0, -1, True), (3, -1, False)], []]
        owner, host, arrays, _, _ = fixture(contacts, self.device)
        self.assert_schedule(owner, host)
        np.testing.assert_array_equal(owner.pair_count.numpy(), [4, 0])
        np.testing.assert_array_equal(owner.unit_count.numpy(), [5, 0])
        arrays["counts"].zero_()
        owner.prepare(arrays["counts"], arrays["counts"], arrays["meta"])
        np.testing.assert_array_equal(owner.pair_count.numpy(), [0, 0])
        np.testing.assert_array_equal(owner.color_offsets.numpy(), 0)
        np.testing.assert_array_equal(owner.tail_count.numpy(), [0, 0])
        shared, shared_host, _, _, _ = fixture([[(0, -1, True), (1, -1, False), (2, -1, True)]], self.device)
        self.assert_schedule(shared, shared_host)
        np.testing.assert_array_equal(shared.pair_color.numpy()[0, :3], [0, 0, 0])

    def test_high_degree_serial_tail_retains_all_rows(self):
        """Solve every overflowing pair and mixed row unit without dropping a contact."""
        owner, host, arrays, mapping, _ = fixture([[(0, body, body % 2 == 0) for body in range(1, 8)]], self.device)
        order = self.assert_schedule(owner, host)
        self.assertEqual(int(owner.tail_count.numpy()[0]), 5)
        self.assertEqual(int(owner.unit_count.numpy()[0]), 7)
        solve(owner, arrays)
        expected_v, expected_impulse = serial(host, mapping, self.device, order)
        np.testing.assert_allclose(arrays["v"].numpy(), expected_v, rtol=3e-5, atol=3e-6)
        np.testing.assert_allclose(arrays["impulses"].numpy(), expected_impulse, rtol=3e-5, atol=3e-6)

    def test_static_endpoint_on_either_side_matches_serial(self):
        """Apply active normal and tangent impulses with either canonical endpoint fixed."""
        owner, host, arrays, mapping, _ = fixture([[(0, -1, True), (-1, 1, True)]], self.device)
        host["v"][mapping[0, 8]] = 0.8
        arrays["v"].assign(host["v"])
        order = self.assert_schedule(owner, host)
        solve(owner, arrays)
        expected_v, expected_impulse = serial(host, mapping, self.device, order)
        np.testing.assert_allclose(arrays["v"].numpy(), expected_v, rtol=3e-5, atol=3e-6)
        np.testing.assert_allclose(arrays["impulses"].numpy(), expected_impulse, rtol=3e-5, atol=3e-6)
        self.assertTrue(np.all(arrays["impulses"].numpy()[0, [0, 3]] > 0.0))
        impulses = arrays["impulses"].numpy()[0].reshape(-1, 3)
        self.assertTrue(np.all(np.linalg.norm(impulses[:, 1:], axis=1) <= 0.4 * impulses[:, 0] + 1e-6))
        # This fixture has uncoupled translation axes: friction alone must
        # reduce each body's tangential kinetic energy.
        tangents = mapping.reshape(-1, 6)[:, :2]
        self.assertTrue(
            np.all(np.sum(arrays["v"].numpy()[tangents] ** 2, axis=1) <= np.sum(host["v"][tangents] ** 2, axis=1))
        )

    def test_large_contact_unit_count_has_no_fixed_4096_limit(self):
        """Assign every normal-only unit and a final friction triple to their common pair."""
        owner, host, _, _, _ = fixture([[(0, 1, False)] * 4097 + [(1, 0, True)]], self.device)
        self.assert_schedule(owner, host)
        self.assertEqual(int(owner.unit_count.numpy()[0]), 4098)
        self.assertEqual(int(owner.pair_count.numpy()[0]), 1)
        self.assertEqual(int(owner.tail_count.numpy()[0]), 0)
        np.testing.assert_array_equal(owner.row_pair.numpy()[0, :4100], 0)

    def test_same_endpoint_combines_both_response_vectors(self):
        """Match the scalar mapping when a synthetic row writes one endpoint twice."""
        owner, host, arrays, mapping, _ = fixture([[(0, 0, True)]], self.device)
        for name in ("J_b", "MiJt_b"):
            host[name][:] = 0.5 * host[name.replace("_b", "_a")]
            arrays[name].assign(host[name])
        host["eff"].fill(1.0 / 2.25)
        host["meta"][:, 1::4] = np.float32(1.0 / 2.25).view(np.int32)
        arrays["meta"].assign(host["meta"])
        order = self.assert_schedule(owner, host)
        solve(owner, arrays)
        expected_v, expected_impulse = serial(host, mapping, self.device, order)
        np.testing.assert_allclose(arrays["v"].numpy(), expected_v, rtol=3e-5, atol=3e-6)
        np.testing.assert_allclose(arrays["impulses"].numpy(), expected_impulse, rtol=3e-5, atol=3e-6)

    def test_unsigned_high_parent_solves_only_the_active_triple(self):
        """Read a parent above signed-int16 range without solving thousands of filler rows."""
        _, host, arrays, mapping, _ = fixture([[(0, 1, True)]], self.device)
        expected_v, expected_impulse = serial(host, mapping, self.device, [[0, 1, 2]])
        parent, capacity = 32768, 32771
        metadata = np.zeros((1, capacity * 4), np.int32)
        metadata[:, 3::4] = pack_endpoints(-1, 3)  # Non-contact filler is not a solve unit.
        metadata[:, parent * 4 :] = host["meta"]
        metadata[0, (parent + 1) * 4 + 3] = pack_endpoints(parent, 2)
        metadata[0, (parent + 2) * 4 + 3] = pack_endpoints(parent, 2)
        arrays["meta"] = wp.array(metadata, dtype=int, device=self.device)
        arrays["counts"] = wp.array([capacity], dtype=int, device=self.device)
        for name in ("J_a", "J_b", "MiJt_a", "MiJt_b", "mu", "w"):
            value = np.zeros((1, capacity, *host[name].shape[2:]), np.float32)
            value[:, parent:] = host[name]
            arrays[name] = wp.array(value, dtype=float, device=self.device)
        arrays["impulses"] = wp.zeros((1, capacity), dtype=float, device=self.device)
        owner = MultiBlockContactSolver(
            1, capacity, 12, wp.array(mapping, dtype=int, device=self.device), self.device, max_colors=2
        )
        owner.prepare(arrays["counts"], arrays["counts"], arrays["meta"])
        self.assertEqual(int(owner.unit_count.numpy()[0]), 1)
        self.assertEqual(int(owner.row_indices.numpy()[0, 0]), parent)
        self.assertEqual(int(owner.unit_lengths.numpy()[0, parent]), 3)
        solve(owner, arrays)
        np.testing.assert_allclose(arrays["v"].numpy(), expected_v, rtol=3e-5, atol=3e-6)
        np.testing.assert_allclose(arrays["impulses"].numpy()[:, parent:], expected_impulse, rtol=3e-5, atol=3e-6)

    def test_projection_matches_serial_with_delay_and_regularization(self):
        """Preserve eight-sweep contact projection, momentum and friction feasibility."""
        for contacts in (
            [(0, 1, True), (2, 3, True)],
            [(1, 0, True), (2, 3, True)],
            [(1, 0, True), (0, 1, True), (2, 3, True)],
            [(0, 1, True), (1, 2, True), (2, 3, True)],
            [(0, 70, True)],
        ):
            for friction_start, regularization in ((0, 0.0), (3, 0.25), (8, 0.0)):
                with self.subTest(contacts=contacts, friction_start=friction_start, regularization=regularization):
                    owner, host, arrays, mapping, masses = fixture([contacts, contacts], self.device)
                    host["w"].fill(1.0 / (1.0 + regularization))
                    arrays["w"].assign(host["w"])
                    order = self.assert_schedule(owner, host)
                    solve(owner, arrays, friction_start=friction_start, regularize=int(regularization != 0))
                    expected_v, expected_impulse = serial(
                        host,
                        mapping,
                        self.device,
                        order,
                        friction_start=friction_start,
                        regularize=int(regularization != 0),
                    )
                    velocity, impulses = arrays["v"].numpy(), arrays["impulses"].numpy()
                    np.testing.assert_allclose(velocity, expected_v, rtol=3e-5, atol=3e-6)
                    np.testing.assert_allclose(impulses, expected_impulse, rtol=3e-5, atol=3e-6)
                    self.assertTrue(np.isfinite(velocity).all())
                    self.assertTrue(np.all(impulses[:, 0] > 0.0))
                    for world in range(2):
                        before = host["v"][mapping[world]].reshape(-1, 6)[:, :3]
                        after = velocity[mapping[world]].reshape(-1, 6)[:, :3]
                        np.testing.assert_allclose(masses @ after, masses @ before, atol=3e-6)
                        for normal in range(0, int(host["counts"][world]), 3):
                            impulse = impulses[world, normal : normal + 3]
                            self.assertGreaterEqual(float(impulse[0]), 0.0)
                            self.assertLessEqual(float(np.linalg.norm(impulse[1:])), 0.4 * float(impulse[0]) + 3e-6)
                    if friction_start == 8:
                        np.testing.assert_array_equal(impulses[:, host["kind"][0] == 2], 0.0)


class TestFeatherPGSMFParallelPrepare(unittest.TestCase):
    """Require scan-based preparation to preserve the serial pair representation exactly."""

    device = "cpu"

    def assert_preparation(self, reference, parallel):
        """Compare active payloads; an unused color schedule is deliberately not equivalent."""
        for name in ("pair_count", "unit_count", "static_pair_count", "body_degree", "row_pair", "unit_lengths"):
            np.testing.assert_array_equal(
                getattr(parallel, name).numpy(), getattr(reference, name).numpy(), err_msg=name
            )
        for world, (pairs, units) in enumerate(
            zip(reference.pair_count.numpy(), reference.unit_count.numpy(), strict=True)
        ):
            for name, count in (
                ("pair_a", pairs),
                ("pair_b", pairs),
                ("pair_offsets", pairs + 1),
                ("row_indices", units),
            ):
                np.testing.assert_array_equal(
                    getattr(parallel, name).numpy()[world, :count],
                    getattr(reference, name).numpy()[world, :count],
                    err_msg=f"world {world}: {name}",
                )
        self.assertFalse(parallel.colors_available)
        np.testing.assert_array_equal(parallel.pair_color.numpy(), -1)
        for name in ("color_count", "color_counts", "color_offsets", "tail_count"):
            np.testing.assert_array_equal(getattr(parallel, name).numpy(), 0, err_msg=name)

    def test_mixed_units_empty_world_prefix_and_reset(self):
        """Subtract prior-world prefixes across adjacent empty worlds and retain mixed contact units."""
        contacts = [
            [],
            [(0, 1, True), (1, 0, False), (0, -1, True), (2, 2, True)],
            [],
            [],
            [(3, -1, False), (0, 2, True)],
            [],
        ]
        for direct_static in (False, True):
            with self.subTest(direct_static_pairs=direct_static):
                fixtures = [
                    fixture(
                        contacts,
                        self.device,
                        mode="mass_split",
                        direct_static_pairs=direct_static,
                        parallel_prepare=parallel,
                    )
                    for parallel in (False, True)
                ]
                reference, parallel = [values[0] for values in fixtures]
                self.assert_preparation(reference, parallel)
                np.testing.assert_array_equal(parallel.pair_count.numpy(), [0, 3, 0, 0, 2, 0])
                np.testing.assert_array_equal(parallel.unit_count.numpy(), [0, 4, 0, 0, 2, 0])
                for loaded in (False, True, False):
                    for owner, host, arrays, _, _ in fixtures:
                        arrays["counts"].assign(host["counts"] if loaded else np.zeros_like(host["counts"]))
                        owner.prepare(arrays["counts"], arrays["counts"], arrays["meta"])
                    self.assert_preparation(reference, parallel)
                    if not loaded:
                        np.testing.assert_array_equal(parallel.pair_count.numpy(), 0)
                        np.testing.assert_array_equal(parallel.body_degree.numpy(), 0)

    def test_all_normal_units_fill_capacity_without_sentinel(self):
        """Handle a world's final active unit when every allocated row is a distinct pair."""
        full = [(0, body, False) for body in range(1, 33)]
        owners = [
            fixture([full, [], full, full], self.device, mode="mass_split", parallel_prepare=parallel)[0]
            for parallel in (False, True)
        ]
        self.assert_preparation(*owners)
        np.testing.assert_array_equal(owners[1].pair_count.numpy(), [32, 0, 32, 32])
        np.testing.assert_array_equal(owners[1].pair_offsets.numpy()[[0, 2, 3], 32], 32)

    def test_unsigned_parent_and_sparse_payload(self):
        """Keep a complete friction triple whose parent exceeds the signed-int16 range."""
        parent, capacity = 32768, 32771
        metadata = np.zeros((1, capacity * 4), np.int32)
        metadata[:, 3::4] = pack_endpoints(-1, 3)
        for row in range(parent, capacity):
            metadata[0, row * 4] = pack_endpoints(0, 6)
            metadata[0, row * 4 + 3] = pack_endpoints(-1, 0) if row == parent else pack_endpoints(parent, 2)
        counts = wp.array([capacity], dtype=int, device=self.device)
        meta = wp.array(metadata, dtype=int, device=self.device)
        mapping = wp.array(np.arange(12, dtype=np.int32).reshape(1, 12), dtype=int, device=self.device)
        owners = [
            MultiBlockContactSolver(1, capacity, 12, mapping, self.device, mode="mass_split", parallel_prepare=parallel)
            for parallel in (False, True)
        ]
        for owner in owners:
            owner.prepare(counts, counts, meta)
        self.assert_preparation(*owners)
        self.assertEqual(int(owners[1].row_indices.numpy()[0, 0]), parent)
        np.testing.assert_array_equal(owners[1].row_pair.numpy()[0, parent:], 0)
        self.assertEqual(int(owners[1].unit_lengths.numpy()[0, parent]), 3)


@unittest.skipUnless(wp.is_cuda_available(), "Parallel preparation replay requires CUDA")
class TestFeatherPGSMFParallelPrepareCUDA(TestFeatherPGSMFParallelPrepare):
    """Exercise the scan and its downstream solve after graph replay changes row counts."""

    device = "cuda:0"

    def test_graph_prepare_and_solve_matches_legacy_after_reset(self):
        """Preserve exact grouping and physical updates across empty and loaded graph replays."""
        contacts = [[], [(0, 1, True), (0, -1, False), (1, 2, True)], [], [(0, -1, True)]]
        fixtures = [
            fixture(contacts, self.device, mode="mass_split", direct_static_pairs=True, parallel_prepare=parallel)
            for parallel in (False, True)
        ]

        def step(values):
            owner, _, arrays, _, _ = values
            owner.prepare(arrays["counts"], arrays["counts"], arrays["meta"])
            arrays["impulses"].zero_()
            arguments = [
                arrays[name] for name in ("meta", "impulses", "J_a", "J_b", "MiJt_a", "MiJt_b", "mu", "w", "v")
            ]
            for _ in range(24):
                owner.solve_mass_split(*arguments, 1.0, 0, 1)

        for values in fixtures:
            step(values)
        with wp.ScopedCapture(self.device) as capture:
            step(fixtures[1])
        for loaded in (False, True, False, True):
            for _, host, arrays, _, _ in fixtures:
                arrays["counts"].assign(host["counts"] if loaded else np.zeros_like(host["counts"]))
                arrays["v"].assign(host["v"])
            step(fixtures[0])
            wp.capture_launch(capture.graph)
            self.assert_preparation(fixtures[0][0], fixtures[1][0])
            for name in ("v", "impulses"):
                np.testing.assert_allclose(
                    fixtures[1][2][name].numpy(), fixtures[0][2][name].numpy(), rtol=3e-5, atol=5e-6, err_msg=name
                )


@unittest.skipUnless(wp.is_cuda_available(), "Multi-block CUDA replay requires CUDA")
class TestFeatherPGSMFColoringCUDA(TestFeatherPGSMFColoring):
    device = "cuda:0"

    def test_dense_contact_and_final_velocity_limit_order(self):
        """Match full GS with dense rows, colored MF contacts, then final velocity limits."""
        device = wp.get_device(self.device)
        _, reference = solve_fixture(device)
        _, colored = solve_fixture(device)
        for values in (reference, colored):
            kinds = values["world_row_type"].numpy()
            kinds[:4, 0] = 4
            values["world_row_type"].assign(kinds)
            rhs = values["rhs_bias"].numpy()
            rhs[:4, 0] = -2.0
            values["rhs_bias"].assign(rhs)
            values["dense_phase_bounds"].assign(np.array([[0, 1]] * 4 + [[0, 0]], np.int32))
            values.update(
                general_world_grid_stride=5,
                use_general_world_queue=0,
                iterations=1,
                omega=1.0,
                regularize=0,
                row_phase=0,
                friction_start_iteration=0,
                iteration_offset=0,
                freeze_drive_rows=0,
                defer_dense_response=0,
            )
        kernel = _get_pgs_solve_mf_gs_kernel(
            8, 6, 29, str(device.arch), has_drive_rows=False, has_dense_velocity_limit_rows=True
        )

        def launch(values, phase, iteration):
            values.update(row_phase=phase, iteration_offset=iteration)
            wp.launch_tiled(
                kernel,
                dim=[5],
                block_dim=32,
                device=device,
                inputs=[values[name] for name in inspect.signature(kernel.func).parameters],
            )

        owner = MultiBlockContactSolver(5, 6, 29, colored["world_dof_indices"], device, max_colors=2)
        owner.prepare(colored["mf_constraint_count"], colored["mf_contact_rows_end"], colored["mf_meta"])
        arguments = [
            colored[name]
            for name in (
                "mf_meta",
                "mf_impulses",
                "mf_J_a",
                "mf_J_b",
                "mf_MiJt_a",
                "mf_MiJt_b",
                "mf_row_mu",
                "mf_row_w",
                "v_out",
            )
        ]
        for iteration in range(8):
            before_mf = colored["mf_impulses"].numpy()
            before_dense = colored["world_impulses"].numpy()
            launch(colored, 6, iteration)
            np.testing.assert_array_equal(colored["mf_impulses"].numpy(), before_mf)
            np.testing.assert_array_equal(colored["world_impulses"].numpy()[:4, 0], before_dense[:4, 0])
            if iteration == 0:
                self.assertTrue(np.any(colored["world_impulses"].numpy()[:4, 1:4] != before_dense[:4, 1:4]))
            for color in range(owner.max_colors):
                owner.solve_color(color, *arguments, 1.0, 0, 1)
            owner.solve_tail(*arguments, 1.0, 0, 1)
            launch(colored, 5, iteration)
            launch(reference, 0, iteration)
        for name in ("v_out", "world_impulses", "mf_impulses"):
            np.testing.assert_allclose(
                colored[name].numpy(), reference[name].numpy(), rtol=5e-5, atol=5e-6, err_msg=name
            )

    def test_graph_replay_rebuilds_current_rows(self):
        """Replay preparation and solve after topology becomes empty and loaded again."""
        owner, host, arrays, mapping, _ = fixture([[(0, 1, True), (1, 2, False), (2, 3, True)]], self.device)

        def step():
            owner.prepare(arrays["counts"], arrays["counts"], arrays["meta"])
            arrays["impulses"].zero_()
            solve(owner, arrays)

        step()
        arrays["v"].assign(host["v"])
        with wp.ScopedCapture(self.device) as capture:
            step()
        for loaded in (False, True, False, True):
            arrays["counts"].assign(host["counts"] if loaded else np.zeros_like(host["counts"]))
            arrays["v"].assign(host["v"])
            wp.capture_launch(capture.graph)
            if loaded:
                order = self.assert_schedule(owner, host)
                expected, _ = serial(host, mapping, self.device, order)
                np.testing.assert_allclose(arrays["v"].numpy(), expected, rtol=3e-5, atol=3e-6)
            else:
                np.testing.assert_array_equal(owner.pair_count.numpy(), 0)
                np.testing.assert_array_equal(arrays["v"].numpy(), host["v"])


@unittest.skipUnless(wp.is_cuda_available(), "Mass-split contact sweeps require CUDA")
class TestFeatherPGSMFMassSplitCUDA(unittest.TestCase):
    """Check physical impulses and convergence, not serial-PGS iteration identity."""

    device = "cuda:0"

    def sweep(self, owner, arrays, *, iterations=24, friction_start=0, regularize=0):
        """Keep the production 24-sweep allowance while averaging pair-local responses."""
        arguments = [arrays[name] for name in ("meta", "impulses", "J_a", "J_b", "MiJt_a", "MiJt_b", "mu", "w", "v")]
        for iteration in range(iterations):
            owner.solve_mass_split(*arguments, 1.0, regularize, int(iteration >= friction_start))

    def assert_physical_response(self, host, arrays, mapping):
        """Require actual velocity to equal the unscaled physical M-inverse J-transpose impulse."""
        expected = host["v"].copy()
        impulse = arrays["impulses"].numpy()
        bodies = mapping.shape[1] // 6
        for world, count in enumerate(host["counts"]):
            for row in range(int(count)):
                for side in ("a", "b"):
                    body = int(host[f"body_{side}"][world, row])
                    if body >= 0:
                        start = 6 * (body % bodies)
                        expected[mapping[world, start : start + 6]] += (
                            host[f"MiJt_{side}"][world, row] * impulse[world, row]
                        )
        np.testing.assert_allclose(arrays["v"].numpy(), expected, rtol=3e-5, atol=5e-6)
        self.assertTrue(np.isfinite(arrays["v"].numpy()).all())
        self.assertTrue(np.isfinite(impulse).all())

    def star(self, *, reversed_endpoints=False):
        """Create an analytically solvable frictionless star with two conflicting pairs."""
        contacts = [(1, 0, False), (2, 0, False)] if reversed_endpoints else [(0, 1, False), (0, 2, False)]
        owner, host, arrays, mapping, masses = fixture([contacts, contacts], self.device, mode="mass_split")
        host["v"].fill(0.0)
        host["v"][mapping[:, 2]] = -1.0
        host["rhs"].fill(0.0)
        host["meta"][:, 2::4] = 0
        arrays["v"].assign(host["v"])
        arrays["meta"].assign(host["meta"])
        return owner, host, arrays, mapping, masses

    def test_conflicting_star_converges_to_analytic_inelastic_solution(self):
        """Reach the unequal-mass consensus while retaining physical impulses and momentum."""
        for reversed_endpoints in (False, True):
            with self.subTest(reversed_endpoints=reversed_endpoints):
                owner, host, arrays, mapping, masses = self.star(reversed_endpoints=reversed_endpoints)
                np.testing.assert_array_equal(owner.body_degree.numpy()[:, [0, 6, 12]], [[2, 1, 1]] * 2)
                self.sweep(owner, arrays)
                self.assert_physical_response(host, arrays, mapping)
                expected_velocity = -masses[0] / masses.sum()
                expected_impulses = -masses[1:] * expected_velocity
                np.testing.assert_allclose(arrays["impulses"].numpy(), [expected_impulses] * 2, atol=3e-5)
                for world in range(2):
                    velocity = arrays["v"].numpy()[mapping[world]].reshape(-1, 6)
                    np.testing.assert_allclose(velocity[:, 2], expected_velocity, atol=3e-5)
                    np.testing.assert_allclose(masses @ velocity[:, :3], [0.0, 0.0, -1.0], atol=5e-6)
                    self.assertLess(float(np.max(np.abs(velocity[0, 2] - velocity[1:, 2]))), 3e-5)

    def test_degree_one_static_reverse_and_self_match_existing_law(self):
        """Reduce to ordinary pair PGS without changing friction delay or regularization."""
        for contacts in (
            [(0, -1, True), (-1, 1, True)],
            [(1, 0, True), (0, 1, True)],
            [(0, 0, True)],
        ):
            for friction_start, regularization in ((0, 0.0), (3, 0.25), (24, 0.0)):
                with self.subTest(contacts=contacts, friction_start=friction_start, regularization=regularization):
                    owner, host, arrays, mapping, _ = fixture([contacts], self.device, mode="mass_split")
                    if contacts[0] == (0, -1, True):
                        host["v"][mapping[0, 8]] = 0.8
                    if contacts[0] == (0, 0, True):
                        for name in ("J_b", "MiJt_b"):
                            host[name][:] = 0.5 * host[name.replace("_b", "_a")]
                        host["eff"].fill(1.0 / 2.25)
                        host["meta"][:, 1::4] = np.float32(1.0 / 2.25).view(np.int32)
                    host["w"].fill(1.0 / (1.0 + regularization))
                    for name in ("v", "w", "J_b", "MiJt_b", "meta"):
                        arrays[name].assign(host[name])
                    self.sweep(owner, arrays, friction_start=friction_start, regularize=int(regularization != 0))
                    self.assert_physical_response(host, arrays, mapping)
                    expected_v, expected_impulse = serial(
                        host,
                        mapping,
                        self.device,
                        [list(range(int(host["counts"][0])))],
                        iterations=24,
                        friction_start=friction_start,
                        regularize=int(regularization != 0),
                    )
                    np.testing.assert_allclose(arrays["v"].numpy(), expected_v, rtol=3e-5, atol=5e-6)
                    np.testing.assert_allclose(arrays["impulses"].numpy(), expected_impulse, rtol=3e-5, atol=5e-6)
                    self.assertTrue(np.all(owner.body_degree.numpy() <= 1))

    def test_conflicting_friction_star_is_feasible_and_dissipative(self):
        """Check bounded normal residual and friction cones rather than treating finite values as convergence."""
        owner, host, arrays, mapping, masses = fixture(
            [[(0, 1, True), (0, 2, True), (0, 3, True)]], self.device, mode="mass_split"
        )
        host["rhs"].fill(0.0)
        host["meta"][:, 2::4] = 0
        arrays["meta"].assign(host["meta"])
        self.sweep(owner, arrays)
        self.assert_physical_response(host, arrays, mapping)
        impulse = arrays["impulses"].numpy()[0].reshape(-1, 3)
        velocity = arrays["v"].numpy()[mapping[0]].reshape(-1, 6)[:, :3]
        before = host["v"][mapping[0]].reshape(-1, 6)[:, :3]
        gap = velocity[0, 2] - velocity[1:, 2]
        self.assertTrue(np.all(impulse[:, 0] >= 0.0))
        self.assertTrue(np.all(np.linalg.norm(impulse[:, 1:], axis=1) <= 0.4 * impulse[:, 0] + 5e-6))
        self.assertGreater(float(gap.min()), -0.004)
        natural_residual = impulse[:, 0] - np.maximum(0.0, impulse[:, 0] - gap * host["eff"][0, ::3])
        self.assertLess(float(np.max(np.abs(natural_residual))), 0.003)
        np.testing.assert_allclose(masses @ velocity, masses @ before, atol=5e-6)
        self.assertLessEqual(
            float(np.sum(masses[:, None] * velocity**2)), float(np.sum(masses[:, None] * before**2)) + 5e-6
        )

    def test_graph_replay_resets_incident_degrees(self):
        """Rebuild a star, a single pair and empty topology without retaining stale consensus state."""
        owner, host, arrays, mapping, masses = self.star()

        def step():
            owner.prepare(arrays["counts"], arrays["counts"], arrays["meta"])
            arrays["impulses"].zero_()
            self.sweep(owner, arrays)

        step()
        with wp.ScopedCapture(self.device) as capture:
            step()
        for count in (0, 1, 2, 0, 2):
            arrays["counts"].assign(np.full(2, count, np.int32))
            arrays["v"].assign(host["v"])
            wp.capture_launch(capture.graph)
            np.testing.assert_array_equal(owner.body_degree.numpy()[:, 0], count)
            for world in range(2):
                velocity = arrays["v"].numpy()[mapping[world]].reshape(-1, 6)[:, 2]
                expected = np.zeros(3)
                expected[: count + 1] = -1.0 / masses[: count + 1].sum()
                np.testing.assert_allclose(velocity, expected, atol=3e-5)

    def anchored_hub(self, *, direct_static_pairs):
        """Couple a high-degree dynamic hub to fixed support beside a static-only world."""
        contacts = [[(0, -1, False), *[(0, body, False) for body in range(1, 9)]], [(-1, 0, False)]]
        owner, host, arrays, mapping, masses = fixture(
            contacts, self.device, mode="mass_split", direct_static_pairs=direct_static_pairs
        )
        host["v"].fill(0.0)
        host["v"][mapping[:, 2]] = [-1.0, 1.0]
        host["rhs"].fill(0.0)
        host["meta"][:, 2::4] = 0
        arrays["v"].assign(host["v"])
        arrays["meta"].assign(host["meta"])
        return owner, host, arrays, mapping, masses

    def test_direct_static_support_is_not_diluted_by_dynamic_degree(self):
        """Finish each sweep with fixed-support response and retain its physical impulse."""
        residual = {}
        for direct in (False, True):
            with self.subTest(direct_static_pairs=direct):
                owner, host, arrays, mapping, masses = self.anchored_hub(direct_static_pairs=direct)
                np.testing.assert_array_equal(owner.static_pair_count.numpy(), [1, 1])
                np.testing.assert_array_equal(owner.body_degree.numpy()[:, 0], [8, 0] if direct else [9, 1])
                self.sweep(owner, arrays)
                self.assert_physical_response(host, arrays, mapping)
                velocity = arrays["v"].numpy()[mapping[0]].reshape(-1, 6)[:, 2]
                residual[direct] = max(0.0, -float(velocity[0]))
                static_impulse = float(arrays["impulses"].numpy()[0, 0])
                np.testing.assert_allclose(masses @ velocity, -1.0 + static_impulse, atol=5e-6)
                self.assertGreater(static_impulse, 0.0)
                self.assertAlmostEqual(float(arrays["v"].numpy()[mapping[1, 2]]), 0.0, delta=3e-6)
        self.assertGreater(residual[False], 0.01)
        self.assertLess(residual[True], 3e-6)

    def test_direct_static_graph_resets_empty_and_static_only_topology(self):
        """Replay mixed, static-only and empty contact sets without stale degrees or impulses."""
        owner, host, arrays, mapping, _ = self.anchored_hub(direct_static_pairs=True)

        def step():
            owner.prepare(arrays["counts"], arrays["counts"], arrays["meta"])
            arrays["impulses"].zero_()
            self.sweep(owner, arrays)

        step()
        with wp.ScopedCapture(self.device) as capture:
            step()
        for counts in ([0, 0], [1, 1], [9, 1], [0, 0], [9, 1]):
            arrays["counts"].assign(np.array(counts, np.int32))
            arrays["v"].assign(host["v"])
            wp.capture_launch(capture.graph)
            self.assert_physical_response(host, arrays, mapping)
            np.testing.assert_array_equal(owner.body_degree.numpy()[:, 0], [max(0, counts[0] - 1), 0])
            np.testing.assert_array_equal(owner.static_pair_count.numpy(), [int(count > 0) for count in counts])
            if counts[0]:
                np.testing.assert_allclose(arrays["v"].numpy()[mapping[:, 2]], 0.0, atol=3e-6)
            else:
                np.testing.assert_array_equal(arrays["v"].numpy(), host["v"])


if __name__ == "__main__":
    unittest.main()
