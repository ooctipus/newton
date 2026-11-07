# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the sparse sweep's contact units against physical sparse PGS references."""

import os
import unittest
from itertools import product
from pathlib import Path
from unittest import mock

import numpy as np
import warp as wp

from newton._src.core.memory import arch_budget
from newton._src.solvers.feather_pgs import class_plan
from newton._src.solvers.feather_pgs.row_kernels import PGS_CONSTRAINT_TYPE_CONTACT as CONTACT
from newton._src.solvers.feather_pgs.row_kernels import PGS_CONSTRAINT_TYPE_FRICTION as FRICTION
from newton._src.solvers.feather_pgs.sparse_contact import CONTACT_RECORD_FLOATS, CONTACT_RING_FLOATS
from newton._src.solvers.feather_pgs.sparse_mass_matrix import _SparseRowSupport
from newton._src.solvers.feather_pgs.sweep import sweep_kernel
from newton._src.solvers.feather_pgs.sweep_routes_sparse import SparseSweep
from newton.solvers import SolverFeatherPGS
from newton.tests.feather_pgs_law_harness import row_lines
from newton.tests.feather_pgs_sparse_scene import equal_dof_trees
from newton.tests.test_feather_pgs_sparse_pgs import _problem, _sweep
from newton.tests.test_feather_pgs_sparse_solver import _build_model, _make_solver

CANDIDATE_OPTIONS = {"contact_triplets": True}


def factory(capacity, dofs, width, *, candidate=True, active=True, linked=False):
    """Return the sparse sweep of the triplet tests: with contact units (``candidate``) or the ordinary one."""
    options = dict(CANDIDATE_OPTIONS) if candidate else {}
    return sweep_kernel(
        SparseSweep(
            capacity,
            dofs,
            width,
            active_rows=active,
            use_support_table=True,
            linked_rows=linked,
            **options,
        )
    )


def pair_metric(a, c, d):
    """Return the paired law's tangent eigenbasis ``(largest, smallest, vx, vy)``: the block's eigenvalues, formed on
    the block scaled by ``max(a, d)`` and scaled back."""
    scale = max(a, d, 1.0e-20)
    a, c, d = a / scale, c / scale, d / scale
    largest = 0.5 * (a + d + np.sqrt((a - d) ** 2 + 4.0 * c * c))
    smallest = max((a * d - c * c) / largest, 0.0) if largest > 0.0 else 0.0
    vx, vy = (largest - d, c) if a >= d else (c, largest - a)
    norm = np.hypot(vx, vy)
    largest, smallest = largest * scale, smallest * scale
    return (largest, smallest, vx / norm, vy / norm) if norm > 0.0 else (largest, smallest, 1.0, 0.0)


def contact_records(data, *, linked=False):
    """Return the block records the sparse contact builder writes for every contact triplet of ``data``.

    The contact units read them; the ring members of a linked sweep are its own.
    """
    worlds, capacity, _ = data["row_factor"].shape
    dofs = data["reference_factor_delta"].shape[1]
    records = np.zeros((worlds, capacity // 3, CONTACT_RECORD_FLOATS + CONTACT_RING_FLOATS * linked), np.float32)
    for world, count in enumerate(data["world_constraint_count"]):
        kinds = data["row_type"][world]
        for row in range(int(count) - 2):
            if (kinds[row], kinds[row + 1], kinds[row + 2]) != (CONTACT, FRICTION, FRICTION):
                continue
            rows = np.zeros((3, dofs))
            for k in range(3):
                indices = data["support_class_dof"][data["support_row_class"][world, row + k]]
                valid = indices >= 0
                rows[k, indices[valid]] = data["row_factor"][world, row + k, valid]
            block = rows @ rows.T
            diagonal = data["diagonal"][world, row : row + 3].astype(np.float64)
            # A normal without a positive diagonal keeps the ordinary update; its record is never read.
            inverse = 1.0 / diagonal[0] if diagonal[0] > 0.0 else 0.0
            records[world, row // 3, :4] = (block[0, 1], block[0, 2], block[1, 2], inverse)
            records[world, row // 3, 4:8] = pair_metric(diagonal[1], block[1, 2], diagonal[2])
    return records


def fixture(*, offset=0, activity=None, seeded=False, omega=1.0, linked=False):
    """Retain the independent physical-coordinate oracle, including its factor."""
    dofs, width = (3, 3) if activity is not None else (18, 12)
    factor, indices, z, initial, bias, kinds, parents, mu, impulses, jacobian, response, incident, diag = _problem(
        dofs,
        width,
        seeded,
        contact_activity=activity,
    )
    if not linked and activity != "linked_load":
        parents[kinds == CONTACT] = -1
    if offset:
        z = np.pad(z, ((offset, 0), (0, 0)))
        jacobian = np.pad(jacobian, ((offset, 0), (0, 0)))
        response = np.pad(response, ((offset, 0), (0, 0)))
        bias = np.pad(bias, (offset, 0), constant_values=1.0)
        incident = np.pad(incident, (offset, 0))
        diag = np.pad(diag, (offset, 0), constant_values=1.0)
        kinds = np.pad(kinds, (offset, 0), constant_values=CONTACT)
        parents = np.pad(np.where(parents >= 0, parents + offset, -1), (offset, 0), constant_values=-1)
        mu, impulses = np.pad(mu, (offset, 0)), np.pad(impulses, (offset, 0))
    count, capacity, worlds = len(kinds), 192, 3
    assert count <= capacity

    def padded(value, dtype=np.float32):
        result = np.zeros((worlds, capacity, *value.shape[1:]), dtype=dtype)
        result[:, : len(value)] = value
        return result

    row_factor = np.zeros((worlds, capacity, width), dtype=np.float32)
    row_factor[:, :count] = z[:, indices]
    data = {
        "world_constraint_count": np.array([count, 0, count], np.int32),
        "rhs_bias": padded(bias),
        "diagonal": padded(diag),
        "initial_impulses": padded(impulses),
        "row_factor": row_factor,
        "row_incident": padded(incident),
        "row_type": padded(kinds, np.int32),
        "row_parent": padded(parents, np.int32),
        "row_mu": padded(mu),
        "support_row_class": np.zeros((worlds, capacity), np.int32),
        "support_class_dof": indices[None].astype(np.int32),
        "iterations": np.array(8),
        "omega": np.array(omega),
        "reference_factor_delta": np.zeros((worlds, dofs), np.float32),
    }
    expected_velocity, expected_impulse = _sweep(
        jacobian,
        response,
        initial,
        bias,
        diag,
        kinds,
        parents,
        mu,
        impulses,
        omega=omega,
        active_rows=True,
        iterations=8,
    )
    physical = {
        "factor": factor,
        "initial": initial,
        "jacobian": jacobian,
        "response": response,
        "expected_velocity": expected_velocity,
        "expected_impulse": expected_impulse,
        "initial_impulse": impulses,
    }
    return data, physical


class DeviceSolve:
    """One ABI adapter shared by synthetic and preserved actual-input tests.

    A kernel with contact units also reads the triplets' block records
    (:func:`contact_records`), which follow the scalar inputs.
    """

    def __init__(self, data, *, candidate, device="cuda:0", active=True, linked=False):
        self.device = wp.get_device(device)
        self.data = data
        self.linked = linked
        self.worlds, self.capacity, self.width = data["row_factor"].shape
        self.dofs = data["reference_factor_delta"].shape[1]
        self.kernel = factory(
            self.capacity,
            self.dofs,
            self.width,
            candidate=candidate,
            active=active,
            linked=linked,
        )
        self.records = None

        def array(value, dtype=float):
            return wp.array(value, dtype=dtype, device=self.device)

        def zeros(shape, dtype=float):
            return wp.zeros(shape, dtype=dtype, device=self.device)

        self.support = _SparseRowSupport()
        self.support.enabled = 1
        self.support.row_class = array(data["support_row_class"], int)
        self.support.class_dof = array(data["support_class_dof"], int)
        self.impulses = array(data["initial_impulses"])
        self.delta = wp.full((self.worlds, self.dofs), float("nan"), device=self.device)
        self.inputs = [
            self.worlds,
            array(data["world_constraint_count"], int),
            row_lines(self.device),
            array(data["rhs_bias"]),
            array(data["diagonal"]),
            self.impulses,
            wp.full((1, 1, 1), -9, dtype=int, device=self.device),
            array(data["row_factor"]),
            array(data["row_incident"]),
            array(data["row_type"], int),
            array(data["row_parent"], int),
            array(data["row_mu"]),
            array(np.arange(self.worlds * self.dofs).reshape(self.worlds, self.dofs), int),
            zeros(self.worlds * self.dofs),
            int(data["iterations"]),
            float(data["omega"]),
        ]
        if "contact_records" in self.kernel.adj.arg_types:
            self.records = array(contact_records(data, linked=self.linked))
            self.inputs.append(self.records)

    @property
    def units(self):
        """Whether the kernel solves contact units (and reads their records)."""
        return self.records is not None

    def rebuild_records(self, data):
        """Write the records of changed rows, as the contact builder does each step."""
        if self.records is not None:
            self.records.assign(contact_records(data, linked=self.linked))

    def launch(self):
        wp.launch_tiled(
            self.kernel,
            dim=[(self.worlds + 1) // 2],
            inputs=self.inputs,
            outputs=[self.delta, self.support],
            block_dim=64,
            device=self.device,
        )

    def reset_impulses(self):
        self.impulses.assign(self.data["initial_impulses"])

    def result(self):
        return self.delta.numpy(), self.impulses.numpy()


def assert_output(test, data, delta, impulses, *, physical=None, reference=None):
    """Check independent response action and physical reference, not kernel naming."""
    counts = data["world_constraint_count"]
    live = np.arange(impulses.shape[1])[None] < counts[:, None]
    test.assertTrue(np.isfinite(delta).all())
    test.assertTrue(np.isfinite(impulses[live]).all())
    if reference is not None:
        np.testing.assert_allclose(delta, reference[0], atol=3e-5, rtol=3e-4)
        np.testing.assert_allclose(impulses[live], reference[1][live], atol=3e-5, rtol=3e-4)
    for world, count in enumerate(counts):
        if count == 0:
            np.testing.assert_array_equal(delta[world], 0.0)
            continue
        if physical is not None:
            velocity = physical["initial"] + np.linalg.solve(physical["factor"].T, delta[world])
            np.testing.assert_allclose(velocity, physical["expected_velocity"], atol=3e-5, rtol=3e-4)
            np.testing.assert_allclose(impulses[world, :count], physical["expected_impulse"], atol=3e-5, rtol=3e-4)
        dense = np.zeros((count, delta.shape[1]), dtype=np.float64)
        indices = data["support_class_dof"][data["support_row_class"][world, :count]]
        for row in range(count):
            valid = indices[row] >= 0
            dense[row, indices[row, valid]] = data["row_factor"][world, row, valid]
        expected_delta = dense.T @ (impulses[world, :count] - data["initial_impulses"][world, :count])
        np.testing.assert_allclose(delta[world], expected_delta, atol=3e-5, rtol=3e-4)
        for row in range(count):
            parent = int(data["row_parent"][world, row])
            if data["row_type"][world, row] != FRICTION or row != parent + 1:
                continue
            if parent < 0 or data["row_parent"][world, parent] >= 0:
                continue  # Linked normal loads retain their existing fallback.
            radius = max(float(data["row_mu"][world, row] * impulses[world, parent]), 0.0)
            test.assertLessEqual(float(np.linalg.norm(impulses[world, row : row + 2])), radius + 3e-5)


# Contact units (region, anchor, normal bias): a region links its normals in one ring, in
# layout order; region None leaves the normal unlinked. An anchor adds its two tangents.
_LINKED_LAYOUTS = {
    "self_loop": ((0, True, -0.2),),
    # The anchor's own normal separates; the other normal of its ring loads it.
    "two_member_ring": ((0, True, 0.15), (0, False, -0.3)),
    "three_member_ring": ((0, True, -0.1), (0, False, 0.2), (0, True, -0.25)),
    # Only the fourth member is violated: selecting the pair, and the fused radius, need
    # the whole ring.
    "far_ring": ((0, True, 0.6), (0, False, 0.6), (0, False, 0.6), (0, False, -0.3)),
    # Four lightly loaded normals hold two sliding anchors, so the fused radius rounds
    # as the ring order sums it.
    "loaded_ring": ((0, True, -0.04), (0, False, -0.07), (0, False, -0.03), (0, True, -0.05)),
    # Patch regions have no member limit. Only the sixth member is violated: selecting the
    # pair, and the full sweep's fused radius, need the whole ring.
    "long_ring": ((0, True, 0.6), (0, False, 0.6), (0, False, 0.6), (0, False, 0.6), (0, False, 0.6), (0, False, -0.3)),
    # Six lightly loaded normals hold anchors at the first and fourth members, so the
    # active scan fuses both pairs with radii pooled over the whole ring.
    "long_loaded_ring": (
        (0, True, -0.04),
        (0, False, -0.07),
        (0, False, -0.03),
        (0, True, -0.05),
        (0, False, -0.06),
        (0, False, -0.02),
    ),
    "mixed_units": (
        (0, True, -0.2),
        (1, False, -0.1),
        (1, True, 0.1),
        (None, True, -0.15),
        (2, False, 0.3),
        (0, False, -0.15),
        (2, True, -0.05),
        (1, False, -0.2),
    ),
}


def linked_fixture(layout, *, pad=0, seeded=False, omega=1.0, active=True):
    """Build ring-linked patch rows in the DeviceSolve layout, with their physical sweep.

    ``pad`` separating, unlinked normals lead the units, moving records and rings
    across 32-row words. Unit masses make the factor rows the physical Jacobian.
    """
    rng = np.random.default_rng(len(layout) + pad)
    dofs = 12
    kinds, parents, mu, bias, rings = [CONTACT] * pad, [-1] * pad, [0.0] * pad, [1.0] * pad, {}
    for region, anchor, normal_bias in layout:
        normal = len(kinds)
        if region is not None:
            rings.setdefault(region, []).append(normal)
        kinds.append(CONTACT)
        parents.append(-1)
        mu.append(0.0)
        bias.append(normal_bias)
        if anchor:
            kinds += [FRICTION] * 2
            parents += [normal] * 2
            mu += [0.5] * 2
            bias += list(rng.uniform(-0.2, 0.2, 2))
    kinds, parents, mu, bias = np.array(kinds, np.int32), np.array(parents, np.int32), np.array(mu), np.array(bias)
    for ring in rings.values():
        parents[ring] = np.roll(ring, -1)
    count = len(kinds)
    # Round the inputs to float32 once, so that the device and the oracle solve one problem.
    rows = rng.normal(scale=0.4, size=(count, dofs)).astype(np.float32).astype(np.float64)
    bias = bias.astype(np.float32).astype(np.float64)
    diagonal = (np.sum(rows * rows, axis=1) + 1.0e-6).astype(np.float32).astype(np.float64)
    impulses = np.zeros(count)
    if seeded:
        impulses[np.flatnonzero(kinds == CONTACT)[1::2]] = 0.0625
        impulses[kinds == FRICTION] = np.tile((0.015625, -0.03125), np.count_nonzero(kinds == FRICTION) // 2)
    capacity, worlds = 192, 3

    def padded(value, dtype=np.float32):
        result = np.zeros((worlds, capacity, *value.shape[1:]), dtype=dtype)
        result[:, : len(value)] = value
        return result

    data = {
        "world_constraint_count": np.array([count, 0, count], np.int32),
        "rhs_bias": padded(bias),
        "diagonal": padded(diagonal),
        "initial_impulses": padded(impulses),
        "row_factor": padded(rows),
        "row_incident": padded(np.zeros(count)),
        "row_type": padded(kinds, np.int32),
        "row_parent": padded(parents, np.int32),
        "row_mu": padded(mu),
        "support_row_class": np.zeros((worlds, capacity), np.int32),
        "support_class_dof": np.arange(dofs, dtype=np.int32)[None],
        "iterations": np.array(8),
        "omega": np.array(omega),
        "reference_factor_delta": np.zeros((worlds, dofs), np.float32),
    }
    velocity, impulse = _sweep(
        rows,
        rows,
        np.zeros(dofs),
        bias,
        diagonal,
        kinds,
        parents,
        mu,
        impulses,
        omega=omega,
        active_rows=active,
        linked_rows=True,
    )
    physical = {
        "factor": np.eye(dofs),
        "initial": np.zeros(dofs),
        "expected_velocity": velocity,
        "expected_impulse": impulse,
    }
    return data, physical


class TestSparseContactTriplets(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "Sparse construction requires CUDA")
    def test_host_admission_and_shared_memory_fallback_cuda(self):
        """Admit compatible triplets, and past the shared memory that holds them keep the sparse route with ordinary
        rows where its full sweep still fits, or leave the class to other owners where it does not."""
        model = equal_dof_trees(4, 3, "cuda:0")
        # The largest capacity whose linked contact units fit the device's static budget, and one more row. At such
        # capacities the active-row scan's bitsets no longer fit, and the law's units pool their rings in the full sweep.
        budget = arch_budget(wp.get_device("cuda:0").arch)

        def staging(rows, **options):
            return SparseSweep(
                rows,
                18,
                12,
                **{
                    "active_rows": False,
                    "use_support_table": True,
                    "contact_triplets": True,
                    "linked_rows": True,
                    "torsion": True,  # The law gives a friction region with a footprint its spin row.
                    **options,
                },
            ).staging

        low, high = 192, 1 << 16
        while high - low > 1:
            middle = (low + high) // 2
            low, high = (middle, high) if staging(middle).fits(budget) else (low, middle)
        # The tiny batch: the largest capacity whose support table (its classes, and a class a row) costs at least the
        # explicit indices (a coordinate an entry), the admission's rule, from this tree's classes.
        probe = SolverFeatherPGS(
            model,
            enable_joint_velocity_limits=False,
            pgs_warmstart=False,
            dense_max_constraints=192,
        )
        support = probe.placement.sparse_row_support
        classes = sum(
            getattr(support, name).numpy().nbytes for name in ("body_class", "pair_class", "class_dof", "limit_class")
        )
        tiny = classes // (4 * model.world_count * (int(probe.spec.classes.packed.max_support) - 1))
        cases = (
            (192, {"contact_units": None}, False),
            (192, {}, True),
            (tiny, {}, False),  # The tiny batch's support table costs more than explicit indices.
            (192, {"active_rows": False}, True),  # The law's units pool their rings in the full sweep too.
            (192, {"row_support": False}, False),
            (low, {}, True),
            (high, {}, False),
        )
        # Past the units' capacity the sparse route keeps ordinary rows only if its plain full sweep fits.
        plain = staging(high, contact_triplets=False).fits(budget)
        for capacity, pins, selected in cases:
            with (
                self.subTest(capacity=capacity, pins=pins),
                mock.patch.dict(os.environ, {}, clear=True),
            ):
                solver = SolverFeatherPGS(
                    model,
                    pins=SolverFeatherPGS.Pins(**pins),
                    enable_joint_velocity_limits=False,
                    pgs_warmstart=False,
                    dense_max_constraints=capacity,
                )
                if capacity == high and not plain:
                    self.assertIsNone(solver.spec.classes.packed_size)
                    continue
                self.assertEqual(solver.spec.classes.packed_size, 18)
                self.assertEqual("contact_triplets" in solver.sweep.sparse_kernel.key, selected)

    def test_factory_on_coupled_fixture_cpu(self):
        """Exercise coupled normal and tangent rows before constructing the opt-in kernel."""
        _, physical = fixture(offset=30, seeded=True)
        test_normal, test_tangent = 31, 32
        cross = physical["jacobian"][test_normal] @ physical["response"][test_tangent]
        self.assertGreater(abs(cross), 1e-4)
        self.assertGreater(np.linalg.norm(physical["expected_velocity"] - physical["initial"]), 1e-3)
        self.assertGreater(np.linalg.norm(physical["expected_impulse"] - physical["initial_impulse"]), 1e-3)
        selected = factory(192, 18, 12)
        old = factory(192, 18, 12, candidate=False)
        self.assertIsNot(selected, old)

    @unittest.skipUnless(wp.is_cuda_available(), "Requires root-owned CUDA lease")
    def test_eight_sweep_physical_reference_cuda(self):
        """Preserve the ordered physical solve across contact activation and friction cases."""
        cases = (
            [{"offset": value, "seeded": True} for value in (0, 29, 30, 184)]
            + [
                {"activity": value}
                for value in (
                    "late_activation",
                    "deferred_activation",
                    "near_singular",
                    "seeded_retraction",
                    "linked_load",
                )
            ]
            + [{"omega": value, "seeded": True} for value in (0.0, 0.7, 1.2, -0.5)]
        )
        for settings in cases:
            with self.subTest(settings=settings):
                data, physical = fixture(**settings)
                control = DeviceSolve(data, candidate=False)
                candidate = DeviceSolve(data, candidate=True)
                self.assertTrue(candidate.units)
                control.launch()
                candidate.launch()
                result = candidate.result()
                assert_output(self, data, *result, physical=physical, reference=control.result())

    @unittest.skipUnless(wp.is_cuda_available(), "Requires root-owned CUDA lease")
    def test_saturated_metric_cache_preserves_uncached_pairs_cuda(self):
        """Solve more distinct active pairs than the compact cache can retain."""
        pairs = 17
        dofs = pairs * 3
        indices = np.arange(dofs, dtype=np.int32).reshape(pairs, 3)
        z = np.zeros((dofs, dofs), dtype=np.float32)
        bias = np.zeros(dofs, dtype=np.float32)
        kinds = np.tile(np.array([CONTACT, FRICTION, FRICTION], np.int32), pairs)
        parents = np.full(dofs, -1, dtype=np.int32)
        mu = np.zeros(dofs, dtype=np.float32)
        for pair in range(pairs):
            row = pair * 3
            z[row, row] = 1.0 + 0.01 * pair
            z[row + 1, row : row + 3] = (0.11, 0.6 + 0.08 * pair, 0.13)
            z[row + 2, row : row + 3] = (-0.07, 0.19, 1.8 - 0.05 * pair)
            bias[row : row + 3] = (-0.2 - 0.01 * pair, 0.4, 0.7 - 0.02 * pair)
            parents[row + 1 : row + 3] = row
            mu[row + 1 : row + 3] = 0.4 + 0.005 * pair
        diag = np.sum(z * z, axis=1) + 0.01
        initial = np.zeros(dofs, dtype=np.float32)
        velocity, impulses = _sweep(
            z,
            z,
            initial,
            bias,
            diag,
            kinds,
            parents,
            mu,
            initial,
            active_rows=True,
            iterations=8,
        )
        self.assertEqual(np.count_nonzero(impulses[::3] > 0), pairs)

        def padded(values):
            result = np.zeros((3, 192, *values.shape[1:]), dtype=values.dtype)
            result[:, :dofs] = values
            return result

        data = {
            "world_constraint_count": np.array([dofs, 0, dofs], np.int32),
            "rhs_bias": padded(bias),
            "diagonal": padded(diag),
            "initial_impulses": padded(initial),
            "row_factor": padded(np.stack([z[row, indices[row // 3]] for row in range(dofs)])),
            "row_incident": padded(initial),
            "row_type": padded(kinds),
            "row_parent": padded(parents),
            "row_mu": padded(mu),
            "support_row_class": padded(np.repeat(np.arange(pairs, dtype=np.int32), 3)),
            "support_class_dof": indices,
            "iterations": np.array(8),
            "omega": np.array(1.0),
            "reference_factor_delta": np.zeros((3, dofs), dtype=np.float32),
        }
        physical = {
            "factor": np.eye(dofs),
            "initial": initial,
            "expected_velocity": velocity,
            "expected_impulse": impulses,
        }
        control = DeviceSolve(data, candidate=False)
        control.launch()
        candidate = DeviceSolve(data, candidate=True)
        candidate.launch()
        assert_output(self, data, *candidate.result(), physical=physical, reference=control.result())

    @unittest.skipUnless(wp.is_cuda_available(), "Requires root-owned CUDA lease")
    def test_finite_fallback_and_distinct_live_worlds_cuda(self):
        """Keep legal fallback rows and isolate each world's records."""
        for mode in (
            "two_live_slots",
            "zero_normal_diag",
            "negative_normal_diag",
            "tail_normal",
            "different_normal_support",
        ):
            with self.subTest(mode=mode):
                data, _ = fixture(seeded=True)
                count = int(data["world_constraint_count"][0])
                # Both warps of block0 must be live and physically different.
                data["world_constraint_count"][1] = count
                data["row_factor"][1] *= 0.55
                data["row_incident"][1] *= 0.55
                data["diagonal"][1] = (data["diagonal"][1] - 1e-6) * 0.55**2 + 0.015
                data["row_mu"][1] *= 0.4
                data["rhs_bias"][1] *= 0.7
                if mode in ("zero_normal_diag", "negative_normal_diag"):
                    # The original kernel skips this normal but still visits
                    # its legal pair at the incoming seeded normal load.
                    data["diagonal"][:, 1] = 0.0 if mode == "zero_normal_diag" else -0.5
                elif mode == "tail_normal":
                    # The last valid row is an ordinary contact normal without
                    # children; do not construct a malformed t1 with no sibling.
                    data["world_constraint_count"][:] = 5
                    self.assertEqual(data["row_type"][0, 4], CONTACT)
                    self.assertEqual(data["row_parent"][0, 4], -1)
                elif mode == "different_normal_support":
                    classes = np.repeat(data["support_class_dof"], 2, axis=0)
                    classes[1, -1] = -1
                    data["support_class_dof"] = classes
                    data["support_row_class"][:, 1] = 1
                    data["row_factor"][:, 1, -1] = 0.0
                    data["diagonal"][:, 1] = np.sum(data["row_factor"][:, 1] ** 2, axis=1) + 1e-6
                    self.assertTrue(np.all(data["support_row_class"][:, 2:4] == 0))
                control = DeviceSolve(data, candidate=False)
                candidate = DeviceSolve(data, candidate=True)
                control.launch()
                candidate.launch()
                assert_output(self, data, *candidate.result(), reference=control.result())

    @unittest.skipUnless(wp.is_cuda_available(), "Requires root-owned CUDA lease")
    def test_graph_empty_regrow_changed_metric_cuda(self):
        """Rewrite records as captured buffers change count, response and CFM."""
        data, _ = fixture(offset=30, seeded=True)
        control = DeviceSolve(data, candidate=False)
        candidate = DeviceSolve(data, candidate=True)
        control.launch()
        candidate.launch()
        with wp.ScopedCapture(device=candidate.device) as captured:
            candidate.launch()
        original = {
            key: data[key].copy() for key in ("world_constraint_count", "row_factor", "diagonal", "row_incident")
        }
        for loaded, factor_scale, cfm in ((True, 1.0, 1e-6), (False, 1.0, 1e-6), (True, 0.6, 0.01), (True, 1.0, 1e-6)):
            data["world_constraint_count"] = original["world_constraint_count"] if loaded else np.zeros(3, np.int32)
            data["row_factor"] = original["row_factor"] * factor_scale
            data["row_incident"] = original["row_incident"] * factor_scale
            data["diagonal"] = (original["diagonal"] - 1e-6) * factor_scale**2 + cfm
            for runner in (control, candidate):
                for slot, key in (
                    (1, "world_constraint_count"),
                    (4, "diagonal"),
                    (7, "row_factor"),
                    (8, "row_incident"),
                ):
                    runner.inputs[slot].assign(data[key])
                # The contact builder rewrites the records with the rows each step.
                runner.rebuild_records(data)
                runner.reset_impulses()
            control.launch()
            wp.capture_launch(captured.graph)
            assert_output(self, data, *candidate.result(), reference=control.result())


class TestSparseContactTripletsPatch(unittest.TestCase):
    """Fuse patch anchor units, pooling their ring loads, in the linked active and full sweeps."""

    def test_linked_factory_cpu(self):
        """Admit ring-linked normals to contact units in either sweep, and keep point variants unlinked."""
        options = {"use_support_table": True, "contact_triplets": True}
        active = sweep_kernel(SparseSweep(192, 12, 12, active_rows=True, linked_rows=True, **options))
        full = sweep_kernel(SparseSweep(192, 12, 12, linked_rows=True, **options))
        point = sweep_kernel(SparseSweep(192, 12, 12, active_rows=True, **options))
        self.assertTrue(active.key.endswith("_active_support_table_contact_triplets_linked"))
        self.assertTrue(full.key.endswith("_12_12_support_table_contact_triplets_linked"), full.key)
        self.assertTrue(point.key.endswith("_active_support_table_contact_triplets"))
        for kernel in (active, full, point):
            self.assertIn("contact_records", kernel.adj.arg_types)
        # Point friction fuses only in the active sweep, and nothing fuses without support tables.
        for unfused in ({"use_support_table": True}, {"active_rows": True, "linked_rows": True}):
            with self.subTest(options=unfused):
                kernel = sweep_kernel(SparseSweep(192, 12, 12, contact_triplets=True, **unfused))
                self.assertNotIn("contact_triplets", kernel.key)

    def test_patch_triplet_admission_cpu(self):
        """Fuse patch triplets in the linked active scan or the full sweep; never in packed sweeps."""
        cases = (
            # Probe changes, active rows, support table, lanes, fits; admitted, reason.
            ({}, True, True, 32, True, True, "active-row scan"),
            ({}, False, True, 32, True, True, "full ordered sweep"),
            ({}, False, True, 16, True, False, "16-lane"),
            ({}, True, False, 32, True, False, "support table"),
            ({}, True, True, 32, False, False, "room for contact units"),
            ({"pins": {"contact_units": None}}, True, True, 32, True, False, "contact_units=None"),
        )
        for changes, active, table, lanes, fits, admitted, phrase in cases:
            with self.subTest(changes=changes, active=active, table=table, lanes=lanes, fits=fits):
                decision, reason = class_plan.contact_triplets_rule(
                    SolverFeatherPGS.Pins(**changes.get("pins", {})),
                    active_rows=active,
                    use_support_table=table,
                    lanes=lanes,
                    fits=lambda fits=fits: fits,
                )
                self.assertEqual(decision, admitted)
                self.assertIn(phrase, reason)

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_linked_triplets_equal_linked_sweeps_cuda(self):
        """Match the same linked sweep without triplets to rounding, and the reference.

        Self-loops, two- and three-member rings, four- and six-member rings loaded only
        through their last member, a four-member loaded ring whose sliding anchors' radii
        round by the ring order, a six-member loaded ring whose anchors the active scan
        fuses, interleaved one- and three-row units and an unlinked anchor, with and
        without a 30-row lead that moves records and rings across 32-row words; zero and
        seeded impulses and relaxation, in the linked active scan and in
        the full sweep. The six-member rings catch a fused radius, in either sweep, or a
        pair selection that stops after up to five members. The exit law stops both sweeps
        within rounding of each other: each change against the same floor as its ordinary update's.
        """
        settings = (
            {},
            {"seeded": True, "omega": 0.7},
            {"seeded": True, "omega": 1.2},
            {"seeded": True, "omega": -0.5},
        )
        for (name, layout), pad, options, active in product(_LINKED_LAYOUTS.items(), (0, 30), settings, (True, False)):
            with self.subTest(layout=name, pad=pad, active=active, **options):
                data, physical = linked_fixture(layout, pad=pad, active=active, **options)
                control = DeviceSolve(data, candidate=False, active=active, linked=True)
                candidate = DeviceSolve(data, candidate=True, active=active, linked=True)
                self.assertIn("_contact_triplets", candidate.kernel.key)
                self.assertIn("_linked", candidate.kernel.key)
                self.assertTrue(candidate.units)
                self.assertNotIn("_contact_triplets", control.kernel.key)
                control.launch()
                candidate.launch()
                delta, impulses = candidate.result()
                assert_output(self, data, delta, impulses, reference=control.result())
                # The oracle pools each ring as the ordinary update does; a fused radius
                # without the ring would leave it.
                assert_output(self, data, delta, impulses, physical=physical)

    @unittest.skipUnless(wp.is_cuda_available(), "Sparse construction requires CUDA")
    def test_patch_host_admission_cuda(self):
        """Build the linked triplet sweep the patch policy selects, and keep a small world's packed sweep."""
        model = equal_dof_trees(4, 3, "cuda:0")
        for pins, suffix in (
            ({}, "_active_support_table_contact_triplets_linked"),
            ({"active_rows": False}, "_18_12_support_table_contact_triplets_linked"),
            ({"contact_units": None}, "_active_support_table_linked"),
            ({"active_rows": False, "contact_units": None}, "_18_12_support_table"),
        ):
            with (
                self.subTest(pins=pins),
                mock.patch.dict(os.environ, {}, clear=True),
            ):
                solver = SolverFeatherPGS(
                    model,
                    pins=SolverFeatherPGS.Pins(**pins),
                    enable_joint_velocity_limits=False,
                    pgs_warmstart=False,
                    dense_max_constraints=192,
                )
                self.assertEqual(solver.spec.classes.packed_size, 18)
                key = solver.sweep.sparse_kernel.key
                # A sphere on each link touches at one point: no region of this model has a footprint, so no
                # contact carries a spin row and the sweep is the plain variant (RowLaw.SPIN).
                self.assertFalse(solver.laws.spin_rows)
                self.assertTrue(key.endswith(suffix), key)
                self.assertEqual(solver.sweep.sparse_contact_triplets, "contact_triplets" in suffix)
        with mock.patch.dict(os.environ, {}, clear=True):
            packed = _make_solver(_build_model(), True)
        self.assertEqual(packed.sweep.sparse_solve_lanes, 16)
        self.assertFalse(packed.sweep.sparse_contact_triplets)
        self.assertIn("16-lane", packed.sweep.sparse_contact_triplets_reason)


def check_frozen_payload(path, *, device="cuda:0"):
    """Root-only diagnostic replay of one actual512-world refresh/held NPZ.

    Source/manifest/hash verification belongs to the external runner; this
    reusable adapter never picks a file implicitly or changes a production test.
    """
    with np.load(Path(path), allow_pickle=False) as payload:
        data = {key: payload[key].copy() for key in payload.files}
    assert int(data["iterations"]) == 8
    assert data["row_factor"].shape[0] == len(data["world_ids"]) <= 512
    control = DeviceSolve(data, candidate=False, device=device)
    candidate = DeviceSolve(data, candidate=True, device=device)
    control.launch()
    candidate.launch()
    test = unittest.TestCase()
    reference = (data["reference_factor_delta"], data["reference_impulses"])
    assert_output(test, data, *control.result(), reference=reference)
    assert_output(test, data, *candidate.result(), reference=control.result())
    return {
        "path": str(path),
        "worlds": len(data["world_ids"]),
        "same_input_output_check": "pass",
        "independent_response_and_cone_check": "pass",
    }


if __name__ == "__main__":
    unittest.main()
