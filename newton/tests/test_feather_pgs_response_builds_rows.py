# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The free-body and factor-row builds of the response operators (CPU checks).

Each build replaces a module-level kernel, function or kernel factory of the solver (:data:`DROP_INS`): it takes
the same parameters in the same order and the same launch. On the CPU, where neither build fuses a multiply-add,
every build reproduces the replaced code's outputs bit for bit on random inputs: free bodies (inverses, responses
and right-hand sides, body maps and the warm-start publication), sparse-diagonal rows (contact slots, diagonals,
compact-boundary rows and limits), the bilateral pre-elimination, and the factor rows (the internal rows' unit
columns, contact rows, the factor and its publication, the paired factor rows). With ``ptxas`` on ``PATH`` the
native builds also assemble offline for CUDA.
"""

import importlib
import inspect
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import (
    compact_contact,
    field_group,
    kernels,
    paired_components,
    solver_feather_pgs,
    sparse_contact,
    sparse_mass_matrix,
)
from newton._src.solvers.feather_pgs import response_builds_rows as builds
from newton._src.solvers.feather_pgs.compact_contact import CompactBoundary
from newton._src.solvers.feather_pgs.contact_input import ContactInput
from newton._src.solvers.feather_pgs.dense_rows import DenseRows
from newton._src.solvers.feather_pgs.friction_patches import FrictionPatches
from newton._src.solvers.feather_pgs.paired_components import PairedComponentData
from newton._src.solvers.feather_pgs.response_operator import FactorRows, FreeBody6, SparseFactor, SparsePairJY
from newton._src.solvers.feather_pgs.sparse_mass_matrix import _SparseRowSupport, factor_indices
from newton._src.solvers.feather_pgs.spatial_laws import inertia_record
from newton._src.solvers.feather_pgs.tree_plan import TreePlan

DEVICE = "cpu"


def _signature(kernel):
    return [(argument.label, wp.types.type_repr(argument.type)) for argument in kernel.adj.args]


def _bits(value: np.ndarray) -> bytes:
    return np.ascontiguousarray(value).tobytes()


def _arrays(values):
    """The arrays among launch arguments (struct members included), by position and member."""
    found = []
    for position, value in enumerate(values):
        if isinstance(value, wp.array):
            found.append((f"{position}", value))
        elif hasattr(value, "_cls"):
            for name in value._cls.vars:
                member = getattr(value, name)
                if isinstance(member, wp.array) and member.ptr:
                    found.append((f"{position}.{name}", member))
    return found


class DifferentialCase(unittest.TestCase):
    """Launch the replaced and the replacing kernel on equal inputs and compare every array bit for bit."""

    def assert_same_launch(self, legacy, replacement, dim, make):
        """Both kernels write equal bits to every array; the replaced kernel writes something (the case is live)."""
        initial = [_bits(array.numpy()) for _, array in _arrays(make())]
        outputs = []
        for kernel in (legacy, replacement):
            values = make()
            wp.launch(kernel, dim=dim, inputs=values, device=DEVICE)
            outputs.append([(name, array.numpy()) for name, array in _arrays(values)])
        self.assertTrue(any(_bits(after) != start for start, (_, after) in zip(initial, outputs[0], strict=True)))
        for (name, before), (_, after) in zip(*outputs, strict=True):
            with self.subTest(argument=name):
                self.assertEqual(before.shape, after.shape)
                self.assertEqual(_bits(before), _bits(after), f"argument {name} differs")


def _array(values, dtype):
    return wp.array(np.ascontiguousarray(values), dtype=dtype, device=DEVICE)


def _spatial_inertia(rng):
    """A random spatial inertia about a point off the center of mass: symmetric positive definite."""
    mass = rng.uniform(0.2, 3.0)
    offset = rng.uniform(-0.3, 0.3, 3)
    axes = rng.normal(size=(3, 3))
    inertia = axes @ axes.T + 0.1 * np.eye(3)
    skew = np.array([[0.0, -offset[2], offset[1]], [offset[2], 0.0, -offset[0]], [-offset[1], offset[0], 0.0]])
    matrix = np.zeros((6, 6))
    matrix[:3, :3] = mass * np.eye(3)
    matrix[:3, 3:] = -mass * skew
    matrix[3:, :3] = mass * skew
    matrix[3:, 3:] = inertia - mass * skew @ skew
    return matrix.astype(np.float32)


# ---------------------------------------------------------------------------------------------------------------------
# Free-body rows
# ---------------------------------------------------------------------------------------------------------------------


class FreeBodyWorlds:
    """Random worlds of free-body rows over a few free bodies (one articulation each)."""

    def __init__(self, seed=3, worlds=3, rows=9, bodies=7):
        rng = np.random.default_rng(seed)
        self.worlds, self.rows, self.bodies = worlds, rows, bodies
        self.count = rng.integers(0, rows + 1, worlds).astype(np.int32)
        self.count[0] = rows
        self.body_a = rng.integers(-1, bodies, (worlds, rows)).astype(np.int32)
        self.body_b = rng.integers(-1, bodies, (worlds, rows)).astype(np.int32)
        self.J_a = rng.normal(size=(worlds, rows, 6)).astype(np.float32)
        self.J_b = rng.normal(size=(worlds, rows, 6)).astype(np.float32)
        self.J_a[0, 0] = 0.0  # a zero row: A_ii = cfm
        self.inertia = np.stack([_spatial_inertia(rng) for _ in range(bodies)])
        self.inverse = np.linalg.inv(self.inertia.astype(np.float64)).astype(np.float32)
        self.phi = rng.uniform(-0.05, 0.05, (worlds, rows)).astype(np.float32)
        self.row_type = rng.choice([0, 0, 2, 4, 3], (worlds, rows)).astype(np.int32)
        self.target = rng.uniform(-0.2, 0.2, (worlds, rows)).astype(np.float32)
        self.restitution = np.where(rng.random((worlds, rows)) < 0.5, rng.uniform(0.1, 0.9, (worlds, rows)), 0.0)
        self.restitution = self.restitution.astype(np.float32)
        self.body_to_art = np.arange(bodies, dtype=np.int32)
        self.dof_start = (6 * np.arange(bodies)).astype(np.int32)
        self.world_offset = (6 * (np.arange(bodies) % 3)).astype(np.int32)
        self.velocity = rng.normal(size=6 * bodies).astype(np.float32)
        self.depenetration = rng.choice([0.0, 0.5, 2.0, np.inf], bodies).astype(np.float32)
        self.impulses = np.where(rng.random((worlds, rows)) < 0.3, 0.0, rng.normal(size=(worlds, rows)))
        self.impulses = self.impulses.astype(np.float32)
        self.MiJt_a = rng.normal(size=(worlds, rows, 6)).astype(np.float32)
        self.MiJt_b = rng.normal(size=(worlds, rows, 6)).astype(np.float32)


class TestFreeBody6Build(DifferentialCase):
    """Free-body rows: inverses, responses with right-hand sides, the body map, coordinates and the publication."""

    def setUp(self):
        self.w = FreeBodyWorlds()

    def test_signatures_are_the_legacy_signatures(self):
        build = FreeBody6.build()
        for legacy, replacement in (
            (kernels.compute_mf_body_Hinv, build.inverse),
            (kernels.compute_mf_effective_mass_and_rhs, build.rows),
            (kernels.build_mf_body_map, build.body_map),
            (kernels.compute_mf_world_dof_offsets, build.coordinates),
            (kernels.apply_mf_warmstart_impulses, build.publish),
        ):
            with self.subTest(kernel=legacy.key):
                self.assertEqual(_signature(replacement), _signature(legacy))

    def test_inverse(self):
        w = self.w
        flags = np.zeros(w.bodies, dtype=np.int32)
        flags[2] = int(kernels.BodyFlags.KINEMATIC)
        free = np.ones(w.bodies, dtype=np.int32)
        free[4] = 0
        body_to_art = w.body_to_art.copy()
        body_to_art[5] = -1

        def make():
            return [
                _array(np.arange(w.bodies, dtype=np.int32), wp.int32),
                _array(w.inertia, wp.spatial_matrix),
                _array(free, wp.int32),
                _array(body_to_art, wp.int32),
                _array(flags, wp.int32),
                _array(np.full((w.bodies, 6, 6), 7.0, np.float32), wp.spatial_matrix),
            ]

        self.assert_same_launch(kernels.compute_mf_body_Hinv, FreeBody6.build().inverse, w.bodies, make)

    def _rows_inputs(self, contact_w, has_target):
        w = self.w

        def make():
            return [
                _array(w.count, wp.int32),
                _array(w.body_a, wp.int32),
                _array(w.body_b, wp.int32),
                _array(w.J_a, wp.float32),
                _array(w.J_b, wp.float32),
                _array(w.inverse, wp.spatial_matrix),
                _array(w.phi, wp.float32),
                _array(w.row_type, wp.int32),
                _array(w.target, wp.float32),
                _array(w.restitution, wp.float32),
                has_target,
                _array(w.body_to_art, wp.int32),
                _array(w.dof_start, wp.int32),
                _array(w.velocity, wp.float32),
                _array(w.depenetration, wp.float32),
                1.0e-4,
                0.3,
                contact_w,
                1.0 / 60.0,
                0.5,
                0.05,
                w.rows,
                _array(np.full((w.worlds, w.rows), -1.0, np.float32), wp.float32),
                _array(np.full((w.worlds, w.rows, 6), -2.0, np.float32), wp.float32),
                _array(np.full((w.worlds, w.rows, 6), -3.0, np.float32), wp.float32),
                _array(np.full((w.worlds, w.rows), -4.0, np.float32), wp.float32),
                _array(np.full((w.worlds, w.rows), -5.0, np.float32), wp.float32),
            ]

        return make

    def test_rows(self):
        for contact_w, has_target in ((1.0, 1), (0.6, 1), (0.6, 0)):
            with self.subTest(contact_w=contact_w, has_target=has_target):
                self.assert_same_launch(
                    kernels.compute_mf_effective_mass_and_rhs,
                    FreeBody6.build().rows,
                    self.w.worlds * self.w.rows,
                    self._rows_inputs(contact_w, has_target),
                )

    def test_body_map(self):
        w = self.w
        for capacity in (64, 3):
            with self.subTest(capacity=capacity):

                def make(capacity=capacity):
                    return [
                        _array(w.count, wp.int32),
                        _array(w.body_a, wp.int32),
                        _array(w.body_b, wp.int32),
                        _array(w.body_to_art, wp.int32),
                        _array(w.dof_start, wp.int32),
                        capacity,
                        _array(np.full((w.worlds, capacity), -9, np.int32), wp.int32),
                        _array(np.full((w.worlds, capacity), -9, np.int32), wp.int32),
                        _array(np.full(w.worlds, -9, np.int32), wp.int32),
                        _array(np.full((w.worlds, w.rows), -9, np.int32), wp.int32),
                        _array(np.full((w.worlds, w.rows), -9, np.int32), wp.int32),
                    ]

                self.assert_same_launch(kernels.build_mf_body_map, FreeBody6.build().body_map, w.worlds, make)

    def test_coordinates(self):
        w = self.w

        def make():
            return [
                _array(w.count, wp.int32),
                _array(w.body_a, wp.int32),
                _array(w.body_b, wp.int32),
                _array(w.body_to_art, wp.int32),
                _array(w.world_offset, wp.int32),
                w.rows,
                _array(np.full((w.worlds, w.rows), -9, np.int32), wp.int32),
                _array(np.full((w.worlds, w.rows), -9, np.int32), wp.int32),
            ]

        self.assert_same_launch(
            kernels.compute_mf_world_dof_offsets, FreeBody6.build().coordinates, w.worlds * w.rows, make
        )

    def test_publish(self):
        w = self.w
        capacity = 8
        bodies = np.zeros(w.worlds, np.int32)
        dof_start = np.zeros((w.worlds, capacity), np.int32)
        local_a = np.full((w.worlds, w.rows), -1, np.int32)
        local_b = np.full((w.worlds, w.rows), -1, np.int32)
        for world in range(w.worlds):
            listed = []
            for row in range(w.count[world]):
                for body, local in ((w.body_a[world, row], local_a), (w.body_b[world, row], local_b)):
                    if body >= 0:
                        if body not in listed:
                            listed.append(body)
                        local[world, row] = listed.index(body)
            bodies[world] = len(listed)
            dof_start[world, : len(listed)] = w.dof_start[listed]
        for delta in (0, 1):
            with self.subTest(set_output_to_delta=delta):

                def make(delta=delta):
                    return [
                        _array(w.count, wp.int32),
                        _array(bodies, wp.int32),
                        _array(dof_start, wp.int32),
                        _array(local_a, wp.int32),
                        _array(local_b, wp.int32),
                        _array(w.MiJt_a, wp.float32),
                        _array(w.MiJt_b, wp.float32),
                        _array(w.impulses, wp.float32),
                        capacity,
                        delta,
                        _array(w.velocity, wp.float32),
                    ]

                self.assert_same_launch(
                    kernels.apply_mf_warmstart_impulses, FreeBody6.build().publish, w.worlds * capacity * 6, make
                )

    def test_meta_word_is_the_consumed_word(self):
        """The packed word's fields are the ones :meth:`.FreeBody6.consume` decodes."""
        word = FreeBody6().pack("w", "a", "b", "inv", "rhs", "kind", "parent", "")
        self.assertIn("w.x = (a << 16) | (b & 0xFFFF);", word)
        self.assertIn("w.y = __float_as_int(inv);", word)
        consume = FreeBody6().consume("i", "n", "")
        self.assertIn("int dof_a = packed_dofs >> 16;", consume)
        self.assertIn("int dof_b = (packed_dofs << 16) >> 16;", consume)
        self.assertIn("float mf_diag = __int_as_float(meta.y);", consume)
        with self.assertRaises(ValueError):
            FreeBody6.build().meta(70000, "sm_120")


# ---------------------------------------------------------------------------------------------------------------------
# Sparse-diagonal rows
# ---------------------------------------------------------------------------------------------------------------------


class ContactWorlds:
    """Random contacts between bodies of single-coordinate (sparse) and six-coordinate (dense) articulations.

    Articulation ``k`` holds body ``k``; the first ``dense`` articulations have six response coordinates, the
    others ``size``; some bodies have no single response coordinate.
    """

    def __init__(self, seed=5, worlds=2, contacts=12, size=2, dense=2, articulations=7, rows=40):
        rng = np.random.default_rng(seed)
        self.worlds, self.contacts, self.size, self.rows = worlds, contacts, size, rows
        dofs = np.where(np.arange(articulations) < dense, 6, size).astype(np.int32)
        self.response_dofs = dofs
        self.dof_start = np.concatenate([[0], np.cumsum(dofs)[:-1]]).astype(np.int32)
        self.world_offset = (self.dof_start % 9).astype(np.int32)
        bodies = articulations
        self.single_dof = np.where(rng.random(bodies) < 0.8, self.dof_start + rng.integers(0, size, bodies), -1).astype(
            np.int32
        )
        self.single_dof[:dense] = -1
        self.body_mask = rng.integers(1, 64, bodies).astype(np.uint32)
        total = int(dofs.sum())
        self.motion = rng.normal(size=(total, 6)).astype(np.float32)
        self.inverse_mass = rng.uniform(0.1, 2.0, total).astype(np.float32)
        self.origin = rng.normal(size=(articulations, 3)).astype(np.float32)
        shapes = bodies + 1
        self.shape_body = np.concatenate([np.arange(bodies), [-1]]).astype(np.int32)
        self.shape0 = rng.integers(0, shapes, contacts).astype(np.int32)
        self.shape1 = rng.integers(0, shapes, contacts).astype(np.int32)
        self.shape1[0] = self.shape0[0]  # both endpoints on one body: the slots merge
        body0, body1 = self.shape_body[self.shape0], self.shape_body[self.shape1]
        self.art0 = np.where(body0 >= 0, body0, -1).astype(np.int32)
        self.art1 = np.where(body1 >= 0, body1, -1).astype(np.int32)
        normal = rng.normal(size=(contacts, 3))
        self.normal = (normal / np.linalg.norm(normal, axis=1, keepdims=True)).astype(np.float32)
        self.point0 = rng.normal(size=(contacts, 3)).astype(np.float32)
        self.point1 = rng.normal(size=(contacts, 3)).astype(np.float32)
        self.margin0 = rng.uniform(0.0, 0.01, contacts).astype(np.float32)
        self.margin1 = rng.uniform(0.0, 0.01, contacts).astype(np.float32)
        self.world = rng.integers(0, worlds, contacts).astype(np.int32)
        self.slots_needed = rng.choice([1, 3], contacts).astype(np.int32)
        self.path = np.where(rng.random(contacts) < 0.15, 1, 0).astype(np.int32)
        self.slot = np.zeros(contacts, np.int32)
        used = np.zeros(worlds, np.int32)
        for c in range(contacts):
            self.slot[c] = used[self.world[c]]
            used[self.world[c]] += self.slots_needed[c]
        self.slot[3] = -1
        quat = rng.normal(size=(bodies, 4))
        quat /= np.linalg.norm(quat, axis=1, keepdims=True)
        self.body_q = np.concatenate([rng.normal(size=(bodies, 3)), quat], axis=1).astype(np.float32)
        self.patch_a = rng.normal(size=(contacts, 3)).astype(np.float32)
        self.patch_b = rng.normal(size=(contacts, 3)).astype(np.float32)
        self.count = np.array([contacts], np.int32)

    def contact_input(self, shared_anchor=0, cfm=1.0e-4):
        w = self
        return field_group.upload(
            ContactInput,
            DEVICE,
            count=w.count,
            point0=_array(w.point0, wp.vec3),
            point1=_array(w.point1, wp.vec3),
            normal=_array(w.normal, wp.vec3),
            shape0=w.shape0,
            shape1=w.shape1,
            margin0=w.margin0,
            margin1=w.margin1,
            world=w.world,
            slot=w.slot,
            art0=w.art0,
            art1=w.art1,
            path=w.path,
            slots_needed=w.slots_needed,
            shape_body=w.shape_body,
            body_q=_array(w.body_q, wp.transform),
            body_mask=w.body_mask,
            response_dofs=w.response_dofs,
            dof_start=w.dof_start,
            origin=_array(w.origin, wp.vec3),
            motion=_array(w.motion, wp.spatial_vector),
            shared_anchor=shared_anchor,
            friction_shared_anchor=0,
            cfm=cfm,
            diag=np.full((w.worlds, w.rows), -7.0, np.float32),
        )

    def patches(self, enabled):
        patches = FrictionPatches()
        patches.enabled = enabled
        patches.point_a = _array(self.patch_a, wp.vec3)
        patches.point_b = _array(self.patch_b, wp.vec3)
        return patches


def _lower_factors(rng, groups, size):
    factor = np.tril(rng.normal(size=(groups, size, size))) * 0.3
    for group in range(groups):
        factor[group][np.diag_indices(size)] = rng.uniform(0.5, 2.0, size)
    factor[0, 2, 2] = 0.0  # a zero pivot: the solves' zero convention
    return factor.astype(np.float32)


@wp.kernel
def _legacy_row_response(
    data: ContactInput,
    boundary: CompactBoundary,
    rows: wp.array2d[int],
    frames: wp.array2d[wp.vec3],
):
    t = wp.tid()
    compact_contact._publish_row_response(
        data,
        boundary,
        rows[t, 0],
        rows[t, 1],
        rows[t, 2],
        rows[t, 3] != 0,
        rows[t, 4],
        rows[t, 5],
        rows[t, 6],
        rows[t, 7],
        frames[t, 0],
        frames[t, 1],
        frames[t, 2],
    )


@wp.kernel
def _row_response(
    data: ContactInput,
    boundary: CompactBoundary,
    rows: wp.array2d[int],
    frames: wp.array2d[wp.vec3],
):
    t = wp.tid()
    builds.sparse_pair_row_response(
        data,
        boundary,
        rows[t, 0],
        rows[t, 1],
        rows[t, 2],
        rows[t, 3] != 0,
        rows[t, 4],
        rows[t, 5],
        rows[t, 6],
        rows[t, 7],
        frames[t, 0],
        frames[t, 1],
        frames[t, 2],
    )


class TestSparsePairBuild(DifferentialCase):
    """Sparse-diagonal rows: contact slots, the sparse diagonal term, compact-boundary rows and the limit prefix."""

    def setUp(self):
        self.w = ContactWorlds()

    def test_signatures_are_the_legacy_signatures(self):
        build = SparsePairJY.build()
        for legacy, replacement in (
            (kernels.populate_sparse_diagonal_contact_response, build.contact_rows),
            (kernels.accumulate_sparse_diagonal_response_diag, build.diagonal),
            (compact_contact.produce_limit_response, build.limit_response),
        ):
            with self.subTest(kernel=legacy.key):
                self.assertEqual(_signature(replacement), _signature(legacy))
        legacy, replacement = compact_contact._publish_row_response, build.row_response
        self.assertEqual(
            [(a.label, wp.types.type_repr(a.type)) for a in legacy.adj.args],
            [(a.label, wp.types.type_repr(a.type)) for a in replacement.adj.args],
        )

    def test_contact_rows(self):
        w = self.w
        for enabled, shared in ((0, 0), (1, 0), (0, 1)):
            with self.subTest(patches=enabled, shared_anchor=shared):

                def make(enabled=enabled, shared=shared):
                    return [
                        w.contact_input(shared_anchor=shared),
                        3,
                        w.size,
                        _array(w.world_offset, wp.int32),
                        _array(w.single_dof, wp.int32),
                        _array(w.inverse_mass, wp.float32),
                        w.patches(enabled),
                        _array(np.full((w.worlds, w.rows, 2), -9, np.int32), wp.int32),
                        _array(np.full((w.worlds, w.rows, 4), -9.0, np.float32), wp.float32),
                    ]

                self.assert_same_launch(
                    kernels.populate_sparse_diagonal_contact_response, SparsePairJY.build().contact_rows, 3, make
                )

    def test_diagonal(self):
        rng = np.random.default_rng(11)
        worlds, rows = 3, 10
        coords = rng.integers(-1, 4, (worlds, rows, 2)).astype(np.int32)
        jy = rng.normal(size=(worlds, rows, 4)).astype(np.float32)
        jy[0, 0] = [0.0, -0.0, -0.0, 0.0]

        def make():
            return [
                _array(np.array([rows, 4, 0], np.int32), wp.int32),
                rows,
                _array(coords, wp.int32),
                _array(jy, wp.float32),
                _array(rng.normal(size=(worlds, rows)).astype(np.float32), wp.float32),
            ]

        rng = np.random.default_rng(11)
        state = rng.bit_generator.state

        def fresh():
            rng.bit_generator.state = state
            return make()

        self.assert_same_launch(
            kernels.accumulate_sparse_diagonal_response_diag, SparsePairJY.build().diagonal, worlds * rows, fresh
        )

    def _boundary(self, rng, groups):
        w = self.w
        return field_group.upload(
            CompactBoundary,
            DEVICE,
            body_single_dof=w.single_dof,
            dof_offset=w.world_offset,
            inverse_mass=w.inverse_mass,
            dense_group=np.arange(w.worlds, dtype=np.int32) % groups,
            factor=_lower_factors(rng, groups, 6),
            sparse_size=w.size,
            J=np.full((groups, w.rows, 6), -9.0, np.float32),
            Y=np.full((groups, w.rows, 6), -9.0, np.float32),
            sparse_dof=np.full((w.worlds, w.rows, 2), -9, np.int32),
            sparse_jy=np.full((w.worlds, w.rows, 4), -9.0, np.float32),
        )

    def test_row_response(self):
        w = self.w
        rng = np.random.default_rng(7)
        threads = 16
        rows = np.stack(
            [
                rng.integers(0, w.worlds, threads),
                np.arange(threads),
                rng.integers(0, 2, threads),
                rng.integers(0, 2, threads),
                rng.integers(-1, 7, threads),
                rng.integers(-1, 7, threads),
                rng.integers(-1, 7, threads),
                rng.integers(-1, 7, threads),
            ],
            axis=1,
        ).astype(np.int32)
        rows[0, 4:8] = [0, 0, 0, 0]  # both endpoints on one dense body
        rows[1, 4:8] = [3, 3, 3, 3]  # both endpoints on one sparse body: the slots merge
        frames = rng.normal(size=(threads, 3, 3)).astype(np.float32)
        seed = rng.bit_generator.state

        def make():
            rng.bit_generator.state = seed
            return [w.contact_input(), self._boundary(rng, 2), _array(rows, wp.int32), _array(frames, wp.vec3)]

        self.assert_same_launch(_legacy_row_response, _row_response, threads, make)

    def test_limit_response(self):
        w = self.w
        rng = np.random.default_rng(9)
        groups, worlds, capacity = 2, 3, 14
        J = rng.normal(size=(groups, capacity, 6)).astype(np.float32)
        J[1, 0] = 0.0
        seed = rng.bit_generator.state

        def make():
            rng.bit_generator.state = seed
            return [
                _array(np.array([[0, 12], [0, 5], [0, 0]], np.int32), wp.int32),
                _array(np.array([14, 9, 3], np.int32), wp.int32),
                _array(np.array([0, 1, 0], np.int32), wp.int32),
                _array(_lower_factors(rng, groups, 6), wp.float32),
                _array(J, wp.float32),
                _array(np.full((groups, capacity, 6), -9.0, np.float32), wp.float32),
                _array(rng.uniform(0.0, 1.0e-3, (worlds, capacity)).astype(np.float32), wp.float32),
                _array(np.full((worlds, capacity), -9.0, np.float32), wp.float32),
            ]

        del w
        self.assert_same_launch(
            compact_contact.produce_limit_response, SparsePairJY.build().limit_response, (worlds, 12), make
        )


# ---------------------------------------------------------------------------------------------------------------------
# Bilateral pre-elimination
# ---------------------------------------------------------------------------------------------------------------------


class BilateralGroup:
    """A size group of three articulations (the last without pre-elimination), each its own world, with mimic and
    connect rows: ``Y = H^-1 J^T`` of a random positive definite ``H`` per articulation."""

    def __init__(self, seed=13, dofs=5, capacity=10):
        rng = np.random.default_rng(seed)
        self.dofs, self.capacity, self.arts = dofs, capacity, 3
        self.counts = np.array([capacity, 8, 6], np.int32)
        J = rng.normal(size=(3, capacity, dofs))
        J[0, 6] = 0.0  # a degenerate connect axis: dropped from the block
        J[0, 9] = 0.0  # another articulation's row: J_B Y_i is zero
        Y = np.zeros_like(J)
        for art in range(3):
            A = rng.normal(size=(dofs, dofs))
            H = A @ A.T + dofs * np.eye(dofs)
            Y[art] = np.linalg.solve(H, J[art].T).T
        self.J, self.Y = J.astype(np.float32), Y.astype(np.float32)
        self.mimic_slot = np.array([2, 1, -1], np.int32)
        self.mimic_art_start = np.array([0, 2, 3, 3], np.int32)
        self.mimic_art_list = np.array([0, 2, 1], np.int32)
        self.connect_slot = np.array([4, 3], np.int32)
        self.connect_art = np.array([0, 1], np.int32)
        self.art_to_preelim = np.array([0, 1, -1], np.int32)
        self.rhs = rng.normal(size=(3, capacity)).astype(np.float32)
        self.velocity = rng.normal(size=3 * dofs).astype(np.float32)

    def setup_inputs(self, n_mimic=3):
        g = self

        def make():
            return [
                _array(np.arange(3, dtype=np.int32), wp.int32),
                _array(g.art_to_preelim, wp.int32),
                _array(g.mimic_slot, wp.int32),
                _array(g.mimic_art_start, wp.int32),
                _array(g.mimic_art_list, wp.int32),
                n_mimic,
                _array(g.connect_slot, wp.int32),
                _array(g.connect_art, wp.int32),
                2,
                _array(g.J, wp.float32),
                _array(g.Y, wp.float32),
                g.dofs,
                1.0e-3,
                1.0e-7,
                _array(np.full(2 * kernels.PREELIM_MAX_ROWS, -9, np.int32), wp.int32),
                _array(np.full(2, -9, np.int32), wp.int32),
                _array(np.zeros(2 * kernels.PREELIM_MAX_ROWS**2, np.float32), wp.float32),
                _array(np.zeros(2 * kernels.PREELIM_MAX_ROWS, np.float32), wp.float32),
                _array(np.zeros(2 * kernels.PREELIM_MAX_ROWS**2, np.float32), wp.float32),
            ]

        return make

    def factored(self):
        """The setup's outputs (slots, block sizes, factors), from the replaced kernel."""
        values = self.setup_inputs()()
        wp.launch(kernels.preelim_setup_for_size, dim=3, inputs=values, device=DEVICE)
        return values[14], values[15], values[18]


class TestBilateralElimination(DifferentialCase):
    """The Schur transform of a DenseJY build: the block's setup, the responses' correction and the projection."""

    def setUp(self):
        self.g = BilateralGroup()

    def test_signatures_are_the_legacy_signatures(self):
        for legacy, replacement in (
            (kernels.preelim_setup_for_size, builds.ELIMINATION.setup),
            (kernels.preelim_correct_Y_for_size, builds.ELIMINATION.correct),
            (kernels.preelim_project_velocity_for_size, builds.ELIMINATION.project),
        ):
            with self.subTest(kernel=legacy.key):
                self.assertEqual(_signature(replacement), _signature(legacy))

    def test_setup(self):
        for n_mimic in (0, 3):
            with self.subTest(n_mimic=n_mimic):
                self.assert_same_launch(
                    kernels.preelim_setup_for_size, builds.ELIMINATION.setup, 3, self.g.setup_inputs(n_mimic)
                )

    def test_correct(self):
        g = self.g
        slots, sizes, factors = g.factored()

        def make():
            return [
                _array(np.arange(3, dtype=np.int32), wp.int32),
                _array(np.arange(3, dtype=np.int32), wp.int32),
                _array(g.art_to_preelim, wp.int32),
                _array(g.counts, wp.int32),
                wp.clone(slots),
                wp.clone(sizes),
                wp.clone(factors),
                _array(g.J, wp.float32),
                g.dofs,
                g.capacity,
                3,
                _array(g.Y, wp.float32),
            ]

        self.assert_same_launch(kernels.preelim_correct_Y_for_size, builds.ELIMINATION.correct, 3 * g.capacity, make)

    def test_project(self):
        g = self.g
        slots, sizes, factors = g.factored()

        def make():
            return [
                _array(np.arange(3, dtype=np.int32), wp.int32),
                _array(np.arange(3, dtype=np.int32), wp.int32),
                _array(g.art_to_preelim, wp.int32),
                _array((g.dofs * np.arange(3)).astype(np.int32), wp.int32),
                wp.clone(slots),
                wp.clone(sizes),
                wp.clone(factors),
                _array(g.J, wp.float32),
                _array(g.Y, wp.float32),
                _array(g.rhs, wp.float32),
                g.dofs,
                _array(g.velocity, wp.float32),
            ]

        self.assert_same_launch(kernels.preelim_project_velocity_for_size, builds.ELIMINATION.project, 3, make)


# ---------------------------------------------------------------------------------------------------------------------
# Factor rows
# ---------------------------------------------------------------------------------------------------------------------


def _ancestor_pattern(rng, n):
    """A random tree's packed lower pattern: ``lookup[row, col]`` is the entry of ``(row, col)`` (-1 for none), with
    every row holding its ancestors (the columns of lower index on its path to the root)."""
    parent = np.array([-1] + [int(rng.integers(0, k)) for k in range(1, n)])
    lookup = np.full((n, n), -1, np.int32)
    entry = 0
    for row in range(n):
        chain, node = [], row
        while node >= 0:
            chain.append(node)
            node = parent[node]
        for col in sorted(chain):
            lookup[row, col] = entry
            entry += 1
    return parent, lookup, entry


class TestFactorPublish(DifferentialCase):
    """The factor rows' publication ``v = v_hat + P^T L^-T u``."""

    def test_publish(self):
        rng = np.random.default_rng(17)
        n, groups = 7, 3
        _parent, lookup, nnz = _ancestor_pattern(rng, n)
        permutation = rng.permutation(n).astype(np.int32)

        def make():
            return [
                _array(np.array([2, 0, 1], np.int32), wp.int32),
                _array(np.array([0, 1, 1], np.int32), wp.int32),
                _array(np.array([0, 0, 7], np.int32), wp.int32),
                _array(np.array([0, 7, 14], np.int32), wp.int32),
                _array(permutation, wp.int32),
                _array(lookup, wp.int32),
                _array(np.random.default_rng(1).normal(size=(groups, nnz)).astype(np.float32), wp.float32),
                _array(np.random.default_rng(2).normal(size=(2, 14)).astype(np.float32), wp.float32),
                _array(np.random.default_rng(3).normal(size=3 * n).astype(np.float32), wp.float32),
                _array(np.full(3 * n, -9.0, np.float32), wp.float32),
            ]

        self.assertEqual(
            _signature(SparseFactor.build().publish), _signature(sparse_contact.apply_sparse_factor_velocity)
        )
        self.assert_same_launch(
            sparse_contact.apply_sparse_factor_velocity, SparseFactor.build().publish, (groups, n), make
        )


class FactorGroup:
    """Two articulations of one factor group (a random tree of ``n`` coordinates), in two worlds, with their packed
    inverse factors and the rows' storage."""

    def __init__(self, seed=19, n=6, capacity=12, support=8):
        rng = np.random.default_rng(seed)
        self.n, self.capacity, self.support, self.worlds = n, capacity, support, 2
        parent, self.lookup, self.nnz = _ancestor_pattern(rng, n)
        self.inverse_permutation = rng.permutation(n).astype(np.int32)
        masks = np.zeros(n, np.uint64)
        for local in range(n):
            column = self.inverse_permutation[local]
            for node in range(n):
                if self.lookup[node, column] >= 0:
                    masks[local] |= np.uint64(1) << np.uint64(node)
        masks[1] |= np.uint64(1) << np.uint64(n - 1)  # a support coordinate outside the pattern: value zero
        self.dof_mask = masks
        self.inverse_factor = rng.normal(size=(2, self.nnz)).astype(np.float32)
        self.v_hat = rng.normal(size=2 * n).astype(np.float32)
        self.joint_q = rng.uniform(-1.0, 1.0, 2 * n).astype(np.float32)
        del parent

    def group(self):
        return [
            _array(np.array([1, 0], np.int32), wp.int32),
            _array(np.array([1, 0], np.int32), wp.int32),
            _array(np.array([0, 2], np.int32), wp.int32),
            _array(np.array([0, self.n], np.int32), wp.int32),
        ]

    def columns(self):
        return [
            _array(self.dof_mask, wp.uint64),
            _array(self.inverse_permutation, wp.int32),
            _array(self.lookup, wp.int32),
            _array(self.inverse_factor, wp.float32),
            _array(self.v_hat, wp.float32),
        ]

    def outputs(self):
        shape = (self.worlds, self.capacity)
        records = {
            field.name: np.full(shape, -9, np.int32 if field.dtype == wp.int32 else np.float32)
            for field in field_group.fields_of(DenseRows)
        }
        return [
            field_group.upload(DenseRows, DEVICE, **records),
            _array(np.full((*shape, self.support), -9, np.int32), wp.int32),
            _array(np.full((*shape, self.support), -9.0, np.float32), wp.float32),
            _array(np.full(shape, -9.0, np.float32), wp.float32),
        ]

    def support_table(self, enabled):
        return field_group.upload(
            _SparseRowSupport,
            DEVICE,
            enabled=enabled,
            body_class=np.zeros(1, np.int32),
            pair_class=np.zeros((1, 1), np.int32),
            class_dof=np.zeros((1, 1), np.int32),
            row_class=np.full((self.worlds, self.capacity), -9, np.int32),
            limit_class=np.arange(self.n, dtype=np.int32) + 3,
        )


class FactorContacts:
    """Contacts between two factored articulations (``n`` coordinates: two chains, each row's columns contiguous as
    a reverse depth-first order packs them), a free rigid body and the world, with their packed inverse factors."""

    def __init__(self, seed=41, n=6, contacts=10, rows=36, support=10):
        rng = np.random.default_rng(seed)
        self.n, self.contacts, self.rows, self.support = n, contacts, rows, support
        chains = (0, 3)
        first = np.array([max(c for c in chains if c <= node) for node in range(n)], np.int32)
        counts = np.arange(n) - first + 1
        self.row_offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
        nnz = int(self.row_offsets[-1])
        self.inverse_factor = rng.normal(size=(2, nnz)).astype(np.float32)
        self.free_inverse_factor = np.tril(rng.normal(size=(1, 6, 6))).astype(np.float32)
        self.permutation = rng.permutation(n).astype(np.int32)
        # Articulations 0 and 1 are factored (groups 0 and 1), 2 is a free rigid body (free group 0).
        self.response_dofs = np.array([n, n, 6], np.int32)
        self.dof_start = np.array([0, n, 2 * n], np.int32)
        self.world_offset = np.array([0, n, 0], np.int32)
        self.art_group = np.array([0, 1, 0], np.int32)
        self.free = np.array([0, 0, 1], np.int32)
        node_bits = lambda nodes: sum(np.uint64(1) << np.uint64(k) for k in nodes)  # noqa: E731
        self.body_mask = np.array(
            [node_bits([0]), node_bits([0, 1, 2]), node_bits([3, 4]), node_bits([0, 1]), node_bits([3, 4, 5]), 0],
            np.uint64,
        )
        self.body_art = np.array([0, 0, 0, 1, 1, 2], np.int32)
        bodies = 6
        self.shape_body = np.concatenate([np.arange(bodies), [-1]]).astype(np.int32)
        self.shape0 = rng.integers(0, bodies + 1, contacts).astype(np.int32)
        self.shape1 = rng.integers(0, bodies + 1, contacts).astype(np.int32)
        self.shape0[:4] = [0, 1, 5, 3]
        self.shape1[:4] = [6, 2, 6, 5]
        body0, body1 = self.shape_body[self.shape0], self.shape_body[self.shape1]
        self.art0 = np.where(body0 >= 0, self.body_art[body0], -1).astype(np.int32)
        self.art1 = np.where(body1 >= 0, self.body_art[body1], -1).astype(np.int32)
        normal = rng.normal(size=(contacts, 3))
        self.normal = (normal / np.linalg.norm(normal, axis=1, keepdims=True)).astype(np.float32)
        self.point0 = rng.normal(size=(contacts, 3)).astype(np.float32)
        self.point1 = rng.normal(size=(contacts, 3)).astype(np.float32)
        self.margin = rng.uniform(0.0, 0.01, contacts).astype(np.float32)
        self.world = np.zeros(contacts, np.int32)
        self.slots_needed = np.full(contacts, 3, np.int32)
        self.slots_needed[[1, 5]] = 1
        self.slot = np.concatenate([[0], np.cumsum(self.slots_needed)[:-1]]).astype(np.int32)
        self.path = np.zeros(contacts, np.int32)
        self.path[6] = 1
        quat = rng.normal(size=(bodies, 4))
        quat /= np.linalg.norm(quat, axis=1, keepdims=True)
        self.body_q = np.concatenate([rng.normal(size=(bodies, 3)), quat], axis=1).astype(np.float32)
        self.body_v = rng.normal(size=(bodies, 6)).astype(np.float32)
        self.motion = rng.normal(size=(2 * n + 6, 6)).astype(np.float32)
        self.v_hat = rng.normal(size=2 * n + 6).astype(np.float32)
        self.origin = rng.normal(size=(3, 3)).astype(np.float32)
        self.mu = rng.uniform(0.2, 1.0, bodies + 1).astype(np.float32)
        self.restitution = np.where(rng.random(bodies + 1) < 0.5, 0.6, 0.0).astype(np.float32)
        self.patch_phi = rng.normal(scale=0.01, size=(contacts, 2)).astype(np.float32)

    def contact_input(self):
        w = self
        records = {
            field.name: np.full((1, w.rows), -9, np.int32 if field.dtype == wp.int32 else np.float32)
            for field in field_group.fields_of(DenseRows)
        }
        return field_group.upload(
            ContactInput,
            DEVICE,
            count=np.array([w.contacts], np.int32),
            point0=_array(w.point0, wp.vec3),
            point1=_array(w.point1, wp.vec3),
            normal=_array(w.normal, wp.vec3),
            shape0=w.shape0,
            shape1=w.shape1,
            margin0=w.margin,
            margin1=w.margin,
            world=w.world,
            slot=w.slot,
            art0=w.art0,
            art1=w.art1,
            path=w.path,
            slots_needed=w.slots_needed,
            shape_body=w.shape_body,
            body_q=_array(w.body_q, wp.transform),
            body_v=_array(w.body_v, wp.spatial_vector),
            response_dofs=w.response_dofs,
            dof_start=w.dof_start,
            origin=_array(w.origin, wp.vec3),
            motion=_array(w.motion, wp.spatial_vector),
            prescribed=np.array([0, 1, 0], np.int32),
            material_mu=w.mu,
            material_restitution=w.restitution,
            incident=w.v_hat,
            shared_anchor=0,
            friction_shared_anchor=0,
            friction_scale=0.5,
            beta=0.2,
            cfm=1.0e-4,
            dt=1.0 / 60.0,
            bias_scale=1.0,
            speculative_scale=0.5,
            restitution_threshold=0.05,
            contact_w=1.0,
            **records,
        )

    def inputs(self, table: bool, records: bool):
        w = self
        patches = FrictionPatches()
        patches.enabled = 1
        patches.point_a = _array(w.point0, wp.vec3)
        patches.point_b = _array(w.point1, wp.vec3)
        patches.phi = _array(w.patch_phi, wp.vec2)
        support = field_group.upload(
            _SparseRowSupport,
            DEVICE,
            enabled=int(table),
            body_class=np.arange(6, dtype=np.int32) % 3,
            pair_class=np.arange(9, dtype=np.int32).reshape(3, 3),
            class_dof=np.zeros((9, w.support), np.int32),
            row_class=np.full((1, w.rows), -9, np.int32),
            limit_class=np.zeros(w.n, np.int32),
        )
        free = (
            np.full((1, w.rows // 3, 8), -9.0, np.float32)
            if records
            else np.full((1, w.rows, w.support), -9.0, np.float32)
        )
        return [
            w.contact_input(),
            3,
            _array(w.art_group, wp.int32),
            _array(w.free, wp.int32),
            _array(w.world_offset, wp.int32),
            _array(w.body_mask, wp.uint64),
            patches,
            _array(w.permutation, wp.int32),
            _array(w.row_offsets, wp.int32),
            _array(w.inverse_factor, wp.float32),
            _array(w.free_inverse_factor, wp.float32),
            0.1,
            _array(np.full((1, w.rows, w.support), -9, np.int32), wp.int32),
            _array(np.full((1, w.rows, w.support), -9.0, np.float32), wp.float32),
            _array(free, wp.float32),
            _array(np.full((1, w.rows), -9.0, np.float32), wp.float32),
            support,
        ]


class TestSparseFactorBuild(DifferentialCase):
    """Factor rows: the internal rows' packed columns of ``L^-1``."""

    def setUp(self):
        self.f = FactorGroup()

    def test_signatures_are_the_legacy_signatures(self):
        build = SparseFactor.build()
        for legacy, replacement in (
            (sparse_contact.build_sparse_joint_limit_rows, build.limit_rows),
            (sparse_contact.build_sparse_joint_velocity_limit_rows, build.velocity_limit_rows),
            (sparse_contact.build_sparse_mimic_rows, build.mimic_rows),
        ):
            with self.subTest(kernel=legacy.key):
                self.assertEqual(_signature(replacement), _signature(legacy))

    def test_factor(self):
        plan = TreePlan.build([-1, 0, 1, 1, 0], [1, 2, 1, 1, 1]).factor
        n, nnz, joints = plan.dof_count, plan.nonzero_count, 5
        rng = np.random.default_rng(43)
        records = []
        for _ in range(2 * joints):
            mass, offset = rng.uniform(0.5, 2.0), rng.uniform(-0.2, 0.2, 3)
            inertia = np.diag(rng.uniform(0.05, 0.2, 3)) + mass * (
                offset @ offset * np.eye(3) - np.outer(offset, offset)
            )
            records.append(np.concatenate([[mass], mass * offset, inertia.ravel()]))
        spd = []
        for _ in range(2):
            a = rng.normal(size=(n, n))
            h = a @ a.T + n * np.eye(n)
            spd.append(h[plan.entry_rows, plan.columns])
        for geometric, update in ((False, (1, 1)), (False, (0, 1)), (True, (1, 1))):
            with self.subTest(geometric=geometric, update=update):
                legacy = sparse_mass_matrix._get_crba_sparse_factor_kernel(
                    n, nnz, warps_per_block=2, geometric=geometric
                )
                replacement = SparseFactor.build().factor(n, nnz, warps_per_block=2, geometric=geometric)
                self.assertEqual(_signature(replacement), _signature(legacy))

                def make(geometric=geometric, update=update):
                    mass = (
                        _array(np.asarray(spd, np.float32), wp.float32)
                        if geometric
                        else _array(np.asarray(records, np.float32), inertia_record)
                    )
                    return [
                        _array(np.array([1, 0], np.int32), wp.int32),
                        _array(np.array(update, np.int32), wp.int32),
                        _array(np.array([0, joints], np.int32), wp.int32),
                        _array(np.array([0, n], np.int32), wp.int32),
                        _array(np.arange(2 * joints, dtype=np.int32), wp.int32),
                        _array(np.random.default_rng(1).normal(size=(2 * n, 6)).astype(np.float32), wp.spatial_vector),
                        mass,
                        _array(np.random.default_rng(2).uniform(4.0, 6.0, (2, n)).astype(np.float32), wp.float32),
                        1,
                        _array(np.array([0, -1, 1, -1, 2, -1] * 2, np.int32), wp.int32),
                        _array(np.array([0.5, -1.0, 2.0], np.float32), wp.float32),
                        factor_indices(plan, DEVICE),
                        _array(np.full((2, nnz), -9.0, np.float32), wp.float32),
                        _array(np.full(2, -9, np.int32), wp.int32),
                    ]

                self.assert_same_launch(legacy, replacement, 64, make)

    def test_contact_rows(self):
        contacts = FactorContacts()
        options = (
            {},
            {"prepare_rows": True},
            {"prepare_rows": True, "use_support_table": True},
            {"prepare_rows": True, "use_support_table": True, "contact_records": True},
            {"prepare_rows": True, "lanes_per_contact": 16, "warps_per_block": 2},
        )
        for kwargs in options:
            with self.subTest(**kwargs):
                legacy = sparse_contact._get_sparse_contact_response_kernel(contacts.n, **kwargs)
                replacement = SparseFactor.build().contact_rows(contacts.n, **kwargs)
                self.assertEqual(_signature(replacement), _signature(legacy))
                lanes = kwargs.get("lanes_per_contact", 8)

                def make(kwargs=kwargs):
                    table = kwargs.get("use_support_table", False)
                    return contacts.inputs(table, kwargs.get("contact_records", False))

                self.assert_same_launch(legacy, replacement, 3 * lanes, make)

    def test_limit_rows(self):
        f = self.f
        rng = np.random.default_rng(23)
        q_index = np.where(rng.random(2 * f.n) < 0.8, np.arange(2 * f.n), -1).astype(np.int32)
        lower = np.where(rng.random(2 * f.n) < 0.8, f.joint_q - rng.uniform(0.0, 0.1, 2 * f.n), -np.inf)
        upper = np.where(rng.random(2 * f.n) < 0.8, f.joint_q + rng.uniform(0.0, 0.1, 2 * f.n), np.inf)
        for enabled, counter in ((0, 0), (1, 0), (0, 9)):
            with self.subTest(support_table=enabled, slots_taken=counter):

                def make(enabled=enabled, counter=counter):
                    return [
                        *f.group(),
                        _array(q_index, wp.int32),
                        _array(lower.astype(np.float32), wp.float32),
                        _array(upper.astype(np.float32), wp.float32),
                        _array(f.joint_q, wp.float32),
                        0.08,
                        0.2,
                        1.0e-5,
                        *f.columns(),
                        _array(np.array([counter, 1], np.int32), wp.int32),
                        *f.outputs(),
                        f.support_table(enabled),
                    ]

                self.assert_same_launch(
                    sparse_contact.build_sparse_joint_limit_rows, SparseFactor.build().limit_rows, 64, make
                )

    def test_velocity_limit_rows(self):
        f = self.f
        rng = np.random.default_rng(29)
        slots = np.where(rng.random(4 * f.n) < 0.6, rng.permutation(4 * f.n) % 14, -1).astype(np.int32)
        signs = rng.choice([-1.0, 1.0], 4 * f.n).astype(np.float32)
        limits = rng.uniform(0.5, 3.0, 2 * f.n).astype(np.float32)
        limits[0] = np.inf
        for enabled in (0, 1):
            with self.subTest(support_table=enabled):

                def make(enabled=enabled):
                    return [
                        *f.group(),
                        _array(limits, wp.float32),
                        _array(slots, wp.int32),
                        _array(signs, wp.float32),
                        1.0e-5,
                        *f.columns(),
                        *f.outputs(),
                        f.support_table(enabled),
                    ]

                self.assert_same_launch(
                    sparse_contact.build_sparse_joint_velocity_limit_rows,
                    SparseFactor.build().velocity_limit_rows,
                    64,
                    make,
                )

    def test_mimic_rows(self):
        f = self.f

        def make():
            return [
                *f.group(),
                _array(np.array([0, 3, 13, 5, -1], np.int32), wp.int32),
                _array(np.array([0, 3, 5], np.int32), wp.int32),
                _array(np.array([0, 1, 4, 2, 3], np.int32), wp.int32),
                _array(np.array([2, 1, 3, 6, 9], np.int32), wp.int32),
                _array(np.array([0, 4, 3, 11, 7], np.int32), wp.int32),
                _array(np.array([2, 1, 3, 6, 9], np.int32), wp.int32),
                _array(np.array([0, 4, 3, 11, 7], np.int32), wp.int32),
                _array(np.array([0.1, -0.2, 0.0, 0.3, 0.0], np.float32), wp.float32),
                _array(np.array([1.5, -0.7, 0.0, 2.0, 1.0], np.float32), wp.float32),
                _array(f.joint_q, wp.float32),
                0.2,
                1.0e-5,
                *f.columns(),
                *f.outputs(),
            ]

        self.assert_same_launch(sparse_contact.build_sparse_mimic_rows, SparseFactor.build().mimic_rows, 64, make)


# ---------------------------------------------------------------------------------------------------------------------
# Paired factor rows
# ---------------------------------------------------------------------------------------------------------------------


class PairedWorlds:
    """Three paired worlds of a ``P``-coordinate articulation and a free body: the first and last without free-body
    rows (factor coordinates), the second with them (physical rows)."""

    def __init__(self, seed=31, P=5, M=7):
        rng = np.random.default_rng(seed)
        self.P, self.S, self.M, self.worlds = P, 6, M, 3
        self.W = P + 6 + 2
        self.primary_Linv = _lower_factors(rng, 3, P)
        self.secondary_Linv = _lower_factors(rng, 3, 6)
        self.primary_J = rng.normal(size=(3, M, P)).astype(np.float32)
        self.secondary_J = rng.normal(size=(3, M, 6)).astype(np.float32)
        self.counts = np.array([M, 4, 9], np.int32)
        self.mf_counts = np.array([0, 3, 0], np.int32)
        self.cfm = rng.uniform(0.0, 1.0e-3, (3, M)).astype(np.float32)

    def inputs(self):
        P, M, W = self.P, self.M, self.W
        return [
            _array(np.zeros((3, P, P), np.float32), wp.float32),
            _array(self.primary_Linv, wp.float32),
            _array(self.primary_J, wp.float32),
            _array(np.array([0, 1, 2], np.int32), wp.int32),
            _array(np.zeros((3, 6, 6), np.float32), wp.float32),
            _array(self.secondary_Linv, wp.float32),
            _array(self.secondary_J, wp.float32),
            _array(np.array([3, 4, 5], np.int32), wp.int32),
            _array(np.array([2, 0, 1], np.int32), wp.int32),
            _array(np.array([0, 1, 2, 0, 1, 2], np.int32), wp.int32),
            _array(np.array([0, 2, 6, P + 2, 0, 1], np.int32), wp.int32),
            _array(self.counts, wp.int32),
            _array(self.mf_counts, wp.int32),
            _array(self.cfm, wp.float32),
            _array(np.full((3, M, W), -9.0, np.float32), wp.float32),
            _array(np.full((3, M, W), -9.0, np.float32), wp.float32),
            _array(np.full((3, M), -9.0, np.float32), wp.float32),
        ]


def _components_data(P, rng, *, qualified=True):
    """Paired component inputs of two worlds: the first qualifies (valid rows on their own components), the second
    has a dense row on the free body's coordinates."""
    M, F, W = 6, 3, P + 6
    J = np.zeros((2, M, W), np.float32)
    Y = np.zeros((2, M, W), np.float32)
    J[:, :, :P] = rng.normal(size=(2, M, P))
    Y[:, :, :P] = rng.normal(size=(2, M, P))
    J[1, 1, P + 2] = 1.0  # couples the components: the world falls back
    row_type = np.array([[0, 2, 2, 3, 0, 3]] * 2, np.int32)
    row_parent = np.array([[-1, 0, 0, -1, -1, -1]] * 2, np.int32)
    meta = np.zeros((2, F * 4), np.int32)
    for row, (kind, parent) in enumerate(((0, 0xFFFF), (2, 0), (2, 0))):
        meta[:, row * 4] = (P << 16) | 0xFFFF
        meta[:, row * 4 + 1] = np.float32(0.5).view(np.int32)
        meta[:, row * 4 + 2] = np.float32(0.1).view(np.int32)
        meta[:, row * 4 + 3] = np.array(kind | (parent << 16), np.uint32).view(np.int32)
    L = _lower_factors(rng, 2, P)
    if not qualified:
        L[0, 0, 0] = np.inf
    return field_group.upload(
        PairedComponentData,
        DEVICE,
        world_count=2,
        counts=np.array([M, M], np.int32),
        mf_counts=np.array([3, 3], np.int32),
        primary_offset=np.zeros(2, np.int32),
        secondary_offset=np.full(2, P, np.int32),
        primary_group=np.array([0, 1], np.int32),
        selector=np.full(2, 7, np.int32),
        J=J,
        Y=Y,
        L=L,
        rhs=rng.normal(size=(2, M)).astype(np.float32),
        diag=rng.uniform(0.1, 1.0, (2, M)).astype(np.float32),
        mu=np.full((2, M), 0.5, np.float32),
        row_type=row_type,
        row_parent=row_parent,
        mf_meta=meta,
        mf_J_a=rng.normal(size=(2, F, 6)).astype(np.float32),
        mf_J_b=rng.normal(size=(2, F, 6)).astype(np.float32),
        mf_MiJt_a=rng.normal(size=(2, F, 6)).astype(np.float32),
        mf_MiJt_b=rng.normal(size=(2, F, 6)).astype(np.float32),
        mf_mu=np.full((2, F), 0.5, np.float32),
    )


class TestFactorRowsBuild(DifferentialCase):
    """Paired factor rows: the rows in factor (or physical) coordinates and the held form of qualified worlds."""

    def test_paired_rows(self):
        pair = PairedWorlds()
        legacy = solver_feather_pgs._get_paired_hinv_jt_kernel(
            pair.P, 6, pair.M, pair.W, "cpu", factor_coordinates=True
        )
        replacement = FactorRows.build().paired(pair.P, 6, pair.M, pair.W, "cpu", factor_coordinates=True)
        self.assertEqual(_signature(replacement), _signature(legacy))
        self.assert_same_launch(legacy, replacement, (3, 128), pair.inputs)
        with self.assertRaises(NotImplementedError):
            FactorRows.build().paired(pair.P, 6, pair.M, pair.W, "cpu")

    def test_components(self):
        for P in (5, 22):
            for qualified in (True, False):
                with self.subTest(primary=P, qualified=qualified):
                    legacy = paired_components.get_prepare_kernel(P)
                    replacement = FactorRows.build().components(P)
                    self.assertEqual(_signature(replacement), _signature(legacy))

                    def make(P=P, qualified=qualified):
                        return [2, _components_data(P, np.random.default_rng(37), qualified=qualified)]

                    self.assert_same_launch(legacy, replacement, (2, 128), make)
                    data = make()[1]
                    wp.launch(replacement, dim=(2, 128), inputs=[2, data], device=DEVICE)
                    self.assertEqual(list(data.selector.numpy()), [-1 if qualified else 3, 3])


class TestDropIns(unittest.TestCase):
    """Every replacement of the table takes the replaced global's parameters, in order."""

    def test_replacements_take_the_replaced_parameters(self):
        for entry in builds.DROP_INS:
            with self.subTest(module=entry.module, name=entry.name):
                module = importlib.import_module(f"newton._src.solvers.feather_pgs.{entry.module}")
                legacy, replacement = getattr(module, entry.name), getattr(builds, entry.replacement)
                if entry.factory:
                    parameters = [
                        [(p.name, p.kind, p.default) for p in inspect.signature(f.__wrapped__).parameters.values()]
                        for f in (legacy, replacement)
                    ]
                    self.assertEqual(parameters[1], parameters[0])
                else:
                    self.assertEqual(
                        [(a.label, wp.types.type_repr(a.type)) for a in replacement.adj.args],
                        [(a.label, wp.types.type_repr(a.type)) for a in legacy.adj.args],
                    )


def _assemble(kernel, arch: int = 120) -> dict:
    """Assemble a kernel's module offline (Warp's PTX, then ``ptxas``) and return its entry's resources."""
    with tempfile.TemporaryDirectory() as directory:
        options = kernel.module.resolve_options(wp.config, block_dim=128)
        kernel.module._compile(None, directory, output_arch=arch, use_ptx=True, options=options)
        ptx = sorted(Path(directory).rglob("*.ptx"))[0]
        source = Path(directory) / "module.ptx"
        source.write_bytes(ptx.read_bytes().rstrip(b"\0"))
        result = subprocess.run(
            ["ptxas", f"-arch=sm_{arch}", "-v", str(source), "-o", str(Path(directory) / "module.cubin")],
            check=True,
            capture_output=True,
            text=True,
        )
    name = f"{kernel.get_mangled_name()}_cuda_kernel_forward"
    entry = result.stderr[result.stderr.index(name) :]
    return {
        "registers": int(re.search(r"Used (\d+) registers", entry).group(1)),
        "spills": int(re.search(r"(\d+) bytes spill stores", entry).group(1)),
    }


@unittest.skipUnless(shutil.which("ptxas"), "ptxas is not on PATH")
class TestNativeBuildsAssemble(unittest.TestCase):
    """The native builds assemble for CUDA with the replaced kernels' registers and no spills."""

    def test_native_builds_assemble_like_the_replaced_kernels(self):
        pairs = {
            "free-body words": (
                lambda: solver_feather_pgs._get_pack_mf_meta_kernel(64, "sm_120"),
                lambda: FreeBody6.build().meta(64, "sm_120"),
            ),
            "limit rows": (
                lambda: sparse_contact.build_sparse_joint_limit_rows,
                lambda: SparseFactor.build().limit_rows,
            ),
            "mimic rows": (lambda: sparse_contact.build_sparse_mimic_rows, lambda: SparseFactor.build().mimic_rows),
            "factor": (
                lambda: sparse_mass_matrix._get_crba_sparse_factor_kernel(12, 30),
                lambda: SparseFactor.build().factor(12, 30),
            ),
            "contact rows": (
                lambda: sparse_contact._get_sparse_contact_response_kernel(
                    12, prepare_rows=True, use_support_table=True, contact_records=True
                ),
                lambda: SparseFactor.build().contact_rows(
                    12, prepare_rows=True, use_support_table=True, contact_records=True
                ),
            ),
            "paired rows": (
                lambda: solver_feather_pgs._get_paired_hinv_jt_kernel(22, 6, 48, 28, "sm_120", factor_coordinates=True),
                lambda: FactorRows.build().paired(22, 6, 48, 28, "sm_120", factor_coordinates=True),
            ),
            "paired components": (
                lambda: paired_components.get_prepare_kernel(22),
                lambda: FactorRows.build().components(22),
            ),
        }
        for name, (legacy, replacement) in pairs.items():
            with self.subTest(build=name):
                expected, actual = _assemble(legacy()), _assemble(replacement())
                self.assertEqual(actual, expected)
                self.assertEqual(actual["spills"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
