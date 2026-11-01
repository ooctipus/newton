# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for FeatherPGS speculative-contact controls."""

import inspect
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import field_group, response_builds_rows
from newton._src.solvers.feather_pgs.contact_input import ContactInput, ContactRoutes
from newton._src.solvers.feather_pgs.dense_rows import DenseRows, MfRows
from newton._src.solvers.feather_pgs.friction_patches import FrictionPatches
from newton._src.solvers.feather_pgs.kernels import (
    BodyFactors,
    allocate_world_contact_slots,
    compute_contact_linear_force_from_impulses,
    compute_world_contact_bias,
    populate_world_J_for_compact_size,
    populate_world_J_for_size,
    prepare_world_contact_rows,
)
from newton.solvers import SolverFeatherPGS
from newton.tests.feather_pgs_law_harness import group_rows, row_lines
from newton.tests.feather_pgs_patch_parity import row_law

PATH_DENSE = 0
PATH_MATRIX_FREE = 1


def _disabled_patches(device):
    patches = FrictionPatches()
    patches.phi = wp.zeros(1, dtype=wp.vec2, device=device)
    return patches


def _launch_contact_allocator(
    *,
    route: int,
    gap: float,
    gate: float,
    responsive: bool = True,
    scoped_gate: float = 0.0,
    pair_gate: float = 0.0,
    enable_friction: bool = False,
    friction_gap: float = float("inf"),
    resolved: bool = False,
):
    """Allocate one contact and return its route metadata and counters."""
    device = "cpu"
    outputs = {
        "world": wp.full((1,), -9, dtype=wp.int32, device=device),
        "slot": wp.full((1,), -9, dtype=wp.int32, device=device),
        "art_a": wp.full((1,), -9, dtype=wp.int32, device=device),
        "art_b": wp.full((1,), -9, dtype=wp.int32, device=device),
        "path": wp.full((1,), -9, dtype=wp.int32, device=device),
        "slots_needed": wp.full((1,), -9, dtype=wp.int32, device=device),
    }
    counters = {
        "dense_count": wp.zeros((1,), dtype=wp.int32, device=device),
        "mf_count": wp.zeros((1,), dtype=wp.int32, device=device),
        "dense_dropped": wp.zeros((1,), dtype=wp.int32, device=device),
        "mf_dropped": wp.zeros((1,), dtype=wp.int32, device=device),
        "dense_first_rejected": wp.full((1,), 2**31 - 1, dtype=wp.int32, device=device),
        "mf_first_rejected": wp.full((1,), 2**31 - 1, dtype=wp.int32, device=device),
    }
    is_free = route == PATH_MATRIX_FREE
    wp.launch(
        allocate_world_contact_slots,
        dim=1,
        inputs=[
            field_group.upload(
                ContactInput,
                device,
                count=wp.array([1], dtype=wp.int32, device=device),
                shape0=wp.array([0], dtype=wp.int32, device=device),
                shape1=wp.array([-1], dtype=wp.int32, device=device),
                point0=wp.array([wp.vec3(gap, 0.0, 0.0)], dtype=wp.vec3, device=device),
                point1=wp.array([wp.vec3(0.0)], dtype=wp.vec3, device=device),
                normal=wp.array([wp.vec3(-1.0, 0.0, 0.0)], dtype=wp.vec3, device=device),
                margin0=wp.zeros((1,), dtype=wp.float32, device=device),
                margin1=wp.zeros((1,), dtype=wp.float32, device=device),
                body_q=wp.array([wp.transform_identity()], dtype=wp.transform, device=device),
                shape_body=wp.array([0], dtype=wp.int32, device=device),
                response_dofs=wp.array([1], dtype=wp.int32, device=device),
                world=outputs["world"],
                slot=outputs["slot"],
                art0=outputs["art_a"],
                art1=outputs["art_b"],
                path=outputs["path"],
                slots_needed=outputs["slots_needed"],
                first=row_lines(device),
            ),
            1,
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.zeros((1,), dtype=wp.int32, device=device),
            wp.array([int(responsive)], dtype=wp.int32, device=device),
            wp.array([int(is_free)], dtype=wp.int32, device=device),
            1,
            gate,
            scoped_gate,
            pair_gate,
            8,
            8,
            int(enable_friction),
            friction_gap,
            0,
            FrictionPatches(),
            wp.array([int(resolved)], dtype=wp.int32, device=device),
        ],
        outputs=[
            counters["dense_count"],
            counters["mf_count"],
            counters["dense_dropped"],
            counters["mf_dropped"],
            counters["dense_first_rejected"],
            counters["mf_first_rejected"],
        ],
        device=device,
    )
    wp.synchronize_device(device)
    return {name: int(array.numpy()[0]) for name, array in (outputs | counters).items()}


def _launch_articulation_pair_contact_allocator(
    *,
    gap: float,
    scoped_gate: float = 0.0,
    pair_gate: float = 0.0,
    cross_articulation: bool = False,
    enable_friction: bool = False,
    friction_gap: float = float("inf"),
):
    """Allocate one contact between two non-free articulated links."""
    device = "cpu"
    body_to_articulation = [0, 1] if cross_articulation else [0, 0]
    articulation_count = 2 if cross_articulation else 1
    contact_slot = wp.full((1,), -9, dtype=wp.int32, device=device)
    contact_path = wp.full((1,), -9, dtype=wp.int32, device=device)
    contact_slots_needed = wp.full((1,), -9, dtype=wp.int32, device=device)
    dense_count = wp.zeros((1,), dtype=wp.int32, device=device)
    dense_dropped = wp.zeros((1,), dtype=wp.int32, device=device)
    mf_dropped = wp.zeros((1,), dtype=wp.int32, device=device)
    wp.launch(
        allocate_world_contact_slots,
        dim=1,
        inputs=[
            field_group.upload(
                ContactInput,
                device,
                count=wp.array([1], dtype=wp.int32, device=device),
                shape0=wp.array([0], dtype=wp.int32, device=device),
                shape1=wp.array([1], dtype=wp.int32, device=device),
                point0=wp.array([wp.vec3(gap, 0.0, 0.0)], dtype=wp.vec3, device=device),
                point1=wp.array([wp.vec3(0.0)], dtype=wp.vec3, device=device),
                normal=wp.array([wp.vec3(-1.0, 0.0, 0.0)], dtype=wp.vec3, device=device),
                margin0=wp.zeros((1,), dtype=wp.float32, device=device),
                margin1=wp.zeros((1,), dtype=wp.float32, device=device),
                body_q=wp.array([wp.transform_identity(), wp.transform_identity()], dtype=wp.transform, device=device),
                shape_body=wp.array([0, 1], dtype=wp.int32, device=device),
                response_dofs=wp.array([1] * articulation_count, dtype=wp.int32, device=device),
                world=wp.full((1,), -9, dtype=wp.int32, device=device),
                slot=contact_slot,
                art0=wp.full((1,), -9, dtype=wp.int32, device=device),
                art1=wp.full((1,), -9, dtype=wp.int32, device=device),
                path=contact_path,
                slots_needed=contact_slots_needed,
                first=row_lines(device),
            ),
            1,
            wp.array(body_to_articulation, dtype=wp.int32, device=device),
            wp.array([0] * articulation_count, dtype=wp.int32, device=device),
            wp.zeros((2,), dtype=wp.int32, device=device),
            wp.ones((2,), dtype=wp.int32, device=device),
            wp.zeros((2,), dtype=wp.int32, device=device),
            0,
            0.0,
            scoped_gate,
            pair_gate,
            8,
            8,
            int(enable_friction),
            friction_gap,
            0,
            FrictionPatches(),
            wp.empty(0, dtype=wp.int32, device=device),
        ],
        outputs=[
            dense_count,
            wp.zeros((1,), dtype=wp.int32, device=device),
            dense_dropped,
            mf_dropped,
            wp.full((1,), 2**31 - 1, dtype=wp.int32, device=device),
            wp.full((1,), 2**31 - 1, dtype=wp.int32, device=device),
        ],
        device=device,
    )
    wp.synchronize_device(device)
    slots_needed = max(0, int(contact_slots_needed.numpy()[0]))
    return int(contact_slot.numpy()[0]), int(contact_path.numpy()[0]), slots_needed


def _dense_speculative_rhs(scale: float) -> float:
    """Return the dense positive-gap RHS for a requested speculative scale."""
    rhs = wp.zeros((1, 1), dtype=wp.float32, device="cpu")
    rows = field_group.upload(
        DenseRows,
        "cpu",
        phi=[[1.0]],
        row_beta=[[0.2]],
        row_type=[[0]],
        target=[[0.0]],
        restitution=[[0.0]],
        rhs=rhs,
        row_w=[[0.0]],
        first=row_lines("cpu"),
    )
    wp.launch(
        compute_world_contact_bias,
        dim=(1, 1),
        inputs=[
            rows,
            wp.zeros_like(rhs),
            wp.array([1], dtype=wp.int32, device="cpu"),
            1,
            1,
            0.5,
            scale,
            1.0,
            0.0,
            0,
            0,
        ],
        device="cpu",
    )
    return float(rhs.numpy()[0, 0])


def _mf_speculative_rhs(scale: float) -> float:
    """Return the MF setup RHS for a requested speculative scale."""
    rhs = wp.zeros((1, 1), dtype=wp.float32, device="cpu")
    wp.launch(
        response_builds_rows.free_body_rows_kernel,
        dim=1,
        inputs=[
            field_group.upload(
                MfRows,
                "cpu",
                body_a=[[-1]],
                body_b=[[-1]],
                Z_a=np.zeros((1, 1, 6), dtype=np.float32),
                Z_b=np.zeros((1, 1, 6), dtype=np.float32),
                phi=[[1.0]],
                row_type=[[0]],
                target=[[0.0]],
                restitution=[[0.0]],
                incident=[[0.0]],
                rhs=rhs,
                first=row_lines("cpu"),
            ),
            wp.array([1], dtype=wp.int32, device="cpu"),
            BodyFactors(),
            0,
            wp.zeros((1,), dtype=wp.float32, device="cpu"),
            wp.array([float("inf")], dtype=wp.float32, device="cpu"),
            1.0,
            0.2,
            1.0,
            0.5,
            scale,
            0.5,
            1,
            0,
            0,
        ],
        outputs=[
            wp.zeros((1, 1), dtype=wp.float32, device="cpu"),
            wp.zeros((1, 1), dtype=wp.float32, device="cpu"),
        ],
        device="cpu",
    )
    return float(rhs.numpy()[0, 0])


def _contact_row_outputs(device: str) -> dict[str, wp.array]:
    """Allocate dense-contact Jacobian and metadata outputs."""
    return {
        "J": wp.zeros((1, 8, 3), dtype=wp.float32, device=device),
        "row_type": wp.full((1, 8), -9, dtype=wp.int32, device=device),
        "row_parent": wp.full((1, 8), -9, dtype=wp.int32, device=device),
        "row_mu": wp.full((1, 8), -9.0, dtype=wp.float32, device=device),
        "row_beta": wp.full((1, 8), -9.0, dtype=wp.float32, device=device),
        "row_cfm": wp.full((1, 8), -9.0, dtype=wp.float32, device=device),
        "phi": wp.full((1, 8), -9.0, dtype=wp.float32, device=device),
        "target_velocity": wp.full((1, 8), -9.0, dtype=wp.float32, device=device),
        "row_restitution": wp.full((1, 8), -9.0, dtype=wp.float32, device=device),
        "sides": wp.zeros((1, 8), dtype=wp.vec2i, device=device),
    }


def _launch_dense_contact_builders(device: str = "cpu") -> tuple[dict[str, wp.array], dict[str, wp.array]]:
    """Build one same-articulation contact through serial and compact paths."""
    contact_count = wp.array([1], dtype=wp.int32, device=device)
    point0 = wp.array([wp.vec3(0.2, 0.1, -0.05)], dtype=wp.vec3, device=device)
    point1 = wp.array([wp.vec3(-0.1, 0.05, 0.2)], dtype=wp.vec3, device=device)
    normal = wp.array([wp.vec3(0.0, 0.0, -1.0)], dtype=wp.vec3, device=device)
    shape0 = wp.array([0], dtype=wp.int32, device=device)
    shape1 = wp.array([1], dtype=wp.int32, device=device)
    thickness0 = wp.array([0.01], dtype=wp.float32, device=device)
    thickness1 = wp.array([0.02], dtype=wp.float32, device=device)
    contact_world = wp.array([0], dtype=wp.int32, device=device)
    contact_slot = wp.array([0], dtype=wp.int32, device=device)
    contact_art_a = wp.array([0], dtype=wp.int32, device=device)
    contact_art_b = wp.array([0], dtype=wp.int32, device=device)
    contact_path = wp.array([PATH_DENSE], dtype=wp.int32, device=device)
    contact_slots_needed = wp.array([3], dtype=wp.int32, device=device)
    response_dof_count = wp.array([3], dtype=wp.int32, device=device)
    art_group_idx = wp.array([0], dtype=wp.int32, device=device)
    art_dof_start = wp.array([0], dtype=wp.int32, device=device)
    articulation_origin = wp.array([wp.vec3(0.1, -0.2, 0.3)], dtype=wp.vec3, device=device)
    body_to_joint = wp.array([0, 1, 2], dtype=wp.int32, device=device)
    joint_ancestor = wp.array([-1, 0, 1], dtype=wp.int32, device=device)
    joint_qd_start = wp.array([0, 1, 2, 3], dtype=wp.int32, device=device)
    joint_s = wp.array(
        [
            wp.spatial_vector(1.0, 0.0, 0.0, 0.0, 0.0, 0.5),
            wp.spatial_vector(0.0, 1.0, 0.0, 0.25, 0.0, 0.0),
            wp.spatial_vector(0.0, 0.0, 1.0, 0.0, -0.5, 0.0),
        ],
        dtype=wp.spatial_vector,
        device=device,
    )
    shape_body = wp.array([2, 1], dtype=wp.int32, device=device)
    body_q = wp.array(
        [
            wp.transform(wp.vec3(0.0), wp.quat_identity()),
            wp.transform(wp.vec3(-0.2, 0.3, 0.1), wp.quat_identity()),
            wp.transform(wp.vec3(0.4, -0.1, 0.2), wp.quat_identity()),
        ],
        dtype=wp.transform,
        device=device,
    )
    body_v = wp.zeros((3,), dtype=wp.spatial_vector, device=device)
    prescribed = wp.zeros((1,), dtype=wp.int32, device=device)
    material_mu = wp.array([0.6, 0.8], dtype=wp.float32, device=device)
    material_restitution = wp.array([0.2, 0.4], dtype=wp.float32, device=device)

    def contact_input(rows):
        return field_group.upload(
            ContactInput,
            device,
            count=contact_count,
            point0=point0,
            point1=point1,
            normal=normal,
            shape0=shape0,
            shape1=shape1,
            margin0=thickness0,
            margin1=thickness1,
            world=contact_world,
            slot=contact_slot,
            art0=contact_art_a,
            art1=contact_art_b,
            path=contact_path,
            slots_needed=contact_slots_needed,
            response_dofs=response_dof_count,
            dof_start=art_dof_start,
            origin=articulation_origin,
            body_mask=wp.array([1, 3, 7], dtype=wp.uint32, device=device),
            motion=joint_s,
            shape_body=shape_body,
            body_q=body_q,
            body_v=body_v,
            prescribed=prescribed,
            material_mu=material_mu,
            material_restitution=material_restitution,
            max_depenetration_velocity=wp.full(body_q.shape[0], np.inf, dtype=float, device=device),
            beta=0.05,
            cfm=1.0e-6,
            row_type=rows["row_type"],
            row_parent=rows["row_parent"],
            row_mu=rows["row_mu"],
            row_beta=rows["row_beta"],
            row_cfm=rows["row_cfm"],
            phi=rows["phi"],
            target=rows["target_velocity"],
            restitution=rows["row_restitution"],
            sides=rows["sides"],
            first=row_lines(device),
        )

    serial = _contact_row_outputs(device)
    wp.launch(
        populate_world_J_for_size,
        dim=1,
        inputs=[
            contact_input(serial),
            1,
            3,
            art_group_idx,
            body_to_joint,
            joint_ancestor,
            joint_qd_start,
            row_law(_disabled_patches(device)),  # point friction
            group_rows(device),
        ],
        outputs=[serial["J"]],
        device=device,
    )

    compact = _contact_row_outputs(device)
    compact_input = contact_input(compact)
    wp.launch(
        prepare_world_contact_rows,
        dim=1,
        inputs=[compact_input, 1, row_law(_disabled_patches(device))],  # point friction
        device=device,
    )
    wp.launch(
        populate_world_J_for_compact_size,
        dim=(1, 32),
        inputs=[compact_input, 1, 3, art_group_idx, row_law(_disabled_patches(device)), group_rows(device)],
        outputs=[compact["J"]],
        device=device,
    )
    wp.synchronize_device(device)
    return serial, compact


class TestFeatherPGSContactControls(unittest.TestCase):
    def test_skipped_contact_publishes_zero_without_reading_stale_rows(self):
        """An invalidated mapping must clear a previous force even with poisoned rows."""
        device = "cpu"
        force = wp.full(1, wp.vec3(91.0), dtype=wp.vec3, device=device)
        integers = wp.array([-1], dtype=wp.int32, device=device)
        poisoned = wp.full((1, 1), float("nan"), dtype=wp.float32, device=device)
        row_metadata = wp.full((1, 1), -91, dtype=wp.int32, device=device)
        wp.launch(
            compute_contact_linear_force_from_impulses,
            dim=1,
            inputs=[
                row_lines(device),
                row_lines(device),
                field_group.upload(
                    ContactRoutes,
                    device,
                    count=[1],
                    normal=wp.full(1, wp.vec3(float("nan")), dtype=wp.vec3, device=device),
                    world=integers,
                    slot=integers,
                    path=integers,
                ),
                poisoned,
                poisoned,
                integers,
                integers,
                row_metadata,
                row_metadata,
                row_metadata,
                row_metadata,
                1,
                240.0,
            ],
            outputs=[force],
            device=device,
        )
        np.testing.assert_array_equal(force.numpy(), np.zeros((1, 3), dtype=np.float32))

    def test_resolved_world_skips_all_contact_routes_and_refreshes_mapping(self):
        """Zero-world allocation must overwrite stale mappings without producing rows."""
        for route in (PATH_DENSE, PATH_MATRIX_FREE):
            with self.subTest(route=route):
                fallback = _launch_contact_allocator(route=route, gap=-0.01, gate=0.0)
                self.assertGreaterEqual(fallback["slot"], 0)
                self.assertEqual(fallback["path"], route)
                skipped = _launch_contact_allocator(route=route, gap=float("nan"), gate=0.0, resolved=True)
                self.assertEqual(skipped["world"], 0)
                self.assertEqual(skipped["art_a"], 0)
                self.assertEqual(skipped["art_b"], -1)
                self.assertEqual(skipped["slot"], -1)
                self.assertEqual(skipped["path"], -1)
                self.assertEqual(skipped["slots_needed"], 0)
                for name in ("dense_count", "mf_count", "dense_dropped", "mf_dropped"):
                    self.assertEqual(skipped[name], 0, name)

    def test_compact_contact_builder_matches_tree_walk(self):
        """Match dense Jacobians and metadata for a same-articulation contact."""
        serial, compact = _launch_dense_contact_builders()

        for name in serial:
            with self.subTest(output=name):
                np.testing.assert_allclose(compact[name].numpy(), serial[name].numpy(), rtol=0.0, atol=1.0e-6)

    def test_solver_exposes_documented_defaults(self):
        """Expose legacy-preserving defaults through the public constructor."""
        solver = SolverFeatherPGS(newton.ModelBuilder().finalize(device="cpu"))

        self.assertEqual(solver.options.contact_speculative_scale, 1.0)
        self.assertEqual(solver.options.contact_gap_gate, 0.0)
        self.assertEqual(solver.options.same_articulation_contact_gap_gate, 0.0)
        self.assertEqual(solver.options.articulation_pair_contact_gap_gate, 0.0)
        self.assertTrue(solver.options.warn_constraint_overflow)
        self.assertTrue(solver.placement.track_row_capacity)
        self.assertFalse(solver.placement.row_watermark)
        quiet_solver = SolverFeatherPGS(
            newton.ModelBuilder().finalize(device="cpu"),
            warn_constraint_overflow=False,
        )
        self.assertFalse(quiet_solver.options.warn_constraint_overflow)
        self.assertFalse(quiet_solver.placement.track_row_capacity)
        self.assertIsNone(quiet_solver.placement.row_overflow_warning_emitted)
        parameters = tuple(inspect.signature(SolverFeatherPGS).parameters)
        self.assertIn("same_articulation_contact_gap_gate", parameters)

    def test_new_contact_controls_preserve_legacy_positional_layout(self):
        """Append new controls without shifting established positional arguments."""
        parameters = tuple(inspect.signature(SolverFeatherPGS).parameters)
        for sequence in (
            ("row_watermark", "restitution_velocity_threshold", "contact_speculative_scale", "contact_gap_gate"),
        ):
            start = parameters.index(sequence[0])
            self.assertEqual(parameters[start : start + len(sequence)], sequence)

        legacy_tail = parameters.index("contact_gap_gate")
        for name in (
            "enable_restitution",
            "same_articulation_contact_gap_gate",
            "articulation_pair_contact_gap_gate",
            "warn_constraint_overflow",
        ):
            self.assertGreater(parameters.index(name), legacy_tail)

    def test_solver_validates_and_stores_contact_controls(self):
        """Accept finite non-negative controls and reject malformed values."""
        model = newton.ModelBuilder().finalize(device="cpu")
        solver = SolverFeatherPGS(
            model,
            contact_speculative_scale=0.0,
            contact_gap_gate=0.001,
            same_articulation_contact_gap_gate=0.002,
            articulation_pair_contact_gap_gate=0.003,
        )
        self.assertEqual(solver.options.contact_speculative_scale, 0.0)
        self.assertEqual(solver.options.contact_gap_gate, 0.001)
        self.assertEqual(solver.options.same_articulation_contact_gap_gate, 0.002)
        self.assertEqual(solver.options.articulation_pair_contact_gap_gate, 0.003)

        for name in (
            "contact_speculative_scale",
            "contact_gap_gate",
            "same_articulation_contact_gap_gate",
            "articulation_pair_contact_gap_gate",
        ):
            for value in (-0.1, float("nan"), float("inf"), "invalid"):
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        SolverFeatherPGS(model, **{name: value})

    def test_scoped_gap_gate_only_drops_distant_same_articulation_contact(self):
        """The scoped gate retains near self-contact while bounding its speculative tail."""
        self.assertEqual(
            _launch_articulation_pair_contact_allocator(gap=0.002, scoped_gate=0.003),
            (0, PATH_DENSE, 1),
        )
        self.assertEqual(
            _launch_articulation_pair_contact_allocator(gap=0.004, scoped_gate=0.003),
            (-1, -1, 0),
        )
        self.assertEqual(
            _launch_articulation_pair_contact_allocator(gap=0.004, scoped_gate=0.0),
            (0, PATH_DENSE, 1),
        )

    def test_scoped_gap_gate_preserves_other_contact_routes(self):
        """Do not shorten predictive contacts for ground, free-body, or cross-articulation rows."""
        for route, counter in ((PATH_DENSE, "dense_count"), (PATH_MATRIX_FREE, "mf_count")):
            with self.subTest(route=route):
                result = _launch_contact_allocator(
                    route=route,
                    gap=0.04,
                    gate=0.0,
                    scoped_gate=0.003,
                )
                self.assertEqual(result["path"], route)
                self.assertEqual(result[counter], 1)

    def test_articulation_pair_gap_gate_drops_distant_pair_contact(self):
        """The pair gate includes same/cross-articulation contact without touching free bodies."""
        for cross_articulation in (False, True):
            with self.subTest(cross_articulation=cross_articulation):
                self.assertEqual(
                    _launch_articulation_pair_contact_allocator(
                        gap=0.004,
                        scoped_gate=0.0,
                        pair_gate=0.003,
                        cross_articulation=cross_articulation,
                    ),
                    (-1, -1, 0),
                )
        result = _launch_contact_allocator(
            route=PATH_MATRIX_FREE,
            gap=0.04,
            gate=0.0,
            pair_gate=0.003,
        )
        self.assertEqual(result["path"], PATH_MATRIX_FREE)
        self.assertEqual(result["mf_count"], 1)

    def test_articulation_pair_friction_filter_reduces_pair_rows(self):
        """Apply the configured friction gap to same- and cross-articulation contacts."""
        for cross_articulation in (False, True):
            with self.subTest(cross_articulation=cross_articulation):
                slot, path, slots_needed = _launch_articulation_pair_contact_allocator(
                    gap=0.004,
                    cross_articulation=cross_articulation,
                    enable_friction=True,
                    friction_gap=0.002,
                )
                self.assertEqual((slot, path, slots_needed), (0, PATH_DENSE, 1))

    def test_negative_friction_gap_delays_tangents(self):
        """Keep a contact's normal while its tangents wait for a gap past the threshold."""
        shallow_pair = _launch_articulation_pair_contact_allocator(
            gap=-0.0005, enable_friction=True, friction_gap=-0.001
        )
        deep_pair = _launch_articulation_pair_contact_allocator(gap=-0.002, enable_friction=True, friction_gap=-0.001)
        self.assertEqual(shallow_pair, (0, PATH_DENSE, 1))
        self.assertEqual(deep_pair, (0, PATH_DENSE, 3))

    def test_speculative_scale_controls_every_position_rhs_family(self):
        """Scale positive-gap position bias on dense and MF rows."""
        rhs_families = {
            "dense": _dense_speculative_rhs,
            "matrix_free": _mf_speculative_rhs,
        }
        for name, compute_rhs in rhs_families.items():
            with self.subTest(family=name):
                self.assertEqual(compute_rhs(0.0), 0.0)
                self.assertAlmostEqual(compute_rhs(1.0), 2.0, places=6)

    def test_gap_gate_prevents_all_route_allocations(self):
        """Drop positive gaps above the gate before any route reserves a slot."""
        for route in (PATH_DENSE, PATH_MATRIX_FREE):
            with self.subTest(route=route):
                allocated = _launch_contact_allocator(route=route, gap=0.002, gate=0.0)
                self.assertEqual(allocated["path"], route)
                self.assertEqual(allocated["slot"], 0)
                self.assertEqual(allocated["world"], 0)
                self.assertEqual(allocated["art_a"], 0)
                self.assertEqual(allocated["art_b"], -1)
                self.assertEqual(allocated["slots_needed"], 1)
                self.assertEqual(
                    (allocated["dense_count"], allocated["mf_count"]),
                    (int(route == PATH_DENSE), int(route == PATH_MATRIX_FREE)),
                )

                dropped = _launch_contact_allocator(route=route, gap=0.002, gate=0.001)
                self.assertEqual(dropped["slot"], -1)
                self.assertEqual(dropped["path"], -1)
                self.assertEqual(dropped["dense_count"], 0)
                self.assertEqual(dropped["mf_count"], 0)

    def test_gap_gate_keeps_contacts_at_threshold(self):
        """Keep a contact whose gap equals the positive gate exactly."""
        for route in (PATH_DENSE, PATH_MATRIX_FREE):
            with self.subTest(route=route):
                result = _launch_contact_allocator(route=route, gap=0.001, gate=0.001)
                self.assertEqual(result["path"], route)
                self.assertEqual(result["slots_needed"], 1)

    def test_allocator_skips_contacts_without_response_dofs(self):
        """Do not allocate a row when neither contact body can change velocity."""
        result = _launch_contact_allocator(route=PATH_DENSE, gap=-0.001, gate=0.0, responsive=False)

        self.assertEqual(result["slot"], -1)
        self.assertEqual(result["path"], -1)
        self.assertEqual(result["dense_count"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
