# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the predictor-resolved world classifier."""

import ast
import dataclasses
import inspect
import os
import textwrap
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import resolved_worlds as rw
from newton._src.solvers.feather_pgs import row_slots
from newton._src.solvers.feather_pgs.friction_patches import CONTACT_PATH_RESOLVED, FrictionPatches, patch_routes
from newton._src.solvers.feather_pgs.solver_feather_pgs import SolverFeatherPGS
from newton.tests.feather_pgs_law_harness import contact_input, plain_laws, row_lines
from newton.tests.feather_pgs_patch_parity import assert_history_close
from newton.tests.test_feather_pgs_friction_patches import _patch_fixture
from newton.tests.test_feather_pgs_mixed_tree_free_body import DT, _add_quadruped, _build


def make_case(*, free=False):
    """Build one responsive body and one prescribed endpoint on CPU."""
    device = "cpu"
    width = 6 if free else 1

    def a(value, dtype=wp.float32):
        return wp.array(value, dtype=dtype, device=device)

    data = rw.ResolvedWorldInput()
    data.world_dof_indices = a([list(range(width))], wp.int32)
    data.world_dof_count = a([width], wp.int32)
    data.limit_q_index = a([-1] * width if free else [0], wp.int32)
    data.lower, data.upper = a([-np.inf] * width), a([np.inf] * width)
    data.prescribed = a([0] * width, wp.int32)
    data.q, data.v_hat = a([0.0]), a([0.0] * width)
    data.body_to_articulation = a([0, 1], wp.int32)
    data.art_to_world = a([0, 0], wp.int32)
    data.free_bodies = wp.array([0] if free else [], dtype=wp.int32, device=device)
    data.predicted_twist = a(np.zeros((2, 6)), wp.spatial_vector)
    data.use_published = 0
    data.articulation_dof_start = a([0, width], wp.int32)
    data.articulation_response_dof_count = a([width, 0], wp.int32)
    data.prescribed_articulation = a([0, 1], wp.int32)
    data.is_free_rigid = a([int(free), 0], wp.int32)
    data.body_flags = a([1, 2], wp.int32)
    data.body_has_response_dofs = a([1, 0], wp.int32)
    data.body_response_dof_mask = a([(1 << width) - 1, 0], wp.uint32)
    data.body_q = a([[0, 0, 0, 0, 0, 0, 1]] * 2, wp.transform)
    data.body_v_s = a([[999, 999, 999, 0, 0, 0], [0, 0, 0, 0, 0, 0]], wp.spatial_vector)
    data.joint_S_s = a(np.eye(6, dtype=np.float32)[:width], wp.spatial_vector)
    data.articulation_origin = a([[0, 0, 0], [0, 0, 0]], wp.vec3)
    data.max_linear_velocity = a([np.inf, np.inf])
    data.max_angular_velocity = a([np.inf, np.inf])
    data.max_depenetration_velocity = a([np.inf, np.inf])
    data.dt, data.beta, data.speculative_scale = 0.1, 0.2, 1.0
    data.activation_gap, data.restitution_threshold = 0.0, 0.5
    data.absolute_margin, data.relative_margin = 0.0, 0.0
    raw = rw.RawContacts()
    raw.count = a([1], wp.int32)
    raw.shape0, raw.shape1 = a([0], wp.int32), a([1], wp.int32)
    raw.point0, raw.point1 = a([[0.1, 0, 0]], wp.vec3), a([[0, 0, 0]], wp.vec3)
    raw.normal = a([[-1, 0, 0]], wp.vec3)
    raw.margin0, raw.margin1 = a([0]), a([0])
    raw.shape_body = a([0, 1], wp.int32)
    raw.shape_mu, raw.shape_restitution = a([0.5, 0.5]), a([0, 0])
    scratch = rw.ResolvedWorldScratch()
    for name in ("rejected", "resolved", "checked_limits", "checked_normals", "global_invalid"):
        setattr(scratch, name, a([0], wp.int32))
    scratch.minimum_residual = a([0])
    scratch.body_twist = a(np.zeros((2, 6)), wp.spatial_vector)
    scratch.body_valid = a([0, 0], wp.int32)
    return data, raw, scratch


def evaluate(data, raw, scratch, *, enabled=1, published=False, row_bound=(100, 100)):
    """Execute the decision kernels on CPU, without initializing CUDA.

    ``published`` reads the predictor twists in ``data.predicted_twist``, as
    on the held-tree owner, instead of running the classifier's body pass.
    ``row_bound`` (dense, matrix-free) bounds the worlds' rows; the default leaves every case room.
    """
    data.use_published = int(published)
    free_normals = wp.zeros(scratch.resolved.shape[0], dtype=wp.int32, device="cpu")
    rw.launch(
        data,
        raw,
        scratch,
        1,
        "cpu",
        enabled=enabled,
        published=published,
        row_bound=row_bound,
        free_normals=free_normals,
        free_line=0,
    )
    return int(scratch.resolved.numpy()[0])


# Synthetic contacts of two worlds. Bodies: 0 an arm link (articulation 0, world 0), 1 a link
# without response DOFs (articulation 1, world 0), 2 an arm link of world 1 (articulation 2) and
# 3 a free box (articulation 3, world 0); shapes 0-3 sit on them and shape 4 is static. Contacts:
# (0) the link on the ground, (1) the unresponsive link on the ground, (2) the link 0.5 m above the
# ground, past the 0.1 m gap gate, (3) world 1's link on the ground, (4) a pair across worlds and
# (5) the free box on the ground.
_ALLOCATION_SHAPES = ((0, 4), (1, 4), (0, 4), (2, 4), (0, 2), (3, 4))
_ALLOCATION_LIFT = (0.0, 0.0, 0.5, 0.0, 0.0, 0.0)


def _allocate(kernel, resolved):
    """Run one allocator serially over the synthetic contacts; return its per-contact outputs and counters."""
    count = len(_ALLOCATION_SHAPES)
    patches = FrictionPatches()
    # Every anchor carries load, and a one-contact region no spin row, so patch friction reserves the rows point
    # friction does.
    patches.weight = wp.ones(count, dtype=float, device="cpu")
    patches.spin_arm = wp.zeros(count, dtype=float, device="cpu")

    def integers(values):
        return wp.array(values, dtype=wp.int32, device="cpu")

    arrays = {
        "contact_count": integers([count]),
        "total_num_threads": 1,
        "contact_shape0": integers([pair[0] for pair in _ALLOCATION_SHAPES]),
        "contact_shape1": integers([pair[1] for pair in _ALLOCATION_SHAPES]),
        "contact_point0": wp.array([[0.0, 0.0, lift] for lift in _ALLOCATION_LIFT], dtype=wp.vec3, device="cpu"),
        "contact_point1": wp.zeros(count, dtype=wp.vec3, device="cpu"),
        "contact_normal": wp.array([[0.0, 0.0, -1.0]] * count, dtype=wp.vec3, device="cpu"),
        "contact_thickness0": wp.zeros(count, dtype=float, device="cpu"),
        "contact_thickness1": wp.zeros(count, dtype=float, device="cpu"),
        "body_q": wp.array([wp.transform_identity()] * 4, dtype=wp.transform, device="cpu"),
        "shape_transform": wp.array([wp.transform_identity()] * 5, dtype=wp.transform, device="cpu"),
        "shape_body": integers([0, 1, 2, 3, -1]),
        "body_to_articulation": integers([0, 1, 2, 3]),
        "art_to_world": integers([0, 0, 1, 0]),
        "articulation_response_dof_count": integers([3, 0, 3, 6]),
        "body_flags": integers([0, 0, 0, 0]),
        "body_has_response_dofs": integers([1, 0, 1, 1]),
        "is_free_rigid": integers([0, 0, 0, 1]),
        "has_free_rigid": 1,
        "contact_gap_gate": 0.1,
        "same_articulation_contact_gap_gate": 0.0,
        "articulation_pair_contact_gap_gate": 0.0,
        "max_constraints": 32,
        "mf_max_constraints": 32,
        "enable_friction": 1,
        "contact_friction_gap_threshold": float("inf"),
        "row_capacity_telemetry": 0,
        "friction_patches": patches,
        "resolved_worlds": integers(resolved),
        "contact_world": integers([-9] * count),
        "contact_slot": integers([-9] * count),
        "contact_art_a": integers([-9] * count),
        "contact_art_b": integers([-9] * count),
        "world_slot_counter": integers([0, 0]),
        "contact_path": integers([-9] * count),
        "mf_slot_counter": integers([0, 0]),
        "contact_slots_needed": integers([-9] * count),
        "dense_dropped_contact_rows": integers([0, 0]),
        "mf_dropped_contact_rows": integers([0, 0]),
        "dense_first_rejected_slot": integers([2**31 - 1] * 2),
        "mf_first_rejected_slot": integers([2**31 - 1] * 2),
        "units": wp.zeros((0, 0), dtype=wp.int32, device="cpu"),
    }
    data, _arrays = contact_input(arrays, "cpu")
    values = {**arrays, "data": data}
    wp.launch(kernel, dim=1, inputs=[values[name] for name in inspect.signature(kernel.func).parameters], device="cpu")
    return {name: value.numpy() for name, value in arrays.items() if isinstance(value, wp.array)}


def _definition(function):
    """Parse the live source of a Warp kernel's or function's Python definition."""
    return ast.parse(textwrap.dedent(inspect.getsource(function.func))).body[0]


class TestResolvedWorlds(unittest.TestCase):
    def test_current_predictor_not_incoming_body_twist(self):
        """Reject a closing predictor despite a separating stale body twist."""
        data, raw, scratch = make_case()
        self.assertEqual(evaluate(data, raw, scratch), 1)
        data.v_hat.assign([-2.0])
        self.assertEqual(evaluate(data, raw, scratch), 0)
        np.testing.assert_array_equal(scratch.body_twist.numpy()[0], [-2, 0, 0, 0, 0, 0])
        self.assertEqual(int(scratch.checked_normals.numpy()[0]), 1)

    def test_position_limits_both_sides_and_speculative_bias(self):
        """Check activated lower and upper rows without constructing J."""
        data, raw, scratch = make_case()
        data.lower.assign([0.0])
        data.q.assign([-0.01])
        self.assertEqual(evaluate(data, raw, scratch), 0)
        data.v_hat.assign([0.03])
        self.assertEqual(evaluate(data, raw, scratch), 1)
        data.q.assign([0.01])
        data.activation_gap = 0.02
        data.v_hat.assign([-0.2])
        self.assertEqual(evaluate(data, raw, scratch), 0)
        data.lower.assign([-np.inf])
        data.upper.assign([0.0])
        data.q.assign([0.01])
        data.v_hat.assign([-0.03])
        self.assertEqual(evaluate(data, raw, scratch), 1)
        self.assertEqual(int(scratch.checked_limits.numpy()[0]), 1)
        # The default activation gap is infinite: every limit is a candidate, none is invalid.
        data.activation_gap = float("inf")
        data.lower.assign([-1.0])
        self.assertEqual(evaluate(data, raw, scratch), 1)
        self.assertEqual(int(scratch.checked_limits.numpy()[0]), 2)
        self.assertEqual(int(scratch.global_invalid.numpy()[0]), 0)

    def test_many_dofs_span_the_lane_stride(self):
        """Lanes stride the world's DOF list; a violation past the first stride still rejects."""
        data, raw, scratch = make_case()
        width = 70
        device = "cpu"
        data.world_dof_indices = wp.array([list(range(width))], dtype=wp.int32, device=device)
        data.world_dof_count = wp.array([width], dtype=wp.int32, device=device)
        data.limit_q_index = wp.array(list(range(width)), dtype=wp.int32, device=device)
        data.lower = wp.array([0.0] * width, dtype=wp.float32, device=device)
        data.upper = wp.array([np.inf] * width, dtype=wp.float32, device=device)
        data.prescribed = wp.zeros(width, dtype=wp.int32, device=device)
        data.q = wp.array([1.0] * width, dtype=wp.float32, device=device)
        data.v_hat = wp.array([0.0] * width, dtype=wp.float32, device=device)
        data.articulation_response_dof_count = wp.array([width, 0], dtype=wp.int32, device=device)
        data.body_has_response_dofs = wp.array([0, 0], dtype=wp.int32, device=device)
        self.assertEqual(evaluate(data, raw, scratch), 1)
        q = np.ones(width, np.float32)
        q[69] = -0.01
        data.q.assign(q)
        self.assertEqual(evaluate(data, raw, scratch), 0)
        self.assertEqual(int(scratch.checked_limits.numpy()[0]), 1)

    def test_prescribed_endpoint_and_restitution(self):
        """Retain known endpoint motion and frozen incident rebound semantics."""
        data, raw, scratch = make_case()
        raw.point0.assign([[0, 0, 0]])
        raw.shape_restitution.assign([1.0, 1.0])
        data.body_v_s.assign([[999, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]])
        self.assertEqual(evaluate(data, raw, scratch), 0)
        self.assertAlmostEqual(float(scratch.minimum_residual.numpy()[0]), -2.0, places=6)
        data.body_v_s.assign([[999, 0, 0, 0, 0, 0], [-1, 0, 0, 0, 0, 0]])
        self.assertEqual(evaluate(data, raw, scratch), 1)

    def test_kinematic_body_of_a_responding_tree_keeps_its_rows(self):
        """A contact with a kinematic body of a responding articulation is never resolved, on either twist source.

        Such a body, a free body made kinematic after construction or a
        kinematic root of a dynamic tree, moves with its prescribed velocity.
        The body pass rejects its world. The published path reads no twist
        for it and took it as still, resolving a world it was closing on.
        Persistent patch friction runs the capacity-bounded decision, whose
        contact check must reject the same contacts under both twist sources.
        """
        data, raw, scratch = make_case()

        def a(value, dtype=wp.float32):
            return wp.array(value, dtype=dtype, device="cpu")

        # Body 1 becomes a one-DOF body of its own responding, non-prescribed articulation.
        data.world_dof_indices, data.world_dof_count = a([[0, 1]], wp.int32), a([2], wp.int32)
        data.limit_q_index = a([-1, -1], wp.int32)
        data.lower, data.upper, data.v_hat = a([-np.inf] * 2), a([np.inf] * 2), a([0.0, 0.0])
        data.joint_S_s = a(np.eye(6, dtype=np.float32)[[0, 0]], wp.spatial_vector)
        data.articulation_response_dof_count = a([1, 1], wp.int32)
        data.prescribed_articulation = a([0, 0], wp.int32)
        data.body_has_response_dofs = a([1, 1], wp.int32)
        data.body_response_dof_mask = a([1, 1], wp.uint32)
        # (flag of body 1, its speed along the contact normal, resolved); positive speed closes the gap.
        for flag, speed, resolved in ((1, 2.0, 0), (1, -2.0, 1), (2, 2.0, 0), (2, -2.0, 0)):
            data.body_flags.assign([1, flag])
            data.v_hat.assign([0.0, speed])
            data.predicted_twist.assign([[0.0] * 6, [speed, 0.0, 0.0, 0.0, 0.0, 0.0]])
            for published in (False, True):
                with self.subTest(kinematic=flag == 2, speed=speed, published=published):
                    self.assertEqual(evaluate(data, raw, scratch, published=published), resolved)
                    self.assertEqual(int(scratch.checked_normals.numpy()[0]), 1)

    def test_zero_friction_radius_allows_slip(self):
        """Accept zero normal impulse without requiring zero tangential speed."""
        data, raw, scratch = make_case(free=True)
        raw.point0.assign([[0, 0, 0]])
        raw.shape_mu.assign([1000, 1000])
        data.v_hat.assign([0, 100, 0, 0, 0, 0])
        self.assertEqual(evaluate(data, raw, scratch), 1)

    def test_rigid_caps_and_uncapped_mf_depenetration(self):
        """Preserve finite speed caps and positive-infinity uncapped bias."""
        data, raw, scratch = make_case(free=True)
        data.v_hat.assign([0, 0, 0, 3, 0, 0])
        data.max_angular_velocity.assign([2, np.inf])
        self.assertEqual(evaluate(data, raw, scratch), 0)
        data.max_angular_velocity.assign([np.inf, np.inf])
        raw.point0.assign([[-1, 0, 0]])
        data.v_hat.assign([1, 0, 0, 0, 0, 0])
        self.assertEqual(evaluate(data, raw, scratch), 0)
        data.max_depenetration_velocity.assign([0.5, np.inf])
        self.assertEqual(evaluate(data, raw, scratch), 1)
        self.assertAlmostEqual(float(scratch.minimum_residual.numpy()[0]), 0.5, places=6)

    def test_rigid_caps_bound_speeds(self):
        """Keep the rows of a free body whose speed exceeds its cap although every component is within it."""
        for part, cap in ((slice(0, 3), "max_linear_velocity"), (slice(3, 6), "max_angular_velocity")):
            with self.subTest(cap=cap):
                data, raw, scratch = make_case(free=True)
                # Without contacts only the cap can keep the world's rows.
                raw.count.assign([0])
                velocity = np.zeros(6, dtype=np.float32)
                velocity[part] = (-0.6, 0.6, 0.0)
                data.v_hat.assign(velocity)
                getattr(data, cap).assign([1.0, np.inf])
                self.assertEqual(evaluate(data, raw, scratch), 1)
                getattr(data, cap).assign([0.8, np.inf])
                self.assertEqual(evaluate(data, raw, scratch), 0)

    def test_one_point_and_large_world_translation(self):
        """Act at the contact's one point, with stable endpoint-relative geometry."""
        data, raw, scratch = make_case()
        data.joint_S_s.assign([[0, 0, 0, 0, 0, 1]])
        data.v_hat.assign([1])
        raw.point0.assign([[0, 1, 0]])
        raw.point1.assign([[0, -1, 0]])
        self.assertEqual(evaluate(data, raw, scratch), 1)
        data.body_q.assign([[1e6, 0, 0, 0, 0, 0, 1]] * 2)
        data.articulation_origin.assign([[1e6, 0, 0]] * 2)
        self.assertEqual(evaluate(data, raw, scratch), 1)
        self.assertEqual(float(scratch.minimum_residual.numpy()[0]), 0.0)

    def test_empty_disabled_and_transient_decisions(self):
        """Clear prior acceptance on disabled and newly violating calls."""
        data, raw, scratch = make_case()
        self.assertEqual(evaluate(data, raw, scratch), 1)
        self.assertEqual(evaluate(data, raw, scratch, enabled=0), 0)
        self.assertEqual(evaluate(data, raw, scratch), 1)
        raw.count.assign([0])
        self.assertEqual(evaluate(data, raw, scratch), 1)
        self.assertEqual(int(scratch.checked_normals.numpy()[0]), 0)

    def test_malformed_prefix_ids_and_nonfinite_input(self):
        """Reject incomplete or invalid raw constraints instead of dropping them."""
        data, raw, scratch = make_case()
        raw.count.assign([2])
        self.assertEqual(evaluate(data, raw, scratch), 0)
        raw.count.assign([1])
        raw.shape0.assign([999])
        self.assertEqual(evaluate(data, raw, scratch), 0)
        raw.shape0.assign([0])
        raw.normal.assign([[0, 0, 0]])
        self.assertEqual(evaluate(data, raw, scratch), 0)
        raw.normal.assign([[np.nan, 0, 0]])
        self.assertEqual(evaluate(data, raw, scratch), 0)
        raw.normal.assign([[-1, 0, 0]])
        data.v_hat.assign([np.inf])
        self.assertEqual(evaluate(data, raw, scratch), 0)
        data.v_hat.assign([0.0])
        data.dt = -1.0
        self.assertEqual(evaluate(data, raw, scratch), 0)

    def test_rejection_margin_is_not_violation_allowance(self):
        """Send numerically ambiguous nonnegative residuals to fallback."""
        data, raw, scratch = make_case()
        raw.point0.assign([[0, 0, 0]])
        self.assertEqual(evaluate(data, raw, scratch), 1)
        data.absolute_margin = 1.0e-5
        self.assertEqual(evaluate(data, raw, scratch), 0)
        data.v_hat.assign([2.0e-5])
        self.assertEqual(evaluate(data, raw, scratch), 1)

    def test_unsupported_constructor_has_empty_mask(self):
        """Keep unsupported solvers disabled without raising or launching."""
        solver = SimpleNamespace(
            model=SimpleNamespace(device=wp.get_device("cpu")),
            spec=SimpleNamespace(pins=SolverFeatherPGS.Pins()),
            placement=SimpleNamespace(keyed_rows=None),
            laws=SimpleNamespace(set=plain_laws(warm_start=True)),
        )
        classifier = rw.ResolvedWorlds(solver)
        self.assertFalse(classifier.enabled)
        self.assertEqual(classifier.resolved.shape, (0,))
        classifier.classify(None, None, None, 0.0)

    def test_patch_allocator_marks_the_contacts_the_complete_path_reserves(self):
        """In a resolved world, mark exactly the contacts the allocator would give rows; leave other worlds alone."""
        complete = _allocate(row_slots.allocate_world_contact_slots, [])
        patch_friction = _allocate(row_slots.allocate_world_contact_slots, [1, 0])
        np.testing.assert_array_equal(complete["contact_slot"] >= 0, [True, False, False, True, False, True])
        np.testing.assert_array_equal(complete["contact_path"], [0, -1, -1, 0, -1, 1])
        world_zero = np.array([True, True, True, False, False, True])
        expected = np.where(complete["contact_slot"] >= 0, int(CONTACT_PATH_RESOLVED), -1)
        np.testing.assert_array_equal(patch_friction["contact_path"][world_zero], expected[world_zero])
        np.testing.assert_array_equal(patch_friction["contact_slot"][world_zero], -1)
        np.testing.assert_array_equal(patch_friction["contact_slots_needed"][world_zero], 0)
        # The resolved world reserves no rows; world 1 and the cross-world pair allocate as before.
        self.assertEqual(int(patch_friction["world_slot_counter"][0]) + int(patch_friction["mf_slot_counter"][0]), 0)
        for name in ("contact_slot", "contact_path", "contact_slots_needed"):
            np.testing.assert_array_equal(patch_friction[name][~world_zero], complete[name][~world_zero], err_msg=name)

    def test_patch_release_repeats_the_allocator_admission(self):
        """Mark a resolved contact by the allocator's own admission: the kernel admits every contact once."""
        allocator = _definition(row_slots.allocate_world_contact_slots)
        admissions = [
            node
            for node in ast.walk(allocator)
            if isinstance(node, ast.Call) and ast.unparse(node.func) == "_admit_contact"
        ]
        self.assertEqual(len(admissions), 1)

    def test_patch_finish_releases_resolved_contacts_only(self):
        """The finish epilogue releases CONTACT_PATH_RESOLVED anchors, keeps unrouted ones and finishes routed ones."""
        model, state, _, patches = _patch_fixture([[-0.2, 0, 0], [0.0, 0, 0], [0.2, 0, 0], [0.4, 0, 0]])
        count = 4
        state.body_qd = wp.zeros(model.body_count, dtype=wp.spatial_vector, device="cpu")
        com = wp.zeros(model.body_count, dtype=wp.vec3, device="cpu")
        world = wp.zeros(count, dtype=int, device="cpu")
        # 0 resolved, 1 capacity-rejected, 2 routed with load, 3 routed without load.
        slot = wp.array([-1, -1, 0, 3], dtype=int, device="cpu")
        path = wp.array([int(CONTACT_PATH_RESOLVED), -1, 0, 0], dtype=int, device="cpu")
        needed = wp.array([0, 0, 3, 3], dtype=int, device="cpu")
        parents = wp.array([[-1, 0, 0, -1, 3, 3, -1, -1]], dtype=int, device="cpu")
        mu = wp.full((1, 8), 0.5, dtype=float, device="cpu")
        impulses = wp.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=float, device="cpu")
        routes = patch_routes(((0, parents, mu, impulses, row_lines(parents.device)),))
        for label, finish_routes, expected in (
            ("finish", routes, [0, 1, 1, 0]),
            ("store only", patch_routes(()), [1, 1, 1, 1]),
        ):
            with self.subTest(label):
                patches.current.valid.assign([1] * count + [0] * (patches.capacity - count))
                patches.finish(state, state, com, world, slot, path, needed, finish_routes, DT)
                np.testing.assert_array_equal(patches.previous.valid.numpy()[:count], expected)

    def test_patch_bound_resolves_only_worlds_whose_rows_fit_each_route(self):
        """Count each checked limit as one dense row and each checked contact as three rows on its route."""
        data, raw, scratch = make_case()
        self.assertEqual(evaluate(data, raw, scratch), 1)
        # The articulated contact takes dense rows.
        self.assertEqual(evaluate(data, raw, scratch, row_bound=(3, 0)), 1)
        self.assertEqual(evaluate(data, raw, scratch, row_bound=(2, 100)), 0)
        data.lower.assign([-1.0])
        data.activation_gap = float("inf")
        self.assertEqual(evaluate(data, raw, scratch, row_bound=(4, 0)), 1)
        self.assertEqual(int(scratch.checked_limits.numpy()[0]), 1)
        self.assertEqual(evaluate(data, raw, scratch, row_bound=(3, 100)), 0)
        # A rejected world stays rejected under any bound.
        data.v_hat.assign([-2.0])
        self.assertEqual(evaluate(data, raw, scratch, row_bound=(100, 100)), 0)
        # A free body's contact with a prescribed endpoint takes matrix-free rows.
        data, raw, scratch = make_case(free=True)
        self.assertEqual(evaluate(data, raw, scratch, row_bound=(0, 3)), 1)
        self.assertEqual(evaluate(data, raw, scratch, row_bound=(100, 2)), 0)

    def test_patch_friction_admission(self):
        """Admit persistent patch friction on the dense route and contact torsion (a separating contact carries no
        spin either), not compliance."""
        solver = _admissible_solver()
        self.assertTrue(rw.supported(solver))
        solver.laws.set = plain_laws(torsion=True)
        self.assertTrue(rw.supported(solver))
        solver.laws.set = plain_laws(compliance=True)
        self.assertFalse(rw.supported(solver))
        solver.spec.classes.case = "K1"
        self.assertFalse(rw.supported(solver))

    def test_classify_bounds_worlds_by_the_solver_capacities(self):
        """The decision gets the solver's dense and matrix-free row capacities."""
        model = _build("cpu", worlds=2)
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        state = model.state()
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        pipeline.collide(state, contacts)
        augmented = SimpleNamespace(
            body_v_s=wp.zeros(model.body_count, dtype=wp.spatial_vector, device="cpu"),
            joint_S_s=wp.zeros(model.joint_dof_count, dtype=wp.spatial_vector, device="cpu"),
        )
        solver = SolverFeatherPGS(model, dense_max_constraints=48, mf_max_constraints=24)
        # The classifier owns only the CUDA matrix-free route; split mode keeps no world DOF map,
        # which the mocked decision launch never reads.
        solver.forest = dataclasses.replace(
            solver.forest,
            world_dof_indices=wp.zeros((model.world_count, 1), dtype=int, device="cpu"),
            world_dof_count=wp.zeros(model.world_count, dtype=int, device="cpu"),
        )
        with patch.object(rw, "decline", return_value=None):
            owner = rw.ResolvedWorlds(solver)
        with patch.object(rw, "launch") as launch:
            owner.classify(state, augmented, contacts, DT)
        launch.assert_called_once()
        options = launch.call_args.kwargs
        # A free body is a one-node tree: its contacts are rows of the line, which hold their capacity (48 + 24), and the
        # free bodies' rows their speed rows alone.
        self.assertEqual(options["row_bound"], (72, solver.placement.mf_max_constraints))
        self.assertEqual(options["free_line"], 1)
        self.assertIs(options["free_normals"], owner.free_normals)
        self.assertEqual(owner.free_normals.shape, (model.world_count,))

    def test_policy_names_no_robot(self):
        """The admission reads row kinds and sizes, never a robot."""
        source = inspect.getsource(rw).lower()
        for token in ("franka", "kuka", "allegro", "size_groups", "[23, 6]"):
            self.assertNotIn(token, source)


def _admissible_solver():
    """A stand-in solver whose policy the classifier admits, on any device."""
    return SimpleNamespace(
        factor=SimpleNamespace(block_count=0),
        placement=SimpleNamespace(keyed_rows=None),
        model=SimpleNamespace(device=SimpleNamespace(is_cuda=False)),
        spec=SimpleNamespace(pins=SolverFeatherPGS.Pins(), classes=SimpleNamespace(case="K6")),
        forest=SimpleNamespace(
            max_world_dofs=29,
            world_dof_indices=1,
            world_dof_count=1,
            body_response_dof_mask=1,
            body_has_response_dofs=1,
        ),
        laws=SimpleNamespace(set=plain_laws(), mimic_count=0, link_count=0, joint_limit_q_index=1),
        options=SimpleNamespace(enable_bilateral_preelimination=False, diagnostics=SolverFeatherPGS.Diagnostics()),
    )


def _rollout(
    model,
    *,
    classifier,
    steps=90,
    published=True,
    history=None,
    solver_kwargs=None,
    toss_every=2,
    friction_anchor_beta=0.2,
):
    """Swing the arms and toss every ``toss_every``-th free body; return the solver, poses and resolved masks.

    ``history`` collects one record per step: the stored anchor validity (``valid``), the count of
    contacts the classifier marked CONTACT_PATH_RESOLVED (``marked``), how many of those held an
    anchor when the patch finish began, which releases it (``released``), and, with the classifier,
    the worlds whose rows all separate (``separating``) and those it resolved (``resolved``).
    """
    # The classifier evaluates the serial matrix-free sweep's law: the pin keeps the Jacobi tiers (case K4, an arm
    # beside a free body, whose keyed rows own the contact rows) off these worlds.
    pins = {"storage": "dense", "predicted_twists": bool(published), "tiers": 0}
    if not classifier:
        pins["claims"] = ()
    solver = SolverFeatherPGS(
        model,
        pins=SolverFeatherPGS.Pins(**pins),
        pgs_iterations=8,
        enable_joint_limits=True,
        update_mass_matrix_interval=2,
        friction_anchor_beta=friction_anchor_beta,
        **(solver_kwargs or {}),
    )
    pipeline = newton.CollisionPipeline(model, deterministic=True)
    contacts = pipeline.contacts()
    states = [model.state() for _ in range(2)]
    newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
    control = model.control()
    control.joint_target_q.assign(np.zeros_like(model.joint_target_q.numpy()))
    qd = np.zeros(model.joint_dof_count, dtype=np.float32)
    types, qd_start = model.joint_type.numpy(), model.joint_qd_start.numpy()
    for joint in np.flatnonzero(types == int(newton.JointType.REVOLUTE))[::3]:
        qd[qd_start[joint]] = 1.5
    # A tossed body starts moving up: its ground contact separates at the predictor.
    for joint in np.flatnonzero(types == int(newton.JointType.FREE))[::toss_every]:
        qd[qd_start[joint] + 2] = 1.0
    states[0].joint_qd.assign(qd)
    held = []
    if history is not None:
        patches = solver.laws.friction_patches
        finish = patches.finish

        def finish_and_record(*args, **kwargs):
            # The anchors the contacts hold before the finish keeps or releases them.
            held.append(patches.current.valid.numpy().copy())
            finish(*args, **kwargs)

        patches.finish = finish_and_record
    poses, masks = [], []
    for _ in range(steps):
        held.clear()
        pipeline.collide(states[0], contacts)
        states[0].clear_forces()
        solver.step(states[0], states[1], control, contacts, DT)
        states.reverse()
        poses.append(states[0].body_q.numpy().copy())
        masks.append(solver.sweep.resolved_worlds.numpy().copy())
        if history is not None:
            count = int(contacts.rigid_contact_count.numpy()[0])
            marked = solver.placement.contact_path.numpy()[:count] == int(CONTACT_PATH_RESOLVED)
            (anchors,) = held
            record = {
                "valid": patches.previous.valid.numpy().copy(),
                "marked": int(np.sum(marked)),
                "released": int(np.sum(marked & (anchors[:count] != 0))),
            }
            owner = solver.sweep.resolved_worlds_owner
            if owner is not None:
                scratch = owner.scratch
                invalid = int(scratch.global_invalid.numpy()[0]) != 0
                record["separating"] = (scratch.rejected.numpy() == 0) & (not invalid)
                record["resolved"] = owner.resolved.numpy() != 0
            history.append(record)
    wp.synchronize_device(model.device)
    return solver, np.array(poses), np.array(masks)


def _tossed_spheres(device, worlds=4):
    """Per world, a two-joint arm hanging clear of the ground and a free sphere 1 cm above it.

    The sphere's one contact holds a patch anchor, whose three rows a matrix-free capacity of two
    rows rejects on the complete path whatever the slot order, so the anchor keeps its history.
    """
    scene = newton.ModelBuilder()
    cfg = scene.default_shape_cfg.copy()
    cfg.mu = 0.6
    parent, joints = -1, []
    for link in range(2):
        body = scene.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * 0.04), lock_inertia=True)
        joints.append(
            scene.add_joint_revolute(
                parent=parent,
                child=body,
                axis=newton.Axis.Y,
                parent_xform=wp.transform(wp.vec3(0.0, 0.0, 1.0 if link == 0 else -0.2), wp.quat_identity()),
                target_ke=30.0,
                target_kd=2.0,
                limit_lower=-1.5,
                limit_upper=1.5,
            )
        )
        parent = body
    scene.add_shape_capsule(parent, radius=0.03, half_height=0.1, cfg=cfg)
    scene.add_articulation(joints)
    sphere = scene.add_body(xform=wp.transform(wp.vec3(0.6, 0.0, 0.06), wp.quat_identity()), mass=0.5)
    scene.add_shape_sphere(sphere, radius=0.05, cfg=cfg)
    builder = newton.ModelBuilder()
    builder.replicate(scene, worlds, spacing=(3.0, 3.0, 0.0))
    builder.add_ground_plane()
    model = builder.finalize(device=device)
    model.rigid_contact_max = 256
    return model


class TestResolvedWorldsDevice(unittest.TestCase):
    """The classifier on the default device: one rule decides which rows exist on the CPU and on CUDA."""

    def test_resolved_worlds_match_the_complete_path(self):
        """Fixed-base arms over free boxes: resolved worlds step identically with no rows built."""
        model = _build(wp.get_device(), worlds=6)
        plain_solver, plain, plain_masks = _rollout(model, classifier=False)
        self.assertIsNone(plain_solver.sweep.resolved_worlds_owner)
        self.assertEqual(plain_masks.shape[1], 0)
        solver, poses, masks = _rollout(model, classifier=True)
        self.assertIsNotNone(solver.sweep.resolved_worlds_owner)
        self.assertEqual(masks.shape, (90, 6))
        resolved_steps = int(masks.sum())
        self.assertGreater(resolved_steps, 0)
        self.assertLess(resolved_steps, masks.size)
        # The tossed boxes resolve while they rise; once a box rests its contact closes the world.
        self.assertTrue(masks[:5, ::2].all())
        self.assertFalse(masks[-30:].all(axis=0).any())
        self.assertTrue(np.isfinite(poses).all())
        np.testing.assert_allclose(poses[:, :, :3], plain[:, :, :3], atol=1.0e-5, rtol=0.0)
        np.testing.assert_allclose(poses[:, :, 3:], plain[:, :, 3:], atol=1.0e-5, rtol=0.0)

    def test_row_watermark_counts_the_rows_it_does_not_choose(self):
        """The row watermark (a diagnostic) resolves the same worlds and steps the same state as the telemetry off, and
        counts the rows the resolved worlds skipped."""
        model = _build(wp.get_device(), worlds=6)
        _solver, poses, masks = _rollout(model, classifier=True, steps=30)
        diagnostics = {"diagnostics": SolverFeatherPGS.Diagnostics(row_watermark=True)}
        solver, watched_poses, watched_masks = _rollout(model, classifier=True, steps=30, solver_kwargs=diagnostics)
        np.testing.assert_array_equal(watched_masks, masks)
        np.testing.assert_array_equal(watched_poses, poses)
        self.assertGreater(int(masks.sum()), 0)
        self.assertGreater(solver.constraint_row_watermarks()["resolved_high_water"], 0)

    def test_patch_resolved_worlds_match_the_complete_path(self):
        """Persistent patch friction: resolved worlds step identically and release the same anchors.

        Boxes stick on the ground, rise (their worlds resolve and their anchors release), and land
        again; arms of three links run the general sweep, arms of seventeen the paired factor sweep.
        """
        # The seventeen-link arm has 34 position limits: keep its rows within capacity as Kuka's recipe does.
        for links, solver_kwargs in ((3, {}), (17, {"dense_max_constraints": 128, "joint_limit_activation_gap": 0.1})):
            with self.subTest(links=links):
                model = _build(wp.get_device(), worlds=6, links=links)
                complete_history, history = [], []
                plain_solver, plain, plain_masks = _rollout(
                    model,
                    classifier=False,
                    friction_anchor_beta=None,
                    history=complete_history,
                    solver_kwargs=solver_kwargs,
                )
                self.assertIsNone(plain_solver.sweep.resolved_worlds_owner)
                self.assertEqual(plain_masks.shape[1], 0)
                solver, poses, masks = _rollout(
                    model, classifier=True, friction_anchor_beta=None, history=history, solver_kwargs=solver_kwargs
                )
                self.assertIsNotNone(solver.sweep.resolved_worlds_owner)
                self.assertEqual(
                    solver.sweep.resolved_worlds_owner.row_bound,
                    (solver.placement.dense_max_constraints, solver.placement.mf_max_constraints),
                )
                self.assertGreater(int(masks.sum()), 0)
                self.assertLess(int(masks.sum()), masks.size)
                # A marked contact need not hold an anchor: count the anchors the finish released.
                self.assertGreater(sum(record["released"] for record in history), 0, "no marked contact held an anchor")
                # Boxes start just above the ground; resting ones hold anchors once they land.
                self.assertGreater(max(int(np.sum(record["valid"])) for record in complete_history), 0, "no anchor")
                for step, (record, expected) in enumerate(zip(history, complete_history, strict=True)):
                    np.testing.assert_array_equal(
                        record["valid"], expected["valid"], err_msg=f"previous.valid at step {step}"
                    )
                assert_history_close(solver, plain_solver)
                self.assertTrue(np.isfinite(poses).all())
                np.testing.assert_allclose(poses[:, :, :3], plain[:, :, :3], atol=1.0e-5, rtol=0.0)
                np.testing.assert_allclose(poses[:, :, 3:], plain[:, :, 3:], atol=1.0e-5, rtol=0.0)

    def test_patch_bound_keeps_worlds_whose_rows_overflow(self):
        """A separating world whose rows overflow a route stays on the complete path, which keeps the anchors.

        Each tossed sphere's contact separates, so the unbounded decision would resolve its world and
        release the anchor; the complete path rejects the contact's three rows for capacity instead and
        keeps the anchor. The capacity bound (three rows per checked contact) must hold the world.
        """
        model = _tossed_spheres(wp.get_device())
        # Thirty steps: each sphere rises about 5 cm and starts falling, inside the 10 cm contact gap.
        # A free body's contacts are rows of the line: two rows a world cannot hold a contact's three.
        options = {
            "steps": 30,
            "friction_anchor_beta": None,
            "solver_kwargs": {"dense_max_constraints": 1, "mf_max_constraints": 1},
        }
        complete_history, history = [], []
        plain_solver, plain, _ = _rollout(model, classifier=False, history=complete_history, toss_every=1, **options)
        solver, poses, _ = _rollout(model, classifier=True, history=history, toss_every=1, **options)
        self.assertEqual(
            solver.sweep.resolved_worlds_owner.row_bound,
            (solver.placement.dense_max_constraints, solver.placement.mf_max_constraints),
        )
        held = sum(int(np.sum(record["separating"] & ~record["resolved"])) for record in history)
        self.assertGreater(held, 0, "no separating world was held for capacity")
        self.assertEqual(sum(record["marked"] for record in history), 0)
        # The complete path keeps the anchor of every rejected contact.
        self.assertGreater(min(int(np.sum(record["valid"])) for record in complete_history), 0, "no anchor kept")
        for step, (record, expected) in enumerate(zip(history, complete_history, strict=True)):
            np.testing.assert_array_equal(record["valid"], expected["valid"], err_msg=f"previous.valid at step {step}")
        assert_history_close(solver, plain_solver)
        np.testing.assert_allclose(poses, plain, atol=1.0e-5, rtol=0.0)

    def test_published_twists_match_the_body_pass(self):
        """The predictor's published twists decide the same worlds as the classifier's own body pass."""
        model = _build(wp.get_device(), worlds=6)
        own_solver, own_poses, own_masks = _rollout(model, classifier=True, published=False)
        solver, poses, masks = _rollout(model, classifier=True)
        self.assertFalse(own_solver.sweep.resolved_worlds_owner.published)
        if not model.device.is_cuda:
            # The predictor publishes its twists from a CUDA register scan (tree_kinetic_owner): the CPU's classifier
            # builds them itself in both runs.
            self.assertFalse(solver.sweep.resolved_worlds_owner.published)
            return
        self.assertTrue(solver.sweep.resolved_worlds_owner.published)
        self.assertGreater(int(masks.sum()), 0)
        # The published twist sums the same axis terms in tree order; a residual within
        # rounding of the margin may decide differently on a handful of steps.
        self.assertGreater(float(np.mean(masks == own_masks)), 0.995)
        np.testing.assert_allclose(poses[:, :, :3], own_poses[:, :, :3], atol=1.0e-5, rtol=0.0)
        np.testing.assert_allclose(poses[:, :, 3:], own_poses[:, :, 3:], atol=1.0e-5, rtol=0.0)


def _walkers(device, worlds=4):
    """One free-root quadruped per world on the ground: a class the general sparse route owns."""
    scene = newton.ModelBuilder()
    cfg = scene.default_shape_cfg.copy()
    cfg.mu = 0.6
    _add_quadruped(scene, cfg)
    builder = newton.ModelBuilder()
    builder.replicate(scene, worlds, spacing=(3.0, 3.0, 0.0))
    builder.add_ground_plane()
    model = builder.finalize(device=device)
    model.rigid_contact_max = 2048
    return model


@unittest.skipUnless(wp.is_cuda_available(), "The sparse route is a CUDA path")
class TestResolvedWorldsSparseRoute(unittest.TestCase):
    def test_sparse_route_allocates_contacts_with_an_empty_mask(self):
        """Removed packed-route flags leave the general sparse route, which declines the classifier."""
        model = _walkers(wp.get_device("cuda:0"))
        # A recipe that still sets the removed packed sparse-factor selectors: unknown names.
        env = {
            "FEATHER_PGS_SPARSE_FACTOR": "1",
            "FEATHER_PGS_SPARSE_LEVEL_UPDATE": "1",
            "FEATHER_PGS_SPARSE_PARALLEL_LIMITS": "1",
        }
        with patch.dict(os.environ, env):
            solver = SolverFeatherPGS(
                model,
                pgs_iterations=8,
                enable_joint_limits=True,
                dense_max_constraints=56,
            )
        self.assertLessEqual(set(env), set(solver.get_path_selection()["environment"]["unknown"]))
        self.assertIsNotNone(solver.spec.classes.packed_size)
        routes = solver.get_path_selection()["routes"]
        self.assertNotIn("research_sparse_factor", routes)
        self.assertEqual(routes["sweep"]["route"], "sparse_pgs")
        self.assertIsNone(solver.sweep.resolved_worlds_owner)
        self.assertEqual(solver.sweep.resolved_worlds.shape, (0,))
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        states = [model.state() for _ in range(2)]
        newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
        control = model.control()
        for _ in range(30):
            pipeline.collide(states[0], contacts)
            states[0].clear_forces()
            solver.step(states[0], states[1], control, contacts, DT)
            states.reverse()
        wp.synchronize_device(model.device)
        poses = states[0].body_q.numpy().reshape(model.world_count, -1, 7)
        self.assertTrue(np.isfinite(poses).all())
        self.assertTrue(np.all(poses[:, 0, 2] > 0.2))
        self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)


if __name__ == "__main__":
    unittest.main()
