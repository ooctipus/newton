# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check complete raw-contact construction and the original G1 physical law."""

import json
import os
import unittest
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import parallel_world, parallel_world_rows
from tools.fpgs_bench import test_g1_kinetic_state as kinetic
from tools.fpgs_bench import test_sparse_metric_tangents as metric
from tools.fpgs_bench.test_body_basis_rows import bind_factor, build, check_rows
from tools.fpgs_bench.test_sparse_contact_block import saved_records
from tools.fpgs_bench.test_sparse_factor import fixture, physical_rows

KEY = "sparse_parallel_world43_s18_c100"
PRODUCER_KEY = "sparse_parallel_world_rows43_s18_c100"
SOLVE_KEY = "sparse_metric_tangent43_s18_c100"
ENV = {
    **kinetic.ENV,
    "FEATHER_PGS_G1_KINETIC_STATE": "1",
    "FEATHER_PGS_G1_CHAIN_SCAN": "1",
    "FEATHER_PGS_COMPILED_COORDINATE_STATE": "0",
    "FEATHER_PGS_BODY_BASIS_ROWS": "0",
    "FEATHER_PGS_SPARSE_PAIRED_GS": "0",
    "FEATHER_PGS_PARALLEL_WORLD": "1",
    "FEATHER_PGS_PARALLEL_WORLD_SPLIT": "0",
}


def row_snapshot(solver):
    """Retain canonical physical metadata before another producer overwrites it."""
    count = int(solver.constraint_count.numpy()[0])
    names = (
        "diag",
        "rhs",
        "row_type",
        "row_parent",
        "row_mu",
        "phi",
        "row_cfm",
        "row_restitution",
        "row_beta",
        "target_velocity",
    )
    result = {name: getattr(solver, name).numpy()[0, :count].copy() for name in names}
    result["count"] = count
    for name in ("contact_path", "contact_slot", "contact_slots_needed"):
        result[name] = getattr(solver, name).numpy().copy()
    return result


def check_row_identity(test, expected, actual, raw_count):
    """Match raw IDs and complete triplets while permitting reordered contact slots."""
    test.assertEqual(actual["count"], expected["count"])
    order = np.arange(actual["count"])
    occupied = []
    for raw in range(raw_count):
        old = int(expected["contact_slot"][raw])
        new = int(actual["contact_slot"][raw])
        old_valid = old >= 0 and expected["contact_path"][raw] == 0
        new_valid = new >= 0 and actual["contact_path"][raw] == 0
        test.assertEqual(new_valid, old_valid, f"raw {raw}")
        if not old_valid:
            continue
        width = int(expected["contact_slots_needed"][raw])
        test.assertEqual(int(actual["contact_slots_needed"][raw]), width)
        test.assertLessEqual(new + width, actual["count"])
        occupied.extend(range(new, new + width))
        order[new : new + width] = np.arange(old, old + width)
        test.assertEqual(int(actual["row_type"][new]), 0)
        for direction in range(1, width):
            test.assertEqual(int(actual["row_type"][new + direction]), 2)
            test.assertEqual(int(actual["row_parent"][new + direction]), new)
        for name in ("diag", "rhs", "row_mu", "phi", "row_cfm", "row_restitution", "row_beta", "target_velocity"):
            np.testing.assert_allclose(
                actual[name][new : new + width],
                expected[name][old : old + width],
                rtol=3e-5,
                atol=3e-6,
                err_msg=f"{name}, raw {raw}",
            )
    test.assertEqual(len(occupied), len(set(occupied)), "Overlapping raw-contact row slots")
    limits = np.flatnonzero(actual["row_type"] == 3)
    old_limits = np.flatnonzero(expected["row_type"] == 3)
    test.assertEqual(len(limits), len(old_limits))
    order[limits] = old_limits
    for name in ("diag", "rhs", "phi", "row_cfm", "row_beta", "target_velocity"):
        np.testing.assert_allclose(actual[name][limits], expected[name][old_limits], rtol=3e-5, atol=3e-6, err_msg=name)
    test.assertEqual(sorted([*occupied, *limits.tolist()]), list(range(actual["count"])))
    return order


def physical_score(test, case, matrix, velocity, impulses, vhat, *, physical_solver=None):
    """Check original-law residuals and momentum using independently reconstructed J."""
    solver = case["solver"]
    rows = row_snapshot(solver)
    view = case if physical_solver is None else {**case, "solver": physical_solver}
    J = physical_rows(view)
    test.assertEqual(J.shape, (rows["count"], 43))
    test.assertTrue(np.isfinite(velocity).all() and np.isfinite(impulses).all())
    test.assertGreaterEqual(float(np.min(impulses[np.isin(rows["row_type"], (0, 3))], initial=0)), -1e-7)
    defect = kinetic.backward_error(matrix, velocity.astype(float) - vhat.astype(float), J.T @ impulses)
    test.assertLess(defect, 2e-6, "held physical momentum")
    score = metric.physical_metrics(
        J,
        rows["diag"],
        rows["rhs"],
        rows["row_type"],
        rows["row_parent"],
        rows["row_mu"],
        vhat,
        velocity,
        impulses,
    )
    test.assertTrue(np.isfinite(list(score.values())).all())
    test.assertLess(score["cone"], 3e-5)
    return rows, score, defect


def held_norm(matrix, velocity):
    """Measure a physical velocity difference in the independent held operator."""
    value = np.asarray(velocity, dtype=float)
    return float(np.sqrt(max(0.0, value @ matrix @ value)))


def physical_reference(J, matrix, rows, vhat):
    """Reuse original metric GS in FP64, without donating iterations to native."""
    J, matrix, cold = np.asarray(J, float), np.asarray(matrix, float), np.asarray(vhat, float)
    response = np.linalg.solve(matrix, J.T).T
    diagonal = np.sum(J * response, axis=1) + rows["row_cfm"]
    arguments = (J, response, diagonal, rows["rhs"], rows["row_type"], rows["row_parent"], rows["row_mu"])

    def score(velocity, impulse):
        return metric.physical_metrics(
            J, diagonal, rows["rhs"], rows["row_type"], rows["row_parent"], rows["row_mu"], cold, velocity, impulse
        )

    eight_velocity, eight_impulse, _ = metric.reference(*arguments, cold, iterations=8)
    velocity, impulse, sweep_bound = cold.copy(), np.zeros(len(J)), 0
    # Fixed diagnostic schedule from the independent row-order investigation.
    # CFM stays denominator-only; residual and momentum use physical J/H.
    for allowance in (64, 192, 768, 3072, 4096):
        velocity, impulse, _ = metric.reference(*arguments, velocity, iterations=allowance, incoming=impulse)
        sweep_bound += allowance
        scores = score(velocity, impulse)
        if max(scores.values()) <= 1e-8:
            break
    return {
        "eight_velocity": eight_velocity,
        "eight_score": score(eight_velocity, eight_impulse),
        "velocity": velocity,
        "score": scores,
        "sweep_bound": sweep_bound,
    }


def held_reference_errors(matrix, vhat, original, candidate, reference, candidate_reference):
    """Compare both native endpoints to one independently converged physical state."""
    scale = 1.0 + held_norm(matrix, reference - vhat)
    return {
        "original": held_norm(matrix, original - reference) / scale,
        "candidate": held_norm(matrix, candidate - reference) / scale,
        "reference_order_agreement": held_norm(matrix, candidate_reference - reference) / scale,
    }


def check_held_reference(test, errors):
    """Use the existing 3e-5 physical-reference allowance, not component monotonicity."""
    test.assertLessEqual(errors["reference_order_agreement"], 3e-5, "row-order reference agreement")
    test.assertLessEqual(errors["candidate"], errors["original"] + 3e-5, "held-H reference error")


def check_eight_reproduction(test, matrix, vhat, velocity, score, reference):
    """Require the unchanged eight-sweep law in each actual native row order."""
    scale = 1.0 + held_norm(matrix, reference["eight_velocity"] - vhat)
    test.assertLessEqual(
        held_norm(matrix, velocity - reference["eight_velocity"]) / scale, 3e-5, "original eight-sweep physical action"
    )
    for name, value in reference["eight_score"].items():
        test.assertLessEqual(abs(score[name] - value), 3e-5 * max(1.0, abs(value)), f"eight-sweep {name}")


class TestParallelWorld(unittest.TestCase):
    def test_split_producer_factory(self):
        """Require the distinct producer without replacing the original metric solver."""
        self.assertEqual(parallel_world_rows.get_producer_kernel().key, PRODUCER_KEY)

    def test_factory(self):
        """Require the complete owner and its fixed-capacity production kernel."""
        self.assertTrue(callable(parallel_world.create))
        self.assertEqual(parallel_world_rows.get_kernel().key, KEY)

    def test_cpu_rejection_keeps_original_storage_and_solver(self):
        """Keep the complete original path when the native owner cannot be installed."""
        with patch.dict(os.environ, ENV):
            case = fixture()
        owner = case["owner"]
        kernel, rows = owner.kernels.solve, owner.data.Z
        self.assertIsNone(owner.parallel_world)
        self.assertIsNone(parallel_world.create(owner))
        self.assertIs(owner.kernels.solve, kernel)
        self.assertIs(owner.data.Z, rows)

    def test_raw_routing_preserves_ids_and_ignores_only_unowned_pairs(self):
        """Keep raw IDs across static gaps and reject malformed descriptors separately."""
        device = "cpu"

        def array(values):
            return wp.array(values, dtype=wp.int32, device=device)

        buckets = parallel_world.ParallelWorldBuckets(2, 4, device=device)
        count = array([4])
        shape0, shape1 = array([0, 2, 0, 1]), array([2, 2, 1, 2])
        shape_body, body_art, art_world = array([0, 1, -1]), array([0, 1]), array([0, 1])
        slot, path, needed = (array([777] * 4) for _ in range(3))

        def build_buckets():
            buckets.build(count, shape0, shape1, shape_body, body_art, art_world, slot=slot, path=path, needed=needed)

        build_buckets()
        np.testing.assert_array_equal(buckets.data.invalid.numpy(), [0])
        np.testing.assert_array_equal(buckets.data.offsets.numpy(), [0, 1, 2])
        np.testing.assert_array_equal(buckets.data.ids.numpy()[:2], [0, 3])
        np.testing.assert_array_equal(buckets.counts.numpy(), [1, 1, 2])
        for values in (slot.numpy(), path.numpy()):
            np.testing.assert_array_equal(values[[1, 2]], [-1, -1])
        np.testing.assert_array_equal(needed.numpy()[[1, 2]], [0, 0])
        for raw_count, raw_shapes in ((4, [0, 99, 0, 1]), (5, [0, 2, 0, 1])):
            count.assign([raw_count])
            shape0.assign(raw_shapes)
            build_buckets()
            self.assertNotEqual(int(buckets.data.invalid.numpy()[0]), 0)

    def test_row_identity_accepts_reordering_not_missing_triplets(self):
        """Accept a complete raw-key permutation and reject aliased contact slots."""
        expected = {
            "count": 7,
            "contact_slot": np.array([1, 4]),
            "contact_path": np.zeros(2, int),
            "contact_slots_needed": np.full(2, 3),
            "row_type": np.array([3, 0, 2, 2, 0, 2, 2]),
            "row_parent": np.array([-1, -1, 1, 1, -1, 4, 4]),
        }
        for name in ("diag", "rhs", "row_mu", "phi", "row_cfm", "row_restitution", "row_beta", "target_velocity"):
            expected[name] = np.zeros(7)
        actual = {name: value.copy() if isinstance(value, np.ndarray) else value for name, value in expected.items()}
        actual["contact_slot"] = np.array([4, 1])
        np.testing.assert_array_equal(check_row_identity(self, expected, actual, 2), [0, 4, 5, 6, 1, 2, 3])
        actual["contact_slot"][:] = 1
        with self.assertRaises(AssertionError):
            check_row_identity(self, expected, actual, 2)

    def test_reference_distance_rejects_worse_endpoint(self):
        """Accept equal physical error and reject a genuinely worse held-H endpoint."""
        matrix = np.diag([4.0, 9.0])
        cold, reference = np.zeros(2), np.zeros(2)
        original = np.array([0.1, 0.0])
        equal = held_reference_errors(matrix, cold, original, original.copy(), reference, reference)
        check_held_reference(self, equal)
        worse = held_reference_errors(matrix, cold, original, np.array([0.15, 0.0]), reference, reference)
        self.assertAlmostEqual(worse["original"], 0.2)
        self.assertAlmostEqual(worse["candidate"], 0.3)
        with self.assertRaises(AssertionError):
            check_held_reference(self, worse)

    def test_disabled_regularization_does_not_write_dummy_weights(self):
        """Require no row-weight writes when admission keeps only the one-value dummy."""
        self.assertNotIn("r.row_w.data", parallel_world_rows._source(True))


@unittest.skipUnless(wp.is_cuda_available(), "Complete parallel-world controls require CUDA")
class TestParallelWorldCUDA(unittest.TestCase):
    device = "cuda:0"
    split = False
    environment = ENV
    producer_key = KEY

    def check_owner(self, solver):
        """Require the selected producer and, for split mode, the retained solver object."""
        sparse = solver._sparse_factor
        self.assertIsNotNone(sparse.parallel_world)
        owner = sparse.parallel_world
        self.assertEqual(owner.split, self.split)
        self.assertEqual(owner.kernel.key, self.producer_key)
        if self.split:
            self.assertIs(owner.solve_kernel, sparse.kernels.solve)
            self.assertEqual(owner.solve_kernel.key, SOLVE_KEY)
        return owner

    def test_empty_rows_publish_predictor_through_final_state(self):
        """Exercise current and held no-contact steps without a global predictor consumer."""
        with patch.dict(os.environ, self.environment):
            model, contacts = kinetic.multiworld_fixture(self.device)
            model.joint_limit_lower.fill_(-np.inf)
            model.joint_limit_upper.fill_(np.inf)
            with patch.dict(os.environ, {"FEATHER_PGS_PARALLEL_WORLD": "0"}):
                original = kinetic.make_solver(model, False)
            candidate = kinetic.make_solver(model, True)
        self.check_owner(candidate)
        contacts.rigid_contact_count.zero_()
        solvers = (original, candidate)
        states = [[model.state(), model.state()] for _ in solvers]
        controls = [model.control() for _ in solvers]
        for pair, control in zip(states, controls, strict=True):
            kinetic.rotated_input(model, pair[0])
            control.joint_f.assign(np.linspace(-0.2, 0.3, model.joint_dof_count, dtype=np.float32))
        for source, destination in ((0, 1), (1, 0)):
            candidate.v_hat.fill_(np.nan)
            candidate.v_out.fill_(np.nan)
            candidate.impulses.fill_(np.nan)
            for solver, pair, control in zip(solvers, states, controls, strict=True):
                solver.step(pair[source], pair[destination], control, contacts, kinetic.DT)
                solver.check_constraint_capacity()
                np.testing.assert_array_equal(solver.constraint_count.numpy(), np.zeros(model.world_count, np.int32))
                kinetic.public_state(self, model, pair[destination])
            expected = original.v_hat.numpy()
            actual = candidate.v_out.numpy()
            self.assertTrue(np.isfinite(actual).all())
            self.assertLess(np.linalg.norm(actual - expected) / (1 + np.linalg.norm(expected)), 7e-4)
            if self.split:
                np.testing.assert_allclose(candidate.v_hat.numpy(), actual, rtol=3e-5, atol=3e-6)
            for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
                expected = getattr(states[0][destination], name).numpy()
                actual = getattr(states[1][destination], name).numpy()
                self.assertLess(np.linalg.norm(actual - expected) / (1 + np.linalg.norm(expected)), 7e-4, name)
        np.testing.assert_array_equal(candidate.mass_update_mask.numpy(), np.zeros(model.world_count, np.int32))

    def test_saved_current_held_rows_and_physical_law(self):
        """Cover all sixteen saved geometries, original raw IDs and held physical operators."""
        replay, records = saved_records()
        tested = 0
        for gpu, record, data in records:
            for index, world in enumerate(data["worlds"]):
                with (
                    self.subTest(gpu=gpu, step=record["step"], world=int(world)),
                    patch.dict(os.environ, self.environment),
                ):
                    with patch.dict(os.environ, {"FEATHER_PGS_PARALLEL_WORLD": "0"}):
                        original = replay.bind_world(record, data, index, self.device)
                    previous, model, state = (original[name] for name in ("solver", "model", "state"))
                    solver = kinetic.make_solver(model, True)
                    sparse, native = solver._sparse_factor, solver._g1_kinetic_state
                    self.check_owner(solver)
                    for name, value in record["settings"].items():
                        setattr(solver, name, value)
                    for name in ("shape_material_mu", "shape_material_restitution"):
                        getattr(solver, name).assign(getattr(previous, name).numpy())
                    output, control = model.state(), model.control()
                    augmented = solver._prepare_augmented_state(state, output, control)
                    captured_pose = state.body_q.numpy().copy()
                    native.begin(state, augmented, state.joint_qd, 0.0025, False)
                    chart_difference = {
                        "screw_abs": float(np.max(np.abs(solver.joint_S_s.numpy() - previous.joint_S_s.numpy()))),
                        "origin_abs": float(
                            np.max(np.abs(solver.articulation_origin.numpy() - previous.articulation_origin.numpy()))
                        ),
                        "pose_abs": float(np.max(np.abs(state.body_q.numpy() - captured_pose))),
                    }
                    # begin initializes the predictor cache AND recomputes the
                    # shared public poses. This saved-row fixture must not mix
                    # those rounded poses/S with the original captured chart.
                    # Complete-step tests separately qualify the recompute path.
                    state.body_q.assign(captured_pose)
                    solver.joint_S_s.assign(previous.joint_S_s.numpy())
                    solver.articulation_origin.assign(previous.articulation_origin.numpy())
                    np.testing.assert_array_equal(state.body_q.numpy(), captured_pose)
                    np.testing.assert_array_equal(solver.joint_S_s.numpy(), previous.joint_S_s.numpy())
                    np.testing.assert_array_equal(
                        solver.articulation_origin.numpy(), previous.articulation_origin.numpy()
                    )
                    case = {**original, "solver": solver, "owner": sparse, "host": sparse.host}
                    bind_factor(original)
                    bind_factor(case)
                    held = sparse.data.W.numpy().copy()
                    # Retain the qualified force/predictor as the independent
                    # control for this held operator; no free supplied vhat.
                    augmented.joint_tau.assign(previous.joint_tau.numpy())
                    control.joint_f.assign(np.linspace(-0.02, 0.03, 43, dtype=np.float32))
                    native.predict(state, augmented, control, state.joint_qd, 0.0025)
                    expected_vhat = solver.v_hat.numpy().copy()
                    previous.v_hat.assign(expected_vhat)
                    build(original)
                    old_v, old_lam, _ = metric.check_native(self, original)
                    expected_rows, expected_score, _ = physical_score(
                        self, original, original["H"], old_v, old_lam, expected_vhat
                    )
                    owner = sparse.parallel_world
                    self.assertTrue(owner.prepare(state, augmented, control, state.joint_qd, case["contacts"], 0.0025))
                    solver.v_hat.fill_(np.nan)
                    solver.v_out.fill_(np.nan)
                    solver.impulses.fill_(np.nan)
                    if self.split:
                        sparse.data.Z.fill_(np.nan)
                        sparse.data.incident.fill_(np.nan)
                    owner.build_rows()
                    owner.solve(8, 1.0, 0)
                    solver.check_constraint_capacity()
                    count = int(solver.constraint_count.numpy()[0])
                    actual_v, actual_lam = solver.v_out.numpy(), solver.impulses.numpy()[0, :count]
                    # Keep original captured S/origin on the physical side;
                    # candidate row construction must not be its own oracle.
                    physical_solver = SimpleNamespace(**vars(solver))
                    physical_solver.joint_S_s = previous.joint_S_s
                    physical_solver.articulation_origin = previous.articulation_origin
                    if self.split:
                        np.testing.assert_allclose(solver.v_hat.numpy(), expected_vhat, rtol=3e-5, atol=3e-6)
                        # FP64 coefficientwise cancellation also fails for the
                        # retained producer on these captured charts. Keep its
                        # structural checks here and compare native publication
                        # by raw ID below, after independent physical checks.
                        check_rows(self, {**case, "solver": physical_solver}, coefficients=False)
                    actual_rows, score, defect = physical_score(
                        self,
                        case,
                        case["H"],
                        actual_v,
                        actual_lam,
                        expected_vhat,
                        physical_solver=physical_solver,
                    )
                    original_reference = physical_reference(
                        physical_rows(original), original["H"], expected_rows, expected_vhat
                    )
                    candidate_reference = physical_reference(
                        physical_rows({**case, "solver": physical_solver}), case["H"], actual_rows, expected_vhat
                    )
                    reference_errors = held_reference_errors(
                        case["H"],
                        expected_vhat,
                        old_v,
                        actual_v,
                        original_reference["velocity"],
                        candidate_reference["velocity"],
                    )
                    np.testing.assert_array_equal(sparse.data.W.numpy(), held)
                    print(
                        "parallel_world_saved "
                        + json.dumps(
                            {
                                "gpu": gpu,
                                "step": record["step"],
                                "world": int(world),
                                "split": self.split,
                                "original": expected_score,
                                "candidate": score,
                                "momentum": defect,
                                "recomputed_vs_captured_chart": chart_difference,
                                "reference_errors": reference_errors,
                                "reference": {
                                    arm: {name: reference[name] for name in ("score", "eight_score", "sweep_bound")}
                                    for arm, reference in (
                                        ("original", original_reference),
                                        ("candidate", candidate_reference),
                                    )
                                },
                            }
                        ),
                        flush=True,
                    )
                    tested += 1
                    order = check_row_identity(
                        self, expected_rows, actual_rows, int(case["contacts"].rigid_contact_count.numpy()[0])
                    )
                    # A legal row permutation need not improve each residual
                    # component at eight sweeps. Require its own original-law
                    # trajectory and nonregression to a converged physical
                    # reference instead; no native allowance is increased.
                    for velocity, scores, reference in (
                        (old_v, expected_score, original_reference),
                        (actual_v, score, candidate_reference),
                    ):
                        self.assertLessEqual(max(reference["score"].values()), 1e-8, "reference did not converge")
                        check_eight_reproduction(self, case["H"], expected_vhat, velocity, scores, reference)
                    check_held_reference(self, reference_errors)
                    if self.split:
                        for name in ("Z", "incident"):
                            expected = getattr(original["owner"].data, name).numpy()[0, :count][order]
                            actual = getattr(sparse.data, name).numpy()[0, :count]
                            np.testing.assert_allclose(
                                actual, expected, rtol=3e-5, atol=3e-6, err_msg=f"original raw-ID matched {name}"
                            )
        self.assertEqual(tested, 16)

    def test_complete_steps_reset_notification_and_graph(self):
        """Reuse complete five-world current/held/reset/notification and two-bank graph controls."""
        constructors, launched = [], []
        original_factory, original_launch = kinetic.make_solver, wp.launch_tiled

        def construct(model, enabled, **overrides):
            with patch.dict(os.environ, {"FEATHER_PGS_PARALLEL_WORLD": str(int(enabled))}):
                solver = original_factory(model, enabled, **overrides)
            constructors.append(solver)
            return solver

        def launch(*args, **kwargs):
            kernel = kwargs.get("kernel", args[0] if args else None)
            launched.append(kernel.key)
            return original_launch(*args, **kwargs)

        with (
            patch.dict(os.environ, self.environment),
            patch.object(kinetic, "make_solver", construct),
            patch.object(wp, "launch_tiled", launch),
        ):
            kinetic.TestG1KineticStateCUDA.test_actual_steps_masked_refresh_reset_and_graph(self)
        self.assertIsNone(constructors[0]._sparse_factor.parallel_world)
        self.check_owner(constructors[1])
        # Four eager lifecycle calls, a replacement input bank, and both
        # captured calls must retain the complete owner after invalidation.
        self.assertGreaterEqual(launched.count(self.producer_key), 7)
        if self.split:
            self.assertNotIn(KEY, launched)
            for index, key in enumerate(launched):
                if key == PRODUCER_KEY:
                    self.assertEqual(launched[index + 1], SOLVE_KEY, "producer must feed the original solver directly")

    def test_overflow_is_visible_and_sticky(self):
        """Report complete limit-plus-contact demand above100 without silently dropping it."""
        with patch.dict(os.environ, self.environment):
            model, contacts = kinetic.multiworld_fixture(self.device)
            solver = kinetic.make_solver(model, True)
        self.check_owner(solver)
        self.assertFalse(solver._regularization_enabled)
        self.assertEqual(solver.row_w.shape, (1, 1))
        solver.row_w.fill_(-13.0)
        state, output, control = model.state(), model.state(), model.control()
        kinetic.rotated_input(model, state)
        indices, q = solver._joint_limit_q_index.numpy(), state.joint_q.numpy()
        active = indices >= 0
        lower = np.full(model.joint_dof_count, -np.inf, np.float32)
        upper = -lower
        lower[active], upper[active] = q[indices[active]] - 0.001, q[indices[active]] + 0.001
        model.joint_limit_lower.assign(lower)
        model.joint_limit_upper.assign(upper)
        # Existing capacity16 is enough:74 real limits plus16 complete
        # triplets demand122 rows in world0, without changing the100-row cap.
        for name in ("shape0", "shape1", "point0", "point1", "normal", "margin0", "margin1"):
            array = getattr(contacts, "rigid_contact_" + name)
            values = array.numpy()
            values[:] = values[0]
            array.assign(values)
        contacts.rigid_contact_count.fill_(16)
        solver.step(state, output, control, contacts, kinetic.DT)
        np.testing.assert_array_equal(solver.row_w.numpy(), [[-13.0]])
        self.assertTrue(solver.constraint_capacity_status()["dense"])
        with self.assertRaises(RuntimeError):
            solver.check_constraint_capacity()
        mask = wp.array([True, False, False, False, False], dtype=wp.bool, device=self.device)
        solver.reset(state, mask)
        self.assertTrue(solver.constraint_capacity_status()["dense"])
        with self.assertRaises(RuntimeError):
            solver.check_constraint_capacity()
        # Malformed raw count is a separate public contact-capacity fault,
        # even though the exact bucket guard also rejects it privately.
        self.assertFalse(solver.constraint_capacity_status()["contacts"])
        contacts.rigid_contact_count.fill_(contacts.rigid_contact_max + 1)
        solver.step(state, output, control, contacts, kinetic.DT)
        self.assertTrue(solver.constraint_capacity_status()["contacts"])
        with self.assertRaises(RuntimeError):
            solver.check_constraint_capacity()
        solver.reset(state, mask)
        self.assertTrue(solver.constraint_capacity_status()["contacts"])
        with self.assertRaises(RuntimeError):
            solver.check_constraint_capacity()


class TestParallelWorldSplitCUDA(TestParallelWorldCUDA):
    """Reuse the same complete physical and lifecycle controls for the split owner."""

    split = True
    environment: ClassVar = {**ENV, "FEATHER_PGS_PARALLEL_WORLD_SPLIT": "1"}
    producer_key = PRODUCER_KEY


if __name__ == "__main__":
    unittest.main()
