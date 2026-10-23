# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test the execution qualifier without simulator imports or GPU execution."""

import builtins
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import qualify_recipe as qualify


class Array:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float).copy()

    def numpy(self):
        return self.values.copy()

    def assign(self, other):
        self.values[:] = other.values


class State:
    def __init__(self):
        self.joint_q, self.joint_qd = Array([0.1, 0.2]), Array([0.3, 0.4])
        self.body_q = Array([[0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1]])
        self.body_qd, self.body_f = Array(np.zeros((2, 6))), Array(np.zeros((2, 6)))

    def assign(self, other):
        for name, value in vars(self).items():
            value.assign(getattr(other, name))


class TestErrors(unittest.TestCase):
    def test_allow_small_roundoff_but_reject_physical_errors_and_nonfinite(self):
        """Use explicit tolerances rather than bit identity or finite-only gates."""
        self.assertTrue(qualify.errors([1.0], [1.00001], atol=1e-4)["pass"])
        self.assertFalse(qualify.errors([1.0], [1.1], atol=1e-4)["pass"])
        self.assertFalse(qualify.errors([1.0], [np.nan], atol=1e-4)["pass"])
        self.assertFalse(qualify.errors([1.0], [np.inf], atol=1e-4)["pass"])
        self.assertFalse(qualify.errors([1.0], [[1.0]], atol=1e-4)["pass"])

    def test_compare_published_pose_without_world_offset_scaled_tolerance(self):
        """Accept quaternion sign but reject stale body positions and velocities."""
        left, right = State(), State()
        right.body_q.values[:, 3:] *= -1
        self.assertTrue(all(item["pass"] for item in qualify.compare_states(left, right).values()))
        left.body_q.values[:, 0] = 10000
        right.body_q.values[:, 0] = 10000.01
        self.assertFalse(qualify.compare_states(left, right)["body_position"]["pass"])
        right.body_qd.values[:] = 0.1
        self.assertFalse(qualify.compare_states(left, right)["body_qd"]["pass"])

    def test_point_force_budget_is_rotation_invariant(self):
        """A small transverse force delta uses the full reference vector scale."""
        left, right = np.array([[100.0, 0, 0]]), np.array([[100.0, 0.03, 0]])
        rotation = np.array([[1, -1, 0], [1, 1, 0], [0, 0, np.sqrt(2)]]) / np.sqrt(2)
        results = [
            qualify.contact_diagnostics(a, b, [0], [1], np.array([0, -1]), [1000], 0.0025)
            for a, b in ((left, right), (left @ rotation, right @ rotation))
        ]
        self.assertFalse(qualify.errors(left, right, atol=0.01, rtol=0.001)["pass"])
        self.assertTrue(all(result["physical_contact_gate_pass"] for result in results))
        self.assertAlmostEqual(
            results[0]["point_vector"]["max_budget_fraction"], results[1]["point_vector"]["max_budget_fraction"]
        )
        self.assertAlmostEqual(
            results[0]["world_net_impulse"]["normalized_error"][0],
            results[1]["world_net_impulse"]["normalized_error"][0],
        )

    def test_world_impulse_routing_and_undefined_weight(self):
        """Cancel internal pairs, route cross-world forces, and reject undefined scales."""
        args = (np.zeros((3, 3)), np.diag([1, 2, 3]), [0, 0, 2], [1, 2, 3], np.array([0, 0, 1, -1]))
        world = qualify.contact_diagnostics(*args, [100, 200], 0.1)["world_net_impulse"]
        np.testing.assert_allclose(world["impulse_error_Ns"], [[0, 0.2, 0], [0, -0.2, 0.3]])
        undefined = qualify.contact_diagnostics(*args, [100, 0], 0.1)["world_net_impulse"]
        self.assertFalse(undefined["pass"])
        self.assertEqual(undefined["undefined_zero_weight_worlds"], [1])
        self.assertIsNone(undefined["normalized_error"][1])
        json.dumps(undefined, allow_nan=False)

    def test_replay_preserves_budgets_contacts_publication_and_reset(self):
        """Exercise the complete replay loop using independent inert solvers."""
        created, calls = [], []
        force_delta = 0.0

        class Solver:
            def __init__(self, model, **kwargs):
                self.kwargs = kwargs
                self.grouped_dynamics = kwargs.get("grouped_dynamics", False)
                self.lazy_kinematics = kwargs.get("lazy_kinematics", False)
                self._grouped_topology = object() if self.grouped_dynamics else None
                self.pgs_mode = kwargs["pgs_mode"]
                self.resets = 0
                created.append(self)

            def step(self, source, target, control, contacts, dt):
                calls.append((id(source), id(control), id(contacts), dt))
                target.assign(source)
                target.joint_q.values += dt * source.joint_qd.values
                target.body_q.values[:] = -999  # Must be repaired by explicit publication.

            def publish_kinematics(self, state):
                state.body_q.values[:] = 0
                state.body_q.values[:, 0] = state.joint_q.values
                state.body_q.values[:, 6] = 1

            def reset(self, state):
                self.resets += 1

            def update_contacts(self, contacts):
                contacts.rigid_contact_force.values[:] = [100, 0, 0]
                if self.grouped_dynamics:
                    contacts.rigid_contact_force.values[:, 1] += force_delta

        model = SimpleNamespace(
            state=State,
            world_count=1,
            body_world=Array([0, -1]),
            body_flags=Array([1, 1]),
            body_mass=Array([100, 0]),
            gravity=Array([[0, 0, -9.81], [0, 0, -9.81]]),
            shape_body=Array([0, -1]),
        )
        contacts = SimpleNamespace(
            rigid_contact_count=Array([3]),
            rigid_contact_force=Array(np.zeros((4, 3))),
            rigid_contact_shape0=Array([0, 0, 0, 0]),
            rigid_contact_shape1=Array([1, 1, 1, 1]),
        )
        pipeline = SimpleNamespace(contacts=lambda: contacts, collide=lambda *args: None)
        kwargs = {
            "pgs_mode": "matrix_free",
            "pgs_iterations": 8,
            "contact_friction_gap_threshold": float("inf"),
            "test_negative_sentinel": float("-inf"),
        }
        manager = SimpleNamespace(
            _model=model,
            _collision_pipeline=pipeline,
            _solver_dt=0.004,
            _num_substeps=2,
            _collision_decimation=1,
            _state_0=State(),
            _control=object(),
            _filter_solver_kwargs=lambda *args: kwargs,
            _solver=Solver(model, **kwargs),
        )
        modules = {
            "warp": SimpleNamespace(copy=lambda target, source: target.assign(source)),
            "newton": SimpleNamespace(BodyFlags=SimpleNamespace(DYNAMIC=1)),
            "newton.solvers": SimpleNamespace(SolverFeatherPGS=Solver),
        }
        report = {}
        with (
            patch.dict(sys.modules, modules),
            patch.object(qualify.checked, "check_solver") as check_solver,
            patch.object(qualify.checked, "check_collision") as check_collision,
        ):
            qualify.qualify(
                manager, object(), report, lambda: json.dumps(report, allow_nan=False), steps=6, require_contacts=True
            )
        self.assertTrue(report["qualified"])
        self.assertTrue(report["loaded_contacts_observed"])
        self.assertEqual(created[1].kwargs, kwargs)
        self.assertEqual(created[2].kwargs, {**kwargs, "grouped_dynamics": True, "lazy_kinematics": True})
        for branch in report["solver_kwargs"].values():
            self.assertEqual(branch["contact_friction_gap_threshold"], "inf")
            self.assertEqual(branch["test_negative_sentinel"], "-inf")
        self.assertEqual([solver.resets for solver in created[1:]], [1, 1])
        self.assertEqual({call[1] for call in calls}, {id(manager._control)})
        self.assertEqual({call[2] for call in calls}, {id(contacts)})
        self.assertEqual({call[3] for call in calls}, {0.004})
        self.assertEqual(len(calls), 48)
        self.assertEqual(check_solver.call_count, 24)
        self.assertEqual(check_collision.call_count, 12)

        # An explicit mode candidate must not also enable grouped/lazy execution.
        for reference_mode, candidate_mode, grouped, lazy in (
            ("split", "matrix_free", False, False),
            ("matrix_free", "split", True, False),
        ):
            with (
                self.subTest(candidate_mode=candidate_mode),
                patch.dict(kwargs, pgs_mode=reference_mode, grouped_dynamics=grouped, lazy_kinematics=lazy),
            ):
                report = {"qualified": False}
                with (
                    patch.dict(sys.modules, modules),
                    patch.object(qualify.checked, "check_solver") as check_solver,
                    patch.object(qualify.checked, "check_collision") as check_collision,
                ):
                    qualify.qualify(
                        manager,
                        object(),
                        report,
                        lambda report=report: json.dumps(report, allow_nan=False),
                        steps=6,
                        require_contacts=True,
                        candidate_pgs_mode=candidate_mode,
                    )
                self.assertTrue(report["qualified"])
                self.assertEqual(created[-2].kwargs, kwargs)
                self.assertEqual(created[-1].kwargs, {**kwargs, "pgs_mode": candidate_mode})
                self.assertEqual(
                    report["recipe_changes"],
                    {"pgs_mode": {"reference": reference_mode, "candidate": candidate_mode}},
                )
                self.assertEqual([entry["pgs_mode"] for entry in report["paths"]], [reference_mode, candidate_mode])
                self.assertEqual([entry["grouped_dynamics"] for entry in report["paths"]], [grouped, grouped])
                self.assertEqual([entry["lazy_kinematics"] for entry in report["paths"]], [lazy, lazy])
                self.assertEqual(check_solver.call_count, 24)
                self.assertEqual(check_collision.call_count, 12)
                self.assertEqual([solver.resets for solver in created[-2:]], [1, 1])

        # A strict component mismatch must not truncate the diagnostic replay.
        force_delta = 0.03
        report = {"qualified": False}
        with (
            patch.dict(sys.modules, modules),
            patch.object(qualify.checked, "check_solver"),
            patch.object(qualify.checked, "check_collision"),
            self.assertRaisesRegex(RuntimeError, "after complete loaded/reset replay"),
        ):
            qualify.qualify(manager, object(), report, lambda: None, steps=6, require_contacts=True)
        self.assertEqual(len(report["observations"]), 12)
        self.assertEqual(
            [(item["phase"], item["step"]) for item in report["observations"]],
            [(phase, step) for phase in ("loaded", "reset_replay") for step in range(6)],
        )
        self.assertTrue(all(value["pass"] for case in report["reset_replay_errors"] for value in case.values()))
        self.assertEqual([solver.resets for solver in created[-2:]], [1, 1])
        self.assertFalse(report["qualified"])
        self.assertFalse(report["strict_contact_gate_pass"])
        self.assertTrue(report["physical_contact_gate_pass"])

    def test_first_boundary_unwinds_context_without_running_timing(self):
        """Stop at the first warmup boundary and close the original environment."""
        source = """from builtins import _qualification_test_events as events
class Closing:
    def __enter__(self): events.append("opened")
    def __exit__(self, *args): events.append("closed")
def main():
    with Closing():
        _model_meta("feather_pgs")
        events.append("timing_started")
"""
        with tempfile.TemporaryDirectory(prefix="recipe-qualify-") as temporary:
            root = Path(temporary)
            lab, newton, output = root / "lab", root / "newton", root / "result.json"
            harness = lab / qualify.checked.HARNESS
            harness.parent.mkdir(parents=True)
            harness.write_text(source)
            modules = {name: SimpleNamespace() for name in qualify.checked.REQUIRED_PACKAGES}
            modules["newton"].CollisionPipeline = object
            modules["isaaclab.physics"] = SimpleNamespace(
                PhysicsManager=SimpleNamespace(_cfg=SimpleNamespace(solver_cfg=None))
            )
            modules["isaaclab_newton.physics"] = SimpleNamespace(NewtonManager=object())
            events = []
            with (
                patch.dict(sys.modules, modules),
                patch.object(builtins, "_qualification_test_events", events, create=True),
                patch.object(qualify.checked, "select_imports", return_value={}),
                patch.object(qualify.checked, "import_provenance", return_value={}),
                patch.object(qualify.checked, "install_collision_storage_overrides"),
                patch.object(qualify.subprocess, "check_output", return_value="abc\n"),
                patch.object(
                    qualify,
                    "qualify",
                    side_effect=lambda manager, cfg, report, save, **kwargs: report.update(qualified=True),
                ) as replay,
            ):
                result = qualify.main(
                    [
                        "--isaaclab",
                        str(lab),
                        "--newton",
                        str(newton),
                        "--task",
                        "Isaac-Ant",
                        "--expected-run-sha256",
                        qualify.checked.sha(harness),
                        "--output",
                        str(output),
                        "--candidate-pgs-mode",
                        "matrix_free",
                    ]
                )
            self.assertEqual(result, 0)
            self.assertEqual(events, ["opened", "closed"])
            self.assertTrue(json.loads(output.read_text())["qualified"])
            self.assertFalse(json.loads(output.read_text())["timing_valid"])
            self.assertEqual(replay.call_args.kwargs["candidate_pgs_mode"], "matrix_free")
            self.assertEqual(json.loads(output.read_text())["candidate_pgs_mode"], "matrix_free")

    def test_reject_invalid_candidate_mode_before_runtime_import(self):
        """Reject unknown solver modes without importing a simulation backend."""
        with patch.object(sys, "stderr", io.StringIO()) as stderr, self.assertRaises(SystemExit) as error:
            qualify.main(
                [
                    "--isaaclab",
                    "/unused/lab",
                    "--newton",
                    "/unused/newton",
                    "--expected-run-sha256",
                    "unused",
                    "--task",
                    "Isaac-Humanoid",
                    "--output",
                    "/unused/output.json",
                    "--candidate-pgs-mode",
                    "dense",
                ]
            )
        self.assertEqual(error.exception.code, 2)
        self.assertIn("invalid choice", stderr.getvalue())
        with self.assertRaisesRegex(ValueError, "Candidate PGS mode"):
            qualify.qualify(None, None, {}, lambda: None, steps=1, require_contacts=False, candidate_pgs_mode="dense")


if __name__ == "__main__":
    unittest.main()
