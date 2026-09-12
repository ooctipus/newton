# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Portable controls and one pinned-runtime CPU constructor integration test."""

import contextlib
import importlib
import inspect
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from typing import ClassVar
from unittest.mock import patch

import allegro_capacity as capacity
import checked_capture as checked
import compare_backends as driver


class TestCompactArguments(unittest.TestCase):
    argv: ClassVar[list[str]] = [
        "--isaaclab",
        "/tmp/lab",
        "--fpgs",
        "/tmp/fpgs",
        "--mjwarp",
        "/tmp/mj",
        "--output-dir",
        "/tmp/output",
        "--task",
        "allegro",
        "--allegro-compact-capacity",
    ]

    def test_explicit_option(self):
        """Require explicit selection of the narrowly scoped storage recipe."""
        args = driver.parse_args(
            [
                "--isaaclab",
                "/tmp/lab",
                "--fpgs",
                "/tmp/fpgs",
                "--mjwarp",
                "/tmp/mj",
                "--output-dir",
                "/tmp/output",
                "--task",
                "allegro",
                "--allegro-compact-capacity",
            ]
        )
        self.assertTrue(args.allegro_compact_capacity)

    def test_wrong_task_scale_unchecked_and_capacity_conflicts(self):
        """Reject unsupported workloads and conflicting capacity arguments."""
        for extra in (
            ["--task", "franka"],
            ["--num-envs", "512"],
            ["--allow-unchecked"],
            ["--capacity", "allegro:fpgs:rigid_contact_max=286720"],
        ):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                driver.parse_args(self.argv + extra)

    def test_wrong_source_rejected(self):
        """Refuse missing or different runtime sources before construction."""
        with self.assertRaises((RuntimeError, FileNotFoundError)):
            capacity.verify_sources(Path(__file__).parent)

    def test_actual_recipe_forwarding_fpgs_only(self):
        """Use the source-pinned Lab recipe, without loading or running simulation."""
        lab = Path("/home/octi/Projects/IsaacLab.wt/fpgs-opt-20260910")
        capture = driver.load_capture(lab)
        args = driver.parse_args(self.argv)
        flags = {0: {"NEWTON_NARROW_PHASE_COHERENT_CONVEX": "reject_only"}}
        pins = {str(capture.HARNESS / "run_profiled.py"): "pin"}
        with tempfile.TemporaryDirectory() as directory:
            for backend in ("fpgs", "mjwarp"):
                run = driver.make_batch(
                    capture,
                    args,
                    {backend: Path("/tmp/source")},
                    flags,
                    [{"index": 0, "uuid": "GPU-test"}],
                    Path(directory),
                    0,
                    "allegro",
                    backend,
                    drivers=pins,
                )[0]
                self.assertEqual(run["command"][4], "Isaac-Reorient-Cube-Allegro")
                self.assertEqual("--allegro-compact-capacity" in run["command"], backend == "fpgs")
                self.assertFalse(any("collision_cfg" in item for item in run["command"]))
                self.assertEqual(run.get("broad_phase_output_max"), 524288 if backend == "fpgs" else None)

    def test_parent_requires_both_constructor_and_boundary_evidence(self):
        """An absent or partial compact record cannot produce an accepted ratio."""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            wrapper = Path(driver.__file__).with_name("checked_capture.py")
            helper = wrapper.with_name("allegro_capacity.py")
            pins = driver.file_hashes([wrapper, helper])
            run = {
                "output_dir": directory,
                "backend": "feather_pgs",
                "mjwarp_linesearch_fix": False,
                "allegro_compact_capacity": True,
                "environment": {"FPGS_BENCH_RUN_SHA256": "harness"},
            }
            report = {
                "complete": True,
                "check_pass": True,
                "boundary_count": 2,
                "boundaries": [
                    {
                        "boundary": i,
                        "physics": "feather_pgs",
                        "check_pass": True,
                        "collision": {"check_pass": True, "status": "checked"},
                    }
                    for i in range(2)
                ],
                "mjwarp_linesearch_fix": False,
                "physics_work_modified": False,
                "physics_budgets_modified": False,
                "checks_output": str(output / "capture_checks.json"),
                "capture_output": str(output / "capture.json"),
                "run_profiled_sha256": "harness",
                "checked_capture_sha256": pins[str(wrapper)],
                "allegro_compact_capacity": True,
            }
            for info in (
                {},
                {
                    "helper_sha256": pins[str(helper)],
                    "constructors": [{"calls": 1, "complete": True, "restored": True}] * 2,
                },
            ):
                report["allegro_capacity"] = info
                (output / "capture_checks.json").write_text(json.dumps(report))
                with self.assertRaises(RuntimeError):
                    driver.check_overflow_result(run, pins)

    def test_sparse_worker_multiplier_is_part_of_snapshot_contract(self):
        """Do not mistake constructor worker counts for the final multiplied grid."""
        source = inspect.getsource(capacity.snapshot)
        self.assertIn('"NEWTON_NARROW_PHASE_THREADS_X") != "4"', source)
        with self.assertRaises(RuntimeError):
            capacity.validate_snapshot({"workers": 96256})


class TestConstructorOwnership(unittest.TestCase):
    def setUp(self):
        """Provide small constructor owners without importing Warp or Isaac Lab."""
        self.manager = NS(_collision_cfg=None)
        self.calls = []
        calls = self.calls

        class Solver:
            def __init__(self, model, *, sentinel=None):
                calls.append(("solver", model.rigid_contact_max, sentinel))
                self._max_contacts_alloc = model.rigid_contact_max

        class Narrow:
            def __init__(self, max_candidate_pairs, *, sparse_gjk_pairs=None, fail=False):
                calls.append(("narrow", sparse_gjk_pairs, fail))
                if fail:
                    raise ValueError("original failure")
                self.max_candidate_pairs, self.sparse_gjk_pairs = max_candidate_pairs, sparse_gjk_pairs

        self.solver, self.narrow = Solver, Narrow
        self.originals = Solver.__init__, Narrow.__init__
        for name in ("verify_sources", "require_constructor"):
            mocked = patch.object(capacity, name)
            mocked.start()
            self.addCleanup(mocked.stop)

    def scope(self, report):
        """Apply the real ownership wrapper to the mock constructor pair."""
        return capacity.scope(self.solver, self.narrow, Path("/tmp"), report, lambda: self.manager)

    def test_original_called_once_and_restored(self):
        """Preserve call arguments, signatures and the default-None configuration."""
        report, model, token = {}, NS(rigid_contact_max=9), object()
        with self.scope(report):
            self.assertEqual(inspect.signature(self.solver.__init__), inspect.signature(self.originals[0]))
            self.solver(model, sentinel=token)
            self.narrow(capacity.BROAD)
            with self.assertRaises(RuntimeError):
                self.solver(model)
        self.assertEqual(self.calls, [("solver", capacity.PUBLIC, token), ("narrow", True, False)])
        self.assertEqual((self.solver.__init__, self.narrow.__init__), self.originals)
        self.assertTrue(all(item["restored"] for item in report["constructors"]))
        self.assertIsNone(self.manager._collision_cfg)

    def test_conflict_and_exception_restore(self):
        """Reject conflicting requests and propagate original constructor errors."""
        for kwargs, error in (({"sparse_gjk_pairs": False}, RuntimeError), ({"fail": True}, ValueError)):
            with self.assertRaises(error), self.scope({}):
                self.narrow(capacity.BROAD, **kwargs)
            self.assertEqual((self.solver.__init__, self.narrow.__init__), self.originals)
        self.manager._collision_cfg = object()
        with self.assertRaises(RuntimeError), self.scope({}):
            self.solver(NS(rigid_contact_max=9))

    def test_foreign_owner_retained_other_wrapper_restored(self):
        """Restore only owned callables and retain an unexpected foreign owner."""

        def foreign(*args):
            return None

        with self.assertRaisesRegex(RuntimeError, "Foreign"), self.scope({}):
            self.narrow.__init__ = foreign
        self.assertIs(self.narrow.__init__, foreign)
        self.assertIs(self.solver.__init__, self.originals[0])


class TestActualConstructor(unittest.TestCase):
    def test_original_cpu_constructor_default_none_and_sparse_specialization(self):
        """Run against the pinned fba checkout; never initialize a CUDA device."""
        np = importlib.import_module("numpy")
        wp = importlib.import_module("warp")
        newton = importlib.import_module("newton")
        NarrowPhase = importlib.import_module("newton.geometry").NarrowPhase
        SolverFeatherPGS = importlib.import_module("newton.solvers").SolverFeatherPGS

        root = Path(newton.__file__).resolve().parents[1]
        capacity.verify_sources(root)
        builder = newton.ModelBuilder()
        for index in range(6):
            body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
            joint = builder.add_joint_prismatic(parent=-1, child=body)
            builder.add_articulation([joint])
            builder.add_shape_box(body, hx=0.01, hy=0.01, hz=0.01, xform=wp.transform((index, 0, 0)))
        model = builder.finalize(device="cpu")
        self.assertFalse(newton.CollisionPipeline(model, broad_phase_output_max=8).narrow_phase.sparse_gjk_pairs)
        manager, report = NS(_collision_cfg=None), {}
        original = newton.CollisionPipeline.__init__
        with patch.object(capacity, "BROAD", 8):
            with capacity.scope(SolverFeatherPGS, NarrowPhase, root, report, lambda: manager):
                try:
                    checked.install_broad_phase_limit(newton.CollisionPipeline, 8, {}, lambda: None)
                    solver = SolverFeatherPGS(model, pgs_mode="split")
                    pipeline = newton.CollisionPipeline(model, broad_phase="explicit")
                    self.assertTrue(pipeline.narrow_phase.sparse_gjk_pairs)
                    self.assertEqual(pipeline.broad_phase_shape_pairs.shape[0], 8)
                    self.assertEqual(pipeline.contacts().rigid_contact_max, capacity.PUBLIC)
                    self.assertEqual(solver._max_contacts_alloc, capacity.PUBLIC)
                    self.assertIsNone(manager._collision_cfg)
                finally:
                    newton.CollisionPipeline.__init__ = original
        self.assertTrue(
            all(item["calls"] == 1 and item["complete"] and item["restored"] for item in report["constructors"])
        )


if __name__ == "__main__":
    unittest.main()
