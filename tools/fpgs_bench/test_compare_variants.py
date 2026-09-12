# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise same-backend variant comparison without importing a GPU runtime."""

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import compare_variants as driver


class TestVariantComparison(unittest.TestCase):
    def setUp(self):
        """Create independent source and output paths for each control."""
        temporary = tempfile.TemporaryDirectory(prefix="variants-unit-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.argv = [
            "--isaaclab",
            str(self.root / "lab"),
            "--baseline",
            str(self.root / "base"),
            "--candidate",
            str(self.root / "candidate"),
            "--task",
            "ant",
            "--output-dir",
            str(self.root / "out"),
        ]

    def test_arguments_and_alternating_order(self):
        """Default to three AB/BA rounds, paired GPUs, checked capture and seed zero."""
        args = driver.parse_args(self.argv)
        self.assertEqual((args.rounds, args.gpus, args.seed), (3, [0, 1], 0))
        self.assertTrue(args.check_overflow)
        self.assertFalse(args.mjwarp_linesearch_fix)
        self.assertEqual(
            list(driver.arm_order(3)),
            [(0, "baseline"), (0, "candidate"), (1, "candidate"), (1, "baseline"), (2, "baseline"), (2, "candidate")],
        )
        for extra in (["--seed", "1"], ["--allow-unchecked"], ["--gpus", "0"], ["--rounds", "0"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                driver.parse_args(self.argv + extra)

    def test_only_fpgs_capacities_and_scoped_variant_flags(self):
        """Reject different-backend storage and implicit solver-budget changes."""
        args = driver.parse_args(
            [
                *self.argv,
                "--capacity",
                "ant:fpgs:dense_max_constraints=704",
                "--candidate-env",
                "FEATHER_PGS_SPARSE_CONTACT_DIRECT=1",
            ]
        )
        self.assertEqual(driver.variant_flags(args, "candidate")[0], {"FEATHER_PGS_SPARSE_CONTACT_DIRECT": "1"})
        self.assertEqual(driver.variant_flags(args, "baseline"), {0: {}, 1: {}})
        for extra in (
            ["--capacity", "ant:mjwarp:njmax=64"],
            ["--capacity", "ant:fpgs:pgs_iterations=1"],
            ["--candidate-env", "PYTHONPATH=wrong"],
        ):
            with self.assertRaises(ValueError):
                driver.parse_args(self.argv + extra)

    def test_real_make_batch_keeps_backend_and_variant_identity_separate(self):
        """Forward identical recipes/capacities while naming both runs as FPGS variants."""
        args = driver.parse_args(
            [
                *self.argv,
                "--capacity",
                "ant:fpgs:dense_max_constraints=704",
                "--capacity",
                "ant:fpgs:broad_phase_output_max=57344",
            ]
        )
        harness = Path(args.isaaclab) / driver.owner.HARNESS_RELATIVE
        harness.mkdir(parents=True)
        capture = SimpleNamespace(HARNESS=harness, COMMON_ENVIRONMENT={}, RECIPES={"ant": ("Isaac-Ant", (), {})})
        devices = [{"index": i, "uuid": f"GPU-{i}"} for i in (0, 1)]
        hashes = {str(harness / "run_profiled.py"): "harness"}
        args.output_dir.mkdir()
        batches = [
            driver.make_variant_batch(capture, args, label, devices, repeat=0, drivers=hashes)
            for label in ("baseline", "candidate")
        ]
        for label, batch in zip(("baseline", "candidate"), batches, strict=True):
            self.assertEqual(len(batch), 2)
            for run in batch:
                self.assertEqual((run["newton"], run["variant"], run["backend"]), (label, label, "feather_pgs"))
                self.assertEqual(run["capture_check_mode"], "checked")
                self.assertEqual(run["environment"]["PYTHONPATH"], str(getattr(args, label)))
                self.assertIn("dense_max_constraints=704", run["command"])
                self.assertIn("--broad-phase-output-max", run["command"])
                self.assertEqual(run["command"][run["command"].index("--seed") + 1], "0")
                self.assertFalse(run["mjwarp_linesearch_fix"])

    def test_physics_and_wall_medians_remain_distinct(self):
        """Reject partial/nonfinite data and never substitute graph gain for wall gain."""
        runs = []
        for repeat in range(1, 4):
            for label in ("baseline", "candidate"):
                for gpu in (0, 1):
                    runs.append(
                        {
                            "round": repeat,
                            "variant": label,
                            "gpu_index": gpu,
                            "result": {
                                "graph_span_us_per_step": 10 if label == "baseline" else 5,
                                "wall_us_per_step": 100 if label == "baseline" else 90,
                            },
                        }
                    )
        results = driver.summarize(runs, [0, 1], 3)
        self.assertEqual(results[0]["physics_baseline_over_candidate"], 2)
        self.assertAlmostEqual(results[0]["wall_baseline_over_candidate"], 100 / 90)
        with self.assertRaises(RuntimeError):
            driver.summarize(runs[:-1], [0, 1], 3)
        runs[0]["result"]["wall_us_per_step"] = float("nan")
        with self.assertRaises(RuntimeError):
            driver.summarize(runs, [0, 1], 3)

    def test_actual_metadata_and_checked_source_hash_are_bound(self):
        """Reject a changed checked-wrapper hash and within-run timestep drift."""
        args = driver.parse_args(self.argv)
        directory = self.root / "capture"
        directory.mkdir()
        wrapper = Path(driver.owner.__file__).with_name("checked_capture.py")
        drivers = driver.owner.file_hashes([wrapper])
        run = {
            "output_dir": str(directory),
            "backend": "feather_pgs",
            "mjwarp_linesearch_fix": False,
            "environment": {"FPGS_BENCH_RUN_SHA256": "source-harness"},
        }
        model = {
            "solver_class": "SolverFeatherPGS",
            "worlds": args.num_envs,
            "num_substeps": 2,
            "solver_dt": 0.0025,
            "pgs_mode": "matrix_free",
            "pgs_iterations": 8,
            "bodies": 17,
            "joints": 17,
            "joint_dofs": 18,
            "joint_coords": 19,
            "articulations": 1,
            "rigid_contact_max": 128,
        }
        data = {
            "task": "Isaac-Ant",
            "physics": "feather_pgs",
            "num_envs": args.num_envs,
            "steps": args.steps,
            "repeats": 1,
            "profile_steps": args.profile_steps,
            "cuda_graph": True,
            "physics_attr": [],
            "solver_attr": [],
            "sim_dt": 0.005,
            "decimation": 4,
            "model": model,
            "model_after": dict(model),
        }
        checks = {
            "complete": True,
            "check_pass": True,
            "boundary_count": 2,
            "boundaries": [
                {
                    "boundary": i,
                    "physics": "feather_pgs",
                    "check_pass": True,
                    "collision": {"check_pass": True, "status": "checked"},
                    "capacities": {
                        "dense_max_constraints": 192,
                        "mf_max_constraints": 32,
                        "propagation_max_constraints": 64,
                    },
                }
                for i in range(2)
            ],
            "mjwarp_linesearch_fix": False,
            "physics_work_modified": False,
            "physics_budgets_modified": False,
            "checks_output": str(directory / "capture_checks.json"),
            "capture_output": str(directory / "capture.json"),
            "run_profiled_sha256": "source-harness",
            "checked_capture_sha256": drivers[str(wrapper)],
        }
        capture = SimpleNamespace(RECIPES={"ant": ("Isaac-Ant", (), {})}, _read_result=lambda _run: {})

        def save():
            for name, value in (("capture.json", data), ("capture_checks.json", checks), ("capture_analysis.json", {})):
                (directory / name).write_text(json.dumps(value))

        save()
        result = driver.read_checked_result(capture, args, run, drivers)
        self.assertEqual(result["model"]["solver_dt"], 0.0025)
        self.assertEqual(len(run["artifacts"]), 3)
        checks["checked_capture_sha256"] = "wrong-source"
        save()
        with self.assertRaisesRegex(RuntimeError, "source-bound overflow"):
            driver.read_checked_result(capture, args, run, drivers)
        checks["checked_capture_sha256"] = drivers[str(wrapper)]
        data["model_after"]["solver_dt"] = 0.005
        save()
        with self.assertRaisesRegex(RuntimeError, "budget/topology"):
            driver.read_checked_result(capture, args, run, drivers)

    def fake_capture(self):
        """Supply only filesystem/process-owner contracts without a simulator import."""
        args = driver.parse_args(self.argv)
        for root in (args.baseline, args.candidate):
            (root / "newton").mkdir(parents=True)
            (root / "newton/__init__.py").write_text("")
        harness = args.isaaclab / driver.owner.HARNESS_RELATIVE
        harness.mkdir(parents=True)
        for name in ("compare_gpus.py", "nsys_run.sh", "run_profiled.py", "analyze_nsys.py"):
            (harness / name).write_text("mock source\n")
        helper = harness / "compare_gpus.py"
        capture = SimpleNamespace(
            __file__=str(helper),
            HARNESS=harness,
            COMMON_ENVIRONMENT={},
            RECIPES={"ant": ("Isaac-Ant", (), {})},
            _loaded_source_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),
            _source=lambda root: {"path": str(root)},
            _gpus=lambda ids: [{"index": i, "uuid": f"GPU-{i}"} for i in ids],
            _require_idle=Mock(),
        )
        return args, capture

    def test_complete_main_runs_paired_ab_ba_and_keeps_quality_unaccepted(self):
        """Use the existing paired owner directly and label both arms honestly."""
        args, capture = self.fake_capture()
        batches = []

        def run(batch, _lab):
            batches.append([(item["variant"], item["gpu_index"]) for item in batch])
            for item in batch:
                item.update(returncode=0, cleanup_signals=[])

        def read(_capture, _args, item, _drivers):
            item["result"] = {
                "graph_span_us_per_step": 10 if item["variant"] == "baseline" else 5,
                "wall_us_per_step": 100,
            }
            return {"same": "actual budgets"}

        with (
            patch.object(driver.owner, "load_capture", return_value=capture),
            patch.object(driver.owner, "tool_checkout", return_value=None),
            patch.object(driver.owner, "runtime_software", return_value={}),
            patch.object(driver.owner, "source_guard"),
            patch.object(driver.owner, "run_batch", side_effect=run),
            patch.object(driver, "read_checked_result", side_effect=read),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(driver.main([*self.argv, "--rounds", "2"]), 0)
        self.assertEqual([batch[0][0] for batch in batches], ["baseline", "candidate", "candidate", "baseline"])
        self.assertTrue(all(len(batch) == 2 for batch in batches))
        report = json.loads((args.output_dir / "manifest.json").read_text())
        self.assertEqual(report["status"], "complete")
        self.assertFalse(report["physical_quality_accepted"])
        self.assertFalse(report["performance_accepted"])
        self.assertFalse(report["repeated_timing_evidence"])
        self.assertTrue(report["final_source_and_idle_guard_pass"])
        self.assertTrue(all(run["returncode"] == 0 for run in report["runs"]))

    def test_first_failure_survives_final_guard_failure(self):
        """Preserve the first failed arm and cleanup metadata without launching another."""
        args, capture = self.fake_capture()

        def run(batch, _lab):
            for item in batch:
                item.update(returncode=1, cleanup_signals=["SIGTERM"])
            raise RuntimeError("first batch failure")

        with (
            patch.object(driver.owner, "load_capture", return_value=capture),
            patch.object(driver.owner, "tool_checkout", return_value=None),
            patch.object(driver.owner, "runtime_software", return_value={}),
            patch.object(driver.owner, "source_guard", side_effect=[None, RuntimeError("later source failure")]),
            patch.object(driver.owner, "run_batch", side_effect=run),
            self.assertRaisesRegex(RuntimeError, "first batch failure"),
        ):
            driver.main(self.argv)
        report = json.loads((args.output_dir / "manifest.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertIn("first batch failure", report["error"])
        self.assertIn("later source failure", report["final_guard_errors"][0]["error"])
        self.assertEqual(len(report["runs"]), 2)
        self.assertNotIn("summary", report)
        self.assertTrue(all(run["cleanup_signals"] == ["SIGTERM"] for run in report["runs"]))

    def test_initialization_failure_is_saved(self):
        """Persist a failed manifest even when the selected Lab helper cannot load."""
        args, _capture = self.fake_capture()
        with (
            patch.object(driver.owner, "tool_checkout", return_value=None),
            patch.object(driver.owner, "load_capture", side_effect=ValueError("initialization failed")),
            self.assertRaisesRegex(ValueError, "initialization failed"),
        ):
            driver.main(self.argv)
        report = json.loads((args.output_dir / "manifest.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["runs"], [])
        self.assertIn("initialization failed", report["error"])


if __name__ == "__main__":
    unittest.main()
