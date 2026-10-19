# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise same-backend variant comparison without importing a GPU runtime."""

import ast
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
        for name in driver.owner.REQUIRED_PACKAGES:
            roots = (
                [self.root / arm / name for arm in ("base", "candidate")]
                if name == "newton"
                else [self.root / "lab/source" / name / name]
            )
            for root in roots:
                root.mkdir(parents=True)
                (root / "__init__.py").write_text("# inert source package\n")
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

    def test_legacy_survey_is_explicit_and_heightfield_flags_are_scoped(self):
        """Keep the older capacity observer opt-in and admit only named geometry flags."""
        self.assertFalse(driver.parse_args(self.argv).legacy_baseline_capacity)
        args = driver.parse_args(
            [*self.argv, "--legacy-baseline-capacity", "--candidate-env", "NEWTON_HEIGHTFIELD_PAIR_CSR=1"]
        )
        self.assertTrue(args.legacy_baseline_capacity)
        self.assertEqual(driver.variant_flags(args, "candidate")[0], {"NEWTON_HEIGHTFIELD_PAIR_CSR": "1"})
        self.assertEqual(driver.variant_flags(args, "baseline")[0], {})

    def test_explicit_solver_recipes_are_narrowly_typed(self):
        """Allow representation recipes without admitting budget or storage changes."""
        args = driver.parse_args(
            [
                *self.argv,
                "--baseline-solver-attr",
                "pgs_mode=split",
                "--candidate-solver-attr",
                "grouped_dynamics=true",
                "--candidate-solver-attr",
                "mf_gs_parallel_rows=128",
            ]
        )
        self.assertEqual(args.solver_recipes["baseline"], {"pgs_mode": "split"})
        self.assertEqual(args.solver_recipes["candidate"], {"grouped_dynamics": True, "mf_gs_parallel_rows": 128})
        for value in (
            "pgs_iterations=1",
            "num_substeps=1",
            "dense_max_constraints=64",
            "pgs_mode=dense",
            "grouped_dynamics=1",
            "mf_gs_parallel_rows=-1",
            "mf_gs_parallel_rows=129",
            "mf_gs_parallel_sweeps=0",
            "mf_gs_parallel_sweeps=129",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                driver.parse_args([*self.argv, "--candidate-solver-attr", value])
        with self.assertRaises(ValueError):
            driver.parse_args(
                [
                    *self.argv,
                    "--candidate-solver-attr",
                    "lazy_kinematics=true",
                    "--candidate-solver-attr",
                    "lazy_kinematics=false",
                ]
            )

    def test_real_make_batch_keeps_backend_and_variant_identity_separate(self):
        """Forward identical recipes/capacities while naming both runs as FPGS variants."""
        args = driver.parse_args(
            [
                *self.argv,
                "--capacity",
                "ant:fpgs:dense_max_constraints=704",
                "--capacity",
                "ant:fpgs:broad_phase_output_max=57344",
                "--candidate-solver-attr",
                "pgs_mode=matrix_free",
                "--candidate-solver-attr",
                "lazy_kinematics=false",
            ]
        )
        harness = Path(args.isaaclab) / driver.owner.HARNESS_RELATIVE
        harness.mkdir(parents=True)
        capture = SimpleNamespace(
            HARNESS=harness, COMMON_ENVIRONMENT={}, RECIPES={"ant": ("Isaac-Ant", ("pgs_mode=split",), {})}
        )
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
                self.assertEqual(
                    run["environment"]["PYTHONPATH"],
                    driver.owner.source_pythonpath(args.isaaclab, getattr(args, label)),
                )
                self.assertIn("dense_max_constraints=704", run["command"])
                self.assertIn("--broad-phase-output-max", run["command"])
                self.assertEqual(run["command"][run["command"].index("--seed") + 1], "0")
                self.assertFalse(run["mjwarp_linesearch_fix"])
                self.assertIn("pgs_mode=split" if label == "baseline" else "pgs_mode=matrix_free", run["command"])
                self.assertEqual(sum(value.startswith("pgs_mode=") for value in run["command"]), 1)
                if label == "candidate":
                    boolean = next(
                        value.partition("=")[2] for value in run["command"] if value.startswith("lazy_kinematics=")
                    )
                    self.assertIs(ast.literal_eval(boolean), False)

    def test_actual_recipe_is_verified_without_relaxing_other_fields(self):
        """Reject ignored requested settings, missing metadata and changed sweep budgets."""
        args = driver.parse_args([*self.argv, "--candidate-solver-attr", "pgs_mode=split"])
        recipe = {
            "grouped_dynamics": False,
            "lazy_kinematics": True,
            "pgs_mode": "split",
            "mf_gs_parallel_rows": 64,
            "mf_gs_parallel_matrix_free": False,
            "mf_gs_parallel_sweeps": 4,
            "mf_gs_parallel_linked_sweeps": "auto",
            "mf_gs_parallel_linked_sweeps_applied": 0,
        }
        run = {"variant": "candidate"}
        checks = {"boundaries": [{"solver_recipe": dict(recipe)} for _ in range(2)]}
        actual, invariant = driver.checked_recipe(args, run, checks, 4)
        self.assertEqual(actual, recipe)
        self.assertNotIn("pgs_mode", invariant)
        self.assertNotIn("mf_gs_parallel_linked_sweeps", invariant)
        self.assertEqual(invariant["mf_gs_parallel_sweeps"], 4)
        checks["boundaries"][1]["solver_recipe"]["mf_gs_parallel_sweeps"] = 3
        with self.assertRaisesRegex(RuntimeError, "recipe changed"):
            driver.checked_recipe(args, run, checks, 4)
        checks["boundaries"][1]["solver_recipe"] = dict(recipe)
        args.solver_recipes["candidate"]["pgs_mode"] = "matrix_free"
        with self.assertRaisesRegex(RuntimeError, "requested solver recipe"):
            driver.checked_recipe(args, run, checks, 4)
        args.solver_recipes["candidate"]["pgs_mode"] = "split"
        for name, value in (("mf_gs_parallel_linked_sweeps", "fast"), ("mf_gs_parallel_linked_sweeps_applied", None)):
            checks["boundaries"][0]["solver_recipe"] = {**recipe, name: value}
            with self.assertRaisesRegex(RuntimeError, "recipe metadata"):
                driver.checked_recipe(args, run, checks, 4)
        del checks["boundaries"][0]["solver_recipe"]
        with self.assertRaisesRegex(RuntimeError, "recipe metadata"):
            driver.checked_recipe(args, run, checks, 4)

    def test_parallel_recipe_cannot_silently_raise_iteration_budget(self):
        """Require explicitly enabled parallel sweeps, linked tiers' included, to retain the task allowance."""
        args = driver.parse_args([*self.argv, "--candidate-solver-attr", "mf_gs_parallel_rows=128"])
        recipe = {
            "grouped_dynamics": False,
            "lazy_kinematics": True,
            "pgs_mode": "matrix_free",
            "mf_gs_parallel_rows": 128,
            "mf_gs_parallel_matrix_free": False,
            "mf_gs_parallel_sweeps": 24,
            "mf_gs_parallel_linked_sweeps": "auto",
            "mf_gs_parallel_linked_sweeps_applied": 0,
        }
        checks = {"boundaries": [{"solver_recipe": dict(recipe)} for _ in range(2)]}
        with self.assertRaisesRegex(RuntimeError, "parallel sweeps"):
            driver.checked_recipe(args, {"variant": "candidate"}, checks, 8)
        args.solver_recipes["candidate"] = {"pgs_mode": "matrix_free"}
        with self.assertRaisesRegex(RuntimeError, "parallel sweeps"):
            driver.checked_recipe(args, {"variant": "candidate"}, checks, 8)
        args.solver_recipes["candidate"]["mf_gs_parallel_sweeps"] = 8
        for boundary in checks["boundaries"]:
            boundary["solver_recipe"]["mf_gs_parallel_sweeps"] = 8
        self.assertEqual(
            driver.checked_recipe(args, {"variant": "candidate"}, checks, 8)[0]["mf_gs_parallel_sweeps"], 8
        )
        # Patch-friction (linked) tiers that sweep the "auto" budget raise the allowance the point tiers keep; at
        # the allowance (FEATHER_PGS_PATCH_PARALLEL_SWEEPS=8) they retain it.
        for applied, allowed in ((30, False), (8, True)):
            for boundary in checks["boundaries"]:
                boundary["solver_recipe"]["mf_gs_parallel_linked_sweeps_applied"] = applied
            if allowed:
                actual, invariant = driver.checked_recipe(args, {"variant": "candidate"}, checks, 8)
                self.assertEqual(actual["mf_gs_parallel_linked_sweeps_applied"], 8)
                # The selecting arm's own budget is checked here, not compared with the other arm's.
                self.assertNotIn("mf_gs_parallel_linked_sweeps_applied", invariant)
            else:
                with self.assertRaisesRegex(RuntimeError, "parallel sweeps"):
                    driver.checked_recipe(args, {"variant": "candidate"}, checks, 8)

    def test_unselected_linked_budget_is_a_shared_physics_budget(self):
        """Fingerprint the applied linked budget when no arm selects the tiers, and never the revision's request."""
        args = driver.parse_args(self.argv)
        recipe = {
            "grouped_dynamics": False,
            "lazy_kinematics": True,
            "pgs_mode": "matrix_free",
            "mf_gs_parallel_rows": 128,
            "mf_gs_parallel_matrix_free": True,
            "mf_gs_parallel_sweeps": 24,
            "mf_gs_parallel_linked_sweeps": None,
            "mf_gs_parallel_linked_sweeps_applied": 0,
        }
        invariants = []
        for request, applied in ((None, 0), ("auto", 0), ("auto", 90)):
            arm = {**recipe, "mf_gs_parallel_linked_sweeps": request, "mf_gs_parallel_linked_sweeps_applied": applied}
            checks = {"boundaries": [{"solver_recipe": dict(arm)} for _ in range(2)]}
            invariants.append(driver.checked_recipe(args, {"variant": "candidate"}, checks, 12)[1])
        # A revision without the option and one with it agree while neither builds linked tiers; a candidate whose
        # recipe now builds them at 90 sweeps differs, so the comparison refuses it.
        self.assertEqual(invariants[0], invariants[1])
        self.assertNotEqual(invariants[1], invariants[2])
        self.assertEqual(invariants[2]["mf_gs_parallel_linked_sweeps_applied"], 90)

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
            "variant": "baseline",
            "output_dir": str(directory),
            "backend": "feather_pgs",
            "mjwarp_linesearch_fix": False,
            "environment": {
                "FPGS_BENCH_RUN_SHA256": "source-harness",
                "FPGS_BENCH_ISAACLAB": str(args.isaaclab),
                "FPGS_BENCH_NEWTON": str(args.baseline),
            },
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
                    "solver_recipe": {
                        "grouped_dynamics": False,
                        "lazy_kinematics": True,
                        "pgs_mode": "matrix_free",
                        "mf_gs_parallel_rows": 64,
                        "mf_gs_parallel_matrix_free": False,
                        "mf_gs_parallel_sweeps": 4,
                        "mf_gs_parallel_linked_sweeps": "auto",
                        "mf_gs_parallel_linked_sweeps_applied": 0,
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
        selected = driver.owner.source_packages(args.isaaclab, args.baseline)
        imported = {
            "selected": selected,
            "resolved": {name: str(Path(value["root"]) / "__init__.py") for name, value in selected.items()},
            "loaded": {
                name: {
                    "file": str(Path(value["root"]) / "__init__.py"),
                    "paths": [value["root"]],
                    "sha256": value["init_sha256"],
                }
                for name, value in selected.items()
            },
        }
        checks.update(import_provenance_before=imported, import_provenance_after=imported)
        capture = SimpleNamespace(RECIPES={"ant": ("Isaac-Ant", (), {})}, _read_result=lambda _run: {})

        def save():
            for name, value in (("capture.json", data), ("capture_checks.json", checks), ("capture_analysis.json", {})):
                (directory / name).write_text(json.dumps(value))

        save()
        result = driver.read_checked_result(capture, args, run, drivers)
        self.assertEqual(result["model"]["solver_dt"], 0.0025)
        self.assertEqual(len(run["artifacts"]), 3)
        args.solver_recipes["candidate"] = {"pgs_mode": "split"}
        baseline = driver.read_checked_result(capture, args, run, drivers)
        run["variant"] = "candidate"
        data["solver_attr"] = ["pgs_mode=split"]
        for metadata in (data["model"], data["model_after"]):
            metadata["pgs_mode"] = "split"
        for boundary in checks["boundaries"]:
            boundary["solver_recipe"]["pgs_mode"] = "split"
        save()
        self.assertEqual(driver.read_checked_result(capture, args, run, drivers), baseline)
        for name in ("num_substeps", "solver_dt", "pgs_iterations", "bodies", "rigid_contact_max"):
            for metadata in (data["model"], data["model_after"]):
                metadata[name] *= 2
            save()
            self.assertNotEqual(driver.read_checked_result(capture, args, run, drivers), baseline)
            for metadata in (data["model"], data["model_after"]):
                metadata[name] /= 2
        for boundary in checks["boundaries"]:
            boundary["capacities"]["dense_max_constraints"] *= 2
        save()
        self.assertNotEqual(driver.read_checked_result(capture, args, run, drivers), baseline)
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
            (root / "newton").mkdir(parents=True, exist_ok=True)
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
