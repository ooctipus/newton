# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise the portable benchmark driver with standard-library CPU mocks only."""

import contextlib
import copy
import hashlib
import io
import json
import os
import signal
import statistics
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import compare_backends as driver


class TestArgumentsAndHelpers(unittest.TestCase):
    def setUp(self):
        """Create isolated source/output paths for argument and helper checks."""
        temporary = tempfile.TemporaryDirectory(prefix="backend-unit-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.lab = self.root / "lab"
        self.harness = self.lab / driver.HARNESS_RELATIVE
        self.harness.mkdir(parents=True)
        self.argv = [
            "--isaaclab",
            str(self.lab),
            "--fpgs",
            str(self.root / "fpgs"),
            "--mjwarp",
            str(self.root / "mjwarp"),
            "--task",
            "ant",
            "--output-dir",
            str(self.root / "output"),
        ]

    def helper_source(self, *, omit=None):
        """Write a portable stand-in exposing the required helper contract."""
        source = (
            "from pathlib import Path\n"
            "REPO=Path(__file__).resolve().parents[3]\n"
            "HARNESS=Path(__file__).resolve().parent\n"
            "RECIPES={'ant': ('Isaac-Ant', (), {})}\n"
            "COMMON_ENVIRONMENT={}\n"
        )
        for name in driver.HELPERS:
            if name != omit:
                source += f"def {name}(*args): return None\n"
        path = self.harness / "compare_gpus.py"
        path.write_text(source)
        return path

    def test_defaults_and_task_deduplication(self):
        """Keep the fixed seed recipe and original sampling defaults."""
        args = driver.parse_args([*self.argv, "--task", "humanoid", "--task", "ant"])
        self.assertEqual(args.task, ["ant", "humanoid"])
        self.assertEqual(args.gpus, [0, 1])
        self.assertEqual(
            (
                args.repeats,
                args.num_envs,
                args.warmup_steps,
                args.steps,
                args.profile_steps,
            ),
            (3, 16384, 200, 40, 40),
        )
        self.assertFalse(hasattr(args, "solver_attr"))
        self.assertTrue(args.check_overflow)
        self.assertFalse(args.allow_unchecked)
        self.assertFalse(args.mjwarp_linesearch_fix)

    def test_unchecked_requires_explicit_mutually_exclusive_choice(self):
        """Permit legacy capture only on an explicit opt-out, never via an omitted option."""
        args = driver.parse_args([*self.argv, "--allow-unchecked"])
        self.assertTrue(args.allow_unchecked)
        self.assertFalse(args.check_overflow)
        self.assertTrue(driver.parse_args([*self.argv, "--check-overflow"]).check_overflow)
        for extra in (["--check-overflow", "--allow-unchecked"], ["--allow-unchecked", "--check-overflow"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                driver.parse_args(self.argv + extra)

    def test_keyboard_recipe_defaults_and_copy_isolation(self):
        """Add the SO101 typing task without changing any Lab recipe or nested flag map."""
        original = {"ant": ("Isaac-Ant", (), {"FEATHER_PGS_EXISTING": "1"})}
        capture = SimpleNamespace(RECIPES=original)
        before = copy.deepcopy(original)
        recipes = driver.recipe_map(capture)
        self.assertEqual(recipes["keyboard-so101"], ("IsaacContrib-Keyboard-SO101", (), {}))
        self.assertEqual(recipes["ant"], before["ant"])
        self.assertIs(capture.RECIPES, original)
        self.assertEqual(capture.RECIPES, before)
        recipes["ant"][2]["FEATHER_PGS_EXISTING"] = "modified-copy"
        self.assertEqual(capture.RECIPES, before)

    def test_keyboard_recipe_conflict_is_rejected(self):
        """Accept an identical Lab alias and reject conflicting task configuration."""
        expected = ("IsaacContrib-Keyboard-SO101", (), {})
        capture = SimpleNamespace(RECIPES={"keyboard-so101": expected})
        self.assertEqual(driver.recipe_map(capture)["keyboard-so101"], expected)
        for conflict in (("Different-Task", (), {}), (expected[0], ("pgs_iterations=1",), {})):
            with self.subTest(conflict=conflict), self.assertRaisesRegex(ValueError, "Conflicting.*keyboard-so101"):
                driver.recipe_map(SimpleNamespace(RECIPES={"keyboard-so101": conflict}))

    def test_invalid_counts_and_gpu_selection(self):
        """Reject invalid samples and duplicate/negative devices before any launch."""
        for extra in (
            ["--repeats", "0"],
            ["--steps", "0"],
            ["--profile-steps", "0"],
            ["--num-envs", "-1"],
            ["--warmup-steps", "-1"],
            ["--gpus", "0", "0"],
            ["--gpus", "-1"],
        ):
            with (
                self.subTest(extra=extra),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                driver.parse_args(self.argv + extra)

    def test_explicit_flags_and_duplicate_rejection(self):
        """Accept only selected-GPU FPGS flags and retain literal empty values."""
        self.assertEqual(
            driver.parse_flags(["0:FEATHER_PGS_INK=1", "1:NEWTON_NARROW_PHASE_THREADS_X="], [0, 1]),
            {0: {"FEATHER_PGS_INK": "1"}, 1: {"NEWTON_NARROW_PHASE_THREADS_X": ""}},
        )
        for entries in (
            ["2:FEATHER_PGS_INK=1"],
            ["0:PYTHONPATH=bad"],
            ["0:NEWTON_UNSCOPED=1"],
            ["0:FEATHER_PGS_INK"],
            ["0:FEATHER_PGS_INK=1", "0:FEATHER_PGS_INK=0"],
        ):
            with self.subTest(entries=entries), self.assertRaises(ValueError):
                driver.parse_flags(entries, [0, 1])

    def test_scoped_capacity_overrides(self):
        """Keep calibrated capacities task/backend-specific without changing budgets."""
        result = driver.parse_capacities(
            ["keyboard-so101:fpgs:dense_max_constraints=704", "keyboard-so101:mjwarp:nconmax=36"],
            ["keyboard-so101", "ant"],
        )
        self.assertEqual(result["keyboard-so101"]["fpgs"], {"dense_max_constraints": 704})
        self.assertEqual(result["keyboard-so101"]["mjwarp"], {"nconmax": 36})
        self.assertEqual(result["ant"], {"fpgs": {}, "mjwarp": {}})

    def test_capacity_overrides_reject_physics_and_ambiguity(self):
        """Reject unselected tasks, wrong owners, duplicate values and physics tuning."""
        for entries in (
            ["other:fpgs:dense_max_constraints=704"],
            ["ant:fpgs:njmax=512"],
            ["ant:mjwarp:ls_iterations=50"],
            ["ant:fpgs:pgs_iterations=1"],
            ["ant:fpgs:rigid_contact_max=0"],
            ["ant:fpgs:rigid_contact_max=1.5"],
            ["ant:fpgs:rigid_contact_max=10", "ant:fpgs:rigid_contact_max=20"],
        ):
            with self.subTest(entries=entries), self.assertRaises(ValueError):
                driver.parse_capacities(entries, ["ant"])

    def test_broad_output_capacity_requires_checked_constructor_route(self):
        """Reject the new key in legacy mode before it can reach Lab from_dict."""
        extra = ["--capacity", "ant:fpgs:broad_phase_output_max=65536"]
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            driver.parse_args(self.argv + extra + ["--allow-unchecked"])
        self.assertTrue(driver.parse_args(self.argv + extra).check_overflow)
        self.assertTrue(driver.parse_args(self.argv + extra + ["--check-overflow"]).check_overflow)

    def test_linesearch_fix_requires_checked_mode(self):
        """Require warning/overflow checks before selecting the numerical compatibility fix."""
        with self.assertRaises(SystemExit):
            driver.parse_args([*self.argv, "--allow-unchecked", "--mjwarp-linesearch-fix"])
        self.assertTrue(driver.parse_args([*self.argv, "--mjwarp-linesearch-fix"]).check_overflow)
        self.assertTrue(
            driver.parse_args([*self.argv, "--check-overflow", "--mjwarp-linesearch-fix"]).mjwarp_linesearch_fix
        )

    def test_clean_inherited_flags_and_project_selection(self):
        """Prevent inherited solver flags or uv paths from leaking into either arm."""
        inherited = {
            "FEATHER_PGS_BAD": "1",
            "FEATHER_OTHER": "1",
            "NEWTON_OTHER": "1",
            "NEWTON_NARROW_PHASE_BAD": "1",
            "FPGS_PROBE_BAD": "1",
            "FPGS_BENCH_ISAACLAB": "bad",
            "FPGS_BENCH_NEWTON": "bad",
            "FPGS_BENCH_RUN_SHA256": "bad",
            "PYTHONPATH": "bad",
            "UV_PROJECT": "bad",
            "UV_PROJECT_ENVIRONMENT": "bad",
            "UV_WORKING_DIR": "bad",
            "UV_NO_PROJECT": "1",
            "UV_ISOLATED": "1",
            "UV_ACTIVE": "1",
            "UV_PYTHON": "bad",
            "VIRTUAL_ENV": "bad",
            "KEEP_ME": "safe",
        }
        with patch.dict(os.environ, inherited, clear=True):
            env = driver.clean_environment({"PYTHONPATH": "selected"}, self.lab)
        self.assertEqual(
            env,
            {
                "KEEP_ME": "safe",
                "UV_PROJECT": str(self.lab),
                "UV_NO_SYNC": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": "selected",
            },
        )

    def test_load_exact_helper_without_bytecode_writes(self):
        """Execute the explicit helper source and create no source-tree cache."""
        path = self.helper_source()
        before = {p.relative_to(self.lab) for p in self.lab.rglob("*")}
        capture = driver.load_capture(self.lab)
        self.assertEqual(capture.REPO, self.lab)
        self.assertEqual(capture._loaded_source_sha256, hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(before, {p.relative_to(self.lab) for p in self.lab.rglob("*")})

    def test_missing_helper_contract_is_rejected(self):
        """Fail clearly when an older Lab checkout lacks a required helper."""
        self.helper_source(omit="_read_result")
        with self.assertRaisesRegex(ValueError, "_read_result"):
            driver.load_capture(self.lab)

    def test_output_inside_source_or_existing_is_rejected(self):
        """Protect source trees and all existing artifacts without exceptions for ignored paths."""
        with self.assertRaises(ValueError):
            driver.validate_output(self.lab / "ignored" / "results", [self.lab])
        existing = self.root / "existing"
        existing.mkdir()
        with self.assertRaises(FileExistsError):
            driver.validate_output(existing, [self.lab])

    def test_runtime_query_uses_selected_lab_environment(self):
        """Verify selected imports with CUDA hidden and no real Newton module loaded."""
        calls = []

        def query(command, **kwargs):
            calls.append((command, kwargs))
            self.assertEqual(command[:4], ["uv", "run", "python", "-c"])
            self.assertEqual(kwargs["cwd"], self.lab)
            self.assertEqual(kwargs["env"]["UV_NO_SYNC"], "1")
            self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
            self.assertEqual(kwargs["env"]["GPU"], "")
            # Execute the literal query with Newton mocked; no simulator or
            # CUDA module is imported by this portable CPU test.
            stdout = io.StringIO()
            newton = SimpleNamespace(__file__=str(Path(kwargs["env"]["PYTHONPATH"]) / "newton/__init__.py"))
            with (
                patch("importlib.metadata.version", return_value="test-version"),
                patch.dict(sys.modules, {"newton": newton}),
                patch.object(sys, "prefix", str(self.lab / ".venv")),
                contextlib.redirect_stdout(stdout),
            ):
                exec(compile(command[4], "<runtime-query>", "exec"), {})
            return stdout.getvalue()

        with patch.object(driver.subprocess, "check_output", side_effect=query):
            result = driver.runtime_software(self.lab, {"fpgs": self.root / "fpgs", "mjwarp": self.root / "mjwarp"})
        self.assertEqual(len(calls), 2)
        self.assertEqual(result["mjwarp"]["distributions"]["mujoco-warp"], "test-version")
        self.assertEqual(calls[0][1]["env"]["PYTHONPATH"], str(self.root / "fpgs"))
        self.assertEqual(calls[1][1]["env"]["PYTHONPATH"], str(self.root / "mjwarp"))

    def test_runtime_wrong_newton_import_is_rejected(self):
        """Reject an installed package taking precedence over the selected checkout."""
        with (
            patch.object(
                driver.subprocess,
                "check_output",
                return_value=json.dumps({"newton_file": "/wrong/newton/__init__.py"}),
            ),
            self.assertRaisesRegex(RuntimeError, "wrong Newton source"),
        ):
            driver.runtime_software(self.lab, {"fpgs": self.root / "fpgs"})

    def test_runtime_wrong_python_environment_is_rejected(self):
        """Reject matching Newton sources imported by a different Python environment."""
        result = {"newton_file": str(self.root / "fpgs/newton/__init__.py"), "python_prefix": "/wrong/venv"}
        with (
            patch.object(driver.subprocess, "check_output", return_value=json.dumps(result)),
            self.assertRaisesRegex(RuntimeError, "prepared Isaac Lab .venv"),
        ):
            driver.runtime_software(self.lab, {"fpgs": self.root / "fpgs"})

    def test_source_and_driver_drift_fail_closed(self):
        """Guard source snapshots including untracked content and the loaded tool bytes."""
        path = self.helper_source()
        capture = SimpleNamespace(_source=Mock(return_value={"source_diff_sha256": "original"}))
        roots = {"lab": self.lab}
        original = {"lab": capture._source(self.lab)}
        hashes = driver.file_hashes([path])
        driver.source_guard(capture, roots, original, hashes)
        capture._source.return_value = {"source_diff_sha256": "untracked-content-changed"}
        with self.assertRaisesRegex(RuntimeError, "checkout changed"):
            driver.source_guard(capture, roots, original, hashes)
        capture._source.return_value = original["lab"]
        path.write_text(path.read_text() + "# drift\n")
        with self.assertRaisesRegex(RuntimeError, "helper/harness changed"):
            driver.source_guard(capture, roots, original, hashes)


class FakeProcess:
    def __init__(self, pid, events, *, code=0, interrupt=False, timeout=False):
        """Create a controllable subprocess stand-in with no operating-system child."""
        self.pid, self.events, self.code = pid, events, code
        self.interrupt, self.timeout = interrupt, timeout
        self.returncode = None

    def wait(self, timeout=None):
        """Record ordinary, interrupted, and timeout waits."""
        self.events.append(("wait", self.pid, timeout))
        if self.interrupt:
            self.interrupt = False
            raise KeyboardInterrupt()
        if timeout is not None and self.timeout:
            self.timeout = False
            raise subprocess.TimeoutExpired("mock", timeout)
        self.returncode = self.code
        return self.returncode

    def poll(self):
        """Report whether the fake child has been reaped."""
        return self.returncode


class TestChildCleanup(unittest.TestCase):
    def setUp(self):
        """Prepare two fake GPU runs with isolated writable logs."""
        temporary = tempfile.TemporaryDirectory(prefix="backend-cleanup-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.events = []
        self.runs = [
            {
                "newton": "fpgs",
                "gpu_index": gpu,
                "command": ["mock-profiler"],
                "environment": {},
                "driver_log": str(self.root / f"gpu{gpu}.log"),
            }
            for gpu in (0, 1)
        ]

    def run_fake(self, factory):
        """Run production child orchestration with process creation and signals mocked."""
        with (
            patch.object(driver.subprocess, "Popen", side_effect=factory),
            patch.object(
                driver.os,
                "killpg",
                side_effect=lambda pid, sig: self.events.append(("kill", pid, sig)),
            ),
            patch.object(driver, "wait_group_gone", return_value=True),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            driver.run_batch(self.runs, self.root)

    def test_both_start_before_either_wait(self):
        """Launch independent GPU children concurrently and reap the complete batch."""

        def factory(command, **kwargs):
            self.assertTrue(kwargs["start_new_session"])
            pid = 100 + len([e for e in self.events if e[0] == "start"])
            self.events.append(("start", pid))
            return FakeProcess(pid, self.events)

        self.run_fake(factory)
        self.assertEqual([e[0] for e in self.events if e[0] != "kill"], ["start", "start", "wait", "wait"])
        self.assertTrue(all(run["returncode"] == 0 for run in self.runs))

    def test_child_failure_still_waits_the_other_gpu(self):
        """Finish both children before reporting an ordinary profiler failure."""
        processes = [
            FakeProcess(100, self.events, code=1),
            FakeProcess(101, self.events),
        ]
        with self.assertRaisesRegex(RuntimeError, "Profiler failed"):
            self.run_fake(processes)
        self.assertEqual([e[1] for e in self.events if e[0] == "wait"], [100, 101])
        self.assertEqual([run["returncode"] for run in self.runs], [1, 0])

    def test_second_spawn_failure_reaps_first_child(self):
        """Terminate an already-running GPU child when the second spawn fails."""
        with self.assertRaisesRegex(OSError, "spawn failed"):
            self.run_fake([FakeProcess(100, self.events), OSError("spawn failed")])
        self.assertIn(("kill", 100, signal.SIGTERM), self.events)
        self.assertIn(("wait", 100, 10), self.events)

    def test_interrupt_reaps_both_children(self):
        """Unwind an interrupted wait and terminate both owned process groups."""
        with self.assertRaises(KeyboardInterrupt):
            self.run_fake(
                [
                    FakeProcess(100, self.events, interrupt=True),
                    FakeProcess(101, self.events),
                ]
            )
        self.assertEqual([e[1] for e in self.events if e[0] == "kill"], [100, 101])
        self.assertTrue(all(run["returncode"] == 0 for run in self.runs))

    def test_cleanup_escalates_after_timeout(self):
        """Escalate a stuck process group from TERM to KILL, then reap it."""
        with self.assertRaises(OSError):
            self.run_fake([FakeProcess(100, self.events, timeout=True), OSError("spawn failed")])
        self.assertEqual(
            [e[2] for e in self.events if e[0] == "kill"],
            [signal.SIGTERM, signal.SIGKILL],
        )
        self.assertEqual([e for e in self.events if e[0] == "wait"], [("wait", 100, 10), ("wait", 100, 10)])

    def test_cleanup_continues_after_one_group_error(self):
        """Attempt every child cleanup even when the first group cannot be signaled."""
        processes = [
            ({}, FakeProcess(100, self.events)),
            ({}, FakeProcess(101, self.events)),
        ]
        signals = []

        def signal_group(pid, sig):
            signals.append(pid)
            if pid == 100:
                raise PermissionError("mock permission error")

        with (
            patch.object(driver.os, "killpg", side_effect=signal_group),
            patch.object(driver, "wait_group_gone", return_value=True),
            self.assertRaisesRegex(RuntimeError, "PID 100"),
        ):
            driver.stop_children(processes)
        self.assertEqual(signals, [100, 101])
        self.assertEqual(processes[1][1].returncode, 0)

    def test_exited_leader_still_terminates_surviving_group(self):
        """Escalate descendants that outlive an already-reaped session leader."""
        process = FakeProcess(100, self.events)
        process.returncode = 0
        run = {}
        with (
            patch.object(driver, "signal_group", return_value=True) as signal_owned,
            patch.object(driver, "wait_group_gone", return_value=False),
        ):
            driver.stop_children([(run, process)], grace_seconds=0.01)
        self.assertEqual(
            signal_owned.call_args_list,
            [unittest.mock.call(100, signal.SIGTERM), unittest.mock.call(100, signal.SIGKILL)],
        )
        self.assertEqual(run["cleanup_signals"], ["SIGTERM", "SIGKILL"])
        self.assertFalse(self.events)

    def test_leader_exit_on_term_does_not_skip_descendant_kill(self):
        """Check the group after its leader exits during the TERM grace period."""
        process = FakeProcess(100, self.events)
        run = {}
        with (
            patch.object(driver, "signal_group", return_value=True) as signal_owned,
            patch.object(driver, "wait_group_gone", return_value=False),
        ):
            driver.stop_children([(run, process)], grace_seconds=0.01)
        self.assertEqual(signal_owned.call_args_list[-1], unittest.mock.call(100, signal.SIGKILL))
        self.assertEqual(process.returncode, 0)

    def test_absent_group_is_benign_but_permission_denied_is_not(self):
        """Distinguish a vanished process group from an unsignalable one."""
        with patch.object(driver.os, "killpg", side_effect=ProcessLookupError):
            self.assertFalse(driver.signal_group(100, 0))
        with patch.object(driver.os, "killpg", side_effect=PermissionError), self.assertRaises(PermissionError):
            driver.signal_group(100, 0)

    def test_group_liveness_polling_has_bounded_grace(self):
        """Poll whole-group liveness and stop at the supplied deadline."""
        with (
            patch.object(driver, "signal_group", return_value=True),
            patch.object(driver.time, "monotonic", side_effect=[0, 0.5, 1.0]),
            patch.object(driver.time, "sleep") as sleep,
        ):
            self.assertFalse(driver.wait_group_gone(100, 1.0))
        sleep.assert_called_once_with(0.05)


class TestCompleteDriver(unittest.TestCase):
    def setUp(self):
        """Provide isolated fake checkouts and the required unchanged-helper interface."""
        temporary = tempfile.TemporaryDirectory(prefix="backend-main-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.lab, self.fpgs, self.mj = (self.root / name for name in ("lab", "fpgs", "mjwarp"))
        self.harness = self.lab / driver.HARNESS_RELATIVE
        self.harness.mkdir(parents=True)
        for name in (
            "compare_gpus.py",
            "nsys_run.sh",
            "run_profiled.py",
            "analyze_nsys.py",
        ):
            (self.harness / name).write_text(f"# frozen {name}\n")
        for root in (self.fpgs, self.mj):
            (root / "newton").mkdir(parents=True)
            (root / "newton/__init__.py").write_text("# fake Newton checkout\n")
        self.output = self.root / "results"
        self.events, self.calls = [], []
        self.source_drift = False
        self.result_error = None
        self.child_failure = False
        self.omit_checks = False
        self.invalid_checks = None
        self.devices = [{"index": gpu, "uuid": f"GPU-{gpu}", "name": f"mock GPU {gpu}"} for gpu in (0, 1)]
        self.capture = SimpleNamespace(
            __file__=str(self.harness / "compare_gpus.py"),
            REPO=self.lab,
            HARNESS=self.harness,
            RECIPES={
                "ant": ("Isaac-Ant", (), {}),
                "anymald": (
                    "Isaac-Velocity-Flat-AnymalD",
                    ("grouped_dynamics=True", "mf_gs_parallel_rows=48"),
                    {"FEATHER_PGS_INK": "1"},
                ),
            },
            COMMON_ENVIRONMENT={
                "FEATHER_PGS_GROUP_LANES": "16",
                "NEWTON_NARROW_PHASE_THREADS_X": "4",
                "UV_NO_SYNC": "1",
                "REPEATS": "1",
                "PYTHONUNBUFFERED": "1",
            },
            _loaded_source_sha256=hashlib.sha256((self.harness / "compare_gpus.py").read_bytes()).hexdigest(),
            _source=self.source,
            _gpus=Mock(return_value=self.devices),
            _require_idle=Mock(),
            _software=Mock(return_value={"driver_python": "mock-driver"}),
            _read_result=Mock(side_effect=self.read_result),
            _summaries=self.summaries,
            _write_json=lambda path, value: path.write_text(json.dumps(value, allow_nan=False)),
        )
        self.argv = [
            "--isaaclab",
            str(self.lab),
            "--fpgs",
            str(self.fpgs),
            "--mjwarp",
            str(self.mj),
            "--task",
            "anymald",
            "--output-dir",
            str(self.output),
        ]

    def source(self, path):
        """Represent the helper's commit and full source-diff snapshot contract."""
        return {
            "path": str(path),
            "sha": "mock-commit",
            "dirty": True,
            "source_diff_sha256": "drift" if self.source_drift and self.calls else "frozen-with-untracked",
        }

    def read_result(self, run):
        """Stand in for the independently maintained strict Lab analyzer/result gate."""
        if self.result_error:
            raise RuntimeError(self.result_error)
        span = (100 if run["newton"] == "fpgs" else 400) + 50 * run["gpu_index"]
        return {
            "graph_span_us_per_step": span,
            "wall_us_per_step": 2000,
            "auxiliary_graph_us_per_step": 3,
            "timing_mode": "graph",
            "state_finite": True,
            "contacts_before": 7,
            "contacts_after": 9,
        }

    def summaries(self, runs):
        """Exercise aggregation consumption without duplicating the Lab analyzer."""
        groups = {}
        for run in runs:
            if "result" in run:
                groups.setdefault((run["task"], run["gpu_index"], run["newton"]), []).append(run["result"])
        return [
            {
                "task": task,
                "gpu_index": gpu,
                "newton": backend,
                "repeats": len(values),
                "median_us": statistics.median(v["graph_span_us_per_step"] for v in values),
                "wall_median_us": statistics.median(v["wall_us_per_step"] for v in values),
                "state_finite": all(v["state_finite"] for v in values),
            }
            for (task, gpu, backend), values in groups.items()
        ]

    def popen(self, command, **kwargs):
        """Capture commands and enforce paired starts without launching a subprocess."""
        self.calls.append((command, kwargs))
        pid = 1000 + len(self.calls)
        self.events.append(("start", pid))
        self.assertEqual(kwargs["cwd"], self.lab)
        self.assertTrue(kwargs["start_new_session"])
        env = kwargs["env"]
        if "FPGS_BENCH_RUN_SHA256" in env and not self.omit_checks:
            directory = Path(env["OUT_DIR"])
            check_path = directory / "capture_checks.json"
            checks = {
                "complete": True,
                "check_pass": True,
                "boundary_count": 2,
                "boundaries": [
                    {
                        "boundary": index,
                        "physics": command[3],
                        "check_pass": True,
                        "collision": {"check_pass": True, "status": "checked"},
                    }
                    for index in (0, 1)
                ],
                "checks_output": str(check_path),
                "capture_output": str(directory / "capture.json"),
                "mjwarp_linesearch_fix": "--mjwarp-linesearch-fix" in command,
                "physics_work_modified": "--mjwarp-linesearch-fix" in command,
                "physics_budgets_modified": False,
                "run_profiled_sha256": env["FPGS_BENCH_RUN_SHA256"],
                "checked_capture_sha256": driver.file_hashes([Path(driver.__file__).with_name("checked_capture.py")])[
                    str(Path(driver.__file__).with_name("checked_capture.py"))
                ],
            }
            if "--mjwarp-linesearch-fix" in command:
                helper = Path(driver.__file__).with_name("mjwarp_linesearch_compat.py")
                checks["mjwarp_linesearch_compat"] = {
                    "helper_sha256": driver.file_hashes([helper])[str(helper)],
                    "installation": {
                        "installed_package_modified": False,
                        "gradient_tolerance_changed": False,
                        "iteration_budget_changed": False,
                    },
                }
                for entry in checks["boundaries"]:
                    entry["mjwarp_linesearch_model_supported"] = True
            if "--broad-phase-output-max" in command:
                requested = int(command[command.index("--broad-phase-output-max") + 1])
                checks["collision_capacity"] = {
                    "requested": requested,
                    "pipelines": [
                        {
                            "construction_pass": True,
                            "broad_phase_mode": "explicit",
                            "full_input_pairs": 100000,
                            "effective": min(requested, 100000),
                        }
                    ],
                }
            if self.invalid_checks is not None:
                self.invalid_checks(checks)
            check_path.write_text(json.dumps(checks))
        return FakeProcess(pid, self.events, code=int(self.child_failure))

    def invoke(self, extra=()):
        """Run the production driver with all external programs and GPU checks mocked."""
        with (
            patch.object(driver, "load_capture", return_value=self.capture),
            patch.object(driver, "tool_checkout", return_value=None),
            patch.object(
                driver,
                "runtime_software",
                return_value={
                    "fpgs": {"distributions": {"mujoco-warp": "test"}},
                    "mjwarp": {},
                },
            ),
            patch.object(driver.shutil, "which", return_value="/mock/executable"),
            patch.object(driver.subprocess, "Popen", side_effect=self.popen),
            patch.object(
                driver.os,
                "killpg",
                side_effect=ProcessLookupError("Finished process group"),
            ),
            patch.dict(
                os.environ,
                {
                    "FEATHER_PGS_POISON": "1",
                    "NEWTON_OTHER_POISON": "1",
                    "PYTHONPATH": "wrong",
                },
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            return driver.main(self.argv + list(extra))

    def test_capacity_commands_are_scoped_and_recorded(self):
        """Forward identical calibrated sizes to both GPUs without cross-backend leakage."""
        entries = [
            "anymald:fpgs:mf_max_constraints=96",
            "anymald:fpgs:rigid_contact_max=32768",
            "anymald:mjwarp:njmax=128",
        ]
        options = ["--repeats", "1"]
        for entry in entries:
            options.extend(["--capacity", entry])
        self.assertEqual(self.invoke(options), 0)
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertEqual(manifest["settings"]["capacity"], entries)
        self.assertEqual(len(self.calls), 4)
        for run in manifest["runs"]:
            command = run["command"]
            if run["newton"] == "fpgs":
                self.assertIn("mf_max_constraints=96", command)
                self.assertIn("env.sim.physics.collision_cfg.rigid_contact_max=32768", command)
                self.assertNotIn("njmax=128", command)
                self.assertIn("mf_gs_parallel_rows=48", command)
            else:
                self.assertIn("njmax=128", command)
                self.assertNotIn("mf_max_constraints=96", command)
                self.assertNotIn("env.sim.physics.collision_cfg.rigid_contact_max=32768", command)

    def test_checked_mode_routes_both_backends_and_pins_every_helper(self):
        """Gate checked results and forward explicit checkout/source pins without recipe changes."""
        self.assertEqual(
            self.invoke(
                [
                    "--repeats",
                    "1",
                    "--capacity",
                    "anymald:fpgs:broad_phase_output_max=65536",
                ]
            ),
            0,
        )
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertTrue(manifest["settings"]["check_overflow"])
        self.assertFalse(manifest["settings"]["allow_unchecked"])
        self.assertEqual(manifest["capture_check_mode"], "checked")
        self.assertEqual(len(manifest["drivers"]), 7)
        for run in manifest["runs"]:
            self.assertEqual(run["capture_check_mode"], "checked")
            self.assertEqual(run["command"][1], str(Path(driver.__file__).with_name("nsys_checked.sh")))
            self.assertEqual(run["environment"]["FPGS_BENCH_ISAACLAB"], str(self.lab))
            self.assertEqual(
                run["environment"]["FPGS_BENCH_NEWTON"], str(self.fpgs if run["newton"] == "fpgs" else self.mj)
            )
            self.assertEqual(
                run["environment"]["FPGS_BENCH_RUN_SHA256"],
                manifest["drivers"][str(self.harness / "run_profiled.py")],
            )
            self.assertTrue(run["overflow_check"]["check_pass"])
            self.assertEqual(run["overflow_check"]["boundary_count"], 2)
            self.assertFalse(any("collision_cfg.broad_phase_output_max" in arg for arg in run["command"]))
            if run["newton"] == "fpgs":
                self.assertEqual(run["command"][5:7], ["--broad-phase-output-max", "65536"])
                self.assertEqual(run["overflow_check"]["collision_capacity"]["pipelines"][0]["effective"], 65536)
            else:
                self.assertNotIn("--broad-phase-output-max", run["command"])
        self.assertEqual(self.capture._read_result.call_count, 4)
        self.assertTrue((self.output / "ratios.json").exists())

    def test_missing_checked_result_prevents_timing_acceptance(self):
        """Reject a successful child that never supplied its required checked report."""
        self.omit_checks = True
        with self.assertRaises(FileNotFoundError):
            self.invoke(["--repeats", "1"])
        self.capture._read_result.assert_not_called()
        self.assertEqual(json.loads((self.output / "manifest.json").read_text())["status"], "failed")
        self.assertFalse((self.output / "ratios.json").exists())

    def test_explicit_unchecked_uses_legacy_and_marks_manifest(self):
        """Retain requested historical diagnosis without implying checked evidence."""
        self.omit_checks = True
        self.assertEqual(self.invoke(["--repeats", "1", "--allow-unchecked"]), 0)
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertEqual(manifest["capture_check_mode"], "unchecked")
        self.assertTrue(manifest["settings"]["allow_unchecked"])
        self.assertFalse(manifest["settings"]["check_overflow"])
        self.assertEqual(len(manifest["drivers"]), 5)
        for run in manifest["runs"]:
            self.assertEqual(run["capture_check_mode"], "unchecked")
            self.assertEqual(run["command"][1], str(self.harness / "nsys_run.sh"))
            self.assertNotIn("overflow_check", run)
            self.assertNotIn("FPGS_BENCH_RUN_SHA256", run["environment"])
            self.assertFalse(run["mjwarp_linesearch_fix"])

    def test_linesearch_fix_only_reaches_mjwarp_and_is_source_bound(self):
        """Record the algorithm change only in the MJ arm while preserving budget declarations."""
        self.assertEqual(self.invoke(["--repeats", "1", "--check-overflow", "--mjwarp-linesearch-fix"]), 0)
        manifest = json.loads((self.output / "manifest.json").read_text())
        helper = Path(driver.__file__).with_name("mjwarp_linesearch_compat.py")
        self.assertIn(str(helper), manifest["drivers"])
        for run in manifest["runs"]:
            fixed = run["newton"] == "mjwarp"
            self.assertEqual(run["mjwarp_linesearch_fix"], fixed)
            self.assertEqual("--mjwarp-linesearch-fix" in run["command"], fixed)
            self.assertEqual(run["overflow_check"]["physics_work_modified"], fixed)
            self.assertIs(run["overflow_check"]["physics_budgets_modified"], False)

    def test_linesearch_fix_metadata_mismatch_prevents_ratios(self):
        """Reject unbound helpers, unsupported models, false work declarations, and changed budgets."""
        mutations = (
            lambda report: report["mjwarp_linesearch_compat"].update(helper_sha256="wrong"),
            lambda report: report["boundaries"][0].update(mjwarp_linesearch_model_supported=False),
            lambda report: report.update(physics_work_modified=False),
            lambda report: report.update(physics_budgets_modified=True),
            lambda report: report.update(mjwarp_linesearch_fix=False),
        )
        for index, mutate in enumerate(mutations):
            self.invalid_checks = lambda report, mutate=mutate: (
                mutate(report) if report["mjwarp_linesearch_fix"] else None
            )
            output = self.root / f"bad-compat-{index}"
            with self.subTest(index=index), self.assertRaisesRegex(RuntimeError, "Checked capture"):
                self.invoke(
                    ["--repeats", "1", "--check-overflow", "--mjwarp-linesearch-fix", "--output-dir", str(output)]
                )
            self.assertFalse((output / "ratios.json").exists())

    def test_unapplied_broad_output_capacity_prevents_ratios(self):
        """Reject a clean overflow report that did not apply its requested constructor size."""
        self.invalid_checks = lambda report: report["collision_capacity"].update(pipelines=[])
        with self.assertRaisesRegex(RuntimeError, "requested broad-phase capacity"):
            self.invoke(
                [
                    "--repeats",
                    "1",
                    "--check-overflow",
                    "--capacity",
                    "anymald:fpgs:broad_phase_output_max=65536",
                ]
            )
        self.capture._read_result.assert_not_called()
        self.assertFalse((self.output / "ratios.json").exists())

    def test_malformed_failed_or_unbound_checks_prevent_ratios(self):
        """Reject failed flags, partial boundaries and another capture/source's clean report."""
        cases = [
            lambda report: report.update(complete=False),
            lambda report: report.update(check_pass=1),
            lambda report: report.update(boundary_count=1),
            lambda report: report["boundaries"].pop(),
            lambda report: report["boundaries"][1].update(check_pass=False),
            lambda report: report["boundaries"][0].update(physics="different-backend"),
            lambda report: report["boundaries"][0].pop("collision"),
            lambda report: report["boundaries"][0]["collision"].update(check_pass=False),
            lambda report: report["boundaries"][0]["collision"].update(status="failed"),
            lambda report: report["boundaries"][0]["collision"].update(status="not_applicable"),
            lambda report: report.update(run_profiled_sha256="wrong-source"),
            lambda report: report.update(checked_capture_sha256="wrong-wrapper"),
            lambda report: report.update(capture_output="/another/capture.json"),
            lambda report: report.update(physics_work_modified=True),
        ]
        for index, mutate in enumerate(cases):
            with self.subTest(case=index):
                self.invalid_checks = mutate
                destination = self.root / f"invalid-checks-{index}"
                with self.assertRaisesRegex(RuntimeError, "Checked capture"):
                    self.invoke(["--repeats", "1", "--output-dir", str(destination)])
                self.assertFalse((destination / "ratios.json").exists())
                self.assertEqual(json.loads((destination / "manifest.json").read_text())["status"], "failed")
        self.capture._read_result.assert_not_called()

    def test_paired_abba_recipes_manifests_and_ratios(self):
        """Preserve per-backend recipes, alternating order, source guards, and median ratios."""
        self.assertEqual(
            self.invoke(
                [
                    "--task",
                    "ant",
                    "--task",
                    "anymald",
                    "--fpgs-env",
                    "0:FEATHER_PGS_REGISTER_WHITENING=1",
                ]
            ),
            0,
        )
        self.assertEqual(len(self.calls), 24)
        for start in range(0, len(self.events), 4):
            self.assertEqual(
                [e[0] for e in self.events[start : start + 4]],
                ["start", "start", "wait", "wait"],
            )
        order = [self.calls[index][0][3] for index in range(0, len(self.calls), 2)]
        self.assertEqual(
            order,
            ["feather_pgs", "newton_mjwarp"] * 2
            + ["newton_mjwarp", "feather_pgs"] * 2
            + ["feather_pgs", "newton_mjwarp"] * 2,
        )
        for command, kwargs in self.calls:
            env, fpgs = kwargs["env"], command[3] == "feather_pgs"
            self.assertNotIn("FEATHER_PGS_POISON", env)
            self.assertNotIn("NEWTON_OTHER_POISON", env)
            self.assertEqual(env["PYTHONPATH"], str(self.fpgs if fpgs else self.mj))
            self.assertEqual(env["GPU"], env["CUDA_VISIBLE_DEVICES"])
            self.assertEqual(env["FPGS_NSYS_TRACE_MODE"], "graph")
            self.assertEqual(env["PYTHONDONTWRITEBYTECODE"], "1")
            self.assertEqual(command[command.index("--seed") + 1], "0")
            self.assertEqual(command[1], str(Path(driver.__file__).with_name("nsys_checked.sh")))
            self.assertIn("FPGS_BENCH_RUN_SHA256", env)
            self.assertNotIn("--physics-attr", command)
            if not fpgs:
                self.assertFalse(any(key.startswith(driver.FLAG_PREFIXES) for key in env))
                self.assertNotIn("--solver-attr", command)
            else:
                self.assertEqual(
                    env.get("FEATHER_PGS_REGISTER_WHITENING"),
                    "1" if env["GPU"] == "GPU-0" else None,
                )
                self.assertEqual(
                    "--solver-attr" in command,
                    command[4] == "Isaac-Velocity-Flat-AnymalD",
                )
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "complete")
        self.assertTrue(manifest["hostname"])
        self.assertEqual(manifest["settings"]["task"], ["anymald", "ant"])
        self.assertEqual(manifest["backends"], driver.BACKENDS)
        self.assertEqual(len(manifest["drivers"]), 7)
        self.assertEqual(set(manifest["source_checkouts"]), {"isaaclab", "fpgs", "mjwarp"})
        self.assertEqual(len(manifest["runs"]), 24)
        self.assertTrue(all(run["returncode"] == 0 for run in manifest["runs"]))
        self.assertEqual(self.capture._read_result.call_count, 24)
        values = {(row["task"], row["gpu"]): row for row in json.loads((self.output / "ratios.json").read_text())}
        self.assertEqual(values["ant", 0]["fpgs_speedup"], 4)
        self.assertEqual(values["ant", 1]["fpgs_speedup"], 3)
        self.assertTrue(all(row["all_states_finite"] for row in values.values()))

    def test_child_failure_is_recorded_without_ratios(self):
        """Keep failure artifacts and publish no final ratios after a child failure."""
        self.child_failure = True
        with self.assertRaisesRegex(RuntimeError, "Profiler failed"):
            self.invoke()
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(json.loads((self.output / "manifest.json").read_text())["status"], "failed")
        self.assertFalse((self.output / "ratios.json").exists())

    def test_keyboard_commands_preserve_both_backend_defaults(self):
        """Launch the SO101 typing task on both GPUs with no task-specific solver overrides."""
        self.argv[self.argv.index("--task") + 1] = "keyboard-so101"
        original = copy.deepcopy(self.capture.RECIPES)
        self.assertEqual(self.invoke(["--repeats", "1", "--num-envs", "32"]), 0)
        self.assertEqual(len(self.calls), 4)
        self.assertEqual({command[3] for command, _ in self.calls}, {"feather_pgs", "newton_mjwarp"})
        for command, kwargs in self.calls:
            self.assertEqual(command[4], "IsaacContrib-Keyboard-SO101")
            self.assertEqual(command[command.index("--num-envs") + 1], "32")
            self.assertNotIn("--solver-attr", command)
            self.assertNotIn("--physics-attr", command)
            self.assertNotIn("FEATHER_PGS_INK", kwargs["env"])
            self.assertNotIn("FEATHER_PGS_TIER_BLOCKS", kwargs["env"])
        self.assertEqual(self.capture.RECIPES, original)
        self.assertNotIn("keyboard-so101", self.capture.RECIPES)

    def test_source_drift_stops_before_second_backend(self):
        """Invalidate completed samples if either checkout changes during a batch."""
        self.source_drift = True
        with self.assertRaisesRegex(RuntimeError, "checkout changed"):
            self.invoke()
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(json.loads((self.output / "manifest.json").read_text())["status"], "failed")
        self.assertFalse((self.output / "ratios.json").exists())

    def test_helper_result_rejections_are_never_bypassed(self):
        """Propagate missing correlations, mixed modes, and nonfinite-state rejection."""
        for error in (
            "Missing CUDA graph launch correlations",
            "Expected graph timing",
            "Nonfinite simulation state",
        ):
            with self.subTest(error=error):
                self.result_error = error
                path = self.root / f"failure-{len(self.calls)}"
                with self.assertRaisesRegex(RuntimeError, error):
                    self.invoke(["--output-dir", str(path)])
                self.assertFalse((path / "ratios.json").exists())

    def test_existing_output_fails_before_runtime_or_gpu_query(self):
        """Refuse an existing output directory before invoking any external process."""
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            self.invoke()
        self.capture._gpus.assert_not_called()
        self.assertFalse(self.calls)

    def test_unknown_recipe_fails_before_gpu_query(self):
        """Reject task aliases not provided by the selected Lab checkout."""
        with self.assertRaisesRegex(ValueError, "Unknown task recipes"):
            self.invoke(["--task", "nonexistent"])
        self.capture._gpus.assert_not_called()
        self.assertFalse(self.calls)

    def test_incomplete_aggregation_has_no_ratio(self):
        """Reject missing or partial backend repetitions."""
        with self.assertRaisesRegex(RuntimeError, "Incomplete"):
            driver.ratios(
                [{"task": "ant", "gpu_index": 0, "newton": "fpgs", "repeats": 3}],
                ["ant"],
                [0],
                3,
            )


if __name__ == "__main__":
    unittest.main()
