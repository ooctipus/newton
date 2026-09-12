# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test portable boundary checks and actual CPU collision constructor evidence."""

import enum
import importlib
import inspect
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import checked_capture as checked
import numpy as np


class FlagArray:
    """Provide the tiny host-read interface without a device runtime."""

    def __init__(self, values):
        self.values = values
        self.shape = (len(values),)
        self.device = SimpleNamespace(is_capturing=False)

    def numpy(self):
        """Return the immutable fake host-copy wrapper."""
        return self

    def tolist(self):
        """Return a copy so test helpers cannot modify the source flags."""
        return self.values.copy()


def fpgs(flags=None):
    """Create a duck-typed current FPGS checker with observable method calls."""
    solver = type("SolverFeatherPGS", (), {})()
    flags = flags or dict.fromkeys(checked.FPGS_FLAGS, False)
    solver.constraint_capacity_status = Mock(return_value=flags)
    solver.check_constraint_capacity = Mock()
    solver.dense_max_constraints, solver.mf_max_constraints, solver.propagation_max_constraints = 704, 64, 64
    return solver


class Bits(enum.IntFlag):
    """Name representative official overflow bits while rejecting every bit."""

    NEFC = 1
    ITERATIONS = 512
    LS_ITERATIONS = 1024


def mjwarp(values):
    """Create a full-DOF nonsleeping MJWarp owner with sticky test flags."""
    solver = type("SolverMuJoCo", (), {})()
    solver.use_mujoco_cpu, solver.enable_sleeping = False, False
    solver.mjw_data = SimpleNamespace(
        overflow=FlagArray(values), nworld=len(values), njmax=300, naconmax=2048, njmax_nnz=600
    )
    solver.mjw_model = SimpleNamespace(opt=SimpleNamespace(warn_overflow=True))
    solver._use_mujoco_contacts = False
    solver.contact_capacity_status = Mock(
        return_value={"negative_count": False, "source_contacts": False, "mjwarp_contacts": False}
    )
    solver.check_contact_capacity = Mock()
    return solver


def collision_pipeline(flags=None):
    """Provide a current rigid-only narrow-phase checker with observable method calls."""
    narrow = SimpleNamespace(
        verify_buffers=True,
        hydroelastic_sdf=None,
        buffer_capacity_status=Mock(return_value=flags or dict.fromkeys(checked.NARROW_FLAGS, False)),
        check_buffer_capacity=Mock(),
    )
    return SimpleNamespace(
        model=SimpleNamespace(particle_count=0),
        narrow_phase=narrow,
        contact_reduction_config=SimpleNamespace(body_pairs=False),
        _body_pair_reducer=None,
        _contact_matcher=None,
        contact_matching="disabled",
        _contact_sorter=None,
        deterministic=False,
    )


class Pipeline:
    """Exercise constructor signature preservation and original mode rejection."""

    def __init__(self, model=None, *, broad_phase_output_max=None, broad_phase="explicit"):
        if broad_phase != "explicit":
            raise ValueError("original constructor rejects unsupported mode")
        self.shape_pairs_filtered = list(range(100))
        self.shape_pairs_max = min(100, broad_phase_output_max or 100)
        self.broad_phase_mode = broad_phase
        self.narrow_phase = SimpleNamespace(
            split_gjk_mpr=False,
            sparse_gjk_pairs=True,
            max_candidate_pairs=self.shape_pairs_max,
            block_dim=64,
            total_num_threads=128,
        )


class TestCheckedCapture(unittest.TestCase):
    def test_sat_actual_cpu_constructor(self):
        """Verify both real constructor specializations and actual Warp pair arrays on CPU."""
        newton = importlib.import_module("newton")
        wp = importlib.import_module("warp")

        with wp.ScopedDevice("cpu"):
            builder = newton.ModelBuilder()
            for index in range(2):
                body = builder.add_body()
                builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, xform=wp.transform((index, 0, 0)))
            model = builder.finalize(device="cpu")
            for value in ("0", "1"):
                pipeline = newton.CollisionPipeline(model, broad_phase="explicit", box_box_sat=value == "1")
                result = {}
                checked.check_collision_sat(pipeline, value, result)
                self.assertTrue(result["check_pass"])
                self.assertIs(result["actual"], value == "1")
                self.assertTrue(result["primitive_module"].endswith("_sat" + value))
                self.assertEqual(result["shape_type_counts"], {"7": 2})
                self.assertEqual(result["filtered_pair_type_counts"], {"7:7": 1})

    def test_sat_main_binds_environment_and_rejects_drift(self):
        """Persist both boundary proofs and reject mutation of the explicit environment."""
        for value, drift in (("0", False), ("1", False), ("1", True)):
            with self.subTest(value=value, drift=drift), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                lab, newton, output = root / "lab", root / "newton", root / "out"
                path = lab / checked.HARNESS
                path.parent.mkdir(parents=True)
                path.write_text(
                    "import os, sys\nfrom pathlib import Path\n"
                    "def _model_meta(physics): return {'state_finite': True}\n"
                    "def main():\n"
                    "    for _ in range(2): _model_meta('feather_pgs')\n"
                    "    Path(sys.argv[sys.argv.index('--output')+1]).write_text('{}')\n"
                    + (f"    os.environ[{checked.SAT_ENV!r}] = '0'\n" if drift else "")
                )
                modules = {
                    "newton": SimpleNamespace(__file__=str(newton / "newton/__init__.py")),
                    "isaaclab_newton.physics": SimpleNamespace(
                        NewtonManager=SimpleNamespace(_solver=fpgs(), _collision_pipeline=self._sat_pipeline(value))
                    ),
                }
                args = [
                    "--isaaclab",
                    str(lab),
                    "--newton",
                    str(newton),
                    "--expected-run-sha256",
                    checked.sha(path),
                    "--checks-output",
                    str(output / "checks.json"),
                    "--",
                    "--physics",
                    "feather_pgs",
                    "--output",
                    str(output / "capture.json"),
                ]
                with patch.dict("sys.modules", modules), patch.dict(os.environ, {checked.SAT_ENV: value}):
                    if drift:
                        with self.assertRaisesRegex(RuntimeError, "environment changed"):
                            checked.main(args)
                    else:
                        self.assertEqual(checked.main(args), 0)
                report = json.loads((output / "checks.json").read_text())
                self.assertIs(report["complete"], not drift)
                self.assertIs(report["box_box_sat"]["check_pass"], not drift)
                self.assertIs(report["box_box_sat"]["requested"], value == "1")
                self.assertEqual(report["box_box_sat"]["environment_value"], value)
                self.assertIs(report["physics_work_modified"], value == "1")
                self.assertIs(report["physics_budgets_modified"], False)
                self.assertEqual(report["boundary_count"], 2)
                for boundary in report["boundaries"]:
                    self.assertIs(boundary["collision"]["box_box_sat"]["actual"], value == "1")

    def test_sat_request_checks_actual_module_and_current_pair_histogram(self):
        """Bind each explicit SAT choice to actual runtime specialization and pair types."""
        for value in ("0", "1"):
            pipeline = self._sat_pipeline(value)
            entry = {}
            checked.check_collision(
                "feather_pgs",
                SimpleNamespace(_solver=fpgs(), _collision_pipeline=pipeline),
                entry,
                sat_requested=value,
            )
            actual = entry["collision"]["box_box_sat"]
            self.assertIs(actual["actual"], value == "1")
            self.assertIs(actual["requested"], value == "1")
            self.assertTrue(actual["check_pass"])
            self.assertEqual(actual["shape_type_counts"], {"7": 2, "10": 1})
            self.assertEqual(actual["filtered_pair_type_counts"], {"7:7": 1, "7:10": 1})
            self.assertEqual(actual["filtered_pair_count"], 2)
            pipeline.narrow_phase.check_buffer_capacity.assert_called_once_with()

    def _sat_pipeline(self, value):
        """Provide immutable NumPy snapshots behind the real Warp array read interface."""
        pipeline = collision_pipeline()
        pipeline.narrow_phase.primitive_kernel = SimpleNamespace(
            module=SimpleNamespace(name=f"narrow_phase_primitive_write_contact_False_True_False_sat{value}")
        )
        pipeline.model.shape_count = 3
        pipeline.model.shape_type = SimpleNamespace(
            shape=(3,),
            device=SimpleNamespace(is_capturing=False),
            numpy=lambda: np.array([7, 7, 10], dtype=np.int32),
        )
        pipeline.shape_pairs_filtered = SimpleNamespace(
            shape=(2,),
            device=SimpleNamespace(is_capturing=False),
            numpy=lambda: np.array([[0, 1], [2, 1]], dtype=np.int32),
        )
        return pipeline

    def test_sat_request_rejects_wrong_missing_and_unreadable_evidence(self):
        """Reject ignored flags and malformed topology without accepting clean capacity flags."""
        for failure in ("wrong", "missing", "unknown", "capture", "pair_index", "pair_shape", "types", "request"):
            pipeline = self._sat_pipeline("1")
            requested = "1"
            if failure == "wrong":
                pipeline.narrow_phase.primitive_kernel.module.name = "narrow_phase_primitive_x_sat0"
            elif failure == "missing":
                del pipeline.narrow_phase.primitive_kernel
            elif failure == "unknown":
                pipeline.narrow_phase.primitive_kernel.module.name = "narrow_phase_primitive_x_sat1_unverified"
            elif failure == "capture":
                pipeline.model.shape_type.device.is_capturing = True
            elif failure == "pair_index":
                pipeline.shape_pairs_filtered.numpy = lambda: np.array([[0, 3], [1, 2]], dtype=np.int32)
            elif failure == "pair_shape":
                pipeline.shape_pairs_filtered.numpy = lambda: np.array([0, 1], dtype=np.int32)
            elif failure == "types":
                pipeline.model.shape_type.numpy = lambda: np.array([7.0, 7.0, 10.0])
            else:
                requested = "true"
            entry = {}
            with self.subTest(failure=failure), self.assertRaises(RuntimeError):
                checked.check_collision(
                    "feather_pgs",
                    SimpleNamespace(_solver=fpgs(), _collision_pipeline=pipeline),
                    entry,
                    sat_requested=requested,
                )
            self.assertFalse(entry["collision"]["check_pass"])

    def test_compat_install_model_admission_and_source_guard(self):
        """Install before execution and reject unsupported models or changed helper bytes."""
        for supported, drift in ((True, False), (False, False), (True, True)):
            with self.subTest(supported=supported, drift=drift), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                lab, newton, output = root / "lab", root / "newton", root / "out"
                path, helper = lab / checked.HARNESS, root / "compat.py"
                path.parent.mkdir(parents=True)
                helper.write_text(
                    "installed=False\n"
                    "def install():\n"
                    "    global installed\n    installed=True\n"
                    "    return dict(installed_package_modified=False, gradient_tolerance_changed=False, iteration_budget_changed=False)\n"
                    "def supports_model(model): return model.supported\n"
                )
                original_hash = checked.sha(helper)
                path.write_text(
                    "import sys\nfrom pathlib import Path\n"
                    "def _model_meta(physics): return {'state_finite': True}\n"
                    "def main():\n"
                    "    assert sys.modules['_newton_checked_mjwarp_linesearch'].installed\n"
                    "    for _ in range(2): _model_meta('newton_mjwarp')\n"
                    "    Path(sys.argv[sys.argv.index('--output')+1]).write_text('{}')\n"
                    + (f"    Path({str(helper)!r}).write_text('# changed')\n" if drift else "")
                )
                solver = mjwarp([0])
                solver.mjw_model.supported = supported
                modules = {
                    "newton": SimpleNamespace(__file__=str(newton / "newton/__init__.py")),
                    "mujoco_warp": SimpleNamespace(OverflowType=Bits),
                    "isaaclab_newton.physics": SimpleNamespace(
                        NewtonManager=SimpleNamespace(_solver=solver, _collision_pipeline=collision_pipeline())
                    ),
                }
                args = [
                    "--isaaclab",
                    str(lab),
                    "--newton",
                    str(newton),
                    "--expected-run-sha256",
                    checked.sha(path),
                    "--checks-output",
                    str(output / "checks.json"),
                    "--mjwarp-linesearch-fix",
                    "--",
                    "--physics",
                    "newton_mjwarp",
                    "--output",
                    str(output / "capture.json"),
                ]
                with patch.dict("sys.modules", modules), patch.object(checked, "COMPAT_PATH", helper):
                    if supported and not drift:
                        self.assertEqual(checked.main(args), 0)
                    else:
                        with self.assertRaisesRegex(RuntimeError, "actual model|helper changed"):
                            checked.main(args)
                report = json.loads((output / "checks.json").read_text())
                self.assertEqual(report["complete"], supported and not drift)
                self.assertIs(report["physics_work_modified"], True)
                self.assertIs(report["physics_budgets_modified"], False)
                self.assertEqual(report["mjwarp_linesearch_compat"]["helper_sha256"], original_hash)
                self.assertEqual(report["boundaries"][0]["mjwarp_linesearch_model_supported"], supported)

    def test_narrow_public_checker_and_no_clear(self):
        """Execute the real checker method after retaining its immutable flag snapshot."""
        pipeline, entry = collision_pipeline(), {}
        checked.check_collision("feather_pgs", SimpleNamespace(_solver=fpgs(), _collision_pipeline=pipeline), entry)
        pipeline.narrow_phase.buffer_capacity_status.assert_called_once_with()
        pipeline.narrow_phase.check_buffer_capacity.assert_called_once_with()
        self.assertTrue(entry["collision"]["check_pass"])

    def test_narrow_failure_is_saved_by_boundary_before_raise(self):
        """Persist a genuine narrow checker failure even when the FPGS rows fit."""
        flags = dict.fromkeys(checked.NARROW_FLAGS, False)
        flags["broad_phase"] = True
        pipeline = collision_pipeline(flags)
        pipeline.narrow_phase.check_buffer_capacity.side_effect = RuntimeError("lost broad pairs")
        manager = SimpleNamespace(_solver=fpgs(), _collision_pipeline=pipeline)
        report, saved = {"boundaries": [], "boundary_count": 0}, []
        harness = SimpleNamespace(_model_meta=lambda physics: {"state_finite": True})
        checked.install_boundary_check(
            harness, report, lambda: saved.append(json.loads(json.dumps(report))), lambda: manager
        )
        with self.assertRaisesRegex(RuntimeError, "lost broad pairs"):
            harness._model_meta("feather_pgs")
        self.assertTrue(saved[0]["boundaries"][0]["collision"]["flags"]["broad_phase"])
        self.assertFalse(saved[0]["boundaries"][0]["check_pass"])

    def test_narrow_missing_disabled_and_uncovered_owners_rejected(self):
        """Refuse unavailable history and exact pipeline owners outside this verifier's scope."""
        for failure in ("missing", "disabled", "particles", "hydro", "body_pairs", "matching", "sort"):
            pipeline = collision_pipeline()
            if failure == "missing":
                del pipeline.narrow_phase.check_buffer_capacity
            elif failure == "disabled":
                pipeline.narrow_phase.verify_buffers = False
            elif failure == "particles":
                pipeline.model.particle_count = 1
            elif failure == "hydro":
                pipeline.narrow_phase.hydroelastic_sdf = object()
            elif failure == "body_pairs":
                pipeline._body_pair_reducer = object()
            elif failure == "matching":
                pipeline.contact_matching = "latest"
            else:
                pipeline._contact_sorter = object()
            entry = {}
            with self.subTest(failure=failure), self.assertRaises(RuntimeError):
                checked.check_collision(
                    "feather_pgs", SimpleNamespace(_solver=fpgs(), _collision_pipeline=pipeline), entry
                )
            self.assertFalse(entry["collision"]["check_pass"])

    def test_only_native_mj_collision_can_have_no_newton_pipeline(self):
        """Distinguish an internal MJ owner from an accidentally missing external pipeline."""
        solver, entry = mjwarp([0]), {}
        solver._use_mujoco_contacts = True
        solver.mjw_model.opt.run_collision_detection = True
        manager = SimpleNamespace(_solver=solver, _collision_pipeline=None)
        checked.check_collision("newton_mjwarp", manager, entry)
        self.assertEqual(entry["collision"]["status"], "not_applicable")
        solver._use_mujoco_contacts = False
        with self.assertRaises(RuntimeError):
            checked.check_collision("newton_mjwarp", manager, {})
        with self.assertRaises(RuntimeError):
            checked.check_collision("feather_pgs", SimpleNamespace(_solver=fpgs(), _collision_pipeline=None), {})

    def test_constructor_injection_preserves_signature_and_rejects_conflicts(self):
        """Keep original validation and full input ownership, recording only constructed output bounds."""
        original, report = Pipeline.__init__, {}
        saved = []
        try:
            self.assertIs(checked.install_broad_phase_limit(Pipeline, 32, report, lambda: saved.append(True)), original)
            self.assertEqual(inspect.signature(Pipeline.__init__), inspect.signature(original))
            result = Pipeline()
            self.assertEqual(result.shape_pairs_max, 32)
            self.assertEqual(len(result.shape_pairs_filtered), 100)
            with self.assertRaisesRegex(ValueError, "Conflicting"):
                Pipeline(broad_phase_output_max=64)
            with self.assertRaisesRegex(ValueError, "original constructor"):
                Pipeline(broad_phase="sap")
            self.assertEqual(len(saved), 3)
            records = report["collision_capacity"]["pipelines"]
            self.assertTrue(records[0]["construction_pass"])
            self.assertEqual(records[0]["narrow_phase"]["max_candidate_pairs"], 32)
            self.assertFalse(records[1]["construction_pass"])
            self.assertFalse(records[2]["construction_pass"])
        finally:
            Pipeline.__init__ = original

    def test_actual_harness_execution_preserves_argv_and_requires_two_boundaries(self):
        """Exercise loading, forwarding, output, and strict boundary completion without a simulator."""
        for boundary_count in (0, 1, 2, 3):
            with self.subTest(boundary_count=boundary_count), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                lab, newton, output = root / "lab", root / "newton", root / "out"
                path = lab / checked.HARNESS
                path.parent.mkdir(parents=True)
                path.write_text(
                    "import json, sys, newton\nfrom pathlib import Path\n"
                    "def _model_meta(physics): return {'state_finite': True}\n"
                    "def main():\n"
                    "    newton.CollisionPipeline()\n"
                    f"    for _ in range({boundary_count}): _model_meta('feather_pgs')\n"
                    "    p = Path(sys.argv[sys.argv.index('--output') + 1])\n"
                    "    p.write_text(json.dumps({'argv': sys.argv[1:]}))\n"
                )
                solver = fpgs()
                modules = {
                    "newton": SimpleNamespace(__file__=str(newton / "newton/__init__.py"), CollisionPipeline=Pipeline),
                    "isaaclab_newton.physics": SimpleNamespace(
                        NewtonManager=SimpleNamespace(_solver=solver, _collision_pipeline=collision_pipeline())
                    ),
                }
                forwarded = ["--output", str(output / "capture.json"), "--solver-attr", "pgs_iterations=8"]
                argv = [
                    "--isaaclab",
                    str(lab),
                    "--newton",
                    str(newton),
                    "--expected-run-sha256",
                    checked.sha(path),
                    "--checks-output",
                    str(output / "checks.json"),
                    "--broad-phase-output-max",
                    "32",
                    "--",
                    *forwarded,
                ]
                original_argv = sys.argv
                original_constructor = Pipeline.__init__
                with patch.dict("sys.modules", modules):
                    if boundary_count == 2:
                        self.assertEqual(checked.main(argv), 0)
                        self.assertEqual(json.loads((output / "capture.json").read_text())["argv"], forwarded)
                        solver.check_constraint_capacity.assert_has_calls([unittest.mock.call(), unittest.mock.call()])
                    else:
                        with self.assertRaisesRegex(RuntimeError, "boundary|boundaries"):
                            checked.main(argv)
                self.assertIs(sys.argv, original_argv)
                self.assertIs(Pipeline.__init__, original_constructor)
                report = json.loads((output / "checks.json").read_text())
                self.assertEqual(report["complete"], boundary_count == 2)
                self.assertEqual(report["check_pass"], boundary_count == 2)
                self.assertEqual(report["boundary_count"], boundary_count)
                self.assertEqual(report["collision_capacity"]["pipelines"][0]["effective"], 32)

    def test_shell_forwards_profile_budget_and_uses_unchanged_analyzer(self):
        """Run the real shell against inert command stubs and inspect its exact argv."""
        for options in (
            ["--broad-phase-output-max", "32768", "--mjwarp-linesearch-fix"],
            ["--mjwarp-linesearch-fix", "--broad-phase-output-max", "32768"],
        ):
            with self.subTest(options=options):
                self._run_shell_forwarding(options)

    def _run_shell_forwarding(self, options):
        """Check either wrapper-flag ordering without invoking a GPU or simulator."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bin_dir, lab, output = root / "bin", root / "lab", root / "out"
            bin_dir.mkdir()
            lab.mkdir()
            record = root / "calls.jsonl"
            source = (
                f"#!{sys.executable}\n"
                "import json, os, sys\n"
                "with open(os.environ['RECORD'], 'a') as f: f.write(json.dumps(sys.argv) + '\\n')\n"
                "print('RESULT inert-command-stub')\n"
            )
            for command in ("nsys", "uv"):
                path = bin_dir / command
                path.write_text(source)
                path.chmod(0o755)
            env = {
                **os.environ,
                "PATH": f"{bin_dir}:{os.environ['PATH']}",
                "RECORD": str(record),
                "FPGS_BENCH_ISAACLAB": str(lab),
                "FPGS_BENCH_NEWTON": str(root / "newton"),
                "FPGS_BENCH_RUN_SHA256": "a" * 64,
                "OUT_DIR": str(output),
                "GPU": "test-uuid",
                "STEPS": "40",
                "PROFILE_STEPS": "17",
                "REPEATS": "3",
                "FPGS_NSYS_TRACE_MODE": "graph",
            }
            subprocess.run(
                [
                    "bash",
                    str(Path(checked.__file__).with_name("nsys_checked.sh")),
                    "capture",
                    "newton_mjwarp",
                    "Task",
                    *options,
                    "--solver-attr",
                    "pgs_iterations=8",
                ],
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            calls = [json.loads(line) for line in record.read_text().splitlines()]
            self.assertEqual(len(calls), 3)
            profile, export, analyzer = calls
            self.assertEqual(profile[1:4], ["profile", "--capture-range=cudaProfilerApi", "--capture-range-end=stop"])
            self.assertIn("--cuda-graph-trace=graph:host-only", profile)
            self.assertEqual(profile[profile.index("--checks-output") + 1], str(output / "capture_checks.json"))
            self.assertEqual(profile[profile.index("--broad-phase-output-max") + 1], "32768")
            self.assertLess(profile.index("--broad-phase-output-max"), profile.index("--"))
            self.assertLess(profile.index("--mjwarp-linesearch-fix"), profile.index("--"))
            forwarded = profile[profile.index("--") + 1 :]
            self.assertEqual(
                forwarded,
                [
                    "--physics",
                    "newton_mjwarp",
                    "--task",
                    "Task",
                    "--device",
                    "cuda:0",
                    "--repeats",
                    "3",
                    "--steps",
                    "40",
                    "--profile-steps",
                    "17",
                    "--output",
                    str(output / "capture.json"),
                    "--solver-attr",
                    "pgs_iterations=8",
                ],
            )
            self.assertEqual(export[1:4], ["export", "--type", "sqlite"])
            self.assertEqual(analyzer[1:4], ["run", "python", str(lab / checked.HARNESS.parent / "analyze_nsys.py")])
            self.assertIn("--require-graph-trace", analyzer)

    def test_fpgs_public_checker_is_executed_and_never_cleared(self):
        """Require the actual public method even when the snapshot is clean."""
        solver, entry = fpgs(), {}
        checked.check_solver("feather_pgs", solver, entry)
        solver.constraint_capacity_status.assert_called_once_with()
        solver.check_constraint_capacity.assert_called_once_with()
        self.assertTrue(entry["check_pass"])

    def test_fpgs_failure_records_flags_before_raising(self):
        """Retain overflow evidence and reject even a defective no-op checker."""
        flags = dict.fromkeys(checked.FPGS_FLAGS, False)
        flags["dense"] = True
        solver, entry = fpgs(flags), {}
        with self.assertRaises(RuntimeError):
            checked.check_solver("feather_pgs", solver, entry)
        self.assertTrue(entry["flags"]["dense"])
        solver.check_constraint_capacity.assert_called_once_with()

    def test_missing_legacy_api_is_not_clean(self):
        """Fail checked mode explicitly on a revision without sticky visibility."""
        solver = fpgs()
        del solver.check_constraint_capacity
        with self.assertRaisesRegex(RuntimeError, "lacks"):
            checked.check_solver("feather_pgs", solver, {})

    def test_all_mj_bits_including_unknown_are_rejected(self):
        """Reject capacity, outer-iteration, line-search, and future unknown bits."""
        with patch.dict("sys.modules", {"mujoco_warp": SimpleNamespace(OverflowType=Bits)}):
            for bit in (1, 512, 1024, 1 << 20):
                entry = {}
                with self.subTest(bit=bit), self.assertRaises(RuntimeError):
                    checked.check_solver("newton_mjwarp", mjwarp([0, bit]), entry)
                self.assertEqual(entry["flags"]["mask"], bit)
                self.assertEqual(entry["flags"]["worlds_nonzero"], 1)
            entry = {}
            checked.check_solver("newton_mjwarp", mjwarp([0, 0]), entry)
            self.assertTrue(entry["check_pass"])

    def test_mj_suppression_and_resettable_history_are_rejected(self):
        """Do not label disabled warnings or resettable sticky history clean."""
        for field in ("warnings", "sleeping"):
            solver = mjwarp([0])
            if field == "warnings":
                solver.mjw_model.opt.warn_overflow = False
            else:
                solver.enable_sleeping = True
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                checked.check_solver("newton_mjwarp", solver, {})

    def test_external_mj_prefix_loss_rejected_with_zero_native_bits(self):
        """Require the converter checker, including native-not-applicable records."""
        with patch.dict("sys.modules", {"mujoco_warp": SimpleNamespace(OverflowType=Bits)}):
            for flag in ("negative_count", "source_contacts", "mjwarp_contacts"):
                solver = mjwarp([0])
                solver.contact_capacity_status.return_value[flag] = True
                entry = {}
                with self.subTest(flag=flag), self.assertRaisesRegex(RuntimeError, "contact"):
                    checked.check_solver("newton_mjwarp", solver, entry)
                self.assertTrue(entry["external_contact_conversion"]["flags"][flag])
                solver.check_contact_capacity.assert_called_once_with()
            for missing in ("contact_capacity_status", "check_contact_capacity"):
                solver = mjwarp([0])
                delattr(solver, missing)
                with self.subTest(missing=missing), self.assertRaisesRegex(RuntimeError, "lacks"):
                    checked.check_solver("newton_mjwarp", solver, {})
            for native in (False, True):
                solver, entry = mjwarp([0]), {}
                solver._use_mujoco_contacts = native
                checked.check_solver("newton_mjwarp", solver, entry)
                solver.check_contact_capacity.assert_called_once_with()
                self.assertEqual(
                    entry["external_contact_conversion"]["status"], "not_applicable" if native else "checked"
                )

    def test_boundary_wrapper_saves_failure_outside_swallowing_lab_helper(self):
        """Persist the second-boundary failure even when Lab metadata succeeds."""
        solver, report, saved = fpgs(), {"boundaries": [], "boundary_count": 0}, []
        harness = SimpleNamespace(_model_meta=lambda physics: {"state_finite": True})
        manager = SimpleNamespace(_solver=solver, _collision_pipeline=collision_pipeline())
        checked.install_boundary_check(
            harness, report, lambda: saved.append(json.loads(json.dumps(report))), lambda: manager
        )
        self.assertTrue(harness._model_meta("feather_pgs")["overflow_check"]["check_pass"])
        solver.check_constraint_capacity.side_effect = RuntimeError("capacity omitted constraints")
        with self.assertRaisesRegex(RuntimeError, "omitted"):
            harness._model_meta("feather_pgs")
        self.assertEqual(report["boundary_count"], 2)
        self.assertEqual(len(saved), 2)
        self.assertFalse(saved[-1]["boundaries"][1]["check_pass"])

    def test_source_pin_failure_writes_unaccepted_report(self):
        """Preserve a failure artifact before importing any simulation module."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lab = root / "lab"
            path = lab / checked.HARNESS
            path.parent.mkdir(parents=True)
            path.write_text("raise AssertionError('must not execute wrong bytes')\n")
            output = root / "out"
            args = [
                "--isaaclab",
                str(lab),
                "--newton",
                str(root / "newton"),
                "--expected-run-sha256",
                "0" * 64,
                "--checks-output",
                str(output / "checks.json"),
                "--",
                "--output",
                str(output / "capture.json"),
            ]
            with self.assertRaisesRegex(RuntimeError, "parent-pinned"):
                checked.main(args)
            report = json.loads((output / "checks.json").read_text())
            self.assertFalse(report["complete"])
            self.assertFalse(report["check_pass"])
            self.assertEqual(report["boundary_count"], 0)


if __name__ == "__main__":
    unittest.main()
