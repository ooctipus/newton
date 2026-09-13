# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU-only provenance and failure-audit controls for the live probe runner."""

import json
import os
import sys
import tempfile
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from tools.fpgs_bench import checked_capture, kinetic_live_probe
from tools.fpgs_bench import run_kinetic_live_probe as runner


def module_at(name, filename=None, *, search_paths=None):
    """Provide an import record without executing any selected package."""
    module = ModuleType(name)
    module.__file__ = str(filename) if filename is not None else None
    if search_paths is not None:
        module.__path__ = [str(path) for path in search_paths]
    return module


class TestRunKineticLiveProbe(unittest.TestCase):
    def test_fixed_backend_with_original_core_and_tasks(self):
        """Select the fixed backend without replacing the task/core checkout."""
        root, lab, backend = Path("/selected/newton"), Path("/selected/lab"), Path("/selected/fixed-lab")
        files = {
            "isaaclab": lab / "source/isaaclab/isaaclab/__init__.py",
            "isaaclab_tasks": lab / "source/isaaclab_tasks/isaaclab_tasks/__init__.py",
            "isaaclab_newton": backend / "source/isaaclab_newton/isaaclab_newton/__init__.py",
        }
        modules = {name: module_at(name, path) for name, path in files.items()}
        with patch.dict(sys.modules, modules, clear=True):
            runner.verify_imports(root, {str(path): "pinned" for path in files.values()}, lab=lab, backend_lab=backend)
        modules["isaaclab_newton"] = module_at(
            "isaaclab_newton", lab / "source/isaaclab_newton/isaaclab_newton/__init__.py"
        )
        with patch.dict(sys.modules, modules, clear=True):
            with self.assertRaisesRegex(ValueError, "Mixed isaaclab_newton"):
                runner.verify_imports(root, {}, lab=lab, backend_lab=backend)

    def test_foreign_probe_helper_is_rejected(self):
        """An otherwise pinned foreign helper cannot bypass Newton selection."""
        root = Path("/selected/newton")
        foreign = Path("/foreign/newton/tools/fpgs_bench/checked_capture.py")
        module = module_at("tools.fpgs_bench.checked_capture", foreign)
        with patch.dict(sys.modules, {module.__name__: module}, clear=True):
            with self.assertRaisesRegex(ValueError, "Mixed"):
                runner.verify_imports(root, {str(foreign): "pinned"}, lab=Path("/selected/lab"))

    def test_foreign_lab_packages_are_rejected(self):
        """Each Lab package must come from its own selected source directory."""
        root, lab = Path("/selected/newton"), Path("/selected/lab")
        for package in ("isaaclab", "isaaclab_newton", "isaaclab_tasks"):
            for prefix in (Path("/foreign/lab/source"), lab / ".venv/lib/site-packages", lab / "source/other"):
                filename = prefix / package / "module.py"
                module = module_at(package + ".module", filename)
                with self.subTest(package=package, filename=filename):
                    with patch.dict(sys.modules, {module.__name__: module}, clear=True):
                        with self.assertRaisesRegex(ValueError, "Mixed " + package):
                            runner.verify_imports(root, {str(filename): "pinned"}, lab=lab)

    def test_protected_namespace_search_paths_cannot_escape(self):
        """Fileless or mixed-path package records cannot hide a foreign root."""
        root, lab = Path("/selected/newton"), Path("/selected/lab")
        for package, selected in (
            ("tools.fpgs_bench", root / "tools/fpgs_bench"),
            ("newton", root / "newton"),
            ("isaaclab", lab / "source/isaaclab/isaaclab"),
        ):
            for search_paths in (None, [selected, Path("/foreign/package")]):
                module = module_at(package, search_paths=search_paths)
                with self.subTest(package=package, search_paths=search_paths):
                    with patch.dict(sys.modules, {package: module}, clear=True):
                        with self.assertRaisesRegex(ValueError, "Mixed"):
                            runner.verify_imports(root, {}, lab=lab)

    def test_selected_packages_and_external_dependencies_are_allowed(self):
        """Do not mistake normal site packages or unrelated tools for a mix."""
        root, lab = Path("/selected/newton"), Path("/selected/lab")
        files = {
            "newton": root / "newton/__init__.py",
            "tools.fpgs_bench.checked_capture": root / "tools/fpgs_bench/checked_capture.py",
            **{
                name: lab / "source" / name / name / "__init__.py"
                for name in ("isaaclab", "isaaclab_newton", "isaaclab_tasks")
            },
            "warp": lab / ".venv/lib/site-packages/warp/__init__.py",
            "numpy": Path("/external/site-packages/numpy/__init__.py"),
            "torch": Path("/external/site-packages/torch/__init__.py"),
            "tools.other_helper": Path("/external/tools/other_helper.py"),
            "isaaclab_assets": Path("/external/site-packages/isaaclab_assets/__init__.py"),
        }
        modules = {name: module_at(name, path) for name, path in files.items()}
        modules["tools.fpgs_bench"] = module_at("tools.fpgs_bench", search_paths=[root / "tools/fpgs_bench"])
        pins = {str(path): "pinned" for path in files.values()}
        with patch.dict(sys.modules, modules, clear=True):
            runner.verify_imports(root, pins, lab=lab)
            with self.assertRaisesRegex(ValueError, "Unpinned local import"):
                runner.verify_imports(root, {}, lab=lab)

    def _mocked_run(self, *, reject_final_imports):
        """Exercise the real report finalizer with no Warp or Lab execution."""
        with tempfile.TemporaryDirectory(prefix="kinetic-probe-test-") as directory:
            base = Path(directory)
            root, lab = base / "newton", base / "lab"
            output, audit, pins = base / "profile.json", base / "audit.json", base / "pins.json"
            pins.write_text(json.dumps({str(Path(runner.__file__).resolve()): "pinned"}))
            reached_capture = False

            @contextmanager
            def observing(_solver, report, **_kwargs):
                """Supply only the exact host observations required by main."""
                report.update(calls=[{"step": step} for step in range(24)], force_exports=1)
                yield

            def capture(_argv):
                """Emit the ordinary recipe/capacity records without physics."""
                nonlocal reached_capture
                output.write_text(json.dumps({"decimation": 4, "sim_dt": 1 / 120, "cuda_graph": False}))
                (base / "capture_checks.json").write_text(
                    json.dumps({"complete": True, "check_pass": True, "boundary_count": 2})
                )
                reached_capture = True

            def imports(*_args, **_kwargs):
                """Fail only the post-capture import check, never initial setup."""
                if reached_capture and reject_final_imports:
                    raise ValueError("late foreign import")

            warp = ModuleType("warp")
            device = SimpleNamespace(uuid=runner.UUIDS[0], name="mock-only")
            warp.get_device = lambda _name: device
            warp.get_cuda_devices = lambda: [device]
            warp.__version__ = "mock-only"
            arguments = [
                "--newton",
                str(root),
                "--isaaclab",
                str(lab),
                "--backend-isaaclab",
                str(base / "fixed-lab"),
                "--output",
                str(output),
                "--audit-output",
                str(audit),
                "--pins-json",
                str(pins),
                "--expected-newton-commit",
                "selected",
            ]
            with ExitStack() as stack:
                stack.enter_context(patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": runner.UUIDS[0]}, clear=True))
                stack.enter_context(patch.dict(sys.modules, {"warp": warp}))
                stack.enter_context(patch.object(sys, "prefix", str(lab / ".venv")))
                stack.enter_context(patch.object(runner, "verify_source"))
                stack.enter_context(patch.object(runner, "verify_pins"))
                stack.enter_context(patch.object(runner, "verify_imports", side_effect=imports))
                stack.enter_context(patch.object(runner, "selected_import_paths"))
                stack.enter_context(patch.object(runner, "sha", return_value="pinned"))
                stack.enter_context(patch.object(checked_capture, "main", side_effect=capture))
                stack.enter_context(patch.object(kinetic_live_probe, "observing", side_effect=observing))
                stack.enter_context(
                    patch.object(
                        runner.importlib, "import_module", return_value=SimpleNamespace(SolverFeatherPGS=object)
                    )
                )
                if reject_final_imports:
                    with self.assertRaisesRegex(ValueError, "late foreign import"):
                        runner.main(arguments)
                else:
                    runner.main(arguments)
            return json.loads(audit.read_text())

    def test_late_import_failure_cannot_publish_success(self):
        """A post-capture exception remains a failed persisted audit."""
        report = self._mocked_run(reject_final_imports=True)
        self.assertTrue(report["source_guard_pass"])
        self.assertFalse(report["success"])
        self.assertFalse(report["complete"])
        self.assertIn("late foreign import", report["error"])

    def test_complete_mocked_run_is_explicitly_untimed(self):
        """Successful host validation still makes no timing/physics acceptance."""
        report = self._mocked_run(reject_final_imports=False)
        self.assertTrue(report["success"])
        self.assertTrue(report["complete"])
        for key in ("timing_accepted", "whole_physics_accepted", "contact_eight_accepted", "convergence_accepted"):
            self.assertFalse(report[key])

    def test_partial_observer_install_restores_owned_hooks(self):
        """A missing late hook restores every hook already installed."""
        solver = type("IncompleteSolver", (), {"step": lambda *args: None})
        first_name = kinetic_live_probe.RETIRED[0]

        def original(*_args):
            """Stand in for the first installed retired-stage hook."""

        setattr(solver, first_name, original)
        original_step = solver.step
        with self.assertRaises(AttributeError):
            with kinetic_live_probe.observing(solver, {}, worlds=512):
                self.fail("An incomplete observer must not enter its body")
        self.assertIs(solver.step, original_step)
        self.assertIs(getattr(solver, first_name), original)

    def test_import_path_restores_after_wrong_selection(self):
        """A stale editable backend fails before imports and restores sys.path."""
        previous = list(sys.path)
        with patch.object(runner.importlib.util, "find_spec", return_value=None):
            with self.assertRaisesRegex(ValueError, "Wrong selected package origin"):
                with runner.selected_import_paths(Path("/newton"), Path("/lab"), Path("/fixed-lab")):
                    self.fail("A wrong import selection must not enter the body")
        self.assertEqual(sys.path, previous)

    def test_only_exact_backend_interpreter_symlink_is_allowed(self):
        """An existing shared venv is permitted, never arbitrary dirty sources."""
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            backend, interpreter = base / "backend", base / "venv"
            backend.mkdir()
            interpreter.mkdir()
            (backend / ".venv").symlink_to(interpreter, target_is_directory=True)
            for dirty, allowed in (("?? .venv\n", True), ("?? .venv\n M code.py\n", False)):
                with patch.object(runner.subprocess, "check_output", side_effect=["selected\n", dirty]):
                    if allowed:
                        self.assertEqual(runner.verify_source(backend, "selected", interpreter=interpreter), "selected")
                    else:
                        with self.assertRaisesRegex(ValueError, "Changed/uncommitted"):
                            runner.verify_source(backend, "selected", interpreter=interpreter)


if __name__ == "__main__":
    unittest.main()
