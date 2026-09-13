# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run actual Lab/kinetic lifecycle checks or a boundary-only discovery capture.

The unchanged Lab core/tasks and harness supply actual models, states, collision
and control callbacks; only isaaclab_newton uses the fixed root-notification
backend. Eager timing includes diagnostic reads and is NOT performance evidence.
Graph/performance modes observe only existing untimed metadata boundaries.
No archived state or scratch experiment is imported by this runner.
"""

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path

UUIDS = (
    "GPU-883586b6-3100-0610-81e5-3b4c26f45639",
    "GPU-ebfac9e8-02d5-d8a9-3bfc-bac64c62ffd4",
)
LAB_COMMIT = "1d8feb82d17dbfab8f0772de56f84deae2cb7974"
BACKEND_COMMIT = "53ee6b44c2334341305dbdf385a3916c6b140799"
HARNESS_SHA = "586aecb18d95e317991f17e62fb2c9527b2eb76220d76d464bd4c67e9ec8b954"
FLAGS = {
    "FEATHER_PGS_GROUP_LANES": "16",
    "FEATHER_PGS_ROWS_MASKED": "1",
    "NEWTON_NARROW_PHASE_THREADS_X": "4",
    "FEATHER_PGS_SIMPLE_WORLD_ZERO": "1",
    "FEATHER_PGS_INDEPENDENT_COMPONENTS": "1",
    "FEATHER_PGS_PAIRED_GENERAL_OVERLAP": "1",
    "FEATHER_PGS_LOCAL_ROW_PACKETS": "1",
    "FEATHER_PGS_KUKA_JOINT_WORLD": "1",
    "FEATHER_PGS_WORLD_SCAN_PUBLICATION": "1",
    "FEATHER_PGS_KUKA_KINETIC_WORLD": "1",
}


def sha(path):
    """Hash the selected runtime or explicit supporting source."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_source(root, expected, *, interpreter=None):
    """Require the clean selected revision, not a moving import tree."""
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"], text=True
    )
    if dirty == "?? .venv\n" and interpreter is not None:
        link = root / ".venv"
        if link.is_symlink() and link.resolve() == Path(interpreter).resolve():
            dirty = ""
    if commit != expected or dirty:
        raise ValueError("Changed/uncommitted source: " + str(root))
    return commit


def verify_pins(pins):
    """Require every declared helper and runtime byte to remain unchanged."""
    if not pins:
        raise ValueError("Require explicit source pins")
    for path, expected in pins.items():
        if not Path(path).is_absolute() or sha(path) != expected:
            raise ValueError("Changed source pin: " + path)


def owned_virtual_module(root, pins, name, module):
    """Recognize only the two virtual objects created by pinned solvers.py."""
    owner = sys.modules.get("newton.solvers")
    fields = vars(owner) if owner is not None else {}
    source = root / "newton/solvers.py"
    if fields.get("__file__") is None or Path(fields["__file__"]).resolve() != source or str(source) not in pins:
        return False
    namespace = fields.get("experimental")
    if namespace is None:
        return False
    if name == "newton.solvers.experimental":
        return module is namespace and vars(namespace).get("__path__") == []
    return (
        name == "newton.solvers.experimental.coupled"
        and module is vars(namespace).get("coupled")
        and type(module) is fields.get("_LazyCoupledModule")
    )


def verify_imports(root, pins, *, lab, backend_lab=None, require_backend=False):
    """Bind owned packages to selected sources, allowing external dependencies."""
    root, lab = Path(root).resolve(), Path(lab).resolve()
    backend_lab = lab if backend_lab is None else Path(backend_lab).resolve()
    package_roots = {
        "newton": root / "newton",
        "tools.fpgs_bench": root / "tools" / "fpgs_bench",
        **{name: lab / "source" / name / name for name in ("isaaclab", "isaaclab_tasks")},
        "isaaclab_newton": backend_lab / "source/isaaclab_newton/isaaclab_newton",
    }
    found = {}
    for name, module in tuple(sys.modules.items()):
        # Do not invoke a lazy module's __getattr__ merely to audit its origin.
        fields = vars(module) if module is not None else {}
        filename = fields.get("__file__")
        if not filename and owned_virtual_module(root, pins, name, module):
            continue
        for package, selected in package_roots.items():
            if name == package or name.startswith(package + "."):
                # Namespace packages have no __file__; every search location
                # must still belong to the selected package, not a mixed tree.
                origins = ([filename] if filename else []) + list(fields.get("__path__", ()))
                if not origins or any(not Path(origin).resolve().is_relative_to(selected) for origin in origins):
                    raise ValueError("Mixed " + package + " import: " + repr(origins))
                break
        if not filename:
            continue
        path = Path(filename).resolve()
        if (
            (path.is_relative_to(root) or path.is_relative_to(package_roots["isaaclab_newton"]))
            and path.suffix == ".py"
            and str(path) not in pins
        ):
            raise ValueError("Unpinned local import: " + str(path))
        if str(path).startswith("/tmp/fpgs-"):
            raise ValueError("Live probe imported a scratch experiment: " + str(path))
        if name.startswith("isaaclab_newton."):
            found[name] = str(path)
    if require_backend:
        required = (
            "isaaclab_newton.physics.newton_manager",
            "isaaclab_newton.assets.articulation.articulation",
        )
        if any(name not in found for name in required):
            raise ValueError("The actual fixed manager/articulation was not observed")
    return found


@contextmanager
def selected_import_paths(root, lab, backend_lab):
    """Select fixed backend before imports and restore the caller's host path."""
    previous = list(sys.path)
    sys.path[:0] = [str(backend_lab / "source/isaaclab_newton"), str(root)]
    try:
        for name, expected in {
            "newton": root / "newton/__init__.py",
            "isaaclab_newton": backend_lab / "source/isaaclab_newton/isaaclab_newton/__init__.py",
            **{name: lab / "source" / name / name / "__init__.py" for name in ("isaaclab", "isaaclab_tasks")},
        }.items():
            spec = importlib.util.find_spec(name)
            if spec is None or spec.origin is None or Path(spec.origin).resolve() != expected:
                raise ValueError("Wrong selected package origin: " + name)
        yield
    finally:
        sys.path[:] = previous


def profile_arguments(worlds, output):
    """Keep task budgets fixed and select explicit untimed eager diagnostics."""
    return [
        "--task",
        "Isaac-Lift-KukaAllegro",
        "--physics",
        "feather_pgs",
        "--device",
        "cuda:0",
        "--num-envs",
        str(worlds),
        "--seed",
        "0",
        "--warmup-steps",
        "0",
        "--steps",
        "3",
        "--repeats",
        "1",
        "--profile-steps",
        "0",
        "--no-graph",
        "--no-nvtx",
        "--output",
        str(output),
    ]


def main(argv=None):
    """Observe cold/reuse/reset/request boundaries in actual live Newton calls."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("newton", "isaaclab", "backend-isaaclab", "output", "audit-output", "pins-json"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-newton-commit", required=True)
    parser.add_argument("--num-envs", type=int, choices=(256, 512, 16384), default=512)
    parser.add_argument("--perturb", action="store_true")
    parser.add_argument("--mode", choices=("eager", "graph", "performance"), default="eager")
    args = parser.parse_args(argv)
    if args.mode != "eager" and args.perturb:
        parser.error("Perturbations belong only to the eager correctness probe")
    if args.mode == "eager" and args.num_envs == 16384:
        parser.error("The eager probe is bounded to 256/512 worlds")
    root, lab = args.newton.resolve(), args.isaaclab.resolve()
    backend_lab = args.backend_isaaclab.resolve()
    output, audit = args.output.resolve(), args.audit_output.resolve()
    if (
        output.parent != audit.parent
        or output == audit
        or any(
            path.exists() or any(path.is_relative_to(tree) for tree in (root, lab, backend_lab))
            for path in (output, audit)
        )
    ):
        raise ValueError("Require fresh distinct paired outputs outside the checkouts")
    if Path(sys.prefix).resolve() != lab / ".venv":
        raise ValueError("Wrong Lab interpreter")
    lease = os.environ.get("CUDA_VISIBLE_DEVICES")
    if lease not in UUIDS:
        raise ValueError("Require one root-owned GPU UUID")
    extra = {
        k: v for k, v in os.environ.items() if k.startswith(("FEATHER_", "NEWTON_", "FPGS_PROBE_")) and k not in FLAGS
    }
    if extra or any(k in os.environ and os.environ[k] != v for k, v in FLAGS.items()):
        raise ValueError("Unrecorded/conflicting experiment flags: " + repr(extra))
    os.environ.update(FLAGS)
    pins = json.loads(args.pins_json.read_text())
    if str(Path(__file__).resolve()) not in pins:
        raise ValueError("The runner is not pinned")
    verify_pins(pins)
    pins[str(args.pins_json.resolve())] = sha(args.pins_json)
    verify_source(root, args.expected_newton_commit)
    verify_source(lab, LAB_COMMIT)
    verify_source(backend_lab, BACKEND_COMMIT, interpreter=lab / ".venv")
    report = {
        "success": False,
        "complete": False,
        "source_guard_pass": False,
        "scope": __doc__,
        "sources": pins,
        "flags": FLAGS,
        "commit": args.expected_newton_commit,
        "lab_core_tasks": {"path": str(lab), "commit": LAB_COMMIT},
        "lab_backend": {"path": str(backend_lab), "commit": BACKEND_COMMIT},
        "perturb": args.perturb,
        "mode": args.mode,
        "timing_accepted": False,
        "whole_physics_accepted": False,
        "contact_eight_accepted": False,
        "convergence_accepted": False,
        "profile_arguments": profile_arguments(args.num_envs, output) if args.mode == "eager" else None,
    }

    def save():
        """Persist both successful and failed diagnostic provenance."""
        audit.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    paths = selected_import_paths(root, lab, backend_lab)
    try:
        paths.__enter__()
        verify_imports(root, pins, lab=lab, backend_lab=backend_lab)
        import warp as wp  # noqa: PLC0415 - flags and runtime authority precede imports

        from tools.fpgs_bench import checked_capture, kinetic_live_measure, kinetic_live_probe  # noqa: PLC0415

        solver_module = importlib.import_module("newton._src.solvers.feather_pgs.solver_feather_pgs")
        verify_imports(root, pins, lab=lab, backend_lab=backend_lab)
        device = wp.get_device("cuda:0")
        if len(wp.get_cuda_devices()) != 1 or device.uuid != lease:
            raise ValueError("The actual GPU differs from the paired lease")
        report["hardware"] = {"uuid": device.uuid, "name": device.name, "warp": wp.__version__}
        checks_path = output.parent / "capture_checks.json"
        if args.mode == "eager":
            observer = kinetic_live_probe.observing(
                solver_module.SolverFeatherPGS, report, worlds=args.num_envs, perturb=args.perturb
            )
        else:
            report["profile_arguments"] = kinetic_live_measure.profile_arguments(args.num_envs, output, args.mode)
            observer = kinetic_live_measure.measuring(
                solver_module.SolverFeatherPGS, report, checked_capture, args.mode
            )
        with observer:
            checked_capture.main(
                [
                    "--isaaclab",
                    str(lab),
                    "--newton",
                    str(root),
                    "--expected-run-sha256",
                    HARNESS_SHA,
                    "--checks-output",
                    str(checks_path),
                    "--",
                    *report["profile_arguments"],
                ]
            )
        if args.mode == "eager" and (
            len(report["calls"]) != 24 or [call["step"] for call in report["calls"]] != list(range(24))
        ):
            raise ValueError("Expected all24 actual cold-through-step23 calls")
        if args.mode == "eager" and not report["force_exports"]:
            raise ValueError("The actual public force API was not exercised")
        if args.perturb and [item["event"] for item in report["perturbations"]] != [
            "force_target",
            "subset_reset",
            "odd_refresh",
            "world_gravity",
        ]:
            raise ValueError("A required lifecycle perturbation did not execute")
        if args.perturb and not report["gravity_perturbation"]["restored"]:
            raise ValueError("The diagnostic gravity input was not restored")
        result = json.loads(output.read_text())
        if result["decimation"] != 4 or result["sim_dt"] != 1 / 120 or result["cuda_graph"] != (args.mode != "eager"):
            raise ValueError("The unchanged task recipe changed")
        if args.mode != "eager" and (
            len(report["graph_boundaries"]) != 2 or not all(item["passed"] for item in report["graph_boundaries"])
        ):
            raise ValueError("Actual graph boundary checks did not pass")
        checks = json.loads(checks_path.read_text())
        if not checks["complete"] or not checks["check_pass"] or checks["boundary_count"] != 2:
            raise ValueError("Actual collision/solver capacity checks failed")
        report["fixed_backend_imports"] = verify_imports(
            root, pins, lab=lab, backend_lab=backend_lab, require_backend=True
        )
        report.update(
            checks=checks,
            complete=True,
            actual_eager_lifecycle_pass=args.mode == "eager",
            graph_boundary_pass=args.mode != "eager",
            public_fk_pass=args.mode != "performance",
            timing_accepted=args.mode == "performance",
        )
    except BaseException as error:
        report.update(complete=False, success=False, error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        paths.__exit__(None, None, None)
        try:
            verify_source(root, args.expected_newton_commit)
            verify_source(lab, LAB_COMMIT)
            verify_source(backend_lab, BACKEND_COMMIT, interpreter=lab / ".venv")
            verify_pins(pins)
            report["source_guard_pass"] = True
        except BaseException as error:
            report.update(complete=False, source_guard_error=repr(error))
            raise
        finally:
            report["success"] = report["complete"] and report["source_guard_pass"]
            save()


if __name__ == "__main__":
    main()
