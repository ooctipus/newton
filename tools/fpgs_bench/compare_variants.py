# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare two FPGS variants with checked, alternating paired-GPU captures.

Both arms use the same existing Isaac Lab recipe, capacities and seed zero.
Physics-graph and unprofiled environment wall times are reported separately.
Neither clean warning flags nor timing gains establish physical equivalence.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import statistics
import sys
import traceback
from pathlib import Path

import compare_backends as owner


def arm_order(rounds: int):
    """Alternate AB and BA while sampling both variants in every round."""
    for repeat in range(rounds):
        for label in ("baseline", "candidate") if repeat % 2 == 0 else ("candidate", "baseline"):
            yield repeat, label


def variant_flags(args: argparse.Namespace, label: str) -> dict:
    """Apply each explicitly requested variant flag identically to both GPUs."""
    entries = [f"{gpu}:{setting}" for gpu in args.gpus for setting in getattr(args, label + "_env")]
    return owner.parse_flags(entries, args.gpus)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Accept sampling and shared capacities, never arbitrary physics overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("isaaclab", "baseline", "candidate", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--baseline-env", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--candidate-env", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--capacity", action="append", default=[], metavar="TASK:fpgs:FIELD=VALUE")
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    parser.add_argument(
        "--seed", type=int, choices=(0,), default=0, help="The existing make_batch recipe fixes seed zero."
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--num-envs", type=int, default=16384)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--profile-steps", type=int, default=40)
    parser.add_argument("--trace-mode", choices=("graph", "node"), default="graph")
    args = parser.parse_args(argv)
    if min(args.rounds, args.num_envs, args.steps, args.profile_steps) < 1 or args.warmup_steps < 0:
        parser.error("Sample counts must be positive and warmup steps nonnegative")
    if len(args.gpus) != 2 or len(set(args.gpus)) != 2 or min(args.gpus) < 0:
        parser.error("Select exactly two distinct nonnegative GPU indices")
    for name in ("isaaclab", "baseline", "candidate", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.task = [args.task]
    capacities = owner.parse_capacities(args.capacity, args.task)
    if any(value["mjwarp"] for value in capacities.values()):
        raise ValueError("Both variants use FPGS; MJWarp capacity overrides are not applicable")
    args.check_overflow = True
    args.mjwarp_linesearch_fix = False
    for label in ("baseline", "candidate"):
        variant_flags(args, label)
    return args


def make_variant_batch(capture, args, label, devices, *, repeat, drivers):
    """Reuse the checked launch owner while keeping backend and variant distinct."""
    directory = args.output_dir / f"round_{repeat + 1:02d}_{label}"
    directory.mkdir()
    batch = owner.make_batch(
        capture,
        args,
        {"fpgs": getattr(args, label)},
        variant_flags(args, label),
        devices,
        directory,
        repeat,
        args.task[0],
        "fpgs",
        drivers=drivers,
    )
    for run in batch:
        run.update(newton=label, variant=label, source_root=str(getattr(args, label)))
        run["environment"]["FPGS_NSYS_TRACE_MODE"] = args.trace_mode
        run["environment"]["REPEATS"] = "1"
    return batch


def read_checked_result(capture, args, run, drivers):
    """Bind actual capture checks, timings and recorded budgets to each arm."""
    owner.check_overflow_result(run, drivers)
    result = capture._read_result(run)
    directory = Path(run["output_dir"])
    data = json.loads((directory / "capture.json").read_text())
    expected = {
        "task": owner.recipe_map(capture)[args.task[0]][0],
        "physics": "feather_pgs",
        "num_envs": args.num_envs,
        "steps": args.steps,
        "repeats": 1,
        "profile_steps": args.profile_steps,
        "cuda_graph": True,
        "physics_attr": [],
    }
    if any(data.get(name) != value for name, value in expected.items()):
        raise RuntimeError("Actual capture changed the selected recipe or sampling contract")
    fields = (
        "solver_class",
        "worlds",
        "num_substeps",
        "solver_dt",
        "pgs_mode",
        "pgs_iterations",
        "bodies",
        "joints",
        "joint_dofs",
        "joint_coords",
        "articulations",
        "rigid_contact_max",
    )
    before = {name: data["model"][name] for name in fields}
    after = {name: data["model_after"][name] for name in fields}
    if before != after or before["solver_class"] != "SolverFeatherPGS":
        raise RuntimeError("Actual solver budget/topology changed within the capture")
    if any(not math.isfinite(float(value)) or float(value) <= 0 for value in (before["solver_dt"], data["sim_dt"])):
        raise RuntimeError("Invalid actual simulation timestep")
    checks = run["overflow_check"]
    capacities = checks["boundaries"][0]["capacities"]
    if capacities != checks["boundaries"][1]["capacities"]:
        raise RuntimeError("Actual row capacities changed within the capture")
    fingerprint = {
        "model": before,
        "row_capacities": capacities,
        "sim_dt": data["sim_dt"],
        "decimation": data["decimation"],
        "solver_attr": data["solver_attr"],
    }
    if "collision_capacity" in checks:
        fingerprint["broad_capacity"] = [
            {name: pipeline[name] for name in ("full_input_pairs", "effective", "broad_phase_mode")}
            for pipeline in checks["collision_capacity"]["pipelines"]
        ]
    run["result"] = result
    run["budget_fingerprint"] = fingerprint
    run["artifacts"] = owner.file_hashes(
        [directory / name for name in ("capture.json", "capture_analysis.json", "capture_checks.json")]
    )
    return fingerprint


def summarize(runs: list[dict], gpus: list[int], rounds: int) -> list[dict]:
    """Report separate median physics and wall ratios only for complete paired data."""
    summary = []
    for gpu in gpus:
        arms = {}
        for label in ("baseline", "candidate"):
            selected = [run for run in runs if run["gpu_index"] == gpu and run["variant"] == label]
            if sorted(run["round"] for run in selected) != list(range(1, rounds + 1)):
                raise RuntimeError("Incomplete or duplicate variant rounds; no valid summary")
            values = {}
            for kind, key in (("physics", "graph_span_us_per_step"), ("wall", "wall_us_per_step")):
                samples = [float(run["result"][key]) for run in selected]
                if any(not math.isfinite(value) or value <= 0 for value in samples):
                    raise RuntimeError("Nonfinite or nonpositive timing sample")
                values[kind + "_values_us"] = samples
                values[kind + "_median_us"] = statistics.median(samples)
            arms[label] = values
        summary.append(
            {
                "gpu_index": gpu,
                **arms,
                "physics_baseline_over_candidate": arms["baseline"]["physics_median_us"]
                / arms["candidate"]["physics_median_us"],
                "wall_baseline_over_candidate": arms["baseline"]["wall_median_us"]
                / arms["candidate"]["wall_median_us"],
            }
        )
    return summary


def main(argv: list[str] | None = None) -> int:
    """Run checked paired arms directly, preserving first failure and final guards."""
    args = parse_args(argv)
    roots = {"isaaclab": args.isaaclab, "baseline": args.baseline, "candidate": args.candidate}
    tool = owner.tool_checkout()
    if tool is not None:
        roots["tool"] = tool
    owner.validate_output(args.output_dir, [*roots.values(), Path(__file__).resolve().parent])
    for root in (args.baseline, args.candidate):
        if not (root / "newton/__init__.py").is_file():
            raise ValueError(f"Not a Newton source checkout: {root}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running",
        "scope": __doc__,
        "capture_check_mode": "checked",
        "selected_backend": "feather_pgs",
        "settings": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "seed": 0,
        "same_capacity_overrides": args.capacity,
        "runs": [],
        "physical_quality_accepted": False,
        "performance_accepted": False,
        "physics_budgets_modified": False,
        "repeated_timing_evidence": False,
        "budget_scope": "Recorded public harness budgets/topology and actual capacity checks; not a proof of every generated algorithm loop or physical equivalence.",
    }
    capture = devices = sources = files = None
    failed = False

    def save():
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")

    save()
    try:
        capture = owner.load_capture(args.isaaclab)
        if args.task[0] not in owner.recipe_map(capture):
            raise ValueError(f"Unknown task recipe: {args.task[0]}")
        files = owner.file_hashes(
            [
                Path(__file__).resolve(),
                Path(owner.__file__).resolve(),
                Path(capture.__file__),
                *(capture.HARNESS / name for name in ("nsys_run.sh", "run_profiled.py", "analyze_nsys.py")),
                *(Path(owner.__file__).with_name(name) for name in ("checked_capture.py", "nsys_checked.sh")),
            ]
        )
        if files[str(Path(capture.__file__))] != capture._loaded_source_sha256:
            raise RuntimeError("Isaac Lab helper changed while loading")
        sources = {name: capture._source(root) for name, root in roots.items()}
        manifest.update(
            files=files,
            sources=sources,
            runtime=owner.runtime_software(args.isaaclab, {label: roots[label] for label in ("baseline", "candidate")}),
        )
        devices = capture._gpus(args.gpus)
        if sorted(device["index"] for device in devices) != sorted(args.gpus):
            raise RuntimeError("Paired device enumeration did not match the selected indices")
        manifest["gpus"] = devices
        fingerprint = None
        for repeat, label in arm_order(args.rounds):
            capture._require_idle(devices)
            owner.source_guard(capture, roots, sources, files)
            batch = make_variant_batch(capture, args, label, devices, repeat=repeat, drivers=files)
            manifest["runs"].extend(batch)
            save()
            owner.run_batch(batch, args.isaaclab)
            for run in batch:
                actual = read_checked_result(capture, args, run, files)
                if fingerprint is not None and actual != fingerprint:
                    raise RuntimeError("Variants/devices/rounds changed actual capacities or recorded physics budgets")
                fingerprint = actual
            manifest["budget_fingerprint"] = fingerprint
            owner.source_guard(capture, roots, sources, files)
            capture._require_idle(devices)
            save()
        artifacts = {path: digest for run in manifest["runs"] for path, digest in run.get("artifacts", {}).items()}
        if artifacts != owner.file_hashes([Path(path) for path in artifacts]):
            raise RuntimeError("Previously recorded capture artifacts changed during comparison")
        manifest.update(
            status="complete",
            summary=summarize(manifest["runs"], args.gpus, args.rounds),
            repeated_timing_evidence=args.rounds >= 3 and args.trace_mode == "graph",
        )
    except BaseException as error:
        failed = True
        manifest.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        errors = []
        for name, ready, check in (
            (
                "source",
                files is not None and sources is not None,
                lambda: owner.source_guard(capture, roots, sources, files),
            ),
            ("idle", devices is not None, lambda: capture._require_idle(devices)),
        ):
            try:
                if not ready:
                    raise RuntimeError("Guard unavailable after incomplete initialization")
                check()
                manifest["final_" + name + "_guard_pass"] = True
            except BaseException as error:
                errors.append({"guard": name, "error": repr(error)})
                manifest["final_" + name + "_guard_pass"] = False
        manifest["final_source_and_idle_guard_pass"] = not errors
        if errors:
            manifest.update(status="failed", final_guard_errors=errors, repeated_timing_evidence=False)
        save()
        if errors and not failed:
            raise RuntimeError(f"Final comparison guards failed: {errors}")
    print(json.dumps(manifest["summary"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, owner.interrupted)
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Comparison interrupted; owned profiler processes were stopped.", file=sys.stderr)
        raise SystemExit(130) from None
