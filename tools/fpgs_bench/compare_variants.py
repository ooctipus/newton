# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare two FPGS variants with checked, alternating paired-GPU captures.

Both arms use the same existing Isaac Lab recipe, capacities and seed zero.
Explicit per-arm solver representations may differ; numerical budgets may not.
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

RECIPE_TYPES = {
    "grouped_dynamics": bool,
    "lazy_kinematics": bool,
    "pgs_mode": str,
    "mf_gs_parallel_rows": int,
    "mf_gs_parallel_matrix_free": bool,
    "mf_gs_parallel_sweeps": int,
}
# Recorded with the recipe, never selected: the linked (patch-friction) tiers' budget request after its environment
# override (None, "auto" or a count), and the largest budget the built linked tiers sweep (0 without them).
LINKED_SWEEPS = "mf_gs_parallel_linked_sweeps"
LINKED_SWEEPS_APPLIED = "mf_gs_parallel_linked_sweeps_applied"


def linked_sweeps_recorded(recipe: dict) -> bool:
    """Require both linked-budget fields: a request of None, "auto" or a count, and an applied count."""
    request, applied = recipe.get(LINKED_SWEEPS), recipe.get(LINKED_SWEEPS_APPLIED)
    valid_request = request is None or request == "auto" or (type(request) is int and request >= 1)
    return LINKED_SWEEPS in recipe and valid_request and type(applied) is int and applied >= 0


def recipe_attributes(entries: list[str]) -> dict:
    """Parse only explicitly supported representation controls."""
    result = {}
    for entry in entries:
        name, separator, raw = entry.partition("=")
        if not separator or name not in RECIPE_TYPES or name in result:
            raise ValueError(f"Unknown or duplicate solver recipe: {entry}")
        kind = RECIPE_TYPES[name]
        if kind is bool and raw.lower() in ("true", "false"):
            value = raw.lower() == "true"
        elif kind is str and raw in ("split", "matrix_free"):
            value = raw
        elif kind is int and raw.isdecimal() and (1 if name.endswith("sweeps") else 0) <= int(raw) <= 128:
            value = int(raw)
        else:
            raise ValueError(f"Invalid solver recipe: {entry}")
        result[name] = value
    return result


def checked_recipe(args, run: dict, checks: dict, iterations: int) -> tuple[dict, dict]:
    """Verify actual per-arm settings and retain all nonselected invariants."""
    recipes = [boundary.get("solver_recipe") for boundary in checks["boundaries"]]
    if len(recipes) != 2 or any(
        not isinstance(recipe, dict)
        or any(type(recipe.get(name)) is not kind for name, kind in RECIPE_TYPES.items())
        or not linked_sweeps_recorded(recipe)
        for recipe in recipes
    ):
        raise RuntimeError("Missing or invalid actual solver recipe metadata")
    actual = recipes[0]
    if actual != recipes[1]:
        raise RuntimeError("Actual solver recipe changed within the capture")
    requested = args.solver_recipes[run["variant"]]
    if any(actual[name] != value for name, value in requested.items()):
        raise RuntimeError("Actual solver differs from the requested solver recipe")
    # Linked tiers sweep their own budget ("auto" by default), so an explicit parallel arm holds it to the
    # allowance too: set FEATHER_PGS_PATCH_PARALLEL_SWEEPS to pgs_iterations on such an arm.
    if (
        actual["pgs_mode"] == "matrix_free"
        and actual["mf_gs_parallel_rows"] > 0
        and (
            requested.get("pgs_mode") == "matrix_free" or any(name.startswith("mf_gs_parallel_") for name in requested)
        )
        and (actual["mf_gs_parallel_sweeps"] != iterations or actual[LINKED_SWEEPS_APPLIED] not in (0, iterations))
    ):
        raise RuntimeError("Explicit parallel sweeps must match the unchanged pgs_iterations budget")
    selected = set().union(*args.solver_recipes.values())
    # The request is each revision's own default; the applied linked budget is the physics budget both arms share,
    # unless an arm selects the parallel tiers and the check above holds its budget to the allowance.
    excluded = selected | {LINKED_SWEEPS}
    if any(name == "pgs_mode" or name.startswith("mf_gs_parallel_") for name in selected):
        excluded.add(LINKED_SWEEPS_APPLIED)
    return actual, {name: value for name, value in actual.items() if name not in excluded}


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
    for label in ("baseline", "candidate"):
        parser.add_argument(f"--{label}-solver-attr", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument(
        "--legacy-baseline-capacity",
        action="store_true",
        help="Use explicitly limited legacy diagnostics for pristine fpgs-main 65ab4b79.",
    )
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
    args.solver_recipes = {
        label: recipe_attributes(getattr(args, label + "_solver_attr")) for label in ("baseline", "candidate")
    }
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
        requested = args.solver_recipes[label]
        run["requested_solver_recipe"] = requested
        command, original, index = [], run["command"], 0
        while index < len(original):
            if original[index] == "--solver-attr" and original[index + 1].partition("=")[0] in requested:
                index += 2
            else:
                command.append(original[index])
                index += 1
        for name, value in requested.items():
            command.extend(["--solver-attr", f"{name}={value}"])
        run["command"] = command
        run["environment"]["FPGS_NSYS_TRACE_MODE"] = args.trace_mode
        run["environment"]["REPEATS"] = "1"
        if label == "baseline" and args.legacy_baseline_capacity:
            run["environment"]["FPGS_BENCH_LEGACY_CAPACITY"] = "1"
            run["capture_check_mode"] = "legacy_warning_screened_boundary_snapshots"
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
    actual_recipe, invariant_recipe = checked_recipe(args, run, checks, before["pgs_iterations"])
    if actual_recipe["pgs_mode"] != before["pgs_mode"]:
        raise RuntimeError("Solver recipe and Lab pgs_mode metadata disagree")
    run["solver_recipe"] = actual_recipe
    selected = set().union(*args.solver_recipes.values())
    capacities = checks["boundaries"][0]["capacities"]
    if capacities != checks["boundaries"][1]["capacities"]:
        raise RuntimeError("Actual row capacities changed within the capture")
    fingerprint = {
        "model": {name: value for name, value in before.items() if name not in selected},
        "solver_recipe": invariant_recipe,
        "row_capacities": capacities,
        "sim_dt": data["sim_dt"],
        "decimation": data["decimation"],
        "solver_attr": [value for value in data["solver_attr"] if value.partition("=")[0] not in selected],
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
        "capture_check_mode": "native_candidate_and_legacy_warning_screened_baseline"
        if args.legacy_baseline_capacity
        else "checked",
        "contact_law_parity_established": False,
        "selected_backend": "feather_pgs",
        "settings": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "seed": 0,
        "same_capacity_overrides": args.capacity,
        "requested_solver_recipes": args.solver_recipes,
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
        recipes = {}
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
                recipe = run.get("solver_recipe")
                if label in recipes and recipes[label] != recipe:
                    raise RuntimeError("Actual solver recipe changed across devices or rounds within one arm")
                recipes[label] = recipe
            manifest["budget_fingerprint"] = fingerprint
            manifest["actual_solver_recipes"] = recipes
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
