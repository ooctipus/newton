# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare existing FPGS and MJWarp recipes using an unchanged Isaac Lab harness.

Physics-graph timings are not solver-accuracy, trajectory-parity, or training
throughput measurements. All benchmark artifacts must be outside source trees.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

BACKENDS = {"fpgs": "feather_pgs", "mjwarp": "newton_mjwarp"}
ADDITIONAL_RECIPES = {"keyboard-so101": ("IsaacContrib-Keyboard-SO101", (), {})}
HARNESS_RELATIVE = Path("scripts/benchmarks/fpgs_profile")
FLAG_PREFIXES = ("FEATHER_", "NEWTON_", "FPGS_PROBE_")
HELPERS = (
    "_source",
    "_gpus",
    "_require_idle",
    "_software",
    "_read_result",
    "_summaries",
    "_write_json",
)


def load_capture(isaaclab: Path) -> ModuleType:
    """Load the selected source bytes without writing an Isaac Lab bytecode cache."""
    path = isaaclab / HARNESS_RELATIVE / "compare_gpus.py"
    module = ModuleType("_fpgs_benchmark_capture")
    module.__file__ = str(path)
    source = path.read_bytes()
    exec(compile(source, str(path), "exec"), module.__dict__)
    for name in HELPERS:
        if not callable(getattr(module, name, None)):
            raise ValueError(f"Isaac Lab comparison helper lacks {name}: {path}")
    if module.REPO.resolve() != isaaclab or module.HARNESS.resolve() != path.parent:
        raise ValueError("Isaac Lab helper resolved a different source root")
    if not isinstance(module.RECIPES, dict) or not isinstance(module.COMMON_ENVIRONMENT, dict):
        raise ValueError("Isaac Lab helper has incompatible recipes/environment")
    module._loaded_source_sha256 = hashlib.sha256(source).hexdigest()
    return module


def recipe_map(capture: ModuleType) -> dict:
    """Merge Newton-only aliases without mutating Lab recipes or accepting conflicts."""
    recipes = copy.deepcopy(capture.RECIPES)
    for name, recipe in ADDITIONAL_RECIPES.items():
        if name in recipes and recipes[name] != recipe:
            raise ValueError(f"Conflicting Isaac Lab recipe for Newton benchmark alias {name!r}")
        recipes[name] = copy.deepcopy(recipe)
    return recipes


def parse_flags(entries: list[str], gpus: list[int]) -> dict[int, dict[str, str]]:
    """Accept only explicit, unique FPGS-side flags for selected GPU indices."""
    flags = {gpu: {} for gpu in gpus}
    for entry in entries:
        gpu, separator, setting = entry.partition(":")
        name, equals, value = setting.partition("=")
        if (
            not separator
            or not gpu.isdecimal()
            or int(gpu) not in flags
            or not equals
            or not re.fullmatch(r"(?:FEATHER_PGS_|NEWTON_NARROW_PHASE_)[A-Z0-9_]+", name)
            or "\0" in value
        ):
            raise ValueError(
                f"Expected selected GPU:FEATHER_PGS_NAME=VALUE or GPU:NEWTON_NARROW_PHASE_NAME=VALUE: {entry!r}"
            )
        if name in flags[int(gpu)]:
            raise ValueError(f"Duplicate per-GPU flag: {entry!r}")
        flags[int(gpu)][name] = value
    return flags


def clean_environment(overrides: dict[str, str], isaaclab: Path) -> dict[str, str]:
    """Discard inherited solver and project-selection flags before explicit setup."""
    removed = {
        "PYTHONPATH",
        "UV_PROJECT",
        "UV_PROJECT_ENVIRONMENT",
        "UV_WORKING_DIR",
        "UV_WORKING_DIRECTORY",
        "UV_NO_PROJECT",
        "UV_ISOLATED",
        "UV_ACTIVE",
        "UV_PYTHON",
        "VIRTUAL_ENV",
    }
    env = {key: value for key, value in os.environ.items() if not key.startswith(FLAG_PREFIXES) and key not in removed}
    env.update(UV_PROJECT=str(isaaclab), UV_NO_SYNC="1", PYTHONDONTWRITEBYTECODE="1")
    env.update(overrides)
    return env


def runtime_software(isaaclab: Path, roots: dict[str, Path]) -> dict:
    """Verify actual Lab runtime imports with CUDA hidden, without GPU initialization."""
    code = """import importlib.metadata as metadata
import json
from pathlib import Path
import sys
import newton
versions = {}
for name in ('warp-lang', 'torch', 'newton', 'mujoco', 'mujoco-warp'):
    try:
        versions[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        versions[name] = None
print(json.dumps({'python': sys.version, 'python_executable': sys.executable, 'python_prefix': sys.prefix,
                  'newton_file': str(Path(newton.__file__).resolve()), 'distributions': versions}))
"""
    result = {}
    for label, root in roots.items():
        info = json.loads(
            subprocess.check_output(
                ["uv", "run", "python", "-c", code],
                cwd=isaaclab,
                env=clean_environment(
                    {"PYTHONPATH": str(root), "CUDA_VISIBLE_DEVICES": "", "GPU": ""},
                    isaaclab,
                ),
                text=True,
            )
        )
        if Path(info["newton_file"]).resolve() != (root / "newton/__init__.py").resolve():
            raise RuntimeError(f"{label} runtime imported the wrong Newton source: {info['newton_file']}")
        if Path(info["python_prefix"]).resolve() != (isaaclab / ".venv").resolve():
            raise RuntimeError(f"{label} runtime did not use the prepared Isaac Lab .venv: {info['python_prefix']}")
        result[label] = info
    return result


def tool_checkout() -> Path | None:
    """Find the tool's owning checkout when installed inside Newton."""
    try:
        return Path(
            subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(Path(__file__).resolve().parent),
                    "rev-parse",
                    "--show-toplevel",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        ).resolve()
    except subprocess.CalledProcessError:
        return None


def validate_output(output: Path, roots: list[Path]) -> None:
    """Reject existing output and every location inside a source tree."""
    if any(output.is_relative_to(root) for root in roots):
        raise ValueError("Benchmark artifacts must be outside every source checkout and the tool directory")
    if output.exists():
        raise FileExistsError(f"Preserve existing benchmark output: {output}")


def file_hashes(paths: list[Path]) -> dict[str, str]:
    """Hash the exact driver, loaded helper, and invoked harness source bytes."""
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def source_guard(capture: ModuleType, roots: dict[str, Path], recorded: dict, files: dict[str, str]) -> None:
    """Fail on commit, staged/unstaged/untracked content, or driver-file changes."""
    if recorded != {name: capture._source(root) for name, root in roots.items()}:
        raise RuntimeError("Source checkout changed during comparison; these samples are invalid")
    if files != file_hashes([Path(path) for path in files]):
        raise RuntimeError("Benchmark driver/helper/harness changed during comparison")


def make_batch(
    capture: ModuleType,
    args: argparse.Namespace,
    roots: dict[str, Path],
    flags: dict,
    devices: list[dict],
    output: Path,
    repeat: int,
    task: str,
    backend: str,
) -> list[dict]:
    """Preserve the helper recipe and add FPGS-only flags without physics overrides."""
    task_name, attributes, recipe_flags = recipe_map(capture)[task]
    batch = []
    for gpu in devices:
        directory = output / f"round_{repeat + 1:02d}_{task}_{backend}_gpu{gpu['index']}"
        directory.mkdir()
        env = {
            key: value
            for key, value in capture.COMMON_ENVIRONMENT.items()
            if backend == "fpgs" or not key.startswith(FLAG_PREFIXES)
        }
        if backend == "fpgs":
            env.update(recipe_flags)
            env.update(flags[gpu["index"]])
        env.update(
            GPU=gpu["uuid"],
            CUDA_VISIBLE_DEVICES=gpu["uuid"],
            CUDA_DEVICE_ORDER="PCI_BUS_ID",
            PYTHONPATH=str(roots[backend]),
            OUT_DIR=str(directory),
            STEPS=str(args.steps),
            PROFILE_STEPS=str(args.profile_steps),
            FPGS_NSYS_TRACE_MODE="graph",
            UV_PROJECT=str(args.isaaclab),
            UV_NO_SYNC="1",
            PYTHONDONTWRITEBYTECODE="1",
        )
        command = [
            "bash",
            str(capture.HARNESS / "nsys_run.sh"),
            "capture",
            BACKENDS[backend],
            task_name,
            "--num-envs",
            str(args.num_envs),
            "--seed",
            "0",
            "--warmup-steps",
            str(args.warmup_steps),
        ]
        if backend == "fpgs":
            for attribute in attributes:
                command.extend(["--solver-attr", attribute])
        batch.append(
            {
                "round": repeat + 1,
                "task": task,
                "newton": backend,
                "backend": BACKENDS[backend],
                "gpu_index": gpu["index"],
                "gpu_uuid": gpu["uuid"],
                "command": command,
                "environment": env,
                "output_dir": str(directory),
                "driver_log": str(directory / "driver.log"),
            }
        )
    return batch


def signal_group(pgid: int, signum: int) -> bool:
    """Signal only a recorded owned group; treat absence, not denial, as completion."""
    try:
        os.killpg(pgid, signum)
        return True
    except ProcessLookupError:
        return False


def wait_group_gone(pgid: int, timeout: float) -> bool:
    """Wait a bounded grace period for all group members, not just their leader."""
    deadline = time.monotonic() + timeout
    while signal_group(pgid, 0):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(0.05, remaining))
    return True


def stop_children(processes: list[tuple[dict, subprocess.Popen]], *, grace_seconds: float = 10.0) -> None:
    """Terminate owned groups and reap direct children, including exited leaders."""
    failures = []
    for run, process in processes:
        try:
            # start_new_session makes this exact recorded PID the owned PGID.
            # An exited leader does not prove that profiler descendants exited.
            deadline = time.monotonic() + grace_seconds
            signaled = signal_group(process.pid, signal.SIGTERM)
            run["cleanup_signals"] = ["SIGTERM"] if signaled else []
            timed_out = False
            if process.poll() is None:
                try:
                    process.wait(timeout=grace_seconds)
                except subprocess.TimeoutExpired:
                    timed_out = True
            if timed_out or (signaled and not wait_group_gone(process.pid, max(0.0, deadline - time.monotonic()))):
                if signal_group(process.pid, signal.SIGKILL):
                    run["cleanup_signals"].append("SIGKILL")
            if process.poll() is None:
                process.wait(timeout=grace_seconds)
            run["returncode"] = process.returncode
        except BaseException as error:
            failures.append(f"PID {process.pid}: {error!r}")
    if failures:
        raise RuntimeError("Could not finish child cleanup: " + "; ".join(failures))


def run_batch(runs: list[dict], isaaclab: Path) -> None:
    """Launch all selected GPUs before waiting, and always clean up owned children."""
    processes = []
    with ExitStack() as logs:
        try:
            for run in runs:
                log = logs.enter_context(Path(run["driver_log"]).open("x"))
                process = subprocess.Popen(
                    run["command"],
                    cwd=isaaclab,
                    env=clean_environment(run["environment"], isaaclab),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                processes.append((run, process))
                run["pid"] = process.pid
                print(
                    f"Started {run['newton']} GPU {run['gpu_index']}: PID {process.pid}",
                    flush=True,
                )
            for run, process in processes:
                run["returncode"] = process.wait()
        finally:
            stop_children(processes)
    if any(run.get("returncode") != 0 for run in runs):
        raise RuntimeError("Profiler failed; inspect " + ", ".join(run["driver_log"] for run in runs))


def ratios(summaries: list[dict], tasks: list[str], gpus: list[int], repeats: int) -> list[dict]:
    """Report per-device median ratios only after both arms complete every round."""
    rows = []
    for task in tasks:
        for gpu in gpus:
            arms = {row["newton"]: row for row in summaries if row["task"] == task and row["gpu_index"] == gpu}
            if set(arms) != set(BACKENDS) or any(row["repeats"] != repeats for row in arms.values()):
                raise RuntimeError("Incomplete backend comparison; no final ratio is valid")
            fpgs, mjwarp = arms["fpgs"], arms["mjwarp"]
            rows.append(
                {
                    "task": task,
                    "gpu": gpu,
                    "fpgs_us": fpgs["median_us"],
                    "mjwarp_us": mjwarp["median_us"],
                    "fpgs_speedup": mjwarp["median_us"] / fpgs["median_us"],
                    "fpgs_wall_us": fpgs["wall_median_us"],
                    "mjwarp_wall_us": mjwarp["wall_median_us"],
                    "all_states_finite": fpgs["state_finite"] and mjwarp["state_finite"],
                }
            )
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Accept benchmark sampling options, not solver or physics parameter tuning."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("isaaclab", "fpgs", "mjwarp", "output-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    parser.add_argument(
        "--task",
        action="append",
        required=True,
        help="Recipe name; repeat for multiple tasks.",
    )
    parser.add_argument("--fpgs-env", action="append", default=[], metavar="GPU:NAME=VALUE")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--num-envs", type=int, default=16384)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--profile-steps", type=int, default=40)
    args = parser.parse_args(argv)
    if min(args.repeats, args.num_envs, args.steps, args.profile_steps) < 1 or args.warmup_steps < 0:
        parser.error("Sample counts must be positive and warmup steps nonnegative")
    if len(set(args.gpus)) != len(args.gpus) or any(gpu < 0 for gpu in args.gpus):
        parser.error("GPU indices must be distinct and nonnegative")
    for name in ("isaaclab", "fpgs", "mjwarp", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.task = list(dict.fromkeys(args.task))
    return args


def main(argv: list[str] | None = None) -> int:
    """Run alternating paired backend rounds with frozen sources and strict results."""
    args = parse_args(argv)
    capture = load_capture(args.isaaclab)
    recipes = recipe_map(capture)
    unknown = set(args.task) - set(recipes)
    if unknown:
        raise ValueError(f"Unknown task recipes {sorted(unknown)}; available: {sorted(recipes)}")
    flags = parse_flags(args.fpgs_env, args.gpus)
    roots = {"fpgs": args.fpgs, "mjwarp": args.mjwarp}
    for root in roots.values():
        if not (root / "newton" / "__init__.py").is_file():
            raise ValueError(f"Expected a Newton checkout with newton/__init__.py: {root}")
    guarded_roots = {"isaaclab": args.isaaclab, **roots}
    own_checkout = tool_checkout()
    if own_checkout is not None:
        guarded_roots["tool"] = own_checkout
    output = args.output_dir
    validate_output(output, [*guarded_roots.values(), Path(__file__).resolve().parent])
    files = [
        Path(__file__).resolve(),
        Path(capture.__file__).resolve(),
        *(capture.HARNESS / name for name in ("nsys_run.sh", "run_profiled.py", "analyze_nsys.py")),
    ]
    drivers = file_hashes(files)
    if drivers[str(Path(capture.__file__).resolve())] != capture._loaded_source_sha256:
        raise RuntimeError("Isaac Lab helper changed while loading")
    sources = {name: capture._source(root) for name, root in guarded_roots.items()}
    for executable in ("uv", "nsys", "nvidia-smi", "bash"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"Required executable is missing: {executable}")
    software = capture._software()
    software["backend_runtime"] = runtime_software(args.isaaclab, roots)
    devices = capture._gpus(args.gpus)
    capture._require_idle(devices)
    source_guard(capture, guarded_roots, sources, drivers)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "scope": "Existing task recipes; physics graphs only, not solver-accuracy, trajectory parity, or training throughput",
        "software": software,
        "isaaclab": sources["isaaclab"],
        "newton": {name: sources[name] for name in roots},
        "source_checkouts": sources,
        "backends": BACKENDS,
        "drivers": drivers,
        "gpus": devices,
        "settings": {
            key: value for key, value in vars(args).items() if key not in ("isaaclab", "fpgs", "mjwarp", "output_dir")
        },
        "output_dir": str(output),
        "runs": [],
        "status": "running",
    }
    print(f"Backend results: {output}", flush=True)
    try:
        capture._write_json(output / "manifest.json", manifest)
        for repeat in range(args.repeats):
            order = list(BACKENDS) if repeat % 2 == 0 else list(reversed(BACKENDS))
            for task in args.task:
                for backend in order:
                    capture._require_idle(devices)
                    source_guard(capture, guarded_roots, sources, drivers)
                    batch = make_batch(
                        capture,
                        args,
                        roots,
                        flags,
                        devices,
                        output,
                        repeat,
                        task,
                        backend,
                    )
                    manifest["runs"].extend(batch)
                    capture._write_json(output / "manifest.json", manifest)
                    print(
                        f"Round {repeat + 1}: {task} / {backend} on GPUs {args.gpus}",
                        flush=True,
                    )
                    run_batch(batch, args.isaaclab)
                    for run in batch:
                        run["result"] = capture._read_result(run)
                    source_guard(capture, guarded_roots, sources, drivers)
                    capture._write_json(output / "manifest.json", manifest)
                    capture._write_json(output / "summary.json", capture._summaries(manifest["runs"]))
        final_ratios = ratios(capture._summaries(manifest["runs"]), args.task, args.gpus, args.repeats)
        source_guard(capture, guarded_roots, sources, drivers)
        capture._write_json(output / "ratios.json", final_ratios)
        manifest["status"] = "complete"
    except BaseException as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        capture._write_json(output / "manifest.json", manifest)
        capture._write_json(output / "summary.json", capture._summaries(manifest["runs"]))
    for row in final_ratios:
        print(json.dumps(row, allow_nan=False), flush=True)
    return 0


def interrupted(signum, frame):
    """Unwind through process cleanup when the driver receives SIGTERM."""
    raise KeyboardInterrupt(f"Received signal {signum}")


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupted)
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(
            "Comparison interrupted; owned profiler processes were stopped.",
            file=sys.stderr,
        )
        raise SystemExit(130) from None
    except (
        OSError,
        ValueError,
        KeyError,
        RuntimeError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"Comparison failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
