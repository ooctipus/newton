# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Profiling for the Newton repro: FPS metering + an Nsight Systems top-k kernel report.

``--profile`` re-execs the run under nsys (graphed, full speed), captures only the steady-state
loop (cudaProfilerApi range), then prints the top-k GPU kernels labelled with their warp module.
``FpsMeter`` prints per-window + summary throughput on every run and owns that capture range.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

import warp as wp

wp.config.quiet = True  # mute Warp's init banner + per-module "... load on device ... took N ms" lines

FPS_EVERY = 50  # print throughput every N control steps
NSYS_INNER = "_REPRO_NSYS_INNER"  # env guard set on the re-exec'd (profiled) child
_KMAP_PATH = os.path.join(tempfile.gettempdir(), "repro_nsys.kmap.json")  # kernel-key -> python module
_REPORT = os.path.join(tempfile.gettempdir(), "repro_nsys")


def add_args(parser: argparse.ArgumentParser) -> None:
    """Add the profiling CLI flags to an existing parser."""
    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="Profile the graphed run under Nsight Systems and print the top-k GPU kernels at exit.",
    )
    parser.add_argument("--topk", type=int, default=15, help="Number of kernels to report with --profile.")


def under_nsys() -> bool:
    """True inside the nsys-profiled child process."""
    return bool(os.environ.get(NSYS_INNER))


def maybe_run_under_nsys(args) -> int | None:
    """If ``--profile`` and not already profiling, re-exec under nsys and return its exit code; else ``None``."""
    if not args.profile or under_nsys():
        return None
    if shutil.which("nsys") is None:
        print("[nsys] not found on PATH; install Nsight Systems.", file=sys.stderr)
        return 1
    for path in (f"{_REPORT}.nsys-rep", f"{_REPORT}.sqlite", _KMAP_PATH):
        if os.path.exists(path):
            os.remove(path)
    cmd = [
        "nsys", "profile",
        "--sample=none",  # only want CUDA kernel timings; skip CPU sampling (avoids a perms warning)
        "--cpuctxsw=none",
        "--force-overwrite=true",
        "--trace=cuda",
        "--cuda-graph-trace=node",  # itemize kernels inside the CUDA graph
        "--capture-range=cudaProfilerApi",  # only the steady-state loop (FpsMeter brackets it)
        "--capture-range-end=stop",
        "-o", _REPORT,
        sys.executable, *sys.argv,  # re-run the exact same command (sys.argv[0] is repro.py)
    ]  # fmt: skip
    # Stream the child's merged output live, dropping nsys's own chatter (capture-range + report-export
    # progress) and the benign USD inertia warnings; real prints/tracebacks pass through. The child is
    # run unbuffered + read line-by-line so the config table / stage markers / [fps] appear as they happen.
    noise = ("Capture range ", "Processing events", "Generated:", "repro_nsys", "possibly invalid inertia")  # fmt: skip
    print("[nsys] profiling under Nsight Systems (graphed, full speed)...", flush=True)
    proc = subprocess.Popen(
        cmd,
        env={**os.environ, NSYS_INNER: "1", "PYTHONUNBUFFERED": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in iter(proc.stdout.readline, ""):
        if "Generating '" in line:
            print("[nsys] generating report (may take a while for large runs)...", flush=True)
        elif not any(token in line for token in noise):
            sys.stdout.write(line)
            sys.stdout.flush()
    proc.wait()
    print("[nsys] computing top kernels from report...", flush=True)
    stats = subprocess.run(
        ["nsys", "stats", "--force-export=true", "--report", "cuda_gpu_kern_sum",
         "--format", "csv", f"{_REPORT}.nsys-rep"],
        capture_output=True,
        text=True,
    )  # fmt: skip
    kmap = {}
    if os.path.exists(_KMAP_PATH):
        with open(_KMAP_PATH) as f:
            kmap = json.load(f)
    _print_top_kernels(stats.stdout, args.topk, kmap)
    return 0


def _dump_kernel_modules(path: str) -> None:
    """Write ``{kernel_key: python_module}`` for the loaded kernels (used to label the top-k table).

    Reads each kernel function's ``__module__`` (the real defining module) rather than the Warp
    module name, which is synthetic for dynamically-generated/closure kernels.
    """
    from warp._src import context as wc

    kmap = {}
    for mod in wc.user_modules.values():
        for key, kern in getattr(mod, "kernels", {}).items():
            kmap[key] = getattr(getattr(kern, "func", None), "__module__", None) or mod.name
    with open(path, "w") as f:
        json.dump(kmap, f)


def _clean_kernel(name: str) -> str:
    """Strip Warp's ``_<hash>_cuda_kernel_(forward|backward)`` suffix from a kernel symbol."""
    return re.sub(r"_[0-9a-f]{6,}_cuda_kernel_(forward|backward)$", "", name)


def _ellipsize(s: str, width: int) -> str:
    return s if len(s) <= width else s[: width - 1] + "…"


def _mid_ellipsize(s: str, width: int) -> str:
    if len(s) <= width:
        return s
    head = (width - 1) // 2
    return s[:head] + "…" + s[-(width - 1 - head) :]


def _print_top_kernels(stats_csv: str, topk: int, kmap: dict) -> None:
    """Print the top-k CUDA kernels by total GPU time as an aligned table, labelled by python module."""
    lines = stats_csv.splitlines()
    header = next((i for i, ln in enumerate(lines) if "Total Time (ns)" in ln and "Name" in ln), None)
    if header is None:
        print("[nsys] could not parse kernel summary:\n" + stats_csv[:500], file=sys.stderr)
        return
    keys = sorted(kmap, key=len, reverse=True)  # longest key that is a substring of the symbol wins

    rows = []
    for r in list(csv.DictReader(lines[header:]))[:topk]:
        module, kernel = next(((kmap[k], k) for k in keys if k in r["Name"]), ("?", _clean_kernel(r["Name"])))
        rows.append(
            (r.get("Time (%)", "?"), float(r["Total Time (ns)"]) / 1e6, r.get("Instances", "?"), module, kernel)
        )

    mod_w = 36
    print(f"\n=== top {len(rows)} GPU kernels by total GPU time ===")
    print(f"  {'#':>2}  {'time%':>6}  {'total ms':>9}  {'calls':>8}  {'module':<{mod_w}}  kernel")
    for i, (pct, ms, calls, module, kernel) in enumerate(rows, 1):
        mod = _ellipsize(module, mod_w)
        print(f"  {i:>2}  {pct:>5}%  {ms:>9.1f}  {calls:>8}  {mod:<{mod_w}}  {_mid_ellipsize(kernel, 52)}")


class FpsMeter:
    """Per-window FPS printer; under nsys it also brackets the cudaProfiler capture range.

    Call :meth:`tick` once per executed control step and :meth:`finish` after the loop.
    """

    def __init__(self, num_envs: int, device: str, every: int = FPS_EVERY):
        self.num_envs = num_envs
        self.device = device
        self.every = every
        self.stepped = 0
        self._win = 0
        self._cuda_profiler = None
        if under_nsys():
            # Start the capture range here so nsys skips the one-time model/SDF build + graph capture.
            import torch.cuda.profiler as cuda_profiler

            self._cuda_profiler = cuda_profiler
            cuda_profiler.start()
        wp.synchronize_device(device)
        self._run_t0 = self._t_win = time.perf_counter()

    def tick(self, step_idx: int) -> None:
        self.stepped += 1
        self._win += 1
        if self._win == self.every:
            wp.synchronize_device(self.device)
            ctrl_s = self.every / (time.perf_counter() - self._t_win)
            env_s = ctrl_s * self.num_envs
            print(f"[fps] step {step_idx + 1}: {env_s:,.0f} env-steps/s ({ctrl_s:.1f} ctrl/s)", flush=True)
            self._win = 0
            wp.synchronize_device(self.device)
            self._t_win = time.perf_counter()

    def finish(self) -> None:
        if self.stepped:
            wp.synchronize_device(self.device)
            ctrl_s = self.stepped / (time.perf_counter() - self._run_t0)
            env_s = ctrl_s * self.num_envs
            print(
                f"[fps] summary: {self.stepped} steps, mean {env_s:,.0f} env-steps/s ({ctrl_s:.1f} ctrl/s)", flush=True
            )
        if self._cuda_profiler is not None:
            self._cuda_profiler.stop()
            _dump_kernel_modules(_KMAP_PATH)
