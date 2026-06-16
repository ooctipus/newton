# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scaling benchmark + PDF report for a Newton repro bundle.

Sweeps ``--num_envs`` (512, 1024, ... doubling) until the GPU runs out of memory, measuring
per size: throughput (env-steps/s), peak GPU memory, per-control-step wall time, and the
top GPU kernels (via ``repro.py --profile``). Writes a multi-page PDF of the scaling curves.

Each size runs in a fresh subprocess so memory is clean and an OOM only kills that one size.
Collision capacity (``rigid_contact_max`` / ``max_triangle_pairs``) is scaled with ``num_envs``
so the sweep hits real GPU OOM rather than the bundle's baked buffer caps.

    python scripts/newton_repro/benchmark.py --env factory_nut_thread --policy hammer -o report.pdf
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import time

_DIR = str(pathlib.Path(__file__).resolve().parent)
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

import warp as wp

wp.config.quiet = True

NCONMAX = 400  # MuJoCo per-env contact cap -> naconmax = NCONMAX * num_envs must fit rigid_contact_max
TRI_PER_ENV = 12000  # narrow-phase triangle pairs to budget per env (factory nut/bolt need ~9.3k)
_JSON_TAG = "BENCHMARK_JSON "
_OOM_MARKERS = ("out of memory", "cuda error", "failed to allocate", "bad_alloc", "outofmemory")


def _capacity_overrides(num_envs: int) -> list[str]:
    return [
        f"collision.rigid_contact_max={NCONMAX * num_envs}",
        f"collision.max_triangle_pairs={TRI_PER_ENV * num_envs}",
    ]


# --------------------------------------------------------------------------- inner (one size)
def _measure(env: str, num_envs: int, device: str, steps: int, warmup: int, policy: str | None) -> dict:
    """Build + step the bundle at ``num_envs`` and return throughput / memory / timing."""
    import inspect

    import torch
    from loader import apply_overrides, build_newton_from_bundle, load_bundle
    from repro import _load_mdp, _resolve_bundle_dir

    bundle = load_bundle(_resolve_bundle_dir(env))
    apply_overrides(bundle.sim_cfg, _capacity_overrides(num_envs))
    sim, env_origins = build_newton_from_bundle(bundle, num_envs=num_envs, device=device)

    decimation = int(bundle.sim_cfg.get("decimation", 1))
    mdp_cls = _load_mdp(bundle.bundle_dir)
    kwargs = dict(
        sim=sim,
        env_origins=env_origins,
        num_envs=env_origins.shape[0],
        physics_dt=sim.physics_dt,
        decimation=decimation,
        episode_length_s=float(bundle.sim_cfg.get("episode_length_s", 0.0)),
        device=device,
        extras_dir=bundle.extras_dir,
    )
    if policy and "policy" in inspect.signature(mdp_cls).parameters:
        kwargs["policy"] = policy
    mdp = mdp_cls(**kwargs)
    sim.capture_graph()

    def _control_step():
        mdp.act()
        for _ in range(decimation):
            mdp.apply_actuator()
            sim.step()
        mdp.forward()
        mdp.reset_done()

    for _ in range(warmup):
        _control_step()
    wp.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(steps):
        _control_step()
    wp.synchronize_device(device)
    dt = time.perf_counter() - t0

    free, total = torch.cuda.mem_get_info(device)
    return {
        "num_envs": num_envs,
        "steps": steps,
        "total_s": dt,
        "ms_per_step": 1000.0 * dt / steps,
        "ctrl_s": steps / dt,
        "env_steps_s": num_envs * steps / dt,
        "peak_gb": (total - free) / 1e9,
        "total_gb": total / 1e9,
    }


# --------------------------------------------------------------------------- orchestration
def _run_measure(env: str, num_envs: int, device: str, steps: int, warmup: int, policy: str | None):
    """Run :func:`_measure` in a fresh subprocess. Returns the metrics dict, ``"OOM"``, or ``None``."""
    cmd = [sys.executable, os.path.abspath(__file__), "--measure", str(num_envs),
           "--env", env, "--device", device, "--steps", str(steps), "--warmup", str(warmup)]  # fmt: skip
    if policy:
        cmd += ["--policy", policy]
    out = subprocess.run(cmd, capture_output=True, text=True)
    blob = out.stdout + out.stderr
    for line in out.stdout.splitlines():
        if line.startswith(_JSON_TAG):
            return json.loads(line[len(_JSON_TAG) :])
    if "captured size" in blob:  # loader caps num_envs at the bundle's captured size
        return "CAP"
    if any(m in blob.lower() for m in _OOM_MARKERS):
        return "OOM"
    print(f"  [warn] measure failed at {num_envs} (rc={out.returncode}):\n{blob[-600:]}", flush=True)
    return None


_ROW = re.compile(r"^\s*\d+\s+([\d.]+)%\s+([\d.]+)\s+(\d+)\s+(\S+)\s+(.+?)\s*$")


def _run_kernels(env: str, num_envs: int, device: str, steps: int, policy: str | None, topk: int) -> dict:
    """Run ``repro.py --profile`` and parse the kernel table -> {kernel: {module, ms_per_step, pct}}."""
    cmd = [sys.executable, os.path.join(_DIR, "repro.py"), "--env", env, "--num_envs", str(num_envs),
           "--device", device, "--steps", str(steps), "--headless", "--profile", "--topk", str(topk)]  # fmt: skip
    if policy:
        cmd += ["--policy", policy]
    for ov in _capacity_overrides(num_envs):
        cmd += ["--set", ov]
    out = subprocess.run(cmd, capture_output=True, text=True)
    kernels = {}
    in_table = False
    for line in out.stdout.splitlines():
        if line.startswith("=== top "):
            in_table = True
            continue
        if not in_table:
            continue
        m = _ROW.match(line)
        if m:
            pct, total_ms, _calls, module, kernel = m.groups()
            kernels[kernel] = {"module": module, "ms_per_step": float(total_ms) / steps, "pct": float(pct)}
    return kernels


def _sweep(args) -> tuple[list[dict], str]:
    sizes = [args.start * 2**i for i in range(args.max_doublings + 1)]
    rows = []
    limit = "completed sweep"
    for n in sizes:
        print(f"[bench] num_envs={n}: measuring ({args.reps} reps)...", flush=True)
        samples, mem, stop = [], None, None
        for _ in range(args.reps):
            m = _run_measure(args.env, n, args.device, args.steps, args.warmup, args.policy)
            if m in ("OOM", "CAP", None):  # m is a dict otherwise
                stop = m
                break
            samples.append(m["ms_per_step"])
            mem = mem or m
        if stop == "OOM":
            print(f"[bench] OOM at num_envs={n} -- stopping sweep.", flush=True)
            rows.append({"num_envs": n, "oom": True})
            limit = f"OOM at {n:,} envs"
            break
        if stop == "CAP":
            limit = f"bundle capped at {rows[-1]['num_envs']:,} envs" if rows else "below bundle size"
            print(f"[bench] reached the bundle's captured size ({limit}) -- stopping.", flush=True)
            break
        if not samples:
            limit = f"measurement failed at {n:,} envs"
            break
        samples.sort()
        med = samples[len(samples) // 2]
        row = {
            "num_envs": n, "reps": len(samples),
            "ms_per_step": med, "ms_min": samples[0], "ms_max": samples[-1],
            "ctrl_s": 1000.0 / med, "env_steps_s": n * 1000.0 / med,
            "peak_gb": mem["peak_gb"], "total_gb": mem["total_gb"],
        }  # fmt: skip
        print(
            f"  median {row['env_steps_s']:,.0f} env-steps/s | {med:.0f} ms/step "
            f"(range {samples[0]:.0f}-{samples[-1]:.0f}) | {row['peak_gb']:.2f} GB",
            flush=True,
        )
        print(f"[bench] num_envs={n}: profiling kernels...", flush=True)
        row["kernels"] = _run_kernels(args.env, n, args.device, args.kernel_steps, args.policy, args.topk)
        rows.append(row)
    return rows, limit


# --------------------------------------------------------------------------- report
def _regen_kernels(args) -> int:
    """Reuse the existing report JSON's speed/memory rows and only re-collect kernels (deeper --topk)
    so the tracked top-3 are present at every size, then rewrite the PDF + JSON."""
    json_path = os.path.splitext(args.output)[0] + ".json"
    with open(json_path) as f:
        rows = json.load(f)
    for r in rows:
        if r.get("oom"):
            continue
        print(f"[regen] num_envs={r['num_envs']}: re-profiling kernels (topk={args.topk})...", flush=True)
        r["kernels"] = _run_kernels(args.env, r["num_envs"], args.device, args.kernel_steps, args.policy, args.topk)
    limit = f"bundle capped at {max(r['num_envs'] for r in rows):,} envs"
    _write_pdf(rows, args.output, args.env, limit)
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"[regen] rewrote {args.output}", flush=True)
    return 0


def _replot_physx(args) -> int:
    """Re-render the report from existing JSON, overlaying a PhysX throughput curve (no sweep).

    ``--replot_physx`` is a JSON map ``{num_envs: env_steps_per_s}`` from a train.py PhysX sweep."""
    json_path = os.path.splitext(args.output)[0] + ".json"
    with open(json_path) as f:
        rows = json.load(f)
    with open(args.replot_physx) as f:
        physx = {int(k): float(v) for k, v in json.load(f).items()}
    limit = f"bundle capped at {max(r['num_envs'] for r in rows if not r.get('oom')):,} envs"
    _write_pdf(rows, args.output, args.env, limit, physx=physx)
    print(f"[replot] rewrote {args.output} with PhysX overlay ({len(physx)} sizes)", flush=True)
    return 0


def _write_pdf(rows: list[dict], path: str, env: str, limit: str, physx: dict | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    ok = [r for r in rows if not r.get("oom")]
    oom_n = next((r["num_envs"] for r in rows if r.get("oom")), None)
    n = [r["num_envs"] for r in ok]

    # Group kernels by subsystem -- individual solver kernels vary too much run-to-run to plot,
    # but the collision-vs-solver split is stable and explains where GPU time goes.
    def _grp(mod):
        if mod.startswith("newton._src.geometry"):
            return "collision"
        if mod.startswith("mujoco_warp"):
            return "solver"
        return "other"

    def _group_ms(r, g):
        return sum(v["ms_per_step"] for v in r["kernels"].values() if _grp(v["module"]) == g)

    with PdfPages(path) as pdf:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle(f"Newton repro scaling — {env}  ({limit})", fontsize=13)

        def _axfmt(ax, ylabel, title, logy=False):
            ax.set(xlabel="num_envs", ylabel=ylabel, title=title, xscale="log")
            if logy:
                ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.3)
            ax.set_xticks(n)
            ax.set_xticklabels([str(x) for x in n], rotation=45)

        reps = ok[0].get("reps", 1) if ok else 1
        # throughput: median over reps + min/max band (speed varies run-to-run with solver iterations)
        nlabel = "Newton (control 25 Hz, physics 200 Hz, substep 8)" if physx else None
        axes[0, 0].plot(n, [r["env_steps_s"] for r in ok], "o-", label=nlabel)
        axes[0, 0].fill_between(
            n,
            [r["num_envs"] * 1000.0 / r.get("ms_max", r["ms_per_step"]) for r in ok],
            [r["num_envs"] * 1000.0 / r.get("ms_min", r["ms_per_step"]) for r in ok],
            alpha=0.2,
        )
        title = f"Throughput per control step (median of {reps}, band = min/max)"
        if physx:
            pn = sorted(physx)
            axes[0, 0].plot(
                pn, [physx[k] for k in pn], "s--", color="C3",
                label="PhysX (control 25 Hz, physics 200 Hz, 192 pos-iters)",
            )  # fmt: skip
            axes[0, 0].legend(fontsize=7, loc="upper left")
            title = "Throughput per control step — Newton vs PhysX*"
        _axfmt(axes[0, 0], "env control-steps / s", title, logy=True)

        # memory: deterministic, with the GPU-total ceiling
        axes[0, 1].plot(n, [r["peak_gb"] for r in ok], "o-")
        if ok:
            tot = ok[0]["total_gb"]
            axes[0, 1].axhline(tot, ls="--", color="r", alpha=0.6, label=f"GPU total {tot:.0f} GB")
            axes[0, 1].legend(fontsize=7, loc="best")
        _axfmt(axes[0, 1], "peak GPU memory [GB]", "Memory")

        # time per control step: median + band
        axes[1, 0].plot(n, [r["ms_per_step"] for r in ok], "o-")
        axes[1, 0].fill_between(
            n,
            [r.get("ms_min", r["ms_per_step"]) for r in ok],
            [r.get("ms_max", r["ms_per_step"]) for r in ok],
            alpha=0.2,
        )
        _axfmt(axes[1, 0], "ms / control-step", "Time per step (median, band = min/max)", logy=True)

        # GPU kernel time by subsystem -- stable, unlike the noisy individual solver kernels
        for g in ("collision", "solver", "other"):
            axes[1, 1].plot(n, [_group_ms(r, g) for r in ok], "o-", label=g)
        axes[1, 1].legend(fontsize=8, loc="best")
        _axfmt(axes[1, 1], "ms / control-step (GPU kernel time)", "GPU kernel time by subsystem", logy=True)

        if physx:
            fig.text(
                0.5, 0.005,
                "*Not directly comparable: Newton = raw Kit-free physics stepping; PhysX = full RL training loop "
                "(physics + obs + reward + Kit) via train.py. Newton repro is also off-distribution "
                "(abnormal-robot watchdog resets ~22% of envs/step).",
                ha="center", va="bottom", fontsize=6.5, style="italic", wrap=True,
            )  # fmt: skip
        fig.tight_layout(rect=(0, 0.03 if physx else 0, 1, 0.96))
        pdf.savefig(fig)
        plt.close(fig)

        # summary table page
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        header = ["num_envs", "env-steps/s (med)", "ms/step (med)", "ms range", "peak GB"]
        cells = [[f"{r['num_envs']:,}", f"{r['env_steps_s']:,.0f}", f"{r['ms_per_step']:.0f}",
                  f"{r.get('ms_min', r['ms_per_step']):.0f}-{r.get('ms_max', r['ms_per_step']):.0f}",
                  f"{r['peak_gb']:.2f}"] for r in ok]  # fmt: skip
        if oom_n:
            cells.append([f"{oom_n:,}", "OOM", "", "", ""])
        table = ax.table(cellText=cells, colLabels=header, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.6)
        ax.set_title(f"Newton repro scaling summary — {env}", fontsize=13, pad=20)
        pdf.savefig(fig)
        plt.close(fig)

    print(f"[bench] wrote {path} ({len(ok)} sizes" + (f", OOM at {oom_n}" if oom_n else "") + ")", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description="Newton repro scaling benchmark + PDF report.")
    p.add_argument("--env", required=True, help="Bundle dir or task name under tasks/.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--policy", default=None)
    p.add_argument("--start", type=int, default=512, help="First num_envs (doubled each step).")
    p.add_argument("--max_doublings", type=int, default=9, help="Max sizes to try after --start.")
    p.add_argument("--reps", type=int, default=3, help="Independent runs per size (speed is noisy -> median + band).")
    p.add_argument("--steps", type=int, default=15, help="Timed control steps per run.")
    p.add_argument("--warmup", type=int, default=6)
    p.add_argument("--kernel_steps", type=int, default=20, help="Control steps for the nsys kernel profile.")
    p.add_argument("--topk", type=int, default=40, help="Kernels to collect per size (top-3 are plotted).")
    p.add_argument("-o", "--output", default=os.path.join(_DIR, "scaling_report.pdf"))
    p.add_argument("--measure", type=int, default=None, help="(internal) measure one size and print JSON.")
    p.add_argument(
        "--regen_kernels",
        action="store_true",
        help="Keep the existing report's speed/memory; only re-collect kernels (deeper --topk) and re-plot.",
    )
    p.add_argument(
        "--replot_physx",
        default=None,
        help="No-sweep re-plot: overlay a PhysX throughput curve from a JSON map {num_envs: env_steps_per_s}.",
    )
    args = p.parse_args()

    if args.replot_physx:
        return _replot_physx(args)

    if args.regen_kernels:
        return _regen_kernels(args)

    if args.measure is not None:  # inner mode: one size, one fresh process
        metrics = _measure(args.env, args.measure, args.device, args.steps, args.warmup, args.policy)
        print(_JSON_TAG + json.dumps(metrics), flush=True)
        return 0

    rows, limit = _sweep(args)
    _write_pdf(rows, args.output, args.env, limit)
    json_path = os.path.splitext(args.output)[0] + ".json"
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
