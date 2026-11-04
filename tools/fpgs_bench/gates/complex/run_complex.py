#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run one (scene, friction law, world count) cell of the FeatherPGS patch-vs-point study.

The solver is constructed exactly as the scene's own code constructs it, except
``friction_anchor_beta`` (point: 0.0; patch: omitted, so Newton's default 0.2 applies). The metric
objects' initial poses get a small seeded per-world perturbation (identical in both arms; world 0
keeps the example's poses), so the worlds are trials, not replicas. The run verifies the law that
was actually built, captures one frame (all substeps) in a CUDA graph and times every graph launch
with a CUDA event pair. Physics samples are copied device-side between frames, outside the event
pairs, and read on the host only every ``--sync-every`` frames; the first frame after each of
those syncs, and the warm-up frames, are left out of the timing statistics. A growable rigid
contact pool is sized before the capture (see ``size_contacts``) and serviced after every frame, as
Isaac Lab does; a frame after a growth (which synchronizes and recaptures) is left out too.

Writes ``results/<label>/<case>-<law>-N<worlds>[-<tag>].json`` and, next to it, the raw per-frame
series and sampled states (``.npz``) from which ``analysis.py`` (and ``compare.py``, when two arms
stopped at different frames) derives every metric.

Example (CPU smoke, no CUDA context):
    CUDA_VISIBLE_DEVICES= WARP_CACHE_PATH=... python run_complex.py --newton <snapshot> --scene pyramid \
        --law patch --world-count 2 --device cpu --cpu-smoke --pyramid-rows 3 --seconds 0.5 --label cpu-smoke
"""

import argparse
import hashlib
import inspect
import json
import os
import platform
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCHEMA = "fpgs-complex-friction/2"
FLAG_PREFIXES = ("FEATHER_PGS_", "IL_NEWTON_FPGS_", "NEWTON_")
DEEP_MM = 0.5  # penetration depth counted as "deep" (analysis.DEEP_MM)
GPU_FIELDS = (
    "name",
    "uuid",
    "driver_version",
    "pstate",
    "clocks.sm",
    "clocks.mem",
    "clocks.max.sm",
    "temperature.gpu",
    "power.draw",
    "utilization.gpu",
    "clocks_event_reasons.active",
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", required=True, choices=("hardware", "coins", "dishes", "cube", "pyramid"))
    p.add_argument("--law", required=True, choices=("patch", "point"))
    p.add_argument("--world-count", type=int, required=True)
    p.add_argument("--label", required=True, help="results/<label>/ subdirectory")
    p.add_argument("--tag", default="", help="optional file-name suffix, e.g. r2 for a repeat")
    p.add_argument("--newton", type=Path, required=True, help="Newton tree to import (GPU: a gpu_run.sh snapshot)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seconds", type=float, default=None, help="simulated seconds (default: per scene)")
    p.add_argument("--frames", type=int, default=None, help="overrides --seconds")
    p.add_argument("--warmup", type=int, default=5, help="initial frames excluded from timing")
    p.add_argument("--sync-every", type=int, default=20, help="frames between host syncs")
    p.add_argument("--sample-every", type=int, default=3, help="frames between state samples")
    p.add_argument("--metric-worlds", type=int, default=32, help="worlds whose state is sampled")
    p.add_argument("--budget-s", type=float, default=0.0, help="safety net: stop the frame loop after this wall time")
    p.add_argument("--warm-only", action="store_true", help="construct, verify, capture, run one frame; no result")
    p.add_argument("--cpu-smoke", action="store_true", help="CPU: split mode, no SDF cook, convex cube")
    p.add_argument("--object-count", type=int, default=None, help="clutter object count override")
    p.add_argument("--example-args", default="", help="extra clutter example CLI arguments (one string)")
    p.add_argument("--pyramid-rows", type=int, default=20)
    p.add_argument("--pyramids", type=int, default=1)
    p.add_argument("--tilt-deg", type=float, default=0.0)
    p.add_argument("--pyramid-iterations", type=int, default=24)
    p.add_argument("--contacts-per-box", type=int, default=16)
    p.add_argument("--perturb-seed", type=int, default=1, help="seed of the per-world initial-pose perturbation")
    p.add_argument("--perturb-mm", type=float, default=0.1, help="x/y perturbation half-range [mm]; 0 disables")
    p.add_argument("--perturb-lift-mm", type=float, default=0.1, help="upward perturbation range [mm]")
    p.add_argument("--perturb-deg", type=float, default=0.1, help="yaw perturbation half-range [deg]")
    p.add_argument("--no-perturb", action="store_true", help="exact replica worlds (outcomes are one sample)")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(os.environ.get("FPGS_OUT", "/home/octi/Projects/IsaacLab.wt/output/fpgs-ws-20261015"))
        / "complex"
        / "results",
    )
    p.add_argument("--no-save-samples", action="store_true", help="skip the .npz (compare cannot re-clip this arm)")
    p.add_argument("--audit", action="store_true", help="the audit: record each world's energy gap and dual violation")
    p.add_argument("--allow-busy-gpu", action="store_true")
    p.add_argument("--allow-env-flags", action="store_true")
    p.add_argument("--allow-live-tree", action="store_true", help="GPU: allow a --newton tree without SNAPSHOT_SHA")
    return p.parse_args(argv)


def nvidia_smi(args, timeout=30):
    """Run nvidia-smi (NVML only: no CUDA context); None when it fails."""
    try:
        run = subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return run.stdout if run.returncode == 0 else None


def compute_apps(visible):
    text = nvidia_smi(["-i", visible, "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"])
    return None if text is None else [line.strip() for line in text.splitlines() if line.strip()]


def foreign_apps(apps, own_pid=None):
    """Compute processes on the GPU other than this one."""
    own = str(os.getpid() if own_pid is None else own_pid)
    return [app for app in apps or [] if app.split(",")[0].strip() != own]


def gpu_state(visible):
    """Driver, clocks, temperature, power and throttle reasons of the pinned GPU."""
    text = nvidia_smi(["-i", visible, "--query-gpu=" + ",".join(GPU_FIELDS), "--format=csv,noheader,nounits"])
    if not text or not text.strip():
        return None
    values = [v.strip() for v in text.strip().splitlines()[0].split(",")]
    return dict(zip(GPU_FIELDS, values, strict=False)) | {"time": time.strftime("%Y-%m-%dT%H:%M:%S")}


def preflight(args):
    """Refuse runs that would touch the shared kernel cache, pick the wrong GPU or tree, or share a busy GPU."""
    cache = os.environ.get("WARP_CACHE_PATH", "")
    shared = (Path.home() / ".cache" / "warp").resolve()
    resolved = Path(cache).expanduser().resolve() if cache else None
    if resolved is None or resolved == shared or shared in resolved.parents:
        sys.exit("WARP_CACHE_PATH must name a private kernel cache (never ~/.cache/warp)")
    flags = sorted(k for k in os.environ if k.startswith(FLAG_PREFIXES))
    if flags and not args.allow_env_flags:
        sys.exit(f"solver/collision environment flags are set: {flags}; unset them or pass --allow-env-flags")
    if not (args.newton / "newton" / "__init__.py").is_file():
        sys.exit(f"--newton {args.newton} holds no newton package")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if args.device == "cpu":
        if visible != "":
            sys.exit("CPU runs must hide CUDA (CUDA_VISIBLE_DEVICES=) so no CUDA context is created")
        return []
    if not visible:
        sys.exit("GPU runs must pin one GPU with CUDA_VISIBLE_DEVICES (UUID)")
    if not (args.newton / "SNAPSHOT_SHA").is_file() and not args.allow_live_tree:
        sys.exit(
            "GPU runs import a frozen snapshot (gpu_run.sh writes SNAPSHOT_SHA); other sessions commit to the live tree"
        )
    busy = compute_apps(visible)
    if busy is None:
        sys.exit(f"nvidia-smi could not query GPU {visible}")
    if busy and not args.allow_busy_gpu:
        sys.exit(f"GPU {visible} has compute processes: {busy}")
    return busy


def graph_substep_problem(is_cuda, substeps):
    """The examples' graphs hold all substeps of one frame; ping-pong states need an even count."""
    if is_cuda and substeps % 2:
        return (
            f"{substeps} substeps: a captured frame with an odd substep count leaves the newest state in the "
            "other buffer, so every replay would start from a stale state (the clutter example only captures "
            "even counts). Use an even --substeps."
        )
    return None


def git_info(path: Path) -> dict:
    """Read-only: HEAD, subject and modified tracked files (no optional index locks).

    A source snapshot made by gpu_run.sh (``git archive``) carries its commit in SNAPSHOT_SHA.
    """
    marker = path / "SNAPSHOT_SHA"
    if marker.is_file():
        return {"head": marker.read_text().strip(), "subject": "git archive snapshot", "dirty": [], "snapshot": True}
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")

    def git(*cmd):
        try:
            run = subprocess.run(
                ["git", "-C", str(path), *cmd], capture_output=True, text=True, timeout=60, env=env, check=False
            )
            return run.stdout.strip()
        except Exception as exc:
            return f"error: {exc}"

    dirty = git("status", "--porcelain", "--untracked-files=no").splitlines()
    return {
        "head": git("rev-parse", "HEAD"),
        "subject": git("log", "-1", "--format=%s"),
        "dirty": dirty[:50],
        "snapshot": False,
    }


def tree_digest(package: Path) -> dict:
    """SHA-256 over the relative paths and contents of every file of the imported newton package."""
    digest = hashlib.sha256()
    files = sorted(p for p in package.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")
    for path in files:
        digest.update(str(path.relative_to(package)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return {"sha256": digest.hexdigest(), "files": len(files)}


def harness_digest() -> str:
    """SHA-256 of the harness files that shape the simulation and its snapshots (not analysis/compare)."""
    digest = hashlib.sha256()
    for name in ("run_complex.py", "scenes.py", "metrics.py"):
        digest.update((HERE / name).read_bytes())
    return digest.hexdigest()


def count_files(directory: Path) -> int:
    return sum(1 for p in directory.rglob("*") if p.is_file()) if directory.is_dir() else 0


def cuda_versions():
    try:
        from warp._src import context as ctx  # noqa: PLC0415
    except ImportError:  # older Warp layouts
        import warp.context as ctx  # noqa: PLC0415
    runtime = getattr(ctx, "runtime", None)
    return {
        "driver": getattr(runtime, "driver_version", None),
        "toolkit": getattr(runtime, "toolkit_version", None),
    }


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if np.isfinite(value) else None
    return value


def same(actual, expected) -> bool:
    if isinstance(expected, float) and isinstance(actual, (int, float)):
        return abs(float(actual) - expected) < 1e-9
    return actual == expected


def verify_law(scene, law, scenes) -> dict:
    """Check every place the constructed solver reports its friction law; any mismatch voids the run."""
    solver = scene.solver
    expected = scenes.EXPECTED_LAW[law]
    selection = solver.get_path_selection()
    friction = selection["routes"]["friction"]
    option = selection["options"].get("friction_anchor_beta", {})
    environment = selection.get("environment", {})
    warnings_text = scene.law_record.get("warnings", [])
    checks = {
        "solver.laws.friction_anchor_beta": (solver.laws.friction_anchor_beta, expected["friction_anchor_beta"]),
        "solver.laws.friction_anchors_enabled": (
            bool(solver.laws.friction_anchors_enabled),
            expected["anchors_enabled"],
        ),
        "solver.laws.friction_anchor_beta_source": (solver.laws.friction_anchor_beta_source, expected["beta_source"]),
        "routes.friction.law": (friction.get("law"), law),
        "routes.friction.anchor_beta_source": (friction.get("anchor_beta_source"), expected["beta_source"]),
        "options.friction_anchor_beta.requested": (option.get("requested"), expected["requested"]),
        "friction_patches.view.enabled": (
            int(solver.laws.friction_patches.view.enabled),
            int(expected["anchors_enabled"]),
        ),
        "constructor_friction_warnings": ([w for w in warnings_text if "friction" in w.lower()], []),
        "environment_flags": (
            {key: environment.get(key) for key in ("applied", "unknown", "retired")},
            {"applied": {}, "unknown": [], "retired": []},
        ),
    }
    if law == "patch":
        checks["patch_capacity_equals_rigid_contact_max"] = (
            int(solver.laws.friction_patches.capacity),
            int(scene.contacts.rigid_contact_max),
        )
    detail = {name: {"actual": a, "expected": e, "ok": same(a, e)} for name, (a, e) in checks.items()}
    return {"ok": all(d["ok"] for d in detail.values()), "checks": detail}, selection


class Recorder:
    """Device-side rings filled between frames (outside the event pairs) and drained at host syncs."""

    def __init__(self, wp, scene, device, metric_worlds, sync_every, sample_every):
        self.wp, self.scene, self.wm = wp, scene, metric_worlds
        self.bpw = scene.bodies_per_world
        self.n = metric_worlds * self.bpw
        self.slots = sync_every // sample_every + 2
        self.q = wp.zeros(self.slots * self.n, dtype=wp.transform, device=device)
        self.qd = wp.zeros(self.slots * self.n, dtype=wp.spatial_vector, device=device)
        self.count = wp.zeros(sync_every, dtype=wp.int32, device=device)
        world_count = scene.world_count
        rows = getattr(scene.solver, "placement", scene.solver)
        sources = {"mf": getattr(rows, "mf_constraint_count", None), "dense": rows.constraint_count}
        self.row_sources = {
            name: array
            for name, array in sources.items()
            if array is not None and array.shape == (world_count,) and array.dtype == wp.int32
        }
        self.rows = {
            name: wp.zeros(sync_every * metric_worlds, dtype=wp.int32, device=device) for name in self.row_sources
        }
        self.pending_times = []
        self.times, self.samples_q, self.samples_qd = [], [], []
        self.counts = []
        self.row_counts = {name: [] for name in self.row_sources}

    # On CPU, wp.array.numpy() is a zero-copy view: every retained read below is an explicit copy.
    def host_sample(self, sim_time):
        state = self.scene.states()[0]
        self.times.append(sim_time)
        self.samples_q.append(np.array(state.body_q.numpy()[: self.n].reshape(self.wm, self.bpw, 7)))
        self.samples_qd.append(np.array(state.body_qd.numpy()[: self.n].reshape(self.wm, self.bpw, 6)))

    def after_frame(self, k, sim_time, sample):
        wp = self.wp
        state = self.scene.states()[0]
        wp.copy(self.count, self.scene.contacts.rigid_contact_count, dest_offset=k, src_offset=0, count=1)
        for name, source in self.row_sources.items():
            wp.copy(self.rows[name], source, dest_offset=k * self.wm, src_offset=0, count=self.wm)
        if sample:
            slot = len(self.pending_times)
            wp.copy(self.q, state.body_q, dest_offset=slot * self.n, src_offset=0, count=self.n)
            wp.copy(self.qd, state.body_qd, dest_offset=slot * self.n, src_offset=0, count=self.n)
            self.pending_times.append(sim_time)

    def drain(self, frames_in_block):
        self.counts.extend(self.count.numpy()[:frames_in_block].tolist())
        for name, ring in self.rows.items():
            block = np.array(ring.numpy()[: frames_in_block * self.wm]).reshape(frames_in_block, self.wm)
            self.row_counts[name].extend(block)
        used = len(self.pending_times)
        if used:
            q = np.array(self.q.numpy()[: used * self.n]).reshape(used, self.wm, self.bpw, 7)
            qd = np.array(self.qd.numpy()[: used * self.n]).reshape(used, self.wm, self.bpw, 6)
            self.times.extend(self.pending_times)
            self.samples_q.extend(q)
            self.samples_qd.extend(qd)
        self.pending_times = []


def penetration_summary(depth_mm):
    if depth_mm.size == 0:
        return {"n": 0}
    return {
        "n": int(depth_mm.size),
        "max_mm": float(depth_mm.max()),
        "p99_mm": float(np.percentile(depth_mm, 99)),
        "deep_count": int(np.count_nonzero(depth_mm > DEEP_MM)),
    }


def snapshot(scene, metric_worlds, sim_time, model_np, metrics, kernels, metric_bodies):
    """Host-side look at the last substep: per-world contacts, penetration, row census, patch anchors.

    Penetration counts only contacts that touch a metric object or the container; the rest (e.g. the
    fixed Franka base resting on the static bench, a constant overlap in both arms) goes to ``other``.
    """
    contacts, solver = scene.contacts, scene.solver
    n = min(int(contacts.rigid_contact_count.numpy()[0]), int(contacts.rigid_contact_max))
    s0 = contacts.rigid_contact_shape0.numpy()[:n]
    s1 = contacts.rigid_contact_shape1.numpy()[:n]
    world = np.maximum(model_np["shape_world"][s0], model_np["shape_world"][s1])
    per_world = np.bincount(world[world >= 0], minlength=scene.world_count)[: scene.world_count]
    keep = (world >= 0) & (world < metric_worlds)
    out = {
        "t": sim_time,
        "contacts_total": n,
        "contacts_per_world_median": float(np.median(per_world)) if per_world.size else 0.0,
        "contacts_per_world_max": int(per_world.max()) if per_world.size else 0,
    }
    if np.any(keep):
        collided = scene.states()[1]  # the last substep's collision input: gaps are self-consistent
        shape_body = model_np["shape_body"]
        phi = metrics.contact_gap(
            collided.body_q.numpy(),
            shape_body,
            s0[keep],
            s1[keep],
            contacts.rigid_contact_point0.numpy()[:n][keep],
            contacts.rigid_contact_point1.numpy()[:n][keep],
            contacts.rigid_contact_normal.numpy()[:n][keep],
            contacts.rigid_contact_margin0.numpy()[:n][keep],
            contacts.rigid_contact_margin1.numpy()[:n][keep],
        )
        depth = np.maximum(-phi, 0.0) * 1e3
        bodies = np.stack((shape_body[s0[keep]], shape_body[s1[keep]]))
        local = np.where(bodies >= 0, bodies % scene.bodies_per_world, -1)
        touches = np.isin(local, metric_bodies).any(axis=0)
        out["penetration"] = {
            "objects": penetration_summary(depth[touches]),
            "other": penetration_summary(depth[~touches]),
        }
    census = {}
    for name, types, counts in (
        ("dense", getattr(solver, "placement", solver).row_type, getattr(solver, "placement", solver).constraint_count),
        (
            "mf",
            getattr(getattr(solver, "placement", solver), "mf_row_type", None),
            getattr(getattr(solver, "placement", solver), "mf_constraint_count", None),
        ),
    ):
        if types is None or counts is None or len(types.shape) != 2:
            continue
        t = types.numpy()[:metric_worlds]
        c = counts.numpy()[:metric_worlds]
        live = np.arange(t.shape[1])[None, :] < c[:, None]
        normal = int(np.count_nonzero(live & (t == kernels.PGS_CONSTRAINT_TYPE_CONTACT)))
        friction = int(np.count_nonzero(live & (t == kernels.PGS_CONSTRAINT_TYPE_FRICTION)))
        census[name] = {"rows": int(c.sum()), "normal": normal, "friction": friction}
    out["census"] = census
    patches = solver.laws.friction_patches
    if solver.laws.friction_anchors_enabled and n:
        weight = patches.view.weight.numpy()[:n]
        source = patches.current.source.numpy()[:n]
        out["anchors"] = {
            "with_rows": int(np.count_nonzero(weight > 0)),
            "carried": int(np.count_nonzero((source >= 0) & (weight > 0))),
        }
    return out


def size_contacts(scene):
    """Size a growable rigid contact pool before the capture; return its service, or None.

    Newton's growable pools start below their bound and grow only in
    ``CollisionPipeline.service_capacity``, outside graph capture. Without this, a replayed graph
    keeps the initial rows however many contacts the scene makes, and drops the rest. A pipeline
    that measures a state (``service_capacity(state)``) sizes the pool from the initial state's
    demand, as Isaac Lab does. The first pools (``ccedc861``) cannot measure without a collision
    pass that overflows and flags the run, so they hold the bound from the start, as the buffers
    did before them.
    """
    service = getattr(scene.pipeline, "service_capacity", None)
    if service is None:
        return None
    if "state" in inspect.signature(service).parameters:
        service(scene.states()[0])
        return service
    # Those pools join a solver's buffers to the pipeline's at its first step, inside the capture,
    # where they cannot grow: join them now, so the reservation reaches the solver too.
    follow = getattr(scene.solver, "_follow_contacts", None)
    if hasattr(scene.solver, "placement"):
        follow = scene.solver.placement.follow_contacts
    if follow is not None:
        follow(scene.contacts)
    scene.contacts.reserve_rigid_contacts(int(scene.contacts.rigid_contact_bound))
    service()  # reports the reservation's growth, which no graph captured yet
    return service


class FrameGraph:
    """One frame (all substeps) captured in a CUDA graph, its only owner, recaptured as Isaac Lab does."""

    def __init__(self, wp, scene, device):
        self.wp, self.scene, self.device = wp, scene, device
        self.graph = None
        self.capture_s = self.capture()

    def capture(self) -> float:
        """(Re)capture the frame; return the seconds it took."""
        wp = self.wp
        # The old graph goes first: a body-pair reducer refuses a second graph on the same contacts.
        self.graph = None
        start = time.perf_counter()
        with wp.ScopedCapture(device=self.device) as capture:
            # FeatherPGS's double-buffer clears wait on events of the previous step; outside this capture
            # (the previous graph's) the wait fails, so seed them here (Isaac Lab's FeatherPGS manager does).
            seed = getattr(self.scene.solver, "seed_double_buffer_events", None)
            if seed is not None:
                seed()
            self.scene.simulate()
        self.graph = capture.graph
        wp.synchronize_device(self.device)
        return time.perf_counter() - start


def contact_rows(scene):
    return int(scene.contacts.rigid_contact_max)


def run_frames(
    wp, scene, args, device, frame_graph, frames, model_np, metrics, kernels, metric_bodies, visible, service
):
    frame_dt = 1.0 / scene.fps
    metric_worlds = min(args.metric_worlds, scene.world_count)
    rec = Recorder(wp, scene, device, metric_worlds, args.sync_every, args.sample_every)
    rec.host_sample(0.0)
    timed_by_events = frame_graph is not None
    if timed_by_events:
        stream = wp.get_stream(device)
        events = [
            (wp.Event(device, enable_timing=True), wp.Event(device, enable_timing=True)) for _ in range(args.sync_every)
        ]
    frame_ms = np.full(frames, np.nan)
    after_sync = np.zeros(frames, dtype=bool)
    snapshots, contention, growths, truncated, done, block_start = [], [], [], False, 0, 0
    loop_start = time.perf_counter()
    for frame in range(frames):
        k = frame - block_start
        scene.before_frame(frame * frame_dt)
        if timed_by_events:
            stream.record_event(events[k][0])
            wp.capture_launch(frame_graph.graph, stream=stream)
            stream.record_event(events[k][1])
        else:
            start = time.perf_counter()
            scene.simulate()
            wp.synchronize_device(device)
            frame_ms[frame] = (time.perf_counter() - start) * 1e3
        done = frame + 1
        rec.after_frame(k, done * frame_dt, sample=done % args.sample_every == 0 or done == frames)
        if service is not None and service():
            # The pool grew, synchronizing the device; later frames replay a graph captured at the new rows.
            recapture_s = frame_graph.capture() if timed_by_events else None
            growths.append({"frame": frame, "rows": contact_rows(scene), "recapture_s": recapture_s})
            if done < frames:
                after_sync[done] = True
        if k + 1 < args.sync_every and done < frames:
            continue
        wp.synchronize_device(device)
        if timed_by_events:
            for j in range(k + 1):
                frame_ms[block_start + j] = wp.get_event_elapsed_time(events[j][0], events[j][1], synchronize=False)
        after_sync[block_start] = True
        rec.drain(k + 1)
        snapshots.append(snapshot(scene, metric_worlds, done * frame_dt, model_np, metrics, kernels, metric_bodies))
        if device.is_cuda:  # outside the event pairs; the next frame is excluded from timing anyway
            foreign = foreign_apps(compute_apps(visible))
            if foreign:
                contention.append({"frames": [block_start, done], "apps": foreign})
        elapsed = time.perf_counter() - loop_start
        block = frame_ms[block_start:done]
        print(
            f"[{scene.case} {args.law} N={scene.world_count}] frame {done}/{frames} t={done * frame_dt:.2f}s "
            f"block median {np.nanmedian(block):.2f} ms/frame, contacts/world "
            f"{snapshots[-1]['contacts_per_world_median']:.0f}, wall {elapsed:.0f}s"
            + (
                f", foreign GPU processes {contention[-1]['apps']}"
                if contention and contention[-1]["frames"][1] == done
                else ""
            ),
            flush=True,
        )
        block_start = done
        if args.budget_s and elapsed > args.budget_s and done < frames:
            truncated = True
            break
    return {
        "timed_by_events": timed_by_events,
        "frame_ms": frame_ms[:done],
        "after_sync": after_sync[:done],
        "frames_run": done,
        "truncated": truncated,
        "loop_wall_s": time.perf_counter() - loop_start,
        "recorder": rec,
        "snapshots": snapshots,
        "metric_worlds": metric_worlds,
        "contention": contention,
        "contact_growths": growths,
    }


def clip_windows(windows, t_plan, fps):
    """Windows the planned run reaches: dropped when they start after it, cut at its end otherwise."""
    tol = 0.5 / fps
    return {name: (t0, min(t1, t_plan)) for name, (t0, t1) in windows.items() if t0 < t_plan - tol}


def raw_arrays(loop, consts):
    rec = loop["recorder"]
    frames = loop["frames_run"]

    def rows(name):
        values = rec.row_counts.get(name)
        return np.asarray(values, dtype=np.int32).reshape(frames, -1) if values else np.zeros(0, dtype=np.int32)

    return {
        "times": np.asarray(rec.times, dtype=np.float64),
        "body_q": np.asarray(rec.samples_q, dtype=np.float32),
        "body_qd": np.asarray(rec.samples_qd, dtype=np.float32),
        "frame_ms": np.asarray(loop["frame_ms"], dtype=np.float64),
        "after_sync": np.asarray(loop["after_sync"], dtype=bool),
        "contacts_total": np.asarray(rec.counts, dtype=np.int64),
        "rows_mf": rows("mf"),
        "rows_dense": rows("dense"),
        **consts,
    }


def run_record(result) -> dict:
    """The run-level records analysis.summary_rows needs (they are not clipped to a horizon)."""
    return {
        "construction": result.get("construction", {}),
        "watermarks": result.get("rows", {}).get("watermarks", {}),
        "flags": result.get("flags", {}),
        "final": result.get("final", {}),
    }


def main(argv=None):
    args = parse_args(argv)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    busy = preflight(args)
    sys.dont_write_bytecode = True  # never leave __pycache__ in the Newton tree
    gpu_start = gpu_state(visible) if args.device != "cpu" else None
    sys.path.insert(1, str(args.newton))
    import warp as wp  # noqa: PLC0415

    wp.config.log_level = wp.LOG_WARNING
    wp.init()
    import newton  # noqa: PLC0415
    import importlib  # noqa: PLC0415
    import pkgutil  # noqa: PLC0415

    import newton._src.solvers.feather_pgs as fpgs  # noqa: PLC0415

    # The row-type codes live in whichever solver module defines them; their owner moves as the solver is restructured.
    kernels = next(
        module
        for module in (
            importlib.import_module(f"{fpgs.__name__}.{m.name}") for m in pkgutil.iter_modules(fpgs.__path__)
        )
        if hasattr(module, "PGS_CONSTRAINT_TYPE_CONTACT")
    )

    newton_file = Path(newton.__file__).resolve()
    if args.newton.resolve() not in newton_file.parents:
        sys.exit(f"imported newton from {newton_file}, expected it under {args.newton}")
    import analysis  # noqa: PLC0415
    import metrics  # noqa: PLC0415
    import scenes  # noqa: PLC0415

    wp.set_device(args.device)
    device = wp.get_device(args.device)
    kernel_cache = Path(wp.config.kernel_cache_dir)
    sdf_cache = Path(tempfile.gettempdir()) / "newton_clutter_sdf_cache"  # the clutter example's SDF cook cache
    cache_before = (count_files(kernel_cache), count_files(sdf_cache))
    modules_before = sum(1 for p in kernel_cache.rglob("*") if p.is_dir()) if kernel_cache.is_dir() else 0

    scenes.AUDIT = args.audit
    scene = scenes.build(
        args.scene,
        args.law,
        args.world_count,
        cpu_smoke=args.cpu_smoke,
        object_count=args.object_count,
        example_args=shlex.split(args.example_args),
        pyramid={
            "rows": args.pyramid_rows,
            "pyramids": args.pyramids,
            "tilt_deg": args.tilt_deg,
            "iterations": args.pyramid_iterations,
            "contacts_per_box": args.contacts_per_box,
        },
    )
    problem = graph_substep_problem(device.is_cuda, scene.substeps)
    if problem:
        sys.exit(problem)
    perturbation = scenes.perturb_initial_poses(
        scene,
        seed=args.perturb_seed,
        xy_mm=0.0 if args.no_perturb else args.perturb_mm,
        lift_mm=0.0 if args.no_perturb else args.perturb_lift_mm,
        yaw_deg=0.0 if args.no_perturb else args.perturb_deg,
    )
    rows_built = contact_rows(scene)
    service = size_contacts(scene)
    rows_captured = contact_rows(scene)
    verification, selection = verify_law(scene, args.law, scenes)
    name = f"{scene.case}-{args.law}-N{args.world_count}" + (f"-{args.tag}" if args.tag else "")
    out_dir = args.out_dir / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = args.frames or round((args.seconds or scene.default_seconds) * scene.fps)
    metric_worlds = min(args.metric_worlds, scene.world_count)
    objects = sorted({b for members in scene.groups.values() for b in members})
    metric_bodies = np.asarray(
        objects + ([scene.support_body] if scene.support_body is not None else []), dtype=np.int64
    )
    windows = clip_windows(scene.windows, frames / scene.fps, scene.fps)
    result = {
        "schema": SCHEMA,
        "label": args.label,
        "case": scene.case,
        "scene": args.scene,
        "law": args.law,
        "world_count": args.world_count,
        "tag": args.tag,
        "cli": sys.argv[1:] if argv is None else list(argv),
        "source": {
            "newton_path": str(args.newton),
            "newton_file": str(newton_file),
            "newton_digest": tree_digest(args.newton / "newton"),
            "harness_digest": harness_digest(),
            "git": git_info(args.newton),
            "warp_version": wp.config.version,
            "cuda": cuda_versions(),
            "python": platform.python_version(),
            "host": platform.node(),
            "device": str(device),
            "device_name": getattr(device, "name", None),
            "device_uuid": str(getattr(device, "uuid", None)),
            "sm_count": getattr(device, "sm_count", None),
            "cuda_visible_devices": visible,
            "warp_cache_path": os.environ.get("WARP_CACHE_PATH"),
            "busy_gpu_processes": busy,
            "gpu_start": gpu_start,
        },
        "protocol": {
            "frames_planned": frames,
            "warmup": args.warmup,
            "sync_every": args.sync_every,
            "sample_every": args.sample_every,
            "metric_worlds": metric_worlds,
            "budget_s": args.budget_s,
            "perturbation": perturbation,
            "windows": windows,
            "windows_nominal": scene.windows,
        },
        "scene_info": {
            "kind": scene.kind,
            "fps": scene.fps,
            "substeps": scene.substeps,
            "bodies_per_world": scene.bodies_per_world,
            "groups": {key: len(members) for key, members in scene.groups.items()},
            "cpu_smoke": args.cpu_smoke,
            **{k: v for k, v in scene.info.items() if k != "construct_s"},
        },
        "solver_kwargs": scene.law_record.get("kwargs"),
        "law_record": scene.law_record,
        "law_verification": verification,
        "path_selection": selection,
    }
    path = out_dir / f"{name}.json"
    if not verification["ok"]:
        result["status"] = "law_mismatch"
        path.write_text(json.dumps(jsonable(result), indent=1) + "\n")
        sys.exit(f"friction law verification failed, see {path}")

    frame_graph = FrameGraph(wp, scene, device) if device.is_cuda else None
    capture_s = frame_graph.capture_s if frame_graph is not None else None

    def construction():
        files_after = (count_files(kernel_cache), count_files(sdf_cache))
        modules_after = sum(1 for p in kernel_cache.rglob("*") if p.is_dir()) if kernel_cache.is_dir() else 0
        return {
            "scene_s": scene.info["construct_s"],
            "solver_init_s": scene.law_record.get("init_s"),
            "capture_s": capture_s,
            # Any file the kernel cache gained (new module directories and binaries added to existing ones,
            # e.g. a CUDA cubin next to a CPU object) means this process compiled: a cold-cache indicator.
            "new_kernel_files": files_after[0] - cache_before[0],
            "new_kernel_modules": modules_after - modules_before,
            "sdf_cache_files_written": files_after[1] - cache_before[1],
        }

    if args.warm_only:
        # Untimed pass that leaves every kernel and SDF of this (scene, law) in the caches, so the
        # measured run's construction is not timed against the other arm's compile.
        scene.before_frame(0.0)
        if frame_graph is not None:
            wp.capture_launch(frame_graph.graph)
        else:
            scene.simulate()
        wp.synchronize_device(device)
        record = {
            "schema": SCHEMA,
            "status": "warm",
            "case": scene.case,
            "law": args.law,
            "world_count": args.world_count,
        }
        record["construction"] = construction()
        warm_dir = out_dir / "warm"
        warm_dir.mkdir(exist_ok=True)
        (warm_dir / f"{name}.json").write_text(json.dumps(jsonable(record), indent=1) + "\n")
        print(
            f"[{scene.case} {args.law} N={args.world_count}] warm pass: {jsonable(record['construction'])}", flush=True
        )
        return

    model = scene.model
    bpw = scene.bodies_per_world
    model_np = {"shape_world": model.shape_world.numpy(), "shape_body": model.shape_body.numpy()}
    consts = {
        "com": model.body_com.numpy()[:bpw].astype(np.float64),
        "mass": model.body_mass.numpy()[:bpw].astype(np.float64),
        "inertia": model.body_inertia.numpy()[:bpw].reshape(bpw, 3, 3).astype(np.float64),
    }
    result["construction"] = construction()
    print(
        f"[{scene.case} {args.law} N={args.world_count}] built in {scene.info['construct_s']:.1f}s "
        f"(solver {scene.law_record.get('init_s', 0):.1f}s, capture {capture_s or 0:.1f}s, kernel cache files "
        f"written {result['construction']['new_kernel_files']}); perturbed {perturbation.get('bodies', 0)} bodies; "
        f"running {frames} frames on {device}",
        flush=True,
    )
    q_start = np.array(scene.states()[0].body_q.numpy())
    loop = run_frames(
        wp, scene, args, device, frame_graph, frames, model_np, metrics, kernels, metric_bodies, visible, service
    )
    wp.synchronize_device(device)

    state = scene.states()[0]
    q_end, qd_end = state.body_q.numpy(), state.body_qd.numpy()
    if scene.kind == "clutter":  # Example.test_post_step's tabletop bounds
        escaped = (np.max(np.abs(q_end[:, :3]), axis=1) > 3.0) | (q_end[:, 2] < -0.10)
    else:  # pyramid: fell through the ground or left the stack's footprint
        escaped = (q_end[:, 2] < -0.10) | (np.linalg.norm(q_end[:, :3] - q_start[:, :3], axis=1) > 3.0)
    solver = scene.solver
    capacity = {"solver": solver.constraint_capacity_status()}
    try:
        solver.check_constraint_capacity()
        capacity["solver_check"] = "ok"
    except RuntimeError as exc:
        capacity["solver_check"] = str(exc)
    capacity.update(scene.end_report())
    capacity["contact_rows"] = {
        "bound": int(getattr(scene.pipeline, "rigid_contact_max", rows_built)),
        "built": rows_built,
        "captured": rows_captured,
        "end": contact_rows(scene),
        "serviced": service is not None,
        "growths": loop["contact_growths"],
    }
    overflow = (
        any(capacity["solver"].values())
        or capacity["solver_check"] != "ok"
        or any(value for key, value in capacity["collision_buffers"].items() if key != "reduction_hash_load")
    )
    reduction = capacity.get("body_pair_reduction")
    if reduction:
        overflow = overflow or any(
            reduction.get(k) for k in ("invariant_violations", "outranked_discards", "input_overflow_frames")
        )

    result["status"] = "ok"
    result["source"]["gpu_end"] = gpu_state(visible) if device.is_cuda else None
    result["source"]["gpu_contention"] = loop["contention"]
    result["final"] = {
        "t": loop["frames_run"] / scene.fps,
        "finite_end": bool(np.isfinite(q_end).all() and np.isfinite(qd_end).all()),
        "escaped_end": int(np.count_nonzero(escaped)),
    }
    result["rows"] = {"watermarks": solver.constraint_row_watermarks()}
    result["capacity"] = capacity
    result["flags"] = {
        "law_verified": verification["ok"],
        "capacity_overflow": bool(overflow),
        "truncated": loop["truncated"],
        "gpu_contention": bool(busy) or bool(loop["contention"]),
        "metric_worlds": loop["metric_worlds"],
        "loop_wall_s": loop["loop_wall_s"],
        "frames_run": loop["frames_run"],
        "timed_by": "cuda_event_pair_per_graph_launch" if loop["timed_by_events"] else "host_perf_counter_synchronous",
    }
    result["analysis_context"] = {
        "case": scene.case,
        "kind": scene.kind,
        "law": args.law,
        "fps": scene.fps,
        "substeps": scene.substeps,
        "world_count": scene.world_count,
        "metric_worlds": loop["metric_worlds"],
        "bodies_per_world": bpw,
        "frames_planned": frames,
        "warmup": args.warmup,
        "windows": windows,
        "groups": scene.groups,
        "focus_group": scene.focus_group,
        "support_body": scene.support_body,
        "tcp_body": scene.tcp_body,
        "shake_axis": scene.shake_axis,
        "container_half_extents": scene.container_half_extents,
        "floor_top": scene.floor_top,
        "special": scene.special,
        "perturbed": bool(perturbation.get("bodies")) and scene.world_count > 1,
    }
    result["snapshots"] = loop["snapshots"]
    raw = raw_arrays(loop, consts)
    raw["snapshots"] = jsonable(loop["snapshots"])
    if args.audit:
        audit = scene.solver.read_audit()
        raw["audit_gap_sum"] = audit["gap_sum"]
        raw["audit_violation_max"] = audit["violation_max"]
        raw["audit_steps"] = audit["steps"]
    ctx = jsonable(result["analysis_context"])
    sections = analysis.analyze(ctx, raw, run_record(jsonable(result)))
    result["timing_frames"] = {
        "per_frame_ms": loop["frame_ms"],
        "excluded_after_sync": np.flatnonzero(loop["after_sync"]),
    }
    result.update(sections)
    result["raw_file"] = None
    if not args.no_save_samples:
        np.savez_compressed(path.with_suffix(".npz"), **{k: v for k, v in raw.items() if k != "snapshots"})
        result["raw_file"] = path.with_suffix(".npz").name
    path.write_text(json.dumps(jsonable(result), indent=1) + "\n")
    t = result["timing"]["windows"].get("all", {})
    print(
        f"[{scene.case} {args.law} N={args.world_count}] done: {loop['frames_run']} frames, "
        f"median {t.get('median_ms', float('nan')):.2f} ms/frame, loop {loop['loop_wall_s']:.0f}s, "
        f"overflow={overflow}, truncated={loop['truncated']}, "
        f"physics flags {result['physics_flags'] or 'none'} -> {path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
