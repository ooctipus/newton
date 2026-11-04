# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The suite runner: Isaac Lab's check_suite scenes, several workers at once; setups overlap, timed loops run alone.

Each scene runs check_suite.py's worker in the Isaac Lab tree (FPGS_LAB) and writes <output>/<scene>/result.json and
<output>/summary.json. One scheduler can run several arms (Newton trees) at once, e.g. both arms of an A/B in one window:

  --newton T --output D [--cache-tag C]         one arm
  --arm NAME,TREE,OUTPUT[,CACHE_TAG]            repeatable instead of --newton/--output: several arms, interleaved
  --jobs N        scene workers at once, over all arms (default 1)
  --mem-budget-gb B
                  start a worker only while the measured peak GPU memory of the running scenes plus the next one
                  fits B GB (default 80). Peaks come from --mem-table (scene|law -> MiB, updated after every run
                  from nvidia-smi samples); a scene not yet measured counts as 16 GB.
  --gpu-token     timed loops run alone (bench_site/sitecustomize.py): at its first timed step a worker takes an
                  exclusive token, SIGSTOPs the other workers, waits --drain seconds for their queued GPU work, times,
                  then resumes them. Default on in perf mode with --jobs > 1.
  --order         start order: "setup" (longest measured setup first, the default) or "given"
  --shared-dir    token file and worker registry (default: the first arm's output); share it with another
                  scheduler running at the same time so their timed loops exclude each other too
  --protocol      JSON overrides of the mode preset, e.g. '{"warmup": 0, "blocks": 6}'
  --retries R     rerun a scene that failed (crash, timeout, error) up to R more times at the end of the queue
                  (default 1); the failed attempt's folder is kept as <scene>.failed-<n>
  --seed-cache S  kernel caches are per tree and tag (~/.cache/fpgs-suite/<python sha>[-<tag>]); a new one starts
                  as a copy of: "auto" (default) another arm's existing cache, else the newest cache with the same
                  tag; a cache directory; or "off" (cold, to measure compiles). Warp's module cache and the CUDA JIT
                  cache are content-addressed (kernel source, referenced constants, options, arch), so a copy only
                  serves kernels that did not change.

The GPU must be free when the run starts (unless --shared-dir is given); afterwards only this run's workers use it.
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

OUT = Path(os.environ.get("FPGS_OUT", "/home/octi/Projects/IsaacLab.wt/output/fpgs-ws-20261015"))
LAB = Path(os.environ.get("FPGS_LAB", "/home/octi/Projects/IsaacLab.wt/isaaclab-fpgs-keyboard-capacity-20261011"))
# The scene checker and suite (check_suite.py, suite21.json): the Isaac Lab tree's, or FPGS_PROFILE_DIR's copy of that
# directory (e.g. a checker change proposed to the owner, applied to both arms of a gate).
PROFILE = Path(os.environ.get("FPGS_PROFILE_DIR", LAB / "scripts/benchmarks/fpgs_profile"))
HERE = Path(__file__).resolve().parent
CACHES = Path.home() / ".cache/fpgs-suite"
SITE = HERE / "bench_site"
sys.path.insert(0, str(PROFILE))
import check_suite  # noqa: E402

# Warm-cache imports + setup seconds per scene (ab-r5a-head-1 on the RTX PRO 6000), for the longest-first start order.
SETUP_S = {
    "keyboard-so101": 108.3,
    "kuka-reorient": 39.1,
    "kuka": 38.8,
    "franka": 27.7,
    "g1": 26.3,
    "g1-flat": 24.0,
    "franka-cabinet": 23.7,
    "allegro": 23.3,
    "go2-rough": 21.4,
    "anymald-rough": 19.2,
    "h1-rough": 19.2,
    "go2-flat": 18.0,
    "anymald": 17.6,
    "humanoid": 16.2,
    "cassie-rough": 15.8,
    "h1-flat": 15.6,
    "cassie-flat": 13.0,
    "ant": 12.7,
    "ur10-reach": 12.5,
    "franka-reach": 11.4,
    "cartpole": 8.9,
}
UNKNOWN_MIB = 16 * 1024

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--newton", type=Path)
parser.add_argument("--output", type=Path)
parser.add_argument("--cache-tag", default="", help="separate kernel cache (seeded per --seed-cache)")
parser.add_argument(
    "--arm",
    action="append",
    default=[],
    help="NAME,TREE,OUTPUT[,CACHE_TAG[,SITE]]; SITE is a directory prepended to this arm's PYTHONPATH (e.g. the "
    "frozen dependency version this arm's tree was built against)",
)
parser.add_argument("--task", action="append", required=True)
parser.add_argument("--gpu", default=os.environ.get("FPGS_PRO_UUID", "GPU-883586b6-3100-0610-81e5-3b4c26f45639"))
parser.add_argument("--mode", choices=check_suite.PRESETS, default="perf")
parser.add_argument("--solver-attr", action="append", default=[], help="key=value applied to every scene")
parser.add_argument("--env", action="append", default=[], help="KEY=VALUE added to every scene environment")
parser.add_argument(
    "--unset-solver-attr", action="append", default=[], help="recipe solver attribute left at its default"
)
parser.add_argument(
    "--unset-env", action="append", default=[], help="recipe environment variable removed (KEY or PREFIX*)"
)
parser.add_argument("--collision-attr", action="append", default=[], help="key=value set on the collision pipeline cfg")
parser.add_argument("--timeout", type=int, default=900)
parser.add_argument("--probe-site", type=Path, help="directory prepended to the scene PYTHONPATH")
parser.add_argument("--jobs", type=int, default=1)
parser.add_argument("--mem-budget-gb", type=float, default=80.0)
parser.add_argument("--mem-table", type=Path, default=OUT / "bench_memory.json")
parser.add_argument("--gpu-token", choices=("auto", "on", "off"), default="auto")
parser.add_argument("--order", choices=("setup", "given"), default="setup")
parser.add_argument("--drain", type=float, default=0.5, help="seconds to let paused workers' GPU work finish")
parser.add_argument("--shared-dir", type=Path, help="token file and worker registry shared with another scheduler")
parser.add_argument("--protocol", default="", help="JSON overrides of the mode preset")
parser.add_argument("--retries", type=int, default=1)
parser.add_argument("--seed-cache", default="auto", help="auto, off, or a cache directory a new kernel cache copies")
args = parser.parse_args()

arms = []
for spec in args.arm:
    parts = spec.split(",")
    if len(parts) not in (3, 4, 5):
        parser.error(f"--arm needs NAME,TREE,OUTPUT[,CACHE_TAG[,SITE]]: {spec}")
    arms.append(
        {
            "name": parts[0],
            "tree": Path(parts[1]).resolve(),
            "output": Path(parts[2]).resolve(),
            "cache_tag": parts[3] if len(parts) >= 4 else "",
            "site": str(Path(parts[4]).resolve()) if len(parts) == 5 and parts[4] else None,
        }
    )
if args.newton or args.output:
    if not (args.newton and args.output) or arms:
        parser.error("use either --newton/--output or --arm")
    arms.append(
        {
            "name": "main",
            "tree": args.newton.resolve(),
            "output": args.output.resolve(),
            "cache_tag": args.cache_tag,
            "site": None,
        }
    )
if not arms:
    parser.error("no arm given")
scenes = {s["id"]: s for s in json.loads((PROFILE / "suite21.json").read_text())["scenes"]}
unknown = set(args.task) - scenes.keys()
if unknown:
    parser.error(f"Unknown scenes: {sorted(unknown)}")
token = args.gpu_token == "on" or (args.gpu_token == "auto" and args.mode == "perf" and args.jobs > 1)
law = "patch" if "friction_anchor_beta" in args.unset_solver_attr else "point"
protocol = dict(check_suite.PRESETS[args.mode], **(json.loads(args.protocol) if args.protocol else {}))
uuid = subprocess.check_output(
    ["nvidia-smi", "-i", args.gpu, "--query-gpu=uuid", "--format=csv,noheader"], text=True
).strip()
running = subprocess.check_output(
    ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True
)
# Owner, 2026-10-31: pass/fail gates may share the GB300, and gpu_locks.sh keeps timing and memory runs alone there.
if (
    uuid in running
    and not args.shared_dir
    and uuid != os.environ.get("GB300", "GPU-ebfac9e8-02d5-d8a9-3bfc-bac64c62ffd4")
):
    raise RuntimeError("Selected GPU is occupied; refusing to contaminate measurements")
lab_identity = check_suite.source_identity(LAB)
for arm in arms:
    arm["output"].mkdir(parents=True, exist_ok=False)
    arm["identity"] = {"newton": check_suite.source_identity(arm["tree"]), "lab": lab_identity}
    arm["cache"] = CACHES / (
        arm["identity"]["newton"]["python_sha256"][:16] + (f"-{arm['cache_tag']}" if arm["cache_tag"] else "")
    )
    arm["cache_existed"] = (arm["cache"] / "warp").is_dir()
    arm["cache"].mkdir(parents=True, exist_ok=True)
    arm["summary"] = {
        "complete": False,
        "mode": args.mode,
        "sources": arm["identity"],
        "overrides": {
            "solver_attrs": args.solver_attr,
            "environment": args.env,
            "unset_solver_attrs": args.unset_solver_attr,
            "unset_environment": args.unset_env,
            "collision_attrs": args.collision_attr,
            "probe_site": str(args.probe_site) if args.probe_site else None,
        },
        "parallel": {
            "jobs": args.jobs,
            "gpu_token": token,
            "order": args.order,
            "drain_s": args.drain,
            "mem_budget_gb": args.mem_budget_gb,
            "arms": [a["name"] for a in arms],
        },
        "protocol": protocol,
        "results": [],
    }
    arm["done"] = {}


def last_used(cache):
    return max((p.stat().st_mtime for p in (cache / "warp").iterdir()), default=0.0)


def seed_source(arm):
    if args.seed_cache == "off":
        return None
    if args.seed_cache != "auto":
        return Path(args.seed_cache).resolve()
    for other in arms:
        if other["cache_existed"]:
            return other["cache"]
    same_tag = [
        c
        for c in CACHES.iterdir()
        if c != arm["cache"] and (c / "warp").is_dir() and c.name.partition("-")[2] == arm["cache_tag"]
    ]
    return max(same_tag, key=last_used, default=None)


for arm in arms:
    source = None if (arm["cache"] / "warp").is_dir() else seed_source(arm)
    if source is not None:
        seeding = time.perf_counter()
        for part in ("warp", "cuda"):
            if (source / part).is_dir():
                shutil.copytree(source / part, arm["cache"] / part, symlinks=True, dirs_exist_ok=True)
        print(
            f"[{arm['name']}] kernel cache seeded from {source.name} ({time.perf_counter() - seeding:.1f}s)", flush=True
        )
    arm["summary"]["kernel_cache"] = {
        "path": str(arm["cache"]),
        "existed": arm["cache_existed"],
        "seeded_from": str(source) if source else None,
    }
shared = (args.shared_dir or arms[0]["output"]).resolve()
shared.mkdir(parents=True, exist_ok=True)
token_file = shared / "gpu.token"
workers_dir = shared / "workers"
mem_table = json.loads(args.mem_table.read_text()) if args.mem_table.exists() else {}

tasks = list(dict.fromkeys(args.task))
if args.order == "setup":
    tasks.sort(key=lambda t: -SETUP_S.get(t, 0.0))
queue = []
for i, task in enumerate(tasks):
    order = arms if i % 2 == 0 else list(reversed(arms))
    queue += [(task, arm) for arm in order]


def need_mib(task):
    return mem_table.get(f"{task}|{law}", UNKNOWN_MIB)


def prepare(task, arm):
    scene = json.loads(json.dumps(scenes[task]))
    for key in args.unset_solver_attr:
        scene["solver_attrs"].pop(key, None)
    for pattern in args.unset_env:
        for key in list(scene["environment"]):
            if key == pattern or (pattern.endswith("*") and key.startswith(pattern[:-1])):
                del scene["environment"][key]
    for item in args.solver_attr:
        key, value = item.split("=", 1)
        scene["solver_attrs"][key] = value
    for item in args.env:
        key, value = item.split("=", 1)
        scene["environment"][key] = value
    for item in args.collision_attr:
        key, value = item.split("=", 1)
        scene.setdefault("storage_at_16384_worlds", {})
        scene["storage_at_16384_worlds"][key] = float(value) if "." in value or "e" in value else int(value)
    folder = arm["output"] / task
    attempt = arm.setdefault("attempts", {}).get(task, 0)
    if folder.exists():
        folder.rename(arm["output"] / f"{task}.failed-{attempt}")
    arm["attempts"][task] = attempt + 1
    folder.mkdir()
    job = {"scene": scene, "protocol": protocol, "mode": args.mode, "newton": str(arm["tree"])}
    check_suite.write_json(folder / "job.json", job)
    environment = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("FEATHER_", "NEWTON_", "FPGS_"))
        and k not in ("PYTHONPATH", "WARP_CACHE_PATH", "CUDA_CACHE_PATH")
    }
    environment.update(scene["environment"])
    sites = ([arm["site"]] if arm["site"] else []) + ([str(args.probe_site)] if args.probe_site else [])
    if token:
        sites.append(str(SITE))
        environment.update(
            FPGS_GPU_TOKEN=str(token_file),
            FPGS_GPU_TOKEN_LOG=str(arm["output"] / "gpu_token.jsonl"),
            FPGS_WORKERS_DIR=str(workers_dir),
            FPGS_DRAIN_S=str(args.drain),
        )
    environment.update(
        CUDA_VISIBLE_DEVICES=uuid,
        PYTHONUNBUFFERED="1",
        PYTHONDONTWRITEBYTECODE="1",
        WARP_CACHE_PATH=str(arm["cache"] / "warp"),
        CUDA_CACHE_PATH=str(arm["cache"] / "cuda"),
        PYTHONPATH=os.pathsep.join(
            sites
            + [str(arm["tree"])]
            + [str(p) for p in sorted((LAB / "source").iterdir()) if p.is_dir() and (p / p.name).is_dir()]
        ),
    )
    log = (folder / "run.log").open("w")
    process = subprocess.Popen(
        [str(LAB / ".venv/bin/python"), str(PROFILE / "check_suite.py"), "--worker", str(folder / "job.json")],
        env=environment,
        cwd=LAB,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return {
        "task": task,
        "arm": arm,
        "folder": folder,
        "process": process,
        "log": log,
        "started": time.perf_counter(),
        "need": need_mib(task),
        "peak": 0,
    }


active = []
lock = threading.Lock()
stop = threading.Event()


def sample_memory():
    """Peak GPU memory per worker from nvidia-smi (MiB), every second."""
    while not stop.is_set():
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
                text=True,
                timeout=10,
            )
            used = {}
            for line in out.strip().splitlines():
                gpu, pid, mib = [x.strip() for x in line.split(",")]
                if gpu == uuid and mib.isdigit():
                    used[int(pid)] = int(mib)
            with lock:
                for item in active:
                    item["peak"] = max(item["peak"], used.get(item["process"].pid, 0))
        except Exception:
            pass
        stop.wait(1.0)


def finish(item, code):
    item["log"].close()
    for name in (str(item["process"].pid), f"{item['process'].pid}.gpu"):
        try:
            (workers_dir / name).unlink()
        except FileNotFoundError:
            pass
    result_path = item["folder"] / "result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else {"scene": item["task"], "complete": False}
    result["returncode"] = code
    result["wall_seconds"] = time.perf_counter() - item["started"]
    result["complete"] = bool(result.get("complete")) and code == 0
    result["gpu_memory_peak_mib"] = item["peak"]
    if result["complete"] and item["peak"] > 0:
        key = f"{item['task']}|{law}"
        mem_table[key] = max(mem_table.get(key, 0), item["peak"])
    return result


sampler = threading.Thread(target=sample_memory, daemon=True)
sampler.start()
budget = args.mem_budget_gb * 1024
started = time.perf_counter()
while queue or active:
    with lock:
        reserved = sum(item["need"] for item in active)
    while queue and len(active) < args.jobs:
        pick = next((i for i, (t, _) in enumerate(queue) if reserved + need_mib(t) <= budget), None)
        if pick is None:
            if active:
                break
            pick = 0  # nothing running: start the next scene whatever it needs
        task, arm = queue.pop(pick)
        item = prepare(task, arm)
        with lock:
            active.append(item)
        reserved += item["need"]
    time.sleep(0.5)
    for item in list(active):
        code = item["process"].poll()
        if code is None and time.perf_counter() - item["started"] > args.timeout:
            os.killpg(item["process"].pid, signal.SIGKILL)
            item["process"].wait()
            code = 124
        if code is None:
            continue
        with lock:
            active.remove(item)
        result = finish(item, code)
        arm = item["arm"]
        result["attempts"] = arm["attempts"][item["task"]]
        if not result["complete"] and result["attempts"] <= args.retries:
            queue.append((item["task"], arm))
            print(f"[{arm['name']}] {item['task']}: FAIL (rc {code}), queued again", flush=True)
            continue
        arm["done"][item["task"]] = result
        timing = result.get("timing", {}).get("mean_ms")
        print(
            f"[{arm['name']}] {item['task']}: {'PASS' if result['complete'] else 'FAIL'} "
            f"{timing if timing is None else f'{timing:.3f} ms'} ({result['wall_seconds']:.1f}s, "
            f"{item['peak'] / 1024:.1f} GB)",
            flush=True,
        )
        arm["summary"]["results"] = [arm["done"][t] for t in args.task if t in arm["done"]]
        arm["summary"]["elapsed_seconds"] = time.perf_counter() - started
        check_suite.write_json(arm["output"] / "summary.json", arm["summary"])
stop.set()
args.mem_table.write_text(json.dumps(dict(sorted(mem_table.items())), indent=1) + "\n")
for arm in arms:
    arm["summary"]["results"] = [arm["done"][t] for t in args.task if t in arm["done"]]
    arm["summary"]["elapsed_seconds"] = time.perf_counter() - started
    arm["summary"]["complete"] = all(r["complete"] for r in arm["summary"]["results"])
    check_suite.write_json(arm["output"] / "summary.json", arm["summary"])
failed = sum(not r["complete"] for arm in arms for r in arm["summary"]["results"])
print(
    f"== {sum(len(a['done']) for a in arms)} scenes over {len(arms)} arm(s) in "
    f"{(time.perf_counter() - started) / 60:.1f} min (jobs {args.jobs}, gpu token {token}, {failed} failed)",
    flush=True,
)
