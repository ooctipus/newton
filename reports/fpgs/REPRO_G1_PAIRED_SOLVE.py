"""Replay the two solve owners on identical retained inputs without privileges."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

if len(sys.argv) != 4:
    raise SystemExit("Usage: REPRO_G1_PAIRED_SOLVE.py GPU_INDEX FRESH_OUTPUT_ROOT CLEAN_E2F_NEWTON_WORKTREE")
ROOT = Path(sys.argv[2]).resolve()
RUNTIME = str(Path(sys.argv[3]).resolve())
ROOT.mkdir(parents=True, exist_ok=True)
MANIFEST = Path("/tmp/fpgs-g1-paired-solve-paired16k-20260915-01/manifest.json")
HELPER = Path(__file__).resolve().parents[2] / "tools/fpgs_bench/diagnose_sparse_paired_worlds.py"
PINS = {
    MANIFEST: "e33d606fcf4fd0bf07ec3d94d100efe7c37a7a6cea5eddee2199391736d3b062",
    HELPER: "2a605df016f06c6f8da5a7c8d8b748cbba1d4b0d4098aeef41b1185734ab9539",
}
UUIDS = ("GPU-883586b6-3100-0610-81e5-3b4c26f45639", "GPU-ebfac9e8-02d5-d8a9-3bfc-bac64c62ffd4")


def guard(gpu, lab, newton):
    assert os.geteuid() != 0
    for path, expected in PINS.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, str(path)
    for path, pin in (
        (newton, "e2f616478fba08a77c7aa6df9fd01a378afd8db4"),
        (lab, "53ee6b44c2334341305dbdf385a3916c6b140799"),
    ):
        assert subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip() == pin
        assert not subprocess.check_output(
            ["git", "-C", path, "status", "--porcelain", "--untracked-files=no"], text=True
        ).strip()
    assert not subprocess.check_output(["git", "-C", newton, "status", "--porcelain"], text=True).strip()
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader"], text=True
    )
    assert not any(line.startswith(UUIDS[gpu]) for line in apps.splitlines()), apps


gpu = int(sys.argv[1])
assert gpu in (0, 1)
data = json.loads(MANIFEST.read_text())
assert data["status"] == "complete" and data["final_source_and_idle_guard_pass"]
run = next(r for r in data["runs"] if r["gpu_index"] == gpu and r["variant"] == "candidate")
env = dict(os.environ)
original_runtime = run["environment"]["FPGS_BENCH_NEWTON"]
env.update({key: value.replace(original_runtime, RUNTIME) for key, value in run["environment"].items()})
lab, newton = env["FPGS_BENCH_ISAACLAB"], env["FPGS_BENCH_NEWTON"]
guard(gpu, lab, newton)
out = ROOT / f"gpu{gpu}"
out.mkdir(exist_ok=False)
env.update(
    FEATHER_PGS_SPARSE_PAIRED_WORLDS="0",
    FPGS_PAIRED_REPLAY_OUTPUT=str(out / "replay.npz"),
    OUT_DIR=str(out),
    PROFILE_STEPS="1",
)
cmd = [
    "/home/octi/.local/bin/uv",
    "run",
    "--no-project",
    "--python",
    lab + "/.venv/bin/python",
    "python",
    str(HELPER),
    "--isaaclab",
    lab,
    "--newton",
    newton,
    "--expected-run-sha256",
    env["FPGS_BENCH_RUN_SHA256"],
    "--checks-output",
    str(out / "checks.json"),
    *run["command"][5:7],
    "--",
    "--physics",
    run["command"][3],
    "--task",
    run["command"][4],
    "--device",
    "cuda:0",
    "--repeats",
    "1",
    "--steps",
    "40",
    "--profile-steps",
    "1",
    "--output",
    str(out / "capture.json"),
    *run["command"][7:],
]
overrides = ("FEATHER_PGS_SPARSE_PAIRED_WORLDS", "FPGS_PAIRED_REPLAY_OUTPUT", "OUT_DIR", "PROFILE_STEPS")
report = {
    "diagnostic_only": True,
    "uid": os.geteuid(),
    "gpu_uuid": UUIDS[gpu],
    "command": cmd,
    "source_manifest": str(MANIFEST),
    "environment": {k: env[k] for k in (*run["environment"], *overrides)},
    "helper_pins": {str(p): sha for p, sha in PINS.items()},
    "complete": False,
}
(out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
with (out / "driver.log").open("w") as log:
    result = subprocess.run(cmd, env=env, cwd=lab, stdout=log, stderr=subprocess.STDOUT, check=False)
report["child_exit"] = result.returncode
guard(gpu, lab, newton)
report["final_source_and_idle_guard_pass"] = True
report["complete"] = result.returncode == 0
(out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report))
raise SystemExit(result.returncode)
