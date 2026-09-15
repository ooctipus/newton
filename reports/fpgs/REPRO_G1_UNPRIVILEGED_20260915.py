# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run a pinned diagnostic: python this_file.py GPU_INDEX FRESH_OUTPUT_ROOT."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

if len(sys.argv) != 3:
    raise SystemExit("Usage: python this_file.py GPU_INDEX FRESH_OUTPUT_ROOT")
ROOT = Path(sys.argv[2]).resolve()
MANIFEST = Path("/tmp/fpgs-g1-pair-terrain-paired16k-20260915-02/manifest.json")
HELPERS = Path("/home/octi/Projects/newton-fpgs-g1-unprivileged-profile-20260915/tools/fpgs_bench")
PINS = {
    MANIFEST: "034c570f6fed6ad085005dbd8217259ca308f6c48bcdd5efadf39b9e921fda59",
    HELPERS / "profile_sparse_metric_clock.py": "25563bf9fdb0974d6958222e5e7bc625d31f4dce4f61c6abf2aebe50d63a538a",
    HELPERS / "profile_sparse_metric_live.py": "7a84f1b0cef3730fc36e6923ca843169d6e386d60501e5382d30d20d37820ed1",
    HELPERS / "profile_sparse_metric_binary.py": "da12069cd62e4b0399ffdd270e4d2ae0a64d3723826ab42a85c50a8a70b7a279",
}
UUIDS = ("GPU-883586b6-3100-0610-81e5-3b4c26f45639", "GPU-ebfac9e8-02d5-d8a9-3bfc-bac64c62ffd4")


def guard(gpu, newton):
    assert os.geteuid() != 0, "Unprivileged diagnostic only"
    for path, expected in PINS.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, str(path)
    assert (
        subprocess.check_output(["git", "-C", newton, "rev-parse", "HEAD"], text=True).strip()
        == "ca0d427af809571bb5501f644c1a6e03990cd2a8"
    )
    assert not subprocess.check_output(["git", "-C", newton, "status", "--porcelain"], text=True).strip()
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader"], text=True
    )
    assert not any(line.startswith(UUIDS[gpu]) for line in apps.splitlines()), apps


gpu = int(sys.argv[1])
assert gpu in (0, 1)
data = json.loads(MANIFEST.read_text())
assert data["status"] == "complete" and data["final_source_and_idle_guard_pass"]
run = next(r for r in data["runs"] if r["gpu_index"] == gpu and r["variant"] == "baseline")
env = dict(os.environ)
env.update(run["environment"])
lab, newton = env["FPGS_BENCH_ISAACLAB"], env["FPGS_BENCH_NEWTON"]
guard(gpu, newton)
ROOT.mkdir(parents=True, exist_ok=True)
out = ROOT / f"gpu{gpu}"
out.mkdir(exist_ok=False)
env["FPGS_CLOCK_OUTPUT"] = str(out / "phase.npz")
env["FPGS_CLOCK_BINARY"] = "1"
env["OUT_DIR"] = str(out)
env["PROFILE_STEPS"] = "1"
cmd = [
    "/home/octi/.local/bin/uv",
    "run",
    "--no-project",
    "--python",
    lab + "/.venv/bin/python",
    "python",
    str(HELPERS / "profile_sparse_metric_live.py"),
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
report = {
    "diagnostic_only": True,
    "uid": os.geteuid(),
    "gpu_uuid": UUIDS[gpu],
    "command": cmd,
    "source_manifest": str(MANIFEST),
    "environment": {
        **run["environment"],
        **{k: env[k] for k in ("OUT_DIR", "PROFILE_STEPS", "FPGS_CLOCK_BINARY", "FPGS_CLOCK_OUTPUT")},
    },
    "helper_pins": {str(p): sha for p, sha in PINS.items()},
    "complete": False,
}
(out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
with (out / "driver.log").open("w") as log:
    result = subprocess.run(cmd, env=env, cwd=lab, stdout=log, stderr=subprocess.STDOUT, check=False)
report["child_exit"] = result.returncode
guard(gpu, newton)
report["final_source_and_idle_guard_pass"] = True
report["complete"] = result.returncode == 0
(out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report))
raise SystemExit(result.returncode)
