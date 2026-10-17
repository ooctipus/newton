# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run the September 19 discovery survey through the existing paired driver.

This is as-delivered source/recipe timing, not matched contact-law validation.
The six original recipes reconstruct documented accepted opt-ins; other tasks
use Lab presets on both sources. Failed tasks remain in the survey manifest.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def flags(*names):
    """Expand explicitly selected solver feature names."""
    return [f"FEATHER_PGS_{name}=1" for name in names]


RECIPES = {
    "franka": (
        [],
        {"rigid_contact_max": 8192},
    ),
    "kuka": (
        [],
        {"rigid_contact_max": 77824},
    ),
    "allegro": (
        [
            *flags("KEYED_ROWS"),
            "NEWTON_NARROW_PHASE_COHERENT_CONVEX=reject_only",
            "NEWTON_NARROW_PHASE_CONVEX_CELLS=1",
        ],
        {"rigid_contact_max": 71680},
    ),
    "anymald": ([], {"rigid_contact_max": 53248, "dense_max_constraints": 72}),
    # The original G1 recipe's packed sparse-factor opt-ins are retired; the
    # general route runs in their place.
    "g1": (
        flags("TREE_CHAIN_SCAN")
        + [
            f"NEWTON_HEIGHTFIELD_{name}=1"
            for name in (
                "CELL_REJECT",
                "FINITE_QUERY",
                "GEOMETRIC_CULL",
                "WELD_FLAT_SEAMS",
                "PAIR_CSR",
                "PAIR_CSR_SHELL",
            )
        ]
        + ["NEWTON_NARROW_PHASE_PACKED_HEIGHTFIELD_PAIRS=1"],
        {"rigid_contact_max": 73728, "dense_max_constraints": 100, "max_triangle_pairs": 442368},
    ),
    "keyboard-so101": (
        flags("SPARSE_CONTACT_DIRECT", "PRISMATIC_PUBLICATION", "COMPACT_CONTACT_BOUNDARY", "PRISMATIC_LINEAR_STATE"),
        {"rigid_contact_max": 147456, "dense_max_constraints": 704},
    ),
    **{
        task: ([], {})
        for task in (
            "anymald-rough",
            "g1-flat",
            "h1-flat",
            "h1-rough",
            "go2-flat",
            "go2-rough",
            "cassie-flat",
            "cassie-rough",
            "cartpole",
            "ant",
            "humanoid",
            "franka-reach",
            "ur10-reach",
            "franka-drawer",
        )
    },
}


def main():
    """Preserve each failed capture and continue only after successful idle guards."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("isaaclab", "baseline", "candidate", "output-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--task", action="append", choices=tuple(RECIPES))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "scope": __doc__,
        "num_envs": 4096,
        "rounds": 1,
        "runs": [],
        "caveats": [
            "Baseline friction patches remain enabled; candidate predates that contact law.",
            "One round is discovery only, not a repeated speedup qualification.",
            "Legacy collision warning/snapshot checks cannot certify absence of every dropped contact.",
            "Original-six contact pools are scaled from prior 16K recipes (Keyboard stays 4K); fresh capacity qualification pending.",
            "Additional fourteen tasks use unchanged Lab presets without newly tuned specialized opt-ins.",
            "Broad-phase pools use checkout defaults because pristine baseline has no output-capacity option.",
            "Device-used memory snapshots include context/cache allocations; they are not peak solver allocation measurements.",
        ],
    }
    manifest = args.output_dir / "survey.json"

    def save():
        manifest.write_text(json.dumps(report, indent=2) + "\n")

    save()
    for task in args.task or RECIPES:
        candidate_flags, capacities = RECIPES[task]
        destination = args.output_dir / task
        command = [
            sys.executable,
            str(Path(__file__).with_name("compare_variants.py")),
            "--isaaclab",
            str(args.isaaclab),
            "--baseline",
            str(args.baseline),
            "--candidate",
            str(args.candidate),
            "--output-dir",
            str(destination),
            "--task",
            task,
            "--legacy-baseline-capacity",
            "--num-envs",
            "4096",
            "--rounds",
            "1",
            "--warmup-steps",
            "200",
            "--steps",
            "40",
            "--profile-steps",
            "40",
            "--gpus",
            "0",
            "1",
        ]
        for setting in candidate_flags:
            command.extend(["--candidate-env", setting])
        for name, value in capacities.items():
            command.extend(["--capacity", f"{task}:fpgs:{name}={value}"])
        entry = {"task": task, "command": command, "status": "running"}
        report["runs"].append(entry)
        save()
        print(f"START {task}", flush=True)
        with (args.output_dir / f"{task}.log").open("w") as log:
            child = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"},
            )
        entry["returncode"] = child.returncode
        result_path = destination / "manifest.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        entry.update(status=result.get("status", "failed"), summary=result.get("summary"), error=result.get("error"))
        save()
        print(f"DONE {task}: {entry['status']} {entry['summary'] or entry['error']}", flush=True)
        if not result.get("final_source_and_idle_guard_pass"):
            report["status"] = "stopped_guard_failure"
            save()
            return 1
    report["status"] = (
        "complete_with_failures" if any(run["status"] != "complete" for run in report["runs"]) else "complete"
    )
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
