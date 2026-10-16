# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Screen execution recipes with the existing paired, checked variant owner."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    """Reuse selected survey pins and capacities without changing physics budgets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--survey", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", action="append")
    parser.add_argument("--rounds", type=int, default=1)
    args = parser.parse_args()
    raw = args.survey.read_bytes()
    survey = json.loads(raw)
    selected = {task: entry for task, entry in survey["tasks"].items() if entry.get("fpgs_optins") == []}
    if args.task and set(args.task) - selected.keys():
        parser.error("Select only generic-recipe tasks present in the frozen survey")
    if args.rounds < 1:
        parser.error("Rounds must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "scope": __doc__,
        "survey": str(args.survey.resolve()),
        "survey_sha256": hashlib.sha256(raw).hexdigest(),
        "status": "running",
        "tasks": [],
    }

    def save():
        (args.output_dir / "survey.json").write_text(json.dumps(report, indent=2) + "\n")

    for task, entry in selected.items():
        if args.task and task not in args.task:
            continue
        evidence = Path(entry["fpgs_manifest"])
        if hashlib.sha256(evidence.read_bytes()).hexdigest() != entry["fpgs_manifest_sha256"]:
            raise RuntimeError(f"Changed prior evidence: {evidence}")
        previous = json.loads(evidence.read_text())["settings"]
        runtime = previous["candidate"]
        for root, pin in ((runtime, survey["pins"]["candidate"]), (previous["isaaclab"], survey["pins"]["isaaclab"])):
            if subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip() != pin:
                raise RuntimeError(f"Selected runtime pin changed: {root}")
        output = args.output_dir / task
        command = [
            sys.executable,
            str(Path(__file__).with_name("compare_variants.py")),
            "--isaaclab",
            previous["isaaclab"],
            "--baseline",
            runtime,
            "--candidate",
            runtime,
            "--output-dir",
            str(output),
            "--task",
            task,
            "--gpus",
            "0",
            "1",
            "--num-envs",
            "4096",
            "--rounds",
            str(args.rounds),
            "--warmup-steps",
            "200",
            "--steps",
            "40",
            "--profile-steps",
            "40",
            "--candidate-solver-attr",
            "grouped_dynamics=True",
        ]
        if task == "g1-flat":
            for flag in ("SPARSE_FACTOR", "SPARSE_LEVEL_UPDATE", "PACKED_TREE_KINETIC_STATE", "G1_CHAIN_SCAN"):
                command.extend(("--candidate-env", f"FEATHER_PGS_{flag}=1"))
        else:
            command.extend(("--candidate-solver-attr", "lazy_kinematics=True"))
        for capacity in previous["capacity"]:
            command.extend(("--capacity", capacity))
        record = {"task": task, "command": command, "status": "running", "manifest": str(output / "manifest.json")}
        report["tasks"].append(record)
        save()
        print("START", task, flush=True)
        with (args.output_dir / f"{task}.log").open("w") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        result = json.loads((output / "manifest.json").read_text()) if (output / "manifest.json").exists() else {}
        record.update(
            status=result.get("status", "failed"),
            returncode=process.returncode,
            summary=result.get("summary"),
            error=result.get("error"),
        )
        save()
        print("DONE", task, record["status"], json.dumps(record["summary"] or record["error"]), flush=True)
        if not result.get("final_source_and_idle_guard_pass"):
            report["status"] = "stopped_guard_failure"
            save()
            return 1
    report["status"] = (
        "complete" if all(t["status"] == "complete" for t in report["tasks"]) else "complete_with_failures"
    )
    save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
