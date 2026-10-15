# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reuse the fixed-import paired protocol with untimed compact-workspace checks."""

import hashlib
import importlib.util
import json
import signal
import sys
from pathlib import Path
from types import ModuleType

ADAPTER = Path("/tmp/fpgs-kinetic-fixed-variants-TZRkIPYw/run.py")
PIN = "c062388f23f82c524690f272fb2d14141ed4126db641de6dcd86f8ade5bfe416"
VARIANTS_PIN = "48406c079588d3088fc25c1cdce2bc2659aaec28539013e5811c9c55378445b5"
HERE = Path(__file__).resolve().parent
FLAG = "FEATHER_PGS_FRANKA_COMPACT_WORKSPACE"


def selected_variants(tools):
    """Permit one idle GPU while preserving every original per-run guard."""
    path = tools / "compare_variants.py"
    source = path.read_text()
    if hashlib.sha256(source.encode()).hexdigest() != VARIANTS_PIN:
        raise RuntimeError("Original paired variant owner changed")
    seam = "len(args.gpus) != 2 or len(set(args.gpus)) != 2"
    if source.count(seam) != 1:
        raise RuntimeError("Unexpected original GPU selection guard")
    source = source.replace(seam, "len(args.gpus) not in (1, 2) or len(set(args.gpus)) != len(args.gpus)")
    source = source.replace(
        "Select exactly two distinct nonnegative GPU indices", "Select one or two distinct nonnegative GPU indices"
    )
    variants = ModuleType("selected_compact_workspace_variants")
    variants.__file__ = str(path)
    exec(compile(source, str(path), "exec"), variants.__dict__)
    if Path(variants.owner.__file__).resolve() != (tools / "compare_backends.py").resolve():
        raise RuntimeError("Wrong original paired capture owner")
    return variants


def main():
    """Preserve original budgets, capture, source guards and cleanup."""
    if hashlib.sha256(ADAPTER.read_bytes()).hexdigest() != PIN:
        raise RuntimeError("The retained fixed-import adapter has changed")
    spec = importlib.util.spec_from_file_location("fixed_compact_workspace_adapter", ADAPTER)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    adapter.EXTRA.extend(HERE / name for name in ("run.py", "checked_compact_workspace.py", "nsys_checked.sh"))
    sys.path.insert(0, str(adapter.TOOLS))
    variants = selected_variants(adapter.TOOLS)
    owner = variants.owner
    adapter.install(owner)
    old_batch, old_check = owner.make_batch, owner.check_overflow_result
    helper, shell = HERE / "checked_compact_workspace.py", HERE / "nsys_checked.sh"

    def make_batch(*args, **kwargs):
        batch = old_batch(*args, **kwargs)
        for run in batch:
            if run["task"] != "franka" or run["environment"].get(FLAG) not in ("0", "1"):
                raise ValueError("Require Franka and an explicit compact-workspace mode on both arms")
            if run["command"][:2] != ["bash", str(adapter.TOOLS / "nsys_checked.sh")]:
                raise RuntimeError("Expected the unchanged checked-capture shell")
            run["command"][1] = str(shell)
        return batch

    def check(run, drivers):
        old_check(run, drivers)
        if str(helper) not in drivers or str(shell) not in drivers:
            raise RuntimeError("The compact-workspace observer/shell was not source-pinned")
        report = json.loads((Path(run["output_dir"]) / "capture_checks.json").read_text())
        requested = run["environment"][FLAG] == "1"
        for entry in report["boundaries"]:
            item = entry.get("compact_workspace", {})
            if (
                item.get("check_pass") is not True
                or item.get("requested") is not requested
                or item.get("observed") is not requested
                or item.get("observer_sha256") != drivers[str(helper)]
                or item.get("status_nonzero") != 0
            ):
                raise RuntimeError("Missing or inconsistent compact-workspace boundary observation")

    owner.make_batch, owner.check_overflow_result = make_batch, check
    signal.signal(signal.SIGTERM, owner.interrupted)
    return variants.main()


if __name__ == "__main__":
    raise SystemExit(main())
