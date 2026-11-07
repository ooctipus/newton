# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Per task, the fields of the solver's path-selection record that differ between two suite runs.

    python3 record_diff.py BASE_DIR CAND_DIR [--routes]

Reads each task's ``result.json`` (its first check's ``selection.solver`` record) in two run_suite_par.py outputs and
prints every differing field, ignoring reasons, environment, measurements, timing, pins and source identity. With
``--routes`` only the route choices are compared (``routes``, ``sparse_mass_matrix``, ``cached_dynamics``,
``pgs_kernel``), kernel names included: the proof that the same code runs. Exit status 1 when any task differs.
"""

import argparse
import json
import os
import sys

IGNORED = ("reason", "environment", "measurements", "pins", "timing", "lane_ms", "seconds", "source_identity")
ROUTE_KEYS = ("routes", "sparse_mass_matrix", "cached_dynamics", "pgs_kernel")


def flatten(value, prefix=""):
    if isinstance(value, dict):
        for key, item in value.items():
            if any(word in key for word in IGNORED):
                continue
            yield from flatten(item, f"{prefix}.{key}" if prefix else key)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from flatten(item, f"{prefix}.{index}")
    else:
        yield prefix, value


def record(directory, task, routes):
    with open(os.path.join(directory, task, "result.json"), encoding="utf-8") as handle:
        result = json.load(handle)
    solver = ((result.get("checks") or [{}])[0].get("selection") or {}).get("solver") or {}
    return dict(flatten({k: v for k, v in solver.items() if k in ROUTE_KEYS} if routes else solver))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("base")
    parser.add_argument("cand")
    parser.add_argument("--routes", action="store_true", help="compare the route choices only")
    args = parser.parse_args()
    same = total = fields = 0
    for task in sorted(os.listdir(args.base)):
        if not all(os.path.exists(os.path.join(d, task, "result.json")) for d in (args.base, args.cand)):
            continue
        total += 1
        base, cand = record(args.base, task, args.routes), record(args.cand, task, args.routes)
        fields += len(base)
        diffs = [
            (key, base.get(key, "<absent>"), cand.get(key, "<absent>"))
            for key in sorted(set(base) | set(cand))
            if base.get(key, "<absent>") != cand.get(key, "<absent>")
        ]
        same += not diffs
        print(f"{task}: {len(base)} fields, {'same' if not diffs else f'{len(diffs)} differ'}")
        for key, a, b in diffs[:40]:
            print(f"    {key}: {json.dumps(a)[:100]} -> {json.dumps(b)[:100]}")
    print(
        f"== {same}/{total} tasks with every {'route' if args.routes else 'selection-record'} field equal "
        f"({fields} fields compared)"
    )
    return 0 if same == total else 1


if __name__ == "__main__":
    sys.exit(main())
