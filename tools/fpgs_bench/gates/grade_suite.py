# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Grade suite runs per task: time against a reference arm and predicted bands, physics against the reference spread.

    python3 grade_suite.py --ref m0-patch-1,m0-patch-2,m0-patch-3 --cand p1-patch-1,p1-patch-2,p1-patch-3 \
        [--bands bands.json] [--other p1-point-1,p1-point-2]

Time: median over runs of each run's mean_ms. Physics: after_run body_velocity_rms, median over runs, against the
reference runs' [min, max]; 'in' when inside it, else the deviation in reference spreads. Contacts: collision_work
raw_contacts (reducer scenes only) median vs reference median, gate +-5%. Flags: any capacity flag in any check of any
candidate run. --other prints a second arm's median and the candidate/other ratio (e.g. patch/point).
"""

import argparse
import json
import math
import statistics
from pathlib import Path


def load(dirs):
    runs = []
    for d in dirs.split(","):
        data = json.loads((Path(d) / "summary.json").read_text())
        runs.append({r["scene"]: r for r in data["results"] if r.get("complete")})
    return runs


def check(result, label):
    return next(c for c in result["checks"] if c["label"] == label)


def flags(result):
    hits = []
    for c in result["checks"]:
        for group, values in (c.get("capacity_flags") or {}).items():
            hits += [f"{c['label']}:{group}.{k}" for k, v in values.items() if v]
    return hits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref", required=True)
    p.add_argument("--cand", required=True)
    p.add_argument("--other")
    p.add_argument("--bands", help="JSON {task: [lo, hi]} of predicted candidate ms")
    p.add_argument(
        "--rel-bands", help="JSON {task: [lo, hi]} of predicted candidate/ref median ratios (same-window A/B)"
    )
    a = p.parse_args()
    ref, cand = load(a.ref), load(a.cand)
    other = load(a.other) if a.other else None
    bands = json.loads(Path(a.bands).read_text()) if a.bands else {}
    rel = json.loads(Path(a.rel_bands).read_text()) if a.rel_bands else {}
    head = f"{'task':15s} {'ref ms':>8s} {'cand ms':>8s} {'d%':>6s} {'band':>13s} {'grade':>6s}  {'runs':22s} "
    head += f"{'rms ref [min,max]':>22s} {'rms cand':>8s} {'phys':>7s} {'contacts d%':>11s} flags"
    if other:
        head += f"  {'other ms':>8s} {'cand/other':>10s}"
    print(head)
    ratios = []
    for task in sorted(cand[0]):
        if not all(task in r for r in ref + cand):
            continue
        rm = statistics.median(r[task]["timing"]["mean_ms"] for r in ref)
        runs = [r[task]["timing"]["mean_ms"] for r in cand]
        cm = statistics.median(runs)
        grade = ""
        band = bands.get(task)
        if task in rel:
            band = [round(rm * rel[task][0], 3), round(rm * rel[task][1], 3)]
        if band:
            lo, hi = band
            grade = "HIT" if lo <= cm <= hi else ("BEAT" if cm < lo else "MISS")
        rr = [check(r[task], "after_run")["body_velocity_rms"] for r in ref]
        cr = statistics.median(check(r[task], "after_run")["body_velocity_rms"] for r in cand)
        lo_r, hi_r = min(rr), max(rr)
        spread = hi_r - lo_r
        if lo_r <= cr <= hi_r:
            phys = "in"
        else:
            dev = (cr - hi_r) if cr > hi_r else (cr - lo_r)
            phys = f"{dev / spread:+.1f}s" if spread > 0 else "out"
        rc = [(check(r[task], "after_run")["collision_work"] or {}).get("raw_contacts") for r in ref]
        cc = [(check(r[task], "after_run")["collision_work"] or {}).get("raw_contacts") for r in cand]
        cdelta = ""
        if all(rc) and all(cc):
            cdelta = f"{100 * (statistics.median(cc) / statistics.median(rc) - 1):+.2f}"
        fl = sorted({f for r in cand for f in flags(r[task])})
        line = f"{task:15s} {rm:8.3f} {cm:8.3f} {100 * (cm / rm - 1):+6.1f} "
        line += (
            f"{(f'{band[0]}-{band[1]}' if band else ''):>13s} {grade:>6s}  {','.join(f'{x:.3f}' for x in runs):22s} "
        )
        line += (
            f"{f'{rm_fmt(rr)} [{lo_r:.4f},{hi_r:.4f}]':>22s} {cr:8.4f} {phys:>7s} {cdelta:>11s} {' '.join(fl) or '-'}"
        )
        if other and all(task in r for r in other):
            om = statistics.median(r[task]["timing"]["mean_ms"] for r in other)
            ratios.append(cm / om)
            line += f"  {om:8.3f} {cm / om:10.3f}"
        print(line)
    if ratios:
        print(f"geomean cand/other over {len(ratios)} tasks: {math.exp(sum(map(math.log, ratios)) / len(ratios)):.3f}")


def rm_fmt(values):
    return f"{statistics.median(values):.4f}"


if __name__ == "__main__":
    main()
