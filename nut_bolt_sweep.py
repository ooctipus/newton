# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run many nut/bolt tunneling configs in one process and tabulate.

Kept separate from the rig so a sweep pays the Warp compile and SDF cook once.

    python nut_bolt_sweep.py discriminate      # can the rig tunnel at all?
    python nut_bolt_sweep.py calibrate         # the four known-truth configs
"""

from __future__ import annotations

import json
import sys
import traceback
from dataclasses import replace

from nut_bolt_tunneling import Cfg, Rig

# Contact stiffness Newton falls back to when nothing is authored. This is the
# "default stiffness" arm of the calibration -- the one Franka hammers through.
DEFAULT_KE = 1.0e4
DEFAULT_KD = 1.0e2


def run(cfg: Cfg) -> dict:
    try:
        return Rig(cfg).run()
    except Exception as exc:  # noqa: BLE001 - a sweep records failures rather than dying
        return {**cfg.__dict__, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-400:]}


def table(rows: list[dict], keys: list[str]) -> str:
    hdr = keys + ["descent_mm", "pitches", "tunneled", "s"]
    out = [" | ".join(f"{h:>12}" for h in hdr), "-" * (15 * len(hdr))]
    for r in rows:
        if "error" in r:
            out.append(" | ".join(f"{str(r.get(k, ''))[:12]:>12}" for k in keys) + f" | ERROR {r['error'][:60]}")
            continue
        vals = [f"{r.get(k, ''):>12}" if not isinstance(r.get(k), float) else f"{r[k]:>12.6g}" for k in keys]
        vals += [
            f"{r['descent_m'] * 1000:>12.3f}",
            f"{r['descent_pitches']:>12.2f}",
            f"{str(r['tunneled']):>12}",
            f"{r['wall_s']:>12.1f}",
        ]
        out.append(" | ".join(vals))
    return "\n".join(out)


def discriminate(base: Cfg) -> list[dict]:
    """Push progressively harder on the known-good config until something gives."""
    rows = []
    for force in (0.0, 100.0, 500.0, 2_000.0, 10_000.0, 50_000.0, 200_000.0):
        r = run(replace(base, force=force, drive="force"))
        rows.append(r)
        print(table([r], ["assembly", "force"]).splitlines()[-1], flush=True)
    for speed in (1.0, 5.0, 20.0, 100.0):
        r = run(replace(base, drive="impulse", impact_speed=speed))
        rows.append(r)
        print(table([r], ["assembly", "impact_speed"]).splitlines()[-1], flush=True)
    return rows


def calibrate(base: Cfg, force: float) -> list[dict]:
    """The four configs whose real-world behavior is already known."""
    arms = {
        "baseline 200Hz/3200Hz/2.56e6": base,
        "100Hz collide": replace(base, collide_hz=100.0),
        "1600Hz solver (substeps 8)": replace(base, substeps=8),
        "default (soft) stiffness": replace(base, ke=DEFAULT_KE, kd=DEFAULT_KD),
    }
    rows = []
    for name, cfg in arms.items():
        r = run(replace(cfg, force=force))
        r["arm"] = name
        rows.append(r)
        got = "ERROR" if "error" in r else ("TUNNELED" if r["tunneled"] else "held")
        extra = "" if "error" in r else f"  descent={r['descent_m'] * 1000:.3f} mm ({r['descent_pitches']:.2f} pitch)"
        print(f"  {name:<32} -> {got}{extra}", flush=True)
    return rows


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "discriminate"
    assembly = sys.argv[2] if len(sys.argv) > 2 else "m16_tight"
    base = Cfg(assembly=assembly, frames=90)

    if mode == "discriminate":
        rows = discriminate(base)
    elif mode == "calibrate":
        force = float(sys.argv[3]) if len(sys.argv) > 3 else 500.0
        print(f"=== calibration @ {force} N, {assembly} ===")
        rows = calibrate(base, force)
    else:
        raise SystemExit(f"unknown mode {mode}")

    out = f"/tmp/nutbolt_{mode}_{assembly}.json"
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=1, sort_keys=True, default=str)
    print(f"\nwrote {out}")
