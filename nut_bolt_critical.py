# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Critical-force metric for nut/bolt tunneling.

A sustained push creeps through any compliant contact eventually, so "did it
tunnel" is only meaningful against a fixed time budget. This bisects the arm
effort ceiling to find the smallest push that drives the nut through the thread
within that budget, turning each contact-parameter set into one number [N] that
can be compared against the 50-1000 N a Franka actually delivers.

    python nut_bolt_critical.py calibrate m16_tight
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace

from nut_bolt_tunneling import Cfg, Rig

DEFAULT_KE, DEFAULT_KD = 2500.0, 100.0  # newton's real ShapeConfig defaults -> solref (0.02, 1.0)
SPEED_LO, SPEED_HI = 20.0, 3000.0  # [N] arm effort-ceiling bracket
BISECT_STEPS = 7


METRIC = "force"  # force -> bisect the arm effort ceiling; speed -> bisect descent speed


def outcome(cfg: Cfg, x: float) -> dict:
    """Probe one disturbance magnitude.

    force: a position-controlled arm pressing toward the bolt base with effort
    ceiling `x` [N]. This is the drive that discriminates, because a blocked nut
    still carries the full ceiling -- a speed source collapses to kd*v_target
    once the nut stops, so it stops loading the thread exactly when it matters.
    """
    if METRIC == "speed":
        return Rig(replace(cfg, drive="speed", press_speed=x)).run()
    return Rig(replace(cfg, drive="press", max_force=x)).run()


def critical_speed(cfg: Cfg, lo: float = SPEED_LO, hi: float = SPEED_HI, steps: int = BISECT_STEPS) -> dict:
    """Slowest descent that drives the nut through the thread inside the budget.

    A blow-up is not a hold: a config that NaNs under load has failed too, and
    scoring it as "held" silently inverts the metric.
    """
    hi_res = outcome(cfg, hi)
    if not (hi_res["tunneled"] or hi_res["blew_up"]):
        # Even the strongest push in the bracket is held.
        return {"v_crit": float("inf"), "held_at": hi, "blew_up": hi_res["blew_up"], "probe": hi_res}
    lo_res = outcome(cfg, lo)
    if lo_res["tunneled"] or lo_res["blew_up"]:
        return {"v_crit": lo, "held_at": 0.0, "blew_up": lo_res["blew_up"], "probe": lo_res}

    best = hi_res
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        res = outcome(cfg, mid)
        if res["tunneled"] or res["blew_up"]:
            hi, best = mid, res
        else:
            lo = mid
    return {"v_crit": hi, "held_at": lo, "blew_up": best["blew_up"], "probe": best}


def arms(base: Cfg) -> dict[str, Cfg]:
    """The four configurations whose real behavior is already known from training."""
    return {
        "baseline 200Hz/3200Hz/ke2.56e6": base,
        "100Hz collide": replace(base, collide_hz=100.0),
        "1600Hz solver (substeps=8)": replace(base, substeps=8),
        "default soft stiffness": replace(base, ke=DEFAULT_KE, kd=DEFAULT_KD),
    }


EXPECTED = {
    "baseline 200Hz/3200Hz/ke2.56e6": "hold",
    "100Hz collide": "tunnel",
    "1600Hz solver (substeps=8)": "tunnel",
    "default soft stiffness": "tunnel",
}
FRANKA_BAND = (50.0, 1000.0)  # [N] contact force observed when Franka hammers


def main():
    assembly = sys.argv[2] if len(sys.argv) > 2 else "m16_tight"
    frames = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    base = Cfg(assembly=assembly, frames=frames, drive="press")
    # One geometric reference for every arm: where the nut rests on a stiff,
    # well-resolved contact. Arms are compared against this, not their own settle.
    ref = Rig(replace(base, drive="none", ke=1.0e7, kd=1.0e4, frames=40)).run()["nut_z_min"]
    base = replace(base, rest_z_ref=ref)
    print(f"reference rest height = {ref:.6f} m")

    print(f"=== critical force, {assembly}, {frames} frames, Franka band {FRANKA_BAND[0]}-{FRANKA_BAND[1]} N ===")
    rows = {}
    for name, cfg in arms(base).items():
        r = critical_speed(cfg)
        rows[name] = r
        f = r["v_crit"]
        # In-band means a Franka can produce it; the baseline must sit above the band.
        verdict = "tunnel" if f <= FRANKA_BAND[1] else "hold"
        ok = "OK " if verdict == EXPECTED[name] else "MISMATCH"
        shown = "inf" if f == float("inf") else f"{f:7.1f}"
        print(f"  [{ok}] {name:<32} crit={shown}   (expected {EXPECTED[name]}, got {verdict})", flush=True)

    out = f"/tmp/nutbolt_critical_{assembly}.json"
    with open(out, "w") as fh:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "probe"} for k, v in rows.items()}, fh, indent=1)
    print(f"\nwrote {out}")

    mismatches = [
        n
        for n, r in rows.items()
        if ("tunnel" if r["v_crit"] <= FRANKA_BAND[1] else "hold") != EXPECTED[n]
    ]
    if mismatches:
        print(f"HARNESS NOT YET VALID -- disagrees with known behavior on: {mismatches}")
    else:
        print("HARNESS VALID -- reproduces all four known outcomes.")


if __name__ == "__main__":
    main()
