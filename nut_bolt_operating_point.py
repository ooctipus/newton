# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Find the cheapest contact setting that survives a Franka-scale load, per thread size.

MuJoCo's REFSAFE clamps the contact time constant to >= 2*dt, and ke/kd map to
solref as (2/kd, (kd/2)/sqrt(ke)). So the stiffest critically-damped contact a
given solver rate can actually hold is

    kd = solver_hz          (time constant sits exactly at the REFSAFE limit)
    ke = (kd/2)**2          (damping ratio 1)

which reproduces the shipping M16 numbers at 3200 Hz: kd=3200, ke=2.56e6. Asking
for more stiffness than the rate supports does not give a stiffer contact, it
gives a silently clamped one.

This walks that family upward until penetration stays under a thread pitch.

    python nut_bolt_operating_point.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace

from nut_bolt_tunneling import THREAD_PITCH, Cfg, Rig

# (collide_hz, substeps) -> solver_hz; kd/ke follow from the rule above.
LADDER = ((200, 16), (200, 32), (400, 16), (400, 32), (800, 32))
FORCES = (100.0, 400.0, 1600.0)
SAFE_PITCHES = 1.0
DIVERGED_PITCHES = 50.0  # beyond this it is numerical divergence, not descent


def rung(collide_hz: int, substeps: int) -> tuple[float, float, float]:
    solver_hz = collide_hz * substeps
    kd = float(solver_hz)
    ke = (kd / 2.0) ** 2
    return solver_hz, ke, kd


def evaluate(base: Cfg, collide_hz: int, substeps: int) -> dict:
    solver_hz, ke, kd = rung(collide_hz, substeps)
    worst, blew, cells = 0.0, False, {}
    for f in FORCES:
        r = Rig(replace(base, collide_hz=float(collide_hz), substeps=substeps, ke=ke, kd=kd, max_force=f)).run()
        if r["blew_up"]:
            blew = True
            cells[f] = "nan"
            continue
        p = r["penetration_pitches"]
        cells[f] = round(p, 3)
        worst = max(worst, p)
    return {
        "collide_hz": collide_hz,
        "substeps": substeps,
        "solver_hz": solver_hz,
        "ke": ke,
        "kd": kd,
        "per_force": cells,
        "worst_pitches": worst,
        "blew_up": blew,
        "safe": (not blew) and worst < SAFE_PITCHES,
        "diverged": worst > DIVERGED_PITCHES,
    }


def main():
    assemblies = sys.argv[1:] or ["m16_tight", "m12_tight", "m8_tight", "m4_tight"]
    out = {}
    for assembly in assemblies:
        pitch = THREAD_PITCH[assembly.split("_")[0]]
        base = Cfg(assembly=assembly, frames=60, drive="press")
        ref = Rig(replace(base, drive="none", ke=1.0e7, kd=1.0e4, frames=40)).run()["nut_z_min"]
        base = replace(base, rest_z_ref=ref)

        print(f"\n=== {assembly} (pitch {pitch * 1000:.2f} mm) ===")
        print(f"{'collide':>8}{'substeps':>10}{'solver':>9}{'ke':>11}{'kd':>8}   worst   verdict")
        rows = []
        for collide_hz, substeps in LADDER:
            r = evaluate(base, collide_hz, substeps)
            rows.append(r)
            worst = "NaN" if r["blew_up"] else (f"{r['worst_pitches']:.2f}p" if not r["diverged"] else "diverged")
            verdict = "SAFE" if r["safe"] else "fails"
            print(
                f"{r['collide_hz']:>8}{r['substeps']:>10}{r['solver_hz']:>9}"
                f"{r['ke']:>11.3g}{r['kd']:>8.0f}{worst:>8}   {verdict}",
                flush=True,
            )
            if r["safe"]:
                break
        out[assembly] = rows
        safe = next((r for r in rows if r["safe"]), None)
        if safe:
            print(
                f"  -> cheapest safe: {safe['collide_hz']} Hz collide x {safe['substeps']} substeps "
                f"= {safe['solver_hz']} Hz solver, ke={safe['ke']:.3g}, kd={safe['kd']:.0f}"
            )
        else:
            print("  -> no rung on this ladder held; needs a finer thread model or another knob")

    path = "/tmp/nutbolt_operating_points.json"
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
