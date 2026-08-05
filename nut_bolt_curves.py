# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Penetration-vs-load curves for nut/bolt contact settings.

A binary "did it tunnel" hides the margin and is sensitive to where the
threshold is drawn. The curve shows how far the nut sinks past its geometric
rest as the arm pushes harder, so configurations can be ranked by how much load
they survive rather than by a single pass/fail.

    python nut_bolt_curves.py m16_tight
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace

from nut_bolt_tunneling import THREAD_PITCH, Cfg, Rig

FORCES = (50.0, 100.0, 200.0, 400.0, 800.0, 1600.0)
DEFAULT_KE, DEFAULT_KD = 2500.0, 100.0  # newton ShapeConfig defaults -> solref (0.02, 1.0)


def arms(base: Cfg) -> dict[str, Cfg]:
    return {
        "baseline 200Hz/3200Hz": base,
        "100Hz collide": replace(base, collide_hz=100.0),
        "1600Hz solver": replace(base, substeps=8),
        "800Hz solver": replace(base, substeps=4),
        "default soft ke": replace(base, ke=DEFAULT_KE, kd=DEFAULT_KD),
    }


def main():
    assembly = sys.argv[1] if len(sys.argv) > 1 else "m16_tight"
    frames = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    pitch = THREAD_PITCH[assembly.split("_")[0]]

    base = Cfg(assembly=assembly, frames=frames, drive="press")
    ref = Rig(replace(base, drive="none", ke=1.0e7, kd=1.0e4, frames=40)).run()["nut_z_min"]
    base = replace(base, rest_z_ref=ref)
    print(f"=== {assembly}: penetration past geometric rest, in thread pitches (pitch={pitch * 1000:.2f} mm) ===")
    print(f"reference rest z = {ref:.6f} m\n")

    hdr = f"{'arm':<24}" + "".join(f"{f:>9.0f}N" for f in FORCES)
    print(hdr)
    print("-" * len(hdr))

    out = {}
    for name, cfg in arms(base).items():
        cells, row = [], {}
        for f in FORCES:
            r = Rig(replace(cfg, max_force=f)).run()
            if r["blew_up"]:
                cells.append(f"{'NaN':>10}")
                row[f] = "nan"
            else:
                cells.append(f"{r['penetration_pitches']:>10.2f}")
                row[f] = round(r["penetration_pitches"], 3)
        print(f"{name:<24}" + "".join(cells), flush=True)
        out[name] = row

    path = f"/tmp/nutbolt_curves_{assembly}.json"
    with open(path, "w") as fh:
        json.dump({"assembly": assembly, "pitch_m": pitch, "rest_z_ref": ref, "curves": out}, fh, indent=1)
    print(f"\n(>1.00 pitch = the nut sank a full turn without rotating)\nwrote {path}")


if __name__ == "__main__":
    main()
