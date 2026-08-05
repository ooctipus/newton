# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Search hydroelastic contact parameters against the same tunneling metric.

Pure SDF has a known-good operating point (ke 2.56e6 / kd 3200 at 200 Hz
collide, 3200 Hz solver). Hydroelastic does not, so this scores candidate
`kh` values on the same penetration-past-rest scale and reports which ones both
stay stable and resist the load a Franka can deliver.

    python nut_bolt_hydro_search.py m16_tight
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace

from nut_bolt_tunneling import THREAD_PITCH, Cfg, Rig

# The hydro example notes kh=1e10 leaves a ~95 ms solref constant (~45 substeps)
# and a visibly cocked nut, and ships 1e11. Bracket around that.
KH_CANDIDATES = (1e9, 1e10, 1e11, 1e12, 1e13)
FORCES = (100.0, 400.0, 1600.0)
# Hydroelastic cooks a pressure field; the example drops SDF resolution to 128.
HYDRO_SDF_RESOLUTION = 128


def main():
    assembly = sys.argv[1] if len(sys.argv) > 1 else "m16_tight"
    frames = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    pitch = THREAD_PITCH[assembly.split("_")[0]]

    base = Cfg(assembly=assembly, frames=frames, drive="press")
    ref = Rig(replace(base, drive="none", ke=1.0e7, kd=1.0e4, frames=40)).run()["nut_z_min"]
    base = replace(base, rest_z_ref=ref, is_hydroelastic=True, sdf_resolution=HYDRO_SDF_RESOLUTION)

    print(f"=== {assembly}: hydroelastic kh search, penetration in pitches (pitch={pitch * 1000:.2f} mm) ===")
    print(f"reference rest z = {ref:.6f} m (measured on the stiff SDF contact)\n")
    hdr = f"{'kh':>10}" + "".join(f"{f:>9.0f}N" for f in FORCES) + "   verdict"
    print(hdr)
    print("-" * len(hdr))

    out = {}
    for kh in KH_CANDIDATES:
        cells, row, worst, blew = [], {}, 0.0, False
        for f in FORCES:
            r = Rig(replace(base, kh=kh, max_force=f)).run()
            if r["blew_up"]:
                cells.append(f"{'NaN':>10}")
                row[f] = "nan"
                blew = True
            else:
                cells.append(f"{r['penetration_pitches']:>10.2f}")
                row[f] = round(r["penetration_pitches"], 3)
                worst = max(worst, r["penetration_pitches"])
        verdict = "UNSTABLE" if blew else ("holds" if worst < 1.0 else f"tunnels ({worst:.1f}p)")
        print(f"{kh:>10.0e}" + "".join(cells) + f"   {verdict}", flush=True)
        out[f"{kh:.0e}"] = {"per_force": row, "worst_pitches": worst, "blew_up": blew, "verdict": verdict}

    path = f"/tmp/nutbolt_hydro_{assembly}.json"
    with open(path, "w") as fh:
        json.dump({"assembly": assembly, "pitch_m": pitch, "rest_z_ref": ref, "kh": out}, fh, indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
