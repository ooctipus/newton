# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reuse the source-pinned G1 paired capture with the chain-owner observer."""

import hashlib
import importlib.util
from pathlib import Path

ORIGINAL = Path("/tmp/fpgs-g1-paired-gs-checked-t0NFbw6H/run_live_sparse_checked.py")
PIN = "23acfade1339b660e03d92c273d92b767081751c7feb2a53eea4d52c2cd5aef8"


def main():
    """Keep all original source, capacity, timing and cleanup checks."""
    if hashlib.sha256(ORIGINAL.read_bytes()).hexdigest() != PIN:
        raise RuntimeError("The retained G1 paired owner has changed")
    spec = importlib.util.spec_from_file_location("retained_g1_paired_capture", ORIGINAL)
    module = importlib.util.module_from_spec(spec)
    source = ORIGINAL.read_text()
    seam = '            if kinetic.get("solve_key") != solve:\n'
    replacement = """            spectral_flag = run["environment"].get("FEATHER_PGS_SPARSE_SPECTRAL_TANGENTS")
            if spectral_flag not in ("0", "1"):
                raise RuntimeError("Require explicit spectral tangent flags on both arms")
            spectral = spectral_flag == "1"
            staged_flag = run["environment"].get("FEATHER_PGS_SPARSE_STAGED_ROWS")
            if staged_flag not in ("0", "1"):
                raise RuntimeError("Require explicit staged-row flags on both arms")
            staged = staged_flag == "1"
            if staged and not spectral:
                raise RuntimeError("Staged-row owner requires the spectral owner")
            if spectral:
                solve = "sparse_spectral_tangent43_s18_c100"
            if staged:
                solve = "sparse_staged_spectral_tangent43_s18_c100"
            if kinetic.get("spectral_tangents") != {"requested": spectral, "observed": spectral,
                    "kernel_key": solve, "check_pass": True}:
                raise RuntimeError("Missing actual spectral tangent owner observation")
            if kinetic.get("staged_rows") != {"requested": staged, "observed": staged,
                    "kernel_key": solve, "fast_capacity": 32, "check_pass": True}:
                raise RuntimeError("Missing actual staged-row owner observation")
            if kinetic.get("solve_key") != solve:
"""
    if source.count(seam) != 1:
        raise RuntimeError("The original solve-key validation seam changed")
    exec(compile(source.replace(seam, replacement), str(ORIGINAL), "exec"), module.__dict__)
    module.FLAGS.add("FEATHER_PGS_SPARSE_SPECTRAL_TANGENTS")
    module.FLAGS.add("FEATHER_PGS_SPARSE_STAGED_ROWS")
    module.FLAGS.add("NEWTON_HEIGHTFIELD_ADAPTIVE_MANIFOLD")
    module.FLAGS.add("NEWTON_HEIGHTFIELD_PAIR_CSR")
    module.FLAGS.add("NEWTON_HEIGHTFIELD_PAIR_CSR_SHELL")
    module.__file__ = str(Path(__file__).resolve())
    module.adapter.EXTRA.extend(
        Path(__file__).with_name(name).resolve() for name in ("run.py", "checked_sparse.py", "nsys_sparse_checked.sh")
    )
    return module.main()


if __name__ == "__main__":
    raise SystemExit(main())
