# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reuse the source-pinned G1 paired capture with the chain-owner observer."""

import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType

ORIGINAL = Path("/tmp/fpgs-g1-paired-gs-checked-t0NFbw6H/run_live_sparse_checked.py")
PIN = "23acfade1339b660e03d92c273d92b767081751c7feb2a53eea4d52c2cd5aef8"
VARIANTS_PIN = "48406c079588d3088fc25c1cdce2bc2659aaec28539013e5811c9c55378445b5"


def selected_variants(module):
    """Permit one idle GPU without weakening original per-run checks."""
    path = module.adapter.TOOLS / "compare_variants.py"
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
    variants = ModuleType("selected_limit_jacobi_variants")
    variants.__file__ = str(path)
    exec(compile(source, str(path), "exec"), variants.__dict__)
    if variants.owner is not module.owner:
        raise RuntimeError("Wrong original paired capture owner")
    return variants


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
            if spectral:
                solve = "sparse_spectral_tangent43_s18_c100"
            limit_flag = run["environment"].get("FEATHER_PGS_SPARSE_LIMIT_JACOBI")
            if limit_flag not in ("0", "1"):
                raise RuntimeError("Require explicit limit-Jacobi policy on both arms")
            limit_policy = limit_flag == "1"
            if limit_policy:
                if not spectral:
                    raise RuntimeError("Limit Jacobi requires spectral tangents")
                solve = "sparse_spectral_limit_jacobi43_s18_c100"
            if kinetic.get("limit_jacobi") != {"requested": limit_policy, "observed": limit_policy,
                    "kernel_key": solve, "check_pass": True}:
                raise RuntimeError("Missing actual limit-Jacobi owner observation")
            if kinetic.get("spectral_tangents") != {"requested": spectral, "observed": spectral,
                    "kernel_key": solve, "check_pass": True}:
                raise RuntimeError("Missing actual spectral tangent owner observation")
            if kinetic.get("solve_key") != solve:
"""
    if source.count(seam) != 1:
        raise RuntimeError("The original solve-key validation seam changed")
    exec(compile(source.replace(seam, replacement), str(ORIGINAL), "exec"), module.__dict__)
    module.variants = selected_variants(module)
    module.FLAGS.add("FEATHER_PGS_SPARSE_SPECTRAL_TANGENTS")
    module.FLAGS.add("FEATHER_PGS_SPARSE_LIMIT_JACOBI")
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
