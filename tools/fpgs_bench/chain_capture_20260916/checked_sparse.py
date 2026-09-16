# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Extend only the original untimed observer's exact kinetic factory keys."""

import hashlib
from pathlib import Path

RETAINED = Path("/tmp/fpgs-g1-paired-gs-checked-t0NFbw6H/checked_sparse.py")
RETAINED_PIN = "5bea2bec51735a02f34a6623b136ca557a84df4678683ff888cab0297bf75046"
source = RETAINED.read_text()
if hashlib.sha256(source.encode()).hexdigest() != RETAINED_PIN:
    raise RuntimeError("The retained G1 boundary observer has changed")
seam = '    if keys != {"repair": "g1_kinetic_repair44", "finish": "g1_kinetic_finish44", "predictor": "g1_kinetic_predict43"}:\n'
replacement = """    chain_flag = os.environ.get("FEATHER_PGS_G1_CHAIN_SCAN")
    if chain_flag not in ("0", "1"):
        raise RuntimeError("Require explicit Boolean chain traversal on both arms")
    chain = chain_flag == "1"
    if getattr(owner, "chain_scan", False) is not chain:
        raise RuntimeError("Requested chain traversal differs from the actual owner")
    suffix = "_chain" if chain else ""
    result["chain_scan"] = {"requested": chain, "observed": chain, "check_pass": True}
    if keys != {"repair": "g1_kinetic_repair44" + suffix, "finish": "g1_kinetic_finish44" + suffix, "predictor": "g1_kinetic_predict43" + suffix}:
"""
if source.count(seam) != 1:
    raise RuntimeError("The original exact-key observation seam changed")
source = source.replace(seam, replacement)
spectral_seam = '    result["solve_key"] = getattr(sparse.kernels.solve, "key", None)\n'
spectral_replacement = """    spectral_flag = os.environ.get("FEATHER_PGS_SPARSE_SPECTRAL_TANGENTS")
    if spectral_flag not in ("0", "1"):
        raise RuntimeError("Require explicit spectral tangent flag on both arms")
    spectral_wanted = spectral_flag == "1"
    spectral_actual = getattr(sparse, "spectral_tangents", False)
    if type(spectral_actual) is not bool or spectral_actual != spectral_wanted:
        raise RuntimeError("Requested spectral tangent owner is not active")
    if spectral_wanted:
        from newton._src.solvers.feather_pgs.sparse_spectral_tangents import get_solve_kernel
        import warp as wp
        solve = "sparse_spectral_tangent43_s18_c100"
        if sparse.kernels.solve is not get_solve_kernel():
            raise RuntimeError("Spectral solve is not the exact cached factory")
        arguments = [argument.label for argument in sparse.kernels.solve.adj.args]
        if len(arguments) != 15 or arguments[5] != "cfm":
            raise RuntimeError("Unexpected spectral current-CFM argument ownership")
        cfm = solver.row_cfm
        if cfm.shape != (int(solver.world_count), 100) or cfm.dtype is not wp.float32:
            raise RuntimeError("Unexpected existing current-CFM array layout")
    result["spectral_tangents"] = {"requested": spectral_wanted, "observed": spectral_actual,
        "kernel_key": solve, "check_pass": True}
    result["solve_key"] = getattr(sparse.kernels.solve, "key", None)
"""
if source.count(spectral_seam) != 1:
    raise RuntimeError("The original solve-factory observation seam changed")
source = source.replace(spectral_seam, spectral_replacement)
# __file__ deliberately remains this wrapper: its digest pins the complete
# source transformation and the retained observer's required SHA256.
exec(compile(source, str(RETAINED), "exec"), globals())
