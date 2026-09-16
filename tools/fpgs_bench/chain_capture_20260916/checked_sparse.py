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
collision_seam = '\n    return {**result, "check_pass": True}\n'
collision_replacement = """
    flag = os.environ.get("NEWTON_HEIGHTFIELD_ADAPTIVE_MANIFOLD")
    if flag not in ("0", "1"):
        raise RuntimeError("Require explicit adaptive manifold flag on both arms")
    wanted = flag == "1"
    actual = getattr(narrow, "_heightfield_adaptive_manifold", False)
    from newton._src.sim.collide import write_contact
    if wanted:
        from newton._src.geometry.heightfield_manifold import create_export_kernel
        expected = create_export_kernel(write_contact)
    else:
        from newton._src.geometry.contact_reduction_global import create_export_reduced_contacts_kernel
        expected = create_export_reduced_contacts_kernel(write_contact)
    # The original factory is not cached; its stable key is the exact dispatch
    # check for the off arm. The new cached factory also permits identity proof.
    observed = narrow.export_reduced_contacts_kernel
    if type(actual) is not bool or actual != wanted or observed.key != expected.key:
        raise RuntimeError("Requested adaptive exporter differs from actual owner")
    if wanted and observed is not expected:
        raise RuntimeError("Adaptive exporter is not the exact cached factory")
    result["adaptive_manifold"] = {"requested": wanted, "observed": actual,
        "kernel_key": observed.key, "check_pass": True}
    return {**result, "check_pass": True}
"""
if source.count(collision_seam) != 1:
    raise RuntimeError("The original collision observation seam changed")
# __file__ deliberately remains this wrapper: its digest pins the complete
# source transformation and the retained observer's required SHA256.
exec(compile(source.replace(collision_seam, collision_replacement), str(RETAINED), "exec"), globals())
