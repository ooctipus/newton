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
    csr_flag = os.environ.get("NEWTON_HEIGHTFIELD_PAIR_CSR")
    if csr_flag not in ("0", "1"):
        raise RuntimeError("Require explicit pair-CSR flag on both arms")
    csr_wanted = csr_flag == "1"
    csr_actual = getattr(narrow, "_heightfield_pair_csr", False)
    if type(csr_actual) is not bool or csr_actual != csr_wanted:
        raise RuntimeError("Requested pair-CSR differs from the bound owner")
    csr_owner = getattr(narrow, "_pair_csr", None)
    if (csr_owner is not None) != csr_wanted:
        raise RuntimeError("Pair-CSR owner existence disagrees with actual dispatch")
    shell_flag = os.environ.get("NEWTON_HEIGHTFIELD_PAIR_CSR_SHELL")
    if shell_flag not in ("0", "1"):
        raise RuntimeError("Require explicit pair-CSR shell flag on both arms")
    shell_wanted = shell_flag == "1"
    shell_actual = getattr(narrow, "_heightfield_pair_csr_shell", False)
    if (type(shell_actual) is not bool or shell_actual != shell_wanted
            or (shell_wanted and not csr_wanted)):
        raise RuntimeError("Requested shell query differs from actual pair-CSR dispatch")
    if csr_wanted and getattr(csr_owner, "shell_support", False) is not shell_wanted:
        raise RuntimeError("Pair-CSR shell owner marker differs from actual dispatch")
    flag = os.environ.get("NEWTON_HEIGHTFIELD_ADAPTIVE_MANIFOLD")
    if flag not in ("0", "1"):
        raise RuntimeError("Require explicit adaptive manifold flag on both arms")
    wanted = flag == "1"
    if wanted and csr_wanted:
        raise RuntimeError("The old-survivor and raw-pool experiments must not be combined")
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
    observed = csr_owner.fallback_export if csr_wanted else narrow.export_reduced_contacts_kernel
    if type(actual) is not bool or actual != wanted or observed.key != expected.key:
        raise RuntimeError("Requested adaptive exporter differs from actual owner")
    if wanted and observed is not expected:
        raise RuntimeError("Adaptive exporter is not the exact cached factory")
    result["adaptive_manifold"] = {"requested": wanted, "observed": actual,
        "kernel_key": observed.key, "check_pass": True}
    csr_result = {"requested": csr_wanted, "observed": csr_actual,
        "shell_requested": shell_wanted, "shell_observed": shell_actual, "check_pass": True}
    if csr_wanted:
        import warp as wp
        from newton._src.geometry.heightfield_pair_csr import create_export_kernel, get_query_kernels
        expected_export = create_export_kernel(write_contact)
        # The preserved measured CSR tree predates the explicit shell argument.
        # Current trees must use the explicit Boolean factory identity.
        expected_queries = (get_query_kernels(shell_wanted)
            if hasattr(narrow, "_heightfield_pair_csr_shell") else get_query_kernels())
        actual_queries = (csr_owner.midphase, csr_owner.finite, csr_owner.generic)
        if (csr_owner.export_kernel is not expected_export
                or narrow.export_reduced_contacts_kernel is not expected_export
                or any(a is not b for a, b in zip(actual_queries, expected_queries))):
            raise RuntimeError("Pair-CSR does not own the exact production factories")
        raw_capacity = narrow.global_contact_reducer.capacity
        pair_capacity = narrow.max_candidate_pairs
        triangle_capacity = narrow.max_triangle_pairs
        sizes = {"triangle_pair": triangle_capacity, "raw_pair": raw_capacity + 1,
                 "ids": raw_capacity, "counts": pair_capacity + 1,
                 "offsets": pair_capacity + 1, "cursors": pair_capacity, "status": 1}
        for name, size in sizes.items():
            array = getattr(csr_owner, name)
            if (array.shape != (size,) or array.dtype is not wp.int32
                    or array.device != wp.get_device(narrow.device)
                    or not array.is_contiguous or array is not getattr(csr_owner.data, name)):
                raise RuntimeError("Unexpected pair-CSR array ownership/layout: " + name)
        separation = csr_owner.separation
        if (separation.shape != (raw_capacity + 1,) or separation.dtype is not wp.vec2
                or separation.device != wp.get_device(narrow.device)
                or not separation.is_contiguous or separation is not csr_owner.data.separation):
            raise RuntimeError("Unexpected pair-CSR separation interval ownership/layout")
        status = csr_owner.status.numpy().tolist()
        if status != [0]:
            raise RuntimeError("Pair-CSR has a sticky routing/capacity failure: " + str(status))
        csr_result.update(kernel_key=expected_export.key,
            query_keys=[kernel.key for kernel in actual_queries], fallback_export_key=observed.key,
            sizes=sizes, separation_size=raw_capacity + 1,
            logical_extra_bytes=4*sum(sizes.values())+8*(raw_capacity+1), status=status,
            scope="Actual factories, integer/interval-buffer ownership and sticky status; not manifold quality")
    result["pair_csr"] = csr_result
    return {**result, "check_pass": True}
"""
if source.count(collision_seam) != 1:
    raise RuntimeError("The original collision observation seam changed")
status_seam = "    return module\n"
status_replacement = """    csr_flag = os.environ.get("NEWTON_HEIGHTFIELD_PAIR_CSR")
    if csr_flag not in ("0", "1"):
        raise RuntimeError("Require explicit pair-CSR status contract on both arms")
    if csr_flag == "1":
        module.NARROW_FLAGS = module.NARROW_FLAGS | {"heightfield_pair_csr_raw_or_membership"}
    return module
"""
if source.count(status_seam) != 1:
    raise RuntimeError("The original capacity-contract loading seam changed")
source = source.replace(status_seam, status_replacement)
source = source.replace(collision_seam, collision_replacement)
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
