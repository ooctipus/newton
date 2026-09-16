# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Extend the original untimed observer with exact replacement-owner checks."""

import hashlib
import os
from pathlib import Path

import numpy as np


def supernodal_snapshot(solver, geometric):
    """Observe the exact refresh replacement without changing its timed path."""
    flag = os.environ.get("FEATHER_PGS_SPARSE_SUPERNODAL", "0")
    if flag not in ("0", "1"):
        raise RuntimeError("Require Boolean supernodal selection")
    requested = flag == "1"
    sparse = solver._sparse_factor
    observed = getattr(sparse, "supernodal", False)
    if type(observed) is not bool or observed != requested:
        raise RuntimeError("Requested supernodal factor differs from the actual owner")
    result = {"requested": requested, "observed": observed, "check_pass": False}
    if requested:
        from newton._src.solvers.feather_pgs import sparse_supernodal  # noqa: PLC0415

        kernel = (
            sparse_supernodal.get_refresh_kernel(geometric=True)
            if geometric
            else sparse_supernodal.get_refresh_kernel()
        )
        key = ("g1_kinetic_" if geometric else "") + "sparse_supernodal43_434"
        if sparse.kernels.refresh is not kernel or kernel.key != key:
            raise RuntimeError("Supernodal factor did not select the exact refresh factory")
        result["refresh_key"] = key
    return {**result, "check_pass": True}


def parallel_snapshot(solver):
    """Require actual device use, not merely an installed experimental owner."""
    flag = os.environ.get("FEATHER_PGS_PARALLEL_WORLD", "0")
    split_flag = os.environ.get("FEATHER_PGS_PARALLEL_WORLD_SPLIT", "0")
    if flag not in ("0", "1") or split_flag not in ("0", "1"):
        raise RuntimeError("Require Boolean parallel-world selection")
    requested = flag == "1"
    split = split_flag == "1"
    if split and not requested:
        raise RuntimeError("Split production requires the parallel-world owner")
    sparse = solver._sparse_factor
    owner = getattr(sparse, "parallel_world", None)
    observed = owner is not None
    result = {"requested": requested, "observed": observed, "split": split, "check_pass": False}
    if observed != requested:
        raise RuntimeError("Requested parallel world differs from the installed owner")
    if not requested:
        return {**result, "check_pass": True}
    from newton._src.solvers.feather_pgs import parallel_world_rows  # noqa: PLC0415

    if getattr(owner, "split", False) is not split:
        raise RuntimeError("Requested split boundary differs from the actual owner")
    factory = parallel_world_rows.get_producer_kernel if split else parallel_world_rows.get_kernel
    kernel = factory(solver._g1_kinetic_state.chain_scan)
    expected_key = "sparse_parallel_world_rows43_s18_c100" if split else "sparse_parallel_world43_s18_c100"
    if owner.active is not True or owner.kernel is not kernel or kernel.key != expected_key:
        raise RuntimeError("Parallel owner fell back or selected an unexpected kernel")
    solve_key = kernel.key
    if split:
        if (
            owner.solve_kernel is not sparse.kernels.solve
            or owner.solve_kernel.key != "sparse_metric_tangent43_s18_c100"
        ):
            raise RuntimeError("Split production did not retain the original metric solver")
        solve_key = owner.solve_kernel.key
    if tuple(owner.status.shape) != (2,):
        raise RuntimeError("Unexpected parallel-world status shape")
    status = owner.status.numpy()
    if int(status[0]) != 0 or int(status[1]) <= 0:
        raise RuntimeError("Parallel-world device guard failed or no successful solve ran")
    buckets = owner.buckets
    worlds, capacity = int(solver.world_count), int(solver._max_contacts_alloc)
    if (
        tuple(buckets.data.ids.shape) != (capacity,)
        or tuple(buckets.data.offsets.shape) != (worlds + 1,)
        or tuple(buckets.counts.shape) != (worlds + 1,)
        or tuple(buckets.data.invalid.shape) != (1,)
    ):
        raise RuntimeError("Parallel-world routing changed the exact allocated capacity")
    if int(buckets.data.invalid.numpy()[0]) != 0:
        raise RuntimeError("Parallel-world raw routing failed coverage validation")
    offsets = buckets.data.offsets.numpy()
    if int(offsets[0]) != 0 or np.any(offsets[1:] < offsets[:-1]) or int(offsets[-1]) > capacity:
        raise RuntimeError("Invalid parallel-world raw bucket offsets")
    return {
        **result,
        "check_pass": True,
        "active": True,
        "kernel_key": kernel.key,
        "solve_kernel_key": solve_key,
        "successful_solve_launches": int(status[1]),
        "status_nonzero": 0,
        "raw_capacity": capacity,
        "routed_raw_count": int(offsets[-1]),
        "retained_fallback_arrays": (
            "Existing Z/support/incident are published and consumed by the original solver"
            if split
            else "Z/support/incident remain allocated; candidate uses private response scratch"
        ),
    }


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
    compiled_flag = os.environ.get("FEATHER_PGS_COMPILED_COORDINATE_STATE", "0")
    if compiled_flag not in ("0", "1"):
        raise RuntimeError("Require Boolean compiled-coordinate selection")
    compiled = compiled_flag == "1"
    if getattr(owner, "compiled_coordinate_state", False) is not compiled:
        raise RuntimeError("Requested compiled coordinates differ from the actual owner")
    if compiled and getattr(owner, "coordinate_plan", None) is None:
        raise RuntimeError("Compiled coordinates are missing their model descriptor")
    result["compiled_coordinate_state"] = {"requested": compiled, "observed": compiled, "check_pass": True}
    state_suffix = suffix + ("_compiled" if compiled else "")
    if keys != {"repair": "g1_kinetic_repair44" + state_suffix, "finish": "g1_kinetic_finish44" + state_suffix, "predictor": "g1_kinetic_predict43" + suffix}:
"""
if source.count(seam) != 1:
    raise RuntimeError("The original exact-key observation seam changed")
source = source.replace(seam, replacement)
parallel_seam = '    result["solve_key"] = getattr(sparse.kernels.solve, "key", None)\n'
parallel_replacement = (
    parallel_seam
    + """    result["parallel_world"] = parallel_snapshot(solver)
    result["executed_solve_key"] = (
        result["parallel_world"]["solve_kernel_key"]
        if result["parallel_world"]["observed"] else result["solve_key"]
    )
"""
)
if source.count(parallel_seam) != 1:
    raise RuntimeError("The original solve-owner observation seam changed")
source = source.replace(parallel_seam, parallel_replacement)
refresh_seam = '    result["refresh_key"] = sparse.kernels.refresh.key\n'
refresh_replacement = (
    '    result["supernodal"] = supernodal_snapshot(solver, requested)\n'
    '    if result["supernodal"]["observed"]:\n'
    '        refresh = result["supernodal"]["refresh_key"]\n' + refresh_seam
)
if source.count(refresh_seam) != 1:
    raise RuntimeError("The original refresh-owner observation seam changed")
source = source.replace(refresh_seam, refresh_replacement)
# __file__ deliberately remains this wrapper: its digest pins the complete
# source transformation and the retained observer's required SHA256.
exec(compile(source, str(RETAINED), "exec"), globals())
