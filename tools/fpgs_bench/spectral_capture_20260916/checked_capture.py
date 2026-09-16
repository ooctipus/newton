# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Append actual spectral owner identity at the original untimed boundaries."""

import hashlib
import os
import sys
from pathlib import Path
from types import ModuleType

ORIGINAL = Path("/home/octi/Projects/newton-fpgs-structural-bench-20260913/tools/fpgs_bench/checked_capture.py")
PIN = "42b289bd0082d194180d5981651a108d11a66058ff4e67649ad7ddb408c9bea6"


def snapshot(solver):
    """Observe factory/configuration, not per-world convergence or admission."""
    value = os.environ.get("FEATHER_PGS_SPECTRAL_GS")
    if value not in ("0", "1"):
        raise RuntimeError("Require explicit spectral GS on both arms")
    wanted = value == "1"
    kernels = solver._pgs_solve_mf_gs_incremental_kernels
    if len(kernels) != 2 or solver._par_tiers != [(32, 0), (48, 32)]:
        raise RuntimeError("Unexpected original ANY18 tier ownership")
    for kernel in kernels:
        if bool(getattr(kernel, "_fpgs_spectral_contact", False)) != wanted:
            raise RuntimeError("Requested spectral factory differs from actual owner")
        if kernel.key.endswith("_sgs24") != wanted:
            raise RuntimeError("Unexpected spectral kernel suffix")
    fields = {
        "max_world_dofs": 18,
        "mf_gs_parallel_sweeps": 24,
        "mf_gs_parallel_matrix_free": True,
        "mf_gs_parallel_nesterov": True,
        "pgs_warmstart": False,
        "_mf_warmstart_enabled": False,
    }
    actual = {key: getattr(solver, key) for key in fields}
    if actual != fields or tuple(solver._ink_sizes) != (18, 0, 0, 0):
        raise RuntimeError("Spectral owner changed original solver settings")
    return {
        "requested": wanted,
        "observed": wanted,
        "check_pass": True,
        "keys": [kernel.key for kernel in kernels],
        "settings": actual,
        "native_source_sha256": [
            hashlib.sha256(kernel._fpgs_spectral_contact_native.encode()).hexdigest() for kernel in kernels
        ]
        if wanted
        else [],
        "scope": "Actual factory and configuration; not dynamic row admission or physical convergence proof",
    }


def main():
    source = ORIGINAL.read_bytes()
    if hashlib.sha256(source).hexdigest() != PIN:
        raise RuntimeError("Original checked observer changed")
    original = ModuleType("_spectral_original_checked")
    original.__file__ = str(ORIGINAL)
    sys.modules[original.__name__] = original
    exec(compile(source, str(ORIGINAL), "exec"), original.__dict__)
    install = original.install_boundary_check

    def combined(harness, report, save, get_manager, compat=None, compact=None):
        install(harness, report, save, get_manager, compat, compact)
        checked = harness._model_meta

        def observed(physics):
            metadata = checked(physics)
            entry = report["boundaries"][-1]
            try:
                if physics != "feather_pgs":
                    raise RuntimeError("This owner screen requires FPGS on both arms")
                entry["spectral_gs"] = snapshot(get_manager()._solver)
                return metadata
            except BaseException as error:
                entry.update(check_pass=False, error=repr(error))
                raise
            finally:
                save()

        harness._model_meta = observed

    original.install_boundary_check = combined
    return original.main()


if __name__ == "__main__":
    raise SystemExit(main())
