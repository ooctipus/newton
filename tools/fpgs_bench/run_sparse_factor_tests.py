# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run the focused actual-tree sparse boundary tests under the paired GPU owner."""

import argparse
import hashlib
import json
import os
import unittest
from pathlib import Path


def main():
    """Require actual CUDA execution, exact runtime import and unchanged sources."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("newton", "isaaclab", "output", "audit-output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    files = [
        args.newton / "newton/_src/solvers/feather_pgs" / name
        for name in ("solver_feather_pgs.py", "sparse_factor.py", "sparse_factor_rows.py")
    ]
    files += [Path(__file__).resolve(), Path(__file__).with_name("test_sparse_factor.py")]
    pins = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    report = {"success": False, "performance_accepted": False, "scope": __doc__, "pins": pins}
    try:
        import warp as wp  # noqa: PLC0415

        import newton  # noqa: PLC0415

        if Path(newton.__file__).resolve() != args.newton.resolve() / "newton/__init__.py":
            raise RuntimeError("Wrong Newton import")
        device = wp.get_device("cuda:0")
        if len(wp.get_cuda_devices()) != 1 or device.uuid != os.environ["CUDA_VISIBLE_DEVICES"]:
            raise RuntimeError("Wrong GPU lease")
        report["gpu"] = {"uuid": device.uuid, "name": device.name}
        suite = unittest.defaultTestLoader.loadTestsFromName("tools.fpgs_bench.test_sparse_factor.TestSparseFactorCUDA")
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        report.update(
            tests_run=result.testsRun,
            skipped=len(result.skipped),
            failures=len(result.failures),
            errors=len(result.errors),
            success=result.wasSuccessful() and not result.skipped and result.testsRun == 2,
        )
        if not report["success"]:
            raise RuntimeError("Sparse physical controls failed")
    finally:
        report["source_guard"] = all(
            hashlib.sha256(Path(path).read_bytes()).hexdigest() == pin for path, pin in pins.items()
        )
        report["success"] &= report["source_guard"]
        for path in (args.audit_output, args.output):
            with path.open("x") as stream:
                json.dump(report, stream, indent=2, allow_nan=False)
        if not report["source_guard"]:
            raise RuntimeError("Sparse sources changed")


if __name__ == "__main__":
    main()
