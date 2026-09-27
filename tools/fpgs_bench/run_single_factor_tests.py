# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run the focused physical representation tests under the paired GPU owner."""

import argparse
import json
import os
import unittest
from pathlib import Path


def main():
    """Require actual CUDA tests on the selected source and exact leased GPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("newton", "isaaclab", "audit-output", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    report = {"success": False, "performance_accepted": False, "scope": __doc__}
    try:
        import warp as wp  # noqa: PLC0415

        import newton  # noqa: PLC0415

        if Path(newton.__file__).resolve() != args.newton.resolve() / "newton/__init__.py":
            raise RuntimeError("Wrong Newton import")
        device = wp.get_device("cuda:0")
        if len(wp.get_cuda_devices()) != 1 or device.uuid != os.environ["CUDA_VISIBLE_DEVICES"]:
            raise RuntimeError("Wrong GPU lease")
        report["gpu"] = {"uuid": device.uuid, "name": device.name}
        suite = unittest.defaultTestLoader.loadTestsFromName("newton.tests.test_feather_pgs_single_factor")
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        report.update(
            tests_run=result.testsRun,
            skipped=len(result.skipped),
            failures=len(result.failures),
            errors=len(result.errors),
            success=result.wasSuccessful() and not result.skipped and result.testsRun == 4,
        )
        if not report["success"]:
            raise RuntimeError("Focused CUDA representation tests failed")
    finally:
        for path in (args.audit_output, args.output):
            with path.open("x") as stream:
                json.dump(report, stream, indent=2, allow_nan=False)
                stream.write("\n")


if __name__ == "__main__":
    main()
