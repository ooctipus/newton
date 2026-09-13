# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run complete terrain-query physical controls under the unchanged paired owner."""

import argparse
import hashlib
import json
import os
import unittest
from pathlib import Path


def main():
    """Require selected sources and exactly one leased GPU; make no timing claim."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("newton", "isaaclab", "audit-output", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    root = args.newton.resolve()
    paths = [
        root / "newton/_src/geometry/heightfield_direct.py",
        root / "newton/_src/geometry/heightfield_cells.py",
        root / "newton/_src/geometry/narrow_phase.py",
        root / "newton/tests/test_heightfield_direct.py",
        root / "tools/fpgs_bench/heightfield_direct_current.py",
        Path(__file__).resolve(),
    ]
    pins = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    report = {"success": False, "performance_accepted": False, "scope": __doc__, "pins": pins}
    try:
        import heightfield_direct_current as current  # noqa: PLC0415
        import warp as wp  # noqa: PLC0415

        import newton  # noqa: PLC0415
        from newton.tests.test_heightfield_direct import TestHeightfieldDirect  # noqa: PLC0415

        if (
            Path(newton.__file__).resolve() != root / "newton/__init__.py"
            or Path(__file__).resolve().parents[2] != root
        ):
            raise RuntimeError("Wrong selected Newton source")
        device = wp.get_device("cuda:0")
        if len(wp.get_cuda_devices()) != 1 or device.uuid != os.environ["CUDA_VISIBLE_DEVICES"]:
            raise RuntimeError("Wrong GPU lease")
        report["gpu"] = {"uuid": device.uuid, "name": device.name}
        TestHeightfieldDirect.device = "cuda:0"
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestHeightfieldDirect)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        report.update(
            tests_run=result.testsRun,
            skipped=len(result.skipped),
            failures=len(result.failures),
            errors=len(result.errors),
        )
        if not result.wasSuccessful() or result.skipped or result.testsRun != 6:
            raise RuntimeError("Direct terrain CUDA controls failed or skipped")
        report["current_geometry"] = [current.check(gpu, "cuda:0") for gpu in (0, 1)]
        report["success"] = True
    finally:
        report["final_source_guard_pass"] = pins == {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
        }
        report["success"] &= report["final_source_guard_pass"]
        for path in (args.audit_output, args.output):
            with path.open("x") as stream:
                json.dump(report, stream, indent=2, allow_nan=False)
                stream.write("\n")
        if not report["final_source_guard_pass"]:
            raise RuntimeError("Changed physical-control source")


if __name__ == "__main__":
    main()
