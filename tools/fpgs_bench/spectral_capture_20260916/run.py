# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reuse the unchanged fixed-backend paired variant protocol."""

import hashlib
import importlib.util
import json
from pathlib import Path

ADAPTER = Path("/tmp/fpgs-kinetic-fixed-variants-TZRkIPYw/run.py")
PIN = "c062388f23f82c524690f272fb2d14141ed4126db641de6dcd86f8ade5bfe416"
HERE = Path(__file__).resolve().parent


def main():
    if hashlib.sha256(ADAPTER.read_bytes()).hexdigest() != PIN:
        raise RuntimeError("Original fixed-backend adapter changed")
    spec = importlib.util.spec_from_file_location("spectral_fixed_adapter", ADAPTER)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    adapter.EXTRA.extend(HERE / name for name in ("run.py", "checked_capture.py", "nsys_checked.sh"))
    old_install = adapter.install

    def install(owner):
        old_install(owner)
        old_batch, old_check = owner.make_batch, owner.check_overflow_result

        def make_batch(*args, **kwargs):
            batch = old_batch(*args, **kwargs)
            for run in batch:
                if run["command"][:2] != ["bash", str(adapter.TOOLS / "nsys_checked.sh")]:
                    raise RuntimeError("Unexpected original capture command")
                if run["environment"].get("FEATHER_PGS_SPECTRAL_RESIDUAL") not in ("0", "1"):
                    raise RuntimeError("Require explicit spectral flags on both arms")
                if run["environment"].get("FEATHER_PGS_SPECTRAL_JACOBI") != "0":
                    raise RuntimeError("The previous spectral Jacobi experiment must stay off")
                if run["environment"].get("FEATHER_PGS_SPECTRAL_GS") != "0":
                    raise RuntimeError("The closed ordered spectral experiment must stay off")
                run["command"][1] = str(HERE / "nsys_checked.sh")
            return batch

        def check(run, drivers):
            old_check(run, drivers)
            if any(str(HERE / name) not in drivers for name in ("run.py", "checked_capture.py", "nsys_checked.sh")):
                raise RuntimeError("Spectral observer files missing from source guard")
            report = json.loads((Path(run["output_dir"]) / "capture_checks.json").read_text())
            wanted = run["environment"]["FEATHER_PGS_SPECTRAL_RESIDUAL"] == "1"
            for boundary in report["boundaries"]:
                item = boundary.get("spectral_residual", {})
                if (
                    item.get("check_pass") is not True
                    or item.get("requested") is not wanted
                    or item.get("observed") is not wanted
                ):
                    raise RuntimeError("Missing or inconsistent spectral owner observation")

        owner.make_batch, owner.check_overflow_result = make_batch, check

    adapter.install = install
    return adapter.main()


if __name__ == "__main__":
    raise SystemExit(main())
