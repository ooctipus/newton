# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reuse fixed-backend sampling on one available GPU or a paired selection."""

import hashlib
import importlib.util
import json
import signal
import sys
from pathlib import Path
from types import ModuleType

ADAPTER = Path("/tmp/fpgs-kinetic-fixed-variants-TZRkIPYw/run.py")
PIN = "c062388f23f82c524690f272fb2d14141ed4126db641de6dcd86f8ade5bfe416"
HERE = Path(__file__).resolve().parent
VARIANTS_PIN = "48406c079588d3088fc25c1cdce2bc2659aaec28539013e5811c9c55378445b5"


def load_variants(adapter):
    """Allow an idle single GPU without changing either arm's capture protocol."""
    path = adapter.TOOLS / "compare_variants.py"
    source = path.read_text()
    if hashlib.sha256(source.encode()).hexdigest() != VARIANTS_PIN:
        raise RuntimeError("Original paired variant owner changed")
    old = "len(args.gpus) != 2 or len(set(args.gpus)) != 2"
    if source.count(old) != 1:
        raise RuntimeError("Unexpected original GPU selection guard")
    source = source.replace(old, "len(args.gpus) not in (1, 2) or len(set(args.gpus)) != len(args.gpus)")
    source = source.replace(
        "Select exactly two distinct nonnegative GPU indices", "Select one or two distinct nonnegative GPU indices"
    )
    sys.path.insert(0, str(adapter.TOOLS))
    variants = ModuleType("selected_krylov_variants")
    variants.__file__ = str(path)
    exec(compile(source, str(path), "exec"), variants.__dict__)
    if Path(variants.owner.__file__).resolve() != adapter.TOOLS / "compare_backends.py":
        raise RuntimeError("Wrong paired owner")
    return variants


def main():
    if hashlib.sha256(ADAPTER.read_bytes()).hexdigest() != PIN:
        raise RuntimeError("Original fixed-backend adapter changed")
    spec = importlib.util.spec_from_file_location("krylov_fixed_adapter", ADAPTER)
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
                if run["environment"].get("FEATHER_PGS_KRYLOV_CHORD") not in ("0", "1"):
                    raise RuntimeError("Require explicit Krylov/chord flags on both arms")
                if run["environment"].get("FEATHER_PGS_SPECTRAL_RESIDUAL") != "0":
                    raise RuntimeError("The closed spectral residual experiment must stay off")
                if run["environment"].get("FEATHER_PGS_SPECTRAL_JACOBI") != "0":
                    raise RuntimeError("The previous spectral Jacobi experiment must stay off")
                if run["environment"].get("FEATHER_PGS_SPECTRAL_GS") != "0":
                    raise RuntimeError("The closed ordered spectral experiment must stay off")
                run["command"][1] = str(HERE / "nsys_checked.sh")
            return batch

        def check(run, drivers):
            old_check(run, drivers)
            if any(str(HERE / name) not in drivers for name in ("run.py", "checked_capture.py", "nsys_checked.sh")):
                raise RuntimeError("Krylov/chord observer files missing from source guard")
            report = json.loads((Path(run["output_dir"]) / "capture_checks.json").read_text())
            wanted = run["environment"]["FEATHER_PGS_KRYLOV_CHORD"] == "1"
            for boundary in report["boundaries"]:
                item = boundary.get("krylov_chord", {})
                if (
                    item.get("check_pass") is not True
                    or item.get("requested") is not wanted
                    or item.get("observed") is not wanted
                ):
                    raise RuntimeError("Missing or inconsistent Krylov/chord owner observation")

        owner.make_batch, owner.check_overflow_result = make_batch, check

    adapter.install = install
    variants = load_variants(adapter)
    adapter.install(variants.owner)
    signal.signal(signal.SIGTERM, variants.owner.interrupted)
    return variants.main()


if __name__ == "__main__":
    raise SystemExit(main())
