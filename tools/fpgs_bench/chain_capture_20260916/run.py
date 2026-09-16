# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reuse the source-pinned G1 paired capture with the chain-owner observer."""

import hashlib
import importlib.util
from pathlib import Path

ORIGINAL = Path("/tmp/fpgs-g1-paired-gs-checked-t0NFbw6H/run_live_sparse_checked.py")
PIN = "23acfade1339b660e03d92c273d92b767081751c7feb2a53eea4d52c2cd5aef8"


def main():
    """Keep all original source, capacity, timing and cleanup checks."""
    if hashlib.sha256(ORIGINAL.read_bytes()).hexdigest() != PIN:
        raise RuntimeError("The retained G1 paired owner has changed")
    spec = importlib.util.spec_from_file_location("retained_g1_paired_capture", ORIGINAL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.FLAGS.add("NEWTON_HEIGHTFIELD_SHELL_SUPPORT")
    module.__file__ = str(Path(__file__).resolve())
    module.adapter.EXTRA.extend(
        Path(__file__).with_name(name).resolve() for name in ("run.py", "checked_sparse.py", "nsys_sparse_checked.sh")
    )
    return module.main()


if __name__ == "__main__":
    raise SystemExit(main())
