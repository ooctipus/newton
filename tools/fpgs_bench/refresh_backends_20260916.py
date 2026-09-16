# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Refresh the three stale backend comparisons with symmetric collision mapping.

Reuse the existing fixed-import, checked, alternating paired-GPU owner. This
adapter changes only the shared narrow-phase worker mapping; it does not change
task budgets, capacities, capture accounting or the MJWarp numerical correction.
The local adapter dependency is part of the retained handoff artifacts.
"""

import hashlib
import runpy
import signal
from pathlib import Path

ADAPTER = Path("/tmp/fpgs-allegro-fixed-backends-EQirljZU/run.py")
ADAPTER_SHA256 = "514137a2b336d1668f501c39488669da768964bb1b30d9386ff41fc45af43427"
SHARED = {"NEWTON_NARROW_PHASE_THREADS_X": "4"}


def main():
    """Apply the same shared collision configuration to both physics backends."""
    if hashlib.sha256(ADAPTER.read_bytes()).hexdigest() != ADAPTER_SHA256:
        raise RuntimeError("The retained fixed-import adapter has changed")
    owner = runpy.run_path(str(ADAPTER), run_name="fixed_backend_adapter")["load_owner"]()
    old_batch, old_hashes = owner.make_batch, owner.file_hashes

    def make_batch(*args, **kwargs):
        batch = old_batch(*args, **kwargs)
        for run in batch:
            if run["task"] not in {"franka", "kuka", "anymald"}:
                raise ValueError("This refresh adapter covers only Franka, Kuka and ANYmal-D")
            run["environment"].update(SHARED)
            run["shared_collision_mapping"] = dict(SHARED)
        return batch

    def file_hashes(paths):
        return old_hashes(list(dict.fromkeys([*paths, Path(__file__).resolve()])))

    owner.make_batch, owner.file_hashes = make_batch, file_hashes
    signal.signal(signal.SIGTERM, owner.interrupted)
    return owner.main()


if __name__ == "__main__":
    raise SystemExit(main())
