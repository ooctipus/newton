# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Declared ``FEATHER_PGS_*`` environment variables of SolverFeatherPGS.

Each entry names the modules that read it; ``newton/tests/test_environment_flags.py``
keeps the declarations equal to the read sites. The solver records its environment
once at construction; a name nothing reads, a removed switch among them, is
undeclared and recorded as unknown, not rejected.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from ...core.environment_flags import EnvironmentFlag, FlagKind, declare, record_environment

PREFIXES = ("FEATHER_PGS_", "IL_NEWTON_FPGS_")

_PACKAGE = "solvers.feather_pgs."
_SOLVER = ("solver_feather_pgs",)


def _flag(name: str, kind: FlagKind, modules: tuple[str, ...], meaning: str) -> EnvironmentFlag:
    return EnvironmentFlag(name, kind, tuple(_PACKAGE + module for module in modules), meaning)


# The flags the solver reads: (name, kind, reading modules, meaning).
_READ = (
    (
        "FEATHER_PGS_DEBUG_SNAPSHOTS",
        "diagnostic",
        _SOLVER,
        "=1 copies the stage-3 joint velocity, acceleration and predicted velocity every substep for NaN dumps.",
    ),
    (
        "FEATHER_PGS_PATCH_FRAME_REUSE",
        "policy",
        ("rows",),
        "=0 sorts the patch pairs of every build, small frames over their whole capacity, instead of keeping them on captured builds that reuse the previous captured build's contact frame (default on).",
    ),
    (
        "FEATHER_PGS_STAGE_CLOCKS",
        "diagnostic",
        ("tree_kinetic_owner",),
        "=1 compiles per-stage cycle counters into the held-tree kernels.",
    ),
    ("FEATHER_PGS_SYNC_TIMINGS", "diagnostic", _SOLVER, "1, true, yes or on prints synchronized per-stage timings."),
    ("FEATHER_PGS_SYNC_TIMINGS_COUNT", "diagnostic", _SOLVER, "Number of steps with synchronized timings (default 1)."),
    ("FEATHER_PGS_SYNC_TIMINGS_START", "diagnostic", _SOLVER, "First step with synchronized timings (default 20)."),
)

FLAGS = declare(*sorted((_flag(*flag) for flag in _READ), key=lambda flag: flag.name))


def record(environ: Mapping[str, str] | None = None) -> dict:
    """Record the solver's environment once at construction."""
    return record_environment(FLAGS, PREFIXES, os.environ if environ is None else environ)
