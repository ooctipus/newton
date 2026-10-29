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
_CACHE = ("held_tree",)


def _flag(name: str, kind: FlagKind, modules: tuple[str, ...], meaning: str) -> EnvironmentFlag:
    return EnvironmentFlag(name, kind, tuple(_PACKAGE + module for module in modules), meaning)


# The flags the solver reads: (name, kind, reading modules, meaning).
_READ = (
    ("FEATHER_PGS_CONTACT_THREADS_X", "policy", _SOLVER, "Multiplier of the contact-build thread budget (default 1)."),
    (
        "FEATHER_PGS_CONTACT_UPDATE",
        "policy",
        _SOLVER,
        "sequential or unit overrides the contact_update argument (unset: the argument).",
    ),
    (
        "FEATHER_PGS_DEBUG_SNAPSHOTS",
        "diagnostic",
        _SOLVER,
        "=1 copies the stage-3 joint velocity, acceleration and predicted velocity every substep for NaN dumps.",
    ),
    (
        "FEATHER_PGS_DENSE_ROW_BUDGETS",
        "policy",
        _SOLVER,
        "tiled_row row budgets: 1 on for 64 or 128 rows, unset automatic (128 rows, 1024+ worlds), else off.",
    ),
    (
        "FEATHER_PGS_FREE_BODY_IMPLICIT_GYRO",
        "experiment",
        _SOLVER,
        "=1 applies an implicit gyroscopic predictor correction to independent free bodies.",
    ),
    ("FEATHER_PGS_KEYED_ROWS", "policy", _SOLVER, "=0 disables the topology-generated keyed contact rows."),
    (
        "FEATHER_PGS_MFGS_STREAM_LAMBDA",
        "reference",
        _SOLVER,
        "1, true, yes or on forces streamed matrix-free impulses instead of the shared-memory choice.",
    ),
    (
        "FEATHER_PGS_MF_EXACT_ROWSUM",
        "experiment",
        _SOLVER,
        "=1 uses the exact A-based step scale in the matrix-free parallel sweep.",
    ),
    (
        "FEATHER_PGS_MIXED_TREE_KINETIC_STATE",
        "policy",
        _CACHE,
        "Values other than 1 disable the dense-route held-tree cache, and the compact diagonal only it evaluates, "
        "under cached_dynamics='auto'.",
    ),
    (
        "FEATHER_PGS_PAIRED_COMPONENTS",
        "policy",
        _SOLVER,
        "Values other than 1 disable independent components in the paired response.",
    ),
    (
        "FEATHER_PGS_PATCH_ACTIVE_ROWS",
        "policy",
        _SOLVER,
        "=0 keeps patch friction on the full ordered sparse sweep instead of the active-row scan (default on); with FEATHER_PGS_SPARSE_CONTACT_TRIPLETS=1 that sweep fuses anchor triplets, so the previous kernel needs both off.",
    ),
    (
        "FEATHER_PGS_PATCH_FRAME_REUSE",
        "policy",
        _SOLVER,
        "=0 sorts the patch pairs of every build, small frames over their whole capacity, instead of keeping them on captured builds that reuse the previous captured build's contact frame (default on).",
    ),
    (
        "FEATHER_PGS_PATCH_PARALLEL_ROWS",
        "policy",
        _SOLVER,
        "=0 keeps patch friction off the parallel tiers and keyed rows, on its serial sweeps (default on).",
    ),
    (
        "FEATHER_PGS_PATCH_PARALLEL_SWEEPS",
        "policy",
        _SOLVER,
        "Deprecated (warns): a positive count or auto overrides the deprecated mf_gs_parallel_linked_sweeps, the linked tiers' own budget; every tier follows one automatic budget rule.",
    ),
    (
        "FEATHER_PGS_PREDICTED_TWISTS",
        "policy",
        ("tree_kinetic_owner",),
        "Values other than 1 stop the tree predictor publishing body twists for the resolved-world classifier.",
    ),
    (
        "FEATHER_PGS_RESOLVED_WORLDS",
        "policy",
        ("resolved_worlds",),
        "Values other than 1 stop skipping worlds whose candidate rows all separate at the predictor.",
    ),
    ("FEATHER_PGS_ROWS_THREADS_X", "policy", _SOLVER, "Multiplier of the row-build thread budget (default 4)."),
    (
        "FEATHER_PGS_ROW_PACKETS",
        "policy",
        _SOLVER,
        "Values other than 1 disable in-place limit-prefix and contact packets for the local solve owners.",
    ),
    (
        "FEATHER_PGS_SCALAR_BRANCHES",
        "policy",
        _SOLVER,
        "=0 keeps a diagonal-mass group of world-fixed roots over one-coordinate leaves out of the held-tree owner's scalar branches, which run it and own the other articulations of its worlds (default on).",
    ),
    (
        "FEATHER_PGS_SPARSE_CONTACT_DIRECT",
        "experiment",
        _SOLVER,
        "=1 launches sparse diagonal contact responses over the full contact capacity.",
    ),
    (
        "FEATHER_PGS_SPARSE_CONTACT_TRIPLETS",
        "experiment",
        _SOLVER,
        "=1 fuses contact triplets in the sparse sweep: point friction's in the active-row scan, patch friction's anchor units in whichever sweep it runs.",
    ),
    (
        "FEATHER_PGS_SPARSE_LOCAL_FACTORS",
        "policy",
        _SOLVER,
        "=0 disables per-world local factor workspaces in the sparse sweep.",
    ),
    ("FEATHER_PGS_SPARSE_SMALL_WORLDS", "policy", _SOLVER, "=0 disables small-world lane packing in the sparse sweep."),
    (
        "FEATHER_PGS_SPARSE_SPLIT",
        "policy",
        _SOLVER,
        "Values other than 1 keep split mode off automatic sparse factors.",
    ),
    (
        "FEATHER_PGS_SPARSE_TRANSACTION",
        "policy",
        _SOLVER,
        "Transaction of fused sparse contact triplets: single (one load group and reduction, per-triplet records) or staged (normal, then tangents, metric cache); auto (default) is single for patch and staged for point friction.",
    ),
    (
        "FEATHER_PGS_SPARSE_VELOCITY_LIMITS",
        "policy",
        _SOLVER,
        "Values other than 1 keep velocity-limit models off automatic sparse factors.",
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
    (
        "FEATHER_PGS_TIER_BLOCKS",
        "experiment",
        _SOLVER,
        "Blocks that grid-stride the parallel sweep tiers over per-tier world lists (default 0, off).",
    ),
    (
        "FEATHER_PGS_TREE_FOREST",
        "policy",
        _CACHE,
        "Held-tree groups: 1 one group per world, other values per-size groups (default by layout).",
    ),
    (
        "FEATHER_PGS_TREE_KINETIC_STATE",
        "policy",
        _CACHE,
        "Values other than 1 disable the sparse-route tree cache, and the native sparse factors only it evaluates, "
        "under cached_dynamics='auto'.",
    ),
    (
        "FEATHER_PGS_TREE_REFERENCE",
        "reference",
        ("tree_kinetic_owner",),
        "=1 selects the serial held solve, level reduction and shared-memory scans in held-tree kernels.",
    ),
    (
        "FEATHER_PGS_TREE_REPAIR_CLASSIFY",
        "policy",
        ("tree_kinetic_owner",),
        "=0 keeps the one-pass held-tree repair launch (default 1 on CUDA: list the blocks to repair, then repair over the list on a capped grid; motion-velocity layouts always launch one pass).",
    ),
    (
        "FEATHER_PGS_TREE_REPAIR_SKIP",
        "policy",
        ("tree_kinetic_owner",),
        "=0 launches every captured held-tree repair (default 1: a repair after a finish of the same capture on the bank it published launches nothing).",
    ),
    (
        "FEATHER_PGS_VELOCITY_LIMIT_LAW",
        "policy",
        _SOLVER,
        "stateless or accumulated overrides the velocity_limit_law argument (unset: the argument).",
    ),
    (
        "IL_NEWTON_FPGS_MF_WARMSTART",
        "policy",
        _SOLVER,
        "Legacy alias: 1, true, yes or on enables contact warm start like mf_warmstart=True.",
    ),
    (
        "IL_NEWTON_FPGS_MF_WARMSTART_DECAY",
        "policy",
        _SOLVER,
        "Legacy alias: warm-start decay used with IL_NEWTON_FPGS_MF_WARMSTART (default mf_warmstart_decay).",
    ),
)

FLAGS = declare(*sorted((_flag(*flag) for flag in _READ), key=lambda flag: flag.name))


def record(environ: Mapping[str, str] | None = None) -> dict:
    """Record the solver's environment once at construction."""
    return record_environment(FLAGS, PREFIXES, os.environ if environ is None else environ)
