# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Declared ``FEATHER_PGS_*`` environment variables of SolverFeatherPGS.

Each entry names the modules that read it; ``newton/tests/test_environment_flags.py``
keeps the declarations equal to the read sites. Retired selectors set to 1 are
rejected here for the solver, except those of :data:`IGNORED_RETIRED`, which
recipes still set: they select nothing and the record lists them as retired. The
solver records its environment once at construction, warns once per process about
each retired name present and records undeclared names instead of rejecting them.
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


# Retired selectors that raise when set to 1, in the order the solver has always checked them, each newly
# retired one last: when several are set, the first one names the error. Nothing reads them.
_RAISING_RETIRED = (
    ("FEATHER_PGS_SINGLE_FACTOR", "Retired single-factor owner; =1 raises at construction."),
    ("FEATHER_PGS_BODY_BASIS_ROWS", "Retired experiment; =1 raises."),
    ("FEATHER_PGS_SPARSE_PACKETS", "Retired experiment; =1 raises."),
    ("FEATHER_PGS_SPARSE_PAIRED_GS", "Retired experiment; =1 raises."),
    ("FEATHER_PGS_COUPLED_CONTACT", "Retired experiment; =1 raises."),
    ("FEATHER_PGS_COUPLED_JACOBI", "Retired experiment; =1 raises."),
    ("FEATHER_PGS_DENSE_ROW_REGISTERS", "Retired register-state copy of the 64-row tiled_row sweep; =1 raises."),
)

# Retired names that are ignored rather than rejected: those Isaac Lab's
# benchmark recipes still set, namely the packed sparse-factor route's
# selectors (480f0f44), the chain-scan and keyed-row aliases (d35949d6), the
# selectors of removed research owners and of the branch-Jacobi sweep
# (790f9782, aa2f332b), the in-sweep world rows' selector, which keyed rows
# declined wherever a recipe set it, the lane count of the removed grouped
# kinematics and torque kernels, and the prismatic opt-ins that scalar
# branches replaced; and the removed diagnostics and experiment that selected
# no route of their own: the sweep-input capture (FEATHER_PGS_CAPTURE_*), the
# row and in-kernel response comparisons (FEATHER_PGS_CHECK_ROWS,
# FEATHER_PGS_INK_CHECK) and the grouped free-body response
# (FEATHER_PGS_GROUPED_HINV_FREE). Nothing in Newton or Isaac Lab reads them,
# and recipes are not Newton's to edit, so the general route runs, the
# construction record lists them as retired and the solver warns once per
# process.
_IGNORED_RETIRED = {
    "FEATHER_PGS_ALLEGRO_KINETIC_ROWS": "Retired alias that turned keyed rows on; ignored (FEATHER_PGS_KEYED_ROWS=0 opts out of them).",
    "FEATHER_PGS_CAPTURE_COUNT": "Retired substep count of the removed sweep-input capture; ignored.",
    "FEATHER_PGS_CAPTURE_DIR": "Retired directory of the removed sweep-input capture; ignored.",
    "FEATHER_PGS_CAPTURE_KERNEL": "Retired capture of the matrix-free sweep's inputs for standalone replay; ignored.",
    "FEATHER_PGS_CAPTURE_PHASE": "Retired solve phase of the removed sweep-input capture; ignored.",
    "FEATHER_PGS_CAPTURE_STEP": "Retired first step of the removed sweep-input capture; ignored.",
    "FEATHER_PGS_CHECK_ROWS": "Retired comparison of the masked contact-row builder with the per-contact builder; ignored.",
    "FEATHER_PGS_FRANKA_COMPACT_WORKSPACE": "Retired compact workspace of a removed robot-specific owner; ignored.",
    "FEATHER_PGS_G1_CHAIN_SCAN": "Retired G1 alias of the held-tree chain scan; ignored.",
    "FEATHER_PGS_GROUPED_HINV_FREE": "Retired grouped response solve of free-body size groups, slower than the per-row kernel; ignored.",
    "FEATHER_PGS_GROUP_LANES": "Retired lanes per articulation of the removed grouped kinematics and torque kernels; ignored.",
    "FEATHER_PGS_INDEPENDENT_COMPONENTS": "Retired independent-components variant of a removed robot-specific owner; ignored.",
    "FEATHER_PGS_INK_CHECK": "Retired write-back of the in-kernel response and diagonal for comparison; ignored.",
    "FEATHER_PGS_KUKA_CURRENT_CONTACT": "Retired current-contact geometry of a removed robot-specific owner; ignored.",
    "FEATHER_PGS_KUKA_JOINT_WORLD": "Retired joint-world prediction of a removed robot-specific owner; ignored.",
    "FEATHER_PGS_LOCAL_ROW_PACKETS": "Retired row-packet producer of a removed robot-specific owner; ignored (see FEATHER_PGS_ROW_PACKETS).",
    "FEATHER_PGS_PACKED_TREE_KINETIC_STATE": "Retired tree cache of the packed sparse-factor route; ignored.",
    "FEATHER_PGS_PAIRED_GENERAL_OVERLAP": "Retired overlap of the paired and general solve streams; ignored.",
    "FEATHER_PGS_PRISMATIC_LINEAR_STATE": "Retired linear state of fixed-root one-body prismatic branches; ignored (scalar branches run them).",
    "FEATHER_PGS_PRISMATIC_PUBLICATION": "Retired body-parallel publication of fixed-root prismatic stars; ignored (scalar branches run them).",
    "FEATHER_PGS_SIMPLE_WORLD_ZERO": "Retired simple-world classifier of a removed robot-specific owner; ignored.",
    "FEATHER_PGS_SPARSE_BRANCH_SWEEPS": "Retired branch-Jacobi sparse sweep; ignored, sparse factors keep their ordered sweeps.",
    "FEATHER_PGS_SPARSE_CONTACT_BLOCK": "Retired contact-block rows of the packed sparse-factor route; ignored.",
    "FEATHER_PGS_SPARSE_FACTOR": "Retired packed sparse-factor route; ignored, the general route (sparse factors where admitted) runs.",
    "FEATHER_PGS_SPARSE_LEVEL_UPDATE": "Retired level-by-level factor update of the packed sparse-factor route; ignored.",
    "FEATHER_PGS_SPARSE_LIMIT_JACOBI": "Retired Jacobi joint-limit solve of the packed sparse-factor route; ignored.",
    "FEATHER_PGS_SPARSE_METRIC_TANGENTS": "Retired metric tangents of the packed sparse-factor route; ignored.",
    "FEATHER_PGS_SPARSE_PARALLEL_LIMITS": "Retired parallel joint-limit prefix of the packed sparse-factor route; ignored.",
    "FEATHER_PGS_SPARSE_SPECTRAL_TANGENTS": "Retired spectral tangents of the packed sparse-factor route; ignored.",
    "FEATHER_PGS_TREE_CHAIN_SCAN": "Retired held-tree chain-scan selector; ignored (the scans' schedules follow from the tree's data).",
    "FEATHER_PGS_TREE_LANE_PACK": "Retired held-tree lane width selector; ignored (the lanes follow from the tree's data).",
    "FEATHER_PGS_WORLD_ROWS": "Retired in-sweep world rows, which keyed rows always declined where recipes set it; ignored.",
    "FEATHER_PGS_WORLD_SCAN_PUBLICATION": "Retired world-scan publication of a removed robot-specific owner; ignored.",
}
IGNORED_RETIRED = frozenset(_IGNORED_RETIRED)

# The flags the solver reads: (name, kind, reading modules, meaning).
_READ = (
    (
        "FEATHER_PGS_COMPACT_CONTACT_BOUNDARY",
        "experiment",
        _SOLVER,
        "=1 selects the complete contact producer for the dense-six plus sparse solve.",
    ),
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
    ("FEATHER_PGS_FK_ID_CACHE", "policy", _SOLVER, "=0 disables the CUDA FK/inverse-dynamics cache."),
    (
        "FEATHER_PGS_FREE_BODY_IMPLICIT_GYRO",
        "experiment",
        _SOLVER,
        "=1 applies an implicit gyroscopic predictor correction to independent free bodies.",
    ),
    (
        "FEATHER_PGS_GROUPED_HINV",
        "policy",
        ("step_plan",),
        "=0 keeps the tiled response solve instead of the grouped one.",
    ),
    (
        "FEATHER_PGS_GROUPED_HINV_MAX",
        "policy",
        ("step_plan",),
        "Largest DOF size that uses the grouped response solve (default 32).",
    ),
    (
        "FEATHER_PGS_INK",
        "experiment",
        _SOLVER,
        "=1 lets the parallel sweep form the response H^-1 J^T and its diagonal in-kernel.",
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
        "Values other than 1 disable the dense-route held-tree cache under cached_dynamics='auto'.",
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
    ("FEATHER_PGS_ROWS_MASKED", "experiment", _SOLVER, "=1 builds contact rows with one thread per contact and DOF."),
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
        "FEATHER_PGS_SPARSE_GROUPED_DYNAMICS",
        "policy",
        _SOLVER,
        "Values other than 1 keep grouped-dynamics solvers on ordinary factors instead of automatic sparse ones.",
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
        "FEATHER_PGS_SPARSE_TRANSACTION_BLOCKS",
        "experiment",
        _SOLVER,
        "Minimum blocks per SM of the single-stage sparse transaction at 128 rows or fewer: 16 (default) or 14.",
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
        "Values other than 1 disable the sparse-route tree cache under cached_dynamics='auto'.",
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

# The raising retired selectors lead; the ignored ones sit in name order with the read flags.
FLAGS = declare(
    *(_flag(name, "retired", ("environment_flags",), meaning) for name, meaning in _RAISING_RETIRED),
    *sorted(
        (
            *(_flag(name, "retired", ("environment_flags",), meaning) for name, meaning in _IGNORED_RETIRED.items()),
            *(_flag(*flag) for flag in _READ),
        ),
        key=lambda flag: flag.name,
    ),
)


def reject_retired(environ: Mapping[str, str] | None = None) -> None:
    """Reject obsolete recipes before they silently select a different owner."""
    environ = os.environ if environ is None else environ
    for name, flag in FLAGS.items():
        if flag.kind == "retired" and name not in IGNORED_RETIRED and environ.get(name) == "1":
            raise ValueError(f"{name} selects a retired experiment; reproduce it from the original research checkout")


def record(environ: Mapping[str, str] | None = None) -> dict:
    """Reject retired selectors, then record the solver's environment once at construction."""
    environ = os.environ if environ is None else environ
    reject_retired(environ)
    return record_environment(FLAGS, PREFIXES, environ)
