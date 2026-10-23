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
_CACHE = ("cached_dynamics",)


def _flag(name: str, kind: FlagKind, modules: tuple[str, ...], meaning: str) -> EnvironmentFlag:
    return EnvironmentFlag(name, kind, tuple(_PACKAGE + module for module in modules), meaning)


# Retired selectors that raise come first, in the order the solver has always
# checked them, with each newly retired one last: when several are set to 1, the
# first one names the error. The ignored ones (IGNORED_RETIRED) sit in name order
# with the other flags.
FLAGS = declare(
    _flag(
        "FEATHER_PGS_SINGLE_FACTOR",
        "retired",
        ("environment_flags", *_SOLVER),
        "Retired single-factor owner; =1 raises at construction (sparse admission still reads it).",
    ),
    _flag("FEATHER_PGS_BODY_BASIS_ROWS", "retired", ("environment_flags",), "Retired experiment; =1 raises."),
    _flag("FEATHER_PGS_SPARSE_PACKETS", "retired", ("environment_flags",), "Retired experiment; =1 raises."),
    _flag("FEATHER_PGS_SPARSE_PAIRED_GS", "retired", ("environment_flags",), "Retired experiment; =1 raises."),
    _flag("FEATHER_PGS_COUPLED_CONTACT", "retired", ("environment_flags",), "Retired experiment; =1 raises."),
    _flag("FEATHER_PGS_COUPLED_JACOBI", "retired", ("environment_flags",), "Retired experiment; =1 raises."),
    _flag(
        "FEATHER_PGS_DENSE_ROW_REGISTERS",
        "retired",
        ("environment_flags",),
        "Retired register-state copy of the 64-row tiled_row sweep; =1 raises.",
    ),
    _flag(
        "FEATHER_PGS_ALLEGRO_KINETIC_ROWS",
        "retired",
        ("environment_flags",),
        "Retired alias that turned keyed rows on; ignored (FEATHER_PGS_KEYED_ROWS=0 opts out of them).",
    ),
    _flag(
        "FEATHER_PGS_CAPTURE_COUNT",
        "diagnostic",
        _SOLVER,
        "Consecutive substeps the kernel capture records (default 1).",
    ),
    _flag(
        "FEATHER_PGS_CAPTURE_DIR",
        "diagnostic",
        _SOLVER,
        "Directory for kernel captures (default /tmp/fpgs_capture).",
    ),
    _flag(
        "FEATHER_PGS_CAPTURE_KERNEL",
        "diagnostic",
        _SOLVER,
        "=1 captures pgs_solve_mf_gs inputs and launch configuration for standalone replay.",
    ),
    _flag(
        "FEATHER_PGS_CAPTURE_PHASE",
        "diagnostic",
        _SOLVER,
        "Solve phase the kernel capture records (default any).",
    ),
    _flag(
        "FEATHER_PGS_CAPTURE_STEP",
        "diagnostic",
        _SOLVER,
        "Step at which the kernel capture starts (default -1, none).",
    ),
    _flag(
        "FEATHER_PGS_CHECK_GROUPED",
        "diagnostic",
        _SOLVER,
        "=1 compares grouped kernels with the serial kernels on the host.",
    ),
    _flag(
        "FEATHER_PGS_CHECK_ROWS",
        "diagnostic",
        _SOLVER,
        "=1 compares the native and masked contact-row builders.",
    ),
    _flag(
        "FEATHER_PGS_CHECK_ROWS_FUSED",
        "diagnostic",
        _SOLVER,
        "=1 compares the fused contact-row builder with the ordinary builder.",
    ),
    _flag(
        "FEATHER_PGS_COLORED_LANES",
        "policy",
        _SOLVER,
        "Lanes per world in the colored propagation warp kernel (default 1).",
    ),
    _flag(
        "FEATHER_PGS_COLORED_WPB",
        "policy",
        _SOLVER,
        "Worlds per block in the colored propagation warp kernel (default 2).",
    ),
    _flag(
        "FEATHER_PGS_COMPACT_CONTACT_BOUNDARY",
        "experiment",
        _SOLVER,
        "=1 selects the complete contact producer for the dense-six plus sparse solve.",
    ),
    _flag(
        "FEATHER_PGS_COMPACT_CONTACT_ORDERING",
        "experiment",
        _SOLVER,
        "=1 orders contacts for the compact contact producer; requires COMPACT_CONTACT_BOUNDARY.",
    ),
    _flag(
        "FEATHER_PGS_CONTACT_THREADS_X",
        "policy",
        _SOLVER,
        "Multiplier of the contact-build thread budget (default 1).",
    ),
    _flag(
        "FEATHER_PGS_CONTACT_UPDATE",
        "policy",
        _SOLVER,
        "sequential or unit overrides the contact_update argument (unset: the argument).",
    ),
    _flag(
        "FEATHER_PGS_DEBUG_CACHE",
        "diagnostic",
        _SOLVER,
        "=1 prints FK/ID cache validity every step.",
    ),
    _flag(
        "FEATHER_PGS_DEBUG_CACHE_CMP",
        "diagnostic",
        _SOLVER,
        "=1 compares the FK/ID cache with a recomputation.",
    ),
    _flag(
        "FEATHER_PGS_DEBUG_CACHE_MODE",
        "diagnostic",
        _SOLVER,
        "FK/ID cache invalidation override for bisection: zero_even or nozero (default ordinary).",
    ),
    _flag(
        "FEATHER_PGS_DEBUG_DELAY",
        "diagnostic",
        _SOLVER,
        "Spin iterations launched at the start of each step to expose stream races (default 0).",
    ),
    _flag(
        "FEATHER_PGS_DEBUG_SNAPSHOTS",
        "diagnostic",
        _SOLVER,
        "=1 copies the stage-3 joint velocity, acceleration and predicted velocity every substep for NaN dumps.",
    ),
    _flag(
        "FEATHER_PGS_DENSE_ROW_BUDGETS",
        "policy",
        _SOLVER,
        "tiled_row row budgets: 1 on for 64 or 128 rows, unset automatic (128 rows, 1024+ worlds), else off.",
    ),
    _flag(
        "FEATHER_PGS_FK_CACHE_SNAPSHOT",
        "reference",
        _SOLVER,
        "=1 restores the per-substep snapshot copy of the FK/ID cache as a bisection aid.",
    ),
    _flag(
        "FEATHER_PGS_FK_ID_CACHE",
        "policy",
        _SOLVER,
        "=0 disables the CUDA FK/inverse-dynamics cache.",
    ),
    _flag(
        "FEATHER_PGS_FRANKA_COMPACT_WORKSPACE",
        "retired",
        ("environment_flags",),
        "Retired compact workspace of a removed robot-specific owner; ignored.",
    ),
    _flag(
        "FEATHER_PGS_FREE_BODY_IMPLICIT_GYRO",
        "experiment",
        _SOLVER,
        "=1 applies an implicit gyroscopic predictor correction to independent free bodies.",
    ),
    _flag(
        "FEATHER_PGS_FUSED_K1",
        "policy",
        _SOLVER,
        "=0 disables the fused world-dynamics kernel.",
    ),
    _flag(
        "FEATHER_PGS_FUSED_K1_MASS",
        "experiment",
        _SOLVER,
        "=1 lets the fused world-dynamics kernel also build and factor the mass matrix.",
    ),
    _flag(
        "FEATHER_PGS_FUSED_PUBLISH",
        "experiment",
        _SOLVER,
        "=1 publishes each general-route articulation in one launch per size group.",
    ),
    _flag(
        "FEATHER_PGS_G1_CHAIN_SCAN",
        "retired",
        ("environment_flags",),
        "Retired G1 alias of the held-tree chain scan; ignored (FEATHER_PGS_TREE_CHAIN_SCAN=0 opts out of it).",
    ),
    _flag(
        "FEATHER_PGS_GROUPED_FK",
        "policy",
        _SOLVER,
        "=0 replaces the grouped template FK kernel with the serial kernel.",
    ),
    _flag(
        "FEATHER_PGS_GROUPED_HINV",
        "policy",
        _SOLVER,
        "=0 keeps the tiled response solve instead of the grouped one.",
    ),
    _flag(
        "FEATHER_PGS_GROUPED_HINV_FREE",
        "experiment",
        _SOLVER,
        "=1 uses the grouped response solve for free-body size groups.",
    ),
    _flag(
        "FEATHER_PGS_GROUPED_HINV_MAX",
        "policy",
        _SOLVER,
        "Largest DOF size that uses the grouped response solve (default 32).",
    ),
    _flag(
        "FEATHER_PGS_GROUPED_MASS",
        "experiment",
        _SOLVER,
        "=1 with GROUPED_TAU_MASS also builds the mass matrix in the grouped kernel.",
    ),
    _flag(
        "FEATHER_PGS_GROUPED_TAU_MASS",
        "experiment",
        _SOLVER,
        "=1 fuses joint torques and mass-matrix assembly in the grouped kernel.",
    ),
    _flag(
        "FEATHER_PGS_GROUP_LANES",
        "policy",
        ("grouped_dynamics",),
        "Lanes per articulation in grouped dynamics (default twice the widest tree level, at most 32).",
    ),
    _flag(
        "FEATHER_PGS_INDEPENDENT_COMPONENTS",
        "retired",
        ("environment_flags",),
        "Retired independent-components variant of a removed robot-specific owner; ignored.",
    ),
    _flag(
        "FEATHER_PGS_INK",
        "experiment",
        _SOLVER,
        "=1 lets the parallel sweep form the response H^-1 J^T and its diagonal in-kernel.",
    ),
    _flag(
        "FEATHER_PGS_INK_CHECK",
        "diagnostic",
        _SOLVER,
        "=1 writes the in-kernel response and diagonal back for comparison.",
    ),
    _flag(
        "FEATHER_PGS_K1_REAL_CHECK",
        "diagnostic",
        _SOLVER,
        "=1 checks the fused world-dynamics kernel against the ordinary stages.",
    ),
    _flag(
        "FEATHER_PGS_K1_TPL_SHARED",
        "reference",
        _SOLVER,
        "=0 reads fused-kernel template tables from global instead of shared memory.",
    ),
    _flag(
        "FEATHER_PGS_KEYED_ROWS",
        "policy",
        _SOLVER,
        "=0 disables the topology-generated keyed contact rows.",
    ),
    _flag(
        "FEATHER_PGS_KUKA_CURRENT_CONTACT",
        "retired",
        ("environment_flags",),
        "Retired current-contact geometry of a removed robot-specific owner; ignored.",
    ),
    _flag(
        "FEATHER_PGS_KUKA_JOINT_WORLD",
        "retired",
        ("environment_flags",),
        "Retired joint-world prediction of a removed robot-specific owner; ignored.",
    ),
    _flag(
        "FEATHER_PGS_LOCAL_ROW_PACKETS",
        "retired",
        ("environment_flags",),
        "Retired row-packet producer of a removed robot-specific owner; ignored (see FEATHER_PGS_ROW_PACKETS).",
    ),
    _flag(
        "FEATHER_PGS_MFGS_STREAM_LAMBDA",
        "reference",
        _SOLVER,
        "1, true, yes or on forces streamed matrix-free impulses instead of the shared-memory choice.",
    ),
    _flag(
        "FEATHER_PGS_MF_EXACT_ROWSUM",
        "experiment",
        _SOLVER,
        "=1 uses the exact A-based step scale in the matrix-free parallel sweep.",
    ),
    _flag(
        "FEATHER_PGS_MIXED_TREE_KINETIC_STATE",
        "policy",
        _CACHE,
        "Values other than 1 disable the dense-route held-tree cache under cached_dynamics='auto'.",
    ),
    _flag(
        "FEATHER_PGS_PACKED_TREE_KINETIC_STATE",
        "retired",
        ("environment_flags",),
        "Retired tree cache of the packed sparse-factor route; ignored.",
    ),
    _flag(
        "FEATHER_PGS_PAIRED_COMPONENTS",
        "policy",
        _SOLVER,
        "Values other than 1 disable independent components in the paired response.",
    ),
    _flag(
        "FEATHER_PGS_PAIRED_GENERAL_OVERLAP",
        "retired",
        ("environment_flags",),
        "Retired overlap of the paired and general solve streams; ignored.",
    ),
    _flag(
        "FEATHER_PGS_PATCH_ACTIVE_ROWS",
        "policy",
        _SOLVER,
        "=0 keeps patch friction on the full ordered sparse sweep instead of the active-row scan (default on); "
        "with FEATHER_PGS_SPARSE_CONTACT_TRIPLETS=1 that sweep fuses anchor triplets, so the previous kernel "
        "needs both off.",
    ),
    _flag(
        "FEATHER_PGS_PATCH_CONDITIONAL_NODES",
        "experiment",
        _SOLVER,
        "=1 sorts captured patch builds that do not keep their pairs under a conditional graph node that skips "
        "empty and already sorted contact frames (default 0: other values keep that sort unconditional).",
    ),
    _flag(
        "FEATHER_PGS_PATCH_DENSE_PATHS",
        "policy",
        _SOLVER,
        "=0 keeps patch friction off the paired factor sweep's contact units, masked rows, the resolved-world "
        "classifier and the dense route for free-body worlds (default on).",
    ),
    _flag(
        "FEATHER_PGS_PATCH_FRAME_REUSE",
        "policy",
        _SOLVER,
        "=0 sorts the patch pairs of every build, small frames over their whole capacity, instead of keeping them "
        "on captured builds that reuse the previous captured build's contact frame (default on).",
    ),
    _flag(
        "FEATHER_PGS_PATCH_PARALLEL_ROWS",
        "policy",
        _SOLVER,
        "=0 keeps patch friction off the parallel tiers and keyed rows, on its serial sweeps (default on).",
    ),
    _flag(
        "FEATHER_PGS_PATCH_PARALLEL_SWEEPS",
        "policy",
        _SOLVER,
        "A positive count or auto overrides mf_gs_parallel_linked_sweeps, the linked tiers' sweep budget.",
    ),
    _flag(
        "FEATHER_PGS_POINT_PAIR_STEP",
        "experiment",
        _SOLVER,
        "isotropic steps each point-friction pair of the parallel tiers with one step (default per_row).",
    ),
    _flag(
        "FEATHER_PGS_PREDICTED_TWISTS",
        "policy",
        ("tree_kinetic_owner",),
        "Values other than 1 stop the tree predictor publishing body twists for the resolved-world classifier.",
    ),
    _flag(
        "FEATHER_PGS_PRISMATIC_LINEAR_STATE",
        "experiment",
        _SOLVER,
        "=1 keeps current linear state for fixed-root one-body prismatic branches; superseded where scalar "
        "branches admit them, unless FEATHER_PGS_SCALAR_BRANCHES=0.",
    ),
    _flag(
        "FEATHER_PGS_PRISMATIC_PUBLICATION",
        "experiment",
        _SOLVER,
        "=1 publishes fixed-root prismatic stars body-parallel; superseded where scalar branches admit them, "
        "unless FEATHER_PGS_SCALAR_BRANCHES=0.",
    ),
    _flag(
        "FEATHER_PGS_RESOLVED_WORLDS",
        "policy",
        ("resolved_worlds",),
        "Values other than 1 stop skipping worlds whose candidate rows all separate at the predictor.",
    ),
    _flag(
        "FEATHER_PGS_ROWS_FUSED",
        "experiment",
        _SOLVER,
        "=1 builds contact rows with one thread per row.",
    ),
    _flag(
        "FEATHER_PGS_ROWS_MASKED",
        "experiment",
        _SOLVER,
        "=1 builds contact rows with one thread per contact and DOF.",
    ),
    _flag(
        "FEATHER_PGS_ROWS_NATIVE",
        "experiment",
        _SOLVER,
        "=1 builds contact rows with one warp per contact.",
    ),
    _flag(
        "FEATHER_PGS_ROWS_THREADS_X",
        "policy",
        _SOLVER,
        "Multiplier of the row-build thread budget (default 4).",
    ),
    _flag(
        "FEATHER_PGS_ROW_PACKETS",
        "policy",
        _SOLVER,
        "Values other than 1 disable in-place limit-prefix and contact packets for the local solve owners.",
    ),
    _flag(
        "FEATHER_PGS_SCALAR_BRANCHES",
        "policy",
        _SOLVER,
        "=0 keeps a diagonal-mass group of world-fixed roots over one-coordinate leaves out of the held-tree "
        "owner's scalar branches, which run it, own the other articulations of its worlds and supersede the "
        "prismatic opt-ins there (default on).",
    ),
    _flag(
        "FEATHER_PGS_SIMPLE_WORLD_ZERO",
        "retired",
        ("environment_flags",),
        "Retired simple-world classifier of a removed robot-specific owner; ignored.",
    ),
    _flag(
        "FEATHER_PGS_SKIP_J_CLEAR",
        "experiment",
        _SOLVER,
        "=1 skips the Jacobian clear when masked or native builders write rows in full.",
    ),
    _flag(
        "FEATHER_PGS_SPARSE_BRANCH_SWEEPS",
        "retired",
        ("environment_flags",),
        "Retired branch-Jacobi sparse sweep; ignored, sparse factors keep their ordered sweeps.",
    ),
    _flag(
        "FEATHER_PGS_SPARSE_CONTACT_BLOCK",
        "retired",
        ("environment_flags",),
        "Retired contact-block rows of the packed sparse-factor route; ignored.",
    ),
    _flag(
        "FEATHER_PGS_SPARSE_CONTACT_DIRECT",
        "experiment",
        _SOLVER,
        "=1 launches sparse diagonal contact responses over the full contact capacity.",
    ),
    _flag(
        "FEATHER_PGS_SPARSE_CONTACT_TRIPLETS",
        "experiment",
        _SOLVER,
        "=1 fuses contact triplets in the sparse sweep: point friction's in the active-row scan, patch "
        "friction's anchor units in whichever sweep it runs.",
    ),
    _flag(
        "FEATHER_PGS_SPARSE_FACTOR",
        "retired",
        ("environment_flags",),
        "Retired packed sparse-factor route; ignored, the general route (sparse factors where admitted) runs.",
    ),
    _flag(
        "FEATHER_PGS_SPARSE_GROUPED_DYNAMICS",
        "policy",
        _SOLVER,
        "Values other than 1 keep grouped-dynamics solvers on ordinary factors instead of automatic sparse ones.",
    ),
    _flag(
        "FEATHER_PGS_SPARSE_LEVEL_UPDATE",
        "retired",
        ("environment_flags",),
        "Retired level-by-level factor update of the packed sparse-factor route; ignored.",
    ),
    _flag(
        "FEATHER_PGS_SPARSE_LIMIT_JACOBI",
        "retired",
        ("environment_flags",),
        "Retired Jacobi joint-limit solve of the packed sparse-factor route; ignored.",
    ),
    _flag(
        "FEATHER_PGS_SPARSE_LOCAL_FACTORS",
        "policy",
        _SOLVER,
        "=0 disables per-world local factor workspaces in the sparse sweep.",
    ),
    _flag(
        "FEATHER_PGS_SPARSE_METRIC_TANGENTS",
        "retired",
        ("environment_flags",),
        "Retired metric tangents of the packed sparse-factor route; ignored.",
    ),
    _flag(
        "FEATHER_PGS_SPARSE_PARALLEL_LIMITS",
        "retired",
        ("environment_flags",),
        "Retired parallel joint-limit prefix of the packed sparse-factor route; ignored.",
    ),
    _flag(
        "FEATHER_PGS_SPARSE_SMALL_WORLDS",
        "policy",
        _SOLVER,
        "=0 disables small-world lane packing in the sparse sweep.",
    ),
    _flag(
        "FEATHER_PGS_SPARSE_SPECTRAL_TANGENTS",
        "retired",
        ("environment_flags",),
        "Retired spectral tangents of the packed sparse-factor route; ignored.",
    ),
    _flag(
        "FEATHER_PGS_SPARSE_SPLIT",
        "policy",
        _SOLVER,
        "Values other than 1 keep split mode off automatic sparse factors.",
    ),
    _flag(
        "FEATHER_PGS_SPARSE_TRANSACTION",
        "policy",
        _SOLVER,
        "Transaction of fused sparse contact triplets: single (one load group and reduction, per-triplet records) or "
        "staged (normal, then tangents, metric cache); auto (default) is single for patch and staged for point "
        "friction.",
    ),
    _flag(
        "FEATHER_PGS_SPARSE_TRANSACTION_BLOCKS",
        "experiment",
        _SOLVER,
        "Minimum blocks per SM of the single-stage sparse transaction at 128 rows or fewer: 16 (default) or 14.",
    ),
    _flag(
        "FEATHER_PGS_SPARSE_VELOCITY_LIMITS",
        "policy",
        _SOLVER,
        "Values other than 1 keep velocity-limit models off automatic sparse factors.",
    ),
    _flag(
        "FEATHER_PGS_STAGE_CLOCKS",
        "diagnostic",
        ("tree_kinetic_owner",),
        "=1 compiles per-stage cycle counters into the held-tree kernels.",
    ),
    _flag(
        "FEATHER_PGS_SYNC_TIMINGS",
        "diagnostic",
        _SOLVER,
        "1, true, yes or on prints synchronized per-stage timings.",
    ),
    _flag(
        "FEATHER_PGS_SYNC_TIMINGS_COUNT",
        "diagnostic",
        _SOLVER,
        "Number of steps with synchronized timings (default 1).",
    ),
    _flag(
        "FEATHER_PGS_SYNC_TIMINGS_START",
        "diagnostic",
        _SOLVER,
        "First step with synchronized timings (default 20).",
    ),
    _flag(
        "FEATHER_PGS_TIER_BLOCKS",
        "experiment",
        _SOLVER,
        "Blocks that grid-stride the parallel sweep tiers over per-tier world lists (default 0, off).",
    ),
    _flag(
        "FEATHER_PGS_TREE_CHAIN_SCAN",
        "policy",
        ("tree_kinetic_owner",),
        "Chain-scan plan for the held-tree owner: 1 on, other values off (default per plan).",
    ),
    _flag(
        "FEATHER_PGS_TREE_FOREST",
        "policy",
        ("mixed_tree_kinetic",),
        "Held-tree groups: 1 one group per world, other values per-size groups (default by layout).",
    ),
    _flag(
        "FEATHER_PGS_TREE_KINETIC_STATE",
        "policy",
        _CACHE,
        "Values other than 1 disable the sparse-route tree cache under cached_dynamics='auto'.",
    ),
    _flag(
        "FEATHER_PGS_TREE_LANE_PACK",
        "policy",
        ("tree_kinetic_plan",),
        "Held-tree lane width: 1 timed at construction (default), 0 one tree per block, 4/8/16/32 fixed.",
    ),
    _flag(
        "FEATHER_PGS_TREE_REFERENCE",
        "reference",
        ("tree_kinetic_owner",),
        "=1 selects the serial held solve, level reduction and shared-memory scans in held-tree kernels.",
    ),
    _flag(
        "FEATHER_PGS_TREE_REPAIR_CLASSIFY",
        "policy",
        ("tree_kinetic_owner",),
        "=0 keeps the one-pass held-tree repair launch (default 1 on CUDA: list the blocks to repair, then repair "
        "over the list on a capped grid; motion-velocity layouts always launch one pass).",
    ),
    _flag(
        "FEATHER_PGS_TREE_REPAIR_SKIP",
        "policy",
        ("tree_kinetic_owner",),
        "=0 launches every captured held-tree repair (default 1: a repair after a finish of the same capture "
        "on the bank it published launches nothing).",
    ),
    _flag(
        "FEATHER_PGS_VELOCITY_LIMIT_LAW",
        "policy",
        _SOLVER,
        "stateless or accumulated overrides the velocity_limit_law argument (unset: the argument).",
    ),
    _flag(
        "FEATHER_PGS_WORLD_ROWS",
        "retired",
        ("environment_flags",),
        "Retired in-sweep world rows, which keyed rows always declined where recipes set it; ignored.",
    ),
    _flag(
        "FEATHER_PGS_WORLD_SCAN_PUBLICATION",
        "retired",
        ("environment_flags",),
        "Retired world-scan publication of a removed robot-specific owner; ignored.",
    ),
    _flag(
        "IL_NEWTON_FPGS_MF_WARMSTART",
        "policy",
        _SOLVER,
        "Legacy alias: 1, true, yes or on enables contact warm start like mf_warmstart=True.",
    ),
    _flag(
        "IL_NEWTON_FPGS_MF_WARMSTART_DECAY",
        "policy",
        _SOLVER,
        "Legacy alias: warm-start decay used with IL_NEWTON_FPGS_MF_WARMSTART (default mf_warmstart_decay).",
    ),
)


# Retired names that Isaac Lab's benchmark recipes still set: the packed
# sparse-factor route's selectors (480f0f44), the chain-scan and keyed-row
# aliases (d35949d6), the selectors of removed research owners and of the
# branch-Jacobi sweep (790f9782, aa2f332b), and the in-sweep world rows'
# selector, which keyed rows declined wherever a recipe set it. Nothing in
# Newton or Isaac Lab reads them, and recipes are not Newton's to edit, so they
# are ignored rather than rejected: the general route runs, the construction
# record lists them as retired and the solver warns once per process.
IGNORED_RETIRED = frozenset(
    (
        "FEATHER_PGS_ALLEGRO_KINETIC_ROWS",
        "FEATHER_PGS_FRANKA_COMPACT_WORKSPACE",
        "FEATHER_PGS_G1_CHAIN_SCAN",
        "FEATHER_PGS_INDEPENDENT_COMPONENTS",
        "FEATHER_PGS_KUKA_CURRENT_CONTACT",
        "FEATHER_PGS_KUKA_JOINT_WORLD",
        "FEATHER_PGS_LOCAL_ROW_PACKETS",
        "FEATHER_PGS_PACKED_TREE_KINETIC_STATE",
        "FEATHER_PGS_PAIRED_GENERAL_OVERLAP",
        "FEATHER_PGS_SIMPLE_WORLD_ZERO",
        "FEATHER_PGS_SPARSE_BRANCH_SWEEPS",
        "FEATHER_PGS_SPARSE_CONTACT_BLOCK",
        "FEATHER_PGS_SPARSE_FACTOR",
        "FEATHER_PGS_SPARSE_LEVEL_UPDATE",
        "FEATHER_PGS_SPARSE_LIMIT_JACOBI",
        "FEATHER_PGS_SPARSE_METRIC_TANGENTS",
        "FEATHER_PGS_SPARSE_PARALLEL_LIMITS",
        "FEATHER_PGS_SPARSE_SPECTRAL_TANGENTS",
        "FEATHER_PGS_WORLD_ROWS",
        "FEATHER_PGS_WORLD_SCAN_PUBLICATION",
    )
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
