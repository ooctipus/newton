# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The device fields SolverFeatherPGS allocates, declared once by group (see :mod:`.field_group`).

Dimension names: ``dof`` (joint DOFs), ``articulation``, ``body``, ``world``,
``row`` (dense rows per world), ``mf_row`` (matrix-free rows per world),
``mf_body`` (matrix-free bodies per world), ``world_dof`` (DOFs of the widest
world), ``art``/``size`` (articulations and DOFs of one size group). A group
that a feature does not use is allocated at the sizes of its placeholder (one
row, one world) or not at all, as each allocation site says.
"""

import dataclasses

import warp as wp

from . import field_group
from .field_group import Field, FieldGroup
from .spatial_laws import inertia_record

ROW_SLOT_UNBOUNDED = 2**31 - 1
"""First rejected slot of a row family that rejected none this step."""

# ---------------------------------------------------------------------------------------------------------------------
# Generalized-coordinate work of the step
# ---------------------------------------------------------------------------------------------------------------------

DOF_WORK = FieldGroup(
    Field("v_hat", wp.float32, ("dof",)),
    Field("v_out", wp.float32, ("dof",)),
    Field("qd_work", wp.float32, ("dof",)),
)
"""Per-DOF velocity work vectors."""

STAGE3_SNAPSHOTS = FieldGroup(
    Field("_debug_stage3_qd_work", wp.float32, ("dof",)),
    Field("_debug_stage3_joint_qdd", wp.float32, ("dof",)),
    Field("_debug_stage3_v_hat", wp.float32, ("dof",)),
)
"""Stage-3 snapshots for external NaN dumpers (opt-in)."""

ARTICULATION_WORK = FieldGroup(
    Field("mass_update_mask", wp.int32, ("articulation",)),
)

BODY_INERTIA = FieldGroup(
    Field("body_X_com", wp.transform, ("body",), fill=None),
    Field("body_I_c", inertia_record, ("body",), fill=None),
    Field("_body_inertia_terms", inertia_record, ("body",), fill=None),
)
"""Body inertia about the articulation origin as inertia records (:data:`.spatial_laws.inertia_record`): each
body's own (``_body_inertia_terms``) and its subtree's composite (``body_I_c``)."""

ARTICULATION_ORIGIN = FieldGroup(Field("articulation_origin", wp.vec3, ("articulation",)))

JOINT_FOREST = FieldGroup(
    Field("_joint_dfs_order", wp.int32, ("joint",)),
    Field("_joint_dfs_position", wp.int32, ("joint",)),
    Field("_joint_dfs_end", wp.int32, ("joint",)),
    Field("_joint_up_flags", wp.int32, ("joint",)),
)
"""The articulation joints' forest (:class:`.tree_plan.TreePlan`) on the device: its depth-first order, each
joint's position and each position's subtree end (the CRBA's ancestor test), and how the up scans meet each
joint (:attr:`.tree_plan.TreePlan.up_flags`)."""

AUGMENTED_DRIVE_ROWS = FieldGroup(
    Field("aug_row_counts", wp.int32, ("articulation",)),
    Field("aug_row_dof_index", wp.int32, ("drive_row",)),
    Field("aug_row_K", wp.float32, ("drive_row",)),
)
"""Augmented drive rows, ``max_dofs`` per articulation."""

STATE_AUX = FieldGroup(
    Field("joint_qdd", wp.float32, ("dof",)),
    Field("joint_tau", wp.float32, ("dof",), fill=None),
    Field("joint_S_s", wp.spatial_vector, ("dof",), fill=None),
    Field("body_q_com", wp.transform, ("body",), fill=None),
    Field("body_I_s", wp.spatial_matrix, ("body",), fill=None),
    Field("body_v_s", wp.spatial_vector, ("body",), fill=None),
    Field("body_a_s", wp.spatial_vector, ("body",), fill=None),
    # Without a memset: the first step may run inside a CUDA graph capture, and a captured memset would erase the
    # cached body forces on every replay while the FK/ID cache stays valid. Stage 1 writes body_f_s before reading it.
    Field("body_f_s", wp.spatial_vector, ("body",), fill=None),
    Field("body_ft_s", wp.spatial_vector, ("body",)),
)
"""Per-state derived dynamics (maximal coordinates), allocated on the state they derive from."""

# ---------------------------------------------------------------------------------------------------------------------
# Size groups
# ---------------------------------------------------------------------------------------------------------------------

SIZE_FACTORS = FieldGroup(
    Field("L", wp.float32, ("factor_art", "factor_size", "factor_size")),
    Field("Y", wp.float32, ("y_art", "y_row", "y_size")),
    Field("R", wp.float32, ("art", "size")),
)
"""One size group's mass factor, response and armature; compact and sparse sizes keep (1, 1, 1) placeholders, and
so does the response of the matrix-free sweeps, which read the packed rows."""

SIZE_GROUPED_SOLVE = FieldGroup(
    Field("tau", wp.float32, ("art", "size", 1)),
    Field("qdd", wp.float32, ("art", "size", 1)),
)
"""Grouped scratch of the dense triangular solve (not of the compact paths)."""

SIZE_SYSTEM = FieldGroup(
    Field("H", wp.float32, ("factor_art", "factor_size", "factor_size")),
    Field("J", wp.float32, ("response_art", "response_row", "response_size")),
)
"""One size group's mass matrix and Jacobian; double-buffered on CUDA streams. The matrix-free sweeps' producers write
the packed rows instead (:mod:`.packed_rows`): their Jacobian is a (1, 1, 1) placeholder."""

# ---------------------------------------------------------------------------------------------------------------------
# Dense rows
# ---------------------------------------------------------------------------------------------------------------------

DENSE_ROWS = FieldGroup(
    Field("rhs", wp.float32, ("world", "row")),
    Field("impulses", wp.float32, ("world", "row")),
    Field("diag", wp.float32, ("world", "row")),
    Field("row_type", wp.int32, ("world", "row")),
    Field("row_parent", wp.int32, ("world", "row"), fill=-1),
    Field("row_mu", wp.float32, ("world", "row")),
    Field("row_restitution", wp.float32, ("world", "row")),
    Field("row_beta", wp.float32, ("world", "row")),
    Field("row_cfm", wp.float32, ("world", "row")),
    Field("phi", wp.float32, ("world", "row")),
    Field("target_velocity", wp.float32, ("world", "row")),
    Field("constraint_count", wp.int32, ("world",)),
    Field("_row_sides", wp.vec2i, ("world", "row")),
)
"""Every world's dense rows: the system and the row records (with each row's sides)."""

DENSE_ROW_WEIGHTS = FieldGroup(Field("row_w", wp.float32, ("world", "row"), fill=1.0))
"""Per-row regularization weights (speculative and restitution rows keep the rigid law)."""

WORLD_RESPONSE = FieldGroup(
    Field("J_world", wp.float32, ("world", "row", "world_dof")),
    Field("Y_world", wp.float32, ("world", "row", "world_dof")),
)
"""World-indexed Jacobian and response of the CUDA (matrix-free) route."""

DENSE_WARMSTART = FieldGroup(
    Field("_ws_prev_dense_impulses", wp.float32, ("world", "row")),
    Field("_ws_prev_dense_row_type", wp.int32, ("world", "row"), fill=-1),
    Field("_ws_prev_dense_row_parent", wp.int32, ("world", "row"), fill=-1),
)
"""Identity-matched warm-start carry of the dense rows."""

FUSED_DIAGONAL_LIMITS = FieldGroup(
    Field("_fused_diagonal_limit_active", wp.int32, ("world", "world_dof")),
    Field("_fused_diagonal_limit_lower_rhs", wp.float32, ("world", "world_dof")),
    Field("_fused_diagonal_limit_upper_rhs", wp.float32, ("world", "world_dof")),
    Field("_fused_diagonal_limit_lower_lambda", wp.float32, ("world", "world_dof")),
    Field("_fused_diagonal_limit_upper_lambda", wp.float32, ("world", "world_dof")),
)
"""Position limits fused into the diagonal-mass sweep, outside the dense row system."""

WORLD_ROW_STATE = FieldGroup(
    Field("slot_counter", wp.int32, ("world",)),
    Field("_dense_first_rejected_slot", wp.int32, ("world",), fill=ROW_SLOT_UNBOUNDED),
    # Per-world dense row-family boundaries, snapshotted from slot_counter each step. The dense slot layout is
    # [mimic][joint-limit][joint-vel-limit][contact/friction], so [world, 0] ends the mimic and position-limit
    # prefix, [world, 1] the velocity-limit segment, and contacts/friction occupy [bounds[world, 1], m_dense).
    Field("dense_phase_bounds", wp.int32, ("world", 2)),
    # First matrix-free slot past the contact/friction rows (start of the velocity-limit tail).
    Field("mf_contact_rows_end", wp.int32, ("world",)),
)
"""Per-world dense row reservation: the slot counter, the first rejected slot and the row-family boundaries."""

CONTACT_RUNS = FieldGroup(
    Field("_contact_run_first", wp.int32, ("world", "run_block"), fill=2**31 - 1),
    Field("_contact_run_last", wp.int32, ("world", "run_block"), fill=-1),
    Field("_contact_run_rows", wp.vec2i, ("world", "run_block")),
)
"""Contact runs of each world and shape block under sorted contacts (``record_world_contact_runs``)."""

VELOCITY_LIMIT_SLOTS = FieldGroup(
    Field("velocity_limit_slot", wp.int32, ("limit_row",), fill=-1, owner="laws"),
    Field("velocity_limit_sign", wp.float32, ("limit_row",), owner="laws"),
)
"""Two unilateral joint velocity-limit rows per DOF (lower ``+e_i``, upper ``-e_i``): their slots and signs."""

PREELIMINATION = FieldGroup(
    Field("_preelim_slots", wp.int32, ("preelim_row",), fill=-1),
    Field("_preelim_nB", wp.int32, ("preelim",)),
    Field("_preelim_S", wp.float32, ("preelim_block",)),
    Field("_preelim_LS", wp.float32, ("preelim_block",)),
    # A projection's block weights (the predictor's, then the leave's).
    Field("_preelim_weight", wp.float32, ("preelim_row",)),
)
"""Bilateral elimination buffers: per eliminating articulation, the model's largest block of rows and its square."""

Preelim = field_group.struct(
    "Preelim",
    (
        Field("art_to_preelim", wp.int32, ("articulation",), attr="_preelim_art_to_idx"),
        *(
            dataclasses.replace(PREELIMINATION[attr], name=attr.removeprefix("_preelim_"), attr=attr)
            for attr in (
                "_preelim_slots",
                "_preelim_nB",
                "_preelim_S",
                "_preelim_LS",
                "_preelim_weight",
            )
        ),
    ),
    __name__,
)
"""The bilateral elimination of :data:`PREELIMINATION`: each articulation's block index, the block's rows (kept, then
dropped) and its kept count, its Gram block and pivoted factor, and a projection's weights."""

SPARSE_FACTOR = FieldGroup(
    Field("_sparse_Linv", wp.float32, ("group", "nonzero")),
    Field("_sparse_mass_matrix_status", wp.int32, ("group",)),
)
"""The native sparse factor's inverse entries and status, per articulation of the factored size."""

SPARSE_ROWS = FieldGroup(
    Field("_sparse_row_dof", wp.int32, ("index_world", "index_row", "index_support"), fill=None),
    Field("_sparse_row_factor", wp.float32, ("world", "row", "support"), fill=None),
    Field("_sparse_row_incident", wp.float32, ("world", "row"), fill=None),
    Field("_sparse_factor_velocity_delta", wp.float32, ("world", "world_dof"), fill=None),
)
"""Each sparse row's factor coordinates (unless the support table holds them), factor entries and incident
velocity, and each world's factor velocity change."""

SPARSE_CONTACT_RECORDS = FieldGroup(Field("_sparse_contact_records", wp.float32, ("world", "triplet", "record")))
"""The single-stage transaction's per-triplet block records (``sparse_contact.CONTACT_RECORD_FLOATS``)."""

# ---------------------------------------------------------------------------------------------------------------------
# Matrix-free rows
# ---------------------------------------------------------------------------------------------------------------------

MF_COUNTERS = FieldGroup(
    Field("mf_constraint_count", wp.int32, ("world",)),
    Field("mf_slot_counter", wp.int32, ("world",)),
    Field("_mf_first_rejected_slot", wp.int32, ("world",), fill=ROW_SLOT_UNBOUNDED),
)
"""Per-world matrix-free row counts, reservation counter and first rejected slot."""

MF_ROWS = FieldGroup(
    Field("mf_body_a", wp.int32, ("world", "mf_row")),
    Field("mf_body_b", wp.int32, ("world", "mf_row")),
    Field("mf_Z_a", wp.float32, ("world", "mf_row", 6)),
    Field("mf_Z_b", wp.float32, ("world", "mf_row", 6)),
    Field("mf_rhs", wp.float32, ("world", "mf_row")),
    Field("mf_impulses", wp.float32, ("world", "mf_row")),
    Field("mf_eff_mass_inv", wp.float32, ("world", "mf_row")),
    Field("mf_row_type", wp.int32, ("world", "mf_row")),
    Field("mf_row_parent", wp.int32, ("world", "mf_row"), fill=-1),
    Field("mf_row_mu", wp.float32, ("world", "mf_row")),
    Field("mf_phi", wp.float32, ("world", "mf_row")),
    Field("mf_row_restitution", wp.float32, ("world", "mf_row")),
    Field("mf_row_incident", wp.float32, ("world", "mf_row")),
    Field("mf_dof_a", wp.int32, ("world", "mf_row")),
    Field("mf_dof_b", wp.int32, ("world", "mf_row")),
    Field("mf_meta_packed", wp.int32, ("world", "mf_meta")),
)
"""Every world's matrix-free (free-body) rows. Each side's six entries hold the row's Jacobian ``J`` as the writers
store it and its whitened form ``Z = L^-1 J^T`` once the rows' build has run (``L`` the body's held factor); a
restitutive contact keeps its incident velocity ``J v_hat``. Packed metadata: four ints per row (dof pair, inverse
effective mass, rhs, type and parent)."""

MF_ROW_WEIGHTS = FieldGroup(Field("mf_row_w", wp.float32, ("world", "mf_row"), fill=1.0))

MF_WARMSTART = FieldGroup(
    Field("_ws_prev_mf_impulses", wp.float32, ("world", "mf_row")),
    Field("_ws_prev_mf_row_type", wp.int32, ("world", "mf_row"), fill=-1),
    Field("_ws_prev_mf_row_parent", wp.int32, ("world", "mf_row"), fill=-1),
)
"""Matrix-free warm-start carry: last step's converged impulses and their row types."""

# ---------------------------------------------------------------------------------------------------------------------
# Capacity telemetry
# ---------------------------------------------------------------------------------------------------------------------

ROW_WATERMARKS = FieldGroup(
    *(
        Field(name, wp.int32, (1,))
        for name in (
            "_row_watermark_dense",
            "_row_watermark_mf",
            "_row_watermark_contact",
            "_row_watermark_dense_raw",
            "_row_watermark_mf_raw",
            "_row_dropped_dense_high_water",
            "_row_dropped_mf_high_water",
            "_row_overflow_dense_excess",
            "_row_overflow_mf_excess",
            "_row_overflow_dense_world_steps",
            "_row_overflow_mf_world_steps",
        )
    )
)
"""Whole-run row high-water telemetry for explicit readback (``row_watermark``)."""

PGS_DIAGNOSTICS = FieldGroup(
    Field("_diag_metrics", wp.float32, ("world", 4)),
    Field("_diag_ncp_metrics", wp.float32, ("world", 6)),
    Field("_diag_prev_impulses", wp.float32, ("world", "row")),
)
"""PGS convergence diagnostics (``pgs_debug``); the NCP/MDP residuals hold six per world."""

PGS_MF_DIAGNOSTICS = FieldGroup(Field("_diag_prev_mf_impulses", wp.float32, ("world", "mf_row")))

# ---------------------------------------------------------------------------------------------------------------------
# Construction plans uploaded once
# ---------------------------------------------------------------------------------------------------------------------

BODY_MAPS = FieldGroup(
    Field("body_to_joint", wp.int32, ("body",), owner="forest"),
    Field("body_to_articulation", wp.int32, ("body",), owner="forest"),
    Field("body_has_response_dofs", wp.int32, ("body",), owner="forest"),
    Field("body_response_dof_mask", wp.uint32, ("body",), owner="forest"),
    Field("body_single_response_dof", wp.int32, ("body",), owner="forest"),
)
"""Each body's tree joint, articulation, response presence, low-32 response support and single response DOF."""

MIMIC_PLAN = FieldGroup(
    Field("_mimic_art_start", wp.int32, ("articulation_bound",), owner="laws"),
    Field("_mimic_art_list", wp.int32, ("mimic",), owner="laws"),
    Field("_mimic_valid", wp.int32, ("mimic",), owner="laws"),
    Field("_mimic_world", wp.int32, ("mimic",), owner="laws"),
    Field("_mimic_dof0", wp.int32, ("mimic",), owner="laws"),
    Field("_mimic_dof1", wp.int32, ("mimic",), owner="laws"),
    Field("_mimic_q0", wp.int32, ("mimic",), owner="laws"),
    Field("_mimic_q1", wp.int32, ("mimic",), owner="laws"),
)
"""The static per-mimic lookup tables: each articulation's valid mimics (CSR) and each mimic's DOFs and coordinates."""

MIMIC_SLOTS = FieldGroup(Field("mimic_slot", wp.int32, ("mimic",), fill=-1, owner="laws"))

CONNECT_PLAN = FieldGroup(
    Field("_connect_art", wp.int32, ("closure",), owner="laws"),
    Field("_connect_body_p", wp.int32, ("closure",), owner="laws"),
    Field("_connect_body_c", wp.int32, ("closure",), owner="laws"),
    Field("_connect_anchor_p", wp.vec3, ("closure",), owner="laws"),
    Field("_connect_anchor_c", wp.vec3, ("closure",), owner="laws"),
    Field("_connect_parent_prescribed", wp.int32, ("closure",), owner="laws"),
    Field("_connect_world", wp.int32, ("closure",), owner="laws"),
    Field("_connect_valid", wp.int32, ("closure",), owner="laws"),
    Field("_connect_enabled", wp.int32, ("closure",), owner="laws"),
)
"""The static per-closure lookup tables of the enforced loop closures."""

CONNECT_SLOTS = FieldGroup(Field("connect_slot", wp.int32, ("closure",), fill=-1, owner="laws"))

Mimics = field_group.struct(
    "Mimics",
    (
        dataclasses.replace(MIMIC_SLOTS["mimic_slot"], name="slot", attr="mimic_slot"),
        *(dataclasses.replace(field, name=field.name.removeprefix("_mimic_"), attr=field.name) for field in MIMIC_PLAN),
        Field("enabled", wp.bool, ("mimic",), owner="model", attr="constraint_mimic_enabled"),
        Field("coef0", wp.float32, ("mimic",), owner="model", attr="constraint_mimic_coef0"),
        Field("coef1", wp.float32, ("mimic",), owner="model", attr="constraint_mimic_coef1"),
    ),
    __name__,
)
"""The mimic rows' tables (:data:`MIMIC_PLAN`), their dense slots and the model's enable flags and coefficients."""

Connects = field_group.struct(
    "Connects",
    (
        dataclasses.replace(CONNECT_SLOTS["connect_slot"], name="slot", attr="connect_slot"),
        *(
            dataclasses.replace(field, name=field.name.removeprefix("_connect_"), attr=field.name)
            for field in CONNECT_PLAN
        ),
    ),
    __name__,
)
"""The enforced loop closures' tables (:data:`CONNECT_PLAN`) and their dense slots."""

# ---------------------------------------------------------------------------------------------------------------------
# Sparse-diagonal contact schedule
# ---------------------------------------------------------------------------------------------------------------------

SPARSE_DIAGONAL_ROWS = FieldGroup(
    Field("_sparse_diagonal_row_dof", wp.int32, ("world", "row", 2), fill=-1),
    Field("_sparse_diagonal_row_z", wp.float32, ("world", "row", 2)),
)
"""Each sparse-diagonal row's two response coordinates and their whitened entries ``J / sqrt(m)``."""

SPARSE_CONTACT_GROUPS = FieldGroup(
    Field("_sparse_contact_group_count", wp.int32, ("world",)),
    Field("_sparse_contact_group_heads", wp.int32, ("world", "world_dof"), fill=None),
)
"""Independent contact groups per world (triples or units), by head row."""

SPARSE_CONTACT_SERIAL = FieldGroup(
    Field("_sparse_contact_serial_count", wp.int32, ("world",)),
    Field("_sparse_contact_serial_normals", wp.int32, ("world", "serial"), fill=None),
)
"""Contacts solved serially after the speculative batches, by normal row."""

# ---------------------------------------------------------------------------------------------------------------------
# Capacity status
# ---------------------------------------------------------------------------------------------------------------------

CAPACITY_STATUS = FieldGroup(
    Field("_constraint_capacity_status", wp.int32, (2,)),
    Field("constraint_overflow", wp.bool, ("world",)),
    Field("_row_dropped_all", wp.int32, (2, "status_world")),
)
"""Sticky capacity flags of the dense and matrix-free rows, the per-world overflow latch and dropped rows."""

OVERFLOW_WARNING = FieldGroup(Field("_row_overflow_warning_emitted", wp.int32, (2,)))
"""The one-shot overflow warning latch of each row family."""


LOCAL_OWNERS = (
    Field("primary_articulation", wp.int32, ("world",), attr="_local_primary_articulation"),
    Field("pair_articulation", wp.int32, ("world",), attr="_local_pair_articulation"),
    Field("residual_pair_articulation", wp.int32, ("world",), attr="_local_residual_pair_articulation"),
    Field("max_rows", wp.int32, attr="_local_solve_max_rows"),
    Field("residual_max_rows", wp.int32, attr="_local_residual_max_rows"),
    Field("residual_mf_max_rows", wp.int32, attr="_local_residual_mf_max_rows"),
    Field("owner", wp.int32, ("world",), attr="_local_solve_owner"),
    Field("general_world_count", wp.int32, (1,), attr="_local_general_world_count"),
    Field("general_worlds", wp.int32, ("world",), attr="_local_general_worlds"),
)
"""Each world's local-solve candidates (its primary articulation and paired articulations), the local row limits,
and the per-step owner of each world with the compact list of the general solver's worlds."""

LocalOwners = field_group.struct("LocalOwners", LOCAL_OWNERS, __name__)
