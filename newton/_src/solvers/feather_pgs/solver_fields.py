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
    Field("v_out", wp.float32, ("dof",), mirror=True),
    Field("qd_work", wp.float32, ("dof",)),
    Field("v_mf_accum", wp.float32, ("dof",)),
    Field("v_out_snap", wp.float32, ("dof",)),
)
"""Per-DOF velocity work vectors."""

STAGE3_SNAPSHOTS = FieldGroup(
    Field("_debug_stage3_qd_work", wp.float32, ("dof",), grad=False),
    Field("_debug_stage3_joint_qdd", wp.float32, ("dof",), grad=False),
    Field("_debug_stage3_v_hat", wp.float32, ("dof",), grad=False),
)
"""Stage-3 snapshots for external NaN dumpers (opt-in); without a gradient, a wp.Tape records no copy."""

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
    Field("_joint_dfs_order", wp.int32, ("joint",), grad=False),
    Field("_joint_dfs_position", wp.int32, ("joint",), grad=False),
    Field("_joint_dfs_end", wp.int32, ("joint",), grad=False),
    Field("_joint_up_flags", wp.int32, ("joint",), grad=False),
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

STATE_GRAD_SCRATCH = FieldGroup(Field("joint_solve_tmp", wp.float32, ("dof",)))
"""Scratch of trisolve_loop's custom gradient, with gradients only."""

# ---------------------------------------------------------------------------------------------------------------------
# Size groups
# ---------------------------------------------------------------------------------------------------------------------

SIZE_FACTORS = FieldGroup(
    Field("L", wp.float32, ("factor_art", "factor_size", "factor_size")),
    Field("Y", wp.float32, ("response_art", "response_row", "response_size")),
    Field("R", wp.float32, ("art", "size")),
)
"""One size group's mass factor, response and armature; compact and sparse sizes keep (1, 1, 1) placeholders."""

SIZE_GROUPED_SOLVE = FieldGroup(
    Field("tau", wp.float32, ("art", "size", 1), grad=False),
    Field("qdd", wp.float32, ("art", "size", 1), grad=False),
)
"""Grouped scratch of the dense triangular solve (not of the compact paths)."""

SIZE_SYSTEM = FieldGroup(
    Field("H", wp.float32, ("factor_art", "factor_size", "factor_size")),
    Field("J", wp.float32, ("response_art", "response_row", "response_size")),
)
"""One size group's mass matrix and Jacobian; double-buffered on CUDA streams."""

# ---------------------------------------------------------------------------------------------------------------------
# Dense rows
# ---------------------------------------------------------------------------------------------------------------------

DENSE_ROWS = FieldGroup(
    Field("rhs", wp.float32, ("world", "row"), mirror=True),
    Field("rhs_unbiased", wp.float32, ("world", "row")),
    Field("impulses", wp.float32, ("world", "row"), mirror=True),
    Field("diag", wp.float32, ("world", "row"), mirror=True),
    Field("row_type", wp.int32, ("world", "row"), mirror=True),
    Field("row_parent", wp.int32, ("world", "row"), fill=-1, mirror=True),
    Field("row_mu", wp.float32, ("world", "row"), mirror=True),
    Field("row_restitution", wp.float32, ("world", "row")),
    Field("row_beta", wp.float32, ("world", "row")),
    Field("row_cfm", wp.float32, ("world", "row")),
    Field("phi", wp.float32, ("world", "row"), mirror=True),
    Field("target_velocity", wp.float32, ("world", "row"), mirror=True),
    Field("drive_stiffness", wp.float32, ("world", "row")),
    Field("drive_damping", wp.float32, ("world", "row")),
    Field("drive_geom_error", wp.float32, ("world", "row")),
    Field("drive_max_force", wp.float32, ("world", "row")),
    Field("drive_target_vel_bias", wp.float32, ("world", "row"), mirror=True),
    Field("drive_vel_multiplier", wp.float32, ("world", "row"), mirror=True),
    Field("drive_impulse_multiplier", wp.float32, ("world", "row"), mirror=True),
    Field("drive_max_impulse", wp.float32, ("world", "row"), mirror=True),
    Field("constraint_count", wp.int32, ("world",), mirror=True),
)
"""Every world's dense rows: the system, the row records and the PhysX drive coefficients."""

DENSE_ROW_WEIGHTS = FieldGroup(Field("row_w", wp.float32, ("world", "row"), fill=1.0, grad=False))
"""Per-row regularization weights (speculative and restitution rows keep the rigid law)."""

DENSE_DRIVE_LIMITS = FieldGroup(Field("drive_vel_limit", wp.float32, ("world", "row")))
"""Per-row joint velocity limits of the fused drive-row clamp."""

DENSE_DELASSUS = FieldGroup(Field("C", wp.float32, ("world", "row", "row")))
"""The assembled Delassus matrices of split mode."""

WORLD_RESPONSE = FieldGroup(
    Field("J_world", wp.float32, ("world", "row", "world_dof"), mirror=True),
    Field("Y_world", wp.float32, ("world", "row", "world_dof"), mirror=True),
)
"""World-indexed Jacobian and response of matrix-free and sparse split modes."""

DENSE_WARMSTART = FieldGroup(
    Field("_ws_prev_dense_impulses", wp.float32, ("world", "row")),
    Field("_ws_prev_dense_row_type", wp.int32, ("world", "row"), fill=-1, grad=False),
    Field("_ws_prev_dense_row_parent", wp.int32, ("world", "row"), fill=-1, grad=False),
)
"""Identity-matched warm-start carry of the dense rows."""

FUSED_DIAGONAL_LIMITS = FieldGroup(
    Field("_fused_diagonal_limit_active", wp.int32, ("world", "world_dof"), grad=False),
    Field("_fused_diagonal_limit_lower_rhs", wp.float32, ("world", "world_dof"), grad=False),
    Field("_fused_diagonal_limit_upper_rhs", wp.float32, ("world", "world_dof"), grad=False),
    Field("_fused_diagonal_limit_lower_lambda", wp.float32, ("world", "world_dof"), grad=False),
    Field("_fused_diagonal_limit_upper_lambda", wp.float32, ("world", "world_dof"), grad=False),
)
"""Position limits fused into the diagonal-mass sweep, outside the dense row system."""

WORLD_ROW_STATE = FieldGroup(
    Field("slot_counter", wp.int32, ("world",)),
    Field("_dense_first_rejected_slot", wp.int32, ("world",), fill=ROW_SLOT_UNBOUNDED, grad=False),
    # Per-world dense row-family boundaries, snapshotted from slot_counter each step. The dense slot layout is
    # [drive][joint-limit][joint-vel-limit][contact/friction], so [world, 0] ends the drive and position-limit
    # prefix (row_phase 3), [world, 1] the velocity-limit segment (row_phase 5), and contacts/friction occupy
    # [bounds[world, 1], m_dense) (row_phase 4).
    Field("dense_phase_bounds", wp.int32, ("world", 2), grad=False),
    # First matrix-free slot past the contact/friction rows (start of the velocity-limit tail).
    Field("mf_contact_rows_end", wp.int32, ("world",), grad=False),
)
"""Per-world dense row reservation: the slot counter, the first rejected slot and the row-family boundaries."""

CONTACT_RUNS = FieldGroup(
    Field("_contact_run_first", wp.int32, ("world", "run_block"), fill=2**31 - 1, grad=False),
    Field("_contact_run_last", wp.int32, ("world", "run_block"), fill=-1, grad=False),
    Field("_contact_run_rows", wp.vec2i, ("world", "run_block"), grad=False),
)
"""Contact runs of each world and shape block under sorted contacts (``record_world_contact_runs``)."""

VELOCITY_LIMIT_SLOTS = FieldGroup(
    Field("velocity_limit_slot", wp.int32, ("limit_row",), fill=-1),
    Field("velocity_limit_sign", wp.float32, ("limit_row",)),
)
"""Two unilateral joint velocity-limit rows per DOF (lower ``+e_i``, upper ``-e_i``): their slots and signs."""

DRIVE_SLOTS = FieldGroup(Field("drive_slot", wp.int32, ("dof",), fill=-1))
"""Each DOF's PhysX drive row slot."""

DENSE_CONTACT_UNITS = FieldGroup(
    Field("_dense_unit_state", wp.float32, ("world", "row", "unit_fields"), grad=False),
    Field("_dense_unit_rows", wp.int32, ("world", "row"), grad=False),
    Field("_dense_unit_block", wp.float32, ("unit_block",), grad=False),
)
"""Per-world contact-unit storage of the dense loop sweep (``contact_update="unit"``)."""

MF_CONTACT_UNITS = FieldGroup(
    Field("_mf_unit_state", wp.float32, ("world", "mf_row", "unit_fields"), grad=False),
    Field("_mf_unit_rows", wp.int32, ("world", "mf_row"), grad=False),
    Field("_mf_unit_block", wp.float32, ("unit_block",), grad=False),
)
"""Per-world contact-unit storage of the free-body loop sweep."""

PREELIMINATION = FieldGroup(
    Field("_preelim_slots", wp.int32, ("preelim_row",), fill=-1, grad=False),
    Field("_preelim_nB", wp.int32, ("preelim",), grad=False),
    Field("_preelim_S", wp.float32, ("preelim_block",), grad=False),
    Field("_preelim_LS", wp.float32, ("preelim_block",), grad=False),
    # Relative diagonal regularization of the S factor, rebuilt each step from the current S diagonal.
    Field("_preelim_reg", wp.float32, ("preelim_row",), grad=False),
)
"""Fixed-capacity bilateral pre-elimination buffers, ``PREELIM_MAX_ROWS`` rows per eliminating articulation."""

Preelim = field_group.struct(
    "Preelim",
    (
        Field("art_to_preelim", wp.int32, ("articulation",), attr="_preelim_art_to_idx"),
        *(
            dataclasses.replace(PREELIMINATION[attr], name=attr.removeprefix("_preelim_"), attr=attr)
            for attr in ("_preelim_slots", "_preelim_nB", "_preelim_S", "_preelim_LS", "_preelim_reg")
        ),
        Field("reg_rel", wp.float32, attr="_preelim_reg_rel"),
        Field("reg_floor", wp.float32, attr="_preelim_reg_floor"),
    ),
    __name__,
)
"""The pre-elimination of :data:`PREELIMINATION`: each articulation's block index, the block's rows and their
count, its Schur complement, factor and regularization, and the regularization's relative scale and floor."""

SPARSE_FACTOR = FieldGroup(
    Field("_sparse_Linv", wp.float32, ("group", "nonzero"), grad=False),
    Field("_sparse_mass_matrix_status", wp.int32, ("group",), grad=False),
)
"""The native sparse factor's inverse entries and status, per articulation of the factored size."""

SPARSE_ROWS = FieldGroup(
    Field("_sparse_row_dof", wp.int32, ("index_world", "index_row", "index_support"), fill=None, grad=False),
    Field("_sparse_row_factor", wp.float32, ("world", "row", "support"), fill=None, grad=False),
    Field("_sparse_row_incident", wp.float32, ("world", "row"), fill=None, grad=False),
    Field("_sparse_factor_velocity_delta", wp.float32, ("world", "world_dof"), fill=None, grad=False),
)
"""Each sparse row's factor coordinates (unless the support table holds them), factor entries and incident
velocity, and each world's factor velocity change."""

SPARSE_CONTACT_RECORDS = FieldGroup(
    Field("_sparse_contact_records", wp.float32, ("world", "triplet", "record"), grad=False)
)
"""The single-stage transaction's per-triplet block records (``sparse_contact.CONTACT_RECORD_FLOATS``)."""

# ---------------------------------------------------------------------------------------------------------------------
# Matrix-free rows
# ---------------------------------------------------------------------------------------------------------------------

MF_COUNTERS = FieldGroup(
    Field("mf_constraint_count", wp.int32, ("world",), mirror=True),
    Field("mf_slot_counter", wp.int32, ("world",)),
    Field("_mf_first_rejected_slot", wp.int32, ("world",), fill=ROW_SLOT_UNBOUNDED, grad=False),
)
"""Per-world matrix-free row counts, reservation counter and first rejected slot."""

MF_ROWS = FieldGroup(
    Field("mf_body_a", wp.int32, ("world", "mf_row"), mirror=True),
    Field("mf_body_b", wp.int32, ("world", "mf_row"), mirror=True),
    Field("mf_J_a", wp.float32, ("world", "mf_row", 6), mirror=True),
    Field("mf_J_b", wp.float32, ("world", "mf_row", 6), mirror=True),
    Field("mf_MiJt_a", wp.float32, ("world", "mf_row", 6), mirror=True),
    Field("mf_MiJt_b", wp.float32, ("world", "mf_row", 6), mirror=True),
    Field("mf_rhs", wp.float32, ("world", "mf_row"), mirror=True),
    Field("mf_rhs_unbiased", wp.float32, ("world", "mf_row")),
    Field("mf_impulses", wp.float32, ("world", "mf_row"), mirror=True),
    Field("mf_eff_mass_inv", wp.float32, ("world", "mf_row"), mirror=True),
    Field("mf_row_type", wp.int32, ("world", "mf_row"), mirror=True),
    Field("mf_row_parent", wp.int32, ("world", "mf_row"), fill=-1, mirror=True),
    Field("mf_row_mu", wp.float32, ("world", "mf_row"), mirror=True),
    Field("mf_phi", wp.float32, ("world", "mf_row"), mirror=True),
    Field("mf_row_restitution", wp.float32, ("world", "mf_row")),
    Field("mf_dof_a", wp.int32, ("world", "mf_row"), mirror=True),
    Field("mf_dof_b", wp.int32, ("world", "mf_row"), mirror=True),
    Field("mf_meta_packed", wp.int32, ("world", "mf_meta"), grad=False),
)
"""Every world's matrix-free (free-body) rows. Packed metadata: four ints per row (dof pair, inverse effective
mass, rhs, type and parent)."""

MF_ROW_WEIGHTS = FieldGroup(Field("mf_row_w", wp.float32, ("world", "mf_row"), fill=1.0, grad=False))

MF_BODY_MAP = FieldGroup(
    Field("mf_body_Hinv", wp.spatial_matrix, ("body",)),
    Field("mf_body_list", wp.int32, ("world", "mf_body")),
    Field("mf_body_dof_start", wp.int32, ("world", "mf_body")),
    Field("mf_body_count", wp.int32, ("world",)),
    Field("mf_local_body_a", wp.int32, ("world", "mf_row")),
    Field("mf_local_body_b", wp.int32, ("world", "mf_row")),
)
"""Body maps and inverse inertia of the tiled matrix-free sweep (split mode)."""

MF_WARMSTART = FieldGroup(
    Field("_ws_prev_mf_impulses", wp.float32, ("world", "mf_row")),
    Field("_ws_prev_mf_row_type", wp.int32, ("world", "mf_row"), fill=-1, grad=False),
    Field("_ws_prev_mf_row_parent", wp.int32, ("world", "mf_row"), fill=-1, grad=False),
)
"""Matrix-free warm-start carry: last step's converged impulses and their row types."""

# ---------------------------------------------------------------------------------------------------------------------
# Capacity telemetry
# ---------------------------------------------------------------------------------------------------------------------

ROW_WATERMARKS = FieldGroup(
    *(
        Field(name, wp.int32, (1,), grad=False)
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
    Field("_diag_metrics", wp.float32, ("world", 4), grad=False),
    Field("_diag_ncp_metrics", wp.float32, ("world", 6), grad=False),
    Field("_diag_prev_impulses", wp.float32, ("world", "row"), grad=False),
)
"""PGS convergence diagnostics (``pgs_debug``); the NCP/MDP residuals hold six per world."""

PGS_MF_DIAGNOSTICS = FieldGroup(Field("_diag_prev_mf_impulses", wp.float32, ("world", "mf_row"), grad=False))

# ---------------------------------------------------------------------------------------------------------------------
# Construction plans uploaded once
# ---------------------------------------------------------------------------------------------------------------------

BODY_MAPS = FieldGroup(
    Field("body_to_joint", wp.int32, ("body",), grad=False),
    Field("body_to_articulation", wp.int32, ("body",), grad=False),
    Field("body_has_response_dofs", wp.int32, ("body",), grad=False),
    Field("body_response_dof_mask", wp.uint32, ("body",), grad=False),
    Field("body_single_response_dof", wp.int32, ("body",), grad=False),
)
"""Each body's tree joint, articulation, response presence, low-32 response support and single response DOF."""

MIMIC_PLAN = FieldGroup(
    Field("_mimic_art_start", wp.int32, ("articulation_bound",), grad=False),
    Field("_mimic_art_list", wp.int32, ("mimic",), grad=False),
    Field("_mimic_valid", wp.int32, ("mimic",), grad=False),
    Field("_mimic_world", wp.int32, ("mimic",), grad=False),
    Field("_mimic_dof0", wp.int32, ("mimic",), grad=False),
    Field("_mimic_dof1", wp.int32, ("mimic",), grad=False),
    Field("_mimic_q0", wp.int32, ("mimic",), grad=False),
    Field("_mimic_q1", wp.int32, ("mimic",), grad=False),
)
"""The static per-mimic lookup tables: each articulation's valid mimics (CSR) and each mimic's DOFs and coordinates."""

MIMIC_SLOTS = FieldGroup(Field("mimic_slot", wp.int32, ("mimic",), fill=-1, grad=False))

CONNECT_PLAN = FieldGroup(
    Field("_connect_art", wp.int32, ("closure",), grad=False),
    Field("_connect_body_p", wp.int32, ("closure",), grad=False),
    Field("_connect_body_c", wp.int32, ("closure",), grad=False),
    Field("_connect_anchor_p", wp.vec3, ("closure",), grad=False),
    Field("_connect_anchor_c", wp.vec3, ("closure",), grad=False),
    Field("_connect_parent_prescribed", wp.int32, ("closure",), grad=False),
    Field("_connect_world", wp.int32, ("closure",), grad=False),
    Field("_connect_valid", wp.int32, ("closure",), grad=False),
    Field("_connect_enabled", wp.int32, ("closure",), grad=False),
)
"""The static per-closure lookup tables of the enforced loop closures."""

CONNECT_SLOTS = FieldGroup(Field("connect_slot", wp.int32, ("closure",), fill=-1, grad=False))

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
    Field("_sparse_diagonal_row_dof", wp.int32, ("world", "row", 2), fill=-1, grad=False),
    Field("_sparse_diagonal_row_jy", wp.float32, ("world", "row", 4), grad=False),
)
"""Each sparse-diagonal row's two response coordinates and their J and Y entries."""

SPARSE_CONTACT_GROUPS = FieldGroup(
    Field("_sparse_contact_group_count", wp.int32, ("world",), grad=False),
    Field("_sparse_contact_group_heads", wp.int32, ("world", "world_dof"), fill=None, grad=False),
)
"""Independent contact groups per world (triples or units), by head row."""

SPARSE_CONTACT_SERIAL = FieldGroup(
    Field("_sparse_contact_serial_count", wp.int32, ("world",), grad=False),
    Field("_sparse_contact_serial_normals", wp.int32, ("world", "serial"), fill=None, grad=False),
)
"""Contacts solved serially after the speculative batches, by normal row."""

# ---------------------------------------------------------------------------------------------------------------------
# Capacity status
# ---------------------------------------------------------------------------------------------------------------------

CAPACITY_STATUS = FieldGroup(
    Field("_constraint_capacity_status", wp.int32, (2,), grad=False),
    Field("constraint_overflow", wp.bool, ("world",), grad=False),
    Field("_row_dropped_all", wp.int32, (2, "status_world"), grad=False),
)
"""Sticky capacity flags of the dense and matrix-free rows, the per-world overflow latch and dropped rows."""

OVERFLOW_WARNING = FieldGroup(Field("_row_overflow_warning_emitted", wp.int32, (2,), grad=False))
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
