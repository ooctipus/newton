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
from .spatial_laws import FACTOR_REQUEST, inertia_record, limited_drive_force

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
    Field("mass_update_mask", wp.int32, ("articulation",), owner="factor"),
)

BODY_INERTIA = FieldGroup(
    Field("body_X_com", wp.transform, ("body",), fill=None, owner="factor"),
    Field("body_inertia_terms", inertia_record, ("body",), fill=None),
)
"""Body inertia about the articulation origin as inertia records (:data:`.spatial_laws.inertia_record`): each
body's own (``body_inertia_terms``) and its subtree's composite (``body_I_c``)."""

ARTICULATION_ORIGIN = FieldGroup(Field("articulation_origin", wp.vec3, ("articulation",), owner="factor"))

AUGMENTED_DRIVE_ROWS = FieldGroup(
    Field("aug_row_counts", wp.int32, ("articulation",), owner="factor"),
    Field("aug_row_dof_index", wp.int32, ("drive_row",), owner="factor"),
    Field("aug_row_K", wp.float32, ("drive_row",), owner="factor"),
    Field("aug_row_Kd", wp.float32, ("drive_row",), owner="factor"),
    Field("aug_row_H", wp.float32, ("drive_row",), owner="factor"),
    Field("aug_row_P", wp.float32, ("drive_row",), owner="factor"),
    Field("aug_row_dv", wp.float32, ("drive_row",), owner="factor"),
    Field("aug_drive", wp.vec4, ("dof",), owner="factor"),
    Field("aug_decided", wp.int32, ("articulation",), owner="factor"),
)
"""Augmented drive rows, ``max_dofs`` per articulation: each row's ``K``, which the trial factor holds; the ``K`` the step's
factor holds once the saturation decision has taken some out (``aug_row_Kd``, the passive damper's on a saturated drive);
the ``H_ii`` (armature in, ``K`` out) its factor last refreshed from; its pivot there without its ``K`` (``aug_row_P``),
which a release of the drive pivots on (:func:`.tree_kinetic_state.release`); what last step's rows did to its coordinate;
per coordinate, what the step's saturation decision reads on every drive
(:data:`.tree_kinetic_state.SATURATE_DRIVES`): ``aug_drive``, the drive's own ``K``, the passive damper's, the effort limit
and the unlimited ``u0``, one read of one record; and per articulation whether its step factor holds a decided set
(``aug_decided``), which the trial factor then keeps all of ``K`` beside."""

DRIVE_ROWS = (
    Field("row_K", wp.float32, ("drive_row",), owner="factor", attr="aug_row_K"),
    Field("row_dv", wp.float32, ("drive_row",), owner="factor", attr="aug_row_dv"),
    Field("drive", wp.vec4, ("dof",), owner="factor", attr="aug_drive"),
    Field("mass_update_requested", wp.int32, ("articulation",), owner="factor"),
    Field("v_out", wp.float32, ("dof",), owner="step"),
    Field("v_hat", wp.float32, ("dof",), owner="step"),
)
"""What a step's drives write on their rows, and last step's velocities (:func:`hold_drive_row`)."""

DriveRows = field_group.struct("DriveRows", DRIVE_ROWS, __name__)


@wp.func
def hold_drive_row(
    rows: DriveRows, row: int, articulation: int, dof: int, u0: float, K: float, passive: float, limit: float
) -> float:
    """Hold a drive's ``u0`` and ``K``, its passive damper's within, and return the limited force the trial solves with.
    A ``K`` the trial factor does not hold (new gains, the first step) requests the articulation's factor refresh; a
    saturation never does, since the decision keeps its ``K`` in the step factor beside it (``aug_row_Kd``). Only a drive
    with an excess reads ``row_dv``, last step's ``v_out - v_hat``, what its rows did to the coordinate.
    """
    if rows.row_K[row] != K:
        rows.row_K[row] = K
        rows.mass_update_requested[articulation] = rows.mass_update_requested[articulation] | FACTOR_REQUEST
    rows.drive[dof] = wp.vec4(K - passive, passive, limit, u0)
    limited = limited_drive_force(u0, limit)
    if limited != u0:
        rows.row_dv[row] = rows.v_out[dof] - rows.v_hat[dof]
    return limited


STATE_AUX = FieldGroup(
    Field("joint_qdd", wp.float32, ("dof",), owner="factor"),
    Field("joint_tau", wp.float32, ("dof",), fill=None, owner="factor"),
    Field("joint_S_s", wp.spatial_vector, ("dof",), fill=None, owner="factor"),
    Field("body_q_com", wp.transform, ("body",), fill=None, owner="factor"),
    Field("body_I_s", wp.spatial_matrix, ("body",), fill=None, owner="factor"),
    Field("body_v_s", wp.spatial_vector, ("body",), fill=None, owner="factor"),
    Field("body_a_s", wp.spatial_vector, ("body",), fill=None, owner="factor"),
    # Without a memset: the first step may run inside a CUDA graph capture, and a captured memset would erase the
    # cached body forces on every replay while the FK/ID cache stays valid. Stage 1 writes body_f_s before reading it.
    Field("body_f_s", wp.spatial_vector, ("body",), fill=None, owner="factor"),
)
"""The augmented state: the step's derived dynamics (maximal coordinates), allocated on the tree factor."""

# ---------------------------------------------------------------------------------------------------------------------
# Size groups
# ---------------------------------------------------------------------------------------------------------------------

SIZE_FACTORS = FieldGroup(
    Field("L", wp.float32, ("factor_art", "factor_size", "factor_size")),
    Field("T", wp.float32, ("trial_art", "trial_size", "trial_size")),
    Field("R", wp.float32, ("art", "size")),
)
"""One size group's mass factor, its trial factor and its armature (the tree factor's); compact and sparse sizes keep
(1, 1, 1) placeholders of the factors, and so does the trial factor of a size whose articulations hold no drive. The
step reads ``L``; ``T`` keeps every drive's ``K`` while ``L`` holds a decided set (:data:`AUGMENTED_DRIVE_ROWS`)."""

SIZE_JACOBIAN = FieldGroup(Field("J", wp.float32, ("response_art", "response_row", "response_size")))
"""One size group's Jacobian. The matrix-free sweeps' producers write the packed rows instead (:mod:`.packed_rows`):
their Jacobian is a (1, 1, 1) placeholder."""

# ---------------------------------------------------------------------------------------------------------------------
# Dense rows
# ---------------------------------------------------------------------------------------------------------------------

DENSE_ROWS = FieldGroup(
    Field("rhs", wp.float32, ("world", "row"), owner="placement"),
    Field("row_incident", wp.float32, ("world", "row"), owner="placement"),
    Field("impulses", wp.float32, ("world", "row"), owner="placement"),
    Field("diag", wp.float32, ("world", "row"), owner="placement"),
    Field("row_type", wp.int32, ("world", "row"), owner="placement"),
    Field("row_parent", wp.int32, ("world", "row"), fill=-1, owner="placement"),
    Field("row_mu", wp.float32, ("world", "row"), owner="placement"),
    Field("row_restitution", wp.float32, ("world", "row"), owner="placement"),
    Field("row_beta", wp.float32, ("world", "row"), owner="placement"),
    Field("row_cfm", wp.float32, ("world", "row"), owner="placement"),
    Field("phi", wp.float32, ("world", "row"), owner="placement"),
    Field("target_velocity", wp.float32, ("world", "row"), owner="placement"),
    Field("constraint_count", wp.int32, ("world",), owner="placement"),
    Field("row_sides", wp.vec2i, ("world", "row"), owner="placement"),
)
"""Every world's dense rows: the system and the row records (with each row's sides). ``row_incident`` is each row's
incident velocity ``J v_hat``, written where its ``J`` is in registers: its producer, or the rows' pass
(:func:`.packed_rows.pack_kernel`)."""

DENSE_ROW_WEIGHTS = FieldGroup(Field("row_w", wp.float32, ("world", "row"), fill=1.0, owner="placement"))
"""Per-row weights of the regularized row law: the relative option's (speculative and restitution rows keep the rigid
law) or a material contact's ``d / (d + eps)``."""

DENSE_ROW_EPS = FieldGroup(Field("row_eps", wp.float32, ("world", "row"), owner="placement"))
"""Per-row compliance of material contact normals (``contact_compliance``); zero elsewhere."""


DENSE_WARMSTART = FieldGroup(
    Field("ws_prev_dense_impulses", wp.float32, ("world", "row"), owner="placement"),
    Field("ws_prev_dense_row_type", wp.int32, ("world", "row"), fill=-1, owner="placement"),
    Field("ws_prev_dense_row_parent", wp.int32, ("world", "row"), fill=-1, owner="placement"),
)
"""Identity-matched warm-start carry of the dense rows."""

FUSED_DIAGONAL_LIMITS = FieldGroup(
    Field("fused_diagonal_limit_active", wp.int32, ("world", "world_dof"), owner="placement"),
    Field("fused_diagonal_limit_lower_rhs", wp.float32, ("world", "world_dof"), owner="placement"),
    Field("fused_diagonal_limit_upper_rhs", wp.float32, ("world", "world_dof"), owner="placement"),
    Field("fused_diagonal_limit_lower_lambda", wp.float32, ("world", "world_dof"), owner="placement"),
    Field("fused_diagonal_limit_upper_lambda", wp.float32, ("world", "world_dof"), owner="placement"),
)
"""Position limits fused into the diagonal-mass sweep, outside the dense row system."""

WORLD_ROW_STATE = FieldGroup(
    Field("slot_counter", wp.int32, ("world",), owner="placement"),
    Field("dense_first_rejected_slot", wp.int32, ("world",), fill=ROW_SLOT_UNBOUNDED, owner="placement"),
    # Per-world dense row-family boundaries, snapshotted from slot_counter each step. The dense slot layout is
    # [mimic][joint-limit][joint-vel-limit][contact/friction], so [world, 0] ends the mimic and position-limit
    # prefix, [world, 1] the velocity-limit segment, and contacts/friction occupy [bounds[world, 1], m_dense).
    Field("dense_phase_bounds", wp.int32, ("world", 2), owner="placement"),
    # First matrix-free slot past the contact/friction rows (start of the velocity-limit tail).
    Field("mf_contact_rows_end", wp.int32, ("world",), owner="placement"),
)
"""Per-world dense row reservation: the slot counter, the first rejected slot and the row-family boundaries."""

CONTACT_RUNS = FieldGroup(
    Field("contact_run_first", wp.int32, ("world", "run_block"), fill=2**31 - 1, owner="placement"),
    Field("contact_run_last", wp.int32, ("world", "run_block"), fill=-1, owner="placement"),
    Field("contact_run_rows", wp.vec2i, ("world", "run_block"), owner="placement"),
)
"""Contact runs of each world and shape block under sorted contacts (``record_world_contact_runs``)."""

VELOCITY_LIMIT_SLOTS = FieldGroup(
    Field("velocity_limit_slot", wp.int32, ("limit_row",), fill=-1, owner="laws"),
    Field("velocity_limit_sign", wp.float32, ("limit_row",), owner="laws"),
)
"""Two unilateral joint velocity-limit rows per DOF (lower ``+e_i``, upper ``-e_i``): their slots and signs."""

PREELIMINATION = FieldGroup(
    Field("_preelim_slots", wp.int32, ("preelim_row",), fill=-1, owner="factor"),
    Field("_preelim_nB", wp.int32, ("preelim",), owner="factor"),
)
"""The tree factor's bilateral blocks: per block, the model's largest block of rows."""

PREELIMINATION_PLAN = FieldGroup(
    Field("_preelim_world", wp.int32, ("preelim",), owner="factor"),
    Field("_preelim_start", wp.int32, ("preelim_bound",), owner="factor"),
    Field("_preelim_entry", wp.int32, ("preelim_entry",), owner="factor"),
)
"""Each block's world and constraints (CSR): a mimic ``m`` as ``m``, a link ``k`` as ``-k - 1``."""

Preelim = field_group.struct(
    "Preelim",
    (
        Field("art_to_preelim", wp.int32, ("articulation",), owner="factor", attr="_preelim_art_to_idx"),
        *(
            dataclasses.replace(field, name=field.name.removeprefix("_preelim_"), attr=field.name)
            for group in (PREELIMINATION, PREELIMINATION_PLAN)
            for field in group
        ),
    ),
    __name__,
)
"""The bilateral elimination of :data:`PREELIMINATION` and :data:`PREELIMINATION_PLAN`: each articulation's block
index, the block's kept rows and their count, and the block's world and constraints."""

SPARSE_FACTOR = FieldGroup(
    Field("sparse_Linv", wp.float32, ("group", "nonzero"), owner="factor"),
    Field("sparse_Linv_trial", wp.float32, ("trial_group", "nonzero"), owner="factor"),
    Field("sparse_mass_matrix_status", wp.int32, ("group",), owner="factor"),
)
"""The native sparse factor's inverse entries, its trial inverse (every drive's ``K``, kept while the step's holds a
decided set; one placeholder group without drives) and its status, per articulation of the factored size."""

SPARSE_ROWS = FieldGroup(
    Field("sparse_row_dof", wp.int32, ("index_world", "index_row", "index_support"), fill=None, owner="placement"),
    Field("sparse_row_factor", wp.float32, ("world", "row", "support"), fill=None, owner="placement"),
    Field("sparse_factor_velocity_delta", wp.float32, ("world", "world_dof"), fill=None, owner="placement"),
)
"""Each sparse row's factor coordinates (unless the support table holds them) and factor entries, and each world's
factor velocity change."""

SPARSE_CONTACT_RECORDS = FieldGroup(
    Field("sparse_contact_records", wp.float32, ("world", "triplet", "record"), owner="placement")
)
"""The single-stage transaction's per-triplet block records (``sparse_contact.CONTACT_RECORD_FLOATS``)."""

# ---------------------------------------------------------------------------------------------------------------------
# Matrix-free rows
# ---------------------------------------------------------------------------------------------------------------------

MF_COUNTERS = FieldGroup(
    Field("mf_constraint_count", wp.int32, ("world",), owner="placement"),
    Field("mf_slot_counter", wp.int32, ("world",), owner="placement"),
    Field("mf_first_rejected_slot", wp.int32, ("world",), fill=ROW_SLOT_UNBOUNDED, owner="placement"),
)
"""Per-world matrix-free row counts, reservation counter and first rejected slot."""

MF_ROWS = FieldGroup(
    Field("mf_body_a", wp.int32, ("world", "mf_row"), owner="placement"),
    Field("mf_body_b", wp.int32, ("world", "mf_row"), owner="placement"),
    Field("mf_Z_a", wp.float32, ("world", "mf_row", 6), owner="placement"),
    Field("mf_Z_b", wp.float32, ("world", "mf_row", 6), owner="placement"),
    Field("mf_rhs", wp.float32, ("world", "mf_row"), owner="placement"),
    Field("mf_impulses", wp.float32, ("world", "mf_row"), owner="placement"),
    Field("mf_eff_mass_inv", wp.float32, ("world", "mf_row"), owner="placement"),
    Field("mf_row_type", wp.int32, ("world", "mf_row"), owner="placement"),
    Field("mf_row_parent", wp.int32, ("world", "mf_row"), fill=-1, owner="placement"),
    Field("mf_row_mu", wp.float32, ("world", "mf_row"), owner="placement"),
    Field("mf_phi", wp.float32, ("world", "mf_row"), owner="placement"),
    Field("mf_row_restitution", wp.float32, ("world", "mf_row"), owner="placement"),
    Field("mf_row_incident", wp.float32, ("world", "mf_row"), owner="placement"),
    Field("mf_dof_a", wp.int32, ("world", "mf_row"), owner="placement"),
    Field("mf_dof_b", wp.int32, ("world", "mf_row"), owner="placement"),
    Field("mf_meta_packed", wp.int32, ("world", "mf_meta"), owner="placement"),
)
"""Every world's matrix-free (free-body) rows. Each side's six entries hold the row's Jacobian ``J`` as the writers
store it and its whitened form ``Z = L^-1 J^T`` once the rows' build has run (``L`` the body's held factor); a
row keeps its incident velocity ``J v_hat``. Packed metadata: four ints per row (dof pair, inverse effective mass, the
residual at zero velocity change, type and parent)."""

MF_ROW_WEIGHTS = FieldGroup(Field("mf_row_w", wp.float32, ("world", "mf_row"), fill=1.0, owner="placement"))

MF_ROW_EPS = FieldGroup(Field("mf_row_eps", wp.float32, ("world", "mf_row"), owner="placement"))

MF_WARMSTART = FieldGroup(
    Field("ws_prev_mf_impulses", wp.float32, ("world", "mf_row"), owner="placement"),
    Field("ws_prev_mf_row_type", wp.int32, ("world", "mf_row"), fill=-1, owner="placement"),
    Field("ws_prev_mf_row_parent", wp.int32, ("world", "mf_row"), fill=-1, owner="placement"),
)
"""Matrix-free warm-start carry: last step's converged impulses and their row types."""

# ---------------------------------------------------------------------------------------------------------------------
# Capacity telemetry
# ---------------------------------------------------------------------------------------------------------------------

ROW_WATERMARKS = FieldGroup(
    *(
        Field(name, wp.int32, (1,), owner="placement")
        for name in (
            "row_watermark_dense",
            "row_watermark_mf",
            "row_watermark_contact",
            "row_watermark_dense_raw",
            "row_watermark_mf_raw",
            "row_dropped_dense_high_water",
            "row_dropped_mf_high_water",
            "row_overflow_dense_excess",
            "row_overflow_mf_excess",
            "row_overflow_dense_world_steps",
            "row_overflow_mf_world_steps",
        )
    )
)
"""Whole-run row high-water telemetry for explicit readback (``row_watermark``)."""

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

LINK_PLAN = FieldGroup(
    Field("_link_primary", wp.int32, ("link",), owner="laws"),
    Field("_link_shadow", wp.int32, ("link",), owner="laws"),
    Field("_link_world", wp.int32, ("link",), owner="laws"),
    Field("_link_enabled", wp.int32, ("link",), owner="laws"),
)
"""Each link's two pieces (:mod:`.body_cut`), world and enable bit."""

LINK_SLOTS = FieldGroup(Field("link_slot", wp.int32, ("link",), fill=-1, owner="laws"))

Links = field_group.struct(
    "Links",
    (
        dataclasses.replace(LINK_SLOTS["link_slot"], name="slot", attr="link_slot"),
        *(dataclasses.replace(field, name=field.name.removeprefix("_link_"), attr=field.name) for field in LINK_PLAN),
    ),
    __name__,
)
"""The links' tables (:data:`LINK_PLAN`) and their dense slots."""

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

# ---------------------------------------------------------------------------------------------------------------------
# Sparse-diagonal contact schedule
# ---------------------------------------------------------------------------------------------------------------------

SPARSE_DIAGONAL_ROWS = FieldGroup(
    Field("sparse_diagonal_row_dof", wp.int32, ("world", "row", 2), fill=-1, owner="placement"),
    Field("sparse_diagonal_row_z", wp.float32, ("world", "row", 2), owner="placement"),
)
"""Each sparse-diagonal row's two response coordinates and their whitened entries ``J / sqrt(m)``."""

SPARSE_CONTACT_GROUPS = FieldGroup(
    Field("sparse_contact_group_count", wp.int32, ("world",), owner="placement"),
    Field("sparse_contact_group_heads", wp.int32, ("world", "world_dof"), fill=None, owner="placement"),
)
"""Independent contact groups per world (triples or units), by head row."""

SPARSE_CONTACT_SERIAL = FieldGroup(
    Field("sparse_contact_serial_count", wp.int32, ("world",), owner="placement"),
    Field("sparse_contact_serial_normals", wp.int32, ("world", "serial"), fill=None, owner="placement"),
)
"""Contacts solved serially after the speculative batches, by normal row."""

# ---------------------------------------------------------------------------------------------------------------------
# Capacity status
# ---------------------------------------------------------------------------------------------------------------------

CAPACITY_STATUS = FieldGroup(
    Field("constraint_capacity_status", wp.int32, (2,), owner="placement"),
    Field("constraint_overflow", wp.bool, ("world",), owner="placement"),
    Field("row_dropped_all", wp.int32, (2, "status_world"), owner="placement"),
)
"""Sticky capacity flags of the dense and matrix-free rows, the per-world overflow latch and dropped rows."""

OVERFLOW_WARNING = FieldGroup(Field("row_overflow_warning_emitted", wp.int32, (2,), owner="placement"))
"""The one-shot overflow warning latch of each row family."""


LOCAL_OWNERS = (
    Field("primary_articulation", wp.int32, ("world",), owner="sweep", attr="local_primary_articulation"),
    Field("pair_articulation", wp.int32, ("world",), owner="sweep", attr="local_pair_articulation"),
    Field("residual_pair_articulation", wp.int32, ("world",), owner="sweep", attr="local_residual_pair_articulation"),
    Field("max_rows", wp.int32, owner="sweep", attr="local_solve_max_rows"),
    Field("residual_max_rows", wp.int32, owner="sweep", attr="local_residual_max_rows"),
    Field("residual_mf_max_rows", wp.int32, owner="sweep", attr="local_residual_mf_max_rows"),
    Field("owner", wp.int32, ("world",), attr="local_solve_owner", owner="placement"),
    Field("general_world_count", wp.int32, (1,), owner="sweep", attr="local_general_world_count"),
    Field("general_worlds", wp.int32, ("world",), owner="sweep", attr="local_general_worlds"),
)
"""Each world's local-solve candidates (its primary articulation and paired articulations), the local row limits,
and the per-step owner of each world with the compact list of the general solver's worlds."""

LocalOwners = field_group.struct("LocalOwners", LOCAL_OWNERS, __name__)
