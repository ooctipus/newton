# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The factor's device fields, declared once by group (see :mod:`..field_group`): the step's mass-update mask, the
bodies' inertia records and the articulations' origins, the augmented drive rows (:func:`hold_drive_row`), the
augmented state, and the bilateral blocks' elimination (:data:`Preelim`)."""

import dataclasses

import warp as wp

from .. import field_group
from ..field_group import Field, FieldGroup
from ..spatial_laws import FACTOR_REQUEST, drive_force, drive_stiffness, inertia_record, limited_drive_force

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
    Field("aug_row_dv", wp.float32, ("drive_row",), owner="factor"),
    Field("aug_drive", wp.vec4, ("dof",), owner="factor"),
)
"""Augmented drive rows, ``max_dofs`` per articulation: each row's ``K``, which the factor holds; the ``K`` the saturation
decision left it (``aug_row_Kd``, the passive damper's on a saturated drive, whose ``K - Kd`` the border releases,
:func:`.tree_kinetic_state.border_solve`); the ``H_ii`` (armature in, ``K`` out) last formed where the tree's
representation forms its mass diagonal; what last step's rows did to its coordinate; per coordinate, what the step's
saturation decision reads on every drive (:data:`.tree_kinetic_state.SATURATE_DRIVES`): ``aug_drive``, the drive's own ``K``, the passive damper's, the effort limit
and the unlimited ``u0``, one read of one record."""

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
    A ``K`` the factor does not hold (new gains, the first step) requests the articulation's factor refresh; a
    saturation never does, since the border releases its ``K`` from the factor's walk (``aug_row_Kd``). Only a drive
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


@wp.func
def drive_terms(
    row: int,
    dof: int,
    q_index: int,
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    dt: float,
    drive_target_index_by_dof: wp.array[int],
    joint_target_ke: wp.array[float],
    joint_target_kd: wp.array[float],
    joint_target_pos: wp.array[float],
    joint_target_vel: wp.array[float],
    damping: wp.array[float],
):
    """Explicit force ``u0`` and implicit stiffness ``K`` of a coordinate's augmented drive on ``row``
    (:func:`.spatial_laws.drive_stiffness`, :func:`.spatial_laws.drive_force`): a coordinate without a row or with a
    nonpositive ``K`` holds no drive, ``(0, 0)``, and reads neither its coordinate ``joint_q[q_index]`` nor its target:
    one law for every drive (the factor's preparation, :func:`.tree_factor.prepare_augmented_drives`, and the
    one-coordinate class's predict, :mod:`.coordinate_trees`)."""
    u0 = float(0.0)
    stiffness = float(0.0)
    if row >= 0:
        ke = joint_target_ke[dof]
        kd = joint_target_kd[dof]
        K = drive_stiffness(ke, kd, damping[dof], dt)
        if K > 0.0:
            u0 = drive_force(
                ke,
                kd,
                joint_q[q_index],
                joint_qd[dof],
                joint_target_pos[drive_target_index_by_dof[dof]],
                joint_target_vel[dof],
                dt,
            )
            stiffness = K
    return u0, stiffness


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

PREELIMINATION = FieldGroup(
    Field("_preelim_slots", wp.int32, ("preelim_row",), fill=-1, owner="factor"),
    Field("_preelim_nB", wp.int32, ("preelim",), owner="factor"),
)
"""The tree factor's bilateral blocks: per block, the model's largest block of rows."""

PREELIMINATION_PLAN = FieldGroup(
    Field("_preelim_world", wp.int32, ("preelim",), owner="factor"),
    Field("_preelim_start", wp.int32, ("preelim_bound",), owner="factor"),
    Field("_preelim_entry", wp.int32, ("preelim_entry",), owner="factor"),
    Field("_preelim_world_start", wp.int32, ("preelim_world_bound",), owner="factor"),
)
"""Each block's world and constraints (CSR): a mimic ``m`` as ``m``, a link ``k`` as ``-k - 1``; each world's blocks
(CSR, blocks ordered by world)."""

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
