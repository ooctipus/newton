# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The rows' kernels outside the one writer.

The row types, the rows' rhs by their law (:func:`row_bias`), the link rows of a cut body and the star's fused diagonal
joint limits.
"""

import dataclasses

import warp as wp

from . import field_group
from .device_preludes import reserve
from .field_group import Field, FieldGroup
from .joint_table import JointTable
from .row_laws import (
    LINK_ROWS,
    ROW_CONTACT,
    ROW_LINK,
    contact_restitution_rhs,
    joint_limit_side,
    limit_rhs,
    row_rhs,
)
from .row_records import UNIT_LINK, DenseRows, RowSupport, row_sides, unit_source

# FeatherPGS takes no gradients: its kernels compile no adjoints.
wp.set_module_options({"enable_backward": False})

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

# Keep launch-geometry-specific kernels out of the large general kernel module: Warp compiles one whole module variant
# per block dimension.
_MASS_DYNAMICS_KERNEL_MODULE = wp.Module(f"{__name__}.mass_dynamics")


@wp.kernel(module=_MASS_DYNAMICS_KERNEL_MODULE)
def prepare_fused_diagonal_joint_limits(
    joints: JointTable,
    world_dof_indices: wp.array2d[int],
    max_world_dofs: int,
    fused_limit_dof_mask: wp.array[int],
    limit_q_index: wp.array[int],
    joint_q: wp.array[float],
    activation_gap: float,
    pgs_beta: float,
    dt: float,
    v_hat: wp.array[float],
    # outputs
    active_sides: wp.array2d[int],
    lower_rhs: wp.array2d[float],
    upper_rhs: wp.array2d[float],
):
    """Prepare independent diagonal-articulation limit projections per world coordinate: each active side's
    right-hand side and its incident velocity, ``+v_hat`` for the lower side's row and ``-v_hat`` for the upper's."""
    element = wp.tid()
    world = element // max_world_dofs
    local_dof = element - world * max_world_dofs
    global_dof = world_dof_indices[world, local_dof]

    active = int(0)
    rhs_lower = float(0.0)
    rhs_upper = float(0.0)
    if global_dof >= 0 and fused_limit_dof_mask[global_dof] != 0 and joints.prescribed[global_dof] == 0:
        q_index = limit_q_index[global_dof]
        if q_index >= 0:
            q = joint_q[q_index]
            lower = joints.joint_limit_lower[global_dof]
            upper = joints.joint_limit_upper[global_dof]
            inv_dt = 1.0 / dt
            lower_active, lower_phi, _lower_sign = joint_limit_side(q, lower, upper, 0, activation_gap)
            if lower_active:
                rhs_lower = limit_rhs(lower_phi, pgs_beta, 0.0, inv_dt, 1.0, 1.0) + v_hat[global_dof]
                active |= 1
            upper_active, upper_phi, _upper_sign = joint_limit_side(q, lower, upper, 1, activation_gap)
            if upper_active:
                rhs_upper = limit_rhs(upper_phi, pgs_beta, 0.0, inv_dt, 1.0, 1.0) - v_hat[global_dof]
                active |= 2

    active_sides[world, local_dof] = active
    lower_rhs[world, local_dof] = rhs_lower
    upper_rhs[world, local_dof] = rhs_upper


# =============================================================================
# Link rows
# =============================================================================
# A link ties a body's primary piece to a shadow (:mod:`.body_cut`): six bilateral rows, one law for every joint the
# body was cut at. The pieces share their center of mass, so the rows read ``twist(primary) - twist(shadow)`` there,
# three linear and three angular, with their pose error ``phi``. No row is scaled: the block's pivot decides rank.


@wp.func
def rotation_log(primary: wp.quat, shadow: wp.quat) -> wp.vec3:
    """``log(R_primary R_shadow^T)`` [rad], the rotation that takes the shadow to the primary, the short way round."""
    error = primary * wp.quat_inverse(shadow)
    v = wp.vec3(error[0], error[1], error[2])
    if error[3] < 0.0:
        return -2.0 * wp.atan2(wp.length(v), -error[3]) * wp.normalize(v)
    return 2.0 * wp.atan2(wp.length(v), error[3]) * wp.normalize(v)


@wp.kernel
def allocate_link_slots(
    links: Links,
    max_constraints: int,
    # outputs
    world_slot_counter: wp.array[int],
    first_rejected_slot: wp.array[int],
):
    """Reserve each enabled link's rows, whole or not at all."""
    k = wp.tid()
    links.slot[k] = -1
    if links.enabled[k] != 0:
        links.slot[k] = reserve(world_slot_counter, links.world[k], LINK_ROWS, max_constraints, first_rejected_slot)


@wp.kernel
def write_link_rows(
    links: Links,
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_to_articulation: wp.array[int],
    pgs_beta: float,
    support: RowSupport,
    # outputs
    rows: DenseRows,
):
    """Write each link row's record and unit source (the one writer walks its two pieces' wrenches,
    :func:`.walk_rows.unit_reader`), on the sides of its two pieces' trees (:func:`.row_records.row_sides`), which may
    be one tree: a thread a row, link ``k``'s row ``r`` thread ``LINK_ROWS k + r``, its unit source's index. A linear
    row's error is the pieces' centre-of-mass gap, an angular row's their rotation's log, each the row's axis of it."""
    i = wp.tid()
    k = i // LINK_ROWS
    row = i - LINK_ROWS * k
    base = links.slot[k]
    if base < 0:
        return
    primary, shadow = links.primary[k], links.shadow[k]
    if row >= 3:
        error = rotation_log(wp.transform_get_rotation(body_q[primary]), wp.transform_get_rotation(body_q[shadow]))
    else:
        error = wp.transform_point(body_q[primary], body_com[primary]) - wp.transform_point(
            body_q[shadow], body_com[shadow]
        )
    axis = row % 3
    line = rows.first[links.world[k]]
    slot = base + row
    support.units[line, slot] = unit_source(UNIT_LINK, i)
    rows.row_type[line, slot] = ROW_LINK
    rows.sides[line, slot] = row_sides(body_to_articulation[primary], body_to_articulation[shadow], support.dofs)
    rows.row_parent[line, slot] = -1
    rows.row_mu[line, slot] = 0.0
    rows.row_beta[line, slot] = pgs_beta
    # The row's axis by selects: written through a runtime index it would sit in local memory.
    rows.phi[line, slot] = wp.where(axis == 0, error[0], wp.where(axis == 1, error[1], error[2]))
    rows.target[line, slot] = 0.0


# =============================================================================
# World-Level PGS and Velocity Kernels for Multi-Articulation
# =============================================================================


@wp.struct
class RowBias:
    """The rows' law's step parameters (:func:`row_bias`): the step ``dt``, a positive gap's speculative scale, the
    contacts' weight ``contact_w``, the restitution threshold, whether rows store their regularization weight
    (``weighted``), and whether contacts carry a material (``material``: their writer stored their rhs and compliance).
    """

    dt: float
    speculative_scale: float
    contact_w: float
    restitution_threshold: float
    weighted: int
    material: int


def _row_bias(stored: bool) -> wp.Function:
    """Row ``i``'s rhs by its row law (:func:`.row_laws.row_rhs`, its parameters ``bias``), and a bouncing contact's
    rebound from its incident velocity ``J v_hat`` (:func:`.row_laws.contact_restitution_rhs`, weight 1); a material
    contact (its writer stored its rhs and compliance) keeps its rhs. ``weighted`` rows store their regularization
    weight. Its law words (``phi``, ``beta``, ``target``, ``restitution``) are its records' where ``stored``
    (:data:`row_bias`, which takes none), else given by its writer (:data:`row_bias_words`: a region member's row,
    whose words no later kernel reads, :func:`.row_build.contact_row_writer`)."""

    def bias_row(
        rows: DenseRows,
        bias: RowBias,
        line: int,
        i: int,
        incident: float,
        phi: float = 0.0,
        beta: float = 0.0,
        target: float = 0.0,
        restitution: float = 0.0,
    ):
        kind = rows.row_type[line, i]
        if bias.material != 0 and kind == ROW_CONTACT:
            if rows.row_eps[line, i] != 0.0:
                return
        inv_dt = 1.0 / bias.dt
        if wp.static(stored):
            phi = rows.phi[line, i]
            target = rows.target[line, i]
            beta = rows.row_beta[line, i]
        rhs, row_w = row_rhs(kind, phi, beta, target, inv_dt, 1.0, bias.speculative_scale, 1.0, bias.contact_w)
        if kind == ROW_CONTACT:
            if wp.static(stored):
                restitution = rows.restitution[line, i]
            if restitution > 0.0:
                fires, bounce = contact_restitution_rhs(
                    phi, incident - target, target, restitution, bias.dt, bias.restitution_threshold
                )
                if fires:
                    rhs = bounce
                    row_w = 1.0
        rows.rhs[line, i] = rhs
        if bias.weighted != 0:
            rows.row_w[line, i] = row_w

    bias_row.__name__ = bias_row.__qualname__ = "row_bias" if stored else "row_bias_words"
    return wp.func(bias_row)


row_bias = _row_bias(stored=True)
row_bias_words = _row_bias(stored=False)
