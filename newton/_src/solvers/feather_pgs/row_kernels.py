# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The rows' kernels outside the one writer.

The row types, the rows' rhs by their law (:func:`row_bias`), the link rows of a cut body, the free rigid
bodies' speed rows' slots, and the star's fused diagonal joint limits.
"""

import dataclasses

import warp as wp

from ...sim import BodyFlags
from . import field_group
from .device_preludes import reserve
from .field_group import Field, FieldGroup
from .joint_table import FreeBodies, JointTable
from .row_laws import LINK_ROWS, contact_restitution_rhs, joint_limit_side, limit_rhs, row_rhs, velocity_limit_active
from .row_records import UNIT_LINK, DenseRows, RowSupport, row_sides, unit_source

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


PGS_CONSTRAINT_TYPE_CONTACT = 0
# A free rigid body's speed row: it bounds the magnitude of the body's linear or angular velocity (type 1 was the
# PhysX drive row; drives fold into the mass matrix).
PGS_CONSTRAINT_TYPE_SPEED = 1
PGS_CONSTRAINT_TYPE_FRICTION = 2
PGS_CONSTRAINT_TYPE_JOINT_LIMIT = 3
# Joint velocity-limit row. Mirrors the PhysX per-DOF velocity clamp (see
# ``notes/investigations/velocity-spike/physx-deep-dive.md`` §4, math
# appendix). Finite limits create two unilateral rows every step: one lower
# bound row and one upper bound row. No Baumgarte / ERP bias.
PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT = 4
# Mimic (joint coupling) row: the bilateral equality
# ``q_follower - coef1 * q_leader - coef0 = 0`` between two 1-DoF joints of the
# same articulation, from ``Model.constraint_mimic_*``. Jacobian has two
# entries (+1 on the follower DOF, -coef1 on the leader DOF); lambda is
# unbounded; drift is removed with the standard Baumgarte ``pgs_beta`` bias.
PGS_CONSTRAINT_TYPE_MIMIC = 5
# Link row: one of the six bilateral rows that tie a body's shadow to its primary piece (:mod:`.body_cut`), unbounded
# lambda, Baumgarte drift correction.
PGS_CONSTRAINT_TYPE_LINK = 6
PGS_CONSTRAINT_TYPE_TORSION = 7
PGS_CONSTRAINT_TYPE_COUNT = 8

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
    """Write each link's rows' records and unit sources (the one writer walks their two pieces' wrenches,
    :func:`.walk_rows.unit_reader`), on the sides of its two pieces' trees (:func:`.row_records.row_sides`), which may
    be one tree."""
    k = wp.tid()
    base = links.slot[k]
    if base < 0:
        return
    primary, shadow = links.primary[k], links.shadow[k]
    point_p = wp.transform_point(body_q[primary], body_com[primary])
    point_s = wp.transform_point(body_q[shadow], body_com[shadow])
    gap = point_p - point_s
    turn = rotation_log(wp.transform_get_rotation(body_q[primary]), wp.transform_get_rotation(body_q[shadow]))
    art_p, art_s = body_to_articulation[primary], body_to_articulation[shadow]
    sides = row_sides(art_p, art_s, support.dofs)
    world = links.world[k]
    line = rows.first[world]
    for row in range(LINK_ROWS):
        slot = base + row
        axis = row % 3
        angular = row >= 3
        support.units[line, slot] = unit_source(UNIT_LINK, LINK_ROWS * k + row)
        rows.row_type[line, slot] = PGS_CONSTRAINT_TYPE_LINK
        rows.sides[line, slot] = sides
        rows.row_parent[line, slot] = -1
        rows.row_mu[line, slot] = 0.0
        rows.row_beta[line, slot] = pgs_beta
        rows.phi[line, slot] = turn[axis] if angular else gap[axis]
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


@wp.func
def row_bias(rows: DenseRows, bias: RowBias, line: int, i: int, incident: float):
    """Row ``i``'s rhs by its row law (:func:`.row_laws.row_rhs`, its parameters ``bias``) from its records, and a
    bouncing contact's rebound from its incident velocity ``J v_hat`` (:func:`.row_laws.contact_restitution_rhs`,
    weight 1); a material contact (its writer stored its rhs and compliance) keeps its rhs. ``weighted`` rows store
    their regularization weight."""
    kind = rows.row_type[line, i]
    if bias.material != 0 and kind == PGS_CONSTRAINT_TYPE_CONTACT:
        if rows.row_eps[line, i] > 0.0:
            return
    inv_dt = 1.0 / bias.dt
    phi = rows.phi[line, i]
    target = rows.target[line, i]
    rhs, row_w = row_rhs(
        kind, phi, rows.row_beta[line, i], target, inv_dt, 1.0, bias.speculative_scale, 1.0, bias.contact_w
    )
    if kind == PGS_CONSTRAINT_TYPE_CONTACT:
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


RIGID_SPEED_ROWS = wp.constant(2)
"""Speed rows of a limited free rigid body: the first bounds its linear speed, the second its angular speed."""


@wp.kernel
def allocate_rigid_velocity_limit_slots(
    bodies: FreeBodies,
    body_flags: wp.array[wp.int32],
    articulation_root_dof_start: wp.array[int],
    joint_qd: wp.array[float],
    velocity_limit_activation_fraction: float,
    mf_max_constraints: int,
    resolved_worlds: wp.array[int],
    # outputs
    rigid_velocity_limit_slot: wp.array[int],
    mf_slot_counter: wp.array[int],
    mf_first_rejected_slot: wp.array[int],
):
    """Allocate one MF speed row per finitely limited speed of a free rigid body.

    Free-rigid generalized velocity is stored as six root DOFs, the center of
    mass velocity and the angular velocity: ``[v, omega]``. PhysX's
    ``maxLinearVelocity`` and ``maxAngularVelocity`` bound their magnitudes,
    ``|v| <= limit`` and ``|omega| <= limit``, so each finite limit owns one
    speed row (:func:`.walk_rows.populate_rigid_velocity_limit_rows`) that the sweeps
    project onto that ball, along the current velocity.

    ``velocity_limit_activation_fraction`` proximity-gates the allocation,
    as the joint velocity limits (:func:`.prefix_rows.velocity_limits`): a positive fraction
    reserves a row only when the speed read from ``joint_qd`` at the free
    root's DOFs satisfies ``|v| >= fraction * limit``. A fraction of ``0.0``
    reserves every finitely limited speed's row without reading ``joint_qd``.
    """
    candidate = wp.tid()
    body = bodies.free_rigid_body_indices[candidate]
    base = RIGID_SPEED_ROWS * candidate

    for row in range(RIGID_SPEED_ROWS):
        rigid_velocity_limit_slot[base + row] = -1

    art = bodies.body_to_articulation[body]
    if art < 0:
        return
    if bodies.is_free_rigid[art] == 0:
        return
    if (body_flags[body] & BodyFlags.KINEMATIC) != 0:
        return

    world = bodies.art_to_world[art]
    if world < 0:
        return
    if world < resolved_worlds.shape[0] and resolved_worlds[world] != 0:
        return

    for speed_axis in range(RIGID_SPEED_ROWS):
        limit = bodies.max_linear_velocity[body]
        if speed_axis == 1:
            limit = bodies.max_angular_velocity[body]
        # An infinite speed limit is no limit; otherwise the velocity-limit activation law, whose proximity
        # gate reserves the row only when the speed reaches ``fraction * limit``.
        if not wp.isfinite(limit):
            continue
        root_dof = articulation_root_dof_start[art] + 3 * speed_axis
        speed = wp.length(wp.vec3(joint_qd[root_dof], joint_qd[root_dof + 1], joint_qd[root_dof + 2]))
        if not velocity_limit_active(limit, speed, velocity_limit_activation_fraction):
            continue

        rigid_velocity_limit_slot[base + speed_axis] = reserve(
            mf_slot_counter, world, 1, mf_max_constraints, mf_first_rejected_slot
        )
