# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Worlds whose every candidate row separates at the predictor need no rows.

Before any row is allocated, this classifier evaluates each activated joint
limit and each raw contact of a world at the unconstrained predictor
velocity. When every residual is strictly positive by a margin, zero impulse
is the exact solution of the world's step: no row would receive an impulse,
so the allocators skip the world, its rows are never built and the response
and the sweeps have nothing to do there. Any ambiguous, unsupported or
malformed row sends the world to the ordinary complete path. The classifier
owns no rows, impulses, factors or publication, and it names no robot and
no coordinate count: it reads the world's DOF list and each body's
response mask.
"""

import math
from types import SimpleNamespace

import numpy as np
import warp as wp

from ...core import memory
from ...core.execution import Work, shape
from . import field_group
from .contact_laws import contact_points, contact_responds, contact_restitution, jacobian_entry
from .field_group import Field, FieldGroup, SolverArrays
from .joint_table import JOINTS
from .row_laws import (
    contact_capped_beta,
    contact_depenetration_cap,
    contact_restitution_rhs,
    contact_rhs,
    joint_limit_side,
    limit_rhs,
)
from .step_plan import first_failed

LANES = 32


RESOLVED_WORLD_INPUT = (
    Field("world_dof_indices", wp.int32, ("world", "world_dof"), owner="forest"),
    Field("world_dof_count", wp.int32, ("world",), owner="forest"),
    Field("limit_q_index", wp.int32, ("dof",), owner="laws", attr="joint_limit_q_index"),
    Field("lower", wp.float32, ("dof",), owner="model", attr="joint_limit_lower"),
    Field("upper", wp.float32, ("dof",), owner="model", attr="joint_limit_upper"),
    FieldGroup(*JOINTS)["prescribed"],
    Field("q", wp.float32, ("coord",), owner="state_in", attr="joint_q"),
    Field("v_hat", wp.float32, ("dof",)),
    Field("body_to_articulation", wp.int32, ("body",), owner="forest"),
    Field("art_to_world", wp.int32, ("articulation",), owner="forest"),
    Field("articulation_dof_start", wp.int32, ("articulation",), owner="forest"),
    Field("articulation_response_dof_count", wp.int32, ("articulation",), owner="forest"),
    Field("prescribed_articulation", wp.int32, ("articulation",), owner="forest"),
    Field("is_free_rigid", wp.int32, ("articulation",), owner="forest"),
    Field("body_flags", wp.int32, ("body",), owner="model"),
    Field("body_has_response_dofs", wp.int32, ("body",), owner="forest"),
    Field("body_response_dof_mask", wp.uint32, ("body",), owner="forest"),
    Field("body_q", wp.transform, ("body",), owner="state_in"),
    Field("body_v_s", wp.spatial_vector, ("body",), owner="state_aug"),
    Field("joint_S_s", wp.spatial_vector, ("dof",), owner="state_aug"),
    Field("articulation_origin", wp.vec3, ("articulation",), owner="factor"),
    Field("max_linear_velocity", wp.float32, ("body",), owner="limits"),
    Field("max_angular_velocity", wp.float32, ("body",), owner="limits"),
    Field("max_depenetration_velocity", wp.float32, ("body",), owner="limits"),
    Field("free_bodies", wp.int32, ("free_body",), owner="classifier"),
    # Predictor body twists published by the held-tree owner (use_published != 0).
    Field("predicted_twist", wp.spatial_vector, ("body",), owner="classifier", attr="twists"),
    Field("use_published", wp.int32, owner="classifier", attr="published"),
    Field("dt", wp.float32, owner="step"),
    Field("beta", wp.float32, owner="options", attr="pgs_beta"),
    Field("speculative_scale", wp.float32, owner="options", attr="contact_speculative_scale"),
    Field("activation_gap", wp.float32, owner="options", attr="joint_limit_activation_gap"),
    Field("restitution_threshold", wp.float32, owner="laws", attr="effective_restitution_velocity_threshold"),
    Field("absolute_margin", wp.float32, owner="classifier"),
    Field("relative_margin", wp.float32, owner="classifier"),
)
"""The classifier's inputs: the solver's maps and limits, the step's state and predictor, its own margins."""

ResolvedWorldInput = field_group.struct("ResolvedWorldInput", RESOLVED_WORLD_INPUT, __name__)

RAW_CONTACTS = (
    Field("count", wp.int32, (1,), owner="contacts", attr="rigid_contact_count"),
    Field("shape0", wp.int32, ("contact",), owner="contacts", attr="rigid_contact_shape0"),
    Field("shape1", wp.int32, ("contact",), owner="contacts", attr="rigid_contact_shape1"),
    Field("point0", wp.vec3, ("contact",), owner="contacts", attr="rigid_contact_point0"),
    Field("point1", wp.vec3, ("contact",), owner="contacts", attr="rigid_contact_point1"),
    Field("normal", wp.vec3, ("contact",), owner="contacts", attr="rigid_contact_normal"),
    Field("margin0", wp.float32, ("contact",), owner="contacts", attr="rigid_contact_margin0"),
    Field("margin1", wp.float32, ("contact",), owner="contacts", attr="rigid_contact_margin1"),
    Field("shape_body", wp.int32, ("shape",), owner="model"),
    Field("shape_mu", wp.float32, ("shape",), owner="laws", attr="shape_material_mu"),
    Field("shape_restitution", wp.float32, ("shape",), owner="laws", attr="shape_material_restitution"),
)
"""The collision pipeline's contacts and the shapes' bodies and materials, before any row exists."""

RawContacts = field_group.struct("RawContacts", RAW_CONTACTS, __name__)


RESOLVED_WORLD_SCRATCH = FieldGroup(
    Field("rejected", wp.int32, ("world",)),
    Field("resolved", wp.int32, ("world",)),
    Field("checked_limits", wp.int32, ("world",)),
    Field("checked_normals", wp.int32, ("world",)),
    Field("minimum_residual", wp.float32, ("world",), fill=None),
    Field("body_twist", wp.spatial_vector, ("body",), fill=None),
    Field("body_valid", wp.int32, ("body",), fill=None),
    Field("global_invalid", wp.int32, (1,)),
)
"""The classifier's per-world verdicts and per-body scratch."""

ResolvedWorldScratch = field_group.struct("ResolvedWorldScratch", RESOLVED_WORLD_SCRATCH, __name__)


@wp.func
def _finite3(v: wp.vec3):
    return wp.isfinite(v[0]) and wp.isfinite(v[1]) and wp.isfinite(v[2])


@wp.func
def _finite6(v: wp.spatial_vector):
    result = True
    for k in range(6):
        result = result and wp.isfinite(v[k])
    return result


@wp.func
def _passes(residual: float, magnitude: float, data: ResolvedWorldInput):
    margin = data.absolute_margin + data.relative_margin * magnitude
    return wp.isfinite(residual) and wp.isfinite(margin) and residual >= margin


@wp.kernel(enable_backward=False)
def reset_world_decisions(enabled: int, data: ResolvedWorldInput, scratch: ResolvedWorldScratch):
    """Reset every decision; a disabled call rejects every world before any check."""
    world = wp.tid()
    scratch.rejected[world] = 1 - wp.min(enabled, 1)
    scratch.resolved[world] = 0
    scratch.checked_limits[world] = 0
    scratch.checked_normals[world] = 0
    scratch.minimum_residual[world] = wp.float32(3.402823466e38)
    if world == 0:
        valid = (
            wp.isfinite(data.dt)
            and data.dt > 0.0
            and wp.isfinite(data.beta)
            and data.beta >= 0.0
            and wp.isfinite(data.speculative_scale)
            and data.speculative_scale >= 0.0
            and not wp.isnan(data.activation_gap)
            and data.activation_gap >= 0.0
            and wp.isfinite(data.restitution_threshold)
            and data.restitution_threshold >= 0.0
            and wp.isfinite(data.absolute_margin)
            and data.absolute_margin >= 0.0
            and wp.isfinite(data.relative_margin)
            and data.relative_margin >= 0.0
        )
        scratch.global_invalid[0] = 1 - wp.int32(valid)


@wp.kernel(enable_backward=False)
def check_world_limits(data: ResolvedWorldInput, scratch: ResolvedWorldScratch):
    """Check every activated signed position limit of a world, one lane per DOF stride."""
    world, lane = wp.tid()
    if scratch.rejected[world] != 0:
        return
    count = data.world_dof_count[world]
    if count < 0 or count > data.world_dof_indices.shape[1]:
        if lane == 0:
            wp.atomic_max(scratch.rejected, world, 1)
        return
    valid = bool(True)
    checked = int(0)
    minimum = wp.float32(3.402823466e38)
    for local in range(lane, count, LANES):
        dof = data.world_dof_indices[world, local]
        if dof < 0 or dof >= data.v_hat.shape[0] or dof >= data.limit_q_index.shape[0]:
            valid = False
            continue
        velocity = data.v_hat[dof]
        if not wp.isfinite(velocity):
            valid = False
            continue
        qi = data.limit_q_index[dof]
        if qi < 0 or data.prescribed[dof] != 0:
            # No limit, or a prescribed coordinate: no limit row (joint_table.JOINTS).
            continue
        if qi >= data.q.shape[0] or dof >= data.lower.shape[0] or dof >= data.upper.shape[0]:
            valid = False
            continue
        q = data.q[qi]
        lower = data.lower[dof]
        upper = data.upper[dof]
        if not wp.isfinite(q) or wp.isnan(lower) or wp.isnan(upper) or lower > upper:
            valid = False
            continue
        for side in range(2):
            active, phi, sign = joint_limit_side(q, lower, upper, side, data.activation_gap)
            if active:
                checked += 1
                signed_velocity = sign * velocity
                bias = limit_rhs(phi, data.beta, 0.0, 1.0 / data.dt, 1.0, 1.0)
                residual = signed_velocity + bias
                minimum = wp.min(minimum, residual)
                valid = valid and _passes(residual, wp.abs(signed_velocity) + wp.abs(bias), data)
    if checked > 0:
        wp.atomic_add(scratch.checked_limits, world, checked)
        wp.atomic_min(scratch.minimum_residual, world, minimum)
    if not valid:
        wp.atomic_max(scratch.rejected, world, 1)


@wp.func
def _within_speed_limits(data: ResolvedWorldInput, body: int, start: int) -> bool:
    """Whether a free body's predicted linear and angular speeds are within its speed limits.

    A limit bounds the magnitude of the velocity (PhysX's ``maxLinearVelocity`` and
    ``maxAngularVelocity``), as the free body's speed rows do. A NaN limit is malformed.
    """
    valid = True
    for speed_row in range(2):
        limit = data.max_linear_velocity[body]
        if speed_row == 1:
            limit = data.max_angular_velocity[body]
        valid = valid and not wp.isnan(limit)
        if wp.isfinite(limit) and limit > 0.0:
            dof = start + 3 * speed_row
            speed = wp.length(wp.vec3(data.v_hat[dof], data.v_hat[dof + 1], data.v_hat[dof + 2]))
            valid = valid and speed <= limit
    return valid


@wp.kernel(enable_backward=False)
def build_predicted_body_twists(data: ResolvedWorldInput, scratch: ResolvedWorldScratch):
    """Contract each body's motion axes against the predictor, never the incoming velocity."""
    body = wp.tid()
    scratch.body_valid[body] = 0
    scratch.body_twist[body] = wp.spatial_vector()
    art = data.body_to_articulation[body]
    if art < 0:
        scratch.body_valid[body] = 1
        return
    if art >= data.art_to_world.shape[0]:
        wp.atomic_max(scratch.global_invalid, 0, 1)
        return
    world = data.art_to_world[art]
    if world < 0 or world >= scratch.resolved.shape[0]:
        wp.atomic_max(scratch.global_invalid, 0, 1)
        return
    if scratch.rejected[world] != 0:
        return
    pose = data.body_q[body]
    rotation = wp.transform_get_rotation(pose)
    valid = _finite3(wp.transform_get_translation(pose)) and _finite3(data.articulation_origin[art])
    norm = float(0.0)
    for k in range(4):
        valid = valid and wp.isfinite(rotation[k])
        norm += rotation[k] * rotation[k]
    valid = valid and wp.isfinite(norm) and norm > 0.0
    twist = wp.spatial_vector()
    responding = data.body_has_response_dofs[body] != 0 and (data.body_flags[body] & 2) == 0
    if data.prescribed_articulation[art] != 0:
        twist = data.body_v_s[body]
    elif responding:
        start = data.articulation_dof_start[art]
        count = data.articulation_response_dof_count[art]
        mask = data.body_response_dof_mask[body]
        valid = valid and count > 0 and count <= 32 and start >= 0
        if count < 32 and count >= 0:
            valid = valid and (mask >> wp.uint32(count)) == wp.uint32(0)
        for local in range(wp.min(count, 32)):
            if (mask & (wp.uint32(1) << wp.uint32(local))) != wp.uint32(0):
                dof = start + local
                if dof >= data.v_hat.shape[0] or dof >= data.joint_S_s.shape[0]:
                    valid = False
                    continue
                axis = data.joint_S_s[dof]
                speed = data.v_hat[dof]
                valid = valid and _finite6(axis) and wp.isfinite(speed)
                twist += axis * speed
        if data.is_free_rigid[art] != 0:
            valid = valid and count == 6 and body < data.max_linear_velocity.shape[0]
            valid = valid and body < data.max_angular_velocity.shape[0]
            if valid:
                valid = _within_speed_limits(data, body, start)
    elif data.articulation_response_dof_count[art] > 0 and (data.body_flags[body] & 2) != 0:
        # Partial prescribed membership needs a different source target model.
        valid = False
    valid = valid and _finite6(twist)
    if valid:
        scratch.body_twist[body] = twist
        scratch.body_valid[body] = 1
    else:
        wp.atomic_max(scratch.rejected, world, 1)


@wp.kernel(enable_backward=False)
def check_free_bodies(data: ResolvedWorldInput, scratch: ResolvedWorldScratch):
    """A free rigid body's predictor must sit inside its velocity caps, or its world keeps its rows."""
    index = wp.tid()
    body = data.free_bodies[index]
    if body < 0 or body >= data.body_to_articulation.shape[0]:
        wp.atomic_max(scratch.global_invalid, 0, 1)
        return
    art = data.body_to_articulation[body]
    if art < 0 or art >= data.art_to_world.shape[0]:
        wp.atomic_max(scratch.global_invalid, 0, 1)
        return
    world = data.art_to_world[art]
    if world < 0 or world >= scratch.resolved.shape[0]:
        wp.atomic_max(scratch.global_invalid, 0, 1)
        return
    if scratch.rejected[world] != 0 or data.prescribed_articulation[art] != 0:
        return
    if data.body_has_response_dofs[body] == 0 or (data.body_flags[body] & 2) != 0:
        return
    start = data.articulation_dof_start[art]
    count = data.articulation_response_dof_count[art]
    valid = count == 6 and start >= 0 and start + 6 <= data.v_hat.shape[0]
    valid = valid and body < data.max_linear_velocity.shape[0] and body < data.max_angular_velocity.shape[0]
    if valid:
        valid = _within_speed_limits(data, body, start)
    if not valid:
        wp.atomic_max(scratch.rejected, world, 1)


@wp.func
def _body_twist(body: int, art: int, data: ResolvedWorldInput, scratch: ResolvedWorldScratch):
    """The endpoint body's predicted twist about its articulation origin.

    Published by the held-tree predictor when every responding articulation
    is tree-owned; otherwise the classifier's own body pass wrote it.
    """
    if data.use_published == 0:
        return scratch.body_twist[body]
    if data.prescribed_articulation[art] != 0:
        return data.body_v_s[body]
    if data.body_has_response_dofs[body] != 0 and (data.body_flags[body] & 2) == 0:
        return data.predicted_twist[body]
    return wp.spatial_vector()


@wp.func
def _published_twist_known(body: int, art: int, data: ResolvedWorldInput):
    """Whether :func:`_body_twist` returns the endpoint body's predicted twist on the published path.

    A kinematic body with response DOFs outside a prescribed articulation, such
    as a free body made kinematic after construction or a kinematic root of a
    dynamic tree, moves at its prescribed joint velocity, which the published
    path does not read. The body pass rejects every world that holds such a
    body; the published path rejects every contact that touches one. A
    kinematic body without response DOFs is welded to the world, so its zero
    twist is exact.
    """
    if data.prescribed_articulation[art] != 0 or data.body_has_response_dofs[body] == 0:
        return True
    return (data.body_flags[body] & 2) == 0


@wp.func
def _endpoint_speed(
    body: int,
    art: int,
    mf: bool,
    lever: wp.vec3,
    normal: wp.vec3,
    twist: wp.spatial_vector,
    data: ResolvedWorldInput,
):
    value = float(0.0)
    if body >= 0 and art >= 0:
        if mf and data.prescribed_articulation[art] == 0 and data.articulation_response_dof_count[art] > 0:
            start = data.articulation_dof_start[art]
            twist = wp.spatial_vector(
                data.v_hat[start],
                data.v_hat[start + 1],
                data.v_hat[start + 2],
                data.v_hat[start + 3],
                data.v_hat[start + 4],
                data.v_hat[start + 5],
            )
        # The lever is already relative to the twist's origin (endpoint-relative arithmetic).
        value = jacobian_entry(twist, normal, lever, wp.vec3(0.0))
    return value


@wp.func
def _check_normal(c: int, raw: RawContacts, data: ResolvedWorldInput, scratch: ResolvedWorldScratch):
    shape_a = raw.shape0[c]
    shape_b = raw.shape1[c]
    if shape_a < -1 or shape_b < -1 or shape_a >= raw.shape_body.shape[0] or shape_b >= raw.shape_body.shape[0]:
        wp.atomic_max(scratch.global_invalid, 0, 1)
        return
    body_a = int(-1)
    body_b = int(-1)
    if shape_a >= 0:
        body_a = raw.shape_body[shape_a]
    if shape_b >= 0:
        body_b = raw.shape_body[shape_b]
    if body_a < -1 or body_b < -1 or body_a >= data.body_q.shape[0] or body_b >= data.body_q.shape[0]:
        wp.atomic_max(scratch.global_invalid, 0, 1)
        return
    art_a = int(-1)
    art_b = int(-1)
    if body_a >= 0:
        art_a = data.body_to_articulation[body_a]
    if body_b >= 0:
        art_b = data.body_to_articulation[body_b]
    world = int(-1)
    if art_a >= 0 and art_a < data.art_to_world.shape[0]:
        world = data.art_to_world[art_a]
    if art_b >= 0 and art_b < data.art_to_world.shape[0]:
        other = data.art_to_world[art_b]
        if world >= 0 and world != other:
            wp.atomic_max(scratch.global_invalid, 0, 1)
            return
        world = other
    if art_a >= data.art_to_world.shape[0] or art_b >= data.art_to_world.shape[0]:
        wp.atomic_max(scratch.global_invalid, 0, 1)
        return
    dofs_a = art_a >= 0 and data.articulation_response_dof_count[art_a] > 0
    dofs_b = art_b >= 0 and data.articulation_response_dof_count[art_b] > 0
    responds_a = contact_responds(
        art_a, body_a, data.articulation_response_dof_count, data.body_has_response_dofs, data.body_flags
    )
    responds_b = contact_responds(
        art_b, body_b, data.articulation_response_dof_count, data.body_has_response_dofs, data.body_flags
    )
    if not responds_a and not responds_b:
        return
    if world < 0 or world >= scratch.resolved.shape[0]:
        wp.atomic_max(scratch.global_invalid, 0, 1)
        return
    wp.atomic_add(scratch.checked_normals, world, 1)
    if scratch.rejected[world] != 0:
        return
    valid = True
    twist_a = wp.spatial_vector()
    twist_b = wp.spatial_vector()
    if body_a >= 0 and art_a >= 0:
        twist_a = _body_twist(body_a, art_a, data, scratch)
        if data.use_published != 0:
            valid = valid and _published_twist_known(body_a, art_a, data) and _finite6(twist_a)
        else:
            valid = valid and scratch.body_valid[body_a] != 0
    if body_b >= 0 and art_b >= 0:
        twist_b = _body_twist(body_b, art_b, data, scratch)
        if data.use_published != 0:
            valid = valid and _published_twist_known(body_b, art_b, data) and _finite6(twist_b)
        else:
            valid = valid and scratch.body_valid[body_b] != 0
    n = -raw.normal[c]
    valid = valid and _finite3(n) and wp.dot(n, n) > 0.0
    valid = valid and _finite3(raw.point0[c]) and _finite3(raw.point1[c])
    valid = valid and wp.isfinite(raw.margin0[c]) and wp.isfinite(raw.margin1[c])
    for endpoint in range(2):
        shape = shape_a
        if endpoint == 1:
            shape = shape_b
        if shape >= 0:
            if shape >= raw.shape_mu.shape[0] or shape >= raw.shape_restitution.shape[0]:
                valid = False
            else:
                valid = valid and wp.isfinite(raw.shape_mu[shape]) and raw.shape_mu[shape] >= 0.0
    if not valid:
        wp.atomic_max(scratch.rejected, world, 1)
        return
    # The rows' gap (contact_points); the lever arms, like it, subtract no rounded world points.
    _normal, translation_a, offset_a, translation_b, offset_b, separation, phi = contact_points(
        raw.normal[c], raw.point0[c], raw.point1[c], raw.margin0[c], raw.margin1[c], body_a, body_b, data.body_q
    )
    lever_a = offset_a
    lever_b = offset_b
    if art_a >= 0:
        lever_a += translation_a - data.articulation_origin[art_a]
    if art_b >= 0:
        lever_b += translation_b - data.articulation_origin[art_b]
    lever_a -= 0.5 * separation
    lever_b += 0.5 * separation
    compatible_a = not dofs_a
    compatible_b = not dofs_b
    if art_a >= 0:
        compatible_a = compatible_a or data.is_free_rigid[art_a] != 0
    if art_b >= 0:
        compatible_b = compatible_b or data.is_free_rigid[art_b] != 0
    mf = compatible_a and compatible_b
    speed_a = _endpoint_speed(body_a, art_a, mf, lever_a, n, twist_a, data)
    speed_b = _endpoint_speed(body_b, art_b, mf, lever_b, n, twist_b, data)
    incident = speed_a - speed_b
    # The row law's bias, at the responding bodies' capped beta.
    cap = contact_depenetration_cap(
        wp.where(responds_a, body_a, -1), wp.where(responds_b, body_b, -1), data.max_depenetration_velocity
    )
    beta = contact_capped_beta(phi, data.beta, cap, data.dt)
    bias, _row_w = contact_rhs(phi, beta, 0.0, 1.0 / data.dt, 1.0, data.speculative_scale, 1.0)
    restitution = contact_restitution(shape_a, shape_b, raw.shape_restitution)
    if restitution > 0.0:
        fires, bounce = contact_restitution_rhs(phi, incident, 0.0, restitution, data.dt, data.restitution_threshold)
        if fires:
            bias = bounce
    residual = incident + bias
    wp.atomic_min(scratch.minimum_residual, world, residual)
    valid = _finite3(separation) and _finite3(lever_a) and _finite3(lever_b) and wp.isfinite(phi)
    valid = valid and _passes(residual, wp.abs(speed_a) + wp.abs(speed_b) + wp.abs(bias), data)
    if not valid:
        wp.atomic_max(scratch.rejected, world, 1)


@wp.func
def _free_route_world(c: int, raw: RawContacts, data: ResolvedWorldInput, worlds: int):
    """Return the world of a responding contact the row allocator routes to matrix-free rows, else -1.

    With free rigid bodies in the model, a contact takes matrix-free rows when
    each endpoint is static, has no response coordinates or is a free rigid
    body; it then counts against the matrix-free capacity instead of the dense
    one. Malformed ids return -1: :func:`_check_normal` rejects the call.
    """
    world = int(-1)
    if data.free_bodies.shape[0] == 0:
        return world
    shape_a = raw.shape0[c]
    shape_b = raw.shape1[c]
    if shape_a < -1 or shape_b < -1 or shape_a >= raw.shape_body.shape[0] or shape_b >= raw.shape_body.shape[0]:
        return world
    body_a = int(-1)
    body_b = int(-1)
    if shape_a >= 0:
        body_a = raw.shape_body[shape_a]
    if shape_b >= 0:
        body_b = raw.shape_body[shape_b]
    if body_a < -1 or body_b < -1 or body_a >= data.body_q.shape[0] or body_b >= data.body_q.shape[0]:
        return world
    art_a = int(-1)
    art_b = int(-1)
    if body_a >= 0:
        art_a = data.body_to_articulation[body_a]
    if body_b >= 0:
        art_b = data.body_to_articulation[body_b]
    if art_a >= data.art_to_world.shape[0] or art_b >= data.art_to_world.shape[0]:
        return world
    if art_a >= 0:
        world = data.art_to_world[art_a]
    if art_b >= 0:
        other = data.art_to_world[art_b]
        if world >= 0 and world != other:
            return int(-1)
        world = other
    dofs_a = art_a >= 0 and data.articulation_response_dof_count[art_a] > 0
    dofs_b = art_b >= 0 and data.articulation_response_dof_count[art_b] > 0
    free_a = art_a >= 0 and data.is_free_rigid[art_a] != 0
    free_b = art_b >= 0 and data.is_free_rigid[art_b] != 0
    responds_a = contact_responds(
        art_a, body_a, data.articulation_response_dof_count, data.body_has_response_dofs, data.body_flags
    )
    responds_b = contact_responds(
        art_b, body_b, data.articulation_response_dof_count, data.body_has_response_dofs, data.body_flags
    )
    if (not responds_a and not responds_b) or world < 0 or world >= worlds:
        return int(-1)
    if (dofs_a and not free_a) or (dofs_b and not free_b):
        return int(-1)
    return world


@wp.kernel(enable_backward=False)
def check_raw_contact_normals(
    threads: int,
    raw: RawContacts,
    data: ResolvedWorldInput,
    scratch: ResolvedWorldScratch,
    free_normals: wp.array[int],
):
    """Check the complete active raw prefix, counting each world's contacts on matrix-free rows; invalid ownership
    rejects the call."""
    tid = wp.tid()
    count = raw.count[0]
    capacity = raw.shape0.shape[0]
    if (
        count < 0
        or raw.shape1.shape[0] != capacity
        or raw.point0.shape[0] != capacity
        or raw.point1.shape[0] != capacity
        or raw.normal.shape[0] != capacity
        or raw.margin0.shape[0] != capacity
        or raw.margin1.shape[0] != capacity
    ):
        if tid == 0:
            wp.atomic_max(scratch.global_invalid, 0, 1)
        return
    for c in range(tid, count, threads):
        _check_normal(c, raw, data, scratch)
        world = _free_route_world(c, raw, data, free_normals.shape[0])
        if world >= 0:
            wp.atomic_add(free_normals, world, 1)


@wp.kernel(enable_backward=False)
def finalize_resolved_worlds(
    scratch: ResolvedWorldScratch, free_normals: wp.array[int], dense_rows: int, free_rows: int
):
    """Publish once the limit, body and contact launches finish on one stream, resolving only worlds whose worst-case
    rows fit each route.

    The friction law releases the anchor of every contact the row
    allocator would reserve rows for, where a capacity rejection would keep it.
    Counting every checked limit as one dense row and every checked contact as
    three rows on its route (``free_normals`` on matrix-free rows, the rest on
    dense rows), a world within both capacities has no such rejection, so the
    release matches the complete path contact by contact.
    """
    world = wp.tid()
    value = int(0)
    if scratch.rejected[world] == 0 and scratch.global_invalid[0] == 0:
        free = free_normals[world]
        dense = scratch.checked_limits[world] + 3 * (scratch.checked_normals[world] - free)
        if dense <= dense_rows and 3 * free <= free_rows:
            value = 1
    scratch.resolved[world] = value


@wp.kernel(enable_backward=False)
def accumulate_resolved_rows(scratch: ResolvedWorldScratch, watermark: wp.array[wp.int32]):
    """The most rows a resolved world skipped (``Diagnostics.row_watermark``): its checked limits and three rows a
    checked contact, a normal and its tangent pair (the bound :func:`finalize_resolved_worlds` resolves under)."""
    world = wp.tid()
    if scratch.resolved[world] != 0:
        wp.atomic_max(watermark, 0, scratch.checked_limits[world] + 3 * scratch.checked_normals[world])


def decline(solver) -> str | None:
    """Why the solver's rows do not follow the law the classifier evaluates, or ``None`` when they do.

    The dense route with point contacts, or with persistent patch friction,
    the plain matrix-free interleaved sweep,
    and only unilateral row kinds whose zero-impulse condition is the
    predictor residual: contacts, their friction and joint position limits.
    A patch tangent pair projects on its ring's pooled normal load, which is
    zero in a world whose normals all separate. Bilateral rows (mimic,
    link, pre-elimination), warm starts, regularization and
    velocity limits keep every world on the complete path.
    """
    laws = solver.laws.set
    # No mimic or link rows.
    laws_exact = {"mimic_count": 0, "link_count": 0}
    required = ("world_dof_indices", "world_dof_count", "body_response_dof_mask", "body_has_response_dofs")
    return first_failed(
        (
            (
                "resolved_worlds" in solver._route_pins.get("claims", ("resolved_worlds",)),
                "A test pin drops the resolved-world claim.",
            ),
            # A separating contact carries no load, so its friction and spin rows (contact torsion) carry nothing either:
            # the classifier's normal law decides a world with spin rows as one without.
            (not laws.compliance, "Material contacts are outside the rigid contact law it classifies."),
            (getattr(solver.placement, "keyed_rows", None) is None, "Keyed rows own the contact rows."),
            (
                not laws.warm_start
                and not laws.regularized
                and not laws.velocity_limits
                and solver.classes.case not in ("K1", "K2")
                and solver.factor.block_count == 0
                and all(getattr(solver.laws, name, object()) == value for name, value in laws_exact.items())
                and solver.options.enable_bilateral_preelimination is False
                and (solver.forest.max_world_dofs or 33) <= 32
                and all(getattr(solver.forest, name) is not None for name in required)
                and solver.laws.joint_limit_q_index is not None,
                "The rows are outside the classified law (route, bilateral or velocity rows, warm start, "
                "regularization or world width).",
            ),
        )
    )


def supported(solver) -> bool:
    """Whether the solver's rows follow the law the classifier evaluates (:func:`decline` names why not)."""
    return decline(solver) is None


class ResolvedWorlds:
    """Own only stable per-world decision scratch; all solver state stays original.

    An unsupported constructor exposes a length-zero mask and performs no
    launches; the allocators treat an empty mask as "no world resolved".
    """

    def __init__(self, solver, *, absolute_margin: float = 1.0e-5, relative_margin: float = 2.0**-20):
        self.solver = solver
        declined = decline(solver)
        self.enabled = declined is None
        if self.enabled:
            self.reason = (
                "Admitted: interleaved matrix-free patch-friction rows whose kinds are all unilateral; a world "
                "resolves only if its checked rows fit every contact route."
            )
        else:
            self.reason = declined
        self.absolute_margin = float(absolute_margin)
        self.relative_margin = float(relative_margin)
        if not all(math.isfinite(v) and v >= 0.0 for v in (self.absolute_margin, self.relative_margin)):
            raise ValueError("Resolved-world rejection margins must be finite and nonnegative")
        device = solver.model.device
        worlds = solver.forest.world_count if self.enabled else 0
        self.scratch = ResolvedWorldScratch()
        if not self.enabled:
            self.resolved = wp.zeros(0, dtype=wp.int32, device=device)
            return
        # A world resolves only if its checked rows fit the dense and matrix-free capacities
        # (finalize_resolved_worlds).
        self.row_bound = (int(solver.placement.dense_max_constraints), int(solver.placement.mf_max_constraints))
        SolverArrays(self.scratch, device).allocate(RESOLVED_WORLD_SCRATCH, world=worlds, body=solver.model.body_count)
        self.resolved = self.scratch.resolved
        # Per-world count of checked contacts on matrix-free rows, for the patch bound.
        self.free_normals = wp.zeros(worlds, dtype=wp.int32, device=device)
        # A model without rigid velocity or depenetration limits bounds nothing.
        self.unbounded = wp.full(solver.model.body_count, float("inf"), dtype=wp.float32, device=device)
        # Predicted body twists: published by the held-tree predictor when every
        # responding, non-prescribed articulation is tree-owned; otherwise the
        # classifier's own body pass builds them.
        twists = getattr(solver, "_predicted_twist", None)
        flags = getattr(solver, "_predicted_twist_arts_host", None)
        self.twists = twists if twists is not None else wp.zeros(1, dtype=wp.spatial_vector, device=device)
        self.published = False
        if twists is not None and flags is not None:
            responding = solver.forest.articulation_response_dof_count.numpy() > 0
            prescribed = solver.forest.prescribed_articulation.numpy() != 0
            self.published = bool(np.all(flags[responding & ~prescribed] != 0))
        free = solver.forest.free_rigid_body_indices
        self.free_bodies = free if free is not None else wp.zeros(0, dtype=wp.int32, device=device)

    def classify(self, state_in, state_aug, contacts, dt: float) -> None:
        """Refresh the decision after the current predictor and collision completion."""
        if not self.enabled:
            if self.resolved.shape[0] > 0:
                self.resolved.zero_()
            return
        solver = self.solver
        if contacts is None:
            self.resolved.zero_()
            return
        # Absent rigid-body speed limits read as unbounded.
        limits = {
            name: self.unbounded
            if getattr(solver.laws, "rigid_body_" + name) is None
            else getattr(solver.laws, "rigid_body_" + name)
            for name in ("max_linear_velocity", "max_angular_velocity", "max_depenetration_velocity")
        }
        data = field_group.bind(
            ResolvedWorldInput,
            solver=solver,
            forest=solver.forest,
            factor=solver.factor,
            laws=solver.laws,
            options=solver.options,
            model=solver.model,
            state_in=state_in,
            state_aug=state_aug,
            limits=SimpleNamespace(**limits),
            classifier=self,
            dt=dt,
        )
        raw = field_group.bind(RawContacts, solver=solver, laws=solver.laws, model=solver.model, contacts=contacts)
        pool = solver.placement.contact_pool
        launch(
            data,
            raw,
            self.scratch,
            pool.launch_rows(),
            solver.model.device,
            published=self.published,
            row_bound=self.row_bound,
            free_normals=self.free_normals,
            pool=pool,
        )

    def record_watermark(self, watermark: wp.array) -> None:
        """Accumulate the rows the resolved worlds skipped into ``watermark`` (:func:`accumulate_resolved_rows`), so the
        row watermark counts them while the diagnostic changes no route."""
        wp.launch(
            accumulate_resolved_rows,
            dim=self.resolved.shape[0],
            inputs=[self.scratch, watermark],
            device=self.solver.model.device,
        )


def launch(
    data,
    raw,
    scratch,
    contact_capacity,
    device,
    *,
    enabled=1,
    published=False,
    row_bound,
    free_normals,
    pool=None,
):
    """Run the decision launches in stream order on ``device``.

    With published twists the body pass is replaced by the free bodies'
    velocity-cap check, and the contact check rejects an endpoint whose twist
    is not published; otherwise the body pass builds the twists itself. The
    ``row_bound`` of (dense, matrix-free) row capacities resolves only worlds
    whose worst-case rows fit both (:func:`finalize_resolved_worlds`),
    counting matrix-free contacts in ``free_normals``. With
    the contacts' ``pool`` the contact checks follow the live count in a
    capture whose launches bind (:meth:`ContactPool.launch`); size
    ``contact_capacity`` with its ``launch_rows()`` then.
    """
    worlds = scratch.resolved.shape[0]
    wp.launch(reset_world_decisions, dim=worlds, inputs=[enabled, data, scratch], device=device)
    wp.launch(check_world_limits, dim=(worlds, LANES), inputs=[data, scratch], device=device)
    if published:
        if data.free_bodies.shape[0] > 0:
            wp.launch(check_free_bodies, dim=data.free_bodies.shape[0], inputs=[data, scratch], device=device)
    else:
        bodies = scratch.body_valid.shape[0]
        wp.launch(build_predicted_body_twists, dim=bodies, inputs=[data, scratch], device=device)
    # The contact checks stride over the contacts on one wave of the device's resident blocks (a thread a contact on
    # the CPU): every verdict they write is an integer atomic or a minimum, so the grid changes no decision.
    grid = shape(Work(max(int(contact_capacity), 1), persistent=True), memory.device(device))
    threads = min(grid.grid * grid.block, max(int(contact_capacity), 1))

    def check(kernel, inputs):
        if pool is None:
            wp.launch(kernel, dim=threads, inputs=inputs, device=device)
        else:
            pool.launch(kernel, threads, inputs=inputs, extent=raw.count, clamp=True)

    free_normals.zero_()
    check(check_raw_contact_normals, [threads, raw, data, scratch, free_normals])
    wp.launch(
        finalize_resolved_worlds,
        dim=worlds,
        inputs=[scratch, free_normals, int(row_bound[0]), int(row_bound[1])],
        device=device,
    )
