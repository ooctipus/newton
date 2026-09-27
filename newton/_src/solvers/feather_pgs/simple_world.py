# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Private pre-allocation FP32 zero-impulse selection.

This is a numerical fast-path selector, not an interval certificate. Every
applicable normal and limit must pass at the current predictor; ambiguous or
unsupported worlds retain the original complete solver. It owns no rows,
impulses, factors, contact publication, or integration. The caller must skip
all row allocators only for ``resolved`` worlds and preserve original fallback.
"""

import math

import warp as wp

from .kernels import contact_restitution_fires, mixed_contact_restitution


@wp.struct
class _SimpleWorldInput:
    world_dof_indices: wp.array2d[int]
    world_dof_count: wp.array[int]
    limit_q_index: wp.array[int]
    lower: wp.array[float]
    upper: wp.array[float]
    q: wp.array[float]
    v_hat: wp.array[float]
    body_to_articulation: wp.array[int]
    art_to_world: wp.array[int]
    articulation_dof_start: wp.array[int]
    articulation_response_dof_count: wp.array[int]
    prescribed_articulation: wp.array[int]
    is_free_rigid: wp.array[int]
    body_flags: wp.array[int]
    body_has_response_dofs: wp.array[int]
    body_response_dof_mask: wp.array[wp.uint32]
    body_q: wp.array[wp.transform]
    body_v_s: wp.array[wp.spatial_vector]
    joint_S_s: wp.array[wp.spatial_vector]
    articulation_origin: wp.array[wp.vec3]
    max_linear_velocity: wp.array[float]
    max_angular_velocity: wp.array[float]
    max_depenetration_velocity: wp.array[float]
    dt: float
    beta: float
    speculative_scale: float
    activation_gap: float
    restitution_threshold: float
    shared_anchor: int
    absolute_margin: float
    relative_margin: float


@wp.struct
class _SimpleRawContacts:
    count: wp.array[int]
    shape0: wp.array[int]
    shape1: wp.array[int]
    point0: wp.array[wp.vec3]
    point1: wp.array[wp.vec3]
    normal: wp.array[wp.vec3]
    margin0: wp.array[float]
    margin1: wp.array[float]
    shape_body: wp.array[int]
    shape_mu: wp.array[float]
    shape_restitution: wp.array[float]


@wp.struct
class _SimpleWorldScratch:
    limit_ok: wp.array[int]
    rejected: wp.array[int]
    resolved: wp.array[int]
    checked_limits: wp.array[int]
    checked_normals: wp.array[int]
    minimum_residual: wp.array[float]
    body_twist: wp.array[wp.spatial_vector]
    body_valid: wp.array[int]
    global_invalid: wp.array[int]


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
def _passes(residual: float, magnitude: float, data: _SimpleWorldInput):
    margin = data.absolute_margin + data.relative_margin * magnitude
    return wp.isfinite(residual) and wp.isfinite(margin) and residual >= margin


@wp.kernel(enable_backward=False)
def prepare_simple_worlds(enabled: int, data: _SimpleWorldInput, scratch: _SimpleWorldScratch):
    """Reset every decision and check every active signed position limit."""
    world = wp.tid()
    scratch.limit_ok[world] = 0
    scratch.rejected[world] = 0
    scratch.resolved[world] = 0
    scratch.checked_limits[world] = 0
    scratch.checked_normals[world] = 0
    scratch.minimum_residual[world] = wp.float32(3.402823466e38)
    if world == 0:
        scratch.global_invalid[0] = 0
    if enabled == 0:
        return
    valid = (
        wp.isfinite(data.dt)
        and data.dt > 0.0
        and wp.isfinite(data.beta)
        and data.beta >= 0.0
        and wp.isfinite(data.speculative_scale)
        and data.speculative_scale >= 0.0
        and wp.isfinite(data.activation_gap)
        and data.activation_gap >= 0.0
        and wp.isfinite(data.restitution_threshold)
        and data.restitution_threshold >= 0.0
        and wp.isfinite(data.absolute_margin)
        and data.absolute_margin >= 0.0
        and wp.isfinite(data.relative_margin)
        and data.relative_margin >= 0.0
    )
    count = data.world_dof_count[world]
    valid = valid and count >= 0 and count <= data.world_dof_indices.shape[1]
    checked = int(0)
    minimum = wp.float32(3.402823466e38)
    if valid:
        for local in range(count):
            dof = data.world_dof_indices[world, local]
            if dof < 0 or dof >= data.v_hat.shape[0] or dof >= data.limit_q_index.shape[0]:
                valid = False
                continue
            velocity = data.v_hat[dof]
            if not wp.isfinite(velocity):
                valid = False
                continue
            qi = data.limit_q_index[dof]
            if qi < 0:
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
                bound = lower
                active = wp.isfinite(lower) and q <= lower + data.activation_gap
                phi = q - lower
                signed_velocity = velocity
                if side == 1:
                    bound = upper
                    active = wp.isfinite(upper) and q >= upper - data.activation_gap
                    phi = upper - q
                    signed_velocity = -velocity
                if active and wp.isfinite(bound):
                    checked += 1
                    bias = phi / data.dt
                    if phi < 0.0:
                        bias = data.beta * phi / data.dt
                    residual = signed_velocity + bias
                    minimum = wp.min(minimum, residual)
                    valid = valid and _passes(residual, wp.abs(signed_velocity) + wp.abs(bias), data)
    scratch.checked_limits[world] = checked
    scratch.minimum_residual[world] = minimum
    if valid:
        scratch.limit_ok[world] = 1


@wp.kernel(enable_backward=False)
def build_predicted_body_twists(data: _SimpleWorldInput, scratch: _SimpleWorldScratch):
    """Contract current motion axes against v_hat, never incoming body velocity."""
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
    if scratch.limit_ok[world] == 0:
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
                for k in range(6):
                    limit = data.max_linear_velocity[body]
                    if k >= 3:
                        limit = data.max_angular_velocity[body]
                    valid = valid and not wp.isnan(limit)
                    if wp.isfinite(limit) and limit > 0.0:
                        valid = valid and wp.abs(data.v_hat[start + k]) <= limit
    elif data.articulation_response_dof_count[art] > 0 and (data.body_flags[body] & 2) != 0:
        # Partial prescribed membership needs a different source target model.
        valid = False
    valid = valid and _finite6(twist)
    if valid:
        scratch.body_twist[body] = twist
        scratch.body_valid[body] = 1
    else:
        wp.atomic_max(scratch.rejected, world, 1)


@wp.func
def _endpoint_speed(
    body: int,
    art: int,
    mf: bool,
    lever: wp.vec3,
    normal: wp.vec3,
    data: _SimpleWorldInput,
    scratch: _SimpleWorldScratch,
):
    value = float(0.0)
    if body >= 0 and art >= 0:
        twist = scratch.body_twist[body]
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
        value = wp.dot(normal, wp.spatial_top(twist) + wp.cross(wp.spatial_bottom(twist), lever))
    return value


@wp.func
def _check_normal(c: int, raw: _SimpleRawContacts, data: _SimpleWorldInput, scratch: _SimpleWorldScratch):
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
    responds_a = False
    responds_b = False
    dofs_a = False
    dofs_b = False
    if art_a >= 0:
        dofs_a = data.articulation_response_dof_count[art_a] > 0
        responds_a = dofs_a and data.body_has_response_dofs[body_a] != 0 and (data.body_flags[body_a] & 2) == 0
    if art_b >= 0:
        dofs_b = data.articulation_response_dof_count[art_b] > 0
        responds_b = dofs_b and data.body_has_response_dofs[body_b] != 0 and (data.body_flags[body_b] & 2) == 0
    if not responds_a and not responds_b:
        return
    if world < 0 or world >= scratch.resolved.shape[0]:
        wp.atomic_max(scratch.global_invalid, 0, 1)
        return
    wp.atomic_add(scratch.checked_normals, world, 1)
    if scratch.limit_ok[world] == 0:
        return
    valid = True
    if body_a >= 0:
        valid = valid and scratch.body_valid[body_a] != 0
    if body_b >= 0:
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
    # Endpoint-relative gap/lever arithmetic avoids subtracting rounded large world points.
    translation_a = wp.vec3(0.0)
    translation_b = wp.vec3(0.0)
    offset_a = raw.point0[c] - raw.margin0[c] * n
    offset_b = raw.point1[c] + raw.margin1[c] * n
    if body_a >= 0:
        pose = data.body_q[body_a]
        translation_a = wp.transform_get_translation(pose)
        offset_a = wp.quat_rotate(wp.transform_get_rotation(pose), raw.point0[c]) - raw.margin0[c] * n
    if body_b >= 0:
        pose = data.body_q[body_b]
        translation_b = wp.transform_get_translation(pose)
        offset_b = wp.quat_rotate(wp.transform_get_rotation(pose), raw.point1[c]) + raw.margin1[c] * n
    separation = (translation_a - translation_b) + (offset_a - offset_b)
    phi = wp.dot(n, separation)
    lever_a = offset_a
    lever_b = offset_b
    if art_a >= 0:
        lever_a += translation_a - data.articulation_origin[art_a]
    if art_b >= 0:
        lever_b += translation_b - data.articulation_origin[art_b]
    if data.shared_anchor != 0:
        lever_a -= 0.5 * separation
        lever_b += 0.5 * separation
    compatible_a = not dofs_a
    compatible_b = not dofs_b
    if art_a >= 0:
        compatible_a = compatible_a or data.is_free_rigid[art_a] != 0
    if art_b >= 0:
        compatible_b = compatible_b or data.is_free_rigid[art_b] != 0
    mf = compatible_a and compatible_b
    speed_a = _endpoint_speed(body_a, art_a, mf, lever_a, n, data, scratch)
    speed_b = _endpoint_speed(body_b, art_b, mf, lever_b, n, data, scratch)
    incident = speed_a - speed_b
    bias = data.speculative_scale * phi / data.dt
    if phi < 0.0:
        bias = data.beta * phi / data.dt
        if mf:
            cap = float(1.0e20)
            if responds_a:
                cap = data.max_depenetration_velocity[body_a]
            if responds_b:
                cap_b = data.max_depenetration_velocity[body_b]
                if cap_b > 0.0 and wp.isfinite(cap_b) and cap_b < cap:
                    cap = cap_b
            if cap > 0.0 and wp.isfinite(cap):
                bias = wp.max(bias, -cap)
    restitution = mixed_contact_restitution(shape_a, shape_b, raw.shape_restitution)
    if restitution > 0.0 and contact_restitution_fires(phi, incident, data.dt, data.restitution_threshold):
        bias = restitution * incident
    residual = incident + bias
    wp.atomic_min(scratch.minimum_residual, world, residual)
    valid = _finite3(separation) and _finite3(lever_a) and _finite3(lever_b) and wp.isfinite(phi)
    valid = valid and _passes(residual, wp.abs(speed_a) + wp.abs(speed_b) + wp.abs(bias), data)
    if not valid:
        wp.atomic_max(scratch.rejected, world, 1)


@wp.kernel(enable_backward=False)
def check_raw_contact_normals(
    threads: int, raw: _SimpleRawContacts, data: _SimpleWorldInput, scratch: _SimpleWorldScratch
):
    """Check the complete active raw prefix; invalid ownership rejects the call."""
    tid = wp.tid()
    count = raw.count[0]
    capacity = raw.shape0.shape[0]
    if (
        count < 0
        or count > capacity
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


@wp.kernel(enable_backward=False)
def finalize_simple_worlds(scratch: _SimpleWorldScratch):
    """Publish only after all limit/body/contact launches finish on one stream."""
    world = wp.tid()
    value = int(0)
    if scratch.limit_ok[world] != 0 and scratch.rejected[world] == 0 and scratch.global_invalid[0] == 0:
        value = 1
    scratch.resolved[world] = value


def _supported(solver) -> bool:
    """Match the original fixed recipe, not arbitrary unnotified topology mutation."""
    exact = {
        "pgs_mode": "matrix_free",
        "drive_mode": "augmented",
        "pgs_schedule": "interleaved",
        "friction_mode": "current",
        "pgs_iterations": 8,
        "pgs_velocity_iterations": 0,
        "max_world_dofs": 29,
        "_paired_response_primary_size": 23,
        "_paired_response_secondary_size": 6,
        "_mimic_count": 0,
        "_connect_count": 0,
        "_preelim_count": 0,
        "mf_gs_incremental_rows": 0,
        "mf_gs_parallel_rows": 0,
        "mf_gs_response_block_rows": 0,
        "articulated_contact_response": "immediate",
        "contact_gap_gate": 0.0,
        "same_articulation_contact_gap_gate": 0.0,
        "articulation_pair_contact_gap_gate": 0.0,
    }
    if any(getattr(solver, name, None) != value for name, value in exact.items()):
        return False
    false = (
        "pgs_warmstart",
        "_mf_warmstart_enabled",
        "_preelim_active",
        "_regularization_enabled",
        "enable_bilateral_preelimination",
        "enable_joint_velocity_limits",
        "fuse_joint_velocity_limits",
        "_fused_diagonal_joint_limits",
        "_sparse_diagonal_contact_solve",
        "_local_internal_fast_path",
        "pgs_debug",
        "_row_watermark",
        "_route_free_free_contacts",
        "propagation_same_articulation_rows",
        "_fused_k1",
        "grouped_dynamics",
        "lazy_kinematics",
        "_compact_contact_jacobian",
    )
    if any(getattr(solver, name, None) is not False for name in false):
        return False
    if not getattr(solver, "enable_joint_limits", False) or not getattr(solver, "_paired_factor_coordinates", False):
        return False
    if getattr(solver, "size_groups", None) != [23, 6] or getattr(solver, "_joint_limit_sizes", None) != frozenset(
        {23}
    ):
        return False
    owner = getattr(solver, "_world_rows_owned_le", None)
    if owner is None or owner() != 0 or getattr(solver.model, "requires_grad", True):
        return False
    if getattr(solver, "drive_slot", None) is not None:
        return False
    required = (
        "body_response_dof_mask",
        "body_has_response_dofs",
        "_joint_limit_q_index",
        "rigid_body_max_linear_velocity",
        "rigid_body_max_angular_velocity",
    )
    return all(getattr(solver, name, None) is not None for name in required)


class SimpleWorldClassifier:
    """Own only stable zero-world scratch; all solver/public state stays original.

    Unsupported constructors expose a length-zero mask and perform no launches.
    No static device-to-host copies or per-notification ownership proof occurs.
    The original model-plan/storage mutation contract remains the caller's.
    """

    def __init__(self, solver, *, absolute_margin: float = 1.0e-5, relative_margin: float = 2.0**-20):
        self.solver = solver
        self.enabled = bool(solver.model.device.is_cuda and _supported(solver))
        self.absolute_margin = float(absolute_margin)
        self.relative_margin = float(relative_margin)
        if not all(math.isfinite(v) and v >= 0.0 for v in (self.absolute_margin, self.relative_margin)):
            raise ValueError("Simple-world rejection margins must be finite and nonnegative")
        device = solver.model.device
        worlds = solver.world_count if self.enabled else 0
        self.resolved = wp.zeros(worlds, dtype=wp.int32, device=device)
        self.scratch = _SimpleWorldScratch()
        if not self.enabled:
            return
        self.scratch.resolved = self.resolved
        for name in ("limit_ok", "rejected", "checked_limits", "checked_normals"):
            setattr(self.scratch, name, wp.zeros(worlds, dtype=wp.int32, device=device))
        self.scratch.minimum_residual = wp.empty(worlds, dtype=wp.float32, device=device)
        self.scratch.body_twist = wp.empty(solver.model.body_count, dtype=wp.spatial_vector, device=device)
        self.scratch.body_valid = wp.empty(solver.model.body_count, dtype=wp.int32, device=device)
        self.scratch.global_invalid = wp.zeros(1, dtype=wp.int32, device=device)

    def classify(self, state_in, state_aug, contacts, dt: float) -> None:
        """Refresh the decision after current predictor and collision completion."""
        if not self.enabled:
            if self.resolved.shape[0] > 0:
                self.resolved.zero_()
            return
        solver = self.solver
        if not _supported(solver) or contacts is None or getattr(state_in, "requires_grad", False):
            self.resolved.zero_()
            return
        data = _SimpleWorldInput()
        for name in (
            "world_dof_indices",
            "world_dof_count",
            "v_hat",
            "body_to_articulation",
            "art_to_world",
            "articulation_dof_start",
            "articulation_response_dof_count",
            "is_free_rigid",
            "body_has_response_dofs",
            "body_response_dof_mask",
            "articulation_origin",
        ):
            setattr(data, name, getattr(solver, name))
        data.limit_q_index = solver._joint_limit_q_index
        data.lower, data.upper, data.q = (
            solver.model.joint_limit_lower,
            solver.model.joint_limit_upper,
            state_in.joint_q,
        )
        data.prescribed_articulation = solver._prescribed_articulation
        data.body_flags, data.body_q = solver.model.body_flags, state_in.body_q
        data.body_v_s, data.joint_S_s = state_aug.body_v_s, state_aug.joint_S_s
        data.max_linear_velocity = solver.rigid_body_max_linear_velocity
        data.max_angular_velocity = solver.rigid_body_max_angular_velocity
        data.max_depenetration_velocity = solver.rigid_body_max_depenetration_velocity
        data.dt, data.beta, data.speculative_scale = dt, solver.pgs_beta, solver.contact_speculative_scale
        data.activation_gap = solver.joint_limit_activation_gap
        data.restitution_threshold = solver._effective_restitution_velocity_threshold
        data.shared_anchor = int(solver.contact_shared_anchor)
        data.absolute_margin, data.relative_margin = self.absolute_margin, self.relative_margin
        raw = _SimpleRawContacts()
        raw.count = contacts.rigid_contact_count
        for name in ("shape0", "shape1", "point0", "point1", "normal", "margin0", "margin1"):
            setattr(raw, name, getattr(contacts, "rigid_contact_" + name))
        raw.shape_body, raw.shape_mu = solver.model.shape_body, solver.shape_material_mu
        raw.shape_restitution = solver.shape_material_restitution
        device = solver.model.device
        wp.launch(prepare_simple_worlds, dim=solver.world_count, inputs=[1, data, self.scratch], device=device)
        wp.launch(build_predicted_body_twists, dim=solver.model.body_count, inputs=[data, self.scratch], device=device)
        threads = max(1, min(contacts.rigid_contact_max, 65536))
        wp.launch(check_raw_contact_normals, dim=threads, inputs=[threads, raw, data, self.scratch], device=device)
        wp.launch(finalize_simple_worlds, dim=solver.world_count, inputs=[self.scratch], device=device)
