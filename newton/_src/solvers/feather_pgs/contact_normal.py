# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared normal admission law with explicit storage and publication policies."""

import hashlib
import linecache
from string import Template

from .kinetic_source import checked_definitions

_ENDPOINT = r"""@wp.func
def ${endpoint_name}(
    body: int,
    art: int,
    mf: bool,
    lever: wp.vec3,
    normal: wp.vec3,
    data: _SimpleWorldInput,
    ${storage_signature},
):
    value = float(0.0)
    if body >= 0 and art >= 0:
${twist_load}        if mf and data.prescribed_articulation[art] == 0 and data.articulation_response_dof_count[art] > 0:
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


"""


_OWNERSHIP = r"""    shape_a = raw.shape0[c]
    shape_b = raw.shape1[c]
    if shape_a < -1 or shape_b < -1 or shape_a >= raw.shape_body.shape[0] or shape_b >= raw.shape_body.shape[0]:
        ${invalid}
        return${rejected}
    body_a = int(-1)
    body_b = int(-1)
    if shape_a >= 0:
        body_a = raw.shape_body[shape_a]
    if shape_b >= 0:
        body_b = raw.shape_body[shape_b]
    if body_a < -1 or body_b < -1 or body_a >= data.body_q.shape[0] or body_b >= data.body_q.shape[0]:
        ${invalid}
        return${rejected}
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
            ${invalid}
            return${rejected}
        world = other
    if art_a >= data.art_to_world.shape[0] or art_b >= data.art_to_world.shape[0]:
        ${invalid}
        return${rejected}
"""


_GEOMETRY = r"""    # Endpoint-relative gap/lever arithmetic avoids subtracting rounded large world points.
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
"""


_NORMAL = r"""@wp.func
def ${normal_name}(c: int, raw: ${raw_type}, data: _SimpleWorldInput, ${storage_signature}${geometry_signature}):
${ownership}    responds_a = False
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
        return${no_response}
    if world < 0 or world >= ${world_count}:
        ${invalid}
        return${rejected}
${checked}${limit_gate}    valid = True
    if body_a >= 0:
        valid = valid and ${valid_a}
    if body_b >= 0:
        valid = valid and ${valid_b}
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
        ${reject}
        return${rejected}
${geometry}    compatible_a = not dofs_a
    compatible_b = not dofs_b
    if art_a >= 0:
        compatible_a = compatible_a or data.is_free_rigid[art_a] != 0
    if art_b >= 0:
        compatible_b = compatible_b or data.is_free_rigid[art_b] != 0
    mf = compatible_a and compatible_b
    speed_a = ${endpoint_name}(body_a, art_a, mf, lever_a, n, data, ${storage_arguments})
    speed_b = ${endpoint_name}(body_b, art_b, mf, lever_b, n, data, ${storage_arguments})
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
${minimum}    valid = _finite3(separation) and _finite3(lever_a) and _finite3(lever_b) and wp.isfinite(phi)
    valid = valid and _passes(residual, wp.abs(speed_a) + wp.abs(speed_b) + wp.abs(bias), data)
${result}
"""


def get_source(policy):
    """Generate the original law for scratch, warp-local, or current geometry."""
    checked_definitions("contact_normal.py", ("_ENDPOINT", "_OWNERSHIP", "_GEOMETRY", "_NORMAL", "get_source", "bind"))
    if policy not in ("scratch", "light", "current"):
        raise ValueError("Unknown normal admission policy: " + policy)
    scratch = policy == "scratch"
    current = policy == "current"
    bindings = {
        "endpoint_name": "_endpoint_speed" if scratch else "_light_endpoint_speed",
        "normal_name": "_check_normal" if scratch else "_light_check_normal",
        "raw_type": "RawRowInput" if current else "_SimpleRawContacts",
        "storage_signature": "scratch: _SimpleWorldScratch"
        if scratch
        else "cache: ContactGeometry, state: RowState"
        if current
        else "address: wp.uint64, plan: JointWorldPlan",
        "storage_arguments": "scratch" if scratch else "cache, state" if current else "address, plan",
        "geometry_signature": ", separation: wp.vec3, lever_a: wp.vec3, lever_b: wp.vec3" if current else "",
        "no_response": "" if scratch else " True",
        "rejected": "" if scratch else " False",
        "invalid": "wp.atomic_max(scratch.global_invalid, 0, 1)" if scratch else "",
        "reject": "wp.atomic_max(scratch.rejected, world, 1)" if scratch else "",
        "world_count": "scratch.resolved.shape[0]"
        if scratch
        else "cache.ready.shape[0]"
        if current
        else "plan.body_ids.shape[0]",
        "checked": "    wp.atomic_add(scratch.checked_normals, world, 1)\n" if scratch else "",
        "limit_gate": "    if scratch.limit_ok[world] == 0:\n        return\n" if scratch else "",
        "valid_a": "scratch.body_valid[body_a] != 0"
        if scratch
        else "True"
        if current
        else "_body_valid(address, plan.body_lane[body_a])",
        "valid_b": "scratch.body_valid[body_b] != 0"
        if scratch
        else "True"
        if current
        else "_body_valid(address, plan.body_lane[body_b])",
        "minimum": "    wp.atomic_min(scratch.minimum_residual, world, residual)\n" if scratch else "",
        "result": "    if not valid:\n        wp.atomic_max(scratch.rejected, world, 1)\n"
        if scratch
        else "    return valid\n",
    }
    bindings["twist_load"] = (
        "        twist = scratch.body_twist[body]\n"
        if scratch
        else "        twist = state.endpoint_twists[body]\n"
        "        if data.prescribed_articulation[art] == 0 and (data.body_has_response_dofs[body] == 0 or (data.body_flags[body] & 2) != 0):\n"
        "            twist = wp.spatial_vector()\n"
        if current
        else "        twist = _body_twist(address, plan.body_lane[body])\n"
    )
    bindings["ownership"] = (
        "    shape_a, shape_b = raw.shape0[c], raw.shape1[c]\n"
        "    body_a, body_b = cache.bodies[c, 0], cache.bodies[c, 1]\n"
        "    art_a, art_b = raw.art_a[c], raw.art_b[c]\n"
        "    world = raw.world[c]\n"
        if current
        else Template(_OWNERSHIP).substitute(bindings)
    )
    bindings["geometry"] = "    phi = wp.dot(n, separation)\n" if current else _GEOMETRY
    return Template(_ENDPOINT).substitute(bindings) + Template(_NORMAL).substitute(bindings)


def bind(policy, namespace):
    """Bind inspectable Warp functions without recovering another consumer's code."""
    source = get_source(policy)
    filename = "<contact-normal-" + hashlib.sha256(source.encode()).hexdigest() + ">"
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    scope = dict(namespace)
    exec(compile(source, filename, "exec"), scope)
    names = (
        ("_endpoint_speed", "_check_normal")
        if policy == "scratch"
        else ("_light_endpoint_speed", "_light_check_normal")
    )
    return scope[names[0]], scope[names[1]]
