# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Complete model-generated state/dynamics ownership with one world per thread.

The initial consumer admission is the retained 13-body kinetic owner. Joint
types and tree dependencies generate code; physical values remain live arrays.
Canonical contact, factor, generalized-state and body-state layouts are intact.
"""

import functools
import hashlib
import linecache
import textwrap

import numpy as np
import warp as wp

from . import franka_kinetic_state as retained
from . import kernels, kuka_joint_world

KineticPlan = retained.KineticPlan
KineticData = retained.KineticData
PublicationData = retained.PublicationData
ForceInput = retained.ForceInput


@wp.struct
class FactorInput:
    mask: wp.array[int]
    armature9: wp.array2d[float]
    armature6: wp.array2d[float]
    drive_row: wp.array[int]
    drive_K: wp.array[float]
    inertia: wp.array[wp.spatial_matrix]


def build_schedule(model, host_plan):
    """Derive a homogeneous typed tree, without specializing physical values."""
    ids = host_plan["joint_ids"][:, :13]
    kinds = model.joint_type.numpy()[ids]
    parents = host_plan["body_parent"][:, :13]
    dofs = host_plan["body_local_dof"][:, :13]
    for value in (kinds, parents, dofs):
        if not np.all(value == value[0]):
            raise ValueError("World-lane state requires homogeneous topology and joint types")
    parent, kind, dof = (tuple(map(int, value[0])) for value in (parents, kinds, dofs))
    if any(p >= i for i, p in enumerate(parent)):
        raise ValueError("World-lane state requires parent-before-child physical IDs")
    roots = tuple(i for i, p in enumerate(parent) if p < 0)
    if len(roots) != 3 or kind[roots[0]] != 3 or any(kind[r] != 4 for r in roots[1:]):
        raise ValueError("World-lane initial admission requires fixed primary and two free roots")
    return parent, kind, dof, roots


@wp.func
def _finite_motion(value: wp.spatial_vector):
    return (
        wp.isfinite(value[0])
        and wp.isfinite(value[1])
        and wp.isfinite(value[2])
        and wp.isfinite(value[3])
        and wp.isfinite(value[4])
        and wp.isfinite(value[5])
    )


@wp.func
def _finite_pose(value: wp.transform):
    return (
        wp.isfinite(value[0])
        and wp.isfinite(value[1])
        and wp.isfinite(value[2])
        and wp.isfinite(value[3])
        and wp.isfinite(value[4])
        and wp.isfinite(value[5])
        and wp.isfinite(value[6])
    )


@wp.func
def _motion(parent_v: wp.spatial_vector, parent_a: wp.spatial_vector, local_v: wp.spatial_vector):
    pv, pw = wp.spatial_top(parent_v), wp.spatial_bottom(parent_v)
    cv, cw = wp.spatial_top(local_v), wp.spatial_bottom(local_v)
    cross = wp.spatial_vector(wp.cross(pw, cv) + wp.cross(pv, cw), wp.cross(pw, cw))
    return parent_v + local_v, parent_a + cross


@wp.func
def _public_velocity(
    body: int,
    local: int,
    world: int,
    pose: wp.transform,
    origin: wp.vec3,
    velocity: wp.spatial_vector,
    data: PublicationData,
    cache: KineticData,
):
    radius = wp.transform_point(pose, data.body_com[body]) - origin
    public = wp.spatial_vector(
        wp.spatial_top(velocity) + wp.cross(wp.spatial_bottom(velocity), radius), wp.spatial_bottom(velocity)
    )
    data.body_qd[body] = public
    cache.com_offset[local, world] = radius
    return public


@wp.func
def _held_terms(
    body: int,
    pose: wp.transform,
    origin: wp.vec3,
    velocity: wp.spatial_vector,
    acceleration: wp.spatial_vector,
    data: PublicationData,
):
    pose_com = pose * data.body_X_com[body]
    radius = wp.transform_get_translation(pose_com) - origin
    rotation = wp.transform_get_rotation(pose_com)
    mass, inertia = data.body_mass[body], data.body_inertia[body]
    linear, omega = wp.spatial_top(velocity), wp.spatial_bottom(velocity)
    a, alpha = wp.spatial_top(acceleration), wp.spatial_bottom(acceleration)
    momentum = wp.quat_rotate(rotation, inertia * wp.quat_rotate_inv(rotation, omega))
    angular_force = wp.quat_rotate(rotation, inertia * wp.quat_rotate_inv(rotation, alpha))
    force = mass * (a + wp.cross(alpha, radius) + wp.cross(omega, linear + wp.cross(omega, radius)) - data.gravity[0])
    torque = angular_force + wp.cross(omega, momentum) + wp.cross(radius, force)
    return wp.spatial_vector(force, torque)


@wp.func
def _refresh_terms(
    body: int,
    pose: wp.transform,
    origin: wp.vec3,
    velocity: wp.spatial_vector,
    acceleration: wp.spatial_vector,
    data: PublicationData,
):
    pose_com = pose * data.body_X_com[body]
    radius = wp.transform_get_translation(pose_com) - origin
    rotation = wp.quat_to_matrix(wp.transform_get_rotation(pose_com))
    mass = data.body_mass[body]
    inertia = rotation * data.body_inertia[body] * wp.transpose(rotation)
    linear, omega = wp.spatial_top(velocity), wp.spatial_bottom(velocity)
    a, alpha = wp.spatial_top(acceleration), wp.spatial_bottom(acceleration)
    force = mass * (a + wp.cross(alpha, radius) + wp.cross(omega, linear + wp.cross(omega, radius)) - data.gravity[0])
    torque = inertia * alpha + wp.cross(omega, inertia * omega) + wp.cross(radius, force)
    at_origin = inertia + mass * (wp.identity(n=3, dtype=float) * wp.dot(radius, radius) - wp.outer(radius, radius))
    return wp.spatial_vector(force, torque), mass, mass * radius, at_origin


@wp.func
def _moment_action(mass: float, moment: wp.vec3, inertia: wp.mat33, axis: wp.spatial_vector):
    linear, angular = wp.spatial_top(axis), wp.spatial_bottom(axis)
    return wp.spatial_vector(mass * linear + wp.cross(angular, moment), wp.cross(moment, linear) + inertia * angular)


@wp.func
def _external(body: int, local: int, world: int, cache: KineticData, force: ForceInput):
    value = wp.spatial_vector()
    if (force.body_flags[body] & 2) == 0:
        authored = force.body_f[body]
        linear = wp.spatial_top(authored)
        value = wp.spatial_vector(
            linear, wp.spatial_bottom(authored) + wp.cross(cache.com_offset[local, world], linear)
        )
    return value


@wp.kernel(module="unique", enable_backward=False)
def _invalidate(world_mask: wp.array[wp.bool], plan: KineticPlan, cache: KineticData):
    world = wp.tid()
    if not world_mask or world_mask[world]:
        for root in range(3):
            cache.current_valid[plan.arts[root, world]] = 0
        cache.geometry_valid[world] = 0


@functools.cache
def _operations():
    """Share exact free-root and generalized integration helpers with the original owner."""
    source = "\n".join(
        kuka_joint_world.operation_source(name)
        for name in ("update_qdd_from_velocity", "remove_free_root_transport_from_qdd", "integrate_generalized_joints")
    )
    namespace = dict(kernels.__dict__)
    filename = "<world-lane-original-operations>"
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    exec(compile(source, filename, "exec"), namespace)
    return {
        "update_qdd": namespace["_light_update_qdd_from_velocity"],
        "remove_transport": namespace["_light_remove_free_root_transport_from_qdd"],
        "integrate": namespace["_light_integrate_generalized_joints"],
    }


def _compile(source, name, extra=None):
    namespace = dict(globals())
    namespace.update(_operations())
    namespace.update(extra or {})
    filename = f"<world-lane-{hashlib.sha256(source.encode()).hexdigest()}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    exec(compile(source, filename, "exec"), namespace)
    return namespace[name]


def _producer_source(schedule, *, finish, refresh, name):
    parent, kinds, dofs, roots = schedule
    primary_end = roots[1]
    lines = [
        "@wp.func",
        f"def {name}(world: int, plan: KineticPlan, data: PublicationData, cache: KineticData):",
        "    good = bool(True)",
    ]

    def emit(source):
        lines.extend("    " + line for line in textwrap.dedent(source).strip().splitlines())

    children = {i: tuple(j for j, p in enumerate(parent) if p == i) for i in range(len(parent))}
    descendants = {}

    def body_code(body, root_index):
        kind, p, dof = kinds[body], parent[body], dofs[body]
        emit(f"joint{body} = plan.joint_ids[{body}, world]\nbody{body} = plan.body_ids[{body}, world]")
        emit(f"start{body} = data.joint_qd_start[joint{body}]\nqs{body} = data.joint_q_start[joint{body}]")
        if kind in (0, 1):
            emit(f"scalar_q{body} = data.joint_q[qs{body}]\nscalar_v{body} = data.joint_qd[start{body}]")
            if finish:
                emit(f"""
                scalar_a{body} = float(0.0)
                if data.kinematic_dof_mask[start{body}] != 0:
                    data.v_out[start{body}] = scalar_v{body}
                else:
                    scalar_a{body} = (data.v_out[start{body}] - scalar_v{body}) * (1.0 / data.dt)
                data.joint_qdd[start{body}] = scalar_a{body}
                if data.kinematic_joint_mask[joint{body}] == 0:
                    scalar_v{body} = scalar_v{body} + scalar_a{body} * data.dt
                    scalar_q{body} = scalar_q{body} + scalar_v{body} * data.dt
                data.joint_q_new[qs{body}] = scalar_q{body}
                data.joint_qd_new[start{body}] = scalar_v{body}
                """)
            motion = (
                f"wp.transform(wp.vec3(), wp.quat_from_axis_angle(data.joint_axis[start{body}], scalar_q{body}))"
                if kind == 1
                else f"wp.transform(data.joint_axis[start{body}] * scalar_q{body}, wp.quat_identity())"
            )
            emit(
                f"relative{body} = data.joint_X_p[joint{body}] * {motion} * wp.transform_inverse(data.joint_X_c[joint{body}])"
            )
        elif kind == 3:
            emit(
                f"relative{body} = data.joint_X_p[joint{body}] * wp.transform_identity() * wp.transform_inverse(data.joint_X_c[joint{body}])"
            )
        else:
            if finish:
                for component in range(6):
                    emit(
                        f"update_qdd(start{body} + {component}, data.joint_qd, data.kinematic_dof_mask, 1.0 / data.dt, data.v_out, data.joint_qdd)"
                    )
                emit(f"""
                remove_transport(plan.root_slots[{root_index - 1}, world], data.free_root_joint_indices,
                                 data.joint_qd_start, data.kinematic_joint_mask, data.joint_qd, data.joint_qdd)
                integrate(joint{body}, data.joint_type, data.joint_parent, data.joint_child,
                          data.joint_q_start, data.joint_qd_start, data.kinematic_joint_mask,
                          data.joint_dof_dim, data.body_com, data.joint_X_c,
                          data.joint_q, data.joint_qd, data.joint_qdd, data.dt,
                          data.angular_damping, data.joint_q_new, data.joint_qd_new)
                """)
            q_array = "data.joint_q_new" if finish else "data.joint_q"
            emit(
                f"relative{body} = data.joint_X_p[joint{body}] * kernels.jcalc_transform({kind}, data.joint_axis, start{body}, 3, 3, {q_array}, qs{body}) * wp.transform_inverse(data.joint_X_c[joint{body}])"
            )
        if body == roots[0]:
            emit(
                f"shift0 = wp.transform_get_translation(relative{body})\npose{body} = wp.transform(wp.vec3(), wp.transform_get_rotation(relative{body}))"
            )
        elif p < 0:
            emit(f"shift{root_index} = wp.vec3()\npose{body} = relative{body}")
        else:
            emit(f"pose{body} = pose{p} * relative{body}")
        if p < 0:
            emit(f"""
            origin{root_index} = wp.transform_point(pose{body}, data.body_com[body{body}])
            data.articulation_origin[plan.arts[{root_index}, world]] = origin{root_index} + shift{root_index}
            anchor{body} = data.joint_X_p[joint{body}]
            anchor{body} = wp.transform(wp.transform_get_translation(anchor{body}) - shift{root_index} - origin{root_index}, wp.transform_get_rotation(anchor{body}))
            """)
        else:
            emit(f"""
            anchor{body} = pose{p} * data.joint_X_p[joint{body}]
            anchor{body} = wp.transform(wp.transform_get_translation(anchor{body}) - origin{root_index}, wp.transform_get_rotation(anchor{body}))
            """)
        if kind in (0, 1):
            screw = (
                f"wp.spatial_vector(wp.vec3(), data.joint_axis[start{body}])"
                if kind == 1
                else f"wp.spatial_vector(data.joint_axis[start{body}], wp.vec3())"
            )
            emit(f"""
            axis{body} = kernels.transform_twist(anchor{body}, {screw})
            data.joint_S_s[start{body}] = axis{body}
            local_v{body} = axis{body} * scalar_v{body}
            """)
        elif kind == 3:
            emit(f"local_v{body} = wp.spatial_vector()")
        else:
            qd_array = "data.joint_qd_new" if finish else "data.joint_qd"
            emit(
                f"local_v{body} = kernels.jcalc_motion({kind}, data.joint_axis, 3, 3, anchor{body}, {qd_array}, start{body}, data.joint_S_s)"
            )
        if p < 0:
            emit(f"velocity{body} = local_v{body}\nacceleration{body} = wp.spatial_vector()")
        else:
            emit(f"velocity{body}, acceleration{body} = _motion(velocity{p}, acceleration{p}, local_v{body})")
        emit(f"""
        public_pose{body} = wp.transform(wp.transform_get_translation(pose{body}) + shift{root_index}, wp.transform_get_rotation(pose{body}))
        data.body_q[body{body}] = public_pose{body}
        good = good and _finite_pose(public_pose{body})
        """)
        if body < primary_end:
            emit(f"""
            public_v{body} = _public_velocity(body{body}, {body}, world, pose{body}, origin0, velocity{body}, data, cache)
            good = good and _finite_motion(public_v{body})
            """)
            # The fixed root's inertial wrench/moments have no responding ancestor.
            if body != roots[0]:
                if refresh:
                    emit(
                        f"wrench{body}, mass{body}, moment{body}, inertia{body} = _refresh_terms(body{body}, pose{body}, origin0, velocity{body}, acceleration{body}, data)"
                    )
                else:
                    emit(
                        f"wrench{body} = _held_terms(body{body}, pose{body}, origin0, velocity{body}, acceleration{body}, data)"
                    )
        else:
            emit(f"""
            data.body_q_com[body{body}] = pose{body} * data.body_X_com[body{body}]
            data.body_v_s[body{body}] = velocity{body}
            data.body_a_s[body{body}] = acceleration{body}
            kernels.finalize_body_dynamics_body(body{body}, data.body_to_articulation,
                data.body_q, data.body_q_com, data.body_com, data.body_mass, data.body_inertia,
                data.is_free_rigid, data.articulation_origin, data.materialize_all_body_inertia,
                data.materialize_body_inertia_terms, data.gravity, data.body_v_s, data.body_a_s,
                data.body_I_s, data.body_inertia_terms, data.body_f_s, data.body_qd)
            cache.com_offset[{body}, world] = wp.transform_point(pose{body}, data.body_com[body{body}]) - origin{root_index}
            good = good and _finite_motion(data.body_qd[body{body}])
            """)
            if root_index == 1:
                emit(f"cache.free_bias[world] = data.body_f_s[body{body}]")
        completed = []
        for child in children[body]:
            body_code(child, root_index)
            completed.extend(descendants[child])
            if body != roots[0]:
                emit(f"wrench{body} = wrench{body} + wrench{child}")
                if refresh:
                    emit(
                        f"mass{body} = mass{body} + mass{child}\nmoment{body} = moment{body} + moment{child}\ninertia{body} = inertia{body} + inertia{child}"
                    )
        if dof >= 0:
            emit(f"""
            projection{body} = data.joint_S_s[start{body}]
            bias{body} = wp.dot(projection{body}, wrench{body})
            cache.bias[{dof}, world] = bias{body}
            good = good and wp.isfinite(bias{body})
            """)
            if refresh:
                emit(f"column{dof} = _moment_action(mass{body}, moment{body}, inertia{body}, projection{body})")
                for descendant in (dof, *completed):
                    row, col = max(dof, descendant), min(dof, descendant)
                    slot = row * (row + 1) // 2 + col
                    emit(
                        f"h{slot} = wp.dot(projection{body}, column{descendant})\ncache.geometric[{slot}, world] = h{slot}\ngood = good and wp.isfinite(h{slot})"
                    )
            completed.append(dof)
        descendants[body] = completed

    for root_index, body in enumerate(roots):
        body_code(body, root_index)
    if refresh:
        n = sum(d >= 0 for d in dofs)
        present = set()
        for body, dof in enumerate(dofs):
            if dof < 0:
                continue
            p = body
            while p >= 0:
                ancestor = dofs[p]
                if ancestor >= 0:
                    present.add((max(dof, ancestor), min(dof, ancestor)))
                p = parent[p]
        for row in range(n):
            for col in range(row + 1):
                if (row, col) not in present:
                    emit(f"cache.geometric[{row * (row + 1) // 2 + col}, world] = 0.0")
    emit("""
    if not good:
        cache.status[world] = cache.status[world] | 1
    cache.generation[world] = cache.generation[world] + wp.int64(1)
    """)
    emit(f"cache.geometry_valid[world] = int({refresh} and cache.status[world] == 0)")
    if refresh:
        emit("cache.geometry_generation[world] = cache.generation[world]")
    for root_index in range(len(roots)):
        emit(f"cache.current_valid[plan.arts[{root_index}, world]] = int(good and cache.status[world] == 0)")
    q, qd = ("data.joint_q_new", "data.joint_qd_new") if finish else ("data.joint_q", "data.joint_qd")
    emit(f"retained._stamp_source(cache, world, {q}, {qd})\nreturn int(good)")
    return "\n".join(lines) + "\n"


@functools.cache
def get_state_kernel(schedule, *, finish=False, refresh=False):
    """Compile actual typed dependencies; never keep a dynamically indexed local state array."""
    tag = hashlib.sha256(repr(schedule).encode()).hexdigest()[:10]
    extra = {}
    for current_refresh in (False, True) if not finish else (refresh,):
        name = f"produce_{tag}_{int(finish)}_{int(current_refresh)}"
        source = _producer_source(schedule, finish=finish, refresh=current_refresh, name=name)
        extra["produce_refresh" if current_refresh else "produce_held"] = _compile(source, name)
    mode = "repair" if not finish else ("finish_refresh" if refresh else "finish_held")
    name = f"world_lane_{mode}13_h45_{tag}"
    lines = [
        '@wp.kernel(module="unique", enable_backward=False)',
        f"def {name}(plan: KineticPlan, data: PublicationData, cache: KineticData, requests: wp.array[int], global_refresh: int):",
        "    world = wp.tid()",
    ]
    if not finish:
        lines.extend(
            textwrap.indent(
                textwrap.dedent("""
        refresh_now = global_refresh != 0 or requests[0] != 0
        valid = cache.current_valid[plan.arts[0, world]] & cache.current_valid[plan.arts[1, world]] & cache.current_valid[plan.arts[2, world]]
        if valid != 0 and retained._same_source(cache, world, data.joint_q, data.joint_qd) != 0:
            if not refresh_now or (cache.geometry_valid[world] != 0 and cache.geometry_generation[world] == cache.generation[world]):
                return
        if refresh_now:
            produce_refresh(world, plan, data, cache)
        else:
            produce_held(world, plan, data, cache)
        """).strip(),
                "    ",
            ).splitlines()
        )
    else:
        lines.append(f"    {'produce_refresh' if refresh else 'produce_held'}(world, plan, data, cache)")
    return _compile("\n".join(lines) + "\n", name, extra)


def _predictor_source(schedule):
    parent, _, dofs, roots = schedule
    primary_end = roots[1]
    name = "world_lane_factor_predict9_6_" + hashlib.sha256(repr(schedule).encode()).hexdigest()[:10]
    lines = [
        '@wp.kernel(module="unique", enable_backward=False)',
        f"def {name}(plan: KineticPlan, cache: KineticData, force: ForceInput, factor: FactorInput):",
        "    world = wp.tid()",
        "    good = wp.isfinite(force.dt) and force.dt > 0.0 and cache.status[world] == 0",
    ]

    def emit(source):
        lines.extend("    " + line for line in textwrap.dedent(source).strip().splitlines())

    for root in range(3):
        emit(f"good = good and cache.current_valid[plan.arts[{root}, world]] != 0")
    # The fixed root has no generalized response, but retain the original
    # nonfinite external-force guard for this authored input.
    emit(
        f"root_external = _external(plan.body_ids[{roots[0]}, world], {roots[0]}, world, cache, force)\ngood = good and _finite_motion(root_external)"
    )
    for body in reversed(range(1, primary_end)):
        emit(
            f"external{body} = _external(plan.body_ids[{body}, world], {body}, world, cache, force)\ngood = good and _finite_motion(external{body})"
        )
        for child, p in enumerate(parent[:primary_end]):
            if p == body:
                emit(f"external{body} = external{body} + external{child}")
        dof = dofs[body]
        if dof >= 0:
            emit(f"""
            d{dof} = plan.dof_ids[{dof}, world]
            tau{dof} = force.joint_f[d{dof}] + force.u0[d{dof}]
            tau{dof} = tau{dof} - cache.bias[{dof}, world] + wp.dot(force.S[d{dof}], external{body}) + force.stiffness[d{dof}] * (force.reference[d{dof}] - force.joint_q[plan.q_index[{dof}, world]]) - force.damping[d{dof}] * force.joint_qd[d{dof}]
            good = good and wp.isfinite(tau{dof})
            """)

    def family(size, offset, root):
        group_name = "primary_group" if root == 0 else "secondary_group"
        emit(f"group{size} = plan.{group_name}[world]\nrefresh{size} = factor.mask[plan.arts[{root}, world]] != 0")
        if root == 1:
            body = roots[1]
            emit(
                f"external_free = _external(plan.body_ids[{body}, world], {body}, world, cache, force)\ngood = good and _finite_motion(external_free)"
            )
            for row in range(size):
                dof = offset + row
                emit(f"""
                d{dof} = plan.dof_ids[{dof}, world]
                tau{dof} = force.joint_f[d{dof}] + force.u0[d{dof}] - wp.dot(force.S[d{dof}], cache.free_bias[world] - external_free)
                good = good and wp.isfinite(tau{dof})
                """)
        for row in range(size):
            for col in range(row + 1):
                emit(f"l{size}_{row}_{col} = float(0.0)")
        emit(f"if refresh{size}:")
        block = []
        if root == 1:
            for col in range(size):
                block.append(f"f{col} = factor.inertia[plan.body_ids[{roots[1]}, world]] * force.S[d{offset + col}]")
        for row in range(size):
            for col in range(row + 1):
                variable = f"l{size}_{row}_{col}"
                value = (
                    f"cache.geometric[{row * (row + 1) // 2 + col}, world]"
                    if root == 0
                    else f"wp.dot(force.S[d{offset + row}], f{col})"
                )
                block.append(f"{variable} = {value}")
                if row == col:
                    block.extend(
                        (
                            f"{variable} = {variable} + factor.armature{size}[group{size}, {row}]",
                            f"drive{size}_{row} = factor.drive_row[d{offset + row}]",
                            f"if drive{size}_{row} >= 0:",
                            f"    coefficient{size}_{row} = factor.drive_K[drive{size}_{row}]",
                            f"    if coefficient{size}_{row} > 0.0:",
                            f"        {variable} = {variable} + coefficient{size}_{row}",
                        )
                    )
        for col in range(size):
            for k in range(col):
                block.append(f"l{size}_{col}_{col} = l{size}_{col}_{col} - l{size}_{col}_{k} * l{size}_{col}_{k}")
            block.append(f"l{size}_{col}_{col} = wp.sqrt(l{size}_{col}_{col})")
            block.append(f"inverse{size}_{col} = 1.0 / l{size}_{col}_{col}")
            for row in range(col + 1, size):
                for k in range(col):
                    block.append(f"l{size}_{row}_{col} = l{size}_{row}_{col} - l{size}_{row}_{k} * l{size}_{col}_{k}")
                block.append(f"l{size}_{row}_{col} = l{size}_{row}_{col} * inverse{size}_{col}")
        for row in range(size):
            for col in range(row + 1):
                block.append(f"force.lower{size}[group{size}, {row}, {col}] = l{size}_{row}_{col}")
        lines.extend("        " + line for line in block)
        emit("else:")
        for row in range(size):
            for col in range(row + 1):
                lines.append(f"        l{size}_{row}_{col} = force.lower{size}[group{size}, {row}, {col}]")
        for row in range(size):
            emit(f"x{size}_{row} = tau{offset + row}")
            for k in range(row):
                emit(f"x{size}_{row} = x{size}_{row} - l{size}_{row}_{k} * x{size}_{k}")
            emit(f"x{size}_{row} = x{size}_{row} / l{size}_{row}_{row} if l{size}_{row}_{row} != 0.0 else 0.0")
        for row in reversed(range(size)):
            for k in range(row + 1, size):
                emit(f"x{size}_{row} = x{size}_{row} - l{size}_{k}_{row} * x{size}_{k}")
            emit(
                f"x{size}_{row} = x{size}_{row} / l{size}_{row}_{row} if l{size}_{row}_{row} != 0.0 else 0.0\ngood = good and wp.isfinite(x{size}_{row})"
            )
        for row in range(size):
            dof = offset + row
            emit(f"""
            acceleration{dof} = x{size}_{row} if force.kinematic_dof[d{dof}] == 0 else 0.0
            """)

    family(9, 0, 0)
    family(6, 9, 1)
    emit("if not good:\n    cache.status[world] = cache.status[world] | 2\n    return")
    for dof in range(15):
        emit(
            f"force.qdd[d{dof}] = acceleration{dof}\nforce.vhat[d{dof}] = force.predictor_qd[d{dof}] + acceleration{dof} * force.dt"
        )
    for dof in range(15, 21):
        emit(
            f"d{dof} = plan.dof_ids[{dof}, world]\nforce.qdd[d{dof}] = 0.0\nforce.vhat[d{dof}] = force.predictor_qd[d{dof}]"
        )
    for root in (1, 2):
        body, offset = roots[root], 9 + 6 * (root - 1)
        emit(f"""
        if force.kinematic_joint[plan.joint_ids[{body}, world]] == 0:
            transport_v{root} = wp.vec3(force.predictor_qd[d{offset}], force.predictor_qd[d{offset + 1}], force.predictor_qd[d{offset + 2}])
            transport_w{root} = wp.vec3(force.predictor_qd[d{offset + 3}], force.predictor_qd[d{offset + 4}], force.predictor_qd[d{offset + 5}])
            transport{root} = wp.cross(transport_w{root}, transport_v{root})
        """)
        for component in range(3):
            lines.append(
                f"        force.vhat[d{offset + component}] = force.vhat[d{offset + component}] + transport{root}[{component}] * force.dt"
            )
    return "\n".join(lines) + "\n", name


@functools.cache
def get_predictor_kernel(schedule):
    """Keep original current-mask Cholesky and held triangular actions in one owner."""
    source, name = _predictor_source(schedule)
    return _compile(source, name)


class WorldLaneState(retained.FrankaKineticState):
    """Use retained admission/invalidation proof with field-major private storage."""

    world_lane_state = True

    def __init__(self, solver):
        super().__init__(solver)
        self.schedule = build_schedule(solver.model, self.host_plan)
        worlds, device = solver.world_count, solver.model.device
        for name, value in self.host_plan.items():
            if value.ndim == 2:
                setattr(self.plan, name, wp.array(np.ascontiguousarray(value.T), dtype=int, device=device))
        self.data.bias = wp.zeros((9, worlds), dtype=float, device=device)
        self.data.com_offset = wp.zeros((13, worlds), dtype=wp.vec3, device=device)
        self.data.geometric = wp.zeros((45, worlds), dtype=float, device=device)
        self.geometric, self.bias = self.data.geometric, self.data.bias
        self.repair_kernel = get_state_kernel(self.schedule)
        self.finish_held_kernel = get_state_kernel(self.schedule, finish=True, refresh=False)
        self.finish_refresh_kernel = get_state_kernel(self.schedule, finish=True, refresh=True)
        self.finish_kernel = self.finish_refresh_kernel
        self.predictor_kernel = get_predictor_kernel(self.schedule)

    def invalidate(self, world_mask=None):
        if world_mask is not None and (
            world_mask.dtype != wp.bool
            or world_mask.shape != (self.solver.world_count,)
            or world_mask.device != self.solver.model.device
        ):
            raise ValueError("World-lane reset requires the original per-world bool mask")
        wp.launch(
            _invalidate,
            dim=self.solver.world_count,
            inputs=[world_mask, self.plan, self.data],
            device=self.solver.model.device,
        )

    def begin(self, state_in, state_aug, dt, global_refresh):
        self._validate_call(state_in, state_aug, dt)
        wp.launch(
            self.repair_kernel,
            dim=self.solver.world_count,
            inputs=[
                self.plan,
                self._publication(state_in, state_aug, state_in, dt),
                self.data,
                self.solver._mass_update_requested,
                int(global_refresh),
            ],
            block_dim=128,
            device=self.solver.model.device,
        )

    def predict(self, state_in, state_aug, control, stage3_qd, dt):
        self._validate_call(state_in, state_aug, dt)
        if stage3_qd.ptr != state_in.joint_qd.ptr:
            raise RuntimeError("World-lane excludes alternate prescaled predictor velocity")
        force, factor = self._predictor_inputs(state_in, state_aug, control, stage3_qd, dt)
        wp.launch(
            self.predictor_kernel,
            dim=self.solver.world_count,
            inputs=[self.plan, self.data, force, factor],
            block_dim=128,
            device=self.solver.model.device,
        )

    def _predictor_inputs(self, state_in, state_aug, control, stage3_qd, dt):
        """Bind only the existing live force and masked factor operands."""
        solver = self.solver
        force = ForceInput()
        force.joint_q, force.joint_qd, force.predictor_qd = state_in.joint_q, state_in.joint_qd, stage3_qd
        force.joint_f, force.body_f, force.body_flags = control.joint_f, state_in.body_f, solver.model.body_flags
        force.stiffness, force.reference, force.damping = (
            solver._passive_spring_stiffness,
            solver._passive_spring_ref,
            solver._passive_joint_damping,
        )
        force.u0, force.S = state_aug.joint_tau, state_aug.joint_S_s
        force.lower9, force.lower6 = solver.L_by_size[9], solver.L_by_size[6]
        force.kinematic_dof, force.kinematic_joint = solver._kinematic_dof_mask, solver._kinematic_joint_mask
        force.qdd, force.vhat, force.dt = state_aug.joint_qdd, solver.v_hat, dt
        factor = FactorInput()
        factor.mask = solver.mass_update_mask
        factor.armature9, factor.armature6 = solver.R_by_size[9], solver.R_by_size[6]
        factor.drive_row, factor.drive_K = solver._augmented_drive_row_by_dof, solver.aug_row_K
        factor.inertia = state_aug.body_I_s
        return force, factor

    def finish(self, state_in, state_aug, state_out, dt, next_refresh):
        self._validate_call(state_in, state_aug, dt)
        self._validate_call(state_out, state_aug, dt)
        if any(
            a.ptr < b.ptr + b.capacity and b.ptr < a.ptr + a.capacity
            for a in (state_in.joint_q, state_in.joint_qd)
            for b in (state_out.joint_q, state_out.joint_qd)
        ):
            raise RuntimeError("World-lane integration requires disjoint generalized state banks")
        kernel = self.finish_refresh_kernel if next_refresh else self.finish_held_kernel
        wp.launch(
            kernel,
            dim=self.solver.world_count,
            inputs=[
                self.plan,
                self._publication(state_in, state_aug, state_out, dt),
                self.data,
                self.solver._mass_update_requested,
                int(next_refresh),
            ],
            block_dim=128,
            device=self.solver.model.device,
        )
