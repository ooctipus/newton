# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Model-derived scalar-joint algebra for complete current-state producers.

Descriptors contain no state, force, mass, or timestep values. Unsupported
joint types/non-rigid constants keep the original local implementation.
"""

import ast
import functools
import hashlib
import inspect
import linecache
import textwrap

import numpy as np
import warp as wp

from ...sim import JointType


@wp.struct
class CoordinatePlan:
    index: wp.array[int]
    mode: wp.array[int]
    cosine: wp.array[wp.quat]
    sine: wp.array[wp.quat]
    translation: wp.array[wp.vec3]
    child_offset: wp.array[wp.vec3]
    screw: wp.array[wp.spatial_vector]


def _multiply(a, b):
    return np.r_[a[3] * b[:3] + b[3] * a[:3] + np.cross(a[:3], b[:3]), a[3] * b[3] - a[:3] @ b[:3]]


def _rotate(q, v):
    return (2 * q[3] * q[3] - 1) * v + 2 * q[:3] * (q[:3] @ v) + 2 * q[3] * np.cross(q[:3], v)


def build_coordinates(model):
    """Deduplicate exact authored constants; never normalize or modify them.

    The unit checks admit ordinary FP32 normalization roundoff, not a change
    to the authored axis. Non-unit axes cannot use screw invariance, although
    their quaternion coefficient identity alone would still be valid.
    """
    kinds = model.joint_type.numpy()
    xp, xc = model.joint_X_p.numpy(), model.joint_X_c.numpy()
    starts, dimensions = model.joint_qd_start.numpy(), model.joint_dof_dim.numpy()
    axes = model.joint_axis.numpy()
    index = np.full(model.joint_count, -1, np.int32)
    records, lookup = [], {}
    tolerance = 8 * np.finfo(np.float32).eps
    for joint, kind in enumerate(kinds):
        if kind not in (int(JointType.FIXED), int(JointType.REVOLUTE), int(JointType.PRISMATIC)):
            continue
        axis = np.zeros(3, np.float32)
        if kind != int(JointType.FIXED):
            if int(dimensions[joint].sum()) != 1:
                continue
            axis = axes[int(starts[joint])]
            if not np.all(np.isfinite(axis)) or abs(float(axis.astype(float) @ axis.astype(float)) - 1) > tolerance:
                continue
        if not np.all(np.isfinite(xp[joint])) or not np.all(np.isfinite(xc[joint])):
            continue
        qp, qc = xp[joint, 3:].astype(float), xc[joint, 3:].astype(float)
        if abs(float(qp @ qp) - 1) > tolerance or abs(float(qc @ qc) - 1) > tolerance:
            continue
        key = (int(kind), xp[joint].tobytes(), xc[joint].tobytes(), axis.tobytes())
        slot = lookup.get(key)
        if slot is None:
            pp, pc, a = xp[joint, :3].astype(float), xc[joint, :3].astype(float), axis.astype(float)
            conjugate = qc * np.array([-1, -1, -1, 1])
            cosine = _multiply(qp, conjugate)
            sine = np.zeros(4)
            translation = pp.copy()
            angular, linear = np.zeros(3), np.zeros(3)
            mode = 0
            if kind == int(JointType.REVOLUTE):
                mode = 1
                sine = _multiply(_multiply(qp, np.r_[a, 0]), conjugate)
                angular = _rotate(qc, a)
                linear = np.cross(pc, angular)
            elif kind == int(JointType.PRISMATIC):
                mode = 2
                translation -= _rotate(cosine, pc)
                sine[:3] = _rotate(qp, a)
                linear = _rotate(qc, a)
            else:
                translation -= _rotate(cosine, pc)
            mode |= 16 if np.all(pc == 0) else 0
            mode |= 32 if np.all(linear == 0) else 0
            slot = len(records)
            lookup[key] = slot
            records.append((mode, cosine, sine, translation, pc, np.r_[linear, angular]))
        index[joint] = slot
    # A zero-entry model still gets valid descriptors; its indices remain -1.
    if not records:
        records.append((0, [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0], [0, 0, 0], [0] * 6))
    plan = CoordinatePlan()
    plan.index = wp.array(index, dtype=int, device=model.device)
    for column, (name, dtype) in enumerate(
        (
            ("mode", int),
            ("cosine", wp.quat),
            ("sine", wp.quat),
            ("translation", wp.vec3),
            ("child_offset", wp.vec3),
            ("screw", wp.spatial_vector),
        )
    ):
        values = np.asarray([record[column] for record in records], dtype=np.int32 if column == 0 else np.float32)
        setattr(plan, name, wp.array(values, dtype=dtype, device=model.device))
    return plan


@wp.func
def local_pose(plan: CoordinatePlan, slot: int, q: float):
    mode = plan.mode[slot]
    kind = mode & 3
    rotation = plan.cosine[slot]
    position = plan.translation[slot]
    if kind == 1:
        half = q * 0.5
        rotation = rotation * wp.cos(half) + plan.sine[slot] * wp.sin(half)
        if (mode & 16) == 0:
            position = position - wp.quat_rotate(rotation, plan.child_offset[slot])
    elif kind == 2:
        direction = plan.sine[slot]
        position = position + wp.vec3(direction[0], direction[1], direction[2]) * q
    return wp.transform(position, rotation)


@wp.func
def current_screw(plan: CoordinatePlan, slot: int, pose: wp.transform, origin: wp.vec3):
    local = plan.screw[slot]
    rotation = wp.transform_get_rotation(pose)
    angular, linear = wp.vec3(), wp.vec3()
    if (plan.mode[slot] & 3) == 1:
        angular = wp.quat_rotate(rotation, wp.spatial_bottom(local))
        linear = wp.cross(wp.transform_get_translation(pose) - origin, angular)
    if (plan.mode[slot] & 32) == 0:
        linear = linear + wp.quat_rotate(rotation, wp.spatial_top(local))
    return wp.spatial_vector(linear, angular)


@wp.func_native("reinterpret_cast<float*>(address)[308+6*body]=value;")
def store_speed(address: wp.uint64, body: int, value: float): ...


@wp.func_native("return reinterpret_cast<float*>(address)[308+6*body];")
def load_speed(address: wp.uint64, body: int) -> float: ...


def _replace(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError(f"Compiled-coordinate source seam changed: {old[:70]!r}")
    return source.replace(old, new, 1)


@functools.cache
def body_terms():
    """Keep every original inertial/bias equation; share the public COM point."""
    from . import g1_kinetic_state  # noqa: PLC0415

    source = textwrap.dedent(inspect.getsource(g1_kinetic_state._body_terms.func))
    source = source.replace("def _body_terms(", "def _compiled_body_terms(", 1)
    source = _replace(
        source,
        "radius = wp.transform_get_translation(pose_com) - origin",
        "radius = wp.transform_point(pose, data.body_com[body]) - origin",
    )
    source = _replace(
        source, "rotation = wp.transform_get_rotation(pose_com)", "rotation = wp.transform_get_rotation(pose)"
    )
    source = _replace(
        source, "public_radius = wp.transform_point(pose, data.body_com[body]) - origin", "public_radius = radius"
    )
    namespace = dict(g1_kinetic_state.__dict__)
    filename = f"<compiled-coordinate-terms-{hashlib.sha256(source.encode()).hexdigest()}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    exec(compile(source, filename, "exec"), namespace)
    return namespace["_compiled_body_terms"]


def compile_g1_state(function):
    """Replace the admitted coordinate producer, preserving the original owner."""
    source = textwrap.dedent(inspect.getsource(function))
    tree = ast.parse(source)
    tree.body[0].args.args.append(ast.arg(arg="coordinates", annotation=ast.Name(id="CoordinatePlan", ctx=ast.Load())))
    source = ast.unparse(tree)
    first = source.index("    if wp.static(finish):")
    last = source.index("    _sync()\n    scan_poses(address, plan)", first)
    source = source[:first] + _COORDINATES + source[last:]
    first = source.index("        anchor = data.joint_X_p[joint]")
    last = source.index("        start = data.joint_qd_start[joint]", first)
    original = source[first:last]
    source = source[:first] + _MOTION + "        else:\n" + textwrap.indent(original, "    ") + source[last:]
    source = _replace(source, "pose * data.body_X_com[body]", "pose")
    source = source.replace("_body_terms(", "_compiled_body_terms(")
    closure = inspect.getclosurevars(function)
    namespace = dict(function.__globals__)
    namespace.update(closure.nonlocals)
    namespace["plan_type"] = function.__annotations__["plan"]
    namespace.update(
        CoordinatePlan=CoordinatePlan,
        _compiled_local_pose=local_pose,
        _compiled_screw=current_screw,
        _compiled_body_terms=body_terms(),
        _store_compiled_speed=store_speed,
        _load_compiled_speed=load_speed,
    )
    filename = f"<compiled-coordinate-state-{hashlib.sha256(source.encode()).hexdigest()}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    exec(compile(source, filename, "exec"), namespace)
    return namespace[function.__name__]


_COORDINATES = """    for local in range(lane, 44, stride):
        joint = js + local
        start = data.joint_qd_start[joint]
        kind = data.joint_type[joint]
        slot = coordinates.index[joint]
        qs = data.joint_q_start[joint]
        scalar_q = float(0.0)
        scalar_qd = float(0.0)
        if kind == kernels.JointType.REVOLUTE or kind == kernels.JointType.PRISMATIC:
            scalar_q = data.joint_q[qs]
            scalar_qd = data.joint_qd[start]
        if wp.static(finish):
            if kind == kernels.JointType.REVOLUTE or kind == kernels.JointType.PRISMATIC:
                scalar_qdd = float(0.0)
                if data.kinematic_dof_mask[start] != 0:
                    data.v_out[start] = scalar_qd
                else:
                    scalar_qdd = (data.v_out[start] - scalar_qd) * (1.0 / data.dt)
                data.joint_qdd[start] = scalar_qdd
                if data.kinematic_joint_mask[joint] == 0:
                    scalar_qd = scalar_qd + scalar_qdd * data.dt
                    scalar_q = scalar_q + scalar_qd * data.dt
                data.joint_q_new[qs] = scalar_q
                data.joint_qd_new[start] = scalar_qd
            else:
                count = data.joint_dof_dim[joint, 0] + data.joint_dof_dim[joint, 1]
                for k in range(count):
                    update_qdd(start + k, data.joint_qd, data.kinematic_dof_mask, 1.0 / data.dt,
                               data.v_out, data.joint_qdd)
                if local == 0:
                    remove_transport(plan.root_slot[group], data.free_root_joint_indices,
                                     data.joint_qd_start, data.kinematic_joint_mask,
                                     data.joint_qd, data.joint_qdd)
                integrate(joint, data.joint_type, data.joint_parent, data.joint_child,
                          data.joint_q_start, data.joint_qd_start, data.kinematic_joint_mask,
                          data.joint_dof_dim, data.body_com, data.joint_X_c,
                          data.joint_q, data.joint_qd, data.joint_qdd, data.dt,
                          data.angular_damping, data.joint_q_new, data.joint_qd_new)
            q = data.joint_q_new
            qd = data.joint_qd_new
        if slot >= 0:
            relative = _compiled_local_pose(coordinates, slot, scalar_q)
            _store_compiled_speed(address, local, scalar_qd)
        else:
            transform = kernels.jcalc_transform(kind, data.joint_axis, start,
                data.joint_dof_dim[joint, 0], data.joint_dof_dim[joint, 1], q, qs)
            relative = data.joint_X_p[joint] * transform * wp.transform_inverse(data.joint_X_c[joint])
        if local == 0:
            _store_origin(address, 1, wp.transform_get_translation(relative))
            relative = wp.transform(wp.vec3(), wp.transform_get_rotation(relative))
        _store_pose(address, local, relative)
"""

_MOTION = """        slot = coordinates.index[joint]
        if slot >= 0:
            velocity = wp.spatial_vector()
            if data.joint_type[joint] != kernels.JointType.FIXED:
                axis = _compiled_screw(coordinates, slot, _load_pose(address, local), origin)
                start = data.joint_qd_start[joint]
                data.joint_S_s[start] = axis
                velocity = axis * _load_compiled_speed(address, local)
        else:
""".removesuffix("        else:\n")
