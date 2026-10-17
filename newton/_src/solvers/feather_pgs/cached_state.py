# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Generate cached state arithmetic from explicit storage and lifetime operands.

The stages delimit ownership: admission, root frames/scans, motion services,
body publication, and validity. They are authored source fragments, never read
from or edited out of another producer. Integration and joint transforms share
one definition; mappings, operators and cache policies remain with their owners.
"""

import hashlib
import linecache
from dataclasses import dataclass
from string import Template
from textwrap import dedent, indent

import warp as wp

from . import kernels
from .publication_data import PublicationData


@wp.func_native("""
#if defined(__CUDA_ARCH__)
    return wp::int64(clock64());
#else
    return wp::int64(0);
#endif
""")
def _clock() -> wp.int64: ...


@dataclass(frozen=True)
class StateOperands:
    """Bind complete physical coordinates, joints and free-root transport."""

    bodies: int
    dofs: int
    joint: str
    dof: str
    root_loop: str
    root_slot: str

    def __post_init__(self):
        if self.bodies <= 0 or self.dofs <= 0:
            raise ValueError("Cached state requires positive physical dimensions")


@dataclass(frozen=True)
class StateStages:
    """Declare cache admission and producer/consumer ownership boundaries.

    Setup establishes world/group, cooperative lane/stride and cache admission.
    Relative adjusts each joint's root frame before storing its local pose.
    Poses completes the pose scan and publishes articulation origins.
    Motion fills axes and relative velocities, then completes the motion scan.
    Bodies consumes complete poses/motion and publishes physical body services.
    Collect completes reductions and stamps current/geometric validity.
    """

    setup: str
    relative: str
    poses: str
    motion: str
    bodies: str
    collect: str


def _compile(source, namespace, label):
    filename = f"<cached-{label}-{hashlib.sha256(source.encode()).hexdigest()}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    exec(compile(source, filename, "exec"), namespace)
    return namespace


def generalized_operations():
    """Bind the retained integration laws once for every cached state mapping."""
    return {
        "update_qdd": kernels._light_update_qdd_from_velocity,
        "remove_transport": kernels._light_remove_free_root_transport_from_qdd,
        "integrate": kernels._light_integrate_generalized_joints,
    }


_INTEGRATION = """
if wp.static(finish):
    for local in range(lane, $DOFS, stride):
        dof = $DOF
        update_qdd(
            dof,
            data.joint_qd,
            data.kinematic_dof_mask,
            1.0 / data.dt,
            data.v_out,
            data.joint_qdd,
        )
    _sync()
    $CLOCKI1
    $ROOT_LOOP:
        remove_transport(
            $ROOT_SLOT,
            data.free_root_joint_indices,
            data.joint_qd_start,
            data.kinematic_joint_mask,
            data.joint_qd,
            data.joint_qdd,
        )
    _sync()
    $CLOCKI2
    for local in range(lane, $BODIES, stride):
        joint = $JOINT
        integrate(
            joint,
            data.joint_type,
            data.joint_parent,
            data.joint_child,
            data.joint_q_start,
            data.joint_qd_start,
            data.kinematic_joint_mask,
            data.joint_dof_dim,
            data.body_com,
            data.joint_X_c,
            data.joint_q,
            data.joint_qd,
            data.joint_qdd,
            data.dt,
            data.angular_damping,
            data.joint_q_new,
            data.joint_qd_new,
        )
    _sync()
    q = data.joint_q_new
    qd = data.joint_qd_new
"""

_RELATIVE = """
for local in range(lane, $BODIES, stride):
    joint = $JOINT
    transform = kernels.jcalc_transform(
        data.joint_type[joint],
        data.joint_axis,
        data.joint_qd_start[joint],
        data.joint_dof_dim[joint, 0],
        data.joint_dof_dim[joint, 1],
        q,
        data.joint_q_start[joint],
    )
    relative = data.joint_X_p[joint] * transform * wp.transform_inverse(data.joint_X_c[joint])
"""


def make_state_kernel(name, signature, operands, stages, namespace, *, finish, clocked=False):
    """Compile one complete repair/finish with the owner's existing launch ABI.

    ``clocked`` adds the stage cycle counters; the caller passes it so that it
    is part of the caller's kernel cache key.
    """
    values = {
        "BODIES": operands.bodies,
        "DOFS": operands.dofs,
        "JOINT": operands.joint,
        "DOF": operands.dof,
        "ROOT_LOOP": operands.root_loop,
        "ROOT_SLOT": operands.root_slot,
        "CLOCKI1": "clock_i1 = _clock()" if clocked else "pass",
        "CLOCKI2": "clock_i2 = _clock()" if clocked else "pass",
    }
    blocks = [
        stages.setup,
        "address = _storage()\nq = data.joint_q\nqd = data.joint_qd",
        Template(_INTEGRATION).substitute(values),
        Template(_RELATIVE).substitute(values).rstrip()
        + "\n"
        + indent(dedent(stages.relative).strip(), "    ")
        + "\n    _store_pose(address, local, relative)",
        "_sync()",
        stages.poses,
        stages.motion,
        stages.bodies,
        stages.collect,
        "_release(address)",
    ]
    if clocked:
        # Stage attribution: cycle counts between stage boundaries, lane 0 of
        # each group, into data.stage_clocks[group, 0:8] (update_qdd, root
        # transport, joint integration, relative, poses, motion, bodies, collect).
        marks = {0: 0, 2: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6}
        timed = []
        for index, block in enumerate(blocks):
            timed.append(block)
            if index in marks:
                timed.append(f"clock_{marks[index]} = _clock()")
            if index == 2:
                timed.append("if not wp.static(finish):\n    clock_i1 = clock_0\n    clock_i2 = clock_0")
        deltas = [
            "    data.stage_clocks[group, 0] = clock_i1 - clock_0",
            "    data.stage_clocks[group, 1] = clock_i2 - clock_i1",
            "    data.stage_clocks[group, 2] = clock_1 - clock_i2",
        ] + [f"    data.stage_clocks[group, {k + 2}] = clock_{k + 1} - clock_{k}" for k in range(1, 6)]
        timed.insert(len(timed) - 1, "if lane == 0 and data.stage_clocks.shape[0] > group:\n" + "\n".join(deltas))
        blocks = timed
    body = "\n".join(dedent(block).strip("\n") for block in blocks)
    source = "def kinetic_state(" + signature + "):\n" + indent(body, "    ") + "\n"
    bindings = dict(
        namespace,
        wp=wp,
        kernels=kernels,
        PublicationData=PublicationData,
        finish=finish,
        _clock=_clock,
        **generalized_operations(),
    )
    function = _compile(source, bindings, "state")["kinetic_state"]
    function.__name__ = function.__qualname__ = name
    return wp.kernel(module="unique", enable_backward=False)(function)


_BODY_TERMS = """
$CLOCKB0
radius = wp.transform_get_translation(pose_com) - origin
rotation = wp.transform_get_rotation(pose_com)
mass = data.body_mass[body]
inertia_local = data.body_inertia[body]
linear, omega = wp.spatial_top(velocity), wp.spatial_bottom(velocity)
a, alpha = wp.spatial_top(acceleration), wp.spatial_bottom(acceleration)
inertia_com = wp.mat33()
angular_momentum = wp.vec3()
angular_acceleration_force = wp.vec3()
if $REFRESH != 0:
    matrix = wp.quat_to_matrix(rotation)
    inertia_com = matrix * inertia_local * wp.transpose(matrix)
    angular_momentum = inertia_com * omega
    angular_acceleration_force = inertia_com * alpha
else:
    angular_momentum = _inertia_action(rotation, inertia_local, omega)
    angular_acceleration_force = _inertia_action(rotation, inertia_local, alpha)
velocity_com = linear + wp.cross(omega, radius)
force = mass * (a + wp.cross(alpha, radius) + wp.cross(omega, velocity_com) - data.gravity[0])
torque = angular_acceleration_force + wp.cross(omega, angular_momentum) + wp.cross(radius, force)
wrench = wp.spatial_vector(force, torque)
$CLOCKB1
public_radius = wp.transform_point(pose, data.body_com[body]) - origin
data.body_qd[body] = wp.spatial_vector(linear + wp.cross(omega, public_radius), omega)
$VELOCITY
$CACHE.com_offset[world, local] = $COM_OFFSET
$CLOCKB2
$STORE
$CLOCKB3
"""


def make_body_terms(signature, namespace, *, refresh, cache, com_offset, store, body_velocity=False, clocked=False):
    """Share body bias/moment arithmetic while retaining each publication contract.

    ``clocked`` adds the body-term cycle counters (see :func:`make_state_kernel`).
    """
    clocks = {f"CLOCKB{k}": (f"clock_b{k} = _clock()" if clocked else "pass") for k in range(4)}
    if clocked:
        clocks["CLOCKB3"] = (
            "clock_b3 = _clock()\n"
            "if data.stage_clocks.shape[0] > world:\n"
            "    column = int(-1)\n"
            "    if local == 0:\n        column = 16\n"
            "    elif data.is_free_rigid[data.body_to_articulation[body]] != 0:\n        column = 19\n"
            "    if column >= 0:\n"
            "        data.stage_clocks[world, column] = clock_b1 - clock_b0\n"
            "        data.stage_clocks[world, column + 1] = clock_b2 - clock_b1\n"
            "        data.stage_clocks[world, column + 2] = clock_b3 - clock_b2"
        )
    body = Template(_BODY_TERMS).substitute(
        REFRESH=refresh,
        CACHE=cache,
        COM_OFFSET=com_offset,
        STORE=store,
        VELOCITY="data.body_v_s[body] = velocity" if body_velocity else "",
        **clocks,
    )
    source = "def _primary_terms(" + signature + "):\n" + indent(body.strip(), "    ") + "\n"
    bindings = dict(namespace, wp=wp, PublicationData=PublicationData, _clock=_clock)
    return wp.func(_compile(source, bindings, "body-terms")["_primary_terms"])
