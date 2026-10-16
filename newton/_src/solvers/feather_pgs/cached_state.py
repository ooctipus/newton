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


def make_state_kernel(name, signature, operands, stages, namespace, *, finish, launch_bounds=None):
    """Compile one complete repair/finish with the owner's existing launch ABI.

    ``launch_bounds`` is the CUDA ``(threads per block, minimum blocks per SM)``
    pair; it caps the kernel's registers at ``65536 / (threads * blocks)``.
    """
    values = {
        "BODIES": operands.bodies,
        "DOFS": operands.dofs,
        "JOINT": operands.joint,
        "DOF": operands.dof,
        "ROOT_LOOP": operands.root_loop,
        "ROOT_SLOT": operands.root_slot,
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
    body = "\n".join(dedent(block).strip("\n") for block in blocks)
    source = "def kinetic_state(" + signature + "):\n" + indent(body, "    ") + "\n"
    bindings = dict(
        namespace, wp=wp, kernels=kernels, PublicationData=PublicationData, finish=finish, **generalized_operations()
    )
    function = _compile(source, bindings, "state")["kinetic_state"]
    function.__name__ = function.__qualname__ = name
    options = {} if launch_bounds is None else {"launch_bounds": tuple(launch_bounds)}
    return wp.kernel(module="unique", enable_backward=False, **options)(function)


_BODY_TERMS = """
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
public_radius = wp.transform_point(pose, data.body_com[body]) - origin
data.body_qd[body] = wp.spatial_vector(linear + wp.cross(omega, public_radius), omega)
$VELOCITY
$CACHE.com_offset[world, local] = $COM_OFFSET
$STORE
"""


def make_body_terms(signature, namespace, *, refresh, cache, com_offset, store, body_velocity=False):
    """Share body bias/moment arithmetic while retaining each publication contract."""
    body = Template(_BODY_TERMS).substitute(
        REFRESH=refresh,
        CACHE=cache,
        COM_OFFSET=com_offset,
        STORE=store,
        VELOCITY="data.body_v_s[body] = velocity" if body_velocity else "",
    )
    source = "def _primary_terms(" + signature + "):\n" + indent(body.strip(), "    ") + "\n"
    bindings = dict(namespace, wp=wp, PublicationData=PublicationData)
    return wp.func(_compile(source, bindings, "body-terms")["_primary_terms"])
